//! swapchain module.

use core::{mem, ptr};
use function_name::named;
use crate::{
  gpu::{AcquireResult, OpaqueNativeHandleInfo, PresentationEngineParams, SwapchainStatus},
  gpu_backends::vulkan::{device::DeviceResource, device::LogicalDevice, utils::NonZeroHandle},
  types::{GpuError, GpuResult},
};
use aethervk_oshal_rlib::log;
use alloc::{string::ToString, vec::Vec};
use ash::vk::{self, Handle};

/// TODO: Document this item
pub(super) const MAX_FRAMES: usize = 8;
/// TODO: Document this item
pub(super) const MAX_DISCARDS: usize = 32;

/// Handles and data relative to an image acquired through a swapchain
struct SwapchainImage {
  pub image: NonZeroHandle<vk::Image>,
  pub image_view: NonZeroHandle<vk::ImageView>,
  pub submission_fence: Option<NonZeroHandle<vk::Fence>>,
  pub acquire_semaphore: Option<NonZeroHandle<vk::Semaphore>>,
  pub present_semaphore: NonZeroHandle<vk::Semaphore>,

  /// Fence signaled by WSI when the OS display compositor is completely done reading
  /// the image. Populated on creation ONLY if VK_EXT_swapchain_maintenance1 is enabled.
  pub present_fence: Option<NonZeroHandle<vk::Fence>>,
  /// Tracks if this present_fence has been successfully submitted to the WSI subsystem and needs waiting
  pub present_fence_in_use: bool,
}

#[derive(Clone)]
struct FrameDiscard {
  discarded_swapchains: Vec<NonZeroHandle<vk::SwapchainKHR>>,
  discarded_semaphores: Vec<NonZeroHandle<vk::Semaphore>>,
  discarded_image_views: Vec<NonZeroHandle<vk::ImageView>>,

  // Fences from vkQueueSubmit
  discarded_fences: Vec<NonZeroHandle<vk::Fence>>,

  // Fences from vkQueuePresentKHR (VK_EXT_swapchain_maintenance1)
  discarded_present_fences_to_wait: Vec<NonZeroHandle<vk::Fence>>,
  discarded_present_fences_to_destroy: Vec<NonZeroHandle<vk::Fence>>,

  // added for windowless
  discarded_images: Vec<NonZeroHandle<vk::Image>>,
  discarded_memories: Vec<NonZeroHandle<vk::DeviceMemory>>,

  /// Legacy Fallback: Delays destruction to give the OS display compositor a grace period
  skip_cycles: u32,
}

#[derive(Clone)]
struct SwapchainFrame {
  pub submission_fence: Option<NonZeroHandle<vk::Fence>>,
  pub acquire_semaphore: Option<NonZeroHandle<vk::Semaphore>>,
}

/// TODO: Document this item
pub(super) enum PresentationState {
  Windowed(WindowedPresentationState),
  Windowless(WindowlessPresentationState),
}

/// TODO: Document this item
pub(super) struct WindowedPresentationState {
  surface_instance: ash::khr::surface::Instance,
  surface_capabilities: ash::khr::get_surface_capabilities2::Instance,
  surface: NonZeroHandle<vk::SurfaceKHR>,

  swapchain_device: ash::khr::swapchain::Device,
  swapchain_maintenance1_device: Option<ash::ext::swapchain_maintenance1::Device>,
  swapchain: NonZeroHandle<vk::SwapchainKHR>,

  images: heapless::Vec<SwapchainImage, MAX_FRAMES>,
  next_image: usize,
  frames: heapless::Vec<SwapchainFrame, MAX_FRAMES>,
  frame_discards: heapless::Vec<FrameDiscard, MAX_FRAMES>,
  current_frame: usize,

  width: u32,
  height: u32,
  pre_transform: vk::SurfaceTransformFlagsKHR,
  surface_format: vk::SurfaceFormatKHR,
  vsync: bool,
  native_handle: OpaqueNativeHandleInfo,

  swapchain_generation: u64,
  physical_device: NonZeroHandle<vk::PhysicalDevice>,
  pending_resize: Option<(u32, u32)>,
  pub archetypes: crate::gpu::vulkan::device::archetypes_struct::Archetypes,
}

trait SwapchainCleanable {
  fn cleanup_windowless(&mut self, device: &ash::Device);
  fn cleanup(&mut self, swapchain_device: &ash::khr::swapchain::Device, device: &ash::Device);
}

struct SwapchainWrapper<'a, T>
where
  T: SwapchainCleanable,
{
  swapchain_device: &'a ash::khr::swapchain::Device,
  wrapped: &'a mut T,
}

impl<'a, T> DeviceResource for SwapchainWrapper<'a, T>
where
  T: SwapchainCleanable,
{
  fn cleanup(&mut self, device: &ash::Device) {
    self.wrapped.cleanup(self.swapchain_device, device);
  }
}

impl DeviceResource for SwapchainImage {
  fn cleanup(&mut self, device: &ash::Device) {
    // Only wait if it was successfully pushed to WSI without errors
    if self.present_fence_in_use {
      if let Some(fence) = self.present_fence {
        unsafe {
          let _ = device.wait_for_fences(&[fence.get()], true, u64::MAX);
        }
      }
    }
    unsafe { device.destroy_semaphore(self.present_semaphore.get(), None) };
    if let Some(fence) = self.present_fence {
      unsafe { device.destroy_fence(fence.get(), None) };
    }
    if let Some(sem) = self.acquire_semaphore {
      unsafe { device.destroy_semaphore(sem.get(), None) };
    }
    if let Some(fence) = self.submission_fence {
      unsafe { device.destroy_fence(fence.get(), None) };
    }
    unsafe { device.destroy_image_view(self.image_view.get(), None) };
  }
}

impl SwapchainCleanable for FrameDiscard {
  fn cleanup_windowless(&mut self, device: &ash::Device) {
    // Legacy fallback wrapper
    if self.skip_cycles > 0 {
      self.skip_cycles -= 1;
      return;
    }

    if !self.discarded_fences.is_empty() {
      unsafe {
        let fences: &[vk::Fence] = core::slice::from_raw_parts(
          self.discarded_fences.as_ptr() as *const vk::Fence,
          self.discarded_fences.len(),
        );
        let _ = device.wait_for_fences(&fences, true, u64::MAX);
      }
    }

    for &sem in self.discarded_semaphores.iter() {
      unsafe { device.destroy_semaphore(sem.get(), None) };
    }
    self.discarded_semaphores.clear();
    for &image_view in self.discarded_image_views.iter() {
      unsafe { device.destroy_image_view(image_view.get(), None) };
    }
    self.discarded_image_views.clear();

    for &fence in self.discarded_fences.iter() {
      unsafe { device.destroy_fence(fence.get(), None) }
    }
    self.discarded_fences.clear();

    for &image in self.discarded_images.iter() {
      unsafe { device.destroy_image(image.get(), None) }
    }
    self.discarded_images.clear();
    for &mem in self.discarded_memories.iter() {
      unsafe { device.free_memory(mem.get(), None) }
    }
    self.discarded_memories.clear();
  }

  fn cleanup(&mut self, swapchain_device: &ash::khr::swapchain::Device, device: &ash::Device) {
    // 1. Skip cleanup if the legacy WSI grace period is still actively protecting these resources
    if self.skip_cycles > 0 {
      self.skip_cycles -= 1;
      return;
    }

    // 2. Await OS WSI Presentation if modern extension fences are attached
    if !self.discarded_present_fences_to_wait.is_empty() {
      unsafe {
        let fences: &[vk::Fence] = core::slice::from_raw_parts(
          self.discarded_present_fences_to_wait.as_ptr() as *const vk::Fence,
          self.discarded_present_fences_to_wait.len(),
        );
        let _ = device.wait_for_fences(&fences, true, u64::MAX);
      }
    }
    self.discarded_present_fences_to_wait.clear();

    // 3. Destroy swapchain BEFORE the semaphores
    for &swapchain in self.discarded_swapchains.iter() {
      #[cfg(test)]
      std::println!("Destroying DISCARDED swapchain: {:?}", swapchain.get());
      unsafe { swapchain_device.destroy_swapchain(swapchain.get(), None) };
    }
    self.discarded_swapchains.clear();

    // 4. Await submission pipeline completion natively
    self.cleanup_windowless(device);

    for &fence in self.discarded_present_fences_to_destroy.iter() {
      unsafe { device.destroy_fence(fence.get(), None) };
    }
    self.discarded_present_fences_to_destroy.clear();
  }
}

impl SwapchainCleanable for SwapchainFrame {
  fn cleanup_windowless(&mut self, device: &ash::Device) {
    if let Some(fence) = self.submission_fence {
      unsafe { device.destroy_fence(fence.get(), None) };
    }
    if let Some(sem) = self.acquire_semaphore {
      unsafe { device.destroy_semaphore(sem.get(), None) };
    }
  }

  fn cleanup(&mut self, swapchain_device: &ash::khr::swapchain::Device, device: &ash::Device) {
    self.cleanup_windowless(device);
  }
}

impl DeviceResource for PresentationState {
  fn cleanup(&mut self, device: &ash::Device) {
    match self {
      Self::Windowed(state) => state.cleanup(device),
      Self::Windowless(state) => state.cleanup(device),
    }
  }
}

impl DeviceResource for WindowedPresentationState {
  fn cleanup(&mut self, device: &ash::Device) {
    for frame_discard in &mut self.frame_discards {
      frame_discard.skip_cycles = 0; // Force immediate absolute cleanup on window destruction
      frame_discard.cleanup(&self.swapchain_device, device);
    }
    for frame in &mut self.frames {
      frame.cleanup(&self.swapchain_device, device);
    }

    // Destroy swapchain before its images/semaphores
    #[cfg(test)]
    std::println!("Destroying MAIN swapchain: {:?}", self.swapchain.get());
    unsafe { self.swapchain_device.destroy_swapchain(self.swapchain.get(), None) };

    for image in &mut self.images {
      image.cleanup(device);
    }
    unsafe { self.surface_instance.destroy_surface(self.surface.get(), None) };
  }
}

unsafe impl Send for PresentationState {}
unsafe impl Sync for PresentationState {}

impl PresentationState {
  /// TODO: Document this item
  #[named]
  pub fn cancel_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<()> {
    match self {
      Self::Windowed(state) => state.cancel_image(device, graphics_queue, image_index, frame_index),
      Self::Windowless(state) => {
        state.cancel_image(device, graphics_queue, image_index, frame_index)
      }
    }
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn resize(
    &mut self,
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    match self {
      Self::Windowed(state) => state.resize(device, physical_device, width, height),
      Self::Windowless(state) => state.resize(instance, device, physical_device, width, height),
    }
  }

  /// TODO: Document this item
  pub(super) fn is_windowless(&self) -> bool {
    matches!(self, Self::Windowless(_))
  }

  pub(super) fn archetypes(&self) -> &crate::gpu::vulkan::device::archetypes_struct::Archetypes {
    match self {
      Self::Windowed(state) => &state.archetypes,
      Self::Windowless(state) => &state.archetypes,
    }
  }

  pub(super) fn archetypes_mut(&mut self) -> &mut crate::gpu::vulkan::device::archetypes_struct::Archetypes {
    match self {
      Self::Windowed(state) => &mut state.archetypes,
      Self::Windowless(state) => &mut state.archetypes,
    }
  }

  /// TODO: Document this item
  pub(super) fn extent(&self) -> (u32, u32) {
    match self {
      Self::Windowed(state) => state.extent(),
      Self::Windowless(state) => state.extent(),
    }
  }

  /// TODO: Document this item
  pub(super) fn format(&self) -> vk::Format {
    match self {
      Self::Windowed(state) => state.format(),
      Self::Windowless(state) => state.format(),
    }
  }

  /// TODO: Document this item
  pub(super) fn swapchain_generation(&self) -> u64 {
    match self {
      Self::Windowed(state) => state.swapchain_generation(),
      Self::Windowless(state) => state.swapchain_generation(),
    }
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn for_each_swapchain_image(
    &self,
    f: impl FnMut(NonZeroHandle<vk::ImageView>) -> GpuResult<()>,
  ) -> GpuResult<()> {
    match self {
      Self::Windowed(state) => state.for_each_swapchain_image(f),
      Self::Windowless(state) => state.for_each_swapchain_image(f),
    }
  }

  /// TODO: Document this item
  #[named]
  pub fn acquire_next_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
  ) -> GpuResult<AcquireResult> {
    match self {
      Self::Windowed(state) => state.acquire_next_image(device),
      Self::Windowless(state) => state.acquire_next_image(device, graphics_queue),
    }
  }

  /// TODO: Document this item
  pub unsafe fn get_frame_resources(
    &self,
    index: usize,
  ) -> (
    Option<NonZeroHandle<vk::Semaphore>>,
    NonZeroHandle<vk::Fence>,
  ) {
    match self {
      Self::Windowed(state) => unsafe { state.get_frame_resources(index) },
      Self::Windowless(state) => unsafe { state.get_frame_resources(index) },
    }
  }

  /// TODO: Document this item
  pub unsafe fn get_image_resources(
    &self,
    index: usize,
  ) -> (
    NonZeroHandle<vk::Image>,
    NonZeroHandle<vk::ImageView>,
    Option<NonZeroHandle<vk::Semaphore>>,
  ) {
    match self {
      Self::Windowed(state) => unsafe { state.get_image_resources(index) },
      Self::Windowless(state) => unsafe { state.get_image_resources(index) },
    }
  }

  /// TODO: Document this item
  #[named]
  pub unsafe fn submit_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<SwapchainStatus> {
    match self {
      Self::Windowed(state) => unsafe {
        state.submit_image(device, graphics_queue, image_index, frame_index)
      },
      Self::Windowless(state) => unsafe {
        state.submit_image(graphics_queue, image_index, frame_index)
      },
    }
  }

  /// TODO: Document this item
  #[named]
  pub fn new(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    swapchain_maintenance1_device: Option<ash::ext::swapchain_maintenance1::Device>,
    params: &PresentationEngineParams,
  ) -> GpuResult<Self> {
    match params.ty {
      crate::gpu::PresentationEngineType::Window => {
        Ok(Self::Windowed(WindowedPresentationState::new(
          entry,
          instance,
          device,
          physical_device,
          swapchain_maintenance1_device,
          params,
        )?))
      }
      crate::gpu::PresentationEngineType::WindowLess => {
        Ok(Self::Windowless(WindowlessPresentationState::new(
          instance,
          device,
          physical_device,
          params.width,
          params.height,
          params.buffer_count,
        )?))
      }
    }
  }
}

impl WindowedPresentationState {
  /// TODO: Document this item
  #[named]
  pub fn cancel_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<()> {
    let image = &mut self.images[image_index as usize];
    let frame = &mut self.frames[frame_index as usize];

    if image.eligible_for_acquisition() || frame.eligible_for_steal() {
      return Ok(());
    }

    let acquire_sem = frame.acquire_semaphore.ok_or(crate::gpu_err_device!())?.get();
    let present_sem = image.present_semaphore.get();
    let fence = frame.submission_fence.ok_or(crate::gpu_err_device!())?.get();

    // Reset explicit presentation fence before reuse if we're using modern tracking
    if self.swapchain_maintenance1_device.is_some() {
      if image.present_fence_in_use {
        let pfence = unsafe { image.present_fence.unwrap_unchecked().get() };
        unsafe {
          let _ = device.wait_for_fences(core::slice::from_ref(&pfence), false, u64::MAX);
          let _ = device.reset_fences(core::slice::from_ref(&pfence));
        }
        image.present_fence_in_use = false;
      }
    }

    let wait_semaphores = [acquire_sem];
    let wait_dst_stage_mask = [vk::PipelineStageFlags::BOTTOM_OF_PIPE];
    let signal_semaphores = [present_sem];

    let submit_info = vk::SubmitInfo::default()
      .wait_semaphores(&wait_semaphores)
      .wait_dst_stage_mask(&wait_dst_stage_mask)
      .signal_semaphores(&signal_semaphores);

    device
      .locked_queue_submit(graphics_queue, core::slice::from_ref(&submit_info), fence)
      .map_err(GpuError::from)?;

    let swapchains = [self.swapchain.get()];
    let image_indices = [image_index];

    let mut present_info = vk::PresentInfoKHR::default()
      .wait_semaphores(&signal_semaphores)
      .swapchains(&swapchains)
      .image_indices(&image_indices);

    // Chain modern WSI tracking fence if available
    let mut present_fence_info = vk::SwapchainPresentFenceInfoEXT::default();
    let mut present_fences = [vk::Fence::null()];
    if self.swapchain_maintenance1_device.is_some() {
      present_fences[0] = unsafe { image.present_fence.unwrap_unchecked().get() };
      present_fence_info = present_fence_info.fences(&present_fences);
      present_info = present_info.push_next(&mut present_fence_info);
    }

    let present_result = unsafe {
      let _guard = device.submission_lock.lock();
      self.swapchain_device.queue_present(graphics_queue, &present_info)
    };

    unsafe { image.reclaim_from_swapchain_frame(frame) };

    match present_result {
      Ok(_) | Err(vk::Result::SUBOPTIMAL_KHR) => {
        if self.swapchain_maintenance1_device.is_some() {
          image.present_fence_in_use = true;
        }
        Ok(())
      }
      Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
        image.present_fence_in_use = false; // Driver doesn't signal on failure
        Ok(())
      }
      Err(e) => {
        image.present_fence_in_use = false;
        Err(e.into())
      }
    }
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn resize(
    &mut self,
    _device: &ash::Device,
    _physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    if self.width != width || self.height != height {
      self.pending_resize = Some((width, height));
    }
    Ok(())
  }

  /// TODO: Document this item
  pub(super) fn extent(&self) -> (u32, u32) {
    (self.width, self.height)
  }

  /// TODO: Document this item
  pub(super) fn format(&self) -> vk::Format {
    self.surface_format.format
  }

  /// TODO: Document this item
  #[named]
  pub fn new(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    swapchain_maintenance1_device: Option<ash::ext::swapchain_maintenance1::Device>,
    params: &PresentationEngineParams,
  ) -> GpuResult<Self> {
    let surface_instance = ash::khr::surface::Instance::new(entry, instance);
    let native_handle = params.window_info;
    let surface = Self::create_surface(&entry, &instance, native_handle)?;
    let swapchain_device = ash::khr::swapchain::Device::new(instance, device);

    let surface_capabilities = ash::khr::get_surface_capabilities2::Instance::new(entry, instance);

    let mut this = Self {
      surface_instance,
      surface_capabilities,
      surface: unsafe { NonZeroHandle::new_unchecked(surface) },
      swapchain_device,
      swapchain_maintenance1_device,
      swapchain: NonZeroHandle::dangling(), // will be well-formed if recreate_swapchain goes through
      images: heapless::Vec::new(),
      next_image: 0,
      frames: heapless::Vec::new(),
      frame_discards: heapless::Vec::new(),
      current_frame: 0,
      width: params.width as _,
      height: params.height as _,
      surface_format: vk::SurfaceFormatKHR::default(),
      pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
      vsync: params.vsync,
      native_handle,
      swapchain_generation: 0,
      physical_device,
      pending_resize: None,
      archetypes: crate::gpu::vulkan::device::archetypes_struct::Archetypes::default(),
    };

    this.recreate_swapchain(device, false, physical_device)?;

    Ok(this)
  }

  // Purely lock-free resize. Uses the extension if available, otherwise relies on 1-frame delayed heuristic
  #[named]
  fn recreate_swapchain(
    &mut self,
    device: &ash::Device,
    use_old_swapchain: bool,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
  ) -> GpuResult<()> {
    debug_assert!(
      self.frame_discards.len() == self.frames.len()
        && self.frame_discards.len() >= self.images.len()
    );

    let (mut swapchain, extent, transform, surface_format, _) =
      self.create_swapchain_internal(physical_device, use_old_swapchain)?;
    let mut swapchain_images =
      self.recreate_swapchain_images(device, swapchain, surface_format.format)?;

    mem::swap(&mut swapchain, &mut self.swapchain);

    if self.frames.len() > self.current_frame {
      let prev_frame = (self.current_frame + self.frames.len() - 1) % self.frames.len();
      let frame_discard = &mut self.frame_discards[prev_frame];

      // HYBRID FIX: Either we have deterministic tracking, or we need a legacy Grace cycle.
      if self.swapchain_maintenance1_device.is_some() {
        frame_discard.skip_cycles = 0;
      } else {
        frame_discard.skip_cycles = 1;
      }

      for frame in &mut self.frames {
        if let Some(fence) = frame.submission_fence.take() {
          let _ = frame_discard.discarded_fences.push(fence);
        }
        if let Some(sem) = frame.acquire_semaphore.take() {
          let _ = frame_discard.discarded_semaphores.push(sem);
        }
      }

      if !self.images.is_empty() {
        frame_discard.discard_swapchain_images(&mut self.images, false);
        self.images.clear();
      }

      if swapchain.get() != self.swapchain.get() {
        unsafe { frame_discard.discard_decommissioned_swapchain(swapchain) };
      }
    }

    let (frame_semaphores, mut frame_fences) =
      self.recreate_swapchain_frame_resources(device, swapchain_images.len())?;

    self.next_image = 0;
    self.current_frame = 0;
    self.update_metadata(extent, transform, surface_format);

    self.images = swapchain_images;

    if self.images.len() > self.frames.len() {
      let _ = self.frames.resize_default(self.images.len());
      let _ = self.frame_discards.resize_default(self.images.len());
    } else if self.images.len() < self.frames.len() {
      // Fold orphaned discards into bin 0
      let new_len = self.images.len();
      for i in new_len..self.frames.len() {
        let mut orphaned = core::mem::take(&mut self.frame_discards[i]);
        self.frame_discards[0].discarded_swapchains.append(&mut orphaned.discarded_swapchains);
        self.frame_discards[0].discarded_semaphores.append(&mut orphaned.discarded_semaphores);
        self.frame_discards[0].discarded_image_views.append(&mut orphaned.discarded_image_views);
        self.frame_discards[0].discarded_fences.append(&mut orphaned.discarded_fences);
        self.frame_discards[0]
          .discarded_present_fences_to_wait
          .append(&mut orphaned.discarded_present_fences_to_wait);
        self.frame_discards[0]
          .discarded_present_fences_to_destroy
          .append(&mut orphaned.discarded_present_fences_to_destroy);
        self.frame_discards[0].discarded_images.append(&mut orphaned.discarded_images);
        self.frame_discards[0].discarded_memories.append(&mut orphaned.discarded_memories);

        // Pass down legacy protections if applicable
        self.frame_discards[0].skip_cycles =
          self.frame_discards[0].skip_cycles.max(orphaned.skip_cycles);

        let mut orphaned_frame = core::mem::take(&mut self.frames[i]);
        if let Some(f) = orphaned_frame.submission_fence.take() {
          let _ = self.frame_discards[0].discarded_fences.push(f);
        }
        if let Some(s) = orphaned_frame.acquire_semaphore.take() {
          let _ = self.frame_discards[0].discarded_semaphores.push(s);
        }
      }

      // Assure folded resources get at least 1 cycle of grace on legacy hardware
      if new_len > 0 && self.swapchain_maintenance1_device.is_none() {
        self.frame_discards[0].skip_cycles = self.frame_discards[0].skip_cycles.max(1);
      }

      self.frames.truncate(new_len);
      self.frame_discards.truncate(new_len);
    }

    for i in 0..self.images.len() {
      unsafe {
        let image = self.images.get_unchecked_mut(i);
        debug_assert!(image.acquire_semaphore.is_none() && image.submission_fence.is_none());
        image.acquire_semaphore = Some(*frame_semaphores.get_unchecked(i));
        let fence = *frame_fences.get_unchecked(i);
        image.submission_fence = if fence.is_null() {
          let frame = self.frames.get_unchecked_mut(i);
          debug_assert!(frame.submission_fence.is_some_and(|f| !f.is_null()));
          frame.submission_fence.take()
        } else {
          Some(NonZeroHandle::new_unchecked(fence))
        };
      }
    }

    if !self.images.is_empty() {
      self.next_image %= self.images.len()
    }
    if !self.frames.is_empty() {
      self.current_frame %= self.frames.len()
    }
    debug_assert!(
      self.frame_discards.len() == self.frames.len()
        && self.frame_discards.len() >= self.images.len()
    );
    self.swapchain_generation += 1;
    Ok(())
  }

  /// TODO: Document this item
  pub(super) fn swapchain_generation(&self) -> u64 {
    self.swapchain_generation
  }

  #[named]
  fn recreate_swapchain_images(
    &self,
    device: &ash::Device,
    swapchain: NonZeroHandle<vk::SwapchainKHR>,
    format: vk::Format,
  ) -> GpuResult<heapless::Vec<SwapchainImage, MAX_FRAMES>> {
    let mut images = heapless::Vec::<vk::Image, MAX_FRAMES>::new();
    let mut count: u32 = unsafe {
      let mut v = 0u32;
      (self.swapchain_device.fp().get_swapchain_images_khr)(
        self.swapchain_device.device(),
        swapchain.get(),
        ptr::from_mut(&mut v),
        ptr::null_mut(),
      )
      .result_with_success(v)
    }?;
    images.resize(count as _, vk::Image::null()).map_err(|_| {
      GpuError::UnsupportedFeatureNamed(
        "Too many images from swapchain. Increase Limit".to_string(),
      )
    })?;
    unsafe {
      (self.swapchain_device.fp().get_swapchain_images_khr)(
        self.swapchain_device.device(),
        swapchain.get(),
        ptr::from_mut(&mut count),
        images.as_mut_ptr(),
      )
      .result()
    }?;

    let sem_create_info = vk::SemaphoreCreateInfo::default();

    // Fences supplied to the WSI extension are just normal, unsignaled core fences
    let pf_create_info = vk::FenceCreateInfo::default();

    let img_view_create_info = vk::ImageViewCreateInfo::default()
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(format)
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_array_layer(0)
          .base_mip_level(0)
          .layer_count(1)
          .level_count(1),
      );

    let mut result = heapless::Vec::<SwapchainImage, MAX_FRAMES>::new();
    for i in 0..count {
      let info = img_view_create_info.image(images[i as usize]);
      unsafe {
        let image_view = NonZeroHandle::new_unchecked(device.create_image_view(&info, None)?);
        let present_semaphore =
          NonZeroHandle::new_unchecked(device.create_semaphore(&sem_create_info, None)?);

        let present_fence = if self.swapchain_maintenance1_device.is_some() {
          Some(NonZeroHandle::new_unchecked(
            device.create_fence(&pf_create_info, None)?,
          ))
        } else {
          None
        };

        result.push_unchecked(SwapchainImage {
          image: NonZeroHandle::new_unchecked(*images.get_unchecked(i as usize)),
          image_view,
          submission_fence: None,
          acquire_semaphore: None,
          present_semaphore,
          present_fence,
          present_fence_in_use: false,
        })
      }
    }
    Ok(result)
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn for_each_swapchain_image(
    &self,
    mut f: impl FnMut(NonZeroHandle<vk::ImageView>) -> GpuResult<()>,
  ) -> GpuResult<()> {
    for image in &self.images {
      (&mut f)(image.image_view)?;
    }
    Ok(())
  }

  #[named]
  fn can_swapchain_image_be_transfer(surf_caps: &vk::SurfaceCapabilities2KHR) -> GpuResult<()> {
    let flags = surf_caps.surface_capabilities.supported_usage_flags;
    if flags.contains(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC) {
      Ok(())
    } else {
      Err(GpuError::UnsupportedFeature)
    }
  }

  #[named]
  fn recreate_swapchain_frame_resources(
    &self,
    device: &ash::Device,
    count: usize,
  ) -> GpuResult<(
    heapless::Vec<NonZeroHandle<vk::Semaphore>, MAX_FRAMES>,
    heapless::Vec<vk::Fence, MAX_FRAMES>,
  )> {
    let mut semaphores = heapless::Vec::<NonZeroHandle<vk::Semaphore>, MAX_FRAMES>::new();
    let mut fences = heapless::Vec::<vk::Fence, MAX_FRAMES>::new();
    let sem_create_info = vk::SemaphoreCreateInfo::default();
    let fence_create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    for i in 0..count {
      unsafe {
        fences.push_unchecked(if i < self.frames.len() {
          let frame = self.frames.get_unchecked(i);
          if frame.submission_fence.is_none() {
            device.create_fence(&fence_create_info, None)?
          } else {
            vk::Fence::null()
          }
        } else {
          device.create_fence(&fence_create_info, None)?
        });
        semaphores.push_unchecked(NonZeroHandle::new_unchecked(
          device.create_semaphore(&sem_create_info, None)?,
        ));
      }
    }

    Ok((semaphores, fences))
  }

  #[named]
  fn create_swapchain_internal(
    &self,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    use_old_swapchain: bool,
  ) -> GpuResult<(
    NonZeroHandle<vk::SwapchainKHR>,
    vk::Extent2D,
    vk::SurfaceTransformFlagsKHR,
    vk::SurfaceFormatKHR,
    vk::PresentModeKHR,
  )> {
    let surface_format = self.select_surface_format(physical_device)?;
    let present_mode = self.select_present_mode(physical_device)?;

    let mut present_mode_ext = vk::SurfacePresentModeEXT::default().present_mode(present_mode);
    let surf_info = vk::PhysicalDeviceSurfaceInfo2KHR::default()
      .surface(self.surface.get())
      .push_next(&mut present_mode_ext);
    let mut surf_caps = vk::SurfaceCapabilities2KHR::default();
    unsafe {
      self.surface_capabilities.get_physical_device_surface_capabilities2(
        physical_device.get(),
        &surf_info,
        &mut surf_caps,
      )
    }?;
    let (extent, transform, image_count) =
      self.extent_transform_imagecount(&surf_caps, present_mode);
    let composite_alpha = Self::get_supported_composite_alpha(&surf_caps)?;

    Self::can_swapchain_image_be_transfer(&surf_caps)?;

    let create_info = vk::SwapchainCreateInfoKHR::default()
      .surface(self.surface.get())
      .min_image_count(image_count)
      .image_format(surface_format.format)
      .image_array_layers(1)
      .image_color_space(surface_format.color_space)
      .image_extent(extent)
      .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
      .image_usage(
        vk::ImageUsageFlags::COLOR_ATTACHMENT
          | vk::ImageUsageFlags::TRANSFER_SRC
          | vk::ImageUsageFlags::TRANSFER_DST,
      )
      .composite_alpha(composite_alpha)
      .clipped(true)
      .old_swapchain(if use_old_swapchain {
        self.swapchain.get()
      } else {
        vk::SwapchainKHR::null()
      })
      .pre_transform(transform)
      .present_mode(present_mode);

    let swapchain = unsafe {
      NonZeroHandle::new_unchecked(self.swapchain_device.create_swapchain(&create_info, None)?)
    };

    Ok((swapchain, extent, transform, surface_format, present_mode))
  }

  fn update_metadata(
    &mut self,
    extent: vk::Extent2D,
    transform: vk::SurfaceTransformFlagsKHR,
    surface_format: vk::SurfaceFormatKHR,
  ) {
    self.width = extent.width;
    self.height = extent.height;
    self.pre_transform = transform;
    self.surface_format = surface_format;
  }

  fn get_supported_composite_alpha(
    surf_caps: &vk::SurfaceCapabilities2KHR,
  ) -> GpuResult<vk::CompositeAlphaFlagsKHR> {
    let supported = surf_caps.surface_capabilities.supported_composite_alpha;

    if supported.contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED) {
      Ok(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
    } else if supported.contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED) {
      Ok(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
    } else if supported.contains(vk::CompositeAlphaFlagsKHR::INHERIT) {
      Ok(vk::CompositeAlphaFlagsKHR::INHERIT)
    } else if supported.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
      Ok(vk::CompositeAlphaFlagsKHR::OPAQUE)
    } else {
      Err(GpuError::UnsupportedFeature)
    }
  }

  fn extent_transform_imagecount(
    &self,
    surf_caps: &vk::SurfaceCapabilities2KHR,
    present_mode: vk::PresentModeKHR,
  ) -> (vk::Extent2D, vk::SurfaceTransformFlagsKHR, u32) {
    let transform = surf_caps.surface_capabilities.current_transform;
    let extent: vk::Extent2D = {
      let mut width = if surf_caps.surface_capabilities.current_extent.width == u32::MAX {
        self.width as u32
      } else {
        surf_caps.surface_capabilities.current_extent.width
      };
      let mut height = if surf_caps.surface_capabilities.current_extent.height == u32::MAX {
        self.height as u32
      } else {
        surf_caps.surface_capabilities.current_extent.height
      };
      if width < surf_caps.surface_capabilities.min_image_extent.width {
        width = surf_caps.surface_capabilities.min_image_extent.width;
      } else if width > surf_caps.surface_capabilities.max_image_extent.width {
        width = surf_caps.surface_capabilities.max_image_extent.width;
      }
      if height < surf_caps.surface_capabilities.min_image_extent.height {
        height = surf_caps.surface_capabilities.min_image_extent.height;
      } else if height > surf_caps.surface_capabilities.max_image_extent.height {
        height = surf_caps.surface_capabilities.max_image_extent.height;
      }
      match transform {
        vk::SurfaceTransformFlagsKHR::ROTATE_90
        | vk::SurfaceTransformFlagsKHR::ROTATE_270
        | vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_90
        | vk::SurfaceTransformFlagsKHR::HORIZONTAL_MIRROR_ROTATE_270 => {
          mem::swap(&mut width, &mut height);
        }
        _ => {}
      }
      vk::Extent2D { width, height }
    };
    let image_count = {
      let mut value = if present_mode == vk::PresentModeKHR::IMMEDIATE {
        2
      } else {
        3
      } as u32;
      if value < surf_caps.surface_capabilities.min_image_count {
        value = surf_caps.surface_capabilities.min_image_count;
      } else if surf_caps.surface_capabilities.max_image_count != 0
        && value > surf_caps.surface_capabilities.max_image_count
      {
        value = surf_caps.surface_capabilities.max_image_count;
      }
      value
    };
    (extent, transform, image_count)
  }

  fn select_surface_format(
    &self,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
  ) -> ash::prelude::VkResult<vk::SurfaceFormatKHR> {
    let formats = unsafe {
      self
        .surface_instance
        .get_physical_device_surface_formats(physical_device.get(), self.surface.get())
    }?;
    if formats.is_empty() {
      return Err(ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED);
    }
    let desired_formats = [
      vk::Format::B8G8R8A8_UNORM,
      vk::Format::R8G8B8A8_UNORM,
      vk::Format::A8B8G8R8_UNORM_PACK32,
    ];
    for format in desired_formats {
      for surface_format in formats.iter() {
        if surface_format.format == format {
          return Ok(*surface_format);
        }
      }
    }
    Ok(if formats[0].format == vk::Format::UNDEFINED {
      vk::SurfaceFormatKHR {
        format: vk::Format::B8G8R8A8_UNORM,
        color_space: formats[0].color_space,
      }
    } else {
      formats[0]
    })
  }

  fn select_present_mode(
    &self,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
  ) -> ash::prelude::VkResult<vk::PresentModeKHR> {
    let present_modes = unsafe {
      self
        .surface_instance
        .get_physical_device_surface_present_modes(physical_device.get(), self.surface.get())
    }?;
    let desired_present_mode = if self.vsync {
      vk::PresentModeKHR::MAILBOX
    } else {
      vk::PresentModeKHR::IMMEDIATE
    };
    if present_modes.contains(&desired_present_mode) {
      Ok(desired_present_mode)
    } else {
      Ok(vk::PresentModeKHR::FIFO)
    }
  }

  fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    native_handle: OpaqueNativeHandleInfo,
  ) -> ash::prelude::VkResult<vk::SurfaceKHR> {
    #[cfg(test)]
    if native_handle.ptr0 == core::ptr::null_mut() && native_handle.ptr1 == core::ptr::null_mut() {
      let headless_instance = ash::ext::headless_surface::Instance::new(entry, instance);
      let create_info = vk::HeadlessSurfaceCreateInfoEXT::default();
      return unsafe { headless_instance.create_headless_surface(&create_info, None) };
    }

    #[cfg(windows)]
    {
      let win32_instance = ash::khr::win32_surface::Instance::new(entry, instance);
      let create_info = vk::Win32SurfaceCreateInfoKHR::default()
        .hinstance(native_handle.ptr0 as _)
        .hwnd(native_handle.ptr1 as _);
      unsafe { win32_instance.create_win32_surface(&create_info, None) }
    }
    #[cfg(all(target_os = "linux", feature = "linux_wayland"))]
    {
      let wayland_instance = ash::khr::wayland_surface::Instance::new(entry, instance);
      let create_info = vk::WaylandSurfaceCreateInfoKHR::default()
        .display(native_handle.ptr0 as _)
        .surface(native_handle.ptr1 as _);
      unsafe { wayland_instance.create_wayland_surface(&create_info, None) }
    }
    #[cfg(all(target_os = "linux", feature = "linux_xcb"))]
    {
      let xcb_instance = ash::khr::xcb_surface::Instance::new(entry, instance);
      let create_info = vk::XcbSurfaceCreateInfoKHR::default()
        .connection(native_handle.ptr0 as _)
        .window(native_handle.ptr1 as _);
      unsafe { xcb_instance.create_xcb_surface(&create_info, None) }
    }
    #[cfg(all(target_os = "linux", feature = "linux_xlib"))]
    {
      let xlib_instance = ash::khr::xlib_surface::Instance::new(entry, instance);
      let create_info = vk::XlibSurfaceCreateInfoKHR::default()
        .dpy(native_handle.ptr0 as _)
        .window(native_handle.ptr1 as _);
      unsafe { xlib_instance.create_xlib_surface(&create_info, None) }
    }
    #[cfg(target_os = "macos")]
    {
      let metal_instance = ash::ext::metal_surface::Instance::new(entry, instance);
      let create_info = vk::MetalSurfaceCreateInfoEXT::default().layer(native_handle.ptr0 as _);
      unsafe { metal_instance.create_metal_surface(&create_info, None) }
    }
  }

  /// TODO: Document this item
  #[named]
  pub fn acquire_next_image(&mut self, device: &LogicalDevice) -> GpuResult<AcquireResult> {
    if let Some((w, h)) = self.pending_resize.take() {
      self.width = w;
      self.height = h;
      self.recreate_swapchain(device, true, self.physical_device)?;
    }

    const FIRST_ATTEMPT_TIMEOUT_NS: u64 = 167;
    let images_count = self.images.len();
    let frame_count = self.frames.len();
    let next_frame = (self.current_frame + 1) % frame_count;
    let swapchain_image = &mut self.images[self.next_image];
    if !swapchain_image.eligible_for_acquisition()
      || !self.frames[self.current_frame].eligible_for_steal()
    {
      return Err(crate::types::GpuError::BackendSpecific(alloc::format!(
        "[Vulkan RenderDevice] window_acquire: not eligible. img_elig={} frame_elig={} current_frame={} next_frame={} next_image={} | image_count={} frame_count={}",
        swapchain_image.eligible_for_acquisition(),
        self.frames[next_frame].eligible_for_steal(),
        self.current_frame,
        next_frame,
        self.next_image,
        images_count,
        frame_count,
      )));
    }
    let fences: &[vk::Fence] = unsafe {
      core::slice::from_ref(&swapchain_image.submission_fence.as_ref().unwrap_unchecked())
    };

    let mut timeout = FIRST_ATTEMPT_TIMEOUT_NS;
    loop {
      let result = unsafe { device.wait_for_fences(fences, false, timeout) };
      if let Err(vk_result) = result {
        if vk_result == vk::Result::TIMEOUT {
          timeout = u64::MAX;
        } else {
          return Err(vk_result.into());
        }
      } else {
        break;
      }
    }

    self.frame_discards[self.current_frame].cleanup(&self.swapchain_device, device);

    unsafe { device.reset_fences(fences) }?;

    let (image_index, vk_result) = unsafe {
      let mut index = 0u32;
      let vk_result = (self.swapchain_device.fp().acquire_next_image_khr)(
        self.swapchain_device.device(),
        self.swapchain.get(),
        u64::MAX,
        swapchain_image.acquire_semaphore.unwrap_unchecked().get(),
        vk::Fence::null(),
        ptr::from_mut(&mut index),
      );
      (index, vk_result)
    };

    match vk_result {
      vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => {
        if image_index as usize != self.next_image {
          let next_fence = self.images[self.next_image].submission_fence.take();
          let next_sem = self.images[self.next_image].acquire_semaphore.take();
          let actual_fence = self.images[image_index as usize].submission_fence.take();
          let actual_sem = self.images[image_index as usize].acquire_semaphore.take();

          self.images[image_index as usize].submission_fence = next_fence;
          self.images[image_index as usize].acquire_semaphore = next_sem;
          self.images[self.next_image].submission_fence = actual_fence;
          self.images[self.next_image].acquire_semaphore = actual_sem;

          if let Some(fence) = self.images[self.next_image].submission_fence {
            unsafe {
              let _ = device.wait_for_fences(&[fence.get()], false, u64::MAX);
            }
          }
        }

        let actual_image = &mut self.images[image_index as usize];
        unsafe { self.frames[self.current_frame].steal_from_swapchain_image(actual_image) };
        let frame_idx_for_submission = self.current_frame;

        self.next_image = (self.next_image + 1) % images_count;
        self.current_frame = next_frame;
        Ok(AcquireResult {
          image_index,
          status: SwapchainStatus::Optimal,
          frame_index: frame_idx_for_submission as u64,
          swapchain_generation: self.swapchain_generation,
        })
      }
      vk::Result::ERROR_OUT_OF_DATE_KHR => Ok(AcquireResult {
        image_index: u32::MAX,
        status: SwapchainStatus::NeedsRecreation,
        frame_index: self.current_frame as u64,
        swapchain_generation: self.swapchain_generation,
      }),
      _ => {
        return Err(vk_result.into());
      }
    }
  }

  /// TODO: Document this item
  pub unsafe fn get_frame_resources(
    &self,
    index: usize,
  ) -> (
    Option<NonZeroHandle<vk::Semaphore>>,
    NonZeroHandle<vk::Fence>,
  ) {
    debug_assert!(self.frames.len() > index);
    let frame = &self.frames[index];
    debug_assert!(!frame.eligible_for_steal());
    unsafe {
      (
        Some(frame.acquire_semaphore.unwrap_unchecked()),
        frame.submission_fence.unwrap_unchecked(),
      )
    }
  }

  /// TODO: Document this item
  pub unsafe fn get_image_resources(
    &self,
    index: usize,
  ) -> (
    NonZeroHandle<vk::Image>,
    NonZeroHandle<vk::ImageView>,
    Option<NonZeroHandle<vk::Semaphore>>,
  ) {
    debug_assert!(self.images.len() > index);
    let image = &self.images[index];
    debug_assert!(!image.eligible_for_acquisition());
    (image.image, image.image_view, Some(image.present_semaphore))
  }

  /// TODO: Document this item
  #[named]
  pub unsafe fn submit_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<SwapchainStatus> {
    let image = &mut self.images[image_index as usize];
    let frame = &mut self.frames[frame_index as usize];
    if image.eligible_for_acquisition() || frame.eligible_for_steal() {
      return Err(crate::gpu_err!("window_submit: still eligible"));
    }

    if self.swapchain_maintenance1_device.is_some() {
      if image.present_fence_in_use {
        let pfence = unsafe { image.present_fence.unwrap_unchecked().get() };
        unsafe {
          let _ = device.wait_for_fences(core::slice::from_ref(&pfence), false, u64::MAX);
          let _ = device.reset_fences(core::slice::from_ref(&pfence));
        }
        image.present_fence_in_use = false;
      }
    }

    let wait_semaphores = [image.present_semaphore.get()];
    let swapchains = [self.swapchain.get()];
    let image_indices = [image_index];

    let mut present_info = vk::PresentInfoKHR::default()
      .wait_semaphores(&wait_semaphores)
      .swapchains(&swapchains)
      .image_indices(&image_indices);

    let mut present_fence_info = vk::SwapchainPresentFenceInfoEXT::default();
    let mut present_fences = [vk::Fence::null()];

    // Chain modern WSI tracking fence if available
    if self.swapchain_maintenance1_device.is_some() {
      present_fences[0] = unsafe { image.present_fence.unwrap_unchecked().get() };
      present_fence_info = present_fence_info.fences(&present_fences);
      present_info = present_info.push_next(&mut present_fence_info);
    }

    let result = {
      let _guard = device.submission_lock.lock();
      unsafe { self.swapchain_device.queue_present(graphics_queue, &present_info) }
    };

    unsafe { image.reclaim_from_swapchain_frame(frame) };

    match result {
      Ok(suboptimal) => {
        if self.swapchain_maintenance1_device.is_some() {
          image.present_fence_in_use = true;
        }
        if suboptimal {
          self.pending_resize = Some((self.width, self.height));
          Ok(SwapchainStatus::Suboptimal)
        } else {
          Ok(SwapchainStatus::Optimal)
        }
      }
      Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
        image.present_fence_in_use = false;
        self.pending_resize = Some((self.width, self.height));
        Ok(SwapchainStatus::NeedsRecreation)
      }
      Err(e) => {
        image.present_fence_in_use = false;
        Err(e.into())
      }
    }
  }
}

impl FrameDiscard {
  fn discard_swapchain_images(
    &mut self,
    swapchain_images: &mut [SwapchainImage],
    is_windowless: bool,
  ) {
    for swapchain_image in swapchain_images {
      unsafe {
        if is_windowless {
          let _ = self.discarded_images.push(swapchain_image.image);
        }
        let _ = self.discarded_image_views.push(swapchain_image.image_view);
        let _ = self.discarded_semaphores.push(swapchain_image.present_semaphore);

        if let Some(sem) = swapchain_image.acquire_semaphore.take() {
          let _ = self.discarded_semaphores.push(sem);
        }
        if let Some(fence) = swapchain_image.submission_fence.take() {
          let _ = self.discarded_fences.push(fence);
        }
        if let Some(fence) = swapchain_image.present_fence.take() {
          if swapchain_image.present_fence_in_use {
            let _ = self.discarded_present_fences_to_wait.push(fence);
          }
          let _ = self.discarded_present_fences_to_destroy.push(fence);
        }
      }
    }
  }

  fn discard_swapchain_frame_keep_fences(&mut self, swapchain_frames: &mut [SwapchainFrame]) {
    for swapchain_frame in swapchain_frames {
      unsafe {
        if let Some(acquire_semaphore) = swapchain_frame.acquire_semaphore.take() {
          let _ = self.discarded_semaphores.push(acquire_semaphore);
        }
      }
    }
  }

  unsafe fn discard_decommissioned_swapchain(
    &mut self,
    swapchain: NonZeroHandle<vk::SwapchainKHR>,
  ) {
    let _ = self.discarded_swapchains.push(swapchain);
  }
}

impl Default for SwapchainFrame {
  fn default() -> Self {
    Self {
      submission_fence: None,
      acquire_semaphore: None,
    }
  }
}

impl Default for FrameDiscard {
  fn default() -> Self {
    Self {
      discarded_swapchains: Vec::new(),
      discarded_semaphores: Vec::new(),
      discarded_image_views: Vec::new(),
      discarded_fences: Vec::new(),
      discarded_present_fences_to_wait: Vec::new(),
      discarded_present_fences_to_destroy: Vec::new(),
      discarded_images: Vec::new(),
      discarded_memories: Vec::new(),
      skip_cycles: 0,
    }
  }
}

impl SwapchainImage {
  fn eligible_for_acquisition(&self) -> bool {
    self.submission_fence.is_some() && self.acquire_semaphore.is_some()
  }

  unsafe fn reclaim_from_swapchain_frame(&mut self, frame: &mut SwapchainFrame) {
    debug_assert!(!self.eligible_for_acquisition() && !frame.eligible_for_steal());
    self.submission_fence = frame.submission_fence.take();
    self.acquire_semaphore = frame.acquire_semaphore.take();
  }
}

impl SwapchainFrame {
  fn eligible_for_steal(&self) -> bool {
    self.submission_fence.is_none() && self.acquire_semaphore.is_none()
  }

  unsafe fn steal_from_swapchain_image(&mut self, swapchain_image: &mut SwapchainImage) {
    debug_assert!(swapchain_image.eligible_for_acquisition() && self.eligible_for_steal());
    self.acquire_semaphore = swapchain_image.acquire_semaphore.take();
    self.submission_fence = swapchain_image.submission_fence.take();
  }
}

/// TODO: Document this item
pub(super) struct WindowlessPresentationState {
  images: heapless::Vec<SwapchainImage, MAX_FRAMES>,
  memories: heapless::Vec<NonZeroHandle<vk::DeviceMemory>, MAX_FRAMES>,
  next_image: usize,
  frames: heapless::Vec<SwapchainFrame, MAX_FRAMES>,
  frame_discards: heapless::Vec<FrameDiscard, MAX_FRAMES>,
  current_frame: usize,

  width: u32,
  height: u32,
  format: vk::Format,

  generation: u64,
  pub submitted_frames: u64,
  pub last_timeline_value: core::sync::atomic::AtomicU64,
  memory_properties: vk::PhysicalDeviceMemoryProperties,
  pending_resize: Option<(u32, u32)>,
  buffer_count: u32,
  archetypes: crate::gpu::vulkan::device::archetypes_struct::Archetypes,
}

impl DeviceResource for WindowlessPresentationState {
  fn cleanup(&mut self, device: &ash::Device) {
    for discard in &mut self.frame_discards {
      discard.skip_cycles = 0; // force cleanup
      discard.cleanup_windowless(device);
    }
    for frame in &mut self.frames {
      if let Some(fence) = frame.submission_fence.take() {
        unsafe { device.destroy_fence(fence.get(), None) };
      }
      if let Some(sem) = frame.acquire_semaphore.take() {
        unsafe { device.destroy_semaphore(sem.get(), None) };
      }
    }
    for image in &mut self.images {
      image.cleanup(device);
      unsafe { device.destroy_image(image.image.get(), None) };
    }
    for mem in &mut self.memories {
      unsafe { device.free_memory(mem.get(), None) };
    }
  }
}

impl WindowlessPresentationState {
  /// TODO: Document this item
  pub fn new(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
    buffer_count: u32,
  ) -> GpuResult<Self> {
    let mut this = Self {
      images: heapless::Vec::new(),
      memories: heapless::Vec::new(),
      next_image: 0,
      frames: heapless::Vec::new(),
      frame_discards: heapless::Vec::new(),
      current_frame: 0,
      width,
      height,
      format: vk::Format::B8G8R8A8_UNORM,
      generation: 0,
      submitted_frames: 0,
      last_timeline_value: core::sync::atomic::AtomicU64::new(0),
      memory_properties: unsafe {
        instance.get_physical_device_memory_properties(physical_device.get())
      },
      pending_resize: None,
      buffer_count,
      archetypes: crate::gpu::vulkan::device::archetypes_struct::Archetypes::default(),
    };
    this.recreate(device, width, height)?;
    Ok(this)
  }

  /// TODO: Document this item
  #[named]
  pub fn cancel_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<()> {
    let image = &mut self.images[image_index as usize];
    let frame = &mut self.frames[frame_index as usize];

    if image.eligible_for_acquisition() || frame.eligible_for_steal() {
      return Ok(());
    }

    let fence = frame.submission_fence.ok_or(crate::gpu_err_device!())?.get();
    let submit_info = vk::SubmitInfo::default();
    unsafe {
      device
        .locked_queue_submit(graphics_queue, core::slice::from_ref(&submit_info), fence)
        .map_err(GpuError::from)?;
      image.reclaim_from_swapchain_frame(frame);
    }

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn resize(
    &mut self,
    _instance: &ash::Instance,
    _device: &ash::Device,
    _physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    if self.width != width || self.height != height {
      self.pending_resize = Some((width, height));
    }
    Ok(())
  }

  /// TODO: Document this item
  pub(super) fn extent(&self) -> (u32, u32) {
    (self.width, self.height)
  }

  /// TODO: Document this item
  pub(super) fn format(&self) -> vk::Format {
    self.format
  }

  /// TODO: Document this item
  pub(super) fn swapchain_generation(&self) -> u64 {
    self.generation
  }

  /// TODO: Document this item
  #[named]
  pub(super) fn for_each_swapchain_image(
    &self,
    mut f: impl FnMut(NonZeroHandle<vk::ImageView>) -> GpuResult<()>,
  ) -> GpuResult<()> {
    for image in &self.images {
      (&mut f)(image.image_view)?;
    }
    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn acquire_next_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
  ) -> GpuResult<AcquireResult> {
    if let Some((w, h)) = self.pending_resize.take() {
      self.width = w;
      self.height = h;
      self.recreate(device, w, h)?;
    }

    let images_count = self.images.len();
    let frame_count = self.frames.len();
    let swapchain_image = &mut self.images[self.next_image];
    if !swapchain_image.eligible_for_acquisition()
      || !self.frames[self.current_frame].eligible_for_steal()
    {
      return Err(crate::gpu_err!("windowless_acquire: not eligible"));
    }
    let fences: &[vk::Fence] = unsafe {
      core::slice::from_ref(&swapchain_image.submission_fence.as_ref().unwrap_unchecked())
    };

    let mut timeout = 167;
    loop {
      let result = unsafe { device.wait_for_fences(fences, false, timeout) };
      if let Err(vk_result) = result {
        if vk_result == vk::Result::TIMEOUT {
          timeout = u64::MAX;
        } else {
          return Err(vk_result.into());
        }
      } else {
        break;
      }
    }

    self.frame_discards[self.current_frame].cleanup_windowless(device);

    unsafe { device.reset_fences(fences) }?;

    unsafe { self.frames[self.current_frame].steal_from_swapchain_image(swapchain_image) };
    let frame_idx_for_submission = self.current_frame;
    let image_index = self.next_image as u32;

    self.next_image = (self.next_image + 1) % images_count;
    self.current_frame = (self.current_frame + 1) % frame_count;
    Ok(AcquireResult {
      image_index,
      status: SwapchainStatus::Optimal,
      frame_index: frame_idx_for_submission as u64,
      swapchain_generation: self.generation,
    })
  }

  /// TODO: Document this item
  pub unsafe fn get_frame_resources(
    &self,
    index: usize,
  ) -> (
    Option<NonZeroHandle<vk::Semaphore>>,
    NonZeroHandle<vk::Fence>,
  ) {
    debug_assert!(self.frames.len() > index);
    let frame = &self.frames[index];
    debug_assert!(!frame.eligible_for_steal());
    unsafe {
      (
        None, // Windowless does not use an acquire semaphore for submission wait
        frame.submission_fence.unwrap_unchecked(),
      )
    }
  }

  /// TODO: Document this item
  pub unsafe fn get_image_resources(
    &self,
    index: usize,
  ) -> (
    NonZeroHandle<vk::Image>,
    NonZeroHandle<vk::ImageView>,
    Option<NonZeroHandle<vk::Semaphore>>,
  ) {
    debug_assert!(self.images.len() > index);
    let image = &self.images[index];
    debug_assert!(!image.eligible_for_acquisition());
    (image.image, image.image_view, None)
  }

  /// TODO: Document this item
  #[named]
  pub unsafe fn submit_image(
    &mut self,
    _graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<SwapchainStatus> {
    let image = &mut self.images[image_index as usize];
    let frame = &mut self.frames[frame_index as usize];
    if image.eligible_for_acquisition() || frame.eligible_for_steal() {
      return Err(crate::gpu_err!("window_submit: still eligible"));
    }
    unsafe { image.reclaim_from_swapchain_frame(frame) };
    self.submitted_frames += 1;
    Ok(SwapchainStatus::Optimal)
  }

  /// TODO: Document this item
  #[named]
  pub fn get_last_submitted_image(&self) -> GpuResult<NonZeroHandle<vk::Image>> {
    if self.submitted_frames == 0 {
      return Err(crate::gpu_err!("get_last: no submissions"));
    }
    if self.images.is_empty() {
      return Err(crate::gpu_err!("get_last: no images"));
    }
    let last_index = (self.next_image + self.images.len() - 1) % self.images.len();
    Ok(self.images[last_index].image)
  }

  /// TODO: Document this item
  pub fn get_last_submitted_timeline_value(&self) -> u64 {
    self.last_timeline_value.load(core::sync::atomic::Ordering::Acquire)
  }

  #[named]
  fn recreate(&mut self, device: &ash::Device, width: u32, height: u32) -> GpuResult<()> {
    self.width = width;
    self.height = height;

    if self.frames.len() > self.current_frame {
      let prev_frame = (self.current_frame + self.frames.len() - 1) % self.frames.len();
      let frame_discard = &mut self.frame_discards[prev_frame];

      // Windowless uses strictly native pipelines, WSI grace delays are not required
      frame_discard.skip_cycles = 0;

      for frame in &mut self.frames {
        if let Some(fence) = frame.submission_fence.take() {
          let _ = frame_discard.discarded_fences.push(fence);
        }
        if let Some(sem) = frame.acquire_semaphore.take() {
          let _ = frame_discard.discarded_semaphores.push(sem);
        }
      }

      if !self.images.is_empty() {
        frame_discard.discard_swapchain_images(&mut self.images, true);
        self.images.clear();
      }

      for mem in &mut self.memories {
        let _ = frame_discard.discarded_memories.push(*mem);
      }
      self.memories.clear();
    }

    self.images.clear();
    self.memories.clear();
    self.frames.clear();

    self.submitted_frames = 0;

    let image_count = self.buffer_count as usize;
    let format = self.format;
    let extent = vk::Extent3D {
      width,
      height,
      depth: 1,
    };

    let image_info = vk::ImageCreateInfo::default()
      .image_type(vk::ImageType::TYPE_2D)
      .format(format)
      .extent(extent)
      .mip_levels(1)
      .array_layers(1)
      .samples(vk::SampleCountFlags::TYPE_1)
      .tiling(vk::ImageTiling::OPTIMAL)
      .usage(
        vk::ImageUsageFlags::COLOR_ATTACHMENT
          | vk::ImageUsageFlags::TRANSFER_SRC
          | vk::ImageUsageFlags::TRANSFER_DST,
      )
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .initial_layout(vk::ImageLayout::UNDEFINED);

    let img_view_create_info = vk::ImageViewCreateInfo::default()
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(format)
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_array_layer(0)
          .base_mip_level(0)
          .layer_count(1)
          .level_count(1),
      );

    let sem_create_info = vk::SemaphoreCreateInfo::default();
    let fence_create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

    let mem_props = self.memory_properties;

    for _ in 0..image_count {
      let image = unsafe { NonZeroHandle::new_unchecked(device.create_image(&image_info, None)?) };
      let mem_reqs = unsafe { device.get_image_memory_requirements(image.get()) };

      let mut mem_type_index = 0;
      let properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;
      for i in 0..mem_props.memory_type_count {
        if (mem_reqs.memory_type_bits & (1 << i)) != 0
          && (mem_props.memory_types[i as usize].property_flags & properties) == properties
        {
          mem_type_index = i;
          break;
        }
      }

      let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type_index);

      let memory =
        unsafe { NonZeroHandle::new_unchecked(device.allocate_memory(&alloc_info, None)?) };
      unsafe { device.bind_image_memory(image.get(), memory.get(), 0)? };

      let view_info = img_view_create_info.image(image.get());
      let image_view =
        unsafe { NonZeroHandle::new_unchecked(device.create_image_view(&view_info, None)?) };
      let present_semaphore =
        unsafe { NonZeroHandle::new_unchecked(device.create_semaphore(&sem_create_info, None)?) };

      unsafe {
        self.images.push_unchecked(SwapchainImage {
          image,
          image_view,
          submission_fence: None,
          acquire_semaphore: None,
          present_semaphore,
          present_fence: None, // strictly un-used in windowless pipelines
          present_fence_in_use: false,
        });
        self.memories.push_unchecked(memory);

        self.frames.push_unchecked(SwapchainFrame {
          submission_fence: None,
          acquire_semaphore: None,
        });

        while self.frame_discards.len() < image_count {
          self.frame_discards.push_unchecked(FrameDiscard::default());
        }
      }
    }

    for i in 0..image_count {
      let frame_fence =
        unsafe { NonZeroHandle::new_unchecked(device.create_fence(&fence_create_info, None)?) };
      let frame_sem =
        unsafe { NonZeroHandle::new_unchecked(device.create_semaphore(&sem_create_info, None)?) };
      self.images[i].acquire_semaphore = Some(frame_sem);
      self.images[i].submission_fence = Some(frame_fence);
    }

    if !self.images.is_empty() {
      self.next_image %= self.images.len()
    }
    if !self.frames.is_empty() {
      self.current_frame %= self.frames.len()
    }

    if self.frame_discards.len() > image_count {
      let new_len = image_count;
      for i in new_len..self.frame_discards.len() {
        let mut orphaned = core::mem::take(&mut self.frame_discards[i]);
        self.frame_discards[0].discarded_swapchains.append(&mut orphaned.discarded_swapchains);
        self.frame_discards[0].discarded_semaphores.append(&mut orphaned.discarded_semaphores);
        self.frame_discards[0].discarded_image_views.append(&mut orphaned.discarded_image_views);
        self.frame_discards[0].discarded_fences.append(&mut orphaned.discarded_fences);
        self.frame_discards[0].discarded_images.append(&mut orphaned.discarded_images);
        self.frame_discards[0].discarded_memories.append(&mut orphaned.discarded_memories);
      }
      self.frame_discards.truncate(new_len);
    }

    self.generation += 1;
    Ok(())
  }
}