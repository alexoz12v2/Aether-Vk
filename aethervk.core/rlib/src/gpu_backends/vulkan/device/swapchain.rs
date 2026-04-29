use core::{mem, ptr};

use alloc::{string::ToString, vec::Vec};
use ash::vk::{self, Handle};
use crate::{
  gpu::{AcquireResult, OpaqueNativeHandleInfo, PresentationEngineParams, SwapchainStatus},
  gpu_backends::vulkan::{device::DeviceResource, utils::NonZeroHandle, device::LogicalDevice},
  types::{GpuError, GpuResult},
};

pub(super) const MAX_FRAMES_IN_FLIGHT: usize = 8;

/// Handles and data relative to an image acquired through a swapchain
/// These data are ephimeral and are associated to a swapchain. Initially, there's
/// a one to one mapping. When swapchain is recreated, the current frame index
/// will be associated to image index 0
struct SwapchainImage {
  /// Image Handle. Automatically freed when the swapchain is freed
  pub image: NonZeroHandle<vk::Image>,
  /// Image View for the whole swapchain image
  pub image_view: NonZeroHandle<vk::ImageView>,
  /// fence which is signaled when render command is submitted. Owned by this
  /// structure. Once a frame is submitted, Ownership is transferred to associated
  /// associated SwapchainFrame. Starts at null. populated on first submission
  /// can be reused when recreating the swapchain
  pub submission_fence: Option<NonZeroHandle<vk::Fence>>,
  /// semaphore which is waited when submitting, signaled on acquire
  /// Once an image has been acquired, Ownership is transferred to associated
  /// SwapchainFrame. Starts at null. Populated on first submission
  /// associated to the image you acquire. cannot be reused
  pub acquire_semaphore: Option<NonZeroHandle<vk::Semaphore>>,
  /// semaphore which is waited when presenting, signaled when submitting
  pub present_semaphore: NonZeroHandle<vk::Semaphore>,
}

/// Mechanism to handle delayed (on next acquire attempt after N discards)
#[derive(Clone)]
struct FrameDiscard {
  discarded_swapchains: Vec<NonZeroHandle<vk::SwapchainKHR>>,
  discarded_semaphores: Vec<NonZeroHandle<vk::Semaphore>>,
  discarded_image_views: Vec<NonZeroHandle<vk::ImageView>>,

  // added
  discarded_fences: Vec<NonZeroHandle<vk::Fence>>,
}

#[derive(Clone)]
struct SwapchainFrame {
  /// fence which is signaled when submission operation has finished for current frame, waited on
  /// for image acquisition. Transfer of ownership happens when acquiring next image
  pub submission_fence: Option<NonZeroHandle<vk::Fence>>,
  /// semaphore which will be waited when submitting, signaled when transferred.
  /// Transfer of ownership happens at image acquisition
  pub acquire_semaphore: Option<NonZeroHandle<vk::Semaphore>>,
}

pub(super) enum PresentationState {
  Windowed(WindowedPresentationState),
  Windowless(WindowlessPresentationState),
}

pub(super) struct WindowedPresentationState {
  surface_instance: ash::khr::surface::Instance,
  surface_capabilities: ash::khr::get_surface_capabilities2::Instance,
  surface: NonZeroHandle<vk::SurfaceKHR>,

  swapchain_device: ash::khr::swapchain::Device,
  swapchain: NonZeroHandle<vk::SwapchainKHR>,

  images: heapless::Vec<SwapchainImage, MAX_FRAMES_IN_FLIGHT>,
  next_image: usize,
  frames: heapless::Vec<SwapchainFrame, MAX_FRAMES_IN_FLIGHT>,
  frame_discards: heapless::Vec<FrameDiscard, MAX_FRAMES_IN_FLIGHT>,
  current_frame: usize,

  width: u32,
  height: u32,
  pre_transform: vk::SurfaceTransformFlagsKHR,
  surface_format: vk::SurfaceFormatKHR,
  vsync: bool,
  native_handle: OpaqueNativeHandleInfo,

  swapchain_generation: u64,
}

trait SwapchainCleanable {
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
    unsafe { device.destroy_semaphore(self.present_semaphore.get(), None) };
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
  fn cleanup(&mut self, swapchain_device: &ash::khr::swapchain::Device, device: &ash::Device) {
    for &swapchain in self.discarded_swapchains.iter() {
      unsafe { swapchain_device.destroy_swapchain(swapchain.get(), None) };
    }
    self.discarded_swapchains.clear();
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
  }
}

impl SwapchainCleanable for SwapchainFrame {
  fn cleanup(&mut self, swapchain_device: &ash::khr::swapchain::Device, device: &ash::Device) {
    if let Some(fence) = self.submission_fence {
      unsafe { device.destroy_fence(fence.get(), None) };
    }
    if let Some(sem) = self.acquire_semaphore {
      unsafe { device.destroy_semaphore(sem.get(), None) };
    }
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
      frame_discard.cleanup(&self.swapchain_device, device);
    }
    for frame in &mut self.frames {
      frame.cleanup(&self.swapchain_device, device);
    }
    for image in &mut self.images {
      image.cleanup(device);
    }
    unsafe {
      self
        .swapchain_device
        .destroy_swapchain(self.swapchain.get(), None)
    };
    unsafe {
      self
        .surface_instance
        .destroy_surface(self.surface.get(), None)
    };
  }
}

unsafe impl Send for PresentationState {}
unsafe impl Sync for PresentationState {}

impl PresentationState {
  /// Assumes command buffer completed, but still not submitted to queue.
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

  pub(super) fn extent(&self) -> (u32, u32) {
    match self {
      Self::Windowed(state) => state.extent(),
      Self::Windowless(state) => state.extent(),
    }
  }

  pub(super) fn format(&self) -> vk::Format {
    match self {
      Self::Windowed(state) => state.format(),
      Self::Windowless(state) => state.format(),
    }
  }

  pub(super) fn swapchain_generation(&self) -> u64 {
    match self {
      Self::Windowed(state) => state.swapchain_generation(),
      Self::Windowless(state) => state.swapchain_generation(),
    }
  }

  pub(super) fn for_each_swapchain_image(
    &self,
    f: impl FnMut(NonZeroHandle<vk::ImageView>) -> GpuResult<()>,
  ) -> GpuResult<()> {
    match self {
      Self::Windowed(state) => state.for_each_swapchain_image(f),
      Self::Windowless(state) => state.for_each_swapchain_image(f),
    }
  }

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

  pub fn new(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    params: &PresentationEngineParams,
  ) -> GpuResult<Self> {
    match params.ty {
      crate::gpu::PresentationEngineType::Window => Ok(Self::Windowed(
        WindowedPresentationState::new(entry, instance, device, physical_device, params)?,
      )),
      crate::gpu::PresentationEngineType::WindowLess => {
        Ok(Self::Windowless(WindowlessPresentationState::new(
          instance,
          device,
          physical_device,
          params.width,
          params.height,
        )?))
      }
    }
  }
}

impl WindowedPresentationState {
  pub fn cancel_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<()> {
    let image = &mut self.images[image_index as usize];
    let frame = &mut self.frames[frame_index as usize];

    // If it hasn't been acquired (e.g., OUT_OF_DATE happened), do nothing
    if image.eligible_for_acquisition() || frame.eligible_for_steal() {
      return Ok(());
    }

    let acquire_sem = frame.acquire_semaphore.unwrap().get();
    let present_sem = image.present_semaphore.get();
    let fence = frame.submission_fence.unwrap().get();

    // 1. Dummy Submit (0 cmd buffers) to advance Semaphores and Fences
    let wait_semaphores = [acquire_sem];
    let wait_dst_stage_mask = [vk::PipelineStageFlags::BOTTOM_OF_PIPE];
    let signal_semaphores = [present_sem];

    let submit_info = vk::SubmitInfo::default()
      .wait_semaphores(&wait_semaphores)
      .wait_dst_stage_mask(&wait_dst_stage_mask)
      .signal_semaphores(&signal_semaphores);

    unsafe {
      device
        .locked_queue_submit(graphics_queue, core::slice::from_ref(&submit_info), fence)
        .map_err(GpuError::from)?;
    }

    // 2. Dummy Present to return WSI WSI image to Swapchain WSI
    let swapchains = [self.swapchain.get()];
    let image_indices = [image_index];
    let present_info = vk::PresentInfoKHR::default()
      .wait_semaphores(&signal_semaphores)
      .swapchains(&swapchains)
      .image_indices(&image_indices);

    let present_result = unsafe {
      let _guard = device.submission_lock.lock();
      self
        .swapchain_device
        .queue_present(graphics_queue, &present_info)
    };

    // 3. Reset internal struct states
    unsafe { image.reclaim_from_swapchain_frame(frame) };

    match present_result {
      Ok(_) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(()),
      Err(e) => Err(e.into()),
    }
  }

  pub(super) fn resize(
    &mut self,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    self.width = width;
    self.height = height;
    unsafe {
      let fences: Vec<_> = self
        .frames
        .iter()
        .filter_map(|f| f.submission_fence.map(|h| h.get()))
        .collect();
      if !fences.is_empty() {
        device.wait_for_fences(&fences, true, u64::MAX)?;
      }
    }
    self.recreate_swapchain(device, true, physical_device)
  }

  pub(super) fn extent(&self) -> (u32, u32) {
    (self.width, self.height)
  }

  pub(super) fn format(&self) -> vk::Format {
    self.surface_format.format
  }

  pub fn new(
    entry: &ash::Entry,
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    params: &PresentationEngineParams,
  ) -> GpuResult<Self> {
    let surface_instance = ash::khr::surface::Instance::new(entry, instance);
    let native_handle = params.window_info;
    let surface = Self::create_surface(&entry, &instance, native_handle)?;
    let swapchain_device = ash::khr::swapchain::Device::new(&instance, &device);
    let surface_capabilities =
      ash::khr::get_surface_capabilities2::Instance::new(&entry, &instance);
    let mut this = Self {
      surface_instance,
      surface_capabilities,
      surface: unsafe { NonZeroHandle::new_unchecked(surface) },
      swapchain_device,
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
    };

    this.recreate_swapchain(device, false, physical_device)?;

    Ok(this)
  }

  // Assumes that user resizing has finished
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
    // transaction like behaviour: everything that can fail happens before mutations
    // - BEGIN
    let (mut swapchain, extent, transform, surface_format, _) =
      self.create_swapchain_internal(physical_device, use_old_swapchain)?;
    let mut swapchain_images =
      self.recreate_swapchain_images(device, swapchain, surface_format.format)?;
    let (frame_semaphores, mut frame_fences) =
      self.recreate_swapchain_frame_resources(device, swapchain_images.len())?;
    // - END

    // bookkeping
    self.next_image = 0;
    mem::swap(&mut swapchain, &mut self.swapchain);
    self.update_metadata(extent, transform, surface_format);

    if self.frames.len() > self.current_frame {
      let frame_discard = &mut self.frame_discards[self.current_frame];
      // discard image resources
      if !self.images.is_empty() {
        frame_discard.discard_swapchain_images(&mut self.images);
        self.images.clear();
      }
      // discard decommissioned swapchain
      unsafe { frame_discard.discard_decommissioned_swapchain(swapchain) };
      // discard all frame acquire semaphores
      frame_discard.discard_swapchain_frame_keep_fences(&mut self.frames);
    }
    // refresh image and frame data
    self.images = swapchain_images;

    if self.images.len() > self.frames.len() {
      let _ = self.frames.resize_default(self.images.len());
      let _ = self.frame_discards.resize_default(self.images.len());
    } else if self.images.len() < self.frames.len() {
      // Clean up orphaned frames so we don't leak Vulkan handles
      for i in self.images.len()..self.frames.len() {
        let mut orphaned_frame = self.frames[i].clone();
        orphaned_frame.cleanup(&self.swapchain_device, device);
      }
      self.frames.truncate(self.images.len());
      self.frame_discards.truncate(self.images.len());
    }

    // Iterate up to the current image count
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

    debug_assert!(
      self.frame_discards.len() == self.frames.len()
        && self.frame_discards.len() >= self.images.len()
    );
    self.swapchain_generation += 1;
    Ok(())
  }

  pub(super) fn swapchain_generation(&self) -> u64 {
    self.swapchain_generation
  }

  fn recreate_swapchain_images(
    &self,
    device: &ash::Device,
    swapchain: NonZeroHandle<vk::SwapchainKHR>,
    format: vk::Format,
  ) -> GpuResult<heapless::Vec<SwapchainImage, MAX_FRAMES_IN_FLIGHT>> {
    let mut images = heapless::Vec::<vk::Image, MAX_FRAMES_IN_FLIGHT>::new();
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

    let mut result = heapless::Vec::<SwapchainImage, MAX_FRAMES_IN_FLIGHT>::new();
    for i in 0..count {
      let info = img_view_create_info.image(images[i as usize]);
      unsafe {
        let image_view = NonZeroHandle::new_unchecked(device.create_image_view(&info, None)?);
        let present_semaphore =
          NonZeroHandle::new_unchecked(device.create_semaphore(&sem_create_info, None)?);

        result.push_unchecked(SwapchainImage {
          image: NonZeroHandle::new_unchecked(*images.get_unchecked(i as usize)),
          image_view,
          submission_fence: None,
          acquire_semaphore: None,
          present_semaphore,
        })
      }
    }
    Ok(result)
  }

  /// Function specifically designed for renderpasses to recreate its framebuffers
  pub(super) fn for_each_swapchain_image(
    &self,
    mut f: impl FnMut(NonZeroHandle<vk::ImageView>) -> GpuResult<()>,
  ) -> GpuResult<()> {
    for image in &self.images {
      (&mut f)(image.image_view)?;
    }
    Ok(())
  }

  fn can_swapchain_image_be_transfer(surf_caps: &vk::SurfaceCapabilities2KHR) -> GpuResult<()> {
    let flags = surf_caps.surface_capabilities.supported_usage_flags;
    if flags.contains(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::TRANSFER_SRC) {
      Ok(())
    } else {
      Err(GpuError::UnsupportedFeature)
    }
  }

  // assumes you discarded all previous acquire semaphores
  fn recreate_swapchain_frame_resources(
    &self,
    device: &ash::Device,
    count: usize,
  ) -> GpuResult<(
    heapless::Vec<NonZeroHandle<vk::Semaphore>, MAX_FRAMES_IN_FLIGHT>,
    heapless::Vec<vk::Fence, MAX_FRAMES_IN_FLIGHT>,
  )> {
    let mut semaphores = heapless::Vec::<NonZeroHandle<vk::Semaphore>, MAX_FRAMES_IN_FLIGHT>::new();
    let mut fences = heapless::Vec::<vk::Fence, MAX_FRAMES_IN_FLIGHT>::new();
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
          // new frame: create fence
          device.create_fence(&fence_create_info, None)?
        });
        semaphores.push_unchecked(NonZeroHandle::new_unchecked(
          device.create_semaphore(&sem_create_info, None)?,
        ));
      }
    }

    Ok((semaphores, fences))
  }

  // note: immutable self
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
    // Surface capabilities
    let surface_format = self.select_surface_format(physical_device)?;
    let present_mode = self.select_present_mode(physical_device)?;

    let mut present_mode_ext = vk::SurfacePresentModeEXT::default().present_mode(present_mode);
    let surf_info = vk::PhysicalDeviceSurfaceInfo2KHR::default()
      .surface(self.surface.get())
      .push_next(&mut present_mode_ext);
    let mut surf_caps = vk::SurfaceCapabilities2KHR::default();
    unsafe {
      self
        .surface_capabilities
        .get_physical_device_surface_capabilities2(
          physical_device.get(),
          &surf_info,
          &mut surf_caps,
        )
    }?;
    let (extent, transform, image_count) =
      self.extent_transform_imagecount(&surf_caps, present_mode);
    let composite_alpha = Self::get_supported_composite_alpha(&surf_caps)?;

    Self::can_swapchain_image_be_transfer(&surf_caps)?;

    // create swapchain
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
      .clipped(true) // pixels can be written from OS outside vulkan. Unless you read back this image, we don't care
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

  /// 1. pCreateInfo->compositeAlpha contains multiple members of VkCompositeAlphaFlagBitsKHR when only a single value is allowed
  /// 2. compositeAlpha must be one of the bits present in the supportedCompositeAlpha member of the
  /// VkSurfaceCapabilitiesKHR structure returned by vkGetPhysicalDeviceSurfaceCapabilitiesKHR for the surface
  fn get_supported_composite_alpha(
    surf_caps: &vk::SurfaceCapabilities2KHR,
  ) -> GpuResult<vk::CompositeAlphaFlagsKHR> {
    let supported = surf_caps.surface_capabilities.supported_composite_alpha;

    // 1. Pre-Multiplied: The gold standard for desktop compositing.
    // Best for macOS Vibrancy, Windows Mica/Acrylic, and Wayland transparency.
    if supported.contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED) {
      Ok(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
    }
    // 2. Post-Multiplied: A solid fallback if Pre-Multiplied isn't supported,
    // though you'll need to ensure your shaders output straight alpha.
    else if supported.contains(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED) {
      Ok(vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED)
    }
    // 3. Inherit: Defers entirely to the OS window manager.
    // Common requirement on Android, and sometimes older X11 setups.
    else if supported.contains(vk::CompositeAlphaFlagsKHR::INHERIT) {
      Ok(vk::CompositeAlphaFlagsKHR::INHERIT)
    }
    // 4. Opaque: The ultimate fallback. No desktop transparency for this window.
    else if supported.contains(vk::CompositeAlphaFlagsKHR::OPAQUE) {
      Ok(vk::CompositeAlphaFlagsKHR::OPAQUE)
    }
    // 5. Unreachable in a spec-compliant Vulkan implementation, but handled for safety.
    else {
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
      // start with 3 if vsync, 2 if not. then check against min, which is >= 1 by spec, and max,
      // Note: if max is 0, then there's no limit
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
      // supported by specification
      Ok(vk::PresentModeKHR::FIFO)
    }
  }

  fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    native_handle: OpaqueNativeHandleInfo,
  ) -> ash::prelude::VkResult<vk::SurfaceKHR> {
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

  /// Note:
  /// - When it returns Ok(AcquireResult) with SwapchainStatus::Optimal -> increments `next_image` when successful and makes frame at `current_frame` eligible for submission
  /// - AcquireResult::image_available_semaphore is not used. State tracked internally by presentation engine
  pub fn acquire_next_image(&mut self, device: &LogicalDevice) -> GpuResult<AcquireResult> {
    const FIRST_ATTEMPT_TIMEOUT_NS: u64 = 167;
    let images_count = self.images.len();
    let frame_count = self.frames.len();
    let swapchain_image = &mut self.images[self.next_image];
    if !swapchain_image.eligible_for_acquisition()
      || !self.frames[self.current_frame].eligible_for_steal()
    {
      return Err(GpuError::InvalidState("swapchain.rs:866"));
    }
    let fences: &[vk::Fence] = unsafe {
      core::slice::from_ref(&swapchain_image.submission_fence.as_ref().unwrap_unchecked())
    };

    // 1. Wait for previous rendering operation on the same frame
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

    // Clean up discarded resources for this frame, since its previous submission is now fully complete
    self.frame_discards[self.current_frame].cleanup(&self.swapchain_device, device);

    // 2. Reset submission hence
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

    match vk_result {
      vk::Result::SUCCESS | vk::Result::SUBOPTIMAL_KHR => {
        let actual_image = &mut self.images[image_index as usize];
        unsafe { self.frames[self.current_frame].steal_from_swapchain_image(actual_image) };
        let frame_idx_for_submission = self.current_frame;
        // this is the only arm in which status is changed
        self.next_image = (self.next_image + 1) % images_count;
        self.current_frame = (self.current_frame + 1) % frame_count;
        Ok(AcquireResult {
          image_index,
          status: SwapchainStatus::Optimal, // Must report Optimal so render loop consumes the signaled semaphore!
          frame_index: frame_idx_for_submission as u64,
        })
      }
      vk::Result::ERROR_OUT_OF_DATE_KHR => Ok(AcquireResult {
        image_index,
        status: SwapchainStatus::NeedsRecreation,
        frame_index: self.current_frame as u64,
      }),
      _ => {
        return Err(vk_result.into());
      }
    }
  }

  /// get SwapchainFrame synchronization resources at frames[index]
  /// Notes:
  /// - `submission_fence`: signaled in a vkQueueSubmit operation
  /// - `acquire_semaphore`: waited in a vkQueueSubmit operation, signaled in a vkAcquireNextImageKHR operation
  /// Safety:
  /// - index < frames.len()
  /// - `swapchain_frame` at index should be !eligible_for_steal
  /// - returned handles should not be freed by caller
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

  /// get SwapchainImage synchronization and image handles at images[index]
  /// Notes:
  /// - `image_view`: used as output color attachment for a subpass
  /// - `present_semaphore`: signaled in a vkQueueSubmit operation, waited in a vkQueuePresentKHR operation
  /// Safety:
  /// - index < images.len()
  /// - `swapchain_image` at index should be !eligible_for_acquisition
  /// - returned handles should not be freed by caller
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

  /// Safety
  /// - `image_index` and `frame_index` should be obtained by `acquire_next_image` without any `recreate_swapchain` or `submit_image` call in between
  /// - handles acquired from `get_image_resources` and `get_frame_resources` shouldn't be used after calling this function, regardless of the result
  /// - `graphics_queue` should be from a GRAPHICS queue family which supports presentation
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
      return Err(GpuError::InvalidState("swapchain.rs:989"));
    }

    let wait_semaphores = [image.present_semaphore.get()];
    let swapchains = [self.swapchain.get()];
    let image_indices = [image_index];

    let present_info = vk::PresentInfoKHR::default()
      .wait_semaphores(&wait_semaphores)
      .swapchains(&swapchains)
      .image_indices(&image_indices);

    let result = {
      let _guard = device.submission_lock.lock();
      unsafe {
        self
          .swapchain_device
          .queue_present(graphics_queue, &present_info)
      }
    };

    match result {
      Ok(suboptimal) if suboptimal => Ok(SwapchainStatus::Suboptimal),
      Ok(_) => {
        unsafe { image.reclaim_from_swapchain_frame(frame) };
        Ok(SwapchainStatus::Optimal)
      }
      Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(SwapchainStatus::NeedsRecreation),
      Err(e) => Err(e.into()),
    }
  }
}

impl FrameDiscard {
  fn discard_swapchain_images(&mut self, swapchain_images: &mut [SwapchainImage]) {
    for swapchain_image in swapchain_images {
      unsafe {
        self.discarded_image_views.push(swapchain_image.image_view);
        self
          .discarded_semaphores
          .push(swapchain_image.present_semaphore);
        debug_assert!(
          !(swapchain_image.submission_fence.is_some()
            ^ swapchain_image.acquire_semaphore.is_some())
        );
        if swapchain_image.eligible_for_acquisition() {
          // there must be only one frame and one frame only which is not eligible
          // for steal. That frame is the one associated with the current image
          self
            .discarded_semaphores
            .push(swapchain_image.acquire_semaphore.unwrap_unchecked());
          self
            .discarded_fences
            .push(swapchain_image.submission_fence.unwrap_unchecked());
        }
      };
    }
  }

  fn discard_swapchain_frame_keep_fences(&mut self, swapchain_frames: &mut [SwapchainFrame]) {
    for swapchain_frame in swapchain_frames {
      unsafe {
        debug_assert!(
          !(swapchain_frame.acquire_semaphore.is_some()
            ^ swapchain_frame.submission_fence.is_some())
        );

        if let Some(acquire_semaphore) = swapchain_frame.acquire_semaphore.take() {
          self.discarded_semaphores.push(acquire_semaphore);
        }
      }
    }
  }

  unsafe fn discard_decommissioned_swapchain(
    &mut self,
    swapchain: NonZeroHandle<vk::SwapchainKHR>,
  ) {
    self.discarded_swapchains.push(swapchain);
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
    }
  }
}

impl SwapchainImage {
  fn eligible_for_acquisition(&self) -> bool {
    self.submission_fence.is_some() && self.acquire_semaphore.is_some()
  }

  /// Safety:
  /// - swapchain_image should not be eligible for image acquisition
  /// - vkQueuePresentKHR should have been already called and it should have returned SUCCESS or SUBOPTIMAL
  /// - swapchain_frame should not be eligible for steal
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

  /// Safety:
  /// - swapchain_image should be eligible for image acquisition
  /// - vkAcquireNextImageKHR should have been already called
  /// - swapchain_frame should be eligible for steal
  unsafe fn steal_from_swapchain_image(&mut self, swapchain_image: &mut SwapchainImage) {
    debug_assert!(swapchain_image.eligible_for_acquisition() && self.eligible_for_steal());
    self.acquire_semaphore = swapchain_image.acquire_semaphore.take();
    self.submission_fence = swapchain_image.submission_fence.take();
  }
}

pub(super) struct WindowlessPresentationState {
  images: heapless::Vec<SwapchainImage, MAX_FRAMES_IN_FLIGHT>,
  memories: heapless::Vec<NonZeroHandle<vk::DeviceMemory>, MAX_FRAMES_IN_FLIGHT>,
  next_image: usize,
  frames: heapless::Vec<SwapchainFrame, MAX_FRAMES_IN_FLIGHT>,
  current_frame: usize,

  width: u32,
  height: u32,
  format: vk::Format,

  generation: u64,
  pub submitted_frames: u64,
  pub last_timeline_value: core::sync::atomic::AtomicU64,
}

impl DeviceResource for WindowlessPresentationState {
  fn cleanup(&mut self, device: &ash::Device) {
    for frame in &mut self.frames {
      if let Some(fence) = frame.submission_fence {
        unsafe { device.destroy_fence(fence.get(), None) };
      }
      if let Some(sem) = frame.acquire_semaphore {
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
  pub fn new(
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
  ) -> GpuResult<Self> {
    let mut this = Self {
      images: heapless::Vec::new(),
      memories: heapless::Vec::new(),
      next_image: 0,
      frames: heapless::Vec::new(),
      current_frame: 0,
      width,
      height,
      format: vk::Format::B8G8R8A8_UNORM,
      generation: 0,
      submitted_frames: 0,
      last_timeline_value: core::sync::atomic::AtomicU64::new(0),
    };
    this.recreate(instance, device, physical_device, width, height)?;
    Ok(this)
  }

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

    let acquire_sem = frame.acquire_semaphore.unwrap().get();
    let fence = frame.submission_fence.unwrap().get();

    // Windowless doesn't present, but still needs to satisfy the semaphores/fence
    let wait_semaphores = [acquire_sem];
    let wait_dst_stage_mask = [vk::PipelineStageFlags::BOTTOM_OF_PIPE];

    let submit_info = vk::SubmitInfo::default()
      .wait_semaphores(&wait_semaphores)
      .wait_dst_stage_mask(&wait_dst_stage_mask);

    unsafe {
      device
        .locked_queue_submit(graphics_queue, core::slice::from_ref(&submit_info), fence)
        .map_err(GpuError::from)?;
      image.reclaim_from_swapchain_frame(frame);
    }

    Ok(())
  }

  pub(super) fn resize(
    &mut self,
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    self.recreate(instance, device, physical_device, width, height)
  }

  pub(super) fn extent(&self) -> (u32, u32) {
    (self.width, self.height)
  }

  pub(super) fn format(&self) -> vk::Format {
    self.format
  }

  pub(super) fn swapchain_generation(&self) -> u64 {
    self.generation
  }

  pub(super) fn for_each_swapchain_image(
    &self,
    mut f: impl FnMut(NonZeroHandle<vk::ImageView>) -> GpuResult<()>,
  ) -> GpuResult<()> {
    for image in &self.images {
      (&mut f)(image.image_view)?;
    }
    Ok(())
  }

  pub fn acquire_next_image(
    &mut self,
    device: &LogicalDevice,
    graphics_queue: vk::Queue,
  ) -> GpuResult<AcquireResult> {
    let images_count = self.images.len();
    let frame_count = self.frames.len();
    let swapchain_image = &mut self.images[self.next_image];
    if !swapchain_image.eligible_for_acquisition()
      || !self.frames[self.current_frame].eligible_for_steal()
    {
      return Err(GpuError::InvalidState("swapchain.rs:1222"));
    }
    if swapchain_image.submission_fence.is_none() {
      unsafe { swapchain_image.reclaim_from_swapchain_frame(&mut self.frames[self.current_frame]) };
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

    unsafe { device.reset_fences(fences) }?;

    unsafe { self.frames[self.current_frame].steal_from_swapchain_image(swapchain_image) };
    let frame_idx_for_submission = self.current_frame;
    let image_index = self.next_image as u32;

    // Signal the acquire semaphore manually since there's no vkAcquireNextImageKHR
    // We can just submit an empty batch to the queue.
    let sem_handle = unsafe {
      self.frames[self.current_frame]
        .acquire_semaphore
        .as_ref()
        .unwrap_unchecked()
        .get()
    };
    let signal_info =
      vk::SubmitInfo::default().signal_semaphores(core::slice::from_ref(&sem_handle));
    device
      .locked_queue_submit(graphics_queue, &[signal_info], vk::Fence::null())
      .map_err(GpuError::from)?;

    self.next_image = (self.next_image + 1) % images_count;
    self.current_frame = (self.current_frame + 1) % frame_count;
    Ok(AcquireResult {
      image_index,
      status: SwapchainStatus::Optimal,
      frame_index: frame_idx_for_submission as u64,
    })
  }

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
    unsafe { (Some(frame.acquire_semaphore.unwrap_unchecked()), frame.submission_fence.unwrap_unchecked()) }
  }

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

  pub unsafe fn submit_image(
    &mut self,
    _graphics_queue: vk::Queue,
    image_index: u32,
    frame_index: u32,
  ) -> GpuResult<SwapchainStatus> {
    let image = &mut self.images[image_index as usize];
    let frame = &mut self.frames[frame_index as usize];
    if image.eligible_for_acquisition() || frame.eligible_for_steal() {
      return Err(GpuError::InvalidState("swapchain.rs:1308"));
    }
    unsafe { image.reclaim_from_swapchain_frame(frame) };
    self.submitted_frames += 1;
    Ok(SwapchainStatus::Optimal)
  }

  pub fn get_last_submitted_image(&self) -> GpuResult<NonZeroHandle<vk::Image>> {
    if self.submitted_frames == 0 {
      return Err(GpuError::InvalidState("swapchain.rs:1317"));
    }
    if self.images.is_empty() {
      return Err(GpuError::InvalidState("swapchain.rs:1320"));
    }
    let last_index = (self.next_image + self.images.len() - 1) % self.images.len();
    Ok(self.images[last_index].image)
  }

  pub fn get_last_submitted_timeline_value(&self) -> u64 {
    self
      .last_timeline_value
      .load(core::sync::atomic::Ordering::Acquire)
  }

  fn recreate(
    &mut self,
    instance: &ash::Instance,
    device: &ash::Device,
    physical_device: NonZeroHandle<vk::PhysicalDevice>,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    self.width = width;
    self.height = height;

    self.cleanup(device);
    self.images.clear();
    self.memories.clear();
    self.frames.clear();

    self.next_image = 0;
    self.current_frame = 0;

    let image_count = 3;
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

    let mem_props =
      unsafe { instance.get_physical_device_memory_properties(physical_device.get()) };

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
        });
        self.memories.push_unchecked(memory);

        self.frames.push_unchecked(SwapchainFrame {
          submission_fence: None,
          acquire_semaphore: None,
        });
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

    self.generation += 1;
    Ok(())
  }
}
