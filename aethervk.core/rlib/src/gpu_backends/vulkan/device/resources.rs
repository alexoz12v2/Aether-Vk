//! resources module.

use crate::{
  gpu::{self, PipelineKey, TextureFlags},
  gpu_backends::vulkan::{
    device::{
      self, DeviceResource, LogicalDevice, VmaDebugNameExt, VulkanDebugNameExt,
      commands::{self, CommandBufferId},
      descriptors::{self, DescriptorPools},
      locks::DebugTrackedRwLock,
      pipelines::GraphicsInfo,
    },
    utils::{self, NonZeroHandle, RwLockable},
  },
  simulation::comet::Texture,
  types::{GpuError, GpuResult},
};
use aethervk_oshal_rlib::{hash::FnvHasher, os::native::ThreadId};
use alloc::{
  boxed::Box,
  collections::{BTreeMap, VecDeque},
  sync,
  vec::Vec,
};
use ash::{vk, vk::Handle};
use core::{
  hash::{Hash, Hasher},
  ptr,
  sync::atomic::AtomicU32,
};
use function_name::named;
use spirv_reflect::{
  ffi::SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS, types::ReflectShaderStageFlags,
};
use vk_mem::{Alloc, AsAllocatorView};

/// Enum Type so that whenever a resource is created or updated in a [`dashmap::DashMap`] of the
/// vulkan device state, it can be transitioned to `Pending` state in a thread safe way and other
/// threads can wait to see the next `Ready` state value
#[derive(Clone, Debug)]
pub enum ResourceState<T> {
  Pending,
  Ready(T),
}

pub(crate) struct ArenaCreationContext<'a, 'b: 'a> {
  pub device: &'a LogicalDevice,
  pub allocator: vk_mem::AllocatorView,
  pub discard_pool: &'a DiscardPool,
  pub queue: Option<&'a device::Queue>,
  pub staging_arena: Option<&'a device::memory::FrameStagingArena>,
  pub vertex_shader: Option<vk::ShaderModule>,
  pub fragment_shader: Option<vk::ShaderModule>,
  pub outline_vertex_shader: Option<vk::ShaderModule>,
  pub outline_fragment_shader: Option<vk::ShaderModule>,
  pub rollback: &'a mut utils::RollbackContext<'b>,
}

impl<'a, 'b: 'a> ArenaCreationContext<'a, 'b> {
  pub fn new_empty(
    device: &'a LogicalDevice,
    allocator: vk_mem::AllocatorView,
    discard_pool: &'a DiscardPool,
    rollback: &'a mut utils::RollbackContext<'b>,
  ) -> Self {
    Self {
      device,
      allocator,
      discard_pool,
      queue: None,
      staging_arena: None,
      vertex_shader: None,
      fragment_shader: None,
      outline_vertex_shader: None,
      outline_fragment_shader: None,
      rollback,
    }
  }
}

pub trait ArchetypeArenaCreate {
  fn new_arena(ctx: &mut ArenaCreationContext) -> crate::types::GpuResult<Self>
  where
    Self: Sized;
}

pub struct TimelineQueue<T> {
  items: VecDeque<(u64, T)>,
}

impl<T> TimelineQueue<T> {
  pub fn with_capacity(cap: usize) -> Self {
    Self {
      items: VecDeque::with_capacity(cap),
    }
  }

  pub fn push(&mut self, timeline: u64, item: T) {
    let mut i = self.items.len();
    while i > 0 && self.items[i - 1].0 > timeline {
      i -= 1;
    }
    self.items.insert(i, (timeline, item));
  }

  pub fn drain_ready<F>(&mut self, current: u64, mut f: F)
  where
    F: FnMut(T),
  {
    while let Some((t, _)) = self.items.front() {
      if *t > current {
        break;
      }

      let (_, item) = self.items.pop_front().unwrap();
      f(item);
    }
  }
}

pub(crate) enum DiscardItem {
  Buffer(BufferDiscard),
  Image(ImageDiscard),
  ImageView(vk::ImageView),
  Pipeline(vk::Pipeline),
  PipelineLayout(vk::PipelineLayout),
  DescriptorSetLayout(vk::DescriptorSetLayout),
  DescriptorPool(vk::DescriptorPool, sync::Arc<descriptors::DescriptorPools>),
  CommandPool(CmdBufDiscard),
  RenderPass(vk::RenderPass),
  Framebuffer(vk::Framebuffer),
  Fence(vk::Fence),
  Semaphore(vk::Semaphore),
  GenericHandle(Box<dyn DeviceResource + Send + Sync + 'static>),
}

#[cfg(debug_assertions)]
impl DiscardItem {
  fn unique_id(&self) -> (u8, u64) {
    use ash::vk::Handle;
    match self {
      Self::Buffer(b) => (0, b.buffer.as_raw()),
      Self::Image(i) => (1, i.image.as_raw()),
      Self::ImageView(v) => (2, v.as_raw()),
      Self::Pipeline(p) => (3, p.as_raw()),
      Self::PipelineLayout(l) => (4, l.as_raw()),
      Self::DescriptorSetLayout(l) => (5, l.as_raw()),
      Self::DescriptorPool(p, _) => (6, p.as_raw()),
      Self::CommandPool(c) => (7, c.command_buffer.as_raw()),
      Self::RenderPass(r) => (8, r.as_raw()),
      Self::Framebuffer(f) => (9, f.as_raw()),
      Self::Fence(f) => (11, f.as_raw()),
      Self::Semaphore(s) => (12, s.as_raw()),
      Self::GenericHandle(h) => {
        let ptr: *const dyn DeviceResource = &**h;
        (10, ptr as *const () as u64)
      }
    }
  }
}

pub(crate) struct BufferDiscard {
  buffer: vk::Buffer,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::AllocatorView,
}
pub(crate) struct ImageDiscard {
  image: vk::Image,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::AllocatorView,
}
pub(crate) struct CmdBufDiscard {
  thread_id: ThreadId,
  command_buffer: vk::CommandBuffer,
  manager: sync::Arc<commands::CommandPools>,
  id: CommandBufferId,
  queue_family_index: u32,
}

/// Structure associated to the main Timeline Semaphore provided by Device
/// Note: this must not outlive device, hence don't expose it outside
pub struct DiscardPool {
  items: crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex<TimelineQueue<DiscardItem>>,
  #[cfg(debug_assertions)]
  queued_handles:
    crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex<hashbrown::HashSet<(u8, u64)>>,
}

impl DiscardPool {
  /// Safety: device and allocator should outlive Self
  pub unsafe fn new(cap: usize) -> Self {
    Self {
      items: crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex::new(
        TimelineQueue::with_capacity(cap),
      ),
      #[cfg(debug_assertions)]
      queued_handles: crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex::new(
        hashbrown::HashSet::with_capacity(cap),
      ),
    }
  }

  fn push_item(&self, timeline: u64, item: DiscardItem) {
    let mut q = crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex::lock(&self.items);
    #[cfg(debug_assertions)]
    {
      let handle = item.unique_id();
      assert!(
        crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex::lock(&self.queued_handles)
          .insert(handle),
        "Resource discarded twice! Type: {}, Handle: {}",
        handle.0,
        handle.1
      );
    }
    q.push(timeline, item);
  }

  pub fn discard_type_erased<T: DeviceResource + Send + Sync + 'static>(
    &self,
    item: T,
    timeline: u64,
  ) {
    aethervk_oshal_rlib::log!("Queuing type_erased discard at timeline {}", timeline);
    self.push_item(timeline, DiscardItem::GenericHandle(Box::new(item)));
  }
  pub fn discard_render_pass(&self, render_pass: vk::RenderPass, timeline: u64) {
    self.push_item(timeline, DiscardItem::RenderPass(render_pass));
  }
  pub fn discard_framebuffer(&self, framebuffer: vk::Framebuffer, timeline: u64) {
    self.push_item(timeline, DiscardItem::Framebuffer(framebuffer));
  }
  pub fn discard_fence(&self, fence: vk::Fence, timeline: u64) {
    self.push_item(timeline, DiscardItem::Fence(fence));
  }
  pub fn discard_semaphore(&self, semaphore: vk::Semaphore, timeline: u64) {
    self.push_item(timeline, DiscardItem::Semaphore(semaphore));
  }
  pub fn discard_buffer(
    &self,
    allocator: vk_mem::AllocatorView,
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    timeline: u64,
  ) {
    self.push_item(
      timeline,
      DiscardItem::Buffer(BufferDiscard {
        buffer,
        alloc,
        allocator,
      }),
    );
  }
  pub fn discard_command_buffer(
    &self,
    thread_id: ThreadId,
    command_buffer_id: CommandBufferId,
    command_buffer: vk::CommandBuffer,
    queue_family_index: u32,
    manager: sync::Arc<commands::CommandPools>,
    timeline: u64,
  ) {
    debug_assert!(sync::Arc::strong_count(&manager) > 1);
    self.push_item(
      timeline,
      DiscardItem::CommandPool(CmdBufDiscard {
        thread_id,
        command_buffer,
        manager,
        id: command_buffer_id,
        queue_family_index,
      }),
    );
  }
  pub fn discard_image(
    &self,
    allocator: vk_mem::AllocatorView,
    image: vk::Image,
    alloc: vk_mem::Allocation,
    timeline: u64,
  ) {
    self.push_item(
      timeline,
      DiscardItem::Image(ImageDiscard {
        image,
        alloc,
        allocator,
      }),
    );
  }
  pub fn discard_image_view(&self, image_view: vk::ImageView, timeline: u64) {
    self.push_item(timeline, DiscardItem::ImageView(image_view));
  }
  pub fn discard_descriptor_set_layout(&self, layout: vk::DescriptorSetLayout, timeline: u64) {
    self.push_item(timeline, DiscardItem::DescriptorSetLayout(layout));
  }
  pub fn discard_descriptor_pool(
    &self,
    pool: vk::DescriptorPool,
    manager: sync::Arc<descriptors::DescriptorPools>,
    timeline: u64,
  ) {
    debug_assert!(sync::Arc::strong_count(&manager) > 1);
    self.push_item(timeline, DiscardItem::DescriptorPool(pool, manager));
  }
  pub fn discard_pipeline(&self, pipeline: vk::Pipeline, timeline: u64) {
    self.push_item(timeline, DiscardItem::Pipeline(pipeline));
  }
  pub fn discard_pipeline_layout(&self, pipeline_layout: vk::PipelineLayout, timeline: u64) {
    self.push_item(timeline, DiscardItem::PipelineLayout(pipeline_layout));
  }

  /// Extracts all ready items from the pool (Requires brief lock). First part of vulkan transaction
  pub fn pop_ready_items(&self, timeline: u64) -> Vec<DiscardItem> {
    let mut items =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex::lock(&self.items);
    #[cfg(debug_assertions)]
    let mut queued_handles =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex::lock(&self.queued_handles);

    let mut ready = Vec::new();
    items.drain_ready(timeline, |item| {
      #[cfg(debug_assertions)]
      {
        queued_handles.remove(&item.unique_id());
      }
      ready.push(item);
    });
    ready
  }

  /// Executes Vulkan API destruction completely lock-free
  pub fn destroy_items_lock_free(
    device: &super::LogicalDevice,
    items: impl IntoIterator<Item = DiscardItem>,
  ) {
    for item in items {
      match item {
        DiscardItem::Buffer(BufferDiscard {
          buffer,
          mut alloc,
          allocator,
        }) => unsafe {
          aethervk_oshal_rlib::log!(
            "DiscardItem::Buffer destroying buffer! alloc: {:?}",
            alloc.get_raw()
          );
          allocator.destroy_buffer(buffer, &mut alloc);
        },
        DiscardItem::Image(ImageDiscard {
          image,
          mut alloc,
          allocator,
        }) => unsafe {
          allocator.destroy_image(image, &mut alloc);
        },
        DiscardItem::Pipeline(pipeline) => {
          unsafe { device.destroy_pipeline(pipeline, None) };
        }
        DiscardItem::PipelineLayout(pipeline_layout) => {
          unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
        }
        DiscardItem::DescriptorSetLayout(layout) => {
          unsafe { device.destroy_descriptor_set_layout(layout, None) };
        }
        DiscardItem::DescriptorPool(pool, manager) => {
          // return the pool to the manager for recycling
          manager.recycle(device, pool);
        }
        DiscardItem::CommandPool(CmdBufDiscard {
          thread_id,
          command_buffer,
          manager,
          id,
          queue_family_index,
        }) => {
          let _x = manager.recycle(device, thread_id, queue_family_index, command_buffer);
          #[cfg(debug_assertions)]
          {
            if let Err(ref e) = _x {
              panic!(
                "command pool recycle failed: tid={:?} id={:?} err={:?}",
                thread_id, id, e
              );
            }
          }
        }
        DiscardItem::ImageView(image_view) => unsafe {
          device.destroy_image_view(image_view, None);
        },
        DiscardItem::RenderPass(render_pass) => unsafe {
          device.destroy_render_pass(render_pass, None);
        },
        DiscardItem::Framebuffer(framebuffer) => unsafe {
          device.destroy_framebuffer(framebuffer, None);
        },
        DiscardItem::Fence(fence) => unsafe {
          device.destroy_fence(fence, None);
        },
        DiscardItem::Semaphore(semaphore) => unsafe {
          device.destroy_semaphore(semaphore, None);
        },
        DiscardItem::GenericHandle(mut handle) => {
          aethervk_oshal_rlib::log!("Destroying GenericHandle");
          handle.cleanup(device);
        }
      }
    }
  }

  /// Used by device cleanup routines
  pub fn destroy_discarded_resources_all(&self, device: &super::LogicalDevice) {
    let items = self.pop_ready_items(u64::MAX);
    aethervk_oshal_rlib::log!(
      "destroy_discarded_resources_all popping {} items",
      items.len()
    );
    Self::destroy_items_lock_free(device, items);
  }
}

impl super::DeviceResource for DiscardPool {
  fn cleanup(&mut self, device: &super::LogicalDevice) {
    self.destroy_discarded_resources_all(device);
  }
}

#[derive(Clone)]
pub(super) struct Buffer {
  pub buffer: NonZeroHandle<vk::Buffer>,
  pub allocation: vk_mem::Allocation,
}

impl Hash for Buffer {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.buffer.hash(state);
  }
}

#[derive(Clone)]
pub(super) struct Image {
  pub image: NonZeroHandle<vk::Image>,
  pub image_view: NonZeroHandle<vk::ImageView>,
  pub allocation: vk_mem::Allocation,
}

impl Image {
  pub fn to_descriptor_image_info(
    &self,
    sampler: NonZeroHandle<vk::Sampler>,
  ) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
      .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
      .image_view(self.image_view.get())
      .sampler(sampler.get())
  }

  #[named]
  pub fn new_storage_2d(
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    width: u32,
    height: u32,
    format: vk::Format,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let image_info = vk::ImageCreateInfo::default()
      .image_type(vk::ImageType::TYPE_2D)
      .extent(vk::Extent3D {
        width,
        height,
        depth: 1,
      })
      .mip_levels(1)
      .array_layers(1)
      .format(format)
      .tiling(vk::ImageTiling::OPTIMAL)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .samples(vk::SampleCountFlags::TYPE_1);

    let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(allocation_create_info);
    allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;

    let (image, alloc, _alloc_info) =
      unsafe { allocator.create_image_with_alloc_info(&image_info, &allocation_create_info) }
        .with_name(device, &alloc::format!("VkImage_Storage_2D_{}", debug_name))?;

    let image_view_info = vk::ImageViewCreateInfo::default()
      .image(image)
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(format)
      .components(vk::ComponentMapping::default())
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_array_layer(0)
          .layer_count(1)
          .base_mip_level(0)
          .level_count(1),
      );
    let image_view = unsafe {
      let res = device.create_image_view(&image_view_info, None).with_name(
        device,
        &alloc::format!("VkImageView_Storage_2D_{}", debug_name),
      );
      if res.is_err() {
        let mut mut_alloc = alloc;
        allocator.destroy_image(image, &mut mut_alloc);
      }
      res
    }?;

    Ok(Self {
      image: unsafe { NonZeroHandle::new_unchecked(image) },
      image_view: unsafe { NonZeroHandle::new_unchecked(image_view) },
      allocation: alloc,
    })
  }

  #[named]
  pub fn new_paint_image(
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    width: u32,
    height: u32,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let image_info = vk::ImageCreateInfo::default()
      .image_type(vk::ImageType::TYPE_2D)
      .extent(vk::Extent3D {
        width,
        height,
        depth: 1,
      })
      .mip_levels(1)
      .array_layers(1)
      .format(vk::Format::R8G8B8A8_UNORM)
      .tiling(vk::ImageTiling::LINEAR)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .usage(
        vk::ImageUsageFlags::SAMPLED
          | vk::ImageUsageFlags::TRANSFER_SRC
          | vk::ImageUsageFlags::TRANSFER_DST,
      )
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .samples(vk::SampleCountFlags::TYPE_1);

    let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
    allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
    allocation_create_info.flags =
      vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;

    let (image, mut alloc, _alloc_info) =
      unsafe { allocator.create_image_with_alloc_info(&image_info, &allocation_create_info) }
        .with_name(device, &alloc::format!("VkImage_Paint_{}", debug_name))?;

    let image_view_info = vk::ImageViewCreateInfo::default()
      .image(image)
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(vk::Format::R8G8B8A8_UNORM)
      .components(vk::ComponentMapping::default())
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_array_layer(0)
          .layer_count(1)
          .base_mip_level(0)
          .level_count(1),
      );
    let image_view = unsafe {
      let res = device
        .create_image_view(&image_view_info, None)
        .with_name(device, &alloc::format!("VkImageView_Paint_{}", debug_name));
      if res.is_err() {
        allocator.destroy_image(image, &mut alloc);
      }
      res
    }?;

    Ok(Self {
      image: unsafe { NonZeroHandle::new_unchecked(image) },
      image_view: unsafe { NonZeroHandle::new_unchecked(image_view) },
      allocation: alloc,
    })
  }

  #[named]
  pub fn new_storage_3d(
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    width: u32,
    height: u32,
    depth: u32,
    format: vk::Format,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let image_info = vk::ImageCreateInfo::default()
      .image_type(vk::ImageType::TYPE_3D)
      .extent(vk::Extent3D {
        width,
        height,
        depth,
      })
      .mip_levels(1)
      .array_layers(1)
      .format(format)
      .tiling(vk::ImageTiling::OPTIMAL)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .usage(
        vk::ImageUsageFlags::STORAGE
          | vk::ImageUsageFlags::SAMPLED
          | vk::ImageUsageFlags::TRANSFER_SRC,
      )
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .samples(vk::SampleCountFlags::TYPE_1);

    let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(allocation_create_info);
    allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;

    let (image, alloc, _alloc_info) =
      unsafe { allocator.create_image_with_alloc_info(&image_info, &allocation_create_info) }
        .with_name(device, &alloc::format!("VkImage_Storage_3D_{}", debug_name))?;

    let image_view_info = vk::ImageViewCreateInfo::default()
      .image(image)
      .view_type(vk::ImageViewType::TYPE_3D)
      .format(format)
      .components(vk::ComponentMapping::default())
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_array_layer(0)
          .layer_count(1)
          .base_mip_level(0)
          .level_count(1),
      );
    let image_view = unsafe {
      let res = device.create_image_view(&image_view_info, None).with_name(
        device,
        &alloc::format!("VkImageView_Storage3D_{}", debug_name),
      );
      if res.is_err() {
        let mut mut_alloc = alloc;
        allocator.destroy_image(image, &mut mut_alloc);
      }
      res
    }?;

    Ok(Self {
      image: unsafe { NonZeroHandle::new_unchecked(image) },
      image_view: unsafe { NonZeroHandle::new_unchecked(image_view) },
      allocation: alloc,
    })
  }

  #[named]
  pub fn new_2d(
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    command_buffer: vk::CommandBuffer,
    staging_arena: &device::memory::FrameStagingArena,
    texture: &Texture,
    usage: vk::ImageUsageFlags,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let image_size = (texture.data.len()) as vk::DeviceSize;
    if image_size == 0 {
      return Err(crate::gpu_invalid_arg!("invalid argument"));
    }

    // 1. Allocate staging memory
    let (staging_offset, staging_ptr) = staging_arena
      .allocate(image_size as usize, 16)
      .ok_or(crate::gpu_err_device!())?;

    unsafe {
      core::ptr::copy_nonoverlapping(texture.data.as_ptr(), staging_ptr, texture.data.len());
    }
    // memory is HOST_COHERENT.

    // 2. Create device image
    let image_info = vk::ImageCreateInfo::default()
      .image_type(vk::ImageType::TYPE_2D)
      .extent(vk::Extent3D {
        width: texture.width,
        height: texture.height,
        depth: 1,
      })
      .mip_levels(1)
      .array_layers(1)
      .format(texture.format.to_vk_format())
      .tiling(vk::ImageTiling::OPTIMAL)
      .initial_layout(vk::ImageLayout::UNDEFINED)
      .usage(usage | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
      .sharing_mode(vk::SharingMode::EXCLUSIVE)
      .samples(vk::SampleCountFlags::TYPE_1);

    let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(allocation_create_info);
    allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;

    let (image, mut alloc, _alloc_info) =
      unsafe { allocator.create_image_with_alloc_info(&image_info, &allocation_create_info) }
        .map_err(|e| {
          aethervk_oshal_rlib::log!("create_image_with_alloc_info failed: {:?}", e);
          e
        })
        .with_name(device, &alloc::format!("VkImage_New2D_{}", debug_name))?;

    // 2.1 Create Image View, then start recording upload data commands
    let image_view_info = vk::ImageViewCreateInfo::default()
      .image(image)
      .view_type(vk::ImageViewType::TYPE_2D)
      .format(texture.format.to_vk_format())
      .components(vk::ComponentMapping::default())
      .subresource_range(
        vk::ImageSubresourceRange::default()
          .aspect_mask(vk::ImageAspectFlags::COLOR)
          .base_array_layer(0)
          .layer_count(1)
          .base_mip_level(0)
          .level_count(1),
      );
    let image_view = unsafe {
      let res = device
        .create_image_view(&image_view_info, None)
        .map_err(|e| {
          aethervk_oshal_rlib::log!("create_image_view failed: {:?}", e);
          e
        })
        .with_name(device, &alloc::format!("VkImageView_New2D_{}", debug_name));
      if res.is_err() {
        allocator.destroy_image(image, &mut alloc);
      }

      res
    }?;

    // 3. Transition layout to TRANSFER_DST_OPTIMAL
    let image_barrier_to_transfer = vk::ImageMemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::NONE)
      .src_access_mask(vk::AccessFlags2::NONE)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
      .old_layout(vk::ImageLayout::UNDEFINED)
      .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
      .image(image)
      .subresource_range(vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
      });

    let dependency_info_to_transfer = vk::DependencyInfo::default()
      .image_memory_barriers(core::slice::from_ref(&image_barrier_to_transfer));
    unsafe {
      device
        .synchronization2
        .cmd_pipeline_barrier2(command_buffer, &dependency_info_to_transfer);
    }

    // 4. Copy buffer to image
    let buffer_image_copy = vk::BufferImageCopy::default()
      .buffer_offset(staging_offset as vk::DeviceSize)
      .buffer_row_length(0)
      .buffer_image_height(0)
      .image_subresource(vk::ImageSubresourceLayers {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        mip_level: 0,
        base_array_layer: 0,
        layer_count: 1,
      })
      .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
      .image_extent(vk::Extent3D {
        width: texture.width,
        height: texture.height,
        depth: 1,
      });

    unsafe {
      device.cmd_copy_buffer_to_image(
        command_buffer,
        staging_arena.buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        &[buffer_image_copy],
      );
    }

    // 5. Transition layout to SHADER_READ_ONLY_OPTIMAL
    let image_barrier_to_shader_read = vk::ImageMemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
      .dst_access_mask(vk::AccessFlags2::SHADER_READ)
      .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
      .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
      .image(image)
      .subresource_range(vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
      });

    let dependency_info_to_shader_read = vk::DependencyInfo::default()
      .image_memory_barriers(core::slice::from_ref(&image_barrier_to_shader_read));
    unsafe {
      device
        .synchronization2
        .cmd_pipeline_barrier2(command_buffer, &dependency_info_to_shader_read);
    }

    Ok(Self {
      image: unsafe { NonZeroHandle::new_unchecked(image) },
      image_view: unsafe { NonZeroHandle::new_unchecked(image_view) },
      allocation: alloc,
    })
  }
}

impl Hash for Image {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.image.hash(state);
  }
}

#[derive(Clone)]
pub(super) struct SunRenderResource {
  pub resolution: (u32, u32, u32),
  pub image: Option<Image>,
  pub descriptor_set: Option<NonZeroHandle<vk::DescriptorSet>>,
  pub is_generated: bool,
  pub params_buffer: Option<vk::Buffer>,
  pub params_alloc: Option<vk_mem::Allocation>,
  pub compute_descriptor_pool: Option<vk::DescriptorPool>,
  pub compute_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
  pub compute_descriptor_set: Option<vk::DescriptorSet>,
  pub compute_pipeline: Option<crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::Pipeline>>,
  pub compute_pipeline_layout: Option<vk::PipelineLayout>,
  pub last_timeline: u64,
}

struct DiscardDestroyPool {
  pub pool: vk::DescriptorPool,
}
impl DeviceResource for DiscardDestroyPool {
  fn cleanup(&mut self, device: &LogicalDevice) {
    unsafe { device.destroy_descriptor_pool(self.pool, None) };
  }
}

impl SunRenderResource {
  pub fn discard(
    &mut self,
    _device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    discard_pool: &DiscardPool,
    frame_timeline: u64,
  ) {
    if let Some(img) = self.image.take() {
      discard_pool.discard_image_view(img.image_view.get(), frame_timeline);
      discard_pool.discard_image(allocator, img.image.get(), img.allocation, frame_timeline);
    }
    if let Some(layout) = self.compute_pipeline_layout.take() {
      discard_pool.discard_pipeline_layout(layout, frame_timeline);
    }
    if let Some(pool) = self.compute_descriptor_pool.take() {
      discard_pool.discard_type_erased(DiscardDestroyPool { pool }, frame_timeline);
    }
    if let Some(set_layout) = self.compute_descriptor_set_layout.take() {
      discard_pool.discard_descriptor_set_layout(set_layout, frame_timeline);
    }
    if let Some(buf) = self.params_buffer.take() {
      if let Some(alloc) = self.params_alloc.take() {
        discard_pool.discard_buffer(allocator, buf, alloc, frame_timeline);
      }
    }
  }
}

#[derive(Clone)]
pub(super) struct ForwardMesh2RenderResource {
  pub allocator: vk_mem::AllocatorView,
  pub position_vertex_buffer: Buffer,
  pub attributes_vertex_buffer: Buffer,
  pub index_buffer: Buffer,

  pub material_buffer: Buffer,
  pub object_buffer: Buffer,

  /// layout(binding = 0) uniform sampler2D albedoMap;
  pub albedo_image: Option<Image>,
  /// layout(binding = 1) uniform sampler2D normalMap;
  pub normal_image: Option<Image>,
  /// layout(binding = 2) uniform sampler2D roughnessMap;
  pub roughness_image: Option<Image>,
  /// layout(binding = 3) uniform sampler2D aoMap;
  pub ao_image: Option<Image>,
  /// layout(binding = 4) uniform sampler2D skyMap;
  pub sky_image: Option<Image>, // Not owned by this. Owned by device!
  /// layout(binding = 5) uniform sampler2D emissivePaintMap;
  pub emissive_paint_image: Option<Image>,

  /// Note: Purposefully leaked! (TODO: if this creates problems, do better.)
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,
}

pub(super) struct ForwardMesh2RenderResourceParams<'a> {
  pub position_data: &'a [f32],
  pub attribute_data: &'a [f32],
  pub index_data: &'a [u32],
  pub material_data: &'a crate::gpu::MaterialData,
  pub object_data: &'a crate::gpu::ObjectData,
  pub albedo_image: Option<Image>,
  pub normal_image: Option<Image>,
  pub roughness_image: Option<Image>,
  pub ao_image: Option<Image>,
  pub sky_image: Option<Image>,
  pub emissive_paint_image: Option<Image>,
  pub sampler: NonZeroHandle<vk::Sampler>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,
  pub dummy_texture: &'a Image,
  pub debug_name: &'a str,
}

impl ForwardMesh2RenderResource {
  /// Manual drop function
  pub fn discard(&mut self, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_buffer(
      self.allocator,
      self.position_vertex_buffer.buffer.get(),
      self.position_vertex_buffer.allocation,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator,
      self.attributes_vertex_buffer.buffer.get(),
      self.attributes_vertex_buffer.allocation,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator,
      self.index_buffer.buffer.get(),
      self.index_buffer.allocation,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator,
      self.material_buffer.buffer.get(),
      self.material_buffer.allocation,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator,
      self.object_buffer.buffer.get(),
      self.object_buffer.allocation,
      timeline,
    );

    if let Some(albedo_image) = &self.albedo_image {
      discard_pool.discard_image(
        self.allocator,
        albedo_image.image.get(),
        albedo_image.allocation,
        timeline,
      );
      discard_pool.discard_image_view(albedo_image.image_view.get(), timeline);
    }
    if let Some(normal_image) = &self.normal_image {
      discard_pool.discard_image(
        self.allocator,
        normal_image.image.get(),
        normal_image.allocation,
        timeline,
      );
      discard_pool.discard_image_view(normal_image.image_view.get(), timeline);
    }
    if let Some(roughness_image) = &self.roughness_image {
      discard_pool.discard_image(
        self.allocator,
        roughness_image.image.get(),
        roughness_image.allocation,
        timeline,
      );
      discard_pool.discard_image_view(roughness_image.image_view.get(), timeline);
    }
    if let Some(ao_image) = &self.ao_image {
      discard_pool.discard_image(
        self.allocator,
        ao_image.image.get(),
        ao_image.allocation,
        timeline,
      );
      discard_pool.discard_image_view(ao_image.image_view.get(), timeline);
    }
    if let Some(emissive_paint_image) = &self.emissive_paint_image {
      discard_pool.discard_image(
        self.allocator,
        emissive_paint_image.image.get(),
        emissive_paint_image.allocation,
        timeline,
      );
      discard_pool.discard_image_view(emissive_paint_image.image_view.get(), timeline);
    }
  }

  pub fn frontend_texture_flags(&self) -> TextureFlags {
    let mut flags = TextureFlags::empty();
    if self.albedo_image.is_some() {
      flags |= TextureFlags::ALBEDO;
    }
    if self.normal_image.is_some() {
      flags |= TextureFlags::NORMAL;
    }
    if self.roughness_image.is_some() {
      flags |= TextureFlags::ROUGHNESS;
    }
    if self.ao_image.is_some() {
      flags |= TextureFlags::AO;
    }
    if self.emissive_paint_image.is_some() {
      // NOTE: We don't have a FLAG_EMISSIVE defined in TextureFlags yet,
      // but if we did, we'd add it here.
    }
    flags
  }

  pub fn buffers_hash(&self) -> u64 {
    let mut hasher = FnvHasher::new();
    self.position_vertex_buffer.hash(&mut hasher);
    self.attributes_vertex_buffer.hash(&mut hasher);
    self.index_buffer.hash(&mut hasher);
    hasher.finish()
  }

  /// SAFETY:
  /// - if `params` contains Some Images, they should have been already registered in the
  ///   [`RollbackContext`] passed into this function
  /// - `command_buffer` must be in the recording state
  #[named]
  pub(super) unsafe fn new(
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    command_buffer: vk::CommandBuffer,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    params: ForwardMesh2RenderResourceParams<'_>,
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<Self> {
    let vma_allocator = allocator.as_allocator_view();
    let mut image_infos = Vec::with_capacity(6);

    let mut rollback_buffer = move |buf: vk::Buffer, mut alloc: vk_mem::Allocation| {
      rollback.defer(move |_device| {
        unsafe { vma_allocator.destroy_buffer(buf, &mut alloc) };
      });
    };
    let mut push_image_fallback =
      |descriptor_index: u32, img: Option<&Image>, fallback: vk::DescriptorImageInfo| {
        if let Some(image) = img {
          image_infos.push((
            descriptor_index,
            image.to_descriptor_image_info(params.sampler),
          ));
        } else {
          image_infos.push((descriptor_index, fallback));
        }
      };

    // Create position buffer
    let position_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      params.position_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
      &alloc::format!("PositionBuffer_{}", params.debug_name),
    )?;
    rollback_buffer(
      position_vertex_buffer.buffer.get(),
      position_vertex_buffer.allocation,
    );

    // Create attributes buffer
    let attributes_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      params.attribute_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
      &alloc::format!("AttributesBuffer_{}", params.debug_name),
    )?;
    rollback_buffer(
      attributes_vertex_buffer.buffer.get(),
      attributes_vertex_buffer.allocation,
    );

    // Create index buffer
    let index_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      params.index_data,
      vk::BufferUsageFlags::INDEX_BUFFER,
      &alloc::format!("IndexBuffer_{}", params.debug_name),
    )?;
    rollback_buffer(index_buffer.buffer.get(), index_buffer.allocation);

    let material_slice = unsafe {
      core::slice::from_raw_parts(
        params.material_data as *const _ as *const u32,
        core::mem::size_of::<crate::gpu::MaterialData>() / 4,
      )
    };
    let material_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      material_slice,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
      &alloc::format!("MaterialBuffer_{}", params.debug_name),
    )?;
    rollback_buffer(material_buffer.buffer.get(), material_buffer.allocation);

    let object_slice = unsafe {
      core::slice::from_raw_parts(
        params.object_data as *const _ as *const u32,
        core::mem::size_of::<crate::gpu::ObjectData>() / 4,
      )
    };
    let object_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      object_slice,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
      &alloc::format!("ObjectBuffer_{}", params.debug_name),
    )?;
    rollback_buffer(object_buffer.buffer.get(), object_buffer.allocation);

    let dummy_info = params.dummy_texture.to_descriptor_image_info(params.sampler);
    push_image_fallback(0, params.albedo_image.as_ref(), dummy_info);
    push_image_fallback(1, params.normal_image.as_ref(), dummy_info);
    push_image_fallback(2, params.roughness_image.as_ref(), dummy_info);
    push_image_fallback(3, params.ao_image.as_ref(), dummy_info);
    push_image_fallback(4, params.sky_image.as_ref(), dummy_info);
    push_image_fallback(5, params.emissive_paint_image.as_ref(), dummy_info);

    let write_descriptor_sets: Vec<_> = image_infos
      .iter()
      .map(|(binding, info)| {
        vk::WriteDescriptorSet::default()
          .dst_set(params.descriptor_set.get())
          .dst_binding(*binding)
          .dst_array_element(0)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .image_info(core::slice::from_ref(info))
      })
      .collect();

    unsafe {
      device.update_descriptor_sets(&write_descriptor_sets, &[]);
    }

    Ok(Self {
      allocator: vma_allocator,
      position_vertex_buffer,
      attributes_vertex_buffer,
      index_buffer,
      material_buffer,
      object_buffer,
      albedo_image: params.albedo_image,
      normal_image: params.normal_image,
      roughness_image: params.roughness_image,
      ao_image: params.ao_image,
      sky_image: params.sky_image,
      emissive_paint_image: params.emissive_paint_image,
      descriptor_set: params.descriptor_set,
    })
  }
}

struct UploadedFont {
  pub texture: Image,
  pub atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
  pub descriptor_index: u32, // The assigned array element
}

pub(super) struct Text2RenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub descriptor_pool: Option<NonZeroHandle<vk::DescriptorPool>>,
  pub descriptor_set: Option<vk::DescriptorSet>,
  pub font_sampler: Option<vk::Sampler>,
  pub uploaded_fonts: hashbrown::HashMap<u64, UploadedFont>,
  pub free_descriptor_indices: Vec<u32>,
  pub next_descriptor_index: u32,
  pub max_fonts: u32,
  pub allocator_raw: Option<vk_mem::AllocatorView>,

  pub glyphs_buffer: NonZeroHandle<vk::Buffer>,
  pub glyphs_alloc: vk_mem::Allocation,
  pub glyphs_ptr: u64,
}

pub(super) struct Text2RenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      Text2RenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl Text2RenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

impl Text2RenderResourceArchetypeArena {
  /// Allocates an index from the arena's free list or increments the linear counter
  #[named]
  pub fn allocate_descriptor_index(&mut self) -> GpuResult<u32> {
    if let Some(idx) = self.free_descriptor_indices.pop() {
      Ok(idx)
    } else if self.next_descriptor_index < self.max_fonts {
      let idx = self.next_descriptor_index;
      self.next_descriptor_index += 1;
      Ok(idx)
    } else {
      Err(crate::gpu_err!(
        "Exceeded descriptor array layout maximum capacity"
      ))
    }
  }

  /// Binds the finalized Vulkan Image into the descriptor set and registers it
  pub fn bind_font_image(
    &mut self,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    font_hash: u64,
    atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
    image: crate::gpu_backends::vulkan::device::resources::Image,
    descriptor_index: u32,
  ) -> GpuResult<()> {
    if let (Some(sampler), Some(set)) = (self.font_sampler, self.descriptor_set) {
      let image_info = [ash::vk::DescriptorImageInfo::default()
        .image_layout(ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(image.image_view.get())
        .sampler(sampler)];

      let write = ash::vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .dst_array_element(descriptor_index)
        .descriptor_type(ash::vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_info);

      unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    self.uploaded_fonts.insert(
      font_hash,
      crate::gpu_backends::vulkan::device::resources::UploadedFont {
        texture: image,
        atlas,
        descriptor_index,
      },
    );

    Ok(())
  }

  /// Extracts the font from the arena and returns it for safe destruction
  #[named]
  pub fn remove_font_atlas(
    &mut self,
    font_hash: u64,
  ) -> GpuResult<crate::gpu_backends::vulkan::device::resources::UploadedFont> {
    if let Some(uploaded) = self.uploaded_fonts.remove(&font_hash) {
      // Create a "Hole" implicitly pushing the index back as structurally available.
      self.free_descriptor_indices.push(uploaded.descriptor_index);
      Ok(uploaded)
    } else {
      Err(crate::gpu_invalid_arg!("atlas not found: {}", font_hash))
    }
  }

  pub fn discard(&mut self, _device: &LogicalDevice, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
    if let Some(sampler) = self.font_sampler {
      struct SamplerDiscard(vk::Sampler);
      impl DeviceResource for SamplerDiscard {
        fn cleanup(&mut self, device: &super::LogicalDevice) {
          unsafe {
            device.destroy_sampler(self.0, None);
          }
        }
      }
      discard_pool.discard_type_erased(SamplerDiscard(sampler), timeline);
    }
    if let Some(pool) = self.descriptor_pool {
      struct PoolDiscard(vk::DescriptorPool);
      impl DeviceResource for PoolDiscard {
        fn cleanup(&mut self, device: &super::LogicalDevice) {
          unsafe {
            device.destroy_descriptor_pool(self.0, None);
          }
        }
      }
      discard_pool.discard_type_erased(PoolDiscard(pool.get()), timeline);
    }
    if let Some(allocator_raw) = self.allocator_raw {
      for (_, uploaded) in self.uploaded_fonts.drain() {
        discard_pool.discard_image_view(uploaded.texture.image_view.get(), timeline);
        discard_pool.discard_image(
          allocator_raw,
          uploaded.texture.image.get(),
          uploaded.texture.allocation,
          timeline,
        );
      }
      discard_pool.discard_buffer(
        allocator_raw,
        self.glyphs_buffer.get(),
        self.glyphs_alloc,
        timeline,
      );
    }
  }
}

impl ArchetypeArenaCreate for Text2RenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;

    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers,
      ),
      ctx.device.max_per_stage_descriptor_samplers,
    );
    let max_fonts = core::cmp::min(256, device_limit);
    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_fonts)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
      .max_sets(1)
      .pool_sizes(&pool_sizes)
      .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;
    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_fonts)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let binding_flags =
      [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];

    let mut binding_flags_info =
      vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
      .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
      .bindings(&bindings)
      .push_next(&mut binding_flags_info);

    let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(pool)
      .set_layouts(core::slice::from_ref(&set_layout));
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?[0];
    let sampler_info = vk::SamplerCreateInfo::default()
      .mag_filter(vk::Filter::LINEAR)
      .min_filter(vk::Filter::LINEAR)
      .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
      .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
      .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
      .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let font_sampler =
      unsafe { device.create_sampler(&sampler_info, None) }.with_name(device, "Text 2 Sampler")?;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::Text2PushConstants>() as u32)];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(core::slice::from_ref(&set_layout))
      .push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_Text2RenderResourceArchetypeArena")?;

    let buffer_size = (100_000 * core::mem::size_of::<crate::gpu::TextGlyphGpu>()) as u64;
    let buffer_info = vk::BufferCreateInfo::default().size(buffer_size).usage(
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
    );

    let mut mem_alloc_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(mem_alloc_info);
    mem_alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    mem_alloc_info.flags |= vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
      | vk_mem::AllocationCreateFlags::MAPPED;
    crate::apply_test_dedicated_alloc!(mem_alloc_info);

    let (glyphs_buffer, glyphs_alloc) =
      unsafe { allocator.create_buffer(&buffer_info, &mem_alloc_info) }
        .map_err(|_| GpuError::OutOfMemory)?;

    device.set_debug_name(glyphs_buffer, "MegaBuffer_TextGlyphs");

    let addr_info = ash::vk::BufferDeviceAddressInfo::default().buffer(glyphs_buffer);
    let glyphs_ptr = unsafe { device.buffer_device_address.get_buffer_device_address(&addr_info) };

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(set_layout) },
      descriptor_pool: Some(unsafe { NonZeroHandle::new_unchecked(pool) }),
      descriptor_set: Some(descriptor_set),
      font_sampler: Some(font_sampler),
      uploaded_fonts: hashbrown::HashMap::new(),
      free_descriptor_indices: Vec::new(),
      next_descriptor_index: 0,
      max_fonts,
      allocator_raw: Some(allocator.as_allocator_view()),
      glyphs_buffer: unsafe { NonZeroHandle::new_unchecked(glyphs_buffer) },
      glyphs_alloc,
      glyphs_ptr,
    })
  }
}
impl Text2RenderResourceArchetypeArena {
  #[named]
  pub fn upload_font_atlas(
    &mut self,
    device: &LogicalDevice,
    _queue: &device::Queue,
    allocator: vk_mem::AllocatorView,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    command_buffer: vk::CommandBuffer,
    font_hash: u64,
    atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
  ) -> GpuResult<u32> {
    if let Some(existing) = self.uploaded_fonts.get(&font_hash) {
      return Ok(existing.descriptor_index);
    }

    let descriptor_index = if let Some(idx) = self.free_descriptor_indices.pop() {
      idx
    } else if self.next_descriptor_index < self.max_fonts {
      let idx = self.next_descriptor_index;
      self.next_descriptor_index += 1;
      idx
    } else {
      return Err(gpu_err!(
        "Exceeded descriptor array layout maximum capacity"
      ));
    };

    let texture = crate::simulation::comet::Texture {
      data: atlas.image_data.clone().into(),
      format: crate::simulation::comet::TexelFormat::R8_UNORM,
      width: atlas.width,
      height: atlas.height,
      has_mipmaps: false,
    };

    let image = Image::new_2d(
      device,
      allocator,
      command_buffer,
      staging_arena,
      &texture,
      vk::ImageUsageFlags::SAMPLED,
      "FontAtlas Dynamic Text2",
    )?;

    if let (Some(sampler), Some(set)) = (self.font_sampler, self.descriptor_set) {
      let image_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(image.image_view.get())
        .sampler(sampler)];

      let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .dst_array_element(descriptor_index)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_info);

      unsafe { device.update_descriptor_sets(&[write], &[]) };
    }

    self.uploaded_fonts.insert(
      font_hash,
      UploadedFont {
        texture: image,
        atlas,
        descriptor_index,
      },
    );

    Ok(descriptor_index)
  }
}

pub(super) struct SphereGizmoRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,

  pub data_buffer: NonZeroHandle<vk::Buffer>,
  pub data_alloc: vk_mem::Allocation,
  pub data_ptr: u64,

  pub allocated_gizmos: hashbrown::HashMap<crate::scene::EntityId, u32>,
  pub free_list: Vec<u32>,
  pub next_index: u32,

  allocator_raw: vk_mem::AllocatorView,
}

pub(super) struct SphereGizmoRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      SphereGizmoRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl SphereGizmoRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

impl SphereGizmoRenderResourceArchetypeArena {
  pub fn allocate_sphere_gizmo_instance(
    &mut self,
    entity: crate::scene::EntityId,
  ) -> GpuResult<u32> {
    if let Some(&idx) = self.allocated_gizmos.get(&entity) {
      return Ok(idx);
    }
    let idx = if let Some(idx) = self.free_list.pop() {
      idx
    } else {
      let idx = self.next_index;
      self.next_index += 1;
      idx
    };
    self.allocated_gizmos.insert(entity, idx);
    Ok(idx)
  }

  pub fn free_sphere_gizmo_instance(&mut self, entity: crate::scene::EntityId) {
    if let Some(idx) = self.allocated_gizmos.remove(&entity) {
      self.free_list.push(idx);
    }
  }

  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
    discard_pool.discard_buffer(
      self.allocator_raw,
      self.data_buffer.get(),
      self.data_alloc,
      timeline,
    );
  }
}
impl ArchetypeArenaCreate for SphereGizmoRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::SphereGizmoPushConstants>() as u32)];

    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);

    unsafe {
      let pipeline_layout = device
        .create_pipeline_layout(&pipeline_layout_info, None)
        .with_name(
          device,
          "VkPipelineLayout_SphereGizmoRenderResourceArchetytpeArena",
        )
        .map_err(|e| {
          aethervk_oshal_rlib::log!("create_pipeline_layout failed: {:?}", e);
          e
        })?;

      let buffer_size = (100_000 * core::mem::size_of::<crate::gpu::SphereGizmoDataGpu>()) as u64;
      let buffer_info = vk::BufferCreateInfo::default().size(buffer_size).usage(
        vk::BufferUsageFlags::STORAGE_BUFFER
          | vk::BufferUsageFlags::TRANSFER_DST
          | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
      );

      let mut mem_alloc_info = vk_mem::AllocationCreateInfo::default();
      crate::apply_test_dedicated_alloc!(mem_alloc_info);
      mem_alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;

      let (data_buffer, data_alloc) = allocator
        .create_buffer(&buffer_info, &mem_alloc_info)
        .map_err(|_| GpuError::OutOfMemory)?;

      device.set_debug_name(data_buffer, "MegaBuffer_SphereGizmoData");

      let addr_info = ash::vk::BufferDeviceAddressInfo::default().buffer(data_buffer);
      let data_ptr = device.buffer_device_address.get_buffer_device_address(&addr_info);

      Ok(Self {
        pipeline_layout: NonZeroHandle::new_unchecked(pipeline_layout),
        data_buffer: NonZeroHandle::new_unchecked(data_buffer),
        data_alloc,
        data_ptr,
        allocated_gizmos: hashbrown::HashMap::new(),
        free_list: Vec::new(),
        next_index: 0,
        allocator_raw: allocator,
      })
    }
  }
}

#[derive(Clone)]
pub(super) struct MeasurementRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub(super) struct MeasurementRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      MeasurementRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for MeasurementRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;

    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::MeasurementPushConstants>() as u32,
    }];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };

    {
      let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
        .with_name(
        device,
        "VkPipelineLayout_MeasurementRenderResourceArchetype",
      )?;

      Ok(Self {
        pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
        push_constant_ranges,
      })
    }
  }
}
impl MeasurementRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, _timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, u64::MAX);
  }
}

impl MeasurementRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

pub(super) struct MarkerRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub(super) struct MarkerRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      MarkerRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for MarkerRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;

    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::MarkerPushConstants>() as u32,
    }];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_MarkerRenderResourceArchetype")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
    })
  }
}

impl MarkerRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl MarkerRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

#[derive(Clone, Debug)]
pub struct Range {
  pub offset: u64,
  pub size: u64,
}

pub struct RangeAllocator {
  pub free_ranges: Vec<Range>,
}

impl RangeAllocator {
  pub fn new(capacity: u64) -> Self {
    Self {
      free_ranges: alloc::vec![Range {
        offset: 0,
        size: capacity
      }],
    }
  }

  pub fn allocate(&mut self, size: u64) -> Option<u64> {
    if size == 0 {
      return Some(0);
    }
    let mut best_idx: Option<usize> = None;
    for (i, range) in self.free_ranges.iter().enumerate() {
      if range.size >= size {
        if best_idx.is_none() || self.free_ranges[best_idx.unwrap()].size > range.size {
          best_idx = Some(i);
        }
      }
    }

    if let Some(i) = best_idx {
      let offset = self.free_ranges[i].offset;
      self.free_ranges[i].offset += size;
      self.free_ranges[i].size -= size;
      if self.free_ranges[i].size == 0 {
        self.free_ranges.remove(i);
      }
      Some(offset)
    } else {
      None
    }
  }

  pub fn free(&mut self, offset: u64, size: u64) {
    if size == 0 {
      return;
    }
    self.free_ranges.push(Range { offset, size });
    self.free_ranges.sort_unstable_by_key(|r| r.offset);

    let mut merged = Vec::new();
    let mut current = self.free_ranges[0].clone();
    for range in self.free_ranges.iter().skip(1) {
      if current.offset + current.size == range.offset {
        current.size += range.size;
      } else {
        merged.push(current);
        current = range.clone();
      }
    }
    merged.push(current);
    self.free_ranges = merged;
  }
}

pub struct CurveAllocation {
  pub segments_offset: u64,
  pub segment_capacity: usize, // Measured in RationalBezierGpu chunks
  pub last_seen_tick: u64,
  pub last_hash: u64,
}

// Fast hash evaluator (Control points & Model Matrix)
pub(super) fn hash_trajectory(
  points: &[[f32; 4]],
  model_mat: &aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
) -> u64 {
  let mut hash = 0xcbf29ce484222325_u64;
  let prime = 0x100000001b3_u64;
  for arr in points {
    for &f in arr {
      hash ^= f.to_bits() as u64;
      hash = hash.wrapping_mul(prime);
    }
  }
  let mat_bytes = unsafe {
    core::slice::from_raw_parts(
      model_mat as *const _ as *const u8,
      core::mem::size_of_val(model_mat),
    )
  };
  for &b in mat_bytes {
    hash ^= b as u64;
    hash = hash.wrapping_mul(prime);
  }
  hash
}

/// Archetype has a list of textures, then each instance, when push constants, chooses one
/// with the textureId.
/// `NonZeroHandle` constraint satisfied until you call `discard`
pub(super) struct UiRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub set_0_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
  pub descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,

  pub elements_buffer: NonZeroHandle<vk::Buffer>,
  pub elements_alloc: vk_mem::Allocation,
  pub elements_ptr: u64,

  pub tick: u64,

  allocator_raw: vk_mem::AllocatorView,
}

pub(super) struct UiRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<UiRenderResourceArchetypeArena>,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl UiRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);
    struct PoolDiscard(vk::DescriptorPool);
    impl DeviceResource for PoolDiscard {
      fn cleanup(&mut self, device: &super::LogicalDevice) {
        unsafe {
          device.destroy_descriptor_pool(self.0, None);
        }
      }
    }
    discard_pool.discard_type_erased(PoolDiscard(self.descriptor_pool.get()), timeline);
    discard_pool.discard_buffer(
      self.allocator_raw,
      self.elements_buffer.get(),
      self.elements_alloc,
      timeline,
    );
  }
}

impl ArchetypeArenaCreate for UiRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers,
      ),
      ctx.device.max_per_stage_descriptor_samplers,
    );
    let max_image_count = core::cmp::min(256, device_limit);
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::UiPushConstants>() as u32,
    }];

    // Create the bindless layout for set = 0
    let bindings = [vk::DescriptorSetLayoutBinding {
      binding: 0,
      descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
      descriptor_count: max_image_count,
      stage_flags: vk::ShaderStageFlags::FRAGMENT,
      p_immutable_samplers: ptr::null(),
      ..Default::default()
    }];

    let binding_flags =
      [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];

    let binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
      binding_count: binding_flags.len() as u32,
      p_binding_flags: binding_flags.as_ptr(),
      ..Default::default()
    };

    let bindless_layout_info = vk::DescriptorSetLayoutCreateInfo {
      flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
      binding_count: bindings.len() as u32,
      p_bindings: bindings.as_ptr(),
      p_next: ptr::from_ref(&binding_flags_info) as *const _,
      ..Default::default()
    };

    let set_0_layout = unsafe { device.create_descriptor_set_layout(&bindless_layout_info, None) }
      .with_name(device, "VkDescriptorSetLayout_UiBindlessTextures")?;

    let set_layouts = [set_0_layout];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_set_layouts: set_layouts.as_ptr(),
      set_layout_count: set_layouts.len() as u32,
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_UiRenderResourceArchetype")?;

    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_image_count)];
    let create_info = vk::DescriptorPoolCreateInfo::default()
      .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
      .max_sets(1)
      .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None) }
      .with_name(device, "VkDescriptorPool_UiArchetype")?;

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool)
      .set_layouts(&set_layouts);

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?[0];

    // Create the mega-buffer for UI elements
    let elements_size = (250_000 * core::mem::size_of::<crate::gpu::UiElementGpu>()) as u64;

    let elements_buffer_info = vk::BufferCreateInfo {
      size: elements_size,
      usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
      sharing_mode: vk::SharingMode::EXCLUSIVE,
      ..Default::default()
    };

    let mut elements_alloc_info = vk_mem::AllocationCreateInfo::default();
    elements_alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
    elements_alloc_info.flags = vk_mem::AllocationCreateFlags::MAPPED
      | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;
    crate::apply_test_dedicated_alloc!(elements_alloc_info);

    let (elements_buffer, elements_alloc) =
      unsafe { allocator.create_buffer(&elements_buffer_info, &elements_alloc_info) }?;

    let elements_ptr = unsafe {
      device
        .buffer_device_address
        .get_buffer_device_address(&vk::BufferDeviceAddressInfo {
          buffer: elements_buffer,
          ..Default::default()
        })
    };

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      set_0_layout: unsafe { NonZeroHandle::new_unchecked(set_0_layout) },
      push_constant_ranges,
      descriptor_pool: unsafe { NonZeroHandle::new_unchecked(descriptor_pool) },
      descriptor_set: unsafe { NonZeroHandle::new_unchecked(descriptor_set) },
      elements_buffer: unsafe { NonZeroHandle::new_unchecked(elements_buffer) },
      elements_alloc,
      elements_ptr,
      tick: 0,
      allocator_raw: allocator,
    })
  }
}

impl UiRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

/// Archetype has a list of textures, then each instance, when push constants, chooses one
/// with the textureId.
/// `NonZeroHandle` constraint satisfied until you call `discard`
pub(super) struct TrajectoryRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub set_0_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
  pub descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,

  pub segments_buffer: NonZeroHandle<vk::Buffer>,
  pub segments_alloc: vk_mem::Allocation,
  pub segments_ptr: u64,

  pub trajectories_buffer: NonZeroHandle<vk::Buffer>,
  pub trajectories_alloc: vk_mem::Allocation,
  pub trajectories_ptr: u64,

  pub map_buffer: NonZeroHandle<vk::Buffer>,
  pub map_alloc: vk_mem::Allocation,
  pub map_ptr: u64,

  pub segment_allocator: RangeAllocator,
  pub curves: BTreeMap<crate::scene::EntityId, CurveAllocation>,
  pub tick: u64,

  allocator_raw: vk_mem::AllocatorView,
}

pub(super) struct TrajectoryRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      TrajectoryRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for TrajectoryRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers,
      ),
      ctx.device.max_per_stage_descriptor_samplers,
    );
    let max_image_count = core::cmp::min(256, device_limit);
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::TrajectoryPushConstants>() as u32,
    }];

    // Create the bindless layout for set = 0
    let bindings = [vk::DescriptorSetLayoutBinding {
      binding: 0,
      descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
      descriptor_count: max_image_count,
      stage_flags: vk::ShaderStageFlags::FRAGMENT,
      p_immutable_samplers: ptr::null(),
      ..Default::default()
    }];

    let binding_flags =
      [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];

    let binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
      binding_count: binding_flags.len() as u32,
      p_binding_flags: binding_flags.as_ptr(),
      ..Default::default()
    };

    let bindless_layout_info = vk::DescriptorSetLayoutCreateInfo {
      flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
      binding_count: bindings.len() as u32,
      p_bindings: bindings.as_ptr(),
      p_next: ptr::from_ref(&binding_flags_info) as *const _,
      ..Default::default()
    };

    let set_0_layout = unsafe { device.create_descriptor_set_layout(&bindless_layout_info, None) }
      .with_name(device, "VkDescriptorSetLayout_TrajectoryBindlessTextures")?;

    let set_layouts = [set_0_layout];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_set_layouts: set_layouts.as_ptr(),
      set_layout_count: set_layouts.len() as u32,
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_TrajectoryRenderResourceArchetype")?;

    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_image_count)];
    let create_info = vk::DescriptorPoolCreateInfo::default()
      .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
      .max_sets(1)
      .pool_sizes(&pool_sizes);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None) }
      .with_name(device, "VkDescriptorPool_TrajectoryArchetype")?;

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool)
      .set_layouts(&set_layouts);

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?[0];

    let segments_size = (1_000_000 * core::mem::size_of::<crate::gpu::RationalBezierGpu>()) as u64;
    let traj_size = (100_000 * core::mem::size_of::<crate::gpu::TrajectoryGpu>()) as u64;
    let map_size = (1_000_000 * core::mem::size_of::<crate::gpu::SegmentMapGpu>()) as u64;

    let create_mega_buffer =
      |size: u64,
       debug_name: &str|
       -> GpuResult<(NonZeroHandle<vk::Buffer>, vk_mem::Allocation, u64)> {
        let buffer_info = vk::BufferCreateInfo::default().size(size).usage(
          vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
        );
        let mut alloc_info = vk_mem::AllocationCreateInfo::default();
        crate::apply_test_dedicated_alloc!(alloc_info);
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;

        let (buffer, alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }
          .map_err(|_| GpuError::OutOfMemory)?;
        device.set_debug_name(buffer, debug_name);

        let addr_info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let ptr = unsafe { device.buffer_device_address.get_buffer_device_address(&addr_info) };
        Ok((unsafe { NonZeroHandle::new_unchecked(buffer) }, alloc, ptr))
      };

    let (segments_buffer, segments_alloc, segments_ptr) =
      create_mega_buffer(segments_size, "MegaBuffer_TrajectorySegments")?;
    let (trajectories_buffer, trajectories_alloc, trajectories_ptr) =
      create_mega_buffer(traj_size, "MegaBuffer_Trajectories")?;
    let (map_buffer, map_alloc, map_ptr) =
      create_mega_buffer(map_size, "MegaBuffer_TrajectoryMaps")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      set_0_layout: unsafe { NonZeroHandle::new_unchecked(set_0_layout) },
      push_constant_ranges,
      descriptor_pool: unsafe { NonZeroHandle::new_unchecked(descriptor_pool) },
      descriptor_set: unsafe { NonZeroHandle::new_unchecked(descriptor_set) },
      segments_buffer,
      segments_alloc,
      segments_ptr,
      trajectories_buffer,
      trajectories_alloc,
      trajectories_ptr,
      map_buffer,
      map_alloc,
      map_ptr,
      segment_allocator: RangeAllocator::new(1_000_000),
      curves: BTreeMap::new(),
      tick: 0,
      allocator_raw: allocator,
    })
  }
}

impl TrajectoryRenderResourceArchetypeArena {
  pub fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);
    unsafe {
      device.destroy_descriptor_pool(self.descriptor_pool.get(), None);
    }
    discard_pool.discard_buffer(
      self.allocator_raw,
      self.segments_buffer.get(),
      self.segments_alloc,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator_raw,
      self.trajectories_buffer.get(),
      self.trajectories_alloc,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator_raw,
      self.map_buffer.get(),
      self.map_alloc,
      timeline,
    );
  }
}

impl TrajectoryRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

pub struct UploadedTexture {
  pub texture: Image,
  pub descriptor_index: u32,
  pub last_used_frame: u64,
}

pub(super) struct BillboardRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub set_0_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub set_1_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
  /// Not using the `super::descriptors::DescriptorPools` because, as this is an archetype,
  /// pool should be persistent
  pub descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,

  pub uploaded_textures: hashbrown::HashMap<u64, UploadedTexture>,
  pub free_descriptor_indices: Vec<u32>,
  pub next_descriptor_index: u32,
  pub max_textures: u32,
  pub allocator_raw: Option<vk_mem::AllocatorView>,
}

pub(super) struct BillboardRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      BillboardRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for BillboardRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator_raw = ctx.allocator;
    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers,
      ),
      ctx.device.max_per_stage_descriptor_samplers,
    );
    let max_image_count = core::cmp::min(256, device_limit);
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::BillboardPushConstants>() as u32,
    }];

    // 1. Create a dummy layout for set = 0, because your shader specifies set = 1.
    // (If you actually have a global camera/scene descriptor at set = 0, use its layout here instead!)
    let empty_layout_info = vk::DescriptorSetLayoutCreateInfo::default();
    let set_0_layout = unsafe { device.create_descriptor_set_layout(&empty_layout_info, None) }
      .with_name(device, "VkDescriptorSetLayout_EmptySet0")?;

    // 2. Create the bindless layout for set = 1
    let bindings = [vk::DescriptorSetLayoutBinding {
      binding: 0,
      descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
      descriptor_count: max_image_count, // Bounded to exactly match the max capacity in your POOL_SIZES
      stage_flags: vk::ShaderStageFlags::FRAGMENT,
      p_immutable_samplers: ptr::null(),
      ..Default::default()
    }];

    // Enable PARTIALLY_BOUND so the shader can use an array without us populating all 256 slots initially.
    // Enable UPDATE_AFTER_BIND so we can write to the descriptors after binding to the command buffer.
    let binding_flags =
      [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];

    let binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
      binding_count: binding_flags.len() as u32,
      p_binding_flags: binding_flags.as_ptr(),
      ..Default::default()
    };

    let bindless_layout_info = vk::DescriptorSetLayoutCreateInfo {
      flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
      binding_count: bindings.len() as u32,
      p_bindings: bindings.as_ptr(),
      p_next: ptr::from_ref(&binding_flags_info) as *const _,
      ..Default::default()
    };

    let set_1_layout = unsafe { device.create_descriptor_set_layout(&bindless_layout_info, None) }
      .with_name(device, "VkDescriptorSetLayout_BindlessTextures")?;

    // Array indices directly map to `set = X` in your GLSL
    let set_layouts = [set_0_layout, set_1_layout];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_set_layouts: set_layouts.as_ptr(),
      set_layout_count: set_layouts.len() as u32,
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_BillboardRenderResourceArchetype")?;

    // Create descriptor pool and descriptor set
    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_image_count)];
    let create_info = vk::DescriptorPoolCreateInfo::default()
      // flag to allow allocations of bindless sets. from VK_EXT_descriptor_indexing
      .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
      .max_sets(1)
      .pool_sizes(&pool_sizes);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None) }.with_name(
      device,
      "VkDescriptorPoolCreateInfo_Dedicated_BillboardRenderResourceArchetype",
    )?;

    let layouts = [set_1_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool)
      .set_layouts(&layouts);

    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?
      .get(0)
      .copied()
      .unwrap();
    device.set_debug_name(
      descriptor_set,
      "VkDescriptorSet_Dedicated_BillboardRenderResourceArchetype",
    );

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
      set_0_layout: unsafe { NonZeroHandle::new_unchecked(set_0_layout) },
      set_1_layout: unsafe { NonZeroHandle::new_unchecked(set_1_layout) },
      descriptor_pool: unsafe { NonZeroHandle::new_unchecked(descriptor_pool) },
      descriptor_set: unsafe { NonZeroHandle::new_unchecked(descriptor_set) },
      uploaded_textures: hashbrown::HashMap::new(),
      free_descriptor_indices: Vec::new(),
      next_descriptor_index: 0,
      max_textures: max_image_count,
      allocator_raw: Some(allocator_raw),
    })
  }
}

impl BillboardRenderResourceArchetypeArena {
  /// Uploads a texture to the GPU and assigns it to a specific index in the bindless array.
  /// Returns the `Image` so the caller can hold onto it (to prevent it from dropping)
  /// and eventually discard it when it is no longer needed.
  #[named]
  pub fn add_texture(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    command_buffer: vk::CommandBuffer,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    texture: &Texture,
    sampler: NonZeroHandle<vk::Sampler>,
    array_index: u32,
    debug_name: &str,
  ) -> GpuResult<Image> {
    // 1. Create the Image using your existing helper.
    // This flawlessly handles the staging buffer, transfer commands,
    // pipeline barriers, and staging cleanup!
    let image = Image::new_2d(
      device,
      allocator,
      command_buffer,
      staging_arena,
      texture,
      vk::ImageUsageFlags::SAMPLED,
      debug_name,
    )?;

    // 2. Update the specific index in the bindless descriptor array.
    let image_info = vk::DescriptorImageInfo::default()
      .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
      .image_view(image.image_view.get())
      .sampler(sampler.get());

    let write = vk::WriteDescriptorSet::default()
      .dst_set(self.descriptor_set.get())
      .dst_binding(0) // Binding 0 from your layout
      .dst_array_element(array_index) // The specific slot in textures[]
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .image_info(core::slice::from_ref(&image_info));

    unsafe {
      device.update_descriptor_sets(core::slice::from_ref(&write), &[]);
    }

    // Return the image so the caller can manage its lifetime/discard later.
    Ok(image)
  }

  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    // we don't care about descriptor set. discard the pool
    discard_pool.discard_type_erased(
      DiscardDestroyPool {
        pool: self.descriptor_pool.get(),
      },
      timeline,
    );

    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.set_1_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);
  }
}

impl BillboardRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

pub(super) struct GizmoRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub set_0_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
  pub descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,
  pub next_index: AtomicU32,
  pub host_buffers: DebugTrackedRwLock<hashbrown::HashMap<u32, Buffer>>,
  pub allocator_raw: vk_mem::AllocatorView,
}

pub(super) struct GizmoRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      GizmoRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl GizmoRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

impl GizmoRenderResourceArchetypeArena {
  pub const MAX_BUFFER_COUNT: u32 = 256;
}

impl ArchetypeArenaCreate for GizmoRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator_raw = ctx.allocator;
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::GizmoPushConstants>() as u32,
    }];

    let bindings = [vk::DescriptorSetLayoutBinding {
      binding: 0,
      descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
      descriptor_count: GizmoRenderResourceArchetypeArena::MAX_BUFFER_COUNT,
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      p_immutable_samplers: ptr::null(),
      ..Default::default()
    }];

    let binding_flags =
      [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];

    let binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
      binding_count: binding_flags.len() as u32,
      p_binding_flags: binding_flags.as_ptr(),
      ..Default::default()
    };

    let bindless_layout_info = vk::DescriptorSetLayoutCreateInfo {
      flags: vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL,
      binding_count: bindings.len() as u32,
      p_bindings: bindings.as_ptr(),
      p_next: ptr::from_ref(&binding_flags_info) as *const _,
      ..Default::default()
    };

    let set_0_layout = unsafe { device.create_descriptor_set_layout(&bindless_layout_info, None) }
      .with_name(device, "VkDescriptorSetLayout_Gizmo")?;

    let set_layouts = [set_0_layout];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_set_layouts: set_layouts.as_ptr(),
      set_layout_count: set_layouts.len() as u32,
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_Gizmo")?;

    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::STORAGE_BUFFER)
      .descriptor_count(Self::MAX_BUFFER_COUNT)];
    let create_info = vk::DescriptorPoolCreateInfo::default()
      .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND)
      .max_sets(1)
      .pool_sizes(&pool_sizes);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None) }
      .with_name(device, "VkDescriptorPool_Gizmo")?;

    let layouts = [set_0_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool)
      .set_layouts(&layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?
      .get(0)
      .copied()
      .unwrap();
    device.set_debug_name(descriptor_set, "VkDescriptorSet_Gizmo");

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
      set_0_layout: unsafe { NonZeroHandle::new_unchecked(set_0_layout) },
      descriptor_pool: unsafe { NonZeroHandle::new_unchecked(descriptor_pool) },
      descriptor_set: unsafe { NonZeroHandle::new_unchecked(descriptor_set) },
      next_index: AtomicU32::new(0),
      host_buffers: DebugTrackedRwLock::new(hashbrown::HashMap::new()),
      allocator_raw,
    })
  }
}

impl GizmoRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_type_erased(
      DiscardDestroyPool {
        pool: self.descriptor_pool.get(),
      },
      timeline,
    );

    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);

    let mut buffers = self.host_buffers.write();
    for (_, buffer) in buffers.drain() {
      discard_pool.discard_buffer(
        self.allocator_raw,
        buffer.buffer.get(),
        buffer.allocation,
        timeline,
      );
    }
  }
}

pub(super) struct CursorRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub(super) struct CursorRenderResourceArchetype {
  pub arena: alloc::sync::Weak<DebugTrackedRwLock<CursorRenderResourceArchetypeArena>>,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl CursorRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl ArchetypeArenaCreate for CursorRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::CursorPushConstants>() as u32,
    }];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_CursorRenderResourceArchetype")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
    })
  }
}

impl CursorRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

pub(super) struct SkyRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub descriptor_set: Option<NonZeroHandle<vk::DescriptorSet>>,
}

pub(super) struct SkyRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<SkyRenderResourceArchetypeArena>,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for SkyRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(1)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::SkyPushConstants>() as u32)]; // mat4 invViewProj

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(core::slice::from_ref(&set_layout))
      .push_constant_ranges(&push_constant_ranges);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_SkyRenderResourceArchetypeArena")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(set_layout) },
      descriptor_set: None,
    })
  }
}

impl SkyRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
  }
}

impl SkyRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

pub(super) struct BackgroundRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
}

pub(super) struct BackgroundRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      BackgroundRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for BackgroundRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::BackgroundPushConstants>() as u32)];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(
        device,
        "VkPipelineLayout_BackgroundRenderResourceArchetypeArena",
      )?;
    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
    })
  }
}

impl BackgroundRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl BackgroundRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

pub(super) struct GridRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
}

pub(super) struct GridRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      GridRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl GridRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl ArchetypeArenaCreate for GridRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::GridPushConstants>() as u32)];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_GridRenderResourceArchetypeArena")?;
    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
    })
  }
}

impl GridRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

pub(super) struct SunRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub(super) struct SunRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<SunRenderResourceArchetypeArena>,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl SunRenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

impl ArchetypeArenaCreate for SunRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let _allocator_raw = ctx.allocator.get_raw();
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::SunPushConstants>() as u32,
    }];

    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER) // Actually, the shader says sampler3D. So it's COMBINED_IMAGE_SAMPLER. Let's look at sun_volume.frag.
      .descriptor_count(1)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let set_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let descriptor_set_layout =
      unsafe { device.create_descriptor_set_layout(&set_layout_info, None) }
        .with_name(device, "VkDescriptorSetLayout_SunRenderResourceArchetype")?;

    let set_layouts = [descriptor_set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .push_constant_ranges(&push_constant_ranges)
      .set_layouts(&set_layouts);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_SunRenderResourceArchetype")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(descriptor_set_layout) },
      push_constant_ranges,
    })
  }
}

impl SunRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
  }
}

/// To be destroyed before descriptor pool
pub(super) struct ForwardMeshRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layouts: Vec<NonZeroHandle<vk::DescriptorSetLayout>>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
  // 0 = vertex, 1 = fragment
  pub specialization_constants: [Vec<vk::SpecializationMapEntry>; 2],
  // 0 = vertex, 1 = fragment
  pub specialization_constants_values: [Vec<u8>; 2],

  pub dummy_texture_handle: Image,
  /// Necessary evil for discard. assumes it outlives this object
  allocator_raw: vk_mem::AllocatorView,
}

/// Reusable helper function to perform the explicit staging buffer upload pattern.
#[named]
pub(super) fn create_buffer_with_staging<T: Copy>(
  device: &LogicalDevice,
  allocator: vk_mem::AllocatorView,
  command_buffer: vk::CommandBuffer,
  staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
  data: &[T],
  usage: vk::BufferUsageFlags,
  debug_name: &str,
) -> GpuResult<Buffer> {
  let buffer_size = (core::mem::size_of::<T>() * data.len()) as vk::DeviceSize;
  if buffer_size == 0 {
    return Err(crate::gpu_invalid_arg!("invalid argument"));
  }

  // 1. Allocate from staging arena
  let (staging_offset, staging_ptr) = staging_arena
    .allocate(buffer_size as usize, core::mem::align_of::<T>())
    .ok_or(crate::gpu_err_device!())?;

  // 2. Create device buffer (GPU-local).
  let (device_buffer, device_allocation) = {
    let device_buffer_info = vk::BufferCreateInfo::default()
      .size(buffer_size)
      .usage(usage | vk::BufferUsageFlags::TRANSFER_DST);
    let device_alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::Auto,
      flags: vk_mem::AllocationCreateFlags::DEDICATED_MEMORY,
      preferred_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
      ..Default::default()
    };
    crate::apply_test_dedicated_alloc!(device_alloc_info);
    let (device_buffer, device_alloc) =
      unsafe { allocator.create_buffer(&device_buffer_info, &device_alloc_info) }
        .map_err(|_| crate::gpu_err!())?;
    aethervk_oshal_rlib::log!("VMA CREATED BUFFER IN RESOURCES: {:?}", device_buffer);
    Ok((device_buffer, device_alloc))
      .with_name(device, &alloc::format!("VkBuffer_{}", debug_name))?
  };

  // 3. Copy data to staging buffer
  unsafe {
    core::ptr::copy_nonoverlapping(data.as_ptr(), staging_ptr as *mut T, data.len());
  }
  // Arena memory is HOST_COHERENT.

  // 4. Record copy command
  let copy_region = vk::BufferCopy::default()
    .src_offset(staging_offset as vk::DeviceSize)
    .dst_offset(0)
    .size(buffer_size);
  unsafe {
    device.cmd_copy_buffer(
      command_buffer,
      staging_arena.buffer,
      device_buffer,
      &[copy_region],
    );
  }

  // 5. Insert a pipeline barrier to synchronize
  let (dst_stage, dst_access) = if usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
    (
      vk::PipelineStageFlags::VERTEX_INPUT,
      vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
    )
  } else if usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
    (
      vk::PipelineStageFlags::VERTEX_INPUT,
      vk::AccessFlags::INDEX_READ,
    )
  } else {
    (
      vk::PipelineStageFlags::TOP_OF_PIPE,
      vk::AccessFlags::empty(),
    )
  };

  if dst_access != vk::AccessFlags::empty() {
    let buffer_barrier = vk::BufferMemoryBarrier::default()
      .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
      .dst_access_mask(dst_access)
      .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
      .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
      .buffer(device_buffer)
      .offset(0)
      .size(buffer_size);

    unsafe {
      device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[],
        &[buffer_barrier],
        &[],
      );
    }
  }

  Ok(Buffer {
    buffer: unsafe { NonZeroHandle::new_unchecked(device_buffer) },
    allocation: device_allocation,
  })
}

impl ArchetypeArenaCreate for ForwardMesh2RenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let staging_arena = ctx.staging_arena.unwrap();
    let queue = &ctx.queue.unwrap();
    const NEVER_DISCARD_TIMELINE: u64 = u64::MAX;

    // --------------------------- 1. Descriptor Sets -------------------------------------------
    let bindings = {
      let mut binding = 0;
      let mut make_descriptor = || -> vk::DescriptorSetLayoutBinding<'_> {
        let b = binding;
        binding += 1;
        vk::DescriptorSetLayoutBinding::default()
          .binding(b)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .descriptor_count(1)
          .stage_flags(vk::ShaderStageFlags::FRAGMENT)
      };
      [
        make_descriptor(),
        make_descriptor(),
        make_descriptor(),
        make_descriptor(),
        make_descriptor(),
        make_descriptor(),
      ]
    };

    let ds_layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let ds_layout = unsafe { device.create_descriptor_set_layout(&ds_layout_info, None) }?;
    ctx
      .rollback
      .defer(move |d| unsafe { d.destroy_descriptor_set_layout(ds_layout, None) });

    // --------------------------- 2. Push Constants --------------------------------------------
    let push_constant_range = vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<gpu::PhysicalMesh2PushConstants>() as _);

    // --------------------------- 3. Pipeline Layout -------------------------------------------
    let layout_create_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(core::slice::from_ref(&ds_layout))
      .push_constant_ranges(core::slice::from_ref(&push_constant_range));

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }
      .with_name(
        device,
        "VkPipelineLayout_ForwardMesh2RenderResourceArchetype",
      )?;

    ctx
      .rollback
      .defer(move |dev| unsafe { dev.destroy_pipeline_layout(pipeline_layout, None) });

    let mut tex = Texture::default();
    tex.data = bytes::Bytes::from_static(&[0, 0, 0, 1]);
    tex.width = 1;
    tex.height = 1;
    tex.format = crate::simulation::comet::TexelFormat::R8G8B8A8_UNORM;

    let cmd_pool_info = vk::CommandPoolCreateInfo::default()
      .queue_family_index(queue.family_index)
      .flags(vk::CommandPoolCreateFlags::TRANSIENT);

    let temp_cmd_pool = match unsafe { device.create_command_pool(&cmd_pool_info, None) } {
      Ok(pool) => pool,
      Err(e) => {
        aethervk_oshal_rlib::log!("ERROR: create_command_pool failed: {:?}", e);
        return Err(e.into());
      }
    };

    let alloc_info = vk::CommandBufferAllocateInfo::default()
      .command_pool(temp_cmd_pool)
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_buffer_count(1);

    let cmd = match unsafe { device.allocate_command_buffers(&alloc_info) } {
      Ok(bufs) => bufs[0],
      Err(e) => {
        aethervk_oshal_rlib::log!("ERROR: allocate_command_buffers failed: {:?}", e);
        return Err(e.into());
      }
    };

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    if let Err(e) = unsafe { device.begin_command_buffer(cmd, &begin_info) } {
      aethervk_oshal_rlib::log!("ERROR: begin_command_buffer failed: {:?}", e);
      return Err(e.into());
    }

    let dummy_texture_handle = match Image::new_2d(
      device,
      allocator,
      cmd,
      staging_arena,
      &tex,
      vk::ImageUsageFlags::SAMPLED,
      "DummyPhysicalMesh2",
    ) {
      Ok(img) => img,
      Err(e) => {
        aethervk_oshal_rlib::log!("ERROR: Image::new_2d failed: {:?}", e);
        unsafe {
          device.destroy_command_pool(temp_cmd_pool, None);
        }
        return Err(e);
      }
    };

    if let Err(e) = unsafe { device.end_command_buffer(cmd) } {
      aethervk_oshal_rlib::log!("ERROR: end_command_buffer failed: {:?}", e);
      return Err(e.into());
    }

    let submit_info = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&cmd));
    let fence = match unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) } {
      Ok(f) => f,
      Err(e) => {
        aethervk_oshal_rlib::log!("ERROR: create_fence failed: {:?}", e);
        return Err(e.into());
      }
    };

    if let Err(e) =
      device.locked_queue_submit(queue.handle, core::slice::from_ref(&submit_info), fence)
    {
      aethervk_oshal_rlib::log!("ERROR: locked_queue_submit failed: {:?}", e);
      return Err(e.into());
    }

    if let Err(e) = unsafe { device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX) }
    {
      aethervk_oshal_rlib::log!("ERROR: wait_for_fences failed: {:?}", e);
      return Err(e.into());
    }

    unsafe {
      device.destroy_fence(fence, None);
      device.free_command_buffers(temp_cmd_pool, core::slice::from_ref(&cmd));
      device.destroy_command_pool(temp_cmd_pool, None);
    }

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      descriptor_set_layout: ds_layout,
      push_constant_range,
      dummy_texture_handle,
      allocator_raw: allocator,
    })
  }
}

/// To be destroyed before descriptor pool
pub(super) struct ForwardMesh2RenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layout: vk::DescriptorSetLayout,
  pub push_constant_range: vk::PushConstantRange,

  pub dummy_texture_handle: Image,
  /// Necessary evil for discard. assumes it outlives this object
  allocator_raw: vk_mem::AllocatorView,
}

pub(super) struct ForwardMesh2RenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      ForwardMesh2RenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
  pub outline_pipeline_key: PipelineKey,
  pub outline_graphics_info: GraphicsInfo,
}

impl ForwardMesh2RenderResourceArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

impl ForwardMesh2RenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_image_view(self.dummy_texture_handle.image_view.get(), timeline);
    aethervk_oshal_rlib::log!(
      "[forward mesh discard] Discarding image {:#X} | alloc {:#X}",
      self.dummy_texture_handle.image.get().as_raw(),
      self.dummy_texture_handle.allocation.get_raw() as u64,
    );
    discard_pool.discard_image(
      self.allocator_raw,
      self.dummy_texture_handle.image.get(),
      self.dummy_texture_handle.allocation,
      timeline,
    );
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout, timeline);
  }
}

fn map_descriptor_type(
  reflect_type: spirv_reflect::types::ReflectDescriptorType,
) -> GpuResult<vk::DescriptorType> {
  use spirv_reflect::types::ReflectDescriptorType;
  Ok(match reflect_type {
    ReflectDescriptorType::Sampler => vk::DescriptorType::SAMPLER,
    ReflectDescriptorType::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    ReflectDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
    ReflectDescriptorType::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
    ReflectDescriptorType::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
    ReflectDescriptorType::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
    ReflectDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
    ReflectDescriptorType::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
    ReflectDescriptorType::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
    ReflectDescriptorType::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
    ReflectDescriptorType::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
    ReflectDescriptorType::AccelerationStructureKHR => {
      vk::DescriptorType::ACCELERATION_STRUCTURE_KHR
    }
    _ => {
      return Err(GpuError::BackendSpecific(alloc::fmt::format(format_args!(
        "Unsupported descriptor type: {:?}",
        reflect_type
      ))));
    }
  })
}

pub(super) trait DerefArchetype {
  type Target: ArchetypeArenaCreate;
  fn deref_arena(
    &self,
  ) -> Option<
    alloc::sync::Arc<crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<Self::Target>>,
  >;
}

macro_rules! impl_deref_archetype {
  ($archetype:ident, $arena:ident) => {
    impl DerefArchetype for $archetype {
      type Target = $arena;
      fn deref_arena(
        &self,
      ) -> Option<
        alloc::sync::Arc<
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<Self::Target>,
        >,
      > {
        self.arena.upgrade()
      }
    }
  };
}

impl_deref_archetype!(
  Text2RenderResourceArchetype,
  Text2RenderResourceArchetypeArena
);
impl_deref_archetype!(
  MeasurementRenderResourceArchetype,
  MeasurementRenderResourceArchetypeArena
);
impl_deref_archetype!(
  MarkerRenderResourceArchetype,
  MarkerRenderResourceArchetypeArena
);
impl_deref_archetype!(UiRenderResourceArchetype, UiRenderResourceArchetypeArena);
impl_deref_archetype!(
  TrajectoryRenderResourceArchetype,
  TrajectoryRenderResourceArchetypeArena
);
impl_deref_archetype!(
  BillboardRenderResourceArchetype,
  BillboardRenderResourceArchetypeArena
);
impl_deref_archetype!(
  GizmoRenderResourceArchetype,
  GizmoRenderResourceArchetypeArena
);
impl_deref_archetype!(
  CursorRenderResourceArchetype,
  CursorRenderResourceArchetypeArena
);
impl_deref_archetype!(SkyRenderResourceArchetype, SkyRenderResourceArchetypeArena);
impl_deref_archetype!(
  BackgroundRenderResourceArchetype,
  BackgroundRenderResourceArchetypeArena
);
impl_deref_archetype!(
  GridRenderResourceArchetype,
  GridRenderResourceArchetypeArena
);
impl_deref_archetype!(SunRenderResourceArchetype, SunRenderResourceArchetypeArena);
impl_deref_archetype!(
  ForwardMesh2RenderResourceArchetype,
  ForwardMesh2RenderResourceArchetypeArena
);

pub struct PrepareFontUpload {
  pub is_already_uploaded: bool,
  pub descriptor_index: u32,
  pub descriptor_set: Option<vk::DescriptorSet>,
  pub font_sampler: Option<vk::Sampler>,
}

pub struct PrepareFontRemove {
  pub descriptor_index: u32,
  pub image_view: vk::ImageView,
  pub image: vk::Image,
  pub allocation: vk_mem::Allocation,
}

macro_rules! impl_font_atlas_arena_transactional {
  ($arena_type:ident) => {
    impl $arena_type {
      /// Step 1: Lock state, resolve descriptor indices, and return parameters for execution
      #[named]
      pub fn prepare_upload_font_atlas(&mut self, font_hash: u64) -> GpuResult<PrepareFontUpload> {
        if let Some(existing) = self.uploaded_fonts.get(&font_hash) {
          return Ok(PrepareFontUpload {
            is_already_uploaded: true,
            descriptor_index: existing.descriptor_index,
            descriptor_set: None,
            font_sampler: None,
          });
        }

        let descriptor_index = if let Some(idx) = self.free_descriptor_indices.pop() {
          idx
        } else if self.next_descriptor_index < self.max_fonts {
          let idx = self.next_descriptor_index;
          self.next_descriptor_index += 1;
          idx
        } else {
          return Err(crate::gpu_err!(
            "Exceeded descriptor array layout maximum capacity"
          ));
        };

        Ok(PrepareFontUpload {
          is_already_uploaded: false,
          descriptor_index,
          descriptor_set: self.descriptor_set,
          font_sampler: self.font_sampler,
        })
      }

      /// Step 2: Lock-free execution of Vulkan creation and descriptor writes
      #[named]
      pub fn execute_upload_font_atlas(
        device: &crate::gpu_backends::vulkan::device::LogicalDevice,
        allocator: vk_mem::AllocatorView,
        command_buffer: vk::CommandBuffer,
        staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
        texture: &crate::simulation::comet::Texture,
        prep: &PrepareFontUpload,
        debug_name: &str,
        rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
      ) -> GpuResult<Image> {
        let image = Image::new_2d(
          device,
          allocator,
          command_buffer,
          staging_arena,
          texture,
          vk::ImageUsageFlags::SAMPLED,
          debug_name,
        )?;

        let img_h = image.image.get();
        let view_h = image.image_view.get();
        let mut alloc_h = image.allocation;

        rollback.defer(move |dev| unsafe {
          dev.destroy_image_view(view_h, None);
          allocator.destroy_image(img_h, &mut alloc_h);
        });

        if let (Some(sampler), Some(set)) = (prep.font_sampler, prep.descriptor_set) {
          let image_info = [vk::DescriptorImageInfo::default()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(view_h)
            .sampler(sampler)];

          let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .dst_array_element(prep.descriptor_index)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info);

          unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        Ok(image)
      }

      /// Step 3: Write lock arena to map the newly allocated resource handle
      pub fn commit_upload_font_atlas(
        &mut self,
        font_hash: u64,
        atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
        texture: Image,
        descriptor_index: u32,
      ) {
        self.uploaded_fonts.insert(
          font_hash,
          UploadedFont {
            texture,
            atlas,
            descriptor_index,
          },
        );
      }

      /// Step 1 for Remove: Extract handles and free descriptor index
      #[named]
      pub fn prepare_remove_font_atlas(&mut self, font_hash: u64) -> GpuResult<PrepareFontRemove> {
        if let Some(uploaded) = self.uploaded_fonts.remove(&font_hash) {
          // Push index back as available
          self.free_descriptor_indices.push(uploaded.descriptor_index);

          Ok(PrepareFontRemove {
            descriptor_index: uploaded.descriptor_index,
            image_view: uploaded.texture.image_view.get(),
            image: uploaded.texture.image.get(),
            allocation: uploaded.texture.allocation,
          })
        } else {
          Err(crate::gpu_invalid_arg!("atlas not found: {}", font_hash))
        }
      }

      /// Step 2 for Remove: Record into the lock-free generic discard pool
      pub fn execute_remove_font_atlas(
        prep: &PrepareFontRemove,
        discard_pool: &DiscardPool,
        allocator_raw: vk_mem::AllocatorView,
        timeline: u64,
      ) {
        discard_pool.discard_image_view(prep.image_view, timeline);
        discard_pool.discard_image(allocator_raw, prep.image, prep.allocation, timeline);
      }
    }
  };
}

impl_font_atlas_arena_transactional!(Text2RenderResourceArchetypeArena);

/// Archetype for `dust.vert/frag` shaders
pub(super) struct DustRenderArchetype {
  pub arena: alloc::sync::Weak<DebugTrackedRwLock<DustRenderArchetypeArena>>,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl DustRenderArchetype {
  pub fn discard(&mut self, _device: &LogicalDevice, _pool: &DiscardPool, _timeline: u64) {
    // Purposefully do nothing.
  }
}

/// Archetype arena for `dust.vert/frag` shaders
pub(super) struct DustRenderArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  // no descriptor sets
}

impl ArchetypeArenaCreate for DustRenderArchetypeArena {
  fn new_arena(ctx: &mut ArenaCreationContext) -> GpuResult<Self> {
    let push_constant_range = vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<gpu::new_particles::DustPushConstants>() as _);
    let create_info = vk::PipelineLayoutCreateInfo::default()
      .push_constant_ranges(core::slice::from_ref(&push_constant_range));

    // create pipeline layout
    let pipeline_layout = unsafe {
      ctx
        .device
        .create_pipeline_layout(&create_info, None)
        .with_name(ctx.device, "VkPipelineLayout_DustRenderArchetypeArena")?
    };
    ctx.rollback.defer(move |dev| unsafe {
      dev.destroy_pipeline_layout(pipeline_layout, None);
    });

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
    })
  }
}

impl DustRenderArchetypeArena {
  pub fn discard(&mut self, _device: &LogicalDevice, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
  }
}

impl_deref_archetype!(DustRenderArchetype, DustRenderArchetypeArena);