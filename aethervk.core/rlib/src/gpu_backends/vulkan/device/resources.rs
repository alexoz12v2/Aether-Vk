//! resources module.

use crate::{
  gpu::vulkan::device::{DebugTrackedRwLock, LogicalDevice},
  gpu_backends::vulkan::{
    self,
    device::{VmaDebugNameExt, VulkanDebugNameExt},
  },
  simulation::comet::Texture,
};
use aethervk_oshal_rlib as oshal;
use alloc::{boxed::Box, collections::VecDeque, sync, vec::Vec};
use ash::{vk, vk::Handle};
use core::{
  hash::{Hash, Hasher},
  ptr,
  sync::atomic::AtomicU32,
};
use function_name::named;
use oshal::{hash::FnvHasher, os::native::ThreadId};
use spirv_reflect::{
  ffi::SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS, types::ReflectShaderStageFlags,
};
use static_assertions as sa;
use vk_mem::Alloc;

use crate::{
  gpu::{PipelineKey, TextureFlags},
  gpu_backends::vulkan::{
    device::{
      DeviceResource, DeviceResourceJanitor, FunctionalDeviceResource,
      commands::{self, CommandBufferId},
      descriptors::{self, DescriptorPools},
      pipelines::GraphicsInfo,
    },
    utils::NonZeroHandle,
  },
  types::{GpuError, GpuResult},
};

#[derive(Clone, Debug)]
pub enum ResourceState<T> {
  Pending,
  Ready(T),
}

pub(crate) struct ArenaCreationContext<'a> {
  pub device: &'a crate::gpu_backends::vulkan::device::LogicalDevice,
  pub allocator: vk_mem::AllocatorView,
  pub discard_pool: &'a DiscardPool,
  pub queue: Option<&'a crate::gpu_backends::vulkan::device::Queue>,
  pub staging_arena: Option<&'a crate::gpu_backends::vulkan::device::memory::FrameStagingArena>,
  pub vertex_shader: Option<&'a crate::gpu_backends::vulkan::device::shader_manager::Shader>,
  pub fragment_shader: Option<&'a crate::gpu_backends::vulkan::device::shader_manager::Shader>,
  pub outline_vertex_shader:
    Option<&'a crate::gpu_backends::vulkan::device::shader_manager::Shader>,
  pub outline_fragment_shader:
    Option<&'a crate::gpu_backends::vulkan::device::shader_manager::Shader>,
}

impl<'a> ArenaCreationContext<'a> {
  #[inline]
  #[track_caller]
  pub fn validate_push_constant_size(&self, rust_size: u32) {
    let mut spv_size = 0;

    let mut check_shader = |shader: &crate::gpu_backends::vulkan::device::shader_manager::Shader| {
      let pcs = shader.spv_module.enumerate_push_constant_blocks(None).unwrap_or_default();
      if let Some(pc_block) = pcs.first() {
        spv_size = spv_size.max(pc_block.size);
      }
    };

    if let Some(s) = self.vertex_shader { check_shader(s); }
    if let Some(s) = self.fragment_shader { check_shader(s); }
    if let Some(s) = self.outline_vertex_shader { check_shader(s); }
    if let Some(s) = self.outline_fragment_shader { check_shader(s); }

    if spv_size > 0 {
      debug_assert_eq!(
        rust_size, spv_size,
        "Push constant size mismatch! Rust struct = {} bytes, SPIR-V reflection = {} bytes",
        rust_size, spv_size
      );
    }
  }
}


pub trait ArchetypeArenaCreate {
  fn new_arena(ctx: &ArenaCreationContext) -> crate::types::GpuResult<Self>
  where
    Self: Sized;
}

/// TODO: Document this item
pub struct TimelineQueue<T> {
  items: VecDeque<(u64, T)>,
}

impl<T> TimelineQueue<T> {
  /// TODO: Document this item
  pub fn with_capacity(cap: usize) -> Self {
    Self {
      items: VecDeque::with_capacity(cap),
    }
  }

  /// TODO: Document this item
  pub fn push(&mut self, timeline: u64, item: T) {
    let mut i = self.items.len();
    while i > 0 && self.items[i - 1].0 > timeline {
      i -= 1;
    }
    self.items.insert(i, (timeline, item));
  }

  /// TODO: Document this item
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
  // TODO other types of resources as needed
  /// Placeholder to use any cleanable resource. Slower than having a specialized type
  GenericHandle(Box<dyn DeviceResource>),
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
  allocator: vk_mem::ffi::VmaAllocator, // non owning copy
}
pub(crate) struct ImageDiscard {
  image: vk::Image,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::ffi::VmaAllocator, // non owning copy
}
pub(crate) struct CmdBufDiscard {
  thread_id: ThreadId,
  command_buffer: vk::CommandBuffer,
  manager: sync::Arc<commands::CommandPools>,
  id: CommandBufferId,
}

/// TODO: Document this item
pub trait DiscardableResource {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64);
}

/// Structure associated to the main Timeline Semaphore provided by Device
/// Note: this must not outlive device, hence don't expose it outside
pub struct DiscardPool {
  items: crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex<TimelineQueue<DiscardItem>>,
  #[cfg(debug_assertions)]
  queued_handles:
    crate::gpu_backends::vulkan::device::locks::DebugTrackedMutex<hashbrown::HashSet<(u8, u64)>>,
}

unsafe impl Sync for DiscardPool {}
unsafe impl Send for DiscardPool {}

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

  /// TODO: Document this item
  pub fn discard_type_erased<T: DeviceResource + 'static>(&self, item: T, timeline: u64) {
    aethervk_oshal_rlib::log!("Queuing type_erased discard at timeline {}", timeline);
    self.push_item(timeline, DiscardItem::GenericHandle(Box::new(item)));
  }

  // TODO all other types of resources as needed
  /// TODO: Document this item
  pub fn discard_render_pass(&self, render_pass: vk::RenderPass, timeline: u64) {
    self.push_item(timeline, DiscardItem::RenderPass(render_pass));
  }

  /// TODO: Document this item
  pub fn discard_framebuffer(&self, framebuffer: vk::Framebuffer, timeline: u64) {
    self.push_item(timeline, DiscardItem::Framebuffer(framebuffer));
  }

  /// TODO: Document this item
  pub fn discard_fence(&self, fence: vk::Fence, timeline: u64) {
    self.push_item(timeline, DiscardItem::Fence(fence));
  }

  /// TODO: Document this item
  pub fn discard_semaphore(&self, semaphore: vk::Semaphore, timeline: u64) {
    self.push_item(timeline, DiscardItem::Semaphore(semaphore));
  }

  /// TODO: Document this item
  pub fn discard_buffer(
    &self,
    allocator: vk_mem::ffi::VmaAllocator,
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

  /// TODO: Document this item
  pub fn discard_command_buffer(
    &self,
    thread_id: ThreadId,
    command_buffer_id: CommandBufferId,
    command_buffer: vk::CommandBuffer,
    manager: sync::Arc<commands::CommandPools>,
    timeline: u64,
  ) {
    self.push_item(
      timeline,
      DiscardItem::CommandPool(CmdBufDiscard {
        thread_id,
        command_buffer,
        manager,
        id: command_buffer_id,
      }),
    );
  }

  /// TODO: Document this item
  pub fn discard_image(
    &self,
    allocator: vk_mem::ffi::VmaAllocator,
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

  /// TODO: Document this item
  pub fn discard_image_view(&self, image_view: vk::ImageView, timeline: u64) {
    self.push_item(timeline, DiscardItem::ImageView(image_view));
  }

  /// TODO: Document this item
  pub fn discard_descriptor_set_layout(&self, layout: vk::DescriptorSetLayout, timeline: u64) {
    self.push_item(timeline, DiscardItem::DescriptorSetLayout(layout));
  }

  /// TODO: Document this item
  pub fn discard_descriptor_pool(
    &self,
    pool: vk::DescriptorPool,
    manager: sync::Arc<descriptors::DescriptorPools>,
    timeline: u64,
  ) {
    self.push_item(timeline, DiscardItem::DescriptorPool(pool, manager));
  }

  /// TODO: Document this item
  pub fn discard_pipeline(&self, pipeline: vk::Pipeline, timeline: u64) {
    self.push_item(timeline, DiscardItem::Pipeline(pipeline));
  }

  /// TODO: Document this item
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
    device: &ash::Device,
    items: impl IntoIterator<Item = DiscardItem>,
  ) {
    for item in items {
      match item {
        DiscardItem::Buffer(BufferDiscard {
          buffer,
          alloc,
          allocator,
        }) => unsafe {
          aethervk_oshal_rlib::log!(
            "DiscardItem::Buffer destroying buffer! alloc: {:?}",
            alloc.get_raw()
          );
          vk_mem::ffi::vmaDestroyBuffer(allocator, buffer, alloc.get_raw());
          core::mem::forget(alloc);
        },
        DiscardItem::Image(ImageDiscard {
          image,
          alloc,
          allocator,
        }) => unsafe {
          vk_mem::ffi::vmaDestroyImage(allocator, image, alloc.get_raw());
          core::mem::forget(alloc);
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
        }) => {
          let _x = manager.recycle(thread_id, id, command_buffer);
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
  pub fn destroy_discarded_resources_all(&self, device: &ash::Device) {
    let items = self.pop_ready_items(u64::MAX);
    aethervk_oshal_rlib::log!(
      "destroy_discarded_resources_all popping {} items",
      items.len()
    );
    Self::destroy_items_lock_free(device, items);
  }
}

impl super::DeviceResource for DiscardPool {
  fn cleanup(&mut self, device: &ash::Device) {
    self.destroy_discarded_resources_all(device);
  }
}

/// TODO: Document this item
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

/// TODO: Document this item
#[derive(Clone)]
pub(super) struct Image {
  pub image: NonZeroHandle<vk::Image>,
  pub image_view: NonZeroHandle<vk::ImageView>,
  pub allocation: vk_mem::Allocation,
}

impl Image {
  /// TODO: Document this item
  pub fn to_descriptor_image_info(
    &self,
    sampler: NonZeroHandle<vk::Sampler>,
  ) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo::default()
      .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
      .image_view(self.image_view.get())
      .sampler(sampler.get())
  }

  /// TODO: Document this item
  #[named]
  pub fn new_storage_2d(
    device: &vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    width: u32,
    height: u32,
    format: vk::Format,
    graphics_queue_family: u32,
    compute_queue_family: u32,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let mut sharing_mode = vk::SharingMode::EXCLUSIVE;
    let queue_family_indices = [graphics_queue_family, compute_queue_family];
    let queue_count = if graphics_queue_family != compute_queue_family {
      sharing_mode = vk::SharingMode::CONCURRENT;
      2
    } else {
      1
    };

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
      .sharing_mode(sharing_mode)
      .queue_family_indices(&queue_family_indices[..queue_count])
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

  /// TODO: Document this item
  #[named]
  pub fn new_paint_image(
    device: &vulkan::device::LogicalDevice,
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

  /// TODO: Document this item
  #[named]
  pub fn new_storage_3d(
    device: &vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    width: u32,
    height: u32,
    depth: u32,
    format: vk::Format,
    graphics_queue_family: u32,
    compute_queue_family: u32,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let mut sharing_mode = vk::SharingMode::EXCLUSIVE;
    let queue_family_indices = [graphics_queue_family, compute_queue_family];
    let queue_count = if graphics_queue_family != compute_queue_family {
      sharing_mode = vk::SharingMode::CONCURRENT;
      2
    } else {
      1
    };

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
      .sharing_mode(sharing_mode)
      .queue_family_indices(&queue_family_indices[..queue_count])
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

  /// TODO: Document this item
  #[named]
  pub fn new_2d(
    device: &vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    command_buffer: vk::CommandBuffer,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
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

#[repr(C)]
/// TODO: Document this item
pub(super) struct ForwardMeshRenderResourcePushData {
  pub model_view_projection: [f32; 16],
  pub model: [f32; 16],
  pub sun_pos: [f32; 3],
  pub texture_flags: TextureFlags,
  pub sun_color: [f32; 4],
  pub camera_pos: [f32; 3],
  pub emissive_intensity: f32,
  pub emissive_color: [f32; 3],
  pub _padding: f32,
}
sa::const_assert!(core::mem::size_of::<ForwardMeshRenderResourcePushData>() == 192);

impl Default for ForwardMeshRenderResourcePushData {
  fn default() -> Self {
    Self {
      model_view_projection: Default::default(),
      model: Default::default(),
      sun_pos: Default::default(),
      texture_flags: TextureFlags::empty(),
      sun_color: Default::default(),
      camera_pos: Default::default(),
      emissive_intensity: 0.0,
      emissive_color: Default::default(),
      _padding: 0.0,
    }
  }
}

/// TODO: Document this item
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

impl SunRenderResource {
  pub fn discard(
    &mut self,
    device: &ash::Device,
    allocator: vk_mem::AllocatorView,
    discard_pool: &DiscardPool,
    frame_timeline: u64,
  ) {
    if let Some(mut img) = self.image.take() {
      discard_pool.discard_image_view(img.image_view.get(), frame_timeline);
      discard_pool.discard_image(
        allocator.get_raw(),
        img.image.get(),
        img.allocation,
        frame_timeline,
      );
    }
    if let Some(layout) = self.compute_pipeline_layout.take() {
      discard_pool.discard_pipeline_layout(layout, frame_timeline);
    }
    if let Some(pool) = self.compute_descriptor_pool.take() {
      discard_pool.discard_type_erased(
        crate::gpu_backends::vulkan::device::resources::FunctionalDeviceResource::new(
          pool,
          |pool, device| unsafe {
            device.destroy_descriptor_pool(pool, None);
          },
        ),
        frame_timeline,
      );
    }
    if let Some(set_layout) = self.compute_descriptor_set_layout.take() {
      discard_pool.discard_descriptor_set_layout(set_layout, frame_timeline);
    }
    if let Some(buf) = self.params_buffer.take() {
      if let Some(alloc) = self.params_alloc.take() {
        discard_pool.discard_buffer(allocator.get_raw(), buf, alloc, frame_timeline);
      }
    }
  }
}

/// TODO: Document this item
#[derive(Clone)]
pub(super) struct ForwardMeshRenderResource {
  pub allocator: vk_mem::ffi::VmaAllocator, // necessary evil. TODO: Edit DeviceResource trait and remove this.
  pub position_vertex_buffer: Buffer,
  pub attributes_vertex_buffer: Buffer,
  pub index_buffer: Buffer,
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
  /// Note: Purposefully leaked! (TODO: if this creates problems, do better.)
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,
}

/// TODO: Document this item
#[derive(Clone)]
pub(super) struct ForwardMesh2RenderResource {
  pub allocator: vk_mem::ffi::VmaAllocator, // necessary evil. TODO: Edit DeviceResource trait and remove this.
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

unsafe impl Sync for ForwardMesh2RenderResource {}
unsafe impl Send for ForwardMesh2RenderResource {}

impl DiscardableResource for ForwardMesh2RenderResource {
  fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
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
}

impl ForwardMesh2RenderResource {
  /// TODO: Document this item
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

  /// TODO: Document this item
  pub fn buffers_hash(&self) -> u64 {
    let mut hasher = FnvHasher::new();
    self.position_vertex_buffer.hash(&mut hasher);
    self.attributes_vertex_buffer.hash(&mut hasher);
    self.index_buffer.hash(&mut hasher);
    hasher.finish()
  }

  /// TODO: Document this item
  #[allow(clippy::too_many_arguments)]
  #[named]
  pub(super) unsafe fn new(
    device: &vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    command_buffer: vk::CommandBuffer,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    position_data: &[f32],
    attribute_data: &[f32],
    index_data: &[u32],
    material_data: &crate::gpu::MaterialData,
    object_data: &crate::gpu::ObjectData,
    albedo_image: Option<Image>,
    normal_image: Option<Image>,
    roughness_image: Option<Image>,
    ao_image: Option<Image>,
    sky_image: Option<Image>,
    emissive_paint_image: Option<Image>,
    sampler: NonZeroHandle<vk::Sampler>,
    descriptor_set: NonZeroHandle<vk::DescriptorSet>,
    dummy_texture: &Image,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let mut janitor = DeviceResourceJanitor::<'_, 9>::new(device);
    let vma_allocator = allocator.get_raw();

    // Create position buffer
    let position_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      position_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
      &alloc::format!("PositionBuffer_{}", debug_name),
    )?;
    let pos_alloc = position_vertex_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        position_vertex_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, pos_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // Create attributes buffer
    let attributes_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      attribute_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
      &alloc::format!("AttributesBuffer_{}", debug_name),
    )?;
    let attr_alloc = attributes_vertex_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        attributes_vertex_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, attr_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // Create index buffer
    let index_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      index_data,
      vk::BufferUsageFlags::INDEX_BUFFER,
      &alloc::format!("IndexBuffer_{}", debug_name),
    )?;
    let idx_alloc = index_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        index_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, idx_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    let material_slice = unsafe {
      core::slice::from_raw_parts(
        material_data as *const _ as *const u32,
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
      &alloc::format!("MaterialBuffer_{}", debug_name),
    )?;
    let mat_alloc = material_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        material_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, mat_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    let object_slice = unsafe {
      core::slice::from_raw_parts(
        object_data as *const _ as *const u32,
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
      &alloc::format!("ObjectBuffer_{}", debug_name),
    )?;
    let obj_alloc = object_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        object_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, obj_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    if let Some(image) = &albedo_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }
    if let Some(image) = &normal_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }
    if let Some(image) = &roughness_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }
    if let Some(image) = &ao_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }
    if let Some(image) = &sky_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }
    if let Some(image) = &emissive_paint_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }

    let mut image_infos = Vec::with_capacity(6);
    let dummy_info = dummy_texture.to_descriptor_image_info(sampler);
    image_infos.push((
      0,
      if let Some(image) = &albedo_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      1,
      if let Some(image) = &normal_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      2,
      if let Some(image) = &roughness_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      3,
      if let Some(image) = &ao_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      4,
      if let Some(image) = &sky_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      5,
      if let Some(image) = &emissive_paint_image {
        image.to_descriptor_image_info(sampler).image_layout(vk::ImageLayout::GENERAL)
      } else {
        dummy_info
      },
    ));
    let write_descriptor_sets: Vec<_> = image_infos
      .iter()
      .map(|(binding, info)| {
        vk::WriteDescriptorSet::default()
          .dst_set(descriptor_set.get())
          .dst_binding(*binding)
          .dst_array_element(0)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .image_info(core::slice::from_ref(info))
      })
      .collect();

    unsafe {
      device.update_descriptor_sets(&write_descriptor_sets, &[]);
    }

    janitor.clear();

    Ok(Self {
      allocator: vma_allocator,
      position_vertex_buffer,
      attributes_vertex_buffer,
      index_buffer,
      material_buffer,
      object_buffer,
      albedo_image,
      normal_image,
      roughness_image,
      ao_image,
      sky_image,
      emissive_paint_image,
      descriptor_set,
    })
  }
}

unsafe impl Sync for ForwardMeshRenderResource {}
unsafe impl Send for ForwardMeshRenderResource {}

impl DiscardableResource for ForwardMeshRenderResource {
  fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
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
  }
}

impl ForwardMeshRenderResource {
  /// TODO: Document this item
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
    flags
  }

  /// TODO: Document this item
  pub fn buffers_hash(&self) -> u64 {
    let mut hasher = FnvHasher::new();
    self.position_vertex_buffer.hash(&mut hasher);
    self.attributes_vertex_buffer.hash(&mut hasher);
    self.index_buffer.hash(&mut hasher);
    hasher.finish()
  }

  /// Safety:
  /// - `descriptor_set` should have been allocated with archetype descriptor set and
  /// match the given arguments
  /// - `sampler` should outlive this object
  #[allow(clippy::too_many_arguments)]
  #[named]
  pub(super) unsafe fn new(
    device: &vulkan::device::LogicalDevice,
    allocator: vk_mem::AllocatorView,
    command_buffer: vk::CommandBuffer,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    position_data: &[f32],
    attribute_data: &[f32],
    index_data: &[u32],
    albedo_image: Option<Image>, // Image creation is complex, pass them in for now
    normal_image: Option<Image>, //
    roughness_image: Option<Image>, //
    ao_image: Option<Image>,     //
    sky_image: Option<Image>,    //
    sampler: NonZeroHandle<vk::Sampler>,
    descriptor_set: NonZeroHandle<vk::DescriptorSet>,
    dummy_texture: &Image,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let mut janitor = DeviceResourceJanitor::<'_, 7>::new(device);
    let vma_allocator = allocator.get_raw();

    // Create position buffer
    let position_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      position_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
      &alloc::format!("PositionBuffer_{}", debug_name),
    )?;
    let pos_alloc = position_vertex_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        position_vertex_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, pos_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // Create attributes buffer
    let attributes_vertex_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      attribute_data,
      vk::BufferUsageFlags::VERTEX_BUFFER,
      &alloc::format!("AttributesBuffer_{}", debug_name),
    )?;
    let attr_alloc = attributes_vertex_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        attributes_vertex_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, attr_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // Create index buffer
    let index_buffer = create_buffer_with_staging(
      device,
      allocator,
      command_buffer,
      staging_arena,
      index_data,
      vk::BufferUsageFlags::INDEX_BUFFER,
      &alloc::format!("IndexBuffer_{}", debug_name),
    )?;
    let idx_alloc = index_buffer.allocation;
    janitor
      .push(FunctionalDeviceResource::new(
        index_buffer.buffer.get(),
        move |h, _| unsafe {
          vk_mem::ffi::vmaDestroyBuffer(vma_allocator, h, idx_alloc.get_raw());
        },
      ))
      .map_err(|s| GpuError::BackendSpecific(s.into()))?;

    // For images, we are still passing them in, but if they were created here,
    // they would also be pushed to the janitor.
    if let Some(image) = &albedo_image {
      let alloc = image.allocation;
      janitor
        .push(FunctionalDeviceResource::new(
          image.image.get(),
          move |h, _| unsafe {
            vk_mem::ffi::vmaDestroyImage(vma_allocator, h, alloc.get_raw());
          },
        ))
        .map_err(|s| GpuError::BackendSpecific(s.into()))?;
    }
    // ... repeat for other optional images ...

    // Now create descriptor set for these images.
    let mut image_infos = Vec::with_capacity(5);
    let dummy_info = dummy_texture.to_descriptor_image_info(sampler);
    image_infos.push((
      0,
      if let Some(image) = &albedo_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      1,
      if let Some(image) = &normal_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      2,
      if let Some(image) = &roughness_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      3,
      if let Some(image) = &ao_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    image_infos.push((
      4,
      if let Some(image) = &sky_image {
        image.to_descriptor_image_info(sampler)
      } else {
        dummy_info
      },
    ));
    let write_descriptor_sets: Vec<_> = image_infos
      .iter()
      .map(|(binding, info)| {
        vk::WriteDescriptorSet::default()
          .dst_set(descriptor_set.get())
          .dst_binding(*binding)
          .dst_array_element(0)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .image_info(core::slice::from_ref(info))
      })
      .collect();

    unsafe {
      device.update_descriptor_sets(&write_descriptor_sets, &[]);
    }

    // Everything was created successfully. Defuse the janitor.
    janitor.clear();

    Ok(Self {
      allocator: vma_allocator,
      position_vertex_buffer,
      attributes_vertex_buffer,
      index_buffer,
      albedo_image,
      normal_image,
      roughness_image,
      ao_image,
      sky_image,
      descriptor_set,
    })
  }
}

/// Structure which is built up per frame and then discarded on submission
/// It holds the vulkan-backend specific draw call data
/// Each frame end, all [`FrameResource`]s are discarded through the [`DiscardableResource`] trait
pub(super) enum FrameResource {
  ForwardMeshRenderResource(ForwardMeshRenderResource),
  ForwardMesh2RenderResource(ForwardMesh2RenderResource),
}

impl DiscardableResource for FrameResource {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    match self {
      Self::ForwardMeshRenderResource(resource) => {
        resource.discard(device, discard_pool, timeline);
      }
      Self::ForwardMesh2RenderResource(resource) => {
        resource.discard(device, discard_pool, timeline);
      }
    }
  }
}

/// TODO: Document this item
pub struct UploadedFont {
  pub texture: Image,
  pub atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
  pub descriptor_index: u32, // The assigned array element
}

/// TODO: Document this item
pub(super) struct TextRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub descriptor_pool: Option<NonZeroHandle<vk::DescriptorPool>>,
  pub descriptor_set: Option<vk::DescriptorSet>,
  pub font_sampler: Option<vk::Sampler>,

  // Bindless Management Map
  pub uploaded_fonts: hashbrown::HashMap<u64, UploadedFont>,
  // TODO substitute with a bitmap. max_fonts should be a function which gives the number of bits
  pub free_descriptor_indices: Vec<u32>,
  pub next_descriptor_index: u32,
  pub max_fonts: u32,

  pub allocator_raw: Option<vk_mem::ffi::VmaAllocator>,
}

pub(super) struct TextRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      TextRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

unsafe impl Sync for TextRenderResourceArchetypeArena {}
unsafe impl Sync for TextRenderResourceArchetype {}
unsafe impl Send for TextRenderResourceArchetypeArena {}
unsafe impl Send for TextRenderResourceArchetype {}

impl TextRenderResourceArchetypeArena {
  // TODO deduplicate with 2
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
  // TODO ENd deduplicate blcok

  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
    if let Some(sampler) = self.font_sampler {
      struct SamplerDiscard(vk::Sampler);
      impl DeviceResource for SamplerDiscard {
        fn cleanup(&mut self, device: &ash::Device) {
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
        fn cleanup(&mut self, device: &ash::Device) {
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
    }
  }
}

impl ArchetypeArenaCreate for TextRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;

    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers
      ),
      ctx.device.max_per_stage_descriptor_samplers
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
    let font_sampler = unsafe { device.create_sampler(&sampler_info, None) }
      .with_name(device, "Text 1 Linear Sampler")?;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::TextPushConstants>() as u32)];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(core::slice::from_ref(&set_layout))
      .push_constant_ranges(&push_constant_ranges);
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::TextPushConstants>() as u32);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

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
      allocator_raw: Some(allocator.get_raw()),
    })
  }
}
impl TextRenderResourceArchetypeArena {
  /// TODO: Document this item
  #[named]
  pub fn upload_font_atlas(
    &mut self,
    device: &vulkan::device::LogicalDevice,
    _queue: &vulkan::device::Queue,
    allocator: vk_mem::AllocatorView,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    command_buffer: vk::CommandBuffer,
    font_hash: u64,
    atlas: alloc::sync::Arc<crate::scene::text::FontAtlas>,
  ) -> GpuResult<u32> {
    // TODO write this function with a rollback (Drop) struct (started cmdbuf, ...)
    if let Some(existing) = self.uploaded_fonts.get(&font_hash) {
      return Ok(existing.descriptor_index);
    }

    // Try tracking a recycled hole before increasing the linear capacity bounds
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
      "FontAtlas Dynamic",
    )?;

    // Overwrite the specific array element natively
    if let (Some(sampler), Some(set)) = (self.font_sampler, self.descriptor_set) {
      let image_info = [vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(image.image_view.get())
        .sampler(sampler)];

      let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .dst_array_element(descriptor_index) // Writes exactly over the targeted array index
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
  pub allocator_raw: Option<vk_mem::ffi::VmaAllocator>,

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

unsafe impl Sync for Text2RenderResourceArchetypeArena {}
unsafe impl Sync for Text2RenderResourceArchetype {}
unsafe impl Send for Text2RenderResourceArchetypeArena {}
unsafe impl Send for Text2RenderResourceArchetype {}

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

  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
    if let Some(sampler) = self.font_sampler {
      struct SamplerDiscard(vk::Sampler);
      impl DeviceResource for SamplerDiscard {
        fn cleanup(&mut self, device: &ash::Device) {
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
        fn cleanup(&mut self, device: &ash::Device) {
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
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;

    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers
      ),
      ctx.device.max_per_stage_descriptor_samplers
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::Text2PushConstants>() as u32);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

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
      allocator_raw: Some(allocator.get_raw()),
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
    device: &vulkan::device::LogicalDevice,
    _queue: &vulkan::device::Queue,
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

impl Text2RenderResourceArchetype {}

/// TODO: Document this item
pub(super) struct BvhRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
}

pub(super) struct BvhRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<BvhRenderResourceArchetypeArena>,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl BvhRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
  }
}

impl ArchetypeArenaCreate for BvhRenderResourceArchetypeArena {
  /// TODO: Document this item
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let _allocator_raw = ctx.allocator.get_raw();
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::BvhPushConstants>() as u32)]; // mat4 (64) + BDA ptr (8) + pad (8) = 80 bytes

    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::BvhPushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .map_err(|e| {
        aethervk_oshal_rlib::log!("create_pipeline_layout failed: {:?}", e);
        e
      })?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
    })
  }
}

impl BvhRenderResourceArchetype {}

pub(super) struct SphereGizmoRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,

  pub data_buffer: NonZeroHandle<vk::Buffer>,
  pub data_alloc: vk_mem::Allocation,
  pub data_ptr: u64,

  pub allocated_gizmos: hashbrown::HashMap<crate::scene::EntityId, u32>,
  pub free_list: Vec<u32>,
  pub next_index: u32,

  allocator_raw: vk_mem::ffi::VmaAllocator,
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

unsafe impl Sync for SphereGizmoRenderResourceArchetypeArena {}
unsafe impl Sync for SphereGizmoRenderResourceArchetype {}
unsafe impl Send for SphereGizmoRenderResourceArchetypeArena {}
unsafe impl Send for SphereGizmoRenderResourceArchetype {}

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
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::SphereGizmoPushConstants>() as u32)];

    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);

    unsafe {
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::SphereGizmoPushConstants>() as u32);
      let pipeline_layout =
        device.create_pipeline_layout(&pipeline_layout_info, None).map_err(|e| {
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
        allocator_raw: allocator.get_raw(),
      })
    }
  }
}

pub(super) struct Bvhwire2RenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,

  pub data_buffer: NonZeroHandle<vk::Buffer>,
  pub data_alloc: vk_mem::Allocation,
  pub data_ptr: u64, // Extracted GPU pointer for Push Constants

  allocator_raw: vk_mem::ffi::VmaAllocator,
}

pub(super) struct Bvhwire2RenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      Bvhwire2RenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

unsafe impl Sync for Bvhwire2RenderResourceArchetypeArena {}
unsafe impl Sync for Bvhwire2RenderResourceArchetype {}
unsafe impl Send for Bvhwire2RenderResourceArchetypeArena {}
unsafe impl Send for Bvhwire2RenderResourceArchetype {}

impl Bvhwire2RenderResourceArchetypeArena {
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
impl ArchetypeArenaCreate for Bvhwire2RenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::Bvhwire2PushConstants>() as u32)]; // 72 bytes

    // Look! Zero set_layouts needed!
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);

    unsafe {
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::Bvhwire2PushConstants>() as u32);
      let pipeline_layout =
        device.create_pipeline_layout(&pipeline_layout_info, None).map_err(|e| {
          aethervk_oshal_rlib::log!("create_pipeline_layout failed: {:?}", e);
          e
        })?;

      // Create the Mega Buffer using DEVICE_ADDRESS flag
      let buffer_size = (100_000 * core::mem::size_of::<crate::gpu::Bvhwire2DataGpu>()) as u64;
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

      device.set_debug_name(data_buffer, "MegaBuffer_BvhData");

      // Extract raw 64-bit pointer
      let addr_info = ash::vk::BufferDeviceAddressInfo::default().buffer(data_buffer);
      let data_ptr = device.buffer_device_address.get_buffer_device_address(&addr_info);

      Ok(Self {
        pipeline_layout: NonZeroHandle::new_unchecked(pipeline_layout),
        data_buffer: NonZeroHandle::new_unchecked(data_buffer),
        data_alloc,
        data_ptr,
        allocator_raw: allocator.get_raw(),
      })
    }
  }
}

impl Bvhwire2RenderResourceArchetype {}

#[derive(Clone)]
/// TODO: Document this item
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

unsafe impl Sync for MeasurementRenderResourceArchetypeArena {}
unsafe impl Sync for MeasurementRenderResourceArchetype {}
unsafe impl Send for MeasurementRenderResourceArchetypeArena {}
unsafe impl Send for MeasurementRenderResourceArchetype {}

impl ArchetypeArenaCreate for MeasurementRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
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

    unsafe {
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::MeasurementPushConstants>() as u32);
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
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, _timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, u64::MAX);
  }
}

impl MeasurementRenderResourceArchetype {}

/// TODO: Document this item
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

unsafe impl Sync for MarkerRenderResourceArchetypeArena {}
unsafe impl Sync for MarkerRenderResourceArchetype {}
unsafe impl Send for MarkerRenderResourceArchetypeArena {}
unsafe impl Send for MarkerRenderResourceArchetype {}

impl ArchetypeArenaCreate for MarkerRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::MarkerPushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_MarkerRenderResourceArchetype")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
    })
  }
}

impl MarkerRenderResourceArchetypeArena {
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl MarkerRenderResourceArchetype {}

/// TODO: Document this item
pub(super) struct MinimapRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
}

pub(super) struct MinimapRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      MinimapRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for MinimapRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;

    let push_constant_ranges = [vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::MinimapPushConstants>() as u32,
    }];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::MinimapPushConstants>() as u32);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .map_err(|e| {
        aethervk_oshal_rlib::log!("create_pipeline_layout failed: {:?}", e);
        e
      })?;
    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
    })
  }
}
impl MinimapRenderResourceArchetypeArena {
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl MinimapRenderResourceArchetype {}

use alloc::collections::BTreeMap;

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

  allocator_raw: vk_mem::ffi::VmaAllocator,
}

pub(super) struct UiRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<UiRenderResourceArchetypeArena>,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

unsafe impl Sync for UiRenderResourceArchetypeArena {}
unsafe impl Sync for UiRenderResourceArchetype {}
unsafe impl Send for UiRenderResourceArchetypeArena {}
unsafe impl Send for UiRenderResourceArchetype {}

impl UiRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);
    struct PoolDiscard(vk::DescriptorPool);
    impl DeviceResource for PoolDiscard {
      fn cleanup(&mut self, device: &ash::Device) {
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
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers
      ),
      ctx.device.max_per_stage_descriptor_samplers
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::UiPushConstants>() as u32);

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
      allocator_raw: allocator.get_raw(),
    })
  }
}

impl UiRenderResourceArchetype {}

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

  allocator_raw: vk_mem::ffi::VmaAllocator,
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

unsafe impl Sync for TrajectoryRenderResourceArchetypeArena {}
unsafe impl Sync for TrajectoryRenderResourceArchetype {}
unsafe impl Send for TrajectoryRenderResourceArchetypeArena {}
unsafe impl Send for TrajectoryRenderResourceArchetype {}

impl ArchetypeArenaCreate for TrajectoryRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers
      ),
      ctx.device.max_per_stage_descriptor_samplers
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::TrajectoryPushConstants>() as u32);

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
      allocator_raw: allocator.get_raw(),
    })
  }
}

impl TrajectoryRenderResourceArchetypeArena {
  /// TODO: Document this item
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

impl TrajectoryRenderResourceArchetype {}

pub struct UploadedTexture {
  pub texture: Image,
  pub descriptor_index: u32,
  pub last_used_frame: u64,
}

/// TODO: Document this item
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
  pub allocator_raw: Option<vk_mem::ffi::VmaAllocator>,
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

unsafe impl Sync for BillboardRenderResourceArchetypeArena {}
unsafe impl Sync for BillboardRenderResourceArchetype {}
unsafe impl Send for BillboardRenderResourceArchetypeArena {}
unsafe impl Send for BillboardRenderResourceArchetype {}

impl ArchetypeArenaCreate for BillboardRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator_raw = ctx.allocator.get_raw();
    let device_limit = core::cmp::min(
      core::cmp::min(
        ctx.device.max_per_stage_descriptor_update_after_bind_samplers,
        ctx.device.max_descriptor_set_update_after_bind_samplers
      ),
      ctx.device.max_per_stage_descriptor_samplers
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::BillboardPushConstants>() as u32);

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
    device: &vulkan::device::LogicalDevice,
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

  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    // we don't care about descriptor set. discard the pool
    discard_pool.discard_type_erased(
      FunctionalDeviceResource::new(self.descriptor_pool.get(), |pool, device| unsafe {
        device.destroy_descriptor_pool(pool, None);
      }),
      timeline,
    );

    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.set_1_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);
  }
}

impl BillboardRenderResourceArchetype {}

/// TODO: Document this item
pub(super) struct GizmoRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub set_0_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
  pub descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,
  pub next_index: AtomicU32,
  pub host_buffers: DebugTrackedRwLock<hashbrown::HashMap<u32, Buffer>>,
  pub allocator_raw: vk_mem::ffi::VmaAllocator,
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

unsafe impl Sync for GizmoRenderResourceArchetypeArena {}
unsafe impl Sync for GizmoRenderResourceArchetype {}
unsafe impl Send for GizmoRenderResourceArchetypeArena {}
unsafe impl Send for GizmoRenderResourceArchetype {}

impl GizmoRenderResourceArchetypeArena {
  pub const MAX_BUFFER_COUNT: u32 = 256;
}

impl ArchetypeArenaCreate for GizmoRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator_raw = ctx.allocator.get_raw();
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::GizmoPushConstants>() as u32);

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
      host_buffers: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(
        hashbrown::HashMap::new(),
      ),
      allocator_raw,
    })
  }
}

impl GizmoRenderResourceArchetypeArena {
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_type_erased(
      FunctionalDeviceResource::new(self.descriptor_pool.get(), |pool, device| unsafe {
        device.destroy_descriptor_pool(pool, None);
      }),
      timeline,
    );

    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);

    let mut buffers =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&self.host_buffers);
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

/// TODO: Document this item
pub(super) struct ParticleRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub set_0_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: alloc::vec::Vec<vk::PushConstantRange>,
  pub descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,

  // Replaces `next_index`: Track allocated slices lock-free in the Mega Buffers
  pub allocated_particles: AtomicU32,
  pub allocated_systems: AtomicU32,

  pub mega_particle_buffer: vk::Buffer,
  pub mega_particle_alloc: vk_mem::Allocation,
  pub mega_indirect_buffer: vk::Buffer,
  pub mega_indirect_alloc: vk_mem::Allocation,

  // Stored purely so DiscardPool can destroy them properly later
  pub allocator_raw: vk_mem::ffi::VmaAllocator,
}

pub(super) struct ParticleRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      ParticleRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

unsafe impl Sync for ParticleRenderResourceArchetypeArena {}
unsafe impl Sync for ParticleRenderResourceArchetype {}
unsafe impl Send for ParticleRenderResourceArchetypeArena {}
unsafe impl Send for ParticleRenderResourceArchetype {}

impl ArchetypeArenaCreate for ParticleRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::ParticlePushConstants>() as u32,
    }];

    let bindings = [vk::DescriptorSetLayoutBinding {
      binding: 0,
      descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
      descriptor_count: 1, // Only 1 static descriptor now!
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      p_immutable_samplers: ptr::null(),
      ..Default::default()
    }];

    let layout_info = vk::DescriptorSetLayoutCreateInfo {
      binding_count: bindings.len() as u32,
      p_bindings: bindings.as_ptr(),
      ..Default::default()
    };

    let set_0_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
      .with_name(device, "VkDescriptorSetLayout_MegaParticleBuffer")?;

    let set_layouts = [set_0_layout];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_set_layouts: set_layouts.as_ptr(),
      set_layout_count: set_layouts.len() as u32,
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::ParticlePushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_ParticleRenderResourceArchetype")?;

    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::STORAGE_BUFFER)
      .descriptor_count(1)];
    let create_info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None) }.with_name(
      device,
      "VkDescriptorPoolCreateInfo_Dedicated_ParticleRenderResourceArchetype",
    )?;

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool)
      .set_layouts(&set_layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?
      .get(0)
      .copied()
      .unwrap();
    device.set_debug_name(
      descriptor_set,
      "VkDescriptorSet_Dedicated_ParticleRenderResourceArchetype",
    );

    // --- SAFE VMA ALLOCATIONS ---
    let alloc_create_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::AutoPreferDevice,
      ..Default::default()
    };
    crate::apply_test_dedicated_alloc!(alloc_create_info);

    // 1. Create Mega Particle Buffer
    let particle_buffer_size = (Self::MAX_PARTICLES as usize
      * core::mem::size_of::<crate::scene::particles::ParticleData>())
      as vk::DeviceSize;
    let particle_buffer_info = vk::BufferCreateInfo::default()
      .size(particle_buffer_size)
      // I kept STORAGE_BUFFER here in case you want Async Compute Shaders to update physics in-place later!
      .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);

    let (mega_particle_buffer, mega_particle_alloc) =
      unsafe { allocator.create_buffer(&particle_buffer_info, &alloc_create_info) }
        .map_err(|_| crate::gpu_err_device!())?;

    // 2. Create Mega Indirect Buffer
    let indirect_buffer_size = (Self::MAX_SYSTEMS as usize
      * core::mem::size_of::<vk::DrawIndirectCommand>())
      as vk::DeviceSize;
    let indirect_buffer_info = vk::BufferCreateInfo::default()
      .size(indirect_buffer_size)
      // Added STORAGE_BUFFER here so Compute Shaders can dynamically write DrawIndirectCommands as well
      .usage(
        vk::BufferUsageFlags::INDIRECT_BUFFER
          | vk::BufferUsageFlags::TRANSFER_DST
          | vk::BufferUsageFlags::STORAGE_BUFFER,
      );

    let (mega_indirect_buffer, mega_indirect_alloc) =
      unsafe { allocator.create_buffer(&indirect_buffer_info, &alloc_create_info) }
        .map_err(|_| crate::gpu_err_device!())?;

    // --- BIND MEGA BUFFER TO DESCRIPTOR SET ONCE FOREVER ---
    let buffer_info = vk::DescriptorBufferInfo::default()
      .buffer(mega_particle_buffer)
      .offset(0)
      .range(vk::WHOLE_SIZE);

    let write = vk::WriteDescriptorSet::default()
      .dst_set(descriptor_set)
      .dst_binding(0)
      .dst_array_element(0)
      .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
      .buffer_info(core::slice::from_ref(&buffer_info));

    unsafe {
      device.update_descriptor_sets(core::slice::from_ref(&write), &[]);
    }

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
      set_0_layout: unsafe { NonZeroHandle::new_unchecked(set_0_layout) },
      descriptor_pool: unsafe { NonZeroHandle::new_unchecked(descriptor_pool) },
      descriptor_set: unsafe { NonZeroHandle::new_unchecked(descriptor_set) },

      allocated_particles: AtomicU32::new(0),
      allocated_systems: AtomicU32::new(0),

      mega_particle_buffer,
      mega_particle_alloc,
      mega_indirect_buffer,
      mega_indirect_alloc,
      allocator_raw: allocator.get_raw(),
    })
  }
}

impl ParticleRenderResourceArchetypeArena {
  pub const MAX_PARTICLES: u32 = 1_000_000;
  pub const MAX_SYSTEMS: u32 = 1000;

  /// Lock-free allocation of a permanent slice inside the Mega Buffers for a Particle System.
  /// Returns: (system_indirect_index, particle_start_index)
  #[named]
  pub fn allocate_system_space(&self, particle_count: u32) -> GpuResult<(u32, u32)> {
    let sys_idx = self.allocated_systems.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    if sys_idx >= Self::MAX_SYSTEMS {
      return Err(gpu_err!("Mega Buffer Full: Max Systems Reached"));
    }

    let particle_offset = self
      .allocated_particles
      .fetch_add(particle_count, core::sync::atomic::Ordering::Relaxed);
    if particle_offset + particle_count > Self::MAX_PARTICLES {
      return Err(gpu_err!("Mega Buffer Full: Max Particles Reached"));
    }

    Ok((sys_idx, particle_offset))
  }

  /// Call this when completely changing levels/scenes to instantly reset the architecture
  /// without re-allocating or freeing memory!
  pub fn reset_allocations(&self) {
    self.allocated_particles.store(0, core::sync::atomic::Ordering::Relaxed);
    self.allocated_systems.store(0, core::sync::atomic::Ordering::Relaxed);
  }

  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_type_erased(
      FunctionalDeviceResource::new(self.descriptor_pool.get(), |pool, device| unsafe {
        device.destroy_descriptor_pool(pool, None);
      }),
      timeline,
    );

    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);

    discard_pool.discard_buffer(
      self.allocator_raw,
      self.mega_particle_buffer,
      self.mega_particle_alloc,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator_raw,
      self.mega_indirect_buffer,
      self.mega_indirect_alloc,
      timeline,
    );
  }
}

impl ParticleRenderResourceArchetype {}

/// TODO: Document this item
pub(super) struct Particle2RenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub set_0_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_constant_ranges: alloc::vec::Vec<vk::PushConstantRange>,
  pub descriptor_pool: NonZeroHandle<vk::DescriptorPool>,
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,

  // Replaces `next_index`: Track allocated slices lock-free in the Mega Buffers
  pub allocated_particles: AtomicU32,
  pub allocated_systems: AtomicU32,

  pub mega_particle_buffer: vk::Buffer,
  pub mega_particle_alloc: vk_mem::Allocation,
  pub mega_indirect_buffer: vk::Buffer,
  pub mega_indirect_alloc: vk_mem::Allocation,

  // Stored purely so DiscardPool can destroy them properly later
  pub allocator_raw: vk_mem::ffi::VmaAllocator,
}

pub(super) struct Particle2RenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      Particle2RenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

unsafe impl Sync for Particle2RenderResourceArchetypeArena {}
unsafe impl Sync for Particle2RenderResourceArchetype {}
unsafe impl Send for Particle2RenderResourceArchetypeArena {}
unsafe impl Send for Particle2RenderResourceArchetype {}

impl Particle2RenderResourceArchetypeArena {
  pub const MAX_PARTICLES: u32 = 1_000_000;
  pub const MAX_SYSTEMS: u32 = 1000;
}

impl ArchetypeArenaCreate for Particle2RenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let allocator = ctx.allocator;
    let push_constant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::Particle2PushConstants>() as u32,
    }];

    let bindings = [vk::DescriptorSetLayoutBinding {
      binding: 0,
      descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
      descriptor_count: 1, // Only 1 static descriptor now!
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      p_immutable_samplers: ptr::null(),
      ..Default::default()
    }];

    let layout_info = vk::DescriptorSetLayoutCreateInfo {
      binding_count: bindings.len() as u32,
      p_bindings: bindings.as_ptr(),
      ..Default::default()
    };

    let set_0_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
      .with_name(device, "VkDescriptorSetLayout_MegaParticleBuffer")?;

    let set_layouts = [set_0_layout];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_set_layouts: set_layouts.as_ptr(),
      set_layout_count: set_layouts.len() as u32,
      p_push_constant_ranges: push_constant_ranges.as_ptr(),
      push_constant_range_count: push_constant_ranges.len() as u32,
      ..Default::default()
    };
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::Particle2PushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_Particle2RenderResourceArchetype")?;

    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::STORAGE_BUFFER)
      .descriptor_count(1)];
    let create_info = vk::DescriptorPoolCreateInfo::default().max_sets(1).pool_sizes(&pool_sizes);
    let descriptor_pool = unsafe { device.create_descriptor_pool(&create_info, None) }.with_name(
      device,
      "VkDescriptorPoolCreateInfo_Dedicated_Particle2RenderResourceArchetype",
    )?;

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool)
      .set_layouts(&set_layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?
      .get(0)
      .copied()
      .unwrap();
    device.set_debug_name(
      descriptor_set,
      "VkDescriptorSet_Dedicated_Particle2RenderResourceArchetype",
    );

    // --- SAFE VMA ALLOCATIONS ---
    let alloc_create_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::AutoPreferDevice,
      ..Default::default()
    };
    crate::apply_test_dedicated_alloc!(alloc_create_info);

    // 1. Create Mega Particle Buffer
    let particle_buffer_size = (Self::MAX_PARTICLES as usize
      * core::mem::size_of::<crate::scene::particles::ParticleData>())
      as vk::DeviceSize;
    let particle_buffer_info = vk::BufferCreateInfo::default()
      .size(particle_buffer_size)
      // I kept STORAGE_BUFFER here in case you want Async Compute Shaders to update physics in-place later!
      .usage(vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST);

    let (mega_particle_buffer, mega_particle_alloc) =
      unsafe { allocator.create_buffer(&particle_buffer_info, &alloc_create_info) }
        .map_err(|_| crate::gpu_err_device!())?;

    // 2. Create Mega Indirect Buffer
    let indirect_buffer_size = (Self::MAX_SYSTEMS as usize
      * core::mem::size_of::<vk::DrawIndirectCommand>())
      as vk::DeviceSize;
    let indirect_buffer_info = vk::BufferCreateInfo::default()
      .size(indirect_buffer_size)
      // Added STORAGE_BUFFER here so Compute Shaders can dynamically write DrawIndirectCommands as well
      .usage(
        vk::BufferUsageFlags::INDIRECT_BUFFER
          | vk::BufferUsageFlags::TRANSFER_DST
          | vk::BufferUsageFlags::STORAGE_BUFFER,
      );

    let (mega_indirect_buffer, mega_indirect_alloc) =
      unsafe { allocator.create_buffer(&indirect_buffer_info, &alloc_create_info) }
        .map_err(|_| crate::gpu_err_device!())?;

    // --- BIND MEGA BUFFER TO DESCRIPTOR SET ONCE FOREVER ---
    let buffer_info = vk::DescriptorBufferInfo::default()
      .buffer(mega_particle_buffer)
      .offset(0)
      .range(vk::WHOLE_SIZE);

    let write = vk::WriteDescriptorSet::default()
      .dst_set(descriptor_set)
      .dst_binding(0)
      .dst_array_element(0)
      .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
      .buffer_info(core::slice::from_ref(&buffer_info));

    unsafe {
      device.update_descriptor_sets(core::slice::from_ref(&write), &[]);
    }

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
      set_0_layout: unsafe { NonZeroHandle::new_unchecked(set_0_layout) },
      descriptor_pool: unsafe { NonZeroHandle::new_unchecked(descriptor_pool) },
      descriptor_set: unsafe { NonZeroHandle::new_unchecked(descriptor_set) },

      allocated_particles: AtomicU32::new(0),
      allocated_systems: AtomicU32::new(0),

      mega_particle_buffer,
      mega_particle_alloc,
      mega_indirect_buffer,
      mega_indirect_alloc,
      allocator_raw: allocator.get_raw(),
    })
  }
}

impl Particle2RenderResourceArchetypeArena {
  /// Lock-free allocation of a permanent slice inside the Mega Buffers for a Particle System.
  /// Returns: (system_indirect_index, particle_start_index)
  #[named]
  pub fn allocate_system_space(&self, particle_count: u32) -> GpuResult<(u32, u32)> {
    let sys_idx = self.allocated_systems.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
    if sys_idx >= Self::MAX_SYSTEMS {
      return Err(gpu_err!("Mega Buffer Full: Max Systems Reached"));
    }

    let particle_offset = self
      .allocated_particles
      .fetch_add(particle_count, core::sync::atomic::Ordering::Relaxed);
    if particle_offset + particle_count > Self::MAX_PARTICLES {
      return Err(gpu_err!("Mega Buffer Full: Max Particles Reached"));
    }

    Ok((sys_idx, particle_offset))
  }

  /// Call this when completely changing levels/scenes to instantly reset the architecture
  /// without re-allocating or freeing memory!
  pub fn reset_allocations(&self) {
    self.allocated_particles.store(0, core::sync::atomic::Ordering::Relaxed);
    self.allocated_systems.store(0, core::sync::atomic::Ordering::Relaxed);
  }

  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_type_erased(
      FunctionalDeviceResource::new(self.descriptor_pool.get(), |pool, device| unsafe {
        device.destroy_descriptor_pool(pool, None);
      }),
      timeline,
    );

    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.set_0_layout.get(), timeline);

    discard_pool.discard_buffer(
      self.allocator_raw,
      self.mega_particle_buffer,
      self.mega_particle_alloc,
      timeline,
    );
    discard_pool.discard_buffer(
      self.allocator_raw,
      self.mega_indirect_buffer,
      self.mega_indirect_alloc,
      timeline,
    );
  }
}

impl Particle2RenderResourceArchetype {}

/// TODO: Document this item
pub(super) struct CursorRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
}

pub(super) struct CursorRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      CursorRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

unsafe impl Sync for CursorRenderResourceArchetypeArena {}
unsafe impl Sync for CursorRenderResourceArchetype {}
unsafe impl Send for CursorRenderResourceArchetypeArena {}
unsafe impl Send for CursorRenderResourceArchetype {}

impl CursorRenderResourceArchetypeArena {
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl ArchetypeArenaCreate for CursorRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::CursorPushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_CursorRenderResourceArchetype")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_constant_ranges,
    })
  }
}

impl CursorRenderResourceArchetype {}

/// TODO: Document this item
pub(super) struct SkyRenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub descriptor_set: Option<NonZeroHandle<vk::DescriptorSet>>,
}
unsafe impl Send for SkyRenderResourceArchetypeArena {}
unsafe impl Sync for SkyRenderResourceArchetypeArena {}

pub(super) struct SkyRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<SkyRenderResourceArchetypeArena>,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
}

impl ArchetypeArenaCreate for SkyRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::SkyPushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(set_layout) },
      descriptor_set: None,
    })
  }
}

impl SkyRenderResourceArchetypeArena {
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
  }
}

impl SkyRenderResourceArchetype {}

/// TODO: Document this item
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
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::BackgroundPushConstants>() as u32)];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::BackgroundPushConstants>() as u32);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;
    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
    })
  }
}

impl BackgroundRenderResourceArchetypeArena {
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl BackgroundRenderResourceArchetype {}

/// TODO: Document this item
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
  /// TODO: Document this item
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

impl ArchetypeArenaCreate for GridRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<crate::gpu::GridPushConstants>() as u32)];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::GridPushConstants>() as u32);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_GridRenderResourceArchetypeArena")?;
    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
    })
  }
}

impl GridRenderResourceArchetype {}

/// TODO: Document this item
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

unsafe impl Sync for SunRenderResourceArchetypeArena {}
unsafe impl Sync for SunRenderResourceArchetype {}
unsafe impl Send for SunRenderResourceArchetypeArena {}
unsafe impl Send for SunRenderResourceArchetype {}

impl ArchetypeArenaCreate for SunRenderResourceArchetypeArena {
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
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
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::SunPushConstants>() as u32);

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
  /// TODO: Document this item
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
  allocator_raw: vk_mem::ffi::VmaAllocator,
}

pub(super) struct ForwardMeshRenderResourceArchetype {
  pub arena: alloc::sync::Weak<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      ForwardMeshRenderResourceArchetypeArena,
    >,
  >,
  pub pipeline_key: PipelineKey,
  pub graphics_info: GraphicsInfo,
  pub outline_pipeline_key: PipelineKey,
  pub outline_graphics_info: GraphicsInfo,
}

/// To be destroyed before descriptor pool
pub(super) struct ForwardMesh2RenderResourceArchetypeArena {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layouts: Vec<NonZeroHandle<vk::DescriptorSetLayout>>,
  pub push_constant_ranges: Vec<vk::PushConstantRange>,
  // 0 = vertex, 1 = fragment
  pub specialization_constants: [Vec<vk::SpecializationMapEntry>; 2],
  // 0 = vertex, 1 = fragment
  pub specialization_constants_values: [Vec<u8>; 2],

  pub dummy_texture_handle: Image,
  /// Necessary evil for discard. assumes it outlives this object
  allocator_raw: vk_mem::ffi::VmaAllocator,
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
    for layout in &self.descriptor_set_layouts {
      discard_pool.discard_descriptor_set_layout(layout.get(), timeline);
    }
  }
}

unsafe impl Sync for ForwardMeshRenderResourceArchetypeArena {}
unsafe impl Sync for ForwardMeshRenderResourceArchetype {}
unsafe impl Send for ForwardMeshRenderResourceArchetypeArena {}
unsafe impl Send for ForwardMeshRenderResourceArchetype {}

impl ForwardMeshRenderResourceArchetypeArena {
  /// TODO: Document this item
  #[named]
  pub fn create_descriptor_set_from_layout_at_index(
    &self,
    device: &vulkan::device::LogicalDevice,
    descriptor_pools: &sync::Arc<DescriptorPools>,
    discard_pool: &DiscardPool,
    index: usize,
    debug_name: &str,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<NonZeroHandle<vk::DescriptorSet>> {
    const NEVER_DISCARD_TIMELINE: u64 = u64::MAX;

    let layout = self
      .descriptor_set_layouts
      .get(index)
      .ok_or(crate::gpu_invalid_arg!("invalid argument"))?
      .get();
    descriptor_pools.allocate(
      device,
      layout,
      discard_pool,
      NEVER_DISCARD_TIMELINE,
      debug_name,
      rollback,
    )
  }
}

impl ArchetypeArenaCreate for ForwardMeshRenderResourceArchetypeArena {
  /// Safety:
  /// - `pipeline_key` must refer to a pipeline created with `vertex_shader` and `fragment_shader`,
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let vertex_shader = ctx.vertex_shader.unwrap();
    let fragment_shader = ctx.fragment_shader.unwrap();
    let outline_vertex_shader = ctx.outline_vertex_shader.unwrap();
    let outline_fragment_shader = ctx.outline_fragment_shader.unwrap();
    let allocator = ctx.allocator;
    let staging_arena = ctx.staging_arena.unwrap();
    let queue = &ctx.queue.unwrap();

    const NEVER_DISCARD_TIMELINE: u64 = u64::MAX;
    // TODO Implement a special value for the janitor which means "DYNAMIC_SIZE"
    let mut janitor = DeviceResourceJanitor::<'_, 256>::new(device);

    if !vertex_shader
      .spv_module
      .get_shader_stage()
      .contains(ReflectShaderStageFlags::VERTEX)
      || !fragment_shader
        .spv_module
        .get_shader_stage()
        .contains(ReflectShaderStageFlags::FRAGMENT)
    {
      return Err(GpuError::InvalidShader);
    }
    // --------------------------- 1. Descriptor Sets -------------------------------------------
    // This will hold the merged layout information.
    // Map<set_number, Map<binding_number, vk::DescriptorSetLayoutBinding>>
    let mut merged_sets: hashbrown::HashMap<
      u32,
      hashbrown::HashMap<u32, vk::DescriptorSetLayoutBinding>,
    > = hashbrown::HashMap::new();

    for shader in [
      vertex_shader,
      fragment_shader,
      outline_vertex_shader,
      outline_fragment_shader,
    ] {
      let shader_stage = shader.shader_stage;
      let sets = shader
        .spv_module
        .enumerate_descriptor_sets(None)
        .map_err(|_| GpuError::InvalidShader)?;

      for set in sets {
        let bindings_map = merged_sets.entry(set.set).or_default();

        for binding in &set.bindings {
          let reflect_binding = binding;
          let new_descriptor_type = map_descriptor_type(reflect_binding.descriptor_type)?;

          if let Some(existing_binding) = bindings_map.get_mut(&reflect_binding.binding) {
            // Binding already exists in another shader stage, check for conflicts.
            if existing_binding.descriptor_type != new_descriptor_type
              || existing_binding.descriptor_count != reflect_binding.count
            {
              return Err(GpuError::BackendSpecific(alloc::fmt::format(format_args!(
                "Descriptor set binding conflict at (set={}, binding={}). Mismatch in descriptor type or count across shader stages.",
                set.set, reflect_binding.binding
              ))));
            }

            // No conflict, so merge the stage flags.
            existing_binding.stage_flags |= shader_stage;
          } else {
            // First time seeing this binding, create a new one.
            let new_binding = vk::DescriptorSetLayoutBinding::default()
              .binding(reflect_binding.binding)
              .descriptor_type(new_descriptor_type)
              .descriptor_count(reflect_binding.count)
              .stage_flags(shader_stage);
            bindings_map.insert(reflect_binding.binding, new_binding);
          }
        }
      }
    }

    // Convert the map of maps into the final structure needed for layout creation.
    // Map<set_number, Vec<vk::DescriptorSetLayoutBinding>>
    let set_layouts: hashbrown::HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = merged_sets
      .into_iter()
      .map(|(set_number, bindings_map)| {
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
          bindings_map.into_values().collect();
        // Sort by binding number for consistency.
        bindings.sort_by_key(|b| b.binding);
        (set_number, bindings)
      })
      .collect();
    // Sort by set number to ensure the final layouts have a deterministic order.
    let mut sorted_layouts: Vec<_> = set_layouts.into_iter().collect();
    sorted_layouts.sort_by_key(|(set, _)| *set);

    let descriptor_set_layouts: Vec<NonZeroHandle<vk::DescriptorSetLayout>> = sorted_layouts
      .into_iter()
      .map(|(_, bindings)| {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
        janitor
          .push(FunctionalDeviceResource::new(layout, |h, d| unsafe {
            d.destroy_descriptor_set_layout(h, None)
          }))
          .map_err(|_| crate::gpu_err_device!())?;
        Ok(unsafe { NonZeroHandle::new_unchecked(layout) })
      })
      .collect::<GpuResult<Vec<_>>>()?;

    // --------------------------- 2. Push Constants --------------------------------------------
    let mut push_constant_ranges = Vec::<vk::PushConstantRange>::new();
    for shader in [vertex_shader, fragment_shader] {
      let blocks = shader
        .spv_module
        .enumerate_push_constant_blocks(None)
        .map_err(|_| GpuError::InvalidShader)?;

      for block in blocks {
        // Find a range with the same offset to merge stage flags and max size.
        if let Some(range) = push_constant_ranges
          .iter_mut()
          .find(|r| r.offset == block.offset)
        {
          // Merge shader stages into the existing range.
          range.stage_flags |= shader.shader_stage;
          range.size = range.size.max(block.size);
        } else {
          // Add a new range for this push constant block.
          push_constant_ranges.push(
            vk::PushConstantRange::default()
              .stage_flags(shader.shader_stage)
              .offset(block.offset)
              .size(block.size),
          );
        }
      }
    }

    // --------------------------- 3. Pipeline Layout -------------------------------------------
    let layout_create_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(unsafe {
        core::slice::from_raw_parts(
          descriptor_set_layouts.as_ptr() as *const _,
          descriptor_set_layouts.len(),
        )
      })
      .push_constant_ranges(&push_constant_ranges);
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::PushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }
      .with_name(
        device,
        "VkPipelineLayout_ForwardMeshRenderResourceArchetype",
      )?;

    janitor
      .push(FunctionalDeviceResource::new(
        pipeline_layout,
        |h, d| unsafe { d.destroy_pipeline_layout(h, None) },
      ))
      .map_err(|_| crate::gpu_err_device!())?;

    // --------------------------- 4. Specialization Infos --------------------------------------
    let mut specialization_constants = [Vec::new(), Vec::new()];
    let mut specialization_constants_values = [Vec::new(), Vec::new()];

    for (i, shader) in [vertex_shader, fragment_shader].iter().enumerate() {
      // NOTE: `spirv-reflect` does not provide the size of specialization constants
      // directly. We are assuming here that all specialization constants are 32-bit (4-byte)
      // values like int, float, or bool. This may need adjustment if other types are used.
      const ASSUMED_SPEC_CONST_SIZE: usize = 4;
      let spv_specialization_constants = unsafe {
        let mut count: u32 = 0;
        let mut res = spirv_reflect::ffi::spvReflectEnumerateSpecializationConstants(
          ptr::from_ref(shader.spv_module.as_raw_unchecked()),
          ptr::from_mut(&mut count),
          ptr::null_mut(),
        );
        if res != SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS {
          return Err(GpuError::InvalidShader);
        }
        let mut the_vec = Vec::new();
        the_vec.resize(
          count as usize,
          ptr::null_mut::<spirv_reflect::ffi::SpvReflectSpecializationConstant>(),
        );
        res = spirv_reflect::ffi::spvReflectEnumerateSpecializationConstants(
          ptr::from_ref(shader.spv_module.as_raw_unchecked()),
          ptr::from_mut(&mut count),
          the_vec.as_mut_ptr(),
        );
        if res != SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS {
          return Err(GpuError::InvalidShader);
        }
        let mut the_vec: Vec<_> = the_vec.iter().map(|c| c.as_ref().unwrap_unchecked()).collect();
        the_vec.sort_by_key(|&c| c.constant_id);
        Ok::<Vec<_>, GpuError>(the_vec)
      }?;
      specialization_constants[i].reserve(spv_specialization_constants.len());

      for spec_const in spv_specialization_constants {
        let offset = specialization_constants_values[i].len() as u32;

        specialization_constants[i].push(
          vk::SpecializationMapEntry::default()
            .constant_id(spec_const.constant_id)
            .offset(offset)
            .size(ASSUMED_SPEC_CONST_SIZE),
        );

        // The reflection does not provide the default value from the shader. We populate
        // the data blob with a default value based on its name.
        let name = unsafe {
          if spec_const.name.is_null() {
            ""
          } else {
            core::ffi::CStr::from_ptr(spec_const.name).to_str().unwrap_or("")
          }
        };

        let default_value_bytes = match name {
          "BASE_ALBEDO_R" => 0.8f32.to_ne_bytes(),
          "BASE_ALBEDO_G" => 0.8f32.to_ne_bytes(),
          "BASE_ALBEDO_B" => 0.8f32.to_ne_bytes(),
          "BASE_ROUGHNESS" => 0.9f32.to_ne_bytes(),
          "BASE_AO" => 1.0f32.to_ne_bytes(),
          _ => [0u8; ASSUMED_SPEC_CONST_SIZE],
        };
        specialization_constants_values[i].extend_from_slice(&default_value_bytes);
      }
    }

    // create dummy 1x1 black texture
    let dummy_texture_handle = {
      let mut inner_janitor = DeviceResourceJanitor::<'_, 64>::new(device);

      // create throwaway command pool and command buffer
      let command_pool = {
        let create_info = vk::CommandPoolCreateInfo::default()
          .queue_family_index(queue.family_index)
          .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        unsafe { device.create_command_pool(&create_info, None) }
      }?;
      inner_janitor
        .push(FunctionalDeviceResource::new(command_pool, |h, d| unsafe {
          d.destroy_command_pool(h, None)
        }))
        .map_err(|_| crate::gpu_err_device!())?;

      let command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
          .command_pool(command_pool)
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(1);
        unsafe { device.allocate_command_buffers(&alloc_info) }?[0]
      };

      let dummy_texture = Texture {
        data: bytes::Bytes::from_static(&[0]),
        format: crate::simulation::comet::TexelFormat::R8_UNORM,
        width: 1,
        height: 1,
        has_mipmaps: false,
      };

      let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
      unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;
      };

      let image = Image::new_2d(
        device,
        allocator,
        command_buffer,
        staging_arena,
        &dummy_texture,
        vk::ImageUsageFlags::SAMPLED,
        "ForwardMeshRenderResourceArchetype_DummyTexture",
      );
      if image.is_err() {
        return Err(unsafe { image.unwrap_err_unchecked() });
      }

      unsafe {
        device.end_command_buffer(command_buffer)?;
        let command_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        let fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        device
          .locked_queue_submit(queue.handle, &submits, fence)
          .map_err(GpuError::from)?;
        device.wait_for_fences(&[fence], true, u64::MAX)?;
        device.destroy_fence(fence, None);
      };

      image
    }?;

    janitor.clear();
    Ok(Self {
      pipeline_layout: NonZeroHandle::new(pipeline_layout).unwrap(),
      descriptor_set_layouts,
      push_constant_ranges,
      specialization_constants,
      specialization_constants_values,
      dummy_texture_handle,
      allocator_raw: allocator.get_raw(),
    })
  }
}

impl ForwardMeshRenderResourceArchetype {}

impl ForwardMeshRenderResourceArchetypeArena {
  pub fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    aethervk_oshal_rlib::log!(
      "ForwardMeshRenderResourceArchetypeArena::discard called for dummy_texture_handle: {:#X}",
      self.dummy_texture_handle.image.get().as_raw()
    );
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_image_view(self.dummy_texture_handle.image_view.get(), timeline);
    discard_pool.discard_image(
      self.allocator_raw,
      self.dummy_texture_handle.image.get(),
      self.dummy_texture_handle.allocation,
      timeline,
    );
    for layout in &self.descriptor_set_layouts {
      discard_pool.discard_descriptor_set_layout(layout.get(), timeline);
    }
  }
}

/// Reusable helper function to perform the explicit staging buffer upload pattern.
#[named]
pub(super) fn create_buffer_with_staging<T: Copy>(
  device: &vulkan::device::LogicalDevice,
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

/// Helper type for getting pipeline information from presentation engine (color attachment format)

/// Helper to map descriptor types from spirv-reflect to ash and handle unsupported cases.
unsafe impl Sync for ForwardMesh2RenderResourceArchetypeArena {}
unsafe impl Sync for ForwardMesh2RenderResourceArchetype {}
unsafe impl Send for ForwardMesh2RenderResourceArchetypeArena {}
unsafe impl Send for ForwardMesh2RenderResourceArchetype {}

impl ForwardMesh2RenderResourceArchetypeArena {
  /// TODO: Document this item
  #[named]
  pub fn create_descriptor_set_from_layout_at_index(
    &self,
    device: &vulkan::device::LogicalDevice,
    descriptor_pools: &sync::Arc<DescriptorPools>,
    discard_pool: &DiscardPool,
    index: usize,
    debug_name: &str,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<NonZeroHandle<vk::DescriptorSet>> {
    const NEVER_DISCARD_TIMELINE: u64 = u64::MAX;

    let layout = self
      .descriptor_set_layouts
      .get(index)
      .ok_or(crate::gpu_invalid_arg!("invalid argument"))?
      .get();
    descriptor_pools.allocate(
      device,
      layout,
      discard_pool,
      NEVER_DISCARD_TIMELINE,
      debug_name,
      rollback,
    )
  }
}

impl ArchetypeArenaCreate for ForwardMesh2RenderResourceArchetypeArena {
  /// TODO: Document this item
  #[named]
  fn new_arena(ctx: &ArenaCreationContext) -> GpuResult<Self> {
    let device = ctx.device;
    let vertex_shader = ctx.vertex_shader.unwrap();
    let fragment_shader = ctx.fragment_shader.unwrap();
    let outline_vertex_shader = ctx.outline_vertex_shader.unwrap();
    let outline_fragment_shader = ctx.outline_fragment_shader.unwrap();
    let allocator = ctx.allocator;
    let staging_arena = ctx.staging_arena.unwrap();
    let queue = &ctx.queue.unwrap();
    const NEVER_DISCARD_TIMELINE: u64 = u64::MAX;
    // TODO Implement a special value for the janitor which means "DYNAMIC_SIZE"
    let mut janitor = DeviceResourceJanitor::<'_, 256>::new(device);

    if !vertex_shader
      .spv_module
      .get_shader_stage()
      .contains(ReflectShaderStageFlags::VERTEX)
      || !fragment_shader
        .spv_module
        .get_shader_stage()
        .contains(ReflectShaderStageFlags::FRAGMENT)
    {
      return Err(GpuError::InvalidShader);
    }
    // --------------------------- 1. Descriptor Sets -------------------------------------------
    // This will hold the merged layout information.
    // Map<set_number, Map<binding_number, vk::DescriptorSetLayoutBinding>>
    let mut merged_sets: hashbrown::HashMap<
      u32,
      hashbrown::HashMap<u32, vk::DescriptorSetLayoutBinding>,
    > = hashbrown::HashMap::new();

    for shader in [
      vertex_shader,
      fragment_shader,
      outline_vertex_shader,
      outline_fragment_shader,
    ] {
      let shader_stage = shader.shader_stage;
      let sets = shader
        .spv_module
        .enumerate_descriptor_sets(None)
        .map_err(|_| GpuError::InvalidShader)?;

      for set in sets {
        let bindings_map = merged_sets.entry(set.set).or_default();

        for binding in &set.bindings {
          let reflect_binding = binding;
          let new_descriptor_type = map_descriptor_type(reflect_binding.descriptor_type)?;

          if let Some(existing_binding) = bindings_map.get_mut(&reflect_binding.binding) {
            // Binding already exists in another shader stage, check for conflicts.
            if existing_binding.descriptor_type != new_descriptor_type
              || existing_binding.descriptor_count != reflect_binding.count
            {
              return Err(GpuError::BackendSpecific(alloc::fmt::format(format_args!(
                "Descriptor set binding conflict at (set={}, binding={}). Mismatch in descriptor type or count across shader stages.",
                set.set, reflect_binding.binding
              ))));
            }

            // No conflict, so merge the stage flags.
            existing_binding.stage_flags |= shader_stage;
          } else {
            // First time seeing this binding, create a new one.
            let new_binding = vk::DescriptorSetLayoutBinding::default()
              .binding(reflect_binding.binding)
              .descriptor_type(new_descriptor_type)
              .descriptor_count(reflect_binding.count)
              .stage_flags(shader_stage);
            bindings_map.insert(reflect_binding.binding, new_binding);
          }
        }
      }
    }

    // Convert the map of maps into the final structure needed for layout creation.
    // Map<set_number, Vec<vk::DescriptorSetLayoutBinding>>
    let set_layouts: hashbrown::HashMap<u32, Vec<vk::DescriptorSetLayoutBinding>> = merged_sets
      .into_iter()
      .map(|(set_number, bindings_map)| {
        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> =
          bindings_map.into_values().collect();
        // Sort by binding number for consistency.
        bindings.sort_by_key(|b| b.binding);
        (set_number, bindings)
      })
      .collect();
    // Sort by set number to ensure the final layouts have a deterministic order.
    let mut sorted_layouts: Vec<_> = set_layouts.into_iter().collect();
    sorted_layouts.sort_by_key(|(set, _)| *set);

    let descriptor_set_layouts: Vec<NonZeroHandle<vk::DescriptorSetLayout>> = sorted_layouts
      .into_iter()
      .map(|(_, bindings)| {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
        janitor
          .push(FunctionalDeviceResource::new(layout, |h, d| unsafe {
            d.destroy_descriptor_set_layout(h, None)
          }))
          .map_err(|_| crate::gpu_err_device!())?;
        Ok(unsafe { NonZeroHandle::new_unchecked(layout) })
      })
      .collect::<GpuResult<Vec<_>>>()?;

    // --------------------------- 2. Push Constants --------------------------------------------
    let mut push_constant_ranges = Vec::<vk::PushConstantRange>::new();
    for shader in [
      vertex_shader,
      fragment_shader,
      outline_vertex_shader,
      outline_fragment_shader,
    ] {
      let blocks = shader
        .spv_module
        .enumerate_push_constant_blocks(None)
        .map_err(|_| GpuError::InvalidShader)?;

      for block in blocks {
        // Find a range with the same offset to merge stage flags and max size.
        if let Some(range) = push_constant_ranges
          .iter_mut()
          .find(|r| r.offset == block.offset)
        {
          // Merge shader stages into the existing range.
          range.stage_flags |= shader.shader_stage;
          range.size = range.size.max(block.size);
        } else {
          // Add a new range for this push constant block.
          push_constant_ranges.push(
            vk::PushConstantRange::default()
              .stage_flags(shader.shader_stage)
              .offset(block.offset)
              .size(block.size),
          );
        }
      }
    }

    // --------------------------- 3. Pipeline Layout -------------------------------------------
    let set_layouts_raw: Vec<vk::DescriptorSetLayout> =
      descriptor_set_layouts.iter().map(|l| l.get()).collect();
    let layout_create_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(&set_layouts_raw)
      .push_constant_ranges(&push_constant_ranges);
    ctx.validate_push_constant_size(core::mem::size_of::<crate::gpu::PhysicalMesh2PushConstants>() as u32);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_create_info, None) }
      .with_name(
        device,
        "VkPipelineLayout_ForwardMesh2RenderResourceArchetype",
      )?;

    janitor
      .push(FunctionalDeviceResource::new(
        pipeline_layout,
        |h, d| unsafe { d.destroy_pipeline_layout(h, None) },
      ))
      .map_err(|_| crate::gpu_err_device!())?;

    // --------------------------- 4. Specialization Infos --------------------------------------
    let mut specialization_constants = [Vec::new(), Vec::new()];
    let mut specialization_constants_values = [Vec::new(), Vec::new()];

    for (i, shader) in [vertex_shader, fragment_shader].iter().enumerate() {
      // NOTE: `spirv-reflect` does not provide the size of specialization constants
      // directly. We are assuming here that all specialization constants are 32-bit (4-byte)
      // values like int, float, or bool. This may need adjustment if other types are used.
      const ASSUMED_SPEC_CONST_SIZE: usize = 4;
      let spv_specialization_constants = unsafe {
        let mut count: u32 = 0;
        let mut res = spirv_reflect::ffi::spvReflectEnumerateSpecializationConstants(
          ptr::from_ref(shader.spv_module.as_raw_unchecked()),
          ptr::from_mut(&mut count),
          ptr::null_mut(),
        );
        if res != SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS {
          return Err(GpuError::InvalidShader);
        }
        let mut the_vec = Vec::new();
        the_vec.resize(
          count as usize,
          ptr::null_mut::<spirv_reflect::ffi::SpvReflectSpecializationConstant>(),
        );
        res = spirv_reflect::ffi::spvReflectEnumerateSpecializationConstants(
          ptr::from_ref(shader.spv_module.as_raw_unchecked()),
          ptr::from_mut(&mut count),
          the_vec.as_mut_ptr(),
        );
        if res != SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS {
          return Err(GpuError::InvalidShader);
        }
        let mut the_vec: Vec<_> = the_vec.iter().map(|c| c.as_ref().unwrap_unchecked()).collect();
        the_vec.sort_by_key(|&c| c.constant_id);
        Ok::<Vec<_>, GpuError>(the_vec)
      }?;
      specialization_constants[i].reserve(spv_specialization_constants.len());

      for spec_const in spv_specialization_constants {
        let offset = specialization_constants_values[i].len() as u32;

        specialization_constants[i].push(
          vk::SpecializationMapEntry::default()
            .constant_id(spec_const.constant_id)
            .offset(offset)
            .size(ASSUMED_SPEC_CONST_SIZE),
        );

        // The reflection does not provide the default value from the shader. We populate
        // the data blob with a default value based on its name.
        let name = unsafe {
          if spec_const.name.is_null() {
            ""
          } else {
            core::ffi::CStr::from_ptr(spec_const.name).to_str().unwrap_or("")
          }
        };

        let default_value_bytes = match name {
          "BASE_ALBEDO_R" => 0.8f32.to_ne_bytes(),
          "BASE_ALBEDO_G" => 0.8f32.to_ne_bytes(),
          "BASE_ALBEDO_B" => 0.8f32.to_ne_bytes(),
          "BASE_ROUGHNESS" => 0.9f32.to_ne_bytes(),
          "BASE_AO" => 1.0f32.to_ne_bytes(),
          _ => [0u8; ASSUMED_SPEC_CONST_SIZE],
        };
        specialization_constants_values[i].extend_from_slice(&default_value_bytes);
      }
    }

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

    if let Err(e) = unsafe {
      device.locked_queue_submit(queue.handle, core::slice::from_ref(&submit_info), fence)
    } {
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

    janitor.clear();

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      descriptor_set_layouts,
      push_constant_ranges,
      specialization_constants,
      specialization_constants_values,
      dummy_texture_handle,
      allocator_raw: allocator.get_raw(),
    })
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
  TextRenderResourceArchetype,
  TextRenderResourceArchetypeArena
);
impl_deref_archetype!(
  Text2RenderResourceArchetype,
  Text2RenderResourceArchetypeArena
);
impl_deref_archetype!(BvhRenderResourceArchetype, BvhRenderResourceArchetypeArena);
impl_deref_archetype!(
  Bvhwire2RenderResourceArchetype,
  Bvhwire2RenderResourceArchetypeArena
);
impl_deref_archetype!(
  MeasurementRenderResourceArchetype,
  MeasurementRenderResourceArchetypeArena
);
impl_deref_archetype!(
  MarkerRenderResourceArchetype,
  MarkerRenderResourceArchetypeArena
);
impl_deref_archetype!(
  MinimapRenderResourceArchetype,
  MinimapRenderResourceArchetypeArena
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
  ParticleRenderResourceArchetype,
  ParticleRenderResourceArchetypeArena
);
impl_deref_archetype!(
  Particle2RenderResourceArchetype,
  Particle2RenderResourceArchetypeArena
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
  ForwardMeshRenderResourceArchetype,
  ForwardMeshRenderResourceArchetypeArena
);
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
        allocator_raw: vk_mem::ffi::VmaAllocator,
        timeline: u64,
      ) {
        discard_pool.discard_image_view(prep.image_view, timeline);
        discard_pool.discard_image(allocator_raw, prep.image, prep.allocation, timeline);
      }
    }
  };
}

impl_font_atlas_arena_transactional!(TextRenderResourceArchetypeArena);
impl_font_atlas_arena_transactional!(Text2RenderResourceArchetypeArena);
