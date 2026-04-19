use core::ptr;
use core::hash::{Hash, Hasher};
use aethervk_oshal_rlib as oshal;
use oshal::{hash::FnvHasher, os::native::ThreadId};
use ash::vk;
use alloc::{boxed::Box, collections::VecDeque, sync, vec::Vec};
use spirv_reflect::{ffi::SpvReflectResult_SPV_REFLECT_RESULT_SUCCESS, types::ReflectShaderStageFlags};
use crate::gpu_backends::vulkan;
use crate::gpu_backends::vulkan::device::{VmaDebugNameExt, VulkanDebugNameExt};
use crate::simulation::comet::Texture;
use vk_mem::Alloc;

use crate::gpu::PipelineKeyable;
use crate::gpu_backends::vulkan::device::commands::{self, CommandBufferId};
use crate::gpu_backends::vulkan::device::pipelines::GraphicsInfo;
use crate::{
  gpu::{PipelineKey},
  gpu_backends::vulkan::{
    device::{
      DeviceResource, DeviceResourceJanitor, FunctionalDeviceResource,
      descriptors::{self, DescriptorPools},
      shader_manager::Shader,
    },
    utils::NonZeroHandle,
  },
  types::{GpuError, GpuResult},
};

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
    self.items.push_back((timeline, item));
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

enum DiscardItem {
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
      Self::GenericHandle(h) => {
        let ptr: *const dyn DeviceResource = &**h;
        (10, ptr as *const () as u64)
      }
    }
  }
}

struct BufferDiscard {
  buffer: vk::Buffer,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::ffi::VmaAllocator, // non owning copy
}
struct ImageDiscard {
  image: vk::Image,
  alloc: vk_mem::Allocation,
  allocator: vk_mem::ffi::VmaAllocator, // non owning copy
}
struct CmdBufDiscard {
  thread_id: ThreadId,
  command_buffer: vk::CommandBuffer,
  manager: sync::Arc<commands::CommandPools>,
  id: CommandBufferId,
}

pub trait DiscardableResource {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64);
}

/// Structure associated to the main Timeline Semaphore provided by Device
/// Note: this must not outlive device, hence don't expose it outside
pub(super) struct DiscardPool {
  items: spin::Mutex<TimelineQueue<DiscardItem>>,
  #[cfg(debug_assertions)]
  queued_handles: spin::Mutex<hashbrown::HashSet<(u8, u64)>>,
}

unsafe impl Sync for DiscardPool {}
unsafe impl Send for DiscardPool {}

impl DiscardPool {
  /// Safety: device and allocator should outlive Self
  pub unsafe fn new(cap: usize) -> Self {
    Self {
      items: spin::Mutex::new(TimelineQueue::with_capacity(cap)),
      #[cfg(debug_assertions)]
      queued_handles: spin::Mutex::new(hashbrown::HashSet::with_capacity(cap)),
    }
  }

  fn push_item(&self, timeline: u64, item: DiscardItem) {
    let mut q = self.items.lock();
    #[cfg(debug_assertions)]
    {
      let handle = item.unique_id();
      assert!(
        self.queued_handles.lock().insert(handle),
        "Resource discarded twice! Type: {}, Handle: {}",
        handle.0,
        handle.1
      );
    }
    q.push(timeline, item);
  }

  pub fn discard_type_erased<T: DeviceResource + 'static>(&self, item: T, timeline: u64) {
    self.push_item(timeline, DiscardItem::GenericHandle(Box::new(item)));
  }

  // TODO all other types of resources as needed
  pub fn discard_render_pass(&self, render_pass: vk::RenderPass, timeline: u64) {
    self.push_item(timeline, DiscardItem::RenderPass(render_pass));
  }

  pub fn discard_framebuffer(&self, framebuffer: vk::Framebuffer, timeline: u64) {
    self.push_item(timeline, DiscardItem::Framebuffer(framebuffer));
  }

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
    self.push_item(timeline, DiscardItem::DescriptorPool(pool, manager));
  }

  pub fn discard_pipeline(&self, pipeline: vk::Pipeline, timeline: u64) {
    self.push_item(timeline, DiscardItem::Pipeline(pipeline));
  }

  pub fn discard_pipeline_layout(&self, pipeline_layout: vk::PipelineLayout, timeline: u64) {
    self.push_item(timeline, DiscardItem::PipelineLayout(pipeline_layout));
  }

  pub fn destroy_discarded_resources_all(&self, device: &ash::Device) {
    self.destroy_discarded_resources_internal(device, u64::MAX);
  }

  /// safety: `sem` needs to be a valid timeline semaphore
  pub unsafe fn destroy_discarded_resources_timeline(
    &self,
    device: &ash::Device,
    sem: vk::Semaphore,
  ) -> ash::prelude::VkResult<()> {
    let timeline = unsafe { device.get_semaphore_counter_value(sem) }?;
    self.destroy_discarded_resources_internal(device, timeline);
    Ok(())
  }

  fn destroy_discarded_resources_internal(&self, device: &ash::Device, timeline: u64) {
    let mut items = self.items.lock();
    #[cfg(debug_assertions)]
    let mut queued_handles = self.queued_handles.lock();

    items.drain_ready(timeline, |item| {
      #[cfg(debug_assertions)]
      {
        queued_handles.remove(&item.unique_id());
      }
      match item {
        DiscardItem::Buffer(BufferDiscard {
          buffer,
          alloc,
          allocator,
        }) => unsafe {
          vk_mem::ffi::vmaDestroyBuffer(allocator, buffer, alloc.get_raw());
        },
        DiscardItem::Image(ImageDiscard {
          image,
          alloc,
          allocator,
        }) => unsafe {
          vk_mem::ffi::vmaDestroyImage(allocator, image, alloc.get_raw());
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
          // leaking if we didn't manage to find it!
          #[cfg(debug_assertions)]
          {
            if _x.is_err() {
              panic!("aaa");
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
        DiscardItem::GenericHandle(mut handle) => {
          handle.cleanup(device);
        }
      }
    });
  }
}

impl super::DeviceResource for DiscardPool {
  fn cleanup(&mut self, device: &ash::Device) {
    self.destroy_discarded_resources_all(device);
  }
}

pub(super) struct Buffer {
  pub buffer: NonZeroHandle<vk::Buffer>,
  pub allocation: vk_mem::Allocation,
}

impl Hash for Buffer {
  fn hash<H: Hasher>(&self, state: &mut H) {
    self.buffer.hash(state);
  }
}

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

  pub fn new_storage_2d(
    device: &vulkan::device::LogicalDevice,
    allocator: &vk_mem::Allocator,
    width: u32,
    height: u32,
    format: vk::Format,
    graphics_queue_family: u32,
    compute_queue_family: u32,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let mut sharing_mode = vk::SharingMode::EXCLUSIVE;
    let mut queue_family_indices = [graphics_queue_family, compute_queue_family];
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

  pub fn new_storage_3d(
    device: &vulkan::device::LogicalDevice,
    allocator: &vk_mem::Allocator,
    width: u32,
    height: u32,
    depth: u32,
    format: vk::Format,
    graphics_queue_family: u32,
    compute_queue_family: u32,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let mut sharing_mode = vk::SharingMode::EXCLUSIVE;
    let mut queue_family_indices = [graphics_queue_family, compute_queue_family];
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
      .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
      .sharing_mode(sharing_mode)
      .queue_family_indices(&queue_family_indices[..queue_count])
      .samples(vk::SampleCountFlags::TYPE_1);

    let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
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

  pub fn new_2d(
    device: &vulkan::device::LogicalDevice,
    allocator: &vk_mem::Allocator,
    command_buffer: vk::CommandBuffer,
    discard_pool: &DiscardPool,
    timeline: u64,
    texture: &Texture,
    usage: vk::ImageUsageFlags,
    debug_name: &str,
  ) -> GpuResult<Self> {
    let image_size = (texture.data.len()) as vk::DeviceSize;
    if image_size == 0 {
      return Err(GpuError::InvalidArgument);
    }

    let vma_allocator = allocator.get_raw();

    // 1. Create staging buffer (CPU-visible) and copy data
    let staging_buffer_info = vk::BufferCreateInfo::default()
      .size(image_size)
      .usage(vk::BufferUsageFlags::TRANSFER_SRC);
    let staging_alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::Auto,
      flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED,
      required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
      ..Default::default()
    };
    let (staging_buffer, staging_allocation, staging_alloc_info) =
      unsafe { allocator.create_buffer_get_info(&staging_buffer_info, &staging_alloc_info) }
        .with_name(
          device,
          &alloc::format!("VkBuffer_New2D_Staging_{}", debug_name),
        )?;

    unsafe {
      core::ptr::copy_nonoverlapping(
        texture.data.as_ptr(),
        staging_alloc_info.mapped_data as *mut u8,
        texture.data.len(),
      );
    }

    if !unsafe { allocator.get_allocation_memory_properties(&staging_allocation) }
      .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
    {
      allocator.flush_allocation(&staging_allocation, 0, vk::WHOLE_SIZE)?;
    }

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
    allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;

    let (image, mut alloc, _alloc_info) =
      unsafe { allocator.create_image_with_alloc_info(&image_info, &allocation_create_info) }
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
      .buffer_offset(0)
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
        staging_buffer,
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

    // 6. Schedule staging buffer for destruction.
    discard_pool.discard_buffer(vma_allocator, staging_buffer, staging_allocation, timeline);

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

// TODO: move up to frame.rs
bitflags::bitflags! {
  pub(super) struct TextureFlags: u32 {
    const FlagAlbedo = 1u32 << 0;
    const FlagNormal = 1u32 << 1;
    const FlagRoughness = 1u32 << 2;
    const FlagAo  = 1u32 << 3;
  }
}

impl TextureFlags {
  pub fn count() -> usize {
    4
  }
}

#[repr(C)]
pub(super) struct ForwardMeshRenderResourcePushData {
  model_view_projection: [f32; 16],
  model: [f32; 16],
  sun_pos: [f32; 3],
  texture_flags: TextureFlags,
  sun_color: [f32; 4],
}
sa::const_assert!(core::mem::size_of::<ForwardMeshRenderResourcePushData>() == 160);

impl Default for ForwardMeshRenderResourcePushData {
  fn default() -> Self {
    Self {
      model_view_projection: Default::default(),
      model: Default::default(),
      sun_pos: Default::default(),
      texture_flags: TextureFlags::empty(),
      sun_color: Default::default(),
    }
  }
}

pub(super) struct SunRenderResource {
  pub resolution: (u32, u32, u32),
  pub image: Option<Image>,
  pub descriptor_set: Option<NonZeroHandle<vk::DescriptorSet>>,
  pub is_generated: bool,
  pub compute_descriptor_pool: Option<vk::DescriptorPool>,
  pub compute_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
  pub compute_descriptor_set: Option<vk::DescriptorSet>,
  pub compute_pipeline: Option<crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::Pipeline>>,
  pub compute_pipeline_layout: Option<vk::PipelineLayout>,
  pub params_buffer: Option<vk::Buffer>,
  pub params_alloc: Option<vk_mem::Allocation>,
}

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
  pub sky_image: Option<Image>,
  /// Note: Purposefully leaked! (TODO: if this creates problems, do better.)
  pub descriptor_set: NonZeroHandle<vk::DescriptorSet>,
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
  pub fn frontend_texture_flags(&self) -> crate::simulation::comet::TextureFlags {
    let mut flags = crate::simulation::comet::TextureFlags::empty();
    if self.albedo_image.is_some() {
      flags |= crate::simulation::comet::TextureFlags::ALBEDO;
    }
    if self.normal_image.is_some() {
      flags |= crate::simulation::comet::TextureFlags::NORMAL;
    }
    if self.roughness_image.is_some() {
      flags |= crate::simulation::comet::TextureFlags::ROUGHNESS;
    }
    if self.ao_image.is_some() {
      flags |= crate::simulation::comet::TextureFlags::AO;
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

  /// Safety:
  /// - `descriptor_set` should have been allocated with archetype descriptor set and
  /// match the given arguments
  /// - `sampler` should outlive this object
  #[allow(clippy::too_many_arguments)]
  pub(super) unsafe fn new(
    device: &vulkan::device::LogicalDevice,
    allocator: &vk_mem::Allocator,
    command_buffer: vk::CommandBuffer,
    discard_pool: &DiscardPool,
    timeline: u64,
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
      discard_pool,
      timeline,
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
      discard_pool,
      timeline,
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
      discard_pool,
      timeline,
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
}

impl DiscardableResource for FrameResource {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    match self {
      Self::ForwardMeshRenderResource(resource) => {
        resource.discard(device, discard_pool, timeline);
      }
    }
  }
}

pub(super) struct TextRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub pipeline_key: Option<PipelineKey>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub descriptor_pool: Option<NonZeroHandle<vk::DescriptorPool>>,
  pub descriptor_set: Option<vk::DescriptorSet>,
  pub font_texture: Option<Image>,
  pub font_sampler: Option<vk::Sampler>,
  pub font_atlas: Option<crate::scene::text::FontAtlas>,
  pub allocator_raw: Option<vk_mem::ffi::VmaAllocator>,
}

unsafe impl Sync for TextRenderResourceArchetype {}
unsafe impl Send for TextRenderResourceArchetype {}

impl DiscardableResource for TextRenderResourceArchetype {
  fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
    if let Some(pool) = self.descriptor_pool.take() {
      discard_pool.discard_type_erased(
        FunctionalDeviceResource::new(
          pool.get(),
          |pool, device| unsafe {
            device.destroy_descriptor_pool(pool, None);
          },
        ),
        timeline,
      );
    }
    if let Some(sampler) = self.font_sampler.take() {
      discard_pool.discard_type_erased(
        FunctionalDeviceResource::new(
          sampler,
          |sampler, device| unsafe {
            device.destroy_sampler(sampler, None);
          },
        ),
        timeline,
      );
    }
    if let Some(mut texture) = self.font_texture.take() {
      if let Some(allocator_raw) = self.allocator_raw {
        discard_pool.discard_image_view(texture.image_view.get(), timeline);
        discard_pool.discard_image(
          allocator_raw,
          texture.image.get(),
          texture.allocation,
          timeline,
        );
      }
    }
  }
}

pub(super) struct BvhRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub pipeline_key: Option<PipelineKey>,
}

impl DiscardableResource for BvhRenderResourceArchetype {
  fn discard(&mut self, _device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_pipeline_layout(self.pipeline_layout.get(), timeline);
  }
}

impl BvhRenderResourceArchetype {
  pub unsafe fn new(device: &vulkan::device::LogicalDevice) -> GpuResult<Self> {
    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(144)]; // mat4 (64) + vec4 * 5 (80) = 144 bytes
      
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .push_constant_ranges(&push_constant_ranges);
      
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      pipeline_key: None,
    })
  }

  pub fn with_pipeline_key(self, pipeline_key: PipelineKey) -> Self {
    Self {
      pipeline_key: Some(pipeline_key),
      ..self
    }
  }
}

pub(super) struct MinimapRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub pipeline_key: Option<PipelineKey>,
}

impl MinimapRenderResourceArchetype {
  pub unsafe fn new(device: &vulkan::device::LogicalDevice) -> GpuResult<Self> {
    let push_constant_ranges = [vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: 544,
    }];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };
    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      pipeline_key: None,
    })
  }

  pub fn with_pipeline_key(self, pipeline_key: PipelineKey) -> Self {
    Self {
      pipeline_key: Some(pipeline_key),
      ..self
    }
  }

  pub fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

pub(super) struct CursorRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub push_contant_ranges: Vec<vk::PushConstantRange>,
  pub graphics_info: Option<GraphicsInfo>,
  pub pipeline_key: Option<PipelineKey>,
}

unsafe impl Sync for CursorRenderResourceArchetype {}
unsafe impl Send for CursorRenderResourceArchetype {}

impl CursorRenderResourceArchetype {
  pub fn with_graphics_info(self, graphics_info: GraphicsInfo) -> Self {
    let pipeline_key = graphics_info.pipeline_key();
    Self {
      graphics_info: Some(graphics_info),
      pipeline_key: Some(pipeline_key),
      ..self
    }
  }

  pub unsafe fn new(device: &vulkan::device::LogicalDevice) -> GpuResult<Self> {
    let push_contant_ranges = alloc::vec![vk::PushConstantRange {
      stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
      offset: 0,
      size: core::mem::size_of::<crate::gpu::CursorPushConstants>() as u32,
    }];

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
      p_push_constant_ranges: push_contant_ranges.as_ptr(),
      push_constant_range_count: push_contant_ranges.len() as u32,
      ..Default::default()
    };

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_CursorRenderResourceArchetype")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      push_contant_ranges,
      graphics_info: None,
      pipeline_key: None,
    })
  }

  pub fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

pub(super) struct SkyRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub pipeline_key: Option<PipelineKey>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub descriptor_set: Option<NonZeroHandle<vk::DescriptorSet>>,
}

impl SkyRenderResourceArchetype {
  pub fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
  }
}

pub(super) struct GridRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub pipeline_key: Option<PipelineKey>,
}

impl GridRenderResourceArchetype {
  pub fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
  }
}

pub(super) struct SunRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layout: NonZeroHandle<vk::DescriptorSetLayout>,
  pub push_contant_ranges: Vec<vk::PushConstantRange>,
  pub graphics_info: Option<GraphicsInfo>,
  pub pipeline_key: Option<PipelineKey>,
}

unsafe impl Sync for SunRenderResourceArchetype {}
unsafe impl Send for SunRenderResourceArchetype {}

impl SunRenderResourceArchetype {
  pub fn with_graphics_info(self, graphics_info: GraphicsInfo) -> Self {
    let pipeline_key = graphics_info.pipeline_key();
    Self {
      graphics_info: Some(graphics_info),
      pipeline_key: Some(pipeline_key),
      ..self
    }
  }

  pub unsafe fn new(device: &vulkan::device::LogicalDevice) -> GpuResult<Self> {
    let push_contant_ranges = alloc::vec![vk::PushConstantRange {
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
      .push_constant_ranges(&push_contant_ranges)
      .set_layouts(&set_layouts);

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }
      .with_name(device, "VkPipelineLayout_SunRenderResourceArchetype")?;

    Ok(Self {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(descriptor_set_layout) },
      push_contant_ranges,
      graphics_info: None,
      pipeline_key: None,
    })
  }

  pub fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    let layout = self.pipeline_layout.get();
    discard_pool.discard_pipeline_layout(layout, timeline);
    discard_pool.discard_descriptor_set_layout(self.descriptor_set_layout.get(), timeline);
  }
}

/// To be destroyed before descriptor pool
pub(super) struct ForwardMeshRenderResourceArchetype {
  pub pipeline_layout: NonZeroHandle<vk::PipelineLayout>,
  pub descriptor_set_layouts: Vec<NonZeroHandle<vk::DescriptorSetLayout>>,
  pub push_contant_ranges: Vec<vk::PushConstantRange>,
  // 0 = vertex, 1 = fragment
  pub specialization_constants: [Vec<vk::SpecializationMapEntry>; 2],
  // 0 = vertex, 1 = fragment
  pub specialization_constants_values: [Vec<u8>; 2],
  /// Populated after with_graphics_info
  pub graphics_info: Option<GraphicsInfo>,
  /// Populated after with_pipeline_key
  pub pipeline_key: Option<PipelineKey>,
  pub outline_pipeline_key: Option<PipelineKey>,

  pub dummy_texture_handle: Image,
  /// Necessary evil for discard. assumes it outlives this object
  allocator_raw: vk_mem::ffi::VmaAllocator,
}

unsafe impl Sync for ForwardMeshRenderResourceArchetype {}
unsafe impl Send for ForwardMeshRenderResourceArchetype {}

impl ForwardMeshRenderResourceArchetype {
  pub fn with_graphics_info(self, graphics_info: GraphicsInfo) -> Self {
    let pipeline_key = graphics_info.pipeline_key();
    Self {
      graphics_info: Some(graphics_info),
      pipeline_key: Some(pipeline_key),
      ..self
    }
  }

  pub fn with_outline_pipeline_key(self, outline_pipeline_key: PipelineKey) -> Self {
    Self {
      outline_pipeline_key: Some(outline_pipeline_key),
      ..self
    }
  }

  /// Safety:
  /// - `pipeline_key` must refer to a pipeline created with `vertex_shader` and `fragment_shader`,
  pub unsafe fn new(
    device: &vulkan::device::LogicalDevice,
    vertex_shader: &Shader,
    fragment_shader: &Shader,
    allocator: &vk_mem::Allocator,
    discard_pool: &DiscardPool,
    queue: &super::Queue,
  ) -> GpuResult<Self> {
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

    for shader in [vertex_shader, fragment_shader] {
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

    let descriptor_set_layouts: Vec<vk::DescriptorSetLayout> = sorted_layouts
      .into_iter()
      .map(|(_, bindings)| {
        let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
        janitor
          .push(FunctionalDeviceResource::new(layout, |h, d| unsafe {
            d.destroy_descriptor_set_layout(h, None)
          }))
          .map_err(|_| GpuError::InvalidState)?;
        Ok(layout)
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
        // Find a range with the same offset and size to merge stage flags.
        if let Some(range) = push_constant_ranges
          .iter_mut()
          .find(|r| r.offset == block.offset && r.size == block.size)
        {
          // Merge shader stages into the existing range.
          range.stage_flags |= shader.shader_stage;
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
      .set_layouts(&descriptor_set_layouts)
      .push_constant_ranges(&push_constant_ranges);

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
      .map_err(|_| GpuError::InvalidState)?;

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
        let mut the_vec: Vec<_> = the_vec
          .iter()
          .map(|c| c.as_ref().unwrap_unchecked())
          .collect();
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
            core::ffi::CStr::from_ptr(spec_const.name)
              .to_str()
              .unwrap_or("")
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
        .map_err(|_| GpuError::InvalidState)?;

      let command_buffer = {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
          .command_pool(command_pool)
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(1);
        unsafe { device.allocate_command_buffers(&alloc_info) }?[0]
      };

      let dummy_texture = Texture {
        data: {
          let mut the_vec = Vec::with_capacity(1);
          the_vec.push(0);
          the_vec
        },
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
        discard_pool,
        NEVER_DISCARD_TIMELINE,
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
        device.queue_submit(queue.handle, &submits, vk::Fence::null())?;
        device.queue_wait_idle(queue.handle)?;
      };

      image
    }?;

    janitor.clear();
    Ok(Self {
      pipeline_layout: NonZeroHandle::new(pipeline_layout).unwrap(),
      descriptor_set_layouts: descriptor_set_layouts
        .into_iter()
        .map(|l| NonZeroHandle::new(l).unwrap())
        .collect(),
      push_contant_ranges: push_constant_ranges,
      specialization_constants,
      specialization_constants_values,
      pipeline_key: None,
      outline_pipeline_key: None,
      graphics_info: None,
      dummy_texture_handle,
      allocator_raw: allocator.get_raw(),
    })
  }

  pub fn create_descriptor_set_from_layout_at_index(
    &self,
    device: &vulkan::device::LogicalDevice,
    descriptor_pools: &sync::Arc<DescriptorPools>,
    discard_pool: &DiscardPool,
    index: usize,
    debug_name: &str,
  ) -> GpuResult<NonZeroHandle<vk::DescriptorSet>> {
    const NEVER_DISCARD_TIMELINE: u64 = u64::MAX;

    let layout = self
      .descriptor_set_layouts
      .get(index)
      .ok_or(GpuError::InvalidArgument)?
      .get();
    descriptor_pools.allocate(
      device,
      layout,
      discard_pool,
      NEVER_DISCARD_TIMELINE,
      debug_name,
    )
  }
}

impl DiscardableResource for ForwardMeshRenderResourceArchetype {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    discard_pool.discard_type_erased(
      FunctionalDeviceResource::new(
        self.pipeline_layout.get(),
        |pipeline_layout, device| unsafe {
          device.destroy_pipeline_layout(pipeline_layout, None);
        },
      ),
      timeline,
    );
    // view discarded _before_ image
    discard_pool.discard_image_view(self.dummy_texture_handle.image_view.get(), timeline);
    discard_pool.discard_image(
      self.allocator_raw,
      self.dummy_texture_handle.image.get(),
      self.dummy_texture_handle.allocation,
      timeline,
    );
    for layout in &self.descriptor_set_layouts {
      unsafe {
        device.destroy_descriptor_set_layout(layout.get(), None);
      }
    }
  }
}

/// Structure which holds vulkan resources which are common to all frame instances of a given
/// render resource type
/// These are destroyed when the [`super::Device`] instance is dropped, ie when the [`DiscardPool`]
/// is dropped, through the [`DiscardableResource`] trait
pub(super) enum FrameResourceArchetype {
  ForwardMeshRenderResource(ForwardMeshRenderResourceArchetype),
}

impl DiscardableResource for FrameResourceArchetype {
  fn discard(&mut self, device: &ash::Device, discard_pool: &DiscardPool, timeline: u64) {
    match self {
      FrameResourceArchetype::ForwardMeshRenderResource(forward_mesh_render_resource_archetype) => {
        forward_mesh_render_resource_archetype.discard(device, discard_pool, timeline);
      }
    }
  }
}

/// Reusable helper function to perform the explicit staging buffer upload pattern.
fn create_buffer_with_staging<T: Copy>(
  device: &vulkan::device::LogicalDevice,
  allocator: &vk_mem::Allocator,
  command_buffer: vk::CommandBuffer,
  discard_pool: &DiscardPool,
  timeline: u64,
  data: &[T],
  usage: vk::BufferUsageFlags,
  debug_name: &str,
) -> GpuResult<Buffer> {
  let buffer_size = (core::mem::size_of::<T>() * data.len()) as vk::DeviceSize;
  if buffer_size == 0 {
    return Err(GpuError::InvalidArgument);
  }

  let vma_allocator = allocator.get_raw();

  // 1. Create staging buffer (CPU-visible)
  let staging_buffer_info = vk::BufferCreateInfo::default()
    .size(buffer_size)
    .usage(vk::BufferUsageFlags::TRANSFER_SRC);
  let staging_alloc_info = vk_mem::AllocationCreateInfo {
    usage: vk_mem::MemoryUsage::Auto,
    flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
      | vk_mem::AllocationCreateFlags::MAPPED,
    required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
    ..Default::default()
  };
  let (staging_buffer, staging_allocation, staging_alloc_info) =
    unsafe { allocator.create_buffer_get_info(&staging_buffer_info, &staging_alloc_info) }
      .with_name(device, &alloc::format!("VkBuffer_Staging_{}", debug_name))?;

  // 2. Create device buffer (GPU-local). In case of failure, we clean up the staging buffer.
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
    match unsafe { allocator.create_buffer(&device_buffer_info, &device_alloc_info) }
      .with_name(device, &alloc::format!("VkBuffer_{}", debug_name))
    {
      Ok(result) => result,
      Err(err) => {
        unsafe {
          vk_mem::ffi::vmaDestroyBuffer(
            vma_allocator,
            staging_buffer,
            staging_allocation.get_raw(),
          );
        }
        return Err(err.into());
      }
    }
  };

  // 3. Copy data to staging buffer
  unsafe {
    core::ptr::copy_nonoverlapping(
      data.as_ptr(),
      staging_alloc_info.mapped_data as *mut T,
      data.len(),
    );
  }
  if !unsafe { allocator.get_allocation_memory_properties(&staging_allocation) }
    .contains(vk::MemoryPropertyFlags::HOST_COHERENT)
  {
    allocator.flush_allocation(&staging_allocation, 0, vk::WHOLE_SIZE)?;
  }

  // 4. Record copy command
  let copy_region = vk::BufferCopy::default().size(buffer_size);
  unsafe {
    device.cmd_copy_buffer(
      command_buffer,
      staging_buffer,
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

  // 6. Schedule staging buffer for destruction.
  discard_pool.discard_buffer(vma_allocator, staging_buffer, staging_allocation, timeline);

  Ok(Buffer {
    buffer: unsafe { NonZeroHandle::new_unchecked(device_buffer) },
    allocation: device_allocation,
  })
}

// Helper to map descriptor types from spirv-reflect to ash and handle unsupported cases.
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
