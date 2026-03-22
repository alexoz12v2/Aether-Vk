use core::ptr;

use ash::vk::{self, PFN_vkGetSemaphoreCounterValue};
use alloc::{collections::VecDeque, sync};
use vk_mem::PoolCreateInfo;

use crate::gpu_backends::vulkan::device::{DeviceResource, descriptors};

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
  DescriptorPool(vk::DescriptorPool, sync::Arc<descriptors::DescriptorPools>),
  // TODO other types of resources as needed
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

/// Structure associated to the main Timeline Semaphore provided by Device
/// Note: this must not outlive device, hence don't expose it outside
pub(super) struct DiscardPool {
  items: spin::Mutex<TimelineQueue<DiscardItem>>,
}

unsafe impl Sync for DiscardPool {}
unsafe impl Send for DiscardPool {}

impl DiscardPool {
  /// Safety: device and allocator should outlive Self
  pub unsafe fn new(cap: usize) -> Self {
    Self {
      items: spin::Mutex::new(TimelineQueue::with_capacity(cap)),
    }
  }

  // TODO all other types of resources as needed
  pub fn discard_buffer(
    &self,
    allocator: vk_mem::ffi::VmaAllocator,
    buffer: vk::Buffer,
    alloc: vk_mem::Allocation,
    timeline: u64,
  ) {
    let mut q = self.items.lock();
    q.push(
      timeline,
      DiscardItem::Buffer(BufferDiscard {
        buffer,
        alloc,
        allocator,
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
    let mut q = self.items.lock();
    q.push(
      timeline,
      DiscardItem::Image(ImageDiscard {
        image,
        alloc,
        allocator,
      }),
    );
  }

  pub fn discard_image_view(&self, image_view: vk::ImageView, timeline: u64) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::ImageView(image_view));
  }

  pub fn discard_descriptor_pool(
    &self,
    pool: vk::DescriptorPool,
    manager: sync::Arc<descriptors::DescriptorPools>,
    timeline: u64,
  ) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::DescriptorPool(pool, manager));
  }

  pub fn discard_pipeline(&self, pipeline: vk::Pipeline, timeline: u64) {
    let mut q = self.items.lock();
    q.push(timeline, DiscardItem::Pipeline(pipeline));
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
    items.drain_ready(timeline, |item| match item {
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
      },
      DiscardItem::DescriptorPool(pool, manager) => {
        // return the pool to the manager for recycling
        manager.recycle(device, pool);
      }
      DiscardItem::ImageView(image_view) => unsafe {
        device.destroy_image_view(image_view, None);
      },
    });
  }
}

impl DeviceResource for DiscardPool {
  fn cleanup(&mut self, device: &ash::Device) {
    self.destroy_discarded_resources_all(device);
  }
}

/// Note: Caller should also provide its own timeline value
pub(super) trait DiscardPoolCaller {
  fn discard_buffer(&self, buffer: vk::Buffer, alloc: vk_mem::Allocation);
  fn discard_image(&self, image: vk::Image, alloc: vk_mem::Allocation);
  fn discard_image_view(&self, image_view: vk::ImageView);
  fn discard_pipeline(&self, pipeline: vk::Pipeline);
  fn discard_descriptor_pool(&self, pool: vk::DescriptorPool);
  fn destroy_discarded_resources(&self);
  fn destroy_discarded_resources_all(&self);
}
