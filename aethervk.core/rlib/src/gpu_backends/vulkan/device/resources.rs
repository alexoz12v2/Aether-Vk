use core::ptr;

use ash::vk::{self, PFN_vkGetSemaphoreCounterValue};
use alloc::{collections::VecDeque, sync};
use vk_mem::PoolCreateInfo;

use crate::gpu_backends::vulkan::device::descriptors;

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
  Pipeline(vk::Pipeline),
  DescriptorPool(vk::DescriptorPool, sync::Arc<descriptors::DescriptorPools>),
  // TODO other types of resources as needed
}

struct BufferDiscard {
  buffer: vk::Buffer,
  alloc: vk_mem::Allocation,
}
struct ImageDiscard {
  image: vk::Image,
  alloc: vk_mem::Allocation,
}

/// Structure associated to the main Timeline Semaphore provided by Device
/// Note: this must not outlive device, hence don't expose it outside
pub(super) struct DiscardPool {
  device: vk::Device,
  allocator: vk_mem::ffi::VmaAllocator, // non owning copy

  // functions I need
  get_semaphore_counter_value: PFN_vkGetSemaphoreCounterValue,

  items: spin::Mutex<TimelineQueue<DiscardItem>>,
}

unsafe impl Sync for DiscardPool {}
unsafe impl Send for DiscardPool {}

impl DiscardPool {
  /// Safety: device and allocator should outlive Self
  pub unsafe fn new(device: &ash::Device, allocator: &vk_mem::Allocator, cap: usize) -> Self {
    Self {
      device: device.handle(),
      get_semaphore_counter_value: device.fp_v1_2().get_semaphore_counter_value,
      allocator: allocator.get_raw(),
      items: spin::Mutex::new(TimelineQueue::with_capacity(cap)),
    }
  }

  // TODO all other types of resources as needed
  pub fn discard_buffer(&self, buffer: vk::Buffer, alloc: vk_mem::Allocation, timeline: u64) {
    let mut q = self.items.lock();
    q.push(
      timeline,
      DiscardItem::Buffer(BufferDiscard { buffer, alloc }),
    );
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

  pub fn destroy_discarded_resources_all(&self) {
    self.destroy_discarded_resources_internal(u64::MAX);
  }

  /// safety: `sem` needs to be a valid timeline semaphore
  pub unsafe fn destroy_discarded_resources_timeline(
    &self,
    sem: vk::Semaphore,
  ) -> ash::prelude::VkResult<()> {
    let mut timeline = 0u64;
    unsafe {
      (self.get_semaphore_counter_value)(self.device, sem, ptr::from_mut(&mut timeline)).result()?
    };
    self.destroy_discarded_resources_internal(timeline);
    Ok(())
  }

  fn destroy_discarded_resources_internal(&self, timeline: u64) {
    let mut items = self.items.lock();
    items.drain_ready(timeline, |item| match item {
      DiscardItem::Buffer(BufferDiscard { buffer, alloc }) => unsafe {
        vk_mem::ffi::vmaDestroyBuffer(self.allocator, buffer, alloc.get_raw());
      },
      DiscardItem::Image(image_discard) => todo!(),
      DiscardItem::Pipeline(pipeline) => todo!(),
      DiscardItem::DescriptorPool(pool, manager) => {
        // return the pool to the manager for recycling
        manager.recycle(pool);
      }
    });
  }
}
