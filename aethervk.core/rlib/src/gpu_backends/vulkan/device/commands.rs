use core::ptr;

use crate::{gpu_backends::vulkan::device::DeviceResource, types::{GpuResult, SpscQueue}};
use aethervk_oshal_rlib::os::thread::ThreadId;
use alloc::{boxed::Box, collections::btree_map::BTreeMap};
use ash::vk;

// TODO: implement trait/function to hash some compile time string
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct CommandBufferId(pub u64);

struct ThreadPools {
  recycled: SpscQueue<vk::CommandPool>,
  active: Option<vk::CommandPool>,
  cmd_cache: BTreeMap<vk::CommandPool, BTreeMap<CommandBufferId, vk::CommandBuffer>>,
}

/// alternative to avoid boxing: Use SlotMap. Drawback for alternative: you need to store
/// a BTreeMap mapping ThreadId to the new_key_type
pub(super) struct CommandPools {
  registry: spin::RwLock<BTreeMap<ThreadId, Box<ThreadPools>>>,
  queue_family_index: u32,
  spsc_capacity: usize,
}

unsafe impl Sync for CommandPools {}
unsafe impl Send for CommandPools {}

impl CommandPools {
  pub(super) fn new(queue_family_index: u32) -> Self {
    Self {
      registry: spin::RwLock::new(BTreeMap::new()),
      queue_family_index,
      spsc_capacity: 64,
    }
  }
  /// Centralized recycling: returns a pool from a discard thread to the owner
  pub(super) fn recycle(&self, device: &ash::Device, pool: vk::CommandPool, owner: ThreadId) {
    let registry = self.registry.read();
    if let Some(tp) = registry.get(&owner) {
      if !tp.recycled.try_push(pool) {
        // Buffer full, destroy handle immediately to prevent leak
        self.destroy_pool_internal(device, pool);
      }
    } else {
      // Owner thread already shut down
      self.destroy_pool_internal(device, pool);
    }
  }

  pub(super) fn allocate_primary(
    &self,
    device: &ash::Device,
    tid: ThreadId,
    id: CommandBufferId,
  ) -> GpuResult<vk::CommandBuffer> {
    self.allocate_internal(device, tid, id, true)
  }

  fn allocate_internal(
    &self,
    device: &ash::Device,
    tid: ThreadId,
    id: CommandBufferId,
    is_primary: bool,
  ) -> GpuResult<vk::CommandBuffer> {
    if is_primary {
      let mut registry = self.registry.write();
      // Pseudo-TLS lookup/initialization
      let tp = registry.entry(tid).or_insert_with(|| {
        Box::new(ThreadPools {
          recycled: SpscQueue::new(self.spsc_capacity),
          active: None,
          cmd_cache: BTreeMap::new(),
        })
      });
      // 1. ensure pool is active
      if tp.active.is_none() {
        tp.active = if let Some(cmd_pool) = tp.recycled.try_pop() {
          Some(cmd_pool)
        } else {
          Some(self.create_pool_internal(device)?)
        };
      }
      let active_pool = unsafe { tp.active.unwrap_unchecked() };

      // 2. Check cache
      let pool_cache = tp.cmd_cache.entry(active_pool).or_default();
      if let Some(&cmd) = pool_cache.get(&id) {
        return Ok(cmd);
      }

      // 3. Allocate new buffer
      let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(active_pool);
      let new_cmd = unsafe {
        let mut cmd: vk::CommandBuffer = vk::CommandBuffer::default();
        let vk_res =
          (device.fp_v1_0().allocate_command_buffers)(device.handle(), &allocate_info, ptr::from_mut(&mut cmd));
        vk_res.result_with_success(cmd)
      }?;
      pool_cache.insert(id, new_cmd);

      Ok(new_cmd)
    } else {
      todo!()
    }
  }

  fn destroy_pool_internal(&self, device: &ash::Device, pool: vk::CommandPool) {
    unsafe { device.destroy_command_pool(pool, None) };
  }

  fn create_pool_internal(&self, device: &ash::Device) -> ash::prelude::VkResult<vk::CommandPool> {
    let create_info = vk::CommandPoolCreateInfo::default()
      .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
      .queue_family_index(self.queue_family_index);
    unsafe {
      device.create_command_pool(
        &create_info,
        None)
    }
  }
}

impl DeviceResource for CommandPools {
  fn cleanup(&mut self, device: &ash::Device) {
    let registry = self.registry.get_mut(); // get_mut if access without locking is enough
    for (_tid, thread_pools) in registry.iter() {
      // destroy active pool if exists
      if let Some(active_pool) = thread_pools.active {
        unsafe { device.destroy_command_pool(active_pool, None) };
      }
      // destroy any recycled pools
      while let Some(recycled_pool) = thread_pools.recycled.try_pop() {
        unsafe { device.destroy_command_pool(recycled_pool, None) };
      }
      // Note: CommandBuffers in cmd_cache do not need explicit destruction because they are implicitly freed when
      // their pool is freed.
    }
  }
}

pub(super) trait CommandPoolsCaller {
  fn allocate_primary(&self, id: CommandBufferId) -> GpuResult<vk::CommandBuffer>;
  fn recycle(&self, pool: vk::CommandPool, owner: ThreadId);
}
