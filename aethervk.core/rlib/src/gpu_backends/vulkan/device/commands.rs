//! commands module.

use core::{fmt, ptr};

use crate::{
  gpu::{CommandBufferHandle, GpuResourceHandle},
  gpu_backends::vulkan::device::DeviceResource,
  types::{GpuError, GpuResult, SpscQueue},
};
use aethervk_oshal_rlib::os::native::ThreadId;
use alloc::{boxed::Box, collections::btree_map::BTreeMap};
use ash::vk;
use core::fmt::{Formatter, Pointer};

// TODO: implement trait/function to hash some compile time string
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// TODO: Document this item
pub(super) struct CommandBufferId(pub u64);

impl From<CommandBufferHandle> for CommandBufferId {
  fn from(value: CommandBufferHandle) -> Self {
    Self(value.0)
  }
}

#[derive(Debug)]
struct ThreadPools {
  recycled: SpscQueue<vk::CommandPool>,
  active: Option<vk::CommandPool>,
  cmd_cache: BTreeMap<vk::CommandPool, BTreeMap<CommandBufferId, (vk::CommandBuffer, bool)>>,
}

/// alternative to avoid boxing: Use SlotMap. Drawback for alternative: you need to store
/// a BTreeMap mapping ThreadId to the new_key_type
pub(super) struct CommandPools {
  registry: spin::RwLock<BTreeMap<ThreadId, Box<ThreadPools>>>,
  queue_family_index: u32,
  spsc_capacity: usize,
}

impl fmt::Debug for CommandPools {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    f.write_str("CommandPools")?;
    f.debug_map().entries(self.registry.read().iter()).finish()?;
    f.write_str(&alloc::format!(
      "queue_family_index: {}",
      self.queue_family_index
    ))?;
    f.write_str(&alloc::format!("spsc_capacity: {}", self.spsc_capacity))
  }
}

unsafe impl Sync for CommandPools {}
unsafe impl Send for CommandPools {}

impl CommandPools {
  /// TODO: Document this item
  pub(super) fn new(queue_family_index: u32) -> Self {
    Self {
      registry: spin::RwLock::new(BTreeMap::new()),
      queue_family_index,
      spsc_capacity: 64,
    }
  }

  /// TODO: Document this item
  pub(super) fn allocate_primary(
    &self,
    device: &ash::Device,
    tid: ThreadId,
    id: CommandBufferId,
  ) -> GpuResult<vk::CommandBuffer> {
    self.allocate_internal(device, tid, id, true)
  }

  /// TODO: Document this item
  pub fn recycle(
    &self,
    tid: ThreadId,
    id: CommandBufferId,
    cmd_buf: vk::CommandBuffer,
  ) -> GpuResult<()> {
    self.recycle_internal(tid, id, cmd_buf)
  }

  fn recycle_internal(
    &self,
    tid: ThreadId,
    id: CommandBufferId,
    cmd_buf: vk::CommandBuffer,
  ) -> GpuResult<()> {
    let mut registry = self.registry.write();
    // Pseudo-TLS lookup/initialization
    let tp = registry.get_mut(&tid).ok_or(GpuError::BackendSpecific(
      "Command Buffer Registry for Thread not initialized".into(),
    ))?;
    // find pool whose buffer is same as this
    let the_buffer = tp
      .cmd_cache
      .iter_mut()
      .flat_map(|(_, map)| map.iter_mut())
      .filter_map(|(the_id, pair)| if *the_id == id { Some(pair) } else { None })
      .find(|(buffer, _)| *buffer == cmd_buf);
    if let Some((_, used)) = the_buffer {
      *used = false;
      Ok(())
    } else {
      Err(crate::gpu_invalid_arg!("invalid argument"))
    }
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
      let active_pool = unsafe { tp.active.as_mut().unwrap_unchecked() };

      // 2. Check cache
      let pool_cache = tp.cmd_cache.entry(*active_pool).or_default();
      
      // Find the first unused command buffer in the cache
      let mut found_recycled = None;
      for (&old_id, pair) in pool_cache.iter() {
        if !pair.1 {
          found_recycled = Some((old_id, pair.0));
          break;
        }
      }

      if let Some((old_id, cmd_buf)) = found_recycled {
        // Remove from old ID and insert with the new ID
        pool_cache.remove(&old_id);
        pool_cache.insert(id, (cmd_buf, true));
        return Ok(cmd_buf);
      }

      // 3. Allocate new buffer
      let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(*active_pool);
      let new_cmd = unsafe {
        let mut cmd: vk::CommandBuffer = vk::CommandBuffer::default();
        let vk_res = (device.fp_v1_0().allocate_command_buffers)(
          device.handle(),
          &allocate_info,
          ptr::from_mut(&mut cmd),
        );
        vk_res.result_with_success(cmd)
      }?;
      // fails if there's already something, don't care
      let _ = pool_cache.insert(id, (new_cmd, true));

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
    unsafe { device.create_command_pool(&create_info, None) }
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
