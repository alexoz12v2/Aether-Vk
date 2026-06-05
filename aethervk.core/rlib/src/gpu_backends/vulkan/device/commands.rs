//! commands module.

use crate::{
  gpu::CommandBufferHandle,
  gpu_backends::vulkan::device::{DeviceResource, LogicalDevice, VulkanDebugNameExt},
  types::{GpuError, GpuResult, SpscQueue},
};
use aethervk_oshal_rlib::os::native::ThreadId;
use alloc::{boxed::Box, collections::btree_map::BTreeMap};
use ash::vk;
use core::{fmt, fmt::Formatter, ptr};
use function_name::named;

// TODO: implement trait/function to hash some compile time string
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
/// TODO: Document this item
pub(crate) struct CommandBufferId(pub u64);

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
pub(crate) struct CommandPools {
  registry: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    BTreeMap<ThreadId, Box<ThreadPools>>,
  >,
  pub queue_family_index: u32,
  spsc_capacity: usize,
}

impl fmt::Debug for CommandPools {
  fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
    f.write_str("CommandPools")?;
    f.debug_map()
      .entries(
        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.registry).iter(),
      )
      .finish()?;
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
  pub(crate) fn new(queue_family_index: u32) -> Self {
    Self {
      registry: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(BTreeMap::new()),
      queue_family_index,
      spsc_capacity: 64,
    }
  }

  /// TODO: Document this item
  #[named]
  pub(crate) fn allocate_primary(
    &self,
    device: &LogicalDevice,
    tid: ThreadId,
    id: CommandBufferId,
  ) -> GpuResult<vk::CommandBuffer> {
    self.allocate_internal(device, tid, id, true)
  }

  /// TODO: Document this item
  #[named]
  pub fn recycle(
    &self,
    tid: ThreadId,
    id: CommandBufferId,
    cmd_buf: vk::CommandBuffer,
  ) -> GpuResult<()> {
    self.recycle_internal(tid, id, cmd_buf)
  }

  #[named]
  fn recycle_internal(
    &self,
    tid: ThreadId,
    _id: CommandBufferId,
    cmd_buf: vk::CommandBuffer,
  ) -> GpuResult<()> {
    let mut registry =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&self.registry);
    // Pseudo-TLS lookup/initialization
    let tp = registry.get_mut(&tid).ok_or(GpuError::BackendSpecific(
      "Command Buffer Registry for Thread not initialized".into(),
    ))?;
    // Find the pool entry containing this cmd_buf handle.
    // We search by handle only (not by CommandBufferId) because the allocator
    // may recycle cmd_buf handles under new IDs between submission and discard
    // pool processing (e.g. via debug_sync_barrier).
    let the_buffer = tp
      .cmd_cache
      .iter_mut()
      .flat_map(|(_, map)| map.iter_mut())
      .find(|(_, (buffer, _))| *buffer == cmd_buf);
    if let Some((_, (_, used))) = the_buffer {
      *used = false;
    }
    // If not found, the cmd_buf was already recycled under a different
    // CommandBufferId by allocate_internal (e.g. via debug_sync_barrier).
    // This is benign — the handle is still tracked in the cache.
    Ok(())
  }

  #[named]
  fn allocate_internal(
    &self,
    device: &LogicalDevice,
    tid: ThreadId,
    id: CommandBufferId,
    is_primary: bool,
  ) -> GpuResult<vk::CommandBuffer> {
    if is_primary {
      crate::gpu_backends::vulkan::utils::VulkanTransaction::new(&self.registry, device)
        .prepare_write(tid, |registry, tid| {
          let tp = registry.entry(tid).or_insert_with(|| {
            Box::new(ThreadPools {
              recycled: SpscQueue::new(self.spsc_capacity),
              active: None,
              cmd_cache: BTreeMap::new(),
            })
          });

          let mut create_pool = false;
          if tp.active.is_none() {
            if let Some(cmd_pool) = tp.recycled.try_pop() {
              tp.active = Some(cmd_pool);
            } else {
              create_pool = true;
            }
          }

          let active_pool = tp.active;

          let mut found_recycled = None;
          if let Some(pool) = active_pool {
            let pool_cache = tp.cmd_cache.entry(pool).or_default();
            for (&old_id, pair) in pool_cache.iter() {
              if !pair.1 {
                found_recycled = Some((old_id, pair.0));
                break;
              }
            }
            if let Some((old_id, cmd_buf)) = found_recycled {
              pool_cache.remove(&old_id);
              pool_cache.insert(id, (cmd_buf, true));
              return Ok((false, active_pool, Some(cmd_buf))); // early out!
            }
          }

          Ok((create_pool, active_pool, None))
        })?
        .execute(|(create_pool, mut active_pool, recycled_cmd), rollback| {
          if let Some(cmd) = recycled_cmd {
            return Ok((active_pool, cmd));
          }

          if create_pool {
            let pool = self.create_pool_internal(&device.handle)?;
            rollback.defer(move |dev| unsafe { dev.destroy_command_pool(pool, None) });
            active_pool = Some(pool);
          }

          let pool = active_pool.unwrap();

          let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_pool(pool);

          let cmd = unsafe {
            let mut cmd: vk::CommandBuffer = vk::CommandBuffer::default();
            let vk_res = (device.fp_v1_0().allocate_command_buffers)(
              (&*device).handle(),
              &allocate_info,
              ptr::from_mut(&mut cmd),
            );
            vk_res.result_with_success(cmd)
          }
          .with_name(device, &alloc::format!("PrimaryCommandBuffer_{}", id.0))?;

          Ok((active_pool, cmd))
        })
        .commit(|registry, result| {
          let (active_pool, cmd) = result?;
          let tp = registry.get_mut(&tid).unwrap();
          if tp.active.is_none() {
            tp.active = active_pool;
          }

          let pool_cache = tp.cmd_cache.entry(active_pool.unwrap()).or_default();
          pool_cache.insert(id, (cmd, true));

          Ok(cmd)
        })
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
    let registry =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::get_mut(&mut self.registry); // get_mut if access without locking is enough
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
