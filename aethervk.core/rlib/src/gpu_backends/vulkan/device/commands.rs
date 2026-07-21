//! commands module.
//! ARM's Mobile/Tiled GPU best practices strongly recommends against using
//! `VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT` in https://developer.arm.com/documentation/101897/0304/CPU-overheads/Optimizing-command-buffers-for-Vulkan?lang=en
//! Therefore, we lose the capability to reset singularly each command buffer and must reset the
//! entire pool by using `vkResetCommandPool`.
//! To do this without stalling the application, we implement a "Chunked Bump-Allocator Strategy"
//!
//! 1. *Queue Family Differentiation:* We group allocations dynamically by a composite key
//!    (ThreadId, QueueFamilyIndex)`.
//! 2. *Chunking & Rotation:* A single pool allocates up to `MAX_BUFFERS_PER_POOL` (eg 32). Once
//!    full, it rotates to a `pending` state, and a fresh pool takes its place
//! 3. *In-Flight Tracking:* Every allocation bumps an `in_flight` counter. Every `DiscardPool`
//!    recycle decrements it.
//! 4. *Bulk O(1) Resets:* When `in_flight == 0`, we know the GPU is 100% finished with all buffers
//!    in that pool. The `DiscardPool` thread instantly triggers a bulk `vkResetCommandPool`. This
//!    optimally resets the pool memory and puts all allocated command buffers back in the
//!    "Initial State" for free reuse

use crate::{
  gpu::{CommandBufferHandle, vulkan::device::locks::DebugTrackedRwLock},
  gpu_backends::vulkan::device::{DeviceResource, LogicalDevice, VulkanDebugNameExt},
  types::{GpuError, GpuResult, SpscQueue},
};
use aethervk_oshal_rlib::os::native::ThreadId;
use alloc::{boxed::Box, collections::btree_map::BTreeMap};
use ash::vk;
use core::ptr;
use function_name::named;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct CommandBufferId(pub u64);

impl From<CommandBufferHandle> for CommandBufferId {
  fn from(value: CommandBufferHandle) -> Self {
    Self(value.0)
  }
}

/// The max buffers a pool can allocate before it is forced to rotate.
/// This guarantees bounds on memory growth in heavily pipelined workloads.
const MAX_BUFFERS_PER_POOL: usize = 32;
/// Cap the number of pool tracking structs in pending state
const MAX_PENDING_TRACKED_POOL: usize = 24;
/// Cap the number of pool tracking structs in free state
const MAX_FREE_TRACKED_POOL: usize = 16;

/// Tracks a single Vulkan Command Pool and its bump-allocated buffers
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct TrackedPool {
  pub id: u64,
  pub pool: vk::CommandPool,
  pub allocated_buffers: [vk::CommandBuffer; MAX_BUFFERS_PER_POOL],
  pub in_flight: usize,
  pub next_free_index: usize,
  // Note: cannot use heapless vec due to repr(C) here. We need layout predictability
  pub allocated_count: usize,
}

/// implementing Pod so that we can use copy_nonoverlapping (zeroable required for pod)
unsafe impl bytemuck::Zeroable for TrackedPool {}
/// implementing Pod so that we can use copy_nonoverlapping
unsafe impl bytemuck::Pod for TrackedPool {}

impl TrackedPool {
  /// SAFETY: `dst` pointer should be a portion of zeroed out memory able to contain this object
  unsafe fn new_at_ptr(
    dst: *mut Self,
    device: &super::LogicalDevice,
    queue_family_index: u32,
    id: u64,
  ) -> ash::prelude::VkResult<()> {
    let create_info = vk::CommandPoolCreateInfo::default()
      // STRICT COMPLIANCE: Intentionally omitting RESET_COMMAND_BUFFER_BIT so the driver can use a
      // single, large, hyper-fast memory allocator (eg ARM Tiled GPU)
      .queue_family_index(queue_family_index);
    let pool = unsafe { device.create_command_pool(&create_info, None) }?;
    // SAFETY: if caller safety contitions are satisfied, then this is correct
    let this_mut = unsafe { dst.as_mut_unchecked() };
    this_mut.id = id;
    this_mut.pool = pool;
    // Zeroed out fields not written here cause caller should already given us a zero init

    Ok(())
  }

  fn reset_memory(&mut self, device: &super::LogicalDevice) -> ash::prelude::VkResult<()> {
    unsafe {
      // Instantly resets all memory in the pool.
      // Command Buffers handles are kept but implicitly returned to the Initial State
      // this assumption ^^^ should be correct, as we are not passing RELEASE flag here
      device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty())?;
    }
    self.next_free_index = 0; // rewind the bump allocator
    self.in_flight = 0;
    Ok(())
  }
}

#[derive(Debug)]
pub(crate) struct QueueFamilyPoolsInner {
  pub active: TrackedPool,
  pub pending: heapless::Vec<TrackedPool, MAX_PENDING_TRACKED_POOL>,
  pub free: heapless::Vec<TrackedPool, MAX_FREE_TRACKED_POOL>,
  /// fast mapping to know which pool a recycled command buffer belongs to
  pub buffer_to_pool:
    heapless::index_map::FnvIndexMap<vk::CommandBuffer, u64, MAX_BUFFERS_PER_POOL>,
  pub next_pool_id: u64,
}

unsafe impl bytemuck::Zeroable for QueueFamilyPoolsInner {}

#[derive(Debug)]
enum MovePoolError {
  Precondition(&'static str),
  Vulkan(ash::vk::Result),
}

impl From<MovePoolError> for GpuError {
  fn from(value: MovePoolError) -> Self {
    match value {
      MovePoolError::Precondition(s) => Self::BackendSpecific(s.to_string()),
      MovePoolError::Vulkan(e) => Self::from(e),
    }
  }
}

impl QueueFamilyPoolsInner {
  /// Moves the pool at the specified `index` from `pending` into `free`,
  /// shifting any subsequent pending pools down to fill the gap.
  /// All operations happen in-place via raw pointers to avoid stack overflow
  pub(crate) fn move_pending_to_free_at(
    &mut self,
    index: usize,
    then_vk_func: Option<impl FnOnce(&mut TrackedPool) -> ash::prelude::VkResult<()>>,
  ) -> Result<(), MovePoolError> {
    use core::ptr::{copy, copy_nonoverlapping};

    let pending_len = self.pending.len();
    let free_len = self.free.len();

    // - Validate preconditions
    if index >= pending_len {
      return Err(MovePoolError::Precondition(
        "Index out of bounds for pending queue",
      ));
    }
    if free_len == self.free.capacity() {
      return Err(MovePoolError::Precondition(
        "Cannot move to free: free pool vector is full",
      ));
    }

    unsafe {
      let pending_ptr = self.pending.as_mut_ptr();
      let free_dst_ptr = self.free.as_mut_ptr().add(free_len);

      // Pointer to the specific element we are removing from `pending`
      let target_pending_ptr = pending_ptr.add(index);

      // 1. Move `pending[index]` to the next available slot in `free`.
      // These memory regions do not overlap, hence `copy_nonoverlapping`
      copy_nonoverlapping(target_pending_ptr, free_dst_ptr, 1);

      if let Some(vk_func) = then_vk_func {
        let pool_mut = free_dst_ptr.as_mut_unchecked();
        if let Err(vk_err) = vk_func(pool_mut) {
          // Rollback: Vulkan function failed. We havent shifted `pending` or touched any lengths
          // yet. This means that we can safely return.
          return Err(MovePoolError::Vulkan(vk_err));
        }
      }

      // 2. Shift elements AFTER the index down by 1 to fill the gap
      // Note: This is O(N). If order doesn't matter, we can take the last element and put it in the
      // hole, breaking the existing ordering, but speeding up the operation into a O(1) swap.
      let elements_to_shift = pending_len - index - 1;
      if elements_to_shift > 0 {
        // Source: the element immediately after `index`
        let shift_src_ptr = pending_ptr.add(index + 1);
        // Destination: the slot at `index` we just vacated
        let shift_dst_ptr = target_pending_ptr;
        // Memory regions do overlap, therefore copy.
        copy(shift_src_ptr, shift_dst_ptr, elements_to_shift);
      }

      // 3. Update vector lengths to reflect the transfer
      self.pending.set_len(pending_len - 1);
      self.free.set_len(free_len + 1);

      Ok(())
    }
  }

  /// Checks if we have the conditions to rotate a pool from `free` to `active`.
  /// Returns true if there is at least 1 free slot in `pending` and at least 1 occupied slot in
  /// `free`.
  pub(crate) fn can_rotate(&self) -> bool {
    self.pending.len() < self.pending.capacity() && !self.free.is_empty()
  }

  // Moves `active` to `pending`, and allocates a new pool directly into `active`.
  // Bypass the stack entirely and safely recovers if Vulkan OOMs (out of memory error)
  pub(crate) fn move_active_to_pending_and_allocate_new(
    &mut self,
    device: &super::LogicalDevice,
    queue_family_index: u32,
    id: u64,
  ) -> GpuResult<()> {
    use core::ptr::copy_nonoverlapping;
    // Ensure free is empty
    debug_assert!(
      self.free.is_empty(),
      "Expected free pools to be empty before allocating a new one"
    );

    // we must have room in the pending to store the current active pool (caller checked)
    if self.pending.len() >= self.pending.capacity() {
      return Err(GpuError::BackendSpecific(
        "Pending queue is full".to_string(),
      ));
    }

    unsafe {
      let pending_len = self.pending.len();
      let pending_dst_ptr = self.pending.as_mut_ptr().add(pending_len);
      let active_ptr: *mut TrackedPool = &mut self.active;

      // Back up the current `active` pool in the next stot in pending.
      copy_nonoverlapping(active_ptr, pending_dst_ptr, 1);

      // Attempt to initialize the new pool directly into the now-available slot in `active`
      match TrackedPool::new_at_ptr(active_ptr, device, queue_family_index, id) {
        Ok(_) => {
          // Success: The new pool is now in `active`.
          // The old pool is safely sitting in `pending`, so we officially publish the new slot
          self.pending.set_len(pending_len + 1);
          Ok(())
        }
        Err(e) => {
          // Vulkan Failure: `active` contains  half-initialized garbage.
          // We must restore the old `active` pool from our backup so that `Drop` cleans it up
          // correctly later
          copy_nonoverlapping(pending_dst_ptr, active_ptr, 1);

          // Return error and do not increase pending len
          Err(GpuError::from(e))
        }
      }
    }
  }

  /// Moves `active` to the end of `pending`, and pops the last element of `free` into `active`. All
  /// memory moves happen strigcly in place via pointers to avoid stack overflow
  pub(crate) fn rotate_active_pool(&mut self) -> Result<(), &'static str> {
    use core::ptr::copy_nonoverlapping;
    // - Verify we have space/elements before doing any unsafe memory operations
    if self.pending.len() == self.pending.capacity() {
      return Err("Cannot move active to pending: pending is full");
    }
    if self.free.is_empty() {
      return Err("Cannot replace active: free pool is empty");
    }

    unsafe {
      let pending_len = self.pending.len();
      let free_len = self.free.len();

      // - Get pointers
      // Pointer to the current active poool
      let active_ptr: *mut TrackedPool = &mut self.active;
      // `as_mut_ptr()` yields a pointer to the start of the backing array.
      // By adding `pending_len`, we get a pointer to the first UNINITIALIZED slot in the `pending`
      // vector
      let pending_dst_ptr = self.pending.as_mut_ptr().add(pending_len);
      // By adding `free_len - 1`, we get a pointer to the LAST INITIALIZED element in the `free`
      // vector (the one we want to pop)
      let free_src_ptr = self.free.as_mut_ptr().add(free_len - 1);

      // - Perform memory moves
      // `copy_nonoverlapping` acts exactly like C's memcpy. It copied bytes from src to dst
      // without any stack intervention.
      // Move `active` -> next available slot in `pending`
      copy_nonoverlapping(active_ptr, pending_dst_ptr, 1);
      // Move `free[last]` -> `active`
      // This safely overwrites the old `active` memory (which conceptually moved to pending)
      copy_nonoverlapping(free_src_ptr, active_ptr, 1);

      // -- update lengths
      self.pending.set_len(pending_len + 1); // Pending gained a pool
      self.free.set_len(free_len - 1); // Free lost a pool
    }

    Ok(())
  }
}

#[derive(Debug)]
pub(crate) struct QueueFamilyPools {
  pub inner: DebugTrackedRwLock<alloc::boxed::Box<QueueFamilyPoolsInner>>,
}

impl QueueFamilyPools {
  /// Function which can insert a new instance of `Self` while constructing it in place without
  /// stack allocation
  pub(crate) fn create_and_insert(
    registry: &dashmap::DashMap<(ThreadId, u32), Self>,
    thread_id: ThreadId,
    queue_family_index: u32,
    id: u64,
    device: &super::LogicalDevice,
  ) -> ash::prelude::VkResult<()> {
    use dashmap::mapref::entry::Entry;
    // - Lock the DashMap shard for this specific key
    match registry.entry((thread_id, queue_family_index)) {
      Entry::Occupied(_) => {
        // The entry already exists. Another thread beat us to insertion. Do nothing
      }
      Entry::Vacant(vacant_entry) => {
        // Now we are holding the shard lock! No other thread can insert this key

        // 1. Allocate a zeroed block of memory from the heap. Done under the hood with `calloc`
        let mut boxed_inner = alloc::boxed::Box::<QueueFamilyPoolsInner>::new_zeroed();
        // SAFETY: `QueueFamilyPoolsInner` is zeroable
        unsafe {
          let ptr = boxed_inner.as_mut_ptr();
          // 2. Initialize the `active` TrackedPool directly into heap allocated object
          TrackedPool::new_at_ptr(
            core::ptr::addr_of_mut!((*ptr).active),
            device,
            queue_family_index,
            id,
          )?;

          // 3. other fields are zero initialized, which should be a valid state for `heapless` structs
          (*ptr).next_pool_id = 1;
        }
        // 4. promise to compiler that we fully initialized objects
        let initialized_inner = unsafe { boxed_inner.assume_init() };
        // 5. Construct the outer object
        let pools = Self {
          inner: DebugTrackedRwLock::new(initialized_inner),
        };
        // 6. Safely insert into the DashMap
        vacant_entry.insert(pools);
      }
    }
    Ok(())
  }
}

#[derive(Debug)]
pub(crate) struct CommandPools {
  /// composite key tracks pool by thread AND queue family index
  registry: dashmap::DashMap<(ThreadId, u32), QueueFamilyPools>,
}

unsafe impl Sync for CommandPools {}
unsafe impl Send for CommandPools {}

impl CommandPools {
  pub(crate) fn new() -> Self {
    Self {
      registry: dashmap::DashMap::with_capacity(4),
    }
  }

  /// Allocate a primary command buffer, first trying from recycled ones, then from the pool
  #[named]
  pub(crate) fn allocate_primary(
    &self,
    device: &super::LogicalDevice,
    tid: ThreadId,
    queue_family_index: u32,
    id: CommandBufferId,
  ) -> GpuResult<vk::CommandBuffer> {
    use super::utils::RwLockable;
    let key = (tid, queue_family_index);
    // - Fast path: try to pp a recycled buffer with just a read lock on the global registry
    let needs_init = !self.registry.contains_key(&key);
    if needs_init {
      QueueFamilyPools::create_and_insert(&self.registry, tid, queue_family_index, id.0, device)?;
    }

    // SAFETY: `needs_init` ensures that the entry exists
    let qfp: dashmap::mapref::one::Ref<_, _> =
      unsafe { self.registry.get(&key).unwrap_unchecked() };
    let mut inner = qfp.inner.write();

    // - Chunk Rotation: Seal and rotate the active pool if full
    if inner.active.next_free_index >= MAX_BUFFERS_PER_POOL {
      if inner.can_rotate() {
        inner
          .rotate_active_pool()
          .map_err(|s| GpuError::BackendSpecific(s.to_string()))?;
      } else {
        inner.move_active_to_pending_and_allocate_new(device, queue_family_index, id.0)?;
      }
    }
    let active = &mut inner.active;

    // Fast-path handle reuse OR slow path native allocation
    let cmd = if active.next_free_index < active.allocated_count {
      // reuse existing handle (implicitly reset cause vkResetCommandPool was called)
      active.allocated_buffers[active.next_free_index]
    } else {
      let allocate_info = vk::CommandBufferAllocateInfo::default()
        .command_buffer_count(1)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(active.pool);

      let cmd = unsafe {
        let mut cmd = vk::CommandBuffer::null();
        let vk_res = (device.fp_v1_0().allocate_command_buffers)(
          device.handle(),
          &allocate_info,
          ptr::from_mut(&mut cmd),
        );
        vk_res.result_with_success(cmd)
      }
      .with_name(
        device,
        &alloc::format!(
          "PrimaryCommandBuffer_qf{}_p{}_{}",
          queue_family_index,
          active.id,
          id.0
        ),
      )?;
      // equivalent to heapless push
      debug_assert!(active.allocated_count < MAX_BUFFERS_PER_POOL);
      active.allocated_buffers[active.allocated_count] = cmd;
      active.allocated_count += 1;
      let id = active.id;
      drop(active);
      let _ = inner.buffer_to_pool.insert(cmd, id);
      cmd
    };

    let active = &mut inner.active;
    active.next_free_index += 1;
    active.in_flight += 1;

    Ok(cmd)
  }

  #[named]
  pub(crate) fn recycle(
    &self,
    device: &super::LogicalDevice,
    tid: ThreadId,
    queue_family_index: u32,
    cmd_buf: vk::CommandBuffer,
  ) -> GpuResult<()> {
    use super::utils::RwLockable;

    let qfp = self.registry.get(&(tid, queue_family_index)).ok_or_else(|| {
      GpuError::BackendSpecific("Command pool state fot found for recycling".to_string())
    })?;
    let mut inner = qfp.inner.write();

    let pool_id = *inner.buffer_to_pool.get(&cmd_buf).ok_or_else(|| {
      GpuError::BackendSpecific("Command Buffer not found in tracking map".to_string())
    })?;

    // Check if it belongs to currently active pool
    if inner.active.id == pool_id {
      inner.active.in_flight -= 1;
      if inner.active.in_flight == 0 {
        // Zero in-flight means ALL buffers from this pool have finished executing on the GPU.
        // It is perfectly safe to reset it now!
        inner.active.reset_memory(device).map_err(GpuError::from)?;
      }
      return Ok(());
    }

    // Otherwise, decrement the matching pending pool
    let mut pending_idx = None;
    for (i, p) in inner.pending.iter_mut().enumerate() {
      if p.id == pool_id {
        p.in_flight -= 1;
        if p.in_flight == 0 {
          pending_idx = Some(i);
        }
        break;
      }
    }

    if let Some(idx) = pending_idx {
      inner.move_pending_to_free_at(
        idx,
        Some(
          |finished_pool: &mut TrackedPool| -> ash::prelude::VkResult<()> {
            finished_pool.reset_memory(device)
          },
        ),
      )?;
    }

    Ok(())
  }
}

impl DeviceResource for CommandPools {
  fn cleanup(&mut self, device: &super::LogicalDevice) {
    use super::utils::RwLockable;
    for ref_mut_multi in self.registry.iter_mut() {
      let inner = ref_mut_multi.inner.write();
      for p in &inner.pending {
        unsafe { device.destroy_command_pool(p.pool, None) };
      }
      for p in &inner.free {
        unsafe { device.destroy_command_pool(p.pool, None) };
      }
    }

    self.registry.clear();
  }
}