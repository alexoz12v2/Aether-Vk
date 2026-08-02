//! timeline_manager module.

use crate::{
  gpu_backends::vulkan::device::{
    TASK_STATUS_FAILED, TASK_STATUS_PENDING, TASK_STATUS_SUCCESS, TaskEntry,
    TimelinePollingWorkload,
  },
  types::{GpuError, GpuResult},
};
use alloc::{collections::BTreeMap, sync::Arc};
use ash::vk;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use function_name::named;

pub struct TimelineManager {
  pub sem_device: ash::khr::timeline_semaphore::Device,
  pub semaphore: crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::Semaphore>,

  /// Polled from the GPU. Represents work officially done.
  pub cached_completed_value: Arc<AtomicU64>,

  /// Tracked on the CPU. The highest value submitted via vkQueueSubmit.
  /// MUST only be incremented inside the submission lock to guarantee monotonicity!
  next_submit_value: AtomicU64,

  /// Registry of CPU tasks waiting for a specific timeline value
  pub task_registry: Arc<
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<BTreeMap<u64, Arc<TaskEntry>>>,
  >,
  next_task_id: AtomicU64,
}

impl TimelineManager {
  /// Constructor
  #[named]
  pub fn new(instance: &ash::Instance, device: &ash::Device) -> GpuResult<Self> {
    let mut sem_type_info = vk::SemaphoreTypeCreateInfo::default()
      .initial_value(0)
      .semaphore_type(vk::SemaphoreType::TIMELINE);

    let sem_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut sem_type_info);

    let semaphore = unsafe { device.create_semaphore(&sem_create_info, None) }
      .map_err(|_| crate::gpu_err_device!())?;

    let sem_device = ash::khr::timeline_semaphore::Device::new(instance, device);

    Ok(Self {
      sem_device,
      semaphore: unsafe {
        crate::gpu_backends::vulkan::utils::NonZeroHandle::new_unchecked(semaphore)
      },
      cached_completed_value: Arc::new(AtomicU64::new(0)),
      next_submit_value: AtomicU64::new(1),
      task_registry: Arc::new(
        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(BTreeMap::new()),
      ),
      next_task_id: AtomicU64::new(1),
    })
  }

  /// Retrieves the explicit timeline value the GPU will reach upon completing the Graphics queue
  /// submission associated to this task.
  /// This will safely spin-loop if the task has been created but not yet submitted.
  #[named]
  pub fn get_task_target_value(&self, task_id: u64) -> GpuResult<u64> {
    use super::utils::RwLockable;
    use aethervk_oshal_rlib::os::native::this_thread;
    use core::sync::atomic::Ordering;
    // 1. Lock briefly just to clone the Arc<TaskEntry> if it exists
    let entry_opt = {
      let registry = self.task_registry.read();
      registry.get(&task_id).cloned() // clone the Arc
    };

    if let Some(entry) = entry_opt {
      // 2. We've dropped the registry lock, therefore we can spin loop
      // Note: `create_task` assigns `u64::MAX` to `target_value`
      let mut val = entry.target_value.load(Ordering::Acquire);

      while val == u64::MAX {
        // Abort the spin loop if the task failed before reaching submission
        if entry.status.load(Ordering::Acquire) == TASK_STATUS_FAILED {
          let err = entry.error.read().clone().unwrap_or(crate::gpu_err_device!());
          return Err(err);
        }

        core::hint::spin_loop();
        this_thread::yield_now();
        val = entry.target_value.load(Ordering::Acquire);
      }

      Ok(val)
    } else if task_id > 0 && task_id < self.next_task_id.load(Ordering::SeqCst) {
      // 3. The task is already finished and the worker thread purged it.
      // Returning the currently completed cached value is safe cause we are sure that we already
      // reached it with queue submission execution
      Ok(self.get_cached_value())
    } else {
      Err(crate::gpu_invalid_arg!("invalid task id"))
    }
  }

  pub fn cleanup(&mut self, device: &ash::Device) {
    unsafe { device.destroy_semaphore(self.semaphore.get(), None) };
  }

  /// Fetches what the GPU has completed (cheap cache read)
  pub fn get_cached_value(&self) -> u64 {
    self.cached_completed_value.load(Ordering::Relaxed)
  }

  /// Polls the GPU and updates the cache safely
  #[named]
  pub fn refresh_cached_value(&self) -> GpuResult<u64> {
    let gpu_value = unsafe { self.sem_device.get_semaphore_counter_value(self.semaphore.get()) }
      .map_err(|_| crate::gpu_err_device!())?;

    self.cached_completed_value.fetch_max(gpu_value, Ordering::Relaxed);
    Ok(gpu_value)
  }

  pub fn get_next_submit_value(&self) -> u64 {
    self.next_submit_value.load(Ordering::SeqCst)
  }

  /// Gets a unique, strictly increasing sequence number for a new submission.
  /// MUST be called inside the vkQueueSubmit lock.
  pub fn allocate_submit_value(&self) -> u64 {
    self.next_submit_value.fetch_add(1, Ordering::SeqCst)
  }

  // --- Task Management API ---
  /// TODO: Document this item
  pub fn create_task(&self) -> u64 {
    let id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
    let entry = Arc::new(TaskEntry {
      target_value: AtomicU64::new(u64::MAX),
      status: AtomicU32::new(TASK_STATUS_PENDING),
      error: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::new(None),
    });
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&self.task_registry)
      .insert(id, entry);
    id
  }

  /// TODO: Document this item
  pub fn assign_task_target(&self, task_id: u64, target_timeline: u64) {
    if let Some(entry) =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.task_registry)
        .get(&task_id)
    {
      entry.target_value.store(target_timeline, Ordering::Release);
    }
  }

  /// TODO: Document this item
  pub fn fail_task(&self, task_id: u64, error: GpuError) {
    if let Some(entry) =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.task_registry)
        .get(&task_id)
    {
      *crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&entry.error) =
        Some(error);
      entry.status.store(TASK_STATUS_FAILED, Ordering::Release);
    }
  }

  /// TODO: Document this item
  pub fn success_task(&self, task_id: u64) {
    if let Some(entry) =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.task_registry)
        .get(&task_id)
    {
      entry.status.store(TASK_STATUS_SUCCESS, Ordering::Release);
    }
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(&self.task_registry)
      .remove(&task_id);
  }

  /// TODO: Document this item
  #[named]
  pub fn is_task_completed(&self, task_id: u64) -> GpuResult<bool> {
    let registry =
      crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&self.task_registry);
    if let Some(entry) = registry.get(&task_id) {
      let status = entry.status.load(Ordering::Acquire);
      if status == TASK_STATUS_SUCCESS {
        Ok(true)
      } else if status == TASK_STATUS_FAILED {
        let err =
          crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&entry.error)
            .clone()
            .unwrap_or(crate::gpu_err_device!());
        Err(err)
      } else {
        Ok(false)
      }
    } else {
      if task_id > 0 && task_id < self.next_task_id.load(Ordering::SeqCst) {
        Ok(true)
      } else {
        Err(crate::gpu_invalid_arg!("invalid argument"))
      }
    }
  }

  /// TODO: Document this item
  pub fn create_polling_workload(
    &self,
    stop_signal: Arc<core::sync::atomic::AtomicBool>,
  ) -> TimelinePollingWorkload {
    TimelinePollingWorkload {
      timeline_sem_device: self.sem_device.clone(),
      timeline_semaphore: self.semaphore.get(),
      timeline_semaphore_cached_value: Arc::clone(&self.cached_completed_value),
      task_registry: Arc::clone(&self.task_registry),
      stop_signal,
    }
  }
}
