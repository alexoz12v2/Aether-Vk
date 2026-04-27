use alloc::sync::Arc;
use core::sync::atomic::{AtomicU64, Ordering, AtomicU32};
use alloc::collections::BTreeMap;
use ash::vk;
use crate::gpu_backends::vulkan::device::{
  TaskEntry, TimelinePollingWorkload, TASK_STATUS_FAILED, TASK_STATUS_PENDING, TASK_STATUS_SUCCESS,
};
use crate::types::{GpuError, GpuResult};

pub(super) struct TimelineManager {
  pub sem_device: ash::khr::timeline_semaphore::Device,
  pub semaphore: crate::gpu_backends::vulkan::utils::NonZeroHandle<vk::Semaphore>,

  /// Polled from the GPU. Represents work officially done.
  pub cached_completed_value: Arc<AtomicU64>,

  /// Tracked on the CPU. The highest value submitted via vkQueueSubmit.
  /// MUST only be incremented inside the submission lock to guarantee monotonicity!
  next_submit_value: AtomicU64,

  /// Registry of CPU tasks waiting for a specific timeline value
  pub task_registry: Arc<spin::RwLock<BTreeMap<u64, Arc<TaskEntry>>>>,
  next_task_id: AtomicU64,
}

impl TimelineManager {
  pub fn new(instance: &ash::Instance, device: &ash::Device) -> GpuResult<Self> {
    let mut sem_type_info = vk::SemaphoreTypeCreateInfo::default()
      .initial_value(0)
      .semaphore_type(vk::SemaphoreType::TIMELINE);

    let sem_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut sem_type_info);

    let semaphore = unsafe { device.create_semaphore(&sem_create_info, None) }
      .map_err(|_| GpuError::InvalidState("Failed to create timeline semaphore"))?;

    let sem_device = ash::khr::timeline_semaphore::Device::new(instance, device);

    Ok(Self {
      sem_device,
      semaphore: unsafe {
        crate::gpu_backends::vulkan::utils::NonZeroHandle::new_unchecked(semaphore)
      },
      cached_completed_value: Arc::new(AtomicU64::new(0)),
      next_submit_value: AtomicU64::new(1),
      task_registry: Arc::new(spin::RwLock::new(BTreeMap::new())),
      next_task_id: AtomicU64::new(1),
    })
  }

  pub fn cleanup(&mut self, device: &ash::Device) {
    unsafe { device.destroy_semaphore(self.semaphore.get(), None) };
  }

  /// Fetches what the GPU has completed (cheap cache read)
  pub fn get_cached_value(&self) -> u64 {
    self.cached_completed_value.load(Ordering::Relaxed)
  }

  /// Polls the GPU and updates the cache safely
  pub fn refresh_cached_value(&self) -> GpuResult<u64> {
    let gpu_value = unsafe {
      self
        .sem_device
        .get_semaphore_counter_value(self.semaphore.get())
    }
    .map_err(|_| GpuError::InvalidState("Failed to get semaphore counter value"))?;

    self
      .cached_completed_value
      .fetch_max(gpu_value, Ordering::Relaxed);
    Ok(gpu_value)
  }

  /// Gets a unique, strictly increasing sequence number for a new submission.
  /// MUST be called inside the vkQueueSubmit lock.
  pub fn allocate_submit_value(&self) -> u64 {
    self.next_submit_value.fetch_add(1, Ordering::SeqCst)
  }

  // --- Task Management API ---
  pub fn create_task(&self) -> u64 {
    let id = self.next_task_id.fetch_add(1, Ordering::SeqCst);
    let entry = Arc::new(TaskEntry {
      target_value: AtomicU64::new(u64::MAX),
      status: AtomicU32::new(TASK_STATUS_PENDING),
      error: spin::RwLock::new(None),
    });
    self.task_registry.write().insert(id, entry);
    id
  }

  pub fn assign_task_target(&self, task_id: u64, target_timeline: u64) {
    if let Some(entry) = self.task_registry.read().get(&task_id) {
      entry.target_value.store(target_timeline, Ordering::Release);
    }
  }

  pub fn fail_task(&self, task_id: u64, error: GpuError) {
    if let Some(entry) = self.task_registry.read().get(&task_id) {
      *entry.error.write() = Some(error);
      entry.status.store(TASK_STATUS_FAILED, Ordering::Release);
    }
  }

  pub fn success_task(&self, task_id: u64) {
    if let Some(entry) = self.task_registry.read().get(&task_id) {
      entry.status.store(TASK_STATUS_SUCCESS, Ordering::Release);
    }
  }

  pub fn is_task_completed(&self, task_id: u64) -> GpuResult<bool> {
    let registry = self.task_registry.read();
    if let Some(entry) = registry.get(&task_id) {
      let status = entry.status.load(Ordering::Acquire);
      if status == TASK_STATUS_SUCCESS {
        Ok(true)
      } else if status == TASK_STATUS_FAILED {
        let err = entry
          .error
          .read()
          .clone()
          .unwrap_or(GpuError::InvalidState("Task Failed"));
        Err(err)
      } else {
        Ok(false)
      }
    } else {
      Err(GpuError::InvalidArgument("Invalid task id"))
    }
  }

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
