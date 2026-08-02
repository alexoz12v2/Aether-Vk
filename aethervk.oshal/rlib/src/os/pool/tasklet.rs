//! tasklet module.

use alloc::{boxed::Box, sync::Arc};
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use super::{ThreadPool, Workload, WorkloadStatus};
use crate::os::NativeResult;

#[derive(Debug)]
pub struct TaskletState<R> {
  result: Mutex<Option<R>>,
  done: AtomicBool,
}

#[derive(Debug)]
pub struct TaskletHandle<R> {
  state: Arc<TaskletState<R>>,
}

unsafe impl<R> Sync for TaskletHandle<R> {}
unsafe impl<R> Send for TaskletHandle<R> {}

impl<R> TaskletHandle<R> {
  /// Spin-wait until the tasklet completes and return its result.
  pub fn wait(self) -> R {
    while !self.state.done.load(Ordering::Acquire) {
      #[cfg(any(unix, target_os = "macos"))]
      unsafe {
        libc::sched_yield()
      };
      #[cfg(not(any(unix, target_os = "macos")))]
      core::hint::spin_loop();
      // TODO windows add YieldProcessor
    }

    self
      .state
      .result
      .lock()
      .take()
      .expect("Result missing (tasklet panicked or was dropped)")
  }

  /// Returns `true` if the tasklet has finished executing (non-blocking).
  #[inline]
  pub fn is_done(&self) -> bool {
    self.state.done.load(Ordering::Acquire)
  }

  /// Spin-wait up to `deadline_us` microseconds for the tasklet to finish.
  ///
  /// Returns `Some(result)` if the tasklet completed within the deadline, or
  /// `None` if it was still running when the deadline expired.
  /// Consuming `self` on `Some` ensures the result is not double-taken.
  pub fn try_wait(self, deadline_us: u64) -> Result<R, Self> {
    let start = super::super::time::get_monotonic_time() as u64;
    loop {
      if self.state.done.load(Ordering::Acquire) {
        let result = self
          .state
          .result
          .lock()
          .take()
          .expect("Result missing (tasklet panicked or was dropped)");
        return Ok(result);
      }
      let elapsed = (super::super::time::get_monotonic_time() as u64).saturating_sub(start);
      if elapsed >= deadline_us {
        return Err(self); // deadline expired — caller retains ownership
      }
      #[cfg(any(unix, target_os = "macos"))]
      unsafe {
        libc::sched_yield()
      };
      #[cfg(not(any(unix, target_os = "macos")))]
      core::hint::spin_loop();
    }
  }
}

struct ClosureWorkload<F, R> {
  func: Option<F>,
  state: Arc<TaskletState<R>>,
  tasklet_id: Option<usize>,
}

/// closure has to be potentially `static cause threads might live forever. Any attempt to shorten
/// lifetime of this closure should be done through type erasure
impl<F, R> Workload for ClosureWorkload<F, R>
where
  F: FnOnce() -> R + Send + 'static,
  R: Send + 'static,
{
  fn execute(&mut self) -> WorkloadStatus {
    // With exclusive &mut self, we effortlessly take the closure out of the Option without locks!
    if let Some(f) = self.func.take() {
      let res = f();
      *self.state.result.lock() = Some(res);
      self.state.done.store(true, Ordering::Release);
    }
    WorkloadStatus::Complete
  }

  fn tasklet_id(&self) -> Option<usize> {
    self.tasklet_id
  }
}

impl<F, R> Drop for ClosureWorkload<F, R> {
  fn drop(&mut self) {
    self.state.done.store(true, Ordering::Release);
  }
}

/// TODO: Document this item
pub trait ThreadPoolExt {
  fn spawn_tasklet<F, R>(&self, tasklet_id: Option<usize>, f: F) -> NativeResult<TaskletHandle<R>>
  where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static;
}

impl ThreadPoolExt for ThreadPool {
  fn spawn_tasklet<F, R>(&self, tasklet_id: Option<usize>, f: F) -> NativeResult<TaskletHandle<R>>
  where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
  {
    let state = Arc::new(TaskletState {
      result: Mutex::new(None),
      done: AtomicBool::new(false),
    });

    let workload: Box<dyn Workload> = Box::new(ClosureWorkload {
      func: Some(f),
      state: Arc::clone(&state),
      tasklet_id,
    });

    self.scatter(alloc::vec![workload])?;

    Ok(TaskletHandle { state })
  }
}
