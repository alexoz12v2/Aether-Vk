//! tasklet module.

use alloc::{boxed::Box, sync::Arc};
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use super::{ThreadPool, Workload, WorkloadStatus};
use crate::os::NativeResult;

/// TODO: Document this item
#[derive(Debug)]
pub struct TaskletState<R> {
  result: Mutex<Option<R>>,
  done: AtomicBool,
}

/// TODO: Document this item
#[derive(Debug)]
pub struct TaskletHandle<R> {
  state: Arc<TaskletState<R>>,
}

impl<R> TaskletHandle<R> {
  /// TODO: Document this item
  pub fn wait(self) -> R {
    while !self.state.done.load(Ordering::Acquire) {
      #[cfg(any(unix, target_os = "macos"))]
      unsafe {
        libc::sched_yield()
      };
      #[cfg(not(any(unix, target_os = "macos")))]
      core::hint::spin_loop();
    }

    self
      .state
      .result
      .lock()
      .take()
      .expect("Result missing (tasklet panicked or was dropped)")
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
