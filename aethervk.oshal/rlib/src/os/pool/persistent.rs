//! persistent module.

use alloc::{boxed::Box, sync::Arc};
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use super::{ThreadPool, Workload, WorkloadStatus};
use crate::os::NativeResult;

/// Controls the lifecycle sequence during persistent tasklet executions.
pub enum PersistentStatus<R> {
  /// Operation is executing across multiple iterations. Relinquishes CPU
  /// resources back to the queue scheduler before resuming.
  Yield,
  /// The iterative loops have successfully finished.
  Complete(R),
}

/// Shared state between the executed persistent job and the handle.
pub struct PersistentState<R> {
  result: Mutex<Option<R>>,
  done: AtomicBool,
}

/// A waitable handle for persistent workloads scattered out on a thread.
pub struct PersistentHandle<R> {
  state: Arc<PersistentState<R>>,
}

impl<R> PersistentHandle<R> {
  /// Blocks the current thread until the persistent machine maps to `PersistentStatus::Complete`.
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
      .expect("Result missing (persistent task panicked or was dropped)")
  }
}

/// An interior FnMut closure generator mapping logically to scheduler loop injections.
struct PersistentWorkload<F, R> {
  func: Option<F>,
  state: Arc<PersistentState<R>>,
  tasklet_id: Option<usize>,
}

/// closure has to be potentially `static cause threads might live forever. Any attempt to shorten
/// lifetime of this closure should be done through type erasure
impl<F, R> Workload for PersistentWorkload<F, R>
where
  F: FnMut() -> PersistentStatus<R> + Send + 'static, // Note: Sync dropped entirely for mutability
  R: Send + 'static,
{
  fn execute(&mut self) -> WorkloadStatus {
    let mut completed = false;

    if let Some(f) = self.func.as_mut() {
      match f() {
        PersistentStatus::Yield => return WorkloadStatus::Yield,
        PersistentStatus::Complete(res) => {
          *self.state.result.lock() = Some(res);
          self.state.done.store(true, Ordering::Release);
          completed = true;
        }
      }
    }

    if completed {
      // Safely eject the closure memory early now that execution is concluded
      self.func = None;
    }

    WorkloadStatus::Complete
  }

  fn tasklet_id(&self) -> Option<usize> {
    self.tasklet_id
  }
}

impl<F, R> Drop for PersistentWorkload<F, R> {
  fn drop(&mut self) {
    self.state.done.store(true, Ordering::Release);
  }
}

/// TODO: Document this item
pub trait ThreadPoolPersistentExt {
  /// Spawns a cooperative, long-lasting mutating closure object (`FnMut`) to cycle on the
  /// thread pool. Returning `PersistentStatus::Yield` will gracefully queue it onto the tail of the scheduling
  /// algorithm without deadlocking the core.
  fn spawn_persistent<F, R>(
    &self,
    tasklet_id: Option<usize>,
    f: F,
  ) -> NativeResult<PersistentHandle<R>>
  where
    F: FnMut() -> PersistentStatus<R> + Send + 'static,
    R: Send + 'static;
}

impl ThreadPoolPersistentExt for ThreadPool {
  fn spawn_persistent<F, R>(
    &self,
    tasklet_id: Option<usize>,
    f: F,
  ) -> NativeResult<PersistentHandle<R>>
  where
    F: FnMut() -> PersistentStatus<R> + Send + 'static,
    R: Send + 'static,
  {
    let state = Arc::new(PersistentState {
      result: Mutex::new(None),
      done: AtomicBool::new(false),
    });

    let workload: Box<dyn Workload> = Box::new(PersistentWorkload {
      func: Some(f),
      state: Arc::clone(&state),
      tasklet_id,
    });

    self.scatter(alloc::vec![workload])?;

    Ok(PersistentHandle { state })
  }
}
