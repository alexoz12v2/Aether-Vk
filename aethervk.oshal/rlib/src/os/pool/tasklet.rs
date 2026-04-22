use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, Ordering};
use spin::Mutex;

use crate::os::NativeResult;
use crate::os::pool::{ThreadPool, Workload};

/// Shared state between the executed workload and the waitable handle.
struct TaskletState<R> {
  result: Mutex<Option<R>>,
  done: AtomicBool,
}

/// A waitable handle for a single tasklet scheduled on the pool.
pub struct TaskletHandle<R> {
  state: Arc<TaskletState<R>>,
}

impl<R> TaskletHandle<R> {
  /// Blocks the current thread until the background workload completes.
  /// Because this consumes `self` and returns the owned value, the result
  /// acts like standard stack data and is natively writable (`mut`).
  pub fn wait(self) -> R {
    // Spin-wait until the background pool thread sets the done flag
    while !self.state.done.load(Ordering::Acquire) {
      #[cfg(any(unix, target_os = "macos"))]
      unsafe {
        libc::sched_yield()
      };

      #[cfg(not(any(unix, target_os = "macos")))]
      core::hint::spin_loop(); // Halts CPU cycles briefly in bare-metal/Windows
    }

    // Cleanly extract the result
    self
      .state
      .result
      .lock()
      .take()
      .expect("Result missing (tasklet panicked or was dropped)")
  }
}

/// Wrapper to translate an owned `FnOnce` into your `&self`-based `Workload` trait.
struct ClosureWorkload<F, R> {
  func: Mutex<Option<F>>,
  state: Arc<TaskletState<R>>,
}

impl<F, R> Workload for ClosureWorkload<F, R>
where
  F: FnOnce() -> R + Send + 'static,
  R: Send + 'static,
{
  fn execute(&self) {
    // Take the closure out of the Option, releasing the lock immediately
    if let Some(f) = self.func.lock().take() {
      let res = f(); // Execute with no locks held
      *self.state.result.lock() = Some(res);
      self.state.done.store(true, Ordering::Release);
    }
  }
}

impl<F, R> Drop for ClosureWorkload<F, R> {
  fn drop(&mut self) {
    // Safety mechanic: Prevent deadlocks if the pool drops the workload unexecuted
    // (e.g. pool is dropped early). Wait() will wake up and panic properly.
    self.state.done.store(true, Ordering::Release);
  }
}

/// Ergonomic extension trait for your thread pool.
pub trait ThreadPoolExt {
  fn spawn_tasklet<F, R>(&self, f: F) -> NativeResult<TaskletHandle<R>>
  where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static;
}

impl ThreadPoolExt for ThreadPool {
  fn spawn_tasklet<F, R>(&self, f: F) -> NativeResult<TaskletHandle<R>>
  where
    F: FnOnce() -> R + Send + 'static,
    R: Send + 'static,
  {
    let state = Arc::new(TaskletState {
      result: Mutex::new(None),
      done: AtomicBool::new(false),
    });

    let workload: Box<dyn Workload> = Box::new(ClosureWorkload {
      func: Mutex::new(Some(f)),
      state: Arc::clone(&state),
    });

    // Scatter a single tasklet
    let mut workloads: Vec<Box<dyn Workload>> = Vec::with_capacity(1);
    workloads.push(workload);
    self.scatter(workloads)?;

    Ok(TaskletHandle { state })
  }
}
