//! A thread pool for scattering and gathering computation workloads.

use core::marker::Send;

use crate::os::NativeResult;

pub mod chunked;
pub mod persistent;
pub mod tasklet;

/// Status returned by a workload to guide the thread scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadStatus {
  /// The workload has finished execution and its memory will be dropped.
  Complete,
  /// The workload wishes to pause execution and requests re-insertion into the queue.
  Yield,
}

/// A unit of work that can be executed by the thread pool.
pub trait Workload: Send {
  /// Executes the workload mutably, returning its scheduling directive.
  fn execute(&mut self) -> WorkloadStatus;

  /// Returns an optional tasklet ID. Workloads with the same tasklet ID
  /// will always be executed sequentially by the same underlying thread in the pool.
  fn tasklet_id(&self) -> Option<usize> {
    None
  }
}

#[cfg(target_os = "windows")]
mod windows_pool {
  use crate::os::NativeError;
  use alloc::{boxed::Box, collections::VecDeque, sync::Arc, vec::Vec};
  use core::{
    ffi::c_void,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
  };
  use spin::Mutex;

  use super::*;
  use windows::Win32::{
    Foundation::{CloseHandle, HANDLE},
    System::Threading::{
      CreateThread, INFINITE, SwitchToThread, THREAD_CREATION_FLAGS, WaitForSingleObject,
    },
  };

  struct ThreadPoolState {
    shared_queue: Mutex<VecDeque<Box<dyn Workload>>>,
    local_queues: Vec<Mutex<VecDeque<Box<dyn Workload>>>>,
    shutdown: AtomicBool,
    pending_tasks: AtomicUsize,
  }

  struct ThreadArg {
    state: Arc<ThreadPoolState>,
    id: usize,
  }

  /// A thread pool for executing workloads.
  pub struct ThreadPool {
    threads: Vec<HANDLE>,
    state: Arc<ThreadPoolState>,
  }

  unsafe impl Send for ThreadPool {}
  unsafe impl Sync for ThreadPool {}

  impl ThreadPool {
    /// TODO: Document this item
    pub fn new(num_threads: usize) -> NativeResult<Self> {
      let mut local_queues = Vec::with_capacity(num_threads);
      for _ in 0..num_threads {
        local_queues.push(Mutex::new(VecDeque::new()));
      }

      let state = Arc::new(ThreadPoolState {
        shared_queue: Mutex::new(VecDeque::new()),
        local_queues,
        shutdown: AtomicBool::new(false),
        pending_tasks: AtomicUsize::new(0),
      });

      let mut threads = Vec::with_capacity(num_threads);

      for i in 0..num_threads {
        let arg = Box::new(ThreadArg {
          state: Arc::clone(&state),
          id: i,
        });

        let raw_arg = Box::into_raw(arg);
        let handle_res = unsafe {
          CreateThread(
            None,
            0,
            Some(thread_func),
            Some(raw_arg as *const c_void),
            THREAD_CREATION_FLAGS(0),
            None,
          )
        };

        match handle_res {
          Ok(handle) => threads.push(handle),
          Err(_) => {
            let _ = unsafe { Box::from_raw(raw_arg) }; // Drop cleanly on failure to avoid leak
            state.shutdown.store(true, Ordering::SeqCst);
            for &thread in &threads {
              unsafe {
                let _ = WaitForSingleObject(thread, INFINITE);
                let _ = CloseHandle(thread);
              }
            }
            return Err(NativeError::OsThreadingError(
              crate::os::ThreadingError::Unknown,
            ));
          }
        }
      }

      Ok(Self { threads, state })
    }

    /// TODO: Document this item
    pub fn scatter(&self, workloads: Vec<Box<dyn Workload>>) -> NativeResult<()> {
      let num_threads = self.state.local_queues.len();
      self.state.pending_tasks.fetch_add(workloads.len(), Ordering::SeqCst);

      for workload in workloads {
        if let Some(id) = workload.tasklet_id() {
          if num_threads > 0 {
            let target = id % num_threads;
            self.state.local_queues[target].lock().push_back(workload);
            continue;
          }
        }
        self.state.shared_queue.lock().push_back(workload);
      }

      Ok(())
    }

    /// TODO: Document this item
    pub fn gather(&self) {
      while self.state.pending_tasks.load(Ordering::Acquire) > 0 {
        unsafe {
          let _ = unsafe { SwitchToThread() };
        }
      }
    }
  }

  impl Drop for ThreadPool {
    fn drop(&mut self) {
      self.state.shutdown.store(true, Ordering::SeqCst);
      for &thread in &self.threads {
        unsafe {
          let _ = WaitForSingleObject(thread, INFINITE);
          let _ = CloseHandle(thread);
        }
      }
    }
  }

  unsafe extern "system" fn thread_func(arg: *mut c_void) -> u32 {
    let thread_arg = Box::from_raw(arg as *mut ThreadArg);
    let state = thread_arg.state.clone();
    let id = thread_arg.id;
    drop(thread_arg);

    let mut tick: u8 = 0;

    while !state.shutdown.load(Ordering::Acquire) {
      tick = tick.wrapping_add(1);

      // Every 4th cycle prioritize popping from the shared queue to prevent
      // queue starvation triggered by heavily persistent local tasklets loop
      let try_shared_first = tick % 4 == 0;

      let workload = if try_shared_first {
        let mut shared = state.shared_queue.lock();
        if let Some(w) = shared.pop_front() {
          Some(w)
        } else {
          drop(shared);
          state.local_queues[id].lock().pop_front()
        }
      } else {
        let mut local = state.local_queues[id].lock();
        if let Some(w) = local.pop_front() {
          Some(w)
        } else {
          drop(local);
          state.shared_queue.lock().pop_front()
        }
      };

      if let Some(mut workload) = workload {
        match workload.execute() {
          WorkloadStatus::Complete => {
            state.pending_tasks.fetch_sub(1, Ordering::Release);
          }
          WorkloadStatus::Yield => {
            let mut target_q = None;
            if let Some(tid) = workload.tasklet_id() {
              let num_threads = state.local_queues.len();
              if num_threads > 0 {
                target_q = Some(tid % num_threads);
              }
            }

            if let Some(target) = target_q {
              state.local_queues[target].lock().push_back(workload);
            } else {
              state.shared_queue.lock().push_back(workload);
            }

            // Yield heuristic to prevent a 100% CPU lock in a persistent while-loop:
            // If the combined size of the queues this thread reads from is <= 1, it means
            // the task we *just* yielded back is the exact same one it'll pull next iteration!
            let local_len = state.local_queues[id].lock().len();
            let shared_len = state.shared_queue.lock().len();
            if local_len + shared_len <= 1 {
              crate::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
            }
          }
        }
      } else {
        crate::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
      }
    }

    0
  }
}

#[cfg(unix)]
mod pthread_pool {
  use super::*;
  use alloc::{boxed::Box, collections::VecDeque, sync::Arc, vec::Vec};
  use core::{
    ffi::c_void,
    ptr,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
  };
  use libc::{pthread_create, pthread_join, pthread_t};
  use spin::Mutex;

  struct ThreadPoolState {
    shared_queue: Mutex<VecDeque<Box<dyn Workload>>>,
    local_queues: Vec<Mutex<VecDeque<Box<dyn Workload>>>>,
    shutdown: AtomicBool,
    pending_tasks: AtomicUsize,
  }

  struct ThreadArg {
    state: Arc<ThreadPoolState>,
    id: usize,
  }

  /// A thread pool for executing workloads.
  pub struct ThreadPool {
    threads: Vec<pthread_t>,
    state: Arc<ThreadPoolState>,
  }

  unsafe impl Send for ThreadPool {}
  unsafe impl Sync for ThreadPool {}

  impl ThreadPool {
    /// TODO: Document this item
    pub fn new(num_threads: usize) -> NativeResult<Self> {
      let mut local_queues = Vec::with_capacity(num_threads);
      for _ in 0..num_threads {
        local_queues.push(Mutex::new(VecDeque::new()));
      }

      let state = Arc::new(ThreadPoolState {
        shared_queue: Mutex::new(VecDeque::new()),
        local_queues,
        shutdown: AtomicBool::new(false),
        pending_tasks: AtomicUsize::new(0),
      });

      let mut threads = Vec::with_capacity(num_threads);

      for i in 0..num_threads {
        let arg = Box::new(ThreadArg {
          state: Arc::clone(&state),
          id: i,
        });

        let raw_arg = Box::into_raw(arg);
        let mut thread: pthread_t = unsafe { core::mem::zeroed() };
        let ret =
          unsafe { pthread_create(&mut thread, ptr::null(), thread_func, raw_arg as *mut _) };

        if ret == 0 {
          threads.push(thread);
        } else {
          let _ = unsafe { Box::from_raw(raw_arg) };
          state.shutdown.store(true, Ordering::SeqCst);
          for &t in &threads {
            unsafe {
              pthread_join(t, ptr::null_mut());
            }
          }
          return Err(crate::os::NativeError::OsThreadingError(
            crate::os::ThreadingError::Unknown,
          ));
        }
      }

      Ok(Self { threads, state })
    }

    /// TODO: Document this item
    pub fn scatter(&self, workloads: Vec<Box<dyn Workload>>) -> NativeResult<()> {
      let num_threads = self.state.local_queues.len();
      self.state.pending_tasks.fetch_add(workloads.len(), Ordering::SeqCst);

      for workload in workloads {
        if let Some(id) = workload.tasklet_id() {
          if num_threads > 0 {
            let target = id % num_threads;
            self.state.local_queues[target].lock().push_back(workload);
            continue;
          }
        }
        self.state.shared_queue.lock().push_back(workload);
      }

      Ok(())
    }

    /// TODO: Document this item
    pub fn gather(&self) {
      while self.state.pending_tasks.load(Ordering::Acquire) > 0 {
        unsafe { libc::sched_yield() };
      }
    }
  }

  impl Drop for ThreadPool {
    fn drop(&mut self) {
      self.state.shutdown.store(true, Ordering::SeqCst);

      // Clear the queues so that any dropped workloads that hold references
      // preventing threads from exiting are released!
      self.state.shared_queue.lock().clear();
      for q in &self.state.local_queues {
        q.lock().clear();
      }

      for &thread in &self.threads {
        unsafe {
          pthread_join(thread, ptr::null_mut());
        }
      }
    }
  }

  extern "C" fn thread_func(arg: *mut c_void) -> *mut c_void {
    let thread_arg = unsafe { Box::from_raw(arg as *mut ThreadArg) };
    let state = thread_arg.state.clone();
    let id = thread_arg.id;
    drop(thread_arg);

    let mut tick: u8 = 0;

    while !state.shutdown.load(Ordering::Acquire) {
      tick = tick.wrapping_add(1);
      let try_shared_first = tick % 4 == 0;

      let workload = if try_shared_first {
        let mut shared = state.shared_queue.lock();
        if let Some(w) = shared.pop_front() {
          Some(w)
        } else {
          drop(shared);
          state.local_queues[id].lock().pop_front()
        }
      } else {
        let mut local = state.local_queues[id].lock();
        if let Some(w) = local.pop_front() {
          Some(w)
        } else {
          drop(local);
          state.shared_queue.lock().pop_front()
        }
      };

      if let Some(mut workload) = workload {
        match workload.execute() {
          WorkloadStatus::Complete => {
            state.pending_tasks.fetch_sub(1, Ordering::Release);
          }
          WorkloadStatus::Yield => {
            if state.shutdown.load(Ordering::Acquire) {
              continue;
            }

            let mut target_q = None;
            if let Some(tid) = workload.tasklet_id() {
              let num_threads = state.local_queues.len();
              if num_threads > 0 {
                target_q = Some(tid % num_threads);
              }
            }

            if let Some(target) = target_q {
              state.local_queues[target].lock().push_back(workload);
            } else {
              state.shared_queue.lock().push_back(workload);
            }

            let local_len = state.local_queues[id].lock().len();
            let shared_len = state.shared_queue.lock().len();
            if local_len + shared_len <= 1 {
              crate::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
            }
          }
        }
      } else {
        crate::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
      }
    }

    ptr::null_mut()
  }
}

#[cfg(unix)]
pub use pthread_pool::ThreadPool;
#[cfg(target_os = "windows")]
pub use windows_pool::ThreadPool;

#[cfg(test)]
mod tests {
  use super::*;
  use crate::os::pool::{
    chunked::ThreadPoolChunkedExt,
    persistent::{PersistentStatus, ThreadPoolPersistentExt},
    tasklet::ThreadPoolExt,
  };
  use alloc::{boxed::Box, sync::Arc, vec::Vec};
  use core::sync::atomic::{AtomicUsize, Ordering};

  struct TestWorkload {
    counter: Arc<AtomicUsize>,
  }

  impl Workload for TestWorkload {
    fn execute(&mut self) -> WorkloadStatus {
      self.counter.fetch_add(1, Ordering::SeqCst);
      WorkloadStatus::Complete
    }
  }

  #[test]
  fn test_thread_pool_scatter_gather() {
    let counter = Arc::new(AtomicUsize::new(0));
    let pool = ThreadPool::new(4).expect("Failed to create thread pool");

    let mut workloads: Vec<Box<dyn Workload>> = Vec::new();
    for _ in 0..100 {
      workloads.push(Box::new(TestWorkload {
        counter: Arc::clone(&counter),
      }));
    }

    pool.scatter(workloads).expect("Failed to scatter workloads");
    pool.gather();

    assert_eq!(counter.load(Ordering::SeqCst), 100);
  }

  #[test]
  fn test_thread_pool_chunked() {
    let pool = ThreadPool::new(4).expect("Failed to create thread pool");
    let counter = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&counter);
    let handle = pool
      .spawn_chunked(10, move |chunk_id| {
        c.fetch_add(chunk_id, Ordering::SeqCst);
      })
      .expect("Failed to spawn chunked");
    handle.wait();
    assert_eq!(counter.load(Ordering::SeqCst), 45); // Sum of 0..9
  }

  #[test]
  fn test_thread_pool_persistent() {
    let pool = ThreadPool::new(4).expect("Failed to create thread pool");
    let mut i = 0;
    let handle = pool
      .spawn_persistent(None, move || {
        if i < 10 {
          i += 1;
          PersistentStatus::Yield
        } else {
          PersistentStatus::Complete(i)
        }
      })
      .expect("Failed to spawn persistent");
    let res = handle.wait();
    assert_eq!(res, 10);
  }

  #[test]
  fn test_thread_pool_tasklet() {
    let pool = ThreadPool::new(4).expect("Failed to create thread pool");
    let handle = pool.spawn_tasklet(None, || 42).expect("Failed to spawn tasklet");
    let res = handle.wait();
    assert_eq!(res, 42);
  }
}
