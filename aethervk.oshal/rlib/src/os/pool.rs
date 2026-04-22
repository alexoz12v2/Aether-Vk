//! A thread pool for scattering and gathering computation workloads.

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::marker::{Send, Sync};

use crate::os::NativeResult;

pub mod tasklet;

/// A unit of work that can be executed by the thread pool.
pub trait Workload: Send + Sync {
  /// Executes the workload.
  fn execute(&self);
}

#[cfg(target_os = "windows")]
mod windows_pool {
  use crate::os::NativeError;
  use spin::Mutex; // Imported spin::Mutex for no_std interior mutability
  use alloc::boxed::Box;
  use alloc::vec::Vec;

  use super::*;
  use windows::Win32::System::Threading::{
    CloseThreadpool, CloseThreadpoolWork, CreateThreadpool, CreateThreadpoolWork,
    PTP_CALLBACK_INSTANCE, PTP_POOL, PTP_WORK, SetThreadpoolThreadMaximum,
    SetThreadpoolThreadMinimum, SubmitThreadpoolWork, TP_CALLBACK_ENVIRON_V3,
    TP_CALLBACK_PRIORITY_NORMAL, WaitForThreadpoolWorkCallbacks,
  };

  /// A thread pool for executing workloads.
  pub struct ThreadPool {
    pool: PTP_POOL,
    callback_environ: TP_CALLBACK_ENVIRON_V3,
    work_items: Mutex<Vec<PTP_WORK>>, // Wrapped in Mutex
  }

  // Explicitly mark ThreadPool as Send and Sync. Win32 Threadpool APIs
  // are designed to be entirely thread-safe for concurrent submissions.
  unsafe impl Send for ThreadPool {}
  unsafe impl Sync for ThreadPool {}

  impl ThreadPool {
    pub fn new(num_threads: usize) -> NativeResult<Self> {
      let pool = unsafe { CreateThreadpool(None) }
        .map_err(|_| NativeError::OsThreadingError(crate::os::ThreadingError::Unknown))?;

      let threads_u32 = num_threads as u32;
      unsafe {
        SetThreadpoolThreadMinimum(pool, threads_u32)
          .map_err(|_| NativeError::OsThreadingError(crate::os::ThreadingError::Unknown))?;
        SetThreadpoolThreadMaximum(pool, threads_u32);
      }

      let mut callback_environ = TP_CALLBACK_ENVIRON_V3::default();
      callback_environ.Version = 3;
      callback_environ.CallbackPriority = TP_CALLBACK_PRIORITY_NORMAL;
      callback_environ.Size = core::mem::size_of::<TP_CALLBACK_ENVIRON_V3>() as u32;
      callback_environ.Pool = pool;

      Ok(Self {
        pool,
        callback_environ,
        work_items: Mutex::new(Vec::new()), // Initialize Mutex
      })
    }

    /// Scatters the given workloads among the threads in the pool.
    pub fn scatter(&self, workloads: Vec<Box<dyn Workload>>) -> NativeResult<()> {
      // Changed to &self
      let mut items = self.work_items.lock();

      for workload in workloads {
        let thin_ptr = Box::into_raw(Box::new(workload));

        let work = unsafe {
          CreateThreadpoolWork(
            Some(work_callback),
            Some(thin_ptr as *mut core::ffi::c_void),
            Some(core::ptr::from_ref(&self.callback_environ)),
          )
        }
        .map_err(|_| NativeError::OsThreadingError(crate::os::ThreadingError::Unknown))?;

        items.push(work);

        unsafe {
          SubmitThreadpoolWork(work);
        }
      }

      Ok(())
    }

    /// Gathers the results of the scattered workloads.
    pub fn gather(&self) {
      // Changed to &self
      // Extract the works quickly so we don't hold the spinlock while waiting
      let works = {
        let mut items = self.work_items.lock();
        core::mem::take(&mut *items)
      };

      for work in works {
        unsafe {
          WaitForThreadpoolWorkCallbacks(work, false.into());
          CloseThreadpoolWork(work);
        }
      }
    }
  }

  impl Drop for ThreadPool {
    fn drop(&mut self) {
      unsafe { CloseThreadpool(self.pool) };
    }
  }

  unsafe extern "system" fn work_callback(
    _instance: PTP_CALLBACK_INSTANCE,
    context: *mut core::ffi::c_void,
    _work: PTP_WORK,
  ) {
    let workload = unsafe { Box::from_raw(context as *mut Box<dyn Workload>) };
    workload.execute();
  }
}

#[cfg(unix)]
mod pthread_pool {
  use super::*;
  use alloc::collections::VecDeque;
  use alloc::sync::Arc;
  use alloc::vec::Vec;
  use alloc::boxed::Box;
  use core::ptr;
  use libc::{pthread_create, pthread_join, pthread_t};
  use spin::Mutex;
  use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
  use core::ffi::c_void;

  struct ThreadPoolState {
    work_queue: Mutex<VecDeque<Box<dyn Workload>>>,
    shutdown: AtomicBool,
    active_threads: AtomicUsize,
  }

  /// A thread pool for executing workloads.
  pub struct ThreadPool {
    threads: Vec<pthread_t>,
    state: Arc<ThreadPoolState>,
  }

  // Explicitly mark as Send/Sync since `pthread_t` might act as a raw pointer
  // on certain unix variants, blocking the auto-trait.
  unsafe impl Send for ThreadPool {}
  unsafe impl Sync for ThreadPool {}

  impl ThreadPool {
    pub fn new(num_threads: usize) -> NativeResult<Self> {
      let state = Arc::new(ThreadPoolState {
        work_queue: Mutex::new(VecDeque::new()),
        shutdown: AtomicBool::new(false),
        active_threads: AtomicUsize::new(0),
      });

      let mut threads = Vec::with_capacity(num_threads);

      for _ in 0..num_threads {
        let state = Arc::clone(&state);
        let mut thread: pthread_t = 0;
        let ret = unsafe {
          pthread_create(
            &mut thread,
            ptr::null(),
            thread_func,
            Arc::into_raw(state) as *mut _,
          )
        };
        if ret == 0 {
          threads.push(thread);
        } else {
          // Handle thread creation error
        }
      }

      Ok(Self { threads, state })
    }

    /// Scatters the given workloads among the threads in the pool.
    pub fn scatter(&self, workloads: Vec<Box<dyn Workload>>) -> NativeResult<()> {
      // Changed to &self
      let mut queue = self.state.work_queue.lock();
      for workload in workloads {
        queue.push_back(workload);
      }

      Ok(())
    }

    /// Gathers the results of the scattered workloads.
    pub fn gather(&self) {
      while !self.state.work_queue.lock().is_empty()
        || self.state.active_threads.load(Ordering::SeqCst) > 0
      {
        unsafe { libc::sched_yield() };
      }
    }
  }

  impl Drop for ThreadPool {
    fn drop(&mut self) {
      self.state.shutdown.store(true, Ordering::SeqCst);
      for thread in &self.threads {
        unsafe {
          pthread_join(*thread, ptr::null_mut());
        }
      }
    }
  }

  extern "C" fn thread_func(arg: *mut c_void) -> *mut c_void {
    let state = unsafe { Arc::from_raw(arg as *mut ThreadPoolState) };

    while !state.shutdown.load(Ordering::SeqCst) {
      let workload = {
        let mut queue = state.work_queue.lock();
        queue.pop_front()
      };

      if let Some(workload) = workload {
        state.active_threads.fetch_add(1, Ordering::SeqCst);
        workload.execute();
        state.active_threads.fetch_sub(1, Ordering::SeqCst);
      } else {
        unsafe { libc::sched_yield() };
      }
    }

    ptr::null_mut()
  }
}

#[cfg(target_os = "windows")]
pub use windows_pool::ThreadPool;
#[cfg(unix)]
pub use pthread_pool::ThreadPool;

#[cfg(test)]
mod tests {
  use super::*;
  use core::sync::atomic::{AtomicUsize, Ordering};
  use alloc::sync::Arc;

  struct TestWorkload {
    counter: Arc<AtomicUsize>,
  }

  impl Workload for TestWorkload {
    fn execute(&self) {
      self.counter.fetch_add(1, Ordering::SeqCst);
    }
  }

  #[test]
  fn test_thread_pool_scatter_gather() {
    let counter = Arc::new(AtomicUsize::new(0));
    let mut pool = ThreadPool::new(4).expect("Failed to create thread pool");

    let mut workloads: Vec<Box<dyn Workload>> = Vec::new();
    for _ in 0..100 {
      workloads.push(Box::new(TestWorkload {
        counter: Arc::clone(&counter),
      }));
    }

    pool
      .scatter(workloads)
      .expect("Failed to scatter workloads");
    pool.gather();

    assert_eq!(counter.load(Ordering::SeqCst), 100);
  }
}
