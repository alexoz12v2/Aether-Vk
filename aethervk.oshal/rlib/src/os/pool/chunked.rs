use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use crate::os::NativeResult;
use super::{ThreadPool, Workload, WorkloadStatus};

pub struct ChunkedState {
  completed: AtomicUsize,
  total: usize,
}

pub struct ChunkedHandle {
  state: Arc<ChunkedState>,
}

impl ChunkedHandle {
  pub fn wait(self) {
    while self.state.completed.load(Ordering::Acquire) < self.state.total {
      #[cfg(any(unix, target_os = "macos"))]
      unsafe {
        libc::sched_yield()
      };
      #[cfg(not(any(unix, target_os = "macos")))]
      core::hint::spin_loop();
    }
  }
}

struct ChunkWorkload<F> {
  func: Arc<F>,
  chunk_id: usize,
  state: Arc<ChunkedState>,
}

impl<F> Workload for ChunkWorkload<F>
where
  // Requires `Sync` exclusively here because `F` is inherently wrapped in an `Arc` for threading dispatch
  F: Fn(usize) + Send + Sync + 'static,
{
  fn execute(&mut self) -> WorkloadStatus {
    (self.func)(self.chunk_id);
    WorkloadStatus::Complete
  }

  fn tasklet_id(&self) -> Option<usize> {
    None
  }
}

impl<F> Drop for ChunkWorkload<F> {
  fn drop(&mut self) {
    self.state.completed.fetch_add(1, Ordering::Release);
  }
}

pub trait ThreadPoolChunkedExt {
  fn spawn_chunked<F>(&self, num_chunks: usize, f: F) -> NativeResult<ChunkedHandle>
  where
    F: Fn(usize) + Send + Sync + 'static;
}

impl ThreadPoolChunkedExt for ThreadPool {
  fn spawn_chunked<F>(&self, num_chunks: usize, f: F) -> NativeResult<ChunkedHandle>
  where
    F: Fn(usize) + Send + Sync + 'static,
  {
    let state = Arc::new(ChunkedState {
      completed: AtomicUsize::new(0),
      total: num_chunks,
    });

    if num_chunks == 0 {
      return Ok(ChunkedHandle { state });
    }

    let func = Arc::new(f);
    let mut workloads: Vec<Box<dyn Workload>> = Vec::with_capacity(num_chunks);

    for chunk_id in 0..num_chunks {
      workloads.push(Box::new(ChunkWorkload {
        func: Arc::clone(&func),
        chunk_id,
        state: Arc::clone(&state),
      }));
    }

    self.scatter(workloads)?;

    Ok(ChunkedHandle { state })
  }
}
