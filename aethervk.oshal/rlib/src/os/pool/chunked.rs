//! chunked module.

use alloc::boxed::Box;
use alloc::sync::Arc;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::{ThreadPool, Workload, WorkloadStatus};
use crate::os::NativeResult;

/// TODO: Document this item
pub struct ChunkedState {
  completed: AtomicUsize,
  total: usize,
}

/// TODO: Document this item
pub struct ChunkedHandle {
  state: Arc<ChunkedState>,
}

impl ChunkedHandle {
  /// TODO: Document this item
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

/// Requires `Sync` exclusively here because `F` is inherently wrapped in an `Arc` for threading dispatch
/// closure has to be potentially `static cause threads might live forever. Any attempt to shorten
/// lifetime of this closure should be done through type erasure
impl<F> Workload for ChunkWorkload<F>
where
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

/// TODO: Document this item
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
