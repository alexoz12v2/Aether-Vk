
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
