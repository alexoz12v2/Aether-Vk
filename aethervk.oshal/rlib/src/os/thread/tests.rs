
use super::*;
use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};

#[test]
fn test_thread_spawn_and_join() {
  let flag = Arc::new(AtomicBool::new(false));
  let flag_clone = Arc::clone(&flag);

  let thread = spawn(move || {
    flag_clone.store(true, Ordering::SeqCst);
  })
  .expect("Failed to spawn thread");

  thread.join();
  assert!(flag.load(Ordering::SeqCst));
}

#[test]
fn test_thread_builder() {
  let thread = Builder::new()
    .name(String::from("test_thread"))
    .stack_size(1024 * 1024)
    .spawn(|| {
      // do nothing
    })
    .expect("Failed to spawn thread with builder");

  thread.join();
}
