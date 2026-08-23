
use super::*;

#[test]
fn test_query_memory_status() {
  let status = query_memory_status();
  // Since different platforms return different things, just check it doesn't crash
  // and if it returns total > 0 on supported platforms
  #[cfg(not(target_os = "linux"))]
  assert!(status.total_bytes > 0);
}

#[test]
fn test_stack_allocator() {
  let allocator = StackAllocator::new();
  let mut buffer = [0u8; 1024];
  let base_ptr = buffer.as_mut_ptr();

  let val_ptr = unsafe { allocator.allocate(base_ptr, 1024, 42u32).unwrap() };
  unsafe { assert_eq!(*val_ptr, 42) };

  let val_ptr2 = unsafe { allocator.allocate(base_ptr, 1024, 100u64).unwrap() };
  unsafe { assert_eq!(*val_ptr2, 100) };
}
