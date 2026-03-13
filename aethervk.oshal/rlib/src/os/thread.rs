
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ThreadId(u64);

pub fn current_id() -> ThreadId {
  let id: ThreadId;
  #[cfg(unix)] {
    id = ThreadId(unsafe { libc::pthread_self() } as _);
  }
  #[cfg(windows)]
  {
    todo!()
  }

  id
}