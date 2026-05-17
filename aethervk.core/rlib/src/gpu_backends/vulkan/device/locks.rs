// no_std, enhance their usage by making their methods take &Self explicitly, like Arc::clone, and make it so that its usage it's like spin::RwLock
use aethervk_oshal_rlib::{hash::FnvHasher, os::native::this_thread};
use core::{
  hash::{Hash, Hasher},
  sync::atomic::{AtomicU64, AtomicUsize, Ordering},
};

const MAX_TRACKED_THREADS: usize = 64;
static THREAD_HASHES: [AtomicU64; MAX_TRACKED_THREADS] =
  [const { AtomicU64::new(0) }; MAX_TRACKED_THREADS];
static THREAD_COUNTS: [AtomicUsize; MAX_TRACKED_THREADS] =
  [const { AtomicUsize::new(0) }; MAX_TRACKED_THREADS];

fn get_thread_hash() -> u64 {
  let mut hasher = FnvHasher::new();
  this_thread::id().hash(&mut hasher);
  let h = hasher.finish();
  if h == 0 { 1 } else { h }
}

#[inline(always)]
pub fn increment_lock_count() {
  #[cfg(debug_assertions)]
  {
    let h = get_thread_hash();
    for i in 0..MAX_TRACKED_THREADS {
      let existing = THREAD_HASHES[i].load(Ordering::Acquire);
      if existing == h {
        THREAD_COUNTS[i].fetch_add(1, Ordering::SeqCst);
        return;
      }
      if existing == 0 {
        if THREAD_HASHES[i].compare_exchange(0, h, Ordering::SeqCst, Ordering::SeqCst).is_ok() {
          THREAD_COUNTS[i].fetch_add(1, Ordering::SeqCst);
          return;
        } else if THREAD_HASHES[i].load(Ordering::Acquire) == h {
          THREAD_COUNTS[i].fetch_add(1, Ordering::SeqCst);
          return;
        }
      }
    }
    panic!("Exceeded MAX_TRACKED_THREADS for lock tracking");
  }
}

#[inline(always)]
pub fn decrement_lock_count() {
  #[cfg(debug_assertions)]
  {
    let h = get_thread_hash();
    for i in 0..MAX_TRACKED_THREADS {
      let existing = THREAD_HASHES[i].load(Ordering::Acquire);
      if existing == h {
        THREAD_COUNTS[i].fetch_sub(1, Ordering::SeqCst);
        return;
      }
    }
  }
}

#[inline(always)]
pub fn assert_no_locks_held() {
  #[cfg(debug_assertions)]
  {
    let h = get_thread_hash();
    for i in 0..MAX_TRACKED_THREADS {
      let existing = THREAD_HASHES[i].load(Ordering::Acquire);
      if existing == h {
        let count = THREAD_COUNTS[i].load(Ordering::SeqCst);
        if count != 0 {
          panic!("Vulkan API called while holding {} locks!", count);
        }
        return;
      }
    }
  }
}

/// pub(super) because this is meant to manage loks under  the vulkan::device module.
#[derive(Debug)]
pub(super) struct DebugTrackedRwLock<T: ?Sized> {
  inner: spin::RwLock<T>,
}

// Methods only available for Sized types
impl<T> DebugTrackedRwLock<T> {
  pub const fn new(value: T) -> Self {
    Self {
      inner: spin::RwLock::new(value),
    }
  }

  pub fn into_inner(this: Self) -> T {
    this.inner.into_inner()
  }
}

// Methods available for all types, including ?Sized (e.g. `[u8]` or `dyn Trait`)
impl<T: ?Sized> DebugTrackedRwLock<T> {
  pub fn read(this: &Self) -> DebugTrackedRwLockReadGuard<'_, T> {
    let inner = this.inner.read();
    increment_lock_count();
    DebugTrackedRwLockReadGuard { inner }
  }

  pub fn try_read(this: &Self) -> Option<DebugTrackedRwLockReadGuard<'_, T>> {
    let inner = this.inner.try_read()?;
    increment_lock_count();
    Some(DebugTrackedRwLockReadGuard { inner })
  }

  pub fn write(this: &Self) -> DebugTrackedRwLockWriteGuard<'_, T> {
    let inner = this.inner.write();
    increment_lock_count();
    DebugTrackedRwLockWriteGuard { inner }
  }

  pub fn try_write(this: &Self) -> Option<DebugTrackedRwLockWriteGuard<'_, T>> {
    let inner = this.inner.try_write()?;
    increment_lock_count();
    Some(DebugTrackedRwLockWriteGuard { inner })
  }

  pub fn get_mut(this: &mut Self) -> &mut T {
    this.inner.get_mut()
  }
}

impl<T: Default> Default for DebugTrackedRwLock<T> {
  fn default() -> Self {
    Self::new(T::default())
  }
}

impl<T> From<T> for DebugTrackedRwLock<T> {
  fn from(value: T) -> Self {
    Self::new(value)
  }
}

pub struct DebugTrackedRwLockReadGuard<'a, T: ?Sized + 'a> {
  inner: spin::RwLockReadGuard<'a, T>,
}
impl<'a, T: ?Sized + 'a> core::ops::Deref for DebugTrackedRwLockReadGuard<'a, T> {
  type Target = T;
  fn deref(&self) -> &T {
    &self.inner
  }
}
impl<'a, T: ?Sized + 'a> Drop for DebugTrackedRwLockReadGuard<'a, T> {
  fn drop(&mut self) {
    decrement_lock_count();
  }
}

pub struct DebugTrackedRwLockWriteGuard<'a, T: ?Sized + 'a> {
  inner: spin::RwLockWriteGuard<'a, T>,
}
impl<'a, T: ?Sized + 'a> core::ops::Deref for DebugTrackedRwLockWriteGuard<'a, T> {
  type Target = T;
  fn deref(&self) -> &T {
    &self.inner
  }
}
impl<'a, T: ?Sized + 'a> core::ops::DerefMut for DebugTrackedRwLockWriteGuard<'a, T> {
  fn deref_mut(&mut self) -> &mut T {
    &mut self.inner
  }
}
impl<'a, T: ?Sized + 'a> Drop for DebugTrackedRwLockWriteGuard<'a, T> {
  fn drop(&mut self) {
    decrement_lock_count();
  }
}

#[derive(Debug)]
pub struct DebugTrackedMutex<T: ?Sized> {
  inner: spin::Mutex<T>,
}

impl<T> DebugTrackedMutex<T> {
  pub const fn new(value: T) -> Self {
    Self {
      inner: spin::Mutex::new(value),
    }
  }

  pub fn into_inner(this: Self) -> T {
    this.inner.into_inner()
  }
}

impl<T: ?Sized> DebugTrackedMutex<T> {
  pub fn lock(this: &Self) -> DebugTrackedMutexGuard<'_, T> {
    let inner = this.inner.lock();
    increment_lock_count();
    DebugTrackedMutexGuard { inner }
  }

  pub fn try_lock(this: &Self) -> Option<DebugTrackedMutexGuard<'_, T>> {
    let inner = this.inner.try_lock()?;
    increment_lock_count();
    Some(DebugTrackedMutexGuard { inner })
  }

  pub fn get_mut(this: &mut Self) -> &mut T {
    this.inner.get_mut()
  }
}

impl<T: Default> Default for DebugTrackedMutex<T> {
  fn default() -> Self {
    Self::new(T::default())
  }
}

impl<T> From<T> for DebugTrackedMutex<T> {
  fn from(value: T) -> Self {
    Self::new(value)
  }
}

pub struct DebugTrackedMutexGuard<'a, T: ?Sized + 'a> {
  inner: spin::MutexGuard<'a, T>,
}
impl<'a, T: ?Sized + 'a> core::ops::Deref for DebugTrackedMutexGuard<'a, T> {
  type Target = T;
  fn deref(&self) -> &T {
    &self.inner
  }
}
impl<'a, T: ?Sized + 'a> core::ops::DerefMut for DebugTrackedMutexGuard<'a, T> {
  fn deref_mut(&mut self) -> &mut T {
    &mut self.inner
  }
}
impl<'a, T: ?Sized + 'a> Drop for DebugTrackedMutexGuard<'a, T> {
  fn drop(&mut self) {
    decrement_lock_count();
  }
}
