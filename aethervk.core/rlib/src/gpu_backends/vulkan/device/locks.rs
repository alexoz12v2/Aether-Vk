use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use core::hash::{Hash, Hasher};
use aethervk_oshal_rlib::os::native::this_thread;
use aethervk_oshal_rlib::hash::FnvHasher;

const MAX_TRACKED_THREADS: usize = 64;
static THREAD_HASHES: [AtomicU64; MAX_TRACKED_THREADS] = [const { AtomicU64::new(0) }; MAX_TRACKED_THREADS];
static THREAD_COUNTS: [AtomicUsize; MAX_TRACKED_THREADS] = [const { AtomicUsize::new(0) }; MAX_TRACKED_THREADS];

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

#[derive(Debug)]
pub struct DebugTrackedRwLock<T> {
    inner: spin::RwLock<T>,
}

impl<T> DebugTrackedRwLock<T> {
    pub const fn new(value: T) -> Self {
        Self { inner: spin::RwLock::new(value) }
    }

    pub fn read(&self) -> DebugTrackedRwLockReadGuard<'_, T> {
        increment_lock_count();
        DebugTrackedRwLockReadGuard {
            inner: self.inner.read(),
        }
    }

    pub fn write(&self) -> DebugTrackedRwLockWriteGuard<'_, T> {
        increment_lock_count();
        DebugTrackedRwLockWriteGuard {
            inner: self.inner.write(),
        }
    }
}

pub struct DebugTrackedRwLockReadGuard<'a, T: ?Sized + 'a> {
    inner: spin::RwLockReadGuard<'a, T>,
}
impl<'a, T: ?Sized + 'a> core::ops::Deref for DebugTrackedRwLockReadGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T { &self.inner }
}
impl<'a, T: ?Sized + 'a> Drop for DebugTrackedRwLockReadGuard<'a, T> {
    fn drop(&mut self) { decrement_lock_count(); }
}

pub struct DebugTrackedRwLockWriteGuard<'a, T: ?Sized + 'a> {
    inner: spin::RwLockWriteGuard<'a, T>,
}
impl<'a, T: ?Sized + 'a> core::ops::Deref for DebugTrackedRwLockWriteGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T { &self.inner }
}
impl<'a, T: ?Sized + 'a> core::ops::DerefMut for DebugTrackedRwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T { &mut self.inner }
}
impl<'a, T: ?Sized + 'a> Drop for DebugTrackedRwLockWriteGuard<'a, T> {
    fn drop(&mut self) { decrement_lock_count(); }
}

#[derive(Debug)]
pub struct DebugTrackedMutex<T> {
    inner: spin::Mutex<T>,
}

impl<T> DebugTrackedMutex<T> {
    pub const fn new(value: T) -> Self {
        Self { inner: spin::Mutex::new(value) }
    }
    
    pub fn lock(&self) -> DebugTrackedMutexGuard<'_, T> {
        increment_lock_count();
        DebugTrackedMutexGuard {
            inner: self.inner.lock(),
        }
    }
}

pub struct DebugTrackedMutexGuard<'a, T: ?Sized + 'a> {
    inner: spin::MutexGuard<'a, T>,
}
impl<'a, T: ?Sized + 'a> core::ops::Deref for DebugTrackedMutexGuard<'a, T> {
    type Target = T;
    fn deref(&self) -> &T { &self.inner }
}
impl<'a, T: ?Sized + 'a> core::ops::DerefMut for DebugTrackedMutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut T { &mut self.inner }
}
impl<'a, T: ?Sized + 'a> Drop for DebugTrackedMutexGuard<'a, T> {
    fn drop(&mut self) { decrement_lock_count(); }
}
