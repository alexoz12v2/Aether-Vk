use core::sync::atomic::{AtomicUsize, Ordering};

use alloc::{boxed::Box, string::String, vec::Vec};

use heapless::index_map::FnvIndexMap;
use thiserror::Error;
use aethervk_oshal_rlib::os::{FsError, NativeError};

pub const RUNTIME_PARAMS_MAX_COUNT: usize = 16;
pub(super) type RuntimeParamsIndex = u32;

pub struct RuntimeParams {
  // TODO add logging stuff
  // backend specific additional parameters (each backend root file defined `RUNTIME_PARAM_<NAME>_*`)
  pub render_backend_params: FnvIndexMap<RuntimeParamsIndex, String, RUNTIME_PARAMS_MAX_COUNT>,
  pub validation_error_callback: Option<fn(&str)>,
}

impl RuntimeParams {
  pub fn new_with_callback(validation_error_callback: Option<fn(&str)>) -> Self {
    Self {
      render_backend_params: FnvIndexMap::new(),
      validation_error_callback,
    }
  }
}

// ---------------------------- Error Types -----------------------------------

#[derive(Error, Debug)]
pub enum EngineError {
  #[error("GPU Error: {0}")]
  Gpu(#[from] GpuError),

  #[error("IO Error: {0}")]
  Io(#[from] IoError),

  #[error("Math Error: {0}")]
  Math(#[from] MathError),

  #[error("Invalid Operation: {0}")]
  InvalidOperation(&'static str),

  #[error("null argument")]
  InvalidNullArgument,
  
  #[error("Native Error: {0}")]
  Native(#[from] NativeError),
}

pub type EngineResult<T> = core::result::Result<T, EngineError>;

// TODO: These will evolve as development progresses

#[derive(Debug, Error, Clone)]
pub enum GpuError {
  #[error("Invalid Input Argument: {0}")]
  InvalidArgument(&'static str),

  #[error("Invalid Object State: {0}")]
  InvalidState(&'static str),

  #[error("Device lost")]
  DeviceLost,

  #[error("Out of Memory")]
  OutOfMemory,

  #[error("Invalid Shader")]
  InvalidShader,

  #[error("Unsupported feature")]
  UnsupportedFeature,

  #[error("Unsupported feature: {0}")]
  UnsupportedFeatureNamed(String),

  #[error("Presentation engine Resize Required")]
  ResizeRequired,

  #[error("Backend error: {0}")]
  BackendSpecific(String),
  
  #[error("Resource not found")]
  NotFound,
}

pub type GpuResult<T> = core::result::Result<T, GpuError>;

#[derive(Debug, Error)]
pub enum IoError {
  #[error("IO Error")]
  Failed,
  #[error("IO Error {0:?}")]
  FileIO(FsError),
  #[error("IO Error (Specific): {0}")]
  Specific(&'static str),
}

impl From<FsError> for IoError {
  fn from(value: FsError) -> Self {
    Self::FileIO(value)
  }
}

pub type IoResult<T> = core::result::Result<T, IoError>;

#[derive(Debug, Error)]
pub enum MathError {}

pub type MathResult<T> = core::result::Result<T, MathError>;

// ---------------------------- Generic Data Structures -----------------------
#[derive(Debug)]
pub struct SpscQueue<T> {
  buffer: Box<[Option<T>]>,
  mask: usize,
  head: AtomicUsize,
  tail: AtomicUsize,
}

impl<T: Copy> SpscQueue<T> {
  pub fn new(cap_pow2: usize) -> Self {
    debug_assert!(cap_pow2 >= 2 && (cap_pow2 & (cap_pow2 - 1)) == 0);
    let mut storage = Vec::with_capacity(cap_pow2);
    for _ in 0..cap_pow2 {
      storage.push(None);
    }

    Self {
      buffer: storage.into_boxed_slice(),
      mask: cap_pow2 - 1,
      head: AtomicUsize::new(0),
      tail: AtomicUsize::new(0),
    }
  }

  pub fn try_push(&self, item: T) -> bool {
    let tail = self.tail.load(Ordering::Relaxed);
    let head = self.head.load(Ordering::Acquire); // see latest head memory

    if ((tail + 1) & self.mask) == (head & self.mask) {
      return false; // Full queue
    }

    // Safety: we own this slot based on SPSC logic
    let slot_ptr = unsafe {
      let ptr = self.buffer.as_ptr() as *mut Option<T>;
      ptr.add(tail & self.mask)
    };
    unsafe {
      *slot_ptr = Some(item);
    };
    self.tail.store(tail + 1, Ordering::Release); // publish memory write
    true
  }

  pub fn try_pop(&self) -> Option<T> {
    let head = self.head.load(Ordering::Relaxed);
    let tail = self.tail.load(Ordering::Acquire); // see latest head memory

    if (head & self.mask) == (tail & self.mask) {
      return None; // empty queue
    }

    let item = self.buffer[head & self.mask];
    self.head.store(head + 1, Ordering::Release); // release memory edits
    item
  }
}
