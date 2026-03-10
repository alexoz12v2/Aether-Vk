use alloc::{string::String};

use heapless::index_map::FnvIndexMap;
use thiserror::Error;

pub const RUNTIME_PARAMS_MAX_COUNT: usize = 16;
pub(super) type RuntimeParamsIndex = u32;

pub struct RuntimeParams {
  // TODO add logging stuff
  // backend specific additional parameters (each backend root file defined `RUNTIME_PARAM_<NAME>_*`)
  pub render_backend_params: FnvIndexMap<RuntimeParamsIndex, String, RUNTIME_PARAMS_MAX_COUNT>,
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
}

pub type EngineResult<T> = core::result::Result<T, EngineError>;

// TODO: These will evolve as development progresses

#[derive(Debug, Error)]
pub enum GpuError {
  #[error("Invalid Input Argument")]
  InvalidArgument,

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

  #[error("Backend error: {0}")]
  BackendSpecific(String),
}

pub type GpuResult<T> = core::result::Result<T, GpuError>;

#[derive(Debug, Error)]
pub enum IoError {}

pub type IoResult<T> = core::result::Result<T, IoError>;

#[derive(Debug, Error)]
pub enum MathError {}

pub type MathResult<T> = core::result::Result<T, MathError>;
