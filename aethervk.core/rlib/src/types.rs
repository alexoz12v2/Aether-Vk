
use alloc::{ string::String };

use thiserror::Error;

// ---------------------------- Error Types -----------------------------------

#[derive(Error, Debug)]
pub(super) enum EngineError {
  #[error("GPU Error: {0}")]
  Gpu(#[from] GpuError),

  #[error("IO Error: {0}")]
  Io(#[from] IoError),

  #[error("Math Error: {0}")]
  Math(#[from] MathError),

  #[error("Invalid Operation: {0}")]
  InvalidOperation(&'static str),
}

pub(super) type EngineResult<T> = core::result::Result<T, EngineError>;

// TODO: These will evolve as development progresses

#[derive(Debug, Error)]
pub(super) enum GpuError {
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

pub(super) type GpuResult<T> = core::result::Result<T, GpuError>;

#[derive(Debug, Error)]
pub(super) enum IoError {
  
}

pub(super) type IoResult<T> = core::result::Result<T, IoError>;

#[derive(Debug, Error)]
pub(super) enum MathError {

}

pub(super) type MathResult<T> = core::result::Result<T, MathError>;
