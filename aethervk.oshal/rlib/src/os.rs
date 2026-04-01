use thiserror::Error;

pub mod debug;
pub mod fs;
pub mod memory;
pub mod native;
pub mod env;
pub mod pool;
pub mod thread;
pub mod time;

#[repr(u32)]
#[derive(Debug, Clone, Copy)]
pub enum ThreadingError {
  /// - Too many threads already exist
  /// - stack allocation fails
  /// - system is low on memory or kernel resources
  /// - system imposed limits (eg job object limits)
  NoResources,
  /// process lacks the rights to create a thread
  Access,
  /// Invalid parameters, such as
  /// - invalid scack size
  /// - bad function pointer
  /// - corrupt or invalid attributes structure
  InvalidArg,
  /// - Start routine pointer is not valid
  /// - memory is not executable/accessible
  BadAddress,
  /// what happened
  Unknown,
}

#[derive(Debug, Error)]
pub enum NativeError {
  #[error("Unknown Error")]
  UnknownError,

  #[error("Invalid Arguments")]
  InvalidArgument,

  #[error("Thread Error {0:?}")]
  OsThreadingError(ThreadingError),
}

impl NativeError {
  pub fn threading_from_raw_os_error(error: i32) -> ThreadingError {
    let threading_error: ThreadingError;
    #[cfg(windows)]
    {
      use windows::Win32::Foundation::*;
      let error = WIN32_ERROR(error as u32);

      threading_error = match error {
        ERROR_NOT_ENOUGH_MEMORY
        | ERROR_OUTOFMEMORY
        | ERROR_NO_SYSTEM_RESOURCES
        | ERROR_NOT_ENOUGH_QUOTA => ThreadingError::NoResources,
        ERROR_ACCESS_DENIED => ThreadingError::BadAddress,
        ERROR_INVALID_PARAMETER => ThreadingError::InvalidArg,
        ERROR_INVALID_ADDRESS => ThreadingError::BadAddress,
        ERROR_MAX_THRDS_REACHED | ERROR_NOT_ENOUGH_QUOTA => ThreadingError::NoResources,
        _ => ThreadingError::Unknown,
      };
    }
    #[cfg(unix)]
    {
      use libc::*;
      threading_error = match error {
        EAGAIN => ThreadingError::NoResources,
        EPERM => ThreadingError::Access,
        EINVAL => ThreadingError::InvalidArg,
        EFAULT => ThreadingError::BadAddress,
        _ => ThreadingError::Unknown,
      };
    }

    threading_error
  }
}

pub type NativeResult<T> = core::result::Result<T, NativeError>;
pub type ThreadingResult<T> = core::result::Result<T, ThreadingError>;

// maybe this is not necessary
impl From<ThreadingError> for NativeError {
  fn from(value: ThreadingError) -> Self {
    value.into()
  }
}
