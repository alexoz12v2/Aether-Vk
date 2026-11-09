use alloc::vec::Vec;
use core::ffi::{c_char, c_int};

use aethervk_cfitsio_sys as sys;

use crate::{FitsError, SESSION};

/// `ffs2tm` output. For a time-only string (`hh:mm:ss`) the date fields are 0; for a date-only
/// string the time fields are 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitsTime {
  pub year: i32,
  pub month: i32,
  pub day: i32,
  pub hour: i32,
  pub minute: i32,
  pub second: f64,
}

/// Parses a FITS date/time string with cfitsio's `ffs2tm`.
///
/// Accepted by cfitsio 4.7.0 (see the tests for the verified behaviour): `dd/mm/yy` (year
/// `1900 + yy`), `YYYY-MM-DD`, `YYYY-MM-DDThh:mm:ss[.fff]` and the time-only `hh:mm:ss[.fff]`.
/// Anything else (`dd/mm/yyyy`, a space instead of `T`, a lowercase `t`, trailing zone letters…)
/// is `Err` with status `BAD_DATE`; a normaliser must run before this call.
///
/// Takes the global session lock: call it with no [`crate::MemFits`] open on this thread (use
/// [`crate::MemFits::str2time`] inside a session).
pub fn str2time(canonical: &str) -> Result<FitsTime, FitsError> {
  let _guard = SESSION.lock();
  str2time_locked(canonical)
}

/// Caller holds `SESSION`.
pub(crate) fn str2time_locked(canonical: &str) -> Result<FitsTime, FitsError> {
  if canonical.len() >= sys::FLEN_VALUE as usize || canonical.bytes().any(|b| b == 0) {
    return Err(FitsError::wrapper(
      sys::BAD_DATE as c_int,
      "date string too long or with an interior NUL",
    ));
  }
  let mut cstr: Vec<u8> = Vec::with_capacity(canonical.len() + 1);
  cstr.extend_from_slice(canonical.as_bytes());
  cstr.push(0);
  // SAFETY: the stack is global; we hold the lock.
  unsafe { sys::ffcmsg() };
  let (mut year, mut month, mut day, mut hour, mut minute): (c_int, c_int, c_int, c_int, c_int) =
    (0, 0, 0, 0, 0);
  let mut second: f64 = 0.0;
  let mut status: c_int = 0;
  // SAFETY: `cstr` is NUL-terminated and every out-pointer is valid; ffs2tm only reads the string.
  unsafe {
    sys::ffs2tm(
      cstr.as_mut_ptr().cast::<c_char>(),
      &mut year,
      &mut month,
      &mut day,
      &mut hour,
      &mut minute,
      &mut second,
      &mut status,
    );
  }
  if status != 0 {
    return Err(FitsError::from_status(status));
  }
  Ok(FitsTime {
    year,
    month,
    day,
    hour,
    minute,
    second,
  })
}
