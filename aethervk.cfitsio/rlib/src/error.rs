use alloc::{string::String, vec::Vec};
use core::ffi::{CStr, c_char, c_int};

use aethervk_cfitsio_sys as sys;

/// A failed cfitsio call: the status code, its `ffgerr` text and the error-message stack the
/// library accumulated (`ffgmsg`, drained and therefore cleared when the error is built).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FitsError {
  pub status: i32,
  pub text: String,
  pub messages: Vec<String>,
}

impl FitsError {
  /// Must be called under the session lock (the message stack is global).
  pub(crate) fn from_status(status: c_int) -> Self {
    let mut buf = [0 as c_char; sys::FLEN_STATUS as usize];
    // SAFETY: ffgerr writes at most FLEN_STATUS bytes including the terminator.
    unsafe { sys::ffgerr(status, buf.as_mut_ptr()) };
    let text = c_to_string(&buf);
    let mut messages = Vec::new();
    let mut msg = [0 as c_char; sys::FLEN_ERRMSG as usize];
    // SAFETY: ffgmsg writes at most FLEN_ERRMSG bytes; it returns 0 when the stack is empty.
    while unsafe { sys::ffgmsg(msg.as_mut_ptr()) } != 0 {
      messages.push(c_to_string(&msg));
      if messages.len() > 64 {
        break;
      }
    }
    FitsError {
      status,
      text,
      messages,
    }
  }

  /// An error raised by the wrapper itself (no library call failed), with an empty message stack.
  pub(crate) fn wrapper(status: c_int, text: &str) -> Self {
    FitsError {
      status,
      text: String::from(text),
      messages: Vec::new(),
    }
  }
}

impl core::fmt::Display for FitsError {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "cfitsio status {} ({})", self.status, self.text)?;
    for (i, m) in self.messages.iter().enumerate() {
      write!(f, "{}{}", if i == 0 { ": " } else { "; " }, m)?;
    }
    Ok(())
  }
}

#[cfg(any(test, feature = "std"))]
impl std::error::Error for FitsError {}

/// A NUL-terminated C buffer (terminated by construction: the last byte is never written by
/// cfitsio) as a String, with trailing blanks trimmed.
pub(crate) fn c_to_string(buf: &[c_char]) -> String {
  let bytes: &[u8] = unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len()) };
  let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
  String::from_utf8_lossy(&bytes[..end]).trim_end().into()
}

/// Length of a NUL-terminated string (bounded by the slice) as a `&str`, lossless for ASCII.
pub(crate) fn cstr_lossy(p: *const c_char) -> String {
  if p.is_null() {
    return String::new();
  }
  // SAFETY: cfitsio hands out NUL-terminated strings it allocated.
  let s = unsafe { CStr::from_ptr(p) };
  String::from_utf8_lossy(s.to_bytes()).into()
}
