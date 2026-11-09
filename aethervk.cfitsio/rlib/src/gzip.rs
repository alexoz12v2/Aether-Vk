//! Whole-file gzip through cfitsio's own in-memory codec (`uncompress2mem_from_mem` /
//! `compress2mem_from_mem` in `zcompress.c`, over the zlib the library is built with): a
//! `.fits.gz` is inflated into a Rust-owned `Vec` first, then opened like any other buffer by
//! [`crate::MemFits::open`]. Nothing else in cfitsio's memory driver reads gzip (`ffomem` wants
//! the plain bytes).

use alloc::vec::Vec;
use core::{
  ffi::{c_char, c_int, c_void},
  ptr,
};

use aethervk_cfitsio_sys as sys;

use crate::{FitsError, SESSION};

unsafe extern "C" {
  pub(crate) fn malloc(size: usize) -> *mut c_void;
  pub(crate) fn realloc(p: *mut c_void, size: usize) -> *mut c_void;
  pub(crate) fn free(p: *mut c_void);
}

/// The most bytes [`gunzip`] inflates (3 GiB): a FITS pixel array over 2 GB is refused by the
/// reader anyway, and a gzip bomb must not grow without bound. `MEMORY_ALLOCATION` past it.
pub const MAX_GUNZIP_BYTES: usize = 3 << 30;

/// The reallocation callback handed to the codec: `realloc` up to [`MAX_GUNZIP_BYTES`]; past
/// the cap it frees the block and returns NULL (the codec then reports `MEMORY_ALLOCATION` and
/// stops; it never frees the block itself).
unsafe extern "C" fn capped_realloc(p: *mut c_void, newsize: usize) -> *mut c_void {
  if newsize > MAX_GUNZIP_BYTES {
    // SAFETY: `p` is the block the codec was handed (malloc'ed by us or by an earlier call here).
    unsafe { free(p) };
    return ptr::null_mut();
  }
  // SAFETY: as above; realloc may move or free it, which the caller expects.
  unsafe { realloc(p, newsize) }
}

/// The two gzip magic bytes (`1f 8b`).
pub fn is_gzip(bytes: &[u8]) -> bool {
  bytes.starts_with(&[0x1f, 0x8b])
}

/// The LZW `compress` magic (`1f 9d`, a `.Z` file): cfitsio reads those from disk only, so the
/// memory path decodes them itself ([`crate::unlzw`]).
pub fn is_unix_compress(bytes: &[u8]) -> bool {
  bytes.starts_with(&[0x1f, 0x9d])
}

/// The bzip2 magic (`BZh`, a `.fits.bz2`): inflated by [`crate::bunzip2`].
pub fn is_bzip2(bytes: &[u8]) -> bool {
  bytes.starts_with(b"BZh")
}

/// Inflates a gzip stream (`.fits.gz`) into a `Vec`; at most [`MAX_GUNZIP_BYTES`]. Takes the
/// global session lock (the codec itself is reentrant zlib, but the error-message stack is not):
/// never call it with a [`crate::MemFits`] open on this thread.
pub fn gunzip(bytes: &[u8]) -> Result<Vec<u8>, FitsError> {
  let _guard = SESSION.lock();
  // SAFETY: the stack is global; we own the lock now.
  unsafe { sys::ffcmsg() };
  // the codec grows the buffer by doubling; start at a plausible FITS size
  let initial = bytes.len().saturating_mul(4).clamp(2880, 1 << 24);
  // SAFETY: plain C allocation, freed below on every path.
  let mut buf = unsafe { malloc(initial) }.cast::<c_char>();
  if buf.is_null() {
    return Err(FitsError::wrapper(sys::MEMORY_ALLOCATION as c_int, "out of memory"));
  }
  let mut size = initial;
  let mut filesize: usize = 0;
  let mut status: c_int = 0;
  // SAFETY: the input is read only (the API takes `char *` by convention), `buf` holds `size`
  // bytes and the callback keeps the pair consistent.
  unsafe {
    sys::uncompress2mem_from_mem(
      bytes.as_ptr().cast_mut().cast::<c_char>(),
      bytes.len(),
      &mut buf,
      &mut size,
      Some(capped_realloc),
      &mut filesize,
      &mut status,
    );
  }
  if status != 0 {
    let e = FitsError::from_status(status);
    if !buf.is_null() {
      // SAFETY: still our block (the codec never frees it; the callback nulls it when it does).
      unsafe { free(buf.cast::<c_void>()) };
    }
    return Err(e);
  }
  let n = filesize.min(size);
  // SAFETY: the codec wrote `filesize` bytes into a buffer of `size` bytes.
  let out = unsafe { core::slice::from_raw_parts(buf.cast::<u8>(), n) }.to_vec();
  // SAFETY: our block.
  unsafe { free(buf.cast::<c_void>()) };
  Ok(out)
}

/// Deflates `bytes` into a gzip stream (`compress2mem_from_mem`): the `.fits.gz` fixtures of the
/// read-path tests. Same lock rule as [`gunzip`].
#[cfg(any(test, feature = "test-writer"))]
pub fn gzip(bytes: &[u8]) -> Result<Vec<u8>, FitsError> {
  let _guard = SESSION.lock();
  // SAFETY: as in `gunzip`.
  unsafe { sys::ffcmsg() };
  let initial = bytes.len().clamp(2880, 1 << 24);
  // SAFETY: as in `gunzip`.
  let mut buf = unsafe { malloc(initial) }.cast::<c_char>();
  if buf.is_null() {
    return Err(FitsError::wrapper(sys::MEMORY_ALLOCATION as c_int, "out of memory"));
  }
  let mut size = initial;
  let mut filesize: usize = 0;
  let mut status: c_int = 0;
  // SAFETY: as in `gunzip`.
  unsafe {
    sys::compress2mem_from_mem(
      bytes.as_ptr().cast_mut().cast::<c_char>(),
      bytes.len(),
      &mut buf,
      &mut size,
      Some(capped_realloc),
      &mut filesize,
      &mut status,
    );
  }
  if status != 0 {
    let e = FitsError::from_status(status);
    if !buf.is_null() {
      // SAFETY: as in `gunzip`.
      unsafe { free(buf.cast::<c_void>()) };
    }
    return Err(e);
  }
  let n = filesize.min(size);
  // SAFETY: as in `gunzip`.
  let out = unsafe { core::slice::from_raw_parts(buf.cast::<u8>(), n) }.to_vec();
  // SAFETY: our block.
  unsafe { free(buf.cast::<c_void>()) };
  Ok(out)
}
