//! Whole-file gzip through cfitsio's own in-memory codec (`uncompress2mem_from_mem` /
//! `compress2mem_from_mem` in `zcompress.c`, over the zlib the library is built with): a
//! `.fits.gz` is inflated into a Rust-owned `Vec` first, then opened like any other buffer by
//! [`crate::MemFits::open`]. Nothing else in cfitsio's memory driver reads gzip (`ffomem` wants
//! the plain bytes).

use alloc::vec::Vec;
use core::{
  ffi::{c_char, c_int, c_void},
  ptr,
  sync::atomic::{AtomicUsize, Ordering},
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

/// Deflate's largest expansion: one stored bit pattern per 1032 output bytes, so a valid stream
/// of `n` bytes (gzip members included) never inflates past `n · 1032` plus a few blocks.
const DEFLATE_MAX_RATIO: usize = 1032;

/// The output cap of the codec call in progress ([`codec_cap`]); set and read only under
/// [`SESSION`], which both codec calls hold.
static CAP: AtomicUsize = AtomicUsize::new(MAX_GUNZIP_BYTES);
/// The real size of the block the codec is filling: the callback grows it geometrically and
/// serves cfitsio's linear `BUFFINCR` requests (28 800 bytes each) from it. Under [`SESSION`].
static CAPACITY: AtomicUsize = AtomicUsize::new(0);

/// The output cap for an input of `len` bytes: [`MAX_GUNZIP_BYTES`], or less when no valid
/// deflate stream of that length can produce more. A truncated stream makes cfitsio's inflate loop
/// ask for `BUFFINCR` more bytes on every `Z_BUF_ERROR` until the callback refuses; with the cap
/// tied to the input that happens after a few requests instead of ~10⁵ (each a copy on Windows,
/// whose `realloc` cannot grow in place: minutes of quadratic copying before the 3 GiB cap).
fn codec_cap(len: usize) -> usize {
  len
    .saturating_mul(DEFLATE_MAX_RATIO)
    .saturating_add(1 << 16)
    .min(MAX_GUNZIP_BYTES)
}

/// The reallocation callback handed to the codec: grows the block to at least `newsize`, up to
/// the call's cap ([`CAP`]), doubling its real size ([`CAPACITY`]) so the codec's linear requests
/// cost amortised O(1) copies; past the cap it frees the block and returns NULL (the codec then
/// reports `MEMORY_ALLOCATION` and stops; it never frees the block itself).
unsafe extern "C" fn capped_realloc(p: *mut c_void, newsize: usize) -> *mut c_void {
  let cap = CAP.load(Ordering::Relaxed);
  if newsize > cap {
    // SAFETY: `p` is the block the codec was handed (malloc'ed by us or by an earlier call here).
    unsafe { free(p) };
    CAPACITY.store(0, Ordering::Relaxed);
    return ptr::null_mut();
  }
  let capacity = CAPACITY.load(Ordering::Relaxed);
  if newsize <= capacity && !p.is_null() {
    return p;
  }
  let grown = newsize.max(capacity.saturating_mul(2)).min(cap);
  // SAFETY: as above; realloc may move or free it, which the caller expects.
  let q = unsafe { realloc(p, grown) };
  if q.is_null() {
    // out of memory: the old block is still ours and the codec drops its pointer to it
    // SAFETY: `realloc` failed, so `p` is untouched and still allocated.
    unsafe { free(p) };
    CAPACITY.store(0, Ordering::Relaxed);
  } else {
    CAPACITY.store(grown, Ordering::Relaxed);
  }
  q
}

/// Allocates the codec's first block of `size` bytes and arms [`capped_realloc`] with the output
/// cap `cap`. Call under [`SESSION`].
fn codec_start(size: usize, cap: usize) -> *mut c_char {
  CAP.store(cap.max(size), Ordering::Relaxed);
  // SAFETY: plain C allocation, freed by the caller on every path.
  let p = unsafe { malloc(size) }.cast::<c_char>();
  CAPACITY.store(if p.is_null() { 0 } else { size }, Ordering::Relaxed);
  p
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
  let mut buf = codec_start(initial, codec_cap(bytes.len()));
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
  // deflate never grows its input by more than 5 bytes per 16 KiB stored block plus the framing
  let cap = bytes.len().saturating_add(bytes.len() / 1000).saturating_add(1 << 16).min(MAX_GUNZIP_BYTES);
  let mut buf = codec_start(initial, cap);
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
