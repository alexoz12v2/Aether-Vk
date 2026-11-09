//! Whole-file bzip2 (`.fits.bz2`) through the libbz2 the library is built with (the cfitsio
//! `bzip2` feature; cfitsio itself only reads bzip2 from a `FILE *`). One call,
//! `BZ2_bzBuffToBuffDecompress`, into a Rust-owned buffer that doubles while the library reports
//! `BZ_OUTBUFF_FULL`, up to [`MAX_GUNZIP_BYTES`].

use alloc::vec::Vec;
use core::ffi::{c_char, c_int, c_uint};

use aethervk_cfitsio_sys as sys;

use crate::{FitsError, MAX_GUNZIP_BYTES};

/// Decodes a bzip2 stream (`BZh` magic). At most [`MAX_GUNZIP_BYTES`] of output (and never more
/// than `u32::MAX` bytes: the libbz2 one-shot API counts in `unsigned int`).
pub fn bunzip2(bytes: &[u8]) -> Result<Vec<u8>, FitsError> {
  if !crate::is_bzip2(bytes) {
    return Err(FitsError::wrapper(sys::DATA_DECOMPRESSION_ERR as c_int, "not a bzip2 stream"));
  }
  let source_len = c_uint::try_from(bytes.len())
    .map_err(|_| FitsError::wrapper(sys::MEMORY_ALLOCATION as c_int, "bzip2 input over 4 GB"))?;
  let cap = MAX_GUNZIP_BYTES.min(c_uint::MAX as usize);
  let mut size = bytes.len().saturating_mul(4).clamp(1 << 16, cap);
  loop {
    let mut out: Vec<u8> = Vec::with_capacity(size);
    let mut dest_len = size as c_uint;
    // SAFETY: `out` has `size` bytes of capacity, which is what `dest_len` tells the library; the
    // source is only read (the API takes `char *` by convention).
    let r = unsafe {
      sys::BZ2_bzBuffToBuffDecompress(
        out.as_mut_ptr().cast::<c_char>(),
        &mut dest_len,
        bytes.as_ptr().cast_mut().cast::<c_char>(),
        source_len,
        0,
        0,
      )
    };
    match r {
      x if x == sys::BZ_OK as c_int => {
        // SAFETY: the library wrote `dest_len` bytes.
        unsafe { out.set_len(dest_len as usize) };
        return Ok(out);
      }
      x if x == sys::BZ_OUTBUFF_FULL as c_int => {
        if size >= cap {
          return Err(FitsError::wrapper(sys::MEMORY_ALLOCATION as c_int, "bzip2 output over the size cap"));
        }
        size = size.saturating_mul(2).min(cap);
      }
      x if x == sys::BZ_MEM_ERROR as c_int => {
        return Err(FitsError::wrapper(sys::MEMORY_ALLOCATION as c_int, "bzip2: out of memory"));
      }
      x if x == sys::BZ_UNEXPECTED_EOF as c_int => {
        return Err(FitsError::wrapper(sys::DATA_DECOMPRESSION_ERR as c_int, "bzip2: truncated stream"));
      }
      _ => {
        return Err(FitsError::wrapper(sys::DATA_DECOMPRESSION_ERR as c_int, "bzip2: corrupt stream"));
      }
    }
  }
}

/// Compresses `bytes` with libbz2 (`BZ2_bzBuffToBuffCompress`, block size 9): the `.fits.bz2`
/// fixtures of the read-path tests.
#[cfg(any(test, feature = "test-writer"))]
pub fn bzip2_compress(bytes: &[u8]) -> Result<Vec<u8>, FitsError> {
  let source_len = c_uint::try_from(bytes.len())
    .map_err(|_| FitsError::wrapper(sys::MEMORY_ALLOCATION as c_int, "bzip2 input over 4 GB"))?;
  // the documented worst case: 1 % more plus 600 bytes
  let size = bytes.len() + bytes.len() / 100 + 600;
  let mut out: Vec<u8> = Vec::with_capacity(size);
  let mut dest_len = size as c_uint;
  // SAFETY: as in `bunzip2`.
  let r = unsafe {
    sys::BZ2_bzBuffToBuffCompress(
      out.as_mut_ptr().cast::<c_char>(),
      &mut dest_len,
      bytes.as_ptr().cast_mut().cast::<c_char>(),
      source_len,
      9,
      0,
      0,
    )
  };
  if r != sys::BZ_OK as c_int {
    return Err(FitsError::wrapper(sys::DATA_COMPRESSION_ERR as c_int, "bzip2 compression failed"));
  }
  // SAFETY: the library wrote `dest_len` bytes.
  unsafe { out.set_len(dest_len as usize) };
  Ok(out)
}
