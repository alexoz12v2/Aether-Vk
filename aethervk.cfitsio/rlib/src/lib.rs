//! Safe, `no_std` wrapper over CFITSIO for read-only, in-memory FITS sessions.
//!
//! * [`MemFits`] opens a byte slice (typically a read-only file mapping) through `ffomem` in
//!   `READONLY` mode with no reallocation callback: cfitsio can neither grow nor free the buffer,
//!   and the borrow keeps the mapping alive for as long as the session exists.
//! * The library is built without thread support (`fits_is_reentrant() == 0`), so every cfitsio
//!   call runs under one global session lock: a [`MemFits`] holds it from `open` to drop, and
//!   [`str2time`] takes it for the duration of the call. A session therefore lives on one thread,
//!   and `str2time` must be called when no session is open on the calling thread (a session's own
//!   [`MemFits::str2time`] exists for that case); otherwise the spin lock never returns.
//! * Every failing call becomes a [`FitsError`] carrying the `ffgerr` text and the drained
//!   `ffgmsg` stack; nothing panics across the FFI boundary.
//!
//! Tile-compressed images (`.fz`, a BINTABLE HDU) are read transparently. Whole-file compression
//! is not understood by the memory driver: inflate first with [`gunzip`] (`.fits.gz`, cfitsio's
//! own zlib codec), [`bunzip2`] (`.fits.bz2`, the libbz2 cfitsio is built with) or [`unlzw`]
//! (`.Z`, Unix compress, pure Rust) — each capped at [`MAX_GUNZIP_BYTES`] — and open the result.
#![cfg_attr(all(not(test), not(feature = "std")), no_std)]
#![deny(unsafe_op_in_unsafe_fn)]

extern crate alloc;

mod bzip2;
mod error;
mod gzip;
mod lzw;
mod mem;
#[cfg(test)]
mod tests;
mod time;
#[cfg(any(test, feature = "test-writer"))]
pub mod writer;

pub use error::FitsError;
#[cfg(any(test, feature = "test-writer"))]
pub use gzip::gzip;
pub use bzip2::bunzip2;
#[cfg(any(test, feature = "test-writer"))]
pub use bzip2::bzip2_compress;
pub use gzip::{MAX_GUNZIP_BYTES, gunzip, is_bzip2, is_gzip, is_unix_compress};
pub use lzw::unlzw;
pub use mem::{HduType, ImageParams, MemFits};
pub use time::{FitsTime, str2time};
#[cfg(any(test, feature = "test-writer"))]
pub use writer::MemFitsWriter;

/// Serialises every cfitsio call (the library is not reentrant). Public only to the crate.
pub(crate) static SESSION: spin::Mutex<()> = spin::Mutex::new(());

/// `ffvers()`: the library's version as cfitsio reports it (`4.07` for release 4.7.0).
pub fn library_version() -> f32 {
  let mut v: f32 = 0.0;
  // SAFETY: ffvers only writes the float it is given and reads a compile-time constant.
  unsafe { aethervk_cfitsio_sys::ffvers(&mut v) }
}
