//! Raw FFI to CFITSIO 4.7.0 (plus the libbz2 it is built with), built statically by `build.rs`
//! (vcpkg, zlib + bzip2, non-reentrant).
//!
//! `src/bindings.rs` is bindgen output committed to the tree; regenerate it with
//! `scripts/regen_cfitsio_bindings.sh` after a cfitsio upgrade. Every allowlisted prototype only
//! uses `int`/`long`/`long long`/`double`/`float`/`size_t`/pointers, which `core::ffi` maps per
//! target at compile time, so one file serves every target. `FITSfile` (whose layout differs per
//! OS) is opaque; `fitsfile` is the only struct exposed.
//!
//! Safety contract for callers: the library is **not** reentrant (`fits_is_reentrant() == 0`);
//! all calls must be serialised by one lock, which `aethervk-cfitsio-rlib` provides.
#![no_std]
#![allow(
  non_camel_case_types,
  non_upper_case_globals,
  non_snake_case,
  clippy::all
)]

include!("bindings.rs");

#[cfg(test)]
mod tests;
