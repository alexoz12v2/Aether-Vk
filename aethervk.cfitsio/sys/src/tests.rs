//! Host-side checks of the committed bindings against the library `build.rs` linked.
extern crate std;

#[test]
fn bindings_header_matches_version() {
  let mut v: f32 = 0.0;
  let r = unsafe { crate::ffvers(&mut v) };
  assert_eq!(r, v);
  // CFITSIO_VERSION is the float 4.07 for release 4.7.0 (major.minor with a two-digit minor, so
  // 4.7.0 and 4.0.7 would collide; the headers' CFITSIO_MAJOR/MINOR tell them apart)
  assert!((v - 4.07).abs() < 0.0005, "ffvers() = {v}");
}

/// `fitsfile { int HDUposition; FITSfile *Fptr; }`: the layout tests are off (target-specific), so
/// check the one struct we expose against the C layout on the host.
#[test]
fn fitsfile_struct_has_the_c_layout() {
  use core::mem::{align_of, size_of};
  let ptr = size_of::<usize>();
  let expected = (size_of::<core::ffi::c_int>().max(ptr)) + ptr;
  assert_eq!(size_of::<crate::fitsfile>(), expected);
  assert_eq!(align_of::<crate::fitsfile>(), ptr);
  assert_eq!(core::mem::offset_of!(crate::fitsfile, Fptr), ptr);
}

/// The vcpkg port is built without the `pthreads` feature, so the library is not reentrant and the
/// wrapper's global session lock is mandatory. A future feature change must be noticed here.
#[test]
fn library_is_not_reentrant() {
  assert_eq!(unsafe { crate::fits_is_reentrant() }, 0);
}
