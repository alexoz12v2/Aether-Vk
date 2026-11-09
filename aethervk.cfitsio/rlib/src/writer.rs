//! Test-only FITS writer over a growable memory file (`ffimem` with the C allocator as the
//! reallocation callback): builds the small fixtures (image extensions, every tile compression,
//! scaled / unsigned integers, long strings, tables, random groups) that the read path is tested
//! against. Compiled only for this crate's tests and under the `test-writer` feature
//! (dependants' test builds); never part of a library build.

use core::{
  ffi::{c_char, c_int, c_long, c_void},
  ptr,
};

use aethervk_cfitsio_sys as sys;

use alloc::{boxed::Box, vec::Vec};

use crate::{
  FitsError, SESSION,
  gzip::{free, malloc, realloc},
  mem::{MemCell, key_cstring},
};

const INITIAL: usize = 2880;

/// Tile compression algorithms, for [`MemFitsWriter::set_compression`].
pub const RICE_1: u32 = sys::RICE_1;
pub const GZIP_1: u32 = sys::GZIP_1;
pub const GZIP_2: u32 = sys::GZIP_2;
pub const PLIO_1: u32 = sys::PLIO_1;
pub const HCOMPRESS_1: u32 = sys::HCOMPRESS_1;
/// Tiles stored uncompressed (`ZCMPTYPE = 'NOCOMPRESS'`); `set_compression` takes it as a
/// negative `i32` cast to `u32`.
pub const NOCOMPRESS: u32 = sys::NOCOMPRESS as u32;
/// Quantisation methods for floating-point tiles, for [`MemFitsWriter::set_quantize_method`].
pub const NO_DITHER: i32 = sys::NO_DITHER;
pub const SUBTRACTIVE_DITHER_1: i32 = sys::SUBTRACTIVE_DITHER_1 as i32;
pub const SUBTRACTIVE_DITHER_2: i32 = sys::SUBTRACTIVE_DITHER_2 as i32;

/// A pixel type `ffppr` / `ffpcl` accept, with its cfitsio datatype code.
pub trait Pixel: Copy {
  const DATATYPE: u32;
}
impl Pixel for u8 {
  const DATATYPE: u32 = sys::TBYTE;
}
impl Pixel for i8 {
  const DATATYPE: u32 = sys::TSBYTE;
}
impl Pixel for i16 {
  const DATATYPE: u32 = sys::TSHORT;
}
impl Pixel for u16 {
  const DATATYPE: u32 = sys::TUSHORT;
}
impl Pixel for i32 {
  const DATATYPE: u32 = sys::TINT;
}
impl Pixel for u32 {
  const DATATYPE: u32 = sys::TUINT;
}
impl Pixel for i64 {
  const DATATYPE: u32 = sys::TLONGLONG;
}
impl Pixel for u64 {
  const DATATYPE: u32 = sys::TULONGLONG;
}
impl Pixel for f32 {
  const DATATYPE: u32 = sys::TFLOAT;
}
impl Pixel for f64 {
  const DATATYPE: u32 = sys::TDOUBLE;
}

/// A NUL-terminated copy of `s` (interior NULs refused as a wrapper error).
fn cstring(s: &str) -> Result<Vec<u8>, FitsError> {
  if s.bytes().any(|b| b == 0) {
    return Err(FitsError::wrapper(sys::WRITE_ERROR as c_int, "interior NUL in a string"));
  }
  let mut v = Vec::with_capacity(s.len() + 1);
  v.extend_from_slice(s.as_bytes());
  v.push(0);
  Ok(v)
}

pub struct MemFitsWriter {
  fptr: *mut sys::fitsfile,
  /// Updated by cfitsio through the pointers it keeps whenever `realloc` moves the buffer.
  cell: Box<MemCell>,
  _guard: spin::MutexGuard<'static, ()>,
}

impl MemFitsWriter {
  pub fn new() -> Result<Self, FitsError> {
    let guard = SESSION.lock();
    unsafe { sys::ffcmsg() };
    let buf = unsafe { malloc(INITIAL) };
    assert!(!buf.is_null());
    let mut cell = Box::new(MemCell { buf, size: INITIAL });
    let mut fptr: *mut sys::fitsfile = ptr::null_mut();
    let mut status: c_int = 0;
    unsafe {
      sys::ffimem(
        &mut fptr,
        &mut cell.buf,
        &mut cell.size,
        INITIAL,
        Some(realloc),
        &mut status,
      )
    };
    if status != 0 {
      let e = FitsError::from_status(status);
      unsafe { free(cell.buf) };
      drop(guard);
      return Err(e);
    }
    Ok(MemFitsWriter {
      fptr,
      cell,
      _guard: guard,
    })
  }

  fn check(&self, status: c_int) -> Result<(), FitsError> {
    if status == 0 {
      Ok(())
    } else {
      Err(FitsError::from_status(status))
    }
  }

  /// Tile compression for the images created afterwards (`fits_set_compression_type` +
  /// `fits_set_tile_dim`); `tile` is per axis, 0 meaning the full axis (an empty `tile` keeps
  /// cfitsio's default: one row per tile).
  pub fn set_compression(&mut self, ctype: u32, tile: &[i64]) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::fits_set_compression_type(self.fptr, ctype as c_int, &mut status) };
    self.check(status)?;
    if tile.is_empty() {
      return Ok(());
    }
    let dims: Vec<c_long> = tile.iter().map(|&t| t as c_long).collect();
    unsafe {
      sys::fits_set_tile_dim(
        self.fptr,
        dims.len() as c_int,
        dims.as_ptr().cast_mut(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `fits_set_hcomp_scale`: 0 is lossless HCOMPRESS, > 0 the lossy scale factor.
  pub fn set_hcomp_scale(&mut self, scale: f32) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::fits_set_hcomp_scale(self.fptr, scale, &mut status) };
    self.check(status)
  }

  /// `fits_set_hcomp_smooth`: smoothing on decompression of a lossy HCOMPRESS image
  /// (`ZNAME = 'SMOOTH'`, `ZVAL = 1`).
  pub fn set_hcomp_smooth(&mut self, smooth: bool) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::fits_set_hcomp_smooth(self.fptr, smooth as c_int, &mut status) };
    self.check(status)
  }

  /// `fits_set_quantize_level`: the quantisation level `q` of floating-point tiles (`ZQUANTIZ`
  /// scale = noise / q); 0 means no quantisation (lossless GZIP of the floats), < 0 an absolute
  /// scale of `-q`.
  pub fn set_quantize_level(&mut self, q: f32) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::fits_set_quantize_level(self.fptr, q, &mut status) };
    self.check(status)
  }

  /// `fits_set_quantize_method`: [`NO_DITHER`], [`SUBTRACTIVE_DITHER_1`] or
  /// [`SUBTRACTIVE_DITHER_2`] (the `ZQUANTIZ` keyword).
  pub fn set_quantize_method(&mut self, method: i32) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::fits_set_quantize_method(self.fptr, method as c_int, &mut status) };
    self.check(status)
  }

  /// `fits_set_dither_seed`: 1 ..= 10000 a fixed seed, 0 the clock, -1 the first tile's checksum
  /// (the `ZDITHER0` keyword).
  pub fn set_dither_seed(&mut self, seed: i32) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::fits_set_dither_seed(self.fptr, seed as c_int, &mut status) };
    self.check(status)
  }

  /// `fits_set_lossy_int`: integer images are converted to floats and quantised too.
  pub fn set_lossy_int(&mut self, lossy: bool) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::fits_set_lossy_int(self.fptr, lossy as c_int, &mut status) };
    self.check(status)
  }

  /// `ffpscl`: the BSCALE / BZERO cfitsio applies (inversely) to the pixels written afterwards,
  /// overriding the header's; `(1, 0)` writes raw values into an HDU whose header carries
  /// scaling keywords.
  pub fn set_scale(&mut self, bscale: f64, bzero: f64) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::ffpscl(self.fptr, bscale, bzero, &mut status) };
    self.check(status)
  }

  /// `ffpnul`: the integer null value the pixel writes map NaN / nulls to (no header change).
  pub fn set_null(&mut self, blank: i64) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::ffpnul(self.fptr, blank as sys::LONGLONG, &mut status) };
    self.check(status)
  }

  /// `ffcrimll`: a primary HDU when the file is empty (`naxes = []` gives `NAXIS = 0`), an IMAGE
  /// extension afterwards.
  pub fn create_image(&mut self, bitpix: i32, naxes: &[i64]) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    let dims: Vec<sys::LONGLONG> = naxes.iter().map(|&n| n as sys::LONGLONG).collect();
    unsafe {
      sys::ffcrimll(
        self.fptr,
        bitpix as c_int,
        dims.len() as c_int,
        dims.as_ptr().cast_mut(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffppr`: writes `data` from pixel `first` (1-based, plane after plane for a cube),
  /// converted to the HDU's BITPIX with the inverse BZERO / BSCALE scaling in force. A
  /// tile-compressed image refuses implicit datatype conversion (status 413): give it the type
  /// of its BITPIX (`u8`, `i16`, `i32`, `i64`, `f32`, `f64`).
  pub fn write_pixels_at<T: Pixel>(&mut self, first: u64, data: &[T]) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe {
      sys::ffppr(
        self.fptr,
        T::DATATYPE as c_int,
        first as sys::LONGLONG,
        data.len() as sys::LONGLONG,
        data.as_ptr().cast_mut().cast::<c_void>(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// [`Self::write_pixels_at`] from the first pixel.
  pub fn write_pixels<T: Pixel>(&mut self, data: &[T]) -> Result<(), FitsError> {
    self.write_pixels_at(1, data)
  }

  /// `ffppre`: floats from the first pixel (see [`Self::write_pixels`]).
  pub fn write_pixels_f32(&mut self, data: &[f32]) -> Result<(), FitsError> {
    self.write_pixels(data)
  }

  /// `ffppri`: 16-bit pixels from the first pixel (see [`Self::write_pixels`]).
  pub fn write_pixels_i16(&mut self, data: &[i16]) -> Result<(), FitsError> {
    self.write_pixels(data)
  }

  /// `ffpkyj`: an integer keyword.
  pub fn write_key_i64(&mut self, key: &str, value: i64, comment: &str) -> Result<(), FitsError> {
    let (k, c) = (key_cstring(key)?, cstring(comment)?);
    let mut status: c_int = 0;
    unsafe {
      sys::ffpkyj(
        self.fptr,
        k.as_ptr().cast::<c_char>(),
        value as sys::LONGLONG,
        c.as_ptr().cast::<c_char>(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffpkyd`: a floating-point keyword with `decimals` digits (negative: `%G` style).
  pub fn write_key_f64(&mut self, key: &str, value: f64, decimals: i32, comment: &str) -> Result<(), FitsError> {
    let (k, c) = (key_cstring(key)?, cstring(comment)?);
    let mut status: c_int = 0;
    unsafe {
      sys::ffpkyd(
        self.fptr,
        k.as_ptr().cast::<c_char>(),
        value,
        decimals as c_int,
        c.as_ptr().cast::<c_char>(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffpkys`: a string keyword (one card; see [`Self::write_long_string`] for CONTINUE).
  pub fn write_key_str(&mut self, key: &str, value: &str, comment: &str) -> Result<(), FitsError> {
    let (k, v, c) = (key_cstring(key)?, cstring(value)?, cstring(comment)?);
    let mut status: c_int = 0;
    unsafe {
      sys::ffpkys(
        self.fptr,
        k.as_ptr().cast::<c_char>(),
        v.as_ptr().cast::<c_char>(),
        c.as_ptr().cast::<c_char>(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffpkyl`: a logical keyword (`T` / `F`).
  pub fn write_key_bool(&mut self, key: &str, value: bool, comment: &str) -> Result<(), FitsError> {
    let (k, c) = (key_cstring(key)?, cstring(comment)?);
    let mut status: c_int = 0;
    unsafe {
      sys::ffpkyl(
        self.fptr,
        k.as_ptr().cast::<c_char>(),
        value as c_int,
        c.as_ptr().cast::<c_char>(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffplsw` + `ffpkls`: the OGIP long-string warning (`LONGSTRN`, once) and a string keyword
  /// split over CONTINUE cards when it does not fit one card.
  pub fn write_long_string(&mut self, key: &str, value: &str, comment: &str) -> Result<(), FitsError> {
    let (k, v, c) = (key_cstring(key)?, cstring(value)?, cstring(comment)?);
    let mut status: c_int = 0;
    unsafe { sys::ffplsw(self.fptr, &mut status) };
    self.check(status)?;
    unsafe {
      sys::ffpkls(
        self.fptr,
        k.as_ptr().cast::<c_char>(),
        v.as_ptr().cast::<c_char>(),
        c.as_ptr().cast::<c_char>(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffpcom`: a COMMENT card.
  pub fn write_comment(&mut self, text: &str) -> Result<(), FitsError> {
    let t = cstring(text)?;
    let mut status: c_int = 0;
    unsafe { sys::ffpcom(self.fptr, t.as_ptr().cast::<c_char>(), &mut status) };
    self.check(status)
  }

  /// `ffphis`: a HISTORY card.
  pub fn write_history(&mut self, text: &str) -> Result<(), FitsError> {
    let t = cstring(text)?;
    let mut status: c_int = 0;
    unsafe { sys::ffphis(self.fptr, t.as_ptr().cast::<c_char>(), &mut status) };
    self.check(status)
  }

  /// `ffprec`: a card written as given (blank-padded to 80 columns by cfitsio), after its own
  /// legality checks (`fftrec`: printable ASCII; a HIERARCH or lowercase keyword passes).
  pub fn write_record(&mut self, card: &str) -> Result<(), FitsError> {
    let t = cstring(card)?;
    let mut status: c_int = 0;
    unsafe { sys::ffprec(self.fptr, t.as_ptr().cast::<c_char>(), &mut status) };
    self.check(status)
  }

  /// `ffdkey`: deletes a keyword of the current header.
  pub fn delete_key(&mut self, key: &str) -> Result<(), FitsError> {
    let k = key_cstring(key)?;
    let mut status: c_int = 0;
    unsafe { sys::ffdkey(self.fptr, k.as_ptr().cast::<c_char>(), &mut status) };
    self.check(status)
  }

  /// `ffcrtb`: a BINTABLE (or ASCII table) extension of `nrows` with `columns` as
  /// `(TTYPEn, TFORMn)`; an empty file gets an empty primary first.
  pub fn create_table(&mut self, binary: bool, nrows: i64, columns: &[(&str, &str)]) -> Result<(), FitsError> {
    let names: Vec<Vec<u8>> = columns.iter().map(|(n, _)| cstring(n)).collect::<Result<_, _>>()?;
    let forms: Vec<Vec<u8>> = columns.iter().map(|(_, f)| cstring(f)).collect::<Result<_, _>>()?;
    let mut ttype: Vec<*mut c_char> = names.iter().map(|n| n.as_ptr().cast_mut().cast::<c_char>()).collect();
    let mut tform: Vec<*mut c_char> = forms.iter().map(|f| f.as_ptr().cast_mut().cast::<c_char>()).collect();
    let mut status: c_int = 0;
    unsafe {
      sys::ffcrtb(
        self.fptr,
        if binary { sys::BINARY_TBL } else { sys::ASCII_TBL } as c_int,
        nrows as sys::LONGLONG,
        columns.len() as c_int,
        ttype.as_mut_ptr(),
        tform.as_mut_ptr(),
        ptr::null_mut(),
        ptr::null(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffpcl`: `data` into column `col` (1-based) from row `first_row`, one element per row for a
  /// scalar column.
  pub fn write_column<T: Pixel>(&mut self, col: u32, first_row: u64, data: &[T]) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe {
      sys::ffpcl(
        self.fptr,
        T::DATATYPE as c_int,
        col as c_int,
        first_row as sys::LONGLONG,
        1,
        data.len() as sys::LONGLONG,
        data.as_ptr().cast_mut().cast::<c_void>(),
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffphprll`: the primary header of a random-groups file (`NAXIS1 = 0`, `GROUPS = T`,
  /// `PCOUNT` parameters and `GCOUNT` groups of a `naxes` array each); must be the first HDU.
  /// The data unit is left to cfitsio's fill on close.
  pub fn create_random_groups(&mut self, bitpix: i32, naxes: &[i64], pcount: i64, gcount: i64) -> Result<(), FitsError> {
    let mut dims: Vec<sys::LONGLONG> = Vec::with_capacity(naxes.len() + 1);
    dims.push(0);
    dims.extend(naxes.iter().map(|&n| n as sys::LONGLONG));
    let mut status: c_int = 0;
    unsafe {
      sys::ffphprll(
        self.fptr,
        1,
        bitpix as c_int,
        dims.len() as c_int,
        dims.as_mut_ptr(),
        pcount as sys::LONGLONG,
        gcount as sys::LONGLONG,
        1,
        &mut status,
      )
    };
    self.check(status)
  }

  /// `ffpcks`: DATASUM/CHECKSUM of the current HDU.
  pub fn write_checksum(&mut self) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    unsafe { sys::ffpcks(self.fptr, &mut status) };
    self.check(status)
  }

  /// Closes the file and returns its bytes (the buffer is exactly the padded file: cfitsio grows a
  /// memory file to the highest written 2880-byte record).
  pub fn finish(self) -> Result<Vec<u8>, FitsError> {
    let mut this = core::mem::ManuallyDrop::new(self);
    let mut status: c_int = 0;
    unsafe { sys::ffclos(this.fptr, &mut status) };
    let result = if status == 0 {
      Ok(())
    } else {
      Err(FitsError::from_status(status))
    };
    let bytes =
      unsafe { core::slice::from_raw_parts(this.cell.buf.cast::<u8>(), this.cell.size).to_vec() };
    unsafe {
      free(this.cell.buf);
      sys::ffcmsg();
      // the cell, then the lock (ManuallyDrop: field by field)
      core::ptr::drop_in_place(&mut this.cell);
      core::ptr::drop_in_place(&mut this._guard);
    }
    result.map(|()| bytes)
  }
}

impl Drop for MemFitsWriter {
  fn drop(&mut self) {
    let mut status: c_int = 0;
    unsafe {
      sys::ffclos(self.fptr, &mut status);
      free(self.cell.buf);
      sys::ffcmsg();
    }
  }
}
