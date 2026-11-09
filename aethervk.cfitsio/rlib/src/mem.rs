use alloc::{boxed::Box, string::String, vec::Vec};
use core::{
  ffi::{c_char, c_int, c_void},
  marker::PhantomData,
  ptr,
};

use aethervk_cfitsio_sys as sys;

use crate::{
  FitsError, SESSION,
  error::cstr_lossy,
  time::{FitsTime, str2time_locked},
};

/// The kind of the current HDU (`ffghdt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HduType {
  Image,
  AsciiTable,
  BinaryTable,
}

impl HduType {
  fn from_c(t: c_int) -> Self {
    match t as u32 {
      sys::ASCII_TBL => HduType::AsciiTable,
      sys::BINARY_TBL => HduType::BinaryTable,
      _ => HduType::Image,
    }
  }
}

/// `ffgiprll` result: the (uncompressed) image geometry of the current HDU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageParams {
  pub bitpix: i32,
  pub naxes: Vec<i64>,
}

/// The `void **buffptr` / `size_t *buffsize` cells handed to `ffomem`. cfitsio keeps those two
/// pointers and dereferences them on every read (and would update them on a reallocation), so the
/// cells must live at a fixed address for the whole session: boxed, never on the moving struct.
#[repr(C)]
pub(crate) struct MemCell {
  pub buf: *mut c_void,
  pub size: usize,
}

/// A read-only cfitsio session over a byte slice.
///
/// Opened with `ffomem` in `READONLY` mode and no reallocation callback, so cfitsio never writes
/// to, grows, or frees the buffer; `ffclos` on drop goes through the memory driver's "keep" path.
/// Holds the global session lock for its whole life, and the raw pointers make it `!Send`/`!Sync`.
pub struct MemFits<'a> {
  fptr: *mut sys::fitsfile,
  /// What cfitsio received and reads through; the library requires `void**` even in read-only
  /// mode. The pointee is never written (READONLY, no realloc callback).
  cell: Box<MemCell>,
  _guard: spin::MutexGuard<'static, ()>,
  _bytes: PhantomData<&'a [u8]>,
}

impl<'a> MemFits<'a> {
  /// Opens `bytes` as a FITS file. The slice does not have to be a multiple of 2880 bytes; every
  /// read is bounded by its length, so a truncated file fails with a read error instead of
  /// touching memory past the slice.
  pub fn open(bytes: &'a [u8]) -> Result<Self, FitsError> {
    let guard = SESSION.lock();
    // SAFETY: the stack is global; we own the lock now.
    unsafe { sys::ffcmsg() };
    let mut fptr: *mut sys::fitsfile = ptr::null_mut();
    // cfitsio takes `void **`; READONLY + no realloc callback guarantees the buffer is never written
    let mut cell = Box::new(MemCell {
      buf: bytes.as_ptr().cast_mut().cast::<c_void>(),
      size: bytes.len(),
    });
    let mut status: c_int = 0;
    // SAFETY: all pointers are valid for the call; the cell is heap-pinned for the session's life
    // and the bytes outlive it because the returned value borrows `bytes` for 'a.
    unsafe {
      sys::ffomem(
        &mut fptr,
        c"mem://".as_ptr(),
        sys::READONLY as c_int,
        &mut cell.buf,
        &mut cell.size,
        0,
        None,
        &mut status,
      );
    }
    if status != 0 || fptr.is_null() {
      // on failure cfitsio has already released its own structures
      let e = FitsError::from_status(status);
      drop(guard);
      return Err(e);
    }
    Ok(MemFits {
      fptr,
      cell,
      _guard: guard,
      _bytes: PhantomData,
    })
  }

  fn check(&self, status: c_int) -> Result<(), FitsError> {
    if status == 0 {
      Ok(())
    } else {
      Err(FitsError::from_status(status))
    }
  }

  /// Number of HDUs in the file (`ffthdu`).
  pub fn hdu_count(&self) -> Result<u32, FitsError> {
    let mut n: c_int = 0;
    let mut status: c_int = 0;
    // SAFETY: valid fptr (open session) and out-pointers.
    unsafe { sys::ffthdu(self.fptr, &mut n, &mut status) };
    self.check(status)?;
    Ok(n.max(0) as u32)
  }

  /// Moves to the 1-based HDU `hdu` (`ffmahd`) and returns its type.
  pub fn move_abs(&mut self, hdu: u32) -> Result<HduType, FitsError> {
    let mut hdutype: c_int = 0;
    let mut status: c_int = 0;
    let n = c_int::try_from(hdu)
      .map_err(|_| FitsError::wrapper(sys::END_OF_FILE as c_int, "HDU number out of range"))?;
    // SAFETY: valid fptr and out-pointers.
    unsafe { sys::ffmahd(self.fptr, n, &mut hdutype, &mut status) };
    self.check(status)?;
    Ok(HduType::from_c(hdutype))
  }

  /// 1-based number of the current HDU (`ffghdn`).
  pub fn current_hdu(&self) -> u32 {
    let mut n: c_int = 0;
    // SAFETY: valid fptr and out-pointer; ffghdn cannot fail.
    unsafe { sys::ffghdn(self.fptr, &mut n) };
    n.max(0) as u32
  }

  /// Type of the current HDU (`ffghdt`).
  pub fn hdu_type(&self) -> Result<HduType, FitsError> {
    let mut t: c_int = 0;
    let mut status: c_int = 0;
    // SAFETY: valid fptr and out-pointers.
    unsafe { sys::ffghdt(self.fptr, &mut t, &mut status) };
    self.check(status)?;
    Ok(HduType::from_c(t))
  }

  /// Whether the current HDU is a tile-compressed image (`fits_is_compressed_image`); such an HDU
  /// reports itself as an image through [`Self::image_params`] and reads decompressed pixels.
  pub fn is_compressed_image(&self) -> bool {
    let mut status: c_int = 0;
    // SAFETY: valid fptr.
    let r = unsafe { sys::fits_is_compressed_image(self.fptr, &mut status) };
    status == 0 && r != 0
  }

  /// BITPIX and NAXISn of the current image HDU (`ffgiprll`; the uncompressed geometry for `.fz`).
  /// `bitpix` is the stored BITPIX (ZBITPIX for a compressed HDU; an unsigned image is BITPIX 16
  /// / 32 / 64 with its BZERO card), and `naxes` keeps every axis, so a cube has
  /// `naxes.len() > 2` and a random-groups primary has `naxes[0] == 0`.
  pub fn image_params(&self) -> Result<ImageParams, FitsError> {
    const MAXDIM: usize = 99;
    let mut bitpix: c_int = 0;
    let mut naxis: c_int = 0;
    let mut naxes = [0 as sys::LONGLONG; MAXDIM];
    let mut status: c_int = 0;
    // SAFETY: `naxes` has MAXDIM slots, which is what we tell the library.
    unsafe {
      sys::ffgiprll(
        self.fptr,
        MAXDIM as c_int,
        &mut bitpix,
        &mut naxis,
        naxes.as_mut_ptr(),
        &mut status,
      )
    };
    self.check(status)?;
    let n = (naxis.max(0) as usize).min(MAXDIM);
    // LONGLONG is c_longlong, i64 on every target
    let naxes: Vec<i64> = naxes[..n].to_vec();
    Ok(ImageParams { bitpix, naxes })
  }

  /// `ffpnul` (`fits_set_imgnull`): the integer value the pixel reads of the current image HDU
  /// treat as undefined, overriding the header's `BLANK` for this session (the header is not
  /// touched: the file is read-only). For a `BLANK` card cfitsio cannot read as an integer
  /// (`BLANK = -32768.0`, as CDS HiPS tiles write it, fails the pixel read with "Error reading
  /// BLANK keyword value as an integer") the caller parses the card and sets the value here.
  pub fn set_image_null(&mut self, blank: i64) -> Result<(), FitsError> {
    let mut status: c_int = 0;
    // SAFETY: valid fptr; ffpnul only records the value in the HDU structure.
    unsafe { sys::ffpnul(self.fptr, blank as sys::LONGLONG, &mut status) };
    self.check(status)
  }

  /// Reads the first `nelem` pixels of the current image HDU as `f32` (`ffgpve`, group 1, element
  /// 1), scaled by BZERO/BSCALE; undefined pixels (BLANK, or NaN in float data) come out as NaN.
  /// `out` is cleared and refilled. Returns whether any pixel was undefined (`anynul`).
  pub fn read_image_f32(&self, nelem: usize, out: &mut Vec<f32>) -> Result<bool, FitsError> {
    self.read_image_f32_from(0, nelem, out)
  }

  /// [`Self::read_image_f32`] starting at the 0-based pixel `first` of the data array (the first
  /// axis fastest, then the second, then every further axis: plane `p` of an `nx × ny` cube starts
  /// at `p · nx · ny`). Works on tile-compressed images too (cfitsio maps the range onto tiles).
  pub fn read_image_f32_from(&self, first: u64, nelem: usize, out: &mut Vec<f32>) -> Result<bool, FitsError> {
    out.clear();
    if nelem == 0 {
      return Ok(false);
    }
    out.resize(nelem, 0.0);
    let n = sys::LONGLONG::try_from(nelem)
      .map_err(|_| FitsError::wrapper(sys::NUM_OVERFLOW as c_int, "pixel count overflow"))?;
    let first = first
      .checked_add(1)
      .and_then(|f| sys::LONGLONG::try_from(f).ok())
      .ok_or_else(|| FitsError::wrapper(sys::NUM_OVERFLOW as c_int, "pixel offset overflow"))?;
    let mut anynul: c_int = 0;
    let mut status: c_int = 0;
    // SAFETY: `out` has `nelem` initialised slots; cfitsio writes at most that many.
    unsafe {
      sys::ffgpve(
        self.fptr,
        1,
        first,
        n,
        f32::NAN,
        out.as_mut_ptr(),
        &mut anynul,
        &mut status,
      )
    };
    if status != 0 {
      out.clear();
      return Err(FitsError::from_status(status));
    }
    Ok(anynul != 0)
  }

  /// Byte offsets of the current HDU in the file (`ffghadll`): the start of its header, the start
  /// of its data unit and the end of its data unit (padding included); every one a multiple of
  /// 2880. The data start tells how much of a file an uncompressed pixel range needs.
  pub fn hdu_address(&self) -> Result<(u64, u64, u64), FitsError> {
    let (mut head, mut data, mut end): (sys::LONGLONG, sys::LONGLONG, sys::LONGLONG) = (0, 0, 0);
    let mut status: c_int = 0;
    // SAFETY: valid fptr and out-pointers.
    unsafe { sys::ffghadll(self.fptr, &mut head, &mut data, &mut end, &mut status) };
    self.check(status)?;
    Ok((head.max(0) as u64, data.max(0) as u64, end.max(0) as u64))
  }

  /// Every 80-column card of the current HDU before `END`, in order, blank-padded
  /// (`ffcnvthdr2str` + `fffree`). For a tile-compressed HDU this is the header of the image it
  /// stands for (ZBITPIX -> BITPIX, ZNAXISn -> NAXISn, the BINTABLE keywords removed); use
  /// [`Self::raw_cards`] for the stored cards.
  pub fn cards(&self) -> Result<Vec<[u8; 80]>, FitsError> {
    let mut header: *mut c_char = ptr::null_mut();
    let mut nkeys: c_int = 0;
    let mut status: c_int = 0;
    // SAFETY: valid fptr; no exclusion list; cfitsio allocates `header` (nkeys * 80 + 1 bytes).
    unsafe {
      sys::ffcnvthdr2str(
        self.fptr,
        0,
        ptr::null_mut(),
        0,
        &mut header,
        &mut nkeys,
        &mut status,
      )
    };
    if status != 0 {
      let e = FitsError::from_status(status);
      if !header.is_null() {
        let mut st2: c_int = 0;
        // SAFETY: cfitsio allocated it for us.
        unsafe { sys::fffree(header.cast::<c_void>(), &mut st2) };
      }
      return Err(e);
    }
    let n = nkeys.max(0) as usize;
    // SAFETY: the string holds exactly nkeys records of 80 bytes (the last one is END).
    let bytes = unsafe { core::slice::from_raw_parts(header.cast::<u8>(), n * 80) };
    let mut out: Vec<[u8; 80]> = bytes.as_chunks::<80>().0.to_vec();
    if out.last().is_some_and(|c| c.starts_with(b"END     ")) {
      out.pop();
    }
    let mut st2: c_int = 0;
    // SAFETY: cfitsio allocated `header` for us to free with fffree.
    unsafe { sys::fffree(header.cast::<c_void>(), &mut st2) };
    Ok(out)
  }

  /// The cards as stored in the file (`ffghsp` + `ffgrec`), before `END`, blank-padded: for a
  /// tile-compressed HDU the BINTABLE header with the `Z` keywords.
  pub fn raw_cards(&self) -> Result<Vec<[u8; 80]>, FitsError> {
    let mut nexist: c_int = 0;
    let mut nmore: c_int = 0;
    let mut status: c_int = 0;
    // SAFETY: valid fptr and out-pointers.
    unsafe { sys::ffghsp(self.fptr, &mut nexist, &mut nmore, &mut status) };
    self.check(status)?;
    let mut card = [0 as c_char; sys::FLEN_CARD as usize];
    // record 0 resets the header pointer to the first card
    // SAFETY: card has FLEN_CARD bytes, the size ffgrec writes at most.
    unsafe { sys::ffgrec(self.fptr, 0, card.as_mut_ptr(), &mut status) };
    self.check(status)?;
    let mut out = Vec::with_capacity(nexist.max(0) as usize);
    for i in 1..=nexist.max(0) {
      card = [0 as c_char; sys::FLEN_CARD as usize];
      // SAFETY: as above.
      unsafe { sys::ffgrec(self.fptr, i, card.as_mut_ptr(), &mut status) };
      self.check(status)?;
      let mut padded = [b' '; 80];
      for (dst, &src) in padded.iter_mut().zip(card.iter()) {
        if src == 0 {
          break;
        }
        *dst = src as u8;
      }
      out.push(padded);
    }
    Ok(out)
  }

  /// A string keyword with its CONTINUE cards joined (`ffgkls` + `fffree`); `None` when absent.
  pub fn long_string(&self, key: &str) -> Result<Option<String>, FitsError> {
    let ckey = key_cstring(key)?;
    let mut value: *mut c_char = ptr::null_mut();
    let mut status: c_int = 0;
    // SAFETY: `ckey` is NUL-terminated; the comment pointer may be NULL.
    unsafe {
      sys::ffgkls(
        self.fptr,
        ckey.as_ptr().cast::<c_char>(),
        &mut value,
        ptr::null_mut(),
        &mut status,
      )
    };
    if status == sys::KEY_NO_EXIST as c_int {
      // SAFETY: the stack is global; we own the lock.
      unsafe { sys::ffcmsg() };
      return Ok(None);
    }
    self.check(status)?;
    let s = cstr_lossy(value);
    let mut st2: c_int = 0;
    // SAFETY: `value` was allocated by cfitsio for us to free with fffree.
    unsafe { sys::fffree(value.cast::<c_void>(), &mut st2) };
    Ok(Some(s))
  }

  /// A keyword as `f64` (`ffgkyd`, which also converts integers and `D` exponents); `None` when
  /// the keyword does not exist.
  pub fn key_f64(&self, key: &str) -> Result<Option<f64>, FitsError> {
    let ckey = key_cstring(key)?;
    let mut v: f64 = 0.0;
    let mut status: c_int = 0;
    // SAFETY: `ckey` is NUL-terminated; the comment pointer may be NULL.
    unsafe {
      sys::ffgkyd(
        self.fptr,
        ckey.as_ptr().cast::<c_char>(),
        &mut v,
        ptr::null_mut(),
        &mut status,
      )
    };
    if status == sys::KEY_NO_EXIST as c_int {
      // SAFETY: as above.
      unsafe { sys::ffcmsg() };
      return Ok(None);
    }
    self.check(status)?;
    Ok(Some(v))
  }

  /// [`crate::str2time`] for use while this session (and therefore the lock) is alive.
  pub fn str2time(&self, canonical: &str) -> Result<FitsTime, FitsError> {
    str2time_locked(canonical)
  }

  /// The buffer length cfitsio sees (what it treats as the file size).
  pub fn len(&self) -> usize {
    self.cell.size
  }

  pub fn is_empty(&self) -> bool {
    self.cell.size == 0
  }

  /// The buffer pointer cfitsio currently holds (tests: it must still be our slice at close).
  #[cfg(test)]
  pub(crate) fn buffer_ptr(&self) -> *const c_void {
    self.cell.buf
  }
}

impl core::fmt::Debug for MemFits<'_> {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("MemFits")
      .field("len", &self.cell.size)
      .field("hdu", &self.current_hdu())
      .finish()
  }
}

impl Drop for MemFits<'_> {
  fn drop(&mut self) {
    let mut status: c_int = 0;
    // SAFETY: fptr is open; a READONLY memory file is closed without touching the buffer.
    unsafe {
      sys::ffclos(self.fptr, &mut status);
      sys::ffcmsg();
    }
  }
}

/// A keyword name as a NUL-terminated buffer (`FLEN_KEYWORD` bound, no interior NUL).
pub(crate) fn key_cstring(key: &str) -> Result<Vec<u8>, FitsError> {
  if key.len() >= sys::FLEN_KEYWORD as usize || key.bytes().any(|b| b == 0) {
    return Err(FitsError::wrapper(
      sys::KEY_NO_EXIST as c_int,
      "invalid keyword name",
    ));
  }
  let mut v = Vec::with_capacity(key.len() + 1);
  v.extend_from_slice(key.as_bytes());
  v.push(0);
  Ok(v)
}
