//! Wrapper tests: the committed 4×3 fixture, hand-built headers, the in-memory writer round trip,
//! the buffer-ownership guarantee and the verified `ffs2tm` behaviour of cfitsio 4.7.0.

use std::string::ToString;

use aethervk_cfitsio_sys as sys;

use crate::{FitsError, HduType, MemFits, library_version, str2time, writer::MemFitsWriter};

/// The committed 4×3 fixture's bytes (downloaded on demand), or skip the calling test.
macro_rules! fixture {
  () => {
    std::fs::read(aethervk_test_assets::test_asset!("fits_lt_67p_header_4x3.fits")).expect("fixture")
  };
}

/// A one-HDU header-only FITS file (NAXIS = 0) from the given cards, blank-padded to 2880 bytes.
fn header_only(cards: &[&str]) -> Vec<u8> {
  let mut v = Vec::new();
  for c in [
    "SIMPLE  =                    T",
    "BITPIX  =                    8",
    "NAXIS   =                    0",
  ]
  .iter()
  .chain(cards.iter())
  .chain(["END"].iter())
  {
    let mut card = [b' '; 80];
    card[..c.len()].copy_from_slice(c.as_bytes());
    v.extend_from_slice(&card);
  }
  v.resize(v.len().div_ceil(2880) * 2880, b' ');
  v
}

const RAMP: [f32; 12] = [
  100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1100.0, 1200.0,
];

#[test]
fn fixture_opens_and_reads_through_cfitsio() {
  let bytes = fixture!();
  let mut f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.len(), bytes.len());
  assert!(core::ptr::eq(f.buffer_ptr().cast::<u8>(), bytes.as_ptr()));
  assert_eq!(f.hdu_count().unwrap(), 1);
  assert_eq!(f.current_hdu(), 1);
  assert_eq!(f.move_abs(1).unwrap(), HduType::Image);
  assert_eq!(f.hdu_type().unwrap(), HduType::Image);
  assert!(!f.is_compressed_image());
  let p = f.image_params().unwrap();
  assert_eq!((p.bitpix, p.naxes.as_slice()), (32, &[4i64, 3][..]));
  let mut px = Vec::new();
  let anynul = f.read_image_f32(12, &mut px).unwrap();
  assert!(!anynul);
  assert_eq!(px, RAMP);
  let cards = f.cards().unwrap();
  assert_eq!(cards.len(), 276); // every card before END
  assert!(cards[0].starts_with(b"SIMPLE  ="));
  assert_eq!(
    cards.iter().filter(|c| c.starts_with(b"COMMENT")).count(),
    2
  );
  assert!(cards.iter().all(|c| c.len() == 80));
  // byte-identical to the file for an uncompressed HDU, and to the ffgrec path
  let file_cards: Vec<[u8; 80]> = bytes[..276 * 80].as_chunks::<80>().0.to_vec();
  assert_eq!(cards, file_cards);
  assert_eq!(f.raw_cards().unwrap(), file_cards);
  assert_eq!(f.key_f64("EXPTIME").unwrap(), Some(20.0));
  assert_eq!(f.key_f64("MJD").unwrap(), Some(59573.005195));
  assert_eq!(f.key_f64("NAXIS1").unwrap(), Some(4.0)); // integer card read as double
  assert_eq!(f.key_f64("NOSUCHKY").unwrap(), None);
  assert_eq!(
    f.long_string("ORIGIN").unwrap().as_deref(),
    Some("Liverpool JMU")
  );
  assert_eq!(f.long_string("OBSTYPE").unwrap().as_deref(), Some("EXPOSE")); // trailing blanks trimmed
  assert_eq!(f.long_string("NOSUCHKY").unwrap(), None);
  assert_eq!(
    f.long_string("DATE-OBS").unwrap().as_deref(),
    Some("2021-12-27T00:14:47.345")
  );
}

#[test]
fn moving_past_the_last_hdu_is_an_error_with_text() {
  let bytes = fixture!();
  let mut f = MemFits::open(&bytes).unwrap();
  let e = f.move_abs(5).unwrap_err();
  assert_eq!(e.status, sys::END_OF_FILE as i32);
  assert_eq!(e.text, "tried to move past end of file");
  // ffmahd pushes nothing on the message stack for this one; the text alone must be readable
  assert!(
    e.to_string().starts_with("cfitsio status 107 (tried to move past end of file)"),
    "{e}"
  );
  // the session is still usable and the stack was drained
  assert_eq!(f.move_abs(1).unwrap(), HduType::Image);
  assert_eq!(f.key_f64("EXPTIME").unwrap(), Some(20.0));
}

#[test]
fn garbage_is_an_error_with_text() {
  let e = MemFits::open(b"NOT A FITS FILE AT ALL").unwrap_err();
  assert!(e.status > 0);
  assert!(!e.text.is_empty());
  assert!(!e.to_string().is_empty());
  assert!(MemFits::open(b"").is_err());
  // a lock left behind by a failed open would hang here
  let bytes = fixture!();
  assert!(MemFits::open(&bytes).is_ok());
}

#[test]
fn truncated_data_fails_the_pixel_read_without_touching_memory_past_the_slice() {
  let bytes = fixture!();
  let cut = &bytes[..23040 + 20]; // header complete, data unit cut inside the first row
  let f = MemFits::open(cut).unwrap();
  assert_eq!(f.image_params().unwrap().naxes, vec![4, 3]);
  let mut px = Vec::new();
  let e = f.read_image_f32(12, &mut px).unwrap_err();
  assert!(e.status > 0, "{e}");
  assert!(px.is_empty());
}

#[test]
fn continue_cards_are_joined_by_long_string() {
  let bytes = header_only(&[
    "LONGSTRN= 'OGIP 1.0'           / The OGIP long string convention may be used.",
    "COMMENT   This FITS file may contain long string keyword values that are",
    "ORIGIN  = 'abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz01234&'",
    "CONTINUE  '56789ABCDEFGHIJKLMNOPQRSTUVWXYZ&'",
    "CONTINUE  'tail' / the end",
    "SHORT   = 'plain   '",
  ]);
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 1);
  assert_eq!(
    f.long_string("ORIGIN").unwrap().as_deref(),
    Some(
      "abcdefghijklmnopqrstuvwxyz0123456789abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZtail"
    )
  );
  assert_eq!(f.long_string("SHORT").unwrap().as_deref(), Some("plain"));
  assert_eq!(f.cards().unwrap().len(), 9);
  assert_eq!(f.image_params().unwrap().naxes, Vec::<i64>::new());
  let mut px = Vec::new();
  assert!(!f.read_image_f32(0, &mut px).unwrap());
}

/// cfitsio must never free or reallocate the caller's buffer: 10 000 sessions over one Rust-owned
/// allocation, the pointer cfitsio holds stays ours, and the Vec is freed by Rust afterwards (a
/// `free()` by cfitsio would double-free here).
#[test]
fn ten_thousand_sessions_never_free_our_buffer() {
  let bytes = fixture!();
  let ptr = bytes.as_ptr();
  for i in 0..10_000 {
    let f = MemFits::open(&bytes).unwrap();
    assert!(core::ptr::eq(f.buffer_ptr().cast::<u8>(), ptr));
    assert_eq!(f.len(), bytes.len());
    if i % 1000 == 0 {
      let mut px = Vec::new();
      f.read_image_f32(12, &mut px).unwrap();
      assert_eq!(px[11], 1200.0);
    }
  }
  assert_eq!(bytes[0], b'S');
  drop(bytes);
}

/// The build is not reentrant; four threads opening at once must serialise on the session lock.
#[test]
fn four_threads_serialise_on_the_session_lock() {
  assert_eq!(
    unsafe { sys::fits_is_reentrant() },
    0,
    "the wrapper's global lock assumes a non-reentrant build"
  );
  let bytes = std::sync::Arc::new(fixture!());
  let handles: Vec<_> = (0..4)
    .map(|t| {
      let bytes = bytes.clone();
      std::thread::spawn(move || {
        for _ in 0..250 {
          let mut f = MemFits::open(&bytes).unwrap();
          assert_eq!(f.move_abs(1).unwrap(), HduType::Image);
          let mut px = Vec::new();
          f.read_image_f32(12, &mut px).unwrap();
          assert_eq!(px, RAMP);
          assert_eq!(f.cards().unwrap().len(), 276);
          // a garbage open in between must not disturb the others' error stacks (after this
          // session is dropped: a second open on the same thread would wait on the lock forever)
          drop(f);
          if t == 0 {
            assert!(MemFits::open(b"garbage").is_err());
          }
        }
        t
      })
    })
    .collect();
  let mut done: Vec<usize> = handles.into_iter().map(|h| h.join().unwrap()).collect();
  done.sort_unstable();
  assert_eq!(done, vec![0, 1, 2, 3]);
}

#[test]
fn version_is_4_7_0() {
  // CFITSIO_VERSION is the float 4.07 for 4.7.0 (two-digit minor)
  assert!(
    (library_version() - 4.07).abs() < 0.0005,
    "{}",
    library_version()
  );
}

/// What cfitsio 4.7.0's `ffs2tm` really does with each form (read from putkey.c and asserted here;
/// the Rust normaliser in front of it must produce only the accepted forms).
#[test]
fn str2time_accepted_and_rejected_forms() {
  let ok = |s: &str| str2time(s).unwrap_or_else(|e| panic!("{s:?}: {e}"));
  let bad = |s: &str| {
    let e = str2time(s).expect_err(s);
    assert_eq!(e.status, sys::BAD_DATE as i32, "{s:?}: {e}");
    assert!(e.text.contains("date"), "{s:?}: {e}");
    e
  };
  // dd/mm/yy: the FITS rule, year 1900 + yy (so 2021-12-27 written that way reads as 1921)
  let t = ok("27/12/21");
  assert_eq!(
    (t.year, t.month, t.day, t.hour, t.minute, t.second),
    (1921, 12, 27, 0, 0, 0.0)
  );
  // ISO date only
  let t = ok("2021-12-27");
  assert_eq!(
    (t.year, t.month, t.day, t.hour, t.minute, t.second),
    (2021, 12, 27, 0, 0, 0.0)
  );
  // ISO date-time, with and without a fraction
  let t = ok("2021-12-27T00:14:47.345");
  assert_eq!(
    (t.year, t.month, t.day, t.hour, t.minute),
    (2021, 12, 27, 0, 14)
  );
  assert!((t.second - 47.345).abs() < 1e-9);
  let t = ok("2021-12-27T23:59:07");
  assert_eq!((t.hour, t.minute, t.second), (23, 59, 7.0));
  // a leap second passes (seconds < 61)
  let t = ok("2021-12-31T23:59:60.5");
  assert_eq!((t.minute, t.second), (59, 60.5));
  // time only: date fields come back as 0
  let t = ok("18:30:00");
  assert_eq!(
    (t.year, t.month, t.day, t.hour, t.minute, t.second),
    (0, 0, 0, 18, 30, 0.0)
  );
  let t = ok("18:30:00.25");
  assert_eq!(t.second, 0.25);
  // a trailing zone letter is accepted only after a fraction (atof stops at the 'Z'): a quirk,
  // not a contract; the normaliser strips it anyway
  let t = ok("2021-12-27T00:14:47.345Z");
  assert_eq!((t.hour, t.minute), (0, 14));

  // rejected (status BAD_DATE = 420): four-digit year with slashes
  bad("27/12/2021");
  // a space instead of 'T', a lowercase 't' (ffs2dt: anything but 'T' after the date)
  bad("2021-12-27 00:14:47");
  bad("2021-12-27t00:14:47");
  // 'Z' right after the seconds (no fraction)
  bad("2021-12-27T00:14:47Z");
  // trailing blank, truncated time, missing seconds
  bad("2021-12-27 ");
  bad("2021-12-27T00:14");
  bad("2021-12-27T00:14:");
  // out-of-range fields
  bad("2021-13-01");
  bad("2021-02-30");
  bad("2021-12-27T24:00:00");
  bad("2021-12-27T00:60:00");
  bad("2021-12-27T00:00:61");
  bad("25:00:00");
  // garbage
  bad("");
  bad("yesterday");
  bad("2021/12/27");
  let e = bad("27/12/21 00:14:47");
  assert!(
    !e.messages.is_empty(),
    "the ffpmsg stack explains the rejection: {e}"
  );
  // wrapper-side guards
  let e = str2time("2021-12-27T00:14:47\0").unwrap_err();
  assert_eq!(e.status, sys::BAD_DATE as i32);
  assert!(e.messages.is_empty());
}

#[test]
fn str2time_works_inside_a_session_too() {
  let bytes = fixture!();
  let f = MemFits::open(&bytes).unwrap();
  let date = f.long_string("DATE-OBS").unwrap().unwrap();
  let t = f.str2time(&date).unwrap();
  assert_eq!(
    (t.year, t.month, t.day, t.hour, t.minute),
    (2021, 12, 27, 0, 14)
  );
  // a failed parse inside the session leaves the session usable
  assert!(f.str2time("nope").is_err());
  assert_eq!(f.key_f64("EXPTIME").unwrap(), Some(20.0));
}

// ───────────────────────────── writer round trips ─────────────────────────────

#[test]
fn writer_primary_image_round_trips() {
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(16, &[4, 3]).unwrap();
    w.write_pixels_f32(&RAMP).unwrap();
    w.write_checksum().unwrap();
    w.finish().unwrap()
  };
  assert_eq!(bytes.len() % 2880, 0);
  assert_eq!(bytes.len(), 2880 * 2, "one header block + one data block");
  assert!(bytes.starts_with(b"SIMPLE  ="));
  let mut f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 1);
  assert_eq!(f.move_abs(1).unwrap(), HduType::Image);
  assert!(!f.is_compressed_image());
  let p = f.image_params().unwrap();
  assert_eq!((p.bitpix, p.naxes.as_slice()), (16, &[4i64, 3][..]));
  let mut px = Vec::new();
  assert!(!f.read_image_f32(12, &mut px).unwrap());
  assert_eq!(px, RAMP);
  let cards = f.cards().unwrap();
  assert!(cards.iter().any(|c| c.starts_with(b"CHECKSUM=")));
  assert!(cards.iter().any(|c| c.starts_with(b"DATASUM =")));
}

#[test]
fn writer_image_extension_round_trips() {
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(8, &[]).unwrap(); // empty primary, NAXIS = 0
    w.create_image(16, &[4, 3]).unwrap(); // XTENSION = 'IMAGE'
    w.write_pixels_f32(&RAMP).unwrap();
    w.finish().unwrap()
  };
  let mut f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 2);
  assert_eq!(f.move_abs(1).unwrap(), HduType::Image);
  assert_eq!(f.image_params().unwrap().naxes, Vec::<i64>::new());
  assert_eq!(f.move_abs(2).unwrap(), HduType::Image);
  assert_eq!(f.current_hdu(), 2);
  assert!(!f.is_compressed_image());
  let p = f.image_params().unwrap();
  assert_eq!((p.bitpix, p.naxes.as_slice()), (16, &[4i64, 3][..]));
  let mut px = Vec::new();
  f.read_image_f32(12, &mut px).unwrap();
  assert_eq!(px, RAMP);
  let cards = f.cards().unwrap();
  assert!(cards[0].starts_with(b"XTENSION= 'IMAGE   '"));
  assert!(cards.iter().any(|c| c.starts_with(b"NAXIS1  =                    4")));
}

#[test]
fn writer_rice_compressed_image_round_trips() {
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.set_compression(sys::RICE_1, &[4, 1]).unwrap(); // one tile per row
    w.create_image(16, &[4, 3]).unwrap();
    // "Implicit datatype conversion is not supported when writing to compressed images"
    // (status 413): a BITPIX 16 compressed image takes 16-bit pixels, not floats
    let e = w.write_pixels_f32(&RAMP).unwrap_err();
    assert_eq!(e.status, 413, "{e}");
    let shorts: Vec<i16> = RAMP.iter().map(|&v| v as i16).collect();
    w.write_pixels_i16(&shorts).unwrap();
    w.finish().unwrap()
  };
  // cfitsio writes an empty primary, then the compressed image as a BINTABLE extension
  let mut f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 2);
  assert_eq!(f.move_abs(1).unwrap(), HduType::Image);
  assert!(!f.is_compressed_image());
  // cfitsio presents the compressed BINTABLE as an image HDU (ffmahd/ffghdt say IMAGE_HDU)
  assert_eq!(f.move_abs(2).unwrap(), HduType::Image);
  assert_eq!(f.hdu_type().unwrap(), HduType::Image);
  assert!(f.is_compressed_image());
  // the uncompressed geometry, as the image it stands for
  let p = f.image_params().unwrap();
  assert_eq!((p.bitpix, p.naxes.as_slice()), (16, &[4i64, 3][..]));
  let mut px = Vec::new();
  assert!(!f.read_image_f32(12, &mut px).unwrap());
  assert_eq!(px, RAMP);
  // `cards` is the translated image header (ZBITPIX -> BITPIX, ZNAXIS1 -> NAXIS1, ...)
  let cards = f.cards().unwrap();
  let text: Vec<String> = cards
    .iter()
    .map(|c| String::from_utf8_lossy(c).trim_end().to_string())
    .collect();
  // (ZSIMPLE = T was recorded, so the decompressed header comes back as a primary one)
  assert!(
    text[0].starts_with("SIMPLE  =                    T"),
    "{text:#?}"
  );
  assert!(
    text.iter().any(|c| c.starts_with("BITPIX  =                   16")),
    "{text:#?}"
  );
  assert!(
    text.iter().any(|c| c.starts_with("NAXIS1  =                    4")),
    "{text:#?}"
  );
  assert!(
    text.iter().any(|c| c.starts_with("NAXIS2  =                    3")),
    "{text:#?}"
  );
  assert!(!text.iter().any(|c| c.starts_with('Z')), "{text:#?}");
  assert!(!text.iter().any(|c| c.starts_with("END")), "{text:#?}");
  // `raw_cards` (ffgrec) is the stored BINTABLE header: cfitsio does not translate record reads
  let raw: Vec<String> = f
    .raw_cards()
    .unwrap()
    .iter()
    .map(|c| String::from_utf8_lossy(c).trim_end().to_string())
    .collect();
  assert!(raw[0].starts_with("XTENSION= 'BINTABLE'"), "{raw:#?}");
  assert!(
    raw.iter().any(|c| c.starts_with("ZBITPIX =                   16")),
    "{raw:#?}"
  );
  assert!(
    raw.iter().any(|c| c.starts_with("ZCMPTYPE= 'RICE_1  '")),
    "{raw:#?}"
  );
  assert!(
    raw.iter().any(|c| c.starts_with("BITPIX  =                    8")),
    "{raw:#?}"
  );
}

#[test]
fn writer_float_nan_reads_back_as_nan_with_anynul() {
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(-32, &[2, 2]).unwrap();
    w.write_pixels_f32(&[1.5, f32::NAN, -2.0, 0.0]).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.image_params().unwrap().bitpix, -32);
  let mut px = Vec::new();
  assert!(f.read_image_f32(4, &mut px).unwrap(), "anynul");
  assert_eq!(px[0], 1.5);
  assert!(px[1].is_nan());
  assert_eq!((px[2], px[3]), (-2.0, 0.0));
}

#[test]
fn fits_error_display_lists_the_messages() {
  let e = FitsError {
    status: 107,
    text: "tried to move past end of file".into(),
    messages: vec!["a".into(), "b".into()],
  };
  assert_eq!(
    e.to_string(),
    "cfitsio status 107 (tried to move past end of file): a; b"
  );
  let e = FitsError {
    status: 1,
    text: "x".into(),
    messages: vec![],
  };
  assert_eq!(e.to_string(), "cfitsio status 1 (x)");
}

// ───────────────────────────── whole-file gzip ─────────────────────────────

#[test]
fn gzip_round_trips_and_the_memory_driver_needs_the_inflated_bytes() {
  use crate::{gunzip, gzip, is_bzip2, is_gzip, is_unix_compress};
  let bytes = fixture!();
  let gz = gzip(&bytes).unwrap();
  assert!(is_gzip(&gz));
  assert!(!is_gzip(&bytes) && !is_unix_compress(&bytes) && !is_bzip2(&bytes));
  assert!(gz.len() < bytes.len() / 2, "{} of {}", gz.len(), bytes.len());
  assert_eq!(gunzip(&gz).unwrap(), bytes);
  // ffomem itself does not inflate
  assert!(MemFits::open(&gz).is_err());
  let inflated = gunzip(&gz).unwrap();
  let f = MemFits::open(&inflated).unwrap();
  assert_eq!(f.image_params().unwrap().naxes, vec![4, 3]);
  drop(f);
  // garbage, a truncated stream, the other magics
  let e = gunzip(b"\x1f\x8b garbage").unwrap_err();
  assert!(e.status > 0, "{e}");
  assert!(gunzip(&gz[..gz.len() / 2]).is_err());
  assert!(gunzip(b"").is_err());
  assert!(is_unix_compress(b"\x1f\x9d\x90"));
  assert!(is_bzip2(b"BZh91AY&SY"));
  // an empty input stream inflates to nothing
  assert_eq!(gunzip(&gzip(b"").unwrap()).unwrap(), Vec::<u8>::new());
}

// ───────────────────────────── the writer matrix ─────────────────────────────

use crate::writer::{
  GZIP_1, GZIP_2, HCOMPRESS_1, NOCOMPRESS, NO_DITHER, PLIO_1, Pixel, RICE_1,
  SUBTRACTIVE_DITHER_1, SUBTRACTIVE_DITHER_2,
};

/// `n` deterministic noisy values in roughly `[0, 1000)` (a linear congruential sequence).
fn noisy(n: usize) -> Vec<f32> {
  let mut x: u32 = 12345;
  (0..n)
    .map(|i| {
      x = x.wrapping_mul(1_103_515_245).wrapping_add(12345);
      (i as f32) * 0.5 + ((x >> 8) % 1000) as f32 * 0.37
    })
    .collect()
}

/// A 16 × 8 image of `bitpix`, tile-compressed with `ctype` (HCOMPRESS needs 2-D tiles), as
/// `(bytes, the values written)`.
fn compressed_image(ctype: u32, bitpix: i32, values: &[f32]) -> Result<Vec<u8>, FitsError> {
  fn conv<T: Pixel + 'static>(w: &mut MemFitsWriter, v: &[f32], f: impl Fn(f32) -> T) -> Result<(), FitsError> {
    let data: Vec<T> = v.iter().map(|&x| f(x)).collect();
    w.write_pixels(&data)
  }
  let mut w = MemFitsWriter::new()?;
  w.set_compression(ctype, &[16, 4])?;
  w.create_image(bitpix, &[16, 8])?;
  match bitpix {
    8 => conv(&mut w, values, |x| x as u8)?,
    16 => conv(&mut w, values, |x| x as i16)?,
    32 => conv(&mut w, values, |x| x as i32)?,
    64 => conv(&mut w, values, |x| x as i64)?,
    -32 => w.write_pixels(values)?,
    _ => conv(&mut w, values, |x| x as f64)?,
  }
  w.finish()
}

/// Reads the whole 16 × 8 image of the first real image HDU as f32 and the raw (stored) cards.
fn decode(bytes: &[u8]) -> (Vec<f32>, Vec<String>, bool) {
  let mut f = MemFits::open(bytes).unwrap();
  let n = f.hdu_count().unwrap();
  let mut hdu = 1;
  while f.image_params().unwrap().naxes.len() < 2 && hdu < n {
    hdu += 1;
    f.move_abs(hdu).unwrap();
  }
  let p = f.image_params().unwrap();
  let count: i64 = p.naxes.iter().product();
  let mut px = Vec::new();
  f.read_image_f32(count as usize, &mut px).unwrap();
  let raw: Vec<String> = f
    .raw_cards()
    .unwrap()
    .iter()
    .map(|c| String::from_utf8_lossy(c).trim_end().to_string())
    .collect();
  let compressed = f.is_compressed_image();
  (px, raw, compressed)
}

fn has_card(raw: &[String], prefix: &str) -> bool {
  raw.iter().any(|c| c.starts_with(prefix))
}

/// Every tile compression cfitsio writes, for every BITPIX it allows: integers round-trip
/// exactly (the tiles are lossless), floats within the quantisation; the three 64-bit
/// refusals are asserted as refusals. One lock per file: writer and reader never overlap.
#[test]
fn every_tile_compression_round_trips_for_every_bitpix() {
  // integer ramps 0..=127 stay exact with every algorithm (PLIO needs non-negative values)
  let ints: Vec<f32> = (0..128).map(|i| i as f32).collect();
  let floats = noisy(128);
  // (NOCOMPRESS is read by cfitsio but not written: `fits_set_compression_type(NOCOMPRESS)` is
  // accepted and the image creation then fails with status 410 "unsupported image compression
  // algorithm", asserted below)
  let algorithms: [(u32, &str); 5] = [
    (RICE_1, "RICE_1"),
    (GZIP_1, "GZIP_1"),
    (GZIP_2, "GZIP_2"),
    (HCOMPRESS_1, "HCOMPRESS_1"),
    (PLIO_1, "PLIO_1"),
  ];
  let mut decoded = 0;
  for (ctype, zname) in algorithms {
    for bitpix in [8i32, 16, 32, 64, -32, -64] {
      let values = if bitpix > 0 { &ints } else { &floats };
      let r = compressed_image(ctype, bitpix, values);
      // 64-bit integers: cfitsio 4.7.0 writes no compressed image from TLONGLONG at all
      // ("writing TLONGLONG to compressed image is not supported", status 413)
      let refused = bitpix == 64;
      let bytes = match r {
        Ok(b) => {
          assert!(!refused, "{zname} BITPIX {bitpix}: expected a refusal");
          b
        }
        Err(e) => {
          assert!(refused, "{zname} BITPIX {bitpix}: {e}");
          continue;
        }
      };
      let (px, raw, compressed) = decode(&bytes);
      assert!(compressed, "{zname} BITPIX {bitpix}");
      assert!(has_card(&raw, "XTENSION= 'BINTABLE'"), "{zname} {bitpix}: {raw:#?}");
      assert!(
        has_card(&raw, &format!("ZCMPTYPE= '{zname}")),
        "{zname} {bitpix}: {raw:#?}"
      );
      assert!(
        has_card(&raw, &format!("ZBITPIX = {bitpix:>20}")),
        "{zname} {bitpix}: {raw:#?}"
      );
      assert_eq!(px.len(), 128, "{zname} {bitpix}");
      if bitpix > 0 {
        assert_eq!(px, *values, "{zname} BITPIX {bitpix}");
      } else {
        // floats are quantised (default q = 4: scale = noise / 4) except by the lossless GZIP
        // of NOCOMPRESS / an unquantisable tile; the noise of `noisy` is ~100, so < 30 absolute
        let worst = px.iter().zip(values).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(worst < 30.0, "{zname} BITPIX {bitpix}: worst error {worst}");
        assert!(px.iter().all(|v| v.is_finite()), "{zname} {bitpix}");
      }
      decoded += 1;
    }
  }
  assert_eq!(decoded, 5 * 5);
  let e = compressed_image(NOCOMPRESS, 16, &ints).unwrap_err();
  assert_eq!(e.status, 410, "{e}");
}

/// The three quantisation methods on floats, with a fixed dither seed, at an absolute scale:
/// the values come back within the scale and the `ZQUANTIZ` / `ZDITHER0` keywords say what was
/// used. `NO_DITHER` reads back the same from a GZIP_1 and a RICE_1 container.
#[test]
fn quantised_floats_dither_methods_round_trip_within_the_scale() {
  let values = noisy(128);
  for (method, zquantiz) in [
    (NO_DITHER, "NO_DITHER"),
    (SUBTRACTIVE_DITHER_1, "SUBTRACTIVE_DITHER_1"),
    (SUBTRACTIVE_DITHER_2, "SUBTRACTIVE_DITHER_2"),
  ] {
    for ctype in [RICE_1, GZIP_1, HCOMPRESS_1] {
      let bytes = {
        let mut w = MemFitsWriter::new().unwrap();
        w.set_compression(ctype, &[16, 4]).unwrap();
        w.set_quantize_method(method).unwrap();
        w.set_quantize_level(-0.01).unwrap(); // absolute scale 0.01
        w.set_dither_seed(42).unwrap();
        w.create_image(-32, &[16, 8]).unwrap();
        w.write_pixels(&values).unwrap();
        w.finish().unwrap()
      };
      let (px, raw, _) = decode(&bytes);
      // HCOMPRESS does not take SUBTRACTIVE_DITHER_2: cfitsio warns and uses DITHER_1
      let zquantiz = if ctype == HCOMPRESS_1 && method == SUBTRACTIVE_DITHER_2 { "SUBTRACTIVE_DITHER_1" } else { zquantiz };
      assert!(has_card(&raw, &format!("ZQUANTIZ= '{zquantiz}")), "{method} {ctype}: {raw:#?}");
      if method != NO_DITHER {
        assert!(has_card(&raw, "ZDITHER0=                   42"), "{method} {ctype}: {raw:#?}");
      }
      let worst = px.iter().zip(&values).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
      assert!(worst <= 0.011, "{method} {ctype}: worst error {worst}");
    }
  }
  // SUBTRACTIVE_DITHER_2 keeps exact zeros exact
  let mut with_zeros = values.clone();
  for v in with_zeros.iter_mut().step_by(7) {
    *v = 0.0;
  }
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.set_compression(RICE_1, &[16, 4]).unwrap();
    w.set_quantize_method(SUBTRACTIVE_DITHER_2).unwrap();
    w.set_quantize_level(-0.5).unwrap();
    w.set_dither_seed(7).unwrap();
    w.create_image(-32, &[16, 8]).unwrap();
    w.write_pixels(&with_zeros).unwrap();
    w.finish().unwrap()
  };
  let (px, ..) = decode(&bytes);
  for (i, v) in with_zeros.iter().enumerate() {
    if *v == 0.0 {
      assert_eq!(px[i], 0.0, "pixel {i}");
    }
  }
}

/// Lossy HCOMPRESS with smoothing: the `SMOOTH` parameter is recorded, the image comes back
/// close to the original (a smooth 2-D gradient compresses with a small error at scale 1).
#[test]
fn hcompress_lossy_with_smoothing_decodes_close() {
  let values: Vec<f32> = (0..128).map(|i| ((i % 16) * 8 + (i / 16) * 4) as f32).collect();
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.set_compression(HCOMPRESS_1, &[16, 8]).unwrap();
    w.set_hcomp_scale(1.0).unwrap();
    w.set_hcomp_smooth(true).unwrap();
    w.create_image(16, &[16, 8]).unwrap();
    let shorts: Vec<i16> = values.iter().map(|&v| v as i16).collect();
    w.write_pixels(&shorts).unwrap();
    w.finish().unwrap()
  };
  let (px, raw, compressed) = decode(&bytes);
  assert!(compressed);
  assert!(has_card(&raw, "ZCMPTYPE= 'HCOMPRESS_1'"), "{raw:#?}");
  assert!(raw.iter().any(|c| c.starts_with("ZNAME") && c.contains("'SMOOTH")), "{raw:#?}");
  assert!(raw.iter().any(|c| c.starts_with("ZNAME") && c.contains("'SCALE")), "{raw:#?}");
  let worst = px.iter().zip(&values).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
  assert!(worst <= 8.0, "worst error {worst}");
  // lossless (scale 0) is exact
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.set_compression(HCOMPRESS_1, &[16, 8]).unwrap();
    w.set_hcomp_scale(0.0).unwrap();
    w.create_image(16, &[16, 8]).unwrap();
    let shorts: Vec<i16> = values.iter().map(|&v| v as i16).collect();
    w.write_pixels(&shorts).unwrap();
    w.finish().unwrap()
  };
  assert_eq!(decode(&bytes).0, values);
}

/// `BLANK` on a tile-compressed integer image reads back as NaN (the card stays `BLANK` in the
/// table header; `ZBLANK` is for quantised floats).
#[test]
fn compressed_integer_blank_is_zblank_and_lossy_int_decodes_close() {
  let mut values: Vec<i16> = (0..128).collect();
  values[5] = -1;
  values[77] = -1;
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.set_compression(RICE_1, &[16, 4]).unwrap();
    w.create_image(16, &[16, 8]).unwrap();
    w.write_key_i64("BLANK", -1, "undefined pixels").unwrap();
    w.write_pixels(&values).unwrap();
    w.finish().unwrap()
  };
  let (px, raw, _) = decode(&bytes);
  // an integer image keeps its BLANK card in the table header (ZBLANK is for quantised floats)
  assert!(has_card(&raw, "BLANK   =                   -1"), "{raw:#?}");
  assert!(px[5].is_nan() && px[77].is_nan(), "{px:?}");
  assert_eq!(px[6], 6.0);
  assert_eq!(px.iter().filter(|v| v.is_nan()).count(), 2);
  // the translated header carries BLANK, not ZBLANK
  let f = MemFits::open(&bytes).unwrap();
  let mut f = f;
  f.move_abs(2).unwrap();
  let cards: Vec<String> = f.cards().unwrap().iter().map(|c| String::from_utf8_lossy(c).trim_end().to_string()).collect();
  assert!(has_card(&cards, "BLANK   =                   -1"), "{cards:#?}");
  assert!(!has_card(&cards, "ZBLANK"), "{cards:#?}");
  drop(f);
}

/// Unsigned integers through BZERO (the `*_IMG` codes write the BZERO card), signed bytes, BSCALE
/// on raw values, BLANK on an uncompressed image, 64-bit values beyond f32 precision, NaN / Inf.
#[test]
fn integer_conventions_read_back_as_physical_values() {
  // uint16: the USHORT_IMG code writes BITPIX 16 + BZERO 32768, and `image_params` reports the
  // stored BITPIX (ffgidt), not the unsigned equivalent
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(sys::USHORT_IMG as i32, &[3, 1]).unwrap();
    w.write_pixels(&[0u16, 65535, 32768]).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.image_params().unwrap().bitpix, 16);
  assert_eq!(f.key_f64("BITPIX").unwrap(), Some(16.0));
  assert_eq!(f.key_f64("BZERO").unwrap(), Some(32768.0));
  let mut px = Vec::new();
  f.read_image_f32(3, &mut px).unwrap();
  assert_eq!(px, [0.0, 65535.0, 32768.0]);
  drop(f);
  // uint32: BZERO 2147483648
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(sys::ULONG_IMG as i32, &[3, 1]).unwrap();
    w.write_pixels(&[0u32, 4_294_967_295, 2_147_483_648]).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.image_params().unwrap().bitpix, 32);
  assert_eq!(f.key_f64("BZERO").unwrap(), Some(2_147_483_648.0));
  let mut px = Vec::new();
  f.read_image_f32(3, &mut px).unwrap();
  assert_eq!(px, [0.0, 4_294_967_295u32 as f32, 2_147_483_648.0]);
  drop(f);
  // int8: BITPIX 8, BZERO -128
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(sys::SBYTE_IMG as i32, &[3, 1]).unwrap();
    w.write_pixels(&[-128i8, 0, 127]).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.image_params().unwrap().bitpix, 8);
  assert_eq!(f.key_f64("BITPIX").unwrap(), Some(8.0));
  assert_eq!(f.key_f64("BZERO").unwrap(), Some(-128.0));
  let mut px = Vec::new();
  f.read_image_f32(3, &mut px).unwrap();
  assert_eq!(px, [-128.0, 0.0, 127.0]);
  drop(f);
  // uint64: BZERO 9223372036854775808
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(sys::ULONGLONG_IMG as i32, &[2, 1]).unwrap();
    w.write_pixels(&[0u64, u64::MAX]).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.image_params().unwrap().bitpix, 64);
  assert_eq!(f.key_f64("BZERO").unwrap(), Some(9_223_372_036_854_775_808.0));
  let mut px = Vec::new();
  f.read_image_f32(2, &mut px).unwrap();
  assert_eq!(px, [0.0, u64::MAX as f32]);
  drop(f);
  // BSCALE / BZERO on raw values, and BLANK → NaN with anynul
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(16, &[4, 1]).unwrap();
    w.write_key_f64("BSCALE", 0.5, 6, "").unwrap();
    w.write_key_f64("BZERO", 10.0, 6, "").unwrap();
    w.write_key_i64("BLANK", -32768, "").unwrap();
    w.set_scale(1.0, 0.0).unwrap(); // write raw values
    w.write_pixels(&[0i16, 4, -2, -32768]).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  let mut px = Vec::new();
  assert!(f.read_image_f32(4, &mut px).unwrap(), "anynul");
  assert_eq!(&px[..3], [10.0, 12.0, 9.0]);
  assert!(px[3].is_nan());
  drop(f);
  // int64 beyond 2^24: the nearest f32
  let big = [1i64 << 53 | 1, -(1i64 << 40) - 3, 16_777_217];
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(64, &[3, 1]).unwrap();
    w.write_pixels(&big).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  let mut px = Vec::new();
  f.read_image_f32(3, &mut px).unwrap();
  assert_eq!(px, [big[0] as f32, big[1] as f32, 16_777_216.0]);
  drop(f);
  // NaN and ±Inf in floats and doubles: NaN is null; cfitsio's float conversion treats an
  // infinity like a NaN (it is "not a number" to `fnan`), so Inf comes back as NaN too
  for bitpix in [-32, -64] {
    let bytes = {
      let mut w = MemFitsWriter::new().unwrap();
      w.create_image(bitpix, &[4, 1]).unwrap();
      w.write_pixels(&[1.5f64, f64::NAN, f64::INFINITY, f64::NEG_INFINITY]).unwrap();
      w.finish().unwrap()
    };
    let f = MemFits::open(&bytes).unwrap();
    let mut px = Vec::new();
    assert!(f.read_image_f32(4, &mut px).unwrap(), "anynul");
    assert_eq!(px[0], 1.5);
    assert!(px[1].is_nan(), "BITPIX {bitpix}: {px:?}");
    assert!(!px[2].is_finite() && !px[3].is_finite(), "BITPIX {bitpix}: {px:?}");
  }
}

/// Header conventions the reader must survive: HIERARCH, CONTINUE, commentary order, a lowercase
/// keyword, trailing garbage after the data, an extension without `EXTEND`.
#[test]
fn header_conventions_cfitsio_accepts() {
  let long = "a".repeat(60) + &"b".repeat(60) + "end";
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(16, &[2, 1]).unwrap();
    w.write_comment("first comment").unwrap();
    w.write_history("first history").unwrap();
    w.write_record("HIERARCH ESO DET DIT = 1.5 / detector integration time").unwrap();
    w.write_long_string("LONGKEY", &long, "joined").unwrap();
    w.write_comment("second comment").unwrap();
    w.write_record("LOWERKEY=                    7 / lowercase keyword").unwrap();
    w.write_history("second history").unwrap();
    w.write_pixels(&[1i16, 2]).unwrap();
    w.finish().unwrap()
  };
  // (ffprec uppercases a keyword on write, so the lowercase card is patched into the bytes)
  let mut bytes = bytes;
  let i = bytes.windows(9).position(|c| c == b"LOWERKEY=").unwrap();
  bytes[i..i + 8].copy_from_slice(b"lowerkey");
  let f = MemFits::open(&bytes).unwrap();
  let cards: Vec<String> = f.cards().unwrap().iter().map(|c| String::from_utf8_lossy(c).trim_end().to_string()).collect();
  let pos = |p: &str| cards.iter().position(|c| c.starts_with(p)).unwrap_or_else(|| panic!("{p}: {cards:#?}"));
  assert!(pos("COMMENT first") < pos("HISTORY first"));
  assert!(pos("HISTORY first") < pos("HIERARCH ESO DET DIT = 1.5"));
  assert!(pos("HIERARCH") < pos("LONGSTRN"));
  assert!(pos("LONGKEY = ") < pos("CONTINUE"));
  assert!(pos("CONTINUE") < pos("COMMENT second"));
  assert!(pos("COMMENT second") < pos("lowerkey="));
  assert!(pos("lowerkey=") < pos("HISTORY second"));
  assert_eq!(f.long_string("LONGKEY").unwrap().as_deref(), Some(long.as_str()));
  assert_eq!(f.key_f64("ESO DET DIT").unwrap(), Some(1.5), "cfitsio reads a HIERARCH keyword by its long name");
  // the lowercase card is tolerated, and cfitsio's keyword search is case-insensitive (ffgcrd
  // uppercases the card's name too), so the value is found under either spelling
  assert_eq!(f.key_f64("LOWERKEY").unwrap(), Some(7.0));
  assert_eq!(f.key_f64("lowerkey").unwrap(), Some(7.0));
  let mut px = Vec::new();
  f.read_image_f32(2, &mut px).unwrap();
  assert_eq!(px, [1.0, 2.0]);
  drop(f);
  // trailing garbage after the last data block: ignored (one HDU)
  let mut trailing = fixture!();
  trailing.extend_from_slice(b"garbage after the data unit, not a header");
  let f = MemFits::open(&trailing).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 1);
  let mut px = Vec::new();
  f.read_image_f32(12, &mut px).unwrap();
  assert_eq!(px, RAMP);
  drop(f);
  // an IMAGE extension behind a primary without EXTEND: still found
  let mut no_extend = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(8, &[]).unwrap();
    w.create_image(16, &[4, 3]).unwrap();
    w.write_pixels(&RAMP).unwrap();
    w.finish().unwrap()
  };
  let i = no_extend.windows(8).position(|c| c == b"EXTEND  ").expect("EXTEND card");
  // blank the card: a header may carry blank cards
  no_extend[i..i + 80].fill(b' ');
  let mut f = MemFits::open(&no_extend).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 2);
  assert_eq!(f.move_abs(2).unwrap(), HduType::Image);
  let mut px = Vec::new();
  f.read_image_f32(12, &mut px).unwrap();
  assert_eq!(px, RAMP);
}

/// Cubes, tables and random groups through the geometry API: a cube reports every axis and its
/// first `nx · ny` pixels are the first plane; tables are not images; a random-groups primary
/// has `naxes[0] == 0`.
#[test]
fn cubes_tables_and_random_groups_geometry() {
  let cube: Vec<i16> = (0..24).collect();
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(16, &[4, 3, 2]).unwrap();
    w.write_pixels(&cube).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.image_params().unwrap().naxes, vec![4, 3, 2]);
  let mut px = Vec::new();
  f.read_image_f32(12, &mut px).unwrap();
  assert_eq!(px, (0..12).map(|i| i as f32).collect::<Vec<_>>());
  drop(f);
  // a BINTABLE and an ASCII table behind an empty primary
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(8, &[]).unwrap();
    w.create_table(true, 3, &[("FLUX", "1E"), ("ID", "1J")]).unwrap();
    w.write_column(1, 1, &[1.0f32, 2.0, 3.0]).unwrap();
    w.write_column(2, 1, &[7i32, 8, 9]).unwrap();
    w.create_table(false, 2, &[("NAME", "A8"), ("VALUE", "E12.4")]).unwrap();
    w.finish().unwrap()
  };
  let mut f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 3);
  assert_eq!(f.move_abs(2).unwrap(), HduType::BinaryTable);
  assert!(!f.is_compressed_image());
  assert_eq!(f.move_abs(3).unwrap(), HduType::AsciiTable);
  drop(f);
  // random groups
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_random_groups(16, &[4, 3], 1, 2).unwrap();
    w.finish().unwrap()
  };
  let f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.hdu_count().unwrap(), 1);
  let p = f.image_params().unwrap();
  assert_eq!(p.naxes, vec![0, 4, 3]);
  let cards: Vec<String> = f.cards().unwrap().iter().map(|c| String::from_utf8_lossy(c).trim_end().to_string()).collect();
  assert!(has_card(&cards, "GROUPS  =                    T"), "{cards:#?}");
  assert!(has_card(&cards, "GCOUNT  =                    2"), "{cards:#?}");
}

// ───────────────────────────── .Z and .bz2 ─────────────────────────────

/// The committed `.Z` twin of the fixture (written by an LZW encoder and checked against
/// `gzip -dc`) inflates to the fixture; the stream is block mode with 16-bit codes, so the
/// width grows through 9 → 14 bits with the format's group padding at every step.
#[test]
fn unlzw_decodes_the_committed_fixture() {
  use crate::unlzw;
  let z = std::fs::read(aethervk_test_assets::test_asset!("fits_lt_67p_header_4x3.fits.Z")).expect(".Z fixture");
  assert!(crate::is_unix_compress(&z));
  assert_eq!(z[2], 0x90, "block mode, 16 bits");
  let out = unlzw(&z).unwrap();
  assert_eq!(out, fixture!());
  // garbage and truncation
  assert!(unlzw(b"\x1f\x9d").is_err());
  assert!(unlzw(b"\x1f\x9d\x88abc").is_err(), "8 bits is below the format's minimum");
  let cut = unlzw(&z[..z.len() / 2]).unwrap();
  assert!(cut.len() < fixture!().len() && fixture!().starts_with(&cut));
  assert!(unlzw(&fixture!()).is_err());
  // the real NASA cube of the FITS corpus when it is around (5 158 080 bytes, CLEAR codes inside)
  if let Some(dir) = std::env::var_os("AETHERVK_FITS_CORPUS") {
    let p = std::path::PathBuf::from(dir).join("nasa-samples/sdn/w0bs0102t_c0h.Z");
    if let Ok(bytes) = std::fs::read(&p) {
      let out = unlzw(&bytes).unwrap();
      assert_eq!(out.len(), 5_158_080);
      assert!(out.starts_with(b"SIMPLE  ="));
      assert_eq!(out.len() % 2880, 0);
    }
  }
}

#[test]
fn bzip2_round_trips() {
  use crate::{bunzip2, bzip2_compress, is_bzip2};
  let bytes = fixture!();
  let bz = bzip2_compress(&bytes).unwrap();
  assert!(is_bzip2(&bz));
  assert!(bz.len() < bytes.len() / 2);
  assert_eq!(bunzip2(&bz).unwrap(), bytes);
  assert!(MemFits::open(&bz).is_err());
  let e = bunzip2(&bz[..bz.len() / 2]).unwrap_err();
  assert!(e.text.contains("truncated"), "{e}");
  assert!(bunzip2(b"BZh9 garbage").is_err());
  assert!(bunzip2(&bytes).is_err());
  // an output much larger than the input (zeros) grows the buffer past the first guess
  let zeros = vec![0u8; 4 << 20];
  let bz = bzip2_compress(&zeros).unwrap();
  assert!(bz.len() < 4096);
  assert_eq!(bunzip2(&bz).unwrap(), zeros);
}

/// A `BLANK` written as a float (`-32768.0`, CDS HiPS tiles) is not honoured by cfitsio's pixel
/// read (the memory path returns the blank as a value; other paths report "Error reading BLANK
/// keyword value as an integer"); after `set_image_null` with the integral value the blank is NaN.
#[test]
fn float_blank_card_needs_set_image_null() {
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(16, &[3, 1]).unwrap();
    w.write_record("BLANK   =             -32768.0 / written as a float").unwrap();
    w.write_pixels(&[5i16, -32768, 7]).unwrap();
    w.finish().unwrap()
  };
  let mut f = MemFits::open(&bytes).unwrap();
  let cards: Vec<String> = f.cards().unwrap().iter().map(|c| String::from_utf8_lossy(c).trim_end().to_string()).collect();
  assert!(cards.iter().any(|c| c.starts_with("BLANK   =             -32768.0")), "{cards:#?}");
  let mut px = Vec::new();
  // cfitsio's memory read ignores the non-integer card: the blank comes back as a value
  assert!(!f.read_image_f32(3, &mut px).unwrap());
  assert_eq!(px, [5.0, -32768.0, 7.0]);
  f.set_image_null(-32768).unwrap();
  assert!(f.read_image_f32(3, &mut px).unwrap(), "anynul");
  assert_eq!((px[0], px[2]), (5.0, 7.0));
  assert!(px[1].is_nan());
}

/// A highly compressible 512 × 512 BITPIX 16 image (a HiPS tile's size) inflates to many times
/// its gzip stream: the codec's growth path (its realloc callback) is what reads it back.
#[test]
fn gunzip_grows_its_buffer_for_a_compressible_image() {
  use crate::{gunzip, gzip};
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(16, &[512, 512]).unwrap();
    w.write_pixels(&vec![7i16; 512 * 512]).unwrap();
    w.finish().unwrap()
  };
  assert_eq!(bytes.len(), 2880 + 524_288usize.div_ceil(2880) * 2880);
  let gz = gzip(&bytes).unwrap();
  assert!(gz.len() * 16 < bytes.len(), "{}", gz.len());
  let back = gunzip(&gz).unwrap_or_else(|e| panic!("{e} (gz {} bytes)", gz.len()));
  assert_eq!(back.len(), bytes.len());
  assert!(back == bytes);
}

/// gzip round trips of a tile-compressed file and of a real gzip-named HiPS tile (corpus).
#[test]
fn gunzip_tile_compressed_and_corpus_tile() {
  use crate::{gunzip, gzip};
  let fz = {
    let mut w = MemFitsWriter::new().unwrap();
    w.set_compression(crate::writer::RICE_1, &[4, 3]).unwrap();
    w.create_image(16, &[4, 3]).unwrap();
    let v: Vec<i16> = (1..=12).map(|x| x * 100).collect();
    w.write_pixels(&v).unwrap();
    w.finish().unwrap()
  };
  let gz = gzip(&fz).unwrap();
  let back = gunzip(&gz).unwrap_or_else(|e| panic!("{e} (fz {} bytes, gz {} bytes)", fz.len(), gz.len()));
  assert!(back == fz, "{} vs {}", back.len(), fz.len());
  // the corpus HiPS tiles: a gzip stream named .fits (AllWISE) inflates to whole FITS blocks; the
  // DSS2 tiles are plain FITS (and 2752 bytes short of a block: the reader pads them)
  if let Some(dir) = std::env::var_os("AETHERVK_FITS_CORPUS") {
    let dir = std::path::PathBuf::from(dir).join("hips");
    if let Ok(bytes) = std::fs::read(dir.join("AllWISE_W1_Norder3_Dir0_Npix300.fits")) {
      assert!(crate::is_gzip(&bytes));
      let out = gunzip(&bytes).unwrap();
      // (also written without its last block's padding: 1 051 456 bytes)
      assert!(out.starts_with(b"SIMPLE  =") && out.len() > 2880, "{}", out.len());
    }
    if let Ok(bytes) = std::fs::read(dir.join("DSS2_red_Norder3_Dir0_Npix300.fits")) {
      assert!(!crate::is_gzip(&bytes) && bytes.starts_with(b"SIMPLE  ="));
      assert_eq!(bytes.len(), 2880 + 512 * 512 * 2);
      assert!(gunzip(&bytes).is_err());
    }
  }
}

/// `read_image_f32_from` reads one plane of a cube (plain and tile-compressed): the pixel offset
/// is the plane times the plane size; past the data array cfitsio reports an error.
#[test]
fn plane_reads_by_pixel_offset() {
  for compressed in [false, true] {
    let bytes = {
      let mut w = MemFitsWriter::new().unwrap();
      if compressed {
        w.set_compression(crate::writer::RICE_1, &[4, 3, 1]).unwrap();
      }
      w.create_image(16, &[4, 3, 3]).unwrap();
      let v: Vec<i16> = (0..36).map(|i| (i % 12) * 10 + (i / 12) * 1000).collect();
      w.write_pixels(&v).unwrap();
      w.finish().unwrap()
    };
    let mut f = MemFits::open(&bytes).unwrap();
    if compressed {
      f.move_abs(2).unwrap();
      assert!(f.is_compressed_image());
    }
    assert_eq!(f.image_params().unwrap().naxes, vec![4, 3, 3]);
    let mut px = Vec::new();
    for plane in 0..3u64 {
      f.read_image_f32_from(plane * 12, 12, &mut px).unwrap();
      let expect: Vec<f32> = (0..12).map(|i| (i * 10 + plane as i32 * 1000) as f32).collect();
      assert_eq!(px, expect, "compressed {compressed} plane {plane}");
    }
    // past the array: an error for a plain image (end of the data unit); a tile-compressed image
    // reports none (cfitsio clips to the tiles it has), so callers check the plane range first
    if !compressed {
      assert!(f.read_image_f32_from(36, 12, &mut px).is_err(), "past the array");
    }
  }
}

/// `hdu_address`: the primary header of a one-block header ends at 2880, its data unit at the
/// next block; an extension starts where the primary ends.
#[test]
fn hdu_addresses_are_block_offsets() {
  let bytes = {
    let mut w = MemFitsWriter::new().unwrap();
    w.create_image(16, &[4, 3]).unwrap();
    w.write_pixels(&[1i16; 12]).unwrap();
    w.create_image(16, &[2, 2]).unwrap();
    w.write_pixels(&[2i16; 4]).unwrap();
    w.finish().unwrap()
  };
  let mut f = MemFits::open(&bytes).unwrap();
  assert_eq!(f.hdu_address().unwrap(), (0, 2880, 5760));
  f.move_abs(2).unwrap();
  assert_eq!(f.hdu_address().unwrap(), (5760, 8640, 11520));
  assert_eq!(bytes.len(), 11520);
}
