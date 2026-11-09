//! HiPS (IVOA Hierarchical Progressive Surveys 1.0, 2017-05-19) tiles and `Allsky` mosaics
//! opened as sky images: the HEALPix NESTED geometry that places every tile pixel on the sky.
//!
//! ## HEALPix (Górski et al. 2005, ApJ 622, 759)
//! `nside = 2^order`, 12 base faces, `12 · nside²` cells. NESTED index = face · nside² + the bit
//! interleave of the face coordinates `(ix, iy)`: `ix` on the even bits, `iy` on the odd bits
//! ([`nest_to_xyf`], [`xyf_to_nest`]). The face axes: `ix` runs from the south corner of the
//! face toward its east corner (north-east), `iy` toward its west corner (north-west);
//! [`xyf_to_loc`] is the continuous face → `(z = sin lat, φ)` map, [`loc_to_xyf`] its inverse.
//! Corners of a cell, in `(dx, dy)` face coordinates: south `(0, 0)`, east `(1, 0)`, north `(1, 1)`,
//! west `(0, 1)`.
//!
//! ## HiPS tiles (standard §4.1, §4.2.1)
//! A tile at order `k` with index `N` (file `Norder{k}/Dir{D}/Npix{N}.fits`, `D = (N / 10000) ·
//! 10000`, §4.1) is the HEALPix NESTED cell `(k, N)`; HiPS uses the NESTED scheme only (§4.1).
//! Its `W × W` pixels (`W = hips_tile_width`, default 512, §4.4.2; `S = log2 W` the "shift
//! order", §4.2.1) are the `4^S` NESTED sub-cells `N · 4^S + s` at order `k + S`, so the tile
//! pixel grid **is** a HEALPix grid: pixel `(x, y)` ↔ face coordinates `(ix, iy)` of the
//! sub-cell inside the tile. The standard shows this packaging in its figure 4 and refers to the
//! HipsGen implementation for the bit order. The convention below was verified on real HipsGen
//! FITS tiles, not read off the figure: the CDS AllWISE W1 order-3 tile 450 against the
//! galactic plane (8 candidate flips / transposes; this one wins on both |b| and |l|), and the
//! CDS DSS2 red order-6 tile 47100 against an independent DSS image (correlation 0.898, the 7
//! other candidates at most 0.29; a `reproject` tile WCS agrees). The tables in
//! `test_assets/fits_corpus/hips_reference/` (astropy-healpix, format in the corpus README)
//! encode the same rule and are checked by `tests::hips_matches_the_reference_tables`.
//! The convention is
//!
//! ```text
//! FITS pixel (p1, p2), 1-based, p2 = 1 the first stored row (FITS bottom-up order):
//!     iy = p1 − 1            (the HEALPix "y" axis runs along the FITS columns, left → right)
//!     ix = W − p2            (the HEALPix "x" axis runs along the rows, top row = ix 0)
//!     s  = interleave(ix on the even bits, iy on the odd bits)
//! ```
//!
//! i.e. the sub-cell's `(ix, iy)` are the FITS `(row, column)` — the x/y swap — and `ix`
//! counts from the **last** stored row: displayed bottom-up as a FITS viewer does, the HEALPix
//! north corner of the tile is at the bottom right (east corner bottom left, west corner top
//! right, south corner top left). JPEG / PNG tiles store the rows top-down (§4.2.1.3 note),
//! which is the same picture; this module only maps FITS tiles. Inside a pixel the map is the
//! continuous [`xyf_to_loc`] (exact HEALPix geometry; pixel centres at `(ix + ½, iy + ½)`).
//!
//! ## Allsky (standard §4.3.2)
//! `Norder{K}/Allsky.fits` (`K ≤ 3`) is the mosaic of all `12 · 4^K` tiles of order `K`, each
//! reduced to a power-of-two width `w` (`NAXIS1 / ncols`), stored side by side left to right,
//! `ncols = ⌊√(12 · 4^K)⌋` per row (27 at order 3), tile `N` at column `N mod ncols` and row
//! `⌊N / ncols⌋` counted from the **top** of the displayed image (the last stored FITS rows hold
//! tiles 0 … 26); the pixels inside a block follow the tile rule above. Verified on the CDS
//! AllWISE `Norder3/Allsky.fits` (1728 × 1856 = 27 × 64 by 29 × 64).
//!
//! ## Frame (standard §4.4.2 `hips_frame`)
//! `equatorial` (ICRS), `galactic`, `ecliptic`, read from the survey's `properties` file (the
//! HiPS root, two or three directories above the tile), else a header keyword (`COORDSYS` with
//! the HEALPix-map letters `C` / `G` / `E`, or `HIPSFRAM` / `FRAME` words), else equatorial.
//! Galactic → ICRS is the IAU 1958 system as realised in FK5 J2000 / ICRS (α_NGP 192.85948°,
//! δ_NGP 27.12825°, l_NCP 122.93192°, the Hipparcos / astropy definition; FK5 J2000 is treated
//! as ICRS, within ~20 mas like the rest of `sky_image`). Ecliptic is the engine's ecliptic J2000
//! (`OBLIQUITY_J2000_RAD`, 84381.448″), what HipsGen means by `ecliptic`.
//!
//! ## Identification
//! HipsGen FITS tiles carry no `ORDER` / `NPIX` cards (a minimal `SIMPLE / BITPIX / NAXIS`
//! header, HipsGen v10), so a tile is recognised by its path: the `Norder{k}`, `Dir{d}`,
//! `Npix{n}` and `Allsky` components separated by `/`, `\` or `_` (the corpus flattens
//! `Norder3/Dir0/Npix300.fits` to `..._Norder3_Dir0_Npix300.fits`); a cube frame suffix
//! `Npix{n}_{f}` is accepted. Header cards `ORDER` + `NPIX` (integers) are honoured first when
//! present.

use alloc::{
  collections::BTreeMap,
  format,
  string::{String, ToString},
  vec::Vec,
};

use super::{
  FitsHeader, SkyImageVertex, dir_of_ra_dec, ecliptic_to_equatorial_j2000,
  equatorial_to_ecliptic_j2000, ra_dec_of,
};

type V3 = [f64; 3];

const PI: f64 = core::f64::consts::PI;
const HALF_PI: f64 = core::f64::consts::FRAC_PI_2;
const DEG: f64 = PI / 180.0;

/// `hips_tile_width` default (standard §4.4.2)
pub const HIPS_DEFAULT_TILE_WIDTH: u32 = 512;
/// deepest tile order this module maps (the standard's table 5 stops at 20; `order + S` must fit
/// 29 for the 64-bit NESTED index and `f64` face coordinates)
pub const HIPS_MAX_ORDER: u8 = 20;
/// highest order an `Allsky` file is defined for (standard §4.3.2)
pub const HIPS_ALLSKY_MAX_ORDER: u8 = 3;

// ─── HEALPix NESTED ──────────────────────────────────────────────────────────

/// ring number of the north corner of each face, in units of `nside`
const JRLL: [u32; 12] = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
/// longitude of the face centre, in units of 45°
const JPLL: [u32; 12] = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

/// `nside` of an order.
pub const fn nside(order: u8) -> u64 {
  1u64 << order
}

/// Number of cells at an order (`12 · 4^order`).
pub const fn npix(order: u8) -> u64 {
  12 * (1u64 << (2 * order))
}

/// Spreads the low 32 bits of `v` to the even bits.
fn spread_bits(v: u64) -> u64 {
  let mut v = v & 0xffff_ffff;
  v = (v | (v << 16)) & 0x0000_ffff_0000_ffff;
  v = (v | (v << 8)) & 0x00ff_00ff_00ff_00ff;
  v = (v | (v << 4)) & 0x0f0f_0f0f_0f0f_0f0f;
  v = (v | (v << 2)) & 0x3333_3333_3333_3333;
  v = (v | (v << 1)) & 0x5555_5555_5555_5555;
  v
}

/// Gathers the even bits of `v`.
fn compact_bits(v: u64) -> u64 {
  let mut v = v & 0x5555_5555_5555_5555;
  v = (v | (v >> 1)) & 0x3333_3333_3333_3333;
  v = (v | (v >> 2)) & 0x0f0f_0f0f_0f0f_0f0f;
  v = (v | (v >> 4)) & 0x00ff_00ff_00ff_00ff;
  v = (v | (v >> 8)) & 0x0000_ffff_0000_ffff;
  v = (v | (v >> 16)) & 0x0000_0000_ffff_ffff;
  v
}

/// NESTED index → `(face, ix, iy)` at `order`.
pub fn nest_to_xyf(order: u8, ipix: u64) -> (u8, u32, u32) {
  let n2 = 1u64 << (2 * order);
  let face = (ipix / n2) as u8;
  let local = ipix % n2;
  (face, compact_bits(local) as u32, compact_bits(local >> 1) as u32)
}

/// `(face, ix, iy)` at `order` → NESTED index.
pub fn xyf_to_nest(order: u8, face: u8, ix: u32, iy: u32) -> u64 {
  (face as u64) * (1u64 << (2 * order)) + spread_bits(ix as u64) + (spread_bits(iy as u64) << 1)
}

/// `(face, ix, iy)` at `order` → RING index (for cross-checks against ring-scheme formulae).
pub fn xyf_to_ring(order: u8, face: u8, ix: u32, iy: u32) -> u64 {
  let ns = nside(order) as i64;
  let nl4 = 4 * ns;
  let jr = JRLL[face as usize] as i64 * ns - ix as i64 - iy as i64 - 1;
  let (nr, n_before, kshift) = if jr < ns {
    (jr, 2 * jr * (jr - 1), 0)
  } else if jr > 3 * ns {
    let nr = nl4 - jr;
    (nr, npix(order) as i64 - 2 * (nr + 1) * nr, 0)
  } else {
    let ncap = 2 * ns * (ns - 1);
    (ns, ncap + (jr - ns) * nl4, (jr - ns) & 1)
  };
  let mut jp = (JPLL[face as usize] as i64 * nr + ix as i64 - iy as i64 + 1 + kshift) / 2;
  if jp > nl4 {
    jp -= nl4;
  }
  if jp < 1 {
    jp += nl4;
  }
  (n_before + jp - 1) as u64
}

/// NESTED → RING index.
pub fn nest_to_ring(order: u8, ipix: u64) -> u64 {
  let (f, x, y) = nest_to_xyf(order, ipix);
  xyf_to_ring(order, f, x, y)
}

/// `(z = sin lat, φ)` (radians, `φ ∈ [0, 2π)`) of the point of face `face` with face-normalised
/// coordinates `x, y ∈ [0, 1]` (`x = (ix + dx) / nside`): the continuous HEALPix geometry
/// (Górski et al. 2005 §4; the `xyf2loc` of the reference implementation).
pub fn xyf_to_loc(face: u8, x: f64, y: f64) -> (f64, f64) {
  let jr = JRLL[face as usize] as f64 - x - y;
  let (nr, z) = if jr < 1.0 {
    (jr, 1.0 - jr * jr / 3.0)
  } else if jr > 3.0 {
    let nr = 4.0 - jr;
    (nr, nr * nr / 3.0 - 1.0)
  } else {
    (1.0, (2.0 - jr) * 2.0 / 3.0)
  };
  let mut tmp = JPLL[face as usize] as f64 * nr + x - y;
  if tmp < 0.0 {
    tmp += 8.0;
  }
  if tmp >= 8.0 {
    tmp -= 8.0;
  }
  let phi = if nr < 1e-15 { 0.0 } else { 0.5 * HALF_PI * tmp / nr };
  (z, phi)
}

/// Inverse of [`xyf_to_loc`]: `(face, x, y)` with `x, y ∈ [0, 1)` face-normalised (the cell at
/// `order` is `(⌊x · nside⌋, ⌊y · nside⌋)`). `φ` any value, `z ∈ [−1, 1]`.
pub fn loc_to_xyf(z: f64, phi: f64) -> (u8, f64, f64) {
  let za = z.abs();
  let tt = (phi.rem_euclid(2.0 * PI)) / HALF_PI; // [0, 4)
  if za <= 2.0 / 3.0 {
    // equatorial belt: lines of constant jp (ascending) / jm (descending)
    let temp1 = 0.5 + tt;
    let temp2 = z * 0.75;
    let jp = temp1 - temp2;
    let jm = temp1 + temp2;
    let ifp = jp.floor() as i64; // in 0..=4
    let ifm = jm.floor() as i64;
    let face = if ifp == ifm {
      (ifp | 4) as u8
    } else if ifp < ifm {
      ifp as u8
    } else {
      (ifm + 8) as u8
    };
    let x = jm - ifm as f64;
    let y = 1.0 - (jp - ifp as f64);
    (face % 12, x.clamp(0.0, 1.0 - f64::EPSILON), y.clamp(0.0, 1.0 - f64::EPSILON))
  } else {
    let ntt = (tt.floor() as i64).min(3);
    let tp = tt - ntt as f64;
    let tmp = (3.0 * (1.0 - za)).sqrt(); // in [0, 1)
    let jp = (tp * tmp).min(1.0 - f64::EPSILON);
    let jm = ((1.0 - tp) * tmp).min(1.0 - f64::EPSILON);
    if z >= 0.0 {
      // the pole itself (jp = jm = 0) is the north corner, shared by the face: keep it inside
      let hi = 1.0 - f64::EPSILON;
      (ntt as u8, (1.0 - jm).min(hi), (1.0 - jp).min(hi))
    } else {
      (ntt as u8 + 8, jp, jm)
    }
  }
}

/// `(lon, lat)` degrees, `lon ∈ [0, 360)`, of `(z, φ)`.
fn lonlat_of_loc(z: f64, phi: f64) -> (f64, f64) {
  let mut lon = phi / DEG;
  if lon >= 360.0 {
    lon -= 360.0;
  }
  if lon < 0.0 {
    lon += 360.0;
  }
  (lon, z.clamp(-1.0, 1.0).asin() / DEG)
}

/// Centre `(lon, lat)` (degrees) of NESTED cell `ipix` at `order`.
pub fn pix_to_lonlat(order: u8, ipix: u64) -> (f64, f64) {
  pix_to_lonlat_at(order, ipix, 0.5, 0.5)
}

/// `(lon, lat)` (degrees) of the point `(dx, dy) ∈ [0, 1]²` of NESTED cell `ipix` at `order`
/// (`dx` along `ix`, `dy` along `iy`; the centre is `(0.5, 0.5)`).
pub fn pix_to_lonlat_at(order: u8, ipix: u64, dx: f64, dy: f64) -> (f64, f64) {
  let (face, ix, iy) = nest_to_xyf(order, ipix);
  let ns = nside(order) as f64;
  let (z, phi) = xyf_to_loc(face, (ix as f64 + dx) / ns, (iy as f64 + dy) / ns);
  lonlat_of_loc(z, phi)
}

/// Corners `[south, east, north, west]` (`(lon, lat)` degrees) of NESTED cell `ipix`.
pub fn pix_corners(order: u8, ipix: u64) -> [(f64, f64); 4] {
  [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)].map(|(dx, dy)| pix_to_lonlat_at(order, ipix, dx, dy))
}

/// NESTED cell at `order` containing `(lon, lat)` (degrees), and the position inside it
/// (`dx, dy ∈ [0, 1)`).
pub fn lonlat_to_pix(order: u8, lon_deg: f64, lat_deg: f64) -> (u64, f64, f64) {
  let (face, x, y) = loc_to_xyf((lat_deg * DEG).sin(), lon_deg * DEG);
  let ns = nside(order) as f64;
  let (fx, fy) = (x * ns, y * ns);
  let (ix, iy) = (fx.floor(), fy.floor());
  (
    xyf_to_nest(order, face, ix as u32, iy as u32),
    fx - ix,
    fy - iy,
  )
}

/// RING-scheme centre `(z, φ)` from the closed forms of Górski et al. 2005 §4.1 (independent of
/// the NESTED path: the cross-check of the tests).
pub fn ring_pix_to_loc(order: u8, ipix: u64) -> (f64, f64) {
  let ns = nside(order);
  let np = npix(order);
  let ncap = 2 * ns * (ns - 1);
  let (z, phi);
  if ipix < ncap {
    // north polar cap
    let iring = (1 + ((1 + 2 * ipix) as u64).isqrt()) / 2;
    let iphi = ipix + 1 - 2 * iring * (iring - 1);
    z = 1.0 - (iring * iring) as f64 / (3 * ns * ns) as f64;
    phi = (iphi as f64 - 0.5) * HALF_PI / iring as f64;
  } else if ipix < np - ncap {
    // equatorial belt
    let ip = ipix - ncap;
    let iring = ip / (4 * ns) + ns;
    let iphi = ip % (4 * ns) + 1;
    let fodd = if (iring + ns) & 1 == 1 { 1.0 } else { 0.5 };
    z = (2 * ns) as f64 - iring as f64;
    let z = z * 2.0 / (3 * ns) as f64;
    phi = (iphi as f64 - fodd) * HALF_PI / ns as f64;
    return (z, phi);
  } else {
    // south polar cap
    let ip = np - ipix;
    let iring = (1 + ((2 * ip - 1) as u64).isqrt()) / 2;
    let iphi = 4 * iring + 1 - (ip - 2 * iring * (iring - 1));
    z = -1.0 + (iring * iring) as f64 / (3 * ns * ns) as f64;
    phi = (iphi as f64 - 0.5) * HALF_PI / iring as f64;
  }
  (z, phi)
}

// ─── frames ──────────────────────────────────────────────────────────────────

/// `hips_frame` (standard §4.4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipsFrame {
  /// ICRS
  Equatorial,
  Galactic,
  /// ecliptic J2000
  Ecliptic,
}

impl HipsFrame {
  /// The `properties` word (`equatorial`, `galactic`, `ecliptic`; `icrs` / `equ` / `gal` / `ecl` /
  /// the HEALPix-map letters `C` / `G` / `E` accepted).
  pub fn parse(s: &str) -> Option<Self> {
    let s = s.trim().to_ascii_lowercase();
    match s.as_str() {
      "equatorial" | "icrs" | "equ" | "c" | "q" | "fk5" | "j2000" => Some(Self::Equatorial),
      "galactic" | "gal" | "g" => Some(Self::Galactic),
      "ecliptic" | "ecl" | "e" => Some(Self::Ecliptic),
      _ => None,
    }
  }

  pub fn name(self) -> &'static str {
    match self {
      Self::Equatorial => "equatorial",
      Self::Galactic => "galactic",
      Self::Ecliptic => "ecliptic",
    }
  }

  /// Unit vector in this frame → ICRS.
  pub fn to_icrs(self, v: V3) -> V3 {
    match self {
      Self::Equatorial => v,
      Self::Galactic => super::wcs::Frame::galactic().to_icrs(v),
      Self::Ecliptic => ecliptic_to_equatorial_j2000(v),
    }
  }

  /// ICRS unit vector → this frame.
  pub fn from_icrs(self, v: V3) -> V3 {
    match self {
      Self::Equatorial => v,
      Self::Galactic => super::wcs::Frame::galactic().from_icrs(v),
      Self::Ecliptic => equatorial_to_ecliptic_j2000(v),
    }
  }
}

/// Rows of the ICRS → galactic rotation (the galactic x, y, z axes in ICRS components), from the
/// defining angles α_NGP = 192.85948°, δ_NGP = 27.12825°, l_NCP = 122.93192° (Hipparcos vol. 1
/// §1.5.3; astropy's `Galactic`). Equals the Hipparcos matrix to 1e-10.
pub fn galactic_matrix() -> [V3; 3] {
  let z = dir_of_ra_dec(192.85948, 27.12825);
  // the north celestial pole projected on the galactic plane: the direction of l = l_NCP
  let ncp = [0.0, 0.0, 1.0];
  let d = dot(ncp, z);
  let p = normalize([ncp[0] - d * z[0], ncp[1] - d * z[1], ncp[2] - d * z[2]]);
  let zp = cross(z, p);
  let (s, c) = (122.93192 * DEG).sin_cos();
  let x = [c * p[0] - s * zp[0], c * p[1] - s * zp[1], c * p[2] - s * zp[2]];
  let y = cross(z, x);
  [x, y, z]
}

/// ICRS → galactic unit vector.
pub fn icrs_to_galactic(v: V3) -> V3 {
  let m = galactic_matrix();
  [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// Galactic → ICRS unit vector.
pub fn galactic_to_icrs(v: V3) -> V3 {
  let m = galactic_matrix();
  [
    m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
    m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
    m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
  ]
}

fn dot(a: V3, b: V3) -> f64 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
  [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
}
fn normalize(a: V3) -> V3 {
  let l = dot(a, a).sqrt();
  [a[0] / l, a[1] / l, a[2] / l]
}

/// Unit vector of `(lon, lat)` degrees (any frame).
fn dir_of_lonlat(lon_deg: f64, lat_deg: f64) -> V3 {
  dir_of_ra_dec(lon_deg, lat_deg)
}

// ─── tile ────────────────────────────────────────────────────────────────────

/// One HiPS tile: HEALPix cell `(order, npix)` of `width × width` pixels in `frame`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipsTile {
  pub order: u8,
  pub npix: u64,
  /// `W`, a power of two
  pub width: u32,
  pub frame: HipsFrame,
  /// `S = log2 W`
  pub shift: u8,
}

impl HipsTile {
  pub fn new(order: u8, npix: u64, width: u32, frame: HipsFrame) -> Result<Self, String> {
    if order > HIPS_MAX_ORDER {
      return Err(format!("HiPS order {order} above {HIPS_MAX_ORDER}"));
    }
    if npix >= self::npix(order) {
      return Err(format!("HiPS tile index {npix} out of range at order {order} ({} tiles)", self::npix(order)));
    }
    if width == 0 || !width.is_power_of_two() || width > 1 << 16 {
      return Err(format!("HiPS tile width {width} is not a power of two"));
    }
    let shift = width.trailing_zeros() as u8;
    if order + shift > 29 {
      return Err(format!("HiPS order {order} with tile width {width} exceeds HEALPix order 29"));
    }
    Ok(Self {
      order,
      npix,
      width,
      frame,
      shift,
    })
  }

  /// Order of the tile's pixels (`order + S`).
  pub fn pixel_order(&self) -> u8 {
    self.order + self.shift
  }

  /// Face and face coordinates (at `pixel_order`, continuous) of a FITS pixel `(p1, p2)`; the
  /// tile pixel convention of the module doc. Pixels off the tile extrapolate the tile's face.
  fn pixel_to_xyf(&self, p1: f64, p2: f64) -> (u8, f64, f64) {
    let (face, tx, ty) = nest_to_xyf(self.order, self.npix);
    let w = self.width as f64;
    let ix = w + 0.5 - p2;
    let iy = p1 - 0.5;
    (face, tx as f64 * w + ix, ty as f64 * w + iy)
  }

  /// HEALPix NESTED cell (at `pixel_order`) of the 0-based tile pixel `(x, y)`, `x` the column
  /// along NAXIS1, `y` the row along NAXIS2 (row 0 = first stored row).
  pub fn pixel_cell(&self, x: u32, y: u32) -> u64 {
    let ix = (self.width - 1 - y.min(self.width - 1)) as u64;
    let iy = x.min(self.width - 1) as u64;
    self.npix * (1u64 << (2 * self.shift)) + spread_bits(ix) + (spread_bits(iy) << 1)
  }

  /// Unit vector in the tile's own frame of FITS pixel `(p1, p2)` (continuous, 1-based).
  pub fn pixel_to_frame_dir(&self, p1: f64, p2: f64) -> V3 {
    let (face, fx, fy) = self.pixel_to_xyf(p1, p2);
    let ns = nside(self.pixel_order()) as f64;
    let (z, phi) = xyf_to_loc(face, fx / ns, fy / ns);
    let (lon, lat) = lonlat_of_loc(z, phi);
    dir_of_lonlat(lon, lat)
  }

  /// ICRS unit vector of FITS pixel `(p1, p2)`.
  pub fn pixel_to_icrs(&self, p1: f64, p2: f64) -> V3 {
    self.frame.to_icrs(self.pixel_to_frame_dir(p1, p2))
  }

  /// ICRS RA / Dec (degrees) of FITS pixel `(p1, p2)`.
  pub fn pixel_to_radec_deg(&self, p1: f64, p2: f64) -> (f64, f64) {
    ra_dec_of(self.pixel_to_icrs(p1, p2))
  }

  /// FITS pixel `(p1, p2)` of an ICRS unit vector; `None` outside the tile. On the tile the
  /// result is in `[0.5, W + 0.5]²`.
  pub fn icrs_to_pixel(&self, d: V3) -> Option<(f64, f64)> {
    let v = self.frame.from_icrs(d);
    let (lon, lat) = ra_dec_of(v);
    let (face, x, y) = loc_to_xyf((lat * DEG).sin(), lon * DEG);
    let (tface, tx, ty) = nest_to_xyf(self.order, self.npix);
    if face != tface {
      return None;
    }
    let ns = nside(self.pixel_order()) as f64;
    let w = self.width as f64;
    let (fx, fy) = (x * ns - tx as f64 * w, y * ns - ty as f64 * w);
    if !(0.0..=w).contains(&fx) || !(0.0..=w).contains(&fy) {
      return None;
    }
    Some((fy + 0.5, w + 0.5 - fx))
  }

  /// FITS pixel of ICRS RA / Dec (degrees); `None` outside the tile.
  pub fn radec_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
    self.icrs_to_pixel(dir_of_ra_dec(ra_deg, dec_deg))
  }

  /// ICRS RA / Dec (degrees) of the tile centre (the cell centre).
  pub fn center_radec_deg(&self) -> (f64, f64) {
    let (lon, lat) = pix_to_lonlat(self.order, self.npix);
    ra_dec_of(self.frame.to_icrs(dir_of_lonlat(lon, lat)))
  }

  /// ICRS RA / Dec (degrees) of the corners `[south, east, north, west]`.
  pub fn corners_radec_deg(&self) -> [(f64, f64); 4] {
    pix_corners(self.order, self.npix).map(|(lon, lat)| ra_dec_of(self.frame.to_icrs(dir_of_lonlat(lon, lat))))
  }

  /// Registration mesh of the tile: `n × n` vertices over `[0.5, W + 0.5]²`, ecliptic J2000 unit
  /// directions (drawn at infinity) with texture coordinates `(p − 0.5) / span`; a triangle list
  /// like `sky_image::grid_vertices`. The HEALPix map is evaluated exactly at every vertex; the
  /// rasteriser interpolates linearly between them (at `n = 32` the chord of a 7° order-3 tile
  /// is 0.24°, a sagitta of 2 µrad: below any registration budget; the texture is sampled in
  /// the tile's own HEALPix grid, bilinear inside a pixel).
  pub fn mesh_vertices(&self, span: [f64; 2], n: usize) -> Vec<SkyImageVertex> {
    let n = n.max(2);
    let w = self.width as f64;
    let mut grid = Vec::with_capacity(n * n);
    for j in 0..n {
      for i in 0..n {
        let p1 = 0.5 + w * i as f64 / (n - 1) as f64;
        let p2 = 0.5 + w * j as f64 / (n - 1) as f64;
        let d = equatorial_to_ecliptic_j2000(self.pixel_to_icrs(p1, p2));
        grid.push(SkyImageVertex {
          pos: [d[0] as f32, d[1] as f32, d[2] as f32],
          uv: [((p1 - 0.5) / span[0]) as f32, ((p2 - 0.5) / span[1]) as f32],
        });
      }
    }
    triangulate(&grid, n)
  }
}

fn triangulate(grid: &[SkyImageVertex], n: usize) -> Vec<SkyImageVertex> {
  let mut tris = Vec::with_capacity((n - 1) * (n - 1) * 6);
  for j in 0..n - 1 {
    for i in 0..n - 1 {
      let a = grid[j * n + i];
      let b = grid[j * n + i + 1];
      let c = grid[(j + 1) * n + i];
      let d = grid[(j + 1) * n + i + 1];
      tris.extend_from_slice(&[a, b, d, a, d, c]);
    }
  }
  tris
}

// ─── Allsky ──────────────────────────────────────────────────────────────────

/// `Norder{K}/Allsky.fits`: the mosaic of every order-`K` tile at reduced width.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HipsAllsky {
  pub order: u8,
  /// width of one tile block (`NAXIS1 / ncols`), a power of two
  pub tile_width: u32,
  /// tiles per row (`⌊√(12 · 4^order)⌋`)
  pub ncols: u32,
  /// rows (`⌈ntiles / ncols⌉`)
  pub nrows: u32,
  pub frame: HipsFrame,
  /// image size (`ncols · tile_width`, `nrows · tile_width`)
  pub nx: u32,
  pub ny: u32,
}

impl HipsAllsky {
  /// Tiles per row of an `Allsky` of `order`.
  pub fn columns_of(order: u8) -> u32 {
    (npix(order) as u64).isqrt() as u32
  }

  /// From the image size; `Err` when the size is not a mosaic of power-of-two blocks.
  pub fn from_dims(order: u8, nx: u32, ny: u32, frame: HipsFrame) -> Result<Self, String> {
    if order > HIPS_ALLSKY_MAX_ORDER {
      return Err(format!("Allsky order {order} above {HIPS_ALLSKY_MAX_ORDER}"));
    }
    let ncols = Self::columns_of(order);
    let ntiles = npix(order) as u32;
    let nrows = ntiles.div_ceil(ncols);
    if nx == 0 || nx % ncols != 0 {
      return Err(format!("Allsky NAXIS1 {nx} is not a multiple of {ncols} tiles"));
    }
    let tile_width = nx / ncols;
    if !tile_width.is_power_of_two() {
      return Err(format!("Allsky tile width {tile_width} is not a power of two"));
    }
    if ny != nrows * tile_width {
      return Err(format!("Allsky NAXIS2 {ny} is not {nrows} rows of {tile_width} px"));
    }
    Ok(Self {
      order,
      tile_width,
      ncols,
      nrows,
      frame,
      nx,
      ny,
    })
  }

  /// The tile of `npix`, its block's first FITS pixel `(p1, p2)` (1-based column / row of the
  /// block's bottom-left pixel) — tile `N` at column `N mod ncols`, row `⌊N / ncols⌋` from the top.
  pub fn block_of(&self, npix: u64) -> Option<(HipsTile, u32, u32)> {
    if npix >= self::npix(self.order) {
      return None;
    }
    let tile = HipsTile::new(self.order, npix, self.tile_width, self.frame).ok()?;
    let c = (npix % self.ncols as u64) as u32;
    let r = (npix / self.ncols as u64) as u32;
    let p1 = c * self.tile_width + 1;
    let p2 = self.ny - (r + 1) * self.tile_width + 1;
    Some((tile, p1, p2))
  }

  /// Tile and tile-local FITS pixel of a mosaic pixel; `None` in the unused blocks after the
  /// last tile.
  pub fn tile_of_pixel(&self, p1: f64, p2: f64) -> Option<(HipsTile, f64, f64)> {
    let w = self.tile_width as f64;
    let c = ((p1 - 0.5) / w).floor();
    let r = ((self.ny as f64 - (p2 - 0.5)) / w).floor();
    if c < 0.0 || r < 0.0 || c >= self.ncols as f64 || r >= self.nrows as f64 {
      return None;
    }
    let n = r as u64 * self.ncols as u64 + c as u64;
    let (tile, b1, b2) = self.block_of(n)?;
    Some((tile, p1 - (b1 as f64 - 1.0), p2 - (b2 as f64 - 1.0)))
  }

  /// ICRS unit vector of a mosaic pixel; `None` in the unused blocks.
  pub fn pixel_to_icrs(&self, p1: f64, p2: f64) -> Option<V3> {
    let (tile, l1, l2) = self.tile_of_pixel(p1, p2)?;
    Some(tile.pixel_to_icrs(l1, l2))
  }

  /// Mosaic pixel of an ICRS unit vector (always on the mosaic: the tiles cover the sky).
  pub fn icrs_to_pixel(&self, d: V3) -> Option<(f64, f64)> {
    let v = self.frame.from_icrs(d);
    let (lon, lat) = ra_dec_of(v);
    let (n, _, _) = lonlat_to_pix(self.order, lon, lat);
    let (tile, b1, b2) = self.block_of(n)?;
    let (l1, l2) = tile.icrs_to_pixel(d)?;
    Some((l1 + b1 as f64 - 1.0, l2 + b2 as f64 - 1.0))
  }

  /// Registration mesh: every tile block as an `n × n` patch (see [`HipsTile::mesh_vertices`]).
  pub fn mesh_vertices(&self, span: [f64; 2], n: usize) -> Vec<SkyImageVertex> {
    let n = n.max(2);
    let w = self.tile_width as f64;
    let mut out = Vec::with_capacity(npix(self.order) as usize * (n - 1) * (n - 1) * 6);
    let mut grid = Vec::with_capacity(n * n);
    for npix in 0..npix(self.order) {
      let Some((tile, b1, b2)) = self.block_of(npix) else {
        continue;
      };
      grid.clear();
      for j in 0..n {
        for i in 0..n {
          let l1 = 0.5 + w * i as f64 / (n - 1) as f64;
          let l2 = 0.5 + w * j as f64 / (n - 1) as f64;
          let d = equatorial_to_ecliptic_j2000(tile.pixel_to_icrs(l1, l2));
          let (p1, p2) = (l1 + b1 as f64 - 1.0, l2 + b2 as f64 - 1.0);
          grid.push(SkyImageVertex {
            pos: [d[0] as f32, d[1] as f32, d[2] as f32],
            uv: [((p1 - 0.5) / span[0]) as f32, ((p2 - 0.5) / span[1]) as f32],
          });
        }
      }
      out.extend(triangulate(&grid, n));
    }
    out
  }
}

// ─── identification ──────────────────────────────────────────────────────────

/// What a path / header identified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipsKind {
  /// `Norder{order}/Dir{d}/Npix{npix}.fits`
  Tile { order: u8, npix: u64 },
  /// `Norder{order}/Allsky.fits`
  Allsky { order: u8 },
}

/// Where the identification came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipsIdentSource {
  /// `ORDER` + `NPIX` header cards
  Header,
  /// the file path components
  Path,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HipsIdent {
  pub kind: HipsKind,
  pub source: HipsIdentSource,
  /// the HiPS root directory (the parent of `Norder{k}`), when the path has one
  pub root: Option<String>,
}

/// Path components split on `/`, `\` and `_` (the corpus flattens directory names with `_`).
fn path_tokens(path: &str) -> Vec<&str> {
  path.split(['/', '\\', '_']).filter(|t| !t.is_empty()).collect()
}

/// `{prefix}{number}` → number.
fn numbered<'a>(token: &'a str, prefix: &str) -> Option<u64> {
  let rest = token.get(..prefix.len())?.eq_ignore_ascii_case(prefix).then(|| &token[prefix.len()..])?;
  if rest.is_empty() || !rest.bytes().all(|b| b.is_ascii_digit()) || rest.len() > 18 {
    return None;
  }
  rest.parse().ok()
}

/// The HiPS root of a tile path: everything before the `Norder{k}` component (only when the
/// components are real path segments).
fn root_of(path: &str) -> Option<String> {
  let bytes = path.as_bytes();
  let mut start = 0usize;
  let mut i = 0usize;
  while i <= bytes.len() {
    if i == bytes.len() || bytes[i] == b'/' || bytes[i] == b'\\' {
      let seg = &path[start..i];
      let is_norder = seg
        .split('_')
        .next()
        .and_then(|t| numbered(t, "Norder"))
        .is_some()
        && seg.len() > 6
        && seg[..6].eq_ignore_ascii_case("Norder");
      if is_norder && start > 0 {
        return Some(path[..start - 1].to_string());
      }
      if is_norder {
        return None;
      }
      start = i + 1;
    }
    i += 1;
  }
  None
}

/// Identifies a HiPS tile / `Allsky` from the header (`ORDER`, `NPIX`) or the path.
pub fn identify(path: Option<&str>, header: &FitsHeader) -> Option<HipsIdent> {
  if let (Some(order), Some(np)) = (header.get_i64("ORDER"), header.get_i64("NPIX"))
    && (0..=HIPS_MAX_ORDER as i64).contains(&order)
    && np >= 0
    && (np as u64) < npix(order as u8)
  {
    return Some(HipsIdent {
      kind: HipsKind::Tile {
        order: order as u8,
        npix: np as u64,
      },
      source: HipsIdentSource::Header,
      root: path.and_then(root_of),
    });
  }
  let path = path?;
  let tokens = path_tokens(path);
  let mut order: Option<u8> = None;
  let mut dir: Option<u64> = None;
  let mut kind = None;
  for (i, t) in tokens.iter().enumerate() {
    if let Some(k) = numbered(t, "Norder") {
      if k <= HIPS_MAX_ORDER as u64 {
        order = Some(k as u8);
        dir = None;
      }
      continue;
    }
    let Some(order_k) = order else {
      continue;
    };
    if let Some(d) = numbered(t, "Dir") {
      dir = Some(d);
      continue;
    }
    // file name: Allsky.fits / Npix{n}.fits / Npix{n}_{frame}.fits
    let stem = t.split('.').next().unwrap_or("");
    if stem.eq_ignore_ascii_case("Allsky") {
      if order_k <= HIPS_ALLSKY_MAX_ORDER {
        kind = Some(HipsKind::Allsky { order: order_k });
      }
      break;
    }
    if let Some(n) = numbered(stem, "Npix") {
      let dir_ok = dir.is_none_or(|d| d == (n / 10000) * 10000);
      if n < npix(order_k) && dir_ok {
        kind = Some(HipsKind::Tile { order: order_k, npix: n });
      }
      break;
    }
    // any other component after Norder{k} that is not Dir: keep looking only if it was a
    // flattened prefix part (no slashes) — a real sub-directory ends the search
    let _ = i;
  }
  Some(HipsIdent {
    kind: kind?,
    source: HipsIdentSource::Path,
    root: root_of(path),
  })
}

// ─── properties ──────────────────────────────────────────────────────────────

/// The HiPS `properties` file (standard §4.4): `key = value` lines, `#` comments.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct HipsProperties {
  pub map: BTreeMap<String, String>,
}

impl HipsProperties {
  pub fn parse(text: &str) -> Self {
    let mut map = BTreeMap::new();
    for line in text.lines() {
      let line = line.trim();
      if line.is_empty() || line.starts_with('#') {
        continue;
      }
      let Some((k, v)) = line.split_once('=') else {
        continue;
      };
      let k = k.trim();
      if !k.is_empty() && !map.contains_key(k) {
        map.insert(k.to_string(), v.trim().to_string());
      }
    }
    Self { map }
  }

  pub fn get(&self, key: &str) -> Option<&str> {
    self.map.get(key).map(String::as_str)
  }

  pub fn frame(&self) -> Option<HipsFrame> {
    self.get("hips_frame").and_then(HipsFrame::parse)
  }

  /// `hips_tile_width` (default 512 when absent, §4.4.2)
  pub fn tile_width(&self) -> Option<u32> {
    self.get("hips_tile_width").and_then(|s| s.trim().parse().ok())
  }

  /// `hips_order` (the deepest order of the survey)
  pub fn order(&self) -> Option<u8> {
    self.get("hips_order").and_then(|s| s.trim().parse().ok())
  }

  /// Reads `<root>/properties`; `None` when the file is missing or unreadable.
  pub fn load(root: &str) -> Option<Self> {
    let path = format!("{root}/properties");
    let m = aethervk_oshal_rlib::os::files::Mmap::open(path.as_str()).ok()?;
    let text = core::str::from_utf8(m.as_ref()).ok()?;
    Some(Self::parse(text))
  }
}

/// Frame of a tile: the `properties` file (when `root` is known), else the header (`COORDSYS`,
/// `HIPSFRAM`, `FRAME`), else equatorial. Returns the frame and its source
/// (`"properties"`, `"header"`, `"assumed"`).
pub fn frame_of(root: Option<&str>, properties: Option<&HipsProperties>, header: &FitsHeader) -> (HipsFrame, &'static str) {
  if let Some(f) = properties.and_then(|p| p.frame()) {
    return (f, "properties");
  }
  if let Some(f) = root.and_then(HipsProperties::load).and_then(|p| p.frame()) {
    return (f, "properties");
  }
  for key in ["HIPSFRAM", "COORDSYS", "FRAME"] {
    if let Some(f) = header.get(key).and_then(HipsFrame::parse) {
      return (f, "header");
    }
  }
  (HipsFrame::Equatorial, "assumed")
}

// ─── the mapping object ──────────────────────────────────────────────────────

/// Pixel ↔ sky of an image: the API a generic WCS module can also implement (ICRS unit
/// vectors; FITS 1-based pixel coordinates).
pub trait PixelSkyMap {
  /// ICRS unit vector of a pixel; `None` where the image has no sky (unused Allsky blocks).
  fn pixel_to_icrs(&self, p1: f64, p2: f64) -> Option<V3>;
  /// Pixel of an ICRS unit vector; `None` off the image.
  fn icrs_to_pixel(&self, d: V3) -> Option<(f64, f64)>;
}

/// A HiPS image's mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HipsImage {
  Tile(HipsTile),
  Allsky(HipsAllsky),
}

impl PixelSkyMap for HipsImage {
  fn pixel_to_icrs(&self, p1: f64, p2: f64) -> Option<V3> {
    match self {
      Self::Tile(t) => Some(t.pixel_to_icrs(p1, p2)),
      Self::Allsky(a) => a.pixel_to_icrs(p1, p2),
    }
  }

  fn icrs_to_pixel(&self, d: V3) -> Option<(f64, f64)> {
    match self {
      Self::Tile(t) => t.icrs_to_pixel(d),
      Self::Allsky(a) => a.icrs_to_pixel(d),
    }
  }
}

impl HipsImage {
  pub fn frame(&self) -> HipsFrame {
    match self {
      Self::Tile(t) => t.frame,
      Self::Allsky(a) => a.frame,
    }
  }

  /// ICRS RA / Dec (degrees) of a pixel; `None` off the sky.
  pub fn pixel_to_radec_deg(&self, p1: f64, p2: f64) -> Option<(f64, f64)> {
    self.pixel_to_icrs(p1, p2).map(ra_dec_of)
  }

  /// Pixel of ICRS RA / Dec (degrees).
  pub fn radec_to_pixel(&self, ra_deg: f64, dec_deg: f64) -> Option<(f64, f64)> {
    self.icrs_to_pixel(dir_of_ra_dec(ra_deg, dec_deg))
  }

  /// Registration mesh (ecliptic J2000 unit directions, texture coordinates over `span` source
  /// pixels): `sky_image::grid_vertices`' counterpart. `n` vertices per side of a tile (the
  /// Allsky gets `n` per block, clamped to 2..=8).
  pub fn mesh_vertices(&self, span: [f64; 2], n: usize) -> Vec<SkyImageVertex> {
    match self {
      Self::Tile(t) => t.mesh_vertices(span, n),
      Self::Allsky(a) => a.mesh_vertices(span, n.clamp(2, 8)),
    }
  }

  /// Ecliptic J2000 unit direction of the image centre (tile centre; Allsky: the north pole of
  /// its frame, a mosaic has no centre on the sky).
  pub fn center_direction_ecliptic(&self) -> V3 {
    match self {
      Self::Tile(t) => {
        let (ra, dec) = t.center_radec_deg();
        equatorial_to_ecliptic_j2000(dir_of_ra_dec(ra, dec))
      }
      Self::Allsky(a) => equatorial_to_ecliptic_j2000(a.frame.to_icrs([0.0, 0.0, 1.0])),
    }
  }
}

/// Header facts of a HiPS image (for the facts JSON / UI).
#[derive(Debug, Clone, PartialEq)]
pub struct HipsFacts {
  /// `true` for `Allsky.fits`
  pub allsky: bool,
  pub order: u8,
  /// tile index (`None` for an Allsky)
  pub npix: Option<u64>,
  /// pixels per tile side (the Allsky's block width)
  pub tile_width: u32,
  pub frame: HipsFrame,
  /// `"properties"`, `"header"`, `"assumed"`
  pub frame_source: &'static str,
  /// `"header"` (`ORDER` / `NPIX` cards) or `"path"`
  pub ident_source: &'static str,
  /// `hips_order` of the survey's `properties`
  pub survey_order: Option<u8>,
  /// `obs_title` / `obs_collection` of the `properties`
  pub survey_title: Option<String>,
  /// the HiPS root directory
  pub root: Option<String>,
  /// ICRS centre of the tile (`None` for an Allsky)
  pub center_ra_deg: Option<f64>,
  pub center_dec_deg: Option<f64>,
  /// angular size of a tile pixel, arcsec (`HEALPix` mean resolution at `order + S`)
  pub pixel_scale_arcsec: f64,
}

impl HipsFacts {
  /// JSON object text of the facts (`fits_metadata::{json_f64, json_str}` conventions).
  pub fn to_json(&self) -> String {
    use super::fits_metadata::{json_f64, json_str};
    let mut s = String::from("{");
    s.push_str(&format!("\"hips\":true,\"allsky\":{},\"order\":{},", self.allsky, self.order));
    s.push_str(&format!(
      "\"npix\":{},\"tile_width\":{},",
      self.npix.map(|n| n.to_string()).unwrap_or_else(|| "null".to_string()),
      self.tile_width
    ));
    s.push_str("\"frame\":");
    json_str(&mut s, self.frame.name());
    s.push_str(",\"frame_source\":");
    json_str(&mut s, self.frame_source);
    s.push_str(",\"ident_source\":");
    json_str(&mut s, self.ident_source);
    s.push_str(&format!(
      ",\"survey_order\":{},",
      self.survey_order.map(|o| o.to_string()).unwrap_or_else(|| "null".to_string())
    ));
    s.push_str("\"survey_title\":");
    match &self.survey_title {
      Some(t) => json_str(&mut s, t),
      None => s.push_str("null"),
    }
    s.push_str(",\"root\":");
    match &self.root {
      Some(t) => json_str(&mut s, t),
      None => s.push_str("null"),
    }
    s.push_str(&format!(
      ",\"center_ra_deg\":{},\"center_dec_deg\":{},\"pixel_scale_arcsec\":{}}}",
      json_f64(self.center_ra_deg),
      json_f64(self.center_dec_deg),
      json_f64(Some(self.pixel_scale_arcsec))
    ));
    s
  }
}

/// Mean angular size of a HEALPix cell at `order`, arcsec (`√(4π / npix)`).
pub fn cell_size_arcsec(order: u8) -> f64 {
  (4.0 * PI / npix(order) as f64).sqrt() / DEG * 3600.0
}

/// Recognises a HiPS FITS tile / Allsky and builds its mapping and facts. `Ok(None)` when the
/// file is not a HiPS tile; `Err` when it is one whose size contradicts the HiPS layout.
pub fn load_mapping(
  path: Option<&str>,
  header: &FitsHeader,
  nx: u32,
  ny: u32,
) -> Result<Option<(HipsImage, HipsFacts)>, String> {
  let Some(ident) = identify(path, header) else {
    return Ok(None);
  };
  let props = ident.root.as_deref().and_then(HipsProperties::load);
  let (frame, frame_source) = frame_of(None, props.as_ref(), header);
  let ident_source = match ident.source {
    HipsIdentSource::Header => "header",
    HipsIdentSource::Path => "path",
  };
  let survey_order = props.as_ref().and_then(|p| p.order());
  let survey_title = props
    .as_ref()
    .and_then(|p| p.get("obs_title").or_else(|| p.get("obs_collection")))
    .map(str::to_string);
  match ident.kind {
    HipsKind::Tile { order, npix } => {
      if nx != ny {
        return Err(format!("HiPS tile {nx} × {ny} is not square"));
      }
      if let Some(w) = props.as_ref().and_then(|p| p.tile_width())
        && w != nx
      {
        return Err(format!("HiPS tile is {nx} px wide, the properties say hips_tile_width = {w}"));
      }
      let tile = HipsTile::new(order, npix, nx, frame)?;
      let (ra, dec) = tile.center_radec_deg();
      let facts = HipsFacts {
        allsky: false,
        order,
        npix: Some(npix),
        tile_width: nx,
        frame,
        frame_source,
        ident_source,
        survey_order,
        survey_title,
        root: ident.root,
        center_ra_deg: Some(ra),
        center_dec_deg: Some(dec),
        pixel_scale_arcsec: cell_size_arcsec(tile.pixel_order()),
      };
      Ok(Some((HipsImage::Tile(tile), facts)))
    }
    HipsKind::Allsky { order } => {
      let a = HipsAllsky::from_dims(order, nx, ny, frame)?;
      let facts = HipsFacts {
        allsky: true,
        order,
        npix: None,
        tile_width: a.tile_width,
        frame,
        frame_source,
        ident_source,
        survey_order,
        survey_title,
        root: ident.root,
        center_ra_deg: None,
        center_dec_deg: None,
        pixel_scale_arcsec: cell_size_arcsec(order + a.tile_width.trailing_zeros() as u8),
      };
      Ok(Some((HipsImage::Allsky(a), facts)))
    }
  }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
  use super::*;
  use std::{eprintln, fs, path::PathBuf, println};

  const TOL_DEG: f64 = 1e-9;

  fn header(cards: &[(&str, &str)]) -> FitsHeader {
    FitsHeader {
      cards: cards.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
    }
  }

  /// angular distance (degrees) between two (lon, lat) in degrees
  fn sep_deg(a: (f64, f64), b: (f64, f64)) -> f64 {
    let (u, v) = (dir_of_lonlat(a.0, a.1), dir_of_lonlat(b.0, b.1));
    let c = cross(u, v);
    dot(c, c).sqrt().atan2(dot(u, v)) / DEG
  }

  /// deterministic pseudo-random u64 stream
  fn lcg(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *seed >> 11
  }

  #[test]
  fn hips_base_pixel_centres_nside_1() {
    let lat_cap = (2.0f64 / 3.0).asin() / DEG; // 41.8103149°
    for f in 0..12u64 {
      let (lon, lat) = pix_to_lonlat(0, f);
      let (elon, elat) = match f {
        0..=3 => (45.0 + 90.0 * f as f64, lat_cap),
        4..=7 => (90.0 * (f - 4) as f64, 0.0),
        _ => (45.0 + 90.0 * (f - 8) as f64, -lat_cap),
      };
      assert!((lon - elon).abs() < TOL_DEG && (lat - elat).abs() < TOL_DEG, "face {f}: ({lon}, {lat}) vs ({elon}, {elat})");
      assert_eq!(nest_to_xyf(0, f), (f as u8, 0, 0));
      assert_eq!(xyf_to_nest(0, f as u8, 0, 0), f);
      assert_eq!(nest_to_ring(0, f), f, "nested = ring at nside 1");
    }
    // corners of the base faces: S, E, N, W
    let c = pix_corners(0, 0);
    assert!(sep_deg(c[0], (45.0, 0.0)) < TOL_DEG, "south {:?}", c[0]);
    assert!(sep_deg(c[1], (90.0, lat_cap)) < TOL_DEG, "east {:?}", c[1]);
    assert!((c[2].1 - 90.0).abs() < TOL_DEG, "north {:?}", c[2]);
    assert!(sep_deg(c[3], (0.0, lat_cap)) < TOL_DEG, "west {:?}", c[3]);
    let c = pix_corners(0, 4);
    assert!(sep_deg(c[0], (0.0, -lat_cap)) < TOL_DEG, "south {:?}", c[0]);
    assert!(sep_deg(c[1], (45.0, 0.0)) < TOL_DEG, "east {:?}", c[1]);
    assert!(sep_deg(c[2], (0.0, lat_cap)) < TOL_DEG, "north {:?}", c[2]);
    assert!(sep_deg(c[3], (315.0, 0.0)) < TOL_DEG, "west {:?}", c[3]);
  }

  #[test]
  fn hips_bit_interleave_round_trips() {
    let mut seed = 7u64;
    for order in 0..=20u8 {
      for _ in 0..200 {
        let ip = lcg(&mut seed) % npix(order);
        let (f, x, y) = nest_to_xyf(order, ip);
        assert!(f < 12 && (x as u64) < nside(order) && (y as u64) < nside(order));
        assert_eq!(xyf_to_nest(order, f, x, y), ip);
      }
    }
    // the first sub-cells of face 0 at order 1
    assert_eq!(nest_to_xyf(1, 0), (0, 0, 0));
    assert_eq!(nest_to_xyf(1, 1), (0, 1, 0));
    assert_eq!(nest_to_xyf(1, 2), (0, 0, 1));
    assert_eq!(nest_to_xyf(1, 3), (0, 1, 1));
  }

  #[test]
  fn hips_nested_centres_match_the_ring_closed_forms() {
    let mut seed = 11u64;
    for order in 0..=8u8 {
      let n = npix(order);
      let count = if n <= 3072 { n } else { 3000 };
      let mut seen = std::collections::BTreeSet::new();
      for k in 0..count {
        let ip = if n <= 3072 { k } else { lcg(&mut seed) % n };
        let ring = nest_to_ring(order, ip);
        assert!(ring < n);
        if n <= 3072 {
          assert!(seen.insert(ring), "nest → ring not injective at order {order}");
        }
        let (z, phi) = ring_pix_to_loc(order, ring);
        let a = lonlat_of_loc(z, phi);
        let b = pix_to_lonlat(order, ip);
        assert!(sep_deg(a, b) < TOL_DEG, "order {order} ipix {ip}: nested {b:?} ring {a:?}");
      }
    }
  }

  #[test]
  fn hips_lonlat_to_pix_inverts_the_centres_and_sub_positions() {
    let mut seed = 3u64;
    for order in 0..=12u8 {
      for _ in 0..300 {
        let ip = lcg(&mut seed) % npix(order);
        let dx = (lcg(&mut seed) % 1000) as f64 / 1000.0 + 0.0005;
        let dy = (lcg(&mut seed) % 1000) as f64 / 1000.0 + 0.0005;
        let (lon, lat) = pix_to_lonlat_at(order, ip, dx, dy);
        let (back, bx, by) = lonlat_to_pix(order, lon, lat);
        assert_eq!(back, ip, "order {order} ipix {ip} ({lon}, {lat}) dx {dx} dy {dy} → {back} ({bx}, {by})");
        let scale = nside(order) as f64;
        assert!((bx - dx).abs() * scale < 1e-6 && (by - dy).abs() * scale < 1e-6, "order {order}: ({bx}, {by}) vs ({dx}, {dy})");
      }
    }
    // poles and the longitude seam
    for (lon, lat) in [(0.0, 90.0), (123.0, -90.0), (359.9999999, 0.0), (0.0, 0.0), (0.0, 41.8), (180.0, -41.9)] {
      let (ip, _, _) = lonlat_to_pix(5, lon, lat);
      let c = pix_to_lonlat(5, ip);
      assert!(sep_deg(c, (lon, lat)) < 2.0 * cell_size_arcsec(5) / 3600.0, "({lon}, {lat}) → {ip} {c:?}");
    }
  }

  #[test]
  fn hips_corners_of_low_order_pixels() {
    // every corner of a cell at order k is a cell corner at order k+1 (the N corner of the
    // (1,1) child, etc.) and the four children tile the parent
    for order in 0..3u8 {
      for ip in 0..npix(order) {
        let c = pix_corners(order, ip);
        let kids: Vec<_> = (0..4).map(|k| pix_corners(order + 1, ip * 4 + k)).collect();
        assert!(sep_deg(c[0], kids[0][0]) < TOL_DEG, "south corner");
        assert!(sep_deg(c[1], kids[1][1]) < TOL_DEG, "east corner (child ix = 1)");
        assert!(sep_deg(c[2], kids[3][2]) < TOL_DEG, "north corner (child 3)");
        assert!(sep_deg(c[3], kids[2][3]) < TOL_DEG, "west corner (child iy = 1)");
        // the parent centre is the N corner of child 0 = S corner of child 3
        let centre = pix_to_lonlat(order, ip);
        assert!(sep_deg(centre, kids[0][2]) < TOL_DEG && sep_deg(centre, kids[3][0]) < TOL_DEG);
      }
    }
    // an equatorial-face cell at order 3: corners differ from the centre by the cell's extent
    let c = pix_corners(3, 4 * 64 + 21);
    let centre = pix_to_lonlat(3, 4 * 64 + 21);
    for k in c {
      let d = sep_deg(k, centre);
      assert!(d > 3.0 && d < 6.0, "corner distance {d}");
    }
  }

  #[test]
  fn hips_tile_round_trip_order_3_width_64() {
    let tile = HipsTile::new(3, 450, 64, HipsFrame::Equatorial).unwrap();
    assert_eq!(tile.pixel_order(), 9);
    for y in 0..64u32 {
      for x in 0..64u32 {
        let (p1, p2) = (x as f64 + 1.0, y as f64 + 1.0);
        let cell = tile.pixel_cell(x, y);
        // the pixel centre is the cell centre
        let (ra, dec) = tile.pixel_to_radec_deg(p1, p2);
        let c = pix_to_lonlat(9, cell);
        assert!(sep_deg((ra, dec), c) < TOL_DEG, "pixel ({x}, {y}) cell {cell}: {:?} vs {c:?}", (ra, dec));
        // the centre maps into the cell at the pixel order and back onto the pixel
        let (back, _, _) = lonlat_to_pix(9, ra, dec);
        assert_eq!(back, cell);
        assert_eq!(back >> 12, 450, "cell lies in the tile");
        let (q1, q2) = tile.radec_to_pixel(ra, dec).expect("on the tile");
        assert!((q1 - p1).abs() < 1e-7 && (q2 - p2).abs() < 1e-7, "({p1}, {p2}) → ({q1}, {q2})");
        // the convention: ix = W − p2 on the even bits, iy = p1 − 1 on the odd bits
        let (_, ix, iy) = nest_to_xyf(9, cell);
        assert_eq!((ix, iy), ((64 - 1 - y) + (nest_to_xyf(3, 450).1 * 64), x + nest_to_xyf(3, 450).2 * 64));
      }
    }
    // a direction off the tile
    let (ra, dec) = tile.center_radec_deg();
    assert!(tile.radec_to_pixel(ra + 60.0, -dec).is_none());
    // corners of the tile are the FITS image's corners: S = top-left (p1 0.5, p2 W+0.5),
    // E = bottom-left, N = bottom-right, W = top-right
    let c = tile.corners_radec_deg();
    assert!(sep_deg(c[0], tile.pixel_to_radec_deg(0.5, 64.5)) < TOL_DEG, "south");
    assert!(sep_deg(c[1], tile.pixel_to_radec_deg(0.5, 0.5)) < TOL_DEG, "east");
    assert!(sep_deg(c[2], tile.pixel_to_radec_deg(64.5, 0.5)) < TOL_DEG, "north");
    assert!(sep_deg(c[3], tile.pixel_to_radec_deg(64.5, 64.5)) < TOL_DEG, "west");
  }

  #[test]
  fn hips_tile_frames_convert_to_icrs() {
    // galactic centre in ICRS: astropy's Galactic is defined through FK5 J2000 and carries the
    // ~23 mas FK5 → ICRS frame bias; the ICRS angles used here (Hipparcos) do not, hence 36 mas
    let gc = galactic_to_icrs([1.0, 0.0, 0.0]);
    let (ra, dec) = ra_dec_of(gc);
    assert!((ra - 266.40498829).abs() < 1e-5 && (dec + 28.93617776).abs() < 1e-5, "GC {ra} {dec}");
    let ngp = galactic_to_icrs([0.0, 0.0, 1.0]);
    let (ra, dec) = ra_dec_of(ngp);
    assert!((ra - 192.85948).abs() < 1e-9 && (dec - 27.12825).abs() < 1e-9, "NGP {ra} {dec}");
    // the Hipparcos matrix (vol. 1, eq. 1.5.11)
    let m = galactic_matrix();
    let hip = [
      [-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
      [0.4941094278755837, -0.4448296299600112, 0.7469822444972189],
      [-0.8676661490190047, -0.1980763734312015, 0.4559837761750669],
    ];
    for i in 0..3 {
      for j in 0..3 {
        assert!((m[i][j] - hip[i][j]).abs() < 1e-9, "m[{i}][{j}] = {} vs {}", m[i][j], hip[i][j]);
      }
    }
    // round trips
    let mut seed = 5u64;
    for frame in [HipsFrame::Equatorial, HipsFrame::Galactic, HipsFrame::Ecliptic] {
      for _ in 0..50 {
        let lon = (lcg(&mut seed) % 360000) as f64 / 1000.0;
        let lat = (lcg(&mut seed) % 180000) as f64 / 1000.0 - 90.0;
        let v = dir_of_lonlat(lon, lat);
        let back = frame.from_icrs(frame.to_icrs(v));
        assert!(sep_deg(ra_dec_of(back), (lon, lat)) < 1e-10);
      }
    }
    // a galactic tile's centre is the cell centre rotated (through the WCS module's galactic
    // frame, astropy's: FK5 J2000 definition plus the frame bias, 36 mas from the matrix above)
    let t = HipsTile::new(3, 450, 64, HipsFrame::Galactic).unwrap();
    let (lon, lat) = pix_to_lonlat(3, 450);
    let expect = ra_dec_of(HipsFrame::Galactic.to_icrs(dir_of_lonlat(lon, lat)));
    assert!(sep_deg(expect, ra_dec_of(galactic_to_icrs(dir_of_lonlat(lon, lat)))) < 2e-5);
    assert!(sep_deg(t.center_radec_deg(), expect) < TOL_DEG);
    let (p1, p2) = t.radec_to_pixel(expect.0, expect.1).unwrap();
    assert!((p1 - 32.5).abs() < 1e-7 && (p2 - 32.5).abs() < 1e-7, "({p1}, {p2})");
  }

  #[test]
  fn hips_identifies_tiles_and_allsky_from_paths_and_headers() {
    let h = FitsHeader::default();
    let id = identify(Some("/data/hips/AllWISE/W1/Norder3/Dir0/Npix450.fits"), &h).unwrap();
    assert_eq!(id.kind, HipsKind::Tile { order: 3, npix: 450 });
    assert_eq!(id.source, HipsIdentSource::Path);
    assert_eq!(id.root.as_deref(), Some("/data/hips/AllWISE/W1"));
    let id = identify(Some("C:\\hips\\Norder6\\Dir10000\\Npix10302.fits"), &h).unwrap();
    assert_eq!(id.kind, HipsKind::Tile { order: 6, npix: 10302 });
    assert_eq!(id.root.as_deref(), Some("C:\\hips"));
    // flattened corpus names
    let id = identify(Some("test_assets/fits_corpus/hips/AllWISE_W1_Norder3_Dir0_Npix300.fits"), &h).unwrap();
    assert_eq!(id.kind, HipsKind::Tile { order: 3, npix: 300 });
    assert_eq!(id.root, None);
    let id = identify(Some("hips/DSS2_red_Norder3_Allsky.fits"), &h).unwrap();
    assert_eq!(id.kind, HipsKind::Allsky { order: 3 });
    let id = identify(Some("/srv/dss/Norder3/Allsky.fits"), &h).unwrap();
    assert_eq!(id.kind, HipsKind::Allsky { order: 3 });
    assert_eq!(id.root.as_deref(), Some("/srv/dss"));
    // cube frame suffix, other FITS extensions
    let id = identify(Some("x/Norder4/Dir0/Npix12_3.fits"), &h).unwrap();
    assert_eq!(id.kind, HipsKind::Tile { order: 4, npix: 12 });
    let id = identify(Some("x/Norder4/Dir0/Npix12.fits.fz"), &h).unwrap();
    assert_eq!(id.kind, HipsKind::Tile { order: 4, npix: 12 });
    // inconsistent Dir, out-of-range index, no Allsky above order 3
    assert!(identify(Some("x/Norder6/Dir0/Npix10302.fits"), &h).is_none());
    assert!(identify(Some("x/Norder3/Dir0/Npix768.fits"), &h).is_none());
    assert!(identify(Some("x/Norder4/Allsky.fits"), &h).is_none());
    assert!(identify(Some("/images/m31.fits"), &h).is_none());
    assert!(identify(None, &h).is_none());
    // header cards win
    let hh = header(&[("ORDER", "5"), ("NPIX", "2741")]);
    let id = identify(Some("/images/m31.fits"), &hh).unwrap();
    assert_eq!(id.kind, HipsKind::Tile { order: 5, npix: 2741 });
    assert_eq!(id.source, HipsIdentSource::Header);
    let id = identify(None, &hh).unwrap();
    assert_eq!(id.kind, HipsKind::Tile { order: 5, npix: 2741 });
  }

  #[test]
  fn hips_properties_and_frame_sources() {
    let p = HipsProperties::parse(
      "# comment\ncreator_did = ivo://CDS/P/allWISE/W1\nhips_order = 8\nhips_frame = galactic\nhips_tile_width = 512\nobs_title=AllWISE W1\nbroken line\n",
    );
    assert_eq!(p.frame(), Some(HipsFrame::Galactic));
    assert_eq!(p.order(), Some(8));
    assert_eq!(p.tile_width(), Some(512));
    assert_eq!(p.get("obs_title"), Some("AllWISE W1"));
    let h = FitsHeader::default();
    assert_eq!(frame_of(None, Some(&p), &h), (HipsFrame::Galactic, "properties"));
    assert_eq!(frame_of(None, None, &h), (HipsFrame::Equatorial, "assumed"));
    assert_eq!(frame_of(None, None, &header(&[("COORDSYS", "G")])), (HipsFrame::Galactic, "header"));
    assert_eq!(frame_of(None, None, &header(&[("COORDSYS", "E")])), (HipsFrame::Ecliptic, "header"));
    assert_eq!(frame_of(None, None, &header(&[("HIPSFRAM", "equatorial")])), (HipsFrame::Equatorial, "header"));
    // a properties file on disk next to a tile tree
    let dir = std::env::temp_dir().join(format!("aethervk_hips_props_{}", std::process::id()));
    let tile_dir = dir.join("Norder3").join("Dir0");
    fs::create_dir_all(&tile_dir).unwrap();
    fs::write(dir.join("properties"), "hips_frame = ecliptic\nhips_order = 3\nhips_tile_width = 64\n").unwrap();
    let tile_path = tile_dir.join("Npix7.fits");
    let path = tile_path.to_str().unwrap();
    let (img, facts) = load_mapping(Some(path), &h, 64, 64).unwrap().expect("a tile");
    assert_eq!(img.frame(), HipsFrame::Ecliptic);
    assert_eq!(facts.frame_source, "properties");
    assert_eq!(facts.survey_order, Some(3));
    assert_eq!((facts.order, facts.npix, facts.tile_width), (3, Some(7), 64));
    // the properties' tile width contradicts the image
    assert!(load_mapping(Some(path), &h, 512, 512).is_err());
    let _ = fs::remove_dir_all(&dir);
  }

  #[test]
  fn hips_allsky_layout_order_3() {
    let a = HipsAllsky::from_dims(3, 1728, 1856, HipsFrame::Equatorial).unwrap();
    assert_eq!((a.ncols, a.nrows, a.tile_width), (27, 29, 64));
    assert!(HipsAllsky::from_dims(3, 1728, 1792, HipsFrame::Equatorial).is_err());
    assert!(HipsAllsky::from_dims(3, 1700, 1856, HipsFrame::Equatorial).is_err());
    assert_eq!(HipsAllsky::columns_of(0), 3);
    assert_eq!(HipsAllsky::columns_of(1), 6);
    assert_eq!(HipsAllsky::columns_of(2), 13);
    // tile 0 is the top-left block (the last stored rows), tile 450 at column 18, row 16
    let (t0, p1, p2) = a.block_of(0).unwrap();
    assert_eq!((t0.npix, p1, p2), (0, 1, 1856 - 64 + 1));
    let (t, p1, p2) = a.block_of(450).unwrap();
    assert_eq!((t.npix, p1, p2), (450, 18 * 64 + 1, 1856 - 17 * 64 + 1));
    assert!(a.block_of(768).is_none());
    // the unused blocks of the last row (768 = 28 · 27 + 12: columns 12.. of row 28 are empty)
    assert!(a.pixel_to_icrs(12.0 * 64.0 + 1.0, 1.0).is_none());
    assert!(a.pixel_to_icrs(11.0 * 64.0 + 1.0, 1.0).is_some());
    // every block pixel maps like the stand-alone tile, and back
    let tile = HipsTile::new(3, 450, 64, HipsFrame::Equatorial).unwrap();
    for (x, y) in [(0u32, 0u32), (63, 0), (0, 63), (63, 63), (17, 40)] {
      let (l1, l2) = (x as f64 + 1.0, y as f64 + 1.0);
      let (g1, g2) = (l1 + (p1 - 1) as f64, l2 + (p2 - 1) as f64);
      let d = a.pixel_to_icrs(g1, g2).unwrap();
      let e = tile.pixel_to_icrs(l1, l2);
      assert!(sep_deg(ra_dec_of(d), ra_dec_of(e)) < TOL_DEG);
      let (q1, q2) = a.icrs_to_pixel(d).unwrap();
      assert!((q1 - g1).abs() < 1e-6 && (q2 - g2).abs() < 1e-6, "({g1}, {g2}) → ({q1}, {q2})");
    }
    // the mosaic covers the sky
    let mut seed = 9u64;
    for _ in 0..500 {
      let lon = (lcg(&mut seed) % 360000) as f64 / 1000.0;
      let lat = (lcg(&mut seed) % 180000) as f64 / 1000.0 - 90.0;
      let (p1, p2) = a.icrs_to_pixel(dir_of_lonlat(lon, lat)).expect("on the mosaic");
      assert!((0.5..=1728.5).contains(&p1) && (0.5..=1856.5).contains(&p2));
      let back = a.pixel_to_icrs(p1, p2).unwrap();
      assert!(sep_deg(ra_dec_of(back), (lon, lat)) < 1e-7, "({lon}, {lat}) → ({p1}, {p2}) → {:?}", ra_dec_of(back));
    }
    let (_, facts) = load_mapping(Some("/x/Norder3/Allsky.fits"), &FitsHeader::default(), 1728, 1856).unwrap().unwrap();
    assert!(facts.allsky && facts.npix.is_none() && facts.tile_width == 64);
    assert!(facts.to_json().contains("\"allsky\":true"));
  }

  #[test]
  fn hips_mesh_vertices_cover_the_tile() {
    let tile = HipsTile::new(3, 450, 512, HipsFrame::Equatorial).unwrap();
    let v = tile.mesh_vertices([512.0, 512.0], 32);
    assert_eq!(v.len(), 31 * 31 * 6);
    for x in &v {
      let l = (x.pos[0] * x.pos[0] + x.pos[1] * x.pos[1] + x.pos[2] * x.pos[2]).sqrt();
      assert!((l - 1.0).abs() < 1e-5);
      assert!((0.0..=1.0).contains(&x.uv[0]) && (0.0..=1.0).contains(&x.uv[1]));
    }
    // the first vertex is the FITS corner (0.5, 0.5): the tile's east corner
    let e = equatorial_to_ecliptic_j2000(dir_of_lonlat(tile.corners_radec_deg()[1].0, tile.corners_radec_deg()[1].1));
    assert!((v[0].pos[0] as f64 - e[0]).abs() < 1e-5 && (v[0].pos[2] as f64 - e[2]).abs() < 1e-5);
    let a = HipsAllsky::from_dims(3, 1728, 1856, HipsFrame::Equatorial).unwrap();
    let v = HipsImage::Allsky(a).mesh_vertices([1728.0, 1856.0], 4);
    assert_eq!(v.len(), 768 * 9 * 6);
    let img = HipsImage::Tile(tile);
    let c = img.center_direction_ecliptic();
    assert!((dot(c, c) - 1.0).abs() < 1e-12);
    let (ra, dec) = img.pixel_to_radec_deg(256.5, 256.5).unwrap();
    assert!(sep_deg((ra, dec), tile.center_radec_deg()) < TOL_DEG);
  }

  /// Reference tables written with astropy-healpix (`test_assets/fits_corpus/hips_reference/
  /// *.tsv`): `tiles.tsv` (order, npix, centre, corners c0..c3 = S, E, N, W) and
  /// `tile_pixels_w{W}.tsv` (order, npix, width, x, y, nested, centre, corners). Missing tables
  /// are reported and skipped.
  #[test]
  fn hips_matches_the_reference_tables() {
    let dir = aethervk_test_assets::test_asset_dir!("fits_corpus/hips_reference");
    let Ok(entries) = fs::read_dir(&dir) else {
      println!("no HiPS reference tables at {}", dir.display());
      return;
    };
    let mut files: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e == "tsv")).collect();
    files.sort();
    if files.is_empty() {
      println!("no .tsv in {}", dir.display());
      return;
    }
    let mut checked = 0usize;
    for file in &files {
      let text = fs::read_to_string(file).unwrap();
      let mut columns: Vec<String> = Vec::new();
      let mut rows = 0usize;
      for line in text.lines() {
        if let Some(rest) = line.strip_prefix('#') {
          let rest = rest.trim();
          if rest.starts_with("order\t") || rest.starts_with("order ") {
            columns = rest.split('\t').map(|c| c.trim().to_string()).collect();
          }
          continue;
        }
        if line.trim().is_empty() {
          continue;
        }
        if columns.is_empty() {
          // a header line without '#'
          if line.starts_with("order") {
            columns = line.split('\t').map(|c| c.trim().to_string()).collect();
            continue;
          }
          panic!("{}: no column header before data", file.display());
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), columns.len(), "{}: {line}", file.display());
        let col = |name: &str| -> Option<&str> { columns.iter().position(|c| c == name).map(|i| f[i].trim()) };
        let num = |name: &str| -> f64 { col(name).unwrap_or_else(|| panic!("column {name}")).parse().unwrap() };
        let order: u8 = col("order").unwrap().parse().unwrap();
        let npix: u64 = col("npix").unwrap().parse().unwrap();
        let (cell_order, cell) = match (col("width"), col("x"), col("y")) {
          (Some(w), Some(x), Some(y)) => {
            let (w, x, y): (u32, u32, u32) = (w.parse().unwrap(), x.parse().unwrap(), y.parse().unwrap());
            let tile = HipsTile::new(order, npix, w, HipsFrame::Equatorial).unwrap();
            let cell = tile.pixel_cell(x, y);
            if let Some(n) = col("nested") {
              assert_eq!(cell, n.parse::<u64>().unwrap(), "{}: {line}", file.display());
            }
            // the tile object agrees with the raw HEALPix call
            let (ra, dec) = tile.pixel_to_radec_deg(x as f64 + 1.0, y as f64 + 1.0);
            assert!(sep_deg((ra, dec), (num("lon"), num("lat"))) < TOL_DEG, "{}: {line}", file.display());
            (tile.pixel_order(), cell)
          }
          _ => (order, npix),
        };
        let c = pix_to_lonlat(cell_order, cell);
        assert!(sep_deg(c, (num("lon"), num("lat"))) < TOL_DEG, "{}: centre {c:?} vs {line}", file.display());
        if col("c0_lon").is_some() {
          let corners = pix_corners(cell_order, cell);
          for (k, corner) in corners.iter().enumerate() {
            let want = (num(&format!("c{k}_lon")), num(&format!("c{k}_lat")));
            assert!(sep_deg(*corner, want) < TOL_DEG, "{}: corner {k} {corner:?} vs {want:?} in {line}", file.display());
          }
        }
        if let Some(r) = col("ring") {
          assert_eq!(nest_to_ring(cell_order, cell), r.parse::<u64>().unwrap(), "{}: {line}", file.display());
        }
        rows += 1;
      }
      eprintln!("{}: {rows} rows checked", file.display());
      checked += rows;
    }
    assert!(checked > 0, "reference tables present but empty");
  }
}
