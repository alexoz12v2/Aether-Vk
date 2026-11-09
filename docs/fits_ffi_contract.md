# FITS texture assets: FFI contract

FITS images (`.fit`, `.fits`, `.fts`, `.fz`, any case) import through the existing asset library entry
point `avkSimulationContext_importAsset` like PNG / JPEG and become **RGBA8 texture assets**.
Everything that already reads texture assets (thumbnails, `avkSimulationContext_copyAssetTexels`,
desktop billboards, exports, content hash) works unchanged. The two functions below are FITS
specific; they live in `aethervk.core/cdylib/src/ffi_fits.rs`.

## Import behaviour

- The first HDU holding a real 2-D image: the primary HDU, or with the engine's `cfitsio`
  feature (the default build) the first image HDU after skipping `NAXIS = 0` primaries, 1-D and
  zero-size images and tables: an `IMAGE` extension or a tile-compressed `.fz` image (`hdu` /
  `compressed` in the JSON). A cube (`NAXIS > 2` with axes longer than 1) imports as its first
  2-D plane (`planes` = the product of the extra axes, `plane` = 0; `planeAxes` lists the extra
  axes when there are two or more) and `avkSimulationContext_setFitsPlane` switches to any other
  plane by its flat index (`NAXIS3` fastest: `i3 + NAXIS3 · i4 …`); singleton extra axes count
  as a plain image. Both builds read cubes. `BITPIX` 8 / 16 / 32 / 64 / −32 / −64, physical value `BZERO + BSCALE · raw`
  (unsigned integers through `BZERO` 32768 / 2147483648 / 2⁶³, signed bytes through −128), integer
  `BLANK` and IEEE NaN are *blank* (CFITSIO also reads ±Inf as blank). The raster is `f32`:
  64-bit integers beyond 2²⁴ keep the nearest float.
  - Tile compression (CFITSIO build): RICE_1, GZIP_1, GZIP_2, HCOMPRESS_1 (lossless or lossy,
    with the `SMOOTH` option), PLIO_1, NOCOMPRESS; quantised floats with any `ZQUANTIZ`
    (`NO_DITHER`, `SUBTRACTIVE_DITHER_1`, `SUBTRACTIVE_DITHER_2`), lossy integers, `ZBLANK`.
    The JSON `cards` are the header of the image the table stands for (`BITPIX`, `NAXISn`,
    `BLANK`; no `Z` keywords).
  - Whole-file compression (CFITSIO build): a gzip stream (`.fits.gz`, magic `1f 8b`; CFITSIO's
    zlib), a bzip2 stream (`.fits.bz2`, `BZh`; the libbz2 CFITSIO is built with) and a Unix
    `compress` stream (`.Z`, `1f 9d`; a pure-Rust LZW decoder) are inflated in memory first,
    capped at 3 GiB of output, whatever the file is named. The pure-Rust build refuses all
    three with a message naming the compression (`… decompress it first`).
  - A data unit written without its last 2880-byte padding (CDS HiPS tiles) is read whole; a
    `BLANK` written as a float (`BLANK = -32768.0`, the same tiles) counts as its integer.
  - Refused with a message: a file with tables only (`no image HDU (tables only)`), a
    random-groups file (`GROUPS = T`, `NAXIS1 = 0`), a side over 2²⁰ pixels (`bad image size`),
    a plane over 2²⁹ pixels (`image too large`: 2 GB of 16-bit pixels, 4 GB of decoded `f32`),
    an uncompressed data unit the file does not hold (`FITS data truncated`).
  - Header leniency, as CFITSIO: `HIERARCH` keywords (the card's key is the long name without the
    prefix), `CONTINUE` cards (kept as cards; the value of the first card ends in `&`),
    `COMMENT` / `HISTORY` in file order, lowercase keywords (kept as written), trailing bytes after
    the last data block, a missing `EXTEND`.
  A build without the feature reads the primary HDU only (no extensions, cubes, compression) and
  refuses the rest with the messages above where they apply.
- **Dates**: the first readable card of `DATE-AVG`, `MJD-AVG`, `DATE-OBS`, `MJD-OBS`, `MJD`,
  `JD`, `DATE-BEG`, `DATE_OBS`, `MJD_OBS`, `JD-OBS`, `JUL-DATE`, `DATE` (the file date, flagged).
  Date strings are normalised before parsing: quotes / blanks, `UTC` / `Z` / `z` and a zero offset
  dropped (a non-zero offset is refused, never shifted), `t` or a blank between date and time →
  `T`, `dd/mm/yyyy` → ISO, `dd/mm/yy` → `20yy` (the FITS standard says `19yy`, but the app's users
  image comets now; the card is flagged `dateTwoDigitYear` and the UI offers `19yy`, shifting the
  date back 100 years), `hh:mm` → `hh:mm:00`, a leap second `:60` → `:59.999999999`. A date-only value takes its time from `TIME-OBS`, `UTSTART`,
  `UT`, `TIME-BEG`, `UTC-OBS` (`HH:MM:SS[.f]`, `HH MM SS`, decimal hours, trailing `Z` / `+00:00`);
  with none readable it stays at midnight (`dateOnly`). Both builds share this code.
- **Orientation**: FITS row 0 is the bottom of the image. The texture is flipped so that texture
  row 0 (the first row `copyAssetTexels` returns) is the **last** FITS row: the texture shows the
  image as a FITS viewer does (north up / east left for a standard WCS). Exception: a header with
  the non-standard `ROWORDER = 'TOP-DOWN'` is not flipped (`flipped: false` in the JSON).
- **Default stretch**: zscale (IRAF, ≤ 600 × 600 grid samples, contrast 0.25), linear transfer.
- Texels: grey `R = G = B`, `A = 255`; blank pixels are black (`0, 0, 0, 255`) and excluded from
  every statistic. A run of one exact value at the image minimum / maximum holding >= 1 % of the
  samples (e.g. Astroart writes -38288 into uncovered stack borders without a `BLANK` card) is
  treated as a fill value and also left out of the zscale / percentile statistics.
- The f32 raster is cached (mapped cache file), so a re-stretch never re-reads the source file.
- Asset identity (`getAssetContentHash`) is fixed at import (default stretch) and does **not**
  change with re-stretches; the stretch is not persisted by the engine (store it in the savefile
  on the C# side and re-apply it after re-import).
- Import errors: a malformed FITS fails the import (`AssetImported` failure; message
  `could not decode FITS: …`).

## Functions

```c
// Header metadata of a FITS texture as UTF-8 JSON, NUL-terminated into `out` (truncated to
// `out_len` on a UTF-8 boundary; out = NULL only sizes). `*out_needed` (nullable) receives the
// full byte length excluding the NUL. Returns false when the asset is unknown, is not a FITS
// texture, or while any writer (an import, an unload, a re-stretch) holds the library: never cache a
// false as "not FITS".
bool avkSimulationContext_getFitsMetadataJson(
    SimulationContext* ctx,
    uint64_t asset_id,
    char* out,
    uint32_t out_len,
    uint32_t* out_needed);

// Re-renders the RGBA8 texels of a FITS texture from its cached raster.
//   mode:     0 = zscale (p0, p1 ignored)
//             1 = percentile (p0 = low %, p1 = high %, both in 0..=100, e.g. 0.5 / 99.5)
//             2 = manual (p0 = black, p1 = white, data units; p0 > p1 renders inverted)
//   transfer: 0 = linear, 1 = log (log(1 + 1000 x) / log(1001)), 2 = asinh (asinh(x/0.1) / asinh(10))
// `*out_black` / `*out_white` (nullable) receive the black / white points used (data units).
// Returns false for an unknown / non-FITS asset, unknown codes, non-finite or out-of-range
// parameters or a cache write failure. Blocks on the library's write lock: call it off the UI thread.
// After success, copyAssetTexels / copyAssetThumbnail return the new image (same size).
bool avkSimulationContext_setFitsStretch(
    SimulationContext* ctx,
    uint64_t asset_id,
    uint32_t mode,
    uint32_t transfer,
    double p0,
    double p1,
    double* out_black,
    double* out_white);

// Shows plane `plane` of a cube FITS texture (0-based flat index over the extra axes, NAXIS3
// fastest; the JSON's "planes" counts them, "planeAxes" gives the axes when there are two or
// more). The source file is read again for that plane; the texels and the raster are replaced,
// the stretch mode / transfer is kept (black / white recomputed on the plane, so a manual stretch
// keeps its points). Afterwards getFitsMetadataJson says "plane": plane and copyAssetTexels /
// copyFitsRaster return that plane (same size). Returns false for an unknown / non-FITS asset,
// plane >= planes, a source file that is gone or unreadable, or a cache write failure (the asset
// is then unchanged). Blocks on the library's write lock: call it off the UI thread.
bool avkSimulationContext_setFitsPlane(
    SimulationContext* ctx,
    uint64_t asset_id,
    uint64_t plane);

// The physical f32 raster (BZERO + BSCALE · raw, blank = NaN) in texture row order (row 0 = the
// top of the displayed image, like copyAssetTexels), for measurements on the data rather than on
// the stretched texels (the C# nucleus detector). out = NULL only sizes (*out_width ·
// *out_height floats); out_len counts floats. Returns the copyAssetTexels status codes:
// 0 ok, 1 invalid argument, 2 unknown asset (also: not FITS), 4 buffer too small, 5 busy.
int32_t avkSimulationContext_copyFitsRaster(
    SimulationContext* ctx,
    uint64_t asset_id,
    float* out,
    uint64_t out_len,
    uint32_t* out_width,
    uint32_t* out_height);
```

C# (same style as `avkSimulationContext_getCompatibilityJson` in
`aethervk.ui-logic/Services/INativeRuntimeService.cs`; Rust `bool` is one byte):

```csharp
[DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
[return: MarshalAs(UnmanagedType.U1)]
public static extern unsafe bool avkSimulationContext_getFitsMetadataJson(
  nint ctx, ulong assetId, byte* outUtf8, uint outLen, uint* outNeeded);

[DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
[return: MarshalAs(UnmanagedType.U1)]
public static extern unsafe bool avkSimulationContext_setFitsStretch(
  nint ctx, ulong assetId, uint mode, uint transfer, double p0, double p1,
  double* outBlack, double* outWhite);

[DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
[return: MarshalAs(UnmanagedType.U1)]
public static extern bool avkSimulationContext_setFitsPlane(nint ctx, ulong assetId, ulong plane);
```

Usage: call `getFitsMetadataJson(ctx, id, null, 0, &needed)`; a `false` result means "not a FITS
texture" (this doubles as the `IsFits` probe). Then allocate `needed + 1` bytes and call again.

## JSON schema

```jsonc
{
  "assetId": 7,
  "textureWidth": 1200, "textureHeight": 1200,
  "rowOrder": "topDown",          // always: texture row 0 is the top of the displayed image
  "flipped": true,                // rows were reversed (false only for ROWORDER = 'TOP-DOWN')
  "stretch": { "mode": 0, "transfer": 0, "black": -1004.76, "white": 1020.78 },
  "naxis1": 1200, "naxis2": 1200, "bitpix": 32,
  "hdu": 1,                       // 1-based HDU the image was read from (2+: an image extension)
  "compressed": false,            // tile-compressed image (.fz), read as the image it stands for
  "planes": 1,                    // 2-D planes in the HDU (a cube: NAXIS3 · NAXIS4 …); 1 for an image
  "plane": 0,                     // the plane the raster holds: 0 at import, setFitsPlane changes it
  // "planeAxes": [n3, n4, …]     // only with two or more extra axes: plane = i3 + n3 · i4 + n3 · n4 · i5 …
  "cards": [                      // every header card before END, file order, duplicates kept
    { "key": "SIMPLE", "value": "T", "comment": "A valid FITS file" },
    { "key": "OBJECT", "value": "67P", "comment": "Actual name of the current observation source" },
    { "key": "COMMENT", "value": "free text", "comment": "" }
    // strings are unquoted ('' → '), trailing blanks dropped; HIERARCH keys without the prefix
  ],
  "facts": {                      // null when missing
    "object": "67P", "telescope": "Liverpool Telescope", "instrument": "IO:O",
    "filter": "SDSS-G",           // FILTER, else FILTER1, else FILTNAME
    "exposureS": 20.0,            // EXPTIME, else EXPOSURE
    "dateObs": "2021-12-27T00:14:47.345",   // raw value of the date card used
    "dateSource": "DATE-OBS",     // DATE-AVG | MJD-AVG | DATE-OBS | MJD-OBS | MJD | JD | DATE-BEG
                                  // | DATE_OBS | MJD_OBS | JD-OBS | JUL-DATE | DATE (first readable)
    "dateIsMid": true,            // card already mid-exposure (DATE-AVG/MJD-AVG, or a DATE-OBS
                                  // whose comment says "averaged"/"mid"); else start + EXPTIME/2
    "dateMidUtc": "2021-12-27T00:14:47.345Z",
    "dateMidTdbCenturies": 0,     // hifitime Duration::to_parts of the TDB duration since J2000:
    "dateMidTdbNanoseconds": 0,   //   the (centuries, nanoseconds) pair the epoch FFI / CTime use
    "dateMidTdbUs": 0,            // TDB microseconds since J2000 TDB (seekEpochSync scale)
    "timesysAssumed": true,       // no TIMESYS: the date was read as UTC
    "pixelScaleArcsec": 0.30368,  // sqrt|det CD| (or CDELT matrix), else PIXSCALE, else SECPIX
    "pixelScaleXArcsec": 0.30368, "pixelScaleYArcsec": 0.30368,
    "pixelScaleSource": "CD",     // CD | CDELT | PIXSCALE | SECPIX
    "fovWidthArcmin": 6.07, "fovHeightArcmin": 6.07,
    "crvalRaDeg": 135.5634, "crvalDecDeg": 28.1569,   // CRVAL1/2 (celestial CTYPE1 only)
    "centerRaDeg": 135.6, "centerDecDeg": 28.1,       // image centre through the TAN WCS
    "pointingRaDeg": 135.56, "pointingDecDeg": 28.16, // RA/DEC (sexagesimal RA = hours) or OBJCTRA/OBJCTDEC
    "radesys": "FK5", "equinox": 2000.0,
    "wcsError": null,             // why the WCS was refused
    "rotationDeg": 0.136,         // position angle of image +Y (texture up), degrees E of N, (-180, 180]
    "mirrored": false,            // det CD > 0: east to the right with north up
    "siteLatDeg": 28.7624,
    "siteLonDeg": -17.8792,       // EAST positive (after the LST check / west-convention rule below)
    "siteElevM": null,            // SITEELEV | ALT-OBS | OBSGEO-H | HEIGHT | from OBSGEO-X/Y/Z
    "siteSource": "LATITUDE/LONGITUD", // SITELAT/SITELONG | LAT-OBS/LONG-OBS | OBSGEO-B/OBSGEO-L
                                  // | OBSGEO-X/Y/Z | LATITUDE/LONGITUD (first present)
    "siteLonWestConvention": false,// negated: comment mentions "west" (and not "east"), or value ends in W
    "siteLonRawDeg": -17.8792,    // as written in the card
    "siteLonFromLst": true,       // the header's LST at UTSTART decided the sign (overrides the wording)
    // date provenance (added later, so after the site keys; absent in JSON from older builds)
    "dateCardRaw": "DATE-OBS= '2021-12-27T00:14:47.345' / UT, averaged",
                                  // the card used, as `KEY     = 'value' / comment`; when no card
                                  // of the chain parses, the first one present (for the UI's hint);
                                  // null with no date card at all
    "dateOnly": false,            // the card had no time of day and no time card gave one: midnight
    "dateIsFileDate": false,      // the date is the file's DATE (when it was written)
    "dateStartUtc": "2021-12-27T00:14:47.345Z",  // start of the exposure: the card itself, or for a
                                  // mid-exposure card DATE-BEG / MJD-BEG, else DATE-OBS / MJD-OBS
                                  // (DATE-AVG / MJD-AVG), else mid − EXPTIME/2; an "averaged"
                                  // DATE-OBS keeps its own instant
    "dateStartTdbUs": 693749756529000, // start, TDB microseconds since J2000 TDB (seekEpochSync scale)
    "dateTwoDigitYear": false,    // the card wrote its year with two digits (dd/mm/yy), read as 20yy;
                                  // the UI offers 19yy (the whole date 100 years earlier)
    // WCS facts (added after the date keys; all null without a celestial WCS)
    "wcsProjection": "TAN",       // projection code (TAN, SIN, ZEA, CAR, …; NCP / GLS read as SIN / SFL)
    "wcsDistortion": "none",      // distortion applied: none | SIP | TPV | …
    "wcsFrame": "FK5",            // ICRS | FK5 | FK4 | FK4-NO-E | galactic | ecliptic
    "wcsEquinox": 2000.0,         // the frame's equinox (Julian years, Besselian for FK4); null for ICRS / galactic
    "wcsFrameAssumed": false      // no RADESYS / EQUINOX card named the frame (ICRS assumed)
  }
  // "hips": { … }                // only for a HiPS tile / Allsky (no WCS; recognised from ORDER / NPIX
                                  // or the Norder*/Dir*/Npix* import path), after "facts":
                                  // { "hips": true, "allsky", "order", "npix" (null for Allsky),
                                  //   "tile_width", "frame" (equatorial | galactic | ecliptic),
                                  //   "frame_source" (properties | header | assumed), "ident_source"
                                  //   (header | path), "survey_order", "survey_title", "root",
                                  //   "center_ra_deg", "center_dec_deg", "pixel_scale_arcsec" }
}
```

Numbers are JSON numbers (non-finite → `null`); the centuries / nanoseconds pair is exact
(`u64` nanoseconds may exceed 2^53: parse as `ulong`).

### Caveat: Liverpool Telescope longitude

The LT header writes `LONGITUD = -17.8792 / [degrees West]`, but La Palma is at 17.88° **W**, so
the value is already east-positive and the comment is wrong. When the header has an `LST` card,
the parser computes the GMST at `UTSTART` (or the observation start) and keeps the sign whose
local sidereal time matches within 3° while the other does not (`siteLonFromLst`); this file
therefore gets `siteLonDeg = -17.8792`. Without an LST the west/east wording decides. The UI still
shows the raw value and lets the user flip the sign.
