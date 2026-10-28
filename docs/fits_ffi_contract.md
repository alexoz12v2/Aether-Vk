# FITS texture assets: FFI contract

FITS images (`.fit`, `.fits`, `.fts`, any case) import through the existing asset library entry
point `avkSimulationContext_importAsset` like PNG / JPEG and become **RGBA8 texture assets**.
Everything that already reads texture assets (thumbnails, `avkSimulationContext_copyAssetTexels`,
desktop billboards, exports, content hash) works unchanged. The two functions below are FITS
specific; they live in `aethervk.core/cdylib/src/ffi_fits.rs`.

## Import behaviour

- Primary HDU only, `NAXIS = 2` (extra axes of length 1 allowed), `BITPIX` 8 / 16 / 32 / 64 /
  −32 / −64, physical value `BZERO + BSCALE · raw`, integer `BLANK` and IEEE NaN are *blank*.
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
    "dateSource": "DATE-OBS",     // DATE-AVG | MJD-AVG | DATE-OBS | MJD-OBS | MJD | JD
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
    "wcsError": null,             // why the TAN WCS was refused (non-TAN, FK4, …)
    "rotationDeg": 0.136,         // position angle of image +Y (texture up), degrees E of N, (-180, 180]
    "mirrored": false,            // det CD > 0: east to the right with north up
    "siteLatDeg": 28.7624,
    "siteLonDeg": -17.8792,       // EAST positive (after the LST check / west-convention rule below)
    "siteElevM": null,            // SITEELEV | ALT-OBS | OBSGEO-H | HEIGHT | from OBSGEO-X/Y/Z
    "siteSource": "LATITUDE/LONGITUD", // SITELAT/SITELONG | LAT-OBS/LONG-OBS | OBSGEO-B/OBSGEO-L
                                  // | OBSGEO-X/Y/Z | LATITUDE/LONGITUD (first present)
    "siteLonWestConvention": false,// negated: comment mentions "west" (and not "east"), or value ends in W
    "siteLonRawDeg": -17.8792,    // as written in the card
    "siteLonFromLst": true        // the header's LST at UTSTART decided the sign (overrides the wording)
  }
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
