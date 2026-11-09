# FITS test corpus

Proves "the app opens any FITS image" on real files, without the files living in the repo. Only
small files are committed here; the images are downloaded.

- `manifest.tsv`: one corpus file per row: `path sha256 size source category expected notes`.
  - `source`: an `http(s)://` URL, or `tar:<archive path>` (extracted from that archive row into
    `<category>/<archive stem>/`).
  - `category`: `astropy-data` (data.astropy.org), `nasa-samples` (science.data.nasa.gov FITS
    samples and the older fits.gsfc.nasa.gov copies, deduplicated by sha256), `heasarc` (HEASARC
    sample files), `calabretta` (the Calabretta & Greisen 1904-66 image in every projection),
    `hips` (CDS HiPS tiles mirrored by IRSA, some gzip-compressed under a `.fits` name), `cds`
    (the CDS fits-rs test set, rows added once it downloads), `wcs_extra` (the WCS forms wcslib
    does not read: the FITS registry TNX / ZPX samples, DSS plate-solution cutouts from STScI and
    ESO — their servers stamp `DATE`, so a fresh download has another sha256 — and helioprojective
    STEREO/SECCHI and SDO/AIA images).
  - `expected`: `image` (must open: `Ok`, `nx, ny > 0`, a finite pixel; cubes are `image`, the
    note says `cube`), `refuse:<text>` (`Err` containing the text), `table-only` (`Err`),
    `either` (deliberately malformed file: `Ok` or `Err`, never a panic), `archive` (not opened).
    Set from the astropy inventory (`scripts/fits_corpus_inventory.py`).
  - `notes`: the image HDU, wrapper (gzip / bzip2 / `.Z`), BITPIX, CTYPEs, frame.
- `wcs_reference/<name>.hdr` + `.tsv`: astropy 8.0.2 / wcslib 8.6 pix → world on a 7×7 grid (corners
  included) and at CRPIX, with the ICRS equivalent and astropy's own round trip; `nan` = outside
  the projection. 28 Calabretta headers (every projection incl. NCP) and 50 synthetic ones (CD,
  PC+CDELT, CDELT+CROTA2, skew, flips, DEC-first axes, CUNIT arcsec, pole, RA wrap, FK4 / FK4-NO-E
  B1950, FK5, ICRS, defaults by EQUINOX / EPOCH / none, galactic, ecliptic, SIP with and without
  AP/BP, TPV, PV of ZPN AZP SZP SIN COD COE COO COP BON CEA AIR HPX, XPH, LONPOLE / LATPOLE, PV1_1/2,
  and a HiPS tile as an HPX WCS). Ecliptic frames: astropy has none for ELON/ELAT, the ICRS columns
  use `BarycentricMeanEcliptic` of EQUINOX (IAU 2006 obliquity). FK4 uses obstime = B1950.
  Forms wcslib does not read have their own oracle (`gen_wcs_reference.py --extra`, the `# oracle:`
  comment of each table): `tnx_*` / `zpx_*` wcstools 3.9.7 `xy2sky -n 14` (the IRAF
  definition; AST agrees only for square half-cross-term surfaces) — the FITS registry samples
  (`wcs_extra/tnx.fits`, `zpx.fits`) and 28 synthetic headers (Chebyshev / Legendre / polynomial ×
  no / full / half cross terms × 4x4, 3x5, 5x3; a ZPX with projp0..5); `dss_*` starlink-pyast
  (AST 9) FitsChan on the three `wcs_extra` DSS cutouts with their GetImage TAN cards removed
  (wcstools cross-check ≤ 2e-6″; AST reports FK5 J2000); `hpc_*` sunpy 8 on the SECCHI COR2 (TAN)
  and HI1 (AZP) and AIA (TAN) samples: pixel → (θx, θy) by astropy.wcs, ICRS as the direction
  (point − observer) in HCRS, `# icrs_tol_deg: 3e-5` (0.1″) for the engine's truncated Earth
  ephemeris (the files' own alternate celestial WCS agree to 3–5″).
- `hips_reference/`: HEALPix nested references (astropy_healpix) for HiPS: `tiles.tsv` (orders
  0–3: centre + 4 corners of each tile), `tile_pixels_w512.tsv` / `_w64.tsv` (sample pixels of a
  tile: their nested index at order + log2 W, centre and 4 corners). Corners are HEALPix face
  offsets `(dx, dy)` = (0,0) (1,0) (1,1) (0,1).

## HiPS tile pixel convention

Pixel `(x, y)` of a FITS tile (`x` column along NAXIS1, `y` row along NAXIS2, both 0-based, row 0 =
first stored row) of tile `N` at order `K`, width `W = 2^S`, is the HEALPix cell of order `K + S`

    nested = N * 4^S + interleave(ix, iy),  ix = W - 1 - y (even bits), iy = x (odd bits)

The HiPS 1.0 standard only draws this (figure 4). It was checked on a real HipsGen tile
(DSS2 red, order 6, tile 47100, `hips/dss2red_Norder6_Npix47100.fits`) against the independent DSS
image `astropy-data/.../tailored_dss.22.29.38.50-20.50.13_60arcmin.fits`: correlation 0.898 for this
mapping, below 0.30 for the 7 other flips / transposes; reproject 0.21's HPX tile WCS gives the same
0.898. Reproduce with `gen_wcs_reference.py --hips-check`. JPEG / PNG tiles store rows top-down,
i.e. flipped in `y` relative to FITS.

## Commands

    scripts/fetch_fits_corpus.sh ~/HDD2/aethervk-fits-corpus            # verify / download all
    scripts/fetch_fits_corpus.sh <dir> --category calabretta,hips       # a subset
    scripts/fetch_fits_corpus.sh <dir> --cds                            # + the CDS tar (~2.2 GB)

    AETHERVK_FITS_CORPUS=<dir> cargo test -p aethervk-core-rlib --features cfitsio --lib \
        fits_corpus_opens_every_image -- --nocapture

| variable | effect |
| --- | --- |
| `AETHERVK_FITS_CORPUS` | corpus directory; unset: the corpus test prints `skipped` and passes |
| `AETHERVK_FITS_CORPUS_FILTER` | substring of the manifest path, or a glob when it holds `*` / `?` |
| `AETHERVK_FITS_CORPUS_CATEGORY` | comma list of categories |
| `AETHERVK_FITS_CORPUS_WCS_STRICT=1` | a WCS the engine refuses is a failure (default: reported) |
| `AETHERVK_FITS_CORPUS_VERBOSE=1` | one line per file |

`fits_corpus_manifest_is_consistent`, `fits_wcs_reference_tables_match` and
`fits_hips_reference_tables_consistent` need no corpus and run in CI.

Regenerating (lab venv `~/aethervk-lab/fits-corpus/venv`: astropy, numpy, astropy-healpix,
reproject; never a build dependency; `-I` so nothing is imported from the corpus):

    venv/bin/python -I scripts/fits_corpus_inventory.py <dir>          # <dir>/INVENTORY.tsv
    venv/bin/python -I scripts/gen_wcs_reference.py --corpus <dir> --hips
    venv/bin/python -I scripts/gen_wcs_reference.py --corpus <dir> --extra --wcstools-bin <dir with xy2sky, sky2xy>
    venv/bin/python -I scripts/fits_corpus_inventory.py <dir> --manifest test_assets/fits_corpus/manifest.tsv \
        --new-rows cds_rows.tsv --category cds --source-prefix tar:cds/fits-rs-test-files.tar

Downloaded but not in the manifest (not FITS): the science.data.nasa.gov `.list` / `.lst` / `.vf`
(fitsverify) / `.txt` / `.pdf` / `.ps` / `.tex` / `.doc` / `.htm` companions and its six `.tar`
files (split HST archives that need their bundled C merge tool; the same data is in the `.fit.gz`
rows).
