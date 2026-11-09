#!/usr/bin/env python3
"""WCS and HiPS reference tables for the engine's FITS tests (lab tool, never a build dependency).

astropy.wcs wraps wcslib: it is the oracle. For every WCS header (the Calabretta & Greisen 1904-66
image in every projection, read from the corpus, plus the synthetic headers below) this writes

  test_assets/fits_corpus/wcs_reference/<name>.hdr   the header, one 80-column card per line
  test_assets/fits_corpus/wcs_reference/<name>.tsv   pix -> world on a 7x7 grid (corners
                                                     included) plus CRPIX, and the round trip

TSV columns (tab separated, `nan` = outside the projection's domain):
  x y            FITS pixel coordinates, 1-based (pixel centre of the first pixel = 1.0)
  lon lat        native celestial coordinates of the header's frame (deg), astropy all_pix2world
                 (SIP / TPV distortions included); lon in [0, 360)
  ra_icrs dec_icrs  the same point in ICRS (deg) through astropy.coordinates (= lon lat for ICRS)
  x_rt y_rt      astropy all_world2pix(lon, lat) (its own round trip, to size tolerances)
Comment lines (`# key: value`) give the frame astropy reports and the versions.

`--extra` (needs `--corpus` with its `wcs_extra/` samples and `--wcstools-bin`): the forms
astropy / wcslib does not read, each against its own independent oracle:
  TNX / ZPX   wcstools 3.9.7 `xy2sky -d -n 14` (the IRAF definition; sky2xy is a cross-check, its
              inverse closes to only ~1e-6 px, so x_rt / y_rt are left nan = the grid pixel; AST only
              agrees for square half-cross-term surfaces, it converts to TPN): the FITS registry
              samples tnx.fits / zpx.fits plus synthetic headers of every surface type
              (Chebyshev / Legendre / polynomial), cross-term mode (none / full / half) and order
              shape; the ZPX case carries projp0..5
  DSS         starlink-pyast (AST 9) FitsChan, the GetImage TAN cards stripped so the plate
              solution is what is read (AST reports FK5 J2000); wcstools xy2sky as a cross-check
  HPLN/HPLT   sunpy: pixel -> (Tx, Ty) through astropy.wcs, the ICRS direction as
              (point - observer) in HCRS; `# icrs_tol_deg:` loosens the ICRS check to 3e-5 deg
              (0.1") for the truncated Earth ephemeris of the engine

HiPS (`--hips`): test_assets/fits_corpus/hips_reference/
  tiles.tsv            order npix  centre and the four corners of the tile's HEALPix cell
  tile_pixels_w<W>.tsv order npix W x y nested(order+log2 W) centre + four corners of that pixel
Tile pixel convention (verified on a real HipsGen FITS tile, see `--hips-check` and README.md):
x = column (0-based, along NAXIS1), y = row (0-based, along NAXIS2, row 0 = first stored row);
the pixel holds the HEALPix cell of order K + S (W = 2^S) with nested index
  N * 4^S + interleave(ix, iy),  ix = W - 1 - y (even bits), iy = x (odd bits)
where (ix, iy) are HEALPix face coordinates (astropy_healpix dx along ix, dy along iy).
Corners are listed as (dx, dy) = (0,0) (1,0) (1,1) (0,1).

Usage (lab venv, -I so nothing is imported from the corpus):
  python3 -I scripts/gen_wcs_reference.py --corpus ~/HDD2/aethervk-fits-corpus [--out test_assets/fits_corpus]
  python3 -I scripts/gen_wcs_reference.py --hips [--out test_assets/fits_corpus]
  python3 -I scripts/gen_wcs_reference.py --hips-check <tile.fits> <image.fits> <order> <npix>
  python3 -I scripts/gen_wcs_reference.py --corpus <dir> --extra --wcstools-bin <dir with xy2sky, sky2xy>
"""
import argparse
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np

warnings.simplefilter("ignore")

from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
import astropy  # noqa: E402
from astropy.wcs import _wcs  # noqa: E402

GRID = 7
CALABRETTA = ["AZP", "SZP", "TAN", "STG", "SIN", "NCP", "ARC", "ZPN", "ZEA", "AIR", "CYP", "CEA", "CAR",
              "MER", "SFL", "PAR", "MOL", "AIT", "COD", "COE", "COO", "COP", "BON", "PCO", "TSC", "CSC",
              "QSC", "HPX"]


# ── synthetic headers ──────────────────────────────────────────────────────────────────────────
def base(nx, ny, ctype, crval, crpix=None, **cards):
    h = {"NAXIS": 2, "NAXIS1": nx, "NAXIS2": ny, "CTYPE1": ctype[0], "CTYPE2": ctype[1],
         "CRVAL1": crval[0], "CRVAL2": crval[1],
         "CRPIX1": crpix[0] if crpix else (nx + 1) / 2, "CRPIX2": crpix[1] if crpix else (ny + 1) / 2}
    h.update(cards)
    return h


def cd_matrix(scale_deg, rot_deg=0.0, east_left=True):
    # CD = R(rot) . diag(-s or s, s): standard north-up east-left image rotated by rot
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    m = np.array([[c, -s], [s, c]]) @ np.diag([-scale_deg if east_left else scale_deg, scale_deg])
    return {"CD1_1": m[0, 0], "CD1_2": m[0, 1], "CD2_1": m[1, 0], "CD2_2": m[1, 1]}


def sip_coeffs():
    a = {"A_ORDER": 3, "A_2_0": 2.1e-6, "A_1_1": -1.3e-6, "A_0_2": 7.0e-7, "A_3_0": 4.0e-10,
         "A_2_1": -2.5e-10, "A_1_2": 1.1e-10, "A_0_3": -3.0e-10}
    b = {"B_ORDER": 3, "B_2_0": -9.0e-7, "B_1_1": 1.7e-6, "B_0_2": -1.9e-6, "B_3_0": -1.5e-10,
         "B_2_1": 3.3e-10, "B_1_2": -2.0e-10, "B_0_3": 2.6e-10}
    return a, b


def fit_inverse_sip(h, order=3):
    """AP/BP by least squares: u = U + AP(U, V), v = V + BP(U, V) over the image."""
    a = {k: v for k, v in h.items() if k.startswith("A_") and k != "A_ORDER"}
    b = {k: v for k, v in h.items() if k.startswith("B_") and k != "B_ORDER"}
    nx, ny = h["NAXIS1"], h["NAXIS2"]
    u, v = np.meshgrid(np.linspace(1, nx, 60) - h["CRPIX1"], np.linspace(1, ny, 60) - h["CRPIX2"])
    u, v = u.ravel(), v.ravel()

    def poly(c, prefix):
        out = np.zeros_like(u)
        for k, val in c.items():
            _, i, j = k.split("_")
            out += val * u ** int(i) * v ** int(j)
        return out
    U, V = u + poly(a, "A"), v + poly(b, "B")
    terms = [(i, j) for i in range(order + 1) for j in range(order + 1 - i)]
    M = np.stack([U ** i * V ** j for i, j in terms], axis=1)
    ap = np.linalg.lstsq(M, u - U, rcond=None)[0]
    bp = np.linalg.lstsq(M, v - V, rcond=None)[0]
    out = {"AP_ORDER": order, "BP_ORDER": order}
    for (i, j), x in zip(terms, ap):
        out[f"AP_{i}_{j}"] = float(x)
    for (i, j), x in zip(terms, bp):
        out[f"BP_{i}_{j}"] = float(x)
    return out


def synthetic():
    s = {}
    px = 1.0 / 3600  # 1 arcsec
    # linear part variants
    s["tan_cd_icrs"] = base(1024, 768, ("RA---TAN", "DEC--TAN"), (150.1, 2.2), RADESYS="ICRS", **cd_matrix(0.6 * px, 23.0))
    s["tan_pc_cdelt_fk5"] = base(1024, 768, ("RA---TAN", "DEC--TAN"), (83.8, -5.4), RADESYS="FK5", EQUINOX=2000.0,
                                  CDELT1=-1.2 * px, CDELT2=1.2 * px, PC1_1=0.9659258263, PC1_2=-0.2588190451,
                                  PC2_1=0.2588190451, PC2_2=0.9659258263)
    s["tan_pc_skew"] = base(800, 600, ("RA---TAN", "DEC--TAN"), (10.68, 41.27), RADESYS="ICRS",
                            CDELT1=-2 * px, CDELT2=2.5 * px, PC1_1=1.0, PC1_2=0.12, PC2_1=-0.05, PC2_2=0.98)
    s["tan_cdelt_crota2"] = base(1000, 1000, ("RA---TAN", "DEC--TAN"), (201.37, -43.02), EQUINOX=2000.0,
                                 CDELT1=-1.5 * px, CDELT2=1.5 * px, CROTA2=-37.5)
    s["tan_cdelt_only_mirrored"] = base(512, 512, ("RA---TAN", "DEC--TAN"), (266.4, -29.0), RADESYS="ICRS",
                                        CDELT1=2 * px, CDELT2=2 * px)
    s["tan_cd_flipped_det_pos"] = base(640, 480, ("RA---TAN", "DEC--TAN"), (56.75, 24.12), RADESYS="ICRS",
                                       **cd_matrix(1.1 * px, -71.0, east_left=False))
    s["tan_dec_first_axis"] = base(600, 400, ("DEC--TAN", "RA---TAN"), (-60.8, 214.0), RADESYS="ICRS",
                                   CD1_1=0.0, CD1_2=1.0 * px * 3, CD2_1=-1.0 * px * 3, CD2_2=0.0)
    s["tan_crpix_outside"] = base(400, 300, ("RA---TAN", "DEC--TAN"), (120.0, 30.0), (-850.5, 1200.25), RADESYS="ICRS",
                                  **cd_matrix(5 * px, 12.0))
    s["tan_near_pole"] = base(2000, 2000, ("RA---TAN", "DEC--TAN"), (37.95, 89.26), RADESYS="ICRS", **cd_matrix(3 * px, 0.0))
    s["tan_ra_wrap"] = base(1200, 900, ("RA---TAN", "DEC--TAN"), (0.05, -12.0), RADESYS="ICRS", **cd_matrix(1.0 * px * 2, 5.0))
    s["tan_cunit_arcsec"] = base(512, 512, ("RA---TAN", "DEC--TAN"), (150.1 * 3600, 2.2 * 3600), RADESYS="ICRS",
                                 CUNIT1="arcsec", CUNIT2="arcsec", CDELT1=-0.5, CDELT2=0.5)
    s["tan_wide_field"] = base(4096, 4096, ("RA---TAN", "DEC--TAN"), (280.0, -60.0), RADESYS="ICRS", **cd_matrix(36 * px, 30.0))
    # frames
    s["tan_fk4_b1950"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.0, -5.4), RADESYS="FK4", EQUINOX=1950.0, **cd_matrix(1.7 * px, 0.0))
    s["tan_fk4_by_equinox"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.0, -5.4), EQUINOX=1950.0, **cd_matrix(1.7 * px, 0.0))
    s["tan_fk4_noe_b1950"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.0, -5.4), RADESYS="FK4-NO-E", EQUINOX=1950.0, **cd_matrix(1.7 * px, 0.0))
    s["tan_fk5_j2000"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.8, -5.4), RADESYS="FK5", EQUINOX=2000.0, **cd_matrix(1.7 * px, 0.0))
    s["tan_fk5_by_equinox"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.8, -5.4), EQUINOX=2000.0, **cd_matrix(1.7 * px, 0.0))
    s["tan_no_frame_cards"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.8, -5.4), **cd_matrix(1.7 * px, 0.0))
    s["tan_epoch_only_1950"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.0, -5.4), EPOCH=1950.0, **cd_matrix(1.7 * px, 0.0))
    s["galactic_tan"] = base(1000, 1000, ("GLON-TAN", "GLAT-TAN"), (30.0, 5.0), **cd_matrix(10 * px, 0.0))
    s["galactic_car_allsky"] = base(720, 360, ("GLON-CAR", "GLAT-CAR"), (0.0, 0.0), CDELT1=-0.5, CDELT2=0.5)
    s["galactic_ait_allsky"] = base(800, 400, ("GLON-AIT", "GLAT-AIT"), (0.0, 0.0), CDELT1=-0.45, CDELT2=0.45)
    s["ecliptic_tan"] = base(900, 900, ("ELON-TAN", "ELAT-TAN"), (100.0, -10.0), EQUINOX=2000.0, **cd_matrix(8 * px, 0.0))
    s["ecliptic_car"] = base(360, 180, ("ELON-CAR", "ELAT-CAR"), (180.0, 0.0), EQUINOX=2000.0, CDELT1=-1.0, CDELT2=1.0)
    # distortions
    a, b = sip_coeffs()
    s["sip_ab"] = base(2048, 2048, ("RA---TAN-SIP", "DEC--TAN-SIP"), (150.1, 2.2), RADESYS="ICRS", **cd_matrix(0.4 * px, 15.0), **a, **b)
    h = base(2048, 2048, ("RA---TAN-SIP", "DEC--TAN-SIP"), (150.1, 2.2), RADESYS="ICRS", **cd_matrix(0.4 * px, 15.0), **a, **b)
    h.update(fit_inverse_sip(h))
    s["sip_ab_apbp"] = h
    s["tpv"] = base(2048, 4096, ("RA---TPV", "DEC--TPV"), (36.0, -4.5), RADESYS="ICRS", **cd_matrix(0.263 * px, 0.0),
                    PV1_0=-0.0029, PV1_1=1.0127, PV1_2=-0.0109, PV1_4=0.0088, PV1_5=-0.0128, PV1_6=0.0068,
                    PV1_7=-0.0099, PV1_8=-0.0052, PV1_9=-0.0084, PV1_10=0.0011,
                    PV2_0=0.0024, PV2_1=1.0119, PV2_2=-0.0089, PV2_4=0.0102, PV2_5=-0.0115, PV2_6=0.0057,
                    PV2_7=-0.0088, PV2_8=-0.0047, PV2_9=-0.0091, PV2_10=0.0013)
    # projection parameters
    s["zpn_pv"] = base(600, 600, ("RA---ZPN", "DEC--ZPN"), (0.0, 90.0), RADESYS="ICRS", CDELT1=-0.1, CDELT2=0.1,
                       PV2_0=0.05, PV2_1=0.975, PV2_2=-0.807, PV2_3=0.337, PV2_4=-0.065, PV2_5=0.01)
    s["azp_pv"] = base(600, 600, ("RA---AZP", "DEC--AZP"), (45.0, 30.0), RADESYS="ICRS", CDELT1=-0.05, CDELT2=0.05, PV2_1=2.0, PV2_2=30.0)
    s["szp_pv"] = base(600, 600, ("RA---SZP", "DEC--SZP"), (45.0, 30.0), RADESYS="ICRS", CDELT1=-0.05, CDELT2=0.05,
                       PV2_1=2.0, PV2_2=180.0, PV2_3=60.0)
    s["sin_slant_pv"] = base(600, 600, ("RA---SIN", "DEC--SIN"), (80.0, 60.0), RADESYS="ICRS", CDELT1=-0.05, CDELT2=0.05,
                             PV2_1=0.1, PV2_2=-0.2)
    s["cod_pv"] = base(600, 400, ("RA---COD", "DEC--COD"), (30.0, 40.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2, PV2_1=45.0, PV2_2=25.0)
    s["coe_pv"] = base(600, 400, ("RA---COE", "DEC--COE"), (30.0, 40.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2, PV2_1=-45.0, PV2_2=25.0)
    s["coo_pv"] = base(600, 400, ("RA---COO", "DEC--COO"), (30.0, 40.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2, PV2_1=45.0, PV2_2=25.0)
    s["cop_pv"] = base(600, 400, ("RA---COP", "DEC--COP"), (30.0, 40.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2, PV2_1=45.0, PV2_2=25.0)
    s["bon_pv"] = base(600, 400, ("RA---BON", "DEC--BON"), (0.0, 0.0), RADESYS="ICRS", CDELT1=-0.5, CDELT2=0.5, PV2_1=45.0)
    s["cea_pv"] = base(720, 360, ("RA---CEA", "DEC--CEA"), (0.0, 0.0), RADESYS="ICRS", CDELT1=-0.5, CDELT2=0.5, PV2_1=0.75)
    s["air_pv"] = base(600, 600, ("RA---AIR", "DEC--AIR"), (0.0, -90.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2, PV2_1=45.0)
    s["hpx_h6_k5"] = base(720, 720, ("RA---HPX", "DEC--HPX"), (0.0, 0.0), RADESYS="ICRS", CDELT1=-0.5, CDELT2=0.5, PV2_1=6, PV2_2=5)
    s["xph"] = base(720, 720, ("RA---XPH", "DEC--XPH"), (0.0, 90.0), RADESYS="ICRS", CDELT1=-0.35, CDELT2=0.35)
    # LONPOLE / LATPOLE
    s["car_oblique_default_lonpole"] = base(400, 300, ("RA---CAR", "DEC--CAR"), (30.0, 40.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2)
    # theta0 = 0 cylindricals: LONPOLE = 30 / 60 leave two valid native poles, LATPOLE picks one
    s["car_oblique_lonpole30"] = base(400, 300, ("RA---CAR", "DEC--CAR"), (30.0, 40.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2, LONPOLE=30.0)
    s["cea_lonpole_latpole_pos"] = base(400, 300, ("RA---CEA", "DEC--CEA"), (120.0, 20.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2,
                                        PV2_1=1.0, LONPOLE=60.0, LATPOLE=60.0)
    s["cea_lonpole_latpole_neg"] = base(400, 300, ("RA---CEA", "DEC--CEA"), (120.0, 20.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2,
                                        PV2_1=1.0, LONPOLE=60.0, LATPOLE=-60.0)
    s["zea_lonpole60"] = base(500, 500, ("RA---ZEA", "DEC--ZEA"), (250.0, -35.0), RADESYS="ICRS", CDELT1=-0.1, CDELT2=0.1, LONPOLE=60.0)
    s["tan_lonpole_nondefault"] = base(800, 800, ("RA---TAN", "DEC--TAN"), (83.8, -5.4), RADESYS="ICRS", LONPOLE=200.0,
                                       **cd_matrix(1.7 * px, 0.0))
    s["sfl_pv1_offset"] = base(400, 300, ("RA---SFL", "DEC--SFL"), (30.0, 40.0), RADESYS="ICRS", CDELT1=-0.2, CDELT2=0.2,
                               PV1_1=10.0, PV1_2=20.0)
    # a HiPS tile as a WCS (reproject's HPX tile header: order 3, tile 300, 64 px)
    try:
        from astropy.coordinates import ICRS
        from reproject.hips._utils import tile_header_2d
        th = tile_header_2d(level=3, index=300, frame=ICRS(), tile_size=64)
        th = th[0] if isinstance(th, tuple) else th
        s["hips_tile_hpx_o3_n300_w64"] = {k: th[k] for k in th if k not in ("COMMENT", "HISTORY", "")}
    except Exception as e:  # noqa: BLE001
        print("skip hips_tile_hpx:", e, file=sys.stderr)
    return s


# ── table writing ──────────────────────────────────────────────────────────────────────────────
def header_text(h):
    lines = [c.image for c in h.cards]
    lines.append("END".ljust(80))
    return "\n".join(lines) + "\n"


def to_header(d):
    h = fits.Header()
    h["SIMPLE"] = True
    h["BITPIX"] = -32
    for k, v in d.items():
        if k in ("SIMPLE", "BITPIX"):
            continue
        h[k] = v
    return h


def frame_info(w):
    from astropy.wcs.utils import wcs_to_celestial_frame
    if w.wcs.lng >= 0 and w.wcs.ctype[w.wcs.lng][:4] == "ELON":
        # astropy maps ELON/ELAT to nothing (or, with EQUINOX, wrongly to FK5): explicit choice
        from astropy.coordinates import BarycentricMeanEcliptic
        eq = w.wcs.equinox if np.isfinite(w.wcs.equinox) else 2000.0
        f = BarycentricMeanEcliptic(equinox=f"J{eq:.1f}")
        return f, (f"{f!r} (chosen here: astropy has no WCS frame for ELON/ELAT; mean ecliptic and "
                   "equinox of EQUINOX, IAU 2006 obliquity)").replace("\n", " ")
    try:
        f = wcs_to_celestial_frame(w)
        return f, repr(f).replace("\n", " ")
    except Exception as e:  # noqa: BLE001
        ct = w.wcs.ctype[w.wcs.lng][:4] if w.wcs.lng >= 0 else ""
        if ct == "ELON":
            from astropy.coordinates import BarycentricMeanEcliptic
            eq = w.wcs.equinox if np.isfinite(w.wcs.equinox) else 2000.0
            f = BarycentricMeanEcliptic(equinox=f"J{eq:.1f}")
            return f, (f"{f!r} (astropy has no WCS mapping for ELON/ELAT: ICRS columns use "
                       "BarycentricMeanEcliptic of EQUINOX, IAU 2006 obliquity)").replace("\n", " ")
        return None, f"none ({e})"


def fmt(v):
    return "nan" if not np.isfinite(v) else f"{v:.12f}"


def write_case(name, header, out_dir, source):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    w = WCS(header)
    nx, ny = header["NAXIS1"], header["NAXIS2"]
    xs = np.linspace(1, nx, GRID)
    ys = np.linspace(1, ny, GRID)
    px = [(x, y) for y in ys for x in xs]
    px.append((float(w.wcs.crpix[0]), float(w.wcs.crpix[1])))
    x = np.array([p[0] for p in px])
    y = np.array([p[1] for p in px])
    try:
        lon, lat = w.all_pix2world(x, y, 1)
    except Exception:  # noqa: BLE001 - some points invalid: do them one by one
        lon, lat = np.full_like(x, np.nan), np.full_like(x, np.nan)
        for i in range(len(x)):
            try:
                lo, la = w.all_pix2world([x[i]], [y[i]], 1)
                lon[i], lat[i] = lo[0], la[0]
            except Exception:  # noqa: BLE001
                pass
    # the longitude axis may be the second one (DEC--TAN first): all_pix2world returns axis order
    if w.wcs.lng == 1:
        lon, lat = lat, lon
    lon = np.where(np.isfinite(lon), np.mod(lon, 360.0), np.nan)
    frame, frame_txt = frame_info(w)
    ra = np.full_like(x, np.nan)
    dec = np.full_like(x, np.nan)
    if frame is not None:
        ok = np.isfinite(lon) & np.isfinite(lat)
        if ok.any():
            c = SkyCoord(lon[ok] * u.deg, lat[ok] * u.deg, frame=frame).icrs
            ra[ok], dec[ok] = c.ra.deg, c.dec.deg
    xr = np.full_like(x, np.nan)
    yr = np.full_like(x, np.nan)
    for i in range(len(x)):
        if not (np.isfinite(lon[i]) and np.isfinite(lat[i])):
            continue
        a1, a2 = (lon[i], lat[i]) if w.wcs.lng == 0 else (lat[i], lon[i])
        try:
            r = w.all_world2pix([[a1, a2]], 1, tolerance=1e-12, maxiter=200, quiet=True)
            xr[i], yr[i] = r[0]
        except Exception:  # noqa: BLE001
            pass
    with open(os.path.join(out_dir, name + ".hdr"), "w") as f:
        f.write(header_text(header))
    with open(os.path.join(out_dir, name + ".tsv"), "w") as f:
        f.write(f"# source: {source}\n")
        f.write(f"# ctype: {w.wcs.ctype[0]} {w.wcs.ctype[1]}\n")
        f.write(f"# frame: {frame_txt}\n")
        f.write(f"# oracle: astropy {astropy.__version__} wcslib {_wcs.WCSLIB_VERSION}\n")
        f.write("# x\ty\tlon\tlat\tra_icrs\tdec_icrs\tx_rt\ty_rt\n")
        for i in range(len(x)):
            f.write("\t".join([fmt(x[i]), fmt(y[i]), fmt(lon[i]), fmt(lat[i]), fmt(ra[i]), fmt(dec[i]),
                               fmt(xr[i]), fmt(yr[i])]) + "\n")
    valid = int(np.isfinite(lon).sum())
    print(f"{name:34s} {w.wcs.ctype[0]:14s} valid {valid}/{len(x)}  {frame_txt[:60]}", file=sys.stderr)


def wcs_tables(corpus, out):
    d = os.path.join(out, "wcs_reference")
    os.makedirs(d, exist_ok=True)
    for p in CALABRETTA:
        path = os.path.join(corpus, "calabretta", "fitswcs_maps", f"1904-66_{p}.fits")
        if not os.path.exists(path):
            print(f"missing {path}", file=sys.stderr)
            continue
        h = fits.getheader(path)
        # keep the cards, drop the long HISTORY / COMMENT blocks (not WCS)
        h2 = fits.Header([c for c in h.cards if c.keyword not in ("HISTORY", "COMMENT", "")])
        write_case(f"calabretta_{p.lower()}", h2, d, f"calabretta/fitswcs_maps/1904-66_{p}.fits")
    for name, cards in synthetic().items():
        try:
            write_case(name, to_header(cards), d, "synthetic (scripts/gen_wcs_reference.py)")
        except Exception as e:  # noqa: BLE001
            print(f"skip {name}: {type(e).__name__}: {e}", file=sys.stderr)


# ── TNX / ZPX / DSS / helioprojective: other oracles ───────────────────────────────────────────
def grid_points(nx, ny, crpix):
    xs = np.linspace(1, nx, GRID)
    ys = np.linspace(1, ny, GRID)
    px = [(x, y) for y in ys for x in xs]
    px.append((float(crpix[0]), float(crpix[1])))
    return np.array([p[0] for p in px]), np.array([p[1] for p in px])


def write_table(name, header, out_dir, source, ctype, frame_txt, oracle, x, y, lon, lat, ra, dec, xr, yr,
                extra_comments=()):
    with open(os.path.join(out_dir, name + ".hdr"), "w") as f:
        f.write(header_text(header))
    with open(os.path.join(out_dir, name + ".tsv"), "w") as f:
        f.write(f"# source: {source}\n")
        f.write(f"# ctype: {ctype[0]} {ctype[1]}\n")
        f.write(f"# frame: {frame_txt}\n")
        f.write(f"# oracle: {oracle}\n")
        for c in extra_comments:
            f.write(f"# {c}\n")
        f.write("# x\ty\tlon\tlat\tra_icrs\tdec_icrs\tx_rt\ty_rt\n")
        for i in range(len(x)):
            f.write("\t".join([fmt(x[i]), fmt(y[i]), fmt(lon[i]), fmt(lat[i]), fmt(ra[i]), fmt(dec[i]),
                               fmt(xr[i]), fmt(yr[i])]) + "\n")
    valid = int(np.isfinite(lon).sum())
    print(f"{name:34s} {ctype[0]:14s} valid {valid}/{len(x)}  {frame_txt[:60]}", file=sys.stderr)


def strip_cards(h, drop=("HISTORY", "COMMENT", "")):
    return fits.Header([c for c in h.cards if c.keyword not in drop])


def header_to_fits(h, path):
    """A tiny FITS file carrying this header (the data are not read by the oracles)."""
    # the data block has to be real: astropy writes NAXISn from the array shape
    hdu = fits.PrimaryHDU(np.zeros((int(h["NAXIS2"]), int(h["NAXIS1"])), dtype=np.uint8))
    for c in h.cards:
        if c.keyword in ("SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "EXTEND", "PCOUNT", "GCOUNT", "XTENSION",
                         "BZERO", "BSCALE"):
            continue
        hdu.header.append(c, end=True)
    hdu.writeto(path, overwrite=True, output_verify="silentfix")
    # the cards as written (astropy may have re-rendered a value)
    return fits.getheader(path)


def wcstools_xy2sky(binary, path, x, y):
    # a list file: a negative coordinate on the command line reads as a flag
    listfile = path + ".xy2sky.txt"
    with open(listfile, "w") as f:
        for p in zip(x, y):
            f.write(f"{float(p[0])!r} {float(p[1])!r}\n")
    args = [os.path.join(binary, "xy2sky"), "-d", "-n", "14", path, "@" + listfile]
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    rows = [l.split() for l in res.stdout.strip().splitlines() if l.strip()]
    if len(rows) != len(x):
        raise RuntimeError(f"xy2sky: {len(rows)} rows for {len(x)} points: {res.stdout[:200]} {res.stderr[:200]}")
    return np.array([float(r[0]) for r in rows]), np.array([float(r[1]) for r in rows]), rows[0][2] if len(rows[0]) > 2 else ""


def wcstools_sky2xy(binary, path, lon, lat):
    import re
    xr = np.full_like(lon, np.nan)
    yr = np.full_like(lon, np.nan)
    idx = [i for i in range(len(lon)) if np.isfinite(lon[i]) and np.isfinite(lat[i])]
    listfile = path + ".sky2xy.txt"
    with open(listfile, "w") as f:
        for i in idx:
            f.write(f"{float(lon[i])!r} {float(lat[i])!r}\n")
    args = [os.path.join(binary, "sky2xy"), "-n", "8", path, "@" + listfile]
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    rows = [l for l in res.stdout.splitlines() if "->" in l]
    if len(rows) != len(idx):
        raise RuntimeError(f"sky2xy: {len(rows)} rows for {len(idx)} points: {res.stderr[:200]}")
    for i, line in zip(idx, rows):
        m = re.match(r"\s*(-?\d+\.\d{8})\s+(-?\d+\.\d{8})", line.split("->", 1)[1])
        if m:
            xr[i], yr[i] = float(m.group(1)), float(m.group(2))
    return xr, yr


def ast_frameset(h):
    import starlink.Ast as Ast
    fc = Ast.FitsChan()
    for c in h.cards:
        fc.putfits(c.image, False)
    fc.clear("Card")
    enc = fc.Encoding  # before read(): the cards are consumed by it
    fs = fc.read()
    if fs is None:
        raise RuntimeError("AST read no WCS")
    return fs, enc


def icrs_from_fk5(lon, lat, equinox=2000.0):
    from astropy.coordinates import SkyCoord, FK5
    import astropy.units as u
    ra = np.full_like(lon, np.nan)
    dec = np.full_like(lon, np.nan)
    ok = np.isfinite(lon) & np.isfinite(lat)
    if ok.any():
        c = SkyCoord(lon[ok] * u.deg, lat[ok] * u.deg, frame=FK5(equinox=f"J{equinox}")).icrs
        ra[ok], dec[ok] = c.ra.deg, c.dec.deg
    return ra, dec


def iraf_wat_cards(axis, function, xorder, yorder, xterms, lims, coeff, which, extra=""):
    """`WATaxis_nnn` cards of one IRAF surface; a piece never ends in a blank (FITS readers strip
    trailing blanks of a string value, which would merge two numbers)."""
    nums = " ".join(f"{v:.17g}" for v in [function, xorder, yorder, xterms, *lims, *coeff])
    s = f"wtype={'zpx' if extra else 'tnx'} axtype={'ra' if axis == 1 else 'dec'} {extra}{which} = \"{nums} \""
    cards = []
    n = 1
    while s:
        cut = 68
        while 1 < cut < len(s) and s[cut - 1] == " ":
            cut -= 1
        cards.append((f"WAT{axis}_{n:03d}", s[:cut]))
        s = s[cut:]
        n += 1
    return cards


def iraf_ncoeff(xo, yo, xt):
    if xt == 0:
        return xo + yo - 1
    if xt == 2:
        o = min(xo, yo)
        return xo * yo - o * (o - 1) // 2
    return xo * yo


def synthetic_iraf(function, xo, yo, xt, zpx=False):
    lims = (-0.3, 0.3, -0.25, 0.25)
    rng = np.random.default_rng(function * 100 + xo * 10 + yo + xt + (1000 if zpx else 0))
    n = iraf_ncoeff(xo, yo, xt)
    c1 = rng.normal(0, 1e-3, n)
    c2 = rng.normal(0, 1e-3, n)
    h = fits.Header()
    h["SIMPLE"] = True
    h["BITPIX"] = 16
    h["NAXIS"] = 2
    h["NAXIS1"] = 2048
    h["NAXIS2"] = 2048
    code = "ZPX" if zpx else "TNX"
    h["CTYPE1"] = f"RA---{code}"
    h["CTYPE2"] = f"DEC--{code}"
    h["CRVAL1"] = 150.1
    h["CRVAL2"] = 2.2
    h["CRPIX1"] = 1024.5
    h["CRPIX2"] = 1024.5
    h["CD1_1"] = -7.3e-5
    h["CD1_2"] = 1.1e-6
    h["CD2_1"] = 0.9e-6
    h["CD2_2"] = 7.3e-5
    h["EQUINOX"] = 2000.0
    h["WAT0_001"] = "system=image"
    extra = "projp0=0. projp1=1. projp2=0. projp3=-50. projp4=0. projp5=2000. " if zpx else ""
    for k, v in iraf_wat_cards(1, function, xo, yo, xt, lims, c1, "lngcor", extra):
        h[k] = v
    for k, v in iraf_wat_cards(2, function, xo, yo, xt, lims, c2, "latcor", extra):
        h[k] = v
    return h


def iraf_case(name, h, out_dir, source, binary, work, cross_ast=True):
    path = os.path.join(work, name + ".fits")
    h = header_to_fits(h, path)
    x, y = grid_points(h["NAXIS1"], h["NAXIS2"], (h["CRPIX1"], h["CRPIX2"]))
    lon, lat, system = wcstools_xy2sky(binary, path, x, y)
    lon = np.mod(lon, 360.0)
    # sky2xy's own inverse closes to only ~1e-6 px (its Newton stop): the table's round trip is
    # the grid pixel itself (x_rt / y_rt left nan), and sky2xy is a recorded cross-check
    xr = np.full_like(x, np.nan)
    yr = np.full_like(x, np.nan)
    sx, sy = wcstools_sky2xy(binary, path, lon, lat)
    equinox = float(h.get("EQUINOX", 2000.0))
    ra, dec = icrs_from_fk5(lon, lat, equinox)
    comments = [f"cross-check: wcstools sky2xy -n 8 recovers the grid pixels to {np.nanmax(np.hypot(sx - x, sy - y)):.1e} px (its own inverse)"]
    if cross_ast:
        try:
            fs, enc = ast_frameset(h)
            out = fs.tran([x, y], True)
            alon, alat = np.degrees(out[0]) % 360, np.degrees(out[1])
            d = np.hypot((alon - lon) * np.cos(np.radians(lat)), alat - lat).max() * 3600
            comments.append(f"cross-check: AST {enc} encoding differs from wcstools by at most {d:.2e} arcsec")
        except Exception as e:  # noqa: BLE001
            comments.append(f"cross-check: AST failed ({type(e).__name__}: {str(e)[:80]})")
    import starlink.Ast as Ast
    write_table(name, h, out_dir, source, (h["CTYPE1"], h["CTYPE2"]),
                f"FK5 J{equinox:g} (wcstools system {system}; EQUINOX {equinox:g})",
                f"wcstools 3.9.7 xy2sky -d -n 14 (AST {Ast.version()} and sky2xy cross-checks)",
                x, y, lon, lat, ra, dec, xr, yr, comments)


DSS_TAN_CARDS = ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2", "CRPIX1", "CRPIX2", "CDELT1", "CDELT2", "CROTA1",
                 "CROTA2", "CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_1", "PC1_2", "PC2_1", "PC2_2", "LONPOLE",
                 "LATPOLE", "RADESYS", "RADECSYS", "CUNIT1", "CUNIT2", "WCSAXES", "DATE-OBS")


def dss_case(name, path, out_dir, source, binary, work):
    import starlink.Ast as Ast
    h = strip_cards(fits.getheader(path), ("HISTORY", "COMMENT", "") + DSS_TAN_CARDS)
    h = header_to_fits(h, os.path.join(work, name + ".fits"))
    fs, enc = ast_frameset(h)
    if enc != "DSS":
        raise RuntimeError(f"AST encoding {enc}, expected DSS")
    # the plate centre in the pixel grid: solve for the pixel of (PLTRA, PLTDEC)
    ra0 = 15 * (h["PLTRAH"] + h["PLTRAM"] / 60 + h["PLTRAS"] / 3600)
    dec0 = (h["PLTDECD"] + h["PLTDECM"] / 60 + h["PLTDECS"] / 3600) * (-1 if str(h["PLTDECSN"]).strip().startswith("-") else 1)
    c0 = fs.tran([[np.radians(ra0)], [np.radians(dec0)]], False)
    x, y = grid_points(h["NAXIS1"], h["NAXIS2"], (c0[0][0], c0[1][0]))
    out = fs.tran([x, y], True)
    lon, lat = np.degrees(out[0]) % 360, np.degrees(out[1])
    back = fs.tran([np.radians(lon), np.radians(lat)], False)
    xr, yr = back[0], back[1]
    system, equinox = fs.System, fs.get("Equinox")
    if system != "FK5":
        raise RuntimeError(f"AST system {system}")
    ra, dec = icrs_from_fk5(lon, lat, float(equinox))
    comments = []
    try:
        wlon, wlat, _ = wcstools_xy2sky(binary, os.path.join(work, name + ".fits"), x, y)
        d = np.hypot((np.mod(wlon, 360) - lon) * np.cos(np.radians(lat)), wlat - lat).max() * 3600
        comments.append(f"cross-check: wcstools xy2sky differs from AST by at most {d:.2e} arcsec")
    except Exception as e:  # noqa: BLE001
        comments.append(f"cross-check: wcstools failed ({type(e).__name__})")
    comments.append("the GetImage TAN cards of the cutout were removed so the plate solution is what is read")
    write_table(name, h, out_dir, source, ("(none: DSS plate solution)", ""),
                f"FK5 J{float(equinox):g} (AST System {system} Equinox {equinox})",
                f"starlink-pyast AST {Ast.version()} FitsChan (Encoding DSS)",
                x, y, lon, lat, ra, dec, xr, yr, comments)


def hpc_case(name, path, out_dir, source, hdu=0):
    import astropy.units as u
    import sunpy
    from sunpy.coordinates.wcs_utils import solar_wcs_frame_mapping
    from astropy.coordinates import SkyCoord, HCRS
    with fits.open(path) as f:
        h = strip_cards(f[hdu].header)
    w = WCS(h)
    frame = solar_wcs_frame_mapping(w)
    x, y = grid_points(w.pixel_shape[0], w.pixel_shape[1], w.wcs.crpix)
    tx, ty = w.all_pix2world(x, y, 1)  # degrees, HPLN / HPLT
    lon = np.where(np.isfinite(tx), np.mod(tx, 360.0), np.nan)
    lat = ty
    obs = frame.observer
    hcrs = HCRS(obstime=frame.obstime)
    obs_hcrs = obs.transform_to(hcrs).cartesian.xyz.to_value(u.au)
    ra = np.full_like(x, np.nan)
    dec = np.full_like(x, np.nan)
    ok = np.isfinite(tx) & np.isfinite(ty)
    pts = SkyCoord(tx[ok] * u.deg, ty[ok] * u.deg, np.full(int(ok.sum()), 0.5) * u.au, frame=frame)
    p = pts.transform_to(hcrs).cartesian.xyz.to_value(u.au)
    d = p - obs_hcrs[:, None]
    d /= np.linalg.norm(d, axis=0)
    ra[ok] = np.degrees(np.arctan2(d[1], d[0])) % 360
    dec[ok] = np.degrees(np.arcsin(np.clip(d[2], -1, 1)))
    xr = np.full_like(x, np.nan)
    yr = np.full_like(x, np.nan)
    for i in range(len(x)):
        if ok[i]:
            try:
                r = w.all_world2pix([[tx[i], ty[i]]], 1, tolerance=1e-12, maxiter=200, quiet=True)
                xr[i], yr[i] = r[0]
            except Exception:  # noqa: BLE001
                pass
    comments = ["icrs_tol_deg: 3e-5",
                f"observer: {obs.lon.deg:.6f} {obs.lat.deg:.6f} deg HGS, {obs.radius.to_value(u.au):.8f} au, obstime {frame.obstime.isot}",
                "ra_icrs dec_icrs: direction (point - observer) in HCRS, the point at 0.5 au from the Sun"]
    # the alternate celestial WCS of SECCHI files, if any: an approximate cross-check
    if "CTYPE1A" in h:
        try:
            wa = WCS(h, key="A")
            ara, adec = wa.all_pix2world(x, y, 1)
            dd = np.hypot((np.mod(ara, 360) - ra) * np.cos(np.radians(dec)), adec - dec)
            comments.append(f"cross-check: the file's alternate '{wa.wcs.ctype[0]}' WCS differs by at most {np.nanmax(dd) * 3600:.1f} arcsec (its own approximation)")
        except Exception as e:  # noqa: BLE001
            comments.append(f"cross-check: alternate WCS failed ({type(e).__name__})")
    write_table(name, h, out_dir, source, (w.wcs.ctype[0], w.wcs.ctype[1]),
                f"{type(frame).__name__} observer HGS lon {obs.lon.deg:.4f} lat {obs.lat.deg:.4f}",
                f"sunpy {sunpy.__version__} / astropy {astropy.__version__} wcslib {_wcs.WCSLIB_VERSION}",
                x, y, lon, lat, ra, dec, xr, yr, comments)


def extra_tables(corpus, out, binary, work=None):
    d = os.path.join(out, "wcs_reference")
    os.makedirs(d, exist_ok=True)
    work = work or tempfile.mkdtemp(prefix="wcs_extra_")
    os.makedirs(work, exist_ok=True)
    ex = os.path.join(corpus, "wcs_extra")
    for name, fn in (("tnx_registry_ctio_mosaic", "tnx.fits"), ("zpx_registry_mosaic2", "zpx.fits")):
        h = strip_cards(fits.getheader(os.path.join(ex, fn)))
        iraf_case(name, h, d, f"wcs_extra/{fn}", binary, work)
    for function, fname in ((1, "cheb"), (2, "legendre"), (3, "poly")):
        for xt, xname in ((0, "none"), (1, "full"), (2, "half")):
            for xo, yo in ((4, 4), (3, 5), (5, 3)):
                iraf_case(f"tnx_{fname}_{xname}_{xo}x{yo}", synthetic_iraf(function, xo, yo, xt), d,
                          "synthetic (scripts/gen_wcs_reference.py)", binary, work, cross_ast=(xt == 2 and xo == yo))
    iraf_case("zpx_poly_half_4x4", synthetic_iraf(3, 4, 4, 2, zpx=True), d, "synthetic (scripts/gen_wcs_reference.py)",
              binary, work)
    for name, fn in (("dss_stsci_poss2_red_m31", "dss_stsci_poss2_red_m31_15min.fits"),
                     ("dss_stsci_poss1_red_m31", "dss_stsci_poss1_red_m31_30min.fits"),
                     ("dss_eso_dss1_cena", "dss_eso_dss1_cena_20min.fits")):
        dss_case(name, os.path.join(ex, fn), d, f"wcs_extra/{fn}", binary, work)
    for name, fn, hdu in (("hpc_secchi_cor2_tan", "secchi_cor2_20130301_000800_d4c2A.fts", 0),
                          ("hpc_secchi_hi1_azp", "secchi_hi1_20130301_000901_s4h1A.fts", 0),
                          ("hpc_aia_171_tan", "aia_20110607_063302_0171_lowres.fits", 1)):
        hpc_case(name, os.path.join(ex, fn), d, f"wcs_extra/{fn}", hdu)


# ── HiPS ───────────────────────────────────────────────────────────────────────────────────────
def interleave(ix, iy, bits):
    out = 0
    for i in range(bits):
        out |= ((ix >> i) & 1) << (2 * i)
        out |= ((iy >> i) & 1) << (2 * i + 1)
    return out


TILES = {0: list(range(12)), 1: [0, 5, 13, 26, 47], 2: [0, 77, 100, 191], 3: [0, 300, 383, 500, 767]}
CORNERS = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def hips_tables(out):
    from astropy_healpix import HEALPix
    d = os.path.join(out, "hips_reference")
    os.makedirs(d, exist_ok=True)
    head = (f"# HEALPix nested, lon/lat in deg, astropy_healpix (astropy {astropy.__version__}).\n"
            "# corners c0..c3 = (dx, dy) (0,0) (1,0) (1,1) (0,1) in HEALPix face coordinates (dx along ix, dy along iy)\n")
    with open(os.path.join(d, "tiles.tsv"), "w") as f:
        f.write(head)
        f.write("# order\tnpix\tlon\tlat\tc0_lon\tc0_lat\tc1_lon\tc1_lat\tc2_lon\tc2_lat\tc3_lon\tc3_lat\n")
        for order in sorted(TILES):
            hp = HEALPix(nside=2 ** order, order="nested")
            for n in range(12 * 4 ** order) if order <= 1 else TILES[order]:
                row = [str(order), str(n)]
                lo, la = hp.healpix_to_lonlat([n])
                row += [fmt(lo.deg[0]), fmt(la.deg[0])]
                for dx, dy in CORNERS:
                    lo, la = hp.healpix_to_lonlat([n], dx=dx, dy=dy)
                    row += [fmt(lo.deg[0] % 360.0), fmt(la.deg[0])]
                f.write("\t".join(row) + "\n")
    for width in (512, 64):
        s = int(np.log2(width))
        picks = sorted({(0, 0), (width - 1, 0), (0, width - 1), (width - 1, width - 1), (width // 2, width // 2),
                        (1, 0), (0, 1), (width // 2 - 1, width // 4), (width // 3, width - 2)})
        with open(os.path.join(d, f"tile_pixels_w{width}.tsv"), "w") as f:
            f.write(head)
            f.write("# tile pixel (x, y): x = column along NAXIS1, y = row along NAXIS2, both 0-based, row 0 = first\n"
                    "# stored row. nested = npix * 4^S + interleave(ix = W-1-y on even bits, iy = x on odd bits),\n"
                    "# S = log2 W, at order + S. Verified on a HipsGen FITS tile (README.md).\n")
            f.write("# order\tnpix\twidth\tx\ty\tnested\tlon\tlat\tc0_lon\tc0_lat\tc1_lon\tc1_lat\tc2_lon\tc2_lat\tc3_lon\tc3_lat\n")
            for order in sorted(TILES):
                hp = HEALPix(nside=2 ** (order + s), order="nested")
                for n in TILES[order]:
                    for (x, y) in picks:
                        nest = n * 4 ** s + interleave(width - 1 - y, x, s)
                        row = [str(order), str(n), str(width), str(x), str(y), str(nest)]
                        lo, la = hp.healpix_to_lonlat([nest])
                        row += [fmt(lo.deg[0]), fmt(la.deg[0])]
                        for dx, dy in CORNERS:
                            lo, la = hp.healpix_to_lonlat([nest], dx=dx, dy=dy)
                            row += [fmt(lo.deg[0] % 360.0), fmt(la.deg[0])]
                        f.write("\t".join(row) + "\n")
    print(f"hips tables -> {d}", file=sys.stderr)


def hips_check(tile_path, img_path, order, npix):
    """Correlates a real HiPS FITS tile with an independent WCS image under the 8 orientations."""
    from astropy_healpix import HEALPix
    t = fits.open(tile_path)[0]
    data = t.data.astype(float)
    if "BLANK" in t.header:
        data[t.data == t.header["BLANK"]] = np.nan
    W = data.shape[0]
    m = int(np.log2(W))
    img = fits.open(img_path)[0]
    iw = WCS(img.header)
    idata = img.data.astype(float)
    yy, xx = np.mgrid[0:W, 0:W]

    def spread(v):
        out = np.zeros_like(v)
        for i in range(m):
            out |= ((v >> i) & 1) << (2 * i)
        return out
    hp = HEALPix(nside=2 ** (order + m), order="nested", frame="icrs")
    variants = {"ix=x iy=y": (xx, yy), "ix=y iy=x": (yy, xx), "ix=W-1-x iy=y": (W - 1 - xx, yy),
                "ix=x iy=W-1-y": (xx, W - 1 - yy), "ix=W-1-x iy=W-1-y": (W - 1 - xx, W - 1 - yy),
                "ix=y iy=W-1-x": (yy, W - 1 - xx), "ix=W-1-y iy=x": (W - 1 - yy, xx),
                "ix=W-1-y iy=W-1-x": (W - 1 - yy, W - 1 - xx)}
    for name, (X, Y) in variants.items():
        sky = hp.healpix_to_skycoord(npix * 4 ** m + (spread(X) | (spread(Y) << 1)))
        px, py = iw.world_to_pixel(sky)
        ix, iy = np.round(px).astype(int), np.round(py).astype(int)
        ok = (ix >= 0) & (iy >= 0) & (ix < idata.shape[1]) & (iy < idata.shape[0]) & np.isfinite(data)
        c = np.corrcoef(data[ok], idata[iy[ok], ix[ok]])[0, 1]
        print(f"{name:20s} corr {c:+.4f}  overlap {ok.mean():.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_assets", "fits_corpus"))
    ap.add_argument("--hips", action="store_true")
    ap.add_argument("--hips-check", nargs=4)
    ap.add_argument("--extra", action="store_true", help="TNX / ZPX / DSS / helioprojective tables (needs --corpus)")
    ap.add_argument("--wcstools-bin", help="directory with the wcstools xy2sky and sky2xy binaries")
    ap.add_argument("--work", help="scratch directory for the FITS files the oracles read (default: a temp dir)")
    a = ap.parse_args()
    if a.hips_check:
        hips_check(a.hips_check[0], a.hips_check[1], int(a.hips_check[2]), int(a.hips_check[3]))
        return
    if a.corpus and not a.extra:
        wcs_tables(a.corpus, a.out)
    if a.extra:
        if not a.corpus or not a.wcstools_bin:
            ap.error("--extra needs --corpus and --wcstools-bin")
        extra_tables(a.corpus, a.out, a.wcstools_bin, a.work)
    if a.hips:
        hips_tables(a.out)


if __name__ == "__main__":
    main()
