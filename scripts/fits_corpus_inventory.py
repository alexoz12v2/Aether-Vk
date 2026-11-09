#!/usr/bin/env python3
"""Inventory of the FITS test corpus (lab tool, never a build dependency).

Opens every file of the corpus with astropy (which reads gzip / bzip2 wrappers and tile-compressed
images) and writes one row per file: size, sha256 and every HDU (type, BITPIX, axes, ZCMPTYPE,
CTYPE1/2, RADESYS, SIP / TPV, HiPS ORDER / NPIX, dates). It also derives the `expected` value the
app must meet, mirroring the engine's read rule (the first IMAGE HDU, tile-compressed included,
with NAXIS >= 2 and NAXIS1, NAXIS2 > 0):

  image              a 2-D image (or a cube: the note says so) the app must open
  table-only         no image HDU at all, only tables
  refuse:no image    no 2-D image but something else (1-D spectra, random groups, empty primary)
  either             astropy cannot read the file strictly (deliberately malformed test files):
                     the app may open or refuse it, it must not panic

Usage (run with the lab venv, `-I` so nothing is imported from the corpus):
  python3 -I scripts/fits_corpus_inventory.py <corpus_dir> [--out INVENTORY.tsv]
      [--manifest test_assets/fits_corpus/manifest.tsv] [--new-rows rows.tsv --category cds
       --source-prefix tar:cds/fits-rs-test-files.tar]

--new-rows writes manifest rows for the corpus files under <corpus_dir>/<category>/ that the
manifest does not list yet, so a new source (the CDS tar once it arrives) can be appended after
review. A `tar:<archive>` prefix is the source of every row and adds the archive's own `archive`
row (fill in its URL); any other prefix is joined with the path inside the category.
"""
import argparse
import hashlib
import os
import sys
import warnings

FITS_SUFFIXES = (".fits", ".fit", ".fts", ".fz", ".fits.gz", ".fit.gz", ".fits.bz2", ".fits.z",
                 ".z", ".hdr", ".pha", ".fits.fz")
SKIP_DIRS = {"_pages"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def is_fits_like(name):
    n = name.lower()
    return n.endswith(FITS_SUFFIXES)


def card(h, key):
    v = h.get(key)
    if v is None:
        return ""
    return str(v).strip().replace("\t", " ")


def hdu_summary(hdu):
    from astropy.io import fits
    h = hdu.header
    kind = type(hdu).__name__.replace("HDU", "")
    if isinstance(hdu, fits.CompImageHDU):
        naxis = h.get("NAXIS", 0)
        axes = [h.get(f"NAXIS{i}", 0) for i in range(1, naxis + 1)]
        zc = card(hdu._header if hasattr(hdu, "_header") else h, "ZCMPTYPE") or "compressed"
    else:
        naxis = h.get("NAXIS", 0)
        axes = [h.get(f"NAXIS{i}", 0) for i in range(1, naxis + 1)]
        zc = card(h, "ZCMPTYPE")
    ctype1, ctype2 = card(h, "CTYPE1"), card(h, "CTYPE2")
    sip = "A_ORDER" in h or "AP_ORDER" in h
    tpv = "TPV" in ctype1 or "TPV" in ctype2 or ("TAN" in ctype1 and any(k.startswith("PV1_") for k in h))
    tnx = "TNX" in ctype1 or "ZPX" in ctype1
    groups = bool(h.get("GROUPS", False))
    return {
        "kind": kind,
        "bitpix": h.get("BITPIX", ""),
        "axes": axes,
        "zcmptype": zc,
        "ctype1": ctype1,
        "ctype2": ctype2,
        "radesys": card(h, "RADESYS") or card(h, "RADECSYS"),
        "equinox": card(h, "EQUINOX") or card(h, "EPOCH"),
        "sip": sip,
        "tpv": tpv,
        "tnx": tnx,
        "groups": groups,
        "order": card(h, "ORDER"),
        "npix": card(h, "NPIX"),
        "pixtype": card(h, "PIXTYPE"),
        "date": card(h, "DATE-OBS") or card(h, "MJD-OBS") or card(h, "DATE"),
    }


def is_image_kind(s):
    return s["kind"] in ("Primary", "Image", "CompImage") and not s["groups"]


def derive_expected(hdus):
    first = None
    for s in hdus:
        if is_image_kind(s) and len(s["axes"]) >= 2 and s["axes"][0] > 0 and s["axes"][1] > 0:
            first = s
            break
    if first is not None:
        extra = [n for n in first["axes"][2:] if n != 1]
        note = f"cube {'x'.join(str(a) for a in first['axes'])}" if extra else ""
        return "image", note
    tables = [s for s in hdus if s["kind"] in ("BinTable", "Table")]
    images = [s for s in hdus if s["kind"] in ("Primary", "Image", "CompImage", "Groups") and s["axes"]]
    if tables and not images:
        return "table-only", ""
    what = []
    if any(s["groups"] for s in hdus):
        what.append("random groups")
    if any(is_image_kind(s) and len(s["axes"]) == 1 for s in hdus):
        what.append("1-D image")
    if tables:
        what.append("tables")
    if not what:
        what.append("empty primary")
    return "refuse:no image", "+".join(what)


def open_fits(path):
    from astropy.io import fits
    if path.lower().endswith(".z"):
        # unix `compress` (.Z): astropy has no reader, gzip -d decodes it
        import io
        import subprocess
        try:
            raw = subprocess.run(["gzip", "-dc", path], check=True, capture_output=True).stdout
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return fits.open(io.BytesIO(raw), lazy_load_hdus=False), "strict(.Z via gzip -d)"
        except Exception as e:  # noqa: BLE001
            return None, "fail(.Z: " + f"{type(e).__name__}: {e}"[:100].replace("\n", " ") + ")"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            hl = fits.open(path, memmap=False, lazy_load_hdus=False)
            return hl, "strict"
        except Exception as e:  # noqa: BLE001 - malformed test files are the point
            err = f"{type(e).__name__}: {e}"
        try:
            hl = fits.open(path, memmap=False, lazy_load_hdus=False, ignore_missing_end=True,
                           ignore_missing_simple=True)
            return hl, "lenient(" + err[:80].replace("\t", " ").replace("\n", " ") + ")"
        except Exception as e2:  # noqa: BLE001
            return None, "fail(" + f"{type(e2).__name__}: {e2}"[:120].replace("\t", " ").replace("\n", " ") + ")"


def inventory_file(path):
    hl, how = open_fits(path)
    hdus = []
    if hl is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                for hdu in hl:
                    hdus.append(hdu_summary(hdu))
            except Exception as e:  # noqa: BLE001
                how += f" hdu-error({type(e).__name__}: {str(e)[:80]})".replace("\n", " ")
            hl.close()
    return how, hdus


def fmt_hdu(i, s):
    flags = []
    if s["sip"]:
        flags.append("SIP")
    if s["tpv"]:
        flags.append("TPV")
    if s["tnx"]:
        flags.append("TNX")
    if s["groups"]:
        flags.append("GROUPS")
    parts = [f"{i}:{s['kind']}", f"bitpix={s['bitpix']}", "axes=" + "x".join(str(a) for a in s["axes"])]
    for key in ("zcmptype", "ctype1", "ctype2", "radesys", "equinox", "order", "npix", "pixtype", "date"):
        if s[key] != "":
            parts.append(f"{key}={s[key]}")
    if flags:
        parts.append("flags=" + "+".join(flags))
    return " ".join(parts)


def walk(root):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for name in sorted(filenames):
            if is_fits_like(name):
                yield os.path.join(dirpath, name)


def read_manifest_paths(path):
    out = set()
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            out.add(line.split("\t")[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("--out")
    ap.add_argument("--manifest")
    ap.add_argument("--new-rows")
    ap.add_argument("--category")
    ap.add_argument("--source-prefix", default="")
    a = ap.parse_args()
    root = os.path.abspath(a.corpus)
    out_path = a.out or os.path.join(root, "INVENTORY.tsv")
    known = read_manifest_paths(a.manifest)
    rows, new_rows = [], []
    for path in walk(root):
        rel = os.path.relpath(path, root)
        size = os.path.getsize(path)
        digest = sha256(path)
        how, hdus = inventory_file(path)
        if how.startswith("fail"):
            expected, note = "either", "astropy cannot read it: " + how
        elif not hdus:
            expected, note = "either", "no HDU read"
        elif not how.startswith("strict") or any(h["kind"].startswith("_") for h in hdus):
            expected, note = "either", "malformed (astropy: " + how + ")"
        else:
            expected, note = derive_expected(hdus)
        rows.append([rel, str(size), digest, how, expected, note, " | ".join(fmt_hdu(i + 1, s) for i, s in enumerate(hdus))])
        if a.new_rows and a.category and rel.startswith(a.category + "/") and rel not in known:
            inner = rel[len(a.category) + 1:]
            src = a.source_prefix if a.source_prefix.startswith("tar:") else (
                f"{a.source_prefix}/{inner}" if a.source_prefix else inner)
            new_rows.append([rel, digest, str(size), src, a.category, expected, note])
        print(f"{expected:16s} {rel}", file=sys.stderr)
    with open(out_path, "w") as f:
        f.write("# path\tsize\tsha256\tastropy_open\tsuggested_expected\tnote\thdus\n")
        for r in rows:
            f.write("\t".join(r) + "\n")
    if a.new_rows and a.source_prefix.startswith("tar:"):
        arch = a.source_prefix[4:]
        full = os.path.join(root, arch)
        if os.path.exists(full) and arch not in known:
            new_rows.insert(0, [arch, sha256(full), str(os.path.getsize(full)), "<archive URL>", a.category,
                                "archive", "extracted to " + a.category + "/" + os.path.basename(arch).split(".tar")[0] + "/"])
    if a.new_rows:
        with open(a.new_rows, "w") as f:
            for r in new_rows:
                f.write("\t".join(r) + "\n")
    print(f"{len(rows)} files -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
