#!/usr/bin/env python3
"""Merge a partial ManualCapture run (AETHERVK_MANUAL_ONLY) into a full one.

    py -3 -I docs/manual/tools/merge_capture.py --base TestResults/e2e/manual/<full> --part TestResults/e2e/manual/<subset>

Every shot of the part that succeeded replaces the base's shot with the same key and state: its PNGs
and callouts file are copied over, the base's --FAILED.png is removed, and the manifest entry is
replaced. Shots that failed again in the part leave the base untouched. Standard library only.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--part", required=True, type=Path)
    args = ap.parse_args()
    base_m = json.loads((args.base / "manifest.json").read_text(encoding="utf-8"))
    part_m = json.loads((args.part / "manifest.json").read_text(encoding="utf-8"))
    theme = base_m.get("theme", "light")
    index = {(s["key"], s["state"]): i for i, s in enumerate(base_m["shots"])}
    merged = 0
    for shot in part_m["shots"]:
        if shot["status"] != "ok":
            continue
        stem = f"{shot['key']}--{shot['state']}--{theme}"
        for src in args.part.glob(stem + "*"):
            shutil.copyfile(src, args.base / src.name)
        failed = args.base / f"{shot['key']}--{shot['state']}--FAILED.png"
        if failed.exists():
            failed.unlink()
        k = (shot["key"], shot["state"])
        if k in index:
            base_m["shots"][index[k]] = shot
        else:
            base_m["shots"].append(shot)
        merged += 1
    (args.base / "manifest.json").write_text(json.dumps(base_m, indent=2, ensure_ascii=False), encoding="utf-8")
    left = [f"{s['key']}--{s['state']}" for s in base_m["shots"] if s["status"] != "ok"]
    print(f"{merged} shots merged into {args.base}; still failed: {', '.join(left) or 'none'}")


if __name__ == "__main__":
    main()
