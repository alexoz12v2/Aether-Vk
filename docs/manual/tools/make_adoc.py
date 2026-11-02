#!/usr/bin/env python3
"""Turn a ManualCapture folder into the manual's AsciiDoc (Blender manual style).

    py -3 -I docs/manual/tools/make_adoc.py --capture TestResults/e2e/manual/<stamp> [--sync]

Reads, from the capture folder, manifest.json and the `<key>--<state>--<theme>.callouts.json`
files; from docs/manual/tools/inventory.json the label and the help-text resx keys of each
AutomationId (run tools/axaml_inventory.py --json first); from
aethervk.ui-logic/Resources/AppStrings.resx the help texts; from docs/manual/descriptions.json
the hand-written English: an intro per shot ("shots": {"<key>" or "<key>--<state>": text}) and a
label and explanation per control ("controls": {"<AutomationId>": {"label", "text"}}), used where the
app has no help text (and as the short line when it has one). Writes docs/manual/adoc/*.adoc
(one file per tab plus index.adoc). With --sync it copies the PNGs and manifest.json into
docs/manual/img/ (only files whose bytes changed). Standard library only.
"""
import argparse
import json
import struct
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANUAL = HERE.parent
REPO = MANUAL.parent.parent

# (file, heading, key prefixes in order). A key belongs to the first file whose prefix matches.
TABS = [
    ("main", "Main window", ("main", "menu-", "docking")),
    ("comet", "Comet tab", ("comet-",)),
    ("timeline", "Timeline tab", ("timeline",)),
    ("model", "Model tab", ("model-",)),
    ("settings", "Settings tab", ("settings-",)),
    ("imports", "Imports tab", ("imports",)),
    ("viewport", "Viewport", ("viewport", "overlay", "billboard-window")),
    ("dialogs", "Dialogs and pop-ups", ("dlg-", "splash", "help-flyout", "combo-dropdown")),
]

TITLES = {
    "main": "Main window",
    "main-modebar": "Mode bar",
    "menu-file": "File menu",
    "menu-edit": "Edit menu",
    "docking": "Docking and tabs",
    "comet-guided": "Guided steps",
    "comet-selection": "Comet selection",
    "comet-simulated-range": "Simulated range",
    "comet-range-advanced": "Simulated range: Advanced",
    "comet-commit": "Reference orbit and commit",
    "comet-nucleus": "Nucleus",
    "comet-rotation": "Rotational model",
    "timeline": "Timeline",
    "model-sun": "Sun",
    "model-nucleus-spin": "Nucleus and spin axis",
    "model-dust": "Dust",
    "model-dust-jet": "Dust: jet",
    "model-dust-defaults": "Dust: default grain and activity",
    "settings-time-scale": "Time scale",
    "settings-general": "General",
    "settings-orbital-elements": "Orbital elements",
    "settings-comet-appearance": "Comet appearance",
    "settings-billboards": "Billboards",
    "settings-billboards-fits": "Billboards: FITS image",
    "settings-particle-rendering": "Particle rendering",
    "settings-input-bindings": "Input bindings",
    "settings-viewport": "Viewport",
    "settings-viewport-persp": "Viewport: perspective projection",
    "settings-viewport-ortho": "Viewport: orthographic projection",
    "settings-viewport-nucleus": "Viewport: nucleus mode",
    "settings-viewport-observer": "Viewport: observer",
    "imports": "Imports",
    "viewport": "Viewport",
    "overlay": "Viewport overlay",
    "billboard-window": "Billboard window",
    "help-flyout": "Help flyout",
    "combo-dropdown": "Drop-down list",
    "dlg-settings": "Settings window",
    "dlg-fatal": "Fatal error window",
    "splash": "Splash window",
}

STATES = {
    "orbit": "Orbit mode",
    "nucleus": "Nucleus mode",
    "model": "Model mode",
    "empty": "no comet loaded",
    "committed": "committed comet (67P, December 2021)",
    "locked": "pending range edit, the range is locked",
    "playing": "playback running",
    "selected": "a jet selected",
    "custom": "custom mesh selected",
    "flyout": "help flyout open",
    "dropdown": "drop-down open",
    "dialog": "dialog open",
    "launch": "at launch",
}
MODE_STATES = {"orbit", "nucleus", "model", "locked", "playing", "selected", "custom", "flyout", "dropdown", "dialog"}

TAB_NAMES = {"settings": "Settings", "comet": "Comet", "model": "Model", "timeline": "Timeline", "imports": "Imports"}


def load_resx(path):
    out = {}
    if not path.exists():
        return out
    for data in ET.parse(path).getroot().iter("data"):
        value = data.find("value")
        if value is not None and value.text is not None:
            out[data.get("name")] = value.text.replace("\r\n", "\n").strip()
    return out


def load_inventory(path):
    items = {}
    if not path.exists():
        print(f"warning: {path} missing; run axaml_inventory.py --json", file=sys.stderr)
        return items
    for row in json.loads(path.read_text(encoding="utf-8")):
        if row.get("id") and row["id"] not in items:
            items[row["id"]] = row
    return items


def tab_of(key):
    for name, _title, prefixes in TABS:
        if any(key == p or key.startswith(p) for p in prefixes):
            return name
    return "dialogs"


def title_of(key):
    if key in TITLES:
        return TITLES[key]
    return key.replace("-", " ").capitalize()


def location_of(key):
    head, _, rest = key.partition("-")
    if head in TAB_NAMES:
        return f"{TAB_NAMES[head]} › {title_of(key).split(':')[0]}" if rest else TAB_NAMES[head]
    if key.startswith("menu-"):
        return f"Menu bar › {title_of(key)}"
    return title_of(key)


def shown_of(state):
    text = STATES.get(state, state)
    if state in MODE_STATES:
        text += ", committed comet"
    return text


def one_line(text, limit=170):
    flat = " ".join(text.split())
    first = flat.split(". ")[0].rstrip(".") + "." if ". " in flat else flat
    if len(first) > limit:
        first = first[: limit - 3].rstrip() + "..."
    return first


def adoc_text(text):
    """A help text as indented description-list body (continuation lines keep the indent)."""
    lines = [ln.strip() for ln in text.split("\n")]
    return "\n".join("  " + ln if ln else "" for ln in lines)


def label_of(callout, inv):
    raw = (callout.get("label") or "").strip()
    cid = callout["id"]
    if raw and raw not in ("?", cid):
        return raw
    row = inv.get(cid, {})
    label = (row.get("label") or "").strip()
    return label if label and label != "?" else cid


def help_of(callout, inv, resx):
    """(header, text) of the property's help button, or (None, None)."""
    cid = callout["id"]
    candidates = [callout.get("help_id"), cid + "Help", cid]
    for hid in candidates:
        row = inv.get(hid) if hid else None
        if row and row.get("helpKey") and resx.get(row["helpKey"]):
            header = resx.get(row.get("helpHeaderKey") or "")
            return header, resx[row["helpKey"]]
    return None, None


# print scale: every UI figure at 150 dpi (1 UI px = 0.169 mm), capped to the text width
PRINT_DPI = 150
TEXT_WIDTH_MM = 170


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != bytes([0x89]) + b"PNG" + bytes([13, 10, 26, 10]):
        return None
    return struct.unpack(">II", head[16:24])


def load_descriptions(path):
    if not path.exists():
        return {"shots": {}, "controls": {}}
    d = json.loads(path.read_text(encoding="utf-8"))
    return {"shots": d.get("shots", {}), "controls": d.get("controls", {})}


def build(capture, resx, inv, out_dir, desc=None):
    desc = desc or {"shots": {}, "controls": {}}
    manifest = json.loads((capture / "manifest.json").read_text(encoding="utf-8"))
    theme = manifest.get("theme", "light")
    shots = [s for s in manifest["shots"] if s["status"] == "ok"]
    problems = [s for s in manifest["shots"] if s["status"] != "ok"]

    files = {name: [] for name, _t, _p in TABS}
    seen_keys = {}
    for shot in shots:
        files[tab_of(shot["key"])].append(shot)

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, heading, _prefixes in TABS:
        lines = [f"== {heading}", ":imagesdir: ..", ""]
        if not files[name]:
            lines.append("// no shots captured for this file yet")
        for shot in files[name]:
            key, state = shot["key"], shot["state"]
            stem = f"{key}--{state}--{theme}"
            callouts_path = capture / f"{stem}.callouts.json"
            callouts = json.loads(callouts_path.read_text(encoding="utf-8")) if callouts_path.exists() else []
            first = key not in seen_keys
            seen_keys.setdefault(key, state)
            anchor = key if first else f"{key}-{state}"
            size = png_size(capture / f"{stem}--tagged.png")
            px = size[0] if size else int(shot["w"])
            width = min(px, 900)
            mm = min(round(px / PRINT_DPI * 25.4), TEXT_WIDTH_MM)
            title = title_of(key)
            intro = desc["shots"].get(f"{key}--{state}") or desc["shots"].get(key)
            lines += [
                f"[[{anchor}]]",
                f"=== {title}",
                f"Location:: {location_of(key)}",
                f"Shown:: {shown_of(state)}",
                "",
                f".{location_of(key)}",
            ]
            if intro:
                lines += [intro.strip(), ""]
            lines += [
                f"image::img/{stem}--tagged.png[{location_of(key)},width={width},pdfwidth={mm}mm]",
                "",
            ]
            props = []
            missing = False
            for c in callouts:
                own = desc["controls"].get(c["id"], {})
                label = own.get("label") or label_of(c, inv)
                header, text = help_of(c, inv, resx)
                short = own.get("text")
                if text:
                    lines.append(f". *{label}* -- {short or one_line(text)}")
                    props.append((header or label, text))
                elif short:
                    lines.append(f". *{label}* -- {short}")
                else:
                    lines.append(f". *{label}* -- // TODO: describe")
                    missing = True
            if callouts:
                lines.append("")
            if props:
                lines += ["==== Properties", ""]
                for term, text in props:
                    lines += [f"{term}::", adoc_text(text), ""]
            elif missing:
                lines += ["// TODO: some controls have no help text and no description", ""]
        (out_dir / f"{name}.adoc").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8", newline="\n")

    version = manifest.get("app_version", "?")
    date = str(manifest.get("date", ""))[:10]
    index = [
        "= AetherVk User Manual",
        ":toc: left",
        ":toclevels: 3",
        ":imagesdir: ..",
        f":screenshots: AetherVk {version}, captured {date}, {manifest.get('culture', '')}, {theme} theme",
        "",
        "Screenshots: {screenshots}.",
        "",
    ]
    index += [f"include::{name}.adoc[]\n" for name, _t, _p in TABS]
    (out_dir / "index.adoc").write_text("\n".join(index), encoding="utf-8", newline="\n")
    return shots, problems


def sync(capture, img_dir):
    img_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in sorted(list(capture.glob("*.png")) + [capture / "manifest.json"]):
        if not src.exists() or src.name.endswith("--FAILED.png"):
            continue
        dest = img_dir / src.name
        if dest.exists() and dest.read_bytes() == src.read_bytes():
            continue
        shutil.copyfile(src, dest)
        copied += 1
    return copied


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--capture", required=True, type=Path, help="TestResults/e2e/manual/<stamp>")
    ap.add_argument("--out", type=Path, default=MANUAL / "adoc")
    ap.add_argument("--img", type=Path, default=MANUAL / "img")
    ap.add_argument("--inventory", type=Path, default=HERE / "inventory.json")
    ap.add_argument("--resx", type=Path, default=REPO / "aethervk.ui-logic" / "Resources" / "AppStrings.resx")
    ap.add_argument("--descriptions", type=Path, default=MANUAL / "descriptions.json")
    ap.add_argument("--sync", action="store_true", help="copy the capture's PNGs and manifest.json into docs/manual/img")
    args = ap.parse_args()

    if not (args.capture / "manifest.json").exists():
        sys.exit(f"{args.capture}\\manifest.json not found")
    shots, problems = build(args.capture, load_resx(args.resx), load_inventory(args.inventory), args.out,
                            load_descriptions(args.descriptions))
    print(f"{len(shots)} shots written to {args.out}")
    for p in problems:
        print(f"  {p['status']}: {p['key']}--{p['state']}: {p.get('reason')}")
    if args.sync:
        print(f"{sync(args.capture, args.img)} files copied to {args.img}")


if __name__ == "__main__":
    main()
