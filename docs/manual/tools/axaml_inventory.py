#!/usr/bin/env python3
"""
First-pass UI inventory for the AetherVk manual.

Walks the Avalonia views (aethervk.ui-app/**/*.axaml), records every Expander (section),
every element with an AutomationProperties.AutomationId, its control type, nesting path,
the visible label (literal text or a {Binding StrX} resolved through AppStrings.resx), and the
PropertyHelpButton help keys next to it. Writes Markdown to stdout (or --out FILE) and,
with --json, a machine-readable list the ManualCapture fixture can consume.

Read-only: it parses source files, it does not build or run anything.

    python -I docs/manual/tools/axaml_inventory.py --repo . --out docs/manual/inventory.md --splice
    python -I docs/manual/tools/axaml_inventory.py --repo . --json docs/manual/tools/inventory.json

Limits (first pass): ItemsControl templates appear once (one billboard, one viewport, one jet);
labels set in code-behind or by converters show as "?"; ids bound at runtime ({Binding ...})
are listed as "{runtime}".
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

AV = "{https://github.com/avaloniaui}"
X = "{http://schemas.microsoft.com/winfx/2006/xaml}"
AID = "AutomationProperties.AutomationId"

# view file -> resx key prefixes its {Binding StrX} resolve against, in order
PREFIXES = {
    "SettingsTabView": ["Tabs_Settings_"],
    "CometTabView": ["Tabs_Comet_"],
    "ModelTabView": ["Tabs_Model_"],
    "TimelineTabView": ["Tabs_Timeline_"],
    "ImportsTabView": ["Tabs_Imports_"],
    "MainWindow": ["ModeBar_"],
    "OverlayWindow": ["Overlay_", "Ephemeris_"],
}

# section order in the manual (files not listed come after, alphabetically)
ORDER = [
    "MainWindow", "TabGroupNodeView", "CometTabView", "TimelineTabView", "ModelTabView",
    "SettingsTabView", "ImportsTabView", "Viewport3DView", "OverlayWindow", "GizmoLegendView",
    "SettingsWindow", "References/HorizonJplView", "References/AssetBrowserView",
    "References/ManageImportsWindow", "References/SpawnMeshDialogWindow",
    "References/SpawnImageDialogWindow", "FatalErrorWindow", "SplashWindow", "ConsoleView",
]

BIND = re.compile(r"\{(?:Reflection)?Binding\s+(?:[^}]*\.)?\(?(?:[\w:]+\)?\.)?(Str\w+)\b[^}]*\}")


def load_resx(path: Path) -> dict[str, str]:
    root = ET.parse(path).getroot()
    return {d.get("name"): (d.findtext("value") or "") for d in root.findall("data")}


def load_generated_map(logic: Path) -> dict[str, set[str]]:
    """StrX -> resx keys, from the LocalizedStrings source generator output (if emitted)."""
    out: dict[str, set[str]] = {}
    pat = re.compile(r"(Str\w+)\s*=\s*_translationService\.GetString\(\"([^\"]+)\"\)")
    for f in logic.rglob("*.LocalizedStrings.g.cs"):
        for prop, key in pat.findall(f.read_text(encoding="utf-8", errors="replace")):
            out.setdefault(prop, set()).add(key)
    return out


class Resolver:
    def __init__(self, strings: dict[str, str], gen: dict[str, set[str]]):
        self.s, self.gen = strings, gen

    def key(self, prop: str, view: str) -> str | None:
        name = prop[3:]  # StrFoo -> Foo
        cands = sorted(self.gen.get(prop, set()))
        for pre in PREFIXES.get(view, []):
            if pre + name in self.s:
                return pre + name
            for c in cands:
                if c.startswith(pre):
                    return c
        if len(cands) == 1:
            return cands[0]
        hits = [k for k in self.s if k.endswith("_" + name)]
        return hits[0] if len(hits) == 1 else None

    def text(self, raw: str | None, view: str) -> tuple[str, str | None]:
        """(display text, resx key) for an attribute value."""
        if not raw:
            return "", None
        m = BIND.search(raw)
        if m:
            k = self.key(m.group(1), view)
            return (self.s.get(k, "?") if k else "?"), k
        if raw.startswith("{"):
            return "?", None
        return raw, None


def local(tag: str) -> str:
    return tag.split("}")[-1]


def walk(path: Path, view: str, res: Resolver):
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        print(f"skip {path}: {e}", file=sys.stderr)
        return []
    parent = {c: p for p in root.iter() for c in p}
    rows = []

    def section_path(el):
        names = []
        p = parent.get(el)
        while p is not None:
            if local(p.tag) == "Expander":
                names.append(header_of(p)[0] or p.get(AID) or "Expander")
            p = parent.get(p)
        return " › ".join(reversed(names))

    def header_of(exp):
        h = exp.get("Header")
        if h:
            return res.text(h, view)
        hdr = exp.find(AV + "Expander.Header")
        if hdr is not None:
            for t in hdr.iter(AV + "TextBlock"):
                if t.get("Text"):
                    return res.text(t.get("Text"), view)
        return "", None

    def label_of(el):
        for attr in ("Content", "Text", "Header", "Watermark"):
            if el.get(attr):
                return res.text(el.get(attr), view)
        # a button / tile whose content is a panel: its first text inside
        if local(el.tag) in ("Button", "Border", "RadioButton", "CheckBox", "ToggleButton"):
            for t in el.iter(AV + "TextBlock"):
                if t.get("Text"):
                    return res.text(t.get("Text"), view)
        p = parent.get(el)
        if p is None:
            return "", None
        # Grid layout: the label sits in column 0 of the same row
        row = el.get("Grid.Row", "0")
        if local(p.tag) == "Grid" and el.get("Grid.Column", "0") != "0":
            for sib in p:
                if sib.get("Grid.Row", "0") == row and sib.get("Grid.Column", "0") == "0":
                    for t in sib.iter(AV + "TextBlock"):
                        if t.get("Text"):
                            return res.text(t.get("Text"), view)
        # StackPanel layout: a TextBlock before it in the same panel
        before = []
        for sib in p:
            if sib is el:
                break
            before.append(sib)
        for sib in reversed(before):
            if local(sib.tag) == "TextBlock" and sib.get("Text"):
                return res.text(sib.get("Text"), view)
        return "", None

    def help_of(el):
        """Help keys of the PropertyHelpButton in the same panel or the label's panel."""
        p = parent.get(el)
        scopes = [p] if p is not None else []
        if p is not None and local(p.tag) == "Grid":
            row = el.get("Grid.Row", "0")
            scopes += [s for s in p if s.get("Grid.Row", "0") == row]
        for sc in scopes:
            for h in sc.iter(AV + "PropertyHelpButton") if sc is not None else []:
                hk = res.text(h.get("HelpHeader"), view)[1]
                ck = res.text(h.get("HelpContent"), view)[1]
                if hk or ck:
                    return hk, ck
        return None, None

    for el in root.iter():
        tag = local(el.tag)
        aid = el.get(AID) or el.get(X + "Name") if tag == "Expander" else el.get(AID)
        if tag == "Expander":
            label, key = header_of(el)
        elif aid:
            if tag == "PropertyHelpButton":
                label, key = res.text(el.get("HelpHeader"), view)
            else:
                label, key = label_of(el)
        else:
            continue
        if aid and aid.startswith("{"):
            aid = "{runtime}"
        hk, ck = (res.text(el.get("HelpHeader"), view)[1], res.text(el.get("HelpContent"), view)[1]) \
            if tag == "PropertyHelpButton" else help_of(el) if tag != "Expander" else (None, None)
        rows.append({
            "view": view, "kind": "section" if tag == "Expander" else "control", "type": tag,
            "id": aid or "", "label": label, "labelKey": key, "section": section_path(el),
            "helpHeaderKey": hk, "helpKey": ck,
            "expanded": el.get("IsExpanded"), "line": None,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--splice", action="store_true",
                    help="keep the hand-written part of --out above the GENERATED marker, replace the rest")
    a = ap.parse_args()
    repo = Path(a.repo).resolve()
    app = repo / "aethervk.ui-app"
    logic = repo / "aethervk.ui-logic"
    res = Resolver(load_resx(logic / "Resources" / "AppStrings.resx"), load_generated_map(logic))

    files = [p for p in app.rglob("*.axaml") if "obj" not in p.parts and "bin" not in p.parts
             and p.parent.name not in ("Styles",) and p.name != "App.axaml"]
    def view_name(p):
        rel = p.relative_to(app).with_suffix("")
        parts = rel.parts[1:] if rel.parts[0] in ("Views", "UI", "Controls") else rel.parts
        return "/".join(parts)
    files.sort(key=lambda p: (ORDER.index(view_name(p)) if view_name(p) in ORDER else 999, view_name(p)))

    all_rows = []
    for f in files:
        all_rows += walk(f, view_name(f), res)

    if a.json:
        Path(a.json).write_text(json.dumps(all_rows, indent=1, ensure_ascii=False), encoding="utf-8")

    out = []
    for f in files:
        v = view_name(f)
        rows = [r for r in all_rows if r["view"] == v]
        if not rows:
            continue
        out.append(f"\n## {v}  (`{f.relative_to(repo).as_posix()}`)\n")
        out.append("| Section path | Kind | Control | AutomationId | Label | Help key |")
        out.append("|---|---|---|---|---|---|")
        for r in rows:
            lab = (r["label"] or "").replace("|", "\\|").replace("\n", " ")
            if len(lab) > 60:
                lab = lab[:57] + "..."
            out.append(f"| {r['section'] or '—'} | {r['kind']} | {r['type']} | `{r['id']}` | {lab or '—'} | "
                       f"{('`' + r['helpKey'] + '`') if r['helpKey'] else '—'} |")
    text = "\n".join(out) + "\n"
    if a.out:
        marker = "<!-- GENERATED BELOW: tools/axaml_inventory.py --splice; do not edit by hand -->"
        if a.splice and Path(a.out).exists():
            head = Path(a.out).read_text(encoding="utf-8").split(marker)[0]
            text = head + marker + "\n" + text
        Path(a.out).write_text(text, encoding="utf-8")
    else:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
