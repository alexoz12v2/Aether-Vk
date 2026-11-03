#!/usr/bin/env python3
"""Check the user manual's sources before a PDF build.

    python3 -I docs/manual/tools/check_manual.py

Fails (exit 1, one file:line per problem) when a docs/manual/adoc/*.adoc file still contains
"TODO", when docs/manual/descriptions.json is not valid JSON, or when one of its "controls" entries
has no "text" (or "help_ids" is not a map of strings). Standard library only.
"""
import json
import sys
from pathlib import Path

MANUAL = Path(__file__).resolve().parent.parent
CONTROLS = '"controls"'
HELP_IDS = '"help_ids"'


def check_adoc(problems):
    files = sorted((MANUAL / "adoc").glob("*.adoc"))
    if not files:
        problems.append(f"{MANUAL / 'adoc'}: no .adoc files")
    for path in files:
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "TODO" in line:
                problems.append(f"{path}:{n}: {line.strip()}")


def line_of(text, needle):
    """1-based line of the first occurrence of needle in text (1 when absent)."""
    i = text.find(needle)
    return text.count("\n", 0, i) + 1 if i >= 0 else 1


def check_descriptions(problems):
    path = MANUAL / "descriptions.json"
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except (OSError, UnicodeDecodeError) as e:
        problems.append(f"{path}:1: unreadable: {e}")
        return
    except json.JSONDecodeError as e:
        problems.append(f"{path}:{e.lineno}: invalid JSON: {e.msg}")
        return
    if not isinstance(data, dict):
        problems.append(f"{path}:1: the top level is not an object")
        return
    controls = data.get("controls", {})
    if not isinstance(controls, dict):
        problems.append(f"{path}:{line_of(text, CONTROLS)}: {CONTROLS} is not an object")
        return
    for cid, entry in controls.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not entry["text"].strip():
            problems.append(f"{path}:{line_of(text, json.dumps(cid, ensure_ascii=False) + ':')}: control {cid!r} has no \"text\"")
    help_ids = data.get("help_ids", {})
    if not isinstance(help_ids, dict) or not all(isinstance(v, str) for v in help_ids.values()):
        problems.append(f"{path}:{line_of(text, HELP_IDS)}: {HELP_IDS} is not a map of strings")


def main():
    problems = []
    check_adoc(problems)
    check_descriptions(problems)
    for p in problems:
        print(p)
    if problems:
        print(f"check_manual: {len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("check_manual: ok (no TODO in adoc/*.adoc, descriptions.json valid)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
