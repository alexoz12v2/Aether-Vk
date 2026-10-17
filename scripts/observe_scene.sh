#!/usr/bin/env bash
# Headless Earth-visual observer: renders the app's scene with the real ephemeris and Earth
# orientation at a few epochs and checks the globe, its polar axis and labels, the per-camera
# render layers and the seasons. Wraps `aethervk.core/rlib/examples/scene_observer.rs`.
#
#   scripts/observe_scene.sh --out DIR [--check] [any scene_observer flag...]
#
# `--out DIR` (or `OBSERVER_OUT`) is required. The run is wrapped in a memory cap and a timeout
# (one GPU run at a time; wait while `pgrep -f dust_observer` shows the dust observer). Environment:
#   OBSERVER_PROFILE=release|debug   cargo profile (default release)
#   OBSERVER_TIMEOUT=SECONDS         outer timeout (default 900)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${OBSERVER_PROFILE:-release}"
TIMEOUT_S="${OBSERVER_TIMEOUT:-900}"

ARGS=("$@")
HAS_OUT=0
for a in "$@"; do [[ "$a" == "--out" ]] && HAS_OUT=1; done
if [[ $HAS_OUT -eq 0 ]]; then
  if [[ -z "${OBSERVER_OUT:-}" ]]; then
    echo "observe_scene.sh: pass --out DIR or set OBSERVER_OUT" >&2
    exit 2
  fi
  ARGS+=(--out "$OBSERVER_OUT")
fi

if pgrep -f dust_observer >/dev/null 2>&1; then
  echo "observe_scene.sh: a dust_observer run is in progress; run again when it is done" >&2
  exit 3
fi

PROFILE_FLAG=(--release)
BIN_DIR="release"
if [[ "$PROFILE" == "debug" ]]; then PROFILE_FLAG=(); BIN_DIR="debug"; fi

cd "$ROOT"
cargo build -p aethervk-core-rlib --example scene_observer --features std "${PROFILE_FLAG[@]}" >&2
exec timeout "$TIMEOUT_S" systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  "$ROOT/target/$BIN_DIR/examples/scene_observer" "${ARGS[@]}"
