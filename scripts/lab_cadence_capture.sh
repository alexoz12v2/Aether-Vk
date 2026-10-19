#!/usr/bin/env bash
# Temporal reuse, step 4: the camera-path captures and their score (the user's image gate).
#
# Every run is one process under the memory cap with its own timeout, one after the other. A
# capture is a 32-frame nucleus-relative orbit (10°/s at 60 fps) of the inside pose at one
# simulation speed (Δt = speed / 60 s per frame), at 1280×800; frames 0 and 31 keep their
# pyramid and truth inputs (the full variant also its snapshots). Then each variant's frames 0
# and 31 are scored against the full variant's truth (8 M grains, drawn once, cached as
# truth8000000_NNN.f64), the rows appended to $ROOT/table.tsv, the Markdown to $ROOT/table.md.
#
#   scripts/lab_cadence_capture.sh --smoke                    # 1 km, 8 frames, s60, full + aa, frame 7
#   scripts/lab_cadence_capture.sh                            # 3 poses × 3 speeds × full + aa
#   scripts/lab_cadence_capture.sh --variants "eps1 eps2 eps4 eps16"   # once the GPU cadence exists
#   scripts/lab_cadence_capture.sh --score-only               # rescore the existing captures
#
# Variants: full (AETHERVK_DUST_CADENCE=0, the fully drawn path, with snapshots), aa (the same
# again: the run-to-run floor), epsE (the cadence at ε = E px), fb (the by-count fallback: its knob
# arrives with step 5; until then the run is skipped). epsE also sets AETHERVK_DUST_CADENCE=1 (the GPU
# cadence is opt-in). A cadenced variant whose run reports no cadence telemetry is scored but marked
# by its path summary (cadence n/a).
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
B="${OBSERVER_BIN:-$REPO/target/release/examples/dust_observer}"
ROOT="${LAB_ROOT:-$HOME/aethervk-lab/lab}"
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
POSES="1km 100m 5000km"
SPEEDS="0 60 86400"
VARIANTS="full aa"
FRAMES=32
TRUTH=8000000
SIZE=1280x800
SCORE_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) POSES="1km"; SPEEDS="60"; VARIANTS="full aa"; FRAMES=8; ROOT="$ROOT/smoke_cad";;
    --poses) POSES="$2"; shift;;
    --speeds) SPEEDS="$2"; shift;;
    --variants) VARIANTS="$2"; shift;;
    --frames) FRAMES="$2"; shift;;
    --truth) TRUTH="$2"; shift;;
    --size) SIZE="$2"; shift;;
    --score-only) SCORE_ONLY=1;;
    *) echo "unknown argument $1"; exit 2;;
  esac
  shift
done
[ -x "$B" ] || { echo "no observer binary at $B (build it, or set OBSERVER_BIN)"; exit 2; }
LAST=$((FRAMES - 1))
pose_args() {
  case "$1" in
    1km) echo "--inside-m 1000";;
    100m) echo "--inside-m 100";;
    5000km) echo "--inside-km 5000";;
    *) echo "unknown pose $1" >&2; exit 2;;
  esac
}
variant_env() {
  case "$1" in
    full|aa) echo "AETHERVK_DUST_CADENCE=0";;
    eps*) echo "AETHERVK_DUST_CADENCE=1 AETHERVK_DUST_CADENCE_EPS=${1#eps}";;
    fb) echo "";;
    *) echo "unknown variant $1" >&2; exit 2;;
  esac
}
FAIL=0
for pose in $POSES; do
  for speed in $SPEEDS; do
    BASE="$ROOT/${pose}_cad/s$speed"
    FULL="$BASE/full"
    for v in $VARIANTS; do
      D="$BASE/$v"
      if [ "$SCORE_ONLY" = 0 ]; then
        if [ "$v" = fb ]; then
          echo "SKIP $pose s$speed fb: the by-count fallback's knob arrives with step 5"
          continue
        fi
        mkdir -p "$D"
        SNAP=""
        [ "$v" = full ] && SNAP="--save-snapshot"
        echo "RUNNING: capture $pose s$speed $v → $D"
        timeout 900 $CAP env OBSERVER_PATH_VARIANT="$v" $(variant_env "$v") "$B" --out "$D" --pose inside $(pose_args "$pose") \
          --prestart --start 2025-10-17 --days 0 --size "$SIZE" --quiet --no-sheet --mc 0 $SNAP \
          --path-frames "$FRAMES" --path-speed "$speed" --orbit-deg-per-s 10 --check > "$D/capture.log" 2>&1
        rc=$?
        echo "capture $pose s$speed $v: exit $rc ($(grep -c 'camera path frame' "$D/log.txt" 2>/dev/null) path lines; $(grep 'camera path:' "$D/log.txt" 2>/dev/null | tail -1))"
        [ $rc = 0 ] || FAIL=1
      fi
    done
    for v in $VARIANTS; do
      D="$BASE/$v"
      [ -f "$D/frame_$(printf %03d $LAST).pyr" ] || continue
      for f in 0 $LAST; do
        echo "RUNNING: score $pose s$speed $v frame $f"
        timeout 900 $CAP "$B" --out "$D" --lab-score "$D" --against "$FULL" --lab-frame "$f" --lab-truth "$TRUTH" \
          --lab-label "$v" --lab-table "$ROOT/table.tsv" --check > "$D/score_$(printf %03d $f).log" 2>&1
        rc=$?
        grep -E '^(score|check)' "$D/score_$(printf %03d $f).log" | sed "s/^/  [$pose s$speed $v $f] /"
        [ $rc = 0 ] || FAIL=1
      done
    done
  done
done
[ -f "$ROOT/table.tsv" ] && "$B" --out "$ROOT" --lab-table-md "$ROOT/table.tsv" > "$ROOT/table.md" && echo "table: $ROOT/table.md"
exit $FAIL
