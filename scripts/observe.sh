#!/usr/bin/env bash
# Headless comet-dust observer: renders the app's dust scene off screen over a month and writes
# PNG frames + numbers. Wraps `aethervk.core/rlib/examples/dust_observer.rs`.
#
#   scripts/observe.sh [--out DIR] [any dust_observer flag...]
#
# Defaults: --days 30 --step-hours 24 --size 1280x800 --pose up-zenith, output under the session
# `--out DIR` (or `OBSERVER_OUT`), required: no default output directory. Environment:
#   OBSERVER_PROFILE=release|debug   cargo profile (default release; debug enables the Vulkan
#                                    validation layers and the engine debug assertions)
#   OBSERVER_TIMEOUT=SECONDS         outer timeout (default 900)
#   AETHERVK_PARTICLES_CPU=1         CPU particle mode (host-evaluated dust, small ring)
#   AETHERVK_DUST_RING=<pow2>        dust ring capacity override
#   AETHERVK_DISABLE_GPU_AV=1        debug profile only: no GPU-assisted validation (it deadlocks
#                                    the Khronos layer on rapid submits, see memory notes)
#   AETHERVK_GPU_LOST_DUMP=0|1       device-loss dump (kernel Xid lines, nvidia-smi, backtraces) to
#                                    /tmp/aethervk_gpu_lost_<pid>.txt; default 1 here, also in release
#   OBSERVER_MC=0|<grains>           --zoom / --inside: the Monte-Carlo truth per frame (default 8000000)
#   OBSERVER_INSIDE_LIST="D D ..."   --inside: the camera distances from the nucleus surface in metres
#                                    (default "1000 100 10"; decimals down to 0.001 = 1 mm accepted)
#   AETHERVK_DUST_SPECKS=0|1         passed through to every run (0 turns the speck pass off)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${OBSERVER_PROFILE:-release}"
TIMEOUT_S="${OBSERVER_TIMEOUT:-900}"
DEFAULT_OUT="${OBSERVER_OUT:-}"

# --zoom: the nine-scale zoom series (5 km .. 1 AU half-height at one epoch, a pre-existing tail so
# the AU scale holds a 5-year trail), one directory per scale under --out, plus zoom_sheet.png and
# the per-scale ripple metric; the lattice detector (`--ripple-max`) runs through --check
if [[ "${1:-}" == "--zoom" ]]; then
  shift
  ZOUT="${OBSERVER_OUT:-}"
  ZARGS=()
  while [[ $# -gt 0 ]]; do case "$1" in --out) ZOUT="$2"; shift 2 ;; *) ZARGS+=("$1"); shift ;; esac; done
  if [[ -z "$ZOUT" ]]; then echo "observe.sh --zoom: pass --out DIR or set OBSERVER_OUT" >&2; exit 2; fi
  mkdir -p "$ZOUT"
  STATUS=0
  # the Monte-Carlo truth per scale (OBSERVER_MC=0 turns it off; OBSERVER_MC=<grains> sets it)
  MC="${OBSERVER_MC:-8000000}"
  MCARGS=(); [[ "$MC" != "0" ]] && MCARGS=(--mc "$MC")
  for H in 5 50 500 5000 50000 500000 5000000 50000000 150000000; do
    echo "=== half-height $H km"
    TIER=(); [[ "$H" -le 500 ]] && TIER=(--mc-tier 0)
    "$0" --out "$ZOUT/h$H" --prestart --start 2025-10-17 --days 0 --step-hours 24 --size 960x600 ${TIER[@]+"${TIER[@]}"} \
      --half-height-km "$H" --check --quiet ${MCARGS[@]+"${MCARGS[@]}"} ${ZARGS[@]+"${ZARGS[@]}"} 2>&1 | grep -E "mc frame 1|ripple frame 1|done:|failure|lost" || STATUS=1
  done
  python3 - "$ZOUT" <<'PY' || true
import sys, os
try:
    from PIL import Image, ImageDraw
except ImportError:
    print("PIL missing: no zoom_sheet.png"); sys.exit(0)
out=sys.argv[1]; hs=[5,50,500,5000,50000,500000,5000000,50000000,150000000]; tiles=[]
for h in hs:
    f=os.path.join(out,f"h{h}","frame_001.png")
    if not os.path.exists(f): f=os.path.join(out,f"h{h}","frame_000.png")
    if not os.path.exists(f): continue
    im=Image.open(f).convert("RGB").resize((480,300)); d=ImageDraw.Draw(im)
    d.rectangle([0,0,250,16],fill=(0,0,0)); d.text((4,2),f"half-height {h} km ({h*2/960:.3g} km/px)",fill=(255,255,255)); tiles.append(im)
W=Image.new("RGB",(480*3,300*((len(tiles)+2)//3)))
for i,t in enumerate(tiles): W.paste(t,((i%3)*480,(i//3)*300))
W.save(os.path.join(out,"zoom_sheet.png")); print("zoom_sheet.png written")
PY
  exit $STATUS
fi

# --inside: the near-field sweep (WP3): a perspective camera D metres sunward of the nucleus
# surface (D + 2 km from the centre) on the anti-solar axis looking through it down the tail (`--pose inside --inside-m D`, fov 60°), one
# process per D under --out (inside_<D>m), the pre-existing tail at one epoch like --zoom, 960x600,
# --check and the Monte-Carlo truth, then inside_sheet.png. OBSERVER_INSIDE_LIST overrides the
# distances (space-separated metres: the later 1 mm .. 1 AU sweep reuses this entry point).
if [[ "${1:-}" == "--inside" ]]; then
  shift
  IOUT="${OBSERVER_OUT:-}"
  IARGS=()
  while [[ $# -gt 0 ]]; do case "$1" in --out) IOUT="$2"; shift 2 ;; *) IARGS+=("$1"); shift ;; esac; done
  if [[ -z "$IOUT" ]]; then echo "observe.sh --inside: pass --out DIR or set OBSERVER_OUT" >&2; exit 2; fi
  mkdir -p "$IOUT"
  STATUS=0
  MC="${OBSERVER_MC:-8000000}"
  MCARGS=(); [[ "$MC" != "0" ]] && MCARGS=(--mc "$MC")
  LIST="${OBSERVER_INSIDE_LIST:-1000 100 10}"
  for D in $LIST; do
    if ! python3 -c 'import sys; v=float(sys.argv[1]); sys.exit(0 if 0.001 <= v <= 1.5e11 else 1)' "$D" 2>/dev/null; then
      echo "observe.sh --inside: distance '$D' m is not a number in [0.001, 1.5e11]" >&2; exit 2
    fi
  done
  if [[ -n "${AETHERVK_DUST_SPECKS:-}" ]]; then export AETHERVK_DUST_SPECKS; echo "[observe.sh] AETHERVK_DUST_SPECKS=$AETHERVK_DUST_SPECKS"; fi
  for D in $LIST; do
    echo "=== inside $D m"
    "$0" --out "$IOUT/inside_${D}m" --pose inside --inside-m "$D" --prestart --start 2025-10-17 --days 0 --step-hours 24 --size 960x600 \
      --check --quiet ${MCARGS[@]+"${MCARGS[@]}"} ${IARGS[@]+"${IARGS[@]}"} 2>&1 | grep -E "inside pose|mc frame|ripple frame|pyramid frame|done:|failure|lost|overflow" || STATUS=1
  done
  python3 - "$IOUT" $LIST <<'PY' || true
import sys, os
try:
    from PIL import Image, ImageDraw
except ImportError:
    print("PIL missing: no inside_sheet.png"); sys.exit(0)
out=sys.argv[1]; ds=sys.argv[2:]; tiles=[]
for d in ds:
    f=os.path.join(out,f"inside_{d}m","frame_001.png")
    if not os.path.exists(f): f=os.path.join(out,f"inside_{d}m","frame_000.png")
    if not os.path.exists(f): continue
    im=Image.open(f).convert("RGB").resize((480,300)); dr=ImageDraw.Draw(im)
    dr.rectangle([0,0,300,16],fill=(0,0,0))
    dr.text((4,2),f"inside {d} m from the surface ({2*(float(d)+2000.0)*0.5773502692/600:.3g} m/px at the nucleus)",fill=(255,255,255)); tiles.append(im)
if not tiles: print("no frames: no inside_sheet.png"); sys.exit(0)
W=Image.new("RGB",(480*3,300*((len(tiles)+2)//3)))
for i,t in enumerate(tiles): W.paste(t,((i%3)*480,(i//3)*300))
W.save(os.path.join(out,"inside_sheet.png")); print("inside_sheet.png written")
PY
  # the sweep's smoothness summary: per D the last frame's displayed mean luminance and white
  # point (mag/arcsec²); a neighbour step > 25 % in either (the white point as a radiance ratio),
  # a black frame (non-black fraction < 0.1 %) or a saturated one (mean luminance > 230/255) is
  # flagged (exit 1); written to inside_summary.txt
  python3 - "$IOUT" $LIST <<'PY' || STATUS=1
import sys, os, json
out = sys.argv[1]; ds = sys.argv[2:]; rows = []; flags = []
for d in ds:
    f = os.path.join(out, f"inside_{d}m", "stats.jsonl")
    try:
        lines = [l for l in open(f) if l.strip()]
        r = json.loads(lines[-1])
    except Exception as e:
        flags.append(f"D {d} m: no stats ({e})"); continue
    ex = r.get("exposure") or {}
    rows.append((d, r.get("mean_luminance"), ex.get("mag_arcsec2"), r.get("nonblack_fraction"), r.get("p99_luminance")))
lines = ["D_m  mean_L  white_mag_arcsec2  nonblack  p99_L"]
for d, l, m, nb, p in rows:
    lines.append(f"{d}  {l if l is None else round(l, 3)}  {m if m is None else round(m, 3)}  {nb}  {p}")
    if nb is not None and nb < 1e-3: flags.append(f"D {d} m: black frame (non-black {nb:.5f})")
    if l is not None and l > 230: flags.append(f"D {d} m: saturated frame (mean L {l:.1f})")
for (da, la, ma, _, _), (db, lb, mb, _, _) in zip(rows, rows[1:]):
    if la and lb and abs(lb / la - 1) > 0.25:
        flags.append(f"D {da} -> {db} m: mean luminance step {la:.2f} -> {lb:.2f} ({100 * (lb / la - 1):+.0f} %)")
    if ma is not None and mb is not None:
        ratio = 10 ** (-0.4 * (mb - ma))
        if abs(ratio - 1) > 0.25:
            flags.append(f"D {da} -> {db} m: white point step {ma:.2f} -> {mb:.2f} mag/arcsec2 ({100 * (ratio - 1):+.0f} % in radiance)")
lines += ["", f"{len(flags)} flag(s)"] + flags
text = "\n".join(lines) + "\n"
print(text, end="")
open(os.path.join(out, "inside_summary.txt"), "w").write(text)
sys.exit(1 if flags else 0)
PY
  exit $STATUS
fi

# split user args: honour --out if given, pass everything through
OUT=""
PASS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out) OUT="$2"; shift 2 ;;
    *) PASS+=("$1"); shift ;;
  esac
done
OUT="${OUT:-$DEFAULT_OUT}"
if [[ -z "$OUT" ]]; then echo "observe.sh: pass --out DIR or set OBSERVER_OUT" >&2; exit 2; fi
mkdir -p "$OUT"

# defaults only when the user did not pass the flag
# (`${PASS[@]+"${PASS[@]}"}` expands to nothing for an empty array under `set -u`; `:-` would
# inject an empty argument, which the binary rejects with a usage error)
has_flag() { local f="$1"; for a in ${PASS[@]+"${PASS[@]}"}; do [[ "$a" == "$f" ]] && return 0; done; return 1; }
# --shape: the automated "watch the tail grow" run (the shape analysis of --check on 6-hour steps)
SHAPE=0
for a in ${PASS[@]+"${PASS[@]}"}; do [[ "$a" == "--shape" ]] && SHAPE=1; done
if [[ $SHAPE -eq 1 ]]; then
  KEEP=(); for a in ${PASS[@]+"${PASS[@]}"}; do [[ -n "$a" && "$a" != "--shape" ]] && KEEP+=("$a"); done
  PASS=(); for a in ${KEEP[@]+"${KEEP[@]}"}; do PASS+=("$a"); done
  has_flag --check || PASS+=(--check)
  has_flag --days || PASS+=(--days 3)
  has_flag --step-hours || PASS+=(--step-hours 6)
  has_flag --size || PASS+=(--size 960x600)
fi
has_flag --days || PASS+=(--days 30)
has_flag --step-hours || PASS+=(--step-hours 24)
has_flag --size || PASS+=(--size 1280x800)
has_flag --pose || PASS+=(--pose up-zenith)

PROFILE_FLAG=()
BIN="$ROOT/target/debug/examples/dust_observer"
if [[ "$PROFILE" == "release" ]]; then
  PROFILE_FLAG=(--release)
  BIN="$ROOT/target/release/examples/dust_observer"
fi

echo "[observe.sh] building ($PROFILE)..."
(cd "$ROOT" && cargo build -p aethervk-core-rlib --example dust_observer --features std "${PROFILE_FLAG[@]}" 2>&1 | grep -E "^(error|warning: unused)|Finished|error\[" || true)
[[ -x "$BIN" ]] || { echo "[observe.sh] build failed: $BIN missing" >&2; exit 2; }

export AETHERVK_GPU_LOST_DUMP="${AETHERVK_GPU_LOST_DUMP:-1}"
echo "[observe.sh] running: $BIN --out $OUT ${PASS[*]}"
set +e
timeout "$TIMEOUT_S" "$BIN" --out "$OUT" "${PASS[@]}" > "$OUT/stdout.txt" 2>&1
CODE=$?
set -e
echo "[observe.sh] exit $CODE; outputs in $OUT"
grep -h "^\[observer\]" "$OUT/log.txt" 2>/dev/null | tail -n 45 || true
grep -h "^\[GPU lost\]\|^\[recycle\]" "$OUT/log.txt" "$OUT/stdout.txt" 2>/dev/null | grep -v "^\[GDB\]" | head -n 40 || true
if [[ $CODE -eq 124 ]]; then
  echo "[observe.sh] TIMEOUT after ${TIMEOUT_S}s (a hung Vulkan process with idle threads is usually the validation-layer lock inversion: retry with AETHERVK_DISABLE_GPU_AV=1)" >&2
fi
exit $CODE
