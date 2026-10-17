#!/usr/bin/env bash
# Long GPU soak of ONE dust_observer run: samples the observer's memory (host and GPU), threads and
# fds every few seconds, then judges leaks, driver faults and frame-time drift.
#
#   scripts/soak.sh <outdir> <label> -- <dust_observer args...>
#
# Writes into <outdir>:
#   <label>.log          the observer's stdout+stderr
#   <label>.csv          one row per sample (see CSV_HEADER below)
#   <label>_kernel.txt   `journalctl -k --since <start>` lines matching Xid|NVRM at the end
#   <label>_summary.txt  exit code, frames, log error counts, regressions, PASS/FAIL per criterion
#   <label>.out/         the observer's --out (only when the args do not carry their own --out)
#
# Environment:
#   SOAK_TIMEOUT       outer `timeout` duration (default 6h)
#   SOAK_PERIOD        sampling period in seconds (default 5)
#   SOAK_SHEET         1 keeps the observer's contact sheet (default: --no-sheet is added, so the
#                      observer holds no frame in memory and RssAnon is the engine's)
#   SOAK_WARMUP_DAYS   samples before start + this many simulated days are excluded from the
#                      regressions (default 10; without a simulated-time column: the first 1/6)
#   OBSERVER_BIN       the observer binary (default: the release build observe.sh uses, built the
#                      same way when missing)
#
# Exit status: the observer's (124 = SOAK_TIMEOUT hit, 137 = SIGKILL, e.g. the 6G scope's OOM
# kill) when non-zero, else 1 when a summary criterion FAILed, else 0.
#
# The observer runs as `systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 timeout
# $SOAK_TIMEOUT <bin> ...`. Ctrl-C / SIGTERM / SIGHUP kill only the observer pid this script found
# (a descendant of the launcher whose /proc/<pid>/exe is the binary), then the summary is written.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { echo "usage: scripts/soak.sh <outdir> <label> -- <dust_observer args...>" >&2; exit 2; }
[[ $# -ge 3 ]] || usage
OUTDIR="$1"; LABEL="$2"; shift 2
[[ "${1:-}" == "--" ]] || usage
shift
OBS_ARGS=("$@")
[[ "$LABEL" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "soak.sh: label must be [A-Za-z0-9._-]+" >&2; exit 2; }

SOAK_TIMEOUT="${SOAK_TIMEOUT:-6h}"
SOAK_PERIOD="${SOAK_PERIOD:-5}"
SOAK_WARMUP_DAYS="${SOAK_WARMUP_DAYS:-10}"

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
LOG="$OUTDIR/$LABEL.log"
CSV="$OUTDIR/$LABEL.csv"
KERN="$OUTDIR/$LABEL""_kernel.txt"
SUMMARY="$OUTDIR/$LABEL""_summary.txt"

# the observer's own --out (its log.txt holds the engine lines even under --quiet)
OBS_OUT=""
# --no-sheet by default (SOAK_SHEET=1 keeps the contact sheet): the observer then holds no frame
# in memory and the anon RSS is the engine's
if [[ "${SOAK_SHEET:-0}" != "1" && " ${OBS_ARGS[*]-} " != *" --no-sheet "* ]]; then
  OBS_ARGS+=(--no-sheet)
fi
for ((k = 0; k < ${#OBS_ARGS[@]}; k++)); do
  if [[ "${OBS_ARGS[$k]}" == "--out" && $((k + 1)) -lt ${#OBS_ARGS[@]} ]]; then OBS_OUT="${OBS_ARGS[$((k + 1))]}"; fi
done
if [[ -z "$OBS_OUT" ]]; then
  OBS_OUT="$OUTDIR/$LABEL.out"
  OBS_ARGS=(--out "$OBS_OUT" ${OBS_ARGS[@]+"${OBS_ARGS[@]}"})
fi

# the binary: OBSERVER_BIN, else observe.sh's release build (built like observe.sh when missing)
if [[ -n "${OBSERVER_BIN:-}" ]]; then
  BIN="$OBSERVER_BIN"
else
  BIN="$ROOT/target/release/examples/dust_observer"
  if [[ ! -x "$BIN" ]]; then
    echo "[soak.sh] building (release)..."
    (cd "$ROOT" && cargo build -p aethervk-core-rlib --example dust_observer --features std --release 2>&1 | grep -E "^(error|warning: unused)|Finished|error\[" || true)
  fi
fi
[[ -x "$BIN" ]] || { echo "[soak.sh] observer binary missing: $BIN" >&2; exit 2; }
BIN_REAL="$(readlink -f "$BIN")"

for tool in systemd-run timeout python3; do
  command -v "$tool" >/dev/null || { echo "[soak.sh] $tool not found" >&2; exit 2; }
done

CSV_HEADER="unix_time,elapsed_s,vmrss_kb,rss_anon_kb,rss_file_kb,rss_shmem_kb,vmsize_kb,threads,fds,gpu_proc_mib,gpu_total_mib,frame,sim_time,frame_wall_s"

# ───── helpers ─────

# the observer: a descendant of the launcher whose exe (or argv[0]) is exactly the binary
find_observer() {
  local queue=("$1") p c exe argv0
  while [[ ${#queue[@]} -gt 0 ]]; do
    p="${queue[0]}"; queue=("${queue[@]:1}")
    for c in $(pgrep -P "$p" 2>/dev/null); do
      exe="$(readlink "/proc/$c/exe" 2>/dev/null || true)"; exe="${exe% (deleted)}"
      argv0="$(tr '\0' '\n' < "/proc/$c/cmdline" 2>/dev/null | head -n 1 || true)"
      if [[ "$exe" == "$BIN_REAL" || "$argv0" == "$BIN" || "$argv0" == "$BIN_REAL" ]]; then
        echo "$c"; return 0
      fi
      queue+=("$c")
    done
  done
  return 1
}

# true while $1 is still our observer (alive, not a zombie, same exe: no pid reuse)
observer_alive() {
  local pid="$1" st exe
  [[ -n "$pid" && -r "/proc/$pid/status" ]] || return 1
  st="$(awk '/^State:/ {print $2; exit}' "/proc/$pid/status" 2>/dev/null)"
  [[ -n "$st" && "$st" != "Z" && "$st" != "X" ]] || return 1
  exe="$(readlink "/proc/$pid/exe" 2>/dev/null || true)"; exe="${exe% (deleted)}"
  [[ -z "$exe" || "$exe" == "$BIN_REAL" ]]
}

# per-process GPU memory (MiB): compute-apps first, then the plain `nvidia-smi` process table
# (Vulkan graphics processes are listed there as type G / C+G, not in compute-apps)
gpu_proc_mib() {
  local pid="$1" v=""
  command -v nvidia-smi >/dev/null || { echo ""; return; }
  v="$(timeout 10 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
      | awk -F', *' -v p="$pid" '$1 == p {s += $2; f = 1} END {if (f) print s}')"
  if [[ -z "$v" ]]; then
    v="$(timeout 10 nvidia-smi 2>/dev/null | awk -v p="$pid" '
      /MiB/ { for (i = 1; i <= NF; i++) if ($i == p) { if (match($0, /[0-9]+MiB[ \t]*\|?[ \t]*$/)) { m = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", m); s += m; f = 1 } break } }
      END { if (f) print s }')"
  fi
  echo "$v"
}

gpu_total_mib() {
  command -v nvidia-smi >/dev/null || { echo ""; return; }
  timeout 10 nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk '{gsub(/ /, ""); if ($0 ~ /^[0-9]+$/) s += $0; f = 1} END {if (f) print s}'
}

# the latest "[observer] frame NNN <epoch> ... | X.Xs" line of the log: "frame,sim_time,wall_s"
latest_frame() {
  awk '
    /^\[observer\] frame [0-9]+ [0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]T/ {
      f = $3 + 0; t = $4; w = ""
      if (match($0, /\| [0-9.]+s$/)) { w = substr($0, RSTART + 2, RLENGTH - 3) }
      last = f "," t "," w
    }
    END { print (last == "" ? ",," : last) }' "$LOG" 2>/dev/null || echo ",,"
}

sample() {
  local pid="$1" now st fds gp gt fr
  now="$(date +%s.%N)"
  st="$(awk '
    /^VmRSS:/ {rss = $2} /^RssAnon:/ {anon = $2} /^RssFile:/ {file = $2} /^RssShmem:/ {shm = $2}
    /^VmSize:/ {vsz = $2} /^Threads:/ {thr = $2}
    END {print rss "," anon "," file "," shm "," vsz "," thr}' "/proc/$pid/status" 2>/dev/null)" || return 0
  [[ -n "$st" && "$st" != ",,,,," ]] || return 0
  fds="$(find "/proc/$pid/fd" -mindepth 1 -maxdepth 1 2>/dev/null | wc -l)"
  gp="$(gpu_proc_mib "$pid")"
  gt="$(gpu_total_mib)"
  fr="$(latest_frame)"
  awk -v n="$now" -v s="$START_S" 'BEGIN {printf "%.3f,%.1f,", n, n - s}' >> "$CSV"
  echo "$st,$fds,$gp,$gt,$fr" >> "$CSV"
}

# ───── run ─────

START_S="$(date +%s.%N)"
START_JOURNAL="$(date -d "@${START_S%.*}" '+%Y-%m-%d %H:%M:%S')"
echo "$CSV_HEADER" > "$CSV"
: > "$LOG"

OBS_PID=""
LAUNCHER=""
INTERRUPTED=""
on_signal() {
  INTERRUPTED="$1"
  echo "[soak.sh] $1: stopping the observer (pid ${OBS_PID:-not found yet})" >&2
  if [[ -n "$OBS_PID" ]] && observer_alive "$OBS_PID"; then kill -TERM "$OBS_PID" 2>/dev/null || true; fi
}
trap 'on_signal SIGINT' INT
trap 'on_signal SIGTERM' TERM
trap 'on_signal SIGHUP' HUP

export AETHERVK_GPU_LOST_DUMP="${AETHERVK_GPU_LOST_DUMP:-1}"
echo "[soak.sh] $(date -Is) running: $BIN ${OBS_ARGS[*]} (timeout $SOAK_TIMEOUT, sample every ${SOAK_PERIOD}s)"
systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 \
  timeout "$SOAK_TIMEOUT" "$BIN" "${OBS_ARGS[@]}" > "$LOG" 2>&1 &
LAUNCHER=$!

# find the observer: up to 60 s (the scope registration and the exec are quick)
for _ in $(seq 1 120); do
  OBS_PID="$(find_observer "$LAUNCHER" || true)"
  [[ -n "$OBS_PID" || -n "$INTERRUPTED" ]] && break
  kill -0 "$LAUNCHER" 2>/dev/null || break
  sleep 0.5
done
if [[ -n "$OBS_PID" ]]; then
  echo "[soak.sh] observer pid $OBS_PID (launcher $LAUNCHER)"
  [[ -n "$INTERRUPTED" ]] && kill -TERM "$OBS_PID" 2>/dev/null
else
  echo "[soak.sh] observer pid not found under launcher $LAUNCHER (it may have exited already)" >&2
fi

# sample while it lives (the sleep runs in the background so a signal is handled at once)
while [[ -n "$OBS_PID" ]] && observer_alive "$OBS_PID"; do
  sample "$OBS_PID"
  sleep "$SOAK_PERIOD" & wait $! 2>/dev/null
  if [[ -n "$INTERRUPTED" ]]; then
    # SIGTERM was sent; escalate to SIGKILL on that pid only after 15 s
    for _ in $(seq 1 30); do observer_alive "$OBS_PID" || break; sleep 0.5; done
    if observer_alive "$OBS_PID"; then
      echo "[soak.sh] observer ignored SIGTERM: SIGKILL $OBS_PID" >&2
      kill -KILL "$OBS_PID" 2>/dev/null || true
    fi
    break
  fi
done

CODE=""
while :; do
  wait "$LAUNCHER" 2>/dev/null; CODE=$?
  # a signal interrupts `wait` (status > 128) while the launcher still runs: wait again
  kill -0 "$LAUNCHER" 2>/dev/null || break
  if [[ -n "$INTERRUPTED" && -n "$OBS_PID" ]] && observer_alive "$OBS_PID"; then kill -KILL "$OBS_PID" 2>/dev/null || true; fi
  sleep 0.5
done
END_S="$(date +%s.%N)"
echo "[soak.sh] $(date -Is) launcher exit $CODE${INTERRUPTED:+ (interrupted: $INTERRUPTED)}"

# new kernel GPU lines (dmesg is not permitted here)
JOURNAL_OK=1
if ! journalctl -k --no-pager -q -o short-iso --since "$START_JOURNAL" 2>/dev/null | grep -E 'Xid|NVRM' > "$KERN"; then
  # grep exits 1 when nothing matched: distinguish an unreadable journal
  journalctl -k --no-pager -q -n 1 >/dev/null 2>&1 || JOURNAL_OK=0
fi

# ───── summary ─────
python3 -I - "$CSV" "$LOG" "$OBS_OUT/log.txt" "$KERN" "$SUMMARY" "$CODE" "${INTERRUPTED:-}" \
  "$SOAK_WARMUP_DAYS" "$START_S" "$END_S" "$JOURNAL_OK" "$BIN" "$LABEL" "${OBS_ARGS[*]}" "${OBS_PID:-}" <<'PY'
import sys, os, re, csv, datetime, statistics

(csv_p, log_p, obslog_p, kern_p, out_p, code, interrupted, warm_days, start_s, end_s, journal_ok,
 bin_p, label, obs_args, obs_pid) = sys.argv[1:16]
warm_days = float(warm_days); start_s = float(start_s); end_s = float(end_s)
code = int(code) if code.strip().lstrip("-").isdigit() else None

def read_lines(p):
    try:
        with open(p, "r", errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return None

def parse_epoch(s):
    m = re.match(r"(\d{4})-(\d\d)-(\d\d)T(\d\d):(\d\d):(\d\d)(?:\.(\d+))?", s or "")
    if not m:
        return None
    y, mo, d, h, mi, se = (int(m.group(k)) for k in range(1, 7))
    frac = float("0." + m.group(7)) if m.group(7) else 0.0
    return datetime.datetime(y, mo, d, h, mi, se).timestamp() + frac  # naive: differences only

log = read_lines(log_p) or []
obslog = read_lines(obslog_p)

# run parameters from the observer's own "run:" line
run_start = None; frames_expected = None; size = None
for l in log + (obslog or []):
    m = re.search(r"\[observer\] run: start (\S+) .*? frames (\d+) step .*? size (\d+)x(\d+)", l)
    if m:
        run_start = parse_epoch(m.group(1)); frames_expected = int(m.group(2))
        size = (int(m.group(3)), int(m.group(4))); break

frame_re = re.compile(r"^\[observer\] frame (\d+) (\d{4}-\d\d-\d\dT\S+)(?: \S+)? .*\| ([0-9.]+)s$")
frames = {}
for l in log:
    m = frame_re.match(l)
    if m:
        frames[int(m.group(1))] = (m.group(2), float(m.group(3)))
if not frames and obslog:
    for l in obslog:
        m = frame_re.match(l)
        if m:
            frames[int(m.group(1))] = (m.group(2), float(m.group(3)))
frames_done = len(frames)

# log line counts per pattern, per file
pats = {
    "device lost": re.compile(r"DEVICE_LOST|DeviceLost|device lost|\[GPU lost\]", re.I),
    "timeout": re.compile(r"timed out|timeout", re.I),
    "download fail": re.compile(r"download failed|download fail", re.I),
    "panic": re.compile(r"panicked|\bpanic\b", re.I),
    "validation error": re.compile(r"VULKAN VALIDATION ERROR"),
    "error": re.compile(r"\berror\b", re.I),
}
counts = {}
for name, lines in (("stdout", log), ("log.txt", obslog)):
    if lines is None:
        continue
    counts[name] = {k: sum(1 for l in lines if p.search(l)) for k, p in pats.items()}
def worst(k):
    return max((c[k] for c in counts.values()), default=0)
examples = {}
for k in ("device lost", "panic", "download fail"):
    for l in log + (obslog or []):
        if pats[k].search(l):
            examples[k] = l.strip()[:200]; break

kern = read_lines(kern_p) or []
# the lines naming the observer's pid ("pid=<n>, name=dust_observer"); the driver's detail lines
# without a pid ("Graphics Exception: ...", logged before the attributed line) are reported only
def _own_line(l):
    m = re.search(r"pid=(\d+)", l)
    if m:
        return m.group(1) == obs_pid if obs_pid else "dust_observer" in l
    return False
kern_own = [l for l in kern if _own_line(l)]

# samples
rows = []
try:
    with open(csv_p, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
except OSError:
    pass
def num(r, k):
    try:
        v = r.get(k, "")
        return float(v) if v not in ("", None) else None
    except ValueError:
        return None

# with --no-sheet (soak.sh's default) the observer keeps no frame: RssAnon is the engine's own
no_sheet = "--no-sheet" in obs_args.split()
frame_bytes_kb = 0.0 if no_sheet else (size[0] * size[1] * 4 / 1024.0 if size else None)
S = []
for r in rows:
    s = {"t": num(r, "unix_time"), "el": num(r, "elapsed_s"), "anon": num(r, "rss_anon_kb"),
         "rss": num(r, "vmrss_kb"), "gp": num(r, "gpu_proc_mib"), "gt": num(r, "gpu_total_mib"),
         "fr": num(r, "frame"), "sim": parse_epoch(r.get("sim_time", "")), "thr": num(r, "threads"),
         "fds": num(r, "fds")}
    # dust_observer keeps every frame's RGBA in memory (frames_rgba, for the contact sheet):
    # (frame + 1) * W*H*4 bytes of anon RSS by design; subtracted for the leak judgement
    if s["anon"] is not None and frame_bytes_kb and s["fr"] is not None:
        s["anon_net"] = s["anon"] - (s["fr"] + 1) * frame_bytes_kb
    elif s["anon"] is not None:
        s["anon_net"] = s["anon"] if not frame_bytes_kb else None
    else:
        s["anon_net"] = None
    S.append(s)

have_sim = run_start is not None and any(s["sim"] is not None for s in S)
if have_sim:
    for s in S:
        s["simd"] = (s["sim"] - run_start) / 86400.0 if s["sim"] is not None else None
    post = [s for s in S if s["simd"] is not None and s["simd"] >= warm_days]
    warm_rule = f"simulated time >= start + {warm_days:g} d"
else:
    for s in S:
        s["simd"] = None
    post = S[len(S) // 6:]
    warm_rule = "after the first 1/6 of the samples (no simulated-time column)"

def slope(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pts) < 3:
        return None
    mx = sum(p[0] for p in pts) / len(pts); my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    if sxx <= 0:
        return None
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx

def growth(vals):
    v = [x for x in vals if x is not None]
    if len(v) < 2:
        return None
    k = max(1, min(5, len(v) // 4))
    a = statistics.median(v[:k]); b = statistics.median(v[-k:])
    return (b - a) / a if a else None

out = []
P = out.append
P(f"soak {label}: {bin_p} {obs_args}")
P(f"wall {end_s - start_s:.0f} s ({(end_s - start_s) / 3600:.2f} h), {len(S)} samples, {len(post)} after warm-up ({warm_rule})")
P(f"exit code: {code}" + (" (timeout)" if code == 124 else " (SIGKILL / OOM in the 6G scope?)" if code == 137 else "")
  + (f", interrupted by {interrupted}" if interrupted else ""))
P(f"frames completed: {frames_done}" + (f" of {frames_expected} expected" if frames_expected else " (expected: unknown, no run: line)"))
if no_sheet:
    P("--no-sheet: the observer keeps no frame in memory; 'net' RssAnon is the raw RssAnon")
elif size:
    P(f"frame size {size[0]}x{size[1]}: the observer retains {frame_bytes_kb / 1024:.2f} MB of RGBA per frame by design (frames_rgba); 'net' RssAnon subtracts it")
P("")
P("log lines matching (stdout | observer log.txt):")
for k in pats:
    P(f"  {k:16s} " + " | ".join(f"{counts[n][k]}" if n in counts else "-" for n in ("stdout", "log.txt")))
for k, l in examples.items():
    P(f"  first {k}: {l}")
P("")
if journal_ok != "1":
    P("kernel journal: UNREADABLE (journalctl -k failed)")
else:
    P(f"new kernel Xid/NVRM lines: {len(kern)} ({len(kern_own)} attributed to the observer pid {obs_pid or '?'}; "
      f"the others are other processes' — this driver also logs Xids for gnome-shell)")
    for l in kern[:40]:
        P(f"  \"{l}\"")
    if len(kern) > 40:
        P(f"  ... {len(kern) - 40} more in {os.path.basename(kern_p)}")
P("")

verdicts = []
def crit(name, ok, detail):
    tag = "SKIP" if ok is None else ("PASS" if ok else "FAIL")
    verdicts.append(tag); P(f"{tag}  {name}: {detail}")

def mem_block(key, label_, to_mb):
    xs_sim = [s["simd"] for s in post]; xs_wall = [s["el"] / 3600.0 if s["el"] is not None else None for s in post]
    ys = [s[key] * to_mb if s[key] is not None else None for s in post]
    if sum(1 for y in ys if y is not None) < 3:
        P(f"{label_}: no data after the warm-up")
        return None, None, None
    ssim = slope(xs_sim, ys) if have_sim else None
    swall = slope(xs_wall, ys)
    g = growth(ys)
    vals = [y for y in ys if y is not None]
    P(f"{label_}: {vals[0]:.1f} -> {vals[-1]:.1f} MB, slope "
      + (f"{ssim:+.3f} MB/sim-day, " if ssim is not None else "")
      + (f"{swall:+.3f} MB/wall-hour" if swall is not None else "n/a")
      + (f", growth {100 * g:+.2f} %" if g is not None else ""))
    return ssim, swall, g

KB = 1 / 1024.0
P("regressions after the warm-up:")
a_sim, a_wall, a_g = mem_block("anon", "  RssAnon (raw)", KB)
n_sim, n_wall, n_g = mem_block("anon_net", "  RssAnon net of retained frames", KB)
mem_block("rss", "  VmRSS", KB)
p_sim, p_wall, p_g = mem_block("gp", "  GPU process memory", 1.0)
t_sim, t_wall, t_g = mem_block("gt", "  GPU total memory", 1.0)
thr = [s["thr"] for s in S if s["thr"] is not None]; fds = [s["fds"] for s in S if s["fds"] is not None]
if thr:
    P(f"  threads {thr[0]:.0f} -> {thr[-1]:.0f} (max {max(thr):.0f}); fds {fds[0]:.0f} -> {fds[-1]:.0f} (max {max(fds):.0f})" if fds else f"  threads {thr[0]:.0f} -> {thr[-1]:.0f}")
P("")

# frame wall time: the observer's own per-frame seconds (frame 0 holds the start-up, excluded);
# fallback: the intervals between the samples where the frame index changed
ft = [frames[k][1] for k in sorted(frames) if k > 0]
src = "per-frame seconds from the frame lines"
if len(ft) < 5:
    ch = []; last = None
    for s in S:
        if s["fr"] is not None and s["fr"] != last:
            ch.append(s["t"]); last = s["fr"]
    ft = [b - a for a, b in zip(ch, ch[1:])]
    src = "intervals between frame-index changes in the samples (± one sample period)"
ft_ratio = None
if len(ft) >= 5:
    q = max(1, len(ft) // 5)
    f0 = statistics.mean(ft[:q]); f1 = statistics.mean(ft[-q:])
    ft_ratio = f1 / f0 if f0 > 0 else None
    P(f"frame wall time ({src}): first quintile {f0:.2f} s, last quintile {f1:.2f} s, ratio {ft_ratio:.3f} ({len(ft)} frames)")
else:
    P("frame wall time: too few frames")
P("")

def mem_crit(name, ssim, g, have):
    if not have:
        crit(name, None, "no data")
        return
    ok = (g is not None and g < 0.05) and (ssim is None or ssim < 1.0)
    gtxt = f"growth {100 * g:+.2f} % (< 5 %)" if g is not None else "growth n/a"
    stxt = f"slope {ssim:+.3f} MB/sim-day (< 1)" if ssim is not None else "slope n/a (no sim time)"
    crit(name, ok, f"{gtxt}, {stxt}")

mem_crit("host anon RSS (net of retained frames)", n_sim, n_g, n_g is not None or n_sim is not None)
mem_crit("GPU process memory", p_sim, p_g, p_g is not None or p_sim is not None)
mem_crit("GPU total memory", t_sim, t_g, t_g is not None or t_sim is not None)
if journal_ok != "1":
    crit("no new Xid/NVRM", None, "journal unreadable")
else:
    crit("no Xid/NVRM attributed to the observer", len(kern_own) == 0,
         f"{len(kern_own)} own line(s), {len(kern) - len(kern_own)} from other processes (reported, not judged)")
crit("no device lost / panic", worst("device lost") == 0 and worst("panic") == 0,
     f"device lost {worst('device lost')}, panic {worst('panic')}")
if frames_expected:
    crit("all frames rendered", frames_done >= frames_expected and worst("download fail") == 0,
         f"{frames_done}/{frames_expected}, download failures {worst('download fail')}")
else:
    crit("all frames rendered", None, f"{frames_done} frames, expected unknown")
crit("frame time last quintile <= 1.25 x first", None if ft_ratio is None else ft_ratio <= 1.25,
     f"ratio {ft_ratio:.3f}" if ft_ratio is not None else "too few frames")
crit("exit code 0", code == 0, f"{code}")
overall = "FAIL" if "FAIL" in verdicts else ("PASS" if "SKIP" not in verdicts else "PASS (with SKIPs)")
P("")
P(f"OVERALL: {overall}")
text = "\n".join(out) + "\n"
with open(out_p, "w") as f:
    f.write(text)
print(text, end="")
sys.exit(1 if overall == "FAIL" else 0)
PY
VERDICT=$?
echo "[soak.sh] summary: $SUMMARY"
# the observer's exit code when it failed, else 1 when a criterion failed
if [[ "${CODE:-1}" != "0" ]]; then exit "${CODE:-1}"; fi
exit "$VERDICT"
