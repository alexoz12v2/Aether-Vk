#!/usr/bin/env bash
# Real-X11 end-to-end tests of the desktop billboards (aethervk.x11.tests, plan B-10 … B-15).
#
#   scripts/x11_e2e.sh [extra dotnet test args...]
#
# Starts a private Xvfb (3840×1200, two 1920-wide RandR monitors via `xrandr --setmonitor`) with a
# compositing manager (xcompmgr, for the alpha test), runs the tests on it with AETHERVK_X11_E2E=1,
# then stops both. Without Xvfb it falls back to Xephyr on the current display. Exit code = dotnet
# test's.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DISP="${X11_E2E_DISPLAY:-:97}"
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT

if command -v Xvfb >/dev/null; then
  # -noreset: without it the server resets whenever its last client disconnects, which throws
  # away the RandR monitors set up below
  Xvfb "$DISP" -screen 0 3840x1200x24 -noreset +extension RANDR +extension Composite -nolisten tcp >/dev/null 2>&1 &
  PIDS+=($!)
elif command -v Xephyr >/dev/null && [[ -n "${DISPLAY:-}" ]]; then
  Xephyr "$DISP" -screen 3840x1200 +extension RANDR +extension Composite >/dev/null 2>&1 &
  PIDS+=($!)
else
  echo "x11_e2e.sh: neither Xvfb nor Xephyr (with a display) is available" >&2
  exit 3
fi

for _ in $(seq 50); do DISPLAY="$DISP" xdpyinfo >/dev/null 2>&1 && break; DISPLAY="$DISP" xwininfo -root >/dev/null 2>&1 && break; sleep 0.1; done

# two side-by-side monitors on the one screen (RandR 1.5 monitors, what Avalonia's Screens reads)
# (the first one takes over the output "screen", which removes its automatic full-width monitor;
# '*' marks it primary). Avalonia's X11 Screens read exactly these (XRRGetMonitors, RandR ≥ 1.5).
DISPLAY="$DISP" xrandr --setmonitor '*X11E2E-LEFT' 1920/508x1200/317+0+0 screen >/dev/null 2>&1 || true
DISPLAY="$DISP" xrandr --setmonitor X11E2E-RIGHT 1920/508x1200/317+1920+0 none >/dev/null 2>&1 || true
DISPLAY="$DISP" xrandr --listmonitors >&2

if command -v xcompmgr >/dev/null; then
  DISPLAY="$DISP" xcompmgr >/dev/null 2>&1 &
  PIDS+=($!)
else
  echo "x11_e2e.sh: xcompmgr not installed: the composited-alpha test will be skipped" >&2
fi
sleep 0.5

cd "$ROOT"
# Xvfb is plain X11: drop the desktop session's Wayland markers (they select the XWayland path;
# the tests force that path themselves with AETHERVK_FORCE_XWAYLAND_PATH=1 where they want it)
MONITORS="$(DISPLAY="$DISP" xrandr --listmonitors 2>/dev/null | sed -n 's/^Monitors: //p')"
env -u WAYLAND_DISPLAY XDG_SESSION_TYPE=x11 DISPLAY="$DISP" AETHERVK_X11_E2E=1 X11_E2E_MONITORS="$MONITORS" \
  dotnet test aethervk.x11.tests --logger "console;verbosity=normal" "$@"
