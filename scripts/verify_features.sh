#!/usr/bin/env bash
# Automated verification of the Earth / desktop billboards / savefile features (plan:
# ~/.claude/plans/while-particles-session-is-immutable-quokka.md, "Testing strategy").
#
#   scripts/verify_features.sh [earth|billboards|savefile|all]   (default: all)
#
# Runs every step one at a time: GPU steps in a memory-capped scope with a timeout, and only while
# no dust observer / GPU dust run is going (`pgrep -f dust_observer`). Prints a PASS/FAIL table and
# exits 1 if any step failed. Logs: $VERIFY_OUT (default: a fresh temp directory).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHAT="${1:-all}"
OUT="${VERIFY_OUT:-$(mktemp -d -t aethervk_verify.XXXXXX)}"
mkdir -p "$OUT"
CAP=(systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0)
# as execute_tests.sh: GPU-assisted validation can deadlock the Khronos layer at context teardown
# (idle futex threads, the 8 s shutdown watchdog fires); core validation stays on
export AETHERVK_DISABLE_GPU_AV="${AETHERVK_DISABLE_GPU_AV:-1}"
RESULTS=()
FAILED=0

wait_for_gpu() {
  while pgrep -f dust_observer >/dev/null 2>&1; do
    echo "  (waiting: a dust_observer / GPU dust run is in progress)"
    sleep 30
  done
}

# step NAME TIMEOUT_S [gpu] -- COMMAND...
step() {
  local name="$1" timeout_s="$2" gpu="$3"
  shift 4
  local log="$OUT/$(echo "$name" | tr ' /:' '___').log"
  echo "=== $name"
  [[ "$gpu" == "gpu" ]] && wait_for_gpu
  if (cd "$ROOT" && timeout "$timeout_s" "${CAP[@]}" "$@") >"$log" 2>&1; then
    # a skipped test is not a verified one: name it next to the PASS
    local skipped
    skipped="$(grep -oE '^\s+Skipped [A-Za-z0-9_.]+' "$log" | awk '{print $2}' | sed 's/.*\.//' | sort -u | paste -sd, -)"
    if [[ -n "$skipped" ]]; then
      RESULTS+=("PASS  $name   (SKIPPED: $skipped — see log)")
    else
      RESULTS+=("PASS  $name")
    fi
  else
    RESULTS+=("FAIL  $name   (log: $log)")
    FAILED=1
  fi
}

earth() {
  step "earth: CPU tests (earth_visual)" 1800 cpu -- \
    cargo test --features std -p aethervk-core-rlib --lib earth_visual -- --test-threads=1
  step "earth: GPU render tests (test_earth_render)" 2400 gpu -- \
    cargo test --features std -p aethervk-core-rlib --lib test_earth_render -- --test-threads=1
  step "earth: render regressions (composite, occlusion, multi-micro, text/ui/gizmo uploads)" 3000 gpu -- \
    cargo test --features std -p aethervk-core-rlib --lib -- --test-threads=1 \
      test_composite_render test_trajectory_occlusion test_lca_render test_multi_micro \
      device::test_render ui_tests scene_conversion indicator text
  step "earth: logic regressions (scene dump, appearance, logic thread)" 3000 gpu -- \
    cargo test --features std -p aethervk-core-rlib --lib -- --test-threads=1 \
      scene_dump comet_appearance logic_thread_tests simulation_api::tests asset_library texture
  step "earth: dust GPU suite (layer clears changed)" 3000 gpu -- \
    cargo test --release --features std -p aethervk-core-rlib --lib gpu_dust -- --test-threads=1
  step "earth: dust draw/frame-time/micro-colour tests" 1200 gpu -- \
    cargo test --release --features std -p aethervk-core-rlib --lib -- --test-threads=1 \
      dust_draw_ignores dust_frame_time micro_color_target
  step "earth: scene_observer --check (real ephemeris, 4 epochs)" 1500 gpu -- \
    bash -c "cargo build -p aethervk-core-rlib --example scene_observer --features std --release && \
      ./target/release/examples/scene_observer --out '$OUT/scene_observer' --check --strict-orientation --quiet"
  step "earth: .NET camera render masks (CameraServiceTests)" 1200 cpu -- \
    dotnet test aethervk.logic.tests --filter "FullyQualifiedName~CameraServiceTests"
}

billboards() {
  step "billboards: Rust texel export (asset library)" 1200 cpu -- \
    cargo test --features std -p aethervk-core-rlib --lib rgba8_export -- --test-threads=1
  step "billboards: FFI copyAssetTexels / DTO flag (cdylib, real context)" 1500 gpu -- \
    cargo test --features std -p aethervk-core-cdylib --lib ffi_assets -- --test-threads=1
  step "billboards: .NET logic (geometry, pointer rules, service, activation, view models)" 1200 cpu -- \
    dotnet test aethervk.logic.tests --filter "FullyQualifiedName~DesktopBillboardTests"
  step "billboards: .NET headless UI (Settings section, window pointer rules and alpha, DI)" 1200 cpu -- \
    dotnet test aethervk.app.tests --filter "FullyQualifiedName~DesktopBillboard"
  step "billboards: real X11 end to end (Xvfb + xcompmgr, xdotool)" 1200 cpu -- \
    "$ROOT/scripts/x11_e2e.sh"
}

savefile() {
  step "savefile: Rust asset export, compatibility JSON, camera dump round trip" 1500 cpu -- \
    cargo test --features std -p aethervk-core-rlib --lib -- --test-threads=1 \
      export_copies_standalone export_extracts_images scene_dump_tests
  step "savefile: Rust dump/restore on the logic thread (real context)" 1500 gpu -- \
    cargo test --features std -p aethervk-core-rlib --lib -- --test-threads=1 \
      test_dump_restore_returns test_seek_epoch
  step "savefile: FFI dump to file / restore from file / version / content hash (cdylib, real context)" 1500 gpu -- \
    cargo test --features std -p aethervk-core-cdylib --lib -- --test-threads=1 \
      dump_to_file_then_restore_from_file ffi_assets
  step "savefile: .NET logic (index, mapping, service, host, File menu, preferences, bindings, tabs)" 1200 cpu -- \
    dotnet test aethervk.logic.tests --filter \
      "FullyQualifiedName~SaveIndex|FullyQualifiedName~SaveFile|FullyQualifiedName~AppSaveFileHost|FullyQualifiedName~FileMenuTests|FullyQualifiedName~Preferences|FullyQualifiedName~InputBindings|FullyQualifiedName~InputRegistry|FullyQualifiedName~SessionRestoredTab|FullyQualifiedName~LocalStorage|FullyQualifiedName~JetViewModelRestore|FullyQualifiedName~CameraServiceTests|FullyQualifiedName~ModelTabViewModelTests"
  # the app tests load the native library from their bin: rebuild it first (the .NET build copies
  # it when newer), so the end-to-end test runs today's engine
  step "savefile: native library for the .NET tests" 2400 cpu -- \
    bash -c "cd aethervk.core/cdylib && cargo build -p aethervk-core-cdylib --target x86_64-unknown-linux-gnu && \
      cd ../.. && dotnet build aethervk.app.tests"
  step "savefile: headless UI (File menu, dirty title, close prompt, dialog, bindings editor)" 1200 gpu -- \
    dotnet test aethervk.app.tests --no-build --filter "FullyQualifiedName~FileMenuView|FullyQualifiedName~InputBindingsView"
  step "savefile: real-native end to end (commit 67P fixture, save, New, open, second state, expiry)" 1200 gpu -- \
    dotnet test aethervk.app.tests --no-build --filter "FullyQualifiedName~SaveFileNativeE2ETests"
  step "savefile: whole app test suite (shared host, real engine)" 1800 gpu -- \
    dotnet test aethervk.app.tests --no-build
}

case "$WHAT" in
  earth) earth ;;
  billboards) billboards ;;
  savefile) savefile ;;
  all) earth; billboards; savefile ;;
  *) echo "usage: $0 [earth|billboards|savefile|all]" >&2; exit 2 ;;
esac

echo
echo "──────── verify_features: $WHAT ────────"
printf '%s\n' "${RESULTS[@]}"
echo "logs: $OUT"
exit $FAILED
