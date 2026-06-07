#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AetherVk - VM Offload Runner for .NET UI App
# 
# Builds the Rust cdylib and the .NET app on the host, then executes the UI app 
# in the UTM macOS Virtual Machine via SSH.
#
# Usage: ./scripts/vm-dotnet-runner.sh [dotnet run arguments...]
# Example: ./scripts/vm-dotnet-runner.sh -c Release
# ─────────────────────────────────────────────────────────────────────────────

VM_HOST=${AETHER_VM_HOST:-"aether-vm.local"}
VM_USER=${AETHER_VM_USER:-"$USER"}
PROJECT_DIR="$(pwd)"

# Default to Debug
CARGO_PROFILE_FLAG=""
TARGET_DIR="debug"
ORIG_ARGS=("$@")

# Parse arguments to determine if we are building Release
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -c|--configuration)
            if [ "${2,,}" == "release" ]; then
                CARGO_PROFILE_FLAG="--release"
                TARGET_DIR="release"
            fi
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "==== [HOST] Building Rust core cdylib ($TARGET_DIR) ===="
cargo build -p aethervk-core-cdylib $CARGO_PROFILE_FLAG || { echo "Cargo build failed"; exit 1; }

echo "==== [HOST] Building .NET App ($TARGET_DIR) ===="
dotnet build aethervk.ui-app "${ORIG_ARGS[@]}" || { echo "Dotnet build failed"; exit 1; }

VM_ENV="export DYLD_LIBRARY_PATH=\$DYLD_LIBRARY_PATH:$PROJECT_DIR/target/$TARGET_DIR;"
if [ -n "$AETHERVK_DISABLE_SYNC_VAL" ]; then
    VM_ENV+="export AETHERVK_DISABLE_SYNC_VAL=$AETHERVK_DISABLE_SYNC_VAL; "
fi

# We use -q (quiet) to suppress SSH banner logs.
# By running this as the logged-in user on the VM, the Avalonia UI window will appear on the VM's desktop!
ssh -q "$VM_USER@$VM_HOST" "cd '$PROJECT_DIR' && $VM_ENV dotnet run --no-build --project aethervk.ui-app ${ORIG_ARGS[@]}"
