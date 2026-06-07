#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AetherVk - VM Offload Runner
# 
# This script is invoked automatically by Cargo when running `cargo run`, 
# `cargo test`, or `cargo nextest run`.
# It executes the built binaries inside the UTM macOS Virtual Machine via SSH.
#
# Ensure you have followed the setup instructions in `docs/vm_offload_guide.md`.
# ─────────────────────────────────────────────────────────────────────────────

# Configuration (override via env vars if needed)
VM_HOST=${AETHER_VM_HOST:-"aether-vm.local"}
VM_USER=${AETHER_VM_USER:-"$USER"}

# Cargo passes the absolute path to the compiled binary as the first argument.

BIN_PATH="$1"
shift

VM_ENV=""
if [ -n "$AETHERVK_DISABLE_SYNC_VAL" ]; then
    VM_ENV+="export AETHERVK_DISABLE_SYNC_VAL=$AETHERVK_DISABLE_SYNC_VAL; "
fi

# We copy the binary to the VM's /tmp/ directory first because virtiofs is notoriously slow
# with the random 4K reads performed by dyld. A sequential rsync/cp is much faster and
# prevents nextest's concurrent test discovery from deadlocking the virtiofs bridge.
BIN_NAME=$(basename "$BIN_PATH")
TMP_BIN_PATH="/tmp/$BIN_NAME"
LOCK_DIR="/tmp/${BIN_NAME}.lock"

ssh -o BatchMode=yes -q "$VM_USER@$VM_HOST" "
    while ! mkdir '$LOCK_DIR' 2>/dev/null; do
        # If the binary is already newer than the source or same size/time, we don't need to wait for the lock forever
        if [ '$TMP_BIN_PATH' -nt '$BIN_PATH' ]; then
            break
        fi
        sleep 0.1
    done
    if [ '$BIN_PATH' -nt '$TMP_BIN_PATH' ] || [ ! -f '$TMP_BIN_PATH' ]; then
        cp '$BIN_PATH' '$TMP_BIN_PATH'
    fi
    rmdir '$LOCK_DIR' 2>/dev/null || true
    cd '$PWD' && $VM_ENV '$TMP_BIN_PATH' $@
"

# Capture the exit code of the SSH command (which reflects the binary's exit code)
EXIT_CODE=$?

# If SSH failed (e.g. 255), print a helpful error message to stderr.
if [ $EXIT_CODE -eq 255 ]; then
    echo "[VM Runner Error] Failed to SSH into $VM_USER@$VM_HOST." >&2
    echo "Please ensure the VM is running and passwordless SSH is configured." >&2
fi

exit $EXIT_CODE
