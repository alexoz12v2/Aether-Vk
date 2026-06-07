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

# We use -q (quiet) to suppress SSH banner logs from cluttering test output.
# We remove BatchMode=yes so that if a password is required, it can prompt.
# We wrap the binary path in quotes in case of spaces in the directory structure.
# CRITICAL: We CD into the exact same directory as the host ($PWD) so relative paths work!
ssh -q "$VM_USER@$VM_HOST" "cd '$PWD' && $VM_ENV '$BIN_PATH' $@"

# Capture the exit code of the SSH command (which reflects the binary's exit code)
EXIT_CODE=$?

# If SSH failed (e.g. 255), print a helpful error message to stderr.
if [ $EXIT_CODE -eq 255 ]; then
    echo "[VM Runner Error] Failed to SSH into $VM_USER@$VM_HOST." >&2
    echo "Please ensure the VM is running and passwordless SSH is configured." >&2
fi

exit $EXIT_CODE
