#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AetherVk - VM Offload Environment Activator
# 
# Usage: source scripts/macos_vm_env.sh [user] [host]
# Example: source scripts/macos_vm_env.sh alessio aether-vm.local
# ─────────────────────────────────────────────────────────────────────────────

# Default to "alessio" and "aether-vm.local" if not provided
export AETHER_VM_USER="${1:-alessio}"
export AETHER_VM_HOST="${2:-aether-vm.local}"
export CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER="scripts/vm-runner.sh"

echo "==== VM Offload Environment Configured ===="
echo "VM User: $AETHER_VM_USER"
echo "VM Host: $AETHER_VM_HOST"
echo "Runner:  $CARGO_TARGET_AARCH64_APPLE_DARWIN_RUNNER"
echo "You can now run 'cargo nextest run' or 'cargo run' and it will natively offload to the VM."
