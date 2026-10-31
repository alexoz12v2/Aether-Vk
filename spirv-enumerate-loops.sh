#!/bin/bash

# Define Vulkan SDK path
# Defaults to the pinned SDK (.vulkan-sdk-version) at its default macOS location.
VULKAN_SDK="${VULKAN_SDK:-$HOME/VulkanSDK/$(tr -d '[:space:]' < "$(dirname "$0")/.vulkan-sdk-version")/macOS}"

if [ ! -f "$VULKAN_SDK/bin/spirv-dis" ]; then
    echo "Error: spirv-dis not found at $VULKAN_SDK/bin/spirv-dis"
    echo "Please ensure the Vulkan SDK is installed and the path is correct."
    exit 1
fi

echo "Counting OpLoopMerge instructions in compiled SPIR-V shaders..."

for f in assets/sim/build_motion_bvh.comp.spv assets/sim/bp_*.comp.spv assets/sim/lbvh_build.comp.spv assets/sim/lbvh_prepass.comp.spv assets/sim/lbvh_collapse.comp.spv; do
  if [ -f "$f" ]; then
    cnt=$("$VULKAN_SDK/bin/spirv-dis" "$f" 2>/dev/null | grep -c "OpLoopMerge")
    echo "$cnt loops: $(basename $f)"
  else
    echo "Warning: File not found: $f"
  fi
done
