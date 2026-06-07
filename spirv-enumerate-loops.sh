#!/bin/bash

# Define Vulkan SDK path
# Note: You may need to adjust this path depending on your system configuration.
VULKAN_SDK="/Volumes/ExtData/alessioext/VulkanSDK/1.4.321.0/macOS"

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
