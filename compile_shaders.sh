#!/usr/bin/env bash
set -e

if [ -z "$VULKAN_SDK" ]; then
    echo "VULKAN_SDK environment variable is not set."
    exit 1
fi

GLSLC="$VULKAN_SDK/bin/glslc"
if [ ! -x "$GLSLC" ]; then
    echo "glslc not found or not executable at $GLSLC"
    exit 1
fi

for file in assets/*.vert assets/*.frag assets/*.comp assets/sim/*.comp; do
    [ -e "$file" ] || continue
    ext="${file##*.}"
    echo "Compiling $file..."
    "$GLSLC" -x glsl --target-env=vulkan1.1 --target-spv=spv1.4 -std=450core -fshader-stage="$ext" -o "$file.spv" "$file"
done

echo "All shaders compiled successfully."