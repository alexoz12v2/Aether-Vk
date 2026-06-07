#!/bin/bash
set -e

# Target directory
DIR="workflow-data/vulkan-synchronization2"

if [ ! -f "$DIR/VkLayer_khronos_synchronization2.json" ] || [ ! -f "$DIR/libVkLayer_khronos_synchronization2.so" ]; then
    echo "Downloading Vulkan SDK 1.4.321.0..."
    mkdir -p "$DIR"
    TMP_DIR=$(mktemp -d)
    
    # Download the linux SDK tarball directly from LunarG
    curl -sL "https://sdk.lunarg.com/sdk/download/1.4.321.0/linux/vulkansdk-linux-x86_64-1.4.321.0.tar.xz" -o "$TMP_DIR/sdk.tar.xz"
    
    echo "Extracting Synchronization2 layer..."
    # Extract only the specific files to save time
    tar -xJf "$TMP_DIR/sdk.tar.xz" -C "$TMP_DIR" \
        1.4.321.0/x86_64/share/vulkan/explicit_layer.d/VkLayer_khronos_synchronization2.json \
        1.4.321.0/x86_64/lib/libVkLayer_khronos_synchronization2.so
    
    # Move them into the repository's tracked workflow-data folder
    cp "$TMP_DIR/1.4.321.0/x86_64/share/vulkan/explicit_layer.d/VkLayer_khronos_synchronization2.json" "$DIR/"
    cp "$TMP_DIR/1.4.321.0/x86_64/lib/libVkLayer_khronos_synchronization2.so" "$DIR/"
    
    rm -rf "$TMP_DIR"
    git add "$DIR"
    echo "Synchronization2 layer files fetched and automatically staged for commit into $DIR."
else
    echo "Synchronization2 layer files already exist in $DIR. No action needed."
fi
