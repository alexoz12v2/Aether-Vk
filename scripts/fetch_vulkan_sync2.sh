#!/bin/bash
set -e

DIR="$(pwd)/workflow-data/vulkan-synchronization2"
mkdir -p "$DIR/linux" "$DIR/windows" "$DIR/macos"

TMP_DIR=$(mktemp -d)
cd "$TMP_DIR"

echo "Downloading Linux SDK 1.3.296.0..."
curl -sL "https://sdk.lunarg.com/sdk/download/1.3.296.0/linux/vulkansdk-linux-x86_64-1.3.296.0.tar.xz" -o sdk_linux.tar.xz
tar -xJf sdk_linux.tar.xz 1.3.296.0/x86_64/share/vulkan/explicit_layer.d/VkLayer_khronos_synchronization2.json 1.3.296.0/x86_64/lib/libVkLayer_khronos_synchronization2.so
cp 1.3.296.0/x86_64/share/vulkan/explicit_layer.d/VkLayer_khronos_synchronization2.json "$DIR/linux/"
cp 1.3.296.0/x86_64/lib/libVkLayer_khronos_synchronization2.so "$DIR/linux/"

echo "Downloading Windows SDK 1.3.296.0..."
curl -sL "https://sdk.lunarg.com/sdk/download/1.3.296.0/windows/VulkanSDK-1.3.296.0-Installer.exe" -o sdk_windows.exe
7z e sdk_windows.exe Bin/VkLayer_khronos_synchronization2.dll Bin/VkLayer_khronos_synchronization2.json
cp VkLayer_khronos_synchronization2.json "$DIR/windows/"
cp VkLayer_khronos_synchronization2.dll "$DIR/windows/"

echo "Downloading macOS SDK 1.3.296.0..."
curl -sL "https://sdk.lunarg.com/sdk/download/1.3.296.0/mac/vulkansdk-macos-1.3.296.0.zip" -o sdk_macos.zip
unzip -q sdk_macos.zip
./InstallVulkan.app/Contents/MacOS/InstallVulkan --accept-licenses --default-answer --confirm-command install -t "$(pwd)/vulkan_mac"
cp vulkan_mac/macOS/share/vulkan/explicit_layer.d/VkLayer_khronos_synchronization2.json "$DIR/macos/"
cp vulkan_mac/macOS/lib/libVkLayer_khronos_synchronization2.dylib "$DIR/macos/"

cd ..
rm -rf "$TMP_DIR"
echo "Synchronization2 layer files for all platforms fetched and staged."
