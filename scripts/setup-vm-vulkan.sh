#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# AetherVk - VM Offload Setup Script
# 
# Run this script INSIDE the UTM macOS Virtual Machine to download and 
# configure the Vulkan SDK.
# ─────────────────────────────────────────────────────────────────────────────

set -e

SDK_VERSION="1.4.321.0"
SDK_URL="https://sdk.lunarg.com/sdk/download/${SDK_VERSION}/mac/vulkansdk-macos-${SDK_VERSION}.zip"
DOWNLOAD_DIR="$HOME/Downloads"
ZIP_PATH="$DOWNLOAD_DIR/vulkan_sdk_${SDK_VERSION}.zip"

echo "==== Downloading Vulkan SDK $SDK_VERSION ===="
mkdir -p "$DOWNLOAD_DIR"
curl -# -L -o "$ZIP_PATH" "$SDK_URL"

echo "==== Extracting Vulkan SDK Installer ===="
# Extract to a temporary directory so we don't clutter the home or downloads folder
TEMP_EXTRACT=$(mktemp -d)
# We use tar -xf because macOS tar (bsdtar) natively supports zips
# and perfectly preserves symlinks, which unzip sometimes breaks!
tar -xf "$ZIP_PATH" -C "$TEMP_EXTRACT"

echo "==== Installing to ~/vulkan_sdk/macOS ===="
# Remove any existing installation
rm -rf ~/vulkan_sdk
mkdir -p ~/vulkan_sdk

# The zip actually contains a macOS .app installer, not the raw files!
# We must execute the Qt Installer Framework headlessly to extract the actual SDK into the target directory.
INSTALLER_APP="$TEMP_EXTRACT/vulkansdk-macOS-${SDK_VERSION}.app/Contents/MacOS/vulkansdk-macOS-${SDK_VERSION}"

# Run the installer headlessly
"$INSTALLER_APP" --root ~/vulkan_sdk --accept-licenses --default-answer --confirm-command install

# The installer extracts a nested 'macOS' folder. We want the contents directly in ~/vulkan_sdk/macOS
if [ -d "$HOME/vulkan_sdk/macOS/macOS" ]; then
    # Move the contents of the nested macOS folder up one level
    mv "$HOME/vulkan_sdk/macOS/macOS"/* "$HOME/vulkan_sdk/macOS/"
    rmdir "$HOME/vulkan_sdk/macOS/macOS"
fi

echo "==== Unquarantining SDK Files ===="
# Remove the com.apple.quarantine attribute recursively from all SDK files.
# If we don't do this, macOS Gatekeeper will block `dlopen` over SSH and wait for a UI popup,
# causing the test runner to hang infinitely!
xattr -cr ~/vulkan_sdk

echo "==== Cleaning up ===="
rm -rf "$TEMP_EXTRACT"
rm "$ZIP_PATH"

echo "==== Generating setup-env.sh ===="
ENV_SCRIPT="$HOME/vulkan_sdk/setup-env.sh"
cat << 'EOF' > "$ENV_SCRIPT"
#!/bin/bash
export VULKAN_SDK="$HOME/vulkan_sdk/macOS"
export VK_DRIVER_FILES="$VULKAN_SDK/share/vulkan/icd.d/MoltenVK_icd.json"
export VK_LAYER_PATH="$VULKAN_SDK/share/vulkan/explicit_layer.d"

if [ -z "$LD_LIBRARY_PATH" ]; then
    export LD_LIBRARY_PATH="$VULKAN_SDK/lib"
else
    export LD_LIBRARY_PATH="$VULKAN_SDK/lib:$LD_LIBRARY_PATH"
fi

if [ -z "$DYLD_LIBRARY_PATH" ]; then
    export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib"
else
    export DYLD_LIBRARY_PATH="$VULKAN_SDK/lib:$DYLD_LIBRARY_PATH"
fi

export PATH="$VULKAN_SDK/bin:$PATH"
EOF
chmod +x "$ENV_SCRIPT"

echo "Vulkan SDK $SDK_VERSION installed successfully!"
echo ""
echo "==== VM Configuration Complete ===="
echo "Execute the setup script from vulkan sdk to load the environment manually:"
echo "    source ~/vulkan_sdk/setup-env.sh"
echo ""
echo "Note: The vm-runner.sh and vm-dotnet-runner.sh scripts will automatically export these variables when executing remote tests."
