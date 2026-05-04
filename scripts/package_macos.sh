#!/bin/bash
set -e

# Build for arm64 and x64 or specified architecture
ARCH=${1:-osx-arm64} # default to osx-arm64
APP_NAME="AetherVk"
BUNDLE_DIR="bin/publish/$APP_NAME.app"
PUBLISH_DIR="aethervk.ui.app/bin/Release/net10.0/$ARCH/publish"

echo "=========================================="
echo " Packaging Aether-Vk for macOS ($ARCH)"
echo "=========================================="

echo "[1/4] Publishing project..."
dotnet publish aethervk.ui.app/AetherVk.csproj -c Release -r "$ARCH" --self-contained true

echo "[2/4] Creating App Bundle structure..."
mkdir -p "$BUNDLE_DIR/Contents/MacOS"
mkdir -p "$BUNDLE_DIR/Contents/Resources"

echo "[3/4] Copying published files to Contents/MacOS..."
cp -R "$PUBLISH_DIR/"* "$BUNDLE_DIR/Contents/MacOS/"

echo "[4/4] Generating Info.plist..."
cat > "$BUNDLE_DIR/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>$APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>com.example.$APP_NAME</string>
    <key>CFBundleName</key>
    <string>$APP_NAME</string>
    <key>CFBundleVersion</key>
    <string>1.0</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>
EOF

echo "=========================================="
echo " Packaging complete: $BUNDLE_DIR"
echo "=========================================="