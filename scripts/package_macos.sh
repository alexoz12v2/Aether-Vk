#!/bin/bash
set -e

# Build for arm64 and x64 or specified architecture
ARCH=${1:-osx-arm64} # default to osx-arm64
CONFIG=${2:-Release}
MODE=${3:-Normal} # Normal or SideBySide

APP_VERSION="1.0"
BASE_APP_NAME="AetherVk"

if [ "$MODE" = "SideBySide" ]; then
    APP_NAME="${BASE_APP_NAME}-${CONFIG}"
    BUNDLE_ID="com.example.${BASE_APP_NAME}.sxs.${APP_VERSION//./-}.${CONFIG,,}"
    DISPLAY_NAME="${BASE_APP_NAME} (SxS $APP_VERSION $CONFIG)"
else
    APP_NAME="$BASE_APP_NAME"
    BUNDLE_ID="com.example.$BASE_APP_NAME"
    DISPLAY_NAME="$BASE_APP_NAME"
fi

BUNDLE_DIR="bin/publish/$APP_NAME.app"
PUBLISH_DIR="aethervk.ui.app/bin/$CONFIG/net10.0/$ARCH/publish"

echo "=========================================="
echo " Packaging Aether-Vk for macOS ($ARCH) - $MODE ($CONFIG)"
echo "=========================================="

echo "[1/4] Publishing project..."
dotnet publish aethervk.ui.app/AetherVk.csproj -c "$CONFIG" -r "$ARCH" --self-contained true

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
    <string>$BASE_APP_NAME</string>
    <key>CFBundleIdentifier</key>
    <string>$BUNDLE_ID</string>
    <key>CFBundleName</key>
    <string>$DISPLAY_NAME</string>
    <key>CFBundleVersion</key>
    <string>$APP_VERSION</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>$APP_VERSION</string>
    <key>LSMinimumSystemVersion</key>
    <string>10.15</string>
</dict>
</plist>
EOF

echo "=========================================="
echo " Packaging complete: $BUNDLE_DIR"
echo "=========================================="