#!/bin/bash
set -e

ARCH=${1:-linux-x64}
DEB_ARCH="amd64"
if [ "$ARCH" == "linux-arm64" ]; then
    DEB_ARCH="arm64"
fi

APP_NAME="AetherVk"
APP_VERSION="1.0.0"
PUBLISH_DIR="aethervk.ui.app/bin/Release/net10.0/$ARCH/publish"
BUILD_DIR="bin/publish/${APP_NAME}_${APP_VERSION}_${DEB_ARCH}"

echo "=========================================="
echo " Packaging Aether-Vk DEB ($DEB_ARCH)"
echo "=========================================="

if [ ! -d "$PUBLISH_DIR" ]; then
    echo "Publish directory $PUBLISH_DIR does not exist. Run package_linux.sh first, or run dotnet publish."
    exit 1
fi

mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/opt/$APP_NAME"
mkdir -p "$BUILD_DIR/usr/bin"
mkdir -p "$BUILD_DIR/usr/share/applications"

# Copy published files
cp -R "$PUBLISH_DIR/"* "$BUILD_DIR/opt/$APP_NAME/"

# Create a symlink in /usr/bin for the executable
ln -s "/opt/$APP_NAME/$APP_NAME" "$BUILD_DIR/usr/bin/aethervk"

# Generate DEBIAN/control file
cat > "$BUILD_DIR/DEBIAN/control" <<EOF
Package: aethervk
Version: $APP_VERSION
Section: utils
Priority: optional
Architecture: $DEB_ARCH
Maintainer: Aether-Vk Team
Description: Aether-Vk application
 A visually appealing, functional prototype containing native Rust binaries
 and rendering backends.
EOF

# Generate desktop entry
cat > "$BUILD_DIR/usr/share/applications/aethervk.desktop" <<EOF
[Desktop Entry]
Name=Aether-Vk
Exec=/opt/$APP_NAME/$APP_NAME
Type=Application
Terminal=false
Categories=Utility;Graphics;
EOF

echo "[4/4] Building .deb package..."
dpkg-deb --build "$BUILD_DIR"

echo "=========================================="
echo " Packaging complete: $BUILD_DIR.deb"
echo "=========================================="