#!/bin/bash
set -e

ARCH=${1:-linux-x64}
APP_NAME="AetherVk"
APP_VERSION="1.0.0"
PUBLISH_DIR="aethervk.ui-app/bin/Release/net10.0/$ARCH/publish"
TAR_FILE="bin/publish/${APP_NAME}_${APP_VERSION}_${ARCH}.tar.gz"

echo "=========================================="
echo " Packaging Aether-Vk for Linux ($ARCH)"
echo "=========================================="

echo "[1/3] Publishing project..."
dotnet publish aethervk.ui-app/aethervk.ui-app.csproj -c Release -r "$ARCH" --self-contained true

echo "[2/3] Creating tarball directory..."
mkdir -p bin/publish

echo "[3/3] Compressing files to $TAR_FILE..."
tar -czvf "$TAR_FILE" -C "$PUBLISH_DIR" .

echo "=========================================="
echo " Packaging complete: $TAR_FILE"
echo "=========================================="