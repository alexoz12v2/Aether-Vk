#!/bin/bash
set -e

# Build for arm64 and x64 or specified architecture
ARCH=${1:-osx-arm64} # default to osx-arm64
CONFIG=${2:-Release}
MODE=${3:-Normal} # Normal or SideBySide

APP_VERSION="1.0.1"
BASE_APP_NAME="AetherVk"

if [ "$MODE" = "SideBySide" ]; then
    APP_NAME="${BASE_APP_NAME}-SxS-${APP_VERSION}-${CONFIG}"
    BUNDLE_ID="com.example.${BASE_APP_NAME}.sxs.${APP_VERSION//./-}.$(echo "$CONFIG" | tr '[:upper:]' '[:lower:]')"
    DISPLAY_NAME="${BASE_APP_NAME} (SxS $APP_VERSION $CONFIG)"
else
    APP_NAME="${BASE_APP_NAME}-${APP_VERSION}"
    BUNDLE_ID="com.example.$BASE_APP_NAME"
    DISPLAY_NAME="$BASE_APP_NAME"
fi

BUNDLE_DIR="bin/publish/$APP_NAME.app"
PUBLISH_DIR="aethervk.ui-app/bin/$CONFIG/net10.0/$ARCH/publish"

echo "=========================================="
echo " Packaging Aether-Vk for macOS ($ARCH) - $MODE ($CONFIG)"
echo "=========================================="

case "$ARCH" in
    osx-arm64) CARGO_TARGET="aarch64-apple-darwin" ;;
    osx-x64) CARGO_TARGET="x86_64-apple-darwin" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac
if [ "$CONFIG" = "Release" ]; then
    CARGO_PROFILE_ARGS=(--profile release)
    CARGO_PROFILE_DIR="release"
else
    CARGO_PROFILE_ARGS=()
    CARGO_PROFILE_DIR="debug"
fi

# Extra cargo flags for the native engine, empty by default. AETHERVK_CARGO_FLAGS=--no-default-features
# builds without CFITSIO (pure-Rust FITS reader) when vcpkg cannot build it for a target.
read -r -a EXTRA_CARGO_FLAGS <<< "${AETHERVK_CARGO_FLAGS:-}"

echo "[1/5] Building the native engine ($CARGO_TARGET) ${AETHERVK_CARGO_FLAGS:-}..."
cargo build --locked -p aethervk-core-cdylib ${CARGO_PROFILE_ARGS[@]+"${CARGO_PROFILE_ARGS[@]}"} --target "$CARGO_TARGET" ${EXTRA_CARGO_FLAGS[@]+"${EXTRA_CARGO_FLAGS[@]}"}
if [ ! -f "target/$CARGO_TARGET/$CARGO_PROFILE_DIR/libaethervk_core_cdylib.dylib" ]; then
    echo "Native library missing: target/$CARGO_TARGET/$CARGO_PROFILE_DIR/libaethervk_core_cdylib.dylib"
    exit 1
fi

echo "[2/5] Publishing project..."
# the engine is built above (with AETHERVK_CARGO_FLAGS), so ui-logic only copies it: a cargo build of
# its own would use the default features. ui-logic (being netstandard) does not receive -r: pass the
# target RID explicitly
dotnet publish aethervk.ui-app/aethervk.ui-app.csproj -c "$CONFIG" -r "$ARCH" --self-contained true -p:SkipNativeBuild=true -p:ResolvedRID="$ARCH"

echo "[3/5] Creating App Bundle structure..."
mkdir -p "$BUNDLE_DIR/Contents/MacOS"
mkdir -p "$BUNDLE_DIR/Contents/Resources"

echo "[4/5] Copying published files to Contents/MacOS..."
cp -R "$PUBLISH_DIR/"* "$BUNDLE_DIR/Contents/MacOS/"

# licence texts: the project's, and those of the static CFITSIO + zlib linked into the engine
cp LICENSE "$BUNDLE_DIR/Contents/Resources/LICENSE"
cp aethervk.cfitsio/sys/THIRD_PARTY_LICENSES.md "$BUNDLE_DIR/Contents/Resources/THIRD_PARTY_LICENSES.md"

echo "[5/5] Generating Info.plist..."
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