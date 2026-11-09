#!/bin/bash
set -e

ARCH=${1:-linux-x64}
APP_NAME="AetherVk"
APP_VERSION="1.0.1"
PUBLISH_DIR="aethervk.ui-app/bin/Release/net10.0/$ARCH/publish"
TAR_FILE="bin/publish/${APP_NAME}_${APP_VERSION}_${ARCH}.tar.gz"

echo "=========================================="
echo " Packaging Aether-Vk for Linux ($ARCH)"
echo "=========================================="

case "$ARCH" in
    linux-x64) CARGO_TARGET="x86_64-unknown-linux-gnu" ;;
    linux-arm64) CARGO_TARGET="aarch64-unknown-linux-gnu" ;;
    *) echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

# Extra cargo flags for the native engine, empty by default. AETHERVK_CARGO_FLAGS=--no-default-features
# builds without CFITSIO (pure-Rust FITS reader) when vcpkg cannot build it for a target.
read -r -a EXTRA_CARGO_FLAGS <<< "${AETHERVK_CARGO_FLAGS:-}"

echo "[1/4] Building the native engine ($CARGO_TARGET) ${AETHERVK_CARGO_FLAGS:-}..."
cargo build --locked -p aethervk-core-cdylib --profile release --target "$CARGO_TARGET" "${EXTRA_CARGO_FLAGS[@]}"
if [ ! -f "target/$CARGO_TARGET/release/libaethervk_core_cdylib.so" ]; then
    echo "Native library missing: target/$CARGO_TARGET/release/libaethervk_core_cdylib.so"
    exit 1
fi

echo "[2/4] Publishing project..."
# the engine is built above (with AETHERVK_CARGO_FLAGS), so ui-logic only copies it: a cargo build of
# its own would use the default features. ui-logic (being netstandard) does not receive -r: pass the
# target RID explicitly
dotnet publish aethervk.ui-app/aethervk.ui-app.csproj -c Release -r "$ARCH" --self-contained true -p:SkipNativeBuild=true -p:ResolvedRID="$ARCH"

# licence texts: the project's, and those of the static CFITSIO + zlib linked into the engine
cp LICENSE "$PUBLISH_DIR/LICENSE"
cp aethervk.cfitsio/sys/THIRD_PARTY_LICENSES.md "$PUBLISH_DIR/THIRD_PARTY_LICENSES.md"

echo "[3/4] Creating tarball directory..."
mkdir -p bin/publish

echo "[4/4] Compressing files to $TAR_FILE..."
tar -czvf "$TAR_FILE" -C "$PUBLISH_DIR" .

echo "=========================================="
echo " Packaging complete: $TAR_FILE"
echo "=========================================="