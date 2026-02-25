#!/usr/bin/env sh

set -e

APP_BUNDLE="$1"
MVK_DYLIB="$2"
VULKAN_DYLIB="$3"

FRAMEWORKS_DIR="$APP_BUNDLE/Contents/Frameworks"
mkdir -p "$FRAMEWORKS_DIR"

make_framework() {
    DYLIB="$1"
    NAME="$2"
    OUT="$3"

    mkdir -p "$OUT/$NAME.framework/Versions/A/Resources"
    cp "$DYLIB" "$OUT/$NAME.framework/Versions/A/$NAME"

    ln -sf "A" "$OUT/$NAME.framework/Versions/Current"
    ln -sf "Versions/Current/$NAME" "$OUT/$NAME.framework/$NAME"
    ln -sf "Versions/Current/Resources" "$OUT/$NAME.framework/Resources"

    # Fix install_name
    install_name_tool -id "@rpath/$NAME.framework/$NAME" "$OUT/$NAME.framework/Versions/A/$NAME"

    # Write Info.plist
    cat > "$OUT/$NAME.framework/Versions/A/Resources/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
	<key>BuildMachineOSBuild</key>
	<string>23H626</string>
	<key>CFBundleDevelopmentRegion</key>
	<string>en</string>
	<key>CFBundleExecutable</key>
	<string>$NAME</string>
	<key>CFBundleIdentifier</key>
	<string>com.$NAME.framework</string>
	<key>CFBundleInfoDictionaryVersion</key>
	<string>6.0</string>
	<key>CFBundleName</key>
	<string>$NAME</string>
	<key>CFBundlePackageType</key>
	<string>FMWK</string>
	<key>CFBundleShortVersionString</key>
	<string>1.4.0</string>
	<key>CFBundleSupportedPlatforms</key>
	<array>
		<string>MacOSX</string>
	</array>
	<key>CFBundleVersion</key>
	<string>1.4.0</string>
	<key>DTCompiler</key>
	<string>com.apple.compilers.llvm.clang.1_0</string>
	<key>DTPlatformBuild</key>
	<string></string>
	<key>DTPlatformName</key>
	<string>macosx</string>
	<key>DTPlatformVersion</key>
	<string>14.5</string>
	<key>DTSDKBuild</key>
	<string>23F73</string>
	<key>DTSDKName</key>
	<string>macosx14.5</string>
	<key>DTXcode</key>
	<string>1540</string>
	<key>DTXcodeBuild</key>
	<string>15F31d</string>
	<key>LSMinimumSystemVersion</key>
	<string>10.15</string>
</dict>
</plist>
EOF

    echo "Created framework: $OUT/$NAME.framework"

    echo "Created framework: $OUT/$NAME.framework"
}

make_framework "$MVK_DYLIB" "MoltenVK" "$FRAMEWORKS_DIR"
make_framework "$VULKAN_DYLIB" "vulkan" "$FRAMEWORKS_DIR"

# Optional: code sign frameworks for ad-hoc
codesign --force --deep --sign - "$FRAMEWORKS_DIR/MoltenVK.framework"
codesign --force --deep --sign - "$FRAMEWORKS_DIR/vulkan.framework"