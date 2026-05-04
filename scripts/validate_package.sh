#!/bin/bash
set -e

TYPE=${1}
FILE=${2}

if [ -z "$TYPE" ] || [ -z "$FILE" ]; then
    echo "Usage: $0 [macos|linux] <path-to-app-or-archive>"
    exit 1
fi

echo "=========================================="
echo " Validating $TYPE package: $FILE"
echo "=========================================="

if [ "$TYPE" == "macos" ]; then
    APP_DIR="$FILE"
    if [ ! -d "$APP_DIR" ]; then
        echo "❌ Error: $APP_DIR is not a directory."
        exit 1
    fi
    
    if [ ! -f "$APP_DIR/Contents/MacOS/AetherVk" ]; then
        echo "❌ Executable missing in $APP_DIR/Contents/MacOS/AetherVk"
        exit 1
    fi

    # Check macOS specific Vulkan structure
    if [ ! -f "$APP_DIR/Contents/MacOS/vulkan/share/vulkan/icd.d/MoltenVK_icd.json" ]; then
        echo "❌ MoltenVK_icd.json missing! Ensure VULKAN_SDK was set during build."
        exit 1
    else
        echo "✅ MoltenVK_icd.json found."
    fi
    
    if [ ! -f "$APP_DIR/Contents/MacOS/vulkan/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json" ]; then
        echo "⚠️  VkLayer_khronos_validation.json missing. Validation layers not bundled."
    else
        echo "✅ Vulkan validation layers found."
    fi

    if [ ! -f "$APP_DIR/Contents/MacOS/libaethervk_core_cdylib.dylib" ]; then
        echo "❌ libaethervk_core_cdylib.dylib missing!"
        exit 1
    else
        echo "✅ Native core library (dylib) found."
    fi
    
    echo "🎉 macOS bundle validation passed."

elif [ "$TYPE" == "linux" ]; then
    if [[ "$FILE" == *.tar.gz ]]; then
        if tar -tf "$FILE" | grep -q "libaethervk_core_cdylib.so"; then
            echo "✅ Core .so found in tarball."
        else
            echo "❌ libaethervk_core_cdylib.so missing!"
            exit 1
        fi
    elif [[ "$FILE" == *.deb ]]; then
        if dpkg-deb -c "$FILE" | grep -q "libaethervk_core_cdylib.so"; then
            echo "✅ Core .so found in deb."
        else
            echo "❌ libaethervk_core_cdylib.so missing!"
            exit 1
        fi
    else
        echo "❌ Unsupported linux file type: $FILE"
        exit 1
    fi
    echo "🎉 Linux archive validation passed."

else
    echo "❌ Unknown type $TYPE. Use 'macos' or 'linux'."
    exit 1
fi
