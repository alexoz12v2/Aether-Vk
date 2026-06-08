#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="$SCRIPT_DIR/assets/planets"

# Create the target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

echo "Downloading SPICE kernels to $TARGET_DIR..."

# Base URLs
PLANETS_URL="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets"
SATELLITES_URL="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites"

# Download planet kernels
echo "Downloading de442.bsp..."
curl -# -L -o "$TARGET_DIR/de442.bsp" "$PLANETS_URL/de442.bsp"

echo "Download complete!"
