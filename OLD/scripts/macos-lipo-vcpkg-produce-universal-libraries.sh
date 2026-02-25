#!/bin/sh

# fail on error
set -e

### --- Usage and parsing ---
if [ "$#" -ne 3 ]; then
  echo  "Usage: $0 <arm64-path> <x64-path> <output-path>"
  exit 1
fi

ARM64_ROOT="$1"
X64_ROOT="$2"
OUT_ROOT="$3"

# required subdirectories in input dirs
SUBDIRS="lib debug/lib"
for sd in $SUBDIRS; do
  if [ ! -d "$ARM64_ROOT/$sd" ]; then
    echo "Error: Missing $sd in ARM64 Path: $ARM64_ROOT/$sd"
    exit 1
  fi
  if [ ! -d "$X64_ROOT/$sd" ]; then
    echo "Error: Missing $sd in x86_64 Path: $X64_ROOT/$sd"
  fi
done

# create output directories
for sd in $SUBDIRS; do
  mkdir -p "$OUT_ROOT/$sd"
done

### --- Function to lipo inside directories ---
process_dir() {
  SUBDIR="$1" # first argument = lib or debug/lib

  ARM64_DIR="$ARM64_ROOT/$SUBDIR"
  X64_DIR="$X64_ROOT/$SUBDIR"
  OUT_DIR="$OUT_ROOT/$SUBDIR"

  # first pass: generate fat binaries for each *non-symlink* files
  for f in "$ARM64_DIR"/*; do
    name=$(basename "$f")
    # skip if symlink or not regular file
    if [ -L "$f" ]; then
      continue
    fi
    if [ ! -f "$f" ]; then
      continue
    fi

    AFILE="$ARM64_DIR/$name"
    XFILE="$X64_DIR/$name"
    # ensure corresponding x64 file exists
    if [ ! -f "$XFILE" ]; then
      echo "Warning: Missing x64 file '$name', skipping..."
      continue
    fi
    LIPOCOMMAND="lipo -create -output \"$OUT_DIR/$name\" -arch arm64 \"$AFILE\" -arch x86_64 \"$XFILE\""
    echo "> $LIPOCOMMAND"
    eval "$LIPOCOMMAND"
  done

  # second pass: reproduce matching symlinks
  for f in "$ARM64_DIR"/*; do
    # skip everything but symlinks
    if [ ! -L "$f" ]; then
      continue
    fi
    name=$(basename "$f")
    # read source file name (should be relative, hence only name)
    # no basename needed I think?
    target=$(readlink "$f")
    # ensure target exists
    if [ ! -e "$OUT_DIR/$target" ]; then
      echo "Warning: target for symlink doesn't exist in output: $name -> $target"
    fi

    ln -sf "$target" "$OUT_DIR/$name"
  done
}

### --- process lib and debug/lib folders ---
process_dir "lib"
process_dir "debug/lib"

echo "All Done."