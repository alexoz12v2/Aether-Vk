#!/usr/bin/env bash
# Packs test asset groups from test_assets/ into deterministic .tar.gz files and writes their rows
# into test_assets/ASSETS.tsv. The tarballs are uploaded to the GitHub release named in
# ASSETS.tsv's `# base` line; tests download them on demand (see test_assets/README.md).
#
#   scripts/pack_test_assets.sh <out dir> <group> [<group> ...]
#
# A group's members are the test_assets-relative paths listed in ASSET_GROUPS below. Packing is
# byte-reproducible (sorted names, fixed mtime/owner, gzip -n), so the same files give the same
# sha256 on any machine.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
src="${AETHERVK_TEST_ASSETS_SRC:-$repo/test_assets}"
out="${1:?usage: pack_test_assets.sh <out dir> <group>...}"
shift
mkdir -p "$out"
manifest="$repo/test_assets/ASSETS.tsv"

declare -A ASSET_GROUPS=(
  [saves]="saves/review_67p_2021-12 saves/review_67p_2021-12.avk saves/review_slow_rotator_2026-11.avk"
  [spk]="spk"
  [fits_fixtures]="fits_lt_67p_header_4x3.fits fits_lt_67p_header_4x3.fits.Z fits_image_extension_4x3.fits fits_tile_compressed_4x3.fz fits_cube_4x3x3.fits fits_cube_4x3x3.fz fits_date_variants"
  [fits_reference]="fits_corpus/wcs_reference fits_corpus/hips_reference"
  [models]="BoxTextured.glb BoxTextured.LICENSE.md cloud.png rain.png sun.png"
)

if [[ ! -f "$manifest" ]]; then
  cat > "$manifest" <<'EOF'
# Test asset groups downloaded on demand by the tests (Rust: aethervk-test-assets; C#: TestAssets.cs).
# base https://github.com/alexoz12v2/Aether-Vk/releases/download/test-assets-1/
# group	paths (test_assets-relative, comma-separated)	file	sha256	size
EOF
fi

for g in "$@"; do
  members="${ASSET_GROUPS[$g]:?unknown group $g}"
  for m in $members; do [[ -e "$src/$m" ]] || { echo "missing $src/$m" >&2; exit 1; }; done
  file="$g.tar.gz"
  # shellcheck disable=SC2086
  tar -C "$src" --sort=name --mtime='2026-01-01 00:00:00Z' --owner=0 --group=0 --numeric-owner \
    --format=gnu -cf - $members | gzip -n -9 > "$out/$file"
  sha=$(sha256sum "$out/$file" | cut -d' ' -f1)
  size=$(stat -c %s "$out/$file")
  paths=$(echo "$members" | tr ' ' ',')
  grep -v -P "^$g\t" "$manifest" > "$manifest.tmp" || true
  printf '%s\t%s\t%s\t%s\t%s\n' "$g" "$paths" "$file" "$sha" "$size" >> "$manifest.tmp"
  mv "$manifest.tmp" "$manifest"
  echo "$g: $file $size bytes sha256 $sha"
done
