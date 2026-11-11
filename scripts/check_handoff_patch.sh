#!/usr/bin/env bash
# Checks that a handoff patch from a platform machine touches only that platform's files.
#
#   scripts/check_handoff_patch.sh <patch-file> windows|macos
#
# The patch's files are its "+++ b/<path>" lines (plus "--- a/<path>" for deletions). Allowed:
#   windows: aethervk.ui.e2e/ and any *.Windows.cs
#   macos:   aethervk.ui.e2e.mac/, scripts/package_macos.sh, TODO_macos_packaging.md and any *.Mac.cs
# Everything shared is outside both lists (aethervk.ui.e2e.shared/, XAML, view models, resx, the
# AutomationIds file, Rust): a change there is made in the main tree, not in a handoff patch.
# Exit 0 when every path is allowed, 1 with the offending paths listed, 2 on usage errors.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <patch-file> windows|macos" >&2
  exit 2
fi
patch=$1
platform=$2
if [[ ! -f $patch ]]; then
  echo "no such patch: $patch" >&2
  exit 2
fi

allowed() {
  local p=$1
  case $platform in
    windows)
      [[ $p == aethervk.ui.e2e/* || $p == *.Windows.cs ]]
      ;;
    macos)
      [[ $p == aethervk.ui.e2e.mac/* || $p == scripts/package_macos.sh || $p == TODO_macos_packaging.md || $p == *.Mac.cs ]]
      ;;
  esac
}

case $platform in
  windows | macos) ;;
  *)
    echo "platform must be windows or macos, not '$platform'" >&2
    exit 2
    ;;
esac

# "+++ b/path" (added/modified/renamed-to) and "--- a/path" (deleted/renamed-from); strip a
# trailing tab + timestamp and the a/ b/ prefixes; /dev/null is not a file.
mapfile -t files < <(
  sed -n -E 's/^(\+\+\+ b|--- a)\/([^\t]*).*$/\2/p' "$patch" | sed -E 's/[[:space:]]+$//' | sort -u
)
if [[ ${#files[@]} -eq 0 ]]; then
  echo "$patch: no '+++ b/' file headers (not a unified diff from git diff / diff -ruN with a/ b/ prefixes?)" >&2
  exit 2
fi

bad=()
for f in "${files[@]}"; do
  allowed "$f" || bad+=("$f")
done

echo "$patch: ${#files[@]} file(s) touched"
if [[ ${#bad[@]} -gt 0 ]]; then
  echo "FAIL: ${#bad[@]} path(s) outside the $platform allowlist:" >&2
  printf '  %s\n' "${bad[@]}" >&2
  exit 1
fi
printf '  %s\n' "${files[@]}"
echo "OK: every path is inside the $platform allowlist"
