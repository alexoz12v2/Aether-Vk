#!/usr/bin/env bash
# Downloads the FITS test corpus listed in test_assets/fits_corpus/manifest.tsv into <dest> and
# verifies every file's sha256. Idempotent and resumable: a file already present with the right
# hash is skipped, a partial download is resumed (curl -C -), a file with a wrong hash is moved
# aside to <file>.bad-<time> (never deleted) and fetched again once.
#
#   scripts/fetch_fits_corpus.sh <dest> [--cds] [--category c1,c2]
#
# Sources:  http(s)://...       downloaded to <dest>/<path>
#           tar:<archive path>  extracted from that manifest archive into
#                               <dest>/<category>/<archive stem>/ (GNU tar: no absolute paths,
#                               no '..', no owners or modes kept)
# --cds     also downloads the CDS fits-rs test set (~2.2 GB) to <dest>/cds/fits-rs-test-files.tar
#           and extracts it into <dest>/cds/fits-rs-test-files/. While the manifest has no row for
#           the tar its hash is printed (append rows with scripts/fits_corpus_inventory.py
#           --new-rows ... --category cds --source-prefix tar:cds/fits-rs-test-files.tar).
#
# Then run the harness: AETHERVK_FITS_CORPUS=<dest> cargo test ... fits_corpus_opens_every_image
set -u -o pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
manifest="$here/test_assets/fits_corpus/manifest.tsv"
cds_url=https://alasky.cds.unistra.fr/Aladin-Lite-test-files/fits-rs-test-files.tar
cds_rel=cds/fits-rs-test-files.tar

dest=""; want_cds=0; cats=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cds) want_cds=1 ;;
    --category) cats="$2"; shift ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) dest="$1" ;;
  esac
  shift
done
[ -n "$dest" ] || { echo "usage: $0 <dest> [--cds] [--category c1,c2]" >&2; exit 2; }
mkdir -p "$dest"
dest=$(cd "$dest" && pwd)

sha() { sha256sum "$1" | cut -d' ' -f1; }
stem() { local b; b=$(basename "$1"); b=${b%.gz}; b=${b%.tgz}; echo "${b%.tar}"; }
in_cats() { [ -z "$cats" ] && return 0; case ",$cats," in *",$1,"*) return 0 ;; esac; return 1; }

fetch_url() { # url file
  mkdir -p "$(dirname "$2")"
  curl -fsSL --retry 3 --connect-timeout 30 -C - -o "$2" "$1"
}

ok=0; fetched=0; bad=0; missing=0
declare -A extracted

extract() { # archive_rel category
  local key="$1" out
  [ -n "${extracted[$key]:-}" ] && return 0
  out="$dest/$2/$(stem "$1")"
  mkdir -p "$out"
  echo "extract $1 -> ${out#"$dest"/}"
  tar --no-same-owner --no-same-permissions -xf "$dest/$1" -C "$out" || return 1
  extracted[$key]=1
}

check_row() { # path sha size source category
  local path="$1" want="$2" src="$4" cat="$5" file="$dest/$1"
  if [ -f "$file" ] && [ "$(sha "$file")" = "$want" ]; then ok=$((ok + 1)); return 0; fi
  if [ -f "$file" ] && [ "${src#tar:}" = "$src" ] && [ "$(stat -c %s "$file")" -ge "$3" ]; then
    mv "$file" "$file.bad-$(date +%s)"   # complete but wrong: keep it aside, fetch again
  fi
  case "$src" in
    tar:*) local arch="${src#tar:}"
           [ -f "$dest/$arch" ] || { echo "MISSING archive $arch for $path"; missing=$((missing + 1)); return 1; }
           extract "$arch" "$cat" || { echo "FAILED extract $arch"; bad=$((bad + 1)); return 1; } ;;
    http*) fetch_url "$src" "$file" || { echo "FAILED download $src"; missing=$((missing + 1)); return 1; } ;;
    *) echo "unknown source $src"; bad=$((bad + 1)); return 1 ;;
  esac
  if [ -f "$file" ] && [ "$(sha "$file")" = "$want" ]; then fetched=$((fetched + 1)); return 0; fi
  echo "BAD sha256 $path (want $want, got $( [ -f "$file" ] && sha "$file" || echo none))"
  bad=$((bad + 1)); return 1
}

if [ $want_cds = 1 ]; then
  if [ -f "$dest/$cds_rel" ] && grep -q "^$cds_rel	" "$manifest"; then
    : # verified with the rows below
  else
    echo "fetching the CDS fits-rs test set (~2.2 GB) -> $cds_rel"
    fetch_url "$cds_url" "$dest/$cds_rel" || echo "FAILED download $cds_url (CDS unreachable? run again later, it resumes)"
  fi
  if [ -f "$dest/$cds_rel" ] && ! grep -q "^$cds_rel	" "$manifest"; then
    tar -tf "$dest/$cds_rel" >/dev/null 2>&1 && extract "$cds_rel" cds
    echo "CDS tar has no manifest row yet: sha256 $(sha "$dest/$cds_rel") size $(stat -c %s "$dest/$cds_rel")"
  fi
fi

# archives first (rows whose `expected` is `archive`), then everything else
for pass in archive rest; do
  while IFS=$'\t' read -r path want size src cat expected notes; do
    case "$path" in ''|'#'*) continue ;; esac
    in_cats "$cat" || continue
    [ "$cat" = cds ] && [ $want_cds = 0 ] && continue
    if [ "$pass" = archive ] && [ "$expected" != archive ]; then continue; fi
    if [ "$pass" = rest ] && [ "$expected" = archive ]; then continue; fi
    check_row "$path" "$want" "$size" "$src" "$cat"
  done < "$manifest"
done

echo "fits corpus in $dest: $ok verified, $fetched fetched, $bad bad, $missing missing"
[ $bad = 0 ] && [ $missing = 0 ]
