# Test assets

Most files the tests read are not committed. `ASSETS.tsv` lists them in groups; each group is a
deterministic `.tar.gz` attached to a GitHub release, and the tests download what they need on
first use, then reuse the copy in a local cache. Without the network a test that needs an asset
skips with a message; in CI it fails instead.

Committed here: `ASSETS.tsv`, this README and the small files the tests read directly
(`fits_corpus/manifest.tsv`, `fits_corpus/README.md`).

Rust tests resolve files through the dev-only crate `aethervk-test-assets`
(`aethervk.test-assets/`); C# tests use `TestAssets.cs`. Both implement the contract below.

## ASSETS.tsv

- Lines starting with `#` are comments, except `# base <url>`: the download base, ending in `/`.
- Each data row has five tab-separated columns:
  1. `group`: the group name (`saves`, `spk`, `fits_fixtures`, `fits_reference`, `models`);
  2. `paths`: comma-separated, test_assets-relative; each entry is a file or a directory prefix;
  3. `file`: the tarball name;
  4. `sha256`: the tarball's SHA-256;
  5. `size`: the tarball's size in bytes.
- Tarballs are rooted at `test_assets/`: `saves.tar.gz` contains `saves/review_67p_2021-12/...`.

## Resolving a test_assets-relative path `P`

1. If `<repo>/test_assets/P` exists, it is used: a committed file or a developer's local copy.
2. Otherwise, find the group with a `paths` entry `e` such that `P == e` or `P` starts with `e/`.
   If no group matches, the asset is unavailable, with the reason "not in ASSETS.tsv".
3. The cache root is `$AETHERVK_TEST_ASSETS_CACHE`, or `<repo>/target/test-assets` if that is
   unset. The group dir is `<cache>/<group>-<first 12 hex digits of sha256>/`. It is complete
   once it contains `.complete`. The resolved path is `<groupdir>/P`.
4. To fill the group dir, the resolver takes an exclusive file lock on `<cache>/<group>.lock`, so
   that parallel threads and nextest processes cannot race. Then it checks `.complete` again. It
   gets the tarball from `$AETHERVK_TEST_ASSETS_MIRROR/<file>` when that is set. Otherwise it
   downloads `<base><file>`, with a 15 s connect timeout and a 300 s total timeout. The size and
   the sha256 must match the row. The resolver extracts into `<cache>/<group>-<sha12>.tmp-<pid>`,
   renames that to the group dir and writes `.complete`.
5. Any failure (offline, DNS, an HTTP error, a hash or size mismatch) makes the asset unavailable,
   with the reason. With `AETHERVK_TEST_ASSETS_REQUIRED=1` it panics with the reason instead.

The repository root is the first ancestor of the test crate's directory that holds
`test_assets/ASSETS.tsv`.

## Environment

| Variable | Effect |
| --- | --- |
| `AETHERVK_TEST_ASSETS_CACHE` | The cache root (default `<repo>/target/test-assets`). |
| `AETHERVK_TEST_ASSETS_MIRROR` | A directory holding the tarballs; copied from there instead of downloaded. |
| `AETHERVK_TEST_ASSETS_BASE` | Overrides the manifest's `# base` URL. |
| `AETHERVK_TEST_ASSETS_REQUIRED` | `1`: a missing asset fails the test instead of skipping it (CI sets this). |

## Offline behaviour

A test whose asset cannot be resolved prints `skipped: <path>: <reason>` and returns, so it
passes. In Rust, `aethervk_test_assets::test_asset!("<path>")` does this. Each test process
tries a failing group only once. The other tests of that run get the same reason without
waiting for the network again. A filled cache works offline, because complete groups are never
fetched again. Set `AETHERVK_TEST_ASSETS_REQUIRED=1` to make sure that nothing skips.

## Adding or changing an asset

1. Edit the files under `test_assets/` locally. A local file always wins over the cache, so the
   tests already see the change. Generated FITS fixtures are written by
   `cargo test -p aethervk-core-rlib --features std write_fits_fixtures -- --ignored`.
2. To add a new group or path, add it to `ASSET_GROUPS` in `scripts/pack_test_assets.sh`.
3. Pack the group: `scripts/pack_test_assets.sh <out dir> <group> [<group> ...]`. Packing is
   byte-reproducible. The script rewrites the group's row in `ASSETS.tsv` with the new sha256
   and size.
4. Upload the tarball to the release named by the `# base` line:
   `gh release upload <tag> <out dir>/<group>.tar.gz --clobber`. For a new release, create it
   first and update the `# base` line. Old checkouts keep working if their tarballs stay on
   their release.
5. Commit `ASSETS.tsv` (and the script, if you changed it), not the asset files themselves. A
   changed tarball gets a new cache dir, because the dir name carries its sha256.

To test against tarballs before uploading them, run with `AETHERVK_TEST_ASSETS_MIRROR=<out dir>`
and a fresh `AETHERVK_TEST_ASSETS_CACHE`. For a run that cannot see the local copies, use a
checkout without them.
