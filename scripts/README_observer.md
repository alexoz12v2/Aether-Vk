# dust_observer — headless comet dust observer

`aethervk.core/rlib/examples/dust_observer.rs` rebuilds the comet dust scene the .NET app shows
(67P from its SPK, one dust jet, the app's default rotation model and view settings), renders it
off screen from a fixed pose at a fixed cadence over a month and writes frames and numbers into a
directory, so the particle system can be observed without the app.

## Run

```sh
scripts/observe.sh                       # 30 days, 24 h steps, 1280x800, up-zenith, output in the scratchpad v3 dir
scripts/observe.sh --out DIR --days 3 --size 640x400 --check
scripts/observe.sh --zoom --out DIR      # the nine-scale zoom series: 5 km .. 1 AU half-height at one epoch
                                         # (a pre-existing tail, so the AU scale holds a 5-year trail), one
                                         # directory per scale, zoom_sheet.png, the ripple metric per scale
scripts/observe.sh --inside --out DIR    # the near-field sweep: `--pose inside` at D = 1 km, 100 m, 10 m from the surface
                                         # (OBSERVER_INSIDE_LIST="D D ..." in metres overrides), same epoch and
                                         # tail as --zoom, 960x600, --check --mc 8000000, one process per D,
                                         # inside_<D>m/ per distance, inside_sheet.png
# or directly:
cargo run -p aethervk-core-rlib --example dust_observer --features std --release -- \
  --out DIR --days 30 --step-hours 24 --size 1280x800 --pose up-zenith
```

`--out DIR` is required (or `OBSERVER_OUT=DIR`): the script has no default output directory. The
Monte-Carlo truth is placed relative to the nucleus centre (`JetState::centre_m`, the render's
anchor), draws the cone through the renderer's direction-kernel bandwidth (σ = aperture/√S per
stream) and clips at the render camera's near/far planes; its frame-mapping check compares the
τ-weighted centroid of the truth's core (above 1 % of its peak) with the render's (within 1.5
nucleus radii + 2 % of the frame), and the shape is seeded at the brightest
pixel within 1.5 nucleus radii of the nucleus pixel (the jet site). `OBSERVER_MC_YOUNG=1` logs the
youngest batches (lit map at u = 1, pilot hits, grain allotment).

Physical photometry (WP1): the Monte-Carlo truth weights every grain by `p · Φ(α) / r_h²` (the
render's radiance factor), the `mc frame` line and `stats.jsonl` (`abs_ratio`) report the absolute
render/truth sum (`--check`: 0.90–1.10), and each frame's `exposure` record gives the view's white
point as a radiance factor and in V mag/arcsec², the EV and the auto softening.
`--afrho-aperture-km R` measures the aperture Afρ (A'Hearn: `4 r_h²/(π ρ) Σ RF·A_px`) of the render
and of the truth drawn through the renderer's direction bandwidth, and prints the recipe's analytic
`Afρ(r_h)·Φ(α)` (the bandwidth leaks a fixed fraction of a 1/r coma past any aperture);
`--isotropic` makes the jet emit over the whole sphere from a non-rotating nucleus (the closure's
geometry): with it `--check` enforces the aperture closure (5 %) and only reports the whole-frame
comparison.

Perspective truth (`--pose earth`, `--pose inside`): the truth's histogram is τ per pixel area, so
each grain adds `σ · w / (A_px(depth) · cos θ_px)` with `A_px = (2·depth·tan(fov/2)/h)²` m², the
area one pixel covers at the grain's depth (a grain near the camera fills more of its pixel; the
orthographic poses keep the constant `m_per_px²`), and `θ_px` the angle between the pixel's ray
and the camera's forward axis (`cos θ_px = depth / |grain − camera|`): `A_px` is measured on
z-planes, which the pixel's column crosses obliquely, so the optical depth along the ray is the
column's cross-section over `A_px · cos θ_px` (10–15 % at the edge of a 60° field; the engine's
test found it). Grains outside the camera's near/far planes are clipped in
every pose, like the render's. The phase angle's eye is the camera point in perspective (the view
axis at infinity in the orthographic poses). `abs_ratio` is then the render's RF sum over the
truth's `Σ σ·w/A_px(depth)` with no frame-wide `m_per_px`; the aperture Afρ is reported for the
orthographic poses only (a perspective frame has no single pixel area; the request is logged and
skipped). Every other per-pixel scale — the nucleus radius on screen (the hue ray's start, the
truth's frame-mapping tolerance, the shape seed), the shape analysis' reach and km columns, the
moments diagnostics — uses the metres per pixel **at the nucleus' depth**, `2·D·tan(fov/2)/h` in
perspective (`camera.m_per_px_nucleus` in `stats.jsonl`). The hue ray stays in pixel space. A frame whose pyramid saturated (the fixed-point unit lags one readback: the frame
after the ignition) is skipped. `AETHERVK_DUST_EXPOSURE=fixed[:μ]` pins the white point to a
surface brightness in mag/arcsec² (20 when omitted).

Flags (`--out` is required for the binary):

| flag | meaning |
|---|---|
| `--days N` / `--step-hours H` | frames at `start + i·H` for `i = 0 .. N·24/H` (31 frames for 30 d / 24 h) |
| `--size WxH` | off-screen swapchain size |
| `--pose up-zenith` | camera one half-height above the comet along ecliptic +Z, looking down (the C# `ComputeSnapAbove` pose), orthographic |
| `--pose anti-sun` | same height, rolled so the anti-sun direction points to screen +x |
| `--pose earth` | perspective camera handed to the native Earth observer (`CometTracking`) at the site `--lat/--lon/--elev`, `--fov-deg` (default 2) |
| `--sky-image-check` | `--pose earth` only: per frame, a synthetic FITS (`sky_check_NNN.fits`: BITPIX −32, 256×256, TAN WCS with `CRVAL` = the camera → comet direction's astrometric RA/Dec through `astrometry::{ecliptic_to_equatorial_j2000, ra_dec_of}`, `CDELT` = fov/2/256 so the image spans half the frame height, north up, east left, a 3×3 grid of 3-px-thick crosses at FITS pixels 64/128/192) is loaded as the scene's sky overlay (`sky_image::SkyImage::load` + `store_insert`, opacity 1, linear stretch), rendered (`sky_check_NNN.png`) and removed; each cross's luminance centroid (window of half the cross spacing around the prediction, above half way from the window's 20th percentile to its peak) is compared with the pixel the observer's own perspective maths gives the WCS direction of its centre (`[observer] sky cross frame N #k … error E px`); `--check` fails a cross beyond 1 px or not found. Use with `--no-jet --hide-sun` |
| `--lat DEG` / `--lon DEG\|auto` / `--elev M` | the `earth` pose's observing site: geodetic latitude, longitude (degrees) and elevation above the WGS84 ellipsoid (metres); default 45, **auto**, 0; passed to `set_earth_observer(.., lat, lon, elev_m, ..)` and printed in the log. `--lon auto` puts the site's meridian through the comet at the first epoch (hour angle 0, from the Earth's body-fixed frame; the log gives the longitude and the elevation at transit): a fixed longitude at UTC midnight puts the comet below the horizon for months (lat 45° lon 0° on 2025-10-02: −45°), where the tracking camera correctly faces the horizon and the frame is empty. Each frame logs `earth pose frame N (before/after settle)`: camera position, forward axis, camera → comet direction, their angle and the comet's elevation; `--check` fails a frame whose comet is below the horizon. The nucleus pixel and the truth are placed at the light-time view the render uses (`LightTimeViewComponent` on the camera: the comet's micro layer drawn at `x + offset_au`, ~24 500 km at 4.6 AU, logged per frame) |
| `--pose inside` | the near field (WP3): a perspective camera `--inside-m D` metres (or `--inside-km D`; default 1000 m) sunward of the nucleus **surface** (D + 2 km from the centre: the camera sits in the dust, never inside the 2 km nucleus mesh) on the anti-solar axis (the comet's heliocentric direction, the hue ray's axis), looking along it through the nucleus down the tail, up = ecliptic north (+Z), the nucleus pixel at the frame centre; `--fov-deg` (default 60); near `D·1e-3`, far `max(1e4·D, 2·coma radius)` through `add_perspective_camera` (AU), the far plane re-applied each frame as the coma radius settles (logged); the pose is re-applied every frame like the other comet-centred poses. The engine floors the near plane at 1 m (`scene_conversion.rs` `DEPTH_NEAR_FLOOR`), so the truth clips at `max(D·1e-3, 1 m)` (both printed in the `inside pose:` log line) |
| `--inside-m D` / `--inside-km D` | the `inside` camera's distance from the nucleus surface (must be > 0); `m_per_px_nucleus` and the nucleus radius on screen use the true centre distance D + 2 km |
| `--fov-deg DEG` | the perspective poses' full field of view (default 2 for `earth`, 60 for `inside`) |
| `--half-height-km KM` | override the orthographic half-height (default `max(3·coma_radius, β_ref·g·T², 2e4 km)`, printed in the log) |
| `--start YYYY-MM-DD` | start epoch (default 2025-10-02, UTC midnight); the scene range is `max(31, days+1)` days |
| `--spk FILE.bsp` | 67P SPK (default: a `~/.aethervk/spk_90000703_<from>_<to>.bsp` covering the range, else a Horizons download into `~/.aethervk/`) |
| `--play SECONDS` | play mode: instead of seeking, run the simulation for `SECONDS` wall seconds per frame at `step/SECONDS` speed, then pause and capture (exercises the tick path) |
| `--reemit-every N` | play mode: toggle the rotation period by 0.1 h every N frames (forces a re-emit) |
| `--check` | exit 1 when a Vulkan validation error fired, a tier is still building after frame 0, the nucleus pixel is black, a tier holds dust before the ignition or older than the time since it (plus one window), a frame's perspective split ran past its budget (`frame N: split budget overflow K children`, pyramid header word 245), a render failed or the device was lost |
| `--ignite-days D` | the jet starts emitting at `start + D` days (default 0): nothing exists before it, the tail is built in simulated time; frames at or before the ignition hold no dust and are skipped by the shape analysis |
| `--prestart` | the legacy pre-existing tail (history filled back 64 TTL before the start, before the scene start too) |
| `--ripple-max R` | `--check`: the largest allowed ripple. With `--mc`: the render's 9-px high-pass rms **minus the truth's** over the truth's region (default 0.05: a lattice is ripple the truth does not have, a fan edge or a lobed cloud has the same on both sides). Without a truth: the absolute high-pass rms of the τ image (default 0.25, logged as "absolute: no truth"; a smooth cloud is ≈ 0, the former lattices 0.3–0.5). Judged on frames whose region holds ≥ 2000 px |
| `--spin-hours H` | nucleus rotation period (default 12.4); `0` = no rotation model: a still nucleus, lit once per orbit (the dark wedge of `the_problem.rdc`; its `unlit_windows` are reported) |
| `--mc N` | per frame, a Monte-Carlo truth of `N` grains drawn from the continuous emission of the engine's own batch descriptors, compared with the frame's τ image over the region where the truth is above 10⁻⁴ of its peak (`mc_NNN.png` = render \| truth \| ratio, `mc_NNN.f32` the raw pair for `scripts/mc_residual.py`; `mc` fields in `stats.jsonl`, a column in `summary.md`); the grains are allotted to the batches by mass × the fraction a 64-grain pilot lands in the frame (unbiased: the weight per grain stays mass/n), so a near frame spends them on the young windows; ~3 s per 8 000 000 grains. `--zoom` passes `--mc 8000000` unless `OBSERVER_MC=0` (or `=N`) |
| `--mc-max R` | `--check`: the largest allowed 2-D rms of a frame against its truth (default 0.15), judged on `mc_rel_rms_judged`: the region where the truth is above 10⁻⁴ of its peak, without the core above 1 % when that core is fewer than 5000 px (an unresolved sub-pixel coma), with the Monte-Carlo noise (1/√grains per 5×5 box) taken out in quadrature |
| `--mc-tier K` | the truth and the render hold tier K only |
| `--mc-age-max S` | the truth and the render hold the dust younger than `S` seconds only (the render through `AETHERVK_DUST_DEBUG_AGE_MAX`, a cap on every tier's age band); `--zoom` passes 2 h / 1 d / 5 d at 5 / 50 / 500 km, where the older dust's kernels cover the frame as a background the Monte Carlo cannot sample (its correctness is the 50 000 km frame's job) |
| `--mc-age-min S` | the truth holds the dust older than `S` seconds only (every tier's band minimum raised to `S`; all truth models gate on it). The render side is `AETHERVK_DUST_DEBUG_AGE_MIN`, which the observer sets but the engine honours only once `gpu/frame.rs` mirrors its `AETHERVK_DUST_DEBUG_AGE_MAX` hook (`frame.ttl[1] = frame.ttl[1].max(S)`); without it the render keeps every age and the comparison is truth(older than S) vs render(all): the residual is the young dust's share, not an error |
| `--view-flags F` | dust view flags: 1 particle tracers, 2 flow pulses (default 0: the plain optical depth, the app's default since 2026-10-10; the flow pulses drew the far "rays") |
| `--no-jet`, `--hide-sun` | isolation switches (dust-free scene / Sun hidden) |
| `--settle-frames N` | distinct render generations to wait after each seek before downloading (default 4); the frame then waits up to 24 more, one at a time, until the exposure's white point changes by less than 2 % between generations (it adapts by half its log gap per frame: a far frame starts 20 mag from the seed and was a third too dark after 4) — then, while the frame's pyramid reports a speck overflow (`dust::specks::PYR_SPECKS_OVERFLOW` > 0: the host moves the speck shell inward ×0.8 per frame over budget), one more generation at a time, the 24 extra generations shared with the exposure — the log line `render ≈ … (N generations, M to settle the exposure, K to settle the speck shell)` |
| `--full-teardown` | drop the `SimulationContext` at exit (default: flush and exit, see below) |
| `--quiet` | do not echo the engine log to stdout |

Hue diagnostics: `OBSERVER_HUE_LEVELS=1` logs, per hue sample of a `--mc` frame, `[observer] hue
levels frame N sample k at (x, y): L0 τ=.. hue=..° C/L=.. | L1 …` — per pyramid level the τ counts
(per pixel area) over the 9×9 box mapped to that level's texels and that level's colour (its words
over `DUST_COLOR_SCALE·τ`, OKLab hue, chroma over lightness relative to the stream colour's);
`OBSERVER_HUE_LEVEL0=1` makes the judged render colour (`pyr_hue_deg`, `pyr_chroma_rel`,
`pyr_tau`) level 0 only (the per-level line still shows all levels).

Environment: `AETHERVK_PARTICLES_CPU=1` (CPU particle mode), `AETHERVK_DUST_RING=<pow2>` (ring
capacity), `AETHERVK_DUST_DEBUG_TIER=<k>` / `AETHERVK_DUST_DEBUG_STREAM=<s>` /
`AETHERVK_DUST_DEBUG_SAMPLE_MOD=<m>` (splat one tier / one stream / one sample in m: which part
of the dust draws a feature), `OBSERVER_PROFILE=debug` (validation layers + engine debug assertions),
`AETHERVK_DISABLE_GPU_AV=1` (debug: no GPU-assisted validation), `OBSERVER_TIMEOUT` (seconds).

GPU-fault bisection (see `.agents/skills/AetherVk/dust_v3.md`, "GPU fault diagnostics"):
`AETHERVK_RENDER_DEBUG_SKIP=background,sky,meshes,measurements,markers,trajectories,sun` skips a
layer draw phase; `AETHERVK_DUST_DEBUG_SKIP=zero,propagate,moments,splat,composite,readback` skips a
dust phase; `AETHERVK_SINGLE_QUEUE=1` runs compute and transfer on the graphics queue;
`AETHERVK_GPU_CHECKPOINTS=1` turns on the NV queue checkpoints in release builds (debug: always);
`AETHERVK_GPU_LOST_DUMP=1` (the script's default) writes `/tmp/aethervk_gpu_lost_<pid>.txt` on a
device loss. The `[GPU lost]` lines in the run's stdout name the faulting address's buffer (live or
`FREED`) and each queue's last checkpoint.

## Outputs (in `--out`)

- `frame_NNN.png` — RGBA frames.
- `contact_sheet.png` — all frames, 6 per row, downscaled x0.25.
- `stats.jsonl` — one JSON object per frame: `epoch`, `epoch_target`, `seek_ok`, `r_helio_au`,
  `comet_au`, `tiers[]` (`live_clusters`, `capacity`, `youngest_age_s`, `oldest_age_s`,
  `band_min_s`, `band_max_s`, `caught_up`), `coma_radius_km`, `camera` (`pos_au`, `quat_xyzw`,
  `half_height_au/km` (the orthographic framing; the perspective poses carry it but do not use
  it), `m_per_px_nucleus`: metres per pixel at the nucleus' depth, `2·half_height/h`
  orthographic, `2·depth·tan(fov/2)/h` perspective — every per-pixel scale derives from it),
  `nucleus_px`, `nucleus_rgb`, `mean_luminance`, `p99_luminance`, `specks` (the speck header
  words: `d_max_m`, `specks`, `specks_overflow`, `n_eye_m3`; see the pyramid line below),
  `nonblack_fraction`, `exposure`, `hue_ramp[]` (10 samples along the on-screen anti-solar
  direction from the nucleus, from 14 px out (the box clears the nucleus) to the end of the lit
  dust: `px`, `level` (display luma / 255), `hue_deg` and `chroma_rel` (OKLab hue, and chroma
  over lightness relative to the stream colour's, 9×9 box of the linear frame bytes), `dark`
  (< 8/255), `faint` (level < 0.2: the colour words' dither and the 8-bit quantisation show),
  `highlight` (the stretch value, level over the colour's luma, above 0.8: the white roll-off),
  and with `--mc` the
  truth's radiance-weighted mean age and age spread at the sample in TTL (`age_ttl`,
  `age_spread_ttl`) and the truth's own colour there (`truth_hue_deg`, `truth_chroma_rel`: its
  grains' age colours mixed by radiance, the mixture the render claims)), `render_task`. With
  `--mc` the samples run to the end of the truth's region along the ray (the displayed extent
  otherwise; it moved between two runs of one frame). Each sample also carries the render's
  accumulated colour read from the pyramid (`pyr_hue_deg`, `pyr_chroma_rel`, `pyr_tau`: the
  colour words over τ in the 9×9 box, every level per pixel area) — the judged quantity, since
  the displayed pixel also shows whatever the scene has behind the dust (the orbit line under
  the 1 AU tail blends in through the dust's transparency). The hue check (`--check` with
  `--mc`): a sample with at least 50 τ counts in its pyramid box, and whose render/truth τ in
  the box is within 2× (`tau_ratio`; beyond it the box is haze — dust the render has where the
  truth has little, counted per frame and not judged), accumulates the truth's colour,
  hue within 15° and at least 75 % of its chroma (the outlier guard), and the frame's median
  chroma over its judged samples is at least 90 % (the guarantee; a pixel of one age keeps the
  stream's chroma, a pixel of many ages greys by their hue spread, and the check follows the
  truth either way); the render's turn never falls by more than 10° between judged samples of
  increasing truth age. `summary.md` has the hue table (turn of the display, of the render and
  of the truth, truth age at the first and last lit sample, the least render chroma over the
  truth's). The `--mc` line and `stats.jsonl` also carry `haze_frac`: the render's τ outside the
  truth's region over its total, what the region rule leaves unjudged (0.8 % at 50 km: the old
  tiers' pooled packets spread over a near frame).
- `log.txt` — every engine log line (logger callback) plus `[observer]` lines. The
  `[observer] pyramid frame N` line summarises the frame's `.pyr` header: levels, `tau_max`, the
  level mask, the tracer counts, the perspective split's words (`split children` = header word
  11, the children made this frame; `overflow` = word 245, the children the per-frame budget
  refused — `--check` fails the frame when it is > 0; `dropped flux` = word 246, f32 bits of the
  largest flux dropped wholly behind the eye; `depth` = word 247, the host-written split depth
  limit of the frame; `d::PYR_SPLIT_*` in `splat.rs`), the speck pass's words (`dust::specks`:
  `specks` = word 248, the specks drawn; `overflow` = word 249, the specks the budget refused —
  `--check` fails the frame when it is > 0 (`frame N: speck budget overflow K specks`); `d_max` =
  f32 bits of word 252, the shell radius in metres; `n_eye` = `(w251 << 32 | w250) · 2⁻³⁶` m⁻³,
  the number density at the eye) and per level `texels/Σcounts/max`. The same speck words go to
  `stats.jsonl` as `specks: {d_max_m, specks, specks_overflow, n_eye_m3}`. The first ~4 frames
  after a start or a seek have `d_max = 0` (the host computes the shell from the moments
  readback, which lags the draw); the speck shell is off by default (`AETHERVK_DUST_SPECKS=1` turns it on: no GPU parity test or observer validation yet). The
  engine's own `[Dust split] …` lines in `log.txt` report the host's split-depth limit (header
  word 247), which adapts so a frame's children stay under the 1 M (2²⁰) budget.
- `summary.md` — run parameters, framing, per-frame table, check results.
- `stdout.txt` — when run through `observe.sh`.

## Scene recipe (mirrors the FFI the app uses)

`set_asset_path(assets)` → `startup` → almanacs `de442.bsp`, `earth_latest_high_prec.bpc`,
`pck00011.pca`, `gm_de431.pca`, 67P SPK → `create_empty_scene2(false, start, end)` →
`TryInitComet` (SPK 1000012, SBDB elements, `OsculatingAtStart`) and wait for
`CometInitialized` → rotation model (pole RA 0°, Dec 90°, 12.4 h, body-fixed) written to the
scene and the cartesian cache → one jet (`tests/emit_particles_test.rs` parameters: aperture
0.5 rad, 2.0 ± 0.5 m/s, 100 µm, 0.533 g/cm³, Afρ₀ 100 cm, power 2, cutoff 15 AU, nucleus 2 km,
site lat 20° lon 0°, colour (1, 0.6, 0.2), seed 7, TTL 30 d) → dust view flags 2, flow speed 1,
softening 1e-2 → windowless presentation engine + camera → per frame `seek_epoch_sync`, wait for
every tier `caught_up`, re-pose the camera from the comet's global AU position, wait for a few
render generations, `download_image` (BGRA8) → PNG.

The nucleus pixel is projected with the example's own camera math (screen right = local −X,
forward = local −Y, up = local +Z; orthographic through the half-height, perspective through
`tan(fov/2)` and the nucleus' depth; it lands at the image centre for the comet-centred poses,
`inside` included).

The `--inside` sweep (`scripts/observe.sh --inside --out DIR`) runs `--pose inside --inside-m D
--prestart --start 2025-10-17 --days 0 --step-hours 24 --size 960x600 --check --mc 8000000` once
per distance from the surface (default 1000, 100 and 10 m; `OBSERVER_INSIDE_LIST` overrides
with any space-separated metres in [0.001, 1.5e11], i.e. 1 mm .. 1 AU, `OBSERVER_MC` the grains,
`AETHERVK_DUST_SPECKS` is passed through to every run), writing `inside_<D>m/`,
`inside_sheet.png` (frame 001 of each, labelled with the metres per pixel at the nucleus, from
the centre distance D + 2 km) and `inside_summary.txt`: the smoothness summary (per D the last
frame's displayed mean luminance, white point in mag/arcsec², non-black fraction and p99; flagged:
a neighbour step > 25 % in the mean luminance or in the white point's radiance, a black frame
(non-black < 0.1 %) or a saturated one (mean luminance > 230/255); any flag makes the sweep exit
1 — a step the scene explains is the reader's call). Extra flags after `--inside` pass through to every run. The
plan's targets: judged rms ≤ 0.15 and `abs_ratio` 0.9–1.1 at each distance.

## Caveats

- Frames are deterministic (seek-based); play mode follows the wall clock.
- Default exit does not drop the context: the render-thread GPU teardown has wedged/segfaulted
  on exit on this machine (NVIDIA Xid 32), which would mask the `--check` exit code.
- `coma_radius_km` is 0 until the youngest tier has a draw state (after the first rendered frame).

## Shape analysis (`--check`, `--shape`)

`--check` also segments every frame's dust from its pyramid dump (`frame_NNN.pyr`, the τ image
rebuilt through the reference `composite_sample` with the unit of header word 15, the fixed-point
unit the frame's counts were written with) with one threshold (what the last frame displays
at 10 %) and measures the growth step by step (`dust::segment_shape` / `dust::shape_step`): the
run fails when dust appears beyond the reach of the previous frame's faint region (the fastest
grains over the step: ejection plus the radiation-pressure drift of the oldest live dust, plus the
splat footprint), when the radius p90 jumps by more than that reach,
when the area ratio leaves `[0.8, 1.5]`, or when more than 5 % of the dust is detached from the
nucleus region. Outputs: `shape.jsonl` (descriptors per frame: area, radii p50/p90/max in px and
km, axis, elongation, grown / lost / beyond-reach pixels, ratios) and `shape.png` (radius p90 in
orange, area in blue, elongation in green, each over its maximum).

`scripts/observe.sh --shape` = `--check --days 3 --step-hours 6 --size 960x600`, the automated
version of "watch the tail grow at 10 s/s". The GPU path is the default (the 2026-10-09 device
losses were the frame staging arena, fixed); CPU particle mode is the reference-path cross-check
and must give the same picture:

```
scripts/observe.sh --check --days 3 --step-hours 24 --size 960x600 --out DIR/gpu
AETHERVK_PARTICLES_CPU=1 scripts/observe.sh --check --days 3 --step-hours 24 --size 960x600 --out DIR/cpu
```

CPU mode defaults to a 32 768-slot ring (16 / 8 / 8 streams, ~18 s per 960×600 frame on this
machine; `AETHERVK_DUST_RING=8192` is 4× faster and ~15 % further off in extent). Pass the same
`AETHERVK_DUST_RING` to both for a strict comparison; `nonblack`, `shape.jsonl` (area, p90,
elongation) and `coma_radius_km` agree within ~20 % / ~7 % at the defaults. A CPU run lighting most
of the frame means the display follows the sampling again (`dust_tau_image_vs_ring_capacity`,
`dust_image_is_independent_of_ring_capacity` in `scene/dust/tests.rs`).

## Soak runs (`scripts/soak.sh`)

`scripts/soak.sh <outdir> <label> -- <dust_observer args...>` wraps **one** observer run for a long
GPU soak (memory leaks, driver faults, frame-time drift). It runs
`systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0 timeout $SOAK_TIMEOUT <bin> ...`
in the background (stdout+stderr in `<outdir>/<label>.log`; `--out <outdir>/<label>.out` is added
when the args carry no `--out`), finds the observer's pid (the launcher's descendant whose
`/proc/<pid>/exe` is the binary) and every `SOAK_PERIOD` seconds appends to `<outdir>/<label>.csv`:
unix time, elapsed s, VmRSS / RssAnon / RssFile / RssShmem / VmSize (kB), threads, open fds, the
process' GPU memory (MiB: `nvidia-smi --query-compute-apps`, else the plain `nvidia-smi` process
table, where a Vulkan process shows as G / C+G), the GPU's total used memory, and the latest
`[observer] frame NNN <epoch> … | X.Xs` line's frame index, simulated epoch and frame seconds.
At the end it writes `<outdir>/<label>_kernel.txt` (`journalctl -k --since <start>` lines matching
`Xid|NVRM`; dmesg is not permitted here) and `<outdir>/<label>_summary.txt`: exit code, frames
completed / expected (from the `run:` line), log-line counts (device lost, timeout, download fail,
panic, validation error, error) in the stdout and in the observer's `log.txt`, the new Xid/NVRM
lines, least-squares slopes after the warm-up (simulated time ≥ start + `SOAK_WARMUP_DAYS`, else
the first 1/6 of the samples dropped) of RssAnon, GPU process and GPU total memory in MB per
simulated day and per wall hour with the relative growth, the frame-time trend (mean of the first
vs the last quintile of the per-frame seconds, frame 0 excluded) and PASS/FAIL per criterion:
anon RSS and GPU memory growth < 5 % and slope < 1 MB/sim-day, no new Xid/NVRM, no device lost /
panic, every frame rendered, last-quintile frame time ≤ 1.25 × the first, exit 0.

The observer keeps every frame's RGBA in memory for the contact sheet (`frames_rgba`: W·H·4 bytes
per frame, 2.2 MiB at 960x600, ~530 MiB over 241 frames, ~9 MB per simulated day at 6 h steps), so
the raw RssAnon grows by design; the anon-RSS criterion judges RssAnon minus `(frame + 1)·W·H·4`
("net of retained frames"), and the summary prints both.

Environment: `SOAK_TIMEOUT` (default `6h`), `SOAK_PERIOD` (5 s), `SOAK_WARMUP_DAYS` (10),
`OBSERVER_BIN` (default `target/release/examples/dust_observer`, built like `observe.sh` when
missing). Ctrl-C / SIGTERM / SIGHUP stop only that observer pid (SIGTERM, SIGKILL after 15 s) and
the summary is still written. Exit status: the observer's when non-zero (124 timeout, 137 SIGKILL /
OOM in the scope), else 1 when a criterion failed.

`--mc` defaults to 0 (no Monte-Carlo truth), so `--mc 0` is accepted and is the same as leaving it
out. `--play SECONDS` advances simulated time continuously: frame 0 seeks to the start, then each
frame runs the simulation for SECONDS wall seconds at `step-hours·3600/SECONDS`× and pauses before
the capture (wall-clock driven, so the epochs drift slightly from `start + i·step`; the frame
count stays `days·24/step-hours + 1`). The planned soaks:

```sh
D=<scratchpad>/soak
# (i) orthographic, 60 days of tail growth, 241 frames
scripts/soak.sh $D zenith60 -- --pose up-zenith --days 60 --step-hours 6 --size 960x600 --mc 0 --quiet
# (ii) the Earth pose at 0.003° fov (perspective, near the nucleus scale), 241 frames
scripts/soak.sh $D earth60 -- --pose earth --lat 45 --fov-deg 0.003 --days 60 --step-hours 6 --size 960x600 --mc 0 --quiet
# (iii) continuous play (the tick path): 6 h of simulated time per 5 s of wall time (4320x), 241 frames
scripts/soak.sh $D play60 -- --pose up-zenith --play 5 --days 60 --step-hours 6 --size 960x600 --mc 0 --quiet
# (iv) the near field, 41 frames: the warm-up must stay below the 10 simulated days
SOAK_WARMUP_DAYS=2 scripts/soak.sh $D inside100 -- --pose inside --inside-m 100 --days 10 --step-hours 6 --size 480x300 --mc 0 --quiet
```

`--quiet` keeps the engine log out of `<label>.log` (it stays in `<label>.out/log.txt`, which the
summary also scans); the frame lines are printed either way.

## Offline replay lab (`--save-snapshot`, `--lab`)

Capture one GPU frame and its truth, then iterate on the CPU without the GPU:

```
CAP="systemd-run --user --scope -q -p MemoryMax=6G -p MemorySwapMax=0"
# capture (GPU): the judged frames' clusters/push constants/header → snapshot_NNN.bin (~60 MB),
# the truth's raw τ image and grain counts → mc_NNN_truth.f32 / mc_NNN_counts.f32
timeout 2400 $CAP env OBSERVER_MC_RAW=$D/mc_001 target/release/examples/dust_observer --out $D \
  --pose inside --inside-m 1000 --prestart --start 2025-10-17 --days 0 --step-hours 24 \
  --size 480x300 --check --quiet --mc 32000000 --save-snapshot
# replay + score (CPU only, all cores; seconds to minutes): fidelity vs the GPU frame, then the observer's metrics
timeout 900 $CAP target/release/examples/dust_observer --out $D --lab $D [--lab-frame 1]
```

Variants (lab only): `OBSERVER_LAB_HARD=<n>` (hard-edged axes, header bits 24–27),
`OBSERVER_LAB_TIME_CELL=1` (bit 28), `OBSERVER_LAB_VBIAS=<n>` (volume bias),
`OBSERVER_LAB_SRAD=<s>` / `OBSERVER_LAB_SLAT=<s>` (every cluster's Σ_v scaled along / across its size
polyline); engine CPU knobs `AETHERVK_DUST_LAB_SIGMA=<s>`, `AETHERVK_DUST_LAB_PIECES=<n>`,
`AETHERVK_DUST_LAB_NO_TIME_CELL=1`, `AETHERVK_DUST_LAB_ALL_VOLUME=1`, `AETHERVK_DUST_LAB_NO_VOLUME=1`,
`AETHERVK_DUST_LAB_NO_SUPPORT=1` (every packet drawn without its support plane). Header overrides:
`OBSERVER_LAB_DEPTH=<n>` (the split depth, bits 0–7) and `OBSERVER_LAB_GATES_OFF=1` (the in-frame
gates off). With both `OBSERVER_LAB_VBIAS=0 OBSERVER_LAB_GATES_OFF=1 AETHERVK_DUST_LAB_ALL_VOLUME=1`
the lab draws every packet by its exact ray integral (the reference of the drawing; ~30 min at 1 km).
**Threads.** The replay's CPU splat runs on all cores (`splat_tier_parallel`: dynamic 64-cluster
blocks, one partial pyramid per thread, merged order-free by `merge_pyramid`). `OBSERVER_THREADS=<n>`
sets the count, and `OBSERVER_THREADS=1` runs the old serial splat. The line
`lab: splat N clusters on T threads, S s; replay pyramid hash H` gives the splat time and an FNV hash of the
replayed pyramid, and H must be the same at any thread count. Before this change a serial replay
with deep splits (the cut probes, since removed) took 30–60 min. `AETHERVK_DUST_PROBE` forces the
serial path because its reports are per call. `OBSERVER_LAB_PACKET_MC` is threaded the same way,
and the result does not depend on the thread count:
- every (tier, cluster, packet, piece) draws its own random stream;
- each tier's clusters form 64 fixed blocks, summed in block order;
- its line ends with `image hash H`.

`OBSERVER_LAB_LOD=<px>` writes the level-of-detail word (`PYR_LOD`, 61: pixels per drawn segment,
0 = off) for the replay; `AETHERVK_DUST_WORK_LOG=1` then prints the packets per tier. (The
captured frames predate the knob and carry 0; a live run, GPU or CPU, draws at
`AETHERVK_DUST_LOD_PX`, default `DUST_LOD_PX_DEFAULT` = 4 px: set `=0` for the full-detail
A/B.) The lab's
frames keep their captured split depth (the far poses' is 0, the host's budget cut, and they are
~27 % dim for it): compare with `OBSERVER_LAB_DEPTH=4` before blaming a kernel change. The
per-packet probe (`OBSERVER_LAB_PACKET_DRAW=1` with `OBSERVER_LAB_PACKET_MC=4
OBSERVER_LAB_SUPPORT_CUT=kernel`) costs ~1 ms a packet: keep it to frames of ≤ 1 M packets
(50 000 km at lod 8, lod 2 took 15 min).

The `pyramid frame N:` line prints the speck pass's `nodes` (tree nodes visited, frame cap
8·2²⁰) and `truncated` (walks that hit a node cap) words beside the specks drawn: walks at the
cap are budget-bound, and truncated ones lose specks.

`OBSERVER_LAB_SPECKS=<d_max m>` (the replay lab): the CPU speck pass on the replayed tiers before
the splat (size law 50 µm / q 3.5, no fade) at that shell radius, printing the walk's cost: bins
walked (the gate), grains found and drawn, nodes per walk and per speck, truncated walks, the
start-level and found-count histograms (`lab specks:` lines).

`OBSERVER_LAB_CADENCE=1` (with `AETHERVK_DUST_WORK_LOG=1`): the temporal-reuse study — every
cluster's on-screen rate per frame and the cadence classes' packet load (fresh + Σ P_k / N_k) per
simulation step (`OBSERVER_LAB_DT=<s>`, default the app's speeds at 60 fps) and tolerance
(`OBSERVER_LAB_CADENCE_EPS=1,2,4,16`), the eye moving `OBSERVER_LAB_DEYE` m per frame across
the view (default 2.9). See `definite_particles.md`, "Temporal reuse, step 1".

Its numbers changed once on 2026-10-12 (new streams, same statistics). It is noisy at small k:
splat-vs-packets rel rms is 0.60 at k=16 and 0.21 at k=64.
**The truth (`--mc`, `--lab-truth`)** now uses all cores whatever the grain allocation:
- Work is split into fixed units of 4 M grains in batch order. Large batches are split, and
  their later pieces get their own streams.
- Units are summed in order, so the truth is bit-identical at any `OBSERVER_THREADS`.
- `OBSERVER_MC_UNIT_GRAINS=<n>` sets the unit size (diagnostic). A size of at least every
  batch's grain count reproduces the pre-2026-10-12 truth exactly.
- By default, truths with batches above 4 M grains moved within noise (1km_v5 at 32 M: Σ
  5.0753e-4 → 5.0913e-4, σ ≈ 0.13 %).
`--probe "x,y;x,y"` on a GPU run attributes the τ at those pixels to clusters (`probe_NNN.txt`).
The truth uses the Neyman allocation by default (`OBSERVER_MC_ALLOC=mass`: the former one).

**The truth offline.** `--save-snapshot` also writes `truth_NNN.bin` (the truth's batch descriptors,
tier bands, the jet time and centre, the albedo and the frame's view, 0.2 MB). With it the lab draws
the truth itself — the same function as the live run, bit-identical at the same grain count:

```
timeout 900 $CAP target/release/examples/dust_observer --out $D --lab $D --lab-truth 32000000
```

**Dust frame budget.** `--dust-budget-ms 8` (the app's budget, agreed 2026-10-14: 8 ms at 1280×800, every pose) lists every frame whose dust GPU time exceeds it among the problems; with `--check` the run fails on it. The GPU test `dust_frame_fits_the_budget` (a full tier from inside the dust at 1280×800) holds the same budget; it is `#[ignore]`d until the splat fits (run it with `--ignored`).

**Dust cost probes (diagnostics).** The second `dust GPU:` line splits the splats (per dispatch, layer then tier) from the specks. `AETHERVK_DUST_TIMING_PROBE=plain_add` makes the splat's saturating texel adds plain `atomicAdd`s (wraps instead of saturating; what the compare-and-swap loop costs), `=no_volume_writes` evaluates the volumetric texels without writing them (the math against the writes), `=no_volume` drops volumetric packets (the screen part alone); GPU only, never in production, the image wrong by design. (A `batched_counters` probe measured no header-counter contention and was removed.) `AETHERVK_DUST_WORK_LOG=1` with `--lab DIR` prints the replayed splat's work per cluster (texel adds, split children, volume evaluations, EWA units: totals, quantiles, the top 0.1 / 1 / 10 % share, the costliest clusters).

**Dust shader-clock profiler (diagnostics, 2026-10-15).** `AETHERVK_DUST_PROFILE=1` makes the device enable `VK_KHR_shader_clock` and `VK_KHR_pipeline_executable_properties` when it has them, load the `.prof` variants of `dust_splat.comp` and `dust_draw_queue.comp` (`assets/sim/dust_profile.glsl`: probes around each stage of a thread's work, the same drawing word for word, test `gpu_dust_profile_counters_are_consistent`), log every dust pipeline's compiled statistics once (`[dust profile] dust_splat[0] … Register Count = 168; Local Memory Size = 5824; Binary Size = 1473344 …` in `log.txt`) and print, after the `dust GPU:` lines of each frame, a `dust profile` table: per stage the lane-cycles, the share, the cycles per thread and per unit (packet, node, leaf, record, texel, volume packet), with the nesting resolved (`subpk'` is the packet construction outside `dust_project_split_scatter_one`, `split'` the stack loop outside its leaves, and so on; `(rest)` what no stage covers). A lane's cycles are the SM's wall cycles while it is resident, so a lane that waits for its warp-mates in a divergent branch charges the stage it is in: the per-unit numbers are latency and divergence included, which is the point. Then the thread tail (log₂ buckets: p50 / p90 / p99 / p99.9 and the slowest thread), the occupancy reading (Σ thread cycles / (SM clock × the splats' time) = the average resident threads, as warps per SM against Pascal's 64; the register-limited residency is 65536 / (registers × 32)) and the queue draw's split (`qload` the record and band setup, `qrows'` the row loop outside the segments, `qseg` the 16-texel segments; records, texels, the slowest workgroup). The SM clock comes from `nvidia-smi` (override `AETHERVK_GPU_SM_MHZ`), the SM count defaults to 15 (`AETHERVK_GPU_SMS`). The counters are in `stats.jsonl` as `dust_profile`. The probe overhead is the profiled run's `splats` time against a plain run's (a few per cent). `scripts/spirv_stats.sh assets/sim/dust_*.comp.spv` prints the blobs' static metrics (functions, calls, function-local variables by kind, dynamic-index access chains, clock reads, words). `AETHERVK_SPIRV_OPT=1 ./compile_shaders.sh` compiles the dust shaders through spirv-opt `-O` (measured: no GPU-time change; off by default).

**Volume packet queue and the split producer (2026-10-15).** `dust_splat.comp` is the enumerator: a warp per cluster with a lane per sub-packet (the slow end bin's two sub-bins, then a bin each; a merged polyline's time pieces one per lane) writes the cluster's packets, in the former per-cluster thread's order, to a per-chunk sub-packet queue (`dust::DUST_SQUEUE_*`, 20 MiB per view); `dust_split.comp` is the splitter: a workgroup per cluster runs the leaf path (`assets/sim/dust_leaf.glsl`: the perspective split, the EWA leaves, the volumetric packets) over the cluster's records, a counting pass finding how many rounding seeds each record draws and a shared-memory prefix giving it its ordinal base, so every packet draws with the seed the per-cluster thread drew (the test's serial reference, `DUST_FRAME_SPLIT_SERIAL`: one lane walking the records in order, is word-identical). Volumetric packets are not drawn in the lane: they are emitted with their seed to a queue after the EWA queue (`dust::DUST_VQUEUE_*`, 32 MiB per view on a GPU with ≥ 4 GiB, less on smaller devices: `Device::dust_vqueue_records`, `AETHERVK_DUST_VQUEUE_RECORDS=<n>` overrides) and drawn one lane per packet by `dust_draw_volume.comp` (`assets/sim/dust_volume_draw.glsl` is the shared arithmetic; the records are structure-of-arrays and the pieces are built in registers per pass, never kept in an array). A tier runs in chunks of at most 1 024 clusters (`Device::cmd_dust_splat`: the queue headers, the enumerator over the range, the splitter per cluster, the consumer indirectly), sized from the previous frames' packets per cluster so a chunk fits the volume queue; a full volume queue draws the rest in the splitter's lane. Every frame logs `[observer] dust volume queue: N packets drawn by the consumer, K overflowed (drawn in-thread), C chunks`. Knobs: `AETHERVK_DUST_VQUEUE=0` draws volumetric packets in-thread (the A/B; test `gpu_dust_volume_queue_draws_the_same_pyramid` holds the on / off / overflowing / 11-chunk pyramids word-identical), `AETHERVK_DUST_VQUEUE_CHUNK=<clusters>` forces the chunk, `AETHERVK_DUST_VQUEUE_NODRAW=1` skips the consumer (the producer alone; image wrong by design), `AETHERVK_DUST_VQUEUE_LOG=1` logs the queue per frame from the device. The profiler's table then shows the producer's lanes (`threads` = lanes with a sub-packet, a mean life of ~0.3 ms) and a `volume consumer` line (its lane-cycles, `cvplan` per packet, `cvdraw` per texel).

**Image reuse and skips.** A frame whose dust draw inputs and header equal the last drawn frame's reuses its image (the `dust GPU:` line ends `(image reused)`, ~0.8 ms) and the next line gives the last drawn frame's cost, which `--dust-budget-ms` checks. `AETHERVK_DUST_NO_REUSE=1` draws (and propagates) every frame (A/B). The speck pass runs only with the specks on, a shell override, or `AETHERVK_DUST_EYE_DENSITY=1` (then `n_eye` in the pyramid line is measured; otherwise 0).

**Packet queue.** Large EWA packets (not volumetric ones) are drawn by workgroups after the splats (`queue_ms` in `stats.jsonl`'s `dust_gpu`). `AETHERVK_DUST_QUEUE=0` draws every packet in its cluster's thread (A/B), `=all` queues every packet; `AETHERVK_DUST_QUEUE_LOG=1` logs `queue N records drawn of M appended` per frame (M > N: the rest overflowed and were drawn in-thread).

**Dust GPU time.** Every frame logs `[observer] dust GPU: propagate … ms, splat+specks … ms, measure … ms, composite … ms, total … ms (average … ms) at WxH` — per-pass GPU timestamps of the last frame measured before the download (release builds too), also in `stats.jsonl` (`dust_gpu`, beside the wall `render_ms`).

Dissection tools (all `--lab`):

| Variable | What it does |
|---|---|
| `OBSERVER_LAB_AGE=lo,hi` (s) | only the render's clusters of that age (flux 0 elsewhere) and, with `--lab-truth`, the truth's grains of that age |
| `OBSERVER_LAB_AGE_T=lo,hi` | the truth's own gate (a cluster draws the time pieces to its older predecessor: its light holds older grains) |
| `OBSERVER_LAB_AGES=1` | the render's clusters by age per tier (count, Σ flux), the youngest ages, the eye against the youngest polylines |
| `OBSERVER_LAB_PACKET_MC=k` | the render's own packets sampled as grains (`k` each) and scored instead of the splat: tells "the packets are wrong" from "the splat draws them wrong"; with the out-of-region light by size bin × piece |
| `OBSERVER_LAB_SUPPORT_CUT=plane\|cone\|apex\|kernel\|trunc` (with `_PACKET_MC`) | the packets sampled with a cut: `plane`/`cone`/`apex` the observer's own cell classification, `kernel` the exact cut at the kernel's planes (renormalised by `support_mass`), `trunc` the kernel's truncated packets (`support_truncated`) |
| `OBSERVER_LAB_RHIST_INVIEW=1` (with `_RHIST`) | the 3-D histograms of the light that lands in view (its τ weight) instead of every grain's cross-section; the line ends with both sides' absolute totals |
| `OBSERVER_MC_LAT_COUPLED=0` (truth) | each grain's direction-bandwidth kick at the bin's mean speed `v_ref·f` instead of its own: a cell's lateral width no longer grows with its distance from the site (the packets' own assumption) |
| `OBSERVER_MC_KICK_MAX=<k>` (truth, packet MC, the `clusters`/`chords` models) | the whitened kicks (speed noise + direction bandwidth; a packet's Gaussian) drawn from the 3-D normal truncated to `|g| ≤ k`; default 3 (the kernels' bound), `0` unbounded (the former draws, bit for bit) |
| `OBSERVER_MC_SIGMA_ANG=<rad>` (truth) | the direction bandwidth instead of the renderer's stream cell; `0` is the hard-edged cone of the emission model |
| `OBSERVER_LAB_RHIST=1`, `OBSERVER_LAB_RHIST_MODE=centre\|eye\|theta\|phi` | the packets' and the truth's cross-section by distance from the centre / the camera, or by direction (polar angle to the spin axis, azimuth from the Sun) |
| `OBSERVER_LAB_RHIST_MODE=speed\|off\|age` (with `_RHIST`, `_PACKET_MC`) | both sides in the young cell's own coordinates: `speed` the distance along the axis in units of f (0.02 bins from −0.5; the truth's speed along the jet axis over v_ref), `off` the angle off the axis (1°; the truth's to the jet axis, the packets' to their cell's), `age` (30 s; the packets' at their piece's centre); the truth logs its jet once; `tail` the draw's whitened 3-D deviation (0.1 bins: the truth's speed and direction kicks, the packets' Gaussian), `view` the angle from the camera's forward axis (1°: where a cluster meets the frame's edge) |
| `AETHERVK_DUST_LAB_SUBBINS=J` (CPU splat and packet MC) | every size bin as J sub-bins (own speed factor, chord J× shorter, share of the 1/f ramp); production draws the slow end bin alone as `DUST_END_SUBBINS` = 2; `1` the former packets (A/B against the GPU breaks parity) |
| `AETHERVK_DUST_LAB_APEX=0` (CPU splat and packet MC) | the former drawing of young cells' packets; production draws a perspective packet with a cell the eye is not near as its cone of grains (`conic_slabs`: slabs along the cell axis, lateral width scaled with the distance from the cell's origin), uncut |
| `AETHERVK_DUST_LEGACY_KERNEL=1` (live runs, GPU and CPU) | the frames draw the kernel before the cone of grains (header bit 29: no `conic_slabs`, no slow-end sub-bins), for A/B timing on one build |
| `OBSERVER_LAB_VIEW_MAX=<deg>` (with `_RHIST`) | both sides' histograms count only what lies within this angle of the camera's forward axis (a cluster's part at the frame's edge); `_RHIST_MODE=fbin` log₁₀ of the speed factor (the truth's grain's, the packets' bin's mid f) |
| `OBSERVER_LAB_CELL_PRINT=1` (with `_PACKET_MC`) | every 16th young cell's speed range, σ_v/v and σ_ang from its moments, its edges' speed factors and its first packets along the axis (f units) with their σ there |
| `OBSERVER_LAB_PIECE_AGE=lo,hi` (with `_PACKET_MC` and `_SUPPORT_CUT` or a cell mode) | only the packets whose time piece's centre age is in [lo, hi): with `OBSERVER_LAB_AGE_T` and `OBSERVER_MC_ALLOC=uniform` (every batch drawn, not only the in-view ones) the pieces' placement against the truth's grains of the same ages |
| `OBSERVER_LAB_PACKET_DRAW=1` (with `_PACKET_MC`, `_SUPPORT_CUT=kernel`) | every packet also drawn alone on a fresh pyramid under the lab's header, against its own grains: drawn/MC by drawing path (uncut eye-near, uncut split, cut eye-near, cut truncated) and each path's largest differences; `OBSERVER_LAB_PACKET_PROBE=<cluster>` prints that cluster's under-drawn packets in full |
| `OBSERVER_LAB_DUMP=<prefix>` | the scored unit-sum boxed images (`_render.f32`, `_truth.f32`) |

The 1 km datasets live in `aethervk.core/rlib/test_assets/lab/` (untracked).

## Camera path and the cadence's image gate (`--path-frames`, `--lab-path`, `--lab-score`)

Temporal reuse, step 4: the cadence is judged on a moving camera, with its frames drawn as at 60 fps.

**The path (GPU, `--pose inside --days 0`).** `--path-frames N` (32) starts from the settled frame 0. Each later frame k is one hold-step:
1. The render budget holds the view (`SimulationContext::set_render_budget(scene, pe, Some(0))`: the logic thread sends that view no frame).
2. The clock seeks, paused, to t₀ + k·Δt, and the tiers catch up.
3. The camera moves to nucleus(t_k) + rel_k. rel_k is rel₀ turned by k·ω about the orbit axis, which is ecliptic north across rel₀, so the view never rolls. rel_k is nucleus-relative in fixed world axes, its length less k·dolly.
4. Exactly one frame is released (`Some(1)`).

The renderer therefore sees Δeye and Δt together, once per frame, as the cadence will at 60 fps. The far plane stays at frame 0's: a moving one would change the projection, which resets the cadence.

Flags:
- `--path-fps F` (60).
- `--path-speed X`: simulation seconds per wall second; Δt = X / F.
- `--path-dt-s T`: Δt directly.
- `--orbit-deg-per-s W` (10).
- `--eye-step-m S`: the orbit by its arc per frame, overriding W.
- `--dolly-m-per-frame D` (0).
- `--path-keep 0,31`: the frames whose `frame_NNN.pyr`, PNG, `truth_NNN.bin` and, with `--save-snapshot`, snapshot are written. Default: the first and the last.

The frame ledger (`dust_frame_ledger`, `gpu/frame.rs`: the last 256 dust frames, each with its sequence, reuse flag, simulation time, eye and nucleus) proves the path. Each step must have exactly one new dust frame, drawn, at t₀ + k·Δt (to 1 ms) with eye − nucleus = rel_k (to 1 mm). `--check` fails otherwise.

Outputs:
- `path.txt`: the path's parameters and the cadence's environment variables.
- `path_summary.txt`: clean, steps, GPU ms per path frame mean / p90 / max, and the cadence's telemetry (`n/a` until the GPU cadence exists).
- In `stats.jsonl`, one `path_frame` line per frame: θ, Δt, eye − nucleus, the ledger's errors, the frame's light Σ, the tiers' live counts, the wall time.

**The score (`--lab-score DIR --against FULL --lab-frame F --lab-truth N`).**
- Frame F of a path capture and the fully drawn frame of the same path (`AETHERVK_DUST_CADENCE=0`) are scored against one truth.
- That truth is `FULL/truth_FFF.bin` drawn once at N grains and cached as `FULL/truthN_FFF.f64` (+ `_counts.f64`).
- Both runs' truth inputs must be byte-equal, otherwise the score exits 2 (not the same frame).
- Printed per frame:
  - the observer's judged value, Σ render/truth, the ratio by truth level and the light outside the region (`run_lab`'s arithmetic; `run_lab` prints a `score check` line proving it is the same);
  - the **60-px block rms**: blocks of 60×60 px with truth ≥ 5 % of the brightest block, sqrt(Σ(R_B − T_B)² / Σ T_B²), absolute; also shown with the shape only and as the truth's count noise at that scale;
  - **x against full directly** (no truth, no noise): Σ ratio, shape rel rms, block60 rms, and the worst block with its block coordinates.
- `--check` fails when either path is not clean, or when |Σ x / Σ full − 1| > 1 % + the ghost share (0 while the cadence reports none).
- `--lab-table T.tsv` appends the row; `--lab-table-md T.tsv` prints the tables (one per source × speed; rows by pose × variant).
- A second fully drawn run (`aa`) gives the run-to-run floor of the direct numbers.

**The recipe.** `scripts/lab_cadence_capture.sh` runs poses 1 km / 100 m / 5 000 km × speeds 0 / 60 / 86 400 × variants `full aa` (later `eps1 eps2 eps4 eps16`, and `fb` once step 5 brings its knob). Every run is one process under the memory cap. Captures go under `~/aethervk-lab/lab/<pose>_cad/s<speed>/<variant>`; the table goes to `~/aethervk-lab/lab/table.md`. `--smoke` runs 1 km at 60× for 8 frames, full + aa only.

**The CPU mirror's path (`--lab DIR --lab-frame F --lab-path N --lab-truth M [--lab-path-eps 1,2,4,16] [--check]`).**
- It replays a snapshot along the same orbit with Δt = 0, using the cadence's CPU state machine (`scene::dust::cadence`) against the full parallel splat on every frame.
- The path's projection comes from the captured mvp: Q = [M₀(3×3) | M₀ col3 + M₀(3×3)·eye_local], and each frame's mvp is Q·ΔRᵀ·[I | anchor − eye_k]. It is checked against every tier's captured mvp at frame 0.
- The last frame is scored against the truth drawn at the last view.
- The check `lab_cadence_scores_the_path`:
  - frame 0 word-identical to the full draw with `OBSERVER_LAB_GATES_OFF=1`;
  - every frame |Σ cadence / Σ full − 1| ≤ 1 %;
  - the owner bytes consistent;
  - at least one class image shown (else vacuous).
- `--lab-table` appends its rows (source `CPU mirror`, speed 0).

## The deferred edge pass and the edge-piece demotion (9A, 2026-10-19)

**The edge pass.** The volume consumer (`dust_draw_volume.comp`) no longer draws small edge pieces itself: it appends a 24-word edge record and moves on. `dust_draw_small_edge.comp` (one lane per record, dispatched indirectly beside the next chunk's producer) runs the fast slices, then either the demotion below or the 4 sigma points, and writes the rows. The records live in two parity-alternating regions of max(n/2, 64) × 96 B each (24 MiB per view at full capacity). A record that does not fit is dropped and counted. The observer prints one line per view:

```
dust edge pass: N small edge pieces drawn, M of them demoted (ε 1e-3), L lost
```

L must be 0. The GPU parity test asserts it, and the production runs at 10 m and 5 000 km measured 0.

**The demotion.** With `DUST_SMALL_DEMOTE_EPS` = 1e-3 (production), an edge piece whose predicted error is below ε of its mass is drawn with one projection instead of the 4 sigma points. The one-projection point carries the second-order correction: mass ×(1+μ), mean +δ, covariance +ΔC (on the CPU in `splat.rs` and on the GPU in `dust_scatter.glsl`, bit-consistent). Before the predictor runs, a cheap conservative gate (centre slice + κ) has to pass: the full predictor is evaluated only when the gate is < `DUST_SMALL_DEMOTE_GATE` = 0.5 ε. ε = 0 gives bit-identical images. Measured at 10 m: 213 532 of 2 841 524 edge pieces demoted (7.5 %). At 5 000 km: 213 435 of 229 019 (93 %).

Tests:
- `small_demotion_calibration` (CPU, `splat.rs`). Two rules, at every ε: sound (worst |ΔL|/(εM) ≤ 1) and accurate (the extra error against the truth ≤ 2e-3). With the correction, ε_cal = 1e-2 passes, with an extra error ≤ 5.2e-5.
- `small_demotion_tail_survey` (CPU). A tail scene from the nucleus (16 streams, a 0.2 rad cone, 200 / 400 d). It needs a candidate pose with ≥ 200 pieces demoted.
- `gpu_dust_small_demotion_matches_reference`. Cases 9056/9057 need ≥ 200 pieces demoted, with the GPU count equal to the CPU count. Every `splat_parity` case asserts |GPU − CPU demoted| ≤ the small margins. The tail frames lower `PARITY_MIN_NONZERO` from 300 to 100.

## The budget loop's fallback and the GPU cadence switch (track 7, T6E)

- **The GPU cadence is opt-in.** It runs only with `AETHERVK_DUST_CADENCE=1`. `scripts/lab_cadence_capture.sh`'s eps variants now set it. Captures taken before 2026-10-19 14:50 drew their eps rows in full (cadence off).
- **The by-count fallback.** When the loop is at its last rung, it keeps the K highest-δ clusters fresh. Under the fallback, class k also needs 2N_k·Δt ≤ θ·age, with θ = `DUST_CADENCE_FALLBACK_SHAPE` = 0.25. So clusters younger than 2N₀Δt/θ (25.6 h at 1 d/s) stay fresh: the near-nucleus dust has the smallest δ and would otherwise go stale first. With θ = 0 (everywhere except under the fallback) the old rule holds word for word, and the CPU and GPU agree bit for bit. The test `fallback_keeps_the_nucleus_fresh` checks that the youngest 1 % and the 1 % nearest the nucleus are 100 % fresh, for a still camera and a following one.
- **The class images keep the frame's level count (T6E).** The 48-frame path check now passes: Σ ratio 0.9993–1.00015 over frames 40–47 (it was 0.985 before).
