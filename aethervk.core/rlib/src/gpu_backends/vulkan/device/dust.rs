//! Dust system v3/v4 device resources and command recording. See [`crate::scene::dust`] and
//! [`crate::scene::dust::splat`].
//!
//! Per particle system:
//! - `clusters`: ring of immutable [`DustCluster`] records (device local). Written only by
//!   `dust_emit.comp` on the compute queue, read by `dust_propagate.comp` on the graphics queue.
//!   Append-only + FIFO expiry, so the only synchronization needed is the graphics submit waiting
//!   on the compute timeline value of the newest emission it draws (no ownership transfer: the
//!   buffer is `CONCURRENT` when the queue families differ).
//! - `render`: per-frame [`DustRenderCluster`]s (positions, predecessor words), written by
//!   propagate, read by `dust_splat.comp` and the dust trace.
//! - `moments`: per-frame [`DustMoments`] (the packets' second moments), written by propagate,
//!   read by `dust_splat.comp`.
//! - `batches`: host-visible ring of [`DUST_BATCH_SLOTS`] [`DustBatch`] descriptors read by
//!   `dust_emit.comp` through its buffer device address.
//!
//! Per presentation engine ([`DustView`]): the fixed-point splat pyramid (`dust::PyramidLayout`)
//! every system of the frame scatters into, zeroed per frame, read by `composite.frag` through
//! its device address; a host-visible readback ring of the header and the white-point level
//! ([`dust::DUST_WHITE_LEVEL`]), read [`DUST_READBACK_SLOTS`] frames later for the white point,
//! the auto softening and the fixed-point unit (eye adaptation).
//!
//! CPU particle mode (`AETHERVK_PARTICLES_CPU=1`) keeps the ring in host memory and runs
//! [`dust::emit_cluster`] / [`dust::packet_moments`] / [`dust::splat_tier`] (the reference
//! implementation) instead of the compute shaders; the pyramid is then a host-visible buffer the
//! host writes directly and the composite reads through the same address. Same layouts, same
//! composite.
/// the GPU mirror of the cadence classes (temporal reuse, step 3; `dust_cadence.rs`)
#[path = "dust_cadence.rs"]
pub mod cadence;
use super::*;
use crate::scene::dust::{
  self, DustBatch, DustCluster, DustEmitPushConstants, DustFrame, DustMoments,
  DustPropagatePushConstants, DustRenderCluster, DustSplatPushConstants, PyramidLayout,
  specks::{self, DustSpecksPushConstants, ShellUpdate, SpeckHeader},
};

/// batch descriptor slots per system; a slot is reused after this many emissions (a tick records
/// at most `logic_thread::utils::DUST_BATCHES_PER_TICK` before it waits for its own submission)
pub const DUST_BATCH_SLOTS: u32 = 4096;
/// ring capacity in CPU particle mode (evaluation is single threaded): the low GPU tier, whose
/// 16 / 8 / 8 streams keep the image within a few % of the 262 144 ring
/// (`dust_tau_image_vs_ring_capacity`); 8 192 (4 / 2 / 2) was ~15 % off in extent
pub const DUST_RING_CAPACITY_CPU: u32 = 32_768;
/// local size of `dust_emit.comp` / `dust_propagate.comp` / `dust_splat.comp` / `dust_measure.comp`
const DUST_WG: u32 = 64;
/// frames between a readback copy and its host read (≥ frames in flight)
pub const DUST_READBACK_SLOTS: usize = 4;

/// GPU timestamps per dust frame ([`DustGpuTimings`]): the frame's start, after the propagates,
/// after the splats and specks, after the measure and readback copy, around the composite; then
/// after the splats (before the specks), after each of the first [`DUST_TS_CALLS`] splat
/// dispatches, and after the packet queue
pub const DUST_TIMESTAMPS: u32 = 27;
/// the most workgroups a dispatch may have per dimension on the baseline devices (Vulkan's
/// required `maxComputeWorkGroupCount`); larger dispatches go two-dimensional
pub const DUST_DISPATCH_MAX_X: u32 = 65_535;
pub const DUST_TS_BEGIN: u32 = 0;
pub const DUST_TS_PROPAGATED: u32 = 1;
pub const DUST_TS_SPLATTED: u32 = 2;
pub const DUST_TS_MEASURED: u32 = 3;
pub const DUST_TS_COMPOSITE_BEGIN: u32 = 4;
pub const DUST_TS_COMPOSITE_END: u32 = 5;
pub const DUST_TS_SPLATS_DONE: u32 = 6;
/// the timestamp after splat dispatch `k` is `DUST_TS_CALL0 + k` (`k < DUST_TS_CALLS`)
pub const DUST_TS_CALL0: u32 = 7;
pub const DUST_TS_CALLS: usize = 8;
/// after the packet queue's draw (`dust_draw_queue.comp`, in [`Device::cmd_dust_frame_end`])
pub const DUST_TS_QUEUED: u32 = DUST_TS_CALL0 + DUST_TS_CALLS as u32;
/// after the speck passes, which run before the splats since 2026-10-16 (they write the walk
/// mask the splats read: `DustMoments::specks_mask`)
pub const DUST_TS_SPECKED: u32 = DUST_TS_QUEUED + 1;
/// after the cadence's begin (fills, control, transitions, compaction, the propagate marks)
pub const DUST_TS_CADENCE: u32 = DUST_TS_SPECKED + 1;
/// after the cadence's classifies, chunk pre-passes and build draws
pub const DUST_TS_BUILDS: u32 = DUST_TS_CADENCE + 1;
/// after the reprojection of the shown class images
pub const DUST_TS_REPROJECTED: u32 = DUST_TS_BUILDS + 1;
/// after the cadence's classifies and chunk pre-passes (before its build draws)
pub const DUST_TS_CHUNKED: u32 = DUST_TS_REPROJECTED + 1;
/// after the build draws of pair `k * 2 + p` (`DUST_TS_PAIR0 + k * 2 + p`, six)
pub const DUST_TS_PAIR0: u32 = DUST_TS_CHUNKED + 1;
const _: () = assert!(DUST_TS_PAIR0 + 6 <= DUST_TIMESTAMPS);

/// What a dust frame cost on the GPU, per pass (ms; `NaN` = not measured: a pass not recorded
/// that frame, or its timestamps not yet available), read [`DUST_READBACK_SLOTS`] frames after
/// it was recorded ([`Device::dust_gpu_timings`]). Release builds included: a few timestamps per
/// frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DustGpuTimings {
  /// the propagates of every system (frame begin → after the last propagate: the header write and
  /// the zeroing of the pyramid included)
  pub propagate_ms: f64,
  /// every layer's splats and specks
  pub splat_ms: f64,
  /// of which the splats (to [`DUST_TS_SPLATS_DONE`]) and the specks after them
  pub splat_only_ms: f64,
  pub specks_ms: f64,
  /// each splat dispatch in recording order (layer, then tier): from the previous dispatch's end
  /// (the propagates' for the first) to its own; `NaN` past the dispatches recorded
  pub call_ms: [f64; DUST_TS_CALLS],
  /// the packet queue's draw (after the splats and specks; `NaN` when not recorded)
  pub queue_ms: f64,
  /// the cadence (`NaN` without it): its begin (inside `propagate_ms`), the classifies and build
  /// draws, the reprojection (inside `splat_ms`)
  pub cadence_ms: f64,
  pub builds_ms: f64,
  pub reproject_ms: f64,
  /// the cadence's classifies and chunk pre-passes, and each pair's build draws (`k * 2 + p`;
  /// from the previous pair's end, the chunks' for the first)
  pub classify_ms: f64,
  pub pair_build_ms: [f64; 6],
  /// the measurement grid and the readback copy
  pub measure_ms: f64,
  /// the composite draw (inside the render pass)
  pub composite_ms: f64,
  /// frame begin → measured, plus the composite
  pub total_ms: f64,
  /// an exponential average of `total_ms` (weight 1/8 per frame)
  pub total_ema_ms: f64,
  /// the view's dust frame number these timings are of, and its pyramid size
  pub frame: u64,
  pub width: u32,
  pub height: u32,
  /// the frame kept the previous frame's image ([`Device::cmd_dust_frame_begin_reusable`]): its
  /// spans are the measure and readback only
  pub reused: bool,
}

/// `ticks` × `period_ns` in ms, `NaN` when either end is missing or out of order
pub fn timestamp_span_ms(t0: Option<u64>, t1: Option<u64>, period_ns: f32) -> f64 {
  match (t0, t1) {
    (Some(a), Some(b)) if b >= a => (b - a) as f64 * period_ns as f64 * 1e-6,
    _ => f64::NAN,
  }
}

impl DustGpuTimings {
  /// The spans of one frame's timestamps (`ts[i]`: `None` when not written or not available)
  pub fn from_timestamps(ts: &[Option<u64>; DUST_TIMESTAMPS as usize], period_ns: f32) -> Self {
    let span = |a: u32, b: u32| timestamp_span_ms(ts[a as usize], ts[b as usize], period_ns);
    let propagate_ms = span(DUST_TS_BEGIN, DUST_TS_PROPAGATED);
    let splat_ms = span(DUST_TS_PROPAGATED, DUST_TS_SPLATTED);
    // the specks run before the splats (`DUST_TS_SPECKED`); a frame without them has no such
    // timestamp and the splats start at the propagate's end
    let after_specks = if ts[DUST_TS_SPECKED as usize].is_some() {
      DUST_TS_SPECKED
    } else if ts[DUST_TS_REPROJECTED as usize].is_some() {
      DUST_TS_REPROJECTED
    } else {
      DUST_TS_PROPAGATED
    };
    let splat_only_ms = span(after_specks, DUST_TS_SPLATS_DONE);
    // (with the cadence the specks follow its builds and reprojection)
    let before_specks = if ts[DUST_TS_REPROJECTED as usize].is_some() { DUST_TS_REPROJECTED } else { DUST_TS_PROPAGATED };
    let specks_ms = span(before_specks, DUST_TS_SPECKED);
    let cadence_ms = span(DUST_TS_BEGIN, DUST_TS_CADENCE);
    let chunked = ts[DUST_TS_CHUNKED as usize].is_some();
    let builds_ms = span(if chunked { DUST_TS_CHUNKED } else { DUST_TS_PROPAGATED }, DUST_TS_BUILDS);
    let classify_ms = span(DUST_TS_PROPAGATED, DUST_TS_CHUNKED);
    let mut prev_pair = DUST_TS_CHUNKED;
    let pair_build_ms = core::array::from_fn(|i| {
      let at = DUST_TS_PAIR0 + i as u32;
      let ms = span(prev_pair, at);
      if ts[at as usize].is_some() {
        prev_pair = at;
      }
      ms
    });
    let reproject_ms = span(DUST_TS_BUILDS, DUST_TS_REPROJECTED);
    let mut prev = after_specks;
    let call_ms = core::array::from_fn(|k| {
      let at = DUST_TS_CALL0 + k as u32;
      let ms = span(prev, at);
      if ts[at as usize].is_some() {
        prev = at;
      }
      ms
    });
    let queue_ms = span(DUST_TS_SPLATTED, DUST_TS_QUEUED);
    let measure_ms = if ts[DUST_TS_QUEUED as usize].is_some() {
      span(DUST_TS_QUEUED, DUST_TS_MEASURED)
    } else {
      span(DUST_TS_SPLATTED, DUST_TS_MEASURED)
    };
    let composite_ms = span(DUST_TS_COMPOSITE_BEGIN, DUST_TS_COMPOSITE_END);
    let frame_ms = span(DUST_TS_BEGIN, DUST_TS_MEASURED);
    let total_ms = frame_ms + if composite_ms.is_nan() { 0.0 } else { composite_ms };
    Self {
      propagate_ms,
      splat_ms,
      splat_only_ms,
      specks_ms,
      call_ms,
      queue_ms,
      cadence_ms,
      builds_ms,
      reproject_ms,
      classify_ms,
      pair_build_ms,
      measure_ms,
      composite_ms,
      total_ms,
      total_ema_ms: total_ms,
      frame: 0,
      width: 0,
      height: 0,
      reused: false,
    }
  }
}
/// alias kept for the dust trace
pub const LOD_READBACK_SLOTS: usize = DUST_READBACK_SLOTS;

/// `AETHERVK_DUST_SPECKS=1` turns the speck shell on (off by default) (the pass still measures the density at
/// the eye); on by default.
fn specks_enabled() -> bool {
  // off by default until the speck pass has a GPU parity test and an observer validation
  // (2026-10-11): `AETHERVK_DUST_SPECKS=1` turns it on
  aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPECKS").is_some_and(|v| v.trim() == "1")
}

/// `AETHERVK_DUST_SPECKS_BUDGET=<n>`: specks per frame, `specks::DUST_SPECKS_BUDGET` by default.
fn specks_budget() -> u32 {
  aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPECKS_BUDGET")
    .and_then(|v| v.trim().parse::<u32>().ok())
    .unwrap_or(specks::DUST_SPECKS_BUDGET)
}

/// Memory of a dust buffer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum DustMem {
  Device,
  /// host writes, device reads
  HostWrite,
  /// device writes (transfer), host reads
  HostRead,
}

pub struct DustBuffer {
  pub buffer: vk::Buffer,
  pub alloc: vk_mem::Allocation,
  pub address: u64,
}

/// Per-frame host state of one system (the dust trace ring)
pub struct DustSystemHost {
  pub frame: u64,
  /// this frame's view aids (`dust::DUST_VIEW_*`) and flow uniform, set by `cmd_dust_system_begin`
  pub view_flags: u32,
  pub flow: dust::DustFlowUniform,
  slot_valid: [bool; DUST_READBACK_SLOTS],
}

pub struct DustGpuSystem {
  pub capacity: u32,
  pub clusters: Option<DustBuffer>,
  pub render: Option<DustBuffer>,
  pub moments: Option<DustBuffer>,
  pub batches: Option<DustBuffer>,
  /// the propagate's solve rows (round 9 step O): one chunk of `DUST_PROP_CHUNK_WGS` workgroups,
  /// 64 rows of `DUST_PROP_ROW_WORDS` words each; the chunks are serialised by barriers
  pub rows: Option<DustBuffer>,
  batches_mapped: *mut u8,
  pub host: spin::Mutex<DustSystemHost>,
  /// dust trace (`gpu::dust_trace`): host-visible copies of the traced render clusters per
  /// readback slot and tier, created on the first traced frame
  trace_rb: spin::Mutex<Option<(DustBuffer, *mut u8)>>,
  /// what each readback slot's traced frame drew with (the line is written when it is read back)
  trace_meta: spin::Mutex<[Option<crate::gpu::dust_trace::TraceFrameMeta>; DUST_READBACK_SLOTS]>,
  /// the inputs of each tier's last recorded propagate (keyed by ring base;
  /// [`Device::cmd_dust_propagate_cached`]): equal inputs, equal output, so the dispatch is skipped
  propagated: spin::Mutex<alloc::collections::BTreeMap<u32, PropagateKey>>,
  /// each tier's propagate generation (keyed by ring base): the inputs it was last bumped for
  /// and the counter, the stamp `dust_propagate.comp` writes to `edges[12].w`
  propagate_gen: spin::Mutex<alloc::collections::BTreeMap<u32, (PropagateGenKey, u32)>>,
  /// the propagate's skip flag (a test hook, [`Device::dust_set_propagate_skip`]): a slot stamped
  /// with the current generation is left as it is
  propagate_skip: core::sync::atomic::AtomicBool,
  /// CPU particle mode ring
  pub cpu_ring: Option<spin::RwLock<alloc::vec::Vec<DustCluster>>>,
  /// CPU particle mode: this frame's evaluated clusters and moments per tier (keyed by ring base)
  cpu_render: spin::Mutex<
    alloc::collections::BTreeMap<
      u32,
      (
        alloc::vec::Vec<DustRenderCluster>,
        alloc::vec::Vec<DustMoments>,
      ),
    >,
  >,
}

impl DustGpuSystem {
  fn empty(capacity: u32) -> Self {
    Self {
      capacity,
      clusters: None,
      render: None,
      moments: None,
      batches: None,
      rows: None,
      batches_mapped: core::ptr::null_mut(),
      host: spin::Mutex::new(DustSystemHost {
        frame: 0,
        view_flags: 0,
        flow: Default::default(),
        slot_valid: [false; DUST_READBACK_SLOTS],
      }),
      trace_rb: spin::Mutex::new(None),
      trace_meta: spin::Mutex::new([const { None }; DUST_READBACK_SLOTS]),
      propagated: spin::Mutex::new(alloc::collections::BTreeMap::new()),
      propagate_gen: spin::Mutex::new(alloc::collections::BTreeMap::new()),
      propagate_skip: core::sync::atomic::AtomicBool::new(false),
      cpu_ring: None,
      cpu_render: spin::Mutex::new(alloc::collections::BTreeMap::new()),
    }
  }
}

/// A frame header as [`Device::cmd_dust_frame_begin_reusable`] compares it: the words the
/// propagates, splats and specks do not read zeroed — the white point and softening (the
/// composite's), and the tracer counts unless the tracers are drawn (`flags`).
pub fn reuse_header(header: &[u32], flags: u32) -> alloc::vec::Vec<u32> {
  let mut h = header.to_vec();
  h[dust::PYR_WHITE as usize] = 0;
  h[dust::PYR_SOFTENING as usize] = 0;
  if flags & dust::DUST_VIEW_TRACERS == 0 {
    h[dust::PYR_TRACER_COUNTS as usize] = 0;
  }
  h
}

/// What a tier's propagate computes from ([`Device::cmd_dust_propagate_cached`]): the per-frame
/// constants, the drawn slot range, the output buffers and the newest emission the frame waits
/// on (`DustDrawState::compute_wait`: every emission or re-emission of a drawn batch is a later
/// timeline value, so the cluster ring under the range is unchanged while it is).
#[derive(Clone, Copy, PartialEq)]
struct PropagateKey {
  frame: DustFrame,
  capacity: u32,
  first_slot: u32,
  live_count: u32,
  render: u64,
  moments: u64,
  emitted: u64,
  /// the size-polyline edge stride (the trajectory detail): a changed knob re-propagates
  stride: u32,
}

/// The inputs a tier's propagate generation follows ([`PropagateKey`]'s notion without the
/// buffers): the generation is bumped whenever they differ from the last propagate's. `emitted`
/// is `None` for an uncached propagate (no emission timeline known).
#[derive(Clone, Copy, PartialEq)]
struct PropagateGenKey {
  frame: DustFrame,
  capacity: u32,
  first_slot: u32,
  live_count: u32,
  emitted: Option<u64>,
  /// the edge stride: a changed knob bumps the generation (the cadence's stamp skip refreshes)
  stride: u32,
}

/// bytes of the traced clusters of one readback slot (all tiers)
pub const TRACE_SLOT_BYTES: u64 = dust::LOD_MAX_TIERS as u64
  * crate::gpu::dust_trace::TRACE_SAMPLES as u64
  * core::mem::size_of::<DustRenderCluster>() as u64;

// SAFETY: `batches_mapped` points into a persistently mapped VMA allocation owned by this struct
unsafe impl Send for DustGpuSystem {}
unsafe impl Sync for DustGpuSystem {}

/// Host side of one view's splat pyramid: the white point feedback (`dust::white_point_from_level`),
/// the auto softening and the fixed-point unit.
pub struct DustViewHost {
  /// the pyramid's last fully drawn frame ([`Device::cmd_dust_frame_begin_reusable`]): the
  /// caller's fingerprint of its draw inputs and its header with the display-only words zeroed
  /// ([`reuse_header`]); `None` after a frame that may not have drawn completely
  pub drawn: Option<(u64, alloc::vec::Vec<u32>)>,
  /// white point (radiance factor per px²), seeded at `DUST_WHITE_SEED_MAG` until measured
  pub white: f32,
  /// median dust texel of the last measured frame (same units), 0 = none
  pub p50: f32,
  /// the white point and median of the 1/ρ-normalised field (`dust::rho_normalised_levels`,
  /// `dust::DUST_VIEW_RHO_NORM`), adapted from every readback whose frame knew its nucleus pixel
  /// alongside the plain pair (so a toggle of the aid swaps settled whites, never blends them)
  pub white_norm: f32,
  pub p50_norm: f32,
  /// the view flags of the frame begun last ([`DustViewHost::display`] picks the white by them)
  pub view_flags: u32,
  /// the nucleus' framebuffer pixel of the frame about to begin ([`Device::dust_set_nucleus_px`];
  /// NaN = unknown) and the one of each readback slot's frame
  nucleus_px_next: [f32; 2],
  /// the coma 1/ρ normalisation is shown (`DUST_VIEW_RHO_NORM`): a display-only option, set
  /// with the nucleus px because the frame flags reaching `frame_begin` have it masked out
  rho_norm: bool,
  slot_nucleus_px: [[f32; 2]; DUST_READBACK_SLOTS],
  /// the white point last logged (`[Dust exposure]`, on a quarter-stop move)
  white_logged: f32,
  /// the readback being absorbed: (its regime, it reused the previous image), for that log
  absorb_tag: (u8, bool),
  /// the readback being absorbed shows a refined image (`dust::dust_refine_factor`): its EWA work
  /// is over the budget by design and does not move the EWA bias
  pub absorb_refined: bool,
  /// static refinement: EWA bias levels below the feedback's this view draws at while its draw
  /// inputs stay unchanged (0 = the budget's bias)
  pub refine: u32,
  /// frames since the inputs last changed or the refinement last stepped
  refine_static: u32,
  /// the warm-up after a reset ([`Self::reset_feedback`]), per side (volume, EWA): from its
  /// coarsest level the side lowers while the remembered step keeps the lower level under the
  /// budget, not only below the band (which stopped the descent a level coarser than a cold
  /// start's ascent: EWA 4 against the live session's 3), by [`warmup_lower`] levels at once
  /// (the raise's rule mirrored: a readback lags `DUST_READBACK_SLOTS` frames); the first
  /// readback that does not lower ends it
  warmup: [bool; 2],
  /// the play pin (handoff P3b, [`Self::play_pin_enter`]): the biases every frame begins with
  /// while the simulation runs with the play quality on, over whatever the feedback absorbed
  pub play: Option<dust::play_quality::PinnedFeedback>,
  /// the paused feedback state saved at the pin's entry, restored at its exit
  /// ([`Self::play_pin_leave`]): the paused path never sees what the play frames taught
  play_saved: Option<SavedFeedback>,
  /// the newest readback's `(EWA bias, texels)`: what the image the view now shows cost
  pub refine_seen: Option<(u32, u64)>,
  /// the last drawn image was refined (a reused frame shows it)
  image_refined: bool,
  /// per readback slot: that frame showed a refined image
  slot_refined: [bool; DUST_READBACK_SLOTS],
  /// largest packet peak optical depth per px² of the last measured frame (raw exposure units,
  /// `dust::PYR_TAU_MAX`): the fixed-point unit follows it, so no view measures as empty
  pub tau_max: f32,
  pub frame: u64,
  /// per readback slot: the unit when it was recorded
  slot_unit: [f32; DUST_READBACK_SLOTS],
  slot_valid: [bool; DUST_READBACK_SLOTS],
  /// per readback slot: the haze detail of the frame's flags (round 9 step B), and the one of the
  /// readback being absorbed ([`DustViewHost::absorb`]: the haze engaged at a top-only bias)
  slot_haze: [u32; DUST_READBACK_SLOTS],
  pub readback_haze: u32,
  /// the view's pixel solid angle (sr; 0 = orthographic or unknown: no speck shell), set by
  /// [`Device::dust_set_shell_camera`] every frame
  pub omega_px_sr: f64,
  /// the speck shell of the last measured frame (`specks::ShellUpdate`): the density at the eye,
  /// the specks drawn / overflowed and the radius the next frame draws with
  pub shell: ShellUpdate,
  /// a fixed shell radius (local m) that replaces the measured one, whatever
  /// `AETHERVK_DUST_SPECKS` says ([`Device::dust_set_shell_override`]; tests, diagnostics)
  pub shell_override: Option<f32>,
  /// the packet split's depth limit the next frame draws with (`PYR_SPLIT_DEPTH`): lowered by
  /// `log2(children / budget)` levels when a measured frame made more children than
  /// `DUST_SPLIT_BUDGET` (each level at most doubles them), raised one level when it made fewer
  /// than a quarter; `AETHERVK_DUST_SPLIT_DEPTH` caps it (diagnostics)
  pub split_depth: u32,
  /// the smallest split depth seen over the budget (`u32::MAX`: none): the depth is never raised
  /// to it again (at depth 0 a frame makes no children and says nothing about depth 1, which
  /// made the feedback oscillate 0 ↔ 1 with 11 M children every other frame)
  pub split_depth_over: u32,
  /// the volumetric packets' level bias the next frame draws with (`PYR_SPLIT_DEPTH` bits 8–15):
  /// one coarser when a measured frame spent more than `DUST_VOLUME_FRAME_BUDGET` ray-integral
  /// evaluations, one finer below an eighth of it (`VolumeStats::suggested_bias`)
  pub volume_bias: u32,
  /// the EWA packets' bias the next frame draws with (`PYR_SPLIT_DEPTH` bits 18–21: the
  /// per-packet texel cap is `DUST_SPLAT_MAX_TEXELS / 4^bias`), from the texels a measured frame
  /// planned (`EwaStats::suggested_bias`)
  pub ewa_bias: u32,
  /// the feedback's memory of the active regime (only its `*_step_q8` / `*_prev` are read; the
  /// bias fields are the ones above)
  pub fb_memory: FeedbackSet,
  /// per readback slot: the timestamps written that frame (bit i = [`DUST_TS_BEGIN`] + i)
  ts_written: [u32; DUST_READBACK_SLOTS],
  /// per readback slot: its frame number when recorded
  ts_frame: [u64; DUST_READBACK_SLOTS],
  /// the slot this frame's timestamps go to, `None` before a frame begin reset it
  ts_slot: Option<usize>,
  /// the last measured frame's GPU timings ([`DustGpuTimings`])
  pub timings: Option<DustGpuTimings>,
  /// the frame recorded since its begin has a packet queue to draw at its end (a GPU frame that
  /// drew; not a reused one, whose queue header still holds the last frame's records)
  queue_pending: bool,
  /// the queue's [`dust::PYR_FLAGS`] bits ([`dust::DUST_FRAME_QUEUE_OFF`] / `_ALL`; a test hook:
  /// [`Device::dust_set_queue_mode`]), else from `AETHERVK_DUST_QUEUE`
  pub queue_mode: Option<u32>,
  /// a capacity below [`dust::DUST_QUEUE_CAPACITY`] (a test hook; 0 = the default)
  pub queue_capacity: u32,
  /// the packet queue of the last frame read back: (records drawn, packets appended)
  pub queue_counts: (u32, u32),
  /// the volume packet queue ([`dust::DUST_VQUEUE_*`]): a test's mode override
  /// (`Some(dust::DUST_FRAME_VQUEUE_OFF)`), else `AETHERVK_DUST_VQUEUE=0`; whether this frame
  /// draws with it off; a capacity below [`dust::DUST_VQUEUE_CAPACITY`] (test hook; 0 = default);
  /// a forced chunk in clusters (`AETHERVK_DUST_VQUEUE_CHUNK` or a test hook; 0 = from the
  /// estimate); the packets per cluster estimate (an EMA of the frames read back: the next
  /// frame's chunk is `capacity / (2 × ppc)`); the frame's chunks and clusters so far; per
  /// readback slot the clusters drawn that frame; the last frame read back: (records drawn by the
  /// consumer from the main queue, overflowed (sent to the volume spill), chunks); the volume
  /// spill's capacity per chunk (test hook; 0 = the view's [`dust::dust_vspill_records`]); the
  /// last frame read back's spill: (records drawn from it, records lost: refused by it too)
  pub vqueue_mode: Option<u32>,
  pub vqueue_off: bool,
  /// the plain consumer blob instead of the `.sg` one (test hook, round 8 step F2)
  pub volume_plain: bool,
  pub vqueue_capacity: u32,
  pub vspill_capacity: u32,
  pub vspill_counts: (u32, u32),
  pub vqueue_chunk: u32,
  pub vqueue_ppc: f32,
  /// the splitter family's chunk feedback (perf9 steps Z / O): the leaves and the candidates per
  /// cluster (EMAs of the leaf queue header's frame-wide words 8 / 9 of the frames read back);
  /// the auto chunk fits the leaf queue and the candidate buffer twice over (`dust_split_auto_chunk`)
  pub lqueue_lpc: f32,
  pub cqueue_cpc: f32,
  pub vqueue_chunks_frame: u32,
  pub vqueue_clusters_frame: u32,
  slot_clusters: [u32; DUST_READBACK_SLOTS],
  slot_chunks: [u32; DUST_READBACK_SLOTS],
  pub vqueue_counts: (u32, u32, u32),
  /// the last measured frame that drew (not reused): what a changing view costs
  pub timings_drawn: Option<DustGpuTimings>,
  /// per readback slot: that frame reused the previous image
  ts_reused: [bool; DUST_READBACK_SLOTS],
  /// the frames draw the legacy kernel (`dust::DUST_FRAME_LEGACY_KERNEL_BIT`;
  /// [`Device::dust_set_legacy_kernel`], or `AETHERVK_DUST_LEGACY_KERNEL=1` for every view)
  pub legacy_kernel: bool,
  /// the frames draw deep chords by the EWA (`dust::DUST_FRAME_DEEP_CHORD`;
  /// [`Device::dust_set_deep_chord`], else `AETHERVK_DUST_DEEP_CHORD=1`; off by default)
  pub deep_chord: Option<bool>,
  /// the cluster level of detail the frames draw with (`dust::PYR_LOD`, pixels per drawn segment,
  /// 0 = off): a test's override ([`Device::dust_set_lod_px`]); else `AETHERVK_DUST_LOD_PX`, else
  /// [`dust::DUST_LOD_PX_DEFAULT`]
  pub lod_px: Option<f32>,
  /// a fixed volumetric level bias (a test hook, `Device::dust_set_volume_bias`; `None`: the
  /// feedback's `volume_bias`, or `AETHERVK_DUST_VOLUME_BIAS`)
  pub volume_bias_override: Option<u32>,
  /// the walk pass records every speck it draws ([`Device::dust_set_specks_trace`],
  /// `dust::DUST_SPECKS_TRACE_*`; a test hook)
  pub specks_trace: bool,
  /// the walk's stack limit in nodes (spill header word 8; a test hook forcing the spill,
  /// [`Device::dust_set_specks_stack_limit`]; 0 = the whole `DUST_SPECKS_WG_STACK`)
  pub specks_stack_limit: u32,
  /// the in-frame gates off for this view (a test hook, [`Device::dust_set_frame_gates`])
  pub frame_gates_off: bool,
  /// the budget loop (temporal reuse step 5, `dust::cadence::BudgetLoop`): `Some` while the view
  /// runs the cadence and `AETHERVK_DUST_BUDGET_LOOP` is not 0
  pub budget: Option<dust::cadence::BudgetLoop>,
  /// the feedback's two regimes ([`FeedbackSet`]): full draws and cadenced frames
  pub fb_full: FeedbackSet,
  pub fb_cad: Option<FeedbackSet>,
  /// per readback slot: the frame's regime (0 full draw, 1 cadenced) and whether its target
  /// probe was recorded ([`Device::cmd_dust_budget_probe`])
  slot_regime: [u8; DUST_READBACK_SLOTS],
  slot_probe: [bool; DUST_READBACK_SLOTS],
  /// the cadenced frames' build work per member ([`full_frame_estimate`]; `None` after a full draw)
  build_rate: Option<BuildRate>,
  /// the last readback's regime and its counters by target
  pub last_regime: u8,
  pub targets: Option<TargetCounters>,
  /// full-draw frames whose biases were restored ([`Device::dust_budget_frame_kind`])
  pub restored: u32,
  /// the scene's dust feedback generation this host's feedback was measured under
  /// (`SceneContext::dust_feedback_gen`, [`Device::dust_sync_feedback_gen`])
  pub feedback_gen: u32,
}

/// The paused feedback state the play pin saves ([`DustViewHost::play_pin_enter`]) and restores
/// ([`DustViewHost::play_pin_leave`]): the fields, the memory, both regimes and the warm-up
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SavedFeedback {
  split_depth: u32,
  split_depth_over: u32,
  volume_bias: u32,
  ewa_bias: u32,
  fb_memory: FeedbackSet,
  fb_full: FeedbackSet,
  fb_cad: Option<FeedbackSet>,
  warmup: [bool; 2],
}

/// One regime's feedback state ([`DustViewHost::absorb`]: the split depth and its over-budget
/// memory, the volumetric and EWA biases). The budget loop keeps two (step 5, 2026-10-18): full
/// draws (the cadence off, a cadence reset or first frame) and cadenced frames, so the cheap
/// cadenced frames never lower the biases a full draw begins with (5 000 km: a reset frame with
/// the cadenced EWA bias 3 drew 2.3e8 EWA texels, 276 ms, against 7.6e7 at bias 5, 111 ms).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeedbackSet {
  pub split_depth: u32,
  pub split_depth_over: u32,
  pub volume_bias: u32,
  pub ewa_bias: u32,
  /// the feedback's memory (round 8 step B): per bias, the work ratio of one step measured at
  /// the last raise into it (8.8 fixed point, 0 = unknown), and the previous readback's (bias,
  /// work) that measures the next raise. A lower is refused while `work × ratio` would pass the
  /// budget: one volume step changes the work up to ~13×, one EWA step up to 25×, both past the
  /// band's 8× (5000 km, 1 d/s, volume detail 1: bias 10 ↔ 11 every four frames, half of them at
  /// the dearer bias, 212 ms against 177)
  pub volume_step_q8: [u32; dust::DUST_VOLUME_BIAS_MAX as usize + 1],
  pub ewa_step_q8: [u32; dust::DUST_EWA_BIAS_MAX as usize + 1],
  pub volume_prev: (u32, u64),
  pub ewa_prev: (u32, u64),
  /// the top plateau's edge (round 9 step B): the work ratio of the raise that engaged the haze
  /// (the last sub-top readback's work over the first top-only one's, 8.8 fixed point, 0 =
  /// unknown), kept apart from the per-bias steps: a raise of k levels into the plateau would
  /// spread the one-off haze-off ↔ haze-on jump (45–200×) over k steps as its k-th root, and the
  /// lower rule then saw `work × 6.7 < budget` and left the plateau, the haze disengaged, the
  /// evaluations jumped back over the budget: a four-readback cycle
  pub volume_edge_q8: u32,
  /// consecutive top-only readbacks under budget / 8 with the haze engaged (the hold before a
  /// lower off the plateau, [`DUST_FEEDBACK_TOP_HOLD`])
  pub top_under: u32,
}

impl Default for FeedbackSet {
  fn default() -> Self {
    Self {
      split_depth: dust::DUST_SPLIT_DEPTH_MAX,
      split_depth_over: u32::MAX,
      volume_bias: 0,
      ewa_bias: 0,
      volume_step_q8: [0; dust::DUST_VOLUME_BIAS_MAX as usize + 1],
      ewa_step_q8: [0; dust::DUST_EWA_BIAS_MAX as usize + 1],
      volume_prev: (0, 0),
      ewa_prev: (0, 0),
      volume_edge_q8: 0,
      top_under: 0,
    }
  }
}

/// The plateau edge the feedback assumes before it measured one (the haze engaged, no raise seen):
/// a lower off the plateau is refused while `work × 64 > budget`
pub const DUST_FEEDBACK_HAZE_EDGE_DEFAULT: f64 = 64.0;
/// How many consecutive under-budget/8 top-only readbacks the feedback holds before it lowers off
/// the plateau with the haze engaged (the readbacks arrive [`DUST_READBACK_SLOTS`] frames late)
pub const DUST_FEEDBACK_TOP_HOLD: u32 = 8;

/// The step ratio the feedback assumes before a raise measured one: a level quarters the work
/// (the log₄ rule of the raises); with the band's lower edge at budget / 8 it keeps today's rule
pub const DUST_FEEDBACK_STEP_DEFAULT: f64 = 4.0;

impl FeedbackSet {
  /// The warm-up start of a reset ([`DustViewHost::reset_feedback`]): the cheapest state (no
  /// split, the coarsest volume bias and feedback EWA bias), memory empty. The feedback lowers
  /// from there to the view's steady state within a few readbacks, so a reset never draws a cold
  /// frame at the finest levels (3 s at 1 km on the GTX 1070, past Windows' 2 s TDR). A new view
  /// still starts cold ([`FeedbackSet::default`]).
  pub fn coarse() -> Self {
    Self {
      split_depth: 0,
      volume_bias: dust::DUST_VOLUME_BIAS_MAX,
      ewa_bias: dust::DUST_EWA_BIAS_FEEDBACK_MAX,
      ..Self::default()
    }
  }

  /// Folds a readback drawn at `bias` with `work` into the memory of `step_q8` / `prev`: the
  /// readback after a raise of `k` steps records the per-step ratio (geometric) at the new bias.
  fn feedback_remember(step_q8: &mut [u32], prev: &mut (u32, u64), bias: u32, work: u64) {
    let (pb, pw) = *prev;
    if bias > pb && pw > 0 && work > 0 && (bias as usize) < step_q8.len() {
      let k = (bias - pb) as f64;
      let r = (pw as f64 / work as f64).powf(1.0 / k);
      step_q8[bias as usize] = (r * 256.0).clamp(256.0, u32::MAX as f64) as u32;
    }
    *prev = (bias, work);
  }

  /// Whether the work measured at `bias` may lower it: `work × ratio ≤ budget`, the ratio the
  /// memory's at `bias` or [`DUST_FEEDBACK_STEP_DEFAULT`]
  fn feedback_allows_lower(step_q8: &[u32], bias: u32, work: u64, budget: u64) -> bool {
    let r = match step_q8.get(bias as usize) {
      Some(&q) if q > 0 => q as f64 / 256.0,
      _ => DUST_FEEDBACK_STEP_DEFAULT,
    };
    work as f64 * r <= budget as f64
  }
}

/// A cadenced frame's work counters by target ([`DUST_PROBE_HEADER_WORDS`]: split children, EWA
/// texels in `DUST_EWA_TEXEL_UNIT`s, volume evaluations): the build draws' (the frame header copied
/// after each class's builds, `Device::cmd_dust_budget_probe`) and the fresh pass's (the frame's
/// total less them). The counters are frame-continuous (the builds' headers are routed through the
/// frame's), so the in-frame gates read the frame's total across both targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TargetCounters {
  pub builds: [u32; 3],
  /// the builds' counters per class (the differences of the class probes)
  pub by_class: [[u32; 3]; PROBE_CLASSES],
  pub fresh: [u32; 3],
}

/// the classes the target probe holds apart (`DUST_CADENCE_MAX_CLASSES`)
pub const PROBE_CLASSES: usize = crate::scene::dust::cadence::DUST_CADENCE_MAX_CLASSES;
/// words per class of the target probe
pub const DUST_PROBE_WORDS: u32 = 4;
/// words per readback slot of the target probe (one class probe after another), appended after the
/// readback slots
pub const DUST_PROBE_SLOT_WORDS: u32 = DUST_PROBE_WORDS * PROBE_CLASSES as u32;
/// the header words the target probe copies: split children, EWA texels, volume evaluations
pub const DUST_PROBE_HEADER_WORDS: [u32; 3] = [dust::PYR_SPLIT_CHILDREN, dust::PYR_EWA_TEXELS, dust::PYR_VOLUME_EVALS];

/// A cadenced frame's counters ([`DUST_PROBE_HEADER_WORDS`]) as a full draw's estimate: the fresh
/// pass's as measured plus, per class, the builds' scaled by `shown / built` (the class's shown
/// members over the members this frame drew into its build images: each shown member was drawn
/// once, at its build, at about the rate of its class's build members). Per class because the
/// classes sample their members at different rates (class k builds each once per N_k frames): a
/// pooled `Σ shown / Σ built` weighs the faster classes' members N_slow / N_fast times too much
/// (round 7, 1 km orbit after the handover: 4 % low, the EWA 1 → 0 switch at f96 against the full
/// draw's f144). A class with shown members and no builds in the window takes the pooled rate.
///
/// The scale `shown / built` is capped at [`DUST_ESTIMATE_SCALE_SLACK`] `· N_k / w` (`w` the
/// frames in the class's window, `N_k` its cycle): a full window builds every list position once
/// per phase, the shown images' members, so its scale is ≈ 1, and a filling one ≈ `N_k / w`. A
/// larger scale means the shown images hold another population than the builds (the handover:
/// class 8's old image of 86 416 ordinary members shown while its builds hold the 6 320 members
/// it keeps, 34 of them the bright young clusters at ~80 EWA units each). Uncapped, that rate
/// times the old image's count put a 4.8e8-texel estimate on a 2.3e7-texel frame (1 km still,
/// τ 0.004, f72), and the EWA bias 2 it set blurred the near-nucleus packets (L1 2.63 %, p99 56 %).
/// A cadenced frame draws a fraction of
/// a full draw's work, and its feedback on the raw counters walks the biases finer than a full
/// draw ever draws: 1 km, still camera, the volume bias 14 → 8 over 60 frames, the class images
/// built at bias 8–10 put 5 % of the light on levels 9–10 the full draw leaves empty (light ratio
/// 1.057). `built == 0`: the raw counters (no build this frame says nothing about the images).
///
/// The builds' rate is a ratio of sums over each class's last cycle ([`BuildRate`]): one frame's
/// ratio swings with its cursors (a cycle's last step is short, a swap frame builds nothing, and
/// the lists' order bunches the costly members: class 16 at 1 km orbit built 9 580 members every
/// frame at 0 to 179 k EWA units), and one 4.3e8-texel estimate at 1 km kicked the EWA bias off
/// the full draw's.
pub fn full_frame_estimate(fresh: [u32; 3], rate: &BuildRate, shown: [u32; PROBE_CLASSES]) -> [u32; 3] {
  let built_all: f64 = rate.built.iter().sum();
  core::array::from_fn(|i| {
    let builds_all: f64 = rate.builds.iter().map(|c| c[i]).sum();
    let b: f64 = (0..PROBE_CLASSES)
      .map(|k| {
        if rate.built[k] > 0.0 {
          let w = rate.window[k].len().max(1) as f64;
          let cap = DUST_ESTIMATE_SCALE_SLACK * rate.frames[k].max(1) as f64 / w;
          rate.builds[k][i] * (shown[k] as f64 / rate.built[k]).min(cap)
        } else if built_all > 0.0 {
          builds_all * shown[k] as f64 / built_all
        } else {
          0.0
        }
      })
      .sum();
    (fresh[i] as f64 + b).min(u32::MAX as f64) as u32
  })
}

/// The cap on [`full_frame_estimate`]'s scale `shown / built` over the one a class's window
/// expects (`N_k / w`): the lists' sizes drift a little within a cycle (1 km still: 1.001 and
/// 0.998), a population mismatch is 10× and more
pub const DUST_ESTIMATE_SCALE_SLACK: f64 = 1.25;

/// The cadenced frames' build work per member built, per class: sums over the class's last `N_k`
/// cadenced frames (one cycle of its lists, where every list position is built once per phase, so
/// the lists' bunching of costly members cancels) of the builds' counters and of the members they
/// drew. A decayed sum over 8 frames held half of class 16's cycle and swung 2× with its order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BuildRate {
  /// the window sums
  pub builds: [[f64; 3]; PROBE_CLASSES],
  pub built: [f64; PROBE_CLASSES],
  window: [alloc::collections::VecDeque<([u32; 3], u32)>; PROBE_CLASSES],
  /// each class's window length `N_k` (the last fold's)
  frames: [usize; PROBE_CLASSES],
}

impl BuildRate {
  /// one cadenced frame folded in: its builds' counters and the members they drew, per class;
  /// `frames[k]` is class k's window (its length `N_k`, at least 1)
  pub fn fold(&mut self, builds: [[u32; 3]; PROBE_CLASSES], built: [u32; PROBE_CLASSES], frames: [usize; PROBE_CLASSES]) {
    self.frames = frames;
    for k in 0..PROBE_CLASSES {
      let w = &mut self.window[k];
      w.push_back((builds[k], built[k]));
      while w.len() > frames[k].max(1) {
        w.pop_front();
      }
      // re-summed (≤ 64 entries): no drift from adding and subtracting floats
      self.builds[k] = core::array::from_fn(|i| w.iter().map(|e| e.0[i] as f64).sum());
      self.built[k] = w.iter().map(|e| e.1 as f64).sum();
    }
  }
}

/// The budget loop's telemetry of a view ([`Device::dust_budget_telemetry`]): the knobs, the
/// loop's state, the feedback regimes, the counters by target and the cadence's own telemetry
#[derive(Debug, Clone, Default)]
pub struct DustBudgetTelemetry {
  pub budget_ms: f64,
  /// `None`: the loop is off (no cadence, or `AETHERVK_DUST_BUDGET_LOOP=0`)
  pub knobs: Option<dust::cadence::BudgetKnobs>,
  pub rung: usize,
  pub rungs: usize,
  pub state: &'static str,
  pub ratio_ema: f64,
  pub moves: u32,
  pub k_target: Option<u32>,
  pub shell_cap_m: Option<f32>,
  /// the class lengths in use and the staleness bound `2 · N_max` (frames)
  pub classes: alloc::vec::Vec<u32>,
  pub staleness_frames: u32,
  /// the last readback's regime (0 full draw, 1 cadenced) and the full-draw frames whose biases
  /// were restored
  pub regime: u8,
  pub restored: u32,
  pub fb_full: FeedbackSet,
  pub fb_cad: Option<FeedbackSet>,
  pub targets: Option<TargetCounters>,
  pub cadence: Option<cadence::CadenceTelemetry>,
}

impl DustBudgetTelemetry {
  /// the observer's `dust GPU:` lines (no prefix)
  pub fn lines(&self) -> alloc::vec::Vec<alloc::string::String> {
    use alloc::format;
    let mut out = alloc::vec::Vec::new();
    let fbs = |f: &FeedbackSet| format!("[depth {}, volume {}, EWA {}]", f.split_depth, f.volume_bias, f.ewa_bias);
    let (fresh, live, frac) = self.cadence.as_ref().map_or((0, 0, 0.0), |c| (c.fresh, c.live, c.fresh_fraction));
    match self.knobs {
      Some(k) => out.push(format!(
        "dust GPU: budget {} ms: loop {} at rung {}/{}{}: ε {} px, lod {} px, budgets ÷{}, classes {:?} (staleness ≤ {} frames), ratio ema {:.2}, K {} vs fresh {} of {} ({:.3}), {} moves",
        self.budget_ms,
        self.state,
        self.rung,
        self.rungs.saturating_sub(1),
        if k.fallback { format!(", by-count fallback (N level {})", k.n_level) } else { alloc::string::String::new() },
        k.eps_px,
        k.lod_px,
        1u32 << k.budget_shift,
        self.classes,
        self.staleness_frames,
        self.ratio_ema,
        self.k_target.map_or("n/a".into(), |k| format!("{k}")),
        fresh,
        live,
        frac,
        self.moves
      )),
      None => out.push(alloc::string::String::from("dust GPU: budget loop off (no cadence, or AETHERVK_DUST_BUDGET_LOOP=0)")),
    }
    if let Some(c) = self.cadence.as_ref() {
      let t = c.timings;
      out.push(format!(
        "dust GPU: cadence: mode {:?}, reset {:?}, admitted {:?}, lists (k, p, [shown, build, pending]) {:?}, images {}, class hist fresh/N… {:?}, D3 {} at build / {} at classify, shown light {:.4}, ghosts {} ({:.4}), aborts rate/budget/prediction/coverage {:?}; ms: builds {} (per pair {}), classify {}, reproject {}, fresh splat {}",
        c.mode,
        c.reset,
        c.admitted,
        c.lists,
        c.images_shown,
        c.class_hist,
        c.d3_rejected,
        c.d3_at_classify,
        c.shown_light_share,
        c.ghost_entries,
        c.ghost_share,
        c.aborts,
        t.map_or("n/a".into(), |t| format!("{:.2}", t.builds_ms)),
        t.map_or("n/a".into(), |t| t.pair_build_ms.iter().map(|v| format!("{v:.2}")).collect::<alloc::vec::Vec<_>>().join(" / ")),
        t.map_or("n/a".into(), |t| format!("{:.2}", t.classify_ms)),
        t.map_or("n/a".into(), |t| format!("{:.2}", t.reproject_ms)),
        t.map_or("n/a".into(), |t| format!("{:.2}", t.splat_only_ms)),
      ));
    }
    out.push(format!(
      "dust GPU: feedback regime {} (full-draw biases restored {} times): full {}, cadenced {}; work by target (split children, EWA texel units, volume evals): {}; specks shell cap {}",
      if self.regime == 1 { "cadenced" } else { "full" },
      self.restored,
      fbs(&self.fb_full),
      self.fb_cad.as_ref().map_or("n/a".into(), fbs),
      self.targets.map_or("n/a".into(), |t| format!("builds {:?}, fresh {:?}", t.builds, t.fresh)),
      self.shell_cap_m.map_or("none".into(), |c| format!("{c:.3} m")),
    ));
    out
  }

  /// `stats.jsonl`'s `dust_budget`
  pub fn to_json(&self) -> alloc::string::String {
    use alloc::format;
    let num = |v: f64| if v.is_finite() { format!("{v}") } else { alloc::string::String::from("null") };
    let fb = |f: &FeedbackSet| format!("{{\"split_depth\":{},\"volume_bias\":{},\"ewa_bias\":{}}}", f.split_depth, f.volume_bias, f.ewa_bias);
    let knobs = self.knobs.map_or("null".into(), |k| {
      format!(
        "{{\"eps_px\":{},\"lod_px\":{},\"budget_shift\":{},\"fallback\":{},\"n_level\":{},\"saturated\":{}}}",
        num(k.eps_px as f64), num(k.lod_px as f64), k.budget_shift, k.fallback, k.n_level, k.saturated
      )
    });
    let cad = self.cadence.as_ref().map_or("null".into(), |c| {
      let t = c.timings;
      format!(
        "{{\"eps_px\":{},\"classes\":{:?},\"mode\":{},\"reset\":{},\"admitted\":{:?},\"timings_frame\":{},\"live\":{},\"fresh\":{},\"fresh_fraction\":{},\"images_shown\":{},\"class_hist\":{:?},\"d3_rejected\":{},\"d3_at_classify\":{},\"shown_light_share\":{},\"ghost_share\":{},\"aborts\":{:?},\"builds_ms\":{},\"pair_build_ms\":[{}],\"classify_ms\":{},\"reproject_ms\":{},\"fresh_splat_ms\":{}}}",
        num(c.eps_px as f64), c.classes,
        c.mode.map_or("null".into(), |m| format!("\"{m:?}\"")),
        c.reset.map_or("null".into(), |r| format!("\"{r:?}\"")),
        c.admitted,
        t.map_or("null".into(), |t| format!("{}", t.frame)),
        c.live, c.fresh, num(c.fresh_fraction as f64), c.images_shown, c.class_hist, c.d3_rejected, c.d3_at_classify,
        num(c.shown_light_share), num(c.ghost_share as f64), c.aborts,
        t.map_or("null".into(), |t| num(t.builds_ms)),
        t.map_or(alloc::string::String::new(), |t| t.pair_build_ms.iter().map(|&v| num(v)).collect::<alloc::vec::Vec<_>>().join(",")),
        t.map_or("null".into(), |t| num(t.classify_ms)),
        t.map_or("null".into(), |t| num(t.reproject_ms)),
        t.map_or("null".into(), |t| num(t.splat_only_ms)),
      )
    });
    format!(
      "{{\"budget_ms\":{},\"knobs\":{},\"rung\":{},\"rungs\":{},\"state\":\"{}\",\"ratio_ema\":{},\"moves\":{},\"k_target\":{},\"shell_cap_m\":{},\"classes\":{:?},\"staleness_frames\":{},\"regime\":{},\"restored\":{},\"fb_full\":{},\"fb_cad\":{},\"targets\":{},\"cadence\":{}}}",
      num(self.budget_ms),
      knobs,
      self.rung,
      self.rungs,
      self.state,
      num(self.ratio_ema),
      self.moves,
      self.k_target.map_or("null".into(), |k| format!("{k}")),
      self.shell_cap_m.map_or("null".into(), |c| num(c as f64)),
      self.classes,
      self.staleness_frames,
      self.regime,
      self.restored,
      fb(&self.fb_full),
      self.fb_cad.as_ref().map_or("null".into(), fb),
      self.targets.map_or("null".into(), |t| format!("{{\"builds\":{:?},\"fresh\":{:?}}}", t.builds, t.fresh)),
      cad,
    )
  }
}

impl DustViewHost {
  fn feedback_now(&self) -> FeedbackSet {
    FeedbackSet {
      split_depth: self.split_depth,
      split_depth_over: self.split_depth_over,
      volume_bias: self.volume_bias,
      ewa_bias: self.ewa_bias,
      volume_step_q8: self.fb_memory.volume_step_q8,
      ewa_step_q8: self.fb_memory.ewa_step_q8,
      volume_prev: self.fb_memory.volume_prev,
      ewa_prev: self.fb_memory.ewa_prev,
      volume_edge_q8: self.fb_memory.volume_edge_q8,
      top_under: self.fb_memory.top_under,
    }
  }

  /// the feedback fields ← regime `regime`'s state (1 cadenced: the full-draw one until a
  /// cadenced frame was measured, the conservative start; else the full-draw one). With the
  /// cadence off only regime 0 is ever loaded and stored: the identity on the fields (today's
  /// single feedback state).
  pub fn feedback_load(&mut self, regime: u8) {
    let s = if regime == 1 { self.fb_cad.unwrap_or(self.fb_full) } else { self.fb_full };
    self.split_depth = s.split_depth;
    self.split_depth_over = s.split_depth_over;
    self.volume_bias = s.volume_bias;
    self.ewa_bias = s.ewa_bias;
    self.fb_memory = s;
  }

  /// regime `regime`'s state ← the feedback fields
  pub fn feedback_store(&mut self, regime: u8) {
    let s = self.feedback_now();
    if regime == 1 {
      self.fb_cad = Some(s);
    } else {
      self.fb_full = s;
    }
  }

  /// the budget loop's shift of the EWA and volume feedback budgets (0 without the loop)
  pub fn budget_shift(&self) -> u32 {
    self.budget.as_ref().map_or(0, |b| b.knobs().budget_shift)
  }

  /// Forgets what the feedback learned: the split depth, the volume and EWA biases and their
  /// memory restart from the warm-up state ([`FeedbackSet::coarse`]: cheap frames, the feedback
  /// lowering the biases back to the steady state), the readbacks in flight are dropped and the
  /// next frame draws. After a discontinuity (a restored save, a seek past a day) the feedback
  /// otherwise keeps what a transient frame measured: the whole history appears in one tick, while
  /// the camera still moves to its restored pose, the EWA bias climbed 0 → 5 against the live
  /// session's 3, and the band's hysteresis kept it there (blocky dust after Open); its memory
  /// is what the reset drops. The white point, the unit and the queue estimates stay: they follow
  /// each readback.
  /// Whether the feedback absorbed a readback since the view's cold start or its last reset
  /// (the volume or EWA memory holds one): [`Device::dust_sync_feedback_gen`] resets only such a
  /// view, so a new view keeps its cold start ([`FeedbackSet::default`]).
  pub fn feedback_measured(&self) -> bool {
    self.fb_memory.volume_prev != (0, 0) || self.fb_memory.ewa_prev != (0, 0)
  }

  pub fn reset_feedback(&mut self) {
    let cold = FeedbackSet::coarse();
    self.split_depth = cold.split_depth;
    self.split_depth_over = cold.split_depth_over;
    self.volume_bias = cold.volume_bias;
    self.ewa_bias = cold.ewa_bias;
    self.fb_memory = cold;
    self.fb_full = cold;
    self.fb_cad = None;
    self.budget = None;
    self.build_rate = None;
    self.slot_valid = [false; DUST_READBACK_SLOTS];
    self.drawn = None;
    self.refine = 0;
    self.refine_static = 0;
    self.refine_seen = None;
    self.warmup = [true; 2];
    // under the pin: what the pause restores is this reset (a paused run would be at it too)
    if self.play_saved.is_some() {
      self.play_saved = Some(self.saved_feedback());
    }
  }

  /// the feedback state the play pin saves and restores
  fn saved_feedback(&self) -> SavedFeedback {
    SavedFeedback {
      split_depth: self.split_depth,
      split_depth_over: self.split_depth_over,
      volume_bias: self.volume_bias,
      ewa_bias: self.ewa_bias,
      fb_memory: self.fb_memory,
      fb_full: self.fb_full,
      fb_cad: self.fb_cad,
      warmup: self.warmup,
    }
  }

  /// Enters (or updates) the play pin: the first call saves the paused feedback state; every
  /// call sets the biases the next frames begin with ([`Self::apply_play_pin`]). Returns whether
  /// the pin was entered (the transition, for the log).
  pub fn play_pin_enter(&mut self, pin: dust::play_quality::PinnedFeedback) -> bool {
    let entered = self.play.is_none();
    if entered {
      self.play_saved = Some(self.saved_feedback());
    }
    self.play = Some(pin);
    entered
  }

  /// Leaves the play pin: the paused feedback state saved at the entry is restored, so the first
  /// paused frame draws with the biases the view had before the play (the readbacks of the play
  /// frames taught the paused state nothing). Returns whether a pin was left.
  pub fn play_pin_leave(&mut self) -> bool {
    let Some(s) = self.play_saved.take() else {
      self.play = None;
      return false;
    };
    self.play = None;
    self.split_depth = s.split_depth;
    self.split_depth_over = s.split_depth_over;
    self.volume_bias = s.volume_bias;
    self.ewa_bias = s.ewa_bias;
    self.fb_memory = s.fb_memory;
    self.fb_full = s.fb_full;
    self.fb_cad = s.fb_cad;
    self.warmup = s.warmup;
    true
  }

  /// The frame's feedback fields under the pin (after [`Self::feedback_load`]): the pinned
  /// biases; without a pin, unchanged
  pub fn apply_play_pin(&mut self) {
    if let Some(p) = self.play {
      self.split_depth = p.split_depth;
      self.volume_bias = p.volume_bias;
      self.ewa_bias = p.ewa_bias;
    }
  }

  /// [`FeedbackSet`] a full-draw frame's header is rewritten from ([`Device::dust_budget_frame_kind`]):
  /// the full regime's, its three biases pinned while the play pin is on
  pub fn full_frame_feedback(&self) -> FeedbackSet {
    match self.play {
      Some(p) => FeedbackSet { split_depth: p.split_depth, volume_bias: p.volume_bias, ewa_bias: p.ewa_bias, ..self.fb_full },
      None => self.fb_full,
    }
  }

  /// Static refinement's step for a frame about to begin: `same_inputs` (its draw inputs equal the
  /// previous frame's) and the knob `factor` (`dust::dust_refine_factor`, 0 = off). A changed
  /// frame returns to the budget's bias at once. After [`dust::DUST_REFINE_AFTER_FRAMES`] unchanged
  /// frames, once a readback of the image now shown is in, one level finer when that image's
  /// texels × 4 (a level's step, at most) stay within `factor` EWA budgets. Returns whether it
  /// stepped (the frame then draws).
  pub fn refine_update(&mut self, same_inputs: bool, factor: u32) -> bool {
    if !same_inputs || factor == 0 {
      self.refine = 0;
      self.refine_static = 0;
      return false;
    }
    self.refine_static = self.refine_static.saturating_add(1);
    let shown = self.ewa_bias.saturating_sub(self.refine);
    if self.refine_static < dust::DUST_REFINE_AFTER_FRAMES || shown == 0 {
      return false;
    }
    let budget = dust::DUST_EWA_FRAME_BUDGET as u64;
    match self.refine_seen {
      Some((bias, texels)) if bias == shown && texels.saturating_mul(4) <= factor as u64 * budget => {
        self.refine += 1;
        self.refine_static = 0;
        true
      }
      _ => false,
    }
  }
}

/// The warm-up's lower bias ([`DustViewHost::reset_feedback`]): from `bias`, whose frame did
/// `work` against `budget`, down by ⌊log₄(budget / work)⌋ levels (a level quadruples the work),
/// at least one (the caller checked the remembered step allows it).
fn warmup_lower(bias: u32, work: u64, budget: u64) -> u32 {
  let down = if work == 0 {
    bias
  } else {
    ((budget as f64 / work as f64).ln() / 4f64.ln()).floor().max(1.0) as u32
  };
  bias.saturating_sub(down)
}

/// how long the pyramid download (diagnostics, `download_dust_pyramid`) waits for the graphics
/// queue: the frames already in flight complete first, and a perspective frame inside the coma
/// takes seconds (5 s lost every dump of the near-field runs)
const DUST_DOWNLOAD_WAIT_NS: u64 = 60_000_000_000;

/// `AETHERVK_DUST_SPLIT_DEPTH=<n>`: caps the packet split's depth (0 = no split; diagnostics)
fn split_depth_cap() -> u32 {
  aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPLIT_DEPTH")
    .and_then(|v| v.trim().parse::<u32>().ok())
    .unwrap_or(dust::DUST_SPLIT_DEPTH_MAX)
    .min(dust::DUST_SPLIT_DEPTH_MAX)
}

impl Default for DustViewHost {
  fn default() -> Self {
    Self {
      white: dust::mag_arcsec2_to_white(dust::DUST_WHITE_SEED_MAG),
      p50: 0.0,
      white_norm: dust::mag_arcsec2_to_white(dust::DUST_WHITE_SEED_MAG),
      p50_norm: 0.0,
      view_flags: 0,
      nucleus_px_next: [f32::NAN; 2],
      rho_norm: false,
      slot_nucleus_px: [[f32::NAN; 2]; DUST_READBACK_SLOTS],
      white_logged: 0.0,
      absorb_tag: (0, false),
      absorb_refined: false,
      refine: 0,
      refine_static: 0,
      warmup: [false; 2],
      play: None,
      play_saved: None,
      refine_seen: None,
      image_refined: false,
      slot_refined: [false; DUST_READBACK_SLOTS],
      tau_max: 0.0,
      frame: 0,
      slot_unit: [0.0; DUST_READBACK_SLOTS],
      slot_valid: [false; DUST_READBACK_SLOTS],
      slot_haze: [dust::DUST_HAZE_DETAIL_DEFAULT; DUST_READBACK_SLOTS],
      readback_haze: dust::DUST_HAZE_DETAIL_DEFAULT,
      omega_px_sr: 0.0,
      shell: ShellUpdate {
        n_eye_m3: 0.0,
        specks: 0,
        overflow: 0,
        truncated: 0,
        nodes: 0,
        d_max_m: 0.0,
      },
      shell_override: None,
      split_depth: dust::DUST_SPLIT_DEPTH_MAX,
      split_depth_over: u32::MAX,
      volume_bias: 0,
      ewa_bias: 0,
      fb_memory: FeedbackSet::default(),
      ts_written: [0; DUST_READBACK_SLOTS],
      ts_frame: [0; DUST_READBACK_SLOTS],
      ts_slot: None,
      timings: None,
      timings_drawn: None,
      queue_counts: (0, 0),
      queue_pending: false,
      queue_mode: None,
      queue_capacity: 0,
      vqueue_mode: None,
      vqueue_off: false,
      volume_plain: false,
      vqueue_capacity: 0,
      vspill_capacity: 0,
      vspill_counts: (0, 0),
      vqueue_chunk: 0,
      vqueue_ppc: 64.0,
      lqueue_lpc: 64.0,
      // (a cold start sized for dense clusters: a refused candidate run loses its cluster)
      cqueue_cpc: 256.0,
      vqueue_chunks_frame: 0,
      vqueue_clusters_frame: 0,
      slot_clusters: [0; DUST_READBACK_SLOTS],
      slot_chunks: [0; DUST_READBACK_SLOTS],
      vqueue_counts: (0, 0, 0),
      ts_reused: [false; DUST_READBACK_SLOTS],
      legacy_kernel: false,
      deep_chord: None,
      lod_px: None,
      volume_bias_override: None,
      specks_trace: false,
      specks_stack_limit: 0,
      frame_gates_off: false,
      drawn: None,
      budget: None,
      fb_full: FeedbackSet::default(),
      fb_cad: None,
      slot_regime: [0; DUST_READBACK_SLOTS],
      slot_probe: [false; DUST_READBACK_SLOTS],
      build_rate: None,
      last_regime: 0,
      targets: None,
      restored: 0,
      feedback_gen: 0,
    }
  }
}

impl DustViewHost {
  /// fixed-point unit: exposure-scaled optical depth × px² per count, relative to the largest
  /// packet peak of the last measured frame (1 before any: the first frames may saturate or read
  /// empty, the maximum is measured regardless, so the unit is right from the next readback on)
  pub fn unit(&self) -> f32 {
    let m = if self.tau_max > 0.0 && self.tau_max.is_finite() {
      self.tau_max
    } else {
      1.0
    };
    m * dust::WHITE_TILE_UNIT_REL
  }

  /// [`DustViewHost::absorb_at`] of a frame whose nucleus pixel is unknown (the plain white only)
  pub fn absorb(&mut self, header: &[u32], level: &[u32], texel_px: u32, unit: f32) {
    self.absorb_at(header, level, 1, texel_px, unit, [f32::NAN; 2]);
  }

  /// Folds one frame's header and white-point level (recorded with `unit`; `level_w` texels
  /// wide) into the white point, the median and the unit; `nucleus_px` (the nucleus' framebuffer
  /// pixel when that frame was drawn, NaN = unknown) feeds the 1/ρ-normalised pair.
  pub fn absorb_at(&mut self, header: &[u32], level: &[u32], level_w: u32, texel_px: u32, unit: f32, nucleus_px: [f32; 2]) {
    // a flat field (`dust::level_is_flat`) measures no white point: the exposure keeps its own
    let flat = dust::level_is_flat(level);
    let measured = if flat {
      None
    } else {
      dust::white_point_from_level(level, texel_px, unit)
    };
    if !flat && nucleus_px[0].is_finite() && nucleus_px[1].is_finite() {
      if let Some((p99, p50)) = dust::rho_normalised_levels(level, level_w, texel_px, unit, Some(nucleus_px)) {
        self.white_norm = dust::adapt_white(self.white_norm, p99);
        self.p50_norm = dust::adapt_white(self.p50_norm, p50);
      }
    }
    if let Some((p99, p50)) = measured {
      self.white = dust::adapt_white(self.white, p99);
      self.p50 = dust::adapt_white(self.p50, p50);
      let moved = !(self.white_logged > 0.0)
        || (dust::exposure_ev(self.white) - dust::exposure_ev(self.white_logged)).abs() >= 0.25;
      if moved {
        self.white_logged = self.white;
        // the raw fixed-point counts behind the percentiles (p99, p50, max, lit texels)
        let counts = {
          let mut v: alloc::vec::Vec<u32> =
            level.chunks_exact(dust::PYRAMID_TEXEL_WORDS as usize).map(|t| t[0]).filter(|&c| c > 0).collect();
          v.sort_unstable();
          let at = |q: f32| v.get(((v.len().max(1) - 1) as f32 * q).round() as usize).copied().unwrap_or(0);
          (at(dust::WHITE_TILE_PERCENTILE), at(0.5), v.last().copied().unwrap_or(0), v.len())
        };
        aethervk_oshal_rlib::log!(
          "[Dust exposure] white {:.3e} ({:.2} mag/arcsec², EV {:+.2}), p50 {:.3e} (p50/white {:.3}), measured p99 {:.3e}, softening {:.3e} | counts p99 {} p50 {} max {} of {} lit texels, unit {:.3e}, texel {} px, tau_max {:.3e} | regime {} reused {}",
          self.white,
          dust::white_to_mag_arcsec2(self.white),
          dust::exposure_ev(self.white),
          self.p50,
          self.p50 / self.white,
          p99,
          dust::auto_softening(self.p50, self.white),
          counts.0,
          counts.1,
          counts.2,
          counts.3,
          unit,
          texel_px,
          f32::from_bits(header[dust::PYR_TAU_MAX as usize]),
          self.absorb_tag.0,
          self.absorb_tag.1
        );
      }
    }
    let v = f32::from_bits(header[dust::PYR_TAU_MAX as usize]);
    if v.is_finite() && v > 0.0 {
      self.tau_max = v;
    }
    // the packet split's depth for the next frames: down by log2(children / budget) levels when
    // over the budget (a level at most doubles the children), up one level below a quarter of it
    let split = dust::SplitStats::load(header);
    if split.children > dust::DUST_SPLIT_BUDGET {
      self.split_depth_over = self.split_depth_over.min(split.depth);
    }
    // (inside the band, or a raise refused, the state stays as it is, not the frame's own
    // depth: a frame read back was drawn DUST_READBACK_SLOTS frames ago, and writing its depth
    // back made the slots DUST_READBACK_SLOTS chains apart, each a fixed point of the band — 1 km,
    // cadenced: the EWA bias 0 on even frames, 1 on odd ones, for a hundred frames)
    let next = if split.children > dust::DUST_SPLIT_BUDGET {
      let ratio = split.children as f32 / dust::DUST_SPLIT_BUDGET as f32;
      let down = (<f32 as aethervk_oshal_rlib::math::FloatLike>::ln(ratio)
        / core::f32::consts::LN_2)
        .ceil() as u32;
      split.depth.saturating_sub(down.max(1))
    } else {
      let up = split.suggested_depth();
      if up == split.depth || up >= self.split_depth_over {
        self.split_depth
      } else {
        up
      }
    };
    if next != self.split_depth {
      aethervk_oshal_rlib::log!(
        "[Dust split] {} children at depth {} (budget {}): depth {} for the next frames",
        split.children,
        split.depth,
        dust::DUST_SPLIT_BUDGET,
        next
      );
    }
    self.split_depth = next;
    let vol = dust::VolumeStats::load(header);
    // the haze knob (round 9 step B): below 16 and without the frame's deep-chord bit it engages
    // at a top-only bias (`dust_haze_detail_of`), where it cuts the evaluations by the plateau's
    // edge; the plateau (every bias from `volume_top_min` on draws the same frame) is one state
    let haze_on = self.readback_haze < dust::DUST_HAZE_DETAIL_DEFAULT && !dust::deep_chord(header);
    let top = dust::volume_top_only_header(header, vol.bias);
    let engaged = haze_on && top;
    let (pb, pw) = self.fb_memory.volume_prev;
    if engaged && vol.bias > pb && pw > 0 && vol.evals > 0 && !dust::volume_top_only_header(header, pb) {
      // the raise that engaged the haze: the edge, apart from the per-bias steps
      let r = pw as f64 / vol.evals as f64;
      self.fb_memory.volume_edge_q8 = (r * 256.0).clamp(256.0, u32::MAX as f64) as u32;
      self.fb_memory.volume_prev = (vol.bias, vol.evals as u64);
    } else {
      FeedbackSet::feedback_remember(
        &mut self.fb_memory.volume_step_q8,
        &mut self.fb_memory.volume_prev,
        vol.bias,
        vol.evals as u64,
      );
    }
    // over the budget: coarser by log4(evals / budget) levels at once (a level divides the
    // texels by 4; inside the coma the first frame asks ~45× the budget, and one level per
    // readback took four slow frames to settle); else the one-step rule, unless the memory says
    // the step would pass the budget again
    // the budget loop's knob (step 5): the target halved `budget_shift` times (0: the constant)
    let vol_budget = dust::DUST_VOLUME_FRAME_BUDGET >> self.budget_shift();
    let edge = match self.fb_memory.volume_edge_q8 {
      0 => DUST_FEEDBACK_HAZE_EDGE_DEFAULT,
      q => q as f64 / 256.0,
    };
    let bias = if vol.evals > vol_budget {
      let ratio = vol.evals as f64 / vol_budget as f64;
      let up = (ratio.ln() / 4f64.ln()).ceil().max(1.0) as u32;
      let raised = (vol.bias + up).min(dust::DUST_VOLUME_BIAS_MAX);
      self.fb_memory.top_under = 0;
      // with the knob below 16 a raise onto the plateau lands on its lowest bias
      if haze_on && dust::volume_top_only_header(header, raised) {
        raised.min(dust::volume_top_min(header)).max(vol.bias)
      } else {
        raised
      }
    } else if engaged && vol.evals < vol_budget / 8 {
      // under the band on the plateau with the haze engaged: off the plateau (one step, where the
      // haze is off) only when the edge keeps the work under the budget, after the hold
      self.fb_memory.top_under += 1;
      if vol.evals as f64 * edge <= vol_budget as f64 && self.fb_memory.top_under >= DUST_FEEDBACK_TOP_HOLD {
        self.fb_memory.top_under = 0;
        dust::volume_top_min(header).saturating_sub(1).min(self.volume_bias)
      } else {
        self.volume_bias
      }
    } else if self.warmup[0]
      && vol.bias > 0
      && FeedbackSet::feedback_allows_lower(&self.fb_memory.volume_step_q8, vol.bias, vol.evals as u64, vol_budget as u64)
    {
      self.fb_memory.top_under = 0;
      warmup_lower(vol.bias, vol.evals as u64, vol_budget as u64).min(self.volume_bias)
    } else if vol.evals < vol_budget / 8
      && FeedbackSet::feedback_allows_lower(&self.fb_memory.volume_step_q8, vol.bias, vol.evals as u64, vol_budget as u64)
    {
      self.fb_memory.top_under = 0;
      vol.suggested_bias_for(vol_budget)
    } else {
      // inside the band: the state stays (as the split depth's)
      self.fb_memory.top_under = 0;
      self.volume_bias
    };
    // (the warm-up ends at its first readback that does not lower)
    self.warmup[0] &= bias < vol.bias;
    if bias != self.volume_bias {
      aethervk_oshal_rlib::log!(
        "[Dust volume] {} ray-integral evaluations at bias {} (budget {}): bias {} for the next frames (haze {}, top {}, edge ×{:.1})",
        vol.evals,
        vol.bias,
        vol_budget,
        bias,
        self.readback_haze,
        top,
        edge
      );
    }
    self.volume_bias = bias;
    let ewa = dust::EwaStats::load(header);
    self.refine_seen = Some((ewa.bias, ewa.texels));
    if self.absorb_refined {
      // a refined image (static refinement): over the budget by design, it teaches the budget's
      // bias nothing
      self.absorb_shell(header);
      return;
    }
    FeedbackSet::feedback_remember(&mut self.fb_memory.ewa_step_q8, &mut self.fb_memory.ewa_prev, ewa.bias, ewa.texels);
    let ewa_budget = (dust::DUST_EWA_FRAME_BUDGET as u64) >> self.budget_shift();
    // inside the band [budget / 8, budget]: the state stays (as the split depth's); below it,
    // only when the memory's step ratio keeps the lower bias under the budget
    let ewa_bias = if ewa.texels > ewa_budget
      || (ewa.texels < ewa_budget / 8
        && FeedbackSet::feedback_allows_lower(&self.fb_memory.ewa_step_q8, ewa.bias, ewa.texels, ewa_budget))
    {
      ewa.suggested_bias_for(ewa_budget)
    } else if self.warmup[1]
      && ewa.bias > 0
      && FeedbackSet::feedback_allows_lower(&self.fb_memory.ewa_step_q8, ewa.bias, ewa.texels, ewa_budget)
    {
      warmup_lower(ewa.bias, ewa.texels, ewa_budget).min(self.ewa_bias)
    } else {
      self.ewa_bias
    };
    self.warmup[1] &= ewa_bias < ewa.bias;
    if ewa_bias != self.ewa_bias {
      aethervk_oshal_rlib::log!(
        "[Dust EWA] {:.3e} texels at bias {} (budget {:.3e}): bias {} for the next frames (memory: step ×{:.1} at {}, previous readback {:?})",
        ewa.texels as f64,
        ewa.bias,
        ewa_budget as f64,
        ewa_bias,
        self.fb_memory.ewa_step_q8.get(ewa.bias as usize).copied().unwrap_or(0) as f64 / 256.0,
        ewa.bias,
        self.fb_memory.ewa_prev
      );
    }
    self.ewa_bias = ewa_bias;
    self.absorb_shell(header);
  }

  /// [`Self::absorb`]'s speck shell update
  fn absorb_shell(&mut self, header: &[u32]) {
    // the speck shell: the density at the eye → the next radius; inward on a budget overflow,
    // outward toward half of it under a quarter, from DUST_SPECKS_START_M before any (a cold start
    // inside the coma cannot overflow)
    let prev = if self.shell.d_max_m > 0.0 {
      self.shell.d_max_m
    } else {
      crate::scene::dust::specks::DUST_SPECKS_START_M
    };
    self.shell = ShellUpdate::from_header(header, self.omega_px_sr, prev);
    if self.shell.truncated > 0 {
      aethervk_oshal_rlib::log!(
        "[Dust specks] {} walks stopped by a node cap ({} nodes): their specks are missing",
        self.shell.truncated,
        self.shell.nodes
      );
    }
    if self.shell.overflow > 0 {
      aethervk_oshal_rlib::log!(
        "[Dust specks] budget overflow: {} specks refused by the budget ({} drawn), shell {:.3} m → {:.3} m",
        self.shell.overflow,
        self.shell.specks,
        prev,
        self.shell.d_max_m
      );
    }
  }

  /// what the composite needs: `(white, auto softening)`; with `dust::DUST_VIEW_RHO_NORM` in
  /// [`DustViewHost::view_flags`] the normalised field's pair, the plain one otherwise
  pub fn display(&self) -> (f32, f32) {
    let (white, p50) = if self.rho_norm {
      (self.white_norm, self.p50_norm)
    } else {
      (self.white, self.p50)
    };
    (dust::display_white_v4(white), dust::auto_softening(p50, white))
  }

  /// the plain measured white as the composite would show it without the view options (the
  /// HUD's physical readout, `DustExposure::of`)
  pub fn display_white_measured(&self) -> f32 {
    dust::display_white_v4(self.white)
  }
}

/// The splat pyramid of one presentation engine.
pub struct DustView {
  pub layout: PyramidLayout,
  pub pyramid: Option<DustBuffer>,
  /// the volume packet queue's records per chunk, as allocated ([`Device::dust_vqueue_records`]:
  /// from the device's memory; at most [`dust::DUST_VQUEUE_CAPACITY`])
  pub vqueue_records: u32,
  /// the speck walk's spill slots this view allocated (`dust::DUST_SPILL_SLOTS` in proportion
  /// to the volume queue's records; 0 in CPU mode)
  pub spill_slots: u32,
  /// words of the cadence's class images between the frame pyramid and the queues (0: none)
  pub image_words: u32,
  /// CPU particle mode: the pyramid is host-visible and written directly; one per frame in
  /// flight (`DUST_READBACK_SLOTS`), since the GPU may still read the previous frames' while the
  /// host zeroes and fills the next (the first entry is `pyramid`'s)
  cpu_pyramids: alloc::vec::Vec<(DustBuffer, *mut u8)>,
  pyramid_mapped: *mut u8,
  pub readback: Option<DustBuffer>,
  readback_mapped: *mut u8,
  /// the view's timestamp queries ([`DUST_READBACK_SLOTS`] × [`DUST_TIMESTAMPS`]; null when the
  /// graphics queue has no timestamps): owned by the manager's pool list
  ts_pool: vk::QueryPool,
  pub host: spin::Mutex<DustViewHost>,
  /// this frame's view words (flags, flow, tracer counts), written into the header
  frame_words: spin::Mutex<[u32; dust::PYRAMID_HEADER_WORDS as usize]>,
  /// what the GPU splat passes drew (diagnostics, [`Device::download_dust_snapshot`]): the
  /// frame being recorded and the last fully recorded one, `(frame, records)` each
  splat_records: spin::Mutex<[(u64, alloc::vec::Vec<DustSplatRecord>); 2]>,
}

/// `bytes` as a vector of `T` (an aligned copy; trailing bytes of a partial element dropped)
fn pod_vec<T: bytemuck::Pod>(bytes: &[u8]) -> alloc::vec::Vec<T> {
  let n = bytes.len() / core::mem::size_of::<T>();
  let mut v = alloc::vec![T::zeroed(); n];
  bytemuck::cast_slice_mut::<T, u8>(&mut v).copy_from_slice(&bytes[..n * core::mem::size_of::<T>()]);
  v
}

/// One tier's GPU splat pass as recorded (host memory only): which buffers and ranges it read
/// and its push constants, so a diagnostic can download the same clusters and replay the frame
/// on the CPU mirror ([`Device::download_dust_snapshot`]).
#[derive(Clone, Copy, Debug)]
pub struct DustSplatRecord {
  pub system: u64,
  pub ring_base: u32,
  pub render: vk::Buffer,
  pub render_offset: u64,
  pub moments: vk::Buffer,
  pub moments_offset: u64,
  /// the tier's live range start (ring slot) and sub-ring size: the buffers are slot indexed
  pub first_slot: u32,
  pub capacity: u32,
  pub pc: DustSplatPushConstants,
  pub flow: dust::DustFlowUniform,
  pub tracer_counts: u32,
}

/// A recorded tier with its clusters downloaded ([`Device::download_dust_snapshot`]).
#[derive(Clone, Debug)]
pub struct DustSplatSnapshot {
  pub pc: DustSplatPushConstants,
  pub flow: dust::DustFlowUniform,
  pub tracer_counts: u32,
  pub render: alloc::vec::Vec<dust::DustRenderCluster>,
  pub moments: alloc::vec::Vec<dust::DustMoments>,
}

// SAFETY: the mapped pointers point into persistently mapped VMA allocations owned by this struct
unsafe impl Send for DustView {}
unsafe impl Sync for DustView {}

impl DustView {
  /// the word where the view's queues start (the EWA queue header): after the frame pyramid and
  /// the class images ([`Self::image_words`]); header word `dust::PYR_QUEUE_BASE` carries it to
  /// the shaders (`dust_queue_base`)
  pub fn queue_word(&self) -> u32 {
    self.layout.total_words + self.image_words
  }

  /// the pyramid whose queue region (from [`Self::queue_word`]) a diagnostic download reads; an
  /// error in CPU particle mode, whose host pyramids end with the frame pyramid (a copy past their
  /// end faulted the copy engine: Xid 31, device lost)
  fn queue_pyramid(&self) -> GpuResult<&DustBuffer> {
    if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      return Err(gpu_err!("CPU particle mode: the dust view has no queues"));
    }
    self.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))
  }

  /// CPU mode: the host pyramid (buffer, mapped words) of frame `frame`
  fn cpu_pyramid(&self, frame: u64) -> Option<(&DustBuffer, *mut u8)> {
    let k = (frame % DUST_READBACK_SLOTS as u64) as usize;
    if k == 0 {
      self.pyramid.as_ref().map(|b| (b, self.pyramid_mapped))
    } else {
      self.cpu_pyramids.get(k - 1).map(|(b, m)| (b, *m))
    }
  }

  /// words copied back per frame: the header and the white-point measurement grid
  /// the header, the measurement grid, then the packet queue's header
  fn readback_words(layout: &PyramidLayout) -> u32 {
    let (_, w, h) = layout.measure;
    dust::PYRAMID_HEADER_WORDS
      + w * h * dust::PYRAMID_TEXEL_WORDS
      + dust::DUST_QUEUE_HEADER_WORDS
      + dust::DUST_VQUEUE_HEADER_WORDS
      // the splitter family (perf9 steps Z / O): the leaf queue's and the candidate buffer's headers
      + 2 * dust::DUST_VQUEUE_HEADER_WORDS
  }
  /// `(level, offset, width, height)` of the measurement grid
  fn white_level(layout: &PyramidLayout) -> (u32, u32, u32, u32) {
    let (off, w, h) = layout.measure;
    (dust::DUST_WHITE_LEVEL, off, w, h)
  }
}

pub struct DustManager {
  pub systems: dashmap::DashMap<u64, DustGpuSystem>,
  pub views: dashmap::DashMap<gpu::PresentationEngineHandle, DustView>,
  /// the views running the GPU cadence ([`Device::dust_cadence_enable`])
  pub cadence: dashmap::DashMap<gpu::PresentationEngineHandle, spin::Mutex<cadence::DustCadenceGpu>>,
  allocator_view: vk_mem::AllocatorView,
  /// every timestamp query pool the views were given (destroyed with the device,
  /// [`DustManager::take_query_pools`]) and those no view holds now: a removed view's pool may
  /// still be written by frames in flight, so it is reused, never destroyed before the device
  query_pools: spin::Mutex<alloc::vec::Vec<vk::QueryPool>>,
  free_query_pools: spin::Mutex<alloc::vec::Vec<vk::QueryPool>>,
  /// the scene's trajectory detail (`dust::clamp_propagate_detail`, 17 / 9 / 5 solved
  /// size-polyline edges; [`Device::dust_set_propagate_detail`], stamped by `prepare_dust` before
  /// the propagates): every propagate of every system runs at its edge stride
  pub propagate_detail: core::sync::atomic::AtomicU32,
}

impl DustManager {
  pub fn new(allocator_view: vk_mem::AllocatorView) -> Self {
    Self {
      systems: dashmap::DashMap::new(),
      views: dashmap::DashMap::new(),
      cadence: dashmap::DashMap::new(),
      allocator_view,
      query_pools: spin::Mutex::new(alloc::vec::Vec::new()),
      free_query_pools: spin::Mutex::new(alloc::vec::Vec::new()),
      propagate_detail: core::sync::atomic::AtomicU32::new(dust::DUST_PROPAGATE_DETAIL_DEFAULT),
    }
  }

  /// Every timestamp pool ever given to a view, for the device's teardown (after its idle wait)
  pub fn take_query_pools(&self) -> alloc::vec::Vec<vk::QueryPool> {
    self.free_query_pools.lock().clear();
    core::mem::take(&mut *self.query_pools.lock())
  }

  /// a removed view's pool back to the free list
  fn release_query_pool(&self, pool: vk::QueryPool) {
    if pool != vk::QueryPool::null() {
      self.free_query_pools.lock().push(pool);
    }
  }

  fn destroy_system(allocator: &vk_mem::AllocatorView, mut sys: DustGpuSystem) {
    let trace = sys.trace_rb.get_mut().take().map(|(b, _)| b);
    for b in [
      trace,
      sys.clusters.take(),
      sys.render.take(),
      sys.moments.take(),
      sys.batches.take(),
      sys.rows.take(),
    ]
    .into_iter()
    .flatten()
    {
      let mut a = b.alloc;
      unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
    }
  }

  fn destroy_view(allocator: &vk_mem::AllocatorView, mut view: DustView) {
    let extra = core::mem::take(&mut view.cpu_pyramids).into_iter().map(|(b, _)| b);
    for b in [view.pyramid.take(), view.readback.take()].into_iter().flatten().chain(extra) {
      let mut a = b.alloc;
      unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
    }
  }
}

impl Drop for DustManager {
  fn drop(&mut self) {
    let ids: alloc::vec::Vec<u64> = self.systems.iter().map(|e| *e.key()).collect();
    for id in ids {
      if let Some((_, sys)) = self.systems.remove(&id) {
        Self::destroy_system(&self.allocator_view, sys);
      }
    }
    let pes: alloc::vec::Vec<_> = self.views.iter().map(|e| *e.key()).collect();
    for pe in pes {
      if let Some((_, v)) = self.views.remove(&pe) {
        Self::destroy_view(&self.allocator_view, v);
      }
    }
    let pes: alloc::vec::Vec<_> = self.cadence.iter().map(|e| *e.key()).collect();
    for pe in pes {
      if let Some((_, c)) = self.cadence.remove(&pe) {
        for b in c.into_inner().into_buffers() {
          let mut a = b.alloc;
          unsafe { self.allocator_view.destroy_buffer(b.buffer, &mut a) };
        }
      }
    }
  }
}

/// What the composite needs of a view's dust this frame.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DustViewParams {
  /// pyramid device address (0: no dust)
  pub pyramid: u64,
  pub levels: u32,
  /// exposure-scaled optical depth × px² per count
  pub unit: f32,
  /// white point (the seed surface brightness before any measurement) the composite divides by:
  /// the plain measured one, or the 1/ρ-normalised field's with `dust::DUST_VIEW_RHO_NORM`
  pub white: f32,
  /// the plain measured white whatever the view options (the HUD's physical readout)
  pub white_measured: f32,
  /// the median dust texel (0 before any measurement)
  pub p50: f32,
  /// auto softening (`dust::auto_softening`)
  pub softening: f32,
  /// the speck shell radius this frame draws with (local m, 0 = no specks)
  pub d_max_m: f32,
  /// the number density at the eye of the last measured frame (m⁻³)
  pub n_eye_m3: f32,
  /// specks drawn / beyond the budget in the last measured frame
  pub specks: u32,
  pub specks_overflow: u32,
}

impl Device {
  /// Micro-layer color target of the compositing pass (`R16G16B16A16_SFLOAT`, or `R8G8B8A8_UNORM`
  /// where unsupported).
  pub fn micro_color_format(&self) -> vk::Format {
    self.micro_color_format
  }

  /// Ring capacity for this device: high tier on discrete GPUs, low tier otherwise, small in CPU
  /// particle mode. `AETHERVK_DUST_RING=<power of two>` overrides it.
  pub fn dust_ring_capacity(&self) -> u32 {
    let env = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_RING")
      .and_then(|s| s.trim().parse::<u32>().ok());
    if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      return dust::ring_capacity(false, env.or(Some(DUST_RING_CAPACITY_CPU)));
    }
    let high_end = self.query_result.physical_device_properties.device_type
      == vk::PhysicalDeviceType::DISCRETE_GPU;
    dust::ring_capacity(high_end, env)
  }

  fn dust_create_buffer(
    &self,
    allocator: &vk_mem::AllocatorView,
    size: u64,
    mem: DustMem,
    concurrent: bool,
    extra_usage: vk::BufferUsageFlags,
    name: &str,
  ) -> GpuResult<(DustBuffer, *mut u8)> {
    let mut info = vk::BufferCreateInfo::default()
      .size(size)
      // TRANSFER_SRC: debug / test readbacks
      .usage(
        vk::BufferUsageFlags::STORAGE_BUFFER
          | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
          | vk::BufferUsageFlags::TRANSFER_SRC
          | extra_usage,
      );
    let families = [
      self.get_compute_queue().family_index,
      self.get_graphics_queue().family_index,
    ];
    if concurrent && families[0] != families[1] {
      info = info.sharing_mode(vk::SharingMode::CONCURRENT).queue_family_indices(&families);
    }
    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    crate::apply_test_dedicated_alloc!(alloc_info);
    match mem {
      DustMem::HostWrite => {
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
          | vk_mem::AllocationCreateFlags::MAPPED;
      }
      DustMem::HostRead => {
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
        alloc_info.flags =
          vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;
      }
      DustMem::Device => {
        alloc_info.usage = vk_mem::MemoryUsage::AutoPreferDevice;
        alloc_info.priority = 1.0;
      }
    }
    let (buffer, alloc, alloc_res) =
      unsafe { allocator.create_buffer_get_info(&info, &alloc_info) }
        .with_name(&self.device, name)?;
    let address = unsafe {
      self
        .device
        .buffer_device_address
        .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
    };
    super::fault::register_address_range(name, address, size);
    Ok((
      DustBuffer {
        buffer,
        alloc,
        address,
      },
      alloc_res.mapped_data.cast(),
    ))
  }

  /// Allocates the dust resources of particle system `id`. Returns the ring capacity.
  pub fn create_particle_system(&self, id: u64) -> GpuResult<u32> {
    let capacity = self.dust_ring_capacity();
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    let sys = if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      let mut sys = DustGpuSystem::empty(capacity);
      sys.cpu_ring = Some(spin::RwLock::new(
        alloc::vec![bytemuck::Zeroable::zeroed(); capacity as usize],
      ));
      sys
    } else {
      let size_c = capacity as u64 * core::mem::size_of::<DustCluster>() as u64;
      let size_r = capacity as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
      let size_m = capacity as u64 * core::mem::size_of::<DustMoments>() as u64;
      let size_b = DUST_BATCH_SLOTS as u64 * core::mem::size_of::<DustBatch>() as u64;
      let none = vk::BufferUsageFlags::empty();
      // on failure, everything created so far is released with the partial system
      let mut sys = DustGpuSystem::empty(capacity);
      let mut make = |size, mem, concurrent, usage, what: &str| {
        self.dust_create_buffer(
          &allocator,
          size,
          mem,
          concurrent,
          usage,
          &alloc::format!("{what}_{id}"),
        )
      };
      let res: GpuResult<()> = (|| {
        sys.clusters = Some(make(size_c, DustMem::Device, true, none, "DustClusters")?.0);
        sys.render = Some(make(size_r, DustMem::Device, false, none, "DustRender")?.0);
        sys.moments = Some(make(size_m, DustMem::Device, false, none, "DustMoments")?.0);
        let size_rows = dust::DUST_PROP_CHUNK_WGS as u64 * DUST_WG as u64 * dust::DUST_PROP_ROW_WORDS as u64 * 4;
        sys.rows = Some(make(size_rows, DustMem::Device, false, none, "DustRows")?.0);
        let (b, mapped) = make(size_b, DustMem::HostWrite, false, none, "DustBatches")?;
        sys.batches = Some(b);
        sys.batches_mapped = mapped;
        Ok(())
      })();
      if let Err(e) = res {
        DustManager::destroy_system(&allocator, sys);
        return Err(e);
      }
      sys
    };
    if let Some(old) = mgr.systems.insert(id, sys) {
      DustManager::destroy_system(&allocator, old);
    }
    aethervk_oshal_rlib::log!("[Dust] system {id}: ring capacity {capacity}");
    Ok(capacity)
  }

  /// Discards the resources of particle system `id`, once both queues are past the given values.
  pub fn discard_particle_system(
    &self,
    id: u64,
    gfx_timeline: u64,
    comp_timeline: u64,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let (_, mut sys) = mgr
      .systems
      .remove(&id)
      .ok_or(gpu_err!("particle system with id {} not found", id))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    // clusters, render, moments: written by compute, read by graphics. Wait (bounded) for the last
    // compute submit, then retire them with the graphics frames that may still read them. If the
    // compute work has not completed within the bound (a heavy frame, GPU-AV), they are leaked
    // instead: freeing them under the compute queue is a use-after-free (a plausible Xid 13/32).
    // Discards are rare.
    let last_compute = comp_timeline.saturating_sub(1);
    let compute_done = last_compute == 0
      || self
        .device
        .wait_for_semaphore_value(self.kernels.timeline, last_compute, 1_000_000_000)
        .is_ok();
    if !compute_done {
      aethervk_oshal_rlib::log!(
        "[Dust] particle system {} removed while its compute work is still in flight: its compute-written buffers are leaked",
        id
      );
    }
    if let Some(b) = sys.clusters.take()
      && compute_done
    {
      res.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        gfx_timeline,
      );
    }
    if let Some(b) = sys.batches.take() {
      self.kernels.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        comp_timeline,
      );
    }
    let trace = sys.trace_rb.get_mut().take().map(|(b, _)| b);
    let (render, moments, rows) = (sys.render.take(), sys.moments.take(), sys.rows.take());
    let compute_written = if compute_done {
      [render, moments, rows]
    } else {
      [None, None, None]
    };
    for b in compute_written.into_iter().chain([trace]).flatten() {
      res.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        gfx_timeline,
      );
    }
    Ok(())
  }

  /// Discards the dust view of presentation engine `pe` (its pyramid), once the graphics queue is
  /// past `gfx_timeline`. No-op without one.
  pub fn discard_dust_view(&self, pe: gpu::PresentationEngineHandle, gfx_timeline: u64) {
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else {
      return;
    };
    let Some((_, mut view)) = mgr.views.remove(&pe) else {
      return;
    };
    mgr.release_query_pool(view.ts_pool);
    let allocator = res.allocator.allocator.as_allocator_view();
    let extra = core::mem::take(&mut view.cpu_pyramids).into_iter().map(|(b, _)| b);
    for b in [view.pyramid.take(), view.readback.take()].into_iter().flatten().chain(extra) {
      res.discard_pool.discard_buffer(
        allocator.as_allocator_view(),
        b.buffer,
        b.alloc,
        gfx_timeline,
      );
    }
  }

  /// Records the emission of `batch` (its `first_index`, `count`, `ring_mask` already set) into
  /// the sub-ring starting at slot `ring_base` (age tier, see `dust::DustSystemState`).
  /// `seq` is the monotonic batch sequence number of the system (selects the descriptor slot).
  /// CPU particle mode: emits on the host immediately, records nothing.
  pub fn cmd_dust_emit(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    ring_base: u32,
    batch: &DustBatch,
    seq: u64,
  ) -> GpuResult<()> {
    if batch.count == 0 {
      return Ok(());
    }
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    if ring_base as u64 + batch.ring_mask as u64 + 1 > sys.capacity as u64 {
      return Err(gpu_err!(
        "dust sub-ring {}+{} exceeds capacity {}",
        ring_base,
        batch.ring_mask + 1,
        sys.capacity
      ));
    }
    if let Some(ring) = sys.cpu_ring.as_ref() {
      let mut ring = ring.write();
      for j in 0..batch.count {
        let slot = (batch.first_index.wrapping_add(j) & batch.ring_mask) as usize;
        ring[ring_base as usize + slot] = dust::emit_cluster(batch, j);
      }
      return Ok(());
    }
    let (Some(clusters), Some(batches)) = (sys.clusters.as_ref(), sys.batches.as_ref()) else {
      return Err(gpu_err!("dust system {} has no GPU buffers", id));
    };
    let slot = (seq % DUST_BATCH_SLOTS as u64) as usize;
    let stride = core::mem::size_of::<DustBatch>();
    unsafe {
      core::ptr::copy_nonoverlapping(
        bytemuck::bytes_of(batch).as_ptr(),
        sys.batches_mapped.add(slot * stride),
        stride,
      );
    }
    res.allocator.allocator.as_allocator_view().flush_allocation(
      &batches.alloc,
      (slot * stride) as u64,
      stride as u64,
    )?;
    let cluster_size = core::mem::size_of::<DustCluster>() as u64;
    let pc = DustEmitPushConstants {
      // the shaders index the sub-ring from its own base address
      clusters: clusters.address + ring_base as u64 * cluster_size,
      batch: batches.address + (slot * stride) as u64,
    };
    let pipelines = &self.kernels.pipelines;
    pipelines.assert_pc_size(pipelines.dust_emit, core::mem::size_of_val(&pc));
    let run = |p: vk::Pipeline| unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
      self.device.cmd_push_constants(
        cmd,
        pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, batch.count.div_ceil(DUST_WG), 1, 1);
    };
    // (perf9_K) the pre-package emission (`dust_emit_mono.comp`, the GTX 1070 default): one
    // dispatch, the same push constants and grid
    if !self.dust_prop_staged() {
      if let Some(m) = pipelines.dust_emit_mono {
        pipelines.assert_pc_size(m, core::mem::size_of_val(&pc));
        run(m);
        return Ok(());
      }
    }
    match pipelines.dust_emit_passes {
      // round 9 step O, the staged emission: 1, (2, 3) × DUST_KEPLER_ITERS_DF, 6, 4, 5r, 5v, 71,
      // 72 over the same threads, a barrier between passes (each reads the record words the
      // previous one wrote)
      Some(passes) => {
        run(passes[0]);
        for _ in 0..dust::DUST_KEPLER_ITERS_DF {
          self.cmd_dust_pre_splat_barrier(cmd);
          run(passes[1]);
          self.cmd_dust_pre_splat_barrier(cmd);
          run(passes[2]);
        }
        for p in &passes[3..] {
          self.cmd_dust_pre_splat_barrier(cmd);
          run(*p);
        }
      }
      None => run(pipelines.dust_emit),
    }
    Ok(())
  }

  /// Barrier before the first dust command of a graphics command buffer: the previous frame's
  /// composite (fragment reads of the pyramid), trace / readback copies (WAR) and host writes of
  /// the batch slots, before this frame's compute writes and the pyramid zeroing.
  pub fn cmd_dust_pre_propagate_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(
        vk::PipelineStageFlags2::FRAGMENT_SHADER
          | vk::PipelineStageFlags2::COMPUTE_SHADER
          | vk::PipelineStageFlags2::TRANSFER,
      )
      .src_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::SHADER_STORAGE_WRITE
          | vk::AccessFlags2::TRANSFER_READ,
      )
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Barrier between the propagates (and the pyramid reset) and the splat passes: compute /
  /// transfer writes → compute reads and atomics.
  pub fn cmd_dust_pre_splat_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// perf9_K: whether the splat runs the stage chain ([`Self::cmd_dust_splat_chain`]) or the
  /// monolithic enumerator (`dust_splat_mono.comp`): the override of
  /// [`Self::dust_set_splat_staged`], else the vendor rule
  /// (`physics::dust_splat_staged_default`, env `AETHERVK_DUST_SPLAT_STAGED`); the chain
  /// wherever the monolith is not built
  pub fn dust_splat_staged(&self) -> bool {
    if self.kernels.pipelines.dust_splat_mono.is_none() {
      return true;
    }
    match self.kernels.pipelines.dust_splat_mode.load(core::sync::atomic::Ordering::Relaxed) {
      1 => true,
      2 => false,
      _ => crate::gpu_backends::vulkan::physics::dust_splat_staged_default(
        self.query_result.physical_device_properties.vendor_id,
      ),
    }
  }

  /// perf9_K: the splat path of this device from now on (true: the stage chain, false: the
  /// monolith, where built), overriding the vendor rule and the env (the parity tests run both)
  pub fn dust_set_splat_staged(&self, staged: bool) {
    self.kernels.pipelines.dust_splat_mode.store(if staged { 1 } else { 2 }, core::sync::atomic::Ordering::Relaxed);
  }

  /// perf9_K: back to the vendor rule and the env
  pub fn dust_reset_splat_staged(&self) {
    self.kernels.pipelines.dust_splat_mode.store(0, core::sync::atomic::Ordering::Relaxed);
  }

  /// perf9_K: logs the splat path once per change (the byte gate's and the observer's logs say
  /// which path ran), with the other staged switches beside it
  fn dust_splat_log_path(&self, mono: bool) {
    let v = if mono { 2u8 } else { 1u8 };
    let p = &self.kernels.pipelines;
    if p.dust_splat_logged.swap(v, core::sync::atomic::Ordering::Relaxed) != v {
      aethervk_oshal_rlib::log!(
        "[dust] STAGED splat={} (AETHERVK_DUST_SPLAT_STAGED={:?}, override {}, vendor {:#X}, monolith built {}); propagate/emit staged={} (family {}), volume stages={}",
        if mono { "mono" } else { "chain" },
        aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPLAT_STAGED"),
        p.dust_splat_mode.load(core::sync::atomic::Ordering::Relaxed),
        self.query_result.physical_device_properties.vendor_id,
        p.dust_splat_mono.is_some(),
        p.dust_propagate_passes.is_some(),
        if self.dust_prop_staged() { "package" } else { "mono" },
        p.dust_draw_volume_stages.is_some()
      );
    }
  }

  /// The enumerator as a stage chain (round 9 step O, `dust_splat.comp` stages): over a chunk's
  /// `n` clusters LIVE, PRE, LOD, with the haze knob engaged HAZE_E, HAZE_A, HAZE_B (the stages
  /// return at once on the clusters without the haze, so the host skips them at the default
  /// detail), then SHAPE, GEO, SUBW, COUNT at [`dust::DUST_SPLAT_STAGE_LANES`] lanes per cluster,
  /// REC_A and REC_B per record, indirect from the sub-packet queue header's args (COUNT atomicMaxes
  /// them). The push constants are the same for every stage (pushed once; the layout is shared);
  /// a barrier between stages: compute writes → compute reads and the indirect args.
  fn cmd_dust_splat_chain(
    &self,
    cmd: vk::CommandBuffer,
    stages: &[vk::Pipeline; dust::DUST_SPLAT_STAGES],
    pc: &DustSplatPushConstants,
    n: u32,
    buffer: vk::Buffer,
    sbase: u64,
  ) {
    let between = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::INDIRECT_COMMAND_READ
          | vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&between));
    let haze = dust::haze_detail_from_flags(pc.flags) < dust::DUST_HAZE_DETAIL_DEFAULT;
    let order: [usize; 12] = [
      dust::DUST_SPLAT_STAGE_LIVE,
      dust::DUST_SPLAT_STAGE_PRE,
      dust::DUST_SPLAT_STAGE_LOD,
      dust::DUST_SPLAT_STAGE_HAZE_E,
      dust::DUST_SPLAT_STAGE_HAZE_A,
      dust::DUST_SPLAT_STAGE_HAZE_B,
      dust::DUST_SPLAT_STAGE_SHAPE,
      dust::DUST_SPLAT_STAGE_GEO,
      dust::DUST_SPLAT_STAGE_SUBW,
      dust::DUST_SPLAT_STAGE_COUNT,
      dust::DUST_SPLAT_STAGE_REC_A,
      dust::DUST_SPLAT_STAGE_REC_B,
    ];
    let mut first = true;
    for s in order {
      let lanes = dust::DUST_SPLAT_STAGE_LANES[s];
      if lanes == 35 || lanes == 70 {
        // the haze trio
        if !haze {
          continue;
        }
      }
      unsafe {
        if !first {
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
        }
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, stages[s]);
        if first {
          self.device.cmd_push_constants(
            cmd,
            self.kernels.pipelines.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(pc),
          );
        }
        if lanes == 0 {
          let args = if s == dust::DUST_SPLAT_STAGE_REC_A { dust::DUST_SQUEUE_REC_A_ARGS } else { dust::DUST_SQUEUE_REC_B_ARGS };
          self.device.cmd_dispatch_indirect(cmd, buffer, sbase + args as u64 * 4);
        } else {
          self.device.cmd_dispatch(cmd, (n * lanes).div_ceil(DUST_WG), 1, 1);
        }
      }
      first = false;
    }
  }

  /// Barrier between the speck walk's passes (round 9 step O): compute writes → compute reads
  /// and atomics, and the indirect args the region holds.
  pub fn cmd_dust_walk_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT)
      .dst_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::SHADER_STORAGE_WRITE
          | vk::AccessFlags2::INDIRECT_COMMAND_READ,
      );
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Barrier after all splat passes: compute writes → the composite's fragment reads and the
  /// readback / trace copies.
  pub fn cmd_dust_post_splat_barrier(&self, cmd: vk::CommandBuffer) {
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::TRANSFER_READ);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe { self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep) };
  }

  /// Starts the frame of system `id`: stores the view aids and the flow uniform, writes the
  /// trace line of the slot read back this frame. Before [`Self::cmd_dust_pre_splat_barrier`].
  pub fn cmd_dust_system_begin(
    &self,
    id: u64,
    flags: u32,
    flow: dust::DustFlowUniform,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    sys.cpu_render.lock().clear();
    let (slot, folded) = {
      let mut host = sys.host.lock();
      host.view_flags = flags;
      host.flow = flow;
      let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
      let folded = host.slot_valid[slot];
      host.slot_valid[slot] = false;
      (slot, folded)
    };
    // the traced frame of this slot (if any) is complete: write its line
    if let Some(meta) = sys.trace_meta.lock()[slot].take()
      && folded
      && let Some((tb, tptr)) = sys.trace_rb.lock().as_ref()
    {
      res.allocator.allocator.as_allocator_view().invalidate_allocation(
        &tb.alloc,
        slot as u64 * TRACE_SLOT_BYTES,
        TRACE_SLOT_BYTES,
      )?;
      let per_tier = crate::gpu::dust_trace::TRACE_SAMPLES;
      let all = unsafe {
        core::slice::from_raw_parts(
          tptr.add(slot * TRACE_SLOT_BYTES as usize).cast::<DustRenderCluster>(),
          dust::LOD_MAX_TIERS as usize * per_tier,
        )
      };
      let clusters: alloc::vec::Vec<alloc::vec::Vec<DustRenderCluster>> = meta
        .tiers
        .iter()
        .map(|t| {
          let base = t.tier as usize * per_tier;
          all[base..base + t.samples.len()].to_vec()
        })
        .collect();
      crate::gpu::dust_trace::append(&crate::gpu::dust_trace::record_line(&meta, &clusters));
    }
    Ok(())
  }

  /// Ends the frame of system `id`: the trace slot of this frame becomes readable in
  /// [`DUST_READBACK_SLOTS`] frames.
  pub fn cmd_dust_system_end(&self, id: u64) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let mut host = sys.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
    host.slot_valid[slot] = sys.render.is_some();
    host.frame += 1;
    Ok(())
  }

  /// Starts the dust frame of presentation engine `pe` of `extent`: creates (or re-creates on a
  /// size change) the view's pyramid, folds the readback slot written [`DUST_READBACK_SLOTS`]
  /// frames ago into the white point, median and unit, then records the header write (level
  /// table, `flags`, `flow`, the tracer height) and the zeroing of the levels. Returns this
  /// frame's [`DustViewParams`] (the unit the splats must use). Before the propagates.
  /// The nucleus' framebuffer pixel (x right, y down; `None` = unknown) of the frame about to
  /// begin on `pe`: the 1/ρ-normalised white point (`dust::DUST_VIEW_RHO_NORM`) measures that
  /// frame's readback around it. Called before `cmd_dust_frame_begin*`.
  pub fn dust_set_nucleus_px(
    &self,
    pe: gpu::PresentationEngineHandle,
    nucleus_px: Option<[f32; 2]>,
    rho_norm: bool,
  ) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      let mut host = view.host.lock();
      host.nucleus_px_next = nucleus_px.unwrap_or([f32::NAN; 2]);
      host.rho_norm = rho_norm;
    }
  }

  pub fn cmd_dust_frame_begin(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    extent: [u32; 2],
    flags: u32,
    flow: dust::DustFlowUniform,
  ) -> GpuResult<DustViewParams> {
    self.frame_begin(cmd, pe, extent, flags, flow, None).map(|(v, _)| v)
  }

  /// [`Self::cmd_dust_frame_begin`] that keeps the last frame's image when nothing it was drawn
  /// from changed (8 ms plan Step 2). `inputs` is the caller's fingerprint of everything its
  /// propagates, splats and specks read besides the header (the push constants, the frames, the
  /// drawn ranges and emissions); the header is compared here, without its display-only words
  /// ([`reuse_header`]). When both equal the last fully drawn frame's, the pyramid is not
  /// cleared — only its white point and softening are rewritten — and `true` is returned: the
  /// caller then records no propagate, splat or specks (the measure and readback still run, so
  /// the exposure keeps adapting). A non-reused frame records `inputs` as drawn; call
  /// [`Self::dust_forget_drawn`] if it then fails to draw completely. CPU mode never reuses.
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_frame_begin_reusable(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    extent: [u32; 2],
    flags: u32,
    flow: dust::DustFlowUniform,
    inputs: u64,
  ) -> GpuResult<(DustViewParams, bool)> {
    self.frame_begin(cmd, pe, extent, flags, flow, Some(inputs))
  }

  /// Resets `pe`'s feedback ([`DustViewHost::reset_feedback`]) when the scene's dust feedback
  /// generation `generation` differs from the one it was measured under; a no-op without a view.
  /// Returns whether it reset.
  /// The play pin of `pe` (handoff P3b; `RenderScene::dust_play`, `prepare_dust`): `Some` while
  /// the simulation runs with the play quality on — the view's frames begin with the play set's
  /// biases ([`dust::play_quality::PlayFeedback::pin`] at the view's level count: split depth 0,
  /// the top-only volume bias, EWA 3) over whatever the feedback absorbed; the paused feedback
  /// state is saved at the first `Some` and restored at the first `None`
  /// ([`DustViewHost::play_pin_enter`] / [`DustViewHost::play_pin_leave`]). Returns whether the
  /// pin changed state.
  pub fn dust_set_play_feedback(&self, pe: gpu::PresentationEngineHandle, play: Option<dust::play_quality::PlayFeedback>) -> bool {
    let res = self.res.read();
    let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) else {
      return false;
    };
    let levels = view.layout.level_count();
    let mut host = view.host.lock();
    match play {
      Some(p) => {
        let pin = p.pin(levels);
        let entered = host.play_pin_enter(pin);
        if entered {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: play feedback pinned (split depth {}, volume bias {} of {} levels, EWA bias {})",
            pe.0, pin.split_depth, pin.volume_bias, levels, pin.ewa_bias
          );
        }
        entered
      }
      None => {
        let left = host.play_pin_leave();
        if left {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: play feedback pin left (paused feedback restored: split depth {}, volume bias {}, EWA bias {})",
            pe.0, host.split_depth, host.volume_bias, host.ewa_bias
          );
        }
        left
      }
    }
  }

  pub fn dust_sync_feedback_gen(&self, pe: gpu::PresentationEngineHandle, generation: u32) -> bool {
    let res = self.res.read();
    let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) else {
      return false;
    };
    let mut host = view.host.lock();
    if host.feedback_gen == generation {
      return false;
    }
    host.feedback_gen = generation;
    // a view that measured nothing yet is at its cold start: it keeps it
    if !host.feedback_measured() {
      return false;
    }
    host.reset_feedback();
    aethervk_oshal_rlib::log!("[Dust] view {}: feedback reset (generation {generation})", pe.0);
    true
  }

  /// Whether `pe` draws refined (static refinement, `dust::dust_refine_factor`): a frame that draws
  /// then draws everything fresh, without the cadence's retained class images (drawn at the
  /// budget's bias)
  pub fn dust_refining(&self, pe: gpu::PresentationEngineHandle) -> bool {
    let res = self.res.read();
    res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)).is_some_and(|v| v.host.lock().refine > 0)
  }

  /// Forgets `pe`'s last drawn frame: the next [`Self::cmd_dust_frame_begin_reusable`] draws
  pub fn dust_forget_drawn(&self, pe: gpu::PresentationEngineHandle) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().drawn = None;
    }
  }

  fn frame_begin(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    extent: [u32; 2],
    flags: u32,
    flow: dust::DustFlowUniform,
    inputs: Option<u64>,
  ) -> GpuResult<(DustViewParams, bool)> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let allocator = res.allocator.allocator.as_allocator_view();
    let cpu = crate::gpu_backends::vulkan::physics::is_cpu_particles_mode();
    let layout = PyramidLayout::new(extent[0], extent[1]);
    // (re)create on first use or a size change
    // (a changed class-image span recreates the view too: the images sit before the queues)
    let want_images = if cpu { 0 } else { mgr.cadence.get(&pe).map_or(0, |c| c.lock().image_span(&layout)) };
    let stale = mgr.views.get(&pe).is_some_and(|v| v.layout != layout || v.image_words != want_images);
    if stale {
      let timeline = res.timeline_manager.get_next_submit_value();
      if let Some((_, mut view)) = mgr.views.remove(&pe) {
        mgr.release_query_pool(view.ts_pool);
        let extra = core::mem::take(&mut view.cpu_pyramids).into_iter().map(|(b, _)| b);
        for b in [view.pyramid.take(), view.readback.take()].into_iter().flatten().chain(extra) {
          res.discard_pool.discard_buffer(
            allocator.as_allocator_view(),
            b.buffer,
            b.alloc,
            timeline,
          );
        }
      }
    }
    if !mgr.views.contains_key(&pe) {
      let bytes = layout.total_words as u64 * 4;
      // the GPU pyramid carries the packet queue after its words (`dust::DUST_QUEUE_WORDS`; its
      // header is the queue draw's indirect dispatch), then the volume packet queue sized from
      // the device's memory
      let vqueue_records = self.dust_vqueue_records();
      // then the sub-packet queue and the speck walk's spill (its slots in proportion to the
      // volume queue's records), placed after the records this view has: the shaders read them
      // from the volume header's `DUST_VQUEUE_RECORDS` word (`dust_squeue_base`)
      // (perf9_O C2 v5) the staged consumer's scratch rows, when its pipelines are built
      let staged = self.kernels.pipelines.dust_draw_volume_stages.is_some();
      let offsets = dust::dust_queue_offsets_for(vqueue_records, staged);
      let spill_slots = offsets.spill_slots;
      let queue_words = offsets.end;
      // the cadence's class images (`DustCadenceGpu::image_span`): between the frame pyramid and
      // the queues
      let image_words = want_images;
      let (pyramid, pmapped) = self.dust_create_buffer(
        &allocator,
        if cpu {
          bytes
        } else {
          bytes + (image_words as u64 + queue_words as u64) * 4
        },
        if cpu {
          DustMem::HostWrite
        } else {
          DustMem::Device
        },
        false,
        if cpu {
          vk::BufferUsageFlags::TRANSFER_DST
        } else {
          vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDIRECT_BUFFER
        },
        &alloc::format!("DustPyramid_{}", pe.0),
      )?;
      let mut cpu_pyramids = alloc::vec::Vec::new();
      if cpu {
        for k in 1..DUST_READBACK_SLOTS {
          match self.dust_create_buffer(
            &allocator,
            bytes,
            DustMem::HostWrite,
            false,
            vk::BufferUsageFlags::TRANSFER_DST,
            &alloc::format!("DustPyramid_{}_{k}", pe.0),
          ) {
            Ok(x) => cpu_pyramids.push(x),
            Err(e) => {
              let mut a = pyramid.alloc;
              unsafe { allocator.destroy_buffer(pyramid.buffer, &mut a) };
              for (b, _) in cpu_pyramids {
                let mut a = b.alloc;
                unsafe { allocator.destroy_buffer(b.buffer, &mut a) };
              }
              return Err(e);
            }
          }
        }
      }
      // (then the target probes, `DUST_PROBE_SLOT_WORDS` per slot)
      let rb_bytes = DUST_READBACK_SLOTS as u64 * DustView::readback_words(&layout) as u64 * 4
        + DUST_READBACK_SLOTS as u64 * DUST_PROBE_SLOT_WORDS as u64 * 4;
      let (readback, rmapped) = match self.dust_create_buffer(
        &allocator,
        rb_bytes,
        DustMem::HostRead,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        &alloc::format!("DustPyramidReadback_{}", pe.0),
      ) {
        Ok(x) => x,
        Err(e) => {
          let mut a = pyramid.alloc;
          unsafe { allocator.destroy_buffer(pyramid.buffer, &mut a) };
          return Err(e);
        }
      };
      aethervk_oshal_rlib::log!(
        "[Dust] view {}: pyramid {}x{} ({} levels, {} MB)",
        pe.0,
        layout.width,
        layout.height,
        layout.level_count(),
        bytes / (1024 * 1024)
      );
      // the timestamp queries: a free pool, else a new one (the graphics queue must count time)
      let ts_pool = match mgr.free_query_pools.lock().pop() {
        Some(p) => p,
        None if self.query_result.graphics_timestamp_bits > 0
          && self.query_result.physical_device_properties.limits.timestamp_period > 0.0 =>
        {
          let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(DUST_READBACK_SLOTS as u32 * DUST_TIMESTAMPS);
          match unsafe { self.device.create_query_pool(&info, None) } {
            Ok(p) => {
              mgr.query_pools.lock().push(p);
              p
            }
            Err(e) => {
              aethervk_oshal_rlib::log!("[Dust] view {}: no timestamp queries ({e:?})", pe.0);
              vk::QueryPool::null()
            }
          }
        }
        None => vk::QueryPool::null(),
      };
      mgr.views.insert(
        pe,
        DustView {
          layout: layout.clone(),
          vqueue_records,
          spill_slots: if cpu { 0 } else { spill_slots },
          image_words: if cpu { 0 } else { image_words },
          pyramid: Some(pyramid),
          cpu_pyramids,
          pyramid_mapped: pmapped,
          readback: Some(readback),
          readback_mapped: rmapped,
          ts_pool,
          host: spin::Mutex::new(DustViewHost::default()),
          frame_words: spin::Mutex::new([0; dust::PYRAMID_HEADER_WORDS as usize]),
          splat_records: spin::Mutex::new([
            (u64::MAX, alloc::vec::Vec::new()),
            (u64::MAX, alloc::vec::Vec::new()),
          ]),
        },
      );
    }
    let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view absent"))?;
    let frame_no = view.host.lock().frame;
    let (pyramid, pyramid_mapped) = if cpu {
      view.cpu_pyramid(frame_no).ok_or(gpu_err!("dust view has no pyramid"))?
    } else {
      (
        view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?,
        core::ptr::null_mut(),
      )
    };
    // fold the readback of DUST_READBACK_SLOTS frames ago
    let (lvl, lvl_off, lw, lh) = DustView::white_level(&view.layout);
    let rb_words = DustView::readback_words(&view.layout) as usize;
    // the view's cadence (the budget loop runs with it) and its newest fresh count (taken
    // before the host lock: the two locks are never held together here)
    let (cad_active, cad_fresh, cad_members, cad_frames) = match mgr.cadence.get(&pe) {
      Some(c) => {
        let c = c.lock();
        let frames: [usize; PROBE_CLASSES] = core::array::from_fn(|k| c.cfg.classes.get(k).map_or(1, |&n| n as usize));
        (true, c.last_state.as_ref().map(|s| s[cadence::CADS_FRESH]), c.work_members(), frames)
      }
      None => (false, None, None, [1; PROBE_CLASSES]),
    };
    let mut host = view.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
    // the regime the frame read back now was drawn in (0 full, 1 cadenced)
    let regime_rb = host.slot_regime[slot];
    if host.slot_valid[slot] && !cpu {
      if let Some(rb) = view.readback.as_ref() {
        res.allocator.allocator.as_allocator_view().invalidate_allocation(
          &rb.alloc,
          (slot * rb_words * 4) as u64,
          (rb_words * 4) as u64,
        )?;
        let words = unsafe {
          core::slice::from_raw_parts(
            view.readback_mapped.add(slot * rb_words * 4).cast::<u32>(),
            rb_words,
          )
        };
        let unit = host.slot_unit[slot];
        let (header, rest) = words.split_at(dust::PYRAMID_HEADER_WORDS as usize);
        let (rest, split_hdrs) = rest.split_at(rest.len() - 2 * dust::DUST_VQUEUE_HEADER_WORDS as usize);
        let (lqueue_hdr, cqueue_hdr) = split_hdrs.split_at(dust::DUST_VQUEUE_HEADER_WORDS as usize);
        let (rest, vqueue) = rest.split_at(rest.len() - dust::DUST_VQUEUE_HEADER_WORDS as usize);
        let (level, queue) = rest.split_at(rest.len() - dust::DUST_QUEUE_HEADER_WORDS as usize);
        // the counters by target (the probe after the builds; fresh = the frame's less them)
        host.targets = if host.slot_probe[slot] {
          let off = DUST_READBACK_SLOTS * rb_words * 4 + slot * DUST_PROBE_SLOT_WORDS as usize * 4;
          res.allocator.allocator.as_allocator_view().invalidate_allocation(&rb.alloc, off as u64, DUST_PROBE_SLOT_WORDS as u64 * 4)?;
          let p = unsafe { core::slice::from_raw_parts(view.readback_mapped.add(off).cast::<u32>(), DUST_PROBE_SLOT_WORDS as usize) };
          // the class probes are cumulative (the counters start the builds at 0): class k's work is
          // probe k less probe k − 1, the builds' total the last probe
          let at = |k: usize, i: usize| p[k * DUST_PROBE_WORDS as usize + i];
          let last = PROBE_CLASSES - 1;
          let builds: [u32; 3] = core::array::from_fn(|i| at(last, i));
          let by_class = core::array::from_fn(|k| core::array::from_fn(|i| if k == 0 { at(0, i) } else { at(k, i).wrapping_sub(at(k - 1, i)) }));
          let tot: [u32; 3] = core::array::from_fn(|i| header[DUST_PROBE_HEADER_WORDS[i] as usize]);
          Some(TargetCounters { builds, by_class, fresh: core::array::from_fn(|i| tot[i].wrapping_sub(builds[i])) })
        } else {
          None
        };
        // the feedback of the regime the frame was drawn in: full draws and cadenced frames learn
        // apart (cheap cadenced frames never lower a full draw's biases); the cadence off loads
        // and stores regime 0 only: today's single state. A cadenced frame learns from its
        // full-draw estimate (`full_frame_estimate`), not its own fraction of the work
        // (`AETHERVK_DUST_FEEDBACK_RAW=1`: the raw counters, the A/B)
        let estimate = match (regime_rb, host.targets, cad_members) {
          (1, Some(t), Some((built, shown)))
            if !aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_FEEDBACK_RAW").is_some_and(|v| v.trim() == "1") =>
          {
            let rate = host.build_rate.get_or_insert_with(BuildRate::default);
            rate.fold(t.by_class, built, cad_frames);
            let mut h = header.to_vec();
            for (i, v) in full_frame_estimate(t.fresh, rate, shown).into_iter().enumerate() {
              h[DUST_PROBE_HEADER_WORDS[i] as usize] = v;
            }
            Some(h)
          }
          (1, ..) => None,
          _ => {
            // a full draw: the next cadenced run measures its builds afresh
            host.build_rate = None;
            None
          }
        };
        // `AETHERVK_DUST_FEEDBACK_LOG=1`: each readback's regime, the counters by target and class,
        // the members per class and the EWA texels absorbed (raw and estimated)
        if aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_FEEDBACK_LOG").is_some_and(|v| v.trim() == "1") {
          let ewa = DUST_PROBE_HEADER_WORDS[1] as usize;
          aethervk_oshal_rlib::log!(
            "[Dust feedback] frame {} slot {slot}: regime {regime_rb}, EWA raw {} absorbed {}, targets {:?}, members (built, shown) {:?}, rate {:?}",
            host.frame,
            header[ewa],
            estimate.as_deref().map_or(header[ewa], |h| h[ewa]),
            host.targets.map(|t| (t.by_class.map(|c| c[1]), t.fresh[1])),
            cad_members,
            host.build_rate.as_ref().map(|r| (r.builds.map(|c| c[1] as u64), r.built.map(|x| x as u64)))
          );
        }
        host.feedback_load(regime_rb);
        host.readback_haze = host.slot_haze[slot];
        host.absorb_tag = (regime_rb, host.ts_reused[slot]);
        host.absorb_refined = host.slot_refined[slot];
        let nucleus_px = host.slot_nucleus_px[slot];
        host.absorb_at(estimate.as_deref().unwrap_or(header), level, lw, 1 << lvl, unit, nucleus_px);
        host.feedback_store(regime_rb);
        host.last_regime = regime_rb;
        // the packet queue of that frame: records drawn, packets appended (the rest in-thread)
        // (header word 0 is the dispatch's x, at most `DUST_QUEUE_DISPATCH_X`: not the count)
        let held = dust::dust_queue_held(queue);
        host.queue_counts = (held, queue[3]);
        if aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_QUEUE_LOG").is_some_and(|v| v.trim() == "1") {
          aethervk_oshal_rlib::log!("[Dust] view {}: queue {} records drawn of {} appended", pe.0, held, queue[3]);
        }
        // the volume packet queue of that frame: the consumer's records, the overflowed ones
        // (sent to the volume spill: drawn from it, or lost), and the packets per cluster that
        // size the next frame's chunks
        let (vdrawn, voverflowed) = (vqueue[dust::DUST_VQUEUE_DRAWN], vqueue[dust::DUST_VQUEUE_OVERFLOWED]);
        let (vspilled, vlost) = (vqueue[dust::DUST_VQUEUE_SPILL_DRAWN], vqueue[dust::DUST_VQUEUE_LOST]);
        host.vqueue_counts = (vdrawn, voverflowed, host.slot_chunks[slot]);
        host.vspill_counts = (vspilled, vlost);
        // never silent: a record the spill refused was not drawn
        if vlost > 0 {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: volume spill lost {} of {} refused records ({} drawn from the spill, {} by the queue): the image misses them",
            pe.0, vlost, voverflowed, vspilled, vdrawn
          );
        }
        let clusters = host.slot_clusters[slot];
        // the splitter family (perf9 steps Z / O), never silent: a leaf the leaf queue and its spill
        // refused (the frame's count) was not drawn; nor were the candidates of a cluster the
        // candidate buffer refused (its header is reset per chunk: the last chunk's count)
        let llost = lqueue_hdr[dust::DUST_LQUEUE_LOST];
        if llost > 0 {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: leaf queue lost {} leaves (the queue and its spill full): the image misses them",
            pe.0, llost
          );
        }
        let clost = cqueue_hdr[dust::DUST_CQUEUE_LOST];
        if clost > 0 {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: candidate buffer refused {} candidates in the last chunk ({} reserved): the image misses them",
            pe.0, clost, cqueue_hdr[3]
          );
        }
        // the splitter family's chunk feedback: the frame's leaves and candidates per cluster
        let (lframe, cframe) =
          (lqueue_hdr[dust::DUST_LQUEUE_FRAME_LEAVES], lqueue_hdr[dust::DUST_LQUEUE_FRAME_CANDIDATES]);
        // fast attack (a higher demand taken at once: a refused run loses its cluster), slow decay;
        // the candidates by the larger of the frame's mean and its densest chunk, doubled after a
        // frame that refused any
        let refused = lqueue_hdr[dust::DUST_LQUEUE_FRAME_REFUSED];
        if clusters > 0 && lframe > 0 {
          let l = lframe as f32 / clusters as f32;
          host.lqueue_lpc = if l > host.lqueue_lpc { l } else { host.lqueue_lpc + (l - host.lqueue_lpc) * 0.25 };
        }
        if clusters > 0 && cframe > 0 {
          let c = (cframe as f32 / clusters as f32).max(lqueue_hdr[dust::DUST_LQUEUE_FRAME_CPC_MAX] as f32);
          host.cqueue_cpc = if c > host.cqueue_cpc { c } else { host.cqueue_cpc + (c - host.cqueue_cpc) * 0.25 };
          if refused > 0 {
            host.cqueue_cpc *= 2.0;
          }
        }
        if refused > 0 {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: candidate buffer refused {} of {} candidates this frame (densest chunk {} a cluster): those clusters are missing; next estimate {:.1} a cluster",
            pe.0, refused, cframe, lqueue_hdr[dust::DUST_LQUEUE_FRAME_CPC_MAX], host.cqueue_cpc
          );
        }
        if aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE_LOG").is_some_and(|v| v.trim() == "1") {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: leaves {} (large {}, spilled {}), candidates {} over {} clusters (estimates {:.1} / {:.1} a cluster)",
            pe.0, lframe, lqueue_hdr[dust::DUST_LQUEUE_FRAME_LARGE], lqueue_hdr[dust::DUST_LQUEUE_FRAME_SPILLED],
            cframe, clusters, host.lqueue_lpc, host.cqueue_cpc
          );
        }
        if clusters > 0 && vdrawn + voverflowed > 0 {
          let ppc = (vdrawn + voverflowed) as f32 / clusters as f32;
          host.vqueue_ppc += (ppc - host.vqueue_ppc) * 0.25;
        }
        if aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE_LOG").is_some_and(|v| v.trim() == "1") {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: vqueue {} packets drawn by the consumer, {} overflowed, {} chunks over {} clusters (ppc estimate {:.1})",
            pe.0, vdrawn, voverflowed, host.slot_chunks[slot], clusters, host.vqueue_ppc
          );
          // (the `.prof` consumer only: 0 0 in production; the mode is `AETHERVK_DUST_PROF_PAIR`)
          let off = dust::PYR_PROF_PAIR as usize;
          let pair_mode = (header[dust::PYR_FLAGS as usize] >> dust::DUST_FRAME_PROF_PAIR_SHIFT) & 3;
          let what = ["offscreen packets, ×256 lane-cycles to their return", "allTop packets, ×256 lane-cycles",
            "allTop pieces, SMALL ones", "merge-eligible packets, ×256 lane-cycles"][pair_mode as usize];
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: profile pair {} ({}): {} {} of {} packets",
            pe.0, pair_mode, what, header[off], header[off + 1], vdrawn + voverflowed
          );
        }
      }
    }
    host.slot_valid[slot] = false;
    // the GPU timings of the frame recorded in this slot DUST_READBACK_SLOTS frames ago, then the
    // slot's queries reset for this frame (outside the render pass)
    if view.ts_pool != vk::QueryPool::null() {
      let written = host.ts_written[slot];
      if written != 0 {
        let mut raw = [[0u64; 2]; DUST_TIMESTAMPS as usize];
        let ok = unsafe {
          (self.device.fp_v1_0().get_query_pool_results)(
            self.device.handle(),
            view.ts_pool,
            slot as u32 * DUST_TIMESTAMPS,
            DUST_TIMESTAMPS,
            core::mem::size_of_val(&raw),
            raw.as_mut_ptr().cast(),
            16,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
          )
        };
        if ok == vk::Result::SUCCESS || ok == vk::Result::NOT_READY {
          let bits = self.query_result.graphics_timestamp_bits;
          let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
          let ts: [Option<u64>; DUST_TIMESTAMPS as usize] = core::array::from_fn(|i| {
            (written & (1 << i) != 0 && raw[i][1] != 0).then_some(raw[i][0] & mask)
          });
          let period = self.query_result.physical_device_properties.limits.timestamp_period;
          let mut t = DustGpuTimings::from_timestamps(&ts, period);
          if !t.total_ms.is_nan() {
            t.frame = host.ts_frame[slot];
            t.reused = host.ts_reused[slot];
            t.width = view.layout.width;
            t.height = view.layout.height;
            if let Some(prev) = host.timings.filter(|p| !p.total_ema_ms.is_nan()) {
              t.total_ema_ms = prev.total_ema_ms + (t.total_ms - prev.total_ema_ms) / 8.0;
            }
            host.timings = Some(t);
            if !t.reused {
              host.timings_drawn = Some(t);
            }
            // the budget loop (step 5): drawn cadenced frames only (a full draw is the known
            // transient: a reset never moves a knob)
            if !t.reused && regime_rb == 1 {
              let now = host.frame;
              let d_max = host.shell.d_max_m;
              if let Some(b) = host.budget.as_mut() {
                let s = dust::cadence::BudgetSample { frame: t.frame, total_ms: t.total_ms, fresh_ms: t.splat_only_ms, fresh: cad_fresh.unwrap_or(0) };
                if b.observe(&s, now) {
                  let k = b.knobs();
                  aethervk_oshal_rlib::log!(
                    "[Dust budget] view {}: {:.2} ms against {} ms: ε {} px, lod {} px, budgets ÷{}, fallback {} (N level {}){}",
                    pe.0, t.total_ms, b.budget_ms, k.eps_px, k.lod_px, 1u32 << k.budget_shift, k.fallback, k.n_level,
                    if k.saturated { ", every knob at its ceiling" } else { "" }
                  );
                }
                if specks_enabled() {
                  b.observe_specks(t.specks_ms, d_max);
                }
              }
            }
          }
        }
      }
      // reset, then this frame's start (the host lock is held: written here, not through
      // `cmd_dust_timestamp`)
      unsafe {
        self.device.cmd_reset_query_pool(cmd, view.ts_pool, slot as u32 * DUST_TIMESTAMPS, DUST_TIMESTAMPS);
        self.device.synchronization2.cmd_write_timestamp2(
          cmd,
          vk::PipelineStageFlags2::ALL_COMMANDS,
          view.ts_pool,
          slot as u32 * DUST_TIMESTAMPS + DUST_TS_BEGIN,
        );
      }
      host.ts_written[slot] = 1 << DUST_TS_BEGIN;
      host.ts_frame[slot] = host.frame;
      host.ts_slot = Some(slot);
    }
    // this frame's regime until `Device::dust_budget_frame_kind` says otherwise: cadenced with the
    // cadence on (reused frames included), full without it
    host.slot_regime[slot] = if cad_active { 1 } else { 0 };
    host.slot_probe[slot] = false;
    // the budget loop lives with the cadence's diagnostic path (`AETHERVK_DUST_CADENCE=1`, or
    // `AETHERVK_DUST_BUDGET_LOOP=1` with the setting): the setting alone keeps its ε
    let loop_wanted = ["AETHERVK_DUST_CADENCE", "AETHERVK_DUST_BUDGET_LOOP"]
      .iter()
      .any(|n| aethervk_oshal_rlib::os::env::var(n).is_some_and(|v| v.trim() == "1"));
    if cad_active && host.budget.is_none() && !cpu && loop_wanted {
      host.budget = dust::cadence::BudgetLoop::from_env();
    } else if !cad_active {
      host.budget = None;
    }
    // the biases this frame begins with: the cadenced regime's with the cadence on (the full one
    // until a cadenced frame was measured); a full-draw frame gets the full regime's back
    host.feedback_load(if cad_active { 1 } else { 0 });
    // the play pin (handoff P3b): over the regime's biases while the simulation runs
    host.apply_play_pin();
    let budget_knobs = host.budget.as_ref().map(|b| b.knobs());
    let shell_cap = host.budget.as_ref().and_then(|b| b.shell_cap_m);
    let unit = host.unit();
    host.slot_unit[slot] = unit;
    host.slot_haze[slot] = dust::haze_detail_from_flags(flags);
    host.slot_nucleus_px[slot] = host.nucleus_px_next;
    host.view_flags = flags;
    let (white, softening) = host.display();
    let white_measured = host.display_white_measured();
    let p50 = host.p50;
    let shell = host.shell;
    let shell_override = host.shell_override;
    let legacy = host.legacy_kernel
      || aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_LEGACY_KERNEL").is_some_and(|v| v.trim() == "1");
    let deep_bits = if host.deep_chord.unwrap_or_else(|| {
      aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_DEEP_CHORD").is_some_and(|v| v.trim() == "1")
    }) {
      dust::DUST_FRAME_DEEP_CHORD
    } else {
      0
    };
    let lod_px = host.lod_px.unwrap_or_else(|| {
      aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_LOD_PX")
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(budget_knobs.map_or(dust::DUST_LOD_PX_DEFAULT, |k| k.lod_px))
    });
    // static refinement (`dust::dust_refine_factor`): unchanged draw inputs since the last frame
    // step the EWA bias below the budget's; a changed frame returns to it
    let refine_factor = if cpu { 0 } else { dust::dust_refine_factor() };
    let same_inputs = inputs.is_some() && host.drawn.as_ref().is_some_and(|(i, _)| Some(*i) == inputs);
    if host.refine_update(same_inputs, refine_factor) {
      aethervk_oshal_rlib::log!(
        "[Dust refine] view {}: static, EWA bias {} for the next frames (budget's {}, image of the last readback {:?})",
        pe.0,
        host.ewa_bias.saturating_sub(host.refine),
        host.ewa_bias,
        host.refine_seen
      );
    }
    let refine = host.refine;
    let split_depth = host.split_depth.min(split_depth_cap());
    // `AETHERVK_DUST_VOLUME_BIAS=<n>`: a fixed volumetric level bias (diagnostics: measures what the
    // frame budget's coarsening costs in accuracy; the per-packet evaluation cap still bounds work)
    let volume_bias = host
      .volume_bias_override
      .or_else(|| {
        aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VOLUME_BIAS").and_then(|v| v.trim().parse::<u32>().ok())
      })
      .unwrap_or(host.volume_bias)
      .min(255);
    // `AETHERVK_DUST_EWA_BIAS=<n>`: a fixed EWA bias (diagnostics)
    let ewa_bias = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_EWA_BIAS")
      .and_then(|v| v.trim().parse::<u32>().ok())
      .unwrap_or(host.ewa_bias.saturating_sub(refine))
      .min(dust::DUST_EWA_BIAS_MAX);
    let (queue_mode, queue_capacity) = (host.queue_mode, host.queue_capacity);
    let gates_hook = host.frame_gates_off;
    // the volume packet queue's mode for this frame (`dust::DUST_FRAME_VQUEUE_OFF`): a test's
    // override, else `AETHERVK_DUST_VQUEUE=0`; the forced chunk (`AETHERVK_DUST_VQUEUE_CHUNK`)
    let vqueue_bits = host.vqueue_mode.unwrap_or_else(|| {
      match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE").as_deref().map(str::trim) {
        Some("0") => dust::DUST_FRAME_VQUEUE_OFF,
        _ => 0,
      }
    });
    host.vqueue_off = vqueue_bits & dust::DUST_FRAME_VQUEUE_OFF != 0;
    if host.vqueue_mode.is_none()
      && let Some(c) = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE_CHUNK")
        .and_then(|v| v.trim().parse::<u32>().ok())
    {
      host.vqueue_chunk = c;
    }
    host.vqueue_chunks_frame = 0;
    host.vqueue_clusters_frame = 0;
    let vqueue_capacity = if host.vqueue_capacity > 0 {
      host.vqueue_capacity.min(view.vqueue_records)
    } else {
      view.vqueue_records
    };
    drop(host);
    let _ = (lvl_off, lw, lh);
    // the header: level table + this frame's view words
    let mut header = view.layout.header();
    // the packet queue's mode (`dust::DUST_FRAME_QUEUE_OFF` / `_ALL`): a test's override, else
    // `AETHERVK_DUST_QUEUE=0` (off, the A/B) or `=all`
    let queue_bits = queue_mode.unwrap_or_else(|| {
      match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_QUEUE").as_deref().map(str::trim) {
        Some("0") => dust::DUST_FRAME_QUEUE_OFF,
        Some("all") => dust::DUST_FRAME_QUEUE_ALL,
        _ => 0,
      }
    });
    // `AETHERVK_DUST_PROF_PAIR=0..3`: what the `.prof` consumer's profile pair counts
    let prof_pair_bits = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PROF_PAIR")
      .and_then(|v| v.trim().parse::<u32>().ok())
      .map_or(0, |m| (m & 3) << dust::DUST_FRAME_PROF_PAIR_SHIFT);
    header[dust::PYR_FLAGS as usize] = flags | queue_bits | vqueue_bits | deep_bits | prof_pair_bits;
    // `AETHERVK_DUST_VOLUME_AXIS=chord|time|grid`: the volumetric packets' piece axes (bits 16–17)
    let volume_axis = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VOLUME_AXIS")
      .map(|v| dust::volume_axis_from_str(&v))
      .unwrap_or(dust::DUST_VOLUME_AXIS_CHORD);
    // `AETHERVK_DUST_FRAME_GATES=0`: the in-frame gates off (diagnostics: unbounded frame work)
    let gates_off = gates_hook
      || aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_FRAME_GATES").is_some_and(|v| v.trim() == "0");
    header[dust::PYR_SPLIT_DEPTH as usize] = split_depth
      | (volume_bias << 8)
      | (volume_axis << 16)
      | (ewa_bias << 18)
      | ((gates_off as u32) << dust::DUST_FRAME_GATES_OFF_BIT)
      | ((dust::jacobian_at_start() as u32) << 23)
      // `AETHERVK_DUST_HARD_AXIS=<n>` (diagnostics): hard-edged Gaussianised axes, n pieces
      | (aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_HARD_AXIS")
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(0)
        .min(15)
        << 24)
      // `AETHERVK_DUST_TIME_CELL=1` (diagnostics): time Gaussians centred on their cells
      | ((aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_TIME_CELL").is_some_and(|v| v.trim() == "1")
        as u32)
        << 28)
      // the kernel before the cone of grains (A/B measurements on one build)
      | ((legacy as u32) << dust::DUST_FRAME_LEGACY_KERNEL_BIT)
      // `AETHERVK_DUST_TIMING_PROBE=plain_add|no_volume_writes|no_volume` (GPU timing diagnostics)
      | aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_TIMING_PROBE").map_or(0, |v| dust::timing_probe_bits(&v));
    // the speck shell of this frame (`AETHERVK_DUST_SPECKS=1` turns the specks on; a test's or a
    // diagnostic's override wins) and the budget (`AETHERVK_DUST_SPECKS_BUDGET`); the size law is
    // per system, in the speck push constants
    let d_max = match shell_override {
      Some(d) => d,
      None if specks_enabled() => shell_cap.map_or(shell.d_max_m, |c| c.min(shell.d_max_m)),
      None => 0.0,
    };
    SpeckHeader::write(&mut header, d_max, specks_budget());
    header[dust::PYR_T_HI as usize] = flow.t_hi.to_bits();
    header[dust::PYR_T_LO as usize] = flow.t_lo.to_bits();
    // tracer dot: a fixed level of the white point, in counts (area 1 px²)
    let tracer_counts = if white > 0.0 {
      (dust::TRACER_LEVEL * white / unit).min(dust::DUST_COUNT_MAX_PER_ADD as f32)
    } else {
      0.0
    };
    header[dust::PYR_TRACER_COUNTS as usize] = tracer_counts.to_bits();
    // the cluster level of detail (`dust::PYR_LOD`: pixels per drawn segment, 0 = off; the word is
    // in the reuse fingerprint, so a changed knob redraws)
    header[dust::PYR_LOD as usize] = lod_px.to_bits();
    // a refined frame is over the EWA budget by design: its in-frame gate moves out of the way
    // (half the refinement's factor in hard-gate units: twice the work it plans)
    header[dust::PYR_EWA_GATE_SCALE as usize] = if refine > 0 { (refine_factor / 2).max(1) } else { 0 };
    // the unit this frame's counts are written with, for dump readers (`download_dust_pyramid`)
    header[dust::PYR_UNIT as usize] = unit.to_bits();
    // where the queues start, for the shaders (`dust_queue_base`): with no class images the
    // value their former formula gave (measure offset + measure words = total_words)
    header[dust::PYR_QUEUE_BASE as usize] = view.queue_word();
    // the white point and softening this frame is shown with (dump readers)
    header[dust::PYR_WHITE as usize] = white.to_bits();
    header[dust::PYR_SOFTENING as usize] = softening.to_bits();
    *view.frame_words.lock() = header;
    // the same draw inputs and header as the last fully drawn frame: its pyramid is this frame's
    let compared = reuse_header(&header, flags);
    let reuse = {
      let mut host = view.host.lock();
      let reuse = !cpu
        && inputs.is_some()
        && host.drawn.as_ref().is_some_and(|(i, h)| Some(*i) == inputs && *h == compared);
      host.drawn = if cpu { None } else { inputs.map(|i| (i, compared)) };
      host.ts_reused[slot] = reuse;
      if !reuse {
        host.image_refined = refine > 0;
      }
      host.slot_refined[slot] = host.image_refined;
      host.queue_pending = !cpu && !reuse;
      reuse
    };
    if reuse {
      // the image is the last drawn frame's: its splat records are the complete ones
      // (`download_dust_snapshot` reads them; a reused frame records none)
      {
        let mut r = view.splat_records.lock();
        if !r[0].1.is_empty() {
          r[1] = core::mem::replace(&mut r[0], (u64::MAX, alloc::vec::Vec::new()));
        }
      }
      // only the display words change (`PYR_WHITE`, `PYR_SOFTENING`: adjacent)
      const _: () = assert!(dust::PYR_SOFTENING == dust::PYR_WHITE + 1);
      unsafe {
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          dust::PYR_WHITE as u64 * 4,
          bytemuck::cast_slice(&[white.to_bits(), softening.to_bits()]),
        );
      }
    } else if cpu {
      // host-written pyramid: zero it and write the header now; the splats follow on the host
      let words = unsafe {
        core::slice::from_raw_parts_mut(
          pyramid_mapped.cast::<u32>(),
          view.layout.total_words as usize,
        )
      };
      words.fill(0);
      words[..dust::PYRAMID_HEADER_WORDS as usize].copy_from_slice(&header);
    } else {
      unsafe {
        self
          .device
          .cmd_update_buffer(cmd, pyramid.buffer, 0, bytemuck::cast_slice(&header));
        self.device.cmd_fill_buffer(
          cmd,
          pyramid.buffer,
          dust::PYRAMID_HEADER_WORDS as u64 * 4,
          (view.layout.total_words - dust::PYRAMID_HEADER_WORDS) as u64 * 4,
          0,
        );
        // the packet queue: empty, its dispatch (0, 1, 1), the capacity override
        let mut q = [0u32; dust::DUST_QUEUE_HEADER_WORDS as usize];
        q[1] = 1;
        q[2] = 1;
        q[4] = queue_capacity;
        // the splitter family (perf9 steps Z / O): the leaf queue, the leaf spill and the large
        // leaves' list, in words from this header (`dust_lqueue_base` and the others)
        let split_offsets = dust::dust_queue_offsets(view.vqueue_records);
        q[dust::DUST_QUEUE_LQUEUE_OFFSET] = split_offsets.lqueue;
        q[dust::DUST_QUEUE_LSPILL_OFFSET] = split_offsets.lspill;
        q[dust::DUST_QUEUE_BLIST_OFFSET] = split_offsets.blist;
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          view.queue_word() as u64 * 4,
          bytemuck::cast_slice(&q),
        );
        // the volume packet queue: the frame's counters zero (the chunk words are written per
        // chunk by `cmd_dust_splat`)
        let mut vq = [0u32; dust::DUST_VQUEUE_HEADER_WORDS as usize];
        vq[1] = 1;
        vq[2] = 1;
        vq[4] = vqueue_capacity;
        // the records this view allocated: the shaders' sub-packet queue and spill follow them
        vq[dust::DUST_VQUEUE_RECORDS] = view.vqueue_records;
        // the volume spill's place (`dust_vspill_base`); its counts (words 13-14) zero
        let offsets = dust::dust_queue_offsets(view.vqueue_records);
        vq[dust::DUST_VQUEUE_SPILL_OFFSET] = offsets.vspill - offsets.vqueue;
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          (view.queue_word() + dust::DUST_QUEUE_WORDS) as u64 * 4,
          bytemuck::cast_slice(&vq),
        );
        // track 9A (t9a2d): the deferred edge pieces' two regions: empty, their dispatch (0, 1, 1),
        // their capacity, the frame's counts (words 5-6) zero (words 0-4 again per chunk)
        let mut ve = [0u32; dust::DUST_VQUEUE_HEADER_WORDS as usize];
        ve[1] = 1;
        ve[2] = 1;
        ve[4] = offsets.vedge_records;
        for base in offsets.vedge {
          self.device.cmd_update_buffer(
            cmd,
            pyramid.buffer,
            (view.queue_word() + base) as u64 * 4,
            bytemuck::cast_slice(&ve),
          );
        }
        // the splitter family (perf9 steps Z / O): the leaf queue, its spill and the large-leaf
        // list empty, their lost counts (word 5) zero for the frame; the leaf queue's words 6-7 the
        // candidate buffer's and the screen-packet slots' places (words from the EWA queue base);
        // the candidate buffer empty, its word 0 (the indirect x) and word 3 zero (again per chunk)
        let mut lq = [0u32; dust::DUST_VQUEUE_HEADER_WORDS as usize];
        lq[1] = 1;
        lq[2] = 1;
        lq[4] = offsets.lqueue_records;
        lq[dust::DUST_LQUEUE_CQUEUE_OFFSET] = offsets.cqueue;
        lq[dust::DUST_LQUEUE_PQUEUE_OFFSET] = offsets.pqueue;
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          (view.queue_word() + offsets.lqueue) as u64 * 4,
          bytemuck::cast_slice(&lq),
        );
        lq[dust::DUST_LQUEUE_CQUEUE_OFFSET] = 0;
        lq[dust::DUST_LQUEUE_PQUEUE_OFFSET] = 0;
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          (view.queue_word() + offsets.lspill) as u64 * 4,
          bytemuck::cast_slice(&lq),
        );
        lq[4] = 2 * offsets.lqueue_records;
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          (view.queue_word() + offsets.blist) as u64 * 4,
          bytemuck::cast_slice(&lq),
        );
        lq[4] = offsets.cqueue_records;
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          (view.queue_word() + offsets.cqueue) as u64 * 4,
          bytemuck::cast_slice(&lq),
        );
      }
    }
    Ok((DustViewParams {
      pyramid: pyramid.address,
      levels: view.layout.level_count(),
      unit,
      white,
      white_measured,
      p50,
      softening,
      d_max_m: d_max,
      n_eye_m3: shell.n_eye_m3 as f32,
      specks: shell.specks,
      specks_overflow: shell.overflow,
    }, reuse))
  }

  /// The budget loop's cadence knobs into `pe`'s cadence (`frame.rs`, before
  /// `cmd_dust_cadence_begin`; temporal reuse step 5): ε and the class lengths under the
  /// fallback's N level (the longest class × 2^n). Returns the cadence fingerprint's salt (the N
  /// level: a changed class length resets the cadence the same frame, as any fingerprint change);
  /// 0 without the loop or the cadence (nothing changed).
  pub fn dust_budget_apply_cadence(&self, pe: gpu::PresentationEngineHandle) -> u64 {
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else { return 0 };
    let Some(view) = mgr.views.get(&pe) else { return 0 };
    // the knobs under the host lock, then the cadence's lock (never both held)
    let (knobs, base) = {
      let host = view.host.lock();
      let Some(b) = host.budget.as_ref() else { return 0 };
      (b.knobs(), b.base_classes.clone())
    };
    let Some(entry) = mgr.cadence.get(&pe) else { return 0 };
    let base = {
      let mut c = entry.lock();
      let base = base.unwrap_or_else(|| c.cfg.classes.clone());
      c.cfg.eps_px = knobs.eps_px;
      // the fallback's shape bound (0 off the fallback: the ε rule alone)
      c.shape_frac = knobs.shape_frac();
      let mut classes = base.clone();
      if let Some(l) = classes.last_mut() {
        *l <<= knobs.n_level;
      }
      c.cfg.classes = classes;
      base
    };
    let mut host = view.host.lock();
    if let Some(b) = host.budget.as_mut()
      && b.base_classes.is_none()
    {
      b.base_classes = Some(base);
    }
    knobs.n_level as u64
  }

  /// What the frame being recorded is for the feedback (`frame.rs`, after
  /// `cmd_dust_cadence_begin`): `full` (a reset or first frame: everything fresh) or a cadenced
  /// frame. A full frame draws with the full-draw regime's biases: header word
  /// [`dust::PYR_SPLIT_DEPTH`]'s depth / volume-bias / EWA-bias bits are rewritten before any splat
  /// (the overrides keep theirs) and its readback updates that regime.
  pub fn dust_budget_frame_kind(&self, cmd: vk::CommandBuffer, pe: gpu::PresentationEngineHandle, full: bool) {
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else { return };
    let Some(view) = mgr.views.get(&pe) else { return };
    let Some(pyramid) = view.pyramid.as_ref() else { return };
    let mut host = view.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
    host.slot_regime[slot] = if full { 0 } else { 1 };
    if !full || crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      return;
    }
    let f = host.full_frame_feedback();
    let env_u32 = |k: &str| aethervk_oshal_rlib::os::env::var(k).and_then(|v| v.trim().parse::<u32>().ok());
    let mut words = view.frame_words.lock();
    let w = words[dust::PYR_SPLIT_DEPTH as usize];
    let mut n = (w & !0xFF) | f.split_depth.min(split_depth_cap());
    if host.volume_bias_override.is_none() && env_u32("AETHERVK_DUST_VOLUME_BIAS").is_none() {
      n = (n & !(0xFF << 8)) | (f.volume_bias.min(255) << 8);
    }
    if env_u32("AETHERVK_DUST_EWA_BIAS").is_none() {
      n = (n & !(0xF << 18)) | (f.ewa_bias.saturating_sub(host.refine).min(dust::DUST_EWA_BIAS_MAX) << 18);
    }
    if n == w {
      return;
    }
    words[dust::PYR_SPLIT_DEPTH as usize] = n;
    host.restored += 1;
    // after the frame begin's header write (transfer → transfer); the splats read it after the
    // frame's pre-splat barrier
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b)));
      self.device.cmd_update_buffer(cmd, pyramid.buffer, dust::PYR_SPLIT_DEPTH as u64 * 4, bytemuck::cast_slice(&[n]));
    }
  }

  /// The work counters after class `k`'s build draws ([`DUST_PROBE_HEADER_WORDS`]) into this
  /// frame's probe `k` (after the readback slots of the view's readback buffer); read with the
  /// frame's readback as [`TargetCounters`] (`frame.rs`: every class `0..PROBE_CLASSES`, in order,
  /// each right after that class's `cmd_dust_cadence_builds`, an empty class included)
  pub fn cmd_dust_budget_probe(&self, cmd: vk::CommandBuffer, pe: gpu::PresentationEngineHandle, k: usize) {
    debug_assert!(k < PROBE_CLASSES);
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else { return };
    let Some(view) = mgr.views.get(&pe) else { return };
    let (Some(pyramid), Some(rb)) = (view.pyramid.as_ref(), view.readback.as_ref()) else { return };
    if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      return;
    }
    let slot = {
      let mut host = view.host.lock();
      let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
      host.slot_probe[slot] = true;
      slot
    };
    let base = DUST_READBACK_SLOTS as u64 * DustView::readback_words(&view.layout) as u64 * 4
      + slot as u64 * DUST_PROBE_SLOT_WORDS as u64 * 4
      + k as u64 * DUST_PROBE_WORDS as u64 * 4;
    let regions: [vk::BufferCopy; 3] = core::array::from_fn(|i| {
      vk::BufferCopy::default().src_offset(DUST_PROBE_HEADER_WORDS[i] as u64 * 4).dst_offset(base + i as u64 * 4).size(4)
    });
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b)));
      self.device.cmd_copy_buffer(cmd, pyramid.buffer, rb.buffer, &regions);
    }
    // the fresh pass's atomics on these words come after the frame's pre-splat barrier (its
    // source stages include TRANSFER: the write-after-read is ordered)
  }

  /// The budget loop's telemetry of `pe` (`None` without a view)
  pub fn dust_budget_telemetry(&self, pe: gpu::PresentationEngineHandle) -> Option<DustBudgetTelemetry> {
    let cad = self.dust_cadence_telemetry(pe);
    let res = self.res.read();
    let view = res.dust_manager.as_ref()?.views.get(&pe)?;
    let host = view.host.lock();
    let classes = cad.as_ref().map_or(alloc::vec::Vec::new(), |c| c.classes.clone());
    let mut t = DustBudgetTelemetry {
      budget_ms: dust::cadence::DUST_BUDGET_MS_DEFAULT,
      staleness_frames: 2 * classes.last().copied().unwrap_or(0),
      classes,
      regime: host.last_regime,
      restored: host.restored,
      fb_full: host.fb_full,
      fb_cad: host.fb_cad,
      targets: host.targets,
      cadence: cad,
      state: "off",
      ratio_ema: f64::NAN,
      ..Default::default()
    };
    if let Some(b) = host.budget.as_ref() {
      t.budget_ms = b.budget_ms;
      t.knobs = Some(b.knobs());
      t.rung = b.rung;
      t.rungs = b.ladder.len();
      t.state = b.state_name();
      t.ratio_ema = b.ema;
      t.moves = b.moves;
      t.k_target = b.k_last;
      t.shell_cap_m = b.shell_cap_m;
    }
    Some(t)
  }

  /// Sets the view's pixel solid angle (sr) for the speck shell: `(3 / (n_eye Ω_px))^{1/3}` is
  /// computed when the header is read back ([`DustViewHost::absorb`]). 0 (an orthographic camera)
  /// disables the shell. After [`Self::cmd_dust_frame_begin`] (the view exists then); it is used
  /// from the next readback on.
  pub fn dust_set_shell_camera(&self, pe: gpu::PresentationEngineHandle, omega_px_sr: f64) {
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else {
      return;
    };
    if let Some(view) = mgr.views.get(&pe) {
      view.host.lock().omega_px_sr = if omega_px_sr.is_finite() && omega_px_sr > 0.0 {
        omega_px_sr
      } else {
        0.0
      };
    }
  }

  /// Fixes the speck shell radius (local m) of view `pe` from its next frame on, whatever the
  /// measurement and `AETHERVK_DUST_SPECKS` (`None` returns to the measured shell). The view
  /// must exist (after the first [`Self::cmd_dust_frame_begin`]). Tests and diagnostics.
  pub fn dust_set_shell_override(&self, pe: gpu::PresentationEngineHandle, d_max: Option<f32>) {
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else {
      return;
    };
    if let Some(view) = mgr.views.get(&pe) {
      view.host.lock().shell_override = d_max.filter(|d| d.is_finite() && *d >= 0.0);
    }
  }

  /// Records (GPU) or performs (CPU mode) the speck pass of one tier of system `id` into the
  /// pyramid of `pe` (see `dust_specks.comp`, mirror `dust::specks::specks_tier`): after the
  /// tier's splat ([`Self::cmd_dust_splat`]), before [`Self::cmd_dust_post_splat_barrier`]; the
  /// two passes only add into the pyramid (atomics), so they need no barrier between them. The
  /// pass always runs (it measures the density at the eye for the next frame's shell); it draws
  /// only when the header's shell radius is positive.
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_specks(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    id: u64,
    ring_base: u32,
    first_slot: u32,
    capacity: u32,
    render: u64,
    moments: u64,
    mut pc: DustSpecksPushConstants,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
    let frame_no = view.host.lock().frame;
    let (pyramid, pyramid_mapped) = if sys.cpu_ring.is_some() {
      view.cpu_pyramid(frame_no).ok_or(gpu_err!("dust view has no pyramid"))?
    } else {
      (
        view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?,
        core::ptr::null_mut(),
      )
    };
    pc.render = render;
    pc.moments = moments;
    pc.pyramid = pyramid.address;
    if sys.cpu_ring.is_some() {
      let mut cpu = sys.cpu_render.lock();
      let Some((clusters, moms)) = cpu.get_mut(&ring_base) else {
        return Ok(());
      };
      let flow = sys.host.lock().flow;
      let words = unsafe {
        core::slice::from_raw_parts_mut(
          pyramid_mapped.cast::<u32>(),
          view.layout.total_words as usize,
        )
      };
      pc.live_count = pc.live_count.min(clusters.len() as u32);
      let stats = specks::specks_tier(moms, clusters, &pc, &flow, &view.layout, words);
      if stats.truncated > 0 {
        aethervk_oshal_rlib::log!(
          "[Dust specks] {} (cluster, bin) walks hit the node cap ({} drawn)",
          stats.truncated,
          stats.drawn
        );
      }
      res.allocator.allocator.as_allocator_view().flush_allocation(
        &pyramid.alloc,
        0,
        view.layout.total_words as u64 * 4,
      )?;
      return Ok(());
    }
    if pc.live_count == 0 {
      return Ok(());
    }
    // the gate pass (a thread per cluster × bin: the density at the eye, the walk gate, the
    // cluster's walk mask), then the breadth-first walk (round 9 step O): setup1 (a thread per
    // marked (cluster, bin): its record in the walk region), setup2a / 2b / 3 / 4 / 5 (a thread
    // per record, indirect from the region header), a step pass per tree level (indirect from
    // the level's args; push word 24 = the level), the many- and single-speck leaves (indirect),
    // the trace fill (the test hook). A barrier after every pass: each reads what the previous
    // one wrote, and the indirect args.
    let pipelines = &self.kernels.pipelines;
    let pipeline = pipelines.dust_specks;
    pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
    pipelines.assert_pc_size(pipelines.dust_specks_walk, core::mem::size_of_val(&pc));
    let threads = pc.live_count.saturating_mul(dust::SIZE_BINS);
    // the spill's header for this pass, before the gate (both passes read its ring words): no
    // slot claimed, the view's slot count, the trace's capacity (a test hook) and count, the
    // tier's live range start (ring slot) and ring mask (the buffers are slot indexed), the
    // cadence's owner bytes, word 8 (the former stack limit: unread), words 9-13 the walk region
    // (its first word, the capacities: records, many-speck leaves, frontier, single-speck
    // leaves). After the previous tier's walk (its region atomics) and before this tier's passes.
    let offsets = dust::dust_queue_offsets(view.vqueue_records);
    let spill_word = view.queue_word() + offsets.spill;
    let spill_offset = spill_word as u64 * 4;
    let trace_cap = if view.host.lock().specks_trace { dust::DUST_SPECKS_TRACE_CAPACITY } else { 0 };
    let stack_limit = view.host.lock().specks_stack_limit;
    // the cadence's owner bytes (SHOWN slots are not walked; 0: no cadence)
    let owner = self.cad_owner_address(mgr, pe, id, ring_base);
    let region = spill_word + dust::DUST_SPILL_HEADER_WORDS;
    let caps = dust::dust_walk_region_caps(view.spill_slots);
    // the region's header: every count and `x` 0, every `y` and `z` 1
    let mut region_header = [0u32; dust::DUST_WR_REGION_HEADER as usize];
    let mut args = |at: usize| {
      region_header[at + 1] = 1;
      region_header[at + 2] = 1;
    };
    args(0);
    args(5);
    for s in 0..dust::DUST_WR_STEPS as usize {
      args(9 + 4 * s);
    }
    args(dust::DUST_WR_LEAF1 as usize + 1);
    let to_update = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT)
      .src_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_WRITE
          | vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::INDIRECT_COMMAND_READ,
      )
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_update)),
      );
      self.device.cmd_update_buffer(
        cmd,
        pyramid.buffer,
        spill_offset,
        bytemuck::cast_slice(&[
          0u32,
          view.spill_slots,
          trace_cap,
          0,
          first_slot,
          capacity.wrapping_sub(1),
          owner as u32,
          (owner >> 32) as u32,
          stack_limit,
          region,
          caps[0],
          caps[1],
          caps[2],
          caps[3],
        ]),
      );
      self.device.cmd_update_buffer(cmd, pyramid.buffer, region as u64 * 4, bytemuck::cast_slice(&region_header));
    }
    self.cmd_dust_pre_splat_barrier(cmd);
    let push = |pc: &DustSpecksPushConstants| unsafe {
      self.device.cmd_push_constants(
        cmd,
        pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(pc),
      );
    };
    let bind = |p: vk::Pipeline| unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
    };
    let indirect = |word: u32| unsafe {
      self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, word as u64 * 4);
    };
    // the gate
    bind(pipeline);
    push(&pc);
    unsafe { self.device.cmd_dispatch(cmd, threads.div_ceil(DUST_WG), 1, 1) };
    // the gate's walk masks → setup1 (direct, the gate's threads)
    // (`AETHERVK_DUST_SETUP_PASSES=n`: a diagnostic cap on the setup dispatches, 0 = the gate only)
    let setup_passes = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SETUP_PASSES")
      .and_then(|v| v.trim().parse::<usize>().ok())
      .unwrap_or(usize::MAX);
    self.cmd_dust_walk_barrier(cmd);
    if setup_passes >= 1 {
      bind(pipelines.dust_specks_setup[0]);
      push(&pc);
      unsafe { self.device.cmd_dispatch(cmd, threads.div_ceil(DUST_WG), 1, 1) };
    }
    // setup 2a, 2b, 3, 4, 5: a thread per record (the args at region word 5)
    for p in pipelines.dust_specks_setup[1..].iter().take(setup_passes.saturating_sub(1)) {
      self.cmd_dust_walk_barrier(cmd);
      bind(*p);
      push(&pc);
      indirect(region + 5);
    }
    // the steps: level s from the frontier s & 1 (its args at region word 9 + 4 s); an empty
    // level is a zero-group dispatch
    // (`AETHERVK_DUST_WALK_STEPS=n`: a diagnostic cap on the step dispatches, the frontiers past it
    // left as they are)
    let steps = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_WALK_STEPS")
      .and_then(|v| v.trim().parse::<u32>().ok())
      .map_or(dust::DUST_WR_STEPS, |n| n.min(dust::DUST_WR_STEPS));
    bind(pipelines.dust_specks_walk);
    for s in 0..steps {
      self.cmd_dust_walk_barrier(cmd);
      let step_pc = DustSpecksPushConstants { live_count: s, ..pc };
      push(&step_pc);
      indirect(region + 9 + 4 * s);
    }
    // the leaves: many-speck (a workgroup each, args at region word 0) and single-speck (a lane
    // each, args after the single leaf count)
    self.cmd_dust_walk_barrier(cmd);
    bind(pipelines.dust_specks_leaves);
    push(&pc);
    indirect(region);
    bind(pipelines.dust_specks_leaves1);
    push(&pc);
    indirect(region + dust::DUST_WR_LEAF1 + 1);
    if trace_cap > 0 {
      // the trace fill (the test hook): a thread per trace entry
      self.cmd_dust_walk_barrier(cmd);
      bind(pipelines.dust_specks_trace);
      push(&pc);
      unsafe { self.device.cmd_dispatch(cmd, trace_cap.div_ceil(DUST_WG), 1, 1) };
    }
    Ok(())
  }

  /// Records (GPU) or performs (CPU mode) the splat pass of one tier of system `id` into the
  /// pyramid of `pe` (see `dust_splat.comp`): `render` / `moments` are the tier's buffers from
  /// [`Self::cmd_dust_propagate`], `pc` the push constants without the buffer addresses, which
  /// are filled here. After [`Self::cmd_dust_pre_splat_barrier`]. `first_slot` / `capacity`
  /// are the tier's live range start and sub-ring size as given to the propagate: the GPU
  /// buffers are slot indexed (`dust_splat.comp` maps the range offset `k` to the slot
  /// `(first_slot + k) & (capacity − 1)`); the CPU mode's are compact in ring order.
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_splat(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    id: u64,
    ring_base: u32,
    first_slot: u32,
    capacity: u32,
    render: u64,
    moments: u64,
    pc: DustSplatPushConstants,
  ) -> GpuResult<()> {
    self.dust_splat_impl(cmd, pe, id, ring_base, first_slot, capacity, render, moments, pc, None, 0, 0, None)
  }

  /// [`Self::cmd_dust_splat`] with the cadence's options: `target` a class image's address in the
  /// view buffer (`None`: the frame pyramid), `list` the address of `n_list` ring slots to draw
  /// (0: the range `[0, live_count)`; ~0 entries are holes), `owner` the owner bytes whose SHOWN
  /// slots are skipped (`None`: the view's cadence, if any; `Some(0)`: none)
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn dust_splat_impl(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
    id: u64,
    ring_base: u32,
    first_slot: u32,
    capacity: u32,
    render: u64,
    moments: u64,
    mut pc: DustSplatPushConstants,
    target: Option<u64>,
    list: u64,
    n_list: u32,
    owner: Option<u64>,
  ) -> GpuResult<()> {
    // the volume spill's drain flag is the host's own (set on the consumer's second dispatch)
    assert!(
      pc.flags & dust::DUST_VIEW_VOLUME_SPILL == 0,
      "cmd_dust_splat: the caller's flags carry the host-internal DUST_VIEW_VOLUME_SPILL bit"
    );
    assert!(
      pc.flags & dust::DUST_VIEW_EDGE_ODD == 0,
      "cmd_dust_splat: the caller's flags carry the host-internal DUST_VIEW_EDGE_ODD bit"
    );
    // the leaf spill's drain flag is the host's own (set on the leaf consumer's second dispatch)
    assert!(
      pc.flags & dust::DUST_VIEW_LEAF_SPILL == 0,
      "cmd_dust_splat: the caller's flags carry the host-internal DUST_VIEW_LEAF_SPILL bit"
    );
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
    let frame_no = view.host.lock().frame;
    let (pyramid, pyramid_mapped) = if sys.cpu_ring.is_some() {
      view.cpu_pyramid(frame_no).ok_or(gpu_err!("dust view has no pyramid"))?
    } else {
      (
        view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?,
        core::ptr::null_mut(),
      )
    };
    pc.render = render;
    pc.moments = moments;
    pc.pyramid = target.unwrap_or(pyramid.address);
    // the record of this pass (diagnostics: `download_dust_snapshot`; the frame's draws only)
    if sys.cpu_ring.is_none()
      && target.is_none()
      && let (Some(rb), Some(mb)) = (sys.render.as_ref(), sys.moments.as_ref())
    {
      let rec = DustSplatRecord {
        system: id,
        ring_base,
        render: rb.buffer,
        render_offset: render.wrapping_sub(rb.address),
        moments: mb.buffer,
        moments_offset: moments.wrapping_sub(mb.address),
        first_slot,
        capacity,
        pc,
        flow: sys.host.lock().flow,
        tracer_counts: f32::from_bits(view.frame_words.lock()[dust::PYR_TRACER_COUNTS as usize])
          as u32,
      };
      let mut r = view.splat_records.lock();
      if r[0].0 != frame_no {
        // a new frame: the one being recorded becomes the last complete one
        let cur = core::mem::replace(&mut r[0], (frame_no, alloc::vec::Vec::new()));
        r[1] = cur;
      }
      r[0].1.push(rec);
    }
    if sys.cpu_ring.is_some() {
      // CPU mode: the reference splat on the clusters propagate evaluated, straight into the
      // host-visible pyramid of this frame
      let cpu = sys.cpu_render.lock();
      let Some((clusters, moms)) = cpu.get(&ring_base) else {
        return Ok(());
      };
      let flow = sys.host.lock().flow;
      let tracer_counts = f32::from_bits(view.frame_words.lock()[dust::PYR_TRACER_COUNTS as usize]);
      let words = unsafe {
        core::slice::from_raw_parts_mut(
          pyramid_mapped.cast::<u32>(),
          view.layout.total_words as usize,
        )
      };
      pc.live_count = pc.live_count.min(clusters.len() as u32);
      dust::splat_tier(
        moms,
        clusters,
        &pc,
        &flow,
        (tracer_counts + 0.5) as u32,
        &view.layout,
        words,
      );
      res.allocator.allocator.as_allocator_view().flush_allocation(
        &pyramid.alloc,
        0,
        view.layout.total_words as u64 * 4,
      )?;
      return Ok(());
    }
    // the clusters the chunks cover: the range, or the list's entries
    let count = if list != 0 { n_list } else { pc.live_count };
    if count == 0 {
      return Ok(());
    }
    let owner = match owner {
      Some(o) => o,
      None => self.cad_owner_address(mgr, pe, id, ring_base),
    };
    // the profiler variants while the request is on (`dust_profile.glsl`; same drawing)
    let profiling = crate::gpu_backends::vulkan::physics::dust_profile_requested();
    let stages = match &self.kernels.pipelines.dust_splat_prof_stages {
      Some(p) if profiling => p,
      _ => &self.kernels.pipelines.dust_splat_stages,
    };
    let producer = stages[0];
    // perf9_K: the splat path (`Self::dust_splat_staged`): `Some` the monolithic enumerator
    // (its profiler variant while the request is on), `None` the stage chain above
    let splat_mono = if self.dust_splat_staged() {
      None
    } else {
      match self.kernels.pipelines.dust_splat_mono_prof {
        Some(p) if profiling => Some(p),
        _ => self.kernels.pipelines.dust_splat_mono,
      }
    };
    self.dust_splat_log_path(splat_mono.is_some());
    if let Some(m) = splat_mono {
      self.kernels.pipelines.assert_pc_size(m, core::mem::size_of_val(&pc));
    }
    let volume_plain = view.host.lock().volume_plain;
    let consumer = match (self.kernels.pipelines.dust_draw_volume_prof, self.kernels.pipelines.dust_draw_volume_plain) {
      (Some(p), _) if profiling => p,
      (_, Some(p)) if volume_plain => p,
      _ => self.kernels.pipelines.dust_draw_volume,
    };
    // (perf9_O C2 v5) the staged consumer, unless the consumer above is the profiler's or the plain
    // blob (the same conditions as that match)
    let staged_stages = if (profiling && self.kernels.pipelines.dust_draw_volume_prof.is_some())
      || (volume_plain && self.kernels.pipelines.dust_draw_volume_plain.is_some())
    {
      None
    } else {
      self.kernels.pipelines.dust_draw_volume_stages
    };
    let splitter = match self.kernels.pipelines.dust_split_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_split,
    };
    // (perf9_K) the split family (`dust_split_staged`): `Some` the pre-split splitter (its profiler
    // variant while the request is on; the GTX 1070 default), `None` the staged chain below
    let split_mono = if self.dust_split_staged() {
      None
    } else {
      match self.kernels.pipelines.dust_split_mono_prof {
        Some(p) if profiling => Some(p),
        _ => self.kernels.pipelines.dust_split_mono,
      }
    };
    // the splitter family (perf9 steps Z / O): PREP → walk → project passes 1-3 → SEQ order →
    // SEQ draw (`cmd_dust_split_chain`), then the leaf consumers beside the volume consumer
    let hard_axis = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_HARD_AXIS")
      .and_then(|v| v.trim().parse::<u32>().ok())
      .unwrap_or(0)
      .min(15)
      != 0;
    let split_prep = match self.kernels.pipelines.dust_split_prep_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_split_prep,
    };
    let split_walk = if hard_axis {
      match self.kernels.pipelines.dust_split_hard_prof {
        Some(p) if profiling => p,
        _ => self.kernels.pipelines.dust_split_hard,
      }
    } else {
      splitter
    };
    let project = [
      (self.kernels.pipelines.dust_project_leaf_p1, self.kernels.pipelines.dust_project_leaf_p1_prof),
      (self.kernels.pipelines.dust_project_leaf_p2, self.kernels.pipelines.dust_project_leaf_p2_prof),
      (self.kernels.pipelines.dust_project_leaf_p3, self.kernels.pipelines.dust_project_leaf_p3_prof),
    ]
    .map(|(plain, prof)| match prof {
      Some(p) if profiling => p,
      _ => plain,
    });
    let seq = match self.kernels.pipelines.dust_split_seq_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_split_seq,
    };
    let seq_draw = match self.kernels.pipelines.dust_split_seq_draw_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_split_seq_draw,
    };
    let leaf = match self.kernels.pipelines.dust_draw_leaf_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_draw_leaf,
    };
    let leaf_wg = match self.kernels.pipelines.dust_draw_leaf_wg_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_draw_leaf_wg,
    };
    let split_stages = [
      if hard_axis { None } else { Some(split_prep) },
      Some(split_walk),
      Some(project[0]),
      Some(project[1]),
      Some(project[2]),
      Some(seq),
      Some(seq_draw),
    ];
    let edge_pass = match self.kernels.pipelines.dust_draw_small_edge_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_draw_small_edge,
    };
    self.kernels.pipelines.assert_pc_size(producer, core::mem::size_of_val(&pc));
    // (perf9_K) the mono draw family (`dust_draw_staged` false, the GTX 1070 default): the
    // pre-perf9_O consumer (the profiler's, the plain or the default blob, chosen as above) and edge
    // pass, never the staged consumer's stages. Their dispatches are the staged family's single
    // blobs' (the same indirect headers, push constants and barriers)
    let (consumer, staged_stages, edge_pass) = if self.dust_draw_staged() {
      (consumer, staged_stages, edge_pass)
    } else {
      (
        // (`dust_draw_staged` is true wherever the family is not built: the fallbacks never run)
        match (self.kernels.pipelines.dust_draw_volume_mono_prof, self.kernels.pipelines.dust_draw_volume_mono_plain) {
          (Some(p), _) if profiling => p,
          (_, Some(p)) if volume_plain => p,
          _ => self.kernels.pipelines.dust_draw_volume_mono.unwrap_or(consumer),
        },
        None,
        match self.kernels.pipelines.dust_draw_small_edge_mono_prof {
          Some(p) if profiling => p,
          _ => self.kernels.pipelines.dust_draw_small_edge_mono.unwrap_or(edge_pass),
        },
      )
    };
    self.kernels.pipelines.assert_pc_size(consumer, core::mem::size_of_val(&pc));
    self.kernels.pipelines.assert_pc_size(splitter, core::mem::size_of_val(&pc));
    if let Some(m) = split_mono {
      self.kernels.pipelines.assert_pc_size(m, core::mem::size_of_val(&pc));
    }
    for p in [split_prep, split_walk, project[0], project[1], project[2], seq, seq_draw, leaf, leaf_wg] {
      self.kernels.pipelines.assert_pc_size(p, core::mem::size_of_val(&pc));
    }
    self.kernels.pipelines.assert_pc_size(edge_pass, core::mem::size_of_val(&pc));
    // the tier in chunks of clusters (`dust::DUST_VQUEUE_*`): per chunk the volume queue and
    // volume spill headers (empty, the range [first, end), the capacities), the producer over
    // the range, the splitter, then the consumer over the queue's records and again over the
    // spill's (`dust::DUST_VIEW_VOLUME_SPILL`); with the queue off every record goes through the
    // spill. `AETHERVK_DUST_VQUEUE_NODRAW=1` skips the consumer (diagnostics: the producer
    // alone; the image wrong by design).
    let (chunk, vqueue_capacity, vspill_capacity, leaf_capacity) = {
      let mut host = view.host.lock();
      host.vqueue_clusters_frame += count;
      let capacity = if host.vqueue_capacity > 0 {
        host.vqueue_capacity.min(view.vqueue_records)
      } else {
        view.vqueue_records
      };
      // (every chunk at most DUST_SQUEUE_CLUSTERS: the sub-packet queue's table and capacity)
      let chunk = if host.vqueue_off {
        // the queue off: every record goes through the volume spill, so the chunk that fits the
        // spill twice over (the spill path's A/B; slower: smaller chunks)
        dust::dust_vqueue_auto_chunk(dust::dust_vspill_records(view.vqueue_records), host.vqueue_ppc)
          .min(dust::DUST_SQUEUE_CLUSTERS)
          .min(count)
      } else {
        // the chunk that fits the queue twice over, at most DUST_VQUEUE_CHUNK_MAX clusters
        // (measured 2026-10-15 at 1 km: 256 / 512 / 1024 / 2400 / 8192 clusters a chunk give
        // 347 / 307 / 293 / 327 / 327 ms: a chunk's records fit the L2 at ~1024, below that the
        // dispatches' fixed cost shows)
        // (the floor scales with the view's records: a small device's chunk fits its queue)
        let c = if host.vqueue_chunk > 0 {
          host.vqueue_chunk.min(dust::DUST_SQUEUE_CLUSTERS)
        } else {
          dust::dust_vqueue_auto_chunk(view.vqueue_records, host.vqueue_ppc)
        };
        c.min(count)
      };
      // the volume spill's capacity per chunk: the view's records or the test hook's, never 0
      // (0 in its header word 4 would read as DUST_VQUEUE_CAPACITY)
      let spill_records = dust::dust_vspill_records(view.vqueue_records);
      let spill_capacity =
        if host.vspill_capacity > 0 { host.vspill_capacity.min(spill_records) } else { spill_records };
      // the splitter family (perf9 steps Z / O): the auto chunk also fits the leaf queue and the
      // candidate buffer twice over at the frame-wide estimates (a refused candidate run loses
      // its cluster's leaves; a refused leaf goes to the spill, then is lost)
      // (perf9_K: the pre-split splitter keeps the volume queue's chunk, as before the family)
      let chunk = if host.vqueue_chunk > 0 || split_mono.is_some() {
        chunk
      } else {
        chunk
          .min(dust::dust_split_auto_chunk(dust::dust_lqueue_records(view.vqueue_records), host.lqueue_lpc))
          .min(dust::dust_split_auto_chunk(dust::dust_cqueue_records(view.vqueue_records), host.cqueue_cpc))
      };
      // the leaf queue's capacity per chunk: the view's records or the queue hook's
      // (`dust_set_queue_capacity`; the rest goes to the leaf spill), never 0
      let leaf_records = dust::dust_lqueue_records(view.vqueue_records);
      let leaf_capacity =
        if host.queue_capacity > 0 { host.queue_capacity.min(leaf_records) } else { leaf_records };
      (chunk.max(1), capacity, spill_capacity, leaf_capacity)
    };
    let nodraw =
      aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE_NODRAW").is_some_and(|v| v.trim() == "1");
    let offsets = dust::dust_queue_offsets(view.vqueue_records);
    let vbase = (view.queue_word() + offsets.vqueue) as u64 * 4;
    let sbase = (view.queue_word() + offsets.squeue) as u64 * 4;
    let vsbase = (view.queue_word() + offsets.vspill) as u64 * 4;
    // track 9A (t9a2d): the deferred edge pieces' regions, by chunk parity
    let vebase = [
      (view.queue_word() + offsets.vedge[0]) as u64 * 4,
      (view.queue_word() + offsets.vedge[1]) as u64 * 4,
    ];
    // the splitter family's regions (perf9 steps Z / O): the leaf queue, the leaf spill, the large
    // leaves' list (the leaf consumers' indirect dispatches) and the candidate buffer (the project
    // passes' and the SEQ draw stage's); the leaf consumer's second dispatch drains the leaf spill
    let lqbase = (view.queue_word() + offsets.lqueue) as u64 * 4;
    let lsbase = (view.queue_word() + offsets.lspill) as u64 * 4;
    let blbase = (view.queue_word() + offsets.blist) as u64 * 4;
    let cqbase = (view.queue_word() + offsets.cqueue) as u64 * 4;
    let mut pc_leaf_spill = pc;
    pc_leaf_spill.flags |= dust::DUST_VIEW_LEAF_SPILL;
    // the consumer's second dispatch of each chunk drains the volume spill (host-internal flag);
    // both drains of an odd chunk append their edge pieces to region 1, the edge pass of an odd
    // chunk drains it (host-internal flag)
    let mut pc_spill = pc;
    pc_spill.flags |= dust::DUST_VIEW_VOLUME_SPILL;
    let with_parity = |p: DustSplatPushConstants, odd: bool| {
      let mut p = p;
      if odd {
        p.flags |= dust::DUST_VIEW_EDGE_ODD;
      }
      p
    };
    // the consumer's (and the previous producer's) storage writes → the header update
    let to_update = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    // the header update → the producer's reads and writes; and (track 9A, t9a2d) the previous
    // chunk's consumer → the edge pass beside this producer: its records and indirect dispatch
    // (the consumer's writes already ordered before the update by `to_update`; named here too)
    let to_producer = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER | vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::INDIRECT_COMMAND_READ
          | vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    // the producer's records and header → the consumer's indirect dispatch and reads
    let to_consumer = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::INDIRECT_COMMAND_READ
          | vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    // (perf9_O C2 v5) between two stages of the staged consumer: a stage's writes (its scratch rows,
    // records and pyramid) → the next stage's reads and writes
    let stage_to_stage = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE);
    let mut first = 0u32;
    let mut chunks = 0u32;
    loop {
      let end = first.saturating_add(chunk).min(count);
      let hdr = [0u32, 1, 1, 0, vqueue_capacity, first, end];
      unsafe {
        // (track 9A, t9a2d: at the first chunk too: the previous call's last edge pass and
        // consumer, recorded back to back with this call by the frame's tier loop with no barrier
        // between, read the headers and regions this update rewrites)
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_update)),
        );
        self.device.cmd_update_buffer(cmd, pyramid.buffer, vbase, bytemuck::cast_slice(&hdr));
        // the tier's ring words: the range offsets → slots (words 7–9 are the frame's counters)
        let ring = [first_slot, capacity.wrapping_sub(1)];
        self.device.cmd_update_buffer(
          cmd,
          pyramid.buffer,
          vbase + dust::DUST_VQUEUE_FIRST_SLOT as u64 * 4,
          bytemuck::cast_slice(&ring),
        );
        // the sub-packet queue: empty (its record counter and overflow count)
        // (words 4-5 the list, 6-7 the owner bytes: dust_splat.comp; round 9 step O: words 8-10
        // and 11-13 the record stages' indirect args, x 0 and y = z = 1, COUNT atomicMaxes x)
        let shdr = [
          0u32,
          dust::DUST_SQUEUE_CAPACITY,
          0,
          0,
          list as u32,
          (list >> 32) as u32,
          owner as u32,
          (owner >> 32) as u32,
          0,
          1,
          1,
          0,
          1,
          1,
        ];
        self.device.cmd_update_buffer(cmd, pyramid.buffer, sbase, bytemuck::cast_slice(&shdr));
        // the volume spill: empty, its dispatch (0, 1, 1), its capacity (covered by the same
        // `to_update` / `to_producer` barriers as the main header)
        let vshdr = [0u32, 1, 1, 0, vspill_capacity];
        self.device.cmd_update_buffer(cmd, pyramid.buffer, vsbase, bytemuck::cast_slice(&vshdr));
        // track 9A (t9a2d): this chunk's edge region (its parity's): empty, its dispatch (0, 1, 1),
        // its capacity (words 5-6, the frame's counts, kept). The other region holds the previous
        // chunk's edge pieces, drained below beside the producer; the chunk before that one's
        // pass, which read this region, finished before `to_update`
        let odd = chunks & 1 == 1;
        let vehdr = [0u32, 1, 1, 0, offsets.vedge_records];
        self.device.cmd_update_buffer(cmd, pyramid.buffer, vebase[odd as usize], bytemuck::cast_slice(&vehdr));
        // the splitter family (perf9 steps Z / O): the leaf queue, its spill and the large-leaf list
        // empty, their dispatch (0, 1, 1), their capacities (words 5-7, the lost count and the
        // offsets, kept for the frame); the candidate buffer empty (word 0 the project passes' and
        // the SEQ draw stage's x, word 3 the reservations, word 5 the refused: the draw stage's
        // granted range is [0, word 3 − word 5) of this chunk). The same barriers as the headers above
        // (perf9_K) not with the pre-split splitter: it feeds none of them
        if split_mono.is_none() {
          let lqhdr = [0u32, 1, 1, 0, leaf_capacity];
          self.device.cmd_update_buffer(cmd, pyramid.buffer, lqbase, bytemuck::cast_slice(&lqhdr));
          let lshdr = [0u32, 1, 1, 0, offsets.lqueue_records];
          self.device.cmd_update_buffer(cmd, pyramid.buffer, lsbase, bytemuck::cast_slice(&lshdr));
          let blhdr = [0u32, 1, 1, 0, 2 * offsets.lqueue_records];
          self.device.cmd_update_buffer(cmd, pyramid.buffer, blbase, bytemuck::cast_slice(&blhdr));
          let cqhdr = [0u32, 1, 1, 0, offsets.cqueue_records, 0];
          self.device.cmd_update_buffer(cmd, pyramid.buffer, cqbase, bytemuck::cast_slice(&cqhdr));
        }
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_producer)),
        );
        // the previous chunk's edge pieces (`dust_draw_small_edge.comp`), beside this producer: no
        // barrier between the two (the pass writes the pyramid's texels and frame words by
        // order-free atomics and reads only its region; the producer touches neither); the
        // splitter's barrier below orders it before this chunk's consumer
        if chunks > 0 && !nodraw {
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, edge_pass);
          self.device.cmd_push_constants(
            cmd,
            self.kernels.pipelines.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&with_parity(pc, !odd)),
          );
          self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, vebase[!odd as usize]);
        }
        // the enumerator: the stage chain over the chunk's clusters (`dust_splat.comp`, round 9
        // step O; `cmd_dust_splat_chain`)
        if let Some(mono) = splat_mono {
          // perf9_K: the monolithic enumerator as before step O: a warp per cluster, a lane per
          // sub-packet (`dust_splat_mono.comp`); the same barriers around it as the chain (the
          // header update before, the splitter's below); the header's words 8-13 (the chain's
          // indirect args) and the plan rows are not read
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mono);
          self.device.cmd_push_constants(
            cmd,
            self.kernels.pipelines.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
          );
          self.device.cmd_dispatch(cmd, ((end - first) * dust::DUST_SPLAT_LANES).div_ceil(DUST_WG), 1, 1);
        } else {
          self.cmd_dust_splat_chain(cmd, stages, &pc, end - first, pyramid.buffer, sbase);
        }
        // the splitter: a workgroup per cluster of the chunk over its records (`dust_split.comp`)
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_consumer)),
        );
        // (perf9 steps Z / O) PREP → walk → project passes 1-3 → SEQ order → SEQ draw
        if let Some(mono) = split_mono {
          // perf9_K: the pre-split splitter (`dust_split_mono.comp`): a workgroup per cluster of the
          // chunk over its records, EWA leaves drawn in-thread or put on the EWA queue (drained at
          // the frame end), volumetric packets to the volume queue / spill; the same barriers
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mono);
          self.device.cmd_push_constants(
            cmd,
            self.kernels.pipelines.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
          );
          self.device.cmd_dispatch(cmd, end - first, 1, 1);
        } else {
          self.cmd_dust_split_chain(cmd, &split_stages, &pc, end - first, pyramid.buffer, cqbase);
        }
        if !nodraw {
          self.device.synchronization2.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_consumer)),
          );
          // (perf9_K) not with the pre-split splitter (no leaf queue)
          if split_mono.is_none() {
            // the leaf consumers (`dust_draw_leaf.comp`): a lane per record of the leaf queue, again
            // over the leaf spill (host-internal flag), a workgroup per entry of the large-leaf list
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, leaf);
            self.device.cmd_push_constants(
              cmd,
              self.kernels.pipelines.pipeline_layout,
              vk::ShaderStageFlags::COMPUTE,
              0,
              bytemuck::bytes_of(&pc),
            );
            self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, lqbase);
            self.device.cmd_push_constants(
              cmd,
              self.kernels.pipelines.pipeline_layout,
              vk::ShaderStageFlags::COMPUTE,
              0,
              bytemuck::bytes_of(&pc_leaf_spill),
            );
            self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, lsbase);
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, leaf_wg);
            self.device.cmd_push_constants(
              cmd,
              self.kernels.pipelines.pipeline_layout,
              vk::ShaderStageFlags::COMPUTE,
              0,
              bytemuck::bytes_of(&pc),
            );
            self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, blbase);
          }
          if let Some(stages) = staged_stages {
            // (perf9_O C2 v5) the seven stages in order (s1 s2 s3a s3b s4a s4b s5), each a main drain
            // then its spill drain, with no barrier between the two drains (their rows and records
            // are per drain); a compute barrier between the stages (6)
            for (k, stage) in stages.into_iter().enumerate() {
              self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, stage);
              self.device.cmd_push_constants(
                cmd,
                self.kernels.pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&with_parity(pc, odd)),
              );
              self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, vbase);
              self.device.cmd_push_constants(
                cmd,
                self.kernels.pipelines.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&with_parity(pc_spill, odd)),
              );
              self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, vsbase);
              if k < 6 {
                self.device.synchronization2.cmd_pipeline_barrier2(
                  cmd,
                  &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&stage_to_stage)),
                );
              }
            }
          } else {
            self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, consumer);
            self.device.cmd_push_constants(
              cmd,
              self.kernels.pipelines.pipeline_layout,
              vk::ShaderStageFlags::COMPUTE,
              0,
              bytemuck::bytes_of(&with_parity(pc, odd)),
            );
            self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, vbase);
            // the same consumer over the chunk's volume spill (the records the queue refused). No
            // barrier between the two drains: the spill's records and dispatch were made visible
            // by the splitter → consumer barrier above, and the drains share only order-free
            // atomics (the in-frame gate's evaluation count is order-dependent past the gate, as
            // among the lanes of one drain)
            self.device.cmd_push_constants(
              cmd,
              self.kernels.pipelines.pipeline_layout,
              vk::ShaderStageFlags::COMPUTE,
              0,
              bytemuck::bytes_of(&with_parity(pc_spill, odd)),
            );
            self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, vsbase);
          }
        }
      }
      chunks += 1;
      first = end;
      if first >= count {
        break;
      }
    }
    // the last chunk's edge pieces, after its consumer (track 9A, t9a2d)
    if !nodraw {
      let odd = (chunks - 1) & 1 == 1;
      unsafe {
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_consumer)),
        );
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, edge_pass);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(&with_parity(pc, odd)),
        );
        self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, vebase[odd as usize]);
      }
    }
    view.host.lock().vqueue_chunks_frame += chunks;
    Ok(())
  }

  /// The splitter family over a chunk's `n` clusters (perf9 steps Z / O; Z DESIGN §4 item 5, §7
  /// item 11, §8.3 item 18, O DESIGN §3 items 3 and 3b): PREP (`stages[0]`; `None` with the
  /// hard-axis walk, which builds its cut itself) and the walk (`stages[1]`), a workgroup per
  /// cluster; the project stage's passes 1, 2, 3 (`stages[2..5]`), a lane per candidate, indirect
  /// from the candidate header's word 0 (`cqbase`, maintained by `dust_cqueue_reserve`); the SEQ
  /// order stage (`stages[5]`), a workgroup per cluster; the SEQ draw stage (`stages[6]`), indirect
  /// from the same word. The same push constants for every stage; a barrier between stages:
  /// compute writes → compute reads and the indirect args. The caller orders the enumerator before
  /// and the consumers after.
  fn cmd_dust_split_chain(
    &self,
    cmd: vk::CommandBuffer,
    stages: &[Option<vk::Pipeline>; 7],
    pc: &DustSplatPushConstants,
    n: u32,
    buffer: vk::Buffer,
    cqbase: u64,
  ) {
    let between = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::INDIRECT_COMMAND_READ
          | vk::AccessFlags2::SHADER_STORAGE_READ
          | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&between));
    let mut first = true;
    for (k, stage) in stages.iter().enumerate() {
      let Some(p) = *stage else { continue };
      unsafe {
        if !first {
          self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
        }
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(pc),
        );
        match k {
          // the project passes and the SEQ draw stage: a lane per candidate
          2 | 3 | 4 | 6 => self.device.cmd_dispatch_indirect(cmd, buffer, cqbase),
          // PREP, the walk, the SEQ order stage: a workgroup per cluster
          _ => self.device.cmd_dispatch(cmd, n, 1, 1),
        }
      }
      first = false;
    }
  }

  /// Test only (fault injection): the splat dispatch of one tier with an explicit pyramid
  /// address, bypassing the view lookup.
  #[cfg(test)]
  pub fn cmd_dust_splat_with_pyramid(
    &self,
    cmd: vk::CommandBuffer,
    _id: u64,
    render: u64,
    moments: u64,
    mut pc: DustSplatPushConstants,
    pyramid: u64,
  ) -> GpuResult<()> {
    pc.render = render;
    pc.moments = moments;
    pc.pyramid = pyramid;
    // the chain's first stage (LIVE, a lane per cluster): its first header read faults as the
    // whole enumerator's did (the later stages need the queue's indirect args: not here)
    let pipeline = self.kernels.pipelines.dust_splat_stages[dust::DUST_SPLAT_STAGE_LIVE];
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, pc.live_count.div_ceil(DUST_WG), 1, 1);
    }
    Ok(())
  }

  /// Ends the dust frame of `pe` (after [`Self::cmd_dust_post_splat_barrier`]): copies the header
  /// and the white-point level into this frame's readback slot. CPU mode folds the measurement
  /// in at once.
  pub fn cmd_dust_frame_end(
    &self,
    cmd: vk::CommandBuffer,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
    let (Some(pyramid), Some(rb)) = (view.pyramid.as_ref(), view.readback.as_ref()) else {
      return Ok(());
    };
    let (lvl, lvl_off, lw, lh) = DustView::white_level(&view.layout);
    let rb_words = DustView::readback_words(&view.layout) as u64;
    let mut host = view.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
    host.slot_clusters[slot] = host.vqueue_clusters_frame;
    host.slot_chunks[slot] = host.vqueue_chunks_frame;
    if crate::gpu_backends::vulkan::physics::is_cpu_particles_mode() {
      let (_, mapped) = view.cpu_pyramid(host.frame).ok_or(gpu_err!("dust view has no pyramid"))?;
      let words = unsafe {
        core::slice::from_raw_parts_mut(mapped.cast::<u32>(), view.layout.total_words as usize)
      };
      // the measurement grid from the finished pyramid (every tier is splatted by now)
      dust::measure_from_pyramid(&view.layout, words);
      let unit = host.slot_unit[slot];
      let nucleus_px = host.slot_nucleus_px[slot];
      let n = (lw * lh * dust::PYRAMID_TEXEL_WORDS) as usize;
      host.absorb_at(
        &words[..dust::PYRAMID_HEADER_WORDS as usize],
        &words[lvl_off as usize..lvl_off as usize + n],
        lw,
        1 << lvl,
        unit,
        nucleus_px,
      );
      host.frame += 1;
      return Ok(());
    }
    host.slot_valid[slot] = true;
    host.frame += 1;
    // the packet queue of this frame (none after a reused begin), and the query of its timestamp
    let queue = core::mem::take(&mut host.queue_pending);
    let queue_ts = match host.ts_slot {
      Some(ts_slot) if queue && view.ts_pool != vk::QueryPool::null() => {
        host.ts_written[ts_slot] |= 1 << DUST_TS_QUEUED;
        Some(ts_slot as u32 * DUST_TIMESTAMPS + DUST_TS_QUEUED)
      }
      _ => None,
    };
    drop(host);
    if queue {
      // the queued packets (`dust_draw_queue.comp`): the splats' appends → the indirect dispatch
      // and its reads; its writes → the measurement, the composite and the copies below
      let qpc = dust::DustMeasurePushConstants {
        pyramid: pyramid.address,
        pad: [0; 2],
      };
      let pipeline = match self.kernels.pipelines.dust_draw_queue_prof {
        Some(p) if crate::gpu_backends::vulkan::physics::dust_profile_requested() => p,
        _ => self.kernels.pipelines.dust_draw_queue,
      };
      // (perf9_K) the mono draw family's queue draw (the pre-perf9_O blob; the same dispatch)
      let pipeline = if self.dust_draw_staged() {
        pipeline
      } else {
        match self.kernels.pipelines.dust_draw_queue_mono_prof {
          Some(p) if crate::gpu_backends::vulkan::physics::dust_profile_requested() => p,
          _ => self.kernels.pipelines.dust_draw_queue_mono.unwrap_or(pipeline),
        }
      };
      self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&qpc));
      let to_queue = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT | vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(
          vk::AccessFlags2::INDIRECT_COMMAND_READ
            | vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_STORAGE_WRITE,
        );
      let after_queue = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(
          vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::FRAGMENT_SHADER
            | vk::PipelineStageFlags2::TRANSFER,
        )
        .dst_access_mask(
          vk::AccessFlags2::SHADER_STORAGE_READ
            | vk::AccessFlags2::SHADER_STORAGE_WRITE
            | vk::AccessFlags2::TRANSFER_READ,
        );
      unsafe {
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_queue)),
        );
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(&qpc),
        );
        self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, view.queue_word() as u64 * 4);
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&after_queue)),
        );
        if let Some(q) = queue_ts {
          self.device.synchronization2.cmd_write_timestamp2(
            cmd,
            vk::PipelineStageFlags2::ALL_COMMANDS,
            view.ts_pool,
            q,
          );
        }
      }
    }
    // the cadence's frame end (the frame's light, the state's readback copy)
    self.cad_frame_end(cmd, mgr, pe, pyramid.address, &view.layout);
    // the measurement grid from the finished pyramid (`dust_measure.comp`): the splats' writes →
    // this pass, then its writes → the readback copy
    let (_, mw, mh) = view.layout.measure;
    let pipeline = self.kernels.pipelines.dust_measure;
    let mpc = dust::DustMeasurePushConstants {
      pyramid: pyramid.address,
      pad: [0; 2],
    };
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&mpc));
    let to_compute = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(
        vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
      );
    let to_copy = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
    unsafe {
      self.device.synchronization2.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_compute)),
      );
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&mpc),
      );
      self.device.cmd_dispatch(cmd, (mw * mh).div_ceil(DUST_WG), 1, 1);
      self.device.synchronization2.cmd_pipeline_barrier2(
        cmd,
        &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_copy)),
      );
    }
    let split_offsets = dust::dust_queue_offsets(view.vqueue_records);
    let regions = [
      vk::BufferCopy::default()
        .src_offset(0)
        .dst_offset(slot as u64 * rb_words * 4)
        .size(dust::PYRAMID_HEADER_WORDS as u64 * 4),
      vk::BufferCopy::default()
        .src_offset(lvl_off as u64 * 4)
        .dst_offset(slot as u64 * rb_words * 4 + dust::PYRAMID_HEADER_WORDS as u64 * 4)
        .size((lw * lh * dust::PYRAMID_TEXEL_WORDS) as u64 * 4),
      vk::BufferCopy::default()
        .src_offset(view.queue_word() as u64 * 4)
        .dst_offset(
          (slot as u64 + 1) * rb_words * 4
            - (dust::DUST_QUEUE_HEADER_WORDS + 3 * dust::DUST_VQUEUE_HEADER_WORDS) as u64 * 4,
        )
        .size(dust::DUST_QUEUE_HEADER_WORDS as u64 * 4),
      vk::BufferCopy::default()
        .src_offset((view.queue_word() + dust::DUST_QUEUE_WORDS) as u64 * 4)
        .dst_offset((slot as u64 + 1) * rb_words * 4 - 3 * dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)
        .size(dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4),
      // the splitter family (perf9 steps Z / O): the leaf queue's and the candidate buffer's headers
      vk::BufferCopy::default()
        .src_offset((view.queue_word() + split_offsets.lqueue) as u64 * 4)
        .dst_offset((slot as u64 + 1) * rb_words * 4 - 2 * dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)
        .size(dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4),
      vk::BufferCopy::default()
        .src_offset((view.queue_word() + split_offsets.cqueue) as u64 * 4)
        .dst_offset((slot as u64 + 1) * rb_words * 4 - dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)
        .size(dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4),
    ];
    let b = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::HOST)
      .dst_access_mask(vk::AccessFlags2::HOST_READ);
    let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
    unsafe {
      self.device.cmd_copy_buffer(cmd, pyramid.buffer, rb.buffer, &regions);
      self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
    }
    drop(view);
    drop(res);
    self.cmd_dust_timestamp(cmd, pe, DUST_TS_MEASURED);
    Ok(())
  }

  /// Writes dust timestamp `index` ([`DUST_TS_BEGIN`] …) of `pe`'s current frame, once all
  /// earlier commands completed. A no-op without timestamp queries, before the frame's begin reset
  /// its slot, or when this index was written this frame already.
  pub fn cmd_dust_timestamp(&self, cmd: vk::CommandBuffer, pe: gpu::PresentationEngineHandle, index: u32) {
    let res = self.res.read();
    let Some(mgr) = res.dust_manager.as_ref() else {
      return;
    };
    let Some(view) = mgr.views.get(&pe) else {
      return;
    };
    if view.ts_pool == vk::QueryPool::null() || index >= DUST_TIMESTAMPS {
      return;
    }
    let mut host = view.host.lock();
    let Some(slot) = host.ts_slot else {
      return;
    };
    if host.ts_written[slot] & (1 << index) != 0 {
      return;
    }
    host.ts_written[slot] |= 1 << index;
    unsafe {
      self.device.synchronization2.cmd_write_timestamp2(
        cmd,
        vk::PipelineStageFlags2::ALL_COMMANDS,
        view.ts_pool,
        slot as u32 * DUST_TIMESTAMPS + index,
      );
    }
  }

  /// `pe`'s frames draw the legacy kernel from the next frame on (`dust::DUST_FRAME_LEGACY_KERNEL_BIT`:
  /// no cone of grains, no slow-end sub-bins); a no-op without a view
  pub fn dust_set_legacy_kernel(&self, pe: gpu::PresentationEngineHandle, on: bool) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().legacy_kernel = on;
    }
  }

  /// `pe`'s frames draw deep chords by the EWA (`dust::DUST_FRAME_DEEP_CHORD`) from the next frame
  /// on; `None` returns to `AETHERVK_DUST_DEEP_CHORD`. A no-op without a view.
  pub fn dust_set_deep_chord(&self, pe: gpu::PresentationEngineHandle, on: Option<bool>) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().deep_chord = on;
    }
  }

  /// `pe`'s frames draw their volumetric packets at the level bias `bias` from the next frame on
  /// (a test hook: the parity of the texel integral at a coarse bias); `None` returns to the
  /// feedback. A no-op without a view.
  pub fn dust_set_volume_bias(&self, pe: gpu::PresentationEngineHandle, bias: Option<u32>) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      let mut host = view.host.lock();
      // a released pin continues from the pinned bias (round 9 step B): the feedback's own state
      // did not run while the frames drew at the pin, and the plateau hold starts afresh
      if let (Some(b), None) = (host.volume_bias_override, bias) {
        host.volume_bias = b;
        host.fb_memory.volume_bias = b;
        host.fb_memory.top_under = 0;
        host.fb_full.volume_bias = b;
        host.fb_full.top_under = 0;
        if let Some(c) = host.fb_cad.as_mut() {
          c.volume_bias = b;
          c.top_under = 0;
        }
      }
      host.volume_bias_override = bias;
    }
  }

  /// `pe`'s frames draw with the cluster level of detail `lod` (pixels per drawn segment, 0 = off)
  /// from the next frame on; `None` returns to `AETHERVK_DUST_LOD_PX` / [`dust::DUST_LOD_PX_DEFAULT`].
  /// A no-op without a view.
  pub fn dust_set_lod_px(&self, pe: gpu::PresentationEngineHandle, lod: Option<f32>) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().lod_px = lod;
    }
  }

  /// `pe`'s walk passes record every speck they draw from the next frame on (a test hook:
  /// `(hash, dp)` per speck, [`Self::download_dust_specks_trace`]); a no-op without a view.
  pub fn dust_set_specks_trace(&self, pe: gpu::PresentationEngineHandle, on: bool) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().specks_trace = on;
    }
  }

  /// `pe`'s walk passes hold their shared stack to `nodes` from the next frame on, spilling past
  /// it (a test hook: the spill path at a scene's own depth; 0 = the whole stack, the default)
  pub fn dust_set_specks_stack_limit(&self, pe: gpu::PresentationEngineHandle, nodes: u32) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().specks_stack_limit = nodes;
    }
  }

  /// The speck walk's spill header after the last walk pass ([`dust::DUST_SPILL_HEADER_WORDS`]
  /// words: 0 the slots claimed (0: the breadth-first walk claims none), 1 the view's slots, 2-3
  /// the trace's capacity and count, 8 the former stack limit (unread), 9-13 the walk region's
  /// first word and capacities). GPU views only.
  pub fn download_dust_spill_header(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<alloc::vec::Vec<u32>> {
    let (buffer, spill_offset) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      let off = (view.queue_word() + dust::dust_queue_offsets(view.vqueue_records).spill) as u64 * 4;
      (pyramid.buffer, off)
    };
    Ok(pod_vec(&self.dust_download_bytes(buffer, spill_offset, dust::DUST_SPILL_HEADER_WORDS as u64 * 4)?))
  }

  /// `n` words of the view's queue allocation from `word` (pyramid-buffer words; diagnostics).
  pub fn download_dust_words(&self, pe: gpu::PresentationEngineHandle, word: u32, n: u32) -> GpuResult<alloc::vec::Vec<u32>> {
    let buffer = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      view.queue_pyramid()?.buffer
    };
    Ok(pod_vec(&self.dust_download_bytes(buffer, word as u64 * 4, n as u64 * 4)?))
  }

  /// The walk region's header after the last walk pass ([`dust::DUST_WR_REGION_HEADER`] words:
  /// 3 the record count, 4 the many-speck leaf count, 8 + 4s the frontier count of step s,
  /// [`dust::DUST_WR_LEAF1`] the single-speck leaf count, the rest indirect args). GPU views only.
  pub fn download_dust_walk_region_header(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<alloc::vec::Vec<u32>> {
    let (buffer, region_offset) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      let off = (view.queue_word() + dust::dust_queue_offsets(view.vqueue_records).spill + dust::DUST_SPILL_HEADER_WORDS) as u64 * 4;
      (pyramid.buffer, off)
    };
    Ok(pod_vec(&self.dust_download_bytes(buffer, region_offset, dust::DUST_WR_REGION_HEADER as u64 * 4)?))
  }

  /// The specks the last frame's walk passes recorded (`(hash, dp)` in local metres from the eye;
  /// [`Self::dust_set_specks_trace`]): the spill header's count, capped at the capacity, then the
  /// records. GPU views only.
  pub fn download_dust_specks_trace(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<alloc::vec::Vec<(u32, [f32; 3])>> {
    Ok(
      self
        .download_dust_specks_trace_records(pe)?
        .iter()
        .map(|w| (w[0], [f32::from_bits(w[1]), f32::from_bits(w[2]), f32::from_bits(w[3])]))
        .collect(),
    )
  }

  /// The whole trace records ([`dust::DUST_SPECKS_TRACE_WORDS`] words: hash, dp xyz, w xyz, the
  /// coordinates xyz, κ, the node level; floats as bits).
  pub fn download_dust_specks_trace_records(
    &self,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<alloc::vec::Vec<[u32; dust::DUST_SPECKS_TRACE_WORDS as usize]>> {
    let (buffer, spill_offset, slots) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      let off = (view.queue_word() + dust::dust_queue_offsets(view.vqueue_records).spill) as u64 * 4;
      (pyramid.buffer, off, view.spill_slots)
    };
    let header: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(buffer, spill_offset, dust::DUST_SPILL_HEADER_WORDS as u64 * 4)?);
    let n = header[3].min(header[2]).min(dust::DUST_SPECKS_TRACE_CAPACITY);
    if n == 0 {
      return Ok(alloc::vec::Vec::new());
    }
    let records_offset = spill_offset
      + (dust::DUST_SPILL_HEADER_WORDS + slots * dust::DUST_SPILL_NODES * dust::DUST_SPILL_NODE_WORDS) as u64 * 4;
    let words: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(buffer, records_offset, n as u64 * dust::DUST_SPECKS_TRACE_WORDS as u64 * 4)?);
    Ok(
      words
        .chunks_exact(dust::DUST_SPECKS_TRACE_WORDS as usize)
        .map(|w| core::array::from_fn(|k| w[k]))
        .collect(),
    )
  }

  /// The GPU timings of `pe`'s last measured frame that drew (not reused, see
  /// [`DustGpuTimings::reused`]): the cost of a changing view
  pub fn dust_gpu_timings_drawn(&self, pe: gpu::PresentationEngineHandle) -> Option<DustGpuTimings> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    let view = mgr.views.get(&pe)?;
    view.host.lock().timings_drawn
  }

  /// Sets `pe`'s packet queue mode from the next frame on (tests, diagnostics):
  /// `Some(dust::DUST_FRAME_QUEUE_OFF)` draws every packet in its cluster's thread,
  /// `Some(dust::DUST_FRAME_QUEUE_ALL)` queues every packet, `None` = `AETHERVK_DUST_QUEUE` / the
  /// default (packets above `dust::DUST_QUEUE_MIN_TEXELS`). A no-op without a view.
  pub fn dust_set_queue_mode(&self, pe: gpu::PresentationEngineHandle, mode: Option<u32>) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().queue_mode = mode;
    }
  }

  /// Sets `pe`'s packet queue capacity from the next frame on (a test hook for the overflow path;
  /// 0 = `dust::DUST_QUEUE_CAPACITY`)
  pub fn dust_set_queue_capacity(&self, pe: gpu::PresentationEngineHandle, records: u32) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().queue_capacity = records;
    }
  }

  /// `pe`'s last frame's packet queue (diagnostics, waits for the GPU): the records drawn and the
  /// packets (row ranges) the producers tried to append; `raw − drawn` overflowed and were drawn
  /// in-thread
  pub fn dust_queue_counts(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<(u32, u32)> {
    let (buffer, base) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      (pyramid.buffer, view.queue_word() as u64 * 4)
    };
    let w: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(buffer, base, 32)?);
    Ok((dust::dust_queue_held(&w), w[3]))
  }

  /// `pe`'s last frame's refused candidates (whole clusters the candidate buffer refused: not
  /// drawn) and its densest chunk (candidates a cluster); diagnostics, waits for the GPU
  pub fn dust_candidates_refused(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<(u32, u32)> {
    let (buffer, base) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      let offsets = dust::dust_queue_offsets(view.vqueue_records);
      (pyramid.buffer, (view.queue_word() + offsets.lqueue) as u64 * 4)
    };
    let w: alloc::vec::Vec<u32> =
      pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)?);
    Ok((w[dust::DUST_LQUEUE_FRAME_REFUSED], w[dust::DUST_LQUEUE_FRAME_CPC_MAX]))
  }

  /// `pe`'s last frame's leaf queue (the splitter family, perf9 steps Z / O; diagnostics, waits
  /// for the GPU): the leaves the sequencer queued, the large ones listed for the workgroup blob,
  /// the ones the queue refused (sent to the leaf spill), the ones the spill refused too (lost)
  pub fn dust_leaf_counts(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<(u32, u32, u32, u32)> {
    let (buffer, base) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      let offsets = dust::dust_queue_offsets(view.vqueue_records);
      (pyramid.buffer, (view.queue_word() + offsets.lqueue) as u64 * 4)
    };
    let w: alloc::vec::Vec<u32> =
      pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)?);
    Ok((
      w[dust::DUST_LQUEUE_FRAME_LEAVES],
      w[dust::DUST_LQUEUE_FRAME_LARGE],
      w[dust::DUST_LQUEUE_FRAME_SPILLED],
      w[dust::DUST_LQUEUE_LOST],
    ))
  }

  /// The volume packet queue's records per chunk for a new view, from the device-local memory
  /// (the chunking makes the capacity a speed knob only: a smaller queue means more chunks, the
  /// same image): [`dust::DUST_VQUEUE_CAPACITY`] (32 MiB) from 4 GiB of device-local memory,
  /// a quarter of that from 2 GiB, a sixteenth below; `AETHERVK_DUST_VQUEUE_RECORDS` overrides
  /// (clamped to 64..=capacity).
  pub fn dust_vqueue_records(&self) -> u32 {
    if let Some(n) = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE_RECORDS")
      .and_then(|v| v.trim().parse::<u32>().ok())
    {
      return n.clamp(64, dust::DUST_VQUEUE_CAPACITY);
    }
    let props = unsafe {
      self
        .instance
        .instance
        .get_physical_device_memory_properties(self.query_result.physical_device)
    };
    let device_local: u64 = props.memory_heaps[..props.memory_heap_count as usize]
      .iter()
      .filter(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
      .map(|h| h.size)
      .sum();
    const GIB: u64 = 1 << 30;
    let records = if device_local >= 4 * GIB {
      dust::DUST_VQUEUE_CAPACITY
    } else if device_local >= 2 * GIB {
      dust::DUST_VQUEUE_CAPACITY / 4
    } else {
      dust::DUST_VQUEUE_CAPACITY / 16
    };
    // (perf9_O C2 v5) the staged consumer's scratch rows (`DUST_VW_SX_WORDS` words each, for the
    // main and the spill drains): at most a sixteenth of the queue (at full capacity 1.03 GB)
    if self.kernels.pipelines.dust_draw_volume_stages.is_some() {
      records.min(dust::DUST_VQUEUE_CAPACITY / 16)
    } else {
      records
    }
  }

  /// Sets `pe`'s volume packet queue mode from the next frame on (tests, diagnostics):
  /// `Some(dust::DUST_FRAME_VQUEUE_OFF)` sends every volumetric packet through the volume spill
  /// (the main queue refuses all; the spill path's A/B), `None` = `AETHERVK_DUST_VQUEUE` / on.
  /// A no-op without a view.
  pub fn dust_set_vqueue_mode(&self, pe: gpu::PresentationEngineHandle, mode: Option<u32>) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().vqueue_mode = mode;
    }
  }

  /// perf9_K: the draw family's vendor rule. The staged family (perf9_O: the workgroup-per-record
  /// volume consumer, the texel-lane queue draw, the grouped edge pass; the zero-spill design) on
  /// Qualcomm (0x5143) and ARM (0x13B5), the mono family (the pre-perf9_O blobs, `*_mono.comp`:
  /// ~14x faster on the GTX 1070) on NVIDIA (0x10DE) and the others. `AETHERVK_DUST_DRAW_STAGED=1`
  /// / `0` overrides it.
  pub fn dust_draw_staged_wanted(vendor_id: u32) -> bool {
    match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_DRAW_STAGED").as_deref().map(str::trim) {
      Some("1") => true,
      Some("0") => false,
      _ => matches!(vendor_id, 0x5143 | 0x13B5),
    }
  }

  /// perf9_K: whether this device's frames draw with the staged family (the test hook's choice,
  /// else [`Self::dust_draw_staged_wanted`], resolved once and logged); always the staged family
  /// where the mono pipelines are not built
  pub fn dust_draw_staged(&self) -> bool {
    use core::sync::atomic::Ordering;
    let p = &self.kernels.pipelines;
    let built = p.dust_draw_volume_mono.is_some() && p.dust_draw_small_edge_mono.is_some() && p.dust_draw_queue_mono.is_some();
    let mode = &p.dust_draw_mode;
    match mode.load(Ordering::Relaxed) {
      1 | 3 => !built,
      2 | 4 => true,
      _ => {
        let vendor = self.query_result.physical_device_properties.vendor_id;
        let staged = !built || Self::dust_draw_staged_wanted(vendor);
        let _ = mode.compare_exchange(0, if staged { 4 } else { 3 }, Ordering::Relaxed, Ordering::Relaxed);
        aethervk_oshal_rlib::log!(
          "[dust] draw family: {} (vendor {vendor:#X}, AETHERVK_DUST_DRAW_STAGED={:?}, mono built {built})",
          if staged { "staged (perf9_O)" } else { "mono (pre-perf9_O)" },
          aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_DRAW_STAGED")
        );
        staged
      }
    }
  }

  /// perf9_K test hook: `Some(true)` / `Some(false)` draws with the staged / mono family from the
  /// next recorded frame on (every view of the device), `None` returns to the vendor rule
  pub fn dust_set_draw_staged(&self, staged: Option<bool>) {
    let m = match staged {
      Some(false) => 1,
      Some(true) => 2,
      None => 0,
    };
    self.kernels.pipelines.dust_draw_mode.store(m, core::sync::atomic::Ordering::Relaxed);
  }

  /// perf9_K (K4): the propagate / emit family's vendor rule. The perf9_O package (the propagate's
  /// lanes-per-cluster solve over the rows buffer and its moments stage, the emission's record-stash
  /// solve; the staged passes where `AETHERVK_DUST_STAGED` picks them) on Qualcomm (0x5143) and ARM
  /// (0x13B5), the pre-package monoliths (`dust_propagate_mono.comp`, `dust_emit_mono.comp`: the
  /// GTX 1070's propagate 12.7 ms against 22.4 ms with the package) on NVIDIA (0x10DE) and the
  /// others. `AETHERVK_DUST_PROP_STAGED=1` / `0` overrides it.
  pub fn dust_prop_staged_wanted(vendor_id: u32) -> bool {
    match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PROP_STAGED").as_deref().map(str::trim) {
      Some("1") => true,
      Some("0") => false,
      _ => matches!(vendor_id, 0x5143 | 0x13B5),
    }
  }

  /// perf9_K (K4): whether this device propagates and emits with the perf9_O package (the test
  /// hook's choice, else [`Self::dust_prop_staged_wanted`], resolved once and logged); always the
  /// package where the mono pipelines are not built
  pub fn dust_prop_staged(&self) -> bool {
    use core::sync::atomic::Ordering;
    let p = &self.kernels.pipelines;
    let built = p.dust_propagate_mono.is_some() && p.dust_emit_mono.is_some();
    let mode = &p.dust_prop_mode;
    match mode.load(Ordering::Relaxed) {
      1 | 3 => !built,
      2 | 4 => true,
      _ => {
        let vendor = self.query_result.physical_device_properties.vendor_id;
        let staged = !built || Self::dust_prop_staged_wanted(vendor);
        let _ = mode.compare_exchange(0, if staged { 4 } else { 3 }, Ordering::Relaxed, Ordering::Relaxed);
        aethervk_oshal_rlib::log!(
          "[dust] propagate/emit family: {} (vendor {vendor:#X}, AETHERVK_DUST_PROP_STAGED={:?}, mono built {built})",
          if staged { "staged (perf9_O package)" } else { "mono (pre-package monoliths)" },
          aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PROP_STAGED")
        );
        staged
      }
    }
  }

  /// perf9_K (K4) test hook: `Some(true)` / `Some(false)` propagates and emits with the package /
  /// the mono pipelines (where built) from the next recorded command on, `None` returns to the
  /// vendor rule
  pub fn dust_set_prop_staged(&self, staged: Option<bool>) {
    let m = match staged {
      Some(false) => 1,
      Some(true) => 2,
      None => 0,
    };
    self.kernels.pipelines.dust_prop_mode.store(m, core::sync::atomic::Ordering::Relaxed);
  }

  /// perf9_K: the split family's vendor rule. The staged leaf path (perf9 steps Z / O: prep, walk,
  /// project passes 1-3, sequencer, leaf consumers; the zero-spill design) on Qualcomm (0x5143) and
  /// ARM (0x13B5), the pre-split splitter (`dust_split_mono.comp`, one dispatch per chunk: the 1 km
  /// still ~0.3 s against several seconds with the family on the GTX 1070, byte gate perf9b_split2)
  /// on NVIDIA (0x10DE) and the others. `AETHERVK_DUST_SPLIT_STAGED=1` / `0` overrides it.
  pub fn dust_split_staged_wanted(vendor_id: u32) -> bool {
    match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPLIT_STAGED").as_deref().map(str::trim) {
      Some("1") => true,
      Some("0") => false,
      _ => matches!(vendor_id, 0x5143 | 0x13B5),
    }
  }

  /// perf9_K: whether this device's frames split with the staged family (the test hook's choice,
  /// else [`Self::dust_split_staged_wanted`], resolved once and logged); always the staged family
  /// where the pre-split splitter is not built
  pub fn dust_split_staged(&self) -> bool {
    use core::sync::atomic::Ordering;
    let built = self.kernels.pipelines.dust_split_mono.is_some();
    let mode = &self.kernels.pipelines.dust_split_mode;
    match mode.load(Ordering::Relaxed) {
      1 | 3 => !built,
      2 | 4 => true,
      _ => {
        let vendor = self.query_result.physical_device_properties.vendor_id;
        let staged = !built || Self::dust_split_staged_wanted(vendor);
        let _ = mode.compare_exchange(0, if staged { 4 } else { 3 }, Ordering::Relaxed, Ordering::Relaxed);
        aethervk_oshal_rlib::log!(
          "[dust] split family: {} (vendor {vendor:#X}, AETHERVK_DUST_SPLIT_STAGED={:?}, mono built {built})",
          if staged { "staged (perf9 Z/O leaf queue)" } else { "mono (pre-split splitter)" },
          aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPLIT_STAGED")
        );
        staged
      }
    }
  }

  /// perf9_K test hook: `Some(true)` / `Some(false)` splits with the staged / mono family from the
  /// next recorded frame on (every view of the device), `None` returns to the vendor rule
  pub fn dust_set_split_staged(&self, staged: Option<bool>) {
    let m = match staged {
      Some(false) => 1,
      Some(true) => 2,
      None => 0,
    };
    self.kernels.pipelines.dust_split_mode.store(m, core::sync::atomic::Ordering::Relaxed);
  }

  /// Runs `pe`'s volume consumer with the plain blob instead of the `.sg` one from the next
  /// frame on (round 8 step F2; a test hook: the baseline devices' path on a device that has
  /// both; a no-op where the plain blob is the only one, or without a view)
  pub fn dust_set_volume_plain(&self, pe: gpu::PresentationEngineHandle, plain: bool) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().volume_plain = plain;
    }
  }

  /// Sets `pe`'s volume queue capacity per chunk from the next frame on (a test hook for the
  /// overflow path; 0 = `dust::DUST_VQUEUE_CAPACITY`)
  pub fn dust_set_vqueue_capacity(&self, pe: gpu::PresentationEngineHandle, records: u32) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().vqueue_capacity = records;
    }
  }

  /// Sets `pe`'s volume spill capacity per chunk from the next frame on (a test hook for the
  /// lost path; 0 = the view's `dust::dust_vspill_records`)
  pub fn dust_set_vspill_capacity(&self, pe: gpu::PresentationEngineHandle, records: u32) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().vspill_capacity = records;
    }
  }

  /// Sets `pe`'s volume queue chunk in clusters from the next frame on (a test hook; 0 = from
  /// the packets-per-cluster estimate)
  pub fn dust_set_vqueue_chunk(&self, pe: gpu::PresentationEngineHandle, clusters: u32) {
    let res = self.res.read();
    if let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) {
      view.host.lock().vqueue_chunk = clusters;
    }
  }

  /// `pe`'s last chunk's sub-packet queue (diagnostics, waits for the GPU): the records the
  /// enumerator wrote and the records that did not fit (never by construction: a test asserts 0)
  pub fn dust_squeue_counts(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<(u32, u32)> {
    let (buffer, base) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      (
        pyramid.buffer,
        (view.queue_word() + dust::dust_queue_offsets(view.vqueue_records).squeue) as u64 * 4,
      )
    };
    let w: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(buffer, base, 16)?);
    Ok((w[0], w[dust::DUST_SQUEUE_OVERFLOWED]))
  }

  /// `pe`'s last frame's volumetric deposits above `DUST_COUNT_MAX_PER_ADD`, split into several
  /// adds (`dust::DUST_VQUEUE_SPLIT_ADDS`; diagnostics, waits for the GPU)
  pub fn dust_volume_split_adds(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<u32> {
    let (buffer, base) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      (pyramid.buffer, (view.queue_word() + dust::DUST_QUEUE_WORDS) as u64 * 4)
    };
    let w: alloc::vec::Vec<u32> =
      pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)?);
    Ok(w[dust::DUST_VQUEUE_SPLIT_ADDS])
  }

  /// `pe`'s last frame's volume packet queue (diagnostics, waits for the GPU): the records the
  /// consumer drew from the main queue, the packets that overflowed a chunk's queue (sent to the
  /// volume spill: [`Self::dust_vqueue_spill_counts`]), the chunks
  pub fn dust_vqueue_counts(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<(u32, u32, u32)> {
    let (buffer, base, chunks) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      (
        pyramid.buffer,
        (view.queue_word() + dust::DUST_QUEUE_WORDS) as u64 * 4,
        view.host.lock().vqueue_chunks_frame,
      )
    };
    let w: alloc::vec::Vec<u32> =
      pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)?);
    Ok((w[dust::DUST_VQUEUE_DRAWN], w[dust::DUST_VQUEUE_OVERFLOWED], chunks))
  }

  /// `pe`'s last frame's volume spill (diagnostics, waits for the GPU): the records the consumer
  /// drew from it, and the records it refused too (lost: not drawn). Overflowed = the two summed.
  pub fn dust_vqueue_spill_counts(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<(u32, u32)> {
    let (buffer, base) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      (pyramid.buffer, (view.queue_word() + dust::DUST_QUEUE_WORDS) as u64 * 4)
    };
    let w: alloc::vec::Vec<u32> =
      pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)?);
    Ok((w[dust::DUST_VQUEUE_SPILL_DRAWN], w[dust::DUST_VQUEUE_LOST]))
  }

  /// `pe`'s last frame's deferred edge pieces (track 9A, t9a2d; diagnostics, waits for the GPU):
  /// the small pieces the edge pass drew, those a full region refused (lost: not drawn) and, of
  /// the drawn, those it demoted (t9a2e), both regions summed ([`dust::DUST_VEDGE_DRAWN`] /
  /// [`dust::DUST_VEDGE_LOST`] / [`dust::DUST_VEDGE_DEMOTED`])
  pub fn dust_vedge_counts(&self, pe: gpu::PresentationEngineHandle) -> GpuResult<(u32, u32, u32)> {
    let (buffer, bases) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      let o = dust::dust_queue_offsets(view.vqueue_records);
      (pyramid.buffer, o.vedge.map(|b| (view.queue_word() + b) as u64 * 4))
    };
    let (mut drawn, mut lost, mut demoted) = (0u32, 0u32, 0u32);
    for base in bases {
      let w: alloc::vec::Vec<u32> =
        pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)?);
      drawn = drawn.wrapping_add(w[dust::DUST_VEDGE_DRAWN]);
      lost = lost.wrapping_add(w[dust::DUST_VEDGE_LOST]);
      demoted = demoted.wrapping_add(w[dust::DUST_VEDGE_DEMOTED]);
    }
    Ok((drawn, lost, demoted))
  }

  /// The shader-clock profiler's counters of `pe`'s last drawn frame
  /// ([`dust::DustProfileCounters`]: the queue header's words 8.., `dust_profile.glsl`), read
  /// straight from the pyramid buffer (a blocking download: diagnostics); all zero on a frame the
  /// production shaders drew. The queue header is reset only by a drawn frame, so after a reused
  /// one they are still the last drawn frame's.
  pub fn dust_profile_counters(
    &self,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<dust::DustProfileCounters> {
    let (buffer, base, frame) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.queue_pyramid()?;
      (pyramid.buffer, view.queue_word() as u64 * 4, view.host.lock().frame)
    };
    let w: alloc::vec::Vec<u32> =
      pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_QUEUE_HEADER_WORDS as u64 * 4)?);
    let mut words = [0u32; dust::DUST_PROFILE_WORDS];
    words.copy_from_slice(&w[8..dust::DUST_QUEUE_HEADER_WORDS as usize]);
    Ok(dust::DustProfileCounters { words, frame })
  }

  /// Whether `pe`'s frames need the speck pass: the specks are on (`AETHERVK_DUST_SPECKS=1`), a
  /// shell override is set (tests, diagnostics), or `AETHERVK_DUST_EYE_DENSITY=1` asks for the
  /// density at the eye (`PYR_N_EYE`) without them. Otherwise its only product is that density,
  /// which nothing reads (the shell radius it feeds is 0 with the specks off), and the frame skips
  /// the pass (8 ms plan Step 2: ~17 ms at the inside poses). `false` without a view.
  pub fn dust_specks_needed(&self, pe: gpu::PresentationEngineHandle) -> bool {
    let res = self.res.read();
    let Some(view) = res.dust_manager.as_ref().and_then(|m| m.views.get(&pe)) else {
      return false;
    };
    let override_set = view.host.lock().shell_override.is_some();
    override_set
      || specks_enabled()
      || aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_EYE_DENSITY").is_some_and(|v| v.trim() == "1")
  }

  /// The last measured GPU timings of `pe`'s dust frame ([`DustGpuTimings`]; `None` before the
  /// first, without a view or without timestamp queries)
  pub fn dust_gpu_timings(&self, pe: gpu::PresentationEngineHandle) -> Option<DustGpuTimings> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    let view = mgr.views.get(&pe)?;
    view.host.lock().timings
  }

  /// The view's current display parameters (`None` without a view).
  pub fn dust_view_params(&self, pe: gpu::PresentationEngineHandle) -> Option<DustViewParams> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    let view = mgr.views.get(&pe)?;
    let host = view.host.lock();
    let (white, softening) = host.display();
    Some(DustViewParams {
      pyramid: view.pyramid.as_ref().map(|b| b.address).unwrap_or(0),
      levels: view.layout.level_count(),
      unit: host.unit(),
      white,
      white_measured: host.display_white_measured(),
      p50: host.p50,
      softening,
      d_max_m: host.shell.d_max_m,
      n_eye_m3: host.shell.n_eye_m3 as f32,
      specks: host.shell.specks,
      specks_overflow: host.shell.overflow,
    })
  }

  /// Diagnostic: the pyramid layout of `pe`'s view.
  pub fn dust_view_layout(&self, pe: gpu::PresentationEngineHandle) -> Option<PyramidLayout> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref()?;
    mgr.views.get(&pe).map(|v| v.layout.clone())
  }

  /// Diagnostic / observer: downloads the whole pyramid of `pe`'s view as written by the last
  /// completed frame (waits for the graphics queue). The first [`dust::PYRAMID_HEADER_WORDS`]
  /// words are the header. Expensive: not for every frame.
  pub fn download_dust_pyramid(
    &self,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<alloc::vec::Vec<u32>> {
    let (buffer, words, mapped_cpu) = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
      let last = view.host.lock().frame.wrapping_sub(1);
      let cpu_mapped = view.cpu_pyramid(last).map(|(_, m)| m).filter(|m| !m.is_null());
      (pyramid.buffer, view.layout.total_words, cpu_mapped)
    };
    if let Some(ptr) = mapped_cpu {
      // CPU mode: the last completed frame's host pyramid (frames are in flight: wait for the
      // graphics queue through a serialized empty submission, never `vkQueueWaitIdle`, which the
      // render thread's submissions would race)
      let (handle, _) = self.get_command_buffer_and_native()?;
      self.begin_command_buffer_all(handle, QueueRole::Graphics)?;
      let (sem, value) =
        self.submit_command_buffer_generic(handle, None, &[], &[], QueueRole::Graphics)?;
      self.device.wait_for_semaphore_value(sem, value, DUST_DOWNLOAD_WAIT_NS)?;
      let v = unsafe { core::slice::from_raw_parts(ptr.cast::<u32>(), words as usize) }.to_vec();
      return Ok(v);
    }
    let bytes = self.dust_download_bytes(buffer, 0, words as u64 * 4)?;
    Ok(pod_vec(&bytes))
  }

  /// `size` bytes of `buffer` from `offset` (diagnostics): a graphics-queue copy after the last
  /// frame's splats and composite into a host-read staging buffer, waited for
  /// ([`DUST_DOWNLOAD_WAIT_NS`]); a copy whose wait failed leaks its staging buffer rather than
  /// freeing it under the GPU. Shared by [`Self::download_dust_pyramid`] and
  /// [`Self::download_dust_snapshot`].
  fn dust_download_bytes(
    &self,
    buffer: vk::Buffer,
    offset: u64,
    size: u64,
  ) -> GpuResult<alloc::vec::Vec<u8>> {
    let res = self.res.read();
    let allocator = res.allocator.allocator.as_allocator_view();
    let (staging, mapped) = self.dust_create_buffer(
      &allocator,
      size,
      DustMem::HostRead,
      false,
      vk::BufferUsageFlags::TRANSFER_DST,
      "DustDiagDownload",
    )?;
    let mut in_flight = false;
    let out: GpuResult<alloc::vec::Vec<u8>> = (|| {
      let (handle, cmd) = self.get_command_buffer_and_native()?;
      self.begin_command_buffer_all(handle, QueueRole::Graphics)?;
      let region = vk::BufferCopy::default().src_offset(offset).size(size);
      // in-order on the graphics queue after the last frame's splats and composite
      let pre = vk::MemoryBarrier2::default()
        .src_stage_mask(
          vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        )
        .src_access_mask(
          vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ,
        )
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
      let b = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::HOST)
        .dst_access_mask(vk::AccessFlags2::HOST_READ);
      unsafe {
        let dep0 = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&pre));
        self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep0);
        self
          .device
          .cmd_copy_buffer(cmd, buffer, staging.buffer, core::slice::from_ref(&region));
        let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
        self.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
      }
      let (sem, value) =
        self.submit_command_buffer_generic(handle, None, &[], &[], QueueRole::Graphics)?;
      in_flight = true;
      self.device.wait_for_semaphore_value(sem, value, DUST_DOWNLOAD_WAIT_NS)?;
      in_flight = false;
      allocator.invalidate_allocation(&staging.alloc, 0, size)?;
      Ok(unsafe { core::slice::from_raw_parts(mapped.cast::<u8>(), size as usize) }.to_vec())
    })();
    if in_flight {
      // the copy was submitted and its wait failed or timed out: the GPU may still write the
      // staging buffer, so it is leaked rather than freed under it (a use-after-free and a
      // plausible GPU fault otherwise); a diagnostics-only path
      aethervk_oshal_rlib::log!(
        "[Dust] download: the copy did not complete; staging buffer leaked ({} bytes)",
        size
      );
    } else {
      let mut a = staging.alloc;
      unsafe { allocator.destroy_buffer(staging.buffer, &mut a) };
    }
    out
  }

  /// What the GPU's splat passes of the last fully recorded frame of `pe` drew: per tier the push
  /// constants, the flow clock, the tracer height and the render clusters and moments they read,
  /// downloaded (diagnostics: the observer's `--probe` replays them on the CPU mirror,
  /// `dust::splat_tier_probed`). The simulation must be paused (the render thread re-draws the
  /// same epoch every frame, so the buffers hold what that frame read). Empty in CPU particle mode.
  pub fn download_dust_snapshot(
    &self,
    pe: gpu::PresentationEngineHandle,
  ) -> GpuResult<alloc::vec::Vec<DustSplatSnapshot>> {
    let records = {
      let res = self.res.read();
      let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
      let view = mgr.views.get(&pe).ok_or(gpu_err!("dust view of PE {} absent", pe.0))?;
      let r = view.splat_records.lock();
      if r[1].1.is_empty() {
        r[0].1.clone()
      } else {
        r[1].1.clone()
      }
    };
    let mut out = alloc::vec::Vec::with_capacity(records.len());
    // the tier's live range gathered from the slot-indexed buffer into compact ring order (two
    // copies when it wraps)
    let gather = |buffer: vk::Buffer, offset: u64, stride: u64, rec: &DustSplatRecord| -> GpuResult<alloc::vec::Vec<u8>> {
      let cap = rec.capacity.max(1);
      let n = rec.pc.live_count.min(cap);
      let first = rec.first_slot & (cap - 1);
      let head = n.min(cap - first);
      if n == 0 {
        return Ok(alloc::vec::Vec::new());
      }
      let mut bytes = self.dust_download_bytes(buffer, offset + first as u64 * stride, head as u64 * stride)?;
      if n > head {
        bytes.extend_from_slice(&self.dust_download_bytes(buffer, offset, (n - head) as u64 * stride)?);
      }
      Ok(bytes)
    };
    for rec in records {
      let rb = gather(
        rec.render,
        rec.render_offset,
        core::mem::size_of::<dust::DustRenderCluster>() as u64,
        &rec,
      )?;
      let mb = gather(
        rec.moments,
        rec.moments_offset,
        core::mem::size_of::<dust::DustMoments>() as u64,
        &rec,
      )?;
      out.push(DustSplatSnapshot {
        pc: rec.pc,
        flow: rec.flow,
        tracer_counts: rec.tracer_counts,
        render: pod_vec(&rb),
        moments: pod_vec(&mb),
      });
    }
    Ok(out)
  }

  /// Traces this frame of system `id` (`gpu::dust_trace`): between the splat passes' post barrier
  /// and [`Self::cmd_dust_system_end`]. `ring_bases[k]` is the sub-ring base of `meta.tiers[k]`.
  /// GPU: copies the traced render clusters into this frame's readback slot (the line is written
  /// when the slot is read back); CPU mode: writes the line now.
  pub fn cmd_dust_trace(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    meta: crate::gpu::dust_trace::TraceFrameMeta,
    ring_bases: &[u32],
  ) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    let stride = core::mem::size_of::<DustRenderCluster>() as u64;
    let Some(render) = sys.render.as_ref() else {
      // CPU mode: the evaluated clusters are still on the host
      let cpu = sys.cpu_render.lock();
      let clusters: alloc::vec::Vec<alloc::vec::Vec<DustRenderCluster>> = meta
        .tiers
        .iter()
        .zip(ring_bases)
        .map(|(t, base)| {
          let v = cpu.get(base).map(|(r, _)| r);
          t.samples
            .iter()
            .map(|&(i, _)| {
              v.and_then(|v| v.get(i as usize).copied())
                .unwrap_or(bytemuck::Zeroable::zeroed())
            })
            .collect()
        })
        .collect();
      crate::gpu::dust_trace::append(&crate::gpu::dust_trace::record_line(&meta, &clusters));
      return Ok(());
    };
    let mut trace = sys.trace_rb.lock();
    if trace.is_none() {
      let allocator = res.allocator.allocator.as_allocator_view();
      *trace = Some(self.dust_create_buffer(
        &allocator,
        DUST_READBACK_SLOTS as u64 * TRACE_SLOT_BYTES,
        DustMem::HostRead,
        false,
        vk::BufferUsageFlags::TRANSFER_DST,
        &alloc::format!("DustTrace_{id}"),
      )?);
    }
    let (tb, _) = trace.as_ref().unwrap();
    let slot = (sys.host.lock().frame % DUST_READBACK_SLOTS as u64) as usize;
    let per_tier = crate::gpu::dust_trace::TRACE_SAMPLES as u64;
    let mut regions = alloc::vec::Vec::new();
    for (t, &base) in meta.tiers.iter().zip(ring_bases) {
      if t.tier >= dust::LOD_MAX_TIERS {
        continue;
      }
      // (the GPU render buffer is slot indexed: the sample's ring slot)
      for (k, &(_, ring_slot)) in t.samples.iter().enumerate().take(per_tier as usize) {
        regions.push(
          vk::BufferCopy::default()
            .src_offset((base as u64 + ring_slot as u64) * stride)
            .dst_offset(
              slot as u64 * TRACE_SLOT_BYTES + (t.tier as u64 * per_tier + k as u64) * stride,
            )
            .size(stride),
        );
      }
    }
    if !regions.is_empty() {
      unsafe { self.device.cmd_copy_buffer(cmd, render.buffer, tb.buffer, &regions) };
    }
    sys.trace_meta.lock()[slot] = Some(meta);
    Ok(())
  }

  /// Records (GPU) or performs (CPU mode) the per-frame evaluation of the live range
  /// `[first_slot, first_slot + live_count)` of the sub-ring `[ring_base, ring_base + capacity)`
  /// (one age tier) of system `id`. Must be outside a render pass.
  /// Returns the device addresses of the compact render and moments buffers (0 in CPU mode,
  /// where the clusters stay on the host for [`Self::cmd_dust_splat`]).
  pub fn cmd_dust_propagate(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    ring_base: u32,
    capacity: u32,
    first_slot: u32,
    live_count: u32,
    frame: &DustFrame,
  ) -> GpuResult<(u64, u64)> {
    self
      .propagate(cmd, id, ring_base, capacity, first_slot, live_count, frame, None, 0)
      .map(|(addrs, _)| addrs)
  }

  /// Sets the scene's trajectory detail (`dust::clamp_propagate_detail`: 17, 9 or 5 solved
  /// size-polyline edges, the rest interpolated) for every propagate from the next on; returns
  /// the value applied (the default without a dust manager). `prepare_dust` stamps the scene's
  /// setting every frame; tests set it directly.
  pub fn dust_set_propagate_detail(&self, detail: u32) -> u32 {
    let d = dust::clamp_propagate_detail(detail);
    if let Some(mgr) = self.res.read().dust_manager.as_ref() {
      mgr.propagate_detail.store(d, core::sync::atomic::Ordering::Relaxed);
    }
    d
  }

  /// The scene's trajectory detail in force ([`Self::dust_set_propagate_detail`])
  pub fn dust_propagate_detail(&self) -> u32 {
    self
      .res
      .read()
      .dust_manager
      .as_ref()
      .map_or(dust::DUST_PROPAGATE_DETAIL_DEFAULT, |m| m.propagate_detail.load(core::sync::atomic::Ordering::Relaxed))
  }

  /// Test hook: the propagate's skip flag for system `id` (a slot whose moments already carry the
  /// tier's current generation stamp is left as it is; off by default)
  pub fn dust_set_propagate_skip(&self, id: u64, on: bool) -> GpuResult<()> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    sys.propagate_skip.store(on, core::sync::atomic::Ordering::Relaxed);
    Ok(())
  }

  /// The propagate generation of the tier at `ring_base` of system `id` (the stamp its last
  /// propagate wrote, `edges[12].w`), `None` before the first
  pub fn dust_propagate_generation(&self, id: u64, ring_base: u32) -> Option<u32> {
    let res = self.res.read();
    let sys = res.dust_manager.as_ref()?.systems.get(&id)?;
    sys.propagate_gen.lock().get(&ring_base).map(|e| e.1)
  }

  /// [`Self::cmd_dust_propagate`], skipped when the tier's inputs equal its last recorded
  /// propagate's ([`PropagateKey`]: the same frame constants, slot range and buffers, and no
  /// emission since `emitted`, the newest emission timeline value the frame waits on). The render
  /// and moments buffers then still hold that propagate's output: a paused or unchanged frame,
  /// a second view of the same tier. Returns the addresses and whether the dispatch was skipped.
  /// CPU mode never skips.
  #[allow(clippy::too_many_arguments)]
  pub fn cmd_dust_propagate_cached(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    ring_base: u32,
    capacity: u32,
    first_slot: u32,
    live_count: u32,
    frame: &DustFrame,
    emitted: u64,
  ) -> GpuResult<((u64, u64), bool)> {
    self.propagate(cmd, id, ring_base, capacity, first_slot, live_count, frame, Some(emitted), 0)
  }

  #[allow(clippy::too_many_arguments)]
  fn propagate(
    &self,
    cmd: vk::CommandBuffer,
    id: u64,
    ring_base: u32,
    capacity: u32,
    first_slot: u32,
    live_count: u32,
    frame: &DustFrame,
    emitted: Option<u64>,
    // the cadence's propagate bitmask (the address of the word holding the tier's first bit; 0 =
    // every live slot): only marked slots are evaluated, the stamp skip on, no dispatch-level cache
    prop_mask: u64,
  ) -> GpuResult<((u64, u64), bool)> {
    let res = self.res.read();
    let mgr = res.dust_manager.as_ref().ok_or(gpu_err!("dust manager absent"))?;
    let sys = mgr.systems.get(&id).ok_or(gpu_err!("dust system {} not found", id))?;
    if !capacity.is_power_of_two() || ring_base as u64 + capacity as u64 > sys.capacity as u64 {
      return Err(gpu_err!(
        "dust sub-ring {}+{} exceeds capacity {}",
        ring_base,
        capacity,
        sys.capacity
      ));
    }
    let live_count = live_count.min(capacity);
    let mask = capacity - 1;
    // the trajectory detail's edge stride (round 8 step D): the same on both mirrors
    let stride = dust::propagate_edge_stride(mgr.propagate_detail.load(core::sync::atomic::Ordering::Relaxed));
    if let Some(ring) = sys.cpu_ring.as_ref() {
      // CPU mode: evaluate on the host; cmd_dust_splat scatters the result
      let ring = ring.read();
      let mut rs = alloc::vec::Vec::with_capacity(live_count as usize);
      let mut ms = alloc::vec::Vec::with_capacity(live_count as usize);
      for i in 0..live_count {
        let slot = first_slot.wrapping_add(i) & mask;
        let (r, m) = dust::packet_moments_detail(
          &ring[(ring_base + slot) as usize],
          slot,
          dust::ring_log2_capacity(mask),
          frame,
          stride,
        );
        rs.push(r);
        ms.push(m);
      }
      sys.cpu_render.lock().insert(ring_base, (rs, ms));
      return Ok(((0, 0), false));
    }
    let (Some(clusters), Some(render), Some(moments)) = (
      sys.clusters.as_ref(),
      sys.render.as_ref(),
      sys.moments.as_ref(),
    ) else {
      return Err(gpu_err!("dust system {} has no GPU buffers", id));
    };
    // each tier evaluates into its own part of the render / moments buffers
    let render_offset = ring_base as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
    let moments_offset = ring_base as u64 * core::mem::size_of::<DustMoments>() as u64;
    let skip_moments = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_DEBUG_SKIP")
      .is_some_and(|v| v.split(',').any(|p| p.trim() == "moments"));
    let key = emitted.map(|emitted| PropagateKey {
      frame: *frame,
      capacity,
      first_slot,
      live_count,
      render: render.address + render_offset,
      moments: if skip_moments { 0 } else { moments.address + moments_offset },
      emitted,
      stride,
    });
    let skipped =
      prop_mask == 0 && live_count > 0 && key.is_some() && sys.propagated.lock().get(&ring_base) == key.as_ref();
    match key {
      // (a masked propagate evaluates part of the range: unknown to the cache)
      Some(k) if live_count > 0 && prop_mask == 0 => {
        sys.propagated.lock().insert(ring_base, k);
      }
      // an uncached or empty propagate leaves the buffers unknown to the cache
      _ => {
        sys.propagated.lock().remove(&ring_base);
      }
    }
    // the generation stamp: bumped when the inputs differ from the last propagate's (never 0,
    // the CPU mirror's stamp)
    let generation = {
      let gk = PropagateGenKey {
        frame: *frame,
        capacity,
        first_slot,
        live_count,
        emitted,
        stride,
      };
      let mut gens = sys.propagate_gen.lock();
      let e = gens.entry(ring_base).or_insert((gk, 1));
      if e.0 != gk {
        e.1 = e.1.wrapping_add(1).max(1);
        e.0 = gk;
      }
      e.1
    };
    // (perf9_K) the propagate family (`dust_prop_staged`): `Some` the pre-package monolith (the
    // GTX 1070 default), `None` the package below
    let prop_mono = if self.dust_prop_staged() { None } else { self.kernels.pipelines.dust_propagate_mono };
    if let (Some(mono), true) = (prop_mono, live_count > 0 && !skipped) {
      // perf9_K: the pre-package dispatch (`dust_propagate_mono.comp`): a thread per live slot, one
      // dispatch, no rows buffer, no moments stage; push word 8 is the ring mask (the package's
      // `rows_lo`), skip.x only the flag and the stride bits
      let pc = DustPropagatePushConstants {
        clusters: clusters.address + ring_base as u64 * core::mem::size_of::<DustCluster>() as u64,
        render: render.address + render_offset,
        moments: if skip_moments {
          0
        } else {
          moments.address + moments_offset
        },
        first_slot,
        live_count,
        rows_lo: mask,
        generation,
        skip: [
          (prop_mask != 0 || sys.propagate_skip.load(core::sync::atomic::Ordering::Relaxed)) as u32
            | dust::propagate_stride_bits(stride),
          prop_mask as u32,
        ],
        // `spin.w` (unused by the shader) carries the bitmask address's high word
        frame: DustFrame {
          spin: [frame.spin[0], frame.spin[1], frame.spin[2], f32::from_bits((prop_mask >> 32) as u32)],
          ..*frame
        },
      };
      self.kernels.pipelines.assert_pc_size(mono, core::mem::size_of_val(&pc));
      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, mono);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(&pc),
        );
        self.device.cmd_dispatch(cmd, live_count.div_ceil(DUST_WG), 1, 1);
      }
    } else if live_count > 0 && !skipped {
      // round 9 step O: 3 clusters per workgroup, each lane's df64 solve state in its row of the
      // rows buffer (one chunk of `DUST_PROP_CHUNK_WGS` workgroups at a time): per chunk the solve
      // stage (the monolith, or the staged passes 1, (2, 3) × iters, 6, 4, 5m, 5s where the
      // monolith does not compile), a barrier, the moments stage over the same groups, a barrier
      // before the next chunk reuses the rows. The shader maps a chunk's offset `i` to the slot
      // `(firstSlot + i) & mask`, so the chunk's `first_slot` advances by its clusters.
      let rows = sys.rows.as_ref().ok_or(gpu_err!("dust system {} has no rows buffer", id))?;
      if rows.address >> dust::DUST_PROP_ROWS_ADDR_BITS != 0 {
        return Err(gpu_err!("dust rows buffer address {:#x} beyond 56 bits", rows.address));
      }
      let rows_hi = ((rows.address >> 32) as u32) & 0x00FF_FFFF;
      let log2cap = capacity.trailing_zeros();
      let skip0 = (prop_mask != 0 || sys.propagate_skip.load(core::sync::atomic::Ordering::Relaxed)) as u32
        | dust::propagate_stride_bits(stride)
        | (log2cap << 3)
        | (rows_hi << 8);
      let per_chunk = dust::DUST_PROP_CHUNK_WGS * dust::DUST_PROP_CLUSTERS;
      let chunks = live_count.div_ceil(per_chunk);
      let pipelines = &self.kernels.pipelines;
      let moments_stage = pipelines.dust_propagate_moments;
      pipelines.assert_pc_size(pipelines.dust_propagate, core::mem::size_of::<DustPropagatePushConstants>());
      pipelines.assert_pc_size(moments_stage, core::mem::size_of::<DustPropagatePushConstants>());
      for c in 0..chunks {
        let live_c = (live_count - c * per_chunk).min(per_chunk);
        let groups = live_c.div_ceil(dust::DUST_PROP_CLUSTERS);
        let x = groups.min(DUST_DISPATCH_MAX_X);
        let y = groups.div_ceil(x);
        let pc = DustPropagatePushConstants {
          clusters: clusters.address + ring_base as u64 * core::mem::size_of::<DustCluster>() as u64,
          render: render.address + render_offset,
          moments: if skip_moments {
            0
          } else {
            moments.address + moments_offset
          },
          first_slot: first_slot.wrapping_add(c * per_chunk),
          live_count: live_c,
          rows_lo: rows.address as u32,
          generation,
          skip: [skip0, prop_mask as u32],
          // `spin.w` (unused by the shader) carries the bitmask address's high word
          frame: DustFrame {
            spin: [frame.spin[0], frame.spin[1], frame.spin[2], f32::from_bits((prop_mask >> 32) as u32)],
            ..*frame
          },
        };
        let run = |p: vk::Pipeline| unsafe {
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, p);
          self.device.cmd_push_constants(
            cmd,
            pipelines.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
          );
          self.device.cmd_dispatch(cmd, x, y, 1);
        };
        match pipelines.dust_propagate_passes {
          Some(passes) => {
            // 1, (2, 3) × DUST_KEPLER_ITERS_DF, 6, 4, 5m, 5s: a barrier between passes (the two
            // pass-5 blobs touch disjoint lanes and words: none between them)
            run(passes[0]);
            for _ in 0..dust::DUST_KEPLER_ITERS_DF {
              self.cmd_dust_pre_splat_barrier(cmd);
              run(passes[1]);
              self.cmd_dust_pre_splat_barrier(cmd);
              run(passes[2]);
            }
            self.cmd_dust_pre_splat_barrier(cmd);
            run(passes[3]);
            self.cmd_dust_pre_splat_barrier(cmd);
            run(passes[4]);
            self.cmd_dust_pre_splat_barrier(cmd);
            run(passes[5]);
            run(passes[6]);
          }
          None => run(pipelines.dust_propagate),
        }
        self.cmd_dust_pre_splat_barrier(cmd);
        run(moments_stage);
        if c + 1 < chunks {
          self.cmd_dust_pre_splat_barrier(cmd);
        }
      }
    }
    Ok((
      (render.address + render_offset, moments.address + moments_offset),
      skipped,
    ))
  }
}

/// The play pin (handoff P3b): host logic, no GPU
#[cfg(test)]
mod play_pin_tests {
  use super::*;
  use dust::play_quality::{PinnedFeedback, PlayFeedback};

  fn taught_host() -> DustViewHost {
    // a view whose feedback learned a paused state (both regimes, a memory, the warm-up over)
    let mut h = DustViewHost::default();
    h.split_depth = 2;
    h.split_depth_over = 3;
    h.volume_bias = 4;
    h.ewa_bias = 1;
    h.fb_memory.ewa_step_q8[1] = 512;
    h.fb_memory.volume_prev = (4, 1234);
    h.fb_full = FeedbackSet { split_depth: 2, volume_bias: 4, ewa_bias: 1, ..FeedbackSet::default() };
    h.fb_cad = Some(FeedbackSet { split_depth: 1, volume_bias: 6, ewa_bias: 2, ..FeedbackSet::default() });
    h.warmup = [false; 2];
    h
  }

  /// entering pins the frame's biases over the regime's; leaving restores the paused state
  /// byte-identical, whatever the play frames' readbacks taught in between
  #[test]
  fn play_pin_enter_apply_leave_restores_the_paused_state() {
    let mut h = taught_host();
    let before = h.saved_feedback();
    let pin = PlayFeedback::default().pin(11);
    assert_eq!(pin, PinnedFeedback { split_depth: 0, volume_bias: 10, ewa_bias: 3 });
    assert!(h.play_pin_enter(pin), "the first call enters");
    assert!(!h.play_pin_enter(pin), "a repeated call only updates");
    // a frame begin: the cadenced regime loaded, then the pin
    h.feedback_load(1);
    h.apply_play_pin();
    assert_eq!((h.split_depth, h.volume_bias, h.ewa_bias), (0, 10, 3));
    assert_eq!(h.full_frame_feedback().volume_bias, 10, "a full frame's header gets the pin too");
    // the play frames' readbacks move the regimes (what `absorb_at` + `feedback_store` do)
    h.volume_bias = 13;
    h.ewa_bias = 5;
    h.split_depth_over = 0;
    h.fb_memory.ewa_step_q8[5] = 9;
    h.feedback_store(1);
    h.feedback_store(0);
    h.warmup = [true; 2];
    assert!(h.play_pin_leave(), "a pin was left");
    assert_eq!(h.saved_feedback(), before, "paused state restored byte-identical");
    assert!(h.play.is_none());
    assert!(!h.play_pin_leave(), "nothing to leave twice");
    // without a pin the frame's biases are the regime's own
    h.feedback_load(0);
    h.apply_play_pin();
    assert_eq!((h.split_depth, h.volume_bias, h.ewa_bias), (2, 4, 1));
    assert_eq!(h.full_frame_feedback(), h.fb_full);
  }

  /// a feedback reset under the pin (a seek past a day while playing) is what the pause restores
  #[test]
  fn play_pin_reset_under_the_pin_is_restored_as_the_reset() {
    let mut h = taught_host();
    assert!(h.play_pin_enter(PlayFeedback::default().pin(11)));
    h.reset_feedback();
    assert!(h.play_pin_leave());
    let cold = FeedbackSet::coarse();
    assert_eq!((h.split_depth, h.volume_bias, h.ewa_bias), (cold.split_depth, cold.volume_bias, cold.ewa_bias));
    assert_eq!(h.fb_full, cold);
    assert_eq!(h.fb_cad, None);
    assert_eq!(h.warmup, [true; 2]);
  }
}
