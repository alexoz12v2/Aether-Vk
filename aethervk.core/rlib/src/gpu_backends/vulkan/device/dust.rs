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
  /// largest packet peak optical depth per px² of the last measured frame (raw exposure units,
  /// `dust::PYR_TAU_MAX`): the fixed-point unit follows it, so no view measures as empty
  pub tau_max: f32,
  pub frame: u64,
  /// per readback slot: the unit when it was recorded
  slot_unit: [f32; DUST_READBACK_SLOTS],
  slot_valid: [bool; DUST_READBACK_SLOTS],
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
  pub vqueue_capacity: u32,
  pub vspill_capacity: u32,
  pub vspill_counts: (u32, u32),
  pub vqueue_chunk: u32,
  pub vqueue_ppc: f32,
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
  /// the in-frame gates off for this view (a test hook, [`Device::dust_set_frame_gates`])
  pub frame_gates_off: bool,
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
      tau_max: 0.0,
      frame: 0,
      slot_unit: [0.0; DUST_READBACK_SLOTS],
      slot_valid: [false; DUST_READBACK_SLOTS],
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
      vqueue_capacity: 0,
      vspill_capacity: 0,
      vspill_counts: (0, 0),
      vqueue_chunk: 0,
      vqueue_ppc: 64.0,
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
      frame_gates_off: false,
      drawn: None,
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

  /// Folds one frame's header and white-point level (recorded with `unit`) into the white
  /// point, the median and the unit.
  pub fn absorb(&mut self, header: &[u32], level: &[u32], texel_px: u32, unit: f32) {
    if let Some((p99, p50)) = dust::white_point_from_level(level, texel_px, unit) {
      self.white = dust::adapt_white(self.white, p99);
      self.p50 = dust::adapt_white(self.p50, p50);
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
    let next = if split.children > dust::DUST_SPLIT_BUDGET {
      let ratio = split.children as f32 / dust::DUST_SPLIT_BUDGET as f32;
      let down = (<f32 as aethervk_oshal_rlib::math::FloatLike>::ln(ratio)
        / core::f32::consts::LN_2)
        .ceil() as u32;
      split.depth.saturating_sub(down.max(1))
    } else {
      let up = split.suggested_depth();
      if up >= self.split_depth_over {
        split.depth
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
    // over the budget: coarser by log4(evals / budget) levels at once (a level divides the
    // texels by 4; inside the coma the first frame asks ~45× the budget, and one level per
    // readback took four slow frames to settle); else the one-step rule
    let bias = if vol.evals > dust::DUST_VOLUME_FRAME_BUDGET {
      let ratio = vol.evals as f64 / dust::DUST_VOLUME_FRAME_BUDGET as f64;
      let up = (ratio.ln() / 4f64.ln()).ceil().max(1.0) as u32;
      (vol.bias + up).min(dust::DUST_VOLUME_BIAS_MAX)
    } else {
      vol.suggested_bias()
    };
    if bias != self.volume_bias {
      aethervk_oshal_rlib::log!(
        "[Dust volume] {} ray-integral evaluations at bias {} (budget {}): bias {} for the next frames",
        vol.evals,
        vol.bias,
        dust::DUST_VOLUME_FRAME_BUDGET,
        bias
      );
    }
    self.volume_bias = bias;
    let ewa = dust::EwaStats::load(header);
    let ewa_bias = ewa.suggested_bias();
    if ewa_bias != self.ewa_bias {
      aethervk_oshal_rlib::log!(
        "[Dust EWA] {:.3e} texels at bias {} (budget {:.3e}): bias {} for the next frames",
        ewa.texels as f64,
        ewa.bias,
        dust::DUST_EWA_FRAME_BUDGET as f64,
        ewa_bias
      );
    }
    self.ewa_bias = ewa_bias;
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

  /// what the composite needs: `(white, auto softening)`
  pub fn display(&self) -> (f32, f32) {
    (
      dust::display_white_v4(self.white),
      dust::auto_softening(self.p50, self.white),
    )
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
  /// white point (the seed surface brightness before any measurement)
  pub white: f32,
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
    let (render, moments) = (sys.render.take(), sys.moments.take());
    let compute_written = if compute_done {
      [render, moments]
    } else {
      [None, None]
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
    let pipeline = self.kernels.pipelines.dust_emit;
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, batch.count.div_ceil(DUST_WG), 1, 1);
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
      let offsets = dust::dust_queue_offsets(vqueue_records);
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
      let rb_bytes = DUST_READBACK_SLOTS as u64 * DustView::readback_words(&layout) as u64 * 4;
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
    let mut host = view.host.lock();
    let slot = (host.frame % DUST_READBACK_SLOTS as u64) as usize;
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
        let (rest, vqueue) = rest.split_at(rest.len() - dust::DUST_VQUEUE_HEADER_WORDS as usize);
        let (level, queue) = rest.split_at(rest.len() - dust::DUST_QUEUE_HEADER_WORDS as usize);
        host.absorb(header, level, 1 << lvl, unit);
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
        if clusters > 0 && vdrawn + voverflowed > 0 {
          let ppc = (vdrawn + voverflowed) as f32 / clusters as f32;
          host.vqueue_ppc += (ppc - host.vqueue_ppc) * 0.25;
        }
        if aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE_LOG").is_some_and(|v| v.trim() == "1") {
          aethervk_oshal_rlib::log!(
            "[Dust] view {}: vqueue {} packets drawn by the consumer, {} overflowed, {} chunks over {} clusters (ppc estimate {:.1})",
            pe.0, vdrawn, voverflowed, host.slot_chunks[slot], clusters, host.vqueue_ppc
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
    let unit = host.unit();
    host.slot_unit[slot] = unit;
    let (white, softening) = host.display();
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
        .unwrap_or(dust::DUST_LOD_PX_DEFAULT)
    });
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
      .unwrap_or(host.ewa_bias)
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
    header[dust::PYR_FLAGS as usize] = flags | queue_bits | vqueue_bits | deep_bits;
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
      None if specks_enabled() => shell.d_max_m,
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
      }
    }
    Ok((DustViewParams {
      pyramid: pyramid.address,
      levels: view.layout.level_count(),
      unit,
      white,
      p50,
      softening,
      d_max_m: d_max,
      n_eye_m3: shell.n_eye_m3 as f32,
      specks: shell.specks,
      specks_overflow: shell.overflow,
    }, reuse))
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
    // cluster's walk mask), a barrier for the mask, then the walk pass (a workgroup of 32 lanes
    // per cluster × bin; the ones not marked return at once)
    let pipeline = self.kernels.pipelines.dust_specks;
    let walk = self.kernels.pipelines.dust_specks_walk;
    self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
    self.kernels.pipelines.assert_pc_size(walk, core::mem::size_of_val(&pc));
    let threads = pc.live_count.saturating_mul(dust::SIZE_BINS);
    // the spill's header for this pass, before the gate (both passes read its ring words): no
    // slot claimed, the view's slot count, the trace's capacity (a test hook) and count, the
    // tier's live range start (ring slot) and ring mask (the buffers are slot indexed). After the
    // previous tier's walk (its spill atomics) and before this tier's passes.
    let spill_offset = (view.queue_word() + dust::dust_queue_offsets(view.vqueue_records).spill) as u64 * 4;
    let trace_cap = if view.host.lock().specks_trace { dust::DUST_SPECKS_TRACE_CAPACITY } else { 0 };
    // the cadence's owner bytes (SHOWN slots are not walked; 0: no cadence)
    let owner = self.cad_owner_address(mgr, pe, id, ring_base);
    let to_update = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
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
        ]),
      );
    }
    self.cmd_dust_pre_splat_barrier(cmd);
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, threads.div_ceil(DUST_WG), 1, 1);
    }
    // the gate's walk masks → the walk pass
    self.cmd_dust_pre_splat_barrier(cmd);
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, walk);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      // a workgroup per cluster × bin in two dimensions: at most DUST_DISPATCH_MAX_X per
      // dimension (Vulkan's required maxComputeWorkGroupCount is 65 535, which the baseline
      // devices report; a tier of 131 072 clusters is 2.1 M workgroups)
      let x = threads.min(DUST_DISPATCH_MAX_X);
      let y = threads.div_ceil(x.max(1));
      self.device.cmd_dispatch(cmd, x, y, 1);
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
    let producer = match self.kernels.pipelines.dust_splat_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_splat,
    };
    let consumer = match self.kernels.pipelines.dust_draw_volume_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_draw_volume,
    };
    let splitter = match self.kernels.pipelines.dust_split_prof {
      Some(p) if profiling => p,
      _ => self.kernels.pipelines.dust_split,
    };
    self.kernels.pipelines.assert_pc_size(producer, core::mem::size_of_val(&pc));
    self.kernels.pipelines.assert_pc_size(consumer, core::mem::size_of_val(&pc));
    self.kernels.pipelines.assert_pc_size(splitter, core::mem::size_of_val(&pc));
    // the tier in chunks of clusters (`dust::DUST_VQUEUE_*`): per chunk the volume queue and
    // volume spill headers (empty, the range [first, end), the capacities), the producer over
    // the range, the splitter, then the consumer over the queue's records and again over the
    // spill's (`dust::DUST_VIEW_VOLUME_SPILL`); with the queue off every record goes through the
    // spill. `AETHERVK_DUST_VQUEUE_NODRAW=1` skips the consumer (diagnostics: the producer
    // alone; the image wrong by design).
    let (chunk, vqueue_capacity, vspill_capacity) = {
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
      (chunk.max(1), capacity, spill_capacity)
    };
    let nodraw =
      aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VQUEUE_NODRAW").is_some_and(|v| v.trim() == "1");
    let offsets = dust::dust_queue_offsets(view.vqueue_records);
    let vbase = (view.queue_word() + offsets.vqueue) as u64 * 4;
    let sbase = (view.queue_word() + offsets.squeue) as u64 * 4;
    let vsbase = (view.queue_word() + offsets.vspill) as u64 * 4;
    // the consumer's second dispatch of each chunk drains the volume spill (host-internal flag)
    let mut pc_spill = pc;
    pc_spill.flags |= dust::DUST_VIEW_VOLUME_SPILL;
    // the consumer's (and the previous producer's) storage writes → the header update
    let to_update = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE | vk::AccessFlags2::SHADER_STORAGE_READ)
      .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE);
    // the header update → the producer's reads and writes
    let to_producer = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
      .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
      .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE);
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
    let mut first = 0u32;
    let mut chunks = 0u32;
    loop {
      let end = first.saturating_add(chunk).min(count);
      let hdr = [0u32, 1, 1, 0, vqueue_capacity, first, end];
      unsafe {
        if first > 0 {
          self.device.synchronization2.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_update)),
          );
        }
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
        // (words 4-5 the list, 6-7 the owner bytes: dust_splat.comp)
        let shdr = [0u32, dust::DUST_SQUEUE_CAPACITY, 0, 0, list as u32, (list >> 32) as u32, owner as u32, (owner >> 32) as u32];
        self.device.cmd_update_buffer(cmd, pyramid.buffer, sbase, bytemuck::cast_slice(&shdr));
        // the volume spill: empty, its dispatch (0, 1, 1), its capacity (covered by the same
        // `to_update` / `to_producer` barriers as the main header)
        let vshdr = [0u32, 1, 1, 0, vspill_capacity];
        self.device.cmd_update_buffer(cmd, pyramid.buffer, vsbase, bytemuck::cast_slice(&vshdr));
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_producer)),
        );
        // the enumerator: a warp per cluster, a lane per sub-packet (`dust_splat.comp`)
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, producer);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(&pc),
        );
        self.device.cmd_dispatch(cmd, ((end - first) * dust::DUST_SPLAT_LANES).div_ceil(DUST_WG), 1, 1);
        // the splitter: a workgroup per cluster of the chunk over its records (`dust_split.comp`)
        self.device.synchronization2.cmd_pipeline_barrier2(
          cmd,
          &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_consumer)),
        );
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, splitter);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(&pc),
        );
        self.device.cmd_dispatch(cmd, end - first, 1, 1);
        if !nodraw {
          self.device.synchronization2.cmd_pipeline_barrier2(
            cmd,
            &vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&to_consumer)),
          );
          self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, consumer);
          self.device.cmd_push_constants(
            cmd,
            self.kernels.pipelines.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&pc),
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
            bytemuck::bytes_of(&pc_spill),
          );
          self.device.cmd_dispatch_indirect(cmd, pyramid.buffer, vsbase);
        }
      }
      chunks += 1;
      first = end;
      if first >= count {
        break;
      }
    }
    view.host.lock().vqueue_chunks_frame += chunks;
    Ok(())
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
    let pipeline = self.kernels.pipelines.dust_splat;
    unsafe {
      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd,
        self.kernels.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytemuck::bytes_of(&pc),
      );
      self.device.cmd_dispatch(cmd, (pc.live_count * dust::DUST_SPLAT_LANES).div_ceil(DUST_WG), 1, 1);
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
      let n = (lw * lh * dust::PYRAMID_TEXEL_WORDS) as usize;
      host.absorb(
        &words[..dust::PYRAMID_HEADER_WORDS as usize],
        &words[lvl_off as usize..lvl_off as usize + n],
        1 << lvl,
        unit,
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
            - (dust::DUST_QUEUE_HEADER_WORDS + dust::DUST_VQUEUE_HEADER_WORDS) as u64 * 4,
        )
        .size(dust::DUST_QUEUE_HEADER_WORDS as u64 * 4),
      vk::BufferCopy::default()
        .src_offset((view.queue_word() + dust::DUST_QUEUE_WORDS) as u64 * 4)
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
      view.host.lock().volume_bias_override = bias;
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
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
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
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
      (pyramid.buffer, view.queue_word() as u64 * 4)
    };
    let w: alloc::vec::Vec<u32> = pod_vec(&self.dust_download_bytes(buffer, base, 32)?);
    Ok((dust::dust_queue_held(&w), w[3]))
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
    if device_local >= 4 * GIB {
      dust::DUST_VQUEUE_CAPACITY
    } else if device_local >= 2 * GIB {
      dust::DUST_VQUEUE_CAPACITY / 4
    } else {
      dust::DUST_VQUEUE_CAPACITY / 16
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
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
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
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
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
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
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
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
      (pyramid.buffer, (view.queue_word() + dust::DUST_QUEUE_WORDS) as u64 * 4)
    };
    let w: alloc::vec::Vec<u32> =
      pod_vec(&self.dust_download_bytes(buffer, base, dust::DUST_VQUEUE_HEADER_WORDS as u64 * 4)?);
    Ok((w[dust::DUST_VQUEUE_SPILL_DRAWN], w[dust::DUST_VQUEUE_LOST]))
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
      let pyramid = view.pyramid.as_ref().ok_or(gpu_err!("dust view has no pyramid"))?;
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
    if let Some(ring) = sys.cpu_ring.as_ref() {
      // CPU mode: evaluate on the host; cmd_dust_splat scatters the result
      let ring = ring.read();
      let mut rs = alloc::vec::Vec::with_capacity(live_count as usize);
      let mut ms = alloc::vec::Vec::with_capacity(live_count as usize);
      for i in 0..live_count {
        let slot = first_slot.wrapping_add(i) & mask;
        let (r, m) = dust::packet_moments_in(
          &ring[(ring_base + slot) as usize],
          slot,
          dust::ring_log2_capacity(mask),
          frame,
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
      };
      let mut gens = sys.propagate_gen.lock();
      let e = gens.entry(ring_base).or_insert((gk, 1));
      if e.0 != gk {
        e.1 = e.1.wrapping_add(1).max(1);
        e.0 = gk;
      }
      e.1
    };
    if live_count > 0 && !skipped {
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
        ring_mask: mask,
        generation,
        skip: [
          (prop_mask != 0 || sys.propagate_skip.load(core::sync::atomic::Ordering::Relaxed)) as u32,
          prop_mask as u32,
        ],
        // `spin.w` (unused by the shader) carries the bitmask address's high word
        frame: DustFrame {
          spin: [frame.spin[0], frame.spin[1], frame.spin[2], f32::from_bits((prop_mask >> 32) as u32)],
          ..*frame
        },
      };
      let pipeline = self.kernels.pipelines.dust_propagate;
      self.kernels.pipelines.assert_pc_size(pipeline, core::mem::size_of_val(&pc));
      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pipeline);
        self.device.cmd_push_constants(
          cmd,
          self.kernels.pipelines.pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          bytemuck::bytes_of(&pc),
        );
        self.device.cmd_dispatch(cmd, live_count.div_ceil(DUST_WG), 1, 1);
      }
    }
    Ok((
      (render.address + render_offset, moments.address + moments_offset),
      skipped,
    ))
  }
}
