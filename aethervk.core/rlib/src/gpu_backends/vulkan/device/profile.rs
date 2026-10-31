//! The dust passes on a Tracy GPU timeline (`profile-tracy` only).
//!
//! The engine already writes a Vulkan timestamp between the dust passes of every frame
//! ([`dust::DUST_TIMESTAMPS`], read back [`dust::DUST_READBACK_SLOTS`] frames later into
//! [`dust::DustGpuTimings`]). Those timestamps are fed here into one Tracy GPU context per device,
//! as nested GPU zones (`dust frame` > `dust propagate` …), so a capture shows the passes on the
//! GPU row under the CPU zones that recorded them. Nothing more is written on the GPU for the
//! profiler: no extra queries, no extra submits.
//!
//! **Calibration.** Tracy places a GPU zone by mapping its GPU timestamp onto the CPU clock with
//! one (CPU time, GPU time) pair, taken when the context is created, and corrected by every later
//! `sync_gpu_time`. The pair is good when the GPU timestamp is read *at* the moment of the call:
//! - with `VK_EXT_calibrated_timestamps` (the usual case on the desktop drivers), the device's
//!   current timestamp comes from `vkGetCalibratedTimestampsEXT` on the calling thread, right
//!   before the Tracy call, so the pair is off by the call itself (microseconds); the context is
//!   re-synchronised the same way about once a second against the GPU clock's drift;
//! - without it, the fallback submits one command buffer that writes a timestamp and waits for it:
//!   the GPU wrote the timestamp somewhere between the submit and the wake-up, while Tracy pairs
//!   it with the CPU time after the wait, so every GPU zone shows late by up to that latency
//!   (logged at creation, typically well under a millisecond on an idle queue), and there is no
//!   re-synchronisation. The log line `[Tracy GPU]` says which path was taken.
//!
//! The GPU zones are emitted at readback, so their CPU-side begin/end (the thin markers Tracy
//! draws on the emitting thread) are on the render tasklet four frames later; their position on
//! the GPU row is the measured one.
use super::{Device, dust};
use crate::types::GpuResult;
use ash::vk;
use core::sync::atomic::{AtomicU64, Ordering};
use tracy_client::{GpuContext, GpuContextType, GpuSpan};

/// Re-synchronise the GPU clock against the CPU clock this often (calibrated timestamps only).
const SYNC_INTERVAL_US: u64 = 1_000_000;

/// A GPU zone begun on this thread, its timestamps uploaded when it is closed.
struct OpenSpan {
  span: GpuSpan,
  start_ns: i64,
  end_ns: i64,
}

/// The Tracy GPU context of a device's graphics queue and what maps its timestamps to it.
pub struct GpuProfiler {
  ctx: GpuContext,
  /// `vkGetCalibratedTimestampsEXT`, when the extension is enabled
  calibrated: Option<ash::ext::calibrated_timestamps::Device>,
  /// ns per timestamp tick (`timestampPeriod`), applied here: Tracy gets nanoseconds (period 1)
  period_ns: f64,
  /// the valid bits of a graphics-queue timestamp
  mask: u64,
  /// monotonic µs of the last `sync_gpu_time`
  last_sync_us: AtomicU64,
}

impl GpuProfiler {
  /// Creates the context for `device`'s graphics queue, calibrated as the module doc says.
  /// `None` when the queue has no timestamps or the one-shot calibration fails (logged).
  pub fn new(device: &Device) -> Option<Self> {
    let bits = device.query_result.graphics_timestamp_bits;
    if bits == 0 {
      aethervk_oshal_rlib::log!("[Tracy GPU] no timestamps on the graphics queue: no GPU zones");
      return None;
    }
    let mask = if bits >= 64 { u64::MAX } else { (1u64 << bits) - 1 };
    let period_ns = device.query_result.physical_device_properties.limits.timestamp_period as f64;
    let calibrated = device
      .query_result
      .optional_extensions
      .contains(super::super::utils::OptionalExtensionSupportFlags::CALIBRATED_TIMESTAMPS)
      .then(|| ash::ext::calibrated_timestamps::Device::new(&device.instance.instance, &device.device));

    let (gpu_now, how) = match &calibrated {
      Some(ext) => match calibrated_device_timestamp(ext) {
        Ok(t) => (t, alloc::string::String::from("vkGetCalibratedTimestampsEXT")),
        Err(e) => {
          aethervk_oshal_rlib::log!("[Tracy GPU] vkGetCalibratedTimestampsEXT failed: {:?}", e);
          return None;
        }
      },
      None => match one_shot_timestamp(device) {
        Ok((t, latency_us)) => (
          t,
          alloc::format!("a one-shot timestamp (zones up to {} µs late)", latency_us),
        ),
        Err(e) => {
          aethervk_oshal_rlib::log!("[Tracy GPU] one-shot calibration timestamp failed: {:?}", e);
          return None;
        }
      },
    };
    let gpu_now_ns = ((gpu_now & mask) as f64 * period_ns) as i64;
    let ctx = match tracy_client::Client::start().new_gpu_context(
      Some("Vulkan graphics"),
      GpuContextType::Vulkan,
      gpu_now_ns,
      1.0,
    ) {
      Ok(ctx) => ctx,
      Err(e) => {
        aethervk_oshal_rlib::log!("[Tracy GPU] context creation failed: {:?}", e);
        return None;
      }
    };
    aethervk_oshal_rlib::log!(
      "[Tracy GPU] context created, calibrated with {} (timestampPeriod {} ns, {} bits)",
      how,
      period_ns,
      bits
    );
    Some(Self {
      ctx,
      calibrated,
      period_ns,
      mask,
      last_sync_us: AtomicU64::new(aethervk_oshal_rlib::os::time::get_monotonic_time() as u64),
    })
  }

  /// A timestamp's nanoseconds on Tracy's GPU clock.
  fn ns(&self, ticks: u64) -> i64 {
    ((ticks & self.mask) as f64 * self.period_ns) as i64
  }

  /// A finished GPU zone over `[t0, t1]` (ticks); nothing when either end is missing or they are
  /// out of order. The zone is begun and ended at once on this thread: the nesting on the GPU row
  /// follows the begin/end order of these calls, so open a parent with [`Self::open`] first.
  fn leaf(&self, name: &str, t0: Option<u64>, t1: Option<u64>) {
    if let Some(span) = self.open(name, t0, t1) {
      self.close(span);
    }
  }

  /// Begins a GPU zone whose children follow (see [`Self::leaf`]); [`Self::close`] ends it.
  fn open(&self, name: &str, t0: Option<u64>, t1: Option<u64>) -> Option<OpenSpan> {
    let (Some(a), Some(b)) = (t0, t1) else { return None };
    if b < a {
      return None;
    }
    let span = self.ctx.span_alloc(name, "dust", file!(), line!()).ok()?;
    Some(OpenSpan { span, start_ns: self.ns(a), end_ns: self.ns(b) })
  }

  /// Ends the zone, then uploads its timestamps: Tracy wants a query's time after the zone event
  /// that named the query.
  fn close(&self, mut open: OpenSpan) {
    open.span.end_zone();
    open.span.upload_timestamp_start(open.start_ns);
    open.span.upload_timestamp_end(open.end_ns);
  }

  /// The dust GPU zones of one frame, from its timestamps as [`dust::DustGpuTimings::from_timestamps`]
  /// reads them (`None`: not written that frame). Called at readback, on the render tasklet.
  pub fn emit_dust_frame(&self, ts: &[Option<u64>; dust::DUST_TIMESTAMPS as usize]) {
    use dust::*;
    let at = |i: u32| ts[i as usize];
    let Some(frame) = self.open("dust frame", at(DUST_TS_BEGIN), at(DUST_TS_MEASURED)) else {
      return;
    };
    // the same spans as `DustGpuTimings`: the propagates (the cadence's begin inside them), the
    // cadence's classifies, builds and reprojection, the specks, the splats, the packet queue, the
    // measurement grid and the readback copy
    if let Some(propagate) = self.open("dust propagate", at(DUST_TS_BEGIN), at(DUST_TS_PROPAGATED)) {
      self.leaf("dust cadence begin", at(DUST_TS_BEGIN), at(DUST_TS_CADENCE));
      self.close(propagate);
    }
    self.leaf("dust classify", at(DUST_TS_PROPAGATED), at(DUST_TS_CHUNKED));
    let chunked = at(DUST_TS_CHUNKED).is_some();
    self.leaf(
      "dust builds",
      at(if chunked { DUST_TS_CHUNKED } else { DUST_TS_PROPAGATED }),
      at(DUST_TS_BUILDS),
    );
    self.leaf("dust reproject", at(DUST_TS_BUILDS), at(DUST_TS_REPROJECTED));
    let before_specks = if at(DUST_TS_REPROJECTED).is_some() { DUST_TS_REPROJECTED } else { DUST_TS_PROPAGATED };
    self.leaf("dust specks", at(before_specks), at(DUST_TS_SPECKED));
    let after_specks = if at(DUST_TS_SPECKED).is_some() {
      DUST_TS_SPECKED
    } else if at(DUST_TS_REPROJECTED).is_some() {
      DUST_TS_REPROJECTED
    } else {
      DUST_TS_PROPAGATED
    };
    self.leaf("dust splat", at(after_specks), at(DUST_TS_SPLATS_DONE));
    self.leaf("dust queue", at(DUST_TS_SPLATTED), at(DUST_TS_QUEUED));
    let before_measure = if at(DUST_TS_QUEUED).is_some() { DUST_TS_QUEUED } else { DUST_TS_SPLATTED };
    self.leaf("dust measure", at(before_measure), at(DUST_TS_MEASURED));
    self.close(frame);
    // the composite draw runs later, inside the render pass: a sibling of the frame
    self.leaf("dust composite", at(DUST_TS_COMPOSITE_BEGIN), at(DUST_TS_COMPOSITE_END));
    self.maybe_sync();
  }

  /// About once a second, re-pairs the GPU clock with the CPU clock (the calibrated path only).
  fn maybe_sync(&self) {
    let Some(ext) = &self.calibrated else { return };
    let now = aethervk_oshal_rlib::os::time::get_monotonic_time() as u64;
    let last = self.last_sync_us.load(Ordering::Relaxed);
    if now.wrapping_sub(last) < SYNC_INTERVAL_US {
      return;
    }
    if self
      .last_sync_us
      .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
      .is_err()
    {
      return;
    }
    if let Ok(t) = calibrated_device_timestamp(ext) {
      self.ctx.sync_gpu_time(self.ns(t));
    }
  }
}

/// The device's current timestamp (ticks) through `vkGetCalibratedTimestampsEXT`.
fn calibrated_device_timestamp(ext: &ash::ext::calibrated_timestamps::Device) -> GpuResult<u64> {
  let infos = [vk::CalibratedTimestampInfoEXT::default().time_domain(vk::TimeDomainEXT::DEVICE)];
  let (timestamps, _max_deviation) = unsafe { ext.get_calibrated_timestamps(&infos) }?;
  Ok(timestamps[0])
}

/// Fallback: one command buffer writing one timestamp, submitted and waited for. Returns the
/// ticks and the submit-to-wake-up latency in µs, the bound of how late the pair is.
fn one_shot_timestamp(device: &Device) -> GpuResult<(u64, u64)> {
  let pool_info = vk::QueryPoolCreateInfo::default().query_type(vk::QueryType::TIMESTAMP).query_count(1);
  let pool = unsafe { device.device.create_query_pool(&pool_info, None) }?;
  let t0 = aethervk_oshal_rlib::os::time::get_monotonic_time() as u64;
  let res = device.run_transient_commands(|cmd| {
    unsafe {
      device.device.cmd_reset_query_pool(cmd, pool, 0, 1);
      device.device.synchronization2.cmd_write_timestamp2(cmd, vk::PipelineStageFlags2::ALL_COMMANDS, pool, 0);
    }
    Ok(())
  });
  let t1 = aethervk_oshal_rlib::os::time::get_monotonic_time() as u64;
  let ticks = res.and_then(|mut transient| {
    let mut raw = [0u64; 1];
    let r = unsafe {
      device.device.get_query_pool_results(pool, 0, &mut raw, vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)
    };
    super::DeviceResource::cleanup(&mut transient, &device.device);
    r.map(|_| raw[0]).map_err(Into::into)
  });
  unsafe { device.device.destroy_query_pool(pool, None) };
  ticks.map(|t| (t, t1.saturating_sub(t0)))
}
