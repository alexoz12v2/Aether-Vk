//! Dust trace recorder: positions over time of the same particles, next to the nucleus and the
//! camera, from the running app (`AETHERVK_DUST_TRACE=<file>`, every
//! `AETHERVK_DUST_TRACE_EVERY` frames, default 10). One NDJSON line per traced frame and system.
//!
//! A single capture cannot tell whether dust moves *with* the comet. A trace can: for a particle
//! with a stable id (its ring slot) the offset from the nucleus grows with age under physics
//! (ejection + radiation pressure), and stays constant if the dust were attached to the comet.
//!
//! The particles are the **GPU-evaluated** render clusters actually drawn (copied after the LOD
//! pass, read back `LOD_READBACK_SLOTS` frames later), the metadata is what the frame drew with.
//! Off by default: one cached env check per frame.
use crate::scene::dust::{self, DustRenderCluster, pcg};
use alloc::{string::String, vec::Vec};

/// particles traced per tier and frame
pub const TRACE_SAMPLES: usize = 64;
/// expected selected slots per live range: a slot is traced iff `pcg(slot ^ salt)` falls below
/// `u32::MAX / next_pow2(live) · TRACE_DENSITY`, so the set is the same in every frame while the
/// live count stays within a power of two (steady state: the ring is full)
pub const TRACE_DENSITY: u64 = 96;
const TRACE_SALT: u32 = 0x7A3C_E115;

/// `AETHERVK_DUST_TRACE` output path, `None` = tracing off (cached).
pub fn trace_path() -> Option<&'static str> {
  static PATH: spin::Once<Option<String>> = spin::Once::new();
  PATH
    .call_once(|| {
      aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_TRACE")
        .map(|s| String::from(s.trim()))
        .filter(|s| !s.is_empty())
    })
    .as_deref()
}

/// traced frame period (`AETHERVK_DUST_TRACE_EVERY`, default 10, ≥ 1)
pub fn trace_every() -> u64 {
  static EVERY: spin::Once<u64> = spin::Once::new();
  *EVERY.call_once(|| {
    aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_TRACE_EVERY")
      .and_then(|s| s.trim().parse::<u64>().ok())
      .unwrap_or(10)
      .max(1)
  })
}

/// Whether ring slot `slot` is traced while `live` clusters are drawn (stable across frames).
pub fn is_traced_slot(slot: u32, live: u32) -> bool {
  let threshold = (u32::MAX as u64 / live.max(1).next_power_of_two() as u64) * TRACE_DENSITY;
  (pcg(slot ^ TRACE_SALT) as u64) < threshold
}

/// Traced `(render index, ring slot)` pairs of a tier drawing the live range
/// `[first_slot, first_slot + live)` of a ring of `capacity`: the slots of [`is_traced_slot`] in
/// range order, at most [`TRACE_SAMPLES`] (about 64 of the expected 96 per range).
pub fn sample_slots(first_slot: u32, live: u32, capacity: u32) -> Vec<(u32, u32)> {
  let mask = capacity.max(1) - 1;
  let mut out = Vec::new();
  let live = live.min(capacity);
  for i in 0..live {
    let slot = first_slot.wrapping_add(i) & mask;
    if is_traced_slot(slot, live) {
      out.push((i, slot));
      if out.len() == TRACE_SAMPLES {
        break;
      }
    }
  }
  out
}

/// Pixel position of particle-system local metres `p` through `mvp` (column major, same matrix as
/// `dust.vert`), in a `viewport` (x right, y down as Vulkan). `None` behind the camera.
pub fn project_px(mvp: &[f32; 16], p: [f64; 3], viewport: [u32; 2]) -> Option<[f64; 2]> {
  let m = |r: usize, c: usize| mvp[c * 4 + r] as f64;
  let clip = |r: usize| m(r, 0) * p[0] + m(r, 1) * p[1] + m(r, 2) * p[2] + m(r, 3);
  let w = clip(3);
  if !(w > 0.0) {
    return None;
  }
  let (x, y) = (clip(0) / w, clip(1) / w);
  Some([
    (x * 0.5 + 0.5) * viewport[0] as f64,
    (y * 0.5 + 0.5) * viewport[1] as f64,
  ])
}

/// What one tier drew with in a traced frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceTierMeta {
  pub tier: u32,
  pub anchor_m: [f64; 3],
  pub t_now_s: f64,
  pub rte_position: [f64; 3],
  pub units_per_m: f64,
  pub mvp: [f32; 16],
  /// `(render index, ring slot)` of the traced particles
  pub samples: Vec<(u32, u32)>,
}

/// What one system drew with in a traced frame.
#[derive(Debug, Clone, PartialEq)]
pub struct TraceFrameMeta {
  pub frame: u64,
  pub wall_us: u64,
  pub sim_time_s: f64,
  pub system: u64,
  pub eye_m: [f64; 3],
  /// nucleus (parent of the jet entity) heliocentric position from the scene graph
  pub nucleus_m: Option<[f64; 3]>,
  pub viewport: [u32; 2],
  pub tiers: Vec<TraceTierMeta>,
}

fn v3(a: [f64; 3]) -> String {
  alloc::format!("[{:e},{:e},{:e}]", a[0], a[1], a[2])
}

fn px(p: Option<[f64; 2]>) -> String {
  match p {
    Some(p) => alloc::format!("[{:.3},{:.3}]", p[0], p[1]),
    None => String::from("null"),
  }
}

/// One NDJSON line: `meta` plus the traced render clusters of each tier (`clusters[t][k]` for
/// `meta.tiers[t].samples[k]`). Heliocentric = anchor + local; screen through the tier's mvp.
pub fn record_line(meta: &TraceFrameMeta, clusters: &[Vec<DustRenderCluster>]) -> String {
  use core::fmt::Write;
  let mut s = String::with_capacity(256 + 220 * TRACE_SAMPLES * meta.tiers.len());
  let _ = write!(
    s,
    "{{\"frame\":{},\"wall_us\":{},\"sim_time_s\":{:e},\"system\":{},\"eye_m\":{},\"nucleus_m\":{},\"viewport\":[{},{}],\"tiers\":[",
    meta.frame,
    meta.wall_us,
    meta.sim_time_s,
    meta.system,
    v3(meta.eye_m),
    meta.nucleus_m.map(v3).unwrap_or_else(|| String::from("null")),
    meta.viewport[0],
    meta.viewport[1],
  );
  for (ti, t) in meta.tiers.iter().enumerate() {
    if ti > 0 {
      s.push(',');
    }
    let nucleus_px = meta.nucleus_m.and_then(|n| {
      let local = [
        n[0] - t.anchor_m[0],
        n[1] - t.anchor_m[1],
        n[2] - t.anchor_m[2],
      ];
      project_px(&t.mvp, local, meta.viewport)
    });
    let _ = write!(
      s,
      "{{\"tier\":{},\"anchor_m\":{},\"t_now_s\":{:e},\"rte\":{},\"units_per_m\":{:e},\"nucleus_px\":{},\"particles\":[",
      t.tier,
      v3(t.anchor_m),
      t.t_now_s,
      v3(t.rte_position),
      t.units_per_m,
      px(nucleus_px),
    );
    let tier_clusters = clusters.get(ti).map(|v| v.as_slice()).unwrap_or(&[]);
    let mut first = true;
    for (k, &(_, slot)) in t.samples.iter().enumerate() {
      let Some(c) = tier_clusters.get(k) else {
        break;
      };
      // the render buffer carries its slot: a mismatch means the range moved under the copy.
      // Clusters culled by `dust_propagate.comp` (outside the tier's age band) have no live bit.
      // Off-screen or budget-dropped ones (flux 0 after the LOD) keep their position: traced,
      // with `drawn: false`.
      let y = c.age_id_dbeta_flux[1];
      if dust::render_slot(y) != slot || !dust::render_live(y) {
        continue;
      }
      let local = [
        c.pos_size[0] as f64,
        c.pos_size[1] as f64,
        c.pos_size[2] as f64,
      ];
      let helio = [
        t.anchor_m[0] + local[0],
        t.anchor_m[1] + local[1],
        t.anchor_m[2] + local[2],
      ];
      if !first {
        s.push(',');
      }
      first = false;
      let _ = write!(
        s,
        "{{\"slot\":{},\"drawn\":{},\"age_s\":{:e},\"spread_m\":{:e},\"local_m\":{},\"helio_m\":{},\"px\":{}}}",
        slot,
        c.age_id_dbeta_flux[3] != 0.0,
        c.age_id_dbeta_flux[0],
        c.pos_size[3],
        v3(local),
        v3(helio),
        px(project_px(&t.mvp, local, meta.viewport)),
      );
    }
    s.push_str("]}");
  }
  s.push_str("]}\n");
  s
}

/// Appends `line` to the trace file (logs once on failure).
pub fn append(line: &str) {
  let Some(path) = trace_path() else {
    return;
  };
  if aethervk_oshal_rlib::os::fs::append_parts(path, &[line.as_bytes()]).is_err() {
    static WARNED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);
    if !WARNED.swap(true, core::sync::atomic::Ordering::Relaxed) {
      aethervk_oshal_rlib::log!("[Dust trace] cannot append to {path}");
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  /// Off by default: without the env var nothing is traced (no file, no extra copies).
  #[test]
  fn tracing_is_off_without_the_env_var() {
    extern crate std;
    if std::env::var_os("AETHERVK_DUST_TRACE").is_none() {
      assert!(trace_path().is_none());
    }
    assert!(trace_every() >= 1);
  }

  #[test]
  fn sampling_is_stable_and_bounded() {
    let cap = 32_768;
    let a = sample_slots(100, 20_000, cap);
    assert!(
      a.len() > TRACE_SAMPLES / 2 && a.len() <= TRACE_SAMPLES,
      "{}",
      a.len()
    );
    // a sparse ring still traces particles
    assert!(sample_slots(0, 4096, 262_144).len() > TRACE_SAMPLES / 2);
    // the range advanced by 500 slots: the traced slots still alive are the same ones
    let b = sample_slots(600, 20_000, cap);
    let sa: Vec<u32> = a.iter().map(|x| x.1).filter(|&s| s >= 600).collect();
    let sb: Vec<u32> = b.iter().map(|x| x.1).collect();
    for s in &sa {
      assert!(
        sb.contains(s) || sb.len() == TRACE_SAMPLES,
        "slot {s} dropped"
      );
    }
    // index ↔ slot consistency (render buffer index = offset in the live range)
    for &(i, s) in &b {
      assert_eq!((600 + i) & (cap - 1), s);
    }
    // wraps the ring
    for &(i, s) in &sample_slots(cap - 10, 5000, cap) {
      assert_eq!((cap - 10 + i) & (cap - 1), s);
    }
  }

  #[test]
  fn projection_matches_the_shader_convention() {
    // ortho-like: x, y scaled by 1/1000, w = 1
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1e-3;
    mvp[5] = 1e-3;
    mvp[10] = 1e-9;
    mvp[15] = 1.0;
    let p = project_px(&mvp, [500.0, -250.0, 0.0], [1000, 500]).unwrap();
    assert!(
      (p[0] - 750.0).abs() < 1e-3 && (p[1] - 187.5).abs() < 1e-3,
      "{p:?}"
    );
    let mut behind = mvp;
    behind[15] = -1.0;
    assert!(project_px(&behind, [0.0; 3], [10, 10]).is_none());
  }

  #[test]
  fn record_line_is_valid_ndjson_with_helio_and_screen() {
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1e-6;
    mvp[5] = 1e-6;
    mvp[15] = 1.0;
    let meta = TraceFrameMeta {
      frame: 7,
      wall_us: 123,
      sim_time_s: 86400.0,
      system: 42,
      eye_m: [1.0, 2.0, 3.0],
      nucleus_m: Some([1.0e11, 0.0, 0.0]),
      viewport: [800, 600],
      tiers: alloc::vec![TraceTierMeta {
        tier: 0,
        anchor_m: [1.0e11, 0.0, 0.0],
        t_now_s: 86400.0,
        rte_position: [0.0; 3],
        units_per_m: 1e-3,
        mvp,
        samples: alloc::vec![(3, 77), (5, 78), (6, 79), (7, 80)],
      }],
    };
    let ok = DustRenderCluster {
      pos_size: [1000.0, 0.0, 0.0, 10.0],
      age_id_dbeta_flux: [3600.0, f32::from_bits(dust::render_word(77, 6)), 0.0, 1.0],
    };
    // off screen (flux 0 after the LOD): kept, not drawn
    let hidden = DustRenderCluster {
      pos_size: [5.0, 0.0, 0.0, 10.0],
      age_id_dbeta_flux: [3600.0, f32::from_bits(dust::render_word(78, 6)), 0.0, 0.0],
    };
    // wrong slot: skipped
    let stale = DustRenderCluster {
      pos_size: [0.0; 4],
      age_id_dbeta_flux: [0.0, f32::from_bits(999), 0.0, 1.0],
    };
    // culled by the propagate (outside the age band): the slot only, skipped
    let culled = DustRenderCluster {
      pos_size: [0.0; 4],
      age_id_dbeta_flux: [0.0, f32::from_bits(79), 0.0, 0.0],
    };
    let line = record_line(&meta, &[alloc::vec![ok, hidden, culled, stale]]);
    assert!(line.ends_with("]}\n"));
    assert_eq!(line.matches("\"slot\":").count(), 2);
    assert!(line.contains("\"slot\":78,\"drawn\":false"), "{line}");
    assert!(line.contains("\"slot\":77,\"drawn\":true"), "{line}");
    assert!(line.contains("\"slot\":77"));
    assert!(
      line.contains("\"helio_m\":[1.00000001e11,0e0,0e0]"),
      "{line}"
    );
    assert!(line.contains("\"nucleus_px\":[400.000,300.000]"), "{line}");
    // balanced braces / brackets
    let open = line.matches('{').count() + line.matches('[').count();
    let close = line.matches('}').count() + line.matches(']').count();
    assert_eq!(open, close);
  }
}
