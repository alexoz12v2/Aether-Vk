//! GPU parity of the cadence (track 5, I1): `dust_classify.comp`'s rate against
//! `cadence::cadence_rate`, and the GPU state machine (owner bytes, lists, cursors, steps, swaps,
//! aborts, the propagate set) against `cadence::DustCadenceState` frame by frame. Needs a Vulkan
//! device and the compiled shaders. The setup helpers are minimal copies of `test_specks.rs`'s.
use super::*;
use crate::gpu_backends::vulkan::device::test_utils::{setup_assets_dir, setup_render_frontend_for_tests};
use crate::scene::dust::{
  self, AU_M, DustBatch, DustFrame, SUN_MU_M3_S2, SizeDistribution, V3, cadence as cad, kepler,
};

fn comet_state() -> (V3, V3) {
  let r = [-1.924 * AU_M, -5.164 * AU_M, -0.204 * AU_M];
  let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
  let vc = (SUN_MU_M3_S2 / rn).sqrt() * 0.9;
  let t = [5.164f64, -1.924, 0.3];
  let tn = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
  (r, [t[0] * vc / tn, t[1] * vc / tn, t[2] * vc / tn])
}

const T_START: f64 = 4.0e8;

fn test_batch(count: u32, dur: f64, first_index: u32, capacity: u32, seed: u32) -> DustBatch {
  let (rc, vc) = comet_state();
  let dist = SizeDistribution::from_diameter_um(100.0);
  let (size_params, vel_params, mass_params) =
    dust::batch_params(&dist, 100.0, 0.533, 0.0213, 2.0, 0.5, 1.0e6, 0.37);
  let mut b: DustBatch = bytemuck::Zeroable::zeroed();
  b.set_comet(rc, vc, T_START, dur);
  b.rot_start = [0.0, 0.0, 0.0, 1.0];
  b.spin = [0.0, 0.0, 1.0, 0.0];
  b.lit = [0.0, 0.0, 0.0, dust::LIT_MODE_ALWAYS];
  b.jet_dir_aperture = [0.35, 0.93, 0.04, 0.6];
  b.size_params = size_params;
  b.vel_params = vel_params;
  b.mass_params = mass_params;
  b.first_index = first_index;
  b.count = count;
  b.ring_mask = capacity - 1;
  b.seed = seed;
  b
}

fn frame_at(days: f64) -> DustFrame {
  let t_now = T_START + days * 86400.0;
  let (rc0, vc0) = comet_state();
  let (rc, _) = kepler::propagate_f64(rc0, vc0, SUN_MU_M3_S2, t_now - T_START);
  DustFrame::new(rc, t_now, [0.0, 0.0, 0.0, 1.0], 1.0e9)
}

/// A perspective camera at `eye` looking along −z, vertical field 60°: the absolute mvp (local
/// metres → clip) and the eye-relative one (as `test_specks::camera`).
fn camera(eye: [f32; 3], w: u32, h: u32) -> ([f32; 16], [f32; 16]) {
  let tan_h = 30.0f32.to_radians().tan();
  let (a, b) = (h as f32 / (w as f32 * tan_h), 1.0 / tan_h);
  let mut rel = [0.0f32; 16];
  rel[0] = a;
  rel[5] = b;
  rel[10] = -1e-3;
  rel[11] = -1.0;
  let mut abs = rel;
  abs[12] = -a * eye[0];
  abs[13] = -b * eye[1];
  abs[14] = 1e-3 * eye[2];
  abs[15] = eye[2];
  (abs, rel)
}

/// a yaw of `deg` about the camera's y axis as a column-major 4×4 view rotation (f64)
fn yaw4(deg: f64) -> [f64; 16] {
  let (s, c) = deg.to_radians().sin_cos();
  [c, 0.0, -s, 0.0, 0.0, 1.0, 0.0, 0.0, s, 0.0, c, 0.0, 0.0, 0.0, 0.0, 1.0]
}

fn with_device(id: u64, f: impl FnOnce(&Device, u32, vk::Buffer, vk::Buffer)) {
  setup_assets_dir();
  assert!(
    !crate::gpu_backends::vulkan::physics::is_cpu_particles_mode(),
    "GPU parity tests need GPU particle mode"
  );
  let (_pool, render_frontend, handle, _) = setup_render_frontend_for_tests(false);
  render_frontend
    .with_device(handle, |dyn_device| {
      let device: &Device = dyn_device.as_any().downcast_ref().unwrap();
      device.create_particle_system(id)?;
      let (capacity, render, moments) = {
        let res = device.res.read();
        let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
        (sys.capacity, sys.render.as_ref().unwrap().buffer, sys.moments.as_ref().unwrap().buffer)
      };
      f(device, capacity, render, moments);
      unsafe { device.device.device_wait_idle() }.unwrap();
      device.discard_particle_system(id, 0, 0)?;
      GpuResult::Ok(())
    })
    .unwrap();
}

fn emit(device: &Device, id: u64, batch: &DustBatch, seq: u64) {
  let mut res = device
    .run_transient_compute_commands(|cmd| device.cmd_dust_emit(cmd, id, 0, batch, seq))
    .unwrap();
  res.cleanup(&device.device);
}

fn records<T: bytemuck::Pod>(device: &Device, buf: vk::Buffer, first: u32, live: u32, cap: u32) -> alloc::vec::Vec<T> {
  crate::gpu_backends::vulkan::device::test_dust::read_tier_records(device, buf, 0, first, live, cap - 1)
}

/// the cloud's mean box and the eye the parity cases look from: `back` box depths above it on +z
fn eye_above(moments: &[dust::DustMoments], back: f32) -> ([f32; 3], f32) {
  // the box of every cluster's mean and polyline, each widened by its 3σ_max (the clusters'
  // dispersion dwarfs their means' spread: an eye above the means alone sat inside every cloud)
  let (mut lo, mut hi) = ([f32::MAX; 3], [f32::MIN; 3]);
  for m in moments.iter().filter(|m| m.flux() > 0.0) {
    let r = 3.0 * cad::sym3_eigen_range(m.cov_v()).1.max(0.0).sqrt();
    for c in core::iter::once(m.mean()).chain((0..dust::SIZE_EDGES as usize).map(|j| m.edge(j))) {
      for i in 0..3 {
        lo[i] = lo[i].min(c[i] - r);
        hi[i] = hi[i].max(c[i] + r);
      }
    }
  }
  let ext = (0..3).map(|i| hi[i] - lo[i]).fold(1.0f32, f32::max);
  ([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), hi[2] + back * ext], ext)
}

/// `dust_classify.comp`'s rate equals `cadence_rate` on the GPU's moments: per live slot the class
/// (outside the boundary band |δ_k · 2N_k − ε| ≤ 10⁻⁴ ε, where the mirrors' sqrt / division ulps
/// may decide), δ_k and g within 10⁻⁴ relative (printed: p50, max). An orthographic and a
/// perspective view (three cloud widths above the cloud, the eye stepping laterally), each case's
/// inputs (Δt, Δe, a_w) scaled by one λ so the CPU's median δ_16 sits at the class-16 bound ε / 32:
/// every term of δ is linear in them (the projected step to first order), so the clusters' own
/// spread (depth, extent, velocity) puts them on both sides of it — at least two classes in each
/// view, asserted (the unscaled scene put every cluster in class 16 in both). A third case is the
/// perspective one under the by-count fallback's shape bound (`RateInputs::shape_frac`, params word
/// `CADP_SHAPE`), θ putting the median age at class 16's bound `2 · 16 · Δt ≤ θ · age`: the same
/// per-slot comparison, the shape bound changes classes there (counted against θ = 0 on the CPU,
/// asserted > 0) and nowhere else (0 in the first two cases).
#[test]
fn gpu_dust_cadence_classify_matches_reference() {
  let id = 9031u64;
  with_device(id, |device, capacity, render_buf, moments_buf| {
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, 0);
    let first = batch.first_index & batch.ring_mask;
    let frame = frame_at(5.0);
    let mut addrs = (0u64, 0u64);
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        addrs = device.cmd_dust_propagate(cmd, id, 0, capacity, first, batch.count, &frame)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let render: alloc::vec::Vec<dust::DustRenderCluster> = records(device, render_buf, first, batch.count, capacity);
    let moments: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, batch.count, capacity);
    let (w, h) = (320u32, 200u32);
    let classes = [8u32, 16, 64];
    let (eye, ext) = eye_above(&moments, 3.0);
    let f_px = 0.5 * h as f32 / 30.0f32.to_radians().tan();
    let d_med = {
      let mut d: alloc::vec::Vec<f32> = moments.iter().filter(|m| m.flux() > 0.0).map(|m| eye[2] - m.mean()[2]).collect();
      d.sort_by(f32::total_cmp);
      d[d.len() / 2]
    };
    let v_med = {
      let mut v: alloc::vec::Vec<f32> = moments
        .iter()
        .filter(|m| m.flux() > 0.0)
        .map(|m| {
          let u = m.helio_velocity();
          (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt()
        })
        .collect();
      v.sort_by(f32::total_cmp);
      v[v.len() / 2]
    };
    // the median age (floored as the rate floors it): the shape case's θ puts it at class 16's bound
    let age_med = {
      let mut a: alloc::vec::Vec<f32> =
        moments.iter().filter(|m| m.flux() > 0.0).map(|m| m.age().max(cad::DUST_CADENCE_AGE_MIN_S)).collect();
      a.sort_by(f32::total_cmp);
      a[a.len() / 2]
    };
    const SHAPE_CASE: &str = "perspective, fallback shape";
    // the ortho view: ±ext around the cloud's centre, the eye far above
    let half = ext;
    let mut ortho = [0.0f32; 16];
    ortho[0] = 1.0 / half;
    ortho[5] = 1.0 / half;
    ortho[10] = 1e-12;
    ortho[12] = -eye[0] / half;
    ortho[13] = -eye[1] / half;
    ortho[15] = 1.0;
    let px_per_m = w as f32 / (2.0 * half);
    let cases: [(&str, [f32; 16], [f32; 3], cad::RateInputs); 3] = [
      (
        "ortho",
        ortho,
        [eye[0], eye[1], eye[2] + 1e9],
        cad::RateInputs {
          // the median cluster at ~1/40 px per frame
          dt_s: 1.0 / (40.0 * v_med.max(1e-3) * px_per_m),
          d_eye_m: [0.0; 3],
          a_w_m: [0.0, 0.0, 0.0],
          eps_px: 1.0,
          frame_px: w.max(h) as f32,
          shape_frac: 0.0,
        },
      ),
      (
        "perspective",
        camera(eye, w, h).0,
        eye,
        cad::RateInputs {
          dt_s: 0.02,
          d_eye_m: [d_med / (40.0 * f_px), 0.0, 0.0],
          a_w_m: [0.0, 0.1 * d_med / (40.0 * f_px), 0.0],
          eps_px: 1.0,
          frame_px: w.max(h) as f32,
          shape_frac: 0.0,
        },
      ),
      (
        // the perspective case under the fallback's shape bound (θ set after the scaling)
        SHAPE_CASE,
        camera(eye, w, h).0,
        eye,
        cad::RateInputs {
          dt_s: 0.02,
          d_eye_m: [d_med / (40.0 * f_px), 0.0, 0.0],
          a_w_m: [0.0, 0.1 * d_med / (40.0 * f_px), 0.0],
          eps_px: 1.0,
          frame_px: w.max(h) as f32,
          shape_frac: 0.0,
        },
      ),
    ];
    for (tag, mvp, e, inp0) in cases {
      // λ: the median δ_16 (the CPU's rate) at the class-16 bound ε / (2 · 16)
      let lam = {
        let mut d: alloc::vec::Vec<f32> = (0..moments.len())
          .filter(|&o| moments[o].flux() > 0.0 && dust::render_live(render[o].age_id_dbeta_flux[1]))
          .map(|o| {
            let m = &moments[o];
            let pred = dust::streak_pred(&render, o).map(|q| &moments[q]).filter(|q| q.propagate_stamp() == m.propagate_stamp());
            cad::cadence_rate(m, pred, &mvp, e, w, h, &classes, &inp0).delta_px[1]
          })
          .filter(|x| x.is_finite() && *x > 0.0)
          .collect();
        d.sort_by(f32::total_cmp);
        if d.is_empty() { 1.0 } else { inp0.eps_px / (2.0 * classes[1] as f32 * d[d.len() / 2]) }
      };
      let inp = cad::RateInputs {
        dt_s: inp0.dt_s * lam,
        d_eye_m: inp0.d_eye_m.map(|x| x * lam),
        a_w_m: inp0.a_w_m.map(|x| x * lam),
        eps_px: inp0.eps_px,
        frame_px: inp0.frame_px,
        // θ with the median age at class 16's shape bound 2 · 16 · Δt = θ · age
        shape_frac: if tag == SHAPE_CASE { 2.0 * classes[1] as f32 * (inp0.dt_s * lam) / age_med } else { 0.0 },
      };
      std::println!("[cadence gpu] classify {tag}: inputs scaled by λ = {lam:.4e} (Δt {:.4e} s)", inp.dt_s);
      let gpu = device
        .dust_cadence_classify_debug(addrs.0, addrs.1, first, capacity, batch.count, mvp, [e[0], e[1], e[2], 0.0], w, h, &classes, &inp, 0.0)
        .unwrap();
      let (mut compared, mut boundary, mut bad) = (0usize, 0usize, 0usize);
      // the slots whose class the shape bound changed (the CPU at θ against θ = 0)
      let mut shaped = 0usize;
      let mut hist = [0usize; 4];
      let (mut rel_d, mut rel_g): (alloc::vec::Vec<f64>, alloc::vec::Vec<f64>) = (alloc::vec::Vec::new(), alloc::vec::Vec::new());
      let (mut worst_abs, mut over) = (0.0f64, 0usize);
      for (o, m) in moments.iter().enumerate() {
        let g = gpu[o];
        let ok = m.flux() > 0.0 && dust::render_live(render[o].age_id_dbeta_flux[1]);
        assert_eq!(g[5] & 1 != 0, ok, "{tag}: offset {o}: live and lit (gpu flags {:#x})", g[5]);
        assert_eq!(g[6], (first + o as u32) & (capacity - 1), "{tag}: offset {o}: the slot");
        if !ok {
          continue;
        }
        let pred = dust::streak_pred(&render, o).map(|q| &moments[q]).filter(|q| q.propagate_stamp() == m.propagate_stamp());
        let r = cad::cadence_rate(m, pred, &mvp, e, w, h, &classes, &inp);
        let r0 = cad::cadence_rate(m, pred, &mvp, e, w, h, &classes, &cad::RateInputs { shape_frac: 0.0, ..inp });
        assert!(r.class <= r0.class, "{tag}: offset {o}: the shape bound raised a class ({} over {})", r.class, r0.class);
        shaped += (r.class != r0.class) as usize;
        compared += 1;
        hist[r.class as usize] += 1;
        // the boundary: the GPU's and the CPU's δ_k · 2N_k on opposite sides of ε for some k (the
        // δ values themselves are held to the bound below, so only a straddling slot may differ)
        let near = (0..3).any(|k| {
          let (xc, xg) = (r.delta_px[k] * 2.0 * classes[k] as f32, f32::from_bits(g[1 + k]) * 2.0 * classes[k] as f32);
          xc.is_finite() && xg.is_finite() && (xc - inp.eps_px) * (xg - inp.eps_px) <= 0.0
        });
        boundary += near as usize;
        if g[0] != r.class as u32 {
          if !near {
            if bad < 5 {
              std::println!(
                "[cadence gpu] classify {tag}: offset {o}: class gpu {} cpu {} (δ gpu {:?} cpu {:?}, g gpu {} cpu {})",
                g[0],
                r.class,
                [f32::from_bits(g[1]), f32::from_bits(g[2]), f32::from_bits(g[3])],
                r.delta_px,
                f32::from_bits(g[4]),
                r.g_px_per_m
              );
            }
            bad += 1;
          }
        }
        for k in 0..3 {
          let (a, b) = (f32::from_bits(g[1 + k]) as f64, r.delta_px[k] as f64);
          if a.is_finite() && b.is_finite() && b.abs() > 0.0 {
            rel_d.push((a / b - 1.0).abs());
            // d1 differences two projected pixel positions: the mirrors' f32 projections (the
            // GPU's fused matrix products) agree to ~1 ulp of the frame's size, which bounds |Δδ|
            let ulp = (w.max(h) as f32).to_bits();
            let ulp = (f32::from_bits(ulp + 1) - f32::from_bits(ulp)) as f64;
            let excess = (a - b).abs() - (1e-4 * b.abs() + 4.0 * ulp);
            worst_abs = worst_abs.max((a - b).abs());
            if excess > 0.0 {
              if over < 3 {
                std::println!("[cadence gpu] classify {tag}: offset {o}: δ_{k} gpu {a:.6e} cpu {b:.6e} beyond 1e-4·δ + 4 ulp({}) px", w.max(h));
              }
              over += 1;
            }
          } else {
            assert_eq!(a.is_finite(), b.is_finite(), "{tag}: offset {o}: δ_{k} finite gpu {a} cpu {b}");
          }
        }
        let (a, b) = (f32::from_bits(g[4]) as f64, r.g_px_per_m as f64);
        if a.is_finite() && b.is_finite() && b > 0.0 {
          rel_g.push((a / b - 1.0).abs());
        }
      }
      rel_d.sort_by(f64::total_cmp);
      rel_g.sort_by(f64::total_cmp);
      let q = |v: &[f64], f: f64| if v.is_empty() { 0.0 } else { v[((v.len() - 1) as f64 * f) as usize] };
      std::println!(
        "[cadence gpu] classify {tag}: {compared} slots, classes fresh/8/16/64 {hist:?}, boundary {boundary}, mismatches outside it {bad}; δ rel p50 {:.2e} max {:.2e}, g rel p50 {:.2e} max {:.2e}",
        q(&rel_d, 0.5),
        q(&rel_d, 1.0),
        q(&rel_g, 0.5),
        q(&rel_g, 1.0)
      );
      assert!(compared >= 500, "{tag}: {compared} live slots");
      assert_eq!(bad, 0, "{tag}: {bad} classes differ outside the boundary band");
      // the bound: |Δδ| ≤ 1e-4 δ + 4 ulp of the frame size (the projections' resolution; the
      // relative error alone is unbounded where δ is a small difference of two pixel positions)
      std::println!("[cadence gpu] classify {tag}: δ |Δ| max {worst_abs:.3e} px, beyond the bound {over}");
      assert_eq!(over, 0, "{tag}: {over} δ values beyond 1e-4·δ + 4 ulp px");
      assert!(q(&rel_g, 1.0) <= 1e-4, "{tag}: g relative error {:.2e}", q(&rel_g, 1.0));
      assert!(hist.iter().filter(|&&n| n > 0).count() >= 2, "{tag}: one class only {hist:?}: the scaled scene spreads nothing");
      std::println!("[cadence gpu] classify {tag}: θ {:.4e}, classes changed by the shape bound {shaped}", inp.shape_frac);
      if tag == SHAPE_CASE {
        assert!(shaped > 0, "{tag}: the shape bound changed no class (θ {})", inp.shape_frac);
      } else {
        assert_eq!(shaped, 0, "{tag}: θ = 0 changed a class");
      }
    }
  });
}

/// The GPU state machine equals the CPU's frame by frame (the perspective scene of the classify
/// case at the full configuration (margin 20 %: 64 px, the budget m − G = 32 px), the eye stepping
/// ~1/40 px of parallax per frame at the median depth; phases from the CPU's state: the constant
/// step until a class image is shown (the class-16 cycle: classified, built, shown 2N later); then
/// a yaw of 1° per frame over 3 frames (≈ 3 px of corner motion: 2N · U = 96 px over the budget,
/// class 16's lists abort — the budget's and the prediction's aborts —, the shown image reprojected
/// rotated); once a list is non-empty again (class 8, admitted 16 frames after the yaw) one frame
/// of a 16× eye step — the rate aborts —, then 4 frames; at most 160 frames; time advancing
/// 0.02 s per frame, a batch emitted at frames 6, 12 and 18, the tail retiring 32 slots at frame
/// 9; nucleus and anchor fixed at the origin, said so). Asserted
/// exactly each frame: the mode, reset and swaps; every owner byte; every list's entries; the
/// steps and the advanced cursors; the abort decisions and reasons (excess within 10⁻⁵); the
/// coverage drops; the exists flags; the classify histogram, the shell count, the chunk entries
/// drawn; the propagate set. g_max within 10⁻⁵ relative. Preconditions: an image shown, a budget
/// or prediction abort after it, at least one rate abort and two classes over the run. Drawing is the
/// CPU's only (state only, I1).
#[test]
fn gpu_dust_cadence_state_matches_reference() {
  let id = 9032u64;
  with_device(id, |device, capacity, render_buf, moments_buf| {
    let mut seq = 0u64;
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, seq);
    seq += 1;
    let mask = capacity - 1;
    let (mut first, mut live) = (batch.first_index & mask, batch.count);
    let mut emitted = 1u64;
    let mut rewritten_from: u32;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 31);
    let (w, h) = (320u32, 200u32);
    let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
    let ttl = 30.0 * 86400.0f32;
    let cfg = cad::CadenceConfig::full();
    let classes = cfg.classes.clone();
    // place the eye from one propagate
    let frame0 = frame_at(5.0);
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_propagate(cmd, id, 0, capacity, first, live, &frame0)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let m0: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, live, capacity);
    let (eye0, _) = eye_above(&m0, 3.0);
    let f_px = 0.5 * h as f32 / 30.0f32.to_radians().tan();
    let d_med = {
      let mut d: alloc::vec::Vec<f32> = m0.iter().filter(|m| m.flux() > 0.0).map(|m| eye0[2] - m.mean()[2]).collect();
      d.sort_by(f32::total_cmp);
      d[d.len() / 2]
    };
    let step = d_med / (40.0 * f_px);
    let rel = camera(eye0, w, h).1;
    let layout = dust::PyramidLayout::new(w, h);
    let mut st = cad::DustCadenceState::new(cfg.clone());
    st.threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    let mut eye = eye0;
    let (mut aborts_seen, mut class_seen) = (0usize, [0u32; 4]);
    let mut reasons_seen = [0usize; 4];
    // the phases (frames, set from the CPU's state as the run goes)
    const FRAMES: u32 = 160;
    const YAW_FRAMES: u32 = 3;
    const YAW_DEG: f64 = 1.0;
    let (mut shown_at, mut yaw_from, mut motion_abort_at, mut jump_at, mut end_at) =
      (None::<u32>, None::<u32>, None::<u32>, None::<u32>, None::<u32>);
    let mut yaw_deg = 0.0f64;
    // the view exists from the first frame_begin; the cadence on from then
    let mut enabled = false;
    for f in 0..FRAMES {
      // what the render thread's `start_frame` does each frame (the idiom of test_commands.rs):
      // recycle the completed submissions' command buffers, or 160 frames of downloads exhaust the
      // 24 pending command pools
      {
        let items = {
          let res = device.res.read();
          let t = res.timeline_manager.refresh_cached_value().unwrap();
          res.discard_pool.pop_ready_items(t)
        };
        crate::gpu_backends::vulkan::device::DiscardPool::destroy_items_lock_free(&device.device, items);
      }
      // the ring: emissions at the head, a retirement at the tail
      if f == 6 || f == 12 || f == 18 {
        let b = test_batch(64, 600.0, first + live, capacity, 0xBEEF + f);
        emit(device, id, &b, seq);
        seq += 1;
        rewritten_from = live;
        live += 64;
        emitted += 1;
      } else {
        rewritten_from = live;
      }
      if f == 9 {
        first = (first + 32) & mask;
        live -= 32;
        rewritten_from = live;
      }
      if f > 0 {
        // one frame of a 16× step: the rate abort of the lists classified at the step
        let dx = if jump_at == Some(f) { 16.0 * step } else { step };
        eye[0] += dx;
      }
      if yaw_from.is_some_and(|y0| f >= y0 && f < y0 + YAW_FRAMES) {
        yaw_deg += YAW_DEG;
      }
      let t_s = T_START + 5.0 * 86400.0 + f as f64 * 0.02;
      let frame = frame_at(5.0 + f as f64 * 0.02 / 86400.0);
      let pre_render: alloc::vec::Vec<dust::DustRenderCluster> = records(device, render_buf, first, live, capacity);
      let mut view = cad::CadenceView {
        view_rot: yaw4(yaw_deg),
        proj: [0.0; 16],
        eye_m: [eye[0] as f64, eye[1] as f64, eye[2] as f64],
        units_per_m: 1.0,
        width: w,
        height: h,
      };
      for i in 0..16 {
        view.proj[i] = rel[i] as f64;
      }
      // both mirrors draw and classify with the view's own mvp (the rotation included)
      let mvp = view.mvp([0.0; 3]);
      let tier = CadenceTierInput {
        key: 0,
        system: id,
        ring_base: 0,
        first_slot: first,
        capacity,
        live_count: live,
        emitted,
        rewritten_from,
        anchor_m: [0.0; 3],
      };
      let mut plan = CadenceGpuPlan::default();
      let mut fm = cad::FrameMeta {
        view,
        t_now_s: t_s,
        speed: 1.0,
        nucleus_m: [0.0; 3],
        flags: 0,
        fingerprint: 0,
        reused: false,
        unit_now: 1.0,
      };
      let mut inv_unit = 1.0f32;
      if !enabled {
        // the view (frame_begin) first, then the cadence
        let mut cmds = device
          .run_transient_commands(|cmd| {
            device.cmd_dust_frame_begin(cmd, pe, [w, h], 0, flow)?;
            device.cmd_dust_frame_end(cmd, pe)?;
            Ok(())
          })
          .unwrap();
        cmds.cleanup(&device.device);
        device.dust_cadence_enable(pe, Some(cfg.clone())).unwrap();
        // the view is recreated with the class images at the next begin: one more frame, then its
        // gates off (the images and the CPU's are compared draw for draw)
        let mut cmds = device
          .run_transient_commands(|cmd| {
            device.cmd_dust_frame_begin(cmd, pe, [w, h], 0, flow)?;
            device.cmd_dust_frame_end(cmd, pe)?;
            Ok(())
          })
          .unwrap();
        cmds.cleanup(&device.device);
        device.dust_set_frame_gates(pe, false);
        enabled = true;
      }
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let v = device.cmd_dust_frame_begin(cmd, pe, [w, h], 0, flow)?;
          fm.unit_now = v.unit;
          inv_unit = 1.0 / v.unit;
          device.cmd_dust_system_begin(id, 0, flow)?;
          plan = device.cmd_dust_cadence_begin(cmd, pe, &fm, &[tier], 0.0)?;
          let (render, moments) = device.cmd_dust_cadence_propagate(cmd, pe, id, 0, capacity, first, live, &frame, emitted)?;
          device.cmd_dust_pre_splat_barrier(cmd);
          device.cmd_dust_cadence_classify(cmd, pe, 0, render, moments, mvp, [eye[0], eye[1], eye[2], ttl])?;
          device.cmd_dust_cadence_chunks(cmd, pe, 0, render, moments)?;
          // the build draws at the frozen views (the CPU's F3), then the fresh splat (owner filter)
          let base = dust::DustSplatPushConstants {
            moments: 0,
            render: 0,
            pyramid: 0,
            live_count: live,
            flags: 0,
            exposure: 1.0,
            inv_unit,
            color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
            albedo: 0.04,
            mvp,
            eye_local: [eye[0], eye[1], eye[2], ttl],
          };
          let draws: alloc::vec::Vec<CadenceBuildDraw> = plan
            .builds
            .iter()
            .map(|b| {
              let e = b.view.eye_local([0.0; 3]);
              let mut pcb = base;
              pcb.mvp = b.view.mvp_image([0.0; 3], &cfg.image_geom(w, h));
              pcb.eye_local = [e[0], e[1], e[2], ttl];
              pcb.inv_unit = if b.unit > 0.0 { 1.0 / b.unit } else { 0.0 };
              CadenceBuildDraw { k: b.k, p: b.p, key: 0, system: id, ring_base: 0, first_slot: first, capacity, render, moments, pc: pcb }
            })
            .collect();
          device.cmd_dust_cadence_builds(cmd, pe, &draws)?;
          device.cmd_dust_cadence_reproject(cmd, pe)?;
          device.cmd_dust_splat(cmd, pe, id, 0, first, capacity, render, moments, base)?;
          device.cmd_dust_cadence_end(cmd, pe)?;
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      let gs = device.download_dust_cadence(pe).unwrap();
      let mut moments: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, live, capacity);
      let render: alloc::vec::Vec<dust::DustRenderCluster> = records(device, render_buf, first, live, capacity);
      // the CPU mirror on the same inputs and the GPU's moments
      let cplan = st.begin(
        &fm,
        &[cad::TierMeta {
          key: 0,
          first_slot: first,
          capacity,
          live_count: live,
          emitted,
          rewritten_from,
          anchor_m: [0.0; 3],
          render: &pre_render,
        }],
        &layout,
      );
      let pc = dust::DustSplatPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: live,
        flags: 0,
        exposure: 1.0,
        inv_unit,
        color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
        albedo: 0.04,
        mvp,
        eye_local: [eye[0], eye[1], eye[2], ttl],
      };
      let mut frame_cpu = alloc::vec![0u32; layout.total_words as usize];
      frame_cpu[..dust::PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
      let gpu_header = device.download_dust_pyramid(pe).unwrap();
      for word in [
        dust::PYR_FLAGS,
        dust::PYR_T_HI,
        dust::PYR_T_LO,
        dust::PYR_TRACER_COUNTS,
        dust::PYR_LOD,
        dust::PYR_SPLIT_DEPTH,
        dust::specks::PYR_SPECKS_DMAX,
        dust::specks::PYR_SPECKS_BUDGET,
      ] {
        frame_cpu[word as usize] = gpu_header[word as usize];
      }
      let rep = st.finish(
        &mut [cad::TierDraw { moments: &mut moments, render: &render, pc, flow, specks_pc: None, stamp: device.dust_propagate_generation(id, 0).unwrap_or(0) }],
        &layout,
        &mut frame_cpu,
      );
      let tag = alloc::format!("frame {f}");
      std::println!(
        "[cadence gpu] {tag}: mode {:?} reset {:?} swaps {:?} aborts {:?} classified {:?} drawn {} d3 {}",
        cplan.mode,
        cplan.reset,
        cplan.swaps,
        cplan.aborts.iter().map(|a| (a.k, a.p, a.list)).collect::<alloc::vec::Vec<_>>(),
        rep.classified,
        rep.drawn_build.len(),
        rep.d3_rejected
      );
      assert_eq!(plan.mode, cplan.mode, "{tag}: mode");
      assert_eq!(plan.reset, cplan.reset, "{tag}: reset");
      assert_eq!(plan.swaps, cplan.swaps, "{tag}: swaps");
        let tc = &st.tiers[&0];
      let gt = &gs.tiers[0];
      // the owner bytes
      let diff: alloc::vec::Vec<usize> = (0..capacity as usize).filter(|&s| gt.owner[s] != tc.owner[s]).collect();
      for &s in diff.iter().take(5) {
        std::println!("[cadence gpu] {tag}: slot {s}: owner gpu {:#04x} cpu {:#04x}", gt.owner[s], tc.owner[s]);
      }
      assert!(diff.is_empty(), "{tag}: {} owner bytes differ", diff.len());
      // the lists, steps, cursors, g_max
      let np = cfg.phases as usize;
      for k in 0..classes.len() {
        for p in 0..np {
          for s in 0..3 {
            assert_eq!(gt.lists[k][p][s], tc.lists[k][p][s], "{tag}: list ({k}, {p}, {s})");
          }
          let tp = cads_tp(gt.t, k, p);
          let (len, cursor, stp) = (gs.state[tp + 1], gs.state[tp + 3], gs.state[tp + 4]);
          assert_eq!(stp, tc.step[k][p], "{tag}: step ({k}, {p})");
          let advanced = if cplan.mode == Some(cad::CadenceMode::Run) { (cursor + stp).min(len) } else { cursor };
          assert_eq!(advanced, tc.cursor[k][p], "{tag}: cursor ({k}, {p})");
          for (list, word) in [(1usize, 5usize), (2, 6)] {
            let (a, b) = (f32::from_bits(gs.state[tp + word]), tc.meta[k][p][list].g_max);
            assert!((a - b).abs() <= 1e-5 * a.abs().max(b.abs()), "{tag}: g_max ({k}, {p}, {list}) gpu {a} cpu {b}");
          }
          // the aborts of this frame
          let flags = gs.state[cads_pair(k, p) + 2];
          for list in [1u8, 2] {
            let c = cplan.aborts.iter().find(|a| a.k == k && a.p == p && a.list == list);
            assert_eq!(flags & list as u32 != 0, c.is_some(), "{tag}: abort ({k}, {p}, {list})");
            if let Some(a) = c {
              let e = f32::from_bits(gs.state[cads_pair(k, p) + 2 + list as usize]);
              assert!((e - a.excess_px).abs() <= 1e-5 * e.abs().max(a.excess_px.abs()), "{tag}: excess gpu {e} cpu {}", a.excess_px);
              assert_eq!(gs.state[cads_pair(k, p) + 8 + list as usize], a.reason as u32, "{tag}: abort reason ({k}, {p}, {list})");
              reasons_seen[a.reason as usize] += 1;
              aborts_seen += 1;
              if a.reason == cad::ABORT_BUDGET || a.reason == cad::ABORT_PREDICTION {
                motion_abort_at.get_or_insert(f);
              }
            }
          }
          let cov = cplan.aborts.iter().any(|a| a.k == k && a.p == p && a.list == 0);
          assert_eq!(gs.state[cads_pair(k, p) + 11] != 0, cov, "{tag}: coverage drop ({k}, {p})");
          if cov {
            reasons_seen[cad::ABORT_COVERAGE as usize] += 1;
          }
          // the images' existence
          let im = &st.images[k * np + p];
          assert_eq!(gs.state[cads_pair(k, p)] != 0, !im.shown.is_empty(), "{tag}: shown image ({k}, {p}) exists");
          assert_eq!(gs.state[cads_pair(k, p) + 1] != 0, !im.build.is_empty(), "{tag}: build image ({k}, {p}) exists");
        }
      }
      // the counters
      for c in 0..4 {
        assert_eq!(gs.state[CADS_HIST + c], rep.class_hist[c], "{tag}: classify histogram [{c}]");
        class_seen[c] += rep.class_hist[c];
      }
      assert_eq!(gs.state[CADS_SHELL_FORCED], rep.shell_forced, "{tag}: shell-forced");
      let drawn: u32 = (0..classes.len()).flat_map(|k| (0..np).map(move |p| (k, p))).map(|(k, p)| gs.state[cads_tp(gt.t, k, p) + 8]).sum();
      assert_eq!(drawn as usize, rep.drawn_build.len(), "{tag}: chunk entries drawn");
      // the propagate set
      let mut cset: alloc::vec::Vec<u32> = cplan.propagate[0].iter().map(|&o| (first + o) & mask).collect();
      cset.sort_unstable();
      assert_eq!(gt.propagate, cset, "{tag}: the propagate set");
      // the class images: the GPU's draws against the CPU's (the splat parity's rules, the images'
      // own layout)
      let ilayout = cfg.image_geom(w, h).layout;
      for (k, p, gshown, gbuild) in device.download_dust_cadence_images(pe).unwrap() {
        let im = &st.images[k * np + p];
        if !im.build.is_empty() {
          let r = cad_pyr_parity(&alloc::format!("{tag}: build image ({k}, {p})"), &ilayout, &gbuild, &im.build);
          if r.1 > 0 {
            std::println!("[cadence gpu] {tag}: build image ({k}, {p}): Σ gpu {} cpu {}, 3×3 beyond {} of {}", r.0, r.1, r.2, r.3);
          }
        }
        if !im.shown.is_empty() {
          cad_pyr_parity(&alloc::format!("{tag}: shown image ({k}, {p})"), &ilayout, &gshown, &im.shown);
        }
      }
      // the frame: the reprojected shown images and the fresh splat, against the CPU's
      {
        let gframe = device.download_dust_pyramid(pe).unwrap();
        let r = cad_pyr_parity(&alloc::format!("{tag}: frame"), &layout, &gframe, &frame_cpu);
        std::println!("[cadence gpu] {tag}: frame Σ gpu {} cpu {}, 3×3 beyond {} of {}, images shown {}", r.0, r.1, r.2, r.3, rep.images_shown);
      }
      if let Err(e) = st.check_owner_bytes() {
        panic!("{tag}: the CPU's owner bytes: {e}");
      }
      // the phases: the yaw the frame after an image is first shown; the jump the frame after a
      // list is non-empty again once the yaw's lists aborted
      if rep.images_shown > 0 && shown_at.is_none() {
        shown_at = Some(f);
        yaw_from = Some(f + 1);
      }
      let tc = &st.tiers[&0];
      let lists_live = (0..classes.len()).any(|k| (0..np).any(|p| !tc.lists[k][p][1].is_empty() || !tc.lists[k][p][2].is_empty()));
      if yaw_from.is_some_and(|y0| f + 1 >= y0 + YAW_FRAMES) && motion_abort_at.is_some() && jump_at.is_none() && lists_live {
        jump_at = Some(f + 1);
        end_at = Some(f + 5);
      }
      if end_at.is_some_and(|e| f >= e) {
        break;
      }
    }
    std::println!(
      "[cadence gpu] state: image first shown at frame {shown_at:?}, the yaw from {yaw_from:?}, the first budget/prediction abort at {motion_abort_at:?}, the eye's jump at {jump_at:?}, the end at {end_at:?}"
    );
    std::println!(
      "[cadence gpu] state: {aborts_seen} list aborts, reasons rate/budget/prediction/coverage {reasons_seen:?}, classes fresh/8/16/64 {class_seen:?} over the run"
    );
    let s = shown_at.unwrap_or_else(|| panic!("no class image shown in {FRAMES} frames: the cycles never completed"));
    assert!(
      motion_abort_at.is_some_and(|a| a > s),
      "no budget or prediction abort after the image shown at frame {s}: the yaw must exceed the margin's motion budget"
    );
    assert!(jump_at.is_some(), "no list non-empty again after the yaw in {FRAMES} frames: the eye's jump never ran");
    assert!(reasons_seen[cad::ABORT_RATE as usize] >= 1, "no rate abort: the eye's jump must abort a list");
    assert!(class_seen.iter().filter(|&&n| n > 0).count() >= 2, "one class only over the run");
    device.dust_cadence_enable(pe, None).unwrap();
  });
}

/// the splat parity's rules on the levels of two pyramids of `layout` (Σ counts within 10⁻³, the
/// 3×3 τ sums within 5 % + 6 on all but 1 % of the neighbourhoods with ≥ 100 counts, τ_max within
/// 10⁻³): `(Σ gpu, Σ cpu, violations, compared)`
fn cad_pyr_parity(tag: &str, layout: &dust::PyramidLayout, gpu: &[u32], cpu: &[u32]) -> (u64, u64, usize, usize) {
  let (mut sg, mut sc, mut bad, mut cmp) = (0u64, 0u64, 0usize, 0usize);
  for &(off, lw, lh) in &layout.levels {
    for t in 0..(lw * lh) as usize {
      let i = off as usize + t * dust::PYRAMID_TEXEL_WORDS as usize;
      sg += gpu[i] as u64;
      sc += cpu[i] as u64;
      let (tx, ty) = ((t as u32 % lw) as i64, (t as u32 / lw) as i64);
      let (mut bg, mut bc) = (0u64, 0u64);
      for dy in -1..=1i64 {
        for dx in -1..=1i64 {
          let (x, y) = (tx + dx, ty + dy);
          if x >= 0 && y >= 0 && x < lw as i64 && y < lh as i64 {
            let j = off as usize + (y as usize * lw as usize + x as usize) * dust::PYRAMID_TEXEL_WORDS as usize;
            bg += gpu[j] as u64;
            bc += cpu[j] as u64;
          }
        }
      }
      if bg.max(bc) >= 100 {
        cmp += 1;
        if bg.abs_diff(bc) as f64 > 6.0 + 5e-2 * bg.max(bc) as f64 {
          if bad < 3 {
            std::println!("[cadence gpu] {tag}: level texel {t} at {off}: 3×3 τ gpu {bg} cpu {bc}");
          }
          bad += 1;
        }
      }
    }
  }
  let (tg, tc) = (f32::from_bits(gpu[dust::PYR_TAU_MAX as usize]), f32::from_bits(cpu[dust::PYR_TAU_MAX as usize]));
  assert!((tg - tc).abs() <= 1e-3 * tg.max(tc), "{tag}: τ_max gpu {tg} cpu {tc}");
  assert!(sc == 0 && sg == 0 || ((sg as f64 / sc.max(1) as f64) - 1.0).abs() < 1e-3, "{tag}: Σ counts gpu {sg} cpu {sc}");
  assert!(bad <= 1 + cmp / 100, "{tag}: 3×3 τ sums beyond 5 % + 6 on {bad} of {cmp}");
  (sg, sc, bad, cmp)
}

/// A deterministic permutation of `0..n` (an LCG shuffle)
fn cad_shuffle(n: usize, seed: u32) -> alloc::vec::Vec<usize> {
  let mut v: alloc::vec::Vec<usize> = (0..n).collect();
  let mut s = seed;
  for i in (1..n).rev() {
    s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    v.swap(i, (s as usize) % (i + 1));
  }
  v
}

/// One drawn frame of a tier (no specks), the splat by range or by `list` (address, entries)
#[allow(clippy::too_many_arguments)]
fn cad_frame(
  device: &Device,
  pe: crate::gpu::PresentationEngineHandle,
  id: u64,
  capacity: u32,
  first: u32,
  live: u32,
  frame: &DustFrame,
  mvp: [f32; 16],
  eye: [f32; 3],
  list: Option<(u64, u32)>,
) {
  let (w, h) = (320u32, 200u32);
  let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
  let mut cmds = device
    .run_transient_commands(|cmd| {
      device.cmd_dust_pre_propagate_barrier(cmd);
      let v = device.cmd_dust_frame_begin(cmd, pe, [w, h], 0, flow)?;
      device.cmd_dust_system_begin(id, 0, flow)?;
      let (render, moments) = device.cmd_dust_propagate(cmd, id, 0, capacity, first, live, frame)?;
      device.cmd_dust_pre_splat_barrier(cmd);
      let pc = dust::DustSplatPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: live,
        flags: 0,
        exposure: 1.0,
        inv_unit: 1.0 / v.unit,
        color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
        albedo: 0.04,
        mvp,
        eye_local: [eye[0], eye[1], eye[2], 30.0 * 86400.0],
      };
      match list {
        None => device.cmd_dust_splat(cmd, pe, id, 0, first, capacity, render, moments, pc)?,
        Some((a, n)) => device.cmd_dust_splat_list(cmd, pe, id, 0, first, capacity, render, moments, pc, a, n)?,
      }
      device.cmd_dust_post_splat_barrier(cmd);
      device.cmd_dust_frame_end(cmd, pe)?;
      device.cmd_dust_system_end(id)?;
      Ok(())
    })
    .unwrap();
  cmds.cleanup(&device.device);
}

/// The enumerator's list mode draws the pyramid of its range (step 2's
/// `index_lists_draw_the_same_pyramid` on the GPU): a permutation of the tier's live slots with a
/// hole (~0) every seventh entry, against the range, word for word (gates off; the sums are
/// order-free).
#[test]
fn gpu_dust_index_lists_draw_the_same_pyramid() {
  let id = 9034u64;
  with_device(id, |device, capacity, _render_buf, moments_buf| {
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, 0);
    let first = batch.first_index & batch.ring_mask;
    let frame = frame_at(5.0);
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_propagate(cmd, id, 0, capacity, first, batch.count, &frame)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let m0: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, batch.count, capacity);
    let (eye, _) = eye_above(&m0, 3.0);
    let mvp = camera(eye, 320, 200).0;
    let (pa, pb) = (crate::gpu::PresentationEngineHandle(u64::MAX - 32), crate::gpu::PresentationEngineHandle(u64::MAX - 33));
    let mut list = alloc::vec::Vec::new();
    for (j, o) in cad_shuffle(batch.count as usize, 0x5EED).into_iter().enumerate() {
      if j % 7 == 3 {
        list.push(u32::MAX);
      }
      list.push((first + o as u32) & (capacity - 1));
    }
    let buf = device.dust_test_upload(&list).unwrap();
    for pe in [pa, pb] {
      // the views exist, their gates off, then the unit follows τ_max (DUST_READBACK_SLOTS + 2
      // frames, as splat_parity): the compared frames hold counts
      for i in 0..DUST_READBACK_SLOTS + 2 {
        cad_frame(device, pe, id, capacity, first, batch.count, &frame, mvp, eye, None);
        if i == 0 {
          device.dust_set_frame_gates(pe, false);
        }
      }
    }
    cad_frame(device, pa, id, capacity, first, batch.count, &frame, mvp, eye, None);
    cad_frame(device, pb, id, capacity, first, batch.count, &frame, mvp, eye, Some((buf.address, list.len() as u32)));
    let (ga, gb) = (device.download_dust_pyramid(pa).unwrap(), device.download_dust_pyramid(pb).unwrap());
    unsafe { device.device.device_wait_idle() }.unwrap();
    device.dust_test_free(buf);
    let diff: alloc::vec::Vec<usize> = (0..ga.len()).filter(|&i| ga[i] != gb[i]).collect();
    let nonzero = ga.iter().skip(dust::PYRAMID_HEADER_WORDS as usize).filter(|&&w| w != 0).count();
    std::println!("[cadence gpu] lists: {} entries ({} holes), {nonzero} non-zero words, {} differ", list.len(), list.len() - batch.count as usize, diff.len());
    for &i in diff.iter().take(5) {
      std::println!("[cadence gpu] lists: word {i}: range {} list {}", ga[i], gb[i]);
    }
    assert!(nonzero > 1000, "{nonzero} non-zero words");
    assert!(diff.is_empty(), "{} words differ between the range and the list", diff.len());
  });
}

/// The cadence's first frame is today's full draw: a view with the cadence on (its queues moved by
/// the class images' span, the propagate through the bitmask, the owner filter on the splat)
/// against a view without it, word for word but the queue word 58 (gates off).
#[test]
fn gpu_dust_cadence_first_frame_is_the_full_draw() {
  let id = 9035u64;
  with_device(id, |device, capacity, render_buf, moments_buf| {
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, 0);
    let first = batch.first_index & batch.ring_mask;
    let live = batch.count;
    let frame = frame_at(5.0);
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_propagate(cmd, id, 0, capacity, first, live, &frame)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let m0: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, live, capacity);
    let (eye, _) = eye_above(&m0, 3.0);
    let (mvp, rel) = camera(eye, 320, 200);
    let (pe_off, pe_on) = (crate::gpu::PresentationEngineHandle(u64::MAX - 34), crate::gpu::PresentationEngineHandle(u64::MAX - 35));
    device.dust_cadence_enable(pe_on, Some(cad::CadenceConfig::full())).unwrap();
    for pe in [pe_off, pe_on] {
      // (the unit follows τ_max after DUST_READBACK_SLOTS + 2 frames; the views' histories equal)
      for i in 0..DUST_READBACK_SLOTS + 2 {
        cad_frame(device, pe, id, capacity, first, live, &frame, mvp, eye, None);
        if i == 0 {
          device.dust_set_frame_gates(pe, false);
        }
      }
    }
    // the reference: today's frame
    cad_frame(device, pe_off, id, capacity, first, live, &frame, mvp, eye, None);
    // the cadence's first frame (everything fresh)
    let (w, h) = (320u32, 200u32);
    let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
    let mut proj = [0.0f64; 16];
    for i in 0..16 {
      proj[i] = rel[i] as f64;
    }
    let view = cad::CadenceView {
      view_rot: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
      proj,
      eye_m: [eye[0] as f64, eye[1] as f64, eye[2] as f64],
      units_per_m: 1.0,
      width: w,
      height: h,
    };
    let tier = CadenceTierInput { key: 0, system: id, ring_base: 0, first_slot: first, capacity, live_count: live, emitted: 1, rewritten_from: live, anchor_m: [0.0; 3] };
    let mut plan = CadenceGpuPlan::default();
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        let v = device.cmd_dust_frame_begin(cmd, pe_on, [w, h], 0, flow)?;
        device.cmd_dust_system_begin(id, 0, flow)?;
        let fm = cad::FrameMeta {
          view,
          t_now_s: frame.t_now_s(),
          speed: 1.0,
          nucleus_m: [0.0; 3],
          flags: 0,
          fingerprint: 0,
          reused: false,
          unit_now: v.unit,
        };
        plan = device.cmd_dust_cadence_begin(cmd, pe_on, &fm, &[tier], 0.0)?;
        let (render, moments) = device.cmd_dust_cadence_propagate(cmd, pe_on, id, 0, capacity, first, live, &frame, 1)?;
        device.cmd_dust_pre_splat_barrier(cmd);
        device.cmd_dust_cadence_classify(cmd, pe_on, 0, render, moments, mvp, [eye[0], eye[1], eye[2], 30.0 * 86400.0])?;
        device.cmd_dust_cadence_chunks(cmd, pe_on, 0, render, moments)?;
        device.cmd_dust_cadence_builds(cmd, pe_on, &[])?;
        let pc = dust::DustSplatPushConstants {
          moments: 0,
          render: 0,
          pyramid: 0,
          live_count: live,
          flags: 0,
          exposure: 1.0,
          inv_unit: 1.0 / v.unit,
          color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
          albedo: 0.04,
          mvp,
          eye_local: [eye[0], eye[1], eye[2], 30.0 * 86400.0],
        };
        device.cmd_dust_splat(cmd, pe_on, id, 0, first, capacity, render, moments, pc)?;
        device.cmd_dust_cadence_end(cmd, pe_on)?;
        device.cmd_dust_post_splat_barrier(cmd);
        device.cmd_dust_frame_end(cmd, pe_on)?;
        device.cmd_dust_system_end(id)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    assert_eq!(plan.mode, Some(cad::CadenceMode::First), "the cadence's first frame");
    let snap = device.download_dust_cadence(pe_on).unwrap();
    let all: alloc::vec::Vec<u32> = (0..live).map(|o| (first + o) & (capacity - 1)).collect();
    let mut all_sorted = all.clone();
    all_sorted.sort_unstable();
    assert_eq!(snap.tiers[0].propagate, all_sorted, "the first frame propagates every live slot");
    let (ga, gb) = (device.download_dust_pyramid(pe_off).unwrap(), device.download_dust_pyramid(pe_on).unwrap());
    let diff: alloc::vec::Vec<usize> = (0..ga.len()).filter(|&i| i != dust::PYR_QUEUE_BASE as usize && ga[i] != gb[i]).collect();
    std::println!(
      "[cadence gpu] first frame: queue words {} / {}, {} words differ",
      ga[dust::PYR_QUEUE_BASE as usize],
      gb[dust::PYR_QUEUE_BASE as usize],
      diff.len()
    );
    for &i in diff.iter().take(5) {
      std::println!("[cadence gpu] first frame: word {i}: off {} on {}", ga[i], gb[i]);
    }
    assert!(gb[dust::PYR_QUEUE_BASE as usize] > ga[dust::PYR_QUEUE_BASE as usize], "the class images move the queues");
    let nonzero = ga.iter().skip(dust::PYRAMID_HEADER_WORDS as usize).filter(|&&w| w != 0).count();
    std::println!("[cadence gpu] first frame: {nonzero} non-zero words");
    assert!(nonzero > 1000, "{nonzero} non-zero words: the frame drew nothing");
    assert!(diff.is_empty(), "{} words differ", diff.len());
    let _ = render_buf;
    device.dust_cadence_enable(pe_on, None).unwrap();
  });
}

/// Builds pause on reused frames (plan item 3): after cadence frames, a frame the view reuses
/// (`cmd_dust_frame_begin_reusable` with unchanged inputs) records a paused cadence frame — the
/// host state, the GPU's state words and owner bytes are unchanged, and no build, classify or
/// reprojection is recorded (the paused plan has no builds).
#[test]
fn gpu_dust_cadence_pauses_on_reused_frames() {
  let id = 9036u64;
  with_device(id, |device, capacity, _render_buf, moments_buf| {
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, 0);
    let first = batch.first_index & batch.ring_mask;
    let live = batch.count;
    let frame = frame_at(5.0);
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_propagate(cmd, id, 0, capacity, first, live, &frame)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let m0: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, live, capacity);
    let (eye, _) = eye_above(&m0, 3.0);
    let (w, h) = (320u32, 200u32);
    let rel = camera(eye, w, h).1;
    let mut proj = [0.0f64; 16];
    for i in 0..16 {
      proj[i] = rel[i] as f64;
    }
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 36);
    device.dust_cadence_enable(pe, Some(cad::CadenceConfig::full())).unwrap();
    let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
    let run = |f: u32, reusable: bool| -> (bool, CadenceGpuPlan) {
      let mut out = (false, CadenceGpuPlan::default());
      let view = cad::CadenceView {
        view_rot: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        proj,
        eye_m: [eye[0] as f64 + f.min(6) as f64, eye[1] as f64, eye[2] as f64],
        units_per_m: 1.0,
        width: w,
        height: h,
      };
      let mvp = view.mvp([0.0; 3]);
      let e = view.eye_local([0.0; 3]);
      let tier = CadenceTierInput { key: 0, system: id, ring_base: 0, first_slot: first, capacity, live_count: live, emitted: 1, rewritten_from: live, anchor_m: [0.0; 3] };
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let (v, reused) = if reusable {
            device.cmd_dust_frame_begin_reusable(cmd, pe, [w, h], 0, flow, 0x5EED)?
          } else {
            (device.cmd_dust_frame_begin(cmd, pe, [w, h], 0, flow)?, false)
          };
          let fm = cad::FrameMeta {
            view,
            t_now_s: frame.t_now_s(),
            speed: 1.0,
            nucleus_m: [0.0; 3],
            flags: 0,
            fingerprint: 0,
            reused,
            unit_now: v.unit,
          };
          let plan = device.cmd_dust_cadence_begin(cmd, pe, &fm, &[tier], 0.0)?;
          if !reused {
            device.cmd_dust_system_begin(id, 0, flow)?;
            let (render, moments) = device.cmd_dust_cadence_propagate(cmd, pe, id, 0, capacity, first, live, &frame, 1)?;
            device.cmd_dust_pre_splat_barrier(cmd);
            device.cmd_dust_cadence_classify(cmd, pe, 0, render, moments, mvp, [e[0], e[1], e[2], 30.0 * 86400.0])?;
            device.cmd_dust_cadence_chunks(cmd, pe, 0, render, moments)?;
            device.cmd_dust_cadence_reproject(cmd, pe)?;
            device.cmd_dust_system_end(id)?;
          }
          device.cmd_dust_cadence_end(cmd, pe)?;
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          out = (reused, plan);
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      out
    };
    // the view's first frame creates it with its class images; then cadence frames (the eye moves
    // for six, then holds still so the readback-driven header words settle: the reuse compares them)
    for f in 0..12u32 {
      run(f, false);
    }
    // two reusable frames at an unchanged view: the second reuses the first's image
    let (r1, _) = run(12, true);
    let before = device.dust_cadence_digest(pe).unwrap();
    let (r2, plan) = run(12, true);
    let after = device.dust_cadence_digest(pe).unwrap();
    std::println!("[cadence gpu] reuse: first reusable frame reused {r1}, second {r2}, mode {:?}, digest {before:#x} → {after:#x}", plan.mode);
    assert!(!r1 && r2, "the second unchanged frame must reuse the image");
    assert_eq!(plan.mode, Some(cad::CadenceMode::Paused), "a reused frame is a paused cadence frame");
    assert!(plan.builds.is_empty() && plan.swaps.is_empty(), "a paused frame records no build or swap");
    assert_eq!(before, after, "a reused frame changes nothing of the cadence");
    device.dust_cadence_enable(pe, None).unwrap();
  });
}

/// The telemetry is consistent with itself and with the plans it reports: after twelve cadence
/// frames (the eye stepping), every live cluster is either fresh or SHOWN in exactly one pair
/// (fresh + Σ SHOWN = live), the valid shown entries are within the shown lists, the reprojected
/// light is a share of the frame's in [0, 1], the mode and swaps are those of the plan four frames
/// before; the line the lab prints is printed.
#[test]
fn gpu_dust_cadence_telemetry_is_consistent() {
  let id = 9037u64;
  with_device(id, |device, capacity, _render_buf, moments_buf| {
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, 0);
    let first = batch.first_index & batch.ring_mask;
    let live = batch.count;
    let frame = frame_at(5.0);
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_propagate(cmd, id, 0, capacity, first, live, &frame)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let m0: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, live, capacity);
    let (eye, _) = eye_above(&m0, 3.0);
    let (w, h) = (320u32, 200u32);
    let rel = camera(eye, w, h).1;
    let mut proj = [0.0f64; 16];
    for i in 0..16 {
      proj[i] = rel[i] as f64;
    }
    let f_px = 0.5 * h as f32 / 30.0f32.to_radians().tan();
    let d = eye[2] - eye_above(&m0, 0.0).0[2];
    let step = (d / (40.0 * f_px)).max(1.0e-3) as f64;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 37);
    device.dust_cadence_enable(pe, Some(cad::CadenceConfig::full())).unwrap();
    let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
    let mut plans: alloc::vec::Vec<CadenceGpuPlan> = alloc::vec::Vec::new();
    for f in 0..14u32 {
      let view = cad::CadenceView {
        view_rot: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        proj,
        eye_m: [eye[0] as f64 + f as f64 * step, eye[1] as f64, eye[2] as f64],
        units_per_m: 1.0,
        width: w,
        height: h,
      };
      let mvp = view.mvp([0.0; 3]);
      let e = view.eye_local([0.0; 3]);
      let tier = CadenceTierInput { key: 0, system: id, ring_base: 0, first_slot: first, capacity, live_count: live, emitted: 1, rewritten_from: live, anchor_m: [0.0; 3] };
      let mut plan = CadenceGpuPlan::default();
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let v = device.cmd_dust_frame_begin(cmd, pe, [w, h], 0, flow)?;
          let fm = cad::FrameMeta { view, t_now_s: frame.t_now_s(), speed: 1.0, nucleus_m: [0.0; 3], flags: 0, fingerprint: 0, reused: false, unit_now: v.unit };
          device.cmd_dust_system_begin(id, 0, flow)?;
          plan = device.cmd_dust_cadence_begin(cmd, pe, &fm, &[tier], 0.0)?;
          let (render, moments) = device.cmd_dust_cadence_propagate(cmd, pe, id, 0, capacity, first, live, &frame, 1)?;
          device.cmd_dust_pre_splat_barrier(cmd);
          let pc = dust::DustSplatPushConstants {
            moments: 0,
            render: 0,
            pyramid: 0,
            live_count: live,
            flags: 0,
            exposure: 1.0,
            inv_unit: 1.0 / v.unit,
            color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
            albedo: 0.04,
            mvp,
            eye_local: [e[0], e[1], e[2], 30.0 * 86400.0],
          };
          device.cmd_dust_cadence_classify(cmd, pe, 0, render, moments, mvp, pc.eye_local)?;
          device.cmd_dust_cadence_chunks(cmd, pe, 0, render, moments)?;
          let geom = plan.geom.clone();
          let draws: alloc::vec::Vec<CadenceBuildDraw> = plan
            .builds
            .iter()
            .map(|b| {
              let mut pcb = pc;
              pcb.mvp = match geom.as_ref() {
                Some(g) => b.view.mvp_image([0.0; 3], g),
                None => b.view.mvp([0.0; 3]),
              };
              let be = b.view.eye_local([0.0; 3]);
              pcb.eye_local = [be[0], be[1], be[2], 30.0 * 86400.0];
              pcb.inv_unit = if b.unit > 0.0 { 1.0 / b.unit } else { 0.0 };
              CadenceBuildDraw { k: b.k, p: b.p, key: 0, system: id, ring_base: 0, first_slot: first, capacity, render, moments, pc: pcb }
            })
            .collect();
          device.cmd_dust_cadence_builds(cmd, pe, &draws)?;
          device.cmd_dust_cadence_reproject(cmd, pe)?;
          device.cmd_dust_splat(cmd, pe, id, 0, first, capacity, render, moments, pc)?;
          device.cmd_dust_cadence_end(cmd, pe)?;
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      plans.push(plan);
    }
    let tel = device.dust_cadence_telemetry(pe).expect("a telemetry after the readbacks");
    // (the readback of the frame four copies before the newest: frame 14 − 1 − 4 = 9's plan)
    let of = &plans[plans.len() - 1 - DUST_READBACK_SLOTS];
    let shown: u32 = tel.shown_valid.iter().sum();
    std::println!(
      "[cadence gpu] telemetry: ε {} classes {:?} × {} margin {:?}; mode {:?} reset {:?} swaps {:?} admitted {:?}; live {} fresh {} ({:.3}); lists {:?}; shown valid {:?}, ghosts {} ({:.4}); images {}; aborts rate/budget/prediction/coverage {:?}; d3 {} / {} at classify, unshifted {}, shell {}, classes {:?}; shown light {:.4}",
      tel.eps_px, tel.classes, tel.phases, tel.margin_px, tel.mode, tel.reset, tel.swaps, tel.admitted, tel.live, tel.fresh, tel.fresh_fraction,
      tel.lists, tel.shown_valid, tel.ghost_entries, tel.ghost_share, tel.images_shown, tel.aborts, tel.d3_rejected, tel.d3_at_classify,
      tel.unshifted_stale, tel.shell_forced, tel.class_hist, tel.shown_light_share
    );
    assert_eq!(tel.live, live, "the live clusters");
    assert_eq!(tel.fresh + shown, tel.live, "every live cluster is fresh or SHOWN in one pair");
    for (i, &(_, _, l)) in tel.lists.iter().enumerate() {
      assert!(tel.shown_valid[i] <= l[0], "pair {i}: {} valid of {} shown entries", tel.shown_valid[i], l[0]);
    }
    assert!((0.0..=1.0001).contains(&tel.shown_light_share), "shown light share {}", tel.shown_light_share);
    assert_eq!(tel.mode, of.mode, "the telemetry's mode is its frame's");
    assert_eq!(tel.swaps, of.swaps, "the telemetry's swaps are its frame's");
    device.dust_cadence_enable(pe, None).unwrap();
  });
}

/// The derived per-word bound of `dust_reproject.comp` against `reproject_add` at frame texel
/// `(l, tx, ty)`. Both mirrors evaluate the same f32 chain in the same order; `precise` keeps the
/// GPU's products and sums unfused and correctly rounded, so the operations whose results differ
/// are the divisions x_s = h_x / h_z and x_s / s (per axis), which GLSL allows 2.5 ulp where the
/// CPU rounds correctly (and x_s / s is exact on the CPU). With u = 2⁻²⁴ and ulp(x) ≤ 2u|x|:
///   |Δq| ≤ (5u + 5u + u) q = 11 u q            q = x_s / s, the tap coordinate on the level's grid
///   ε_x = |Δa_x| ≤ |Δq| + 2u (q + ½) + 2u ≤ u (13 q + 3)   (fx = q − ½ rounded on each side;
///          a_x = fx − ⌊fx⌋ exact for fx ≥ 0, rounded once for fx ∈ [−½, 0))
/// The bilinear sum is a continuous piecewise-bilinear function of (fx, fy) (across a floor flip
/// too), Lipschitz in fx with constant ≤ max |B − A| over adjacent taps ≤ D, D the largest word
/// of the 4 × 4 block around the taps (the words are ≥ 0; a tap without count is skipped by both,
/// i.e. a zero): per image i, |Δv| ≤ r_i (ε_x + ε_y) D_i. The roundings relative to v (every term
/// ≥ 0): the weight chain (1 − a, the product, × r: 3u per side → 6u), the products (2u) and the
/// T = 4n sums (2(T − 1)u) → ≤ (2T + 8) u v. The dither ⌊v + u_h⌋ (the same u_h) moves by at
/// most ⌈|Δv|⌉: + 1. An image whose pre-image lies within the first division's error (6u, taken
/// as 8u) of its grid's edge may be taken by one mirror only: its whole r_i D_i. h_z ≤ 0 and the
/// level mask decide identically (no division).
///   |Δword_q| ≤ 1 + (2T + 8) u v_q + Σ_i r_i [ε_x + ε_y + edge_i] D_{i,q}
/// (at the margin's 448-texel grid ε_x + ε_y ≤ 3.5e-4: a tap error of the order of the word
/// itself, a wrong tap or texel, stays far outside it.)
fn cad_reproject_bound(
  l: usize,
  tx: u32,
  ty: u32,
  imgs: &[(&[u32], [f32; 9], f32, u32)],
  unit_now: f32,
  ilayout: &dust::PyramidLayout,
  vmax: [f64; 4],
) -> [f64; 4] {
  const U: f64 = 1.0 / 16_777_216.0;
  let s = (1u32 << l) as f32;
  let (cx, cy) = ((tx as f32 + 0.5) * s, (ty as f32 + 0.5) * s);
  let terms = 4.0 * imgs.len() as f64;
  let mut b = vmax.map(|v| 1.0 + (2.0 * terms + 8.0) * U * v);
  for &(words, h, unit, _) in imgs {
    let ratio = if unit > 0.0 { unit / unit_now } else { 0.0 };
    if words[dust::PYR_LEVEL_MASK as usize] & (1 << l) == 0 || !(ratio > 0.0) {
      continue;
    }
    let Some(&(ioff, iw, ih)) = ilayout.levels.get(l) else { continue };
    // the CPU's own f32 (h_z's sign is shared; the pre-image in f64 for the band)
    let hx = h[0] * cx + h[1] * cy + h[2];
    let hy = h[3] * cx + h[4] * cy + h[5];
    let hz = h[6] * cx + h[7] * cy + h[8];
    if !(hz > 0.0) {
      continue;
    }
    let (xs, ys) = (hx as f64 / hz as f64, hy as f64 / hz as f64);
    let (gw, gh) = (iw as f64 * s as f64, ih as f64 * s as f64);
    let near = |x: f64, g: f64| x.abs() <= 8.0 * U * g || (x - g).abs() <= 8.0 * U * g.max(x.abs());
    let edge = near(xs, gw) || near(ys, gh);
    if !edge && !(xs >= 0.0 && ys >= 0.0 && xs <= gw && ys <= gh) {
      continue;
    }
    let (qx, qy) = ((xs / s as f64).max(0.0), (ys / s as f64).max(0.0));
    let eps = U * (13.0 * qx + 3.0) + U * (13.0 * qy + 3.0);
    let (x0, y0) = ((qx - 0.5).floor() as i64, (qy - 0.5).floor() as i64);
    for (q, bq) in b.iter_mut().enumerate() {
      let mut d = 0u32;
      for dy in -1..=2i64 {
        for dx in -1..=2i64 {
          let sx = (x0 + dx).clamp(0, iw as i64 - 1) as u32;
          let sy = (y0 + dy).clamp(0, ih as i64 - 1) as u32;
          let i = (ioff + (sy * iw + sx) * dust::PYRAMID_TEXEL_WORDS) as usize;
          if words[i] != 0 {
            d = d.max(words[i + q]);
          }
        }
      }
      *bq += ratio as f64 * (eps + if edge { 1.0 } else { 0.0 }) * d as f64;
    }
  }
  b
}

/// a rotation of `deg` about axis `axis` (0 x, 1 y, 2 z) as a column-major 4×4 (f64)
fn rot4(axis: usize, deg: f64) -> [f64; 16] {
  let (s, c) = deg.to_radians().sin_cos();
  let mut m = [0.0f64; 16];
  m[15] = 1.0;
  let (a, b) = match axis {
    0 => (1, 2),
    1 => (2, 0),
    _ => (0, 1),
  };
  m[axis * 4 + axis] = 1.0;
  m[a * 4 + a] = c;
  m[b * 4 + b] = c;
  m[a * 4 + b] = s;
  m[b * 4 + a] = -s;
  m
}

/// one class image drawn by the CPU mirror at `view` with the unit following its τ_max (the
/// host's rule): (words, unit)
fn cad_image(
  moments: &[dust::DustMoments],
  render: &[dust::DustRenderCluster],
  view: &cad::CadenceView,
  layout: &dust::PyramidLayout,
  keep: &dyn Fn(usize) -> bool,
) -> (alloc::vec::Vec<u32>, f32) {
  let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
  let e = view.eye_local([0.0; 3]);
  let mut render_k = render.to_vec();
  for (i, r) in render_k.iter_mut().enumerate() {
    if !keep(i) {
      r.age_id_dbeta_flux[1] = f32::from_bits(r.age_id_dbeta_flux[1].to_bits() & !0x8000_0000);
    }
  }
  let draw = |unit: f32| {
    let pc = dust::DustSplatPushConstants {
      moments: 0,
      render: 0,
      pyramid: 0,
      live_count: moments.len() as u32,
      flags: 0,
      exposure: 1.0,
      inv_unit: 1.0 / unit,
      color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
      albedo: 0.04,
      mvp: view.mvp([0.0; 3]),
      eye_local: [e[0], e[1], e[2], 30.0 * 86400.0],
    };
    let mut w = alloc::vec![0u32; layout.total_words as usize];
    w[..dust::PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
    w[dust::PYR_UNIT as usize] = unit.to_bits();
    dust::splat_tier(moments, &render_k, &pc, &flow, 0, layout, &mut w);
    w
  };
  let first = draw(1.0);
  let tau = f32::from_bits(first[dust::PYR_TAU_MAX as usize]);
  let unit = if tau > 0.0 && tau.is_finite() { tau * crate::scene::dust::WHITE_TILE_UNIT_REL } else { 1.0 };
  (draw(unit), unit)
}

/// `dust_reproject.comp` equals `cadence::reproject_add` (Unit) on the same image words, maps,
/// units and generations: the identity word for word (the frame's header folds included); a
/// perspective yaw of 5° with a second image (half the clusters, a different unit and
/// generation) and an orthographic pan + roll + zoom 1.1 within the f32 chain's derived bound
/// ([`cad_reproject_bound`], per image) on every written texel (the old ±(1 + 10⁻⁵ v) count printed), Σ
/// counts within 10⁻⁵, the depth words equal where both mirrors wrote, the folds exact; the same
/// at the full configuration's margin (images 448 × 328 drawn at their image views, the map
/// `map_to_image`, the taps on each image's own grid; the identity a translation by the margin,
/// word-exact).
#[test]
fn gpu_dust_reproject_matches_reference() {
  let id = 9033u64;
  with_device(id, |device, capacity, render_buf, moments_buf| {
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, 0);
    let first = batch.first_index & batch.ring_mask;
    let frame = frame_at(5.0);
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_propagate(cmd, id, 0, capacity, first, batch.count, &frame)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let render: alloc::vec::Vec<dust::DustRenderCluster> = records(device, render_buf, first, batch.count, capacity);
    let moments: alloc::vec::Vec<dust::DustMoments> = records(device, moments_buf, first, batch.count, capacity);
    let (w, h) = (320u32, 200u32);
    let layout = dust::PyramidLayout::new(w, h);
    let (eye, ext) = eye_above(&moments, 3.0);
    let persp = |rot: [f64; 16], e: [f32; 3]| {
      let rel = camera([0.0; 3], w, h).1;
      let mut proj = [0.0f64; 16];
      for i in 0..16 {
        proj[i] = rel[i] as f64;
      }
      cad::CadenceView { view_rot: rot, proj, eye_m: [e[0] as f64, e[1] as f64, e[2] as f64], units_per_m: 1.0, width: w, height: h }
    };
    let ortho = |rot: [f64; 16], e: [f32; 3], half: f64| {
      let mut proj = [0.0f64; 16];
      proj[0] = 1.0 / half;
      proj[5] = 1.0 / half;
      proj[10] = 1e-12;
      proj[15] = 1.0;
      cad::CadenceView { view_rot: rot, proj, eye_m: [e[0] as f64, e[1] as f64, e[2] as f64], units_per_m: 1.0, width: w, height: h }
    };
    let id4 = rot4(2, 0.0);
    // the margin's geometry (the full configuration's 20 %: 448 × 328 images at 320 × 200)
    let mgeom = cad::CadenceConfig::full().image_geom(w, h);
    let cases: [(&str, cad::CadenceView, cad::CadenceView, bool, bool); 5] = [
      ("identity", persp(id4, eye), persp(id4, eye), false, false),
      ("yaw 5° + a second image", persp(id4, eye), persp(rot4(1, 5.0), eye), true, false),
      (
        "ortho pan + roll + zoom",
        ortho(id4, eye, ext as f64),
        ortho(rot4(2, 3.0), [eye[0] + 0.03 * ext, eye[1] - 0.02 * ext, eye[2]], ext as f64 / 1.1),
        false,
        false,
      ),
      ("margin identity", persp(id4, eye), persp(id4, eye), false, true),
      ("margin yaw 5° + a second image", persp(id4, eye), persp(rot4(1, 5.0), eye), true, true),
    ];
    for (tag, frozen, current, two, margin) in cases {
      let geom = margin.then(|| mgeom.clone());
      // an image drawn at its image view (the margin's projection) in its own layout
      let (iview, ilayout) = match geom.as_ref() {
        Some(g) => (frozen.image_view(g), g.layout.clone()),
        None => (frozen, layout.clone()),
      };
      let (img_a, unit_a) = cad_image(&moments, &render, &iview, &ilayout, &|_| true);
      let (img_b, unit_b) = cad_image(&moments, &render, &iview, &ilayout, &|i| i % 2 == 0);
      let map = match geom.as_ref() {
        Some(g) => current.map_to_image(&frozen, g).expect("a map"),
        None => current.map_from(&frozen).expect("a map"),
      };
      let unit_now = unit_a * 1.25;
      let mut imgs: alloc::vec::Vec<(&[u32], [f32; 9], f32, u32)> = alloc::vec![(&img_a, map, unit_a, 7)];
      if two {
        imgs.push((&img_b, map, unit_b * 0.5, 11));
      }
      let seed = reproject_seed(imgs.iter().map(|x| x.3));
      let shown: alloc::vec::Vec<cad::ShownImage<'_>> =
        imgs.iter().map(|&(words, map, unit, _)| cad::ShownImage { words, map, unit, layout: geom.as_ref().map(|g| &g.layout) }).collect();
      let mut cpu = alloc::vec![0u32; layout.total_words as usize];
      cpu[..dust::PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
      let header = cpu[..dust::PYRAMID_HEADER_WORDS as usize].to_vec();
      cad::reproject_add(&layout, &mut cpu, &shown, unit_now, seed, cad::ReprojectFactor::Unit);
      let gpu = device.dust_reproject_test(&layout, &header, &imgs, unit_now).unwrap();
      for word in [dust::PYR_TAU_MAX, dust::PYR_LEVEL_MASK] {
        assert_eq!(gpu[word as usize], cpu[word as usize], "{tag}: header word {word}");
      }
      let (mut written, mut beyond, mut depth_cmp, mut depth_bad) = (0usize, 0usize, 0usize, 0usize);
      let (mut over_bound, mut worst_ratio, mut worst_rel, mut worst_abs) = (0usize, 0.0f64, 0.0f64, 0u32);
      let (mut sg, mut sc) = (0f64, 0f64);
      let mut exact = true;
      for (l, &(off, lw, lh)) in layout.levels.iter().enumerate() {
        for t in 0..(lw * lh) as usize {
          let i = off as usize + t * dust::PYRAMID_TEXEL_WORDS as usize;
          let (g, c) = (&gpu[i..i + 5], &cpu[i..i + 5]);
          if g != c {
            exact = false;
          }
          if g[0] == 0 && c[0] == 0 && g[4] == 0 && c[4] == 0 {
            continue;
          }
          written += 1;
          sg += g[0] as f64;
          sc += c[0] as f64;
          // the old empirical band (printed only)
          beyond += (0..4).any(|q| g[q].abs_diff(c[q]) as f64 > 1.0 + 1e-5 * g[q].max(c[q]) as f64) as usize;
          // the derived bound
          let vmax = [0usize, 1, 2, 3].map(|q| g[q].max(c[q]) as f64);
          let bound = cad_reproject_bound(l, t as u32 % lw, t as u32 / lw, &imgs, unit_now, &ilayout, vmax);
          for q in 0..4 {
            let d = g[q].abs_diff(c[q]);
            worst_abs = worst_abs.max(d);
            worst_ratio = worst_ratio.max(d as f64 / bound[q]);
            if vmax[q] > 0.0 {
              worst_rel = worst_rel.max(d as f64 / vmax[q]);
            }
          }
          if (0..4).any(|q| g[q].abs_diff(c[q]) as f64 > bound[q]) {
            if over_bound < 3 {
              std::println!("[cadence gpu] reproject {tag}: word {i}: gpu {g:?} cpu {c:?} beyond the bound {bound:?}");
            }
            over_bound += 1;
          }
          if g[4] != 0 && c[4] != 0 {
            depth_cmp += 1;
            depth_bad += (g[4] != c[4]) as usize;
          }
        }
      }
      std::println!(
        "[cadence gpu] reproject {tag}: {written} texels written, {beyond} beyond ±(1 + 1e-5 v), depth differs on {depth_bad} of {depth_cmp}, Σ counts gpu {sg:.0} cpu {sc:.0}, word-exact {exact}"
      );
      let inz = |im: &[u32]| (dust::PYRAMID_HEADER_WORDS as usize..im.len()).step_by(dust::PYRAMID_TEXEL_WORDS as usize).filter(|&i| im[i] != 0).count();
      std::println!("[cadence gpu] reproject {tag}: image A {} non-zero texels, image B {}", inz(&img_a), inz(&img_b));
      assert!(written > 300, "{tag}: {written} texels");
      if tag == "identity" || tag == "margin identity" {
        assert!(exact, "{tag}: the identity map must be word-exact");
      }
      std::println!(
        "[cadence gpu] reproject {tag}: the derived bound: {over_bound} texels beyond it, max |Δ| / bound {worst_ratio:.3}, max |Δ| {worst_abs}, max |Δ| / v {worst_rel:.2e}"
      );
      assert_eq!(over_bound, 0, "{tag}: {over_bound} of {written} texels beyond the derived bound");
      assert!(((sg - sc) / sc.max(1.0)).abs() <= 1e-5, "{tag}: Σ counts gpu {sg} cpu {sc}");
      assert!(depth_bad <= depth_cmp / 1000, "{tag}: depth differs on {depth_bad} of {depth_cmp}");
    }
  });
}
