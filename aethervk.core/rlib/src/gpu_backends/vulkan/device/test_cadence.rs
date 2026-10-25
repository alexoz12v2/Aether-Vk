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
        dust::PYR_EWA_GATE_SCALE,
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
      // the incoming bytes (the handover)
      let idiff: alloc::vec::Vec<usize> = (0..capacity as usize).filter(|&s| gt.incoming[s] != tc.incoming[s]).collect();
      for &s in idiff.iter().take(5) {
        std::println!("[cadence gpu] {tag}: slot {s}: incoming gpu {:#04x} cpu {:#04x}", gt.incoming[s], tc.incoming[s]);
      }
      assert!(idiff.is_empty(), "{tag}: {} incoming bytes differ", idiff.len());
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

/// The handover's GPU parity (round 7): `gpu_dust_cadence_state_matches_reference`'s scene and
/// per-frame comparisons (mode, swaps, every owner and incoming byte, the lists, steps, cursors,
/// aborts, histogram, chunk entries, the propagate set, the build/shown images and the frame), with
/// the eye fixed and a yaw over frames 1–24 that admits class 8 only, then 446 still frames: class 16
/// and class 64 are admitted and the members move 8 → 16 → 64 (the observer's perf7c: the GPU's
/// light fell 0.8 % per class-16 window in the first two class-64 windows). The collapse and drop
/// counts match the CPU's every frame.
#[test]
fn gpu_dust_cadence_handover_matches_reference() {
  handover_parity(9038, 38, None);
}

/// Round 7c's τ 0.004 defect (observer: 2.6 % shape L1 from f≈64–96 with the handover on, none
/// with it off; the CPU mirror passes the same sweep, `cadence_hands_members_over_as_the_light_bound_moves`):
/// the handover parity with class 16's light bound swept across the clusters' ages (τ off over the
/// yaw; then the bound admits from the oldest cluster to every cluster over frames 60–220, holds,
/// and returns over 260–420), so members move between classes one by one as in the observer.
/// Every per-frame comparison of the handover parity holds.
#[test]
fn gpu_dust_cadence_handover_light_sweep_matches_reference() {
  fn q(f: u32) -> f64 {
    match f {
      0..60 => 1.0,
      60..220 => 1.0 - (f - 60) as f64 / 160.0,
      220..260 => 0.0,
      _ => ((f - 260) as f64 / 160.0).min(1.0),
    }
  }
  handover_parity(9039, 39, Some(q));
}

/// The handover parity's body: system `id`, presentation engine `u64::MAX − pe_off`; `light`: the
/// age quantile class 16's light bound admits from, per frame after the yaw (τ = 1.5 · 16 · Δt /
/// age_q, Δt the run's 0.02 s; τ 0 over the yaw), or `None` (the configuration's τ throughout)
fn handover_parity(id: u64, pe_off: u64, light: Option<fn(u32) -> f64>) {
  with_device(id, |device, capacity, render_buf, moments_buf| {
    let mut seq = 0u64;
    let batch = test_batch(1024, 3600.0, 3, capacity, 0xC0FFEE);
    emit(device, id, &batch, seq);
    seq += 1;
    let mask = capacity - 1;
    let (mut first, mut live) = (batch.first_index & mask, batch.count);
    let mut emitted = 1u64;
    let mut rewritten_from: u32;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - pe_off);
    let (w, h) = (320u32, 200u32);
    let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
    let ttl = 30.0 * 86400.0f32;
    let mut cfg = cad::CadenceConfig::full();
    if light.is_some() {
      cfg.light_tol = 0.0;
    }
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
    // the light sweep's ages (sorted; they grow by 0.02 s a frame, negligible against days)
    let ages: alloc::vec::Vec<f32> = {
      let mut a: alloc::vec::Vec<f32> = m0.iter().filter(|m| m.flux() > 0.0).map(|m| m.age()).collect();
      a.sort_by(f32::total_cmp);
      a
    };
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
    const FRAMES: u32 = 470;
    const YAW_FRAMES: u32 = 24;
    // the yaw per frame: a corner motion U with 2·8·U ≤ m − G < 2·16·U (class 8 admitted while it
    // yaws, class 16 not), set from the CPU's motion budget after frame 0
    let mut yaw_step_deg = 0.0f64;
    let (mut collapsed_gpu, mut dropped_gpu, mut admitted_at) = (0u64, 0u64, [None::<u32>; 4]);
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
      // the eye fixed (no step): only the yaw moves the view
      let _ = step;
      if f >= 1 && f <= YAW_FRAMES {
        yaw_deg += yaw_step_deg;
      }
      let t_s = T_START + 5.0 * 86400.0 + f as f64 * 0.02;
      let frame = frame_at(5.0 + f as f64 * 0.02 / 86400.0);
      if let Some(q) = light
        && enabled
      {
        let tau = if f <= YAW_FRAMES {
          0.0
        } else {
          let qq = q(f).clamp(0.0, 1.0);
          let age_q = ages[((qq * (ages.len() - 1) as f64) as usize).min(ages.len() - 1)].max(cad::DUST_CADENCE_AGE_MIN_S);
          (1.5 * classes[1] as f64 * 0.02 / age_q as f64) as f32
        };
        st.cfg.light_tol = tau;
        device.dust_cadence_set_light_tol(pe, tau);
      }
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
      if f == 0 {
        let b = st.motion_budget();
        assert!(b[0].is_finite() && b[1].is_finite(), "no motion budget: every class admitted, no handover");
        let u = b[0].min(b[1]) as f64 / 24.0;
        let half_w = 0.5 * w as f64 / f_px as f64;
        yaw_step_deg = (u / (f_px as f64 * (1.0 + half_w * half_w))).to_degrees();
        std::println!("[cadence gpu] handover: budget {b:?} px, yaw {yaw_step_deg:.5}° per frame");
      }
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
        dust::PYR_EWA_GATE_SCALE,
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
      // the incoming bytes (the handover)
      let idiff: alloc::vec::Vec<usize> = (0..capacity as usize).filter(|&s| gt.incoming[s] != tc.incoming[s]).collect();
      for &s in idiff.iter().take(5) {
        std::println!("[cadence gpu] {tag}: slot {s}: incoming gpu {:#04x} cpu {:#04x}", gt.incoming[s], tc.incoming[s]);
      }
      assert!(idiff.is_empty(), "{tag}: {} incoming bytes differ", idiff.len());
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
      // the handover's collapses (cumulative on the CPU, per frame on the GPU)
      collapsed_gpu += gs.state[CADS_HANDOVER] as u64;
      dropped_gpu += gs.state[CADS_HANDOVER + 1] as u64;
      assert_eq!(collapsed_gpu, st.handover_collapsed as u64, "{tag}: handover collapses");
      assert_eq!(dropped_gpu, st.handover_dropped as u64, "{tag}: handover dropped collapses");
      for (k, &ad) in rep.admitted.iter().enumerate().take(4) {
        if ad && admitted_at[k].is_none() && f > YAW_FRAMES {
          admitted_at[k] = Some(f);
        }
      }
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
      let _ = (jump_at, end_at, lists_live);
    }
    std::println!(
      "[cadence gpu] handover (light sweep {}): classes fresh/8/16/64 {class_seen:?}, admitted after the yaw at {admitted_at:?}, collapsed {collapsed_gpu} dropped {dropped_gpu}, aborts {aborts_seen} {reasons_seen:?}, motion abort {motion_abort_at:?}",
      light.is_some()
    );
    assert!(shown_at.is_some(), "no class image shown in {FRAMES} frames");
    assert!(admitted_at[1].is_some() && admitted_at[2].is_some(), "classes 16 and 64 not both admitted after the yaw: {admitted_at:?}");
    assert!(collapsed_gpu > 0, "no handover collapsed");
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

/// a cadenced frame's full-draw estimate: fresh as measured, each class's builds at its rate per
/// member over its last cycle times its shown members (class 8 builds 12 000 a frame, its two
/// images show the cycle's 96 000; the first frame scales its one frame by 8); one odd frame
/// moves the rate by at most 1 / N of its weight; no build ever: fresh alone; saturating at
/// u32::MAX
#[test]
fn the_full_frame_estimate_scales_the_builds() {
  let one = |b: [u32; 3]| [b, [0; 3], [0; 3]];
  let in0 = |n: u32| [n, 0, 0];
  let frames = [8, 16, 64];
  let mut rate = BuildRate::default();
  assert_eq!(full_frame_estimate([5, 50, 7], &rate, in0(1000)), [5, 50, 7]);
  rate.fold(one([10, 200, 1_000_000]), in0(12_000), frames);
  assert_eq!(full_frame_estimate([5, 50, 7_000_000], &rate, in0(96_000)), [5 + 80, 50 + 1600, 7_000_000 + 8_000_000]);
  // a steady rate stays the rate
  for _ in 0..40 {
    rate.fold(one([10, 200, 1_000_000]), in0(12_000), frames);
  }
  let steady = full_frame_estimate([0, 0, 0], &rate, in0(96_000));
  assert!(steady.iter().zip([80u32, 1600, 8_000_000]).all(|(&a, b)| a.abs_diff(b) <= b / 1000 + 1), "{steady:?}");
  // a frame that built 100 members at the cost of 12 000 (a short last step) barely moves it
  rate.fold(one([10, 200, 1_000_000]), in0(100), frames);
  let after = full_frame_estimate([0, 0, 0], &rate, in0(96_000))[2] as f64;
  assert!(after < 8_000_000.0 * 1.2, "{after}");
  let mut big = BuildRate::default();
  big.fold(one([0, 0, u32::MAX]), in0(1), frames);
  assert_eq!(full_frame_estimate([0, 0, 1], &big, in0(4))[2], u32::MAX);
}

/// each class's rate spans its own cycle (round 7): class 16 builds 9 580 members every frame but
/// its EWA work comes in bursts (the lists' order: 3 of every 16 frames carry it all, as at 1 km
/// orbit). Over one cycle the estimate is the same every frame (within 0.1 %); the old decayed
/// 8-frame sum swung by more than 2× with the same input.
#[test]
fn the_build_rate_spans_each_class_cycle() {
  let burst = |f: usize| if f % 16 < 3 { 160_000u32 } else { 0 };
  let mut rate = BuildRate::default();
  let (mut lo, mut hi) = (u32::MAX, 0u32);
  let (mut dlo, mut dhi, mut d) = (f64::MAX, 0.0f64, [0.0f64; 2]);
  for f in 0..160 {
    rate.fold([[0; 3], [0, burst(f), 0], [0; 3]], [0, 9_580, 0], [8, 16, 64]);
    // the old rule: a decayed sum with weight 7/8
    d = [d[0] * 0.875 + burst(f) as f64, d[1] * 0.875 + 9_580.0];
    if f >= 32 {
      let e = full_frame_estimate([0, 0, 0], &rate, [0, 153_000, 0])[1];
      lo = lo.min(e);
      hi = hi.max(e);
      let old = d[0] * 153_000.0 / d[1];
      dlo = dlo.min(old);
      dhi = dhi.max(old);
    }
  }
  assert!((hi - lo) as f64 <= 0.001 * hi as f64, "the estimate swings {lo}..{hi} over class 16's cycle");
  assert!(dhi > 2.0 * dlo, "the decayed 8-frame sum {dlo}..{dhi} (the test's premise)");
  // the window is the class's: a class-8 burst pattern is flat over 8 frames
  let mut r8 = BuildRate::default();
  for f in 0..64 {
    r8.fold([[0, if f % 8 == 0 { 8_000 } else { 0 }, 0], [0; 3], [0; 3]], [1_000, 0, 0], [8, 16, 64]);
  }
  assert_eq!(r8.builds[0][1], 8_000.0);
  assert_eq!(r8.built[0], 8_000.0);
}

/// the estimate weighs the classes apart (round 7): class 8 shows 10 000 members at a cost of 1
/// each and builds 1 250 a frame, class 16 shows 150 000 at a cost of 3 and builds 9 375. The true
/// full draw is 10 000 + 450 000 = 460 000; the pooled `Σ shown / Σ built` gives
/// (1 250 + 28 125) · 160 000 / 10 625 = 442 353 (4 % low, the orbit's early EWA switch); per
/// class it is exact. A class with shown members and no builds takes the pooled rate.
#[test]
fn the_full_frame_estimate_weighs_the_classes_apart() {
  let mut rate = BuildRate::default();
  for _ in 0..64 {
    rate.fold([[0, 1_250, 0], [0, 28_125, 0], [0; 3]], [1_250, 9_375, 0], [8, 16, 64]);
  }
  let e = full_frame_estimate([0, 0, 0], &rate, [10_000, 150_000, 0])[1];
  assert!(e.abs_diff(460_000) <= 460, "per class {e}, the full draw 460 000");
  let pooled = (1_250.0 + 28_125.0) * 160_000.0 / 10_625.0;
  assert!((e as f64 - pooled).abs() > 0.03 * 460_000.0, "per class {e} equals the pooled {pooled}");
  // class 64 shows 6 400 members it has not built in the window: the pooled rate over the classes'
  // cycles (each window holds every member of its class once: (8 · 1 250 + 16 · 28 125) /
  // (8 · 1 250 + 16 · 9 375) = 2.875 per member)
  let e64 = full_frame_estimate([0, 0, 0], &rate, [10_000, 150_000, 6_400])[1];
  let want = 460_000.0 + 6_400.0 * 2.875;
  assert!((e64 as f64 - want).abs() <= 0.001 * want, "with class 64 {e64}, want {want}");
}

/// the handover's population mismatch (1 km still, τ 0.004, f72, `AETHERVK_DUST_FEEDBACK_LOG`):
/// class 8's window built 6 320 members (the bright young clusters among them) for 507 681 EWA
/// units while its old image still showed 86 416 ordinary members; class 16 built 160 272 for
/// 454 952 and showed 80 111; the fresh pass 356 971. The uncapped scale (13.7) gave 7.5e6 units,
/// over the frame budget (the EWA bias 2 and its blur); capped at 1.25 the estimate sits inside the
/// feedback's band [budget / 8, budget], so the bias stays. A full window with consistent counts is
/// unchanged (within the lists' drift).
#[test]
fn the_full_frame_estimate_caps_a_population_mismatch() {
  let budget = (dust::DUST_EWA_FRAME_BUDGET as f64 / dust::DUST_EWA_TEXEL_UNIT as f64) as u32;
  let mut rate = BuildRate::default();
  for f in 0..64usize {
    let b8 = if f % 8 == 0 { 507_681 } else { 0 };
    let n8 = if f % 8 == 0 { 6_320 } else { 0 };
    let b16 = if f % 16 == 0 { 454_952 } else { 0 };
    let n16 = if f % 16 == 0 { 160_272 } else { 0 };
    rate.fold([[0, b8, 0], [0, b16, 0], [0; 3]], [n8, n16, 0], [8, 16, 64]);
  }
  let shown = [86_416, 80_111, 0];
  let uncapped = 356_971.0 + 507_681.0 * 86_416.0 / 6_320.0 + 454_952.0 * 80_111.0 / 160_272.0;
  assert!(uncapped > budget as f64, "the premise: {uncapped} over the budget {budget}");
  let e = full_frame_estimate([0, 356_971, 0], &rate, shown)[1];
  let want = 356_971.0 + 507_681.0 * 1.25 + 454_952.0 * 80_111.0 / 160_272.0;
  assert!((e as f64 - want).abs() <= 1.0, "capped {e}, want {want}");
  assert!(e > budget / 8 && e < budget, "{e} outside the band [{}, {budget}]", budget / 8);
  // consistent counts (the lists' drift, scale 1.001): the uncapped estimate
  let e1 = full_frame_estimate([0, 0, 0], &rate, [6_326, 160_272, 0])[1];
  let want1 = 507_681.0 * 6_326.0 / 6_320.0 + 454_952.0;
  assert!((e1 as f64 - want1).abs() <= 1.0, "consistent {e1}, want {want1}");
}

/// the feedback's band keeps the current state: frames drawn at another bias (read back
/// DUST_READBACK_SLOTS frames late) whose counters sit inside the band do not write their own
/// bias back, so the in-flight slots cannot settle as separate chains
#[test]
fn the_feedback_band_keeps_the_state() {
  use crate::scene::dust::{DUST_EWA_FRAME_BUDGET, DUST_EWA_TEXEL_UNIT, DUST_VOLUME_FRAME_BUDGET, EwaStats};
  let header_at = |ewa_bias: u32, ewa_texels: u64, vol_bias: u32, evals: u32| {
    let mut h = alloc::vec![0u32; dust::PYRAMID_HEADER_WORDS as usize];
    EwaStats::store_bias(ewa_bias, &mut h);
    h[dust::PYR_EWA_TEXELS as usize] = (ewa_texels / DUST_EWA_TEXEL_UNIT as u64) as u32;
    let w = &mut h[dust::PYR_SPLIT_DEPTH as usize];
    *w = (*w & !(0xFF << 8)) | (vol_bias << 8);
    h[dust::PYR_VOLUME_EVALS as usize] = evals;
    // the split inside its band at depth 0 (children between a quarter and the budget)
    h[dust::PYR_SPLIT_CHILDREN as usize] = dust::DUST_SPLIT_BUDGET / 2;
    h
  };
  let level = alloc::vec![0u32; 16];
  let mut host = DustViewHost::default();
  host.ewa_bias = 1;
  host.volume_bias = 14;
  host.split_depth = 2;
  let mid_ewa = DUST_EWA_FRAME_BUDGET as u64 / 4;
  let mid_vol = DUST_VOLUME_FRAME_BUDGET / 4;
  // drawn at EWA 0 / volume 12 (an older state), inside both bands: the state stays
  host.absorb(&header_at(0, mid_ewa, 12, mid_vol), &level, 1, 1.0);
  assert_eq!((host.ewa_bias, host.volume_bias, host.split_depth), (1, 14, 2));
  // below the band: one finer than the frame's own bias
  host.absorb(&header_at(2, DUST_EWA_FRAME_BUDGET as u64 / 16, 13, DUST_VOLUME_FRAME_BUDGET / 16), &level, 1, 1.0);
  assert_eq!((host.ewa_bias, host.volume_bias), (1, 12));
  // over the budget: coarser than the frame's own bias
  host.absorb(&header_at(0, DUST_EWA_FRAME_BUDGET as u64 * 2, 11, DUST_VOLUME_FRAME_BUDGET * 2), &level, 1, 1.0);
  assert_eq!((host.ewa_bias, host.volume_bias), (1, 12));
}

/// the feedback after a savefile open (`DustViewHost::reset_feedback`): the reopened N2 run read
/// back one transient frame per bias while its cameras flew to their saved poses (1.07e10 texels
/// at EWA bias 0 → 3, 8.3e8 at 3 → 4, 2.9e8 at 4 → 5), then the settled view, whose texels at
/// bias 5 sit under budget / 8 but whose remembered step (×28) refuses every lower: bias 5 for
/// good, where the live session that grew the same dust settled at 3 (blocky dust after Open).
/// A reset once the view settled measures it cold and lands where the live session did.
#[test]
fn the_feedback_reset_forgets_a_transient() {
  use crate::scene::dust::{DUST_EWA_FRAME_BUDGET, DUST_EWA_TEXEL_UNIT, DUST_VOLUME_FRAME_BUDGET, EwaStats};
  let header_at = |ewa_bias: u32, ewa_texels: u64| {
    let mut h = alloc::vec![0u32; dust::PYRAMID_HEADER_WORDS as usize];
    EwaStats::store_bias(ewa_bias, &mut h);
    h[dust::PYR_EWA_TEXELS as usize] = (ewa_texels / DUST_EWA_TEXEL_UNIT as u64) as u32;
    let w = &mut h[dust::PYR_SPLIT_DEPTH as usize];
    *w = (*w & !(0xFF << 8)) | (2 << 8);
    // the volume and the split inside their bands: only the EWA side moves
    h[dust::PYR_VOLUME_EVALS as usize] = DUST_VOLUME_FRAME_BUDGET / 4;
    h[dust::PYR_SPLIT_CHILDREN as usize] = dust::DUST_SPLIT_BUDGET / 2;
    h
  };
  let level = alloc::vec![0u32; 16];
  let eb = DUST_EWA_FRAME_BUDGET as u64;
  // the settled view: 40 budgets at bias 0, a level quarters it
  let settled = |bias: u32| eb * 40 / 4u64.pow(bias);
  let settle = |host: &mut DustViewHost| {
    for _ in 0..12 {
      let b = host.ewa_bias;
      host.absorb(&header_at(b, settled(b)), &level, 1, 1.0);
    }
  };
  // the live session: the dust grows over many frames, the view the same
  let mut live = DustViewHost::default();
  live.volume_bias = 2;
  for k in 1..=40u64 {
    let b = live.ewa_bias;
    live.absorb(&header_at(b, eb * k / 4u64.pow(b)), &level, 1, 1.0);
  }
  settle(&mut live);
  assert_eq!(live.ewa_bias, 3, "the live session's level");
  // the reopen: the whole history at once, read back during the flight
  let mut reopened = DustViewHost::default();
  reopened.volume_bias = 2;
  for (b, texels) in [(0, 10_690_000_000u64), (3, 828_700_000), (4, 292_900_000)] {
    assert_eq!(reopened.ewa_bias, b);
    reopened.absorb(&header_at(b, texels), &level, 1, 1.0);
  }
  settle(&mut reopened);
  assert_eq!(reopened.ewa_bias, 5, "the premise: the transient's memory keeps the coarse level");
  // the view settled: a cold start measures it again
  reopened.drawn = Some((1, alloc::vec![0]));
  reopened.reset_feedback();
  assert_eq!((reopened.ewa_bias, reopened.volume_bias), (0, 0));
  assert_eq!(reopened.split_depth, dust::DUST_SPLIT_DEPTH_MAX);
  assert_eq!(reopened.split_depth_over, u32::MAX);
  assert_eq!(reopened.fb_memory, FeedbackSet::default());
  assert_eq!(reopened.fb_full, FeedbackSet::default());
  assert!(reopened.fb_cad.is_none() && reopened.budget.is_none());
  assert!(reopened.drawn.is_none(), "the next frame draws (and is read back), even paused");
  reopened.volume_bias = 2;
  settle(&mut reopened);
  assert_eq!(reopened.ewa_bias, live.ewa_bias, "path independent after the reset");
}

/// the EWA feedback stops at `DUST_EWA_BIAS_FEEDBACK_MAX` (5): the 67P Earth view (1.9·10⁵ km
/// half-width) read 5.4e8 texels at 5, twice the budget, and went to 6, where the frame collapsed
/// to a flat field and the exposure turned it into a grey wall. 6 stays for the override only.
#[test]
fn the_ewa_feedback_stops_at_bias_5() {
  use crate::scene::dust::{DUST_EWA_BIAS_FEEDBACK_MAX, DUST_EWA_FRAME_BUDGET, DUST_EWA_TEXEL_UNIT, EwaStats};
  assert_eq!(DUST_EWA_BIAS_FEEDBACK_MAX, 5);
  let cold = EwaStats { texels: 16_820_000_000, bias: 0 };
  assert_eq!(cold.suggested_bias(), 3);
  let at3 = EwaStats { texels: 1_792_000_000, bias: 3 };
  assert_eq!(at3.suggested_bias(), 5, "⌈log₄ 6.7⌉ = 2 levels");
  let at5 = EwaStats { texels: 544_900_000, bias: 5 };
  assert_eq!(at5.suggested_bias(), 5, "over the budget at 5: no 6");
  let mut h = alloc::vec![0u32; dust::PYRAMID_HEADER_WORDS as usize];
  EwaStats::store_bias(5, &mut h);
  h[dust::PYR_EWA_TEXELS as usize] = (544_900_000u64 / DUST_EWA_TEXEL_UNIT as u64) as u32;
  h[dust::PYR_SPLIT_CHILDREN as usize] = dust::DUST_SPLIT_BUDGET / 2;
  let mut host = DustViewHost::default();
  host.ewa_bias = 5;
  host.absorb(&h, &alloc::vec![0u32; 16], 1, 1.0);
  assert_eq!(host.ewa_bias, 5);
  assert!(544_900_000 > DUST_EWA_FRAME_BUDGET as u64, "the premise: over the budget");
}

/// A flat white-point level (every lit texel the same count: the EWA bias 6 frame of the 67P Earth
/// view, 262 143 or 582 841 counts) moves no white point; a dust image (its densest texels ×4–×8
/// above the 99th percentile) does.
#[test]
fn a_flat_level_moves_no_white_point() {
  let words = dust::PYRAMID_TEXEL_WORDS as usize;
  let level_of = |counts: &[u32]| {
    let mut l = alloc::vec![0u32; counts.len() * words];
    for (i, &c) in counts.iter().enumerate() {
      l[i * words] = c;
    }
    l
  };
  let flat = level_of(&[582_841; 3124]);
  let image: alloc::vec::Vec<u32> = (0..3124u32).map(|i| 90_000 + i * i * 6).collect();
  let image = level_of(&image);
  assert!(dust::level_is_flat(&flat));
  assert!(!dust::level_is_flat(&image));
  assert!(!dust::level_is_flat(&level_of(&[582_841; 10])), "a few texels are no field");
  let h = alloc::vec![0u32; dust::PYRAMID_HEADER_WORDS as usize];
  let mut host = DustViewHost::default();
  let seed = host.white;
  host.absorb(&h, &flat, 16, 1e-13);
  assert_eq!(host.white, seed, "flat: the exposure keeps its own");
  host.absorb(&h, &image, 16, 1e-13);
  assert_ne!(host.white, seed, "an image adapts it");
}

/// Static refinement (`dust::dust_refine_factor`): after `DUST_REFINE_AFTER_FRAMES` unchanged
/// frames and a readback of the image shown, one EWA level finer while that image's texels × 4
/// stay within the factor's budgets; refined readbacks never move the budget's bias; a changed
/// frame returns to it at once. The 67P Earth view's measured costs (5.4e8 at 5, ×1.7 a level).
#[test]
fn static_refinement_steps_within_its_factor() {
  use crate::scene::dust::{DUST_EWA_FRAME_BUDGET, DUST_EWA_TEXEL_UNIT, DUST_REFINE_AFTER_FRAMES, EwaStats};
  let header_at = |ewa_bias: u32, texels: u64| {
    let mut h = alloc::vec![0u32; dust::PYRAMID_HEADER_WORDS as usize];
    EwaStats::store_bias(ewa_bias, &mut h);
    h[dust::PYR_EWA_TEXELS as usize] = (texels / DUST_EWA_TEXEL_UNIT as u64) as u32;
    h[dust::PYR_SPLIT_CHILDREN as usize] = dust::DUST_SPLIT_BUDGET / 2;
    h
  };
  let level = alloc::vec![0u32; 16];
  let factor = 16;
  let wait = |host: &mut DustViewHost| -> bool {
    let mut stepped = false;
    for _ in 0..DUST_REFINE_AFTER_FRAMES {
      stepped |= host.refine_update(true, factor);
    }
    stepped
  };
  let mut host = DustViewHost::default();
  host.ewa_bias = 5;
  assert!(!wait(&mut host), "no readback of the shown image yet");
  host.absorb(&header_at(5, 544_900_000), &level, 1, 1.0);
  assert!(wait(&mut host));
  assert_eq!(host.refine, 1);
  // the refined image's readback: over the budget, yet the budget's bias stays
  host.absorb_refined = true;
  host.absorb(&header_at(4, 930_000_000), &level, 1, 1.0);
  assert_eq!(host.ewa_bias, 5);
  assert!(wait(&mut host), "9.3e8 × 4 ≤ 16 budgets");
  assert_eq!(host.refine, 2);
  host.absorb(&header_at(3, 1_600_000_000), &level, 1, 1.0);
  assert!(1_600_000_000u64 * 4 > factor as u64 * DUST_EWA_FRAME_BUDGET as u64);
  assert!(!wait(&mut host), "the next level would pass the factor");
  assert_eq!((host.refine, host.ewa_bias), (2, 5));
  // the view moves: the budget's bias at once
  assert!(!host.refine_update(false, factor));
  assert_eq!(host.refine, 0);
  // off: never refines
  host.absorb_refined = false;
  host.absorb(&header_at(5, 544_900_000), &level, 1, 1.0);
  for _ in 0..3 * DUST_REFINE_AFTER_FRAMES {
    assert!(!host.refine_update(true, 0));
  }
  assert_eq!(host.refine, 0);
}

/// the feedback's memory (round 8 step B): the 5000 km, 1 d/s, volume detail 1 oscillation
/// (`perf8e_5000km_vd1`: 97 M evaluations at bias 10 → 11, 7.3 M at 11 < budget / 8 → 10, every
/// four frames). After the raise the memory holds the measured step (13.3×), so 7.3 M at 11 is
/// refused (7.3 M × 13.3 > the budget); a drift down to 4 M (× 13.3 ≤ the budget) is lowered.
/// Without a measured step the default 4× keeps today's rule (`the_feedback_band_keeps_the_state`).
/// The same for the EWA side (1 km: 6.7e8 texels at 0 → 1, 2.7e7 at 1 → 0, a 25× step).
#[test]
fn the_feedback_refuses_a_lower_that_would_raise_again() {
  use crate::scene::dust::{DUST_EWA_FRAME_BUDGET, DUST_EWA_TEXEL_UNIT, DUST_VOLUME_FRAME_BUDGET, EwaStats};
  let header_at = |ewa_bias: u32, ewa_texels: u64, vol_bias: u32, evals: u32| {
    let mut h = alloc::vec![0u32; dust::PYRAMID_HEADER_WORDS as usize];
    EwaStats::store_bias(ewa_bias, &mut h);
    h[dust::PYR_EWA_TEXELS as usize] = (ewa_texels / DUST_EWA_TEXEL_UNIT as u64) as u32;
    let w = &mut h[dust::PYR_SPLIT_DEPTH as usize];
    *w = (*w & !(0xFF << 8)) | (vol_bias << 8);
    h[dust::PYR_VOLUME_EVALS as usize] = evals;
    h[dust::PYR_SPLIT_CHILDREN as usize] = dust::DUST_SPLIT_BUDGET / 2;
    h
  };
  let level = alloc::vec![0u32; 16];
  let vb = DUST_VOLUME_FRAME_BUDGET;
  let eb = DUST_EWA_FRAME_BUDGET as u64;
  let mid_ewa = eb / 4;
  let mut host = DustViewHost::default();
  host.volume_bias = 10;
  host.ewa_bias = 0;
  // the raise: 97 M at 10 (over the budget 67 M) → 11
  host.absorb(&header_at(0, mid_ewa, 10, 97_030_085), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 11);
  // 7.3 M at 11, below budget / 8 (8.4 M): refused, the measured step is 13.3×
  host.absorb(&header_at(0, mid_ewa, 11, 7_295_987), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 11, "a lower that would raise again");
  let q = host.fb_memory.volume_step_q8[11];
  assert!((q as f64 / 256.0 - 13.3).abs() < 0.1, "step ratio {}", q as f64 / 256.0);
  // the pose drifts: 4 M at 11 (× 13.3 = 53 M ≤ the budget) is lowered
  host.absorb(&header_at(0, mid_ewa, 11, 4_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 10);
  // a two-step raise records the per-step (geometric) ratio: 16× over two steps is 4× a step
  let mut host = DustViewHost::default();
  host.volume_bias = 3;
  host.absorb(&header_at(0, mid_ewa, 3, vb * 16), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 5);
  host.absorb(&header_at(0, mid_ewa, 5, vb), &level, 1, 1.0);
  assert_eq!(host.fb_memory.volume_step_q8[5], 4 * 256);
  // the EWA side: 6.7e8 at 0 → 1, then 2.7e7 at 1 (< budget / 8 = 3.4e7) is refused (× 25 > budget)
  let mut host = DustViewHost::default();
  host.ewa_bias = 0;
  host.volume_bias = 2;
  host.absorb(&header_at(0, 666_100_000, 2, vb / 4), &level, 1, 1.0);
  assert_eq!(host.ewa_bias, 1);
  host.absorb(&header_at(1, 27_160_000, 2, vb / 4), &level, 1, 1.0);
  assert_eq!(host.ewa_bias, 1, "the EWA lower that would raise again");
  // the memory follows the regime: stored with it and loaded back
  host.feedback_store(1);
  host.fb_memory = FeedbackSet::default();
  host.feedback_load(1);
  assert!(host.fb_memory.ewa_step_q8[1] > 0);
}

/// a feedback header with `levels` levels, the last 1 × 1 (the top-only predicate live), the EWA
/// side in its band and the split in its band (round 9 step B)
fn feedback_header(levels: u32, vol_bias: u32, evals: u32) -> alloc::vec::Vec<u32> {
  use crate::scene::dust::{DUST_EWA_FRAME_BUDGET, DUST_EWA_TEXEL_UNIT, EwaStats};
  let mut h = alloc::vec![0u32; dust::PYRAMID_HEADER_WORDS as usize];
  EwaStats::store_bias(0, &mut h);
  h[dust::PYR_EWA_TEXELS as usize] = (DUST_EWA_FRAME_BUDGET as u64 / 4 / DUST_EWA_TEXEL_UNIT as u64) as u32;
  let w = &mut h[dust::PYR_SPLIT_DEPTH as usize];
  *w = (*w & !(0xFF << 8)) | (vol_bias << 8);
  h[dust::PYR_VOLUME_EVALS as usize] = evals;
  h[dust::PYR_SPLIT_CHILDREN as usize] = dust::DUST_SPLIT_BUDGET / 2;
  h[dust::PYR_LEVELS as usize] = levels;
  for l in 0..levels {
    let t = (dust::PYR_TABLE + 3 * l) as usize;
    h[t + 1] = 512 >> l;
    h[t + 2] = 512 >> l;
  }
  h
}

/// with the haze knob below 16 a raise onto the top plateau lands on its lowest bias (`levels − 1`:
/// every bias from it on draws the same frame), at 16 the raise is today's (round 9 step B)
#[test]
fn the_feedback_raises_into_the_lowest_top_bias() {
  let level = alloc::vec![0u32; 16];
  let mut host = DustViewHost::default();
  host.volume_bias = 7;
  host.readback_haze = 1;
  // 300 M at 7 (4.5× the budget: two levels) would land at 9 = the plateau's first bias
  host.absorb(&feedback_header(10, 7, 300_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 9);
  // 3 000 M at 7 (three levels → 10): capped at 9
  let mut host = DustViewHost::default();
  host.volume_bias = 7;
  host.readback_haze = 1;
  host.absorb(&feedback_header(10, 7, 3_000_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 9);
  // on the plateau and still over the budget: stays (no raise past the lowest top bias)
  host.absorb(&feedback_header(10, 9, 76_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 9);
  // the knob at 16: today's raise
  let mut host = DustViewHost::default();
  host.volume_bias = 7;
  host.absorb(&feedback_header(10, 7, 3_000_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 10);
  host.absorb(&feedback_header(10, 10, 76_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 11);
}

/// with the haze engaged (the knob below 16 at a top-only bias) the feedback remembers the plateau's
/// edge (the raise's full ratio) and refuses a lower off the plateau while `work × edge > budget`;
/// a drift under it is lowered one step off the plateau only after the hold, and an in-band readback
/// off the plateau keeps the state (round 9 step B)
#[test]
fn the_feedback_holds_the_top_bias_while_the_haze_is_engaged() {
  use crate::scene::dust::DUST_VOLUME_FRAME_BUDGET;
  let level = alloc::vec![0u32; 16];
  let mut host = DustViewHost::default();
  host.volume_bias = 7;
  host.readback_haze = 1;
  host.absorb(&feedback_header(10, 7, 300_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 9);
  // the first top-only readback: 1 M (the haze cut 300×): the edge, the lower refused
  host.absorb(&feedback_header(10, 9, 1_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 9, "a lower that would raise again (edge 300×)");
  assert!((host.fb_memory.volume_edge_q8 as f64 / 256.0 - 300.0).abs() < 0.5, "edge {}", host.fb_memory.volume_edge_q8 as f64 / 256.0);
  assert_eq!(host.fb_memory.volume_step_q8[9], 0, "the edge is not a step");
  // the pose drifts: 0.1 M (× 300 = 30 M ≤ the budget) is held DUST_FEEDBACK_TOP_HOLD readbacks
  for k in 0..(DUST_FEEDBACK_TOP_HOLD - 2) {
    host.absorb(&feedback_header(10, 9, 100_000), &level, 1, 1.0);
    assert_eq!(host.volume_bias, 9, "held readback {k}");
  }
  host.absorb(&feedback_header(10, 9, 100_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 8, "one step off the plateau after the hold");
  assert_eq!(host.fb_memory.top_under, 0);
  // off the plateau, in the band: the state stays
  host.absorb(&feedback_header(10, 8, DUST_VOLUME_FRAME_BUDGET / 2), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 8);
  // without a measured edge the default 64× refuses 2 M (128 M > the budget), lowers 0.5 M
  let mut host = DustViewHost::default();
  host.volume_bias = 9;
  host.readback_haze = 4;
  for _ in 0..DUST_FEEDBACK_TOP_HOLD {
    host.absorb(&feedback_header(10, 9, 2_000_000), &level, 1, 1.0);
  }
  assert_eq!(host.volume_bias, 9, "the default edge refuses");
  for _ in 0..DUST_FEEDBACK_TOP_HOLD {
    host.absorb(&feedback_header(10, 9, 500_000), &level, 1, 1.0);
  }
  assert_eq!(host.volume_bias, 8, "the default edge lowers after the hold");
}

/// with the knob at 16 the memory keeps today's per-step (geometric) ratio and no edge: the raise
/// 7 → 9 by 300 M records √300 at 9, and 1 M at 9 is lowered as before (round 9 step B)
#[test]
fn the_feedback_memory_keeps_the_haze_edge_apart_from_the_steps() {
  let level = alloc::vec![0u32; 16];
  let mut host = DustViewHost::default();
  host.volume_bias = 7;
  host.absorb(&feedback_header(10, 7, 300_000_000), &level, 1, 1.0);
  assert_eq!(host.volume_bias, 9);
  host.absorb(&feedback_header(10, 9, 1_000_000), &level, 1, 1.0);
  assert_eq!(host.fb_memory.volume_edge_q8, 0);
  assert!((host.fb_memory.volume_step_q8[9] as f64 / 256.0 - 300f64.sqrt()).abs() < 0.1);
  assert_eq!(host.volume_bias, 8, "today's lower at 16");
  // the memory follows the regime
  host.fb_memory.volume_edge_q8 = 7 * 256;
  host.fb_memory.top_under = 3;
  host.feedback_store(1);
  host.fb_memory = FeedbackSet::default();
  host.feedback_load(1);
  assert_eq!((host.fb_memory.volume_edge_q8, host.fb_memory.top_under), (7 * 256, 3));
}

/// the build members of a frame are `min(step, length − cursor)` per tier slot and pair (a
/// finished cursor builds nothing); the shown entries are summed over every pair and tier
#[test]
fn cads_work_members_reads_the_cursors() {
  let mut st = alloc::vec![0u32; CADS_WORDS];
  let set = |st: &mut [u32], t: usize, k: usize, p: usize, w: [u32; 5]| {
    st[cads_tp(t, k, p)..cads_tp(t, k, p) + 5].copy_from_slice(&w)
  };
  // shown, build, pending, cursor, step
  set(&mut st, 0, 0, 0, [800, 640, 0, 0, 80]);
  set(&mut st, 0, 0, 1, [700, 640, 5, 600, 80]);
  set(&mut st, 0, 2, 1, [6400, 6400, 0, 6400, 100]);
  set(&mut st, 1, 1, 0, [1600, 1600, 0, 100, 100]);
  assert_eq!(cads_work_members(&st, 2, 3, 2), ([80 + 40, 100, 0], [800 + 700, 1600, 6400]));
  // a tier slot or class beyond the counts is not read
  assert_eq!(cads_work_members(&st, 1, 2, 2), ([120, 0, 0], [1500, 0, 0]));
}

// ── temporal reuse step 5, B2 (brief 07; records/budget-loop.md "Unfinished" 1): the GPU tests of
// the budget loop's host wiring — the frame kind restores the full-draw biases, the target probe
// splits the counters, the gate counters route through the frame, the loop moves its knobs in the
// ladder's order on real timings. Every frame is recorded in `frame.rs`'s order
// (`prepare_dust`): the loop's cadence knobs, the cadence begin with the salted fingerprint, the
// frame kind, the cadence propagate, classify, chunks, the builds class by class each followed by
// its target probe, the reprojection, the fresh splat, the timestamps the loop reads.

/// The scene of one budget-test view: a tier `[first, first + live)` of system `id`, its frame,
/// the viewport and the view flags (0: `DUST_VIEW_FLOW` would turn the cadence off)
struct BudgetScene {
  id: u64,
  capacity: u32,
  first: u32,
  live: u32,
  frame: DustFrame,
  w: u32,
  h: u32,
}

/// the view's host under its lock (`None` without the view); never held across a device call
fn with_view_host<R>(device: &Device, pe: crate::gpu::PresentationEngineHandle, f: impl FnOnce(&mut crate::gpu_backends::vulkan::device::dust::DustViewHost) -> R) -> Option<R> {
  let res = device.res.read();
  let view = res.dust_manager.as_ref()?.views.get(&pe)?;
  let mut host = view.host.lock();
  Some(f(&mut host))
}

/// One cadenced frame of `sc` seen by `view`, recorded as `frame.rs` does. `after_begin` runs after
/// the frame begin (the view exists then) and before the cadence's begin and the frame kind (the
/// tests' hooks into the host: the loop, a forced full-draw regime). Returns the plan.
fn budget_frame(
  device: &Device,
  pe: crate::gpu::PresentationEngineHandle,
  sc: &BudgetScene,
  view: cad::CadenceView,
  fingerprint: u64,
  after_begin: &dyn Fn(),
) -> CadenceGpuPlan {
  use crate::gpu_backends::vulkan::device::dust::{DUST_TS_BUILDS, DUST_TS_CADENCE, DUST_TS_CHUNKED, DUST_TS_PROPAGATED, DUST_TS_REPROJECTED, DUST_TS_SPLATS_DONE, DUST_TS_SPLATTED, PROBE_CLASSES};
  let flow = dust::DustFlowUniform { t_hi: 1.0e8, t_lo: 0.0, speed: 1.0 };
  let (id, capacity, first, live) = (sc.id, sc.capacity, sc.first, sc.live);
  let mvp = view.mvp([0.0; 3]);
  let e = view.eye_local([0.0; 3]);
  let tier = CadenceTierInput { key: 0, system: id, ring_base: 0, first_slot: first, capacity, live_count: live, emitted: 1, rewritten_from: live, anchor_m: [0.0; 3] };
  let mut plan = CadenceGpuPlan::default();
  let mut cmds = device
    .run_transient_commands(|cmd| {
      device.cmd_dust_pre_propagate_barrier(cmd);
      let v = device.cmd_dust_frame_begin(cmd, pe, [sc.w, sc.h], 0, flow)?;
      after_begin();
      device.cmd_dust_system_begin(id, 0, flow)?;
      let salt = device.dust_budget_apply_cadence(pe);
      let fm = cad::FrameMeta {
        view,
        t_now_s: sc.frame.t_now_s(),
        speed: 0.0,
        nucleus_m: [0.0; 3],
        flags: 0,
        fingerprint: fingerprint ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15),
        reused: false,
        unit_now: v.unit,
      };
      plan = device.cmd_dust_cadence_begin(cmd, pe, &fm, &[tier], v.d_max_m)?;
      let full = plan.reset.is_some() || plan.mode != Some(cad::CadenceMode::Run);
      device.dust_budget_frame_kind(cmd, pe, full);
      device.cmd_dust_timestamp(cmd, pe, DUST_TS_CADENCE);
      let (render, moments) = device.cmd_dust_cadence_propagate(cmd, pe, id, 0, capacity, first, live, &sc.frame, 1)?;
      device.cmd_dust_timestamp(cmd, pe, DUST_TS_PROPAGATED);
      device.cmd_dust_pre_splat_barrier(cmd);
      let pc = dust::DustSplatPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: live,
        flags: 0,
        exposure: 1.0,
        inv_unit: if v.unit > 0.0 { 1.0 / v.unit } else { 0.0 },
        color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
        albedo: 0.04,
        mvp,
        eye_local: [e[0], e[1], e[2], 30.0 * 86400.0],
      };
      device.cmd_dust_cadence_classify(cmd, pe, 0, render, moments, mvp, pc.eye_local)?;
      device.cmd_dust_cadence_chunks(cmd, pe, 0, render, moments)?;
      device.cmd_dust_timestamp(cmd, pe, DUST_TS_CHUNKED);
      let geom = plan.geom.clone();
      let mut draws: alloc::vec::Vec<CadenceBuildDraw> = plan
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
      draws.sort_by_key(|d| (d.k, d.p));
      for k in 0..PROBE_CLASSES {
        let (lo, hi) = (draws.partition_point(|d| d.k < k), draws.partition_point(|d| d.k <= k));
        device.cmd_dust_cadence_builds(cmd, pe, &draws[lo..hi])?;
        device.cmd_dust_budget_probe(cmd, pe, k);
      }
      device.cmd_dust_timestamp(cmd, pe, DUST_TS_BUILDS);
      device.cmd_dust_cadence_reproject(cmd, pe)?;
      device.cmd_dust_timestamp(cmd, pe, DUST_TS_REPROJECTED);
      device.cmd_dust_pre_splat_barrier(cmd);
      device.cmd_dust_splat(cmd, pe, id, 0, first, capacity, render, moments, pc)?;
      device.cmd_dust_timestamp(cmd, pe, DUST_TS_SPLATS_DONE);
      device.cmd_dust_timestamp(cmd, pe, DUST_TS_SPLATTED);
      device.cmd_dust_post_splat_barrier(cmd);
      device.cmd_dust_cadence_end(cmd, pe)?;
      device.cmd_dust_frame_end(cmd, pe)?;
      device.cmd_dust_system_end(id)?;
      Ok(())
    })
    .unwrap();
  cmds.cleanup(&device.device);
  plan
}

/// the small parity scene of the telemetry test: 1 024 clusters, the eye three cloud depths above
/// them stepping laterally `step(f)` metres per frame at 320 × 200
fn budget_small_scene(device: &Device, id: u64, capacity: u32, moments_buf: vk::Buffer) -> (BudgetScene, [f32; 3], [f64; 16], f64) {
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
  (BudgetScene { id, capacity, first, live, frame, w, h }, eye, proj, step)
}

fn budget_small_view(eye: [f32; 3], proj: [f64; 16], dx: f64, w: u32, h: u32) -> cad::CadenceView {
  cad::CadenceView {
    view_rot: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    proj,
    eye_m: [eye[0] as f64 + dx, eye[1] as f64, eye[2] as f64],
    units_per_m: 1.0,
    width: w,
    height: h,
  }
}

/// The header word [`dust::PYR_SPLIT_DEPTH`]'s feedback fields: split depth (bits 0–7), volume
/// bias (8–15), EWA bias (18–21)
fn budget_bias_bits(w: u32) -> (u32, u32, u32) {
  (w & 0xFF, (w >> 8) & 0xFF, (w >> 18) & 0xF)
}

/// A frame the cadence draws in full (a reset) begins with the full-draw regime's biases
/// (`Device::dust_budget_frame_kind`, brief 07's two regimes): after twelve cadenced frames, the
/// full regime is set by hand to values distinct from the cadenced regime's (the hook runs after
/// the frame begin, which wrote the cadenced biases into the header), the next frame changes the
/// fingerprint (a reset): its header carries the forced full-draw split depth / volume / EWA bias,
/// and the restore count rises by exactly one; the cadenced frame before it was a `Run` frame that
/// restored nothing. (With `AETHERVK_DUST_VOLUME_BIAS`, `_EWA_BIAS` or `_SPLIT_DEPTH` set the
/// pinned fields keep theirs: the test then skips those fields, printed.)
#[test]
fn gpu_dust_reset_frame_restores_the_full_draw_biases() {
  let id = 9040u64;
  with_device(id, |device, capacity, _render_buf, moments_buf| {
    let (sc, eye, proj, step) = budget_small_scene(device, id, capacity, moments_buf);
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 40);
    device.dust_cadence_enable(pe, Some(cad::CadenceConfig::full())).unwrap();
    let mut last = CadenceGpuPlan::default();
    for f in 0..12u32 {
      let before = with_view_host(device, pe, |h| h.restored).unwrap_or(0);
      last = budget_frame(device, pe, &sc, budget_small_view(eye, proj, f as f64 * step, sc.w, sc.h), 0, &|| {});
      let after = with_view_host(device, pe, |h| h.restored).unwrap_or(0);
      std::println!("[budget gpu] restore f{f}: mode {:?} reset {:?}, restored {before} → {after}", last.mode, last.reset);
    }
    let restored_before = with_view_host(device, pe, |h| h.restored).unwrap();
    assert_eq!(last.mode, Some(cad::CadenceMode::Run), "frame 11 must be a cadenced (Run) frame");
    // the forced full-draw regime: every field distinct from the cadenced regime's
    let cad_now = with_view_host(device, pe, |h| h.fb_cad.unwrap_or(h.fb_full)).unwrap();
    let forced = crate::gpu_backends::vulkan::device::dust::FeedbackSet {
      split_depth: if cad_now.split_depth == dust::DUST_SPLIT_DEPTH_MAX { dust::DUST_SPLIT_DEPTH_MAX - 1 } else { dust::DUST_SPLIT_DEPTH_MAX },
      volume_bias: (cad_now.volume_bias + 3) % (dust::DUST_VOLUME_BIAS_MAX + 1),
      ewa_bias: (cad_now.ewa_bias + 2) % (dust::DUST_EWA_BIAS_MAX + 1),
      ..cad_now
    };
    let plan = budget_frame(device, pe, &sc, budget_small_view(eye, proj, 12.0 * step, sc.w, sc.h), 1, &|| {
      with_view_host(device, pe, |h| h.fb_full = forced);
    });
    let restored_after = with_view_host(device, pe, |h| h.restored).unwrap();
    let pyr = device.download_dust_pyramid(pe).unwrap();
    let got = budget_bias_bits(pyr[dust::PYR_SPLIT_DEPTH as usize]);
    let pinned = |k: &str| aethervk_oshal_rlib::os::env::var(k).is_some();
    std::println!(
      "[budget gpu] reset frame: reset {:?} mode {:?}; cadenced regime (depth, volume, EWA) ({}, {}, {}), forced full ({}, {}, {}), header ({}, {}, {}); restored {restored_before} → {restored_after}",
      plan.reset, plan.mode, cad_now.split_depth, cad_now.volume_bias, cad_now.ewa_bias, forced.split_depth, forced.volume_bias, forced.ewa_bias, got.0, got.1, got.2
    );
    assert_eq!(plan.reset, Some(cad::CadenceReset::Fingerprint), "the changed fingerprint resets the cadence");
    if !pinned("AETHERVK_DUST_SPLIT_DEPTH") {
      assert_eq!(got.0, forced.split_depth, "the reset frame's split depth is the full regime's");
    }
    if !pinned("AETHERVK_DUST_VOLUME_BIAS") {
      assert_eq!(got.1, forced.volume_bias, "the reset frame's volume bias is the full regime's");
    }
    if !pinned("AETHERVK_DUST_EWA_BIAS") {
      assert_eq!(got.2, forced.ewa_bias, "the reset frame's EWA bias is the full regime's");
    }
    assert_eq!(restored_after, restored_before + 1, "one restore for the one reset frame");
    device.dust_cadence_enable(pe, None).unwrap();
  });
}

/// The target probe splits a cadenced frame's counters (`cmd_dust_budget_probe`, `TargetCounters`)
/// and the gate counters run through the frame (the builds' headers routed in and out,
/// `CAD_ROUTED_WORDS`): over 64 frames of the stepping parity scene (one class of 8 frames in two phases), every readback's class
/// probes are cumulative (no class's difference wraps, Σ by class = builds), the builds never
/// exceed the frame's total (fresh = total − builds does not wrap), and at least one cadenced
/// readback has build work on a counter (non-vacuity: with the builds counting into their own
/// headers only, the probes would read 0 and fresh would equal the total). Printed per readback:
/// the regime, builds, fresh, by class.
#[test]
fn gpu_dust_budget_probe_splits_the_counters() {
  let id = 9041u64;
  with_device(id, |device, capacity, _render_buf, moments_buf| {
    let (sc, eye, proj, step) = budget_small_scene(device, id, capacity, moments_buf);
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 41);
    // one class of 8 frames in two phases (perf9_T fix 1): with `full()` a near-still eye folds
    // every cluster into class 16 / 64, first classified at frame 16..64, and the builds came too late
    let cfg = cad::CadenceConfig { classes: alloc::vec![8], phases: 2, eps_px: 1.0, margin_frac: 0.2, retain: true, handover: true, light_tol: 0.0 };
    device.dust_cadence_enable(pe, Some(cfg)).unwrap();
    let mut with_builds = 0usize;
    let mut readbacks = 0usize;
    let mut built_frames = 0usize;
    for f in 0..64u32 {
      let plan = budget_frame(device, pe, &sc, budget_small_view(eye, proj, f as f64 * step, sc.w, sc.h), 0, &|| {});
      built_frames += (!plan.builds.is_empty()) as usize;
      let (regime, targets) = with_view_host(device, pe, |h| (h.last_regime, h.targets)).unwrap();
      let Some(t) = targets else { continue };
      readbacks += 1;
      std::println!("[budget gpu] probe f{f}: regime {regime}, builds {:?}, fresh {:?}, by class {:?}", t.builds, t.fresh, t.by_class);
      for i in 0..3 {
        let sum: u64 = t.by_class.iter().map(|c| c[i] as u64).sum();
        assert!(t.by_class.iter().all(|c| c[i] < 1 << 31), "f{f} counter {i}: a class probe went backwards (wrapped): {:?}", t.by_class);
        assert_eq!(sum, t.builds[i] as u64, "f{f} counter {i}: Σ by class ≠ builds");
        assert!(t.fresh[i] < 1 << 31, "f{f} counter {i}: the builds {} exceed the frame's total (fresh wrapped to {})", t.builds[i], t.fresh[i]);
      }
      with_builds += (regime == 1 && t.builds.iter().any(|&b| b > 0)) as usize;
    }
    std::println!("[budget gpu] probe: {readbacks} readbacks with a probe, {with_builds} cadenced with build work; {built_frames} frames recorded builds");
    assert!(built_frames > 0, "the scene never built a class image");
    assert!(readbacks > 0, "no readback carried the target probe");
    assert!(with_builds > 0, "no cadenced readback counted build work: the builds' counters did not reach the frame");
    device.dust_cadence_enable(pe, None).unwrap();
  });
}

/// The budget loop on real timings: a full ring (every slot emitted, mixed ages, as
/// `dust_frame_fits_the_budget`) seen from inside the dust (200 m from the nucleus) at 1280 × 800,
/// paused (Δt 0), the eye orbiting the nucleus `orbit_deg` per frame; the cadence with the loop (set
/// on the view's host after its first frame, as `AETHERVK_DUST_CADENCE=1` does, without the env).
/// Every frame the loop's knobs obey the ladder's order (both ways): lod above its default only
/// with ε at its ceiling, a budget shift only with lod at its ceiling, the fallback only at the
/// last budget shift, an N level only under the fallback. Returns the last drawn frame's GPU total
/// and the telemetry; prints the telemetry's lines every 25 frames.
fn budget_loop_run(tag: &str, id: u64, pe_off: u64, orbit_deg: f64, frames: u32) -> (f64, crate::gpu_backends::vulkan::device::dust::DustBudgetTelemetry) {
  use crate::scene::dust::cadence::{BudgetLoop, BudgetPins, DUST_BUDGET_SHIFT_MAX, DUST_CADENCE_EPS_MAX};
  use crate::scene::dust::{DUST_LOD_PX_DEFAULT, DUST_LOD_PX_MAX};
  let mut out = (f64::NAN, crate::gpu_backends::vulkan::device::dust::DustBudgetTelemetry::default());
  with_device(id, |device, capacity, _, _| {
    if device.query_result.graphics_timestamp_bits == 0 {
      std::println!("[budget gpu] {tag}: the graphics queue has no timestamps: skipped");
      return;
    }
    let per = 4096u32;
    let (mut emitted, mut seq) = (0u32, 0u64);
    while emitted < capacity {
      let n = per.min(capacity - emitted);
      let batch = test_batch(n, 3600.0 * (1.0 + (seq % 7) as f64), emitted, capacity, 0xB0D6 ^ seq as u32);
      emit(device, id, &batch, seq);
      emitted += n;
      seq += 1;
    }
    let sc = BudgetScene { id, capacity, first: 0, live: capacity, frame: frame_at(5.0), w: 1280, h: 800 };
    let rel = camera([0.0; 3], sc.w, sc.h).1;
    let mut proj = [0.0f64; 16];
    for i in 0..16 {
      proj[i] = rel[i] as f64;
    }
    let d = 200.0f64;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - pe_off);
    device.dust_cadence_enable(pe, Some(cad::CadenceConfig::full())).unwrap();
    let budget_ms = crate::scene::dust::cadence::DUST_BUDGET_MS_DEFAULT;
    let mut moves_seen = 0u32;
    for f in 0..frames {
      let th = orbit_deg * f as f64;
      let (s, c) = th.to_radians().sin_cos();
      let view = cad::CadenceView { view_rot: yaw4(-th), proj, eye_m: [d * s, 0.0, d * c], units_per_m: 1.0, width: sc.w, height: sc.h };
      budget_frame(device, pe, &sc, view, 0, &|| {
        with_view_host(device, pe, |h| {
          if h.budget.is_none() {
            h.budget = Some(BudgetLoop::new(budget_ms, BudgetPins::default()));
          }
        });
      });
      let tel = device.dust_budget_telemetry(pe).expect("the view's telemetry");
      if let Some(k) = tel.knobs {
        let at = |what: &str| alloc::format!("{tag} f{f}: {what} out of the ladder's order: {k:?}");
        assert!(k.lod_px <= DUST_LOD_PX_DEFAULT || k.eps_px >= DUST_CADENCE_EPS_MAX, "{}", at("lod"));
        assert!(k.budget_shift == 0 || k.lod_px >= DUST_LOD_PX_MAX, "{}", at("budget shift"));
        assert!(!k.fallback || k.budget_shift == DUST_BUDGET_SHIFT_MAX, "{}", at("fallback"));
        assert!(k.n_level == 0 || k.fallback, "{}", at("N level"));
      }
      if tel.moves != moves_seen || f % 25 == 0 || f + 1 == frames {
        moves_seen = tel.moves;
        for l in tel.lines() {
          std::println!("[budget gpu] {tag} f{f}: {l}");
        }
      }
      out.1 = tel;
    }
    let t = device.dust_gpu_timings_drawn(pe).expect("no dust GPU timings");
    out.0 = t.total_ms;
    std::println!(
      "[budget gpu] {tag}: {capacity} clusters at {}x{}, {frames} frames: total {:.2} ms (propagate {:.2}, cadence {:.2}, classify {:.2}, builds {:.2}, reproject {:.2}, fresh splat {:.2}, measure {:.2}) against {budget_ms} ms; loop {} rung {}/{} moves {}, knobs {:?}, K {:?}",
      sc.w, sc.h, t.total_ms, t.propagate_ms, t.cadence_ms, t.classify_ms, t.builds_ms, t.reproject_ms, t.splat_only_ms, t.measure_ms,
      out.1.state, out.1.rung, out.1.rungs, out.1.moves, out.1.knobs, out.1.k_target
    );
    device.dust_cadence_enable(pe, None).unwrap();
  });
  out
}

/// The budget loop's verdict run (brief 07): paused and orbiting (10°/s at 60 fps), 240 frames
/// each, the ladder's order asserted every frame (`budget_loop_run`); a frame over the budget at
/// the end must have moved the loop (moves ≥ 1). The verdict line per case: fits, or over at the
/// state the loop reached (the "saturated" state is the brief's verdict: the next plan is the
/// per-packet price, not more knobs).
#[test]
fn gpu_dust_budget_loop_keeps_the_ladder_order() {
  for (tag, id, pe_off, orbit) in [("paused", 9042u64, 42u64, 0.0f64), ("orbit 10°/s", 9043, 43, 10.0 / 60.0)] {
    let (ms, tel) = budget_loop_run(tag, id, pe_off, orbit, 240);
    if ms.is_nan() {
      continue;
    }
    let over = ms > tel.budget_ms;
    std::println!(
      "[budget gpu] verdict {tag}: {:.2} ms against {} ms: {}",
      ms,
      tel.budget_ms,
      if !over { "fits".into() } else { alloc::format!("over, loop {} after {} moves", tel.state, tel.moves) }
    );
    assert!(tel.knobs.is_some(), "{tag}: the budget loop is not running");
    assert!(!over || tel.moves >= 1, "{tag}: {ms:.2} ms over the budget and the loop never moved");
  }
}

/// The dust frame fits `test_dust::DUST_FRAME_BUDGET_MS` with the cadence and the
/// budget loop, paused and orbiting at 1280 × 800 (brief 07's un-ignored case; 600 frames each:
/// the ladder's ≤ 36 moves × 7 frames and two cycles of the longest class). Ignored like
/// `dust_frame_fits_the_budget`: the loop saturates at 172–179 ms (2026-10-19, CP2 budget runs).
#[test]
#[ignore = "the cadenced dust frame saturates the budget loop at 172–179 ms against 8 ms (2026-10-19): the per-packet price is the next plan"]
fn dust_frame_fits_the_budget_with_the_cadence() {
  for (tag, id, pe_off, orbit) in [("paused", 9044u64, 44u64, 0.0f64), ("orbit 10°/s", 9045, 45, 10.0 / 60.0)] {
    let (ms, tel) = budget_loop_run(tag, id, pe_off, orbit, 600);
    if ms.is_nan() {
      continue;
    }
    assert!(ms <= tel.budget_ms, "{tag}: the cadenced dust frame costs {ms:.2} ms, over the {} ms budget (loop {})", tel.budget_ms, tel.state);
  }
}

/// The in-frame gates per target (brief 07, decided 2026-10-18 in records/budget-loop.md): the hard
/// gates read the frame's running counters across both targets because the build draws' headers
/// route the gate counters frame → image → frame. The routed ranges hold every probed counter
/// (split children, EWA texels, volume evaluations) and the gate word (`PYR_SPLIT_DEPTH`, its
/// gates-off bit) and none of the image's own words (geometry, level table, τ_max, level mask,
/// unit, owner address).
#[test]
fn the_gate_counters_route_through_the_frame() {
  let routed = |w: u32| CAD_ROUTED_WORDS.iter().any(|&(lo, hi)| (lo..hi).contains(&w));
  for w in crate::gpu_backends::vulkan::device::dust::DUST_PROBE_HEADER_WORDS {
    assert!(routed(w), "probed counter word {w} is not routed through the frame");
  }
  assert!(routed(dust::PYR_SPLIT_DEPTH), "the gate word is not routed");
  for w in [dust::PYR_TAU_MAX, dust::PYR_LEVEL_MASK, dust::PYR_UNIT, 58] {
    assert!(!routed(w), "the image's own word {w} is routed");
  }
  for w in (0..=2).chain(16..=57) {
    assert!(!routed(w), "geometry word {w} is routed");
  }
}
