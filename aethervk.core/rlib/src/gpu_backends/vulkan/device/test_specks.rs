//! GPU parity of the physical specks (`dust_specks.comp`) against the CPU mirror
//! (`dust::specks::specks_tier`), on the splat's pyramid. Needs a Vulkan device and the compiled
//! shaders in `assets/sim`. The setup helpers are minimal copies of `test_dust.rs`'s (private
//! there).
use super::{
  test_utils::{setup_assets_dir, setup_render_frontend_for_tests},
  *,
};
use crate::scene::dust::{
  self, AU_M, DustBatch, DustFrame, SUN_MU_M3_S2, SizeDistribution, V3, kepler, specks,
};

/// 67P-like comet at 5.5 AU (as `test_dust.rs`)
fn comet_state() -> (V3, V3) {
  let r = [-1.924 * AU_M, -5.164 * AU_M, -0.204 * AU_M];
  let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
  let vc = (SUN_MU_M3_S2 / rn).sqrt() * 0.9;
  let t = [5.164f64, -1.924, 0.3];
  let tn = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
  (r, [t[0] * vc / tn, t[1] * vc / tn, t[2] * vc / tn])
}

const T_START: f64 = 4.0e8;

fn test_batch(count: u32, dur: f64, first_index: u32, capacity: u32) -> DustBatch {
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
  b.seed = 0xC0FFEE;
  b
}

fn frame_after(days: f64) -> DustFrame {
  let t_now = T_START + days * 86400.0;
  let (rc0, vc0) = comet_state();
  let (rc, _) = kepler::propagate_f64(rc0, vc0, SUN_MU_M3_S2, t_now - T_START);
  DustFrame::new(rc, t_now, [0.0, 0.0, 0.0, 1.0], 1.0e9)
}

/// Copies `bytes` bytes of a device buffer into host memory (synchronous, graphics queue).
#[allow(dead_code)] // (the tier records go through `test_dust::read_tier_records`, slot indexed)
fn read_back(device: &Device, src: vk::Buffer, bytes: u64) -> alloc::vec::Vec<u8> {
  let allocator = device.res.read().allocator.allocator.as_allocator_view();
  let info = vk::BufferCreateInfo::default()
    .size(bytes)
    .usage(vk::BufferUsageFlags::TRANSFER_DST);
  let mut alloc_info = vk_mem::AllocationCreateInfo::default();
  alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
  alloc_info.flags =
    vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;
  let (dst, mut dst_alloc, dst_info) =
    unsafe { allocator.create_buffer_get_info(&info, &alloc_info) }.unwrap();
  let mut res = device
    .run_transient_commands(|cmd| {
      let b = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ);
      let dep = vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&b));
      unsafe {
        device.device.synchronization2.cmd_pipeline_barrier2(cmd, &dep);
        device
          .device
          .cmd_copy_buffer(cmd, src, dst, &[vk::BufferCopy::default().size(bytes)]);
      }
      Ok(())
    })
    .unwrap();
  res.cleanup(&device.device);
  allocator.invalidate_allocation(&dst_alloc, 0, bytes).unwrap();
  let out =
    unsafe { core::slice::from_raw_parts(dst_info.mapped_data.cast::<u8>(), bytes as usize) }
      .to_vec();
  unsafe { allocator.destroy_buffer(dst, &mut dst_alloc) };
  out
}

/// `(capacity, render buffer, moments buffer)` of system `id`
fn buffers(device: &Device, id: u64) -> (u32, vk::Buffer, vk::Buffer) {
  let res = device.res.read();
  let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
  (
    sys.capacity,
    sys.render.as_ref().unwrap().buffer,
    sys.moments.as_ref().unwrap().buffer,
  )
}

/// A perspective camera at `eye` looking along −z, vertical field 60°, `w × h` px: the absolute
/// mvp (local metres → clip, the splat's) and the eye-relative one (the specks').
fn camera(eye: [f32; 3], w: u32, h: u32) -> ([f32; 16], [f32; 16]) {
  let tan_h = 30.0f32.to_radians().tan();
  let (a, b) = (h as f32 / (w as f32 * tan_h), 1.0 / tan_h);
  let mut rel = [0.0f32; 16];
  rel[0] = a; // x_clip = a·dx
  rel[5] = b; // y_clip = b·dy
  rel[10] = -1e-3; // depth (unused by the dust)
  rel[11] = -1.0; // w = −dz
  let mut abs = rel;
  abs[12] = -a * eye[0];
  abs[13] = -b * eye[1];
  abs[14] = 1e-3 * eye[2];
  abs[15] = eye[2];
  (abs, rel)
}

/// The speck pass on the GPU equals the CPU mirror on the same moments and header: a 60°
/// perspective camera at the mean of a five-day-old cluster of a 1024-cluster batch (inside the
/// cloud: hundreds of kilometres wide, 5·10⁹ grains per cluster), the shell radius fixed so the
/// ball holds ≈ 2·10⁵ grains of the CPU's own number density at the eye, 320×200 px.
///
/// Expected: the trees are identical where the quantised counts are (13 bits: a count whose
/// float evaluation differs by ulps changes it on ~10⁻³ of the bins) and the leaves differ by
/// the mirrors' `log`/`exp`/`sqrt` ulps, so a speck flips across the ball or the eye plane only
/// within ~10⁻⁶ of the radius; ≈ 10 % of the in-front specks are on screen (the frustum's
/// ~1.4 sr of the 2π in front): ~10⁴ drawn. Tolerances: the speck words (drawn, counts, nodes)
/// within 2 % + a few, `n_eye` within 0.1 % (a float sum over 16 384 bins), no overflow, and the
/// level-0 texels under the splat parity's rules (Σ counts 0.1 %, 3×3 τ sums 5 % + 6 on all but
/// 1 % of the neighbourhoods, the hue ratio where τ ≥ 64).
#[test]
fn gpu_dust_specks_match_reference() {
  setup_assets_dir();
  assert!(
    !crate::gpu_backends::vulkan::physics::is_cpu_particles_mode(),
    "GPU parity tests need GPU particle mode"
  );
  let id = 9021u64;
  let (_pool, render_frontend, handle, _) = setup_render_frontend_for_tests(false);
  render_frontend
    .with_device(handle, |dyn_device| {
      let device: &Device = dyn_device.as_any().downcast_ref().unwrap();
      device.create_particle_system(id)?;
      run_specks_parity(device, id);
      unsafe { device.device.device_wait_idle() }.unwrap();
      device.discard_particle_system(id, 0, 0)?;
      GpuResult::Ok(())
    })
    .unwrap();
}

fn run_specks_parity(device: &Device, id: u64) {
  let (capacity, render_buf, moments_buf) = buffers(device, id);
  let batch = test_batch(1024, 3600.0, 3, capacity);
  let mut res = device
    .run_transient_compute_commands(|cmd| device.cmd_dust_emit(cmd, id, 0, &batch, 0))
    .unwrap();
  res.cleanup(&device.device);
  let first_slot = batch.first_index & batch.ring_mask;
  let frame = frame_after(5.0);
  let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 21);
  let (w, h) = (320u32, 200u32);
  let flow = dust::DustFlowUniform {
    t_hi: 1.0e8,
    t_lo: 0.0,
    speed: 1.0,
  };
  let ttl = 30.0 * 86400.0f32;
  let s_ref = 50.0f32;
  let q = 3.5f32;
  // the slot-indexed GPU records in the CPU mirror's compact ring order
  let fetch = |device: &Device| {
    let render: alloc::vec::Vec<dust::DustRenderCluster> = super::test_dust::read_tier_records(
      device,
      render_buf,
      0,
      first_slot,
      batch.count,
      capacity - 1,
    );
    let moments: alloc::vec::Vec<dust::DustMoments> = super::test_dust::read_tier_records(
      device,
      moments_buf,
      0,
      first_slot,
      batch.count,
      capacity - 1,
    );
    (render, moments)
  };
  // one frame of moments (no splat) to place the camera
  let mut cmds = device
    .run_transient_commands(|cmd| {
      device.cmd_dust_pre_propagate_barrier(cmd);
      device.cmd_dust_frame_begin(cmd, pe, [w, h], 0, flow)?;
      device.cmd_dust_system_begin(id, 0, flow)?;
      device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
      device.cmd_dust_pre_splat_barrier(cmd);
      device.cmd_dust_post_splat_barrier(cmd);
      device.cmd_dust_frame_end(cmd, pe)?;
      device.cmd_dust_system_end(id)?;
      Ok(())
    })
    .unwrap();
  cmds.cleanup(&device.device);
  let (render0, moments0) = fetch(device);
  let mut live: alloc::vec::Vec<usize> =
    (0..moments0.len()).filter(|&i| moments0[i].flux() > 0.0).collect();
  assert!(live.len() > 500, "{} live clusters", live.len());
  live.sort_by(|&a, &b| moments0[a].mean()[0].total_cmp(&moments0[b].mean()[0]));
  let eye = moments0[live[live.len() / 2]].mean();
  let n_eye = specks::number_density_at(&moments0, &render0, batch.count, eye, s_ref, q, ttl, 0.0);
  let target = 2.0e5f64;
  let d_max = ((target / (4.0 / 3.0 * core::f64::consts::PI * n_eye.max(1e-30))).cbrt())
    .clamp(1.0, 1.0e4) as f32;
  std::println!(
    "[specks gpu] eye at {eye:?} m, n_eye {n_eye:.3e} m⁻³, shell {d_max:.2} m (≈ {:.2e} grains in the ball)",
    n_eye * 4.0 / 3.0 * core::f64::consts::PI * (d_max as f64).powi(3)
  );
  let (mvp_abs, mvp_rel) = camera(eye, w, h);
  let flags = dust::DUST_VIEW_FLOW;
  let mut splat_pc = dust::DustSplatPushConstants {
    moments: 0,
    render: 0,
    pyramid: 0,
    live_count: batch.count,
    flags,
    exposure: 1.0,
    inv_unit: 0.0,
    color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
    albedo: 0.04,
    mvp: mvp_abs,
    eye_local: [eye[0], eye[1], eye[2], ttl],
  };
  let mut speck_pc = specks::DustSpecksPushConstants {
    moments: 0,
    render: 0,
    pyramid: 0,
    live_count: batch.count,
    size_fade: specks::pack_size_fade(s_ref, 0.0),
    exposure: 1.0,
    inv_unit: 0.0,
    color: specks::pack_color_q(dust::pack_color([1.0, 0.6, 0.2, 1.0]), q),
    albedo: 0.04,
    mvp: mvp_rel,
    eye_local: [eye[0], eye[1], eye[2], ttl],
  };
  // the shell from the work bounds: the frame's node account and the per-walk cap are what the
  // pass may spend (`DUST_SPECKS_FRAME_NODES`, `DUST_SPECKS_MAX_NODES`); a 4-D leaf tree (the
  // chord as a cube axis) needs more nodes per speck than the 3-D Gaussian of the first version,
  // so the 2·10⁵-grain shell is moved inward ×0.8 while the GPU frame overflows — the production
  // rule (`ShellUpdate`) — and the nodes used are printed against the budget
  let mut d_max = d_max;
  let mut tries = 0;
  let gpu = loop {
    device.dust_set_shell_override(pe, Some(d_max));
    // the unit follows the view after DUST_READBACK_SLOTS frames (as the splat parity)
    for _ in 0..(super::dust::DUST_READBACK_SLOTS + 2) {
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], flags, flow)?;
          splat_pc.inv_unit = 1.0 / view.unit;
          speck_pc.inv_unit = 1.0 / view.unit;
          device.cmd_dust_system_begin(id, flags, flow)?;
          let (render, moments) =
            device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
          device.cmd_dust_pre_splat_barrier(cmd);
          // the specks before the splat: their walk mask decides the splat's hand-over
          device.cmd_dust_specks(cmd, pe, id, 0, first_slot, capacity, render, moments, speck_pc)?;
          device.cmd_dust_pre_splat_barrier(cmd);
          device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, splat_pc)?;
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
    }
    let g = device.download_dust_pyramid(pe).unwrap();
    std::println!(
      "[specks gpu] shell {d_max:.3} m: GPU nodes {} of the frame budget {} (per walk ≤ {}), drawn {}, overflow {}",
      g[specks::PYR_SPECKS_NODES as usize],
      specks::DUST_SPECKS_FRAME_NODES,
      specks::DUST_SPECKS_MAX_NODES,
      g[specks::PYR_SPECKS as usize],
      g[specks::PYR_SPECKS_OVERFLOW as usize]
    );
    if g[specks::PYR_SPECKS_OVERFLOW as usize] == 0 || tries >= 16 {
      break g;
    }
    d_max *= specks::DUST_SPECKS_SHRINK;
    tries += 1;
  };
  let layout = device.dust_view_layout(pe).unwrap();
  let (render, mut moments) = fetch(device);
  // the CPU reference on the GPU's moments and the GPU frame's host-written words
  let mut cpu = alloc::vec![0u32; layout.total_words as usize];
  cpu[..dust::PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
  for word in [
    dust::PYR_FLAGS,
    dust::PYR_T_HI,
    dust::PYR_T_LO,
    dust::PYR_SPLIT_DEPTH,
    specks::PYR_SPECKS_DMAX,
    specks::PYR_SPECKS_BUDGET,
  ] {
    cpu[word as usize] = gpu[word as usize];
  }
  assert_eq!(f32::from_bits(gpu[specks::PYR_SPECKS_DMAX as usize]), d_max);
  dust::splat_tier(&moments, &render, &splat_pc, &flow, 0, &layout, &mut cpu);
  let splat_only = cpu.clone();
  let stats = specks::specks_tier(&mut moments, &render, &speck_pc, &flow, &layout, &mut cpu);
  let word = |p: &[u32], k: u32| p[k as usize];
  let (dg, dc) = (
    word(&gpu, specks::PYR_SPECKS),
    word(&cpu, specks::PYR_SPECKS),
  );
  let (cg, cc) = (
    word(&gpu, specks::PYR_SPECKS_COUNTS),
    word(&cpu, specks::PYR_SPECKS_COUNTS),
  );
  let (ng, nc) = (
    word(&gpu, specks::PYR_SPECKS_NODES),
    word(&cpu, specks::PYR_SPECKS_NODES),
  );
  let n_eye_of = |p: &[u32]| specks::ShellUpdate::from_header(p, 0.0, 0.0).n_eye_m3;
  let (eg, ec) = (n_eye_of(&gpu), n_eye_of(&cpu));
  let level0_sum = |p: &[u32]| -> u64 {
    let (off, lw, lh) = layout.levels[0];
    (0..(lw * lh) as usize)
      .map(|t| p[off as usize + t * dust::PYRAMID_TEXEL_WORDS as usize] as u64)
      .sum()
  };
  let (l0_splat, l0_cpu) = (level0_sum(&splat_only), level0_sum(&cpu));
  std::println!(
    "[specks gpu] drawn gpu {dg} cpu {dc} (enumerated {}, overflow gpu {} cpu {}), counts gpu {cg} cpu {cc}, nodes gpu {ng} cpu {nc}, n_eye gpu {eg:.4e} cpu {ec:.4e}; level 0: splat {l0_splat} + specks {} counts",
    stats.enumerated,
    word(&gpu, specks::PYR_SPECKS_OVERFLOW),
    word(&cpu, specks::PYR_SPECKS_OVERFLOW),
    l0_cpu - l0_splat
  );
  assert!(dc >= 1000, "only {dc} specks drawn on the CPU");
  assert_eq!(word(&gpu, specks::PYR_SPECKS_OVERFLOW), 0, "GPU overflow");
  assert_eq!(word(&cpu, specks::PYR_SPECKS_OVERFLOW), 0, "CPU overflow");
  assert!(
    dg.abs_diff(dc) as f32 <= 8.0 + 0.02 * dc as f32,
    "specks drawn gpu {dg} cpu {dc}"
  );
  assert!(
    cg.abs_diff(cc) as f32 <= 64.0 + 0.02 * cc as f32,
    "speck counts gpu {cg} cpu {cc}"
  );
  assert!(
    ng.abs_diff(nc) as f32 <= 512.0 + 0.02 * nc as f32,
    "speck nodes gpu {ng} cpu {nc}"
  );
  assert!((eg / ec - 1.0).abs() < 1e-3, "n_eye gpu {eg} cpu {ec}");
  assert!(l0_cpu > l0_splat, "the specks must deposit on level 0");
  // the whole pyramid's counts (the splat's own parity rule) and the level-0 texels
  let total = |p: &[u32]| -> u64 {
    layout
      .levels
      .iter()
      .map(|&(off, lw, lh)| {
        (0..(lw * lh) as usize)
          .map(|t| p[off as usize + t * dust::PYRAMID_TEXEL_WORDS as usize] as u64)
          .sum::<u64>()
      })
      .sum()
  };
  let (sg, sc) = (total(&gpu), total(&cpu));
  assert!(
    (sg as f64 / sc as f64 - 1.0).abs() < 1e-3,
    "Σ counts gpu {sg} cpu {sc}"
  );
  let (off, lw, lh) = layout.levels[0];
  let at = |p: &[u32], x: i64, y: i64| -> u64 {
    if x < 0 || y < 0 || x >= lw as i64 || y >= lh as i64 {
      return 0;
    }
    p[off as usize + (y as usize * lw as usize + x as usize) * dust::PYRAMID_TEXEL_WORDS as usize]
      as u64
  };
  // diagnostics: the CPU's specks alone (a pyramid with the same header, no splat) and the
  // number of CPU specks landing on each level-0 texel (the walk replayed with the pass's view)
  let mut specks_only = alloc::vec![0u32; layout.total_words as usize];
  specks_only[..dust::PYRAMID_HEADER_WORDS as usize]
    .copy_from_slice(&cpu[..dust::PYRAMID_HEADER_WORDS as usize]);
  for k in [
    specks::PYR_SPECKS,
    specks::PYR_SPECKS_OVERFLOW,
    specks::PYR_N_EYE,
    specks::PYR_N_EYE_HI,
    specks::PYR_SPECKS_COUNTS,
    specks::PYR_SPECKS_NODES,
  ] {
    specks_only[k as usize] = 0;
  }
  specks::specks_tier(
    &mut moments,
    &render,
    &speck_pc,
    &flow,
    &layout,
    &mut specks_only,
  );
  // every CPU speck in front: screen position, depth, the counts it deposits (its own
  // `scatter_speck` on a scratch pyramid) and the floor `speck_floor` (printed for information)
  let mut landing = alloc::vec![0u32; (lw * lh) as usize];
  let mut recs: alloc::vec::Vec<SpeckRec> = alloc::vec::Vec::new();
  // every CPU speck's offset from the eye by hash, for the GPU's trace
  // (with its coordinates and its cluster: the per-axis split and the map's own error)
  let mut cpu_dp: std::collections::HashMap<u32, ([f32; 3], [f32; 3], [f32; 3], specks::SpeckCluster)> =
    std::collections::HashMap::new();
  let f_px = (mvp_rel[0].abs() * 0.5 * w as f32).max(mvp_rel[5].abs() * 0.5 * h as f32);
  {
    let view = specks::SpeckView::new(&speck_pc, &layout);
    let mut scratch = alloc::vec![0u32; layout.total_words as usize];
    scratch[..dust::PYRAMID_HEADER_WORDS as usize]
      .copy_from_slice(&cpu[..dust::PYRAMID_HEADER_WORDS as usize]);
    for i in 0..moments.len() {
      let m = &moments[i];
      if !(m.flux() > 0.0) {
        continue;
      }
      // the bins' time segments as both mirrors' passes form them (`bin_time_segment`: the bin
      // edges' midpoints of the predecessor and the cluster; the cluster means' difference this
      // used until 2026-10-16 is another leaf — the trace's metres and its unmatched specks)
      let pred = dust::streak_pred(&render, i).map(|p| &moments[p]);
      let slot = dust::render_slot(render[i].age_id_dbeta_flux[1]);
      let exposure = speck_pc.exposure * dust::flow_factor(m.age(), &flow);
      let v = specks::SpeckView { exposure, ..view };
      let phot = specks::SpeckPhotometry {
        albedo: speck_pc.albedo,
        sun: m.sun_dir(),
        r_h_au: m.helio_au(),
      };
      for b in 0..dust::SIZE_BINS as usize {
        // the walk gate (`DUST_SPECKS_WALK_MIN`): the bins the pass did not walk draw no speck
        if !m.specks_walked(b) {
          continue;
        }
        let seg = specks::bin_time_segment(m, pred, b);
        let sc = specks::SpeckCluster::new(m, b, seg, slot, s_ref, q, ttl, 0.0);
        sc.enumerate(
          eye,
          d_max,
          view.fwd,
          specks::DUST_SPECKS_MAX_NODES,
          &mut |sp| {
            let c = |r: usize| {
              mvp_rel[r] * sp.dp[0]
                + mvp_rel[4 + r] * sp.dp[1]
                + mvp_rel[8 + r] * sp.dp[2]
                + mvp_rel[12 + r]
            };
            let cw = c(3);
            cpu_dp.insert(sp.hash, (sp.dp, sp.xi, sp.w, sc));
            let dist = (sp.dp[0] * sp.dp[0] + sp.dp[1] * sp.dp[1] + sp.dp[2] * sp.dp[2]).sqrt();
            let (_, counts) = specks::scatter_speck(
              sp.dp,
              sp.xsec_m2,
              sp.hash,
              [1.0; 3],
              &phot,
              &v,
              &layout,
              &mut scratch,
            );
            if cw > 0.0 {
              let (nx, ny) = (c(0) / cw, c(1) / cw);
              let (px, py) = ((nx * 0.5 + 0.5) * w as f32, (ny * 0.5 + 0.5) * h as f32);
              let ddp = speck_floor(&sc, sp.xi);
              recs.push(SpeckRec {
                cluster: i,
                bin: b,
                dist,
                mu: [px, py],
                counts,
                pred_px: f_px * ddp * (1.0 + nx.abs().max(ny.abs())) / cw,
                pred_amp: 2.0 * ddp / cw,
              });
              if px >= 0.0 && py >= 0.0 && px < w as f32 && py < h as f32 {
                landing[py as usize * lw as usize + px as usize] += 1;
              }
            } else {
              recs.push(SpeckRec {
                cluster: i,
                bin: b,
                dist,
                mu: [f32::NAN; 2],
                counts: 0,
                pred_px: 0.0,
                pred_amp: 0.0,
              });
            }
            true
          },
        );
      }
    }
  }
  // the 3×3 rule judges the neighbourhoods with no speck footprint within 2 px: a speck's
  // footprint reaches 2.5 px from its centre, the box 1 px, so texels within 5.5 px of a speck
  // are its; only specks of ≥ 6 counts are counted (a speck below the rule's 6-count slack
  // cannot break it alone: the mirrors place specks ~1 px apart, see the isolated comparison)
  let mut masked = alloc::vec![false; (lw * lh) as usize];
  let mut bright = 0usize;
  for r in &recs {
    if !r.mu[0].is_finite() || r.counts < 6 {
      continue;
    }
    bright += 1;
    let rad = 5.5f32;
    let (x0, x1) = (
      (r.mu[0] - rad).floor() as i64,
      (r.mu[0] + rad).ceil() as i64,
    );
    let (y0, y1) = (
      (r.mu[1] - rad).floor() as i64,
      (r.mu[1] + rad).ceil() as i64,
    );
    for y in y0.max(0)..=y1.min(lh as i64 - 1) {
      for x in x0.max(0)..=x1.min(lw as i64 - 1) {
        masked[y as usize * lw as usize + x as usize] = true;
      }
    }
  }
  let mut worst: alloc::vec::Vec<(u64, i64, i64, u64, u64)> = alloc::vec::Vec::new();
  let (mut compared, mut violations, mut excluded) = (0usize, 0usize, 0usize);
  for y in 0..lh as i64 {
    for x in 0..lw as i64 {
      let (mut bg, mut bc) = (0u64, 0u64);
      for dy in -1..=1 {
        for dx in -1..=1 {
          bg += at(&gpu, x + dx, y + dy);
          bc += at(&cpu, x + dx, y + dy);
        }
      }
      if bg.max(bc) >= 100 && masked[y as usize * lw as usize + x as usize] {
        // a speck's neighbourhood: the specks are judged one by one below
        excluded += 1;
      } else if bg.max(bc) >= 100 {
        compared += 1;
        if bg.abs_diff(bc) as f64 > 6.0 + 5e-2 * bg.max(bc) as f64 {
          violations += 1;
          worst.push((bg.abs_diff(bc), x, y, bg, bc));
        }
      }
      let i = off as usize + (y as usize * lw as usize + x as usize) * 5;
      let (g0, c0) = (gpu[i], cpu[i]);
      if g0 >= 64 && c0 >= 64 {
        for k in 1..4 {
          let (hg, hc) = (
            gpu[i + k] as f32 / g0 as f32 / dust::DUST_COLOR_SCALE,
            cpu[i + k] as f32 / c0 as f32 / dust::DUST_COLOR_SCALE,
          );
          assert!(
            (hg - hc).abs() <= 0.03 + 8.0 / c0 as f32,
            "texel ({x},{y}) word {k}: hue gpu {hg:.4} cpu {hc:.4}"
          );
        }
      }
    }
  }
  // the worst neighbourhoods: is the difference the specks' (specks-only τ ~ the difference, a
  // few specks landing) or the splat's (splat-only τ carries it)?
  worst.sort_unstable_by(|a, b| b.0.cmp(&a.0));
  for &(d, x, y, bg, bc) in worst.iter().take(8) {
    let (mut tsplat, mut tspeck, mut n) = (0u64, 0u64, 0u32);
    for dy in -1..=1i64 {
      for dx in -1..=1i64 {
        tsplat += at(&splat_only, x + dx, y + dy);
        tspeck += at(&specks_only, x + dx, y + dy);
        let (xx, yy) = (x + dx, y + dy);
        if xx >= 0 && yy >= 0 && xx < lw as i64 && yy < lh as i64 {
          n += landing[yy as usize * lw as usize + xx as usize];
        }
      }
    }
    std::println!(
      "[specks gpu] level 0 ({x},{y}): 3×3 τ gpu {bg} cpu {bc} (|Δ| {d}); cpu splat-only {tsplat}, cpu specks-only {tspeck} from {n} specks"
    );
  }
  std::println!(
    "[specks gpu] level-0 3×3 τ sums beyond 5 % + 6 on {violations} of {compared} speck-free neighbourhoods; {excluded} neighbourhoods within 2 px of the footprints of {bright} specks of ≥ 6 counts (of {} in front) judged per speck",
    recs.len()
  );
  // per speck. Shells around single specks cannot isolate them across the mirrors: inside the
  // ball the specks' distances are ~1 mm apart at 160 m (N ∝ D³) while a speck's cross-mirror
  // offset is ~0.3 m (0.79 px at 73 m) — the first version's "τ gpu 0 cpu 101" at 161 m was the
  // GPU's copy of that speck just outside its own two shells. Instead one shell, grown until
  // the specks ≥ 1 m inside it give ≥ 16 well-separated on-screen candidates, and a frame with
  // no specks (the splat alone, identical in every frame): their difference holds the nearest
  // specks only, sparse on the screen. Each candidate (footprint
  // inside the frame, ≥ 60 counts for a centroid above the continuum's own change between the
  // two frames (the hand-over inside the shell is in the difference too: at ≥ 20 counts seven
  // of 16 candidates measured 0 or negative and one 19 of 27, 2026-10-16), no other speck of the
  // shell within 8 px) is
  // compared in the 9×9 window about its CPU position (its 2.5 px footprint + ≤ 1.5 px offset).
  // Bounds: the first isolated speck differed by 0.79 px and 1.1 % (one count of 91: the
  // stochastic rounding's granularity); the mirrors' inverse CDFs differ by a few f32 ulps,
  // amplified by `‖L‖ / depth` (module doc of `dust::specks`). Centroid within 1.5 px (≈ 2× the
  // measured offset), τ within 3 % + 2 counts (each mirror rounds `⌊counts + u⌉` to ±1 count).
  // `speck_floor` (a worst-case bound, far looser) is printed for information.
  // the GPU's speck positions against the CPU's, by hash (`dust_set_specks_trace`: one more
  // frame at the shell with the walk pass recording every speck it draws): the cross-mirror
  // offset of a speck is the mirrors' transcendental ulps through Φ⁻¹ and the quantile; the
  // user's millimetre is a speck's consistency in time within a mirror, measured on the CPU
  // (`specks_hold_to_a_millimetre_floors`), and this is the cross-mirror figure
  device.dust_set_specks_trace(pe, true);
  {
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], flags, flow)?;
        splat_pc.inv_unit = 1.0 / view.unit;
        speck_pc.inv_unit = 1.0 / view.unit;
        device.cmd_dust_system_begin(id, flags, flow)?;
        let (render, moments) =
          device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
        device.cmd_dust_pre_splat_barrier(cmd);
        device.cmd_dust_specks(cmd, pe, id, 0, first_slot, capacity, render, moments, speck_pc)?;
        device.cmd_dust_pre_splat_barrier(cmd);
        device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, splat_pc)?;
        device.cmd_dust_post_splat_barrier(cmd);
        device.cmd_dust_frame_end(cmd, pe)?;
        device.cmd_dust_system_end(id)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
  }
  device.dust_set_specks_trace(pe, false);
  let records = device.download_dust_specks_trace_records(pe).unwrap();
  let fb = |w: u32| f32::from_bits(w);
  let trace: alloc::vec::Vec<(u32, [f32; 3])> =
    records.iter().map(|w| (w[0], [fb(w[1]), fb(w[2]), fb(w[3])])).collect();
  {
    // the leaf identity per matched speck: the GPU's probabilities `w`, coordinates and κ
    // against the CPU's (the same hash is the same node: `w` must agree to its fma ulp)
    let (mut dprob, mut dcoord, mut dkap): (alloc::vec::Vec<f64>, alloc::vec::Vec<f64>, alloc::vec::Vec<f64>) =
      Default::default();
    let mut by_err: alloc::vec::Vec<(f64, usize)> = alloc::vec::Vec::new();
    let mut unmatched_at: alloc::vec::Vec<usize> = alloc::vec::Vec::new();
    let mut offs: alloc::vec::Vec<f64> = alloc::vec::Vec::new();
    let mut unmatched = 0usize;
    // per coordinate: |Δc_k| = |(M⁻¹ Δdp)_k| (the CPU's map: the GPU's own map differs by its
    // ulps too, which shows as a Δc proportional to |c|), and its world share |Δc_k| ‖M_k‖
    let mut dc: [alloc::vec::Vec<f64>; 3] = Default::default();
    let mut dw: [alloc::vec::Vec<f64>; 3] = Default::default();
    let mut worst = (0.0f64, [0.0f64; 3], [0.0f32; 3], 0.0f32);
    let mut rel: alloc::vec::Vec<f64> = alloc::vec::Vec::new();
    let (mut map_err, mut cond, mut edge): (alloc::vec::Vec<f64>, alloc::vec::Vec<f64>, alloc::vec::Vec<f64>) =
      Default::default();
    let mut worst_cond = (0.0f64, 0.0f64);
    for (ti, (hash, dp)) in trace.iter().enumerate() {
      match cpu_dp.get(hash) {
        Some((c, xi, wc, sc)) => {
          let (mi, cn, kappa) = (&sc.minv, &sc.col_norm, &sc.kappa);
          let rec = &records[ti];
          let (mut a, mut b) = (0.0f64, 0.0f64);
          for k in 0..3 {
            a = a.max((fb(rec[4 + k]) as f64 - wc[k] as f64).abs());
            b = b.max((fb(rec[7 + k]) as f64 - xi[k] as f64).abs());
          }
          dprob.push(a);
          dcoord.push(b);
          dkap.push(((fb(rec[10]) as f64 - *kappa as f64) / (*kappa as f64).abs().max(1e-30)).abs());
          // the CPU's own f32 map against the f64 map of the same f32 leaf: if the cross-mirror
          // offset is the f32 factor's conditioning (a near rank-1 leaf: its small pivots are
          // differences of the large), this error has the same size as the mirrors' |Δdp|
          let m64 = leaf_map64(sc.leaf.cov, sc.leaf.seg);
          let mut e = [0.0f64; 3];
          for k in 0..3 {
            for j in 0..3 {
              e[k] += (sc.chol[j][k] as f64 - m64[j][k]) * xi[j] as f64;
            }
          }
          map_err.push((e[0] * e[0] + e[1] * e[1] + e[2] * e[2]).sqrt());
          let fro = (cn[0] as f64).hypot(cn[1] as f64).hypot(cn[2] as f64);
          cond.push(fro / (sc.sigma_lo as f64).max(1e-30));
          let d = [
            (dp[0] - c[0]) as f64,
            (dp[1] - c[1]) as f64,
            (dp[2] - c[2]) as f64,
          ];
          let n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
          offs.push(n);
          by_err.push((n, ti));
          // relative to the speck's own `‖M‖ (1 + |c|)` (an f32 ulp of the map or of the
          // coordinates is 6e-8 of it; the tolerance 1e-5)
          let cmax = cn[0].max(cn[1]).max(cn[2]) as f64;
          let xmax = xi[0].abs().max(xi[1].abs()).max(xi[2].abs()) as f64;
          rel.push(n / (cmax * (1.0 + xmax)).max(1e-30));
          let mut dck = [0.0f64; 3];
          for k in 0..3 {
            dck[k] = mi[3 * k] as f64 * d[0] + mi[3 * k + 1] as f64 * d[1] + mi[3 * k + 2] as f64 * d[2];
            dc[k].push(dck[k].abs());
            dw[k].push(dck[k].abs() * cn[k] as f64);
          }
          if n > worst.0 {
            worst = (n, dck, *xi, *kappa);
            worst_cond = (*cond.last().unwrap(), *map_err.last().unwrap());
          }
        }
        None => {
          unmatched += 1;
          // a different leaf, or a speck the CPU placed just outside the shell?
          let r = ((dp[0] as f64).powi(2) + (dp[1] as f64).powi(2) + (dp[2] as f64).powi(2)).sqrt();
          edge.push(r - d_max as f64);
          unmatched_at.push(ti);
        }
      }
    }
    offs.sort_by(|a, b| a.total_cmp(b));
    let pick = |v: &alloc::vec::Vec<f64>, p: f64| v.get(((v.len().saturating_sub(1)) as f64 * p) as usize).copied().unwrap_or(f64::NAN);
    for k in 0..3 {
      dc[k].sort_by(|a, b| a.total_cmp(b));
      dw[k].sort_by(|a, b| a.total_cmp(b));
      std::println!(
        "[specks gpu] trace axis {k}: |Δc| p50 {:.2e} p99 {:.2e} max {:.2e}; world |Δc|‖M_k‖ p50 {:.2e} m p99 {:.2e} m max {:.2e} m",
        pick(&dc[k], 0.5),
        pick(&dc[k], 0.99),
        pick(&dc[k], 1.0),
        pick(&dw[k], 0.5),
        pick(&dw[k], 0.99),
        pick(&dw[k], 1.0)
      );
    }
    for v in [&mut map_err, &mut cond, &mut edge] {
      v.sort_by(|a, b| a.total_cmp(b));
    }
    std::println!(
      "[specks gpu] trace map: CPU f32 M c vs f64 M c of the same leaf p50 {:.2e} m p90 {:.2e} m p99 {:.2e} m max {:.2e} m; cond ‖M‖‖M⁻¹‖ p50 {:.2e} p99 {:.2e} max {:.2e}; the worst speck's cond {:.2e}, its f32-map error {:.2e} m",
      pick(&map_err, 0.5),
      pick(&map_err, 0.9),
      pick(&map_err, 0.99),
      pick(&map_err, 1.0),
      pick(&cond, 0.5),
      pick(&cond, 0.99),
      pick(&cond, 1.0),
      worst_cond.0,
      worst_cond.1
    );
    for v in [&mut dprob, &mut dcoord, &mut dkap] {
      v.sort_by(|a, b| a.total_cmp(b));
    }
    std::println!(
      "[specks gpu] trace leaf: max |Δw| p50 {:.2e} p99 {:.2e} max {:.2e} ({} of {} non-zero); max |Δcoord| p50 {:.2e} p99 {:.2e} max {:.2e}; |Δκ|/κ p50 {:.2e} p99 {:.2e} max {:.2e}",
      pick(&dprob, 0.5),
      pick(&dprob, 0.99),
      pick(&dprob, 1.0),
      dprob.iter().filter(|&&x| x > 0.0).count(),
      dprob.len(),
      pick(&dcoord, 0.5),
      pick(&dcoord, 0.99),
      pick(&dcoord, 1.0),
      pick(&dkap, 0.5),
      pick(&dkap, 0.99),
      pick(&dkap, 1.0)
    );
    by_err.sort_by(|a, b| b.0.total_cmp(&a.0));
    for &(n, ti) in by_err.iter().take(5) {
      let rec = &records[ti];
      let (_, xi, wc, sc) = &cpu_dp[&rec[0]];
      std::println!(
        "[specks gpu] trace worst {n:.2e} m: gpu w {:?} coord {:?} κ {} level {}; cpu w {:?} coord {:?} κ {} (bin {})",
        [fb(rec[4]), fb(rec[5]), fb(rec[6])],
        [fb(rec[7]), fb(rec[8]), fb(rec[9])],
        fb(rec[10]),
        rec[11],
        wc,
        xi,
        sc.kappa,
        sc.bin
      );
    }
    for &ti in unmatched_at.iter().take(5) {
      let rec = &records[ti];
      let r = (fb(rec[1]).powi(2) + fb(rec[2]).powi(2) + fb(rec[3]).powi(2)).sqrt();
      std::println!(
        "[specks gpu] trace unmatched hash {:#010x}: gpu |dp| {r:.2} m (shell {d_max:.2}), w {:?} coord {:?} κ {} level {}",
        rec[0],
        [fb(rec[4]), fb(rec[5]), fb(rec[6])],
        [fb(rec[7]), fb(rec[8]), fb(rec[9])],
        fb(rec[10]),
        rec[11]
      );
    }
    std::println!(
      "[specks gpu] trace unmatched: |dp_gpu| − d_max min {:.2e} m p50 {:.2e} m max {:.2e} m (of {} unmatched)",
      pick(&edge, 0.0),
      pick(&edge, 0.5),
      pick(&edge, 1.0),
      edge.len()
    );
    rel.sort_by(|a, b| a.total_cmp(b));
    std::println!(
      "[specks gpu] trace worst: |Δdp| {:.2e} m, Δc {:?}, ξ {:?}, κ {}; |Δdp| / (‖M‖(1+|c|)) p50 {:.2e} p99 {:.2e} max {:.2e}",
      worst.0,
      worst.1,
      worst.2,
      worst.3,
      pick(&rel, 0.5),
      pick(&rel, 0.99),
      pick(&rel, 1.0)
    );
    let q = |p: f64| pick(&offs, p);
    std::println!(
      "[specks gpu] trace: {} specks recorded, {} matched by hash with the CPU's {} ({unmatched} unmatched); |Δdp| p50 {:.2e} m, p90 {:.2e} m, p99 {:.2e} m, max {:.2e} m",
      trace.len(),
      offs.len(),
      cpu_dp.len(),
      q(0.5),
      q(0.9),
      q(0.99),
      q(1.0)
    );
    assert!(trace.len() >= 1000, "the trace recorded {} specks", trace.len());
    assert!(unmatched <= trace.len() / 100, "{unmatched} GPU specks have no CPU speck of their hash");
    // measured 2026-10-16 on this five-day scene (clouds of 10⁵–10⁶ m): every speck matched,
    // identical `w`, |Δdp| p50 8.7 cm, p99 29 cm, max 1.5 m, relative to `‖M‖(1 + |c|)` p50
    // 1.4e-7, p99 4.8e-7, max 1.7e-6 — the f32 coordinates and map (a few ulps of a 4e5 m
    // column); the tail is the ζ quantile's upper tail (Δz 1–2e-5 at w ≈ 0.993: the CDF's f32
    // absolute error over its small density). (The metres of the first version, p50 1.2 m and
    // 26 unmatched, were this test's own reference: the cluster means' difference for the bins'
    // time segment instead of `bin_time_segment`.) Bounded at 4× the measured relative p99 and
    // max; centimetres on such clouds need the df64 placed-speck path (plan 1b (a0)).
    assert!(pick(&rel, 0.99) < 2e-6, "cross-mirror relative p99 {}", pick(&rel, 0.99));
    assert!(pick(&rel, 1.0) < 7e-6, "cross-mirror relative max {}", pick(&rel, 1.0));
  }
  recs.sort_by(|a, b| a.dist.total_cmp(&b.dist));
  let on_screen = |r: &SpeckRec| {
    r.mu[0].is_finite()
      && r.mu[0] >= 0.0
      && r.mu[1] >= 0.0
      && r.mu[0] < w as f32
      && r.mu[1] < h as f32
  };
  let footprint_inside = |r: &SpeckRec| {
    r.mu[0] >= 4.5 && r.mu[1] >= 4.5 && r.mu[0] <= w as f32 - 4.5 && r.mu[1] <= h as f32 - 4.5
  };
  // the shell keeps a margin of 1 m on both sides of its boundary: a speck's cross-mirror offset
  // is ~0.3 m (0.79 px at 73 m with 173 px per radian), so a speck within 1 m of the boundary
  // may be inside on one mirror only; candidates lie ≥ 1 m inside, and "alone within 8 px" is
  // checked against every speck up to 1 m outside
  let margin = 1.0f32;
  let upto = |d: f32| recs.partition_point(|r| r.dist <= d);
  // the walk gate (`DUST_SPECKS_WALK_MIN`) decides per bin and per shell: a candidate's bin must
  // still be walked at the isolation shell, else neither mirror draws it there
  let bin_walked = |cluster: usize, bin: usize, shell: f32| -> bool {
    let m = &moments[cluster];
    let pred = dust::streak_pred(&render, cluster).map(|p| &moments[p]);
    let seg = specks::bin_time_segment(m, pred, bin);
    let slot = dust::render_slot(render[cluster].age_id_dbeta_flux[1]);
    let sc = specks::SpeckCluster::new(m, bin, seg, slot, s_ref, q, ttl, 0.0);
    sc.expected_in_shell(&specks::ShellCut::new(eye, shell)) >= specks::DUST_SPECKS_WALK_MIN
  };
  let candidates = |shell: f32| -> alloc::vec::Vec<usize> {
    let (n_in, n_all) = (upto(shell - margin), upto(shell + margin));
    (0..n_in)
      .filter(|&j| {
        let r = &recs[j];
        on_screen(r)
          && footprint_inside(r)
          && r.counts >= 60
          && bin_walked(r.cluster, r.bin, shell)
          && (0..n_all).all(|k| {
            k == j
              || !on_screen(&recs[k])
              || recs[k].counts == 0
              || (recs[k].mu[0] - r.mu[0]).hypot(recs[k].mu[1] - r.mu[1]) > 8.0
          })
      })
      .collect()
  };
  // (the walk gate draws no lone speck of a bin expecting under a quarter grain, which were the
  // easiest to isolate: the search may run over every enumerated speck)
  let n_max = recs.len();
  let mut n = 1usize;
  let mut shell = recs[0].dist + margin;
  let mut chosen = candidates(shell);
  while n < n_max && chosen.len() < 16 {
    n = (n * 5 / 4).max(n + 1).min(n_max);
    shell = recs[n - 1].dist + margin;
    chosen = candidates(shell);
  }
  std::println!(
    "[specks gpu] isolation shell {shell:.3} m ({n} specks within it less the margin): {} candidates",
    chosen.len()
  );
  let level0 = |p: &[u32]| -> alloc::vec::Vec<i64> {
    (0..(lw * lh) as usize)
      .map(|t| p[off as usize + t * dust::PYRAMID_TEXEL_WORDS as usize] as i64)
      .collect()
  };
  // one frame of both mirrors with the shell `rk`: the level-0 τ words; the GPU header's
  // `PYR_SPECKS_DMAX` of that frame is printed (the override reaches it in the same frame)
  let frame_pair = |rk: f32| -> (alloc::vec::Vec<i64>, alloc::vec::Vec<i64>) {
    device.dust_set_shell_override(pe, Some(rk));
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_frame_begin(cmd, pe, [w, h], flags, flow)?;
        device.cmd_dust_system_begin(id, flags, flow)?;
        let (render, moments) =
          device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
        device.cmd_dust_pre_splat_barrier(cmd);
        // the specks before the splat: their walk mask decides the splat's hand-over
        device.cmd_dust_specks(cmd, pe, id, 0, first_slot, capacity, render, moments, speck_pc)?;
        device.cmd_dust_pre_splat_barrier(cmd);
        device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, splat_pc)?;
        device.cmd_dust_post_splat_barrier(cmd);
        device.cmd_dust_frame_end(cmd, pe)?;
        device.cmd_dust_system_end(id)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let pyr = device.download_dust_pyramid(pe).unwrap();
    let hd = f32::from_bits(pyr[specks::PYR_SPECKS_DMAX as usize]);
    std::println!(
      "[specks gpu] isolation frame: shell {rk:.4} m, GPU header d_max {hd:.4} m, GPU specks drawn {}",
      pyr[specks::PYR_SPECKS as usize]
    );
    assert_eq!(hd, rk, "the shell override must reach the frame's header");
    let g = level0(&pyr);
    let mut c = alloc::vec![0u32; layout.total_words as usize];
    c[..dust::PYRAMID_HEADER_WORDS as usize]
      .copy_from_slice(&specks_only[..dust::PYRAMID_HEADER_WORDS as usize]);
    specks::SpeckHeader::write(&mut c, rk, gpu[specks::PYR_SPECKS_BUDGET as usize]);
    let mut mm = moments.clone();
    specks::specks_tier(&mut mm, &render, &speck_pc, &flow, &layout, &mut c);
    (g, level0(&c))
  };
  let (g0, c0) = frame_pair(0.5 * recs[0].dist);
  let (g1, c1) = frame_pair(shell);
  // the speck's footprint: the shell frame less the speck-free frame in the 9×9 window
  let window = |a: &[i64], b: &[i64], mu: [f32; 2]| -> (f64, [f64; 2]) {
    let (cx, cy) = (mu[0].floor() as i64, mu[1].floor() as i64);
    let (mut s, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
    for y in (cy - 4).max(0)..=(cy + 4).min(lh as i64 - 1) {
      for x in (cx - 4).max(0)..=(cx + 4).min(lw as i64 - 1) {
        let t = y as usize * lw as usize + x as usize;
        let d = (a[t] - b[t]) as f64;
        s += d;
        sx += d * (x as f64 + 0.5);
        sy += d * (y as f64 + 0.5);
      }
    }
    (
      s,
      if s != 0.0 {
        [sx / s, sy / s]
      } else {
        [f64::NAN; 2]
      },
    )
  };
  let mut isolated = 0usize;
  for &j in chosen.iter().take(16) {
    let r = &recs[j];
    let (sg, cg) = window(&g1, &g0, r.mu);
    let (sc, cc) = window(&c1, &c0, r.mu);
    let dpx = ((cg[0] - cc[0]).powi(2) + (cg[1] - cc[1]).powi(2)).sqrt();
    std::println!(
      "[specks gpu] speck at {:.3} m, cpu ({:.2}, {:.2}), {} counts: τ gpu {sg:.0} cpu {sc:.0} ({:+.2} %), centroid gpu ({:.3}, {:.3}) cpu ({:.3}, {:.3}), |Δ| {dpx:.3} px; worst-case floor {:.3} px, {:.2} %",
      r.dist,
      r.mu[0],
      r.mu[1],
      r.counts,
      100.0 * (sg / sc - 1.0),
      cg[0],
      cg[1],
      cc[0],
      cc[1],
      r.pred_px,
      100.0 * r.pred_amp
    );
    if !(sc > 0.0) {
      continue;
    }
    // a window that holds under half the speck's counts, or whose CPU centroid lies over 3 px
    // from the record, is clipped (the footprint at the window's edge, the continuum's hand-over
    // in the difference): not a measurement of the speck (2026-10-16: a 27-count speck drawn
    // 4 px from its record, 19 counts in the CPU window and 12 in the GPU's)
    let off_px = (cc[0] - r.mu[0] as f64).hypot(cc[1] - r.mu[1] as f64);
    if sc < 0.5 * r.counts as f64 || off_px > 4.0 {
      std::println!(
        "[specks gpu] speck at {:.3} m: window clipped ({sc:.0} of {} counts, centroid {off_px:.2} px off the record), not judged",
        r.dist, r.counts
      );
      continue;
    }
    isolated += 1;
    assert!(
      (sg - sc).abs() <= 0.03 * sc + 2.0,
      "speck at {} m: τ gpu {sg} cpu {sc}",
      r.dist
    );
    assert!(
      dpx <= 1.5,
      "speck at {} m: centroids {dpx} px apart",
      r.dist
    );
  }
  std::println!(
    "[specks gpu] {isolated} specks compared of {} candidates",
    chosen.len()
  );
  assert!(isolated >= 8, "only {isolated} on-screen specks isolated");
  // the speck-free neighbourhoods (stated in the print above; in this geometry every
  // neighbourhood lies within 2 px of a speck's footprint, so there may be none: the splat has its
  // own parity tests); the specks themselves are judged one by one and in aggregate (Σ counts
  // 0.1 %, the speck words)
  assert!(
    violations <= 1 + compared / 100,
    "3×3 τ sums differ on {violations} of {compared}"
  );
  device.dust_set_shell_override(pe, None);
  device.discard_dust_view(pe, 0);
}

/// One CPU speck of the parity frame: distance from the eye, screen position (NaN behind the
/// eye), the counts it deposits, and the worst-case cross-mirror floor in px and in relative
/// amplitude ([`speck_floor`], information only).
struct SpeckRec {
  dist: f32,
  mu: [f32; 2],
  counts: u32,
  pred_px: f32,
  pred_amp: f32,
  /// the speck's cluster (compact index) and size bin: the walk gate decides per bin and per
  /// shell whether it is drawn at all (`SpeckCluster::expected_in_shell`)
  cluster: usize,
  bin: usize,
}

/// A worst-case cross-mirror uncertainty (m) of a speck's offset `L ξ` (printed only: its
/// `1/φ(ξ)` term is far too pessimistic in the tails; the measured offsets are ~1 px): the GPU's `log`/`exp`/`sqrt`
/// differ from the CPU's by a few ulps, so `ξ = Φ⁻¹(w)` differs. In the central branch the
/// Halley step sets `x ← x − (Φ(x) − p)/φ(x)`: `Φ(x)` (from `erf`, an `exp`) differs by up to
/// ~8 ulps of 1 = 2⁻²¹, moving `x` by 2⁻²¹/φ(x); in the tails `√(−2 ln p)` differs by a few
/// ulps of `|x|`. So `Δξ_k ≤ 2⁻²¹ (1/φ(ξ_k) + |ξ_k|)` and `|Δ(L ξ)| ≤ Σ_k ‖L_k‖ Δξ_k`; the f32
/// `L` itself is computed by the same correctly rounded `+ − × ÷ √` on both sides (`÷` and `√`
/// to 2.5 ulps on the GPU: 3·10⁻⁷ of ‖L‖, inside the bound). On the screen that is
/// `f_px · Δ / w` px (× (1 + |ndc|) off axis) and `2 Δ / w` of the amplitude: a speck 1 m from
/// the eye in a cloud of 10⁵ m sits 1–10 px apart on the two mirrors. The same mirror is exact
/// from frame to frame (same inputs, same ops): persistence is not affected.
/// `specks::leaf_map` in f64 from the same f32 leaf (columns of `M`): the frame along `seg`,
/// the Cholesky factor with `cholesky3`'s floor, mapped back. Only the map's `M`, not `κ`.
fn leaf_map64(cov: [f32; 6], seg: [f32; 3]) -> [[f64; 3]; 3] {
  let c: [f64; 6] = core::array::from_fn(|i| cov[i] as f64);
  let chol = |s: [f64; 6]| -> [[f64; 3]; 3] {
    let tr = s[0] + s[3] + s[5];
    let fl = (tr * 1e-12).max(1e-20);
    let (sxx, sxy, sxz, syy, syz, szz) = (s[0] + fl, s[1], s[2], s[3] + fl, s[4], s[5] + fl);
    let l00 = sxx.max(0.0).sqrt();
    let (l10, l20) = if l00 > 0.0 { (sxy / l00, sxz / l00) } else { (0.0, 0.0) };
    let d1 = syy - l10 * l10;
    let l11 = if d1 > 0.0 { d1.sqrt() } else { 0.0 };
    let l21 = if l11 > 0.0 { (syz - l20 * l10) / l11 } else { 0.0 };
    let d2 = szz - l20 * l20 - l21 * l21;
    let l22 = if d2 > 0.0 { d2.sqrt() } else { 0.0 };
    [[l00, l10, l20], [0.0, l11, l21], [0.0, 0.0, l22]]
  };
  let s: [f64; 3] = core::array::from_fn(|i| seg[i] as f64);
  let slen = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
  if !(slen > 0.0) {
    return chol(c);
  }
  let cross = |a: [f64; 3], b: [f64; 3]| {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
  };
  let e3 = [s[0] / slen, s[1] / slen, s[2] / slen];
  let a = if e3[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
  let e1 = cross(a, e3);
  let n1 = (e1[0] * e1[0] + e1[1] * e1[1] + e1[2] * e1[2]).sqrt();
  let e1 = [e1[0] / n1, e1[1] / n1, e1[2] / n1];
  let e2 = cross(e3, e1);
  let sv = |v: [f64; 3]| {
    [
      c[0] * v[0] + c[1] * v[1] + c[2] * v[2],
      c[1] * v[0] + c[3] * v[1] + c[4] * v[2],
      c[2] * v[0] + c[4] * v[1] + c[5] * v[2],
    ]
  };
  let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  let (s1, s2, s3) = (sv(e1), sv(e2), sv(e3));
  let lb = chol([dot(e1, s1), dot(e1, s2), dot(e1, s3), dot(e2, s2), dot(e2, s3), dot(e3, s3)]);
  let world = |l: [f64; 3]| -> [f64; 3] {
    core::array::from_fn(|k| e1[k] * l[0] + e2[k] * l[1] + e3[k] * l[2])
  };
  [world(lb[0]), world(lb[1]), world(lb[2])]
}

fn speck_floor(sc: &specks::SpeckCluster, xi: [f32; 3]) -> f32 {
  let mut d = 0.0f32;
  for k in 0..3 {
    let x = xi[k];
    let phi = (-0.5 * x * x).exp() / (2.0 * core::f32::consts::PI).sqrt();
    let dxi = (1.0 / phi.max(1e-30) + x.abs()) * 2f32.powi(-21);
    d += sc.col_norm[k] * dxi;
  }
  d
}
