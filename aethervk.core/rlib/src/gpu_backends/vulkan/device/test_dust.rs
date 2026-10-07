//! Dust v3 GPU parity tests: `dust_emit.comp` / `dust_propagate.comp` (df64, no shaderFloat64)
//! against the Rust reference in `crate::scene::dust`. They need a Vulkan device (any, Lavapipe
//! included) and the compiled shaders in `assets/sim`.
use super::{
  test_utils::{setup_assets_dir, setup_render_frontend_for_tests},
  *,
};
use crate::scene::dust::{
  self, AU_M, DustBatch, DustCluster, DustFrame, DustRenderCluster, SUN_MU_M3_S2, SizeDistribution,
  V3, kepler,
};

/// 67P-like comet at 5.5 AU, ~0.9 of circular speed, slightly inclined
fn comet_state() -> (V3, V3) {
  let r = [-1.924 * AU_M, -5.164 * AU_M, -0.204 * AU_M];
  let rn = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
  let vc = (SUN_MU_M3_S2 / rn).sqrt() * 0.9;
  let t = [5.164f64, -1.924, 0.3];
  let tn = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
  (r, [t[0] * vc / tn, t[1] * vc / tn, t[2] * vc / tn])
}

const T_START: f64 = 4.0e8; // ~12.7 years after the epoch: exercises df64 time

fn test_batch(count: u32, dur: f64, first_index: u32, capacity: u32) -> DustBatch {
  let (rc, vc) = comet_state();
  let dist = SizeDistribution::from_diameter_um(100.0);
  let (size_params, vel_params, mass_params) =
    dust::batch_params(&dist, 100.0, 0.533, 0.0213, 2.0, 0.5, 1.0e6, 0.37);
  let mut b: DustBatch = bytemuck::Zeroable::zeroed();
  b.set_comet(rc, vc, T_START, dur);
  b.rot_start = [0.0, 0.0, 0.0, 1.0];
  // 45° about z over the batch: exercises the spin rotation
  b.spin = [
    0.0,
    0.0,
    1.0,
    (core::f64::consts::FRAC_PI_4 / dur.max(1.0)) as f32,
  ];
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

/// Copies `bytes` bytes of a device buffer into host memory (synchronous).
fn read_back(
  device: &Device,
  src: vk::Buffer,
  compute_queue: bool,
  bytes: u64,
) -> alloc::vec::Vec<u8> {
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
  let record = |cmd: vk::CommandBuffer| -> GpuResult<()> {
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
  };
  let mut res = if compute_queue {
    device.run_transient_compute_commands(record)
  } else {
    device.run_transient_commands(record)
  }
  .unwrap();
  res.cleanup(&device.device);
  allocator.invalidate_allocation(&dst_alloc, 0, bytes).unwrap();
  let out =
    unsafe { core::slice::from_raw_parts(dst_info.mapped_data.cast::<u8>(), bytes as usize) }
      .to_vec();
  unsafe { allocator.destroy_buffer(dst, &mut dst_alloc) };
  out
}

fn buffers(device: &Device, id: u64) -> (u32, vk::Buffer, vk::Buffer) {
  let res = device.res.read();
  let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
  (
    sys.capacity,
    sys.clusters.as_ref().unwrap().buffer,
    sys.render.as_ref().unwrap().buffer,
  )
}

fn emit(device: &Device, id: u64, batch: &DustBatch) {
  let mut res = device
    .run_transient_compute_commands(|cmd| device.cmd_dust_emit(cmd, id, 0, batch, 0))
    .unwrap();
  res.cleanup(&device.device);
}

fn propagate(device: &Device, id: u64, first_slot: u32, live: u32, frame: &DustFrame) {
  let capacity = buffers(device, id).0;
  let mut res = device
    .run_transient_commands(|cmd| {
      device.cmd_dust_pre_propagate_barrier(cmd);
      device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, live, frame)?;
      Ok(())
    })
    .unwrap();
  res.cleanup(&device.device);
}

fn norm3(a: [f32; 3]) -> f32 {
  (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
}

/// Runs `f` on a fresh headless device with dust system `id` allocated (GPU mode).
fn with_dust_device(id: u64, f: impl FnOnce(&Device, u32, vk::Buffer, vk::Buffer)) {
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
      let (capacity, clusters, render) = buffers(device, id);
      f(device, capacity, clusters, render);
      unsafe { device.device.device_wait_idle() }.unwrap();
      device.discard_particle_system(id, 0, 0)?;
      GpuResult::Ok(())
    })
    .unwrap();
}

#[test]
fn gpu_dust_emit_matches_reference() {
  with_dust_device(9001, |device, capacity, clusters, _| {
    // starts 100 slots before the ring end: exercises the wrap-around
    let single = test_batch(512, 1800.0, capacity - 100, capacity);
    // 64 streams on a site going in and out of daylight, after a missing window: stream breaks;
    // a size rotation (size wraps break too) and a site 2 km off the centre turning with the spin
    let mut streams = test_batch(512, 6.0 * 3600.0, capacity - 100, capacity);
    streams.mass_params[3] = dust::batch_word(6, true, false, 37);
    streams.spin[3] = (2.0 * core::f64::consts::PI / (2.0 * 3600.0)) as f32;
    streams.site_offset = [2000.0, 0.0, 0.0, 0.0];
    let (rc, _) = comet_state();
    let rn = (rc[0] * rc[0] + rc[1] * rc[1] + rc[2] * rc[2]).sqrt();
    let sun = [-rc[0] / rn, -rc[1] / rn, -rc[2] / rn];
    streams.lit = dust::LitWindow::new(
      sun,
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 1.0],
      streams.spin[3] as f64,
      6.0 * 3600.0,
    )
    .to_gpu();
    assert_eq!(streams.lit[3], dust::LIT_MODE_PERIODIC);
    for batch in [single, streams] {
      emit(device, 9001, &batch);
      let mut breaks = 0;
      let bytes = read_back(
        device,
        clusters,
        true,
        capacity as u64 * core::mem::size_of::<DustCluster>() as u64,
      );
      let ring: &[DustCluster] = bytemuck::cast_slice(&bytes);
      let mut worst_r = 0.0f64;
      for j in 0..batch.count {
        let slot = (batch.first_index.wrapping_add(j) & batch.ring_mask) as usize;
        let gpu = &ring[slot];
        let cpu = dust::emit_cluster(&batch, j);
        // emission time: df64 identical up to f32 rounding of u_t (same ops)
        assert!(
          // lit-time batches: an f32 offset, ≤ 2 ulps of the window length apart
          (gpu.t0().to_f64() - cpu.t0().to_f64()).abs()
            < 1e-3 + 2.0 * f32::EPSILON as f64 * batch.comet_v_dur_hi[3] as f64,
          "t0 slot {slot}: gpu {} cpu {}",
          gpu.t0().to_f64() - T_START,
          cpu.t0().to_f64() - T_START
        );
        // position: GPU df64 Kepler vs CPU df64 Kepler (cm level; only `/` and `sqrt` may differ)
        let dr = {
          let (a, b) = (gpu.r0().to_f64(), cpu.r0().to_f64());
          ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        worst_r = worst_r.max(dr);
        // lit-time batches emit at an f32 offset (2 ms ulp over 6 h): a sub-ulp difference of GPU
        // `/` moves the emission point along the comet's ~13 km/s path
        let dt0 = (gpu.t0().to_f64() - cpu.t0().to_f64()).abs();
        assert!(
          dr < 0.5 + 2.0e4 * dt0,
          "r0 slot {slot}: |Δ| = {dr} m (Δt0 {dt0} s)"
        );
        let dv = {
          let (a, b) = (gpu.v0().to_f64(), cpu.v0().to_f64());
          ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
        };
        // velocity: ejection direction/speed use f32 transcendental functions (cos, log, pow)
        assert!(dv < 1e-3, "v0 slot {slot}: |Δ| = {dv} m/s");
        let rel = |a: f32, b: f32| (a - b).abs() / b.abs().max(1e-30);
        assert!(
          rel(gpu.beta(), cpu.beta()) < 1e-4,
          "beta slot {slot}: gpu {} cpu {} (s_um gpu {} cpu {})",
          gpu.beta(),
          cpu.beta(),
          gpu.misc[1],
          cpu.misc[1]
        );
        assert!(rel(gpu.mass_g(), cpu.mass_g()) < 1e-4, "mass slot {slot}");
        for k in 0..3 {
          assert!(
            rel(gpu.misc[k], cpu.misc[k]) < 1e-4,
            "misc[{k}] slot {slot}"
          );
        }
        // β half-spread: its low mantissa bits are the child-pattern id (exact u32 hash), the rest
        // may land on the other side of the truncation by an ulp of `log` / `pow`
        assert_eq!(
          dust::child_id(gpu.misc[3]),
          dust::child_id(cpu.misc[3]),
          "child id slot {slot}"
        );
        assert!(rel(gpu.misc[3], cpu.misc[3]) < 2e-2, "misc[3] slot {slot}");
        assert_eq!(
          dust::stream_break(gpu.misc[3]),
          dust::stream_break(cpu.misc[3]),
          "break slot {slot}"
        );
        breaks += dust::stream_break(cpu.misc[3]) as u32;
      }
      std::println!(
        "[dust gpu] emit parity ({} streams): worst |Δr0| = {worst_r:.4} m, {breaks} stream breaks",
        1 << dust::batch_streams(&batch).0
      );
      // slots outside the batch are untouched (the ring starts zeroed only by chance, so check a
      // slot just before the batch is not one of ours)
      let before = (batch.first_index.wrapping_sub(1) & batch.ring_mask) as usize;
      assert_ne!(
        ring[before].t0().to_f64(),
        dust::emit_cluster(&batch, 0).t0().to_f64()
      );
    }
  });
}

#[test]
fn gpu_dust_propagate_matches_reference() {
  with_dust_device(9002, |device, capacity, _, render| {
    let batch = test_batch(1024, 3600.0, 7, capacity);
    emit(device, 9002, &batch);
    let frame = frame_after(10.0);
    let first_slot = batch.first_index & batch.ring_mask;
    propagate(device, 9002, first_slot, batch.count, &frame);
    let bytes = read_back(
      device,
      render,
      false,
      batch.count as u64 * core::mem::size_of::<DustRenderCluster>() as u64,
    );
    let out: &[DustRenderCluster] = bytemuck::cast_slice(&bytes);
    let mut worst = 0.0f32;
    for (i, gpu) in out.iter().enumerate() {
      let slot = (first_slot + i as u32) & batch.ring_mask;
      assert_eq!(
        dust::render_slot(gpu.age_id_dbeta_flux[1]),
        slot,
        "compact index {i} must carry its slot"
      );
      let cpu = dust::evaluate_cluster(&dust::emit_cluster(&batch, i as u32), slot, &frame);
      let d = [
        gpu.pos_size[0] - cpu.pos_size[0],
        gpu.pos_size[1] - cpu.pos_size[1],
        gpu.pos_size[2] - cpu.pos_size[2],
      ];
      let p = norm3([cpu.pos_size[0], cpu.pos_size[1], cpu.pos_size[2]]);
      // tail displacements reach 1e7 m; β from f32 pow differs by ulps between GPU and CPU
      let err = norm3(d);
      worst = worst.max(err);
      assert!(
        err < 2.0 + 2e-6 * p,
        "slot {slot}: |Δpos| = {err} m at |pos| = {p} m"
      );
      assert!(
        (gpu.age_id_dbeta_flux[0] - cpu.age_id_dbeta_flux[0]).abs() < 1e-2,
        "age slot {slot}"
      );
      let flux_rel = (gpu.age_id_dbeta_flux[3] - cpu.age_id_dbeta_flux[3]).abs()
        / cpu.age_id_dbeta_flux[3].max(1e-30);
      assert!(flux_rel < 1e-3, "flux slot {slot}");
      // the child-pattern id survives the propagation bit for bit
      assert_eq!(
        dust::child_id(gpu.age_id_dbeta_flux[2]),
        dust::child_id(cpu.age_id_dbeta_flux[2]),
        "child id slot {slot}"
      );
    }
    std::println!("[dust gpu] propagate parity: worst |Δpos| = {worst:.3} m");
  });
}

#[test]
fn gpu_dust_df64_keeps_zero_beta_cluster_on_comet() {
  // β = 0 and no ejection: the cluster IS the comet. Its offset from the comet after days at
  // 5.5 AU (8e11 m) is pure round-off of the GPU df64 path: f32 would give ~1e5 m, df64 < 1 m.
  // Fails if the driver contracts the error-free transformations into FMAs.
  with_dust_device(9003, |device, capacity, _, render| {
    let mut batch = test_batch(64, 600.0, 0, capacity);
    batch.vel_params[0] = 0.0; // no ejection
    batch.vel_params[3] = 0.0; // β = 0
    emit(device, 9003, &batch);
    for days in [0.5, 5.0, 30.0] {
      let frame = frame_after(days);
      propagate(device, 9003, 0, batch.count, &frame);
      let bytes = read_back(
        device,
        render,
        false,
        batch.count as u64 * core::mem::size_of::<DustRenderCluster>() as u64,
      );
      let out: &[DustRenderCluster] = bytemuck::cast_slice(&bytes);
      let worst = out
        .iter()
        .map(|c| norm3([c.pos_size[0], c.pos_size[1], c.pos_size[2]]))
        .fold(0.0f32, f32::max);
      std::println!("[dust gpu] β=0 after {days} d: worst offset {worst:.4} m");
      assert!(
        worst < 1.0,
        "{days} d: β=0 cluster drifted {worst} m from the comet (df64 broken?)"
      );
    }
  });
}

#[test]
fn gpu_dust_emit_lit_time_matches_reference() {
  // spinning nucleus, jet site in and out of daylight over 2.5 rotations
  const P_ROT: f64 = 12.4 * 3600.0;
  let omega = 2.0 * core::f64::consts::PI / P_ROT;
  let unit = |a: V3| {
    let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
    [a[0] / n, a[1] / n, a[2] / n]
  };
  let dot = |a: V3, b: V3| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  let axis = unit([0.2, 0.1, 0.97]);
  let n0 = unit([1.0, -0.3, 0.1]);
  let dur = 2.5 * P_ROT;
  with_dust_device(9004, |device, capacity, clusters, _| {
    let (rc, vc) = comet_state();
    let sun = unit([-rc[0], -rc[1], -rc[2]]);
    let mut batch = test_batch(2048, dur, 0, capacity);
    batch.rot_start = [0.0, 0.0, 0.0, 1.0];
    batch.jet_dir_aperture = [n0[0] as f32, n0[1] as f32, n0[2] as f32, 0.6];
    batch.spin = [axis[0] as f32, axis[1] as f32, axis[2] as f32, omega as f32];
    batch.lit = dust::LitWindow::new(sun, n0, axis, omega, dur).to_gpu();
    assert_eq!(batch.lit[3], dust::LIT_MODE_PERIODIC);
    emit(device, 9004, &batch);
    let bytes = read_back(
      device,
      clusters,
      true,
      capacity as u64 * core::mem::size_of::<DustCluster>() as u64,
    );
    let ring: &[DustCluster] = bytemuck::cast_slice(&bytes);
    let mut boundary_outliers = 0u32;
    let mut worst_dt = 0.0f64;
    for j in 0..batch.count {
      let gpu = &ring[(j & batch.ring_mask) as usize];
      let cpu = dust::emit_cluster(&batch, j);
      let dt_gpu = gpu.t0().to_f64() - T_START;
      assert!(dt_gpu >= 0.0 && dt_gpu <= dur + 1e-2, "dt {dt_gpu}");
      // every GPU cluster is emitted in daylight
      let n = dust::rotate_axis_angle(n0, axis, omega * dt_gpu);
      assert!(dot(n, sun) > -1e-3, "cluster {j} emitted in the dark");
      // GPU division / floor may differ by ulps: tiny time offsets, or (at an arc end) the
      // next lit arc
      let ddt = (dt_gpu - (cpu.t0().to_f64() - T_START)).abs();
      if ddt > 0.1 {
        boundary_outliers += 1;
        continue;
      }
      worst_dt = worst_dt.max(ddt);
      // r0 lies on the comet orbit at the GPU's own emission time
      let (r_exp, v_comet) = kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, dt_gpu);
      let r0 = gpu.r0().to_f64();
      let dr =
        ((r0[0] - r_exp[0]).powi(2) + (r0[1] - r_exp[1]).powi(2) + (r0[2] - r_exp[2]).powi(2))
          .sqrt();
      assert!(dr < 0.5, "r0 cluster {j}: |Δ| = {dr} m");
      // ejection velocity (relative to the comet) matches the reference
      let (cg, cc) = (gpu.v0().to_f64(), cpu.v0().to_f64());
      let (_, v_comet_cpu) =
        kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, cpu.t0().to_f64() - T_START);
      let dv = (0..3)
        .map(|k| ((cg[k] - v_comet[k]) - (cc[k] - v_comet_cpu[k])).powi(2))
        .sum::<f64>()
        .sqrt();
      assert!(dv < 1e-3, "ejection cluster {j}: |Δ| = {dv} m/s");
    }
    std::println!(
      "[dust gpu] lit-time emit parity: worst |Δt0| = {worst_dt:.4} s, {boundary_outliers} arc-boundary outliers"
    );
    assert!(
      boundary_outliers <= batch.count / 200,
      "{boundary_outliers} clusters off the reference lit arc"
    );
  });
}

/// Age tiers on the GPU: a batch emitted into a sub-ring at `ring_base` and evaluated with an age
/// band lands in the tier's part of the render buffer, and the band gate (min age, no fade) matches
/// the CPU reference cluster by cluster.
#[test]
fn gpu_dust_tier_sub_ring_and_age_band_match_reference() {
  with_dust_device(9004, |device, capacity, _, render| {
    let (base, sub) = (capacity / 2, capacity / 4);
    let mut batch = test_batch(256, 3600.0, 3, capacity);
    batch.ring_mask = sub - 1;
    let mut res = device
      .run_transient_compute_commands(|cmd| device.cmd_dust_emit(cmd, 9004, base, &batch, 0))
      .unwrap();
    res.cleanup(&device.device);
    // ages 10 d − [0, 1 h]: a band min in the middle of the batch culls about half of it
    let frame = frame_after(10.0).with_band((10.0 * 86400.0 - 1800.0) as f32, false);
    let first_slot = batch.first_index & batch.ring_mask;
    let mut res = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        let addr =
          device.cmd_dust_propagate(cmd, 9004, base, sub, first_slot, batch.count, &frame)?;
        assert!(addr > 0);
        Ok(())
      })
      .unwrap();
    res.cleanup(&device.device);
    let stride = core::mem::size_of::<DustRenderCluster>();
    let bytes = read_back(
      device,
      render,
      false,
      ((base + batch.count) as usize * stride) as u64,
    );
    let out: &[DustRenderCluster] = bytemuck::cast_slice(&bytes[base as usize * stride..]);
    let (mut culled, mut drawn) = (0, 0);
    for (i, gpu) in out.iter().enumerate() {
      let slot = (first_slot + i as u32) & batch.ring_mask;
      let cpu = dust::evaluate_cluster(&dust::emit_cluster(&batch, i as u32), slot, &frame);
      assert_eq!(
        gpu.age_id_dbeta_flux[3] > 0.0,
        cpu.age_id_dbeta_flux[3] > 0.0,
        "band gate differs at {i}: age {}",
        cpu.age_id_dbeta_flux[0]
      );
      if cpu.age_id_dbeta_flux[3] > 0.0 {
        drawn += 1;
        let rel =
          (gpu.age_id_dbeta_flux[3] - cpu.age_id_dbeta_flux[3]).abs() / cpu.age_id_dbeta_flux[3];
        assert!(rel < 1e-3, "flux {i}");
      } else {
        culled += 1;
      }
    }
    assert!(
      culled > 32 && drawn > 32,
      "band should split the batch: {culled} culled, {drawn} drawn"
    );
    // over-capacity sub-rings are refused
    let mut res = device
      .run_transient_commands(|cmd| {
        assert!(device.cmd_dust_propagate(cmd, 9004, capacity - 1, sub, 0, 1, &frame).is_err());
        Ok(())
      })
      .unwrap();
    res.cleanup(&device.device);
  });
}

/// The micro-layer color target is RGBA16F where supported (dust optical depth below 1/255 per
/// splat accumulates), RGBA8 otherwise or with `AETHERVK_DUST_8BIT=1`.
#[test]
fn micro_color_target_format_selection() {
  use crate::gpu_backends::vulkan::device::choose_micro_color_format as choose;
  use ash::vk::{Format, FormatFeatureFlags as F};
  let full = F::COLOR_ATTACHMENT | F::COLOR_ATTACHMENT_BLEND | F::SAMPLED_IMAGE;
  assert_eq!(choose(full, true, false), Format::R16G16B16A16_SFLOAT);
  assert_eq!(
    choose(F::COLOR_ATTACHMENT, true, false),
    Format::R8G8B8A8_UNORM,
    "no blend"
  );
  assert_eq!(
    choose(full, false, false),
    Format::R8G8B8A8_UNORM,
    "no transient input usage"
  );
  assert_eq!(choose(full, true, true), Format::R8G8B8A8_UNORM, "forced");
  // this machine's device: R16G16B16A16_SFLOAT is in the required-format table, unless the
  // 8-bit fallback is forced for the whole run
  let forced =
    aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_8BIT").is_some_and(|v| v.trim() == "1");
  with_dust_device(9005, |device, _, _, _| {
    let want = if forced {
      Format::R8G8B8A8_UNORM
    } else {
      Format::R16G16B16A16_SFLOAT
    };
    assert_eq!(device.micro_color_format(), want);
  });
}

/// `AETHERVK_DUST_8BIT=1` forces the RGBA8 micro target (own process under nextest).
#[test]
fn micro_color_target_8bit_override() {
  unsafe { std::env::set_var("AETHERVK_DUST_8BIT", "1") };
  with_dust_device(9006, |device, _, _, _| {
    assert_eq!(device.micro_color_format(), ash::vk::Format::R8G8B8A8_UNORM);
  });
}

/// `dust_lod.comp` against `dust::lod_evaluate`: a camera that sees about half of a propagated
/// ring; the indirect command stays within the budget, every drawn cluster has exactly `k` list
/// entries with `flux / k`, off-screen clusters none, and demand / tiles match the mirror.
#[test]
fn gpu_dust_lod_matches_reference() {
  use super::dust::DustLodDraw;
  with_dust_device(9006, |device, capacity, _, render| {
    let id = 9006;
    // 64 streams: short streaks between a stream's samples, point spreads for the first ones
    let mut batch = test_batch(4096, 86400.0, 0, capacity);
    batch.mass_params[3] = dust::batch_streams_word(6, false);
    emit(device, id, &batch);
    let frame = frame_after(10.0).with_streams(6);
    propagate(device, id, 0, batch.count, &frame);
    let rbytes = batch.count as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
    let before: alloc::vec::Vec<DustRenderCluster> =
      bytemuck::cast_slice(&read_back(device, render, false, rbytes)).to_vec();
    // orthographic view centred on the comet, half-width = median |x|: about half on screen
    let mut xs: alloc::vec::Vec<f32> = before.iter().map(|c| c.pos_size[0].abs()).collect();
    xs.sort_by(f32::total_cmp);
    let half = xs[xs.len() / 2].max(1.0);
    let units = 1e-3f32;
    let p = 1.0 / (half * units);
    let mut mvp = [0.0f32; 16];
    mvp[0] = p * units;
    mvp[5] = p * units;
    mvp[10] = 1e-12;
    mvp[15] = 1.0;
    // a 2160 px tall view: λ = 1 asks for more than the budget
    let params = [units, p, p, 2.0 / 2160.0];
    let exposure = 1.0e7f32;
    let budget = capacity * dust::CHILDREN_PER_CLUSTER;
    let render_addr = {
      let res = device.res.read();
      let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
      sys.render.as_ref().unwrap().address
    };
    // the LOD rewrites the flux: restore the propagated clusters before each run
    let run_lod = |lambda: f32| -> (DustLodDraw, f32) {
      propagate(device, id, 0, batch.count, &frame);
      let unit = {
        let res = device.res.read();
        let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
        let mut h = sys.lod_host.lock();
        h.lambda[0] = lambda;
        h.tile_unit()
      };
      let mut draw = None;
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          device.cmd_dust_lod_begin(cmd, id, 0, Default::default(), 0.0)?;
          // the readback of an earlier run must not move λ under the test
          {
            let res = device.res.read();
            let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
            sys.lod_host.lock().lambda[0] = lambda;
          }
          device.cmd_dust_pre_lod_barrier(cmd);
          draw = Some(device.cmd_dust_lod(
            cmd,
            id,
            0,
            0,
            capacity,
            render_addr,
            batch.count,
            exposure,
            mvp,
            params,
          )?);
          device.cmd_dust_post_propagate_barrier(cmd);
          device.cmd_dust_lod_end(cmd, id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      (draw.unwrap(), unit)
    };

    // λ up to 1 (production): the LOD takes this frame's share from this frame's demand, so the
    // first frame already fills 80–95 % of the budget with nothing dropped (no adjustment phase),
    // with the share of the mirror
    let (_, _) = run_lod(1.0);
    {
      let words = device.dust_lod_readback_latest(id).unwrap();
      let fill = words[5] as f64 / budget as f64;
      assert!((0.8..=0.95).contains(&fill), "first-frame fill {fill}");
      assert_eq!(words[1], words[5], "nothing dropped at the budget");
      let mut cpu = before.clone();
      let pc = dust::DustLodPushConstants {
        render: 0,
        header: 0,
        tiles: 0,
        list: 0,
        live_count: batch.count,
        budget,
        lambda: 1.0,
        tile_scale: 1.0,
        mvp,
        params,
      };
      let mut t = alloc::vec![0u32; dust::DUST_TILE_COUNT as usize];
      let mut l = alloc::vec::Vec::new();
      let cpu_out = dust::lod_evaluate(&mut cpu, &pc, 0, 0.0, &mut t, &mut l);
      let gpu_lambda = f32::from_bits(words[dust::LOD_HEADER_LAMBDA as usize]);
      std::println!(
        "[dust gpu] same-frame share: gpu λ {gpu_lambda} cpu λ {} fill {fill:.3}",
        cpu_out.lambda
      );
      assert!(
        (gpu_lambda / cpu_out.lambda - 1.0).abs() < 1e-3,
        "λ gpu {gpu_lambda} cpu {}",
        cpu_out.lambda
      );
      assert_eq!(
        words[dust::LOD_HEADER_ON_SCREEN as usize],
        cpu_out.on_screen
      );
      assert!(
        words[1] <= budget,
        "instances {} over budget {budget}",
        words[1]
      );
      let (lod_buffer, lod_bytes) = device.dust_lod_list_buffer(id).unwrap();
      let lod: alloc::vec::Vec<u32> =
        bytemuck::cast_slice(&read_back(device, lod_buffer, false, lod_bytes)).to_vec();
      let after: alloc::vec::Vec<DustRenderCluster> =
        bytemuck::cast_slice(&read_back(device, render, false, rbytes)).to_vec();
      let mut per_cluster = alloc::vec![0u32; batch.count as usize];
      let w0 = dust::LOD_LIST_WORD0 as usize;
      for &e in &lod[w0..w0 + words[1] as usize] {
        per_cluster[(e & ((1 << dust::LOD_CLUSTER_BITS) - 1)) as usize] += 1;
      }
      for i in 0..batch.count as usize {
        let k = per_cluster[i];
        if k == 0 {
          assert_eq!(
            after[i].age_id_dbeta_flux[3], 0.0,
            "dropped cluster {i} still drawable"
          );
        } else {
          // flux per child × k = the cluster flux, a streak's visible fraction of it
          let total = after[i].age_id_dbeta_flux[3] * k as f32;
          let r = total / before[i].age_id_dbeta_flux[3];
          assert!(r > 0.0 && r <= 1.0 + 1e-5, "cluster {i}: {r}");
        }
      }
    }

    // under budget: exact parity with the mirror (λ chosen on the mirror: below 90 % of it)
    let lambda = [0.01f32, 1e-3, 1e-4, 1e-5]
      .into_iter()
      .find(|&l| {
        let mut c = before.clone();
        let mut pc = dust::DustLodPushConstants {
          render: 0,
          header: 0,
          tiles: 0,
          list: 0,
          live_count: batch.count,
          budget,
          lambda: l,
          tile_scale: 1.0,
          mvp,
          params,
        };
        pc.budget = u32::MAX;
        let mut t = alloc::vec![0u32; dust::DUST_TILE_COUNT as usize];
        let mut l2 = alloc::vec::Vec::new();
        (dust::lod_evaluate(&mut c, &pc, 0, 0.0, &mut t, &mut l2).attempted as f64)
          < 0.9 * budget as f64
      })
      .unwrap();
    let (draw, unit) = run_lod(lambda);
    assert!(draw.indirect.is_some());

    // CPU mirror on the same clusters
    let mut cpu = before.clone();
    let pc = dust::DustLodPushConstants {
      render: 0,
      header: 0,
      tiles: 0,
      list: 0,
      live_count: batch.count,
      budget,
      lambda,
      tile_scale: exposure / unit,
      mvp,
      params,
    };
    let mut cpu_tiles = alloc::vec![0u32; dust::DUST_TILE_COUNT as usize];
    let mut cpu_list = alloc::vec::Vec::new();
    let cpu_out = dust::lod_evaluate(&mut cpu, &pc, 0, 0.0, &mut cpu_tiles, &mut cpu_list);
    let mut cpu_k = alloc::vec![0u32; batch.count as usize];
    for &e in &cpu_list {
      cpu_k[(e & ((1 << dust::LOD_CLUSTER_BITS) - 1)) as usize] += 1;
    }

    let words = device.dust_lod_readback_latest(id).unwrap();
    let (lod_buffer, lod_bytes) = device.dust_lod_list_buffer(id).unwrap();
    let lod: alloc::vec::Vec<u32> =
      bytemuck::cast_slice(&read_back(device, lod_buffer, false, lod_bytes)).to_vec();
    assert_eq!(
      &words[..],
      &lod[..dust::LOD_READBACK_WORDS as usize],
      "readback copy"
    );
    let instances = words[1];
    assert_eq!(words[0], 6, "vertexCount");
    assert!(instances <= budget);
    assert!(instances > 0);
    let rel = |a: u32, b: u32| (a as f64 - b as f64).abs() / (b as f64).max(1.0);
    assert!(
      rel(instances, cpu_out.instances) < 0.01,
      "instances {instances} vs {}",
      cpu_out.instances
    );
    assert!(
      rel(words[4], cpu_out.demand) < 0.01,
      "demand {} vs {}",
      words[4],
      cpu_out.demand
    );
    assert!(rel(words[5], cpu_out.attempted) < 0.01);

    let after: alloc::vec::Vec<DustRenderCluster> =
      bytemuck::cast_slice(&read_back(device, render, false, rbytes)).to_vec();
    let list =
      &lod[dust::LOD_LIST_WORD0 as usize..dust::LOD_LIST_WORD0 as usize + instances as usize];
    let mut per_cluster = alloc::vec![0u32; batch.count as usize];
    let mut children_seen = alloc::vec![0u64; batch.count as usize];
    for &e in list {
      let c = (e & ((1 << dust::LOD_CLUSTER_BITS) - 1)) as usize;
      assert!(c < batch.count as usize);
      per_cluster[c] += 1;
      children_seen[c] |= 1u64 << ((e >> dust::LOD_CLUSTER_BITS).min(63));
    }
    let (mut on, mut off, mut mismatched) = (0, 0, 0);
    for i in 0..batch.count as usize {
      let k = per_cluster[i];
      if k == 0 {
        off += 1;
        assert_eq!(
          after[i].age_id_dbeta_flux[3], 0.0,
          "cluster {i}: not drawn, flux kept"
        );
        continue;
      }
      on += 1;
      // flux per child × k = the cluster flux, a streak's visible fraction of it
      let total = after[i].age_id_dbeta_flux[3] * k as f32;
      let r = total / before[i].age_id_dbeta_flux[3];
      assert!(r > 0.0 && r <= 1.0 + 1e-5, "cluster {i}: {r}");
      if k < 64 {
        assert_eq!(
          children_seen[i],
          (1u64 << k) - 1,
          "cluster {i}: children 0..{k}"
        );
      }
      let ck = cpu_k[i];
      let cf = cpu[i].age_id_dbeta_flux[3];
      if ck != k || ((after[i].age_id_dbeta_flux[3] - cf) / cf).abs() > 1e-3 {
        if mismatched < 5 {
          std::println!(
            "[dust gpu] cluster {i}: gpu k {k} cpu k {ck} spread {}",
            before[i].pos_size[3]
          );
        }
        mismatched += 1;
      }
    }
    std::println!(
      "[dust gpu] instances {instances} attempted {} budget {budget} cpu {cpu_out:?} lambda {lambda}",
      words[5]
    );
    assert!(
      on > 100 && off > 100,
      "on {on} off {off}: the view must cut the ring"
    );
    assert!(
      mismatched * 100 <= on,
      "{mismatched} of {on} clusters differ from the mirror"
    );

    let t0 = dust::LOD_TILE_WORD0 as usize;
    let gpu_tiles = &words[t0..t0 + dust::DUST_TILE_COUNT as usize];
    let sum = |t: &[u32]| t.iter().map(|&v| v as f64).sum::<f64>();
    let (gs, cs) = (sum(gpu_tiles), sum(&cpu_tiles));
    assert!(
      cs > 0.0 && (gs / cs - 1.0).abs() < 0.02,
      "tiles {gs} vs {cs}"
    );
    let (gw, cw) = (
      dust::white_point_from_tiles(gpu_tiles, unit).unwrap(),
      dust::white_point_from_tiles(&cpu_tiles, unit).unwrap(),
    );
    assert!((gw / cw - 1.0).abs() < 0.05, "white {gw} vs {cw}");
    std::println!(
      "[dust gpu] LOD: {on} on / {off} off screen, {instances} instances (budget {budget}), {mismatched} k mismatches, white {gw:.3e}"
    );
  });
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoupling: particles are not children of the comet
// ─────────────────────────────────────────────────────────────────────────────

/// Scene root → comet subtree frame (micro, layer 1) → comet body → jet entity carrying a real
/// `ParticleSystemComponent`, a camera child of root near the comet, and a dust history ticked
/// from the comet's own orbit. Returns (scene, camera, subtree, body, jet, t_now).
fn decoupling_scene(
  render_frontend: crate::gpu::RenderFrontend,
  handle: crate::gpu::RenderDeviceHandle,
) -> (
  crate::scene::Scene,
  EntityId,
  EntityId,
  EntityId,
  EntityId,
  f64,
) {
  use crate::{
    scene::{
      CameraComponent, CameraProjection, HighResTransformComponent, ReferenceFrameComponent,
      ReferenceFrameType, Scene, TransformComponent,
      dust::{DustEmitConfig, JetState},
      particles::{ParticleSystemComponent, ParticleSystemDrawParams, ParticleSystemEmitParams},
    },
    simulation::texture_cache::TextureCache,
  };
  use aethervk_oshal_rlib::math::vector::{
    vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat, vec4f64::Quat64,
  };
  const AU_TO_KM: f64 = 149_597_870.7;
  let scene = Scene::new(alloc::sync::Arc::new(parking_lot::RwLock::new(
    TextureCache::new("dust_decoupling"),
  )));
  scene.register_all_crate_components();
  let root = scene.spawn_entity("root");
  scene
    .add_component(
      root,
      ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Macro,
        scale: 1.0,
        soi_radius: f32::MAX,
        depth_layer: 0,
      },
    )
    .unwrap();
  // comet heliocentric state at t_now, split like `compute_macro_and_residual`: frame on an f32 AU
  // grid point, body residual in km
  let t_now = 10.0 * 86400.0;
  let (rc0, vc0) = comet_state();
  let (rc, _) = kepler::propagate_f64(rc0, vc0, SUN_MU_M3_S2, t_now);
  let frame_au = [
    (rc[0] / AU_M) as f32,
    (rc[1] / AU_M) as f32,
    (rc[2] / AU_M) as f32,
  ];
  let subtree = scene.spawn_entity("comet_subtree");
  scene.set_parent(subtree, Some(root));
  scene
    .add_component(
      subtree,
      TransformComponent {
        position: Vec3f32::from_components(frame_au[0], frame_au[1], frame_au[2]),
        rotation: Quat::identity(),
        scale: Vec3f32::one(),
      },
    )
    .unwrap();
  scene
    .add_component(
      subtree,
      ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Micro,
        scale: (1.0 / AU_TO_KM) as f32,
        soi_radius: 1.0,
        depth_layer: 1,
      },
    )
    .unwrap();
  let body = scene.spawn_entity("comet_body");
  scene.set_parent(body, Some(subtree));
  let residual_km = [
    (rc[0] / 1e3 - frame_au[0] as f64 * AU_TO_KM) as f32,
    (rc[1] / 1e3 - frame_au[1] as f64 * AU_TO_KM) as f32,
    (rc[2] / 1e3 - frame_au[2] as f64 * AU_TO_KM) as f32,
  ];
  scene
    .add_component(
      body,
      TransformComponent {
        position: Vec3f32::from_components(residual_km[0], residual_km[1], residual_km[2]),
        rotation: Quat::identity(),
        scale: Vec3f32::one(),
      },
    )
    .unwrap();
  let jet = scene.spawn_entity("jet");
  scene.set_parent(jet, Some(body));
  scene
    .add_component(
      jet,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 2.0),
        rotation: Quat::identity(),
        scale: Vec3f32::one(),
      },
    )
    .unwrap();
  let ps = ParticleSystemComponent::new(
    render_frontend,
    handle,
    jet,
    bytemuck::Zeroable::zeroed(),
    ParticleSystemDrawParams {
      stream_color: [1.0, 0.8, 0.5, 1.0],
    },
    (30.0 * 86400.0 * 1e6) as _,
  )
  .unwrap();
  // a history ticked from the comet's own orbit (hourly, 2 days), all submitted
  {
    let cfg = DustEmitConfig {
      q_dust_kgs: 1.5e-3,
      ttl_s: 30.0 * 86400.0,
      dist: SizeDistribution::from_diameter_um(100.0),
      diameter_um: 100.0,
      density_gcm3: 0.533,
      beta_ref: 0.0213,
      v_mean: 2.0,
      v_std: 0.5,
      jet_dir: [0.0, 0.0, 1.0],
      aperture_rad: 0.5,
      seed: 42,
    };
    let jet_at = |t: f64| -> Option<JetState> {
      let (r, v) = kepler::propagate_f64(rc0, vc0, SUN_MU_M3_S2, t);
      let n = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
      Some(JetState {
        t_s: t,
        r_m: r,
        v_ms: v,
        rot: [0.0, 0.0, 0.0, 1.0],
        site_normal: [-r[0] / n, -r[1] / n, -r[2] / n],
        spin: Some([0.0, 0.0, 1.0, 0.0]),
        site_offset_m: [0.0; 3],
      })
    };
    let mut sys = ps.dust.lock();
    let mut t = t_now - 2.0 * 86400.0;
    let mut seq = 1;
    while t <= t_now + 1.0 {
      sys.tick(t, &jet_at, &|_| cfg);
      sys.mark_submitted(seq);
      seq += 1;
      t += 3600.0;
    }
  }
  scene.add_component(jet, ps).unwrap();
  // camera 500 km from the comet, child of root (heliocentric, AU, f64)
  let cam = scene.spawn_entity("camera");
  scene.set_parent(cam, Some(root));
  scene
    .add_component(
      cam,
      HighResTransformComponent {
        position: Vec3f64::from_components((rc[0] + 5.0e5) / AU_M, rc[1] / AU_M, rc[2] / AU_M),
        rotation: Quat64::identity(),
        scale: Vec3f32::one(),
      },
    )
    .unwrap();
  scene
    .add_component(
      cam,
      CameraComponent {
        projection: CameraProjection::Perspective {
          fov: 60.0f32.to_radians(),
          aspect_ratio: 16.0 / 9.0,
          near: 1e-9,
          far: 10.0,
        },
        focus_distance: 1.0,
      },
    )
    .unwrap();
  scene.set_sim_time_s(t_now);
  (scene, cam, subtree, body, jet, t_now)
}

/// Draw translation `rte / units + eye` (heliocentric anchor) and evaluation frame of each call.
fn dust_call_signature(
  scene: &crate::scene::Scene,
  cam: EntityId,
) -> alloc::vec::Vec<(u32, [f64; 3], [f64; 3], DustFrame)> {
  let p = scene.global_transform_f64(cam).unwrap().position;
  let eye = [p.x() * AU_M, p.y() * AU_M, p.z() * AU_M];
  scene
    .dust_draw_calls(cam)
    .into_iter()
    .flat_map(|(layer, calls)| {
      calls.into_iter().map(move |c| {
        let u = c.units_per_m;
        let world = [
          c.rte_position[0] / u + eye[0],
          c.rte_position[1] / u + eye[1],
          c.rte_position[2] / u + eye[2],
        ];
        (layer, world, c.rte_position, c.state.frame)
      })
    })
    .collect()
}

/// The comet and jet entity transforms never enter the dust draw: teleporting / spinning the comet
/// body, moving its frame or the jet entity leaves every dust draw call bit-identical. The
/// translation is `anchor − eye` (heliocentric), and moving only the camera shifts it by exactly
/// the camera displacement.
#[test]
fn dust_draw_ignores_comet_and_jet_transforms() {
  use crate::scene::TransformComponent;
  use aethervk_oshal_rlib::math::vector::{vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat};
  setup_assets_dir();
  let (_pool, render_frontend, handle, _) = setup_render_frontend_for_tests(false);
  let (scene, cam, subtree, body, jet, t_now) = decoupling_scene(render_frontend, handle);

  let before = dust_call_signature(&scene, cam);
  assert!(!before.is_empty(), "the history must be drawable");
  for (layer, world, _, frame) in &before {
    assert_eq!(
      *layer, 1,
      "drawn in the comet's layer (occlusion against the nucleus)"
    );
    let a = frame.anchor_m();
    let d =
      ((world[0] - a[0]).powi(2) + (world[1] - a[1]).powi(2) + (world[2] - a[2]).powi(2)).sqrt();
    assert!(
      d < 1e-3,
      "rte / units + eye must be the heliocentric anchor ({d} m off)"
    );
    assert!((frame.t_now_s() - t_now).abs() < 1e-6);
  }

  // teleport + spin the comet body, move its frame, move the jet on the nucleus
  scene
    .with_component_mut(body, |t: &mut TransformComponent| {
      t.position = t.position + Vec3f32::from_components(1.0e5, -3.0e4, 7.0e3);
      t.rotation = Quat::from_components(0.0, 0.0, 0.7071068, 0.7071068);
    })
    .unwrap();
  scene
    .with_component_mut(subtree, |t: &mut TransformComponent| {
      t.position = t.position + Vec3f32::from_components(0.25, 0.0, 0.0);
    })
    .unwrap();
  scene
    .with_component_mut(jet, |t: &mut TransformComponent| {
      t.position = Vec3f32::from_components(1.5, -1.0, 0.3);
    })
    .unwrap();
  let after = dust_call_signature(&scene, cam);
  assert_eq!(before.len(), after.len());
  for (b, a) in before.iter().zip(after.iter()) {
    assert_eq!(b.2, a.2, "rte moved with the comet");
    assert_eq!(b.3, a.3, "frame moved with the comet");
  }

  // the camera alone moves the draw: by exactly its displacement
  let delta_au = Vec3f64::from_components(2.0e-6, -1.0e-6, 5.0e-7);
  scene
    .with_component_mut(cam, |t: &mut crate::scene::HighResTransformComponent| {
      t.position = t.position + delta_au;
    })
    .unwrap();
  let moved = dust_call_signature(&scene, cam);
  for (b, m) in before.iter().zip(moved.iter()) {
    let u = 1e-3;
    for k in 0..3 {
      let expect = b.2[k] - [delta_au.x(), delta_au.y(), delta_au.z()][k] * AU_M * u;
      assert!(
        (m.2[k] - expect).abs() < 1e-4,
        "axis {k}: {} vs {expect}",
        m.2[k]
      );
    }
    assert_eq!(b.3, m.3, "the evaluation does not depend on the camera");
  }
}

/// The logic thread stores the newest emission (tₙ) in the host state before it commits the
/// transforms of tₙ: an extraction in between used to draw tₙ dust at the comet of tₙ₋₁ (whole
/// cloud offset by v·Δt). Now dust is evaluated at the committed scene time, whatever the host
/// state's own time.
#[test]
fn dust_frame_time_is_the_committed_scene_time() {
  setup_assets_dir();
  let (_pool, render_frontend, handle, _) = setup_render_frontend_for_tests(false);
  let (scene, cam, _, _, _, t_now) = decoupling_scene(render_frontend, handle);
  // scene committed one tick (10 min) behind the host state
  let t_prev = t_now - 600.0;
  scene.set_sim_time_s(t_prev);
  let behind = dust_call_signature(&scene, cam);
  assert!(!behind.is_empty());
  for (_, _, _, f) in &behind {
    assert!(
      (f.t_now_s() - t_prev).abs() < 1e-6,
      "evaluated at {} not {t_prev}",
      f.t_now_s()
    );
  }
  scene.set_sim_time_s(t_now);
  for (_, _, _, f) in dust_call_signature(&scene, cam) {
    assert!((f.t_now_s() - t_now).abs() < 1e-6);
  }
  // clusters emitted after the scene time are culled (negative age), older ones evaluated at it
  let s = behind[0].3;
  let c_new = dust::DustCluster {
    r0_t0_hi: [1.0, 0.0, 0.0, (t_now - 60.0) as f32],
    r0_t0_lo: [0.0; 4],
    v0_hi_beta: [0.0, 0.0, 0.0, 0.0],
    v0_lo_mass: [0.0, 0.0, 0.0, 1.0],
    misc: [0.0, 1.0, 1.0, 0.0],
  };
  assert_eq!(
    dust::evaluate_cluster(&c_new, 0, &s).age_id_dbeta_flux[3],
    0.0
  );
}

/// GPU path of the anchor invariant: the same ring propagated relative to two anchors 1e6 km apart
/// lands on the same heliocentric points (`render.pos + A`).
#[test]
fn gpu_dust_world_positions_ignore_the_anchor() {
  with_dust_device(9007, |device, capacity, _, render| {
    let batch = test_batch(1024, 3600.0, 0, capacity);
    emit(device, 9007, &batch);
    let f0 = frame_after(10.0);
    let a0 = f0.anchor_m();
    let a1 = [a0[0] + 1.0e9, a0[1] - 3.0e8, a0[2] + 1.0e8];
    let f1 = DustFrame::new(a1, f0.t_now_s(), [0.0, 0.0, 0.0, 1.0], 1.0e9);
    let bytes = batch.count as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
    let mut worlds = alloc::vec::Vec::new();
    for (f, a) in [(f0, a0), (f1, a1)] {
      propagate(device, 9007, 0, batch.count, &f);
      let out: alloc::vec::Vec<DustRenderCluster> =
        bytemuck::cast_slice(&read_back(device, render, false, bytes)).to_vec();
      worlds.push(
        out
          .iter()
          .map(|c| {
            [
              a[0] + c.pos_size[0] as f64,
              a[1] + c.pos_size[1] as f64,
              a[2] + c.pos_size[2] as f64,
            ]
          })
          .collect::<alloc::vec::Vec<_>>(),
      );
    }
    let mut worst = 0.0f64;
    for (i, (w0, w1)) in worlds[0].iter().zip(worlds[1].iter()).enumerate() {
      let d = ((w0[0] - w1[0]).powi(2) + (w0[1] - w1[1]).powi(2) + (w0[2] - w1[2]).powi(2)).sqrt();
      let far =
        ((w0[0] - a1[0]).powi(2) + (w0[1] - a1[1]).powi(2) + (w0[2] - a1[2]).powi(2)).sqrt();
      worst = worst.max(d);
      assert!(
        d < 2.0 + 2e-6 * far,
        "cluster {i}: {d} m between anchors (|r − A| {far} m)"
      );
    }
    std::println!("[dust gpu] anchor invariance: worst {worst:.2} m over 1e6 km");
  });
}

/// Extracts `(slot, helio_m)` of every particle of an NDJSON trace line (test-only scanner).
fn trace_particles(line: &str) -> alloc::vec::Vec<(u32, [f64; 3])> {
  let mut out = alloc::vec::Vec::new();
  let mut rest = line;
  while let Some(i) = rest.find("\"slot\":") {
    rest = &rest[i + 7..];
    let slot: u32 = rest[..rest.find(',').unwrap()].parse().unwrap();
    let h = rest.find("\"helio_m\":[").unwrap();
    let body = &rest[h + 11..];
    let v: alloc::vec::Vec<f64> =
      body[..body.find(']').unwrap()].split(',').map(|x| x.parse().unwrap()).collect();
    out.push((slot, [v[0], v[1], v[2]]));
  }
  out
}

/// The trace recorder end to end on the GPU path: two traced frames 1 h apart write two NDJSON lines
/// once their readback slots are folded. The traced particles are the drawn (GPU-evaluated)
/// clusters with stable slots: their heliocentric positions match the reference evaluation, and
/// their offset from the nucleus grows between the frames (what the trace analysis looks for).
#[test]
fn gpu_dust_trace_records_particles_over_time() {
  extern crate std;
  use crate::gpu::dust_trace::{TraceFrameMeta, TraceTierMeta, sample_slots};
  let path = std::env::temp_dir().join(alloc::format!(
    "aethervk_dust_trace_{}.ndjson",
    std::process::id()
  ));
  let _ = std::fs::remove_file(&path);
  // SAFETY: set before any other thread reads the environment (nextest: one test per process)
  unsafe { std::env::set_var("AETHERVK_DUST_TRACE", &path) };
  let id = 9008;
  let mut expected: alloc::vec::Vec<(DustFrame, [f64; 3])> = alloc::vec::Vec::new();
  let mut batch_used = None;
  with_dust_device(id, |device, capacity, _, _| {
    let batch = test_batch(4096, 3600.0, 0, capacity);
    emit(device, id, &batch);
    batch_used = Some(batch);
    let render_addr = {
      let res = device.res.read();
      let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
      sys.render.as_ref().unwrap().address
    };
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1e-9;
    mvp[5] = 1e-9;
    mvp[10] = 1e-12;
    mvp[15] = 1.0;
    // consistent with mvp = P · scale(units): P = 1e-6 per km
    let params = [1e-3, 1e-6, 1e-6, 2.0 / 720.0];
    let run_frame = |traced: Option<(DustFrame, [f64; 3])>| {
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          device.cmd_dust_lod_begin(cmd, id, 0, Default::default(), 0.0)?;
          if let Some((frame, nucleus)) = traced {
            device.cmd_dust_propagate(cmd, id, 0, capacity, 0, batch.count, &frame)?;
            device.cmd_dust_pre_lod_barrier(cmd);
            device.cmd_dust_lod(
              cmd,
              id,
              0,
              0,
              capacity,
              render_addr,
              batch.count,
              1.0,
              mvp,
              params,
            )?;
            device.cmd_dust_post_propagate_barrier(cmd);
            let anchor = frame.anchor_m();
            let meta = TraceFrameMeta {
              frame: 0,
              wall_us: 0,
              sim_time_s: frame.t_now_s(),
              system: id,
              eye_m: [anchor[0] + 1e9, anchor[1], anchor[2]],
              nucleus_m: Some(nucleus),
              viewport: [1280, 720],
              tiers: alloc::vec![TraceTierMeta {
                tier: 0,
                anchor_m: anchor,
                t_now_s: frame.t_now_s(),
                rte_position: [0.0; 3],
                units_per_m: 1e-3,
                mvp,
                samples: sample_slots(0, batch.count, capacity),
              }],
            };
            device.cmd_dust_trace(cmd, id, meta, &[0])?;
          }
          device.cmd_dust_lod_end(cmd, id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
    };
    for hours in [24.0, 25.0] {
      let f = frame_after(hours / 24.0);
      let nucleus = f.anchor_m();
      expected.push((f, nucleus));
      run_frame(Some((f, nucleus)));
      // fold the readback slot: the line is written LOD_READBACK_SLOTS frames later
      for _ in 0..super::dust::LOD_READBACK_SLOTS {
        run_frame(None);
      }
    }
  });
  let text = std::fs::read_to_string(&path).expect("trace file written");
  let _ = std::fs::remove_file(&path);
  let lines: alloc::vec::Vec<&str> = text.lines().collect();
  assert_eq!(lines.len(), 2, "one line per traced frame");
  let batch = batch_used.unwrap();
  let recs: alloc::vec::Vec<_> = lines.iter().map(|l| trace_particles(l)).collect();
  assert!(
    recs[0].len() > 10,
    "only {} traced particles",
    recs[0].len()
  );
  let mut grew = 0;
  let mut common = 0;
  for (slot, h0) in &recs[0] {
    // parity with the reference evaluation
    for (k, rec) in recs.iter().enumerate() {
      if let Some((_, h)) = rec.iter().find(|(s, _)| s == slot) {
        let (f, _) = expected[k];
        let c = dust::evaluate_cluster(&dust::emit_cluster(&batch, *slot), *slot, &f);
        let a = f.anchor_m();
        let r = [
          a[0] + c.pos_size[0] as f64,
          a[1] + c.pos_size[1] as f64,
          a[2] + c.pos_size[2] as f64,
        ];
        let d = ((h[0] - r[0]).powi(2) + (h[1] - r[1]).powi(2) + (h[2] - r[2]).powi(2)).sqrt();
        assert!(d < 5.0, "slot {slot} record {k}: {d} m from the reference");
      }
    }
    if let Some((_, h1)) = recs[1].iter().find(|(s, _)| s == slot) {
      common += 1;
      let off = |h: &[f64; 3], n: [f64; 3]| {
        ((h[0] - n[0]).powi(2) + (h[1] - n[1]).powi(2) + (h[2] - n[2]).powi(2)).sqrt()
      };
      if off(h1, expected[1].1) > off(h0, expected[0].1) {
        grew += 1;
      }
    }
  }
  assert!(
    common > 10,
    "the same particles must be traced in both frames ({common})"
  );
  assert!(
    grew * 10 >= common * 9,
    "offset from the nucleus grew for {grew} of {common}"
  );
}

/// Streaks on the GPU at telescope zoom: a 34 km field inside the coma of a 64-stream batch on a
/// spinning nucleus. `dust_propagate.comp` writes the stream word (slot, shift, live) and
/// `dust_lod.comp` links, clips and splits the same streaks as `dust::lod_evaluate`: same
/// per-child flux (visible fraction / k), so the same children counts.
#[test]
fn gpu_dust_lod_draws_streaks_like_the_reference_at_telescope_zoom() {
  with_dust_device(9010, |device, capacity, _, render| {
    let id = 9010;
    let mut batch = test_batch(4096, 86400.0, 0, capacity);
    batch.mass_params[3] = dust::batch_streams_word(6, false);
    emit(device, id, &batch);
    let frame = frame_after(2.0).with_streams(6);
    propagate(device, id, 0, batch.count, &frame);
    let rbytes = batch.count as u64 * core::mem::size_of::<DustRenderCluster>() as u64;
    let before: alloc::vec::Vec<DustRenderCluster> =
      bytemuck::cast_slice(&read_back(device, render, false, rbytes)).to_vec();
    for (i, c) in before.iter().enumerate() {
      if c.age_id_dbeta_flux[3] > 0.0 {
        assert_eq!(
          c.age_id_dbeta_flux[1].to_bits(),
          dust::render_word(i as u32, 6),
          "render {i}"
        );
      }
    }
    // the view: 34 km around the middle of the coma
    let live: alloc::vec::Vec<&DustRenderCluster> =
      before.iter().filter(|c| c.age_id_dbeta_flux[3] > 0.0).collect();
    let mut xs: alloc::vec::Vec<f32> = live.iter().map(|c| c.pos_size[0]).collect();
    let mut ys: alloc::vec::Vec<f32> = live.iter().map(|c| c.pos_size[1]).collect();
    xs.sort_by(|a, b| a.total_cmp(b));
    ys.sort_by(|a, b| a.total_cmp(b));
    let (cx, cy) = (xs[xs.len() / 2], ys[ys.len() / 2]);
    let half = 17.0e3f32;
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1.0 / half;
    mvp[5] = 1.0 / half;
    mvp[10] = 1e-12;
    mvp[12] = -cx / half;
    mvp[13] = -cy / half;
    mvp[15] = 1.0;
    let params = [1.0, 1.0 / half, 1.0 / half, 2.0 / 720.0];
    let lambda = 0.05;
    let render_addr = {
      let res = device.res.read();
      let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
      sys.lod_host.lock().lambda[0] = lambda;
      sys.render.as_ref().unwrap().address
    };
    // tracers on, a flow clock: the header words dust.vert reads
    let flags = dust::DUST_VIEW_TRACERS | dust::DUST_VIEW_FLOW;
    // solar gravity at ~5 AU: the beta extent enters the footprints (dust::dust_extent)
    const SUN_G: f32 = 2.0e-4;
    let mut clock = dust::DustFlowClock::default();
    clock.set_speed(10.0);
    clock.sync(0, 0);
    clock.sync(1_000_000, 3_600_000_000);
    let flow = dust::DustFlowUniform::new(&clock, 4.0e8 + 0.123);
    let mut draw = None;
    let mut cmds = device
      .run_transient_commands(|cmd| {
        device.cmd_dust_pre_propagate_barrier(cmd);
        device.cmd_dust_lod_begin(cmd, id, flags, flow, SUN_G)?;
        {
          let res = device.res.read();
          let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
          sys.lod_host.lock().lambda[0] = lambda;
        }
        device.cmd_dust_pre_lod_barrier(cmd);
        draw = Some(device.cmd_dust_lod(
          cmd,
          id,
          0,
          0,
          capacity,
          render_addr,
          batch.count,
          1.0,
          mvp,
          params,
        )?);
        device.cmd_dust_post_propagate_barrier(cmd);
        device.cmd_dust_lod_end(cmd, id)?;
        Ok(())
      })
      .unwrap();
    cmds.cleanup(&device.device);
    let words = device.dust_lod_readback_latest(id).unwrap();
    let budget = capacity * dust::CHILDREN_PER_CLUSTER;
    assert!(
      words[5] <= budget,
      "test must stay under budget ({} > {budget})",
      words[5]
    );
    let after: alloc::vec::Vec<DustRenderCluster> =
      bytemuck::cast_slice(&read_back(device, render, false, rbytes)).to_vec();

    let mut cpu = before.clone();
    let pc = dust::DustLodPushConstants {
      render: 0,
      header: 0,
      tiles: 0,
      list: 0,
      live_count: batch.count,
      budget,
      lambda,
      tile_scale: 1.0,
      mvp,
      params,
    };
    let mut tiles = alloc::vec![0u32; dust::DUST_TILE_COUNT as usize];
    let mut list = alloc::vec::Vec::new();
    let out = dust::lod_evaluate(&mut cpu, &pc, flags, SUN_G, &mut tiles, &mut list);
    // header words for dust.vert: the instance list (written by the LOD), clock and flags
    let draw = draw.unwrap();
    let h = dust::LOD_HEADER_LIST as usize;
    assert_eq!(
      words[h] as u64 | (words[h + 1] as u64) << 32,
      draw.list,
      "list address"
    );
    assert_eq!(words[dust::LOD_HEADER_FLOW_SPEED as usize], flow.speed.to_bits());
    assert_eq!(words[dust::LOD_HEADER_FLAGS as usize], flags);
    assert_eq!(words[dust::LOD_HEADER_T_HI as usize], flow.t_hi.to_bits());
    assert_eq!(words[dust::LOD_HEADER_T_LO as usize], flow.t_lo.to_bits());
    // T = t_sim + (K − 1)·1 h
    assert!(flow.speed == 10.0 && flow.t_lo != 0.0);
    assert!(((flow.t_hi as f64 + flow.t_lo as f64) - (4.0e8 + 0.123 + 9.0 * 3600.0)).abs() < 1e-3);
    assert_eq!(
      f32::from_bits(words[dust::LOD_HEADER_SUN_G as usize]),
      SUN_G
    );
    // tracer instances: the same clusters as the mirror
    let (lod_buffer, lod_bytes) = device.dust_lod_list_buffer(id).unwrap();
    let lod: alloc::vec::Vec<u32> =
      bytemuck::cast_slice(&read_back(device, lod_buffer, false, lod_bytes)).to_vec();
    let w0 = dust::LOD_LIST_WORD0 as usize;
    let tracers = |l: &[u32]| {
      let mut t: alloc::vec::Vec<u32> = l
        .iter()
        .filter(|&&e| e >> dust::LOD_CLUSTER_BITS == dust::TRACER_CHILD)
        .map(|&e| e & dust::RENDER_SLOT_MASK)
        .collect();
      t.sort_unstable();
      t
    };
    let (gt, ct) = (tracers(&lod[w0..w0 + words[1] as usize]), tracers(&list));
    std::println!("[dust gpu] tracers: gpu {} cpu {}", gt.len(), ct.len());
    assert!(!ct.is_empty(), "tracers in view");
    assert_eq!(gt, ct, "tracer clusters");
    let (mut streaks, mut mismatch) = (0, 0);
    for i in 0..batch.count as usize {
      let (g, c) = (after[i].age_id_dbeta_flux[3], cpu[i].age_id_dbeta_flux[3]);
      if c > 0.0 && dust::streak_pred(&before, i).is_some() {
        streaks += 1;
      }
      let same = (g == 0.0 && c == 0.0) || (g * c > 0.0 && ((g - c) / c).abs() < 1e-3);
      if !same {
        mismatch += 1;
      }
    }
    std::println!(
      "[dust gpu] telescope zoom: {streaks} streaks in view, {} children (cpu), gpu {} attempted, {mismatch} mismatches",
      out.attempted,
      words[5]
    );
    assert!(streaks > 50, "streaks must cross the view ({streaks})");
    assert!(
      mismatch * 100 <= batch.count as usize,
      "{mismatch} clusters differ from the reference"
    );
    let rel = (words[5] as f64 - out.attempted as f64).abs() / out.attempted.max(1) as f64;
    assert!(
      rel < 0.01,
      "attempted children: gpu {} vs cpu {}",
      words[5],
      out.attempted
    );
  });
}
