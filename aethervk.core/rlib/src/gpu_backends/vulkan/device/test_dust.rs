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
    .run_transient_compute_commands(|cmd| device.cmd_dust_emit(cmd, id, batch, 0))
    .unwrap();
  res.cleanup(&device.device);
}

fn propagate(device: &Device, id: u64, first_slot: u32, live: u32, frame: &DustFrame) {
  let mut res = device
    .run_transient_commands(|cmd| {
      device.cmd_dust_pre_propagate_barrier(cmd);
      device.cmd_dust_propagate(cmd, id, first_slot, live, frame)?;
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
    let batch = test_batch(512, 1800.0, capacity - 100, capacity);
    emit(device, 9001, &batch);
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
        (gpu.t0().to_f64() - cpu.t0().to_f64()).abs() < 1e-3,
        "t0 slot {slot}"
      );
      // position: GPU df64 Kepler vs CPU df64 Kepler (cm level; only `/` and `sqrt` may differ)
      let dr = {
        let (a, b) = (gpu.r0().to_f64(), cpu.r0().to_f64());
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
      };
      worst_r = worst_r.max(dr);
      assert!(dr < 0.5, "r0 slot {slot}: |Δ| = {dr} m");
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
      for k in 0..4 {
        assert!(
          rel(gpu.misc[k], cpu.misc[k]) < 1e-4,
          "misc[{k}] slot {slot}"
        );
      }
    }
    std::println!("[dust gpu] emit parity: worst |Δr0| = {worst_r:.4} m");
    // slots outside the batch are untouched (the ring starts zeroed only by chance, so check a
    // slot just before the batch is not one of ours)
    let before = (batch.first_index.wrapping_sub(1) & batch.ring_mask) as usize;
    assert_ne!(
      ring[before].t0().to_f64(),
      dust::emit_cluster(&batch, 0).t0().to_f64()
    );
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
        gpu.age_id_dbeta_flux[1].to_bits(),
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
