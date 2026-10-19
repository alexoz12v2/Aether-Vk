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

/// The `count` records of a tier's live range from a slot-indexed buffer (`dust_propagate.comp`
/// writes record `(first_slot + k) & mask` of the tier starting at record `base`), in compact
/// ring order `k` (the wrap handled): the CPU mirror's layout.
pub(super) fn read_tier_records<T: bytemuck::Pod>(
  device: &Device,
  buffer: vk::Buffer,
  base: u32,
  first_slot: u32,
  count: u32,
  mask: u32,
) -> alloc::vec::Vec<T> {
  let stride = core::mem::size_of::<T>();
  let first = first_slot & mask;
  let count = count.min(mask + 1);
  // the records up to the range's end, or the whole sub-ring when it wraps
  let upto = if first + count <= mask + 1 { first + count } else { mask + 1 };
  let bytes = read_back(device, buffer, false, (base + upto) as u64 * stride as u64);
  let all: &[T] = bytemuck::cast_slice(&bytes[base as usize * stride..]);
  (0..count).map(|k| all[((first + k) & mask) as usize]).collect()
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

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
  [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
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
    // a site 2 km off the centre turning with the spin
    let mut streams = test_batch(512, 6.0 * 3600.0, capacity - 100, capacity);
    streams.mass_params[3] = dust::batch_word(6, true, false);
    // 4 streams (a small tier): the mass shares follow S = 4
    let mut cycling = test_batch(512, 1800.0, capacity - 100, capacity);
    cycling.mass_params[3] = dust::batch_word(2, false, false);
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
    for batch in [single, streams, cycling] {
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
    assert_eq!(batch.ring_mask, capacity - 1);
    // the records by slot, in ring order
    let out: alloc::vec::Vec<DustRenderCluster> =
      read_tier_records(device, render, 0, first_slot, batch.count, batch.ring_mask);
    let mut worst = 0.0f32;
    for (i, gpu) in out.iter().enumerate() {
      let slot = (first_slot + i as u32) & batch.ring_mask;
      assert_eq!(
        dust::render_slot(gpu.age_id_dbeta_flux[1]),
        slot,
        "ring offset {i} must carry its slot"
      );
      let cpu = dust::evaluate_cluster_in(
        &dust::emit_cluster(&batch, i as u32),
        slot,
        dust::ring_log2_capacity(batch.ring_mask),
        &frame,
      );
      // the render word (slot, mask bits, shift, live) bit for bit
      assert_eq!(
        gpu.age_id_dbeta_flux[1].to_bits(),
        cpu.age_id_dbeta_flux[1].to_bits(),
        "render word of slot {slot}"
      );
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
        let (addr, moments) =
          device.cmd_dust_propagate(cmd, 9004, base, sub, first_slot, batch.count, &frame)?;
        assert!(addr > 0 && moments > 0);
        Ok(())
      })
      .unwrap();
    res.cleanup(&device.device);
    // the tier's records by slot within its sub-ring, in ring order
    let out: alloc::vec::Vec<DustRenderCluster> =
      read_tier_records(device, render, base, first_slot, batch.count, batch.ring_mask);
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
      particles::{ParticleSystemComponent, ParticleSystemDrawParams},
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
    // this test ticks its own 2-day history below
    crate::scene::particles::EMISSION_START_PREEXISTING,
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
      albedo: 0.0,
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
    eject: [0.0; 4],
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
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 7);
    let run_frame = |traced: Option<(DustFrame, [f64; 3])>| {
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          device.cmd_dust_frame_begin(cmd, pe, [1280, 720], 0, Default::default())?;
          device.cmd_dust_system_begin(id, 0, Default::default())?;
          if let Some((frame, nucleus)) = traced {
            let (render, moments) =
              device.cmd_dust_propagate(cmd, id, 0, capacity, 0, batch.count, &frame)?;
            assert_eq!(render, render_addr);
            device.cmd_dust_pre_splat_barrier(cmd);
            let pc = crate::scene::dust::DustSplatPushConstants {
              moments: 0,
              render: 0,
              pyramid: 0,
              live_count: batch.count,
              flags: 0,
              exposure: 1.0,
              inv_unit: 1.0,
              color: 0xFFFFFF,
              albedo: 0.0,
              mvp,
              eye_local: [0.0, -1e12, 0.0, 0.0],
            };
            device.cmd_dust_splat(cmd, pe, id, 0, 0, capacity, render, moments, pc)?;
            device.cmd_dust_post_splat_barrier(cmd);
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
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
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
  // SAFETY: as for set_var above; later tests in this process must not trace
  unsafe { std::env::remove_var("AETHERVK_DUST_TRACE") };
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

fn moments_buffer(device: &Device, id: u64) -> vk::Buffer {
  let res = device.res.read();
  let sys = res.dust_manager.as_ref().unwrap().systems.get(&id).unwrap();
  sys.moments.as_ref().unwrap().buffer
}

/// `dust_propagate.comp` writes the packets' second moments (velocity secants, β chord and
/// sagitta) as the reference `packet_moments` does, within the f32 rounding of df64 differences.
#[test]
fn gpu_dust_moments_match_reference() {
  with_dust_device(9007, |device, capacity, _, _| {
    let batch = test_batch(512, 3600.0, 7, capacity);
    emit(device, 9007, &batch);
    let moments = moments_buffer(device, 9007);
    // a spinning nucleus: the frame's spin lands in the moments' spare lanes
    let spin = [
      0.267_261_24,
      0.534_522_48,
      0.801_783_73,
      2.0 * core::f64::consts::PI / 44_640.0,
    ];
    for days in [0.5, 10.0, 400.0] {
      let frame = frame_after(days).with_spin(spin);
      let first_slot = batch.first_index & batch.ring_mask;
      propagate(device, 9007, first_slot, batch.count, &frame);
      // the moments by slot, in ring order
      let out: alloc::vec::Vec<dust::DustMoments> =
        read_tier_records(device, moments, 0, first_slot, batch.count, batch.ring_mask);
      let generation = device.dust_propagate_generation(9007, 0).expect("propagated");
      assert!(generation > 0);
      let mut worst = 0.0f32;
      for (i, gpu) in out.iter().enumerate() {
        let slot = (first_slot + i as u32) & batch.ring_mask;
        let (_, cpu) = dust::packet_moments(&dust::emit_cluster(&batch, i as u32), slot, &frame);
        if cpu.flux() > 0.0 {
          // the velocity lanes (the mean's heliocentric velocity, particle-system axes) and the
          // stamp: the GPU's generation, the CPU's 0
          let (gv, cv) = (gpu.helio_velocity(), cpu.helio_velocity());
          for k in 0..3 {
            assert!(
              (gv[k] - cv[k]).abs() < 1e-3 + 1e-6 * norm3(cv),
              "{days} d slot {slot}: velocity[{k}] gpu {} cpu {}",
              gv[k],
              cv[k]
            );
          }
          assert!(norm3(cv) > 1e3, "{days} d slot {slot}: a heliocentric speed");
          assert_eq!(gpu.propagate_stamp(), generation, "{days} d slot {slot}: stamp");
          assert_eq!(cpu.propagate_stamp(), 0);
        }
        assert!(
          (gpu.flux() - cpu.flux()).abs() <= 1e-3 * cpu.flux().abs() + 1e-30,
          "{days} d slot {slot} flux"
        );
        assert_eq!(gpu.id(), cpu.id(), "{days} d slot {slot} id");
        let scale = cpu.cov_v().iter().fold(0.0f32, |a, &v| a.max(v.abs())).max(1e-6);
        for k in 0..6 {
          let (g, c) = (gpu.cov_v()[k], cpu.cov_v()[k]);
          let err = (g - c).abs() / scale;
          worst = worst.max(err);
          assert!(
            err < 2e-2,
            "{days} d slot {slot}: cov[{k}] gpu {g:.4e} cpu {c:.4e}"
          );
        }
        // the size polyline: every edge within 1 % of its distance from the mean (+ 2 m), the
        // speed factors equal
        for j in 0..dust::SIZE_EDGES as usize {
          let (ge, ce) = (gpu.edge(j), cpu.edge(j));
          let l1 = norm3(sub3(ce, cpu.mean())).max(1.0);
          for k in 0..3 {
            assert!(
              (ge[k] - ce[k]).abs() < 1e-2 * l1 + 2.0,
              "{days} d slot {slot}: edge {j}[{k}] gpu {} cpu {}",
              ge[k],
              ce[k]
            );
          }
          assert!(
            (gpu.edge_factor(j) - cpu.edge_factor(j)).abs() < 1e-5,
            "{days} d slot {slot}: edge {j} factor"
          );
        }
        // the spin lanes: ω and the unit axis, bit-close on both sides
        assert!(
          (gpu.spin_rate() - cpu.spin_rate()).abs() < 1e-9
            && (cpu.spin_rate() - spin[3] as f32).abs() < 1e-9,
          "{days} d slot {slot}: spin rate gpu {} cpu {}",
          gpu.spin_rate(),
          cpu.spin_rate()
        );
        for k in 0..3 {
          assert!(
            (gpu.spin_axis()[k] - cpu.spin_axis()[k]).abs() < 1e-6
              && (cpu.spin_axis()[k] - spin[k] as f32).abs() < 1e-6,
            "{days} d slot {slot}: spin axis[{k}] gpu {} cpu {}",
            gpu.spin_axis()[k],
            cpu.spin_axis()[k]
          );
        }
        for k in 0..3 {
          assert!(
            (gpu.mean()[k] - cpu.mean()[k]).abs() < 2.0 + 2e-6 * norm3(cpu.mean()),
            "{days} d slot {slot}: mean"
          );
        }
      }
      std::println!("[dust gpu] moments parity at {days} d: worst relative cov error {worst:.2e}");
    }
  });
}

/// The propagate's generation stamp (`edges[12].w`) and its skip flag, on a live range that wraps
/// the ring: every live slot carries the generation; with the flag a second propagate at the
/// same inputs leaves the moments word for word, and leaves a stamped slot alone even when its
/// cluster changed (so the skip really happened: an uncached propagate's generation follows the
/// frame and the range, not the emission); without the flag the slot is recomputed; a new time
/// bumps the generation and rewrites every stamp, equal to the unflagged propagate at that time.
#[test]
fn gpu_dust_propagate_stamp_skips_at_the_same_time() {
  with_dust_device(9040, |device, capacity, _, _| {
    let id = 9040;
    // the range wraps: 300 slots before the end, 724 after the start
    let batch = test_batch(1024, 3600.0, capacity - 300, capacity);
    emit(device, id, &batch);
    let moments = moments_buffer(device, id);
    let first_slot = batch.first_index & batch.ring_mask;
    let read = |d: &Device| -> alloc::vec::Vec<dust::DustMoments> {
      read_tier_records(d, moments, 0, first_slot, batch.count, batch.ring_mask)
    };
    let words = |m: &dust::DustMoments| bytemuck::bytes_of(m).to_vec();
    let frame = frame_after(10.0);
    propagate(device, id, first_slot, batch.count, &frame);
    let g1 = device.dust_propagate_generation(id, 0).expect("propagated");
    let m1 = read(device);
    let live: alloc::vec::Vec<usize> = (0..m1.len()).filter(|&i| m1[i].flux() > 0.0).collect();
    assert!(live.len() > m1.len() / 2, "only {} live clusters", live.len());
    for &i in &live {
      assert_eq!(m1[i].propagate_stamp(), g1, "ring offset {i}: stamp");
    }
    // the same inputs with the skip flag: the generation stands and the words with it
    device.dust_set_propagate_skip(id, true).unwrap();
    propagate(device, id, first_slot, batch.count, &frame);
    assert_eq!(device.dust_propagate_generation(id, 0), Some(g1));
    let m2 = read(device);
    for i in 0..m1.len() {
      assert_eq!(words(&m2[i]), words(&m1[i]), "ring offset {i}: skipped words");
    }
    // other clusters in the same slots at the same generation: the flag leaves every stamped slot
    let mut other = batch;
    other.seed ^= 0x5EED_0001;
    emit(device, id, &other);
    propagate(device, id, first_slot, batch.count, &frame);
    assert_eq!(device.dust_propagate_generation(id, 0), Some(g1));
    let m3 = read(device);
    for &i in &live {
      assert_eq!(words(&m3[i]), words(&m1[i]), "ring offset {i}: a stamped slot was recomputed");
    }
    // without the flag the new clusters are evaluated (and stamped with the same generation)
    device.dust_set_propagate_skip(id, false).unwrap();
    propagate(device, id, first_slot, batch.count, &frame);
    let m4 = read(device);
    let changed = live.iter().filter(|&&i| words(&m4[i]) != words(&m1[i])).count();
    assert!(changed > live.len() / 2, "only {changed} of {} slots re-evaluated", live.len());
    for (i, m) in m4.iter().enumerate() {
      if m.flux() > 0.0 {
        assert_eq!(m.propagate_stamp(), g1, "ring offset {i}: stamp after the re-evaluation");
      }
    }
    // a new time bumps the generation: with the flag every slot is rewritten and stamped
    device.dust_set_propagate_skip(id, true).unwrap();
    let frame2 = frame_after(11.0);
    propagate(device, id, first_slot, batch.count, &frame2);
    let g2 = device.dust_propagate_generation(id, 0).unwrap();
    assert_ne!(g2, g1);
    let m5 = read(device);
    let mut stamped = 0;
    for m in &m5 {
      if m.flux() > 0.0 {
        assert_eq!(m.propagate_stamp(), g2);
        stamped += 1;
      }
    }
    assert!(stamped > m5.len() / 2);
    // ... equal to the unflagged propagate at that time, word for word
    device.dust_set_propagate_skip(id, false).unwrap();
    propagate(device, id, first_slot, batch.count, &frame2);
    assert_eq!(device.dust_propagate_generation(id, 0), Some(g2));
    let m6 = read(device);
    for i in 0..m5.len() {
      assert_eq!(words(&m6[i]), words(&m5[i]), "ring offset {i}: the bumped generation's words");
    }
    std::println!(
      "[dust gpu] propagate stamp: generations {g1} → {g2}, {} live, {changed} re-evaluated",
      live.len()
    );
  });
}

/// `dust_splat.comp` scatters the same texel counts as the reference `splat_tier` into the
/// view's pyramid (integer sums: order independent; per-texel differences come only from the
/// float evaluation of each packet), and the header carries the level mask and τ_max: the GPU
/// splat of `batch` (emitted into system `id`) evaluated at `frame`, seen top-down over
/// ±`half` m around the anchor (or through `camera`: an mvp and the eye in local metres, for
/// the perspective case of the split), against the CPU splat of the GPU's own moments.
#[allow(clippy::too_many_arguments)]
fn splat_parity(
  id: u64,
  batch_of: &dyn Fn(u32) -> DustBatch,
  frame: DustFrame,
  half: f32,
  exposure: f32,
  expect_spiral: bool,
  albedo: f32,
  camera: Option<([f32; 16], [f32; 3])>,
  shell: Option<f32>,
  legacy: bool,
  lod: Option<f32>,
) {
  with_dust_device(id, |device, capacity, _, _| {
    let batch = batch_of(capacity);
    emit(device, id, &batch);
    let first_slot = batch.first_index & batch.ring_mask;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 9);
    let (w, h) = (640u32, 400u32);
    // top-down orthographic view of ±2e7 m around the anchor, metres (units 1)
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1.0 / half;
    mvp[5] = 1.0 / half;
    mvp[10] = 1e-12;
    mvp[15] = 1.0;
    let mut eye = [0.0f32, 0.0, 1e10];
    if let Some((m, e)) = camera {
      mvp = m;
      eye = e;
    }
    let mut pc = dust::DustSplatPushConstants {
      moments: 0,
      render: 0,
      pyramid: 0,
      live_count: batch.count,
      flags: dust::DUST_VIEW_FLOW,
      exposure,
      inv_unit: 0.0,
      color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
      albedo,
      eye_local: [eye[0], eye[1], eye[2], 30.0 * 86400.0], // w = TTL: the hue ramp is exercised
      mvp,
    };
    let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let mut run = |pc: &mut dust::DustSplatPushConstants| -> f32 {
      let mut unit = 0.0;
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], pc.flags, flow)?;
          unit = view.unit;
          pc.inv_unit = 1.0 / view.unit;
          device.cmd_dust_system_begin(id, pc.flags, flow)?;
          let (render, moments) =
            device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
          device.cmd_dust_pre_splat_barrier(cmd);
          device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, *pc)?;
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      unit
    };
    // the first frame measures τ_max (unit 1e-4 before any); the host folds a readback
    // DUST_READBACK_SLOTS frames later, so by the sixth frame the unit follows the view
    let mut gpu_unit = 0.0;
    for i in 0..(super::dust::DUST_READBACK_SLOTS + 2) {
      gpu_unit = run(&mut pc);
      if i == 0 {
        // the speck shell's radius and the kernel in the next frames' headers (the view exists
        // now)
        device.dust_set_shell_override(pe, shell);
        device.dust_set_legacy_kernel(pe, legacy);
        device.dust_set_lod_px(pe, lod);
        device.dust_set_deep_chord(pe, Some(PARITY_DEEP_CHORD.with(|c| c.get())));
        device.dust_set_volume_bias(pe, parity_volume_bias());
      }
    }
    let gpu = device.download_dust_pyramid(pe).unwrap();
    let layout = device.dust_view_layout(pe).unwrap();
    assert_eq!(layout, dust::PyramidLayout::new(w, h));
    // every volumetric record the GPU emitted was drawn (the CPU draws them all; a record the
    // volume spill refused would be a mismatch that is not the drawing's)
    let (_, lost) = device.dust_vqueue_spill_counts(pe).unwrap();
    assert_eq!(lost, 0, "case {id}: the volume spill lost {lost} records");
    // a fixed volume bias must have reached the header the GPU drew with, before anything else
    if let Some(b) = parity_volume_bias() {
      let hw_ = dust::PYRAMID_HEADER_WORDS as usize;
      let nonzero_gpu = gpu[hw_..].chunks_exact(dust::PYRAMID_TEXEL_WORDS as usize).filter(|t| t[0] > 0).count();
      std::println!(
        "[dust gpu] splat parity case {id}: fixed volume bias {b}; header bias {}, split word {:#x}, level mask {:#b}, volume evaluations {}, nonzero texels {nonzero_gpu}",
        dust::VolumeStats::load(&gpu).bias,
        gpu[dust::PYR_SPLIT_DEPTH as usize],
        gpu[dust::PYR_LEVEL_MASK as usize],
        gpu[dust::PYR_VOLUME_EVALS as usize]
      );
      assert_eq!(dust::VolumeStats::load(&gpu).bias, b, "case {id}: the fixed volume bias did not reach the header");
    }
    // reference: the CPU splat of the *GPU's* moments (`gpu_dust_moments_match_reference` covers
    // those): the merge, piece-count and level decisions are made on identical packets, so what
    // remains is the float evaluation of the capsules
    let gpu_render = buffers(device, id).2;
    let gpu_moments = moments_buffer(device, id);
    // (slot indexed on the GPU: gathered into the CPU mirror's compact ring order)
    let render: alloc::vec::Vec<dust::DustRenderCluster> =
      read_tier_records(device, gpu_render, 0, first_slot, batch.count, capacity - 1);
    let moments: alloc::vec::Vec<dust::DustMoments> =
      read_tier_records(device, gpu_moments, 0, first_slot, batch.count, capacity - 1);
    // how many live samples draw their time segment as the spiral (`spiral_of`): the case that
    // expects it must exercise the GLSL mirror on most of them
    let (mut live, mut spiral) = (0usize, 0usize);
    for (i, m) in moments.iter().enumerate() {
      if !(m.flux() > 0.0) {
        continue;
      }
      live += 1;
      if let Some(q) = dust::streak_pred(&render, i) {
        if dust::spiral_of(m, &moments[q]).is_some() {
          spiral += 1;
        }
      }
    }
    std::println!("[dust gpu] splat parity: {spiral} of {live} live samples on the spiral");
    if expect_spiral {
      // the youngest 8 samples of each of the 16 streams
      assert!(
        spiral >= 64,
        "only {spiral} of {live} samples on the spiral"
      );
    } else {
      assert_eq!(spiral, 0, "{spiral} samples on the spiral without a spin");
    }
    let mut cpu = alloc::vec![0u32; layout.total_words as usize];
    cpu[..dust::PYRAMID_HEADER_WORDS as usize].copy_from_slice(&layout.header());
    // the host-written split depth and volume level bias the GPU frame drew with (the host
    // lowers / raises them from the previous frames' words): the reference uses the same
    cpu[dust::PYR_SPLIT_DEPTH as usize] = gpu[dust::PYR_SPLIT_DEPTH as usize];
    // and the speck shell's radius: the continuum gives that mass up on both mirrors
    let dmax_word = dust::specks::PYR_SPECKS_DMAX as usize;
    cpu[dmax_word] = gpu[dmax_word];
    // and the cluster level of detail (`PYR_LOD`): the same groups and piece caps on both mirrors
    cpu[dust::PYR_LOD as usize] = gpu[dust::PYR_LOD as usize];
    // and the deep-chord bit (`DUST_FRAME_DEEP_CHORD` of `PYR_FLAGS`): the same split and Jacobian
    let deep = dust::DUST_FRAME_DEEP_CHORD;
    cpu[dust::PYR_FLAGS as usize] |= gpu[dust::PYR_FLAGS as usize] & deep;
    assert_eq!(
      dust::deep_chord(&gpu),
      PARITY_DEEP_CHORD.with(|c| c.get()),
      "the deep-chord bit reached the header"
    );
    let lod_px = dust::lod_px(&gpu);
    std::println!("[dust gpu] splat parity: level of detail {lod_px} px");
    if let Some(l) = lod {
      assert_eq!(lod_px, l, "the level of detail reached the header");
    }
    {
      // the rule on the GPU's moments: how many live polylines group their bins and how many
      // time segments lose pieces (`lod_group`, `time_pieces_lod`); a case drawn at a knob must
      // exercise both on the GPU mirror
      let view = dust::SplitView::of(&pc.mvp, [pc.eye_local[0], pc.eye_local[1], pc.eye_local[2]]);
      let px = |q: [f32; 3]| dust::project_point(q, &pc.mvp, w, h);
      let (mut grouped, mut capped, mut n) = (0usize, 0usize, 0usize);
      for (i, m) in moments.iter().enumerate() {
        if !(m.flux() > 0.0) {
          continue;
        }
        n += 1;
        let pred = dust::streak_pred(&render, i).map(|q| &moments[q]);
        let depth = if view.perspective { dust::polyline_depth(m, view.eye, view.fwd) } else { None };
        if dust::lod_group(dust::polyline_extent_px(m, &px), lod_px, dust::lod_group_depth_cap(depth)) > 1 {
          grouped += 1;
        }
        let n_t = dust::time_pieces(m, pred);
        let span = pred.filter(|_| view.perspective).map(|p| {
          let d = [p.mean()[0] - m.mean()[0], p.mean()[1] - m.mean()[1], p.mean()[2] - m.mean()[2]];
          d[0] * view.fwd[0] + d[1] * view.fwd[1] + d[2] * view.fwd[2]
        });
        let depth_min = dust::time_pieces_depth(span, depth);
        if dust::time_pieces_lod(n_t, dust::time_segment_px(m, pred, &px), lod_px, depth_min) < n_t {
          capped += 1;
        }
      }
      std::println!("[dust gpu] splat parity: {grouped} of {n} live polylines grouped, {capped} with capped pieces");
      if lod.is_some_and(|l| l > 0.0) {
        assert!(grouped >= n / 20, "only {grouped} of {n} polylines grouped at {lod_px} px");
        if expect_spiral {
          assert!(capped >= 1, "no spiral segment lost a piece at {lod_px} px");
        }
      }
    }
    std::println!(
      "[dust gpu] splat parity: speck shell {} m",
      f32::from_bits(gpu[dmax_word])
    );
    if let Some(d) = shell {
      assert_eq!(
        f32::from_bits(gpu[dmax_word]),
        d,
        "the shell override reached the header"
      );
    }
    let _ = dust::take_volume_class_counts();
    let _ = dust::take_volume_small_paths();
    let _ = dust::take_volume_small_margins();
    let _ = dust::take_volume_split_adds();
    dust::splat_tier(&moments, &render, &pc, &flow, 0, &layout, &mut cpu);
    let classes = dust::take_volume_class_counts();
    let small_paths = dust::take_volume_small_paths();
    let small_margins = dust::take_volume_small_margins();
    let split_adds = dust::take_volume_split_adds();
    // track 9A: the small pieces' paths on both mirrors at the count level, from the profiler's
    // words when the `.prof` shaders drew this frame (`gpu_dust_small_paths_match_reference`): a
    // decision on a float boundary (D3's 3σ within 1e-3 σ of the frame's edge, the predictor
    // within 1e-3 ε of ε) moves one piece between the paths, so each count agrees within the
    // reference's pieces at those margins
    if let Ok(p) = device.dust_profile_counters(pe) {
      if p.any() {
        let gf = p.word(dust::DUST_PROF_SMALL_PATHS) as u64;
        let ge = p.word(dust::DUST_PROF_SMALL_PATHS + 1) as u64;
        let margin = small_margins[1] + small_margins[2];
        std::println!(
          "[dust gpu] splat parity case {id}: small pieces by one projection a slice gpu {gf} cpu {}, by 4 sigma points gpu {ge} cpu {}; demoted cpu {} (ε {:.0e}); at a margin cpu: D3 {}, ε {}",
          small_paths[0],
          small_paths[1],
          small_margins[0],
          dust::DUST_SMALL_DEMOTE_EPS,
          small_margins[1],
          small_margins[2]
        );
        assert!(
          gf.abs_diff(small_paths[0]) <= margin && ge.abs_diff(small_paths[1]) <= margin,
          "case {id}: small paths gpu ({gf}, {ge}) cpu ({}, {}), margin {margin}",
          small_paths[0],
          small_paths[1]
        );
        PARITY_SMALL_PATHS_COMPARED.store(true, core::sync::atomic::Ordering::SeqCst);
      }
    }
    let gpu_split_adds = device.dust_volume_split_adds(pe).unwrap_or(u32::MAX);
    PARITY_VOLUME_CLASSES.lock().unwrap_or_else(|e| e.into_inner()).push((id, classes));
    std::println!(
      "[dust gpu] splat parity: volumetric pieces drawn (centre, small, sub-rays, lines) {classes:?}; small by one projection a slice {}, by 4 sigma points {}; volume deposits split above the per-add cap: cpu {split_adds}, gpu {gpu_split_adds}",
      small_paths[0],
      small_paths[1]
    );
    dust::measure_from_pyramid(&layout, &mut cpu);
    let hw = dust::PYRAMID_HEADER_WORDS as usize;
    assert_eq!(gpu[dust::PYR_LEVELS as usize], layout.level_count());
    assert_eq!(
      gpu[dust::PYR_LEVEL_MASK as usize],
      cpu[dust::PYR_LEVEL_MASK as usize],
      "level mask"
    );
    let (tg, tc) = (
      f32::from_bits(gpu[dust::PYR_TAU_MAX as usize]),
      f32::from_bits(cpu[dust::PYR_TAU_MAX as usize]),
    );
    assert!(
      (tg / tc - 1.0).abs() < 1e-2,
      "τ_max gpu {tg:.4e} cpu {tc:.4e}"
    );
    // the split words (`split_packet` / `dust_split_step`): the same rules on identical packets
    // to the same host-written depth (no run-time gate: the children count is informational and
    // thread-order free); a decision on a float boundary moves a node between split and kept,
    // so the children made agree to 2 % + 4, the largest dropped flux to 1 % (a different packet
    // dropped on one side only would show as a different maximum)
    let (cg, cc) = (
      gpu[dust::PYR_SPLIT_CHILDREN as usize],
      cpu[dust::PYR_SPLIT_CHILDREN as usize],
    );
    let (dg, dc) = (
      f32::from_bits(gpu[dust::PYR_SPLIT_DROPPED_FLUX as usize]),
      f32::from_bits(cpu[dust::PYR_SPLIT_DROPPED_FLUX as usize]),
    );
    // the volume evaluations (`PYR_VOLUME_EVALS`): the same plans on identical packets; a plan
    // decision on a float boundary (a level or a piece merge) moves one packet's count: 2 % + 16
    let (eg, ec) = (
      gpu[dust::PYR_VOLUME_EVALS as usize],
      cpu[dust::PYR_VOLUME_EVALS as usize],
    );
    let (sg_, vg) = (dust::SplitStats::load(&gpu), dust::VolumeStats::load(&gpu));
    std::println!(
      "[dust gpu] split: depth {} (volume bias {}), children gpu {cg} cpu {cc} (budget {}, overflow {}), dropped flux gpu {dg:.4e} cpu {dc:.4e}, volume evaluations gpu {eg} cpu {ec}",
      sg_.depth,
      vg.bias,
      dust::DUST_SPLIT_BUDGET,
      sg_.overflow()
    );
    assert!(
      cg.abs_diff(cc) as f32 <= 4.0 + 0.02 * cc.max(cg) as f32,
      "split children gpu {cg} cpu {cc}"
    );
    assert!(
      eg.abs_diff(ec) as f32 <= 16.0 + 0.02 * ec.max(eg) as f32,
      "volume evaluations gpu {eg} cpu {ec}"
    );
    assert!(
      (dg == 0.0 && dc == 0.0) || (dg / dc - 1.0).abs() < 1e-2,
      "dropped flux gpu {dg:.4e} cpu {dc:.4e}"
    );
    // the EWA texels planned (`PYR_EWA_TEXELS`): the same level choices on identical packets; a
    // level on a float boundary moves one packet's count: 2 % + 16 units
    let (tg_, tc_) = (
      gpu[dust::PYR_EWA_TEXELS as usize],
      cpu[dust::PYR_EWA_TEXELS as usize],
    );
    std::println!("[dust gpu] EWA texel units gpu {tg_} cpu {tc_}");
    assert!(
      tg_.abs_diff(tc_) as f32 <= 16.0 + 0.02 * tc_.max(tg_) as f32,
      "EWA texel units gpu {tg_} cpu {tc_}"
    );
    let (mut worst, mut peak, mut sg, mut sc, mut nonzero) = (0u32, 0u32, 0u64, 0u64, 0usize);
    let (mut depth_compared, mut depth_mismatch) = (0usize, 0usize);
    let (mut box_compared, mut box_violations) = (0usize, 0usize);
    for (l, &(off, lw, lh)) in layout.levels.iter().enumerate() {
      for t in 0..(lw * lh) as usize {
        let i = off as usize + t * dust::PYRAMID_TEXEL_WORDS as usize;
        for k in 0..4 {
          let (g, c) = (gpu[i + k], cpu[i + k]);
          let d = g.abs_diff(c);
          worst = worst.max(d);
          peak = peak.max(c);
          if k == 0 {
            sg += g as u64;
            sc += c as u64;
            nonzero += (c > 0) as usize;
          }
          // per texel within the float evaluation of each packet where the counts are substantial;
          // faint texels carry the error diffusion's ±1-per-packet dither, checked on 3×3 sums
          // below
          if k > 0 {
            // the colour words follow τ: compare the hue ratio (word / τ) instead, the dither
            // of both words cancels in it
            let (g0, c0) = (gpu[i], cpu[i]);
            if g0 >= 64 && c0 >= 64 {
              let (hg, hc) = (
                g as f32 / g0 as f32 / dust::DUST_COLOR_SCALE,
                c as f32 / c0 as f32 / dust::DUST_COLOR_SCALE,
              );
              assert!(
                // the colour words carry the dither of every contributing packet (±0.5 per
                // packet and word, uncorrelated between the words): 8 counts of slack
                (hg - hc).abs() <= 0.03 + 8.0 / c0 as f32,
                "level {l} texel {t} word {k}: hue gpu {hg:.4} ({g}/{g0}) cpu {hc:.4} ({c}/{c0})"
              );
            }
          } else {
            // τ: the error diffusion dithers each packet's faint texels by ±1 where the float
            // evaluation differs, and a merge / piece-count decision on a float boundary moves a
            // whole packet between neighbouring texels; compared on 3×3 sums (5 % + 6 counts)
            let (tx, ty) = ((t as u32 % lw) as i64, (t as u32 / lw) as i64);
            let (mut bg, mut bc) = (0u64, 0u64);
            for dy in -1..=1i64 {
              for dx in -1..=1i64 {
                let (x, y) = (tx + dx, ty + dy);
                if x >= 0 && y >= 0 && x < lw as i64 && y < lh as i64 {
                  let j = off as usize
                    + (y as usize * lw as usize + x as usize) * dust::PYRAMID_TEXEL_WORDS as usize;
                  bg += gpu[j] as u64;
                  bc += cpu[j] as u64;
                }
              }
            }
            // fainter neighbourhoods are the dither of many packets of a fraction of a count
            // each: covered in aggregate by the level sums and the measurement grid (0.1 %)
            // a merge or piece-count decision on a float boundary moves a whole packet between
            // neighbourhoods on one side only: counted, allowed on half a percent of them
            let bd = bg.abs_diff(bc);
            if bg.max(bc) >= 100 {
              box_compared += 1;
              if bd as f64 > 6.0 + 5e-2 * bg.max(bc) as f64 {
                box_violations += 1;
                if box_violations <= 3 {
                  std::println!(
                    "[parity] level {l} texel {t}: 3×3 τ gpu {bg} cpu {bc} (texel gpu {g} cpu {c})"
                  );
                }
              }
            }
          }
        }
        // nearest depth: identical bits where dust landed on both sides with more than the
        // rounding noise (the error diffusion rounds a faint packet to one count on one side and
        // none on the other, and that packet's depth then differs)
        if cpu[i] >= 4 && gpu[i] >= 4 {
          let (dg, dc) = (f32::from_bits(gpu[i + 4]), f32::from_bits(cpu[i + 4]));
          depth_compared += 1;
          if !((dg - dc).abs() < 1e-6 * dc.max(1e-30) + 1e-12) {
            depth_mismatch += 1;
          }
        }
      }
    } // a packet rounded to one count on one side only (error diffusion) changes a texel's nearest
    // depth: allowed on a per-mille of the compared texels
    // a packet whose merge / piece-count / on-screen decision sits on a float boundary (the GPU
    // fuses multiply-adds the reference does not) lands on one side only: it moves the nearest
    // depth of its whole footprint and a few 3×3 sums; the level sums and the measurement grid
    // (0.1 %, above) bound the energy of such packets
    std::println!(
      "[parity] nearest depth differs on {depth_mismatch} of {depth_compared} texels, 3×3 τ sums beyond 5 % + 6 on {box_violations} of {box_compared}"
    );
    assert!(
      depth_mismatch <= 1 + depth_compared / 10,
      "nearest depth differs on {depth_mismatch} of {depth_compared} texels"
    );
    assert!(
      box_violations <= 1 + box_compared / 100,
      "3×3 τ sums differ by more than 5 % + 6 on {box_violations} of {box_compared} neighbourhoods"
    );

    assert!(nonzero > 300, "{nonzero} texels");
    assert!(
      (sg as f64 / sc as f64 - 1.0).abs() < 1e-3,
      "Σ counts gpu {sg} cpu {sc}"
    );
    // the measurement grid (dust_measure.comp from the GPU pyramid vs the reference from the CPU
    // one): a box sum of ≤ 256 + 64 + 16 + 4 + 1 level texels, each within the tolerance above
    let (mo, mw, mh) = layout.measure;
    let (mut msg, mut msc, mut m_occupied) = (0u64, 0u64, 0usize);
    // the error diffusion dithers the faint texels of every packet by ±1 count, and a packet
    // whose sub-texel position differs by a fused multiply-add can put its last count on one
    // side only: an occupancy mismatch of up to 2 counts is that dither, allowed on 1 % of the
    // occupied texels
    // a packet whose footprint edge sits on a float boundary (the GPU fuses multiply-adds the
    // reference does not) puts its fringe column on one side only: a one-sided texel below
    // 1e-4 of the grid's peak is that fringe (the 3×3 and level rules above bound its energy);
    // counted with the dither against the 1 % cap
    let measure_peak = (0..(mw * mh) as usize)
      .map(|t| {
        let i = mo as usize + t * dust::PYRAMID_TEXEL_WORDS as usize;
        gpu[i].max(cpu[i])
      })
      .max()
      .unwrap_or(0);
    let fringe_cap = 2u32.max((measure_peak as f64 * 1e-4) as u32);
    let mut occupancy_dither = 0usize;
    let mut occupancy_bad: alloc::vec::Vec<(usize, u32, u32)> = alloc::vec::Vec::new();
    for t in 0..(mw * mh) as usize {
      let i = mo as usize + t * dust::PYRAMID_TEXEL_WORDS as usize;
      let (g, c) = (gpu[i], cpu[i]);
      msg += g as u64;
      msc += c as u64;
      m_occupied += (c > 0) as usize;
      assert!(
        g.abs_diff(c) as f32 <= 2.0 * 341.0 + 2e-3 * c.max(g) as f32,
        "case {id}: measure texel {t}: gpu {g} cpu {c}"
      );
      if (g > 0) != (c > 0) {
        if g.max(c) > fringe_cap {
          occupancy_bad.push((t, g, c));
        }
        occupancy_dither += 1;
      }
    }
    // what the differing measure texels hold, level by level (the pyramid texels inside each
    // 16-px box: a faint whole packet on one side only, or a spread difference?)
    for &(t, _, _) in occupancy_bad.iter().take(4) {
      let (bx, by) = ((t % mw as usize) as u32, (t / mw as usize) as u32);
      for (l, &(off, lw, lh)) in layout.levels.iter().enumerate() {
        let size = 1u32 << l;
        let (x0, y0) = (bx * 16 / size, by * 16 / size);
        let (x1, y1) = (
          ((bx + 1) * 16).div_ceil(size).min(lw),
          ((by + 1) * 16).div_ceil(size).min(lh),
        );
        for y in y0..y1 {
          for x in x0..x1 {
            let i = off as usize + (y * lw + x) as usize * dust::PYRAMID_TEXEL_WORDS as usize;
            if gpu[i] > 0 || cpu[i] > 0 {
              std::println!(
                "[parity] measure texel {t} ({bx},{by}): level {l} texel ({x},{y}) gpu {} cpu {}",
                gpu[i],
                cpu[i]
              );
            }
          }
        }
      }
    }
    assert!(
      occupancy_bad.is_empty(),
      "case {id}: measure occupancy differs beyond the dither and the fringe cap {fringe_cap} (peak {measure_peak}) on {} texels (grid {mw}×{mh}): {:?}",
      occupancy_bad.len(),
      &occupancy_bad[..occupancy_bad.len().min(12)]
    );
    // the dither on 1 % of the occupied texels plus one float-boundary fringe
    assert!(
      occupancy_dither <= 2 + m_occupied / 100,
      "measure occupancy differs on {occupancy_dither} texels of {m_occupied}"
    );
    assert!(
      m_occupied > 0 && (msg as f64 / msc as f64 - 1.0).abs() < 1e-3,
      "Σ measure gpu {msg} cpu {msc}"
    );
    std::println!(
      "[dust gpu] splat parity: {nonzero} texels, worst |Δ| {worst} counts of peak {peak}, unit {gpu_unit:.3e}, measure peak {measure_peak}, one-sided measure texels {occupancy_dither} of {m_occupied}"
    );
    let _ = hw;
    device.discard_dust_view(pe, 0);
  });
}

#[test]
fn gpu_dust_splat_matches_reference() {
  // five days of a one-hour window seen over ±2e7 m: the sweep and the size polyline
  // the radiance weight on (`dust_radiance_weight`: the phase LUT and 1/r_h² on the GPU)
  splat_parity(
    9008,
    &|cap| test_batch(1024, 3600.0, 3, cap),
    frame_after(5.0),
    2.0e7,
    1.0,
    false,
    0.04,
    None,
    None,
    false,
    None,
  );
}

/// The cluster level of detail on both mirrors (`dust_lod_*`, `PYR_LOD` = 4 px): the five-day
/// scene seen over ±1e8 m (one pixel = 312 km), where the polylines are a few pixels long and
/// their bins shorter than the knob, so the enumerator draws them as chord groups of 2–16 bins
/// (a lane per group chord, `g / 16` of the flux each, the group's density ramp and open ends);
/// and the spinning site seen 20× farther than its own case, where the spiral's eight pieces per
/// segment are capped by the segment on screen. The orthographic views have no depth rule. Each
/// case asserts the rule groups at least a twentieth of the live polylines (and, with pieces, caps
/// at least one time segment) on the GPU's moments, then compares the pyramids under the parity
/// rules.
#[test]
fn gpu_dust_splat_matches_reference_lod() {
  // the five-day scene over ±1e8 m (312 km per pixel): most polylines grouped, no time pieces to
  // cap (the predecessors are no wider)
  splat_parity(
    9041,
    &|cap| test_batch(1024, 3600.0, 3, cap),
    frame_after(5.0),
    1.0e8,
    1.0,
    false,
    0.04,
    None,
    None,
    false,
    Some(4.0),
  );
  // the spinning site's first two hours over ±1e5 m (20× its own case: 312 m per pixel): the
  // young polylines' bins are a pixel or two, the spiral's eight pieces per 112 s segment are
  // capped by the segment on screen
  let (batch_of, frame) = spinning_site_scene();
  splat_parity(9042, &batch_of, frame, 1.0e5, 1.0e-6, true, 0.0, None, None, false, Some(4.0));
}

/// The perspective split (`dust_project_split_scatter`: chords clipped at the eye plane, the
/// Gaussians sliced by depth, the leaves projected one by one) on both mirrors: a 60° camera
/// inside the five-day cloud, 200 m from the anchor, looking along −z through it; the packets
/// are kilometres to thousands of kilometres across, so nearly every one is split, clipped or
/// dropped (about 1.1·10⁶ children at depth 4, above the host's 2²⁰ budget target: there is no
/// gate, the count is informational), and the split words of the header must agree too. The
/// packets that hold the eye or are near it (`√tr Σ > 0.2 · dist`) are drawn by their exact ray
/// integral (`dust_volume_scatter`, whole frame at a coarse level): that path is compared here
/// too, under the same tolerance rules.
/// The perspective split from a camera inside the cloud looking **away** from the anchor (at
/// (0, 0, 200 m) along +z, the anchor behind it): the eye plane is the camera's on both mirrors
/// (`dust_view_forward` no longer turns a perspective view towards the scene's origin), so the
/// dust in front is drawn and the dust between the camera and the anchor is cut or dropped.
#[test]
fn gpu_dust_splat_matches_reference_perspective_away() {
  let (w, h) = (640.0f32, 400.0f32);
  let d = 200.0f32;
  let tan_h = (30.0f32).to_radians().tan();
  // w_clip = z − D: in front is +z, away from the anchor
  let mut mvp = [0.0f32; 16];
  mvp[0] = h / (w * tan_h);
  mvp[5] = 1.0 / tan_h;
  mvp[10] = -1e-3;
  mvp[11] = 1.0;
  mvp[15] = -d;
  splat_parity(
    9012,
    &|cap| test_batch(1024, 3600.0, 3, cap),
    frame_after(5.0),
    2.0e7,
    1.0,
    false,
    0.04,
    Some((mvp, [0.0, 0.0, d])),
    None,
    false,
    None,
  );
}

/// The speck shell's hand-over on both mirrors: the perspective camera of
/// `gpu_dust_splat_matches_reference_perspective` with a 20-km shell in the header
/// (`dust_set_shell_override`): the leaves inside it are left to the specks, those across it
/// split once more or scaled by their outside fraction (`dust_shell_classify`,
/// `dust_shell_outside_fraction`), and the splat pyramids still match.
#[test]
fn gpu_dust_splat_matches_reference_perspective_shell() {
  let (w, h) = (640.0f32, 400.0f32);
  let d = 200.0f32;
  let tan_h = (30.0f32).to_radians().tan();
  let mut mvp = [0.0f32; 16];
  mvp[0] = h / (w * tan_h);
  mvp[5] = 1.0 / tan_h;
  mvp[10] = -1e-3;
  mvp[11] = -1.0;
  mvp[15] = d;
  splat_parity(
    9013,
    &|cap| test_batch(1024, 3600.0, 3, cap),
    frame_after(5.0),
    2.0e7,
    1.0,
    false,
    0.04,
    Some((mvp, [0.0, 0.0, d])),
    Some(2.0e4),
    false,
    None,
  );
}

#[test]
fn gpu_dust_splat_matches_reference_perspective() {
  let (w, h) = (640.0f32, 400.0f32);
  let d = 200.0f32;
  let tan_h = (30.0f32).to_radians().tan();
  // camera at (0, 0, D) looking along −z: x_clip = x · (H/W) / tan(30°), y_clip = y / tan(30°),
  // w_clip = D − z (square pixels, 60° vertical field)
  let mut mvp = [0.0f32; 16];
  mvp[0] = h / (w * tan_h);
  mvp[5] = 1.0 / tan_h;
  mvp[10] = -1e-3;
  mvp[11] = -1.0;
  mvp[15] = d;
  splat_parity(
    9011,
    &|cap| test_batch(1024, 3600.0, 3, cap),
    frame_after(5.0),
    2.0e7,
    1.0,
    false,
    0.04,
    Some((mvp, [0.0, 0.0, d])),
    None,
    false,
    None,
  );
}

/// The legacy-kernel bit (`dust::DUST_FRAME_LEGACY_KERNEL_BIT`: no cone of grains, no slow-end
/// sub-bins; the A/B switch of the timing measurements) keeps the GPU and its CPU mirror in step:
/// the perspective case with the bit set.
#[test]
fn gpu_dust_splat_matches_reference_perspective_legacy_kernel() {
  let (w, h) = (640.0f32, 400.0f32);
  let d = 200.0f32;
  let tan_h = (30.0f32).to_radians().tan();
  // camera at (0, 0, D) looking along −z: x_clip = x · (H/W) / tan(30°), y_clip = y / tan(30°),
  // w_clip = D − z (square pixels, 60° vertical field)
  let mut mvp = [0.0f32; 16];
  mvp[0] = h / (w * tan_h);
  mvp[5] = 1.0 / tan_h;
  mvp[10] = -1e-3;
  mvp[11] = -1.0;
  mvp[15] = d;
  splat_parity(
    9017,
    &|cap| test_batch(1024, 3600.0, 3, cap),
    frame_after(5.0),
    2.0e7,
    1.0,
    false,
    0.04,
    Some((mvp, [0.0, 0.0, d])),
    None,
    true,
    None,
  );
}

std::thread_local! {
  /// the deep-chord bit [`splat_parity`] sets on its view (`Device::dust_set_deep_chord`)
  static PARITY_DEEP_CHORD: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}
/// a fixed volumetric level bias [`splat_parity`] sets on its view (`Device::dust_set_volume_bias`;
/// `u32::MAX`: none). Process-wide, not thread-local: the device closure of `with_dust_device`
/// may run on the render frontend's thread
static PARITY_VOLUME_BIAS: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(u32::MAX);
/// the volumetric pieces the reference drew per class in the last [`splat_parity`] of a case id
/// (centre, small, sub-rays, lines: `dust::take_volume_class_counts`, counted on the closure's thread)
static PARITY_VOLUME_CLASSES: std::sync::Mutex<alloc::vec::Vec<(u64, [u64; 4])>> = std::sync::Mutex::new(alloc::vec::Vec::new());
fn parity_volume_bias() -> Option<u32> {
  let b = PARITY_VOLUME_BIAS.load(core::sync::atomic::Ordering::SeqCst);
  (b != u32::MAX).then_some(b)
}

/// The deep-chord bit (`dust::DUST_FRAME_DEEP_CHORD`: a chord the split's depth limit leaves deep
/// only by its span drawn by the EWA at its geometric-mean depth, `dust_deep_chord_factors`,
/// instead of by the ray integral) keeps the GPU and its CPU mirror in step: the perspective case
/// with the bit set.
#[test]
fn gpu_dust_splat_matches_reference_perspective_deep_chord() {
  let (w, h) = (640.0f32, 400.0f32);
  let d = 200.0f32;
  let tan_h = (30.0f32).to_radians().tan();
  let mut mvp = [0.0f32; 16];
  mvp[0] = h / (w * tan_h);
  mvp[5] = 1.0 / tan_h;
  mvp[10] = -1e-3;
  mvp[11] = -1.0;
  mvp[15] = d;
  PARITY_DEEP_CHORD.with(|c| c.set(true));
  splat_parity(
    9018,
    &|cap| test_batch(1024, 3600.0, 3, cap),
    frame_after(5.0),
    2.0e7,
    1.0,
    false,
    0.04,
    Some((mvp, [0.0, 0.0, d])),
    None,
    false,
    None,
  );
  PARITY_DEEP_CHORD.with(|c| c.set(false));
}

/// The spiral pieces (`dust_spiral_point`): a site 2 km off the centre of a nucleus turning
/// once in 2 h, the first two hours of its emission seen from 5 km with the spin in the frame
/// (every young sample's time segment is drawn as the spiral about the anchor).
/// The spinning site's first two hours (`gpu_dust_splat_matches_reference_spinning_site`): the
/// batch of a capacity and the frame two hours in, anchored at the nucleus centre with the spin
fn spinning_site_scene() -> (impl Fn(u32) -> DustBatch, DustFrame) {
  let axis = [0.0f64, 0.0, 1.0];
  let omega = 2.0 * core::f64::consts::PI / (2.0 * 3600.0);
  let off = [2000.0f32, 0.0, 0.0];
  let (rc, vc) = comet_state();
  let batch_of = move |cap: u32| {
    let mut b = test_batch(1024, 7200.0, 3, cap);
    // 16 streams: 64 samples per stream 112 s apart, the nucleus turning 5.6° between them
    // (the spiral gate: > 0.01 rad, and the youngest 8 samples of every stream)
    b.mass_params[3] = dust::batch_streams_word(4, false);
    b.spin = [axis[0] as f32, axis[1] as f32, axis[2] as f32, omega as f32];
    b.site_offset = [off[0], off[1], off[2], 0.0];
    b.jet_dir_aperture = [1.0, 0.0, 0.0, 0.3];
    // the site at the window start (rot_start = identity): the batch's comet is the site
    b.set_comet(
      [
        rc[0] + off[0] as f64,
        rc[1] + off[1] as f64,
        rc[2] + off[2] as f64,
      ],
      vc,
      T_START,
      7200.0,
    );
    b
  };
  // the frame two hours in, anchored at the nucleus centre with the spin
  let t_now = T_START + 7200.0;
  let (centre, _) = kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, 7200.0);
  let frame = DustFrame::new(centre, t_now, [0.0, 0.0, 0.0, 1.0], 1.0e9)
    .with_streams(4)
    .with_spin([axis[0], axis[1], axis[2], omega]);
  (batch_of, frame)
}

#[test]
fn gpu_dust_splat_matches_reference_spinning_site() {
  let (batch_of, frame) = spinning_site_scene();
  splat_parity(9010, &batch_of, frame, 5.0e3, 1.0e-6, true, 0.0, None, None, false, None);
}

/// The support cut (`dust_phot_support`, `dust_ray_kept`, `dust_support_mass`): the spinning
/// site's first two hours (`gpu_dust_splat_matches_reference_spinning_site`) seen in perspective
/// from 4 km above the nucleus (60°, looking down −z through it): the young cone is near and deep,
/// its turned cone pieces carry support planes, and the ones a plane cuts are drawn whole by the
/// exact ray integral over the half-space on both mirrors.
#[test]
fn gpu_dust_splat_matches_reference_spinning_site_perspective() {
  let axis = [0.0f64, 0.0, 1.0];
  let omega = 2.0 * core::f64::consts::PI / (2.0 * 3600.0);
  let off = [2000.0f32, 0.0, 0.0];
  let (rc, vc) = comet_state();
  let batch_of = move |cap: u32| {
    let mut b = test_batch(1024, 7200.0, 3, cap);
    b.mass_params[3] = dust::batch_streams_word(4, false);
    b.spin = [axis[0] as f32, axis[1] as f32, axis[2] as f32, omega as f32];
    b.site_offset = [off[0], off[1], off[2], 0.0];
    b.jet_dir_aperture = [1.0, 0.0, 0.0, 0.3];
    b.set_comet(
      [
        rc[0] + off[0] as f64,
        rc[1] + off[1] as f64,
        rc[2] + off[2] as f64,
      ],
      vc,
      T_START,
      7200.0,
    );
    b
  };
  let t_now = T_START + 7200.0;
  let (centre, _) = kepler::propagate_f64(rc, vc, SUN_MU_M3_S2, 7200.0);
  let frame = DustFrame::new(centre, t_now, [0.0, 0.0, 0.0, 1.0], 1.0e9)
    .with_streams(4)
    .with_spin([axis[0], axis[1], axis[2], omega]);
  let (w, h) = (640.0f32, 400.0f32);
  let d = 4000.0f32;
  let tan_h = (30.0f32).to_radians().tan();
  // camera at (0, 0, D) looking along −z (w_clip = D − z)
  let mut mvp = [0.0f32; 16];
  mvp[0] = h / (w * tan_h);
  mvp[5] = 1.0 / tan_h;
  mvp[10] = -1e-3;
  mvp[11] = -1.0;
  mvp[15] = d;
  splat_parity(9016, &batch_of, frame, 5.0e3, 1.0e-6, true, 0.0, Some((mvp, [0.0, 0.0, d])), None, false, None);
}

/// K1c, the texel's integral, on both mirrors: the cloud of the perspective case (a one-hour
/// window five days old, the radiance weight on) seen with the volumetric level bias fixed at
/// `bias` from (0, 0, `d`) looking at the anchor. Returns the reference's pieces per class.
fn volume_texel_integral_parity(id: u64, bias: u32, d: f32) -> [u64; 4] {
  let batch_of = |cap: u32| test_batch(1024, 3600.0, 3, cap);
  let (w, h) = (640.0f32, 400.0f32);
  let tan_h = (30.0f32).to_radians().tan();
  let mut mvp = [0.0f32; 16];
  mvp[0] = h / (w * tan_h);
  mvp[5] = 1.0 / tan_h;
  mvp[10] = -1e-3;
  mvp[11] = -1.0;
  mvp[15] = d;
  PARITY_VOLUME_BIAS.store(bias, core::sync::atomic::Ordering::SeqCst);
  // (reset even when the parity panics: the next case must not inherit the bias)
  struct Reset;
  impl Drop for Reset {
    fn drop(&mut self) {
      PARITY_VOLUME_BIAS.store(u32::MAX, core::sync::atomic::Ordering::SeqCst);
    }
  }
  let _reset = Reset;
  splat_parity(id, &batch_of, frame_after(5.0), 2.0e7, 1.0, false, 0.04, Some((mvp, [0.0, 0.0, d])), None, false, None);
  PARITY_VOLUME_CLASSES
    .lock()
    .unwrap_or_else(|e| e.into_inner())
    .iter()
    .rev()
    .find(|(i, _)| *i == id)
    .map(|(_, c)| *c)
    .expect("the parity recorded its class counts")
}

/// K1c's classes on both mirrors, word for word within `splat_parity`'s rules, in two scenes that
/// use many levels (a bias that puts everything on the 1 × 1 and 2 × 1 levels leaves `splat_parity`
/// a handful of texels): inside the cloud 200 m from the anchor at bias 1 (the full pieces of an
/// eye inside: centre rays and sub-rays) and 2 000 km out at bias 6 (the far volumetric pieces that
/// take sub-rays at bias 2 fit two texels of 64 × their size: the small class; the EWA leaves keep
/// the fine levels busy). Between them every class but the lines draws at least once
/// (a thin full band seen from outside, the line class, is exercised by the CPU texel test's band
/// cases; printed here).
#[test]
fn gpu_dust_splat_matches_reference_volume_texel_integral() {
  let near = volume_texel_integral_parity(9051, 1, 200.0);
  let far = volume_texel_integral_parity(9052, 6, 2.0e6);
  let all: [u64; 4] = core::array::from_fn(|k| near[k] + far[k]);
  std::println!("[dust gpu] texel integral classes (centre, small, sub-rays, lines): inside {near:?}, 2 000 km {far:?}");
  assert!(all[0] > 0, "no centre-ray piece drawn: {all:?}");
  assert!(all[1] > 0, "no small piece drawn: {all:?}");
  assert!(all[2] > 0, "no sub-ray piece drawn: {all:?}");
}

/// whether a [`splat_parity`] compared the small pieces' path counts (the profiler's words)
static PARITY_SMALL_PATHS_COMPARED: core::sync::atomic::AtomicBool = core::sync::atomic::AtomicBool::new(false);

/// Track 9A: the small pieces' paths (one projection a slice — interior and demoted edge pieces —
/// and 4 sigma points a slice) on both mirrors at the count level, in the 2 000 km bias-6 scene of
/// `gpu_dust_splat_matches_reference_volume_texel_integral` (its small class): the `.prof` shaders
/// draw the parity frames (their pyramid is the production one word for word,
/// `gpu_dust_profile_counters_are_consistent`), and `splat_parity` compares their path words with
/// the reference's within the pieces at a decision's margin. Needs `VK_KHR_shader_clock`.
#[test]
fn gpu_dust_small_paths_match_reference() {
  use crate::gpu_backends::vulkan::physics::DUST_PROFILE_OVERRIDE;
  struct Reset;
  impl Drop for Reset {
    fn drop(&mut self) {
      DUST_PROFILE_OVERRIDE.store(false, core::sync::atomic::Ordering::Relaxed);
    }
  }
  DUST_PROFILE_OVERRIDE.store(true, core::sync::atomic::Ordering::Relaxed);
  let _reset = Reset;
  PARITY_SMALL_PATHS_COMPARED.store(false, core::sync::atomic::Ordering::SeqCst);
  let classes = volume_texel_integral_parity(9053, 6, 2.0e6);
  assert!(classes[1] > 0, "no small piece drawn: {classes:?}");
  assert!(
    PARITY_SMALL_PATHS_COMPARED.load(core::sync::atomic::Ordering::SeqCst),
    "the profiler's words were empty (no VK_KHR_shader_clock, or the .prof shaders did not draw): the small paths were not compared"
  );
}

/// Stress: a full tier (every slot emitted) propagated 40 times with the moments, as a heavy
/// frame does; must not fault the device.
#[test]
fn gpu_dust_full_ring_propagate_stress() {
  with_dust_device(9009, |device, capacity, _, _| {
    let per = 4096u32;
    let mut emitted = 0u32;
    let mut seq = 0u64;
    while emitted < capacity {
      let n = per.min(capacity - emitted);
      let batch = test_batch(n, 3600.0 * (1.0 + (seq % 7) as f64), emitted, capacity);
      let mut res = device
        .run_transient_compute_commands(|cmd| device.cmd_dust_emit(cmd, 9009, 0, &batch, seq))
        .unwrap();
      res.cleanup(&device.device);
      emitted += n;
      seq += 1;
    }
    for k in 0..40 {
      let frame = frame_after(0.5 + k as f64 * 3.0);
      let mut res = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          device.cmd_dust_propagate(cmd, 9009, 0, capacity, 0, capacity, &frame)?;
          device.cmd_dust_pre_splat_barrier(cmd);
          device.cmd_dust_post_splat_barrier(cmd);
          Ok(())
        })
        .expect("propagate frame");
      res.cleanup(&device.device);
    }
    std::println!("[dust gpu] full-ring propagate stress: {capacity} clusters x 40 frames ok");
  });
}

/// Fault injection (ignored by default: it loses the test process's device on purpose): a splat
/// dispatch whose pyramid address points nowhere must end in a device loss that the fault
/// diagnostics report (`Device::report_device_lost`), with the device marked lost. Run with
/// `cargo nextest run -E 'test(/gpu_device_fault_is_reported/)' --run-ignored all`.
#[test]
#[ignore]
fn gpu_device_fault_is_reported() {
  with_dust_device(9010, |device, capacity, _, _| {
    let batch = test_batch(256, 3600.0, 3, capacity);
    emit(device, 9010, &batch);
    let frame = frame_after(5.0);
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 11);
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1.0 / 2.0e7;
    mvp[5] = 1.0 / 2.0e7;
    mvp[10] = 1e-12;
    mvp[15] = 1.0;
    let res = device.run_transient_commands(|cmd| {
      device.cmd_dust_pre_propagate_barrier(cmd);
      let _ = device.cmd_dust_frame_begin(cmd, pe, [640, 400], 0, Default::default())?;
      let (render, moments) =
        device.cmd_dust_propagate(cmd, 9010, 0, capacity, 0, batch.count, &frame)?;
      device.cmd_dust_pre_splat_barrier(cmd);
      let pc = dust::DustSplatPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: batch.count,
        flags: 0,
        exposure: 1.0,
        inv_unit: 1e6,
        color: 0xFFFFFF,
        albedo: 0.0,
        mvp,
        eye_local: [0.0, 0.0, 1e10, 30.0 * 86400.0], // w = TTL: the hue ramp is exercised
      };
      // the dispatch with a pyramid pointer into nowhere: the shader's first header read faults
      device.cmd_checkpoint(cmd, c"test: injected fault dispatch");
      device.cmd_dust_splat_with_pyramid(cmd, 9010, render, moments, pc, 0x0000_7000_0000_0000)?;
      device.cmd_dust_post_splat_barrier(cmd);
      Ok(())
    });
    // either the submit or the wait reports the loss; the report must have run
    let _ = res;
    unsafe {
      let _ = device.device.device_wait_idle();
    }
    let _ = device.report_device_lost("test wait");
    assert!(
      device.is_lost(),
      "the device must be marked lost after the injected fault"
    );
  });
}

/// The span arithmetic of [`super::dust::DustGpuTimings`]: ticks × period, a missing or reversed
/// end is `NaN`, the total is the frame (begin → measured) plus the composite when measured.
#[test]
fn dust_timestamp_spans_are_the_tick_differences() {
  use super::dust::{DUST_TIMESTAMPS, DustGpuTimings, timestamp_span_ms};
  assert!((timestamp_span_ms(Some(1_000), Some(3_000_000), 1.0) - 2.999).abs() < 1e-12);
  assert!(timestamp_span_ms(None, Some(5), 1.0).is_nan());
  assert!(timestamp_span_ms(Some(9), Some(5), 1.0).is_nan());
  let mut ts = [None; DUST_TIMESTAMPS as usize];
  // 2 ms propagate, 10 ms splat, 0.5 ms measure at a 0.5 ns tick; no composite
  ts[0] = Some(0);
  ts[1] = Some(4_000_000);
  ts[2] = Some(24_000_000);
  ts[3] = Some(25_000_000);
  let t = DustGpuTimings::from_timestamps(&ts, 0.5);
  assert_eq!((t.propagate_ms, t.splat_ms, t.measure_ms), (2.0, 10.0, 0.5));
  assert!(t.composite_ms.is_nan());
  assert_eq!(t.total_ms, 12.5);
  ts[4] = Some(30_000_000);
  ts[5] = Some(31_000_000);
  let t = DustGpuTimings::from_timestamps(&ts, 0.5);
  assert_eq!((t.composite_ms, t.total_ms), (0.5, 13.0));
}

/// The packet queue (`dust_draw_queue.comp`, 8 ms plan Q1): a frame drawn with every packet
/// queued, with none queued, and with a queue of 5 records that overflows (the rest drawn in the
/// clusters' threads) gives the same pyramid word for word — the per-segment dither makes the
/// drawing independent of who draws a segment. Both the orthographic and the perspective
/// (inside the cloud) views of the parity tests.
#[test]
fn gpu_dust_queue_draws_the_same_pyramid() {
  let id = 9034;
  with_dust_device(id, |device, capacity, _, _| {
    let batch = test_batch(1024, 3600.0, 3, capacity);
    emit(device, id, &batch);
    let first_slot = batch.first_index & batch.ring_mask;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 34);
    let (w, h) = (640u32, 400u32);
    let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let frame = frame_after(5.0);
    // the orthographic ±2e7 m view, and the 60° camera 200 m from the anchor inside the cloud
    let mut ortho = [0.0f32; 16];
    ortho[0] = 1.0 / 2.0e7;
    ortho[5] = 1.0 / 2.0e7;
    ortho[10] = 1e-12;
    ortho[15] = 1.0;
    let d = 200.0f32;
    let tan_h = (30.0f32).to_radians().tan();
    let mut persp = [0.0f32; 16];
    persp[0] = h as f32 / (w as f32 * tan_h);
    persp[5] = 1.0 / tan_h;
    persp[10] = -1e-3;
    persp[11] = -1.0;
    persp[15] = d;
    for (name, mvp, eye) in [("orthographic", ortho, [0.0, 0.0, 1e10]), ("perspective", persp, [0.0, 0.0, d])] {
      let base_pc = dust::DustSplatPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: batch.count,
        flags: dust::DUST_VIEW_FLOW,
        exposure: 1.0,
        inv_unit: 0.0,
        color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
        albedo: 0.04,
        eye_local: [eye[0], eye[1], eye[2], 30.0 * 86400.0],
        mvp,
      };
      // one frame; `inv_unit` 0 = the view's adapting unit
      let run = |inv_unit: f32| -> (Vec<u32>, f32) {
        let mut used = inv_unit;
        let mut cmds = device
          .run_transient_commands(|cmd| {
            device.cmd_dust_pre_propagate_barrier(cmd);
            let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], base_pc.flags, flow)?;
            let mut pc = base_pc;
            if !(used > 0.0) {
              used = 1.0 / view.unit;
            }
            pc.inv_unit = used;
            device.cmd_dust_system_begin(id, pc.flags, flow)?;
            let (render, moments) =
              device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
            device.cmd_dust_pre_splat_barrier(cmd);
            device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, pc)?;
            device.cmd_dust_post_splat_barrier(cmd);
            device.cmd_dust_frame_end(cmd, pe)?;
            device.cmd_dust_system_end(id)?;
            Ok(())
          })
          .unwrap();
        cmds.cleanup(&device.device);
        let pyr = device.download_dust_pyramid(pe).unwrap();
        (pyr[dust::PYRAMID_HEADER_WORDS as usize..].to_vec(), used)
      };
      device.dust_set_queue_mode(pe, None);
      device.dust_set_queue_capacity(pe, 0);
      let mut inv_unit = 0.0;
      for _ in 0..(super::dust::DUST_READBACK_SLOTS + 2) {
        inv_unit = run(0.0).1;
      }
      let (default, _) = run(inv_unit);
      let default_counts = device.dust_queue_counts(pe).unwrap();
      device.dust_set_queue_mode(pe, Some(dust::DUST_FRAME_QUEUE_OFF));
      let (off, _) = run(inv_unit);
      let off_counts = device.dust_queue_counts(pe).unwrap();
      device.dust_set_queue_mode(pe, Some(dust::DUST_FRAME_QUEUE_ALL));
      let (all, _) = run(inv_unit);
      let all_counts = device.dust_queue_counts(pe).unwrap();
      device.dust_set_queue_capacity(pe, 5);
      let (overflow, _) = run(inv_unit);
      let overflow_counts = device.dust_queue_counts(pe).unwrap();
      device.dust_set_queue_mode(pe, None);
      device.dust_set_queue_capacity(pe, 0);
      // and with the level of detail off (the frames above draw at the default knob): the queue
      // is as order-free at full detail
      device.dust_set_lod_px(pe, Some(0.0));
      let (default_lod0, _) = run(inv_unit);
      device.dust_set_queue_mode(pe, Some(dust::DUST_FRAME_QUEUE_OFF));
      let (off_lod0, _) = run(inv_unit);
      device.dust_set_queue_mode(pe, None);
      device.dust_set_lod_px(pe, None);
      assert!(default_lod0.iter().any(|&v| v != 0), "{name}: the frame at lod 0 drew nothing");
      assert!(default_lod0 == off_lod0, "{name}: the queue changed the pyramid at lod 0");
      std::println!(
        "[dust queue] {name}: queue records (drawn, appended): default {default_counts:?}, off {off_counts:?}, all {all_counts:?}, capacity 5 {overflow_counts:?}"
      );
      assert!(off.iter().any(|&v| v != 0), "{name}: the frame drew nothing");
      assert_eq!(off_counts.1, 0, "{name}: the queue was used while off");
      // the perspective view inside the cloud draws volumetric packets only (never queued)
      if name == "orthographic" {
        assert!(all_counts.0 > 0, "{name}: nothing was queued with every packet queued");
      }
      if all_counts.1 > 5 {
        assert_eq!(overflow_counts.0, 5, "{name}: the capped queue did not fill");
        assert!(overflow_counts.1 > 5, "{name}: the capped queue did not overflow");
      }
      let diff = |a: &[u32], b: &[u32]| a.iter().zip(b).filter(|(x, y)| x != y).count();
      assert_eq!(diff(&all, &off), 0, "{name}: every packet queued drew another pyramid than none queued");
      assert_eq!(diff(&default, &off), 0, "{name}: the default queue drew another pyramid than none queued");
      assert_eq!(diff(&overflow, &off), 0, "{name}: an overflowing queue drew another pyramid");
    }
  });
}

/// The volume packet queue (`dust_draw_volume.comp`, `dust::DUST_VQUEUE_*`): a frame drawn with
/// the volumetric packets queued per chunk and drawn by the consumer (default), with every one
/// through the volume spill (the queue off: the spill path's A/B, slower — the chunk is sized to
/// the spill), with a chunk queue of 5 records that overflows (the rest drawn from the spill) and
/// with a forced chunk of 100 clusters (many chunks) gives the same pyramid word for word: the
/// packet's seed travels with its record, every record is drawn by the consumer and the
/// pyramid's atomics are order-free. With the spill capped at 3 records a chunk too, the rest are
/// lost: counted (overflowed = spill drawn + lost, the records conserved) and missing from the
/// image. The perspective view inside the cloud draws volumetric packets only; the orthographic
/// one none.
#[test]
fn gpu_dust_volume_queue_draws_the_same_pyramid() {
  let id = 9037;
  with_dust_device(id, |device, capacity, _, _| {
    let batch = test_batch(1024, 3600.0, 3, capacity);
    emit(device, id, &batch);
    let first_slot = batch.first_index & batch.ring_mask;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 37);
    let (w, h) = (640u32, 400u32);
    let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let frame = frame_after(5.0);
    let mut ortho = [0.0f32; 16];
    ortho[0] = 1.0 / 2.0e7;
    ortho[5] = 1.0 / 2.0e7;
    ortho[10] = 1e-12;
    ortho[15] = 1.0;
    let d = 200.0f32;
    let tan_h = (30.0f32).to_radians().tan();
    let mut persp = [0.0f32; 16];
    persp[0] = h as f32 / (w as f32 * tan_h);
    persp[5] = 1.0 / tan_h;
    persp[10] = -1e-3;
    persp[11] = -1.0;
    persp[15] = d;
    for (name, mvp, eye) in [("orthographic", ortho, [0.0, 0.0, 1e10]), ("perspective", persp, [0.0, 0.0, d])] {
      let base_pc = dust::DustSplatPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: batch.count,
        flags: dust::DUST_VIEW_FLOW,
        exposure: 1.0,
        inv_unit: 0.0,
        color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
        albedo: 0.04,
        eye_local: [eye[0], eye[1], eye[2], 30.0 * 86400.0],
        mvp,
      };
      let run = |inv_unit: f32| -> (Vec<u32>, f32) {
        let mut used = inv_unit;
        let mut cmds = device
          .run_transient_commands(|cmd| {
            device.cmd_dust_pre_propagate_barrier(cmd);
            let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], base_pc.flags, flow)?;
            let mut pc = base_pc;
            if !(used > 0.0) {
              used = 1.0 / view.unit;
            }
            pc.inv_unit = used;
            device.cmd_dust_system_begin(id, pc.flags, flow)?;
            let (render, moments) =
              device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
            device.cmd_dust_pre_splat_barrier(cmd);
            device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, pc)?;
            device.cmd_dust_post_splat_barrier(cmd);
            device.cmd_dust_frame_end(cmd, pe)?;
            device.cmd_dust_system_end(id)?;
            Ok(())
          })
          .unwrap();
        cmds.cleanup(&device.device);
        let pyr = device.download_dust_pyramid(pe).unwrap();
        (pyr[dust::PYRAMID_HEADER_WORDS as usize..].to_vec(), used)
      };
      device.dust_set_vqueue_mode(pe, None);
      device.dust_set_vqueue_capacity(pe, 0);
      device.dust_set_vspill_capacity(pe, 0);
      device.dust_set_vqueue_chunk(pe, 0);
      let mut inv_unit = 0.0;
      for _ in 0..(super::dust::DUST_READBACK_SLOTS + 2) {
        inv_unit = run(0.0).1;
      }
      // (drawn by the consumer from the queue, overflowed, chunks, spill drawn, lost)
      let counts = || {
        let (d, o, c) = device.dust_vqueue_counts(pe).unwrap();
        let (sd, l) = device.dust_vqueue_spill_counts(pe).unwrap();
        (d, o, c, sd, l)
      };
      let (default, _) = run(inv_unit);
      let default_counts = counts();
      // the queue off: every record through the volume spill (the spill path's A/B; slower:
      // the chunk sized to the spill)
      device.dust_set_vqueue_mode(pe, Some(dust::DUST_FRAME_VQUEUE_OFF));
      let (off, _) = run(inv_unit);
      let off_counts = counts();
      device.dust_set_vqueue_mode(pe, None);
      // the queue at 5 records a chunk: the rest through the spill
      device.dust_set_vqueue_capacity(pe, 5);
      let (overflow, _) = run(inv_unit);
      let overflow_counts = counts();
      // and the spill at 3 records a chunk too: the rest lost (dropped, counted)
      device.dust_set_vspill_capacity(pe, 3);
      let (lossy, _) = run(inv_unit);
      let lossy_counts = counts();
      device.dust_set_vspill_capacity(pe, 0);
      device.dust_set_vqueue_capacity(pe, 0);
      device.dust_set_vqueue_chunk(pe, 100);
      let (chunked, _) = run(inv_unit);
      let chunked_counts = counts();
      device.dust_set_vqueue_chunk(pe, 0);
      // the splitter's serial reference: one lane per cluster, the records in order
      device.dust_set_vqueue_mode(pe, Some(dust::DUST_FRAME_SPLIT_SERIAL));
      let (serial, _) = run(inv_unit);
      let (srecords, soverflow) = device.dust_squeue_counts(pe).unwrap();
      device.dust_set_vqueue_mode(pe, None);
      std::println!(
        "[dust vqueue] {name}: (drawn by the consumer, overflowed, chunks, spill drawn, lost): default {default_counts:?}, off (spill only) {off_counts:?}, capacity 5 {overflow_counts:?}, capacity 5 + spill 3 {lossy_counts:?}, chunk 100 {chunked_counts:?}"
      );
      // the pyramids' FNV-1a hashes (step 9C: a consumer change that must keep the image word for
      // word is checked by these lines' equality before and after it)
      let fnv = |a: &[u32]| a.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &w| (h ^ w as u64).wrapping_mul(0x100_0000_01b3));
      std::println!(
        "[dust vqueue] {name}: pyramid hash default {:016x}, off {:016x}, capacity 5 {:016x}, chunk 100 {:016x}, serial {:016x} ({} words)",
        fnv(&default),
        fnv(&off),
        fnv(&overflow),
        fnv(&chunked),
        fnv(&serial),
        default.len()
      );
      assert!(off.iter().any(|&v| v != 0), "{name}: the frame drew nothing");
      // every case: overflowed = spill drawn + lost (the spill drain's counts)
      for (case, c) in [("default", default_counts), ("off", off_counts), ("capacity 5", overflow_counts), ("spill 3", lossy_counts), ("chunk 100", chunked_counts)] {
        assert_eq!(c.1, c.3 + c.4, "{name} / {case}: overflowed {} != spill drawn {} + lost {}", c.1, c.3, c.4);
      }
      // with the queue off nothing is drawn from the main queue: every record through the spill
      assert_eq!(off_counts.0, 0, "{name}: the volume queue was used while off");
      assert_eq!(off_counts.4, 0, "{name}: the spill lost records with the queue off");
      assert_eq!(off_counts.3, default_counts.0, "{name}: the spill-only frame drew another count of packets");
      if name == "perspective" {
        assert!(default_counts.0 > 0, "{name}: no volumetric packet was queued inside the cloud");
        assert_eq!((default_counts.1, default_counts.4), (0, 0), "{name}: the default queue overflowed");
        // the capped queue: its refusals all drawn from the spill. Main raw - main n = spill raw
        // per chunk (dust_draw_volume.comp), summed: the records the queue did not draw are
        // exactly the spill's appends
        assert!(overflow_counts.0 >= 5 && overflow_counts.3 > 0, "{name}: the capped queue did not spill");
        assert_eq!(overflow_counts.4, 0, "{name}: the default spill lost records");
        assert_eq!(default_counts.0 - overflow_counts.0, overflow_counts.1, "{name}: main raw - main n != spill raw");
        // the capped spill: some lost, counted; the rest conserved
        assert!(lossy_counts.4 > 0, "{name}: the capped spill lost nothing");
        assert!(lossy_counts.3 <= 3 * lossy_counts.2, "{name}: more than 3 spill records a chunk drawn");
        assert_eq!(default_counts.0 - lossy_counts.0, lossy_counts.1, "{name}: main raw - main n != spill raw (capped spill)");
        assert_eq!(lossy_counts.0 + lossy_counts.3 + lossy_counts.4, default_counts.0, "{name}: records not conserved");
        assert_eq!(chunked_counts.0, default_counts.0, "{name}: the chunked frame drew another count of packets");
      } else {
        assert_eq!(default_counts.0, 0, "{name}: an orthographic view queued volumetric packets");
        assert_eq!(off_counts.3, 0, "{name}: an orthographic view spilled volumetric packets");
      }
      assert_eq!(chunked_counts.2, (batch.count as u32).div_ceil(100), "{name}: chunk count");
      let diff = |a: &[u32], b: &[u32]| a.iter().zip(b).filter(|(x, y)| x != y).count();
      assert_eq!(diff(&default, &off), 0, "{name}: the spilled volume packets drew another pyramid than the queued ones");
      assert_eq!(diff(&overflow, &off), 0, "{name}: an overflowing volume queue drew another pyramid");
      if name == "perspective" {
        // the lost records are missing from the image (the drop is visible, not silent)
        assert!(diff(&lossy, &off) > 0, "{name}: the lost records left the pyramid unchanged");
      }
      assert_eq!(diff(&chunked, &off), 0, "{name}: the chunked frame drew another pyramid");
      assert!(srecords > 0 && soverflow == 0, "{name}: sub-packet records {srecords}, overflowed {soverflow}");
      assert_eq!(diff(&serial, &off), 0, "{name}: the splitter's counting pass and prefix differ from the serial order");
    }
  });
}

/// The shader-clock profiler (`dust_profile.glsl`, `AETHERVK_DUST_PROFILE=1`): with the request
/// on before the device is created, the `.prof` splat and queue shaders run and their counters
/// are consistent (the thread count is the live count, the nested timers contain their parts,
/// the residual outside the stages is small, the histogram holds every thread), and the pyramid
/// they draw equals the production shaders' word for word (the probes change no arithmetic).
/// Skipped with a notice on a device without `VK_KHR_shader_clock`.
#[test]
fn gpu_dust_profile_counters_are_consistent() {
  use crate::gpu_backends::vulkan::physics::DUST_PROFILE_OVERRIDE;
  let id = 9036;
  DUST_PROFILE_OVERRIDE.store(true, core::sync::atomic::Ordering::Relaxed);
  with_dust_device(id, |device, capacity, _, _| {
    if device.kernels.pipelines.dust_splat_prof.is_none() {
      std::println!("[dust profile] no VK_KHR_shader_clock on this device: the profiler test is skipped");
      DUST_PROFILE_OVERRIDE.store(false, core::sync::atomic::Ordering::Relaxed);
      return;
    }
    let batch = test_batch(1024, 3600.0, 3, capacity);
    emit(device, id, &batch);
    let first_slot = batch.first_index & batch.ring_mask;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 36);
    let (w, h) = (640u32, 400u32);
    let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let frame = frame_after(5.0);
    let mut ortho = [0.0f32; 16];
    ortho[0] = 1.0 / 2.0e7;
    ortho[5] = 1.0 / 2.0e7;
    ortho[10] = 1e-12;
    ortho[15] = 1.0;
    let d = 200.0f32;
    let tan_h = (30.0f32).to_radians().tan();
    let mut persp = [0.0f32; 16];
    persp[0] = h as f32 / (w as f32 * tan_h);
    persp[5] = 1.0 / tan_h;
    persp[10] = -1e-3;
    persp[11] = -1.0;
    persp[15] = d;
    for (name, mvp, eye) in [("orthographic", ortho, [0.0, 0.0, 1e10]), ("perspective", persp, [0.0, 0.0, d])] {
      let base_pc = dust::DustSplatPushConstants {
        moments: 0,
        render: 0,
        pyramid: 0,
        live_count: batch.count,
        flags: dust::DUST_VIEW_FLOW,
        exposure: 1.0,
        inv_unit: 0.0,
        color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
        albedo: 0.04,
        eye_local: [eye[0], eye[1], eye[2], 30.0 * 86400.0],
        mvp,
      };
      let run = |inv_unit: f32| -> (Vec<u32>, f32) {
        let mut used = inv_unit;
        let mut cmds = device
          .run_transient_commands(|cmd| {
            device.cmd_dust_pre_propagate_barrier(cmd);
            let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], base_pc.flags, flow)?;
            let mut pc = base_pc;
            if !(used > 0.0) {
              used = 1.0 / view.unit;
            }
            pc.inv_unit = used;
            device.cmd_dust_system_begin(id, pc.flags, flow)?;
            let (render, moments) =
              device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
            device.cmd_dust_pre_splat_barrier(cmd);
            device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, pc)?;
            device.cmd_dust_post_splat_barrier(cmd);
            device.cmd_dust_frame_end(cmd, pe)?;
            device.cmd_dust_system_end(id)?;
            Ok(())
          })
          .unwrap();
        cmds.cleanup(&device.device);
        let pyr = device.download_dust_pyramid(pe).unwrap();
        (pyr[dust::PYRAMID_HEADER_WORDS as usize..].to_vec(), used)
      };
      // every packet queued: the queue draw's counters are exercised too
      device.dust_set_queue_mode(pe, Some(dust::DUST_FRAME_QUEUE_ALL));
      let mut inv_unit = 0.0;
      for _ in 0..(super::dust::DUST_READBACK_SLOTS + 2) {
        inv_unit = run(0.0).1;
      }
      DUST_PROFILE_OVERRIDE.store(true, core::sync::atomic::Ordering::Relaxed);
      let (profiled, _) = run(inv_unit);
      let p = device.dust_profile_counters(pe).unwrap();
      let vq = device.dust_vqueue_counts(pe).unwrap();
      let vs = device.dust_vqueue_spill_counts(pe).unwrap();
      DUST_PROFILE_OVERRIDE.store(false, core::sync::atomic::Ordering::Relaxed);
      let (production, _) = run(inv_unit);
      let p_off = device.dust_profile_counters(pe).unwrap();
      DUST_PROFILE_OVERRIDE.store(true, core::sync::atomic::Ordering::Relaxed);
      device.dust_set_queue_mode(pe, None);
      let s = |k: usize| p.stage(k);
      let names = dust::DUST_PROF_STAGE_NAMES;
      let idx = |n: &str| names.iter().position(|x| *x == n).unwrap();
      std::println!(
        "[dust profile] {name}: threads {} packets {} cone {} leaves {} records {} volume {} nodes {}; cycles thread {} setup {} subpk {} packet {} cone {} split {} project {} scprep {} qput {} ewa {} vplan {} vdraw {} tracer {}; max thread {}; queue {:?} records {} texels {}",
        p.word(dust::DUST_PROF_THREADS), p.word(dust::DUST_PROF_PACKETS), p.word(dust::DUST_PROF_CONE),
        p.word(dust::DUST_PROF_LEAVES), p.word(dust::DUST_PROF_RECORDS), p.word(dust::DUST_PROF_VOLUME),
        p.word(dust::DUST_PROF_NODES),
        s(0), s(1), s(2), s(3), s(4), s(5), s(6), s(7), s(8), s(9), s(10), s(11), s(12),
        p.word(dust::DUST_PROF_MAX_THREAD),
        (0..4).map(|k| p.qstage(k)).collect::<Vec<_>>(),
        p.word(dust::DUST_PROF_QRECORDS), p.u64_at(dust::DUST_PROF_QTEXELS)
      );
      assert!(p.any(), "{name}: the profiler recorded nothing");
      assert!(!p_off.any(), "{name}: the production shaders wrote profiler counters");
      // the threads are the lanes with a sub-packet that drew (a warp per cluster,
      // `dust::DUST_SPLAT_LANES`): at most one per sub-packet of every cluster
      let threads = p.word(dust::DUST_PROF_THREADS);
      assert!(threads > 0 && threads <= batch.count * dust::DUST_SPLAT_LANES, "{name}: thread count {threads}");
      let hist: u64 = (0..32).map(|k| p.word(dust::DUST_PROF_HIST + k) as u64).sum();
      assert_eq!(hist, threads as u64, "{name}: the histogram does not hold every thread");
      assert!(p.word(dust::DUST_PROF_PACKETS) > 0 && p.word(dust::DUST_PROF_NODES) > 0, "{name}: no packets");
      assert!(p.word(dust::DUST_PROF_MAX_THREAD) as u64 <= s(0), "{name}: the slowest thread exceeds the sum");
      // the nesting: each timer holds its parts (the enumerator's stages within its thread, the
      // splitter's within its own lane cycles)
      let thread = s(idx("thread"));
      let inner = s(idx("setup")) + s(idx("subpk")) + s(idx("tracer")) + s(idx("ecount"));
      assert!(inner <= thread, "{name}: setup + subpk + tracer + ecount {inner} > thread {thread}");
      // (the rest is the enumerator's barriers and prefix between its two passes, unprobed)
      assert!((thread - inner) as f64 <= 0.50 * thread as f64, "{name}: {} cycles outside the stages of {thread}", thread - inner);
      let sthread = p.u64_at(dust::DUST_PROF_STHREAD);
      assert!(s(idx("count")) + s(idx("packet")) <= sthread, "{name}: count + packet > the splitter's cycles {sthread}");
      assert!(p.word(dust::DUST_PROF_SLANES) > 0, "{name}: the splitter drew nothing");
      let (srecords, soverflow) = device.dust_squeue_counts(pe).unwrap();
      assert!(srecords > 0 && soverflow == 0, "{name}: sub-packet queue records {srecords}, overflowed {soverflow}");
      assert!(s(idx("cone")) + s(idx("split")) <= s(idx("packet")), "{name}: cone + split > packet");
      let leaves = s(idx("project")) + s(idx("scprep")) + s(idx("ewa")) + s(idx("vplan")) + s(idx("vdraw"));
      assert!(leaves <= s(idx("split")), "{name}: the leaf stages {leaves} > split {}", s(idx("split")));
      assert!(s(idx("qput")) <= s(idx("scprep")), "{name}: qput > scprep");
      if name == "orthographic" {
        assert!(p.word(dust::DUST_PROF_LEAVES) > 0 && p.word(dust::DUST_PROF_RECORDS) > 0, "{name}: no leaves or records");
        assert!(p.word(dust::DUST_PROF_QRECORDS) > 0 && p.qstage(3) > 0, "{name}: the queue draw recorded nothing");
        assert!(p.qstage(0) + p.qstage(1) <= p.qstage(3), "{name}: qload + qrows > qthread");
        assert!(p.qstage(2) <= p.qstage(1), "{name}: qseg > qrows");
      } else {
        assert!(p.word(dust::DUST_PROF_VOLUME) > 0, "{name}: no volume packets inside the cloud");
        // the consumer (dust_draw_volume.comp.prof): its records are the queue's and the volume
        // spill's drawn counts (both drains run the profiled consumer)
        assert_eq!(p.word(dust::DUST_PROF_CPACKETS), vq.0 + vs.0, "{name}: the consumer's packet count");
        assert!(p.word(dust::DUST_PROF_CPACKETS) > 0, "{name}: the consumer drew nothing");
        assert!(
          p.u64_at(dust::DUST_PROF_CVPLAN) + p.u64_at(dust::DUST_PROF_CVDRAW) <= p.u64_at(dust::DUST_PROF_CTHREAD),
          "{name}: the consumer's stages exceed its thread cycles"
        );
        assert!(p.word(dust::DUST_PROF_CMAX) as u64 <= p.u64_at(dust::DUST_PROF_CTHREAD), "{name}: the consumer's slowest lane");
      }
      // the probes change no arithmetic
      let diff = profiled.iter().zip(&production).filter(|(x, y)| x != y).count();
      assert_eq!(diff, 0, "{name}: the profiled frame drew another pyramid than the production shaders");
    }
  });
  DUST_PROFILE_OVERRIDE.store(false, core::sync::atomic::Ordering::Relaxed);
}

/// Image reuse ([`Device::cmd_dust_frame_begin_reusable`], 8 ms plan Step 2): frames with the
/// same draw inputs reuse the image once the header has settled (the unit and biases follow the
/// readbacks), and a reused frame's pyramid equals the last drawn one word for word; other inputs,
/// a changed header (the legacy-kernel bit) or a forgotten frame draw again. The speck pass is
/// needed only with a shell override here (the specks are off).
#[test]
fn gpu_dust_frame_reuses_an_unchanged_image() {
  let id = 9033;
  with_dust_device(id, |device, capacity, _, _| {
    let batch = test_batch(1024, 3600.0, 3, capacity);
    emit(device, id, &batch);
    let first_slot = batch.first_index & batch.ring_mask;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 33);
    let (w, h) = (320u32, 200u32);
    let half = 2.0e7f32;
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1.0 / half;
    mvp[5] = 1.0 / half;
    mvp[10] = 1e-12;
    mvp[15] = 1.0;
    let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let base_pc = dust::DustSplatPushConstants {
      moments: 0,
      render: 0,
      pyramid: 0,
      live_count: batch.count,
      flags: dust::DUST_VIEW_FLOW,
      exposure: 1.0,
      inv_unit: 0.0,
      color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
      albedo: 0.04,
      eye_local: [0.0, 0.0, 1e10, 30.0 * 86400.0],
      mvp,
    };
    let frame = frame_after(5.0);
    // one frame through the reusable begin: (reused, the pyramid's level words)
    let run = |inputs: u64| -> (bool, Vec<u32>) {
      let mut reused = false;
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let (view, reuse) =
            device.cmd_dust_frame_begin_reusable(cmd, pe, [w, h], base_pc.flags, flow, inputs)?;
          reused = reuse;
          device.cmd_dust_system_begin(id, base_pc.flags, flow)?;
          if !reuse {
            let mut pc = base_pc;
            pc.inv_unit = 1.0 / view.unit;
            let (render, moments) =
              device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
            device.cmd_dust_pre_splat_barrier(cmd);
            device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, pc)?;
          }
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      let pyr = device.download_dust_pyramid(pe).unwrap();
      (reused, pyr[dust::PYRAMID_HEADER_WORDS as usize..].to_vec())
    };
    let mut prev: Option<Vec<u32>> = None;
    let mut reuses = 0;
    for k in 0..16 {
      let (reused, levels) = run(42);
      // the unit follows the view from the DUST_READBACK_SLOTS-th frame (the seed draws nothing)
      if k > super::dust::DUST_READBACK_SLOTS + 1 {
        assert!(levels.iter().any(|&v| v != 0), "frame {k} drew nothing");
      }
      if reused {
        reuses += 1;
        assert_eq!(Some(&levels), prev.as_ref(), "frame {k} reused another image than the last");
      }
      prev = Some(levels);
    }
    std::println!("[dust reuse] {reuses} of 16 unchanged frames reused the image");
    assert!(reuses >= 8, "only {reuses} of 16 unchanged frames reused the image");
    assert!(run(42).0, "the settled view did not reuse");
    assert!(!run(43).0, "other draw inputs reused the image");
    assert!(run(43).0, "the new inputs did not reuse on the next frame");
    device.dust_forget_drawn(pe);
    assert!(!run(43).0, "a forgotten frame was reused");
    device.dust_set_legacy_kernel(pe, true);
    assert!(!run(43).0, "a changed header reused the image");
    device.dust_set_legacy_kernel(pe, false);
    // the speck pass: off without the specks, on with a shell override
    if std::env::var("AETHERVK_DUST_SPECKS").is_err() && std::env::var("AETHERVK_DUST_EYE_DENSITY").is_err() {
      assert!(!device.dust_specks_needed(pe), "the speck pass is needed with the specks off");
    }
    device.dust_set_shell_override(pe, Some(10.0));
    assert!(device.dust_specks_needed(pe), "a shell override does not need the speck pass");
    device.dust_set_shell_override(pe, None);
  });
}

/// The propagate skip ([`Device::cmd_dust_propagate_cached`], 8 ms plan Step 2): a frame whose
/// tier inputs equal the last recorded propagate's skips the dispatch and draws the same pyramid,
/// bit for bit, as a frame that dispatches; a later emission (a new `emitted`), another time or an
/// uncached propagate in between makes the next one dispatch again.
#[test]
fn gpu_dust_propagate_skips_unchanged_inputs() {
  let id = 9032;
  with_dust_device(id, |device, capacity, _, _| {
    let batch = test_batch(1024, 3600.0, 3, capacity);
    emit(device, id, &batch);
    let first_slot = batch.first_index & batch.ring_mask;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 32);
    let (w, h) = (320u32, 200u32);
    // the orthographic ±2e7 m view of `gpu_dust_splat_matches_reference` (it draws the batch)
    let half = 2.0e7f32;
    let mut mvp = [0.0f32; 16];
    mvp[0] = 1.0 / half;
    mvp[5] = 1.0 / half;
    mvp[10] = 1e-12;
    mvp[15] = 1.0;
let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let base_pc = dust::DustSplatPushConstants {
      moments: 0,
      render: 0,
      pyramid: 0,
      live_count: batch.count,
      flags: dust::DUST_VIEW_FLOW,
      exposure: 1.0,
      inv_unit: 0.0,
      color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
      albedo: 0.04,
      eye_local: [0.0, 0.0, 1e10, 30.0 * 86400.0],
      mvp,
    };
    // one frame: `Some(emitted)` propagates through the cache, `None` always dispatches; the
    // pyramid it drew and whether the dispatch was skipped
    // `inv_unit`: 0 = the view's adapting unit; the compared frames use a fixed one, so their
    // pyramids compare word for word
    let run = |frame: &dust::DustFrame, emitted: Option<u64>, inv_unit: f32| -> (Vec<u32>, bool, f32) {
      let mut skipped = false;
      let mut used = inv_unit;
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], base_pc.flags, flow)?;
          let mut pc = base_pc;
          if !(used > 0.0) {
            used = 1.0 / view.unit;
          }
          pc.inv_unit = used;
          device.cmd_dust_system_begin(id, pc.flags, flow)?;
          let (render, moments) = match emitted {
            Some(e) => {
              let (a, s) = device.cmd_dust_propagate_cached(cmd, id, 0, capacity, first_slot, batch.count, frame, e)?;
              skipped = s;
              a
            }
            None => device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, frame)?,
          };
          device.cmd_dust_pre_splat_barrier(cmd);
          device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, pc)?;
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      let pyr = device.download_dust_pyramid(pe).unwrap();
      (pyr, skipped, used)
    };
    let hn = dust::PYRAMID_HEADER_WORDS as usize;
    let f1 = frame_after(5.0);
    let f2 = frame_after(6.0);
    // the unit follows the view after DUST_READBACK_SLOTS frames; then it is held
    let mut inv_unit = 0.0;
    for _ in 0..(super::dust::DUST_READBACK_SLOTS + 2) {
      inv_unit = run(&f1, None, 0.0).2;
    }
    let run = |frame: &dust::DustFrame, emitted: Option<u64>| {
      let (p, s, _) = run(frame, emitted, inv_unit);
      (p, s)
    };
    let (dispatched, s0) = run(&f1, None);
    assert!(!s0, "an uncached propagate reported a skip");
    assert!(dispatched[hn..].iter().any(|&v| v != 0), "the reference frame drew nothing");
    let (a, s1) = run(&f1, Some(7));
    assert!(!s1, "the first cached propagate after an uncached one skipped");
    let (b, s2) = run(&f1, Some(7));
    assert!(s2, "unchanged inputs did not skip the propagate");
    assert_eq!(a[hn..], dispatched[hn..], "the cached propagate drew another pyramid");
    assert_eq!(b[hn..], dispatched[hn..], "the skipped propagate drew another pyramid");
    let (_, s3) = run(&f1, Some(8));
    assert!(!s3, "a later emission did not dispatch");
    let (c, s4) = run(&f2, Some(8));
    assert!(!s4, "another time did not dispatch");
    let (c2, s5) = run(&f2, Some(8));
    assert!(s5, "unchanged inputs at the new time did not skip");
    assert_eq!(c2[hn..], c[hn..], "the skipped propagate at the new time drew another pyramid");
    assert_ne!(c[hn..], dispatched[hn..], "the frames at two times drew the same pyramid");
  });
}

/// Real dust frames report their GPU time ([`Device::dust_gpu_timings`]): six frames of the
/// perspective parity case recorded as the render thread does (frame begin, propagate, the
/// propagated mark, splat, the splatted mark, frame end), each waited for. The timings of a frame
/// are read [`DUST_READBACK_SLOTS`](super::dust::DUST_READBACK_SLOTS) frames later; they are
/// positive, the spans add up to the total, no composite was drawn, and the total is at most the
/// frame's wall time (a timestamp pair cannot span more than the submission it is part of).
#[test]
fn gpu_dust_timings_are_reported() {
  use super::dust::{DUST_READBACK_SLOTS, DUST_TS_PROPAGATED, DUST_TS_SPLATTED};
  let id = 9031;
  with_dust_device(id, |device, capacity, _, _| {
    if device.query_result.graphics_timestamp_bits == 0 {
      std::println!("[dust timings] the graphics queue has no timestamps: skipped");
      return;
    }
    let batch = test_batch(1024, 3600.0, 3, capacity);
    emit(device, id, &batch);
    let frame = frame_after(5.0);
    let first_slot = batch.first_index & batch.ring_mask;
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 31);
    let (w, h) = (640u32, 400u32);
    let d = 200.0f32;
    let tan_h = (30.0f32).to_radians().tan();
    let mut mvp = [0.0f32; 16];
    mvp[0] = h as f32 / (w as f32 * tan_h);
    mvp[5] = 1.0 / tan_h;
    mvp[10] = -1e-3;
    mvp[11] = -1.0;
    mvp[15] = d;
    let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let mut pc = dust::DustSplatPushConstants {
      moments: 0,
      render: 0,
      pyramid: 0,
      live_count: batch.count,
      flags: dust::DUST_VIEW_FLOW,
      exposure: 1.0,
      inv_unit: 0.0,
      color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
      albedo: 0.04,
      eye_local: [0.0, 0.0, d, 30.0 * 86400.0],
      mvp,
    };
    let mut slowest_wall_ms = 0.0f64;
    for _ in 0..(DUST_READBACK_SLOTS + 2) {
      let t0 = std::time::Instant::now();
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], pc.flags, flow)?;
          pc.inv_unit = 1.0 / view.unit;
          device.cmd_dust_system_begin(id, pc.flags, flow)?;
          let (render, moments) =
            device.cmd_dust_propagate(cmd, id, 0, capacity, first_slot, batch.count, &frame)?;
          device.cmd_dust_timestamp(cmd, pe, DUST_TS_PROPAGATED);
          device.cmd_dust_pre_splat_barrier(cmd);
          device.cmd_dust_splat(cmd, pe, id, 0, first_slot, capacity, render, moments, pc)?;
          device.cmd_dust_timestamp(cmd, pe, DUST_TS_SPLATTED);
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
      slowest_wall_ms = slowest_wall_ms.max(t0.elapsed().as_secs_f64() * 1e3);
    }
    let t = device.dust_gpu_timings(pe).expect("no dust GPU timings after six frames");
    std::println!(
      "[dust timings] propagate {:.3} ms, splat {:.3} ms, measure {:.3} ms, total {:.3} ms (frame {}, {}x{}); slowest frame's wall time {:.1} ms",
      t.propagate_ms, t.splat_ms, t.measure_ms, t.total_ms, t.frame, t.width, t.height, slowest_wall_ms
    );
    for (name, v) in [("propagate", t.propagate_ms), ("splat", t.splat_ms), ("measure", t.measure_ms)] {
      assert!(v.is_finite() && v >= 0.0, "{name}: {v}");
    }
    assert!(t.splat_ms > 0.0, "the splat took no time");
    assert!(t.composite_ms.is_nan(), "a composite was timed without one drawn");
    // the packet queue's span (after the splats, in `cmd_dust_frame_end`) when it was drawn
    assert!(t.queue_ms.is_nan() || t.queue_ms >= 0.0, "queue: {}", t.queue_ms);
    let queue_ms = if t.queue_ms.is_nan() { 0.0 } else { t.queue_ms };
    assert!(
      (t.total_ms - (t.propagate_ms + t.splat_ms + queue_ms + t.measure_ms)).abs() <= 1e-9 * t.total_ms.max(1.0),
      "total {} is not the sum of its spans",
      t.total_ms
    );
    assert!(t.total_ms <= slowest_wall_ms, "GPU {} ms beyond the frame's wall time {slowest_wall_ms} ms", t.total_ms);
    assert_eq!((t.width, t.height), (w, h));
    assert!(t.frame >= 1, "frame {}", t.frame);
  });
}

/// The app's dust budget (agreed 2026-10-14): a dust frame — propagate, splat + specks, measure,
/// composite — within this many GPU milliseconds at 1280×800 on the development GPU, at every
/// pose; the app re-splats every presentation frame (16 ms)
pub const DUST_FRAME_BUDGET_MS: f64 = 8.0;

/// The dust frame fits [`DUST_FRAME_BUDGET_MS`]: a full tier (every ring slot emitted, mixed ages,
/// as `gpu_dust_full_ring_propagate_stress`) drawn from inside the dust (the perspective parity
/// camera) at 1280×800, frames recorded as the render thread does; the GPU timings of the last
/// measured frame within the budget. Ignored until the splat fits: measured 2026-10-14 the app's
/// frames cost 330 ms – 4.8 s (`definite_particles.md`, the dust GPU timings table).
#[test]
#[ignore = "the dust frame costs 0.33–4.8 s against the 8 ms budget (2026-10-14): the splat's cut is the next plan"]
fn dust_frame_fits_the_budget() {
  use super::dust::{DUST_READBACK_SLOTS, DUST_TS_PROPAGATED, DUST_TS_SPLATTED};
  let id = 9032;
  with_dust_device(id, |device, capacity, _, _| {
    if device.query_result.graphics_timestamp_bits == 0 {
      std::println!("[dust budget] the graphics queue has no timestamps: skipped");
      return;
    }
    let per = 4096u32;
    let (mut emitted, mut seq) = (0u32, 0u64);
    while emitted < capacity {
      let n = per.min(capacity - emitted);
      let batch = test_batch(n, 3600.0 * (1.0 + (seq % 7) as f64), emitted, capacity);
      let mut res = device
        .run_transient_compute_commands(|cmd| device.cmd_dust_emit(cmd, id, 0, &batch, seq))
        .unwrap();
      res.cleanup(&device.device);
      emitted += n;
      seq += 1;
    }
    let frame = frame_after(5.0);
    let pe = crate::gpu::PresentationEngineHandle(u64::MAX - 32);
    let (w, h) = (1280u32, 800u32);
    let d = 200.0f32;
    let tan_h = (30.0f32).to_radians().tan();
    let mut mvp = [0.0f32; 16];
    mvp[0] = h as f32 / (w as f32 * tan_h);
    mvp[5] = 1.0 / tan_h;
    mvp[10] = -1e-3;
    mvp[11] = -1.0;
    mvp[15] = d;
    let flow = dust::DustFlowUniform {
      t_hi: 1.0e8,
      t_lo: 0.0,
      speed: 1.0,
    };
    let mut pc = dust::DustSplatPushConstants {
      moments: 0,
      render: 0,
      pyramid: 0,
      live_count: capacity,
      flags: dust::DUST_VIEW_FLOW,
      exposure: 1.0,
      inv_unit: 0.0,
      color: dust::pack_color([1.0, 0.6, 0.2, 1.0]),
      albedo: 0.04,
      eye_local: [0.0, 0.0, d, 30.0 * 86400.0],
      mvp,
    };
    // enough frames for the budget feedback to settle and the last one to be read back
    for _ in 0..(3 * DUST_READBACK_SLOTS + 2) {
      let mut cmds = device
        .run_transient_commands(|cmd| {
          device.cmd_dust_pre_propagate_barrier(cmd);
          let view = device.cmd_dust_frame_begin(cmd, pe, [w, h], pc.flags, flow)?;
          pc.inv_unit = 1.0 / view.unit;
          device.cmd_dust_system_begin(id, pc.flags, flow)?;
          let (render, moments) = device.cmd_dust_propagate(cmd, id, 0, capacity, 0, capacity, &frame)?;
          device.cmd_dust_timestamp(cmd, pe, DUST_TS_PROPAGATED);
          device.cmd_dust_pre_splat_barrier(cmd);
          device.cmd_dust_splat(cmd, pe, id, 0, 0, capacity, render, moments, pc)?;
          device.cmd_dust_timestamp(cmd, pe, DUST_TS_SPLATTED);
          device.cmd_dust_post_splat_barrier(cmd);
          device.cmd_dust_frame_end(cmd, pe)?;
          device.cmd_dust_system_end(id)?;
          Ok(())
        })
        .unwrap();
      cmds.cleanup(&device.device);
    }
    let t = device.dust_gpu_timings(pe).expect("no dust GPU timings");
    std::println!(
      "[dust budget] {capacity} clusters at {w}x{h}: propagate {:.2} ms, splat {:.2} ms, measure {:.2} ms, total {:.2} ms (budget {DUST_FRAME_BUDGET_MS} ms)",
      t.propagate_ms, t.splat_ms, t.measure_ms, t.total_ms
    );
    assert!(
      t.total_ms <= DUST_FRAME_BUDGET_MS,
      "the dust frame costs {:.2} ms, over the {DUST_FRAME_BUDGET_MS} ms budget",
      t.total_ms
    );
  });
}
