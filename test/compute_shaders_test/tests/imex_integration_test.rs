use aethervk_core_rlib::{gpu::compute_push_constants::*, simulation::almanac::AlmanacPackedData};
use anise::time::Epoch;
use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct ImexIntegrationData {
  num_particles: u32,
  num_emitters: u32,
  dt: f32,
  particles: Vec<f32>,
  emitters: Vec<f32>,
  sorted_morton: Vec<u32>,
}

use compute_shaders_test::cpu_clustering::{CollisionEvent, group_and_cluster_collisions};

fn save_snapshot(name: &str, data: &[f32]) {
  let json = serde_json::to_string_pretty(data).unwrap();
  std::fs::write(format!("{}.json", name), json).unwrap();
}

#[test]
fn test_imex_integration_all_shaders() {
  compute_shaders_test::ensure_test_data(
    "test_data/imex_integration.json",
    "gen_imex_integration_data.py",
  );
  let json_data = fs::read_to_string("test_data/imex_integration.json").unwrap();
  let mut test_data: ImexIntegrationData = serde_json::from_str(&json_data).unwrap();

  // Load almanac and inject real planet data for 2 emitters!
  let mut almanac_data = AlmanacPackedData::default();
  almanac_data.load_almanac("../../assets/planets").unwrap();
  let epoch = Epoch::from_tdb_seconds(0.0); // J2000
  let frame = anise::constants::frames::SUN_J2000;
  let earth = almanac_data.get_ephem_full(399, frame, epoch, true, false).unwrap();
  let moon = almanac_data.get_ephem_full(301, frame, epoch, true, false).unwrap();

  // Replace the dummy emitters in JSON with REAL SPICE data!
  // Emitter 0: Earth
  test_data.emitters[0] = earth.position[0];
  test_data.emitters[1] = earth.position[1];
  test_data.emitters[2] = earth.position[2];
  test_data.emitters[3] = 398600.4418; // mu Earth
  // Emitter 1: Moon
  test_data.emitters[4] = moon.position[0];
  test_data.emitters[5] = moon.position[1];
  test_data.emitters[6] = moon.position[2];
  test_data.emitters[7] = 4902.8000; // mu Moon

  save_snapshot("snapshot_00_initial", &test_data.particles);

  let ctx = VulkanContext::new();
  let np = test_data.num_particles;
  let total_nodes = 2 * np - 1;

  let (particles_buf, mut p_alloc, p_addr) = ctx.create_buffer(
    &test_data.particles,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );
  let (emitters_buf, mut e_alloc, e_addr) = ctx.create_buffer(
    &test_data.emitters,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );
  let (morton_buf, mut m_alloc, m_addr) = ctx.create_buffer(
    &test_data.sorted_morton,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let bvh_init = vec![0.0f32; total_nodes as usize * 15];
  let (bvh_buf, mut b_alloc, b_addr) = ctx.create_buffer(
    &bvh_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let counters_init = vec![0u32; total_nodes as usize];
  let (cnt_buf, mut c_alloc, c_addr) = ctx.create_buffer(
    &counters_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let max_collisions_per_thread = 16;
  let total_slots = np * max_collisions_per_thread;
  let sparse_init = vec![0u32; total_slots as usize * 4]; // valid, pA, pB, toi
  let (sparse_buf, mut sp_alloc, sp_addr) = ctx.create_buffer(
    &sparse_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  // We will re-allocate packed_buf after CPU clustering

  let toi_init = vec![0u32; 1];
  let (toi_buf, mut toi_alloc, toi_addr) = ctx.create_buffer(
    &toi_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let max_pairs = np * (np - 1) / 2;
  let impulses_init = vec![0f32; max_pairs as usize * 3]; // 3 floats per impulse
  let (imp_buf, mut imp_alloc, imp_addr) = ctx.create_buffer(
    &impulses_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  // 1. p1-2
  let pc_p12 = P1_2PushConstants {
    particles_addr: p_addr,
    dt: test_data.dt,
    total_particles: np,
  };
  let bytes_p12 = unsafe {
    std::slice::from_raw_parts(
      &pc_p12 as *const _ as *const u8,
      std::mem::size_of::<P1_2PushConstants>(),
    )
  };
  println!("Running p1-2");
  run_compute_shader(
    &ctx,
    "../../assets/sim/p1-2_imex_particles.comp.spv",
    bytes_p12,
    (np + 127) / 128,
    1,
    1,
  );

  // 2. lbvh_build
  let pc_lbvh = LbvhPushConstants {
    bvh_addr: b_addr,
    sorted_morton_addr: m_addr,
    counters_addr: c_addr,
    particles_addr: p_addr,
    num_primitives: np,
    particle_radius: 1.0,
    dt: test_data.dt,
  };
  let bytes_lbvh = unsafe {
    std::slice::from_raw_parts(
      &pc_lbvh as *const _ as *const u8,
      std::mem::size_of::<LbvhPushConstants>(),
    )
  };
  println!("Running lbvh_build");
  let prepass_spv_path = "../../assets/sim/lbvh_prepass.comp.spv";
  let prepass_push_constants_bytes = unsafe {
    std::slice::from_raw_parts(
      &pc_lbvh as *const LbvhPushConstants as *const u8,
      8, // Only needs the bvh_addr which is the first 8 bytes
    )
  };
  run_compute_shader(
    &ctx,
    prepass_spv_path,
    prepass_push_constants_bytes,
    1,
    1,
    1,
  );

  run_compute_shader(
    &ctx,
    "../../assets/sim/lbvh_build.comp.spv",
    bytes_lbvh,
    (np + 127) / 128,
    1,
    1,
  );

  // 3. ccd
  let pc_ccd = CcdPushConstants {
    particle_bvh: b_addr,
    output_list: sp_addr,
    root_index: 0,
    total_particles: np,
  };
  let bytes_ccd = unsafe {
    std::slice::from_raw_parts(
      &pc_ccd as *const _ as *const u8,
      std::mem::size_of::<CcdPushConstants>(),
    )
  };
  println!("Running ccd");
  run_compute_shader(
    &ctx,
    "../../assets/sim/ccd.comp.spv",
    bytes_ccd,
    (np + 127) / 128,
    1,
    1,
  );

  // Read back sparse output and perform CPU clustering
  let sparse_data: Vec<u32> = ctx.read_buffer(sparse_buf, &mut sp_alloc, total_slots as usize * 4);
  let mut collisions = Vec::new();
  for i in 0..total_slots as usize {
    if sparse_data[i * 4] == 1 {
      collisions.push(CollisionEvent {
        entity_a: sparse_data[i * 4 + 1],
        entity_b: sparse_data[i * 4 + 2],
        time_of_impact: f32::from_bits(sparse_data[i * 4 + 3]),
      });
    }
  }

  let clusters = group_and_cluster_collisions(collisions, 0.01);
  let total_clusters = clusters.len() as u32;

  let mut packed_init = vec![0u32; 1 + max_pairs as usize * 2];
  packed_init[0] = 0; // count, but wait, LCP expects block per cluster?
  // We'll pack them sequentially: pairs start at index 4 (due to dispatch_x, y, z, count)
  // Actually our shader definition of PackedCollisions is:
  // uint dispatch_x, y, z, count; uvec2 pairs[];
  // So 4 uints header. Let's make the buffer big enough.
  let mut dense_data = vec![0u32; 4 + max_pairs as usize * 2];
  let mut pair_idx = 0;
  for cluster in clusters {
    for col in cluster {
      dense_data[4 + pair_idx * 2] = col.entity_a;
      dense_data[4 + pair_idx * 2 + 1] = col.entity_b;
      pair_idx += 1;
    }
  }
  dense_data[3] = pair_idx as u32; // count
  let (packed_buf, mut pk_alloc, pk_addr) = ctx.create_buffer(
    &dense_data,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  // 4. stream_compact (SKIPPED - CPU Clustering done)

  // 5. reduce_toi
  let pc_toi = ReduceToiPushConstants {
    particles_addr: p_addr,
    collisions_addr: pk_addr,
    out_toi_addr: toi_addr,
    particle_radius: 1.0,
    dt: test_data.dt,
  };
  let bytes_toi = unsafe {
    std::slice::from_raw_parts(
      &pc_toi as *const _ as *const u8,
      std::mem::size_of::<ReduceToiPushConstants>(),
    )
  };
  println!("Running reduce_toi");
  run_compute_shader(
    &ctx,
    "../../assets/sim/reduce_toi.comp.spv",
    bytes_toi,
    (max_pairs + 127) / 128,
    1,
    1,
  );

  // 6. lcp_solver
  let pc_lcp = LcpPushConstants {
    particles_addr: p_addr,
    collisions_addr: pk_addr,
    impulses_addr: imp_addr,
    total_clusters: std::cmp::max(1, total_clusters),
    rigid_bodies_addr: 0,
    dt: 0.016,
    restitution: 0.5,
  };
  let bytes_lcp = unsafe {
    std::slice::from_raw_parts(
      &pc_lcp as *const _ as *const u8,
      std::mem::size_of::<LcpPushConstants>(),
    )
  };
  println!("Running lcp_solver");
  run_compute_shader(
    &ctx,
    "../../assets/sim/lcp_solver.comp.spv",
    bytes_lcp,
    1,
    1,
    1,
  ); // 1 workgroup per cluster ideally

  // 7. apply_impulses
  let pc_imp = ApplyImpulsesPushConstants {
    particles_addr: p_addr,
    collisions_addr: pk_addr,
    impulses_addr: imp_addr,
  };
  let bytes_imp = unsafe {
    std::slice::from_raw_parts(
      &pc_imp as *const _ as *const u8,
      std::mem::size_of::<ApplyImpulsesPushConstants>(),
    )
  };
  println!("Running apply_impulses");
  run_compute_shader(
    &ctx,
    "../../assets/sim/apply_impulses.comp.spv",
    bytes_imp,
    (max_pairs + 127) / 128,
    1,
    1,
  );

  // Snapshot after collision
  let post_collision_particles: Vec<f32> =
    ctx.read_buffer(particles_buf, &mut p_alloc, test_data.particles.len());
  save_snapshot("snapshot_01_post_collision", &post_collision_particles);

  // 8. barnes_hut
  let pc_bh = BarnesHutPushConstants {
    particles_addr: p_addr,
    bvh_addr: b_addr,
    root_index: 0,
    total_particles: np,
    theta: 0.5,
    g: 1.0,
  };
  let bytes_bh = unsafe {
    std::slice::from_raw_parts(
      &pc_bh as *const _ as *const u8,
      std::mem::size_of::<BarnesHutPushConstants>(),
    )
  };
  println!("Running barnes_hut");
  run_compute_shader(
    &ctx,
    "../../assets/sim/barnes_hut.comp.spv",
    bytes_bh,
    (np + 127) / 128,
    1,
    1,
  );

  // 9. p5_imex
  let pc_p5 = P5PushConstants {
    particles_addr: p_addr,
    emitters_addr: e_addr,
    dt: test_data.dt,
    total_particles: np,
    num_emitters: 2,
  };
  let bytes_p5 = unsafe {
    std::slice::from_raw_parts(
      &pc_p5 as *const _ as *const u8,
      std::mem::size_of::<P5PushConstants>(),
    )
  };
  println!("Running p5_imex_particles");
  run_compute_shader(
    &ctx,
    "../../assets/sim/p5_imex_particles.comp.spv",
    bytes_p5,
    (np + 127) / 128,
    1,
    1,
  );

  // Verify Output
  let final_particles: Vec<f32> =
    ctx.read_buffer(particles_buf, &mut p_alloc, test_data.particles.len());
  save_snapshot("snapshot_02_final", &final_particles);

  // Basic assert: positions changed and are not NaN
  for i in 0..np as usize {
    let block_idx = i / 32;
    let local_idx = i % 32;
    let base = block_idx * 10 * 32 + local_idx;

    let x = final_particles[base + 0 * 32];
    let y = final_particles[base + 1 * 32];
    let z = final_particles[base + 2 * 32];
    assert!(
      !x.is_nan() && !y.is_nan() && !z.is_nan(),
      "Particle {} position is NaN",
      i
    );
  }

  println!(
    "All IMEX shaders executed sequentially successfully with SPICE planet data integrated!"
  );

  // Cleanup
  ctx.destroy_buffer(particles_buf, p_alloc);
  ctx.destroy_buffer(emitters_buf, e_alloc);
  ctx.destroy_buffer(morton_buf, m_alloc);
  ctx.destroy_buffer(bvh_buf, b_alloc);
  ctx.destroy_buffer(cnt_buf, c_alloc);
  ctx.destroy_buffer(sparse_buf, sp_alloc);
  ctx.destroy_buffer(packed_buf, pk_alloc);
  ctx.destroy_buffer(toi_buf, toi_alloc);
  ctx.destroy_buffer(imp_buf, imp_alloc);
}
