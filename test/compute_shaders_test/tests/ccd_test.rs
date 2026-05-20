use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct CcdData {
  total_particles: u32,
  root_index: u32,
  bvh_nodes: Vec<f32>,
  particles: Vec<f32>,
  particle_radius: f32,
  dt: f32,
  expected_count: u32,
  expected_pairs: Vec<u32>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
  particle_bvh: u64,
  output_list: u64,
  particles: u64,
  root_index: u32,
  total_particles: u32,
  particle_radius: f32,
  dt: f32,
}

#[test]
fn test_ccd() {
  compute_shaders_test::ensure_test_data("test_data/ccd.json", "gen_ccd_data.py");
  let json_data = fs::read_to_string("test_data/ccd.json").expect("Failed to read JSON");
  let test_data: CcdData = serde_json::from_str(&json_data).expect("Failed to parse JSON");

  let ctx = VulkanContext::new();
  let (bvh_buffer, mut bvh_alloc, bvh_addr) = ctx.create_buffer(
    &test_data.bvh_nodes,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let (particles_buffer, mut particles_alloc, particles_addr) = ctx.create_buffer(
    &test_data.particles,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  // We now output to a SparseCollisions array, max 16 collisions per particle.
  let max_collisions_per_thread = 16;
  let total_slots = test_data.total_particles * max_collisions_per_thread;
  let mut packed_out_init = vec![0u32; total_slots as usize * 11];

  let (output_buffer, mut output_alloc, output_addr) = ctx.create_buffer(
    &packed_out_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let push_constants = PushConstants {
    particle_bvh: bvh_addr,
    output_list: output_addr,
    particles: particles_addr,
    root_index: test_data.root_index,
    total_particles: test_data.total_particles,
    particle_radius: test_data.particle_radius,
    dt: test_data.dt,
  };

  let push_constants_bytes = unsafe {
    std::slice::from_raw_parts(
      &push_constants as *const PushConstants as *const u8,
      std::mem::size_of::<PushConstants>(),
    )
  };

  let spv_path = "../../assets/sim/ccd.comp.spv";
  let dispatch_x = (test_data.total_particles + 127) / 128;

  run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

  let output_data: Vec<u32> =
    ctx.read_buffer(output_buffer, &mut output_alloc, total_slots as usize * 11);

  ctx.destroy_buffer(bvh_buffer, bvh_alloc);
  ctx.destroy_buffer(particles_buffer, particles_alloc);
  ctx.destroy_buffer(output_buffer, output_alloc);

  let mut actual_pairs = Vec::new();
  for i in 0..total_slots as usize {
    let valid = output_data[i * 11];
    if valid == 1 {
      actual_pairs.push((output_data[i * 11 + 1], output_data[i * 11 + 2]));
    }
  }
  actual_pairs.sort_by_key(|p| p.0);

  let count = actual_pairs.len() as u32;

  assert_eq!(
    count, test_data.expected_count,
    "Expected {} pairs, but got {}",
    test_data.expected_count, count
  );

  for i in 0..count as usize {
    let (actual_a, actual_b) = actual_pairs[i];
    let expected_a = test_data.expected_pairs[i * 2];
    let expected_b = test_data.expected_pairs[i * 2 + 1];

    assert_eq!(
      actual_a, expected_a,
      "Mismatch at pair {}: actual_a {}, expected_a {}",
      i, actual_a, expected_a
    );
    assert_eq!(
      actual_b, expected_b,
      "Mismatch at pair {}: actual_b {}, expected_b {}",
      i, actual_b, expected_b
    );
  }
}
