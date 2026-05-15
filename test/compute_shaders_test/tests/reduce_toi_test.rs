use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct ReduceToiData {
  dt: f32,
  particle_radius: f32,
  total_pairs: u32,
  particles: Vec<f32>,
  pairs: Vec<u32>,
  expected_min_tc_uint: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
  particles_addr: u64,
  collisions_addr: u64,
  out_toi_addr: u64,
  particle_radius: f32,
  dt: f32,
}

#[test]
fn test_reduce_toi() {
  compute_shaders_test::ensure_test_data("test_data/reduce_toi.json", "gen_reduce_toi_data.py");
  let json_data = fs::read_to_string("test_data/reduce_toi.json").expect("Failed to read JSON");
  let test_data: ReduceToiData = serde_json::from_str(&json_data).expect("Failed to parse JSON");

  let ctx = VulkanContext::new();

  let (particles_buffer, mut particles_alloc, particles_addr) = ctx.create_buffer(
    &test_data.particles,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let mut packed_collisions = vec![0, 0, 0, test_data.total_pairs];
  packed_collisions.extend_from_slice(&test_data.pairs);

  let (collisions_buffer, mut collisions_alloc, collisions_addr) = ctx.create_buffer(
    &packed_collisions,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let out_toi_init = vec![f32::to_bits(test_data.dt)];
  let (toi_buffer, mut toi_alloc, toi_addr) = ctx.create_buffer(
    &out_toi_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let push_constants = PushConstants {
    particles_addr,
    collisions_addr,
    out_toi_addr: toi_addr,
    particle_radius: test_data.particle_radius,
    dt: test_data.dt,
  };

  let push_constants_bytes = unsafe {
    std::slice::from_raw_parts(
      &push_constants as *const PushConstants as *const u8,
      std::mem::size_of::<PushConstants>(),
    )
  };

  let spv_path = "../../assets/sim/reduce_toi.comp.spv";
  let dispatch_x = (test_data.total_pairs + 127) / 128;

  run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

  let output_toi: Vec<u32> = ctx.read_buffer(toi_buffer, &mut toi_alloc, 1);

  ctx.destroy_buffer(particles_buffer, particles_alloc);
  ctx.destroy_buffer(collisions_buffer, collisions_alloc);
  ctx.destroy_buffer(toi_buffer, toi_alloc);

  let expected = test_data.expected_min_tc_uint;
  let actual = output_toi[0];

  let expected_f = f32::from_bits(expected);
  let actual_f = f32::from_bits(actual);

  let diff = (actual_f - expected_f).abs();
  assert!(
    diff < 1e-5,
    "Expected min TC {}, got {}",
    expected_f,
    actual_f
  );
}
