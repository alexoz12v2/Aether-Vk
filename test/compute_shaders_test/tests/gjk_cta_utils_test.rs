use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct SphereData {
  center: [f32; 3],
  radius: f32,
  velocity: [f32; 3],
  _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct PushConstants {
  input_a: u64,
  input_b: u64,
  out_tois: u64,
  num_pairs: u32,
}

#[test]
fn test_gjk_cta_utils() {
  let ctx = VulkanContext::new();

  // Setup Test Cases
  // 1. Direct hit (Dist=5, Vrel=10 -> toi=0.3)
  let a1 = SphereData {
    center: [0.0, 0.0, 0.0],
    radius: 1.0,
    velocity: [10.0, 0.0, 0.0],
    _pad: 0.0,
  };
  let b1 = SphereData {
    center: [5.0, 0.0, 0.0],
    radius: 1.0,
    velocity: [0.0, 0.0, 0.0],
    _pad: 0.0,
  };

  // 2. Parallel miss
  let a2 = SphereData {
    center: [0.0, 5.0, 0.0],
    radius: 1.0,
    velocity: [10.0, 0.0, 0.0],
    _pad: 0.0,
  };
  let b2 = SphereData {
    center: [5.0, 0.0, 0.0],
    radius: 1.0,
    velocity: [0.0, 0.0, 0.0],
    _pad: 0.0,
  };

  // 3. Already overlapping (hit at t=0)
  let a3 = SphereData {
    center: [0.0, 0.0, 0.0],
    radius: 1.0,
    velocity: [5.0, 0.0, 0.0],
    _pad: 0.0,
  };
  let b3 = SphereData {
    center: [0.5, 0.0, 0.0],
    radius: 1.0,
    velocity: [0.0, 0.0, 0.0],
    _pad: 0.0,
  };

  let input_a = vec![a1, a2, a3];
  let input_b = vec![b1, b2, b3];
  let num_pairs = input_a.len() as u32;

  let (buf_a, mut alloc_a, addr_a) =
    ctx.create_buffer(&input_a, vk::BufferUsageFlags::STORAGE_BUFFER);
  let (buf_b, mut alloc_b, addr_b) =
    ctx.create_buffer(&input_b, vk::BufferUsageFlags::STORAGE_BUFFER);

  let initial_tois = vec![-2.0f32; num_pairs as usize];
  let (buf_out, mut alloc_out, addr_out) = ctx.create_buffer(
    &initial_tois,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let pc = PushConstants {
    input_a: addr_a,
    input_b: addr_b,
    out_tois: addr_out,
    num_pairs,
  };
  let pc_bytes = unsafe {
    std::slice::from_raw_parts(
      &pc as *const _ as *const u8,
      std::mem::size_of::<PushConstants>(),
    )
  };

  run_compute_shader(
    &ctx,
    "../../assets/sim/gjk_cta_utils_test.comp.spv",
    pc_bytes,
    num_pairs,
    1,
    1,
  );

  let final_tois: Vec<f32> = ctx.read_buffer(buf_out, &mut alloc_out, num_pairs as usize);

  assert!(
    (final_tois[0] - 0.3).abs() < 0.05,
    "Expected hit around 0.3, got {}",
    final_tois[0]
  );
  assert_eq!(
    final_tois[1], -1.0,
    "Expected miss (-1.0), got {}",
    final_tois[1]
  );
  assert_eq!(
    final_tois[2], 0.0,
    "Expected overlap (0.0), got {}",
    final_tois[2]
  );

  ctx.destroy_buffer(buf_a, alloc_a);
  ctx.destroy_buffer(buf_b, alloc_b);
  ctx.destroy_buffer(buf_out, alloc_out);
}
