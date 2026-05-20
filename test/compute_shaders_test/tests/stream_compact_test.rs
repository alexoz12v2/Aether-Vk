use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use serde::Deserialize;
use std::fs;

#[derive(Deserialize)]
struct StreamCompactData {
  total_elements: u32,
  sparse_in: Vec<u32>,
  expected_count: u32,
  expected_pairs: Vec<u32>,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
  sparse_in_addr: u64,
  packed_out_addr: u64,
  total_elements: u32,
}

#[test]
fn test_stream_compact() {
  compute_shaders_test::ensure_test_data(
    "test_data/stream_compact.json",
    "gen_stream_compact_data.py",
  );
  let json_data = fs::read_to_string("test_data/stream_compact.json").expect("Failed to read JSON");
  let test_data: StreamCompactData =
    serde_json::from_str(&json_data).expect("Failed to parse JSON");

  let ctx = VulkanContext::new();

  let (sparse_buffer, mut sparse_alloc, sparse_addr) = ctx.create_buffer(
    &test_data.sparse_in,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  // output array size: dispatch (3) + count (1) + pairs (12 uints * total_elements)
  // we initialize it to 0
  let mut packed_out_init = vec![0u32; 4 + test_data.total_elements as usize * 12];

  let (packed_buffer, mut packed_alloc, packed_addr) = ctx.create_buffer(
    &packed_out_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let push_constants = PushConstants {
    sparse_in_addr: sparse_addr,
    packed_out_addr: packed_addr,
    total_elements: test_data.total_elements,
  };

  let push_constants_bytes = unsafe {
    std::slice::from_raw_parts(
      &push_constants as *const PushConstants as *const u8,
      std::mem::size_of::<PushConstants>(),
    )
  };

  let spv_path = "../../assets/sim/stream_compact.comp.spv";
  let dispatch_x = (test_data.total_elements + 127) / 128;

  run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

  let output_data: Vec<u32> = ctx.read_buffer(
    packed_buffer,
    &mut packed_alloc,
    4 + test_data.total_elements as usize * 12,
  );

  ctx.destroy_buffer(sparse_buffer, sparse_alloc);
  ctx.destroy_buffer(packed_buffer, packed_alloc);

  let count = output_data[3];
  println!("First 10 elements: {:?}", &output_data[0..10]);
  assert_eq!(
    count, test_data.expected_count,
    "Expected {} items, but got {}",
    test_data.expected_count, count
  );

  let mut actual_pairs = Vec::new();
  for i in 0..count as usize {
    actual_pairs.push((output_data[4 + i * 12 + 1], output_data[4 + i * 12 + 3]));
  }
  actual_pairs.sort_by_key(|p| p.0);

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
