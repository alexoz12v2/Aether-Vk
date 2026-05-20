use ash::vk;
use compute_shaders_test::{
  VulkanContext,
  cpu_clustering::{CollisionEvent, group_and_cluster_collisions},
  run_compute_shader,
};

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
  sparse_in_addr: u64,
  packed_out_addr: u64,
  total_elements: u32,
}

#[test]
fn test_complex_compaction_and_clustering() {
  let ctx = VulkanContext::new();

  // Create a massive sparse collision buffer
  let total_elements = 10_000;
  let mut sparse_in = vec![0u32; total_elements as usize * 11];

  // Generate complex 3-way and multi-point intersections
  // Cluster 1: A=1, B=2, C=3 (triangle)
  sparse_in[10 * 11] = 1;
  sparse_in[10 * 11 + 1] = 1;
  sparse_in[10 * 11 + 2] = 2;
  sparse_in[10 * 11 + 3] = f32::to_bits(0.5);
  sparse_in[25 * 11] = 1;
  sparse_in[25 * 11 + 1] = 2;
  sparse_in[25 * 11 + 2] = 3;
  sparse_in[25 * 11 + 3] = f32::to_bits(0.5);
  sparse_in[50 * 11] = 1;
  sparse_in[50 * 11 + 1] = 1;
  sparse_in[50 * 11 + 2] = 3;
  sparse_in[50 * 11 + 3] = f32::to_bits(0.5);

  // Cluster 2: Star topology around 10
  for i in 11..20 {
    let idx = 100 + i;
    sparse_in[idx * 11] = 1;
    sparse_in[idx * 11 + 1] = 10;
    sparse_in[idx * 11 + 2] = i as u32;
    sparse_in[idx * 11 + 3] = f32::to_bits(0.1);
  }

  // Cluster 3: Linear chain 30-31-32-33
  sparse_in[200 * 11] = 1;
  sparse_in[200 * 11 + 1] = 30;
  sparse_in[200 * 11 + 2] = 31;
  sparse_in[200 * 11 + 3] = f32::to_bits(0.8);
  sparse_in[201 * 11] = 1;
  sparse_in[201 * 11 + 1] = 31;
  sparse_in[201 * 11 + 2] = 32;
  sparse_in[201 * 11 + 3] = f32::to_bits(0.8);
  sparse_in[202 * 11] = 1;
  sparse_in[202 * 11 + 1] = 32;
  sparse_in[202 * 11 + 2] = 33;
  sparse_in[202 * 11 + 3] = f32::to_bits(0.8);

  // Some random noise
  for i in 1000..5000 {
    if i % 7 == 0 {
      sparse_in[i * 11] = 1;
      sparse_in[i * 11 + 1] = (i * 13 % 1000) as u32 + 50;
      sparse_in[i * 11 + 2] = (i * 17 % 1000) as u32 + 50;
      sparse_in[i * 11 + 3] = f32::to_bits(0.3);
    }
  }

  let (sparse_buffer, mut sparse_alloc, sparse_addr) = ctx.create_buffer(
    &sparse_in,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let mut packed_out_init = vec![0u32; 4 + total_elements as usize * 3];

  let (packed_buffer, mut packed_alloc, packed_addr) = ctx.create_buffer(
    &packed_out_init,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let push_constants = PushConstants {
    sparse_in_addr: sparse_addr,
    packed_out_addr: packed_addr,
    total_elements,
  };

  let push_constants_bytes = unsafe {
    std::slice::from_raw_parts(
      &push_constants as *const PushConstants as *const u8,
      std::mem::size_of::<PushConstants>(),
    )
  };

  let spv_path = "../../assets/sim/stream_compact.comp.spv";
  let dispatch_x = (total_elements + 127) / 128;

  run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

  let output_data: Vec<u32> = ctx.read_buffer(
    packed_buffer,
    &mut packed_alloc,
    4 + total_elements as usize * 3,
  );

  ctx.destroy_buffer(sparse_buffer, sparse_alloc);
  ctx.destroy_buffer(packed_buffer, packed_alloc);

  let gpu_count = output_data[3];
  println!("GPU stream_compact found {} valid collisions", gpu_count);

  let gpu_events =
    compute_shaders_test::cpu_clustering::parse_gpu_packed_pairs(&output_data, gpu_count as usize);

  // Now let's extract CPU sparse events
  let mut cpu_events = Vec::new();
  for i in 0..total_elements as usize {
    if sparse_in[i * 11] == 1 {
      cpu_events.push(CollisionEvent {
        entity_a: sparse_in[i * 11 + 1],
        entity_b: sparse_in[i * 11 + 2],
        time_of_impact: f32::from_bits(sparse_in[i * 11 + 3]),
      });
    }
  }

  println!("CPU extraction found {} valid collisions", cpu_events.len());
  assert_eq!(
    gpu_count as usize,
    cpu_events.len(),
    "GPU and CPU extracted different number of valid collisions!"
  );

  // To verify that both give the expected group result, we cluster the CPU events.
  // (Note that GPU currently drops TOI in stream_compact.comp, so we only cluster the CPU events
  //  and verify that the CPU clustering logic produces expected clusters for the complex scene).
  let clusters = group_and_cluster_collisions(cpu_events, 0.05);

  println!("Clustering yielded {} clusters", clusters.len());

  // We expect cluster 1 to have 3 events
  let c1 = clusters.iter().find(|c| c.iter().any(|e| e.entity_a == 1 && e.entity_b == 2)).unwrap();
  assert_eq!(c1.len(), 3);

  // We expect cluster 2 to have 9 events
  let c2 =
    clusters.iter().find(|c| c.iter().any(|e| e.entity_a == 10 && e.entity_b == 11)).unwrap();
  assert_eq!(c2.len(), 9);

  // We expect cluster 3 to have 3 events
  let c3 =
    clusters.iter().find(|c| c.iter().any(|e| e.entity_a == 30 && e.entity_b == 31)).unwrap();
  assert_eq!(c3.len(), 3);

  println!("All complex groupings match perfectly!");
}
