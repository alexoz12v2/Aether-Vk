use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct AABB {
  min_bounds: [f32; 3],
  max_bounds: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct BVHNodeAABB {
  bound: AABB,
  left_child_or_primitive_offset: u32,
  right_child_offset: u32,
  primitive_count: u32,
  node_type: u32,
  parent_idx: u32,
  mass: f32,
  center_of_mass: [f32; 3],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct MortonEntry {
  morton_code: u32,
  primitive_index: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct PushConstants {
  bvh_addr: u64,
  sorted_morton_addr: u64,
  counters_addr: u64,
  particles_addr: u64,
  num_primitives: u32,
  particle_radius: f32,
  dt: f32,
}

#[test]
fn test_lbvh_build() {
  let ctx = VulkanContext::new();

  let num_primitives: u32 = 16;
  let subgroup_size: u32 = 32;

  let mut morton_entries = Vec::new();
  for i in 0..num_primitives {
    morton_entries.push(MortonEntry {
      morton_code: i * 10,
      primitive_index: i,
    });
  }

  let mut particles = vec![
    0.0f32;
    (10 * subgroup_size * ((num_primitives + subgroup_size - 1) / subgroup_size))
      as usize
  ];
  for i in 0..num_primitives {
    let block_idx = i / subgroup_size;
    let local_idx = i % subgroup_size;
    let base = block_idx * 10 * subgroup_size + local_idx;

    particles[(base + 0 * subgroup_size) as usize] = i as f32;
    particles[(base + 1 * subgroup_size) as usize] = i as f32 * 2.0;
    particles[(base + 2 * subgroup_size) as usize] = i as f32 * 3.0;

    particles[(base + 6 * subgroup_size) as usize] = 10.0; // mass
  }

  let total_nodes = 2 * num_primitives - 1;
  let bvh_nodes = vec![
    BVHNodeAABB {
      bound: AABB {
        min_bounds: [0.0; 3],
        max_bounds: [0.0; 3]
      },
      left_child_or_primitive_offset: 0,
      right_child_offset: 0,
      primitive_count: 0,
      node_type: 0,
      parent_idx: 0,
      mass: 0.0,
      center_of_mass: [0.0; 3],
    };
    total_nodes as usize
  ];

  let atomic_counters = vec![0u32; total_nodes as usize];

  let (morton_buffer, mut morton_alloc, morton_addr) = ctx.create_buffer(
    &morton_entries,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let (particles_buffer, mut particles_alloc, particles_addr) = ctx.create_buffer(
    &particles,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let (bvh_buffer, mut bvh_alloc, bvh_addr) = ctx.create_buffer(
    &bvh_nodes,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let (counters_buffer, mut counters_alloc, counters_addr) = ctx.create_buffer(
    &atomic_counters,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let push_constants = PushConstants {
    bvh_addr,
    sorted_morton_addr: morton_addr,
    counters_addr,
    particles_addr,
    num_primitives,
    particle_radius: 1.0,
    dt: 0.016,
  };

  let push_constants_bytes = unsafe {
    std::slice::from_raw_parts(
      &push_constants as *const PushConstants as *const u8,
      std::mem::size_of::<PushConstants>(),
    )
  };

  let spv_path = "../../assets/sim/lbvh_build.comp.spv";
  let dispatch_x = (num_primitives + 127) / 128;

  run_compute_shader(&ctx, spv_path, push_constants_bytes, dispatch_x, 1, 1);

  let output_bvh: Vec<BVHNodeAABB> =
    ctx.read_buffer(bvh_buffer, &mut bvh_alloc, total_nodes as usize);

  ctx.destroy_buffer(morton_buffer, morton_alloc);
  ctx.destroy_buffer(particles_buffer, particles_alloc);
  ctx.destroy_buffer(bvh_buffer, bvh_alloc);
  ctx.destroy_buffer(counters_buffer, counters_alloc);

  // Basic sanity checks
  // Internal nodes should have primitive_count = 0
  for i in 0..(num_primitives - 1) as usize {
    assert_eq!(
      output_bvh[i].primitive_count, 0,
      "Internal node {} should have primitive_count = 0",
      i
    );
    assert!(
      output_bvh[i].mass > 0.0,
      "Internal node {} mass should be > 0.0",
      i
    );
  }

  // Leaf nodes should have primitive_count = 1
  for i in (num_primitives - 1) as usize..total_nodes as usize {
    assert_eq!(
      output_bvh[i].primitive_count, 1,
      "Leaf node {} should have primitive_count = 1",
      i
    );
    assert_eq!(
      output_bvh[i].mass, 10.0,
      "Leaf node {} mass should be 10.0",
      i
    );
  }
}
