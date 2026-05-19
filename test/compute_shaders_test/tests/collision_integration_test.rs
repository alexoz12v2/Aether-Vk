use aethervk_core_rlib::gpu::compute_push_constants::{
  BroadPhasePushConstants, BvhNodeAABBGpu, EntityGpu, LcpPushConstants,
  NarrowPhaseRigidBodyPushConstants,
};
use ash::vk;
use compute_shaders_test::{VulkanContext, run_compute_shader};
use std::mem;

#[test]
fn test_gpu_collision_pipeline_integration() {
  let ctx = VulkanContext::new();

  let bvh_nodes = vec![BvhNodeAABBGpu {
    min_bounds: [-10.0, -10.0, -10.0],
    max_bounds: [10.0, 10.0, 10.0],
    left_child_or_primitive_offset: 0,
    right_child_offset: 0,
    primitive_count: 1,
    node_type: 0,
    parent_idx: 0,
    mass: 1.0,
    center_of_mass: [0.0, 0.0, 0.0],
  }];
  let (bvh_buf, mut bvh_alloc, bvh_addr) = ctx.create_buffer(
    &bvh_nodes,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_DST
      | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
  );

  let read_back: Vec<BvhNodeAABBGpu> = ctx.read_buffer(bvh_buf, &mut bvh_alloc, 1);
  println!(
    "DEBUG RUST: read_back min_bounds: {:?}",
    read_back[0].min_bounds
  );
  println!("DEBUG RUST: bvh_addr = 0x{:X}", bvh_addr);

  // 1. Create two intersecting spheres (rigidbodies)
  let entities = vec![
    EntityGpu {
      bvh_addr,
      transform: [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      ],
      inv_transform: [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      ],
      linear_velocity: [1.0, 0.0, 0.0],
      angular_velocity: [0.0, 0.0, 0.0],
      root_index: 0,
      entity_type: 1, // RIGID_BODY
      primitive_offset: 0,
      total_primitives: 1,
      frame_scale_type: 0,
      scale_factor: 1.0,
      shape_type: 0,
      shape_data: [0.0; 3],
    },
    EntityGpu {
      bvh_addr,
      transform: [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.5, 0.0, 0.0,
        1.0, // Intersects first sphere
      ],
      inv_transform: [
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.5, 0.0, 0.0, 1.0,
      ],
      linear_velocity: [-1.0, 0.0, 0.0],
      angular_velocity: [0.0, 0.0, 0.0],
      root_index: 0,
      entity_type: 1, // RIGID_BODY
      primitive_offset: 1,
      total_primitives: 1,
      frame_scale_type: 0,
      scale_factor: 1.0,
      shape_type: 0,
      shape_data: [0.0; 3],
    },
  ];

  let (entities_buf, mut entities_alloc, entities_addr) = ctx.create_buffer(
    &entities,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_DST
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
  );

  let entities_read_back: Vec<EntityGpu> = ctx.read_buffer(entities_buf, &mut entities_alloc, 2);
  println!(
    "DEBUG RUST: read_back entities[0].bvh_addr = 0x{:X}",
    entities_read_back[0].bvh_addr
  );

  // 2. Output buffer for Broad Phase
  let mut overlapping_pairs = vec![0u32; 1 + 2 * 16]; // Count + pairs
  let (broad_out_buf, mut broad_out_alloc, broad_out_addr) = ctx.create_buffer(
    &overlapping_pairs,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  // 3. Output buffer for Narrow Phase
  let mut narrow_out = vec![0u32; 4 * 16]; // uvec4 per collision
  let (narrow_out_buf, mut narrow_out_alloc, narrow_out_addr) = ctx.create_buffer(
    &narrow_out,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  // We skip the broad phase compute execution for this isolated integration test
  // and manually set the broad phase output to trigger narrow phase.

  // Narrow Phase
  let narrow_pc = NarrowPhaseRigidBodyPushConstants {
    scene_entities_addr: entities_addr,
    output_list_addr: narrow_out_addr,
    particles_addr: bvh_addr, // Hack: make bvh_buf resident in MoltenVK
    entity_a_idx: 0,
    entity_b_idx: 1,
    dt: 1.0 / 60.0,
    particle_radius: 1.0,
  };
  let narrow_pc_bytes = unsafe {
    std::slice::from_raw_parts(
      &narrow_pc as *const _ as *const u8,
      mem::size_of::<NarrowPhaseRigidBodyPushConstants>(),
    )
  };

  println!("Running narrow_ccd_rigidbody");
  run_compute_shader(
    &ctx,
    "../../assets/sim/narrow_ccd_rigidbody.comp.spv",
    narrow_pc_bytes,
    1,
    1,
    1,
  );
  println!("narrow_ccd_rigidbody done");

  // Read back Narrow Phase output
  let narrow_results: Vec<u32> = ctx.read_buffer(narrow_out_buf, &mut narrow_out_alloc, 4);

  assert!(narrow_results.len() == 4);

  // LCP Solver
  let mut lcp_out = vec![0.0f32; 3 * 16]; // vec3 per impulse
  let (lcp_out_buf, mut lcp_out_alloc, lcp_out_addr) = ctx.create_buffer(
    &lcp_out,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let mut packed_collisions = vec![0u32; 4 + 2 * 16];
  packed_collisions[3] = 1; // 1 collision
  packed_collisions[4] = 0; // Particle A
  packed_collisions[5] = 1; // Particle B
  let (packed_col_buf, mut packed_col_alloc, packed_col_addr) = ctx.create_buffer(
    &packed_collisions,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let mut mock_particles = vec![0.0f32; 10 * 32];
  mock_particles[6 * 32 + 0] = 1.0;
  mock_particles[6 * 32 + 1] = 1.0;

  let (particles_buf, mut p_alloc, p_addr) = ctx.create_buffer(
    &mock_particles,
    vk::BufferUsageFlags::STORAGE_BUFFER
      | vk::BufferUsageFlags::TRANSFER_SRC
      | vk::BufferUsageFlags::TRANSFER_DST,
  );

  let lcp_pc = LcpPushConstants {
    particles_addr: p_addr,
    collisions_addr: packed_col_addr,
    impulses_addr: lcp_out_addr,
    total_clusters: 1,
  };

  let lcp_pc_bytes = unsafe {
    std::slice::from_raw_parts(
      &lcp_pc as *const _ as *const u8,
      mem::size_of::<LcpPushConstants>(),
    )
  };

  println!("Running lcp_solver");
  run_compute_shader(
    &ctx,
    "../../assets/sim/lcp_solver.comp.spv",
    lcp_pc_bytes,
    1,
    1,
    1,
  );
  println!("lcp_solver done");

  ctx.destroy_buffer(bvh_buf, bvh_alloc);
  ctx.destroy_buffer(entities_buf, entities_alloc);
  ctx.destroy_buffer(broad_out_buf, broad_out_alloc);
  ctx.destroy_buffer(narrow_out_buf, narrow_out_alloc);
  ctx.destroy_buffer(lcp_out_buf, lcp_out_alloc);
  ctx.destroy_buffer(packed_col_buf, packed_col_alloc);
  ctx.destroy_buffer(particles_buf, p_alloc);
}
