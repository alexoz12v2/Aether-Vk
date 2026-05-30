#[cfg(test)]
mod tests {
  use crate::{
    gpu::{
      compute_push_constants::{
        LbvhPushConstants, MotionBoundsPushConstants, MotionRefitPushConstants,
      },
      vulkan::{device::LogicalDevice, physics::VulkanCommandBuffer},
    },
    gpu_backends::vulkan::physics::VulkanComputeKernels,
  };

  #[test]
  // Placeholder stub — body is empty.  Implement and un-ignore once a real
  // end-to-end GPU pipeline test (motion_bounds + lbvh_prepass + lbvh_build)
  // has been written.  Run with: cargo test -- --ignored
  #[ignore]
  fn test_motion_blas_full_pipeline() {
    // This is an integration test to run the full Motion BLAS pipeline end-to-end:
    // 1. Allocation (respecting is_list)
    // 2. Leaf generation (motion_bounds.comp)
    // 3. lbvh_prepass.comp (writing 0xFFFFFFFF to root)
    // 4. Hierarchy build / Refit (motion_refit.comp)
    // It verifies that the shaders run without crashing and the layouts match.
  }

  #[test]
  fn test_physics_scene_tlas_blas_hierarchy() {
    use crate::{
      math::collision::{
        bounds::AABB,
        linear_bvh::{LinearBVH, LinearBVHNode, LinearBound},
      },
      physics::physics_scene::{PhysicsScene, RbNode, pack_meta, *},
      scene::{
        KinematicComponent, PhysicalMeshComponent, ReferenceFrameComponent, ReferenceFrameType,
        Scene, TransformComponent,
        particles::{ParticleData, ParticleSystemComponent},
      },
      simulation::comet::{Comet, Vertex},
    };
    use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
    use alloc::sync::Arc;
    use spin::RwLock;

    let mut scene = Scene::new(Arc::new(RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("test"),
    )));
    scene.register_all_crate_components();

    // 1. Setup Macro Frame
    let macro_entity = scene.spawn_entity("macro");
    scene.add_component(macro_entity, TransformComponent::default()).unwrap();
    scene
      .add_component(
        macro_entity,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Macro,
          scale: 1.0,
          soi_radius: 1000.0,
          depth_layer: 0,
        },
      )
      .unwrap();

    // 2. Setup Micro Frame
    let micro_entity = scene.spawn_entity("micro");
    scene
      .add_component(
        micro_entity,
        TransformComponent {
          position: Vec3f32::from_array([10.0, 0.0, 0.0]),
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        micro_entity,
        ReferenceFrameComponent {
          frame_type: ReferenceFrameType::Micro,
          scale: 1.0,
          soi_radius: 100.0,
          depth_layer: 1,
        },
      )
      .unwrap();
    scene.set_parent(micro_entity, Some(macro_entity));

    let create_dummy_mesh_component = || -> PhysicalMeshComponent {
      let vertices = alloc::vec![
        Vertex {
          position: [-1.0, -1.0, 0.0],
          normal: [0.0, 0.0, 1.0],
          uv: [0.0, 0.0],
          tangent: [1.0, 0.0, 0.0, 1.0]
        },
        Vertex {
          position: [1.0, -1.0, 0.0],
          normal: [0.0, 0.0, 1.0],
          uv: [1.0, 0.0],
          tangent: [1.0, 0.0, 0.0, 1.0]
        },
        Vertex {
          position: [0.0, 1.0, 0.0],
          normal: [0.0, 0.0, 1.0],
          uv: [0.5, 1.0],
          tangent: [1.0, 0.0, 0.0, 1.0]
        },
      ];
      let indices = alloc::vec![0, 1, 2];

      let bvh_nodes = alloc::vec![LinearBVHNode {
        center_of_mass: [0.0, 0.0, 0.0],
        mass: 0.0,
        bound: LinearBound::AABB(AABB::new(
          Vec3f32::from_array([-1.0, -1.0, -0.1]),
          Vec3f32::from_array([1.0, 1.0, 0.1]),
        )),
        left_child_or_primitive_offset: 0,
        right_child_offset: u32::MAX,
        primitive_count: 1,
      }];

      let bvh = LinearBVH {
        header: crate::math::collision::linear_bvh::LinearBVHHeader {
          preciseness: 0,
          node_count: bvh_nodes.len() as u32,
          primitive_count: 1,
        },
        nodes: bvh_nodes,
        primitives: alloc::vec![0],
      };

      let comet = Comet {
        id: 1,
        vertices,
        indices,
        albedo_map: None,
        normal_map: None,
        roughness_map: None,
        ao_map: None,
        mass_properties: unsafe { core::mem::zeroed() },
        bvh: Some(bvh),
        pa_basis_bf: None,
        bf_to_pa: None,
      };

      PhysicalMeshComponent {
        asset_path: "".into(),
        mesh: alloc::sync::Arc::new(comet),
        emissive_intensity: 0.0,
        emissive_color: [0.0, 0.0, 0.0],
        use_new_path: false,
        paint_display_mode: 0,
        sphere_center: [0.0, 0.0, 0.0],
        sphere_radius: 1.0,
        grid_color: [0.0, 0.0, 0.0],
        grid_density: 1.0,
      }
    };

    // 3. Add Mesh in Macro Frame
    let macro_mesh = scene.spawn_entity("mesh");
    scene.add_component(macro_mesh, TransformComponent::default()).unwrap();
    scene
      .add_component(
        macro_mesh,
        KinematicComponent {
          velocity: Vec3f32::from_array([1.0, 0.0, 0.0]),
          ..Default::default()
        },
      )
      .unwrap();
    scene.add_component(macro_mesh, create_dummy_mesh_component()).unwrap();
    scene.set_parent(macro_mesh, Some(macro_entity));

    // 4. Add Particle System in Micro Frame
    let micro_particles = scene.spawn_entity("particles");
    scene.add_component(micro_particles, TransformComponent::default()).unwrap();
    let sys = ParticleSystemComponent::new(10);
    {
      let mut ps = sys.particles.write();
      ps.push(ParticleData {
        id_low: 0,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [0.0, 0.0, 0.0],
        velocity: [0.0, 1.0, 0.0],
        mass: 1.0,
        active: 1,
      });
      ps.push(ParticleData {
        id_low: 1,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [1.0, 1.0, 1.0],
        velocity: [0.0, 0.0, 1.0],
        mass: 1.0,
        active: 1,
      });
    }
    scene.add_component(micro_particles, sys).unwrap();
    scene.set_parent(micro_particles, Some(micro_entity));

    let dt = 0.016;
    let physics_scene = PhysicsScene::build_from_scene(&scene, dt);

    // Assert macro_tlas parent encompasses children
    fn assert_encompasses(nodes: &[RbNode], idx: u32) {
      if idx == u32::MAX {
        return;
      }
      let node = &nodes[idx as usize];
      if let Some(left) = node.left {
        let left_node = &nodes[left as usize];
        assert!(node.min[0] <= left_node.min[0]);
        assert!(node.min[1] <= left_node.min[1]);
        assert!(node.min[2] <= left_node.min[2]);
        assert!(node.max[0] >= left_node.max[0]);
        assert!(node.max[1] >= left_node.max[1]);
        assert!(node.max[2] >= left_node.max[2]);
        assert_encompasses(nodes, left);
      }
      if let Some(right) = node.right {
        let right_node = &nodes[right as usize];
        assert!(node.min[0] <= right_node.min[0]);
        assert!(node.min[1] <= right_node.min[1]);
        assert!(node.min[2] <= right_node.min[2]);
        assert!(node.max[0] >= right_node.max[0]);
        assert!(node.max[1] >= right_node.max[1]);
        assert!(node.max[2] >= right_node.max[2]);
        assert_encompasses(nodes, right);
      }
    }

    if !physics_scene.macro_tlas.nodes.is_empty() {
      assert_encompasses(&physics_scene.macro_tlas.nodes, 0);
    }

    fn unpack_meta(meta: u32) -> (bool, u32, u32, u32) {
      let is_leaf = (meta & 0x8000_0000) != 0;
      let frame = (meta >> 29) & 0x3;
      let shape = (meta >> 27) & 0x3;
      let index = meta & 0x07FF_FFFF;
      (is_leaf, frame, shape, index)
    }

    // Assert that leaves of Scene TLAS is either particles, body or micro frame
    for node in &physics_scene.macro_tlas.nodes {
      if let Some(meta) = node.leaf_meta {
        let (is_leaf, frame, shape, _idx) = unpack_meta(meta);
        assert!(is_leaf);
        assert!(
          shape == BVH_SHAPE_AABB || shape == BVH_SHAPE_SPHERE || shape == BVH_SHAPE_SUB_TLAS,
          "Invalid shape: {} for frame: {}",
          shape,
          frame
        );
        assert_eq!(frame, BVH_FRAME_MACRO);
      }
    }

    // Assert that micro frame is a proper nested TLAS
    for (_frame_idx, micro_tlas) in &physics_scene.micro_tlases {
      if !micro_tlas.nodes.is_empty() {
        assert_encompasses(&micro_tlas.nodes, 0);
      }
      for node in &micro_tlas.nodes {
        if let Some(meta) = node.leaf_meta {
          let (is_leaf, frame, shape, _idx) = unpack_meta(meta);
          assert!(is_leaf);
          // Leaves in micro frame are only particles (SPHERE) or body (AABB)
          assert!(
            shape == BVH_SHAPE_AABB || shape == BVH_SHAPE_SPHERE,
            "Invalid shape: {} for frame: {}",
            shape,
            frame
          );
          assert_eq!(frame, BVH_FRAME_MICRO);
        }
      }
    }

    // Assert particles count in BLAS is what we expected
    assert_eq!(physics_scene.particle_blases.len(), 1);
    let particle_blas = physics_scene.particle_blases[0].as_ref().unwrap();
    // 2 particles were added
    let leaf_count = particle_blas.nodes.iter().filter(|n| n.primitive_count > 0).count();
    assert_eq!(leaf_count, 2);
  }
}
