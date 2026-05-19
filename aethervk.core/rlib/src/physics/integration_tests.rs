#[cfg(test)]
mod tests {
  use crate::scene::PhysicalMeshComponent;
  use crate::simulation::comet::Comet;
  use crate::{
    gpu::{DynamicBody, Kernels},
    gpu_backends::simulation_step,
    physics::{
      cpu_kernels::{CpuScalarKernels, CpuSimdKernels},
      physics_scene::PhysicsScene,
    },
    scene::{
      ColliderComponent, ColliderShape, EntityId, KinematicComponent, ReferenceFrameComponent,
      ReferenceFrameType, Scene, TransformComponent,
      particles::{ParticleData, ParticleSystemComponent},
    },
  };
  use aethervk_oshal_rlib::{
    math::{
      quaternion::Quaternion,
      vector::{Vector, Vector3, vec3::Vec3f32},
    },
    os::pool::ThreadPool,
  };
  use polyhedral_mass_properties::{MassProperties, TriangleContrib};
  use std::sync::Arc;

  fn dummy_mesh() -> PhysicalMeshComponent {
    let tri_contrib = TriangleContrib::new([-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.5, 1.0, 0.0]);
    PhysicalMeshComponent {
      asset_path: "".into(),
      mesh: Arc::new(Comet {
        id: 0,
        vertices: alloc::vec![],
        indices: alloc::vec![],
        albedo_map: None,
        normal_map: None,
        roughness_map: None,
        ao_map: None,
        mass_properties: MassProperties::from_contrib_sum(tri_contrib).unwrap(),
        bvh: None,
        pa_basis_bf: None,
        bf_to_pa: None,
      }),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
      use_new_path: false,
      paint_display_mode: 0,
      sphere_center: [0.0; 3],
      sphere_radius: 1.0,
      grid_color: [0.0; 3],
      grid_density: 0.0,
    }
  }

  fn run_simulation<K>(kernels: &K, scene: &mut Scene, duration_secs: f32)
  where
    K: Kernels,
  {
    let mut ps = PhysicsScene::build_from_scene(scene);
    let dt_us = 16_667; // 60 FPS
    let iterations = (duration_secs * 60.0) as usize;

    for _ in 0..iterations {
      let _ = simulation_step(kernels, &mut ps, scene, 0, dt_us, true);
    }
  }

  #[test]
  fn test_single_frame_all_shapes() {
    let mut scene = Scene::new(std::sync::Arc::new(crate::gpu::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_all_crate_components();

    let root = scene.spawn_reference_frame(
      "MicroFrame",
      None,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
      ReferenceFrameType::Micro,
      0.1,
      100.0,
    );

    // Spawn a Sphere RigidBody
    let sphere = scene.spawn_entity("Sphere");
    scene.set_parent(sphere, Some(root));
    scene
      .add_component(
        sphere,
        TransformComponent {
          position: Vec3f32::from_components(-5.0, 0.0, 0.0),
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.add_component(sphere, dummy_mesh()).unwrap();
    scene
      .add_component(
        sphere,
        ColliderComponent {
          shape: ColliderShape::Sphere { radius: 1.0 },
          mass: 10.0,
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        sphere,
        KinematicComponent {
          velocity: Vec3f32::from_components(2.0, 0.0, 0.0),
          ..Default::default()
        },
      )
      .unwrap();

    // Spawn an OBB RigidBody
    let obb = scene.spawn_entity("OBB");
    scene.set_parent(obb, Some(root));
    scene
      .add_component(
        obb,
        TransformComponent {
          position: Vec3f32::from_components(5.0, 0.0, 0.0),
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.add_component(obb, dummy_mesh()).unwrap();
    scene
      .add_component(
        obb,
        ColliderComponent {
          shape: ColliderShape::OBB {
            half_extents: Vec3f32::from_components(1.0, 1.0, 1.0),
          },
          mass: 10.0,
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        obb,
        KinematicComponent {
          velocity: Vec3f32::from_components(-2.0, 0.0, 0.0),
          ..Default::default()
        },
      )
      .unwrap();

    let scalar_kernels = CpuScalarKernels {};
    run_simulation(&scalar_kernels, &mut scene, 5.0);

    let t_sphere = scene.global_transform(sphere).unwrap();
    let t_obb = scene.global_transform(obb).unwrap();

    // They should have bounced
    let v_sphere = scene.with_component(sphere, |k: &KinematicComponent| k.velocity).unwrap();
    let v_obb = scene.with_component(obb, |k: &KinematicComponent| k.velocity).unwrap();

    assert!(v_sphere.x() < 0.0, "Sphere should have bounced back");
    assert!(v_obb.x() > 0.0, "OBB should have bounced back");
    assert!(t_sphere.position.x() < 0.0);
    assert!(t_obb.position.x() > 0.0);
  }

  #[test]
  fn test_cross_frame_lca_collision() {
    let mut scene = Scene::new(std::sync::Arc::new(crate::gpu::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_all_crate_components();

    // Macro Frame
    let macro_frame = scene.spawn_reference_frame(
      "MacroFrame",
      None,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
      ReferenceFrameType::Macro,
      1.0,
      10000.0,
    );

    // Micro Frame offset by (10, 0, 0)
    let micro_frame = scene.spawn_reference_frame(
      "MicroFrame",
      Some(macro_frame),
      TransformComponent {
        position: Vec3f32::from_components(10.0, 0.0, 0.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
      ReferenceFrameType::Micro,
      0.1,
      100.0,
    );

    // Spawn an Object in Macro Frame at (5, 0, 0) moving towards +X
    let obj_macro = scene.spawn_entity("ObjMacro");
    scene.set_parent(obj_macro, Some(macro_frame));
    scene
      .add_component(
        obj_macro,
        TransformComponent {
          position: Vec3f32::from_components(5.0, 0.0, 0.0),
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.add_component(obj_macro, dummy_mesh()).unwrap();
    scene
      .add_component(
        obj_macro,
        ColliderComponent {
          shape: ColliderShape::Sphere { radius: 1.0 },
          mass: 10.0,
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        obj_macro,
        KinematicComponent {
          velocity: Vec3f32::from_components(2.0, 0.0, 0.0),
          ..Default::default()
        },
      )
      .unwrap();

    // Spawn an Object in Micro Frame at (-5, 0, 0) local -> which is (5, 0, 0) in Macro Frame
    // So they are basically inside each other. Wait, we want them to collide.
    // Let's place it at (-2, 0, 0) local -> which is (8, 0, 0) Macro
    // It's moving towards -X
    let obj_micro = scene.spawn_entity("ObjMicro");
    scene.set_parent(obj_micro, Some(micro_frame));
    scene
      .add_component(
        obj_micro,
        TransformComponent {
          position: Vec3f32::from_components(-2.0, 0.0, 0.0),
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.add_component(obj_micro, dummy_mesh()).unwrap();
    scene
      .add_component(
        obj_micro,
        ColliderComponent {
          shape: ColliderShape::Sphere { radius: 1.0 },
          mass: 10.0,
          ..Default::default()
        },
      )
      .unwrap();
    // Micro's velocity is scaled? No, velocity is always local units per second.
    // If it's -2.0 local units, it moves towards macro frame.
    scene
      .add_component(
        obj_micro,
        KinematicComponent {
          velocity: Vec3f32::from_components(-2.0, 0.0, 0.0),
          ..Default::default()
        },
      )
      .unwrap();

    let scalar_kernels = CpuScalarKernels {};
    run_simulation(&scalar_kernels, &mut scene, 5.0);

    let v_macro = scene.with_component(obj_macro, |k: &KinematicComponent| k.velocity).unwrap();
    let v_micro = scene.with_component(obj_micro, |k: &KinematicComponent| k.velocity).unwrap();

    println!("v_macro: {:?}", v_macro);
    println!("v_micro: {:?}", v_micro);

    // They should have collided and bounced due to LCA resolution
    assert!(v_macro.x() < 0.0, "Macro object should have bounced back (-X)");
    assert!(v_micro.x() > 0.0, "Micro object should have bounced back (+X)");
  }

  #[test]
  fn test_deeply_nested_gravity() {
    let mut scene = Scene::new(std::sync::Arc::new(crate::gpu::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_all_crate_components();

    let macro_frame = scene.spawn_reference_frame(
      "MacroFrame",
      None,
      TransformComponent {
        position: Vec3f32::from_components(0.0, 0.0, 0.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
      ReferenceFrameType::Macro,
      1.0,
      10000.0,
    );
    let micro_frame = scene.spawn_reference_frame(
      "MicroFrame",
      Some(macro_frame),
      TransformComponent {
        position: Vec3f32::from_components(10.0, 0.0, 0.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
      ReferenceFrameType::Micro,
      0.1,
      100.0,
    );
    let sub_micro_frame = scene.spawn_reference_frame(
      "SubMicroFrame",
      Some(micro_frame),
      TransformComponent {
        position: Vec3f32::from_components(0.0, 10.0, 0.0),
        rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        scale: Vec3f32::from_components(1.0, 1.0, 1.0),
      },
      ReferenceFrameType::Micro,
      0.01,
      10.0,
    );

    // Planet in Macro Frame
    let planet = scene.spawn_entity("Planet");
    scene.set_parent(planet, Some(macro_frame));
    scene
      .add_component(
        planet,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();
    scene.add_component(planet, dummy_mesh()).unwrap();
    scene
      .add_component(
        planet,
        crate::scene::AlmanacPlanet {
          mu: 1_000_000.0,
          naif_id: 399, // Earth
          rot_period: 0.0,
          bf_to_pa: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
        },
      )
      .unwrap();
    scene.add_component(planet, KinematicComponent::default()).unwrap(); // needed to be kinematic body

    // Particle System in SubMicro Frame
    let particle_system = scene.spawn_entity("ParticleSystem");
    scene.set_parent(particle_system, Some(sub_micro_frame));
    scene
      .add_component(
        particle_system,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .unwrap();

    let p1 = ParticleData {
      id_low: 0,
      id_high: 0,
      age_low: 0,
      age_high: 0,
      position: [0.0, 0.0, 0.0],
      mass: 1.0,
      velocity: [0.0, 0.0, 0.0],
      active: 1,
    };
    let mut sys = ParticleSystemComponent {
      particles: std::sync::Arc::new(spin::RwLock::new(vec![p1])),
      bvh: None,
      accumulator: 0,
      next_id: 1,
    };
    sys.update_bvh(1.0);
    scene.add_component(particle_system, sys).unwrap();

    let scalar_kernels = CpuScalarKernels {};
    run_simulation(&scalar_kernels, &mut scene, 5.0);

    let sys = scene
      .with_component(particle_system, |sys: &ParticleSystemComponent| {
        sys.particles.read().clone()
      })
      .unwrap();

    // Planet is at Macro (0,0,0). SubMicro is at Macro (10, 1.0, 0)
    // The gravity should pull the particle towards the planet.
    assert!(
      sys[0].velocity[0] < 0.0 || sys[0].velocity[1] < 0.0,
      "Particle should be pulled by gravity"
    );
  }

  #[test]
  fn test_backend_determinism() {
    let pool = Arc::new(ThreadPool::new(4).unwrap());

    let setup_scene = || -> (Scene, EntityId, EntityId) {
      let mut scene = Scene::new(std::sync::Arc::new(crate::gpu::RwLock::new(
        crate::simulation::texture_cache::TextureCache::new("AetherVk"),
      )));
      scene.register_all_crate_components();

      let root = scene.spawn_reference_frame(
        "MicroFrame",
        None,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
        ReferenceFrameType::Micro,
        0.1,
        100.0,
      );

      let sphere = scene.spawn_entity("Sphere");
      scene.set_parent(sphere, Some(root));
      scene
        .add_component(
          sphere,
          TransformComponent {
            position: Vec3f32::from_components(-5.0, 0.0, 0.0),
            rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
            scale: Vec3f32::from_components(1.0, 1.0, 1.0),
          },
        )
        .unwrap();
      scene.add_component(sphere, dummy_mesh()).unwrap();
      scene
        .add_component(
          sphere,
          ColliderComponent {
            shape: ColliderShape::Sphere { radius: 1.0 },
            mass: 10.0,
            ..Default::default()
          },
        )
        .unwrap();
      scene
        .add_component(
          sphere,
          KinematicComponent {
            velocity: Vec3f32::from_components(2.0, 0.0, 0.0),
            ..Default::default()
          },
        )
        .unwrap();

      let obb = scene.spawn_entity("OBB");
      scene.set_parent(obb, Some(root));
      scene
        .add_component(
          obb,
          TransformComponent {
            position: Vec3f32::from_components(5.0, 0.0, 0.0),
            rotation: aethervk_oshal_rlib::math::vector::vec4::Quat::identity(),
            scale: Vec3f32::from_components(1.0, 1.0, 1.0),
          },
        )
        .unwrap();
      scene.add_component(obb, dummy_mesh()).unwrap();
      scene
        .add_component(
          obb,
          ColliderComponent {
            shape: ColliderShape::OBB {
              half_extents: Vec3f32::from_components(1.0, 1.0, 1.0),
            },
            mass: 10.0,
            ..Default::default()
          },
        )
        .unwrap();
      scene
        .add_component(
          obb,
          KinematicComponent {
            velocity: Vec3f32::from_components(-2.0, 0.0, 0.0),
            ..Default::default()
          },
        )
        .unwrap();

      (scene, sphere, obb)
    };

    let (mut scene_scalar, sphere_scalar, obb_scalar) = setup_scene();
    let (mut scene_simd, sphere_simd, obb_simd) = setup_scene();

    let scalar_kernels = CpuScalarKernels {};
    run_simulation(&scalar_kernels, &mut scene_scalar, 5.0);

    let simd_kernels = CpuSimdKernels { thread_pool: pool };
    run_simulation(&simd_kernels, &mut scene_simd, 5.0);

    let v_sphere_scalar =
      scene_scalar.with_component(sphere_scalar, |k: &KinematicComponent| k.velocity).unwrap();
    let v_sphere_simd =
      scene_simd.with_component(sphere_simd, |k: &KinematicComponent| k.velocity).unwrap();

    let v_obb_scalar =
      scene_scalar.with_component(obb_scalar, |k: &KinematicComponent| k.velocity).unwrap();
    let v_obb_simd =
      scene_simd.with_component(obb_simd, |k: &KinematicComponent| k.velocity).unwrap();

    approx::assert_abs_diff_eq!(v_sphere_scalar.x(), v_sphere_simd.x(), epsilon = 1e-4);
    approx::assert_abs_diff_eq!(v_obb_scalar.x(), v_obb_simd.x(), epsilon = 1e-4);
  }
}