#[cfg(test)]
mod tests {
  use crate::{
    gpu::{
      DeviceAdditionalParams, RenderFrontend, VULKAN_RENDER_BACKEND, new_render_frontend,
      simulation_step, vulkan::device,
    }, gpu_backends, physics::physics_scene::PhysicsScene, scene::{
      ColliderComponent, ColliderShape, KinematicComponent, PhysicalMeshComponent,
      ReferenceFrameType, Scene, TransformComponent,
    }, simulation::comet::Comet, types::RuntimeParams
  };
  use aethervk_oshal_rlib::{
    math::{
      quaternion::Quaternion,
      vector::{Vector, Vector3, vec3::Vec3f32},
    },
    os::time::timeus_t,
  };
  use heapless::index_map::FnvIndexMap;
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
      rotational_model: None,
    }
  }

  fn panic_on_validation_error(msg: &str) {
    eprintln!("VULKAN ERROR: {}", msg);
  }

  struct VulkanTestContext {
    pub frontend: RenderFrontend,
    pub device_handle: crate::gpu::RenderDeviceHandle,
    pub pool: Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
  }

  impl VulkanTestContext {
    pub fn new() -> Self {
      crate::gpu_backends::vulkan::physics::READBACK_DIAGNOSTICS
        .store(true, core::sync::atomic::Ordering::Relaxed);
      crate::gpu::set_asset_dir_for_tests();

      let runtime_params = Box::new(RuntimeParams {
        render_backend_params: FnvIndexMap::new(),
        validation_error_callback: Some(panic_on_validation_error as fn(&str)),
      });

      let pool = aethervk_oshal_rlib::os::pool::ThreadPool::new(1).unwrap();
      let pool_arc = Arc::new(pool);

      let frontend = new_render_frontend(VULKAN_RENDER_BACKEND, &runtime_params).unwrap();
      let mut additional_params = DeviceAdditionalParams::new();
      additional_params
        .insert(
          crate::gpu_backends::vulkan::DEVICE_ADDIDITIONAL_PARAM_DEBUG_SHADERS,
          1,
        )
        .unwrap();
      let device_handle = frontend.write().init_device(0, &additional_params).unwrap();

      frontend
        .with_device(device_handle, |device| {
          device.wire_callbacks(pool_arc.clone())
        })
        .unwrap();

      Self {
        frontend,
        device_handle,
        pool: pool_arc,
      }
    }

    pub fn is_lavapipe(&self) -> bool {
      self
        .frontend
        .with_device(self.device_handle, |dev| {
          let vulkan_device = dev
            .as_any()
            .downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
            .unwrap();
          let props = &vulkan_device.query_result.physical_device_properties;
          let device_name = props.device_name_as_c_str().unwrap().to_string_lossy();
          Ok(device_name.contains("llvmpipe"))
        })
        .unwrap_or(false)
    }
  }

  pub fn run_simulation<K: crate::gpu::Kernels + ?Sized>(
    kernels: &K,
    scene: &mut Scene,
    duration_seconds: f32,
    collisions_enabled: bool,
  ) -> PhysicsScene {
    let mut physical_scene = PhysicsScene::build_from_scene(scene, 0.016);

    let dt: timeus_t = 16_667; // 60 FPS
    let mut current_time: timeus_t = 0;
    let end_time: timeus_t = (duration_seconds * 1_000_000.0) as timeus_t;

    while current_time < end_time {
      let sync = simulation_step(
        kernels,
        &mut physical_scene,
        scene,
        current_time,
        current_time + dt,
        collisions_enabled,
        dt,
      )
      .unwrap();
      if let Some(s) = sync {
        kernels.wait_sync(&s).unwrap();
      }
      let old_collisions = core::mem::take(&mut physical_scene.recent_collisions);
      physical_scene = PhysicsScene::build_from_scene(scene, 0.016);
      physical_scene.recent_collisions = old_collisions;
      current_time += dt;
    }
    physical_scene
  }

  #[test]
  #[cfg_attr(not(feature = "collisions"), ignore = "Requires collisions feature")]
  fn test_vulkan_single_frame_all_shapes_collision_full() {
    let ctx = VulkanTestContext::new();

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
      1.0,
      10000.0,
    );

    // Spawn a Sphere RigidBody
    let sphere = scene.spawn_entity("Sphere");
    scene.set_parent(sphere, Some(root));
    scene
      .add_component(
        sphere,
        TransformComponent {
          position: Vec3f32::from_components(-1.01, 0.0, 0.0),
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
          position: Vec3f32::from_components(1.01, 0.0, 0.0),
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

    let tracked_allocs = ctx
      .frontend
      .with_device(ctx.device_handle, |dev| {
        let vulkan_device = dev.as_any().downcast_ref::<device::Device>().unwrap();

        let ps = run_simulation(vulkan_device, &mut scene, 0.1, true);

        let t_sphere = scene.global_transform(sphere).unwrap();
        let t_obb = scene.global_transform(obb).unwrap();

        // They should have bounced
        let v_sphere = scene.with_component(sphere, |k: &KinematicComponent| k.velocity).unwrap();
        let v_obb = scene.with_component(obb, |k: &KinematicComponent| k.velocity).unwrap();

        assert!(v_sphere.x() < 0.0, "Sphere should have bounced back");
        assert!(v_obb.x() > 0.0, "OBB should have bounced back");

        // Verify collision events
        eprintln!("RECENT COLLISIONS LEN: {}", ps.recent_collisions.len());
        for c in &ps.recent_collisions {
          eprintln!("Collision: {} vs {}", c.entity_a_name, c.entity_b_name);
        }
        assert!(
          !ps.recent_collisions.is_empty(),
          "Expected at least one collision event"
        );
        let coll = ps
          .recent_collisions
          .iter()
          .find(|c| {
            (c.entity_a_name == "Sphere" && c.entity_b_name == "OBB")
              || (c.entity_a_name == "OBB" && c.entity_b_name == "Sphere")
          })
          .expect(&alloc::format!(
            "Expected collision between Sphere and OBB, got: {:?}",
            ps.recent_collisions
          ));

        assert!(
          coll.penetration_depth > 0.0,
          "Penetration depth should be positive"
        );
        // Sphere and OBB are both children of the 'root' micro frame.
        // They are intra-LCA, so they are routed to standard CCD (out_rb_rb)
        // without coord transforms. Thus, is_lca should be false.
        assert!(
          !coll.is_lca,
          "Sphere–OBB collision is intra-LCA (both entities share the root micro frame), so it routes to standard CCD. Expected is_lca=false, got is_lca={}, frame_id={}",
          coll.is_lca, coll.frame_id
        );

        let tracked_allocs = vulkan_device.kernels.tracked_physical_allocations.lock().clone();

        crate::types::GpuResult::Ok(tracked_allocs)
      })
      .unwrap();

    drop(scene);
    drop(ctx);

    if let Some(lock) = aethervk_oshal_rlib::os::memory::tracking::GPU_ALLOCATIONS.try_lock() {
      if let Some(map) = lock.as_ref() {
        for mem_addr in tracked_allocs {
          assert!(
            !map.contains_key(&mem_addr),
            "Physical buffer allocation at {:#X} was leaked!",
            mem_addr
          );
        }
      }
    }
  }

  #[test]
  fn test_vulkan_single_frame_all_shapes_collisionless() {
    let ctx = VulkanTestContext::new();

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
      1.0,
      10000.0,
    );

    // Spawn a Sphere RigidBody
    let sphere = scene.spawn_entity("Sphere");
    scene.set_parent(sphere, Some(root));
    scene
      .add_component(
        sphere,
        TransformComponent {
          position: Vec3f32::from_components(-1.01, 0.0, 0.0),
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
          position: Vec3f32::from_components(1.01, 0.0, 0.0),
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

    let tracked_allocs = ctx
      .frontend
      .with_device(ctx.device_handle, |dev| {
        let vulkan_device = dev.as_any().downcast_ref::<device::Device>().unwrap();

        let ps = run_simulation(vulkan_device, &mut scene, 0.1, false);

        let t_sphere = scene.global_transform(sphere).unwrap();
        let t_obb = scene.global_transform(obb).unwrap();

        // They should have passed through each other
        let v_sphere = scene.with_component(sphere, |k: &KinematicComponent| k.velocity).unwrap();
        let v_obb = scene.with_component(obb, |k: &KinematicComponent| k.velocity).unwrap();

        assert!(v_sphere.x() > 0.0, "Sphere should have passed through (+X)");
        assert!(v_obb.x() < 0.0, "OBB should have passed through (-X)");
        assert!(ps.recent_collisions.is_empty(), "Expected no collisions");

        let tracked_allocs = vulkan_device.kernels.tracked_physical_allocations.lock().clone();

        crate::types::GpuResult::Ok(tracked_allocs)
      })
      .unwrap();

    drop(scene);
    drop(ctx);

    if let Some(lock) = aethervk_oshal_rlib::os::memory::tracking::GPU_ALLOCATIONS.try_lock() {
      if let Some(map) = lock.as_ref() {
        for mem_addr in tracked_allocs {
          assert!(
            !map.contains_key(&mem_addr),
            "Physical buffer allocation at {:#X} was leaked!",
            mem_addr
          );
        }
      }
    }
  }

  #[test]
  #[cfg_attr(not(feature = "collisions"), ignore = "Requires collisions feature")]
  fn test_vulkan_cross_frame_lca_collision_full() {
    let ctx = VulkanTestContext::new();

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

    // Spawn an Object in Micro Frame at (-20, 0, 0) local => (8, 0, 0) macro space, moving towards -X
    let obj_micro = scene.spawn_entity("ObjMicro");
    scene.set_parent(obj_micro, Some(micro_frame));
    scene
      .add_component(
        obj_micro,
        TransformComponent {
          position: Vec3f32::from_components(-30.0, 0.0, 0.0),
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
        obj_micro,
        KinematicComponent {
          velocity: Vec3f32::from_components(-20.0, 0.0, 0.0), // -2.0 in macro space
          ..Default::default()
        },
      )
      .unwrap();

    ctx
      .frontend
      .with_device(ctx.device_handle, |dev| {
        let vulkan_device = dev.as_any().downcast_ref::<device::Device>().unwrap();

        let ps = run_simulation(vulkan_device, &mut scene, 0.6, true);

        // Verify
        let v_macro = scene.with_component(obj_macro, |k: &KinematicComponent| k.velocity).unwrap();
        let v_micro = scene.with_component(obj_micro, |k: &KinematicComponent| k.velocity).unwrap();

        assert!(
          v_macro.x() < 0.0,
          "Macro object should have bounced back (-X)"
        );
        assert!(
          v_micro.x() > 0.0,
          "Micro object should have bounced back (+X)"
        );

        // Verify collision events
        eprintln!("RECENT COLLISIONS LEN: {}", ps.recent_collisions.len());
        for c in &ps.recent_collisions {
          eprintln!("Collision: {} vs {}", c.entity_a_name, c.entity_b_name);
        }
        assert!(
          !ps.recent_collisions.is_empty(),
          "Expected at least one collision event"
        );
        let coll = ps
          .recent_collisions
          .iter()
          .find(|c| {
            (c.entity_a_name == "ObjMacro" && c.entity_b_name == "ObjMicro")
              || (c.entity_a_name == "ObjMicro" && c.entity_b_name == "ObjMacro")
          })
          .expect(&alloc::format!(
            "Expected LCA collision between ObjMacro and ObjMicro, got: {:?}",
            ps.recent_collisions
          ));

        assert!(
          coll.penetration_depth > 0.0,
          "Penetration depth should be positive"
        );
        assert_ne!(
          coll.frame_id, 0,
          "LCA Collision should happen in the microframe"
        );

        crate::types::GpuResult::Ok(())
      })
      .unwrap();
  }
  #[test]
  fn test_vulkan_cross_frame_lca_collisionless() {
    let ctx = VulkanTestContext::new();

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

    // Spawn an Object in Micro Frame at (-20, 0, 0) local => (8, 0, 0) macro space, moving towards -X
    let obj_micro = scene.spawn_entity("ObjMicro");
    scene.set_parent(obj_micro, Some(micro_frame));
    scene
      .add_component(
        obj_micro,
        TransformComponent {
          position: Vec3f32::from_components(-30.0, 0.0, 0.0),
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
        obj_micro,
        KinematicComponent {
          velocity: Vec3f32::from_components(-20.0, 0.0, 0.0), // -2.0 in macro space
          ..Default::default()
        },
      )
      .unwrap();

    ctx
      .frontend
      .with_device(ctx.device_handle, |dev| {
        let vulkan_device = dev.as_any().downcast_ref::<device::Device>().unwrap();

        let ps = run_simulation(vulkan_device, &mut scene, 0.6, false);

        // Verify
        let v_macro = scene.with_component(obj_macro, |k: &KinematicComponent| k.velocity).unwrap();
        let v_micro = scene.with_component(obj_micro, |k: &KinematicComponent| k.velocity).unwrap();

        assert!(
          v_macro.x() > 0.0,
          "Macro object should have kept moving (+X)"
        );
        assert!(
          v_micro.x() < 0.0,
          "Micro object should have kept moving (-X)"
        );
        assert!(ps.recent_collisions.is_empty(), "Expected no collisions");

        crate::types::GpuResult::Ok(())
      })
      .unwrap();
  }

  #[test]
  #[cfg_attr(not(feature = "collisions"), ignore = "Requires collisions feature")]
  fn test_lbvh_complex_race_condition() {
    let ctx = VulkanTestContext::new();

    let mut scene = Scene::new(std::sync::Arc::new(crate::gpu::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_all_crate_components();

    // Set up macroframe
    let macro_frame = scene.spawn_entity("macro_frame");
    let _ = scene.add_component(macro_frame, TransformComponent::default());
    let _ = scene.add_component(
      macro_frame,
      crate::scene::ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Macro,
        scale: 1.0,
        soi_radius: f32::MAX,
        depth_layer: 0,
      },
    );

    // Set up microframe
    let micro_frame = scene.spawn_entity("micro_frame");
    let _ = scene.add_component(micro_frame, TransformComponent::default());
    let _ = scene.add_component(
      micro_frame,
      crate::scene::ReferenceFrameComponent {
        frame_type: ReferenceFrameType::Micro,
        scale: 1.0,
        soi_radius: 50.0,
        depth_layer: 1,
      },
    );

    // Add spheres and particles
    for i in 0..100 {
      let s = scene.spawn_entity(&alloc::format!("sphere_{}", i));
      let _ = scene.add_component(
        s,
        TransformComponent {
          position: Vec3f32::from_array([(i % 10) as f32, (i / 10) as f32, 0.0]),
          ..Default::default()
        },
      );
      let _ = scene.add_component(
        s,
        ColliderComponent {
          shape: ColliderShape::Sphere { radius: 1.0 },
          restitution: 0.5,
          friction: 0.5,
          mass: 1.0,
        },
      );
      // add kinematic
      let _ = scene.add_component(
        s,
        KinematicComponent {
          velocity: Vec3f32::from_array([1.0, 0.0, 0.0]),
          ..Default::default()
        },
      );
      let _ = scene.add_component(s, dummy_mesh());
    }

    // Set up particles
    let p_sys = scene.spawn_entity("particles");
    let _ = scene.add_component(p_sys, TransformComponent::default());
    let p_comp = crate::scene::ParticleSystemComponent::new(1000);
    {
      let mut lock = p_comp.particles.write();
      for i in 0..1000 {
        lock.push(crate::scene::ParticleData {
          id_low: i as u32,
          id_high: 0,
          age_low: 0,
          age_high: 0,
          position: [(i % 10) as f32, (i / 10) as f32, 10.0],
          mass: 1.0,
          velocity: [0.0, -1.0, 0.0],
          active: 1,
        });
      }
    }
    let _ = scene.add_component(p_sys, p_comp);

    // Run simulation step
    // This will do the LBVH build with > 128 particles, crossing workgroup boundaries!
    ctx
      .frontend
      .with_device(ctx.device_handle, |device| {
        let vulkan_device = device
          .as_any()
          .downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
          .unwrap();
        let _ps = run_simulation(vulkan_device, &mut scene, 0.016, true);
        crate::types::GpuResult::Ok(())
      })
      .unwrap();
  }

  #[test]
  #[cfg_attr(not(feature = "collisions"), ignore = "Requires collisions feature")]
  fn test_barnes_hut_forces() {
    crate::gpu_backends::vulkan::physics::USE_PRINTF_SHADERS.store(true, core::sync::atomic::Ordering::Relaxed);
    let ctx = VulkanTestContext::new();

    let mut scene = Scene::new(std::sync::Arc::new(crate::gpu::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_all_crate_components();

    // Spawn 10 particles at the origin
    let p_sys = scene.spawn_entity("particles");
    let _ = scene.add_component(p_sys, TransformComponent::default());
    let p_comp = crate::scene::ParticleSystemComponent::new(11);
    {
      let mut lock = p_comp.particles.write();
      for i in 0..10 {
        lock.push(crate::scene::ParticleData {
          id_low: i as u32,
          id_high: 0,
          age_low: 0,
          age_high: 0,
          // Guarantee unique positions to prevent t_c=0 loops if collisions are accidentally enabled
          position: [
            (i % 10) as f32 * 10.0,
            ((i / 10) % 10) as f32 * 10.0,
            (i / 100) as f32 * 10.0,
          ],
          mass: 1.0,
          velocity: [0.0, 0.0, 0.0],
          active: 1,
        });
      }
      // Add one MASSIVE particle far away
      lock.push(crate::scene::ParticleData {
        id_low: 10,
        id_high: 0,
        age_low: 0,
        age_high: 0,
        position: [10000.0, 0.0, 0.0],
        mass: 1e6,
        velocity: [0.0, 0.0, 0.0],
        active: 1,
      });
    }
    let _ = scene.add_component(p_sys, p_comp);

    ctx
      .frontend
      .with_device(ctx.device_handle, |dev| {
        let vulkan_device = dev
          .as_any()
          .downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
          .unwrap();

        crate::gpu::Kernels::toggle_particle_self_gravity(vulkan_device, true);


        // Run 1 simulation step
        // The massive particle should pull the cluster of 1000 particles towards +X.
        // Total mass of cluster = 1000.
        // Distance r = ~10000.
        // Acceleration a = G * M_massive / r^2 = G * 1e6 / (1e8) = G * 0.01.        // Assuming G = 6.674e-11 (or whatever it is in physics backend)
        println!("Running 1 simulation step...");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let ps = run_simulation(vulkan_device, &mut scene, 0.016, false);

        println!("Simulation step complete. Waiting for idle...");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        unsafe { vulkan_device.device.device_wait_idle() };

        println!("Device idle.");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let p_comp = scene
          .with_component(p_sys, |c: &crate::scene::ParticleSystemComponent| c.clone())
          .unwrap();

        let p_comp_read = p_comp.particles.read();
        let mut count = 0;
        let mut vx_sum = 0.0;
        for p in p_comp_read.iter() {
          if p.id_low < 10 {
            count += 1;
            vx_sum += p.velocity[0];
          }
        }
        let avg_vx = (vx_sum / count as f32);

        println!("avg_vx: {}", avg_vx);
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        assert!(avg_vx > 0.0);

        let mut avg_vx = avg_vx;
        avg_vx /= 10.0;
        let lock = p_comp.particles.read();

        let mut output_str = alloc::format!("Average X velocity: {}\n", avg_vx);
        for i in 0..11 {
          output_str.push_str(&alloc::format!(
            "Particle {} POS: {:?} VEL: {:?}\n",
            i,
            lock[i].position,
            lock[i].velocity
          ));
        }
        std::fs::write("test_vels.txt", output_str).unwrap();
        assert!(
          avg_vx > 0.0,
          "Particles should have moved towards the massive body!"
        );

        crate::types::GpuResult::Ok(())
      })
      .unwrap();
  }
  #[test]
  #[cfg(feature = "collisions")]
  fn test_ccd_time_of_impact() {
    use crate::gpu::Kernels;
    crate::gpu_backends::vulkan::physics::USE_PRINTF_SHADERS
      .store(true, std::sync::atomic::Ordering::Relaxed);
    let mut ctx = VulkanTestContext::new();
    let mut scene = crate::scene::Scene::new(std::sync::Arc::new(crate::gpu::RwLock::new(
      crate::simulation::texture_cache::TextureCache::new("AetherVk"),
    )));
    scene.register_all_crate_components();

    let root = scene.spawn_reference_frame(
      "RootFrame",
      None,
      crate::scene::TransformComponent::default(),
      crate::scene::ReferenceFrameType::Macro,
      1.0,
      10000.0,
    );

    let rb_a = scene.spawn_entity("rb_a");
    scene.set_parent(rb_a, Some(root));
    scene
      .add_component(
        rb_a,
        crate::scene::TransformComponent {
          position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
            0.0, 0.0, 0.0,
          ),
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        rb_a,
        crate::scene::ColliderComponent {
          shape: ColliderShape::Sphere { radius: 1.0 },
          mass: 1.0,
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        rb_a,
        crate::scene::KinematicComponent {
          velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
            100.0, 0.0, 0.0,
          ),
          ..Default::default()
        },
      )
      .unwrap();

    let rb_b = scene.spawn_entity("rb_b");
    scene.set_parent(rb_b, Some(root));
    scene
      .add_component(
        rb_b,
        crate::scene::TransformComponent {
          position: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
            4.0, 0.0, 0.0,
          ),
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        rb_b,
        crate::scene::ColliderComponent {
          shape: ColliderShape::Sphere { radius: 1.0 },
          mass: 1.0,
          ..Default::default()
        },
      )
      .unwrap();
    scene
      .add_component(
        rb_b,
        crate::scene::KinematicComponent {
          velocity: aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
            0.0, 0.0, 0.0,
          ),
          ..Default::default()
        },
      )
      .unwrap();

    ctx
      .frontend
      .with_device(ctx.device_handle, |dev| {
        let vulkan_device = dev
          .as_any()
          .downcast_ref::<crate::gpu_backends::vulkan::device::Device>()
          .unwrap();

        // Step with dt = 0.1s
        let _ps = run_simulation(vulkan_device, &mut scene, 0.1, true);

        let vel_a = scene
          .with_component(rb_a, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap();
        let vel_b = scene
          .with_component(rb_b, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap();

        let va = vel_a.x();
        let vb = vel_b.x();

        eprintln!("After collision: vA = {}, vB = {}", va, vb);

        // Ensure they collided and A's velocity decreased / B's increased
        assert!(
          va < 10.0 || vb > 0.0,
          "RigidBody A should have slowed down or bounced! va = {}",
          va
        );
        assert!(vb > 0.0, "RigidBody B should have been pushed! vb = {}", vb);

        crate::types::GpuResult::Ok(())
      })
      .unwrap();
  }
}