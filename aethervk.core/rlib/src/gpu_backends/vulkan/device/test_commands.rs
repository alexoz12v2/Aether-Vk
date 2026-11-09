#[cfg(test)]
use super::*;
#[cfg(test)]
use crate::gpu::{RenderDevice, ScopedCommandBuffer};
#[cfg(test)]
use crate::gpu_backends::vulkan::device::test_utils::{
  setup_assets_dir, setup_render_frontend_for_tests,
};
#[cfg(test)]
use aethervk_oshal_rlib::os::native::this_thread;

#[test]
fn stress_test_command_pool_rotation() {
  setup_assets_dir();
  let (_pool_arc, render_frontend, render_device_handle, _) =
    setup_render_frontend_for_tests(false);

  let res = render_frontend.with_device(render_device_handle, |device| {
    let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
      device.as_any().downcast_ref().unwrap();

    // Submit 100 commands (far exceeding MAX_BUFFERS_PER_POOL of 32)
    // This will force the CommandPool allocator to rotate `active` to `pending`
    // multiple times, and allocate brand new pools.
    for _ in 0..100 {
      let task_id = device.create_task();

      // Get compute queue command buffer
      let (cmd_buffer_handle, _cmd_native) = vulkan_device.get_command_buffer_and_native()?;
      let cmd_scope = ScopedCommandBuffer::new(device, cmd_buffer_handle, Some(task_id))?;

      // We could record a `cmd_copy_buffer` here, but simply opening, closing, and
      // submitting the command buffer is enough to put it in flight and stress the pool allocation ring.
      cmd_scope.submit()?;

      // Simulate the LogicThread doing a GPU timeline poll to recycle discarded command buffers.
      // This prevents exhaustion of the 24 max pending pools limit.
      let gpu_timeline_val = unsafe {
        vulkan_device
          .device
          .timeline_semaphore
          .get_semaphore_counter_value(vulkan_device.kernels.timeline)
      }
      .unwrap_or(0);

      let items = vulkan_device.kernels.discard_pool.pop_ready_items(gpu_timeline_val);
      if !items.is_empty() {
        crate::gpu_backends::vulkan::device::DiscardPool::destroy_items_lock_free(
          &vulkan_device.device,
          items,
        );
      }
    }

    // Final synchronization to wait for remaining GPU tasks
    let final_task_id = device.create_task();
    let (cmd, _) = vulkan_device.get_command_buffer_and_native()?;
    let scope = ScopedCommandBuffer::new(device, cmd, Some(final_task_id))?;
    scope.submit()?;

    // Wait for device to idle
    unsafe { vulkan_device.device.device_wait_idle() }.unwrap();

    crate::types::GpuResult::Ok(())
  });

  assert!(res.is_ok(), "Command pool stress test failed");
}

/// `AETHERVK_SINGLE_QUEUE=1` maps every role to the graphics queue and routes compute submissions
/// through the graphics submission lock.
#[test]
fn single_queue_mode_maps_every_role_to_the_graphics_queue() {
  setup_assets_dir();
  unsafe { std::env::set_var("AETHERVK_SINGLE_QUEUE", "1") };
  let (_pool_arc, render_frontend, render_device_handle, _) =
    setup_render_frontend_for_tests(false);
  render_frontend
    .with_device(render_device_handle, |device| {
      let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
        device.as_any().downcast_ref().unwrap();
      let g = vulkan_device.get_graphics_queue();
      let c = vulkan_device.get_compute_queue();
      assert_eq!(g.handle, c.handle, "compute must be the graphics queue");
      assert_eq!(g.family_index, c.family_index);
      assert!(vulkan_device.device.single_queue);
      assert!(core::ptr::eq(
        vulkan_device.device.compute_submission_lock(),
        &vulkan_device.device.submission_lock
      ));
      // a compute submission still works on the shared queue
      let (handle, _) = vulkan_device.get_compute_command_buffer_and_native()?;
      vulkan_device.begin_command_buffer_all(handle, QueueRole::Compute)?;
      let (sem, value) =
        vulkan_device.submit_command_buffer_generic(handle, None, &[], &[], QueueRole::Compute)?;
      vulkan_device.device.wait_for_semaphore_value(sem, value, 2_000_000_000)?;
      unsafe { vulkan_device.device.device_wait_idle() }.unwrap();
      crate::types::GpuResult::Ok(())
    })
    .unwrap();
  unsafe { std::env::remove_var("AETHERVK_SINGLE_QUEUE") };
}

/// The fault report names the buffer a faulting address falls in.
#[test]
fn fault_address_map_names_registered_buffers() {
  use crate::gpu_backends::vulkan::device::fault::{
    describe_address, register_address_range, unregister_address_range, with_isolated_address_maps,
  };
  with_isolated_address_maps(|| {
    register_address_range("TestBufA", 0x1000_0000, 0x1000);
    register_address_range("TestBufB", 0x2000_0000, 0x100);
    assert_eq!(
      describe_address(0x1000_0230),
      "TestBufA + 0x230 (size 0x1000)"
    );
    assert_eq!(
      describe_address(0x2000_0100 + 0x40),
      "0x40 past the end of TestBufB"
    );
    assert_eq!(describe_address(0x9000_0000_0000), "no registered buffer");
    unregister_address_range("TestBufA");
    unregister_address_range("TestBufB");
  });
}

/// A burst of 20 pools' worth of graphics command buffers (more than the 17 the old free cap of 16
/// took back) returns at once, three times: after each drain no pending pool is left with nothing
/// in flight, `pending` is empty, and the next burst allocates (with the old cap, 3 pools stuck
/// idle in `pending` per drain and the allocation failed once they filled it).
#[test]
fn command_pools_take_back_every_pool_after_a_burst() {
  setup_assets_dir();
  let (_pool_arc, render_frontend, render_device_handle, _) =
    setup_render_frontend_for_tests(false);
  let res = render_frontend.with_device(render_device_handle, |device| {
    let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
      device.as_any().downcast_ref().unwrap();
    let tid = this_thread::id();
    let family = vulkan_device.get_graphics_queue().family_index;
    let pools = vulkan_device.res.read().command_pools.clone().unwrap();
    for round in 0..3 {
      for _ in 0..20 * 32 {
        let (cmd, _native) = vulkan_device.get_command_buffer_and_native()?;
        ScopedCommandBuffer::new(device, cmd, None)?.submit()?;
      }
      unsafe { vulkan_device.device.device_wait_idle() }.unwrap();
      // everything has completed: return every discarded command buffer
      let items = vulkan_device.res.read().discard_pool.pop_ready_items(u64::MAX);
      crate::gpu_backends::vulkan::device::DiscardPool::destroy_items_lock_free(
        &vulkan_device.device,
        items,
      );
      let (pending, idle, free) = pools.pool_counts(tid, family).unwrap();
      std::println!("round {round}: pending {pending} (idle {idle}), free {free}");
      assert_eq!(idle, 0, "round {round}: {idle} idle pools stuck in pending");
      assert_eq!(pending, 0, "round {round}: {pending} pools left in pending");
    }
    crate::types::GpuResult::Ok(())
  });
  assert!(res.is_ok(), "burst allocation failed");
}

/// A nucleus mesh whose GPU resource is retired by the logic thread (a scene restore's
/// `cleanup_stale_scene_resources`) while the render thread is between building the draw list
/// and recording the draw, in the same frame that re-uploaded its position buffer twice (two
/// nucleus-radius changes). The frame must not fail, the mesh draw is skipped, and nothing the
/// frame reads or copies into may be queued for destruction before the frame's own timeline
/// value, even when another graphics submission takes the next value first.
fn drive_mesh_retired_mid_frame(legacy: bool) -> Result<(), alloc::string::String> {
  use crate::{
    gpu::{
      RenderableInstanceId,
      frame::{DrawCall, RenderLayer, RenderScene},
    },
    scene::{CameraComponent, StaticMeshComponent, TransformComponent},
  };
  use aethervk_oshal_rlib::math::{
    matrix::{SquareMatrix, mat4::Mat4x4f32},
    quaternion::Quaternion,
    vector::{Vector, vec3::Vec3f32, vec4::Quat},
  };

  setup_assets_dir();
  LEGACY_MESH_RETIRE.store(legacy, core::sync::atomic::Ordering::Relaxed);
  let skipped_before = SKIPPED_MESH_DRAWS.load(core::sync::atomic::Ordering::Relaxed);
  let (_pool_arc, render_frontend, render_device_handle, pe) =
    setup_render_frontend_for_tests(true);
  let pe = pe.unwrap();

  let outcome = render_frontend.with_device(render_device_handle, |device| {
    let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
      device.as_any().downcast_ref().unwrap();
    let [width, height] = device.get_presentation_engine_extent(pe)?;

    // the nucleus: a UV sphere like the comet's, 60 km
    let mut sphere = crate::simulation::comet::generate_uv_sphere(60.0, 12, 24, 1.0, false);
    sphere.id = 0x6E75_636C_6575_73; // a stable cache key, as `Comet::id` is for the nucleus
    let mesh_id = sphere.id;
    let index_count = sphere.indices.len() as u32;
    let component = StaticMeshComponent {
      asset_path: alloc::string::String::new(),
      mesh: alloc::sync::Arc::new(sphere),
      emissive_color: [0.0; 4],
      is_visible: true,
    };
    let positions = |radius: f32| -> alloc::vec::Vec<f32> {
      component
        .mesh
        .vertices
        .iter()
        .flat_map(|v| [v.normal[0] * radius, v.normal[1] * radius, v.normal[2] * radius])
        .collect()
    };
    let new_scene = || {
      let mut scene = RenderScene::new(
        (
          TransformComponent {
            position: Vec3f32::from_array([0.0, 0.0, 300.0]),
            rotation: Quat::identity(),
            scale: Vec3f32::from_array([1.0, 1.0, 1.0]),
          },
          CameraComponent::new_persp(45.0f32.to_radians(), 1.0, 0.1, 10000.0),
        ),
        0,
        0,
        [width, height],
      );
      scene.depth_layers.push(RenderLayer::new(0, 1.0, Vec3f32::zero(), 0.1, 10000.0));
      scene
    };
    let draw_call = |res| {
      DrawCall::from_handles_and_matrix(
        res,
        index_count,
        None,
        Mat4x4f32::identity(),
        [0.0; 3],
        0.0,
        [0.0; 3],
        true,
        0,
        0,
      )
    };
    let key = RenderableInstanceId::from_physical_mesh(mesh_id);
    let mesh_buffers = || -> Option<[ash::vk::Buffer; 3]> {
      let res = vulkan_device.res.read();
      let entry = res.physical_mesh2_resources.get(&key)?;
      match entry.value() {
        resources::ResourceState::Ready(r) => Some([
          r.position_vertex_buffer.buffer.get(),
          r.attributes_vertex_buffer.buffer.get(),
          r.index_buffer.buffer.get(),
        ]),
        _ => None,
      }
    };
    let wait_task = |task: u64| -> GpuResult<()> {
      let t0 = std::time::Instant::now();
      while !device.is_task_completed(task)? {
        if t0.elapsed().as_secs() > 20 {
          return Err(gpu_err!("frame did not complete"));
        }
        this_thread::sleep_for(core::time::Duration::from_millis(2));
      }
      Ok(())
    };

    // ── frame 1: the mesh is uploaded and drawn ────────────────────────────────
    let task1 = device.create_task();
    device.start_frame()?;
    let acquire = device.acquire_next_image(pe)?;
    let cmd = device.get_command_buffer()?;
    device.set_command_buffer_presentation_engine(cmd, pe)?;
    {
      let scope = ScopedCommandBuffer::new(device, cmd, Some(task1))?;
      let res = device.create_physical_mesh2_resources(cmd, mesh_id, &component, pe, "nucleus")?;
      let mut scene = new_scene();
      scene.depth_layers[0].draw_calls.push(draw_call(res));
      device.begin_compositing_render_pass(cmd, pe, &acquire)?;
      let rp = crate::gpu::ScopedRenderPass::new(device, cmd);
      let extent = device.get_presentation_engine_extent(pe)?;
      device.set_viewport(cmd, &crate::gpu::Viewport::from_extent(extent))?;
      device.set_scissor(cmd, &crate::gpu::Rect2D::from_extent(extent))?;
      crate::gpu::frame::render_frame(device, cmd, pe, &scene)?;
      rp.end()?;
      scope.submit()?;
    }
    device.present(pe, acquire.image_index as usize, acquire.frame_index as usize)?;
    wait_task(task1)?;
    device.success_task(task1);
    let [pos0, attr, idx] = mesh_buffers().ok_or(gpu_err!("mesh not ready after frame 1"))?;

    // ── frame 2: two radius changes, the draw list resolves the mesh, then the restore
    //    retires it before the draw is recorded ──────────────────────────────────
    let task2 = device.create_task();
    device.start_frame()?;
    let acquire = device.acquire_next_image(pe)?;
    let cmd = device.get_command_buffer()?;
    device.set_command_buffer_presentation_engine(cmd, pe)?;
    let (pos1, pos2, render_result) = {
      let scope = ScopedCommandBuffer::new(device, cmd, Some(task2))?;
      device.enqueue_mesh_position_update(mesh_id, positions(61.0));
      device.flush_pending_mesh_updates(cmd)?;
      let [pos1, _, _] = mesh_buffers().ok_or(gpu_err!("mesh not ready after swap 1"))?;
      device.enqueue_mesh_position_update(mesh_id, positions(62.0));
      device.flush_pending_mesh_updates(cmd)?;
      let [pos2, _, _] = mesh_buffers().ok_or(gpu_err!("mesh not ready after swap 2"))?;
      assert!(pos0 != pos1 && pos1 != pos2, "each update must install a new position buffer");

      // the draw list of this frame names the (Ready) mesh
      let res = device.get_physical_mesh2_resources(mesh_id, pe)?;
      let mut scene = new_scene();
      scene.depth_layers[0].draw_calls.push(draw_call(res));

      // the logic thread restores a scene without this mesh
      vulkan_device.cleanup_stale_scene_resources(&alloc::collections::BTreeSet::new(), &[])?;
      assert!(mesh_buffers().is_none(), "the restore must retire the mesh resource");

      // another graphics submission takes the next timeline value before this frame does
      let other = device.get_command_buffer()?;
      ScopedCommandBuffer::new(device, other, None)?.submit()?;

      device.begin_compositing_render_pass(cmd, pe, &acquire)?;
      let rp = crate::gpu::ScopedRenderPass::new(device, cmd);
      let extent = device.get_presentation_engine_extent(pe)?;
      device.set_viewport(cmd, &crate::gpu::Viewport::from_extent(extent))?;
      device.set_scissor(cmd, &crate::gpu::Rect2D::from_extent(extent))?;
      let render_result = crate::gpu::frame::render_frame(device, cmd, pe, &scene);
      rp.end()?;
      scope.submit()?;
      (pos1, pos2, render_result)
    };
    device.present(pe, acquire.image_index as usize, acquire.frame_index as usize)?;
    let frame_value = vulkan_device.res.read().timeline_manager.get_task_target_value(task2)?;

    // what the frame reads (attributes, indices) and copies into (pos1, pos2) must outlive it
    let mut too_early = alloc::vec::Vec::new();
    {
      let res = vulkan_device.res.read();
      for (name, buffer) in [("pos0", pos0), ("pos1", pos1), ("pos2", pos2), ("attr", attr), ("idx", idx)] {
        match res.discard_pool.queued_timeline_of_buffer(buffer) {
          Some(t) if t < frame_value => too_early.push(alloc::format!("{name} at {t}")),
          Some(_) => {}
          None => too_early.push(alloc::format!("{name} not queued (leaked or already destroyed)")),
        }
      }
    }
    wait_task(task2)?;
    device.success_task(task2);

    let mut problems = alloc::vec::Vec::new();
    if let Err(e) = render_result {
      problems.push(alloc::format!("render_frame failed: {e:?}"));
    }
    if !too_early.is_empty() {
      problems.push(alloc::format!(
        "queued for destruction before the frame's value {frame_value}: {}",
        too_early.join(", ")
      ));
    }
    let skipped = SKIPPED_MESH_DRAWS.load(core::sync::atomic::Ordering::Relaxed) - skipped_before;
    if problems.is_empty() && skipped == 0 {
      problems.push("the retired mesh draw was not skipped".into());
    }
    Ok::<_, GpuError>(problems)
  });

  LEGACY_MESH_RETIRE.store(false, core::sync::atomic::Ordering::Relaxed);
  drop(render_frontend);
  let errors = crate::gpu_backends::vulkan::utils::VULKAN_ERROR_MESSAGES.lock().unwrap().clone();
  let problems = outcome.map_err(|e| alloc::format!("{e:?}"))?;
  let mut all = problems;
  if !errors.is_empty() {
    all.push(alloc::format!("validation errors: {errors:?}"));
  }
  if all.is_empty() { Ok(()) } else { Err(all.join("; ")) }
}

#[test]
fn mesh_retired_mid_frame_is_skipped_and_freed_after_the_frame() {
  if let Err(problems) = drive_mesh_retired_mid_frame(false) {
    panic!("{problems}");
  }
}

/// The same drive on the previous lifetime handling: the frame fails ("resource missing") and
/// the buffers the frame copies into are queued one value too early. Documents what the test
/// above detects; ignored because it exercises the bug on purpose.
#[test]
#[ignore]
fn mesh_retired_mid_frame_legacy_handling_shows_the_bug() {
  let problems = drive_mesh_retired_mid_frame(true).expect_err("the legacy handling must fail");
  println!("legacy handling: {problems}");
  assert!(problems.contains("resource missing"), "{problems}");
  assert!(problems.contains("queued for destruction before"), "{problems}");
}
