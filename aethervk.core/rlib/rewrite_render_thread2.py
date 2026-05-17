import re

with open("src/simulation_api/render_thread.rs", "r") as f:
    content = f.read()

# 1. Update `process_command` call
old_call = """          if let Err(e) = render_frontend.with_device(render_device_handle, |render_device| {
            process_command(cmd, render_device, &render_params, &mut first_render_map)
          }) {"""
new_call = """          if let Err(e) = render_frontend.with_device(render_device_handle, |render_device| {
            process_command(cmd, render_device, &render_params, &mut first_render_map, render_frontend.clone(), render_device_handle)
          }) {"""
content = content.replace(old_call, new_call)

# 2. Update `process_command` signature
old_sig = """fn process_command(
  cmd: RenderCommand,
  render_device: &dyn RenderDevice,
  ctx: &RenderThreadContext,
  first_render_map: &mut hashbrown::HashMap<PresentationEngineHandle, bool>,
) -> GpuResult<()> {"""
new_sig = """fn process_command(
  cmd: RenderCommand,
  render_device: &dyn RenderDevice,
  ctx: &RenderThreadContext,
  first_render_map: &mut hashbrown::HashMap<PresentationEngineHandle, bool>,
  render_frontend: alloc::sync::Arc<gpu::RenderFrontend>,
  render_device_handle: gpu::RenderDeviceHandle,
) -> GpuResult<()> {"""
content = content.replace(old_sig, new_sig)

# 3. Replace RenderCommand::RenderFrames handling
# We will just replace everything from `RenderCommand::RenderFrames` down to the end of the function.
start_idx = content.find("    RenderCommand::RenderFrames")
if start_idx == -1:
    print("Could not find RenderCommand::RenderFrames")

new_logic = """    RenderCommand::RenderFrames(render_frames) => {
      render_device.start_frame()?;
      
      let mut handles = alloc::vec::Vec::new();
      
      for render_frame in render_frames {
        let task_id_feedback = alloc::sync::Arc::clone(&render_frame.task_id);
        let extent_res = render_device.get_presentation_engine_extent(render_frame.presentation_engine_handle);
        let extent = match extent_res {
          Ok(e) => e,
          Err(_) => {
            task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
            continue;
          }
        };
        if extent[0] == 0 || extent[1] == 0 {
          task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
          continue;
        }

        let acquire_result = render_device.acquire_next_image(render_frame.presentation_engine_handle);
        let acquire_result = match acquire_result {
          Ok(res) => res,
          Err(_) => {
            task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
            continue;
          }
        };
        
        if acquire_result.status.needs_resize() {
          task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
          continue;
        }

        let is_first_render = if first_render_map.contains_key(&render_frame.presentation_engine_handle) {
          *unsafe { first_render_map.get(&render_frame.presentation_engine_handle).unwrap_unchecked() }
        } else {
          let _ = first_render_map.insert(render_frame.presentation_engine_handle, true);
          true
        };
        
        let frontend = render_frontend.clone();
        let handle = render_device_handle;
        let thread_pool = alloc::sync::Arc::clone(&ctx.thread_pool);
        
        let tasklet = ctx.thread_pool.spawn_tasklet(None, move || -> GpuResult<(gpu::CommandBufferHandle, bool, bool)> {
            frontend.with_device(handle, |render_device| {
               let task_id = render_device.create_task();
               
               let present_guard = gpu::FrameCancelGuard::new(render_device, render_frame.presentation_engine_handle, acquire_result);

               let extracted_scene = render_frame.extract_scene(extent, Some(&thread_pool))?;
               
               let cmd_buffer = render_device.get_command_buffer()?;
               render_device.set_command_buffer_presentation_engine(cmd_buffer, render_frame.presentation_engine_handle)?;
               let cmd_scope = gpu::ScopedCommandBuffer::new(render_device, cmd_buffer, Some(task_id))?;
               
               let time_readings = render_frame.scene.read().time_state.read().time_info.read().current();
               let debug_name = render_frame.scene.read().debug_name.clone();

               let mut render_scene = extracted_scene.build_render_scene(
                 render_device,
                 render_frame.presentation_engine_handle,
                 cmd_buffer,
                 time_readings,
                 extent.into(),
                 &debug_name,
               )?;
               
               if let Some(sun_call) = &render_scene.sun_call {
                 render_device.update_sun(cmd_buffer, render_frame.presentation_engine_handle, sun_call.entity, (128, 128, 128), sun_call.radius)?;
               }

               render_device.upload_particle_systems(cmd_buffer, render_frame.presentation_engine_handle, &mut render_scene.particle_calls)?;
               render_device.upload_particle2_systems(cmd_buffer, render_frame.presentation_engine_handle, &mut render_scene.particle2_calls)?;

               if is_first_render && render_frame.custom_render_callback.is_some() {
                 let c = unsafe { render_frame.custom_render_callback.as_ref().unwrap_unchecked() };
                 (c.on_first_render_fn)(render_device, cmd_buffer, render_frame.presentation_engine_handle, &render_scene, c.user_data.0)?
               }

               render_device.begin_render_pass(cmd_buffer, render_frame.presentation_engine_handle, &acquire_result)?;
               let render_pass_scope = gpu::ScopedRenderPass::new(render_device, cmd_buffer);

               render_device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))?;
               render_device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))?;

               gpu::frame::render_frame(render_device, cmd_buffer, &render_scene, render_frame.presentation_engine_handle)?;

               if render_frame.custom_render_callback.is_some() {
                 let c = unsafe { render_frame.custom_render_callback.as_ref().unwrap_unchecked() };
                 (c.after_render_frame_fn)(render_device, cmd_buffer, render_frame.presentation_engine_handle, &render_scene, c.user_data.0)?;
               }

               render_pass_scope.end()?;
               
               let is_windowless = unsafe {
                 render_device.is_presentation_engine_windowless(render_frame.presentation_engine_handle).unwrap_unchecked()
               };
               if is_windowless {
                 if let Err(e) = render_device.record_windowless_download(cmd_buffer, render_frame.presentation_engine_handle, task_id) {
                   return Err(e);
                 }
               }
               
               cmd_scope.submit()?;
               present_guard.defuse();
               
               let task_id_feedback = alloc::sync::Arc::clone(&render_frame.task_id);
               task_id_feedback.store(task_id, core::sync::atomic::Ordering::Release);
               
               Ok((cmd_buffer, is_windowless, is_first_render))
            })
        });
        
        if let Ok(handle) = tasklet {
            handles.push((handle, render_frame.presentation_engine_handle, acquire_result));
        } else {
            let task_id_feedback = alloc::sync::Arc::clone(&render_frame.task_id);
            task_id_feedback.store(u64::MAX, core::sync::atomic::Ordering::Release);
        }
      }
      
      // Step 2: Synchronize and submit sequentially
      for (handle, pe_handle, acquire_result) in handles {
          handle.wait();
          if let Ok(Ok(Ok((_cmd_buffer, _is_windowless, is_first_render)))) = handle.get() {
             if is_first_render {
                 *unsafe { first_render_map.get_mut(&pe_handle).unwrap_unchecked() } = false;
             }
             if crate::gpu::SwapchainStatus::Optimal != render_device.present(pe_handle, acquire_result.image_index as usize, acquire_result.frame_index as usize)? {
                oshal::log!("[Render Thread] Warning: render_device.present isn't optimal. Might not be an error");
             }
          }
      }
      Ok(())
    }
    RenderCommand::Resize(resize_cmd) => render_device.resize_presentation_engine(
      resize_cmd.presentation_engine_handle,
      resize_cmd.width,
      resize_cmd.height,
    ),
    RenderCommand::GenerateSky => render_device.generate_sky(),
  }
}
"""

end_idx = content.find("fn do_render_scene_async")
if end_idx == -1:
    # already removed?
    end_idx = len(content)

content = content[:start_idx] + new_logic + content[end_idx:]

with open("src/simulation_api/render_thread.rs", "w") as f:
    f.write(content)
