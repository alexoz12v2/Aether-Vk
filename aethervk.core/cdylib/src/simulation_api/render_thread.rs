use alloc::vec;
use alloc::vec::Vec;
use thingbuf::mpsc;
use aethervk_oshal_rlib as oshal;
use aethervk_core_rlib::gpu;
use aethervk_core_rlib::gpu::frame::{CameraRenderData, ResourceUploadResult};
use aethervk_core_rlib::gpu::{PresentationEngineHandle, RenderDevice, RenderScene, SwapchainStatus};
use aethervk_core_rlib::gpu::scene_conversion::{RenderSceneExtraction, SceneConversionExt};
use aethervk_core_rlib::math::collision::linear_bvh::{LinearBVHNode, LinearBound};
use aethervk_core_rlib::scene::{
  BillboardType, BvhDebugComponent, CameraComponent, EntityId, FollowingComponent, GridComponent,
  HiddenComponent, ImageBillboardComponent, MarkersComponent, MeasurementComponent,
  PhysicalMeshComponent, RenderableDataRef, Scene, SelectedComponent, SkyComponent, SunComponent,
  TransformComponent,
};
use aethervk_core_rlib::types::{EngineError, EngineResult, GpuError, GpuResult};
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::matrix::{Matrix4, MatrixVectorMul, SquareMatrix};
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::vector::{Vector3, Vector4};
use aethervk_oshal_rlib::os;
use aethervk_oshal_rlib::os::{thread, NativeError, ThreadingError};
use aethervk_oshal_rlib::os::thread::Thread;
use crate::SimulationContext;
use crate::structs::{RenderCommand, RenderFeedback, RenderTaskStatus, RenderThreadContext};

/// Retries a channel send operation. Evaluates to `Ok(())` on success,
/// or `Err($err)` if max attempts are reached.
macro_rules! try_send_with_limit {
  ($action:expr, $attempts:expr, $delay:expr$(,)?) => {{
    let success = channel_utils::retry_with_limit($action, $attempts, $delay);
    if success {
      Ok(())
    } else {
      Err(GpuError::InvalidState(
        "[Render Thread] process_command | failed to send feedback",
      ))
    }
  }};
}

pub fn start_render_thread(
  render_rx: mpsc::Receiver<RenderCommand>,
  render_params: RenderThreadContext,
) -> EngineResult<Thread> {
  thread::spawn(move || {
    assert!(render_params.is_render_single_ownership());
    let render_device_handle = render_params.render_device_handle;
    let render_frontend = {
      let r = render_params
        .render_frontend
        .try_borrow_mut()
        .map_err(|_| EngineError::InvalidOperation("Failed to borrow render_frontend"));
      if let Err(e) = r {
        oshal::log!("render_thread | render_frontend borrow: {:?}", e);
        return;
      }
      let frontend = unsafe { r.unwrap_unchecked() }
        .take()
        .ok_or(EngineError::InvalidOperation(
          "render_frontend was already None",
        ));
      if let Ok(render_frontend) = frontend {
        render_frontend
      } else {
        oshal::log!("render_thread | render_frontend acquire: {:?}", unsafe {
          frontend.unwrap_err_unchecked()
        });
        return;
      }
    };
    loop {
      match render_rx.try_recv() {
        Ok(cmd) => {
          if let RenderCommand::Shutdown = cmd {
            break;
          }
          if let Err(e) = render_frontend.with_device(render_device_handle, |render_device| {
            process_command(cmd, render_device, &render_params)
          }) {
            oshal::log!("render_thread | process_command failed: {:?}", e);
          }
          // match cmd.data {
          //   RenderCommandContent::RenderFrame {
          //     mut packet,
          //     scene,
          //     task_id,
          //   } => {
          //     packet.clear_color = clear_color;
          //     let scene_guard = scene.as_ref();
          //     let cursor_ent = packet.cursor_entity;
          //     let sun_ent = packet.sun_entity;
          //     let mut c_payload = RenderPayloadData {
          //       packet: &mut packet,
          //       presentation_engine,
          //       scene: &scene_guard,
          //       cursor_entity: cursor_ent,
          //       sun_entity: sun_ent,
          //       task_id,
          //     };

          //     let res = frontend.take_and(|context| {
          //       context
          //           .deref_device_and(
          //             render_device_handle,
          //             &mut c_payload as *mut _ as *mut core::ffi::c_void,
          //             render_payload_ffi,
          //           )
          //           .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
          //           .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
          //     });

          //     if let Some(Err(e)) = res {
          //       oshal::log!(
          //       "[RenderThread] task_id={} | Error on render_payload_ffi={}",
          //       task_id,
          //       e.to_string()
          //     );
          //       // Report failure to task registry
          //       let _ = frontend.take_and(|context| {
          //         let _ = context
          //             .deref_device_and(
          //               render_device_handle,
          //               &mut (task_id, e) as *mut _ as *mut core::ffi::c_void,
          //               |device, data| {
          //                 let (tid, err) =
          //                     unsafe { &*(data as *mut (u64, aethervk_core_rlib::types::EngineError)) };
          //                 if let aethervk_core_rlib::types::EngineError::Gpu(gpu_err) = err {
          //                   device.fail_task(*tid, gpu_err.clone());
          //                 } else {
          //                   device.fail_task(
          //                     *tid,
          //                     aethervk_core_rlib::types::GpuError::InvalidState("invalid state"),
          //                   );
          //                 }
          //                 Ok(())
          //               },
          //             )
          //             .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
          //             .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from));
          //         Ok(())
          //       });
          //     }
          //   }
          //   RenderCommandContent::DownloadImage {
          //     buffer,
          //     buffer_size,
          //     success,
          //     done_signal,
          //   } => {
          //     let slice = unsafe { core::slice::from_raw_parts_mut(buffer.0, buffer_size) };
          //     let mut payload = (presentation_engine, slice);
          //     let res = frontend.take_and(|context| {
          //       context
          //           .deref_device_and(
          //             render_device_handle,
          //             &mut payload as *mut _ as *mut core::ffi::c_void,
          //             |device, data| {
          //               let (engine, buf) =
          //                   unsafe { &mut *(data as *mut (gpu::PresentationEngineHandle, &mut [u8])) };
          //               device.download_windowless_image(*engine, *buf, None)
          //             },
          //           )
          //           .ok_or(aethervk_core_rlib::types::EngineError::InvalidNullArgument)
          //           .and_then(|r| r.map_err(aethervk_core_rlib::types::EngineError::from))
          //     });
          //     unsafe { *(success.0) = matches!(res, Some(Ok(()))) };
          //     done_signal.store(true, core::sync::atomic::Ordering::Release);
          //   }
          //   RenderCommandContent::SetClearColor(color) => {
          //     clear_color = color;
          //   }
          //   RenderCommandContent::Resize { width, height } => {
          //     let mut data = (presentation_engine, width, height);
          //     let _ = frontend.take_and(|context| {
          //       let _ = context.deref_device_and(
          //         render_device_handle,
          //         &mut data as *mut _ as *mut core::ffi::c_void,
          //         |device, data_ptr| {
          //           let (pe, w, h) =
          //               unsafe { &mut *(data_ptr as *mut (gpu::PresentationEngineHandle, u32, u32)) };
          //           device.resize_presentation_engine(*pe, *w, *h)
          //         },
          //       );
          //       Ok(())
          //     });
          //   }
          //   RenderCommandContent::GenerateSky => {
          //     let _ = frontend.take_and(|context| {
          //       let _ = context.deref_device_and(
          //         render_device_handle,
          //         core::ptr::null_mut(),
          //         |device, _| device.generate_sky(),
          //       );
          //       Ok(())
          //     });
          //   }
          //   RenderCommandContent::Shutdown => break,
          // },
        }
        Err(e) => {
          if let thingbuf::mpsc::errors::TryRecvError::Closed = e {
            break;
          }
          // Avoid pegging CPU if no commands
          oshal::os::native::this_thread::sleep_for(core::time::Duration::from_millis(1));
        }
      }
    }
  })
  .map_err(|err| <ThreadingError as Into<NativeError>>::into(err))
  .map_err(|err| <NativeError as Into<EngineError>>::into(err))
}

fn render_payload_ffi(device: &dyn RenderDevice, data: *mut core::ffi::c_void) -> GpuResult<()> {
  let payload = unsafe { &mut *(data as *mut RenderPayloadData) };

  device.start_frame()?;
  let acquire_result = device.acquire_next_image(payload.presentation_engine)?;
  if acquire_result.status.needs_resize() {
    // handled via resize command or next frame
    device.success_task(payload.task_id);
    return Ok(());
  }

  let mut render_scene = gpu::frame::RenderScene::new((
    payload.packet.camera_transform,
    payload.packet.camera_component,
  ));

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SunComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        if let Some(transform) = payload.scene.global_transform(entity) {
          render_scene.sun = Some((entity, *comp, transform));
        }
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        render_scene.sky = Some((entity, *comp));
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::GridComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        render_scene.grid = Some((entity, *comp));
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::CursorComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        if let Some(transform) = payload.scene.global_transform(entity) {
          let _ = render_scene.add_renderable(
            device,
            entity,
            transform.to_mat4(),
            RenderableDataRef::Cursor(comp),
            payload.presentation_engine,
            "Cursor",
            false,
            [1.0, 1.0, 1.0, 1.0],
          );
        }
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::MeasurementComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        let _ = render_scene.add_renderable(
          device,
          entity,
          aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::identity(),
          RenderableDataRef::Measurement(comp),
          payload.presentation_engine,
          "Measurement",
          false,
          [1.0, 1.0, 1.0, 1.0],
        );
      }
    });

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::ImageBillboardComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        let mut model_matrix = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::identity();
        if let Some(transform) = payload.scene.global_transform(entity) {
          model_matrix = transform.to_mat4();
        }
        let _ = render_scene.add_renderable(
          device,
          entity,
          model_matrix,
          RenderableDataRef::ImageBillboard(comp),
          payload.presentation_engine,
          "ImageBillboard",
          false,
          [1.0, 1.0, 1.0, 1.0],
        );
      }
    });

  for item in &payload.packet.render_items {
    let is_hidden = payload
      .scene
      .with_component(
        item.entity_id,
        |_c: &aethervk_core_rlib::scene::HiddenComponent| {},
      )
      .is_some();
    if is_hidden {
      continue;
    }
    let _ = payload.scene.with_component(
      item.entity_id,
      |mesh: &PhysicalMeshComponent| -> GpuResult<()> {
        let mut draw_outline = payload.packet.outlines_enabled;
        let mut outline_color = [0.0, 0.0, 0.0, 0.0];

        let is_selected = payload
          .scene
          .with_component(
            item.entity_id,
            |_c: &aethervk_core_rlib::scene::SelectedComponent| {},
          )
          .is_some();
        let is_following = payload
          .scene
          .with_component(
            item.entity_id,
            |_c: &aethervk_core_rlib::scene::FollowingComponent| {},
          )
          .is_some();

        if is_selected {
          draw_outline = true;
          outline_color = [1.0, 1.0, 1.0, 1.0];
        } else if is_following {
          draw_outline = true;
          outline_color = [0.2, 0.5, 1.0, 1.0];
        } else if payload.packet.outlines_enabled {
          draw_outline = true;
          outline_color = [0.2, 0.5, 1.0, 0.5];
        }

        let _ = render_scene.add_renderable(
          device,
          item.entity_id,
          item.model_matrix,
          RenderableDataRef::PhysicalMesh(mesh),
          payload.presentation_engine,
          &alloc::format!("Comet_{:?}", item.entity_id),
          draw_outline,
          outline_color,
        );
        Ok(())
      },
    );
  }

  // BVH debug rendering
  let mut all_bvh_nodes = Vec::new();
  for item in &payload.packet.render_items {
    let is_hidden = payload
      .scene
      .with_component(
        item.entity_id,
        |_c: &aethervk_core_rlib::scene::HiddenComponent| {},
      )
      .is_some();
    if is_hidden {
      continue;
    }
    let mut dbg_states = None;
    payload.scene.with_component(
      item.entity_id,
      |dbg: &aethervk_core_rlib::scene::BvhDebugComponent| {
        dbg_states = Some(dbg.node_render_states.clone());
      },
    );

    if let Some(states) = dbg_states {
      payload
        .scene
        .with_component(item.entity_id, |mesh: &PhysicalMeshComponent| {
          if let Some(bvh) = &mesh.mesh.bvh {
            for (i, &render) in states.iter().enumerate() {
              if render && i < bvh.nodes.len() {
                all_bvh_nodes.push((bvh.nodes[i].bound.clone(), item.model_matrix));
              }
            }
          }
        });
    }
  }

  let mut sun_opt = None;
  if let Some(sun_entity) = payload.sun_entity {
    if payload
      .scene
      .with_component(
        sun_entity,
        |_: &aethervk_core_rlib::scene::HiddenComponent| {},
      )
      .is_none()
    {
      payload.scene.with_component(
        sun_entity,
        |sun_comp: &aethervk_core_rlib::scene::SunComponent| {
          sun_opt = Some(*sun_comp);
        },
      );
    }
    if let Some(sun_comp) = sun_opt {
      if let Some(sun_transform) = payload.scene.global_transform(sun_entity) {
        render_scene.sun = Some((sun_entity, sun_comp, sun_transform.into()));
      }
    }
  }

  let mut sky_opt = None;
  payload
    .scene
    .query1::<aethervk_core_rlib::scene::SkyComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        sky_opt = Some((entity, *comp));
      }
    });

  let mut grid_opt = None;
  payload
    .scene
    .query1::<aethervk_core_rlib::scene::GridComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        grid_opt = Some((entity, *comp));
      }
    });

  if let Some((id, comp)) = sky_opt {
    render_scene.sky = Some((id, comp));
  }
  if let Some((id, comp)) = grid_opt {
    render_scene.grid = Some((id, comp));
  }

  let cmd_buffer = device.get_command_buffer()?;
  device.begin_command_buffer(cmd_buffer)?;
  if let Some(sun_comp) = sun_opt {
    // safety: if we found the sun_comp then there's a sun entity
    let sun_entity = unsafe { payload.sun_entity.unwrap_unchecked() };
    device.update_sun(cmd_buffer, sun_entity, (128, 128, 128))?;
  }
  device.begin_render_pass(cmd_buffer, payload.presentation_engine, &acquire_result)?;

  let extent = device.get_presentation_engine_extent(payload.presentation_engine)?;
  let root_viewport = gpu::Viewport {
    x: 0.0,
    y: 0.0,
    width: extent[0] as f32,
    height: extent[1] as f32,
    min_depth: 0.0,
    max_depth: 1.0,
  };
  device.set_viewport(cmd_buffer, &root_viewport)?;
  let _ = device.set_scissor(
    cmd_buffer,
    &gpu::Rect2D {
      offset: [0, 0],
      extent,
    },
  );

  let _ = device.render_ui_rect(
    cmd_buffer,
    payload.packet.clear_color,
    [-1.0, -1.0],
    [2.0, 2.0],
    payload.presentation_engine,
  );

  device.render_frame(cmd_buffer, &render_scene)?;

  // Compute view matrix to print
  let view =
      <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_columns(
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 1.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_quat_custom_frame(
        payload.packet.camera_transform.rotation.conjugate(),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::translation(payload.packet.camera_transform.position * -1.0);

  let view_proj = payload.packet.camera_component.projection * view;

  // if !all_bvh_nodes.is_empty() {
  //   let _ = device.render_bvh(
  //     cmd_buffer,
  //     &all_bvh_nodes,
  //     view_proj.into(),
  //     payload.presentation_engine,
  //   );
  // }

  payload
    .scene
    .query1::<aethervk_core_rlib::scene::MeasurementComponent, _>(|entity, comp| {
      if payload
        .scene
        .with_component(entity, |_c: &aethervk_core_rlib::scene::HiddenComponent| {})
        .is_none()
      {
        let mid = comp.pos1 + (comp.pos2 - comp.pos1) * 0.5;
        let mid_vec4 = aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
          mid.x(),
          mid.y(),
          mid.z(),
          1.0,
        );
        let mut clip = view_proj.mul_vector(mid_vec4);
        if clip.w() > 0.0 {
          clip = clip / clip.w();
          if clip.z() >= 0.0 && clip.z() <= 1.0 {
            let ndc_x = clip.x();
            let ndc_y = clip.y();

            let screen_x = (ndc_x * 0.5 + 0.5) * payload.packet.window_width as f32;
            let screen_y = (ndc_y * 0.5 + 0.5) * payload.packet.window_height as f32;

            let distance = (comp.pos2 - comp.pos1).length();
            let text = alloc::format!("{:.3} m", distance);

            let _ = device.render_text(
              cmd_buffer,
              &text,
              24.0,
              [1.0, 1.0, 1.0, 1.0],
              [screen_x, screen_y],
            );
          }
        }
      }
    });

  device.end_render_pass(cmd_buffer)?;
  device.submit_command_buffer(cmd_buffer, Some(payload.task_id))?;

  let _ = device.present(
    payload.presentation_engine,
    acquire_result.image_index as usize,
    acquire_result.frame_index as usize,
  );

  Ok(())
}

// fn collect_render_packet(ctx: &SimulationContext) -> Option<RenderPacket> {
//   let mut render_items = Vec::new();
//   let mut matrix_stack = vec![Mat4x4f32::identity()];
//
//   let active = ctx.active_scene()?;
//   if active.active_camera_entity.is_none() {
//     return None;
//   }
//   let active_camera_entity = unsafe { active.active_camera_entity.unwrap_unchecked() };
//
//   active.scene.traverse_with_hooks(
//     active.root_entity,
//     &mut matrix_stack,
//     &mut |stack: &mut Vec<Mat4x4f32>,
//           entity: EntityId,
//           transform_opt: Option<TransformComponent>,
//           mesh_opt: Option<&PhysicalMeshComponent>| {
//       let local_transform = transform_opt
//         .map(|c| {
//           Mat4x4f32::translation(c.position)
//             * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(c.rotation)
//             * Mat4x4f32::from_scale(c.scale)
//         })
//         .unwrap_or(Mat4x4f32::identity());
//
//       if let Some(parent_transform) = stack.last() {
//         let global_transform = *parent_transform * local_transform;
//
//         if mesh_opt.is_some() {
//           render_items.push(RenderItem {
//             entity_id: entity,
//             model_matrix: global_transform,
//           });
//         }
//         stack.push(global_transform);
//       }
//       true
//     },
//     &mut |stack: &mut Vec<Mat4x4f32>, _| {
//       stack.pop();
//     },
//   );
//
//   let mut camera_transform = TransformComponent {
//     position: Vec3f32::from_components(0.0, 0.0, 0.0),
//     rotation: Quat::identity(),
//     scale: Vec3f32::from_components(1.0, 1.0, 1.0),
//   };
//   let mut camera_component = CameraComponent {
//     projection: Mat4x4f32::identity(),
//     near_plane: 0.1,
//     far_plane: 10000.0,
//   };
//
//   if let Some(global) = active.scene.global_transform(active_camera_entity) {
//     camera_transform = global;
//   }
//   let _ = active
//     .scene
//     .with_component(active_camera_entity, |c| camera_component = *c);
//
//   Some(RenderPacket {
//     render_items,
//     camera_transform,
//     camera_component,
//     window_width: ctx.window_width,
//     window_height: ctx.window_height,
//     outlines_enabled: active
//       .outlines_enabled
//       .load(core::sync::atomic::Ordering::Relaxed),
//     clear_color: ctx.clear_color,
//     sun_entity: active.sun_entity,
//     cursor_entity: active.cursor_entity,
//   })
// }

fn process_command(
  cmd: RenderCommand,
  render_device: &dyn RenderDevice,
  ctx: &RenderThreadContext,
) -> GpuResult<()> {
  let _1ms = core::time::Duration::from_millis(1);
  let max_attempts = 10;
  match cmd {
    // this is processed in render_thread function
    RenderCommand::Shutdown => Ok(()),
    RenderCommand::RenderFrame(render_frame) => {
      // The FFI Caller thread, before launching this command, should have already
      // updated the camera's projection matrix.
      let render_scene = render_frame.prepare_scene(render_device)?;
      let task_id = render_device.create_task();
      // `render_device.success_task` will be called by thread pool when timeline advances
      match do_render_scene_async(
        render_device,
        render_scene,
        render_frame.presentation_engine_handle,
        task_id,
      ) {
        Ok(()) => {
          try_send_with_limit!(
            || ctx
              .render_feedback_tx
              .try_send(RenderFeedback::TaskCreated(Some(task_id))),
            max_attempts,
            _1ms,
          )
        }
        Err(err) => {
          render_device.fail_task(task_id, err.clone());
          channel_utils::retry_until_success(
            || {
              ctx
                .render_feedback_tx
                .try_send(RenderFeedback::TaskCreated(None))
            },
            _1ms,
          );
          Err(err)
        }
      }
    }
    RenderCommand::DownloadImage(download_image) => {
      // 1. Check completion. If true, try reading the download.
      // Both steps must succeed to return Ok(true). Any failure returns Err(e).
      let task_status = render_device
        .is_task_completed(download_image.task_id)
        .and_then(|is_completed| {
          if is_completed {
            render_device
              .read_windowless_download(download_image.task_id, unsafe {
                core::slice::from_raw_parts_mut(download_image.buffer.0, download_image.buffer_size)
              })
              .map(|_| true) // Map Ok(()) to Ok(true) to feed into the match
          } else {
            Ok(false)
          }
        });

      // 2. Handle the combined result
      match task_status {
        Ok(true) => {
          try_send_with_limit!(
            || ctx
              .render_feedback_tx
              .try_send(RenderFeedback::TaskQueryStatus(RenderTaskStatus::Completed)),
            max_attempts,
            _1ms
          )
        }
        Ok(false) => {
          try_send_with_limit!(
            || ctx
              .render_feedback_tx
              .try_send(RenderFeedback::TaskQueryStatus(RenderTaskStatus::Pending)),
            max_attempts,
            _1ms
          )
        }
        Err(err) => {
          // Catches errors from both `is_task_completed` and `read_windowless_download`
          channel_utils::retry_until_success(
            || {
              ctx
                .render_feedback_tx
                .try_send(RenderFeedback::TaskQueryStatus(RenderTaskStatus::Error(
                  err.clone(),
                )))
            },
            _1ms,
          );
          Err(err)
        }
      }
    }
    RenderCommand::Resize(_) => {
      todo!();
    }
    // TODO move to logic thread which will dispatch this to an affinity thread for compute
    RenderCommand::GenerateSky => todo!(),
  }
}

fn do_render_scene_async(
  render_device: &dyn RenderDevice,
  render_scene: RenderScene,
  presentation_engine_handle: PresentationEngineHandle,
  task_id: u64,
) -> GpuResult<()> {
  render_device.start_frame()?;

  let acquire_result = render_device.acquire_next_image(presentation_engine_handle)?;
  if acquire_result.status.needs_resize() {
    // handled via resize command or next frame
    render_device.success_task(task_id);
    return Ok(());
  }

  let cmd_buffer = render_device.get_command_buffer()?;
  let cmd_scope = gpu::ScopedCommandBuffer::new(render_device, cmd_buffer, Some(task_id))?;
  if let Some(sun_call) = &render_scene.sun_call {
    // TODO move to kernels
    render_device.update_sun(cmd_buffer, sun_call.entity, (128, 128, 128))?;
  }

  render_device.begin_render_pass(cmd_buffer, presentation_engine_handle, &acquire_result)?;
  let render_pass_scope = gpu::ScopedRenderPass::new(render_device, cmd_buffer);

  let extent = render_device.get_presentation_engine_extent(presentation_engine_handle)?;
  render_device.set_viewport(cmd_buffer, &gpu::Viewport::from_extent(extent))?;
  render_device.set_scissor(cmd_buffer, &gpu::Rect2D::from_extent(extent))?;

  // TODO: 2) Text not included in measurement now (inside render_frame)
  render_device.render_frame(cmd_buffer, &render_scene)?;

  // on `DownloadImage` Command, Query task status and copy data if completed with `render_device.read_windowless_download`
  render_device.record_windowless_download(cmd_buffer, presentation_engine_handle, task_id)?;

  // present and submit
  render_pass_scope.end()?;
  cmd_scope.submit()?;

  if SwapchainStatus::Optimal
    != render_device.present(
      presentation_engine_handle,
      acquire_result.image_index as usize,
      acquire_result.frame_index as usize,
    )?
  {
    oshal::log!(
      "[Render Thread] Warning: render_device.present isn't optimal. Might not be an error"
    );
  }

  Ok(())
}


// TODO possibly, group by pipeline if necessary
impl super::structs::RenderFrame {
  pub fn prepare_scene(&self, device: &dyn RenderDevice) -> GpuResult<gpu::RenderScene> {
    let render_extraction: RenderSceneExtraction = {
      let scene = self.scene.read();
      scene.scene.convert_scene(self.camera_entity, self.render_physical_meshes_outline)
    }?; // <-- THE ECS RWLOCK IS SAFELY DROPPED HERE!

    // --- PASS 2: VULKAN TRANSLATION ---
    render_extraction.build_render_scene(device, self.presentation_engine_handle)
  }
}

pub mod channel_utils {
  use super::*;

  /// Repeatedly attempts an action (like sending a message) up to `max_attempts`.
  /// Returns `true` if successful, `false` if all attempts were exhausted.
  pub fn retry_with_limit<F, E>(
    mut action: F,
    max_attempts: usize,
    delay: core::time::Duration,
  ) -> bool
  where
    F: FnMut() -> Result<(), E>,
  {
    for _ in 0..max_attempts {
      if action().is_ok() {
        return true;
      }
      os::native::this_thread::sleep_for(delay);
    }
    false
  }

  /// Repeatedly attempts an action infinitely until it succeeds.
  pub fn retry_until_success<F, E>(mut action: F, delay: core::time::Duration)
  where
    F: FnMut() -> Result<(), E>,
  {
    loop {
      if action().is_ok() {
        break;
      }
      os::native::this_thread::sleep_for(delay);
    }
  }
}

/// Function used to get, from the midpoint (slightly above) from a measurement,
/// The screen space coordinates to render some text
/// TODO move into rlib
fn from_world_space_to_screen_space(
  mid: Vec3f32,
  view_proj: Mat4x4f32,
  window_extent: (f32, f32),
) -> Option<(f32, f32)> {
  let mid_vec4 = mid.to_point();
  let mut clip = view_proj.mul_vector(mid_vec4);
  if clip.w() > 0.0 {
    clip = clip / clip.w();
    if clip.z() >= 0.0 && clip.z() <= 1.0 {
      let ndc_x = clip.x();
      let ndc_y = clip.y();

      let screen_x = (ndc_x * 0.5 + 0.5) * window_extent.0;
      let screen_y = (ndc_y * 0.5 + 0.5) * window_extent.1;

      return Some((screen_x, screen_y));
    }
  }

  None
}
