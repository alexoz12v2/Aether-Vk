//! Several live views of one scene through the logic and render threads: a second presentation
//! engine beside the first one, a camera that draws no dust, a camera without the UI layer, a
//! paused view. Windowless engines; skipped when no Vulkan device is available.
//!
//! Guarantees:
//! - two engines of one scene are both rendered by the render thread, and one can be removed
//!   while the other keeps rendering;
//! - a camera with `CAMERA_VIEW_NO_DUST` builds no dust draw call: its engine has no dust view,
//!   no entry in the dust frame ledger and no pyramid, while the other engine of the scene has
//!   all three;
//! - the dust view's image is the same, byte for byte, without a second view, with a running
//!   no-dust view and with a paused one;
//! - a paused view gets no frame at all (none recorded, none presented) and renders again when
//!   resumed;
//! - a camera with `CAMERA_VIEW_NO_UI_LAYER` uploads and draws neither the UI nor the text batch.

use super::review_67p_repro_tests::{
  SAVE_JETS, SPK_COMMITTED, add_particle_system, committed_scene, committed_start,
  load_almanac_files, save_dto, wait_caught_up,
};
use crate::{
  gpu::PresentationEngineHandle,
  scene::{
    CAMERA_VIEW_NO_DUST, CAMERA_VIEW_NO_UI_LAYER, EntityId, RENDER_LAYER_DEFAULT,
    particles::emission_start_us_from_epoch,
  },
  simulation_api::{
    SimulationContext,
    structs::{CustomRenderCallback, LogicCommand, SendPtrMut, TaskStatusCode},
  },
};
use hifitime::{Duration, Epoch};
use std::{
  collections::BTreeMap,
  sync::Mutex,
  time::{Duration as StdDuration, Instant},
};

// ─────────────────────────────────────────────────────────────────────────────
// harness
// ─────────────────────────────────────────────────────────────────────────────

/// A context with a render device, or `None` (the test is skipped).
fn startup() -> Option<Box<SimulationContext>> {
  let asset_dir = format!("{}/../../assets", env!("CARGO_MANIFEST_DIR"));
  SimulationContext::set_asset_path(&asset_dir);
  match SimulationContext::startup(None) {
    Ok(ctx) if ctx.render_frontend().is_some() => Some(ctx),
    _ => {
      println!("Skipping test: Vulkan backend could not be initialized");
      None
    }
  }
}

fn shutdown(ctx: &SimulationContext) {
  let _ = ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown);
}

/// Polls `done` every 5 ms for at most `timeout_s`; whether it became true.
fn wait_until(timeout_s: f64, mut done: impl FnMut() -> bool) -> bool {
  let t0 = Instant::now();
  loop {
    if done() {
      return true;
    }
    if t0.elapsed().as_secs_f64() > timeout_s {
      return false;
    }
    std::thread::sleep(StdDuration::from_millis(5));
  }
}

/// The app's empty scene (Sun, sky, Earth), one day long.
fn empty_scene(ctx: &SimulationContext) -> u64 {
  ctx
    .create_empty_scene(
      true,
      Epoch::from_gregorian_at_midnight(2020, 12, 25, hifitime::TimeScale::UTC),
      Epoch::from_gregorian_at_midnight(2020, 12, 26, hifitime::TimeScale::UTC),
    )
    .expect("empty scene")
}

fn play(ctx: &SimulationContext, scene_id: u64) {
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::PlayScene {
      scene_id,
      speed: aethervk_oshal_rlib::os::time::v2::SimSpeed::Realtime,
    })
    .unwrap();
}

/// A windowless engine of `width` × `height` with a perspective camera. The engine is held (render
/// budget 0) from before its camera exists until `configure` has run on the camera, so no frame
/// of it is ever built with the camera half set up.
fn add_view(
  ctx: &SimulationContext,
  scene_id: u64,
  name: &str,
  (width, height): (u32, u32),
  (fov_rad, near, far): (f32, f32, f32),
  configure: impl FnOnce(EntityId),
) -> (PresentationEngineHandle, EntityId) {
  let pe = ctx.create_presentation_engine(scene_id, width, height).expect("presentation engine");
  assert!(ctx.set_render_budget(scene_id, pe, Some(0)));
  let cam = ctx
    .add_perspective_camera(scene_id, pe, name, fov_rad, near, far)
    .expect("camera");
  let cam = EntityId::from_ffi(cam.get());
  configure(cam);
  assert!(ctx.set_render_budget(scene_id, pe, None));
  (pe, cam)
}

/// What the render thread recorded for one engine (`CustomRenderCallback::after_render_frame_fn`,
/// called once per recorded frame with that frame's `RenderScene`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ViewProbe {
  /// frames recorded
  frames: u64,
  /// frames with an uploaded UI batch / text batch
  ui_batches: u64,
  text_batches: u64,
  /// frames drawn with the UI layer off
  ui_hidden: u64,
  /// dust draw calls over all frames
  dust_calls: u64,
  /// frames with a dust pyramid
  dust_pyramids: u64,
  /// frames flagged as drawing no dust
  no_dust: u64,
}

static PROBES: Mutex<BTreeMap<u64, ViewProbe>> = Mutex::new(BTreeMap::new());

fn probe_after_frame(
  _device: &dyn crate::gpu::RenderDevice,
  _cmd: crate::gpu::CommandBufferHandle,
  pe: PresentationEngineHandle,
  render_scene: &crate::gpu::RenderScene,
  _user: *mut core::ffi::c_void,
) -> crate::types::GpuResult<()> {
  let mut probes = PROBES.lock().unwrap_or_else(|e| e.into_inner());
  let p = probes.entry(pe.0).or_default();
  p.frames += 1;
  p.ui_batches += render_scene.ui_call.is_some() as u64;
  p.text_batches += render_scene.text2_call.is_some() as u64;
  p.ui_hidden += render_scene.hide_ui_labels as u64;
  p.dust_calls += render_scene.depth_layers.iter().map(|l| l.dust_calls.len() as u64).sum::<u64>();
  p.dust_pyramids += (render_scene.dust_view.pyramid != 0) as u64;
  p.no_dust += render_scene.no_dust as u64;
  Ok(())
}

fn probe_first_frame(
  _device: &dyn crate::gpu::RenderDevice,
  _cmd: crate::gpu::CommandBufferHandle,
  _pe: PresentationEngineHandle,
  _render_scene: &crate::gpu::RenderScene,
  _user: *mut core::ffi::c_void,
) -> crate::types::GpuResult<()> {
  Ok(())
}

/// Records every frame of `scene_id` in [`PROBES`], by engine (engine handles are unique in the
/// process, so tests running side by side do not mix).
fn install_probe(ctx: &SimulationContext, scene_id: u64) {
  let scene = ctx.get_scene(scene_id).expect("scene");
  scene.write().register_custom_render_callback(Some(CustomRenderCallback {
    after_render_frame_fn: probe_after_frame,
    on_first_render_fn: probe_first_frame,
    user_data: SendPtrMut(core::ptr::null_mut()),
  }));
}

fn probe(pe: PresentationEngineHandle) -> ViewProbe {
  PROBES.lock().unwrap_or_else(|e| e.into_inner()).get(&pe.0).copied().unwrap_or_default()
}

/// Whether the device holds per-view dust state for `pe`.
fn has_dust_view(ctx: &SimulationContext, pe: PresentationEngineHandle) -> bool {
  ctx
    .with_device(|device| {
      let vulkan_device: &crate::gpu_backends::vulkan::device::Device =
        device.as_any().downcast_ref().ok_or(crate::gpu_err!("not a Vulkan device"))?;
      Ok(vulkan_device.dust_view_params(pe).is_some())
    })
    .unwrap_or(false)
}

/// The committed 67P scene of the test save with its −10° jet ignited one lifetime before the
/// committed start, the dust history built at the committed start (one seek away and back, as
/// the app does). Returns `(scene_id, comet body)`.
fn dust_scene(ctx: &SimulationContext) -> (u64, EntityId) {
  let (scene_id, comet) = committed_scene(ctx);
  let start = committed_start();
  let ignition = emission_start_us_from_epoch(start - Duration::from_days(30.0));
  add_particle_system(ctx, scene_id, &save_dto(&SAVE_JETS[2], ignition))
    .unwrap_or_else(|e| panic!("jet refused: {e}"));
  assert!(ctx.seek_epoch_sync(scene_id, start + Duration::from_days(1.0)), "seek start + 1 d");
  wait_caught_up(ctx, scene_id);
  assert!(ctx.seek_epoch_sync(scene_id, start), "seek start");
  let tiers = wait_caught_up(ctx, scene_id);
  assert!(
    tiers.iter().any(|t| t.live_clusters > 0),
    "the scene has no dust at the committed start: {tiers:?}"
  );
  (scene_id, comet)
}

/// Places `cam` at `comet + rel_au` looking at the comet, screen up along `up`.
fn look_at_comet(
  ctx: &SimulationContext,
  scene_id: u64,
  cam: EntityId,
  comet: EntityId,
  rel_au: [f64; 3],
  up: [f64; 3],
) {
  use aethervk_oshal_rlib::math::vector::{vec3f64::Vec3f64, vec4f64::Quat64};
  let scene = ctx.get_scene(scene_id).expect("scene");
  let g = scene.write();
  let c: [f64; 3] = g.scene.global_transform_f64(comet).expect("comet transform").position.into();
  let q = crate::simulation_api::earth_observer::look_at_origin_from(rel_au, Some(up));
  assert!(
    g.scene
      .set_global_transform_f64(
        cam,
        Vec3f64::from_array([c[0] + rel_au[0], c[1] + rel_au[1], c[2] + rel_au[2]]),
        Quat64::from_components(q[0], q[1], q[2], q[3]),
      )
      .is_ok(),
    "camera pose"
  );
  g.mark_component_changed(
    cam.as_ffi(),
    <crate::scene::HighResTransformComponent as crate::scene::ForeignSerializable>::COMPONENT_ID,
  );
}

const AU_KM: f64 = 149_597_870.7;
/// the dust view: 0.02 AU from the nucleus, a 4° field (the 30 d tail fits several times)
const DUST_VIEW_DISTANCE_AU: f64 = 0.02;
const DUST_VIEW_FOV_RAD: f32 = 0.0698;

/// The dust view's offset from the comet at step `k`: a slow orbit in the world X-Z plane
/// (0.05° a step), so that no frame repeats the previous one's inputs.
fn dust_view_offset(k: u32) -> [f64; 3] {
  let a = (k as f64 * 0.05).to_radians();
  [DUST_VIEW_DISTANCE_AU * a.sin(), 0.0, DUST_VIEW_DISTANCE_AU * a.cos()]
}

// ─────────────────────────────────────────────────────────────────────────────
// two engines
// ─────────────────────────────────────────────────────────────────────────────

/// Two windowless engines of one scene are both rendered by the render thread (their presented
/// frame counts advance together); removing one mid-run leaves the other rendering, and an engine
/// added afterwards renders too.
#[test]
fn two_presentation_engines_render_concurrently_through_the_render_thread() {
  let Some(ctx) = startup() else {
    return;
  };
  let scene_id = empty_scene(&ctx);
  assert!(ctx.threads.render_thread_running());
  let (pe_a, _) = add_view(&ctx, scene_id, "Camera A", (64, 64), (1.0, 0.1, 1000.0), |_| {});
  let (pe_b, _) = add_view(&ctx, scene_id, "Camera B", (96, 48), (1.0, 0.1, 1000.0), |_| {});
  assert_ne!(pe_a, pe_b);
  play(&ctx, scene_id);
  let count = |pe| ctx.viewport_frame_count(pe);

  assert!(
    wait_until(60.0, || count(pe_a) >= 10 && count(pe_b) >= 10),
    "both engines must present frames: A {} B {}",
    count(pe_a),
    count(pe_b)
  );
  // over one interval, both
  let (a0, b0) = (count(pe_a), count(pe_b));
  assert!(
    wait_until(60.0, || count(pe_a) >= a0 + 10 && count(pe_b) >= b0 + 10),
    "both engines must keep presenting: A {} → {}, B {} → {}",
    a0,
    count(pe_a),
    b0,
    count(pe_b)
  );

  // one goes mid-run: the other keeps rendering
  ctx.destroy_presentation_engine(scene_id, pe_b).expect("destroy B");
  assert_eq!(count(pe_b), 0, "a removed engine has no frame count");
  assert!(!ctx.set_viewport_active(scene_id, pe_b, true), "a removed engine cannot be resumed");
  let a1 = count(pe_a);
  assert!(
    wait_until(60.0, || count(pe_a) >= a1 + 10),
    "A must keep presenting after B was removed: {} → {}",
    a1,
    count(pe_a)
  );

  // and one added mid-run joins in
  let (pe_c, _) = add_view(&ctx, scene_id, "Camera C", (48, 48), (1.0, 0.1, 1000.0), |_| {});
  let a2 = count(pe_a);
  assert!(
    wait_until(60.0, || count(pe_c) >= 10 && count(pe_a) >= a2 + 10),
    "an engine added mid-run must present beside A: C {}, A {} → {}",
    count(pe_c),
    a2,
    count(pe_a)
  );
  shutdown(&ctx);
}

/// Every frame of the scene costs the render thread this long (a render thread slower than the
/// logic thread's 60 Hz display slots).
const SLOW_FRAME_MS: u64 = 80;

fn slow_after_frame(
  _device: &dyn crate::gpu::RenderDevice,
  _cmd: crate::gpu::CommandBufferHandle,
  _pe: PresentationEngineHandle,
  _render_scene: &crate::gpu::RenderScene,
  _user: *mut core::ffi::c_void,
) -> crate::types::GpuResult<()> {
  std::thread::sleep(StdDuration::from_millis(SLOW_FRAME_MS));
  Ok(())
}

/// A render thread slower than the display rate never falls behind the logic thread by more than
/// `MAX_RENDER_BATCHES_IN_FLIGHT` frame batches, so a view added mid-run renders its first frame
/// within a few frames, not after a backlog of old batches (it used to wait for up to the
/// channel's 128 batches: about 10 s at this frame cost).
#[test]
fn slow_render_thread_keeps_no_backlog_and_a_new_view_renders_at_once() {
  use super::structs::MAX_RENDER_BATCHES_IN_FLIGHT;
  let Some(ctx) = startup() else {
    return;
  };
  let scene_id = empty_scene(&ctx);
  let (pe_a, _) = add_view(&ctx, scene_id, "Camera A", (64, 64), (1.0, 0.1, 1000.0), |_| {});
  ctx.get_scene(scene_id).expect("scene").write().register_custom_render_callback(Some(
    CustomRenderCallback {
      after_render_frame_fn: slow_after_frame,
      on_first_render_fn: probe_first_frame,
      user_data: SendPtrMut(core::ptr::null_mut()),
    },
  ));
  play(&ctx, scene_id);
  let count = |pe| ctx.viewport_frame_count(pe);
  assert!(wait_until(60.0, || count(pe_a) >= 3), "A must present frames: {}", count(pe_a));

  // 3 s of slow frames: the batches sent and not finished stay bounded
  let in_flight = || ctx.threads.render_batches_in_flight.load(core::sync::atomic::Ordering::Acquire);
  let mut most = 0;
  let t0 = Instant::now();
  while t0.elapsed() < StdDuration::from_secs(3) {
    most = most.max(in_flight());
    std::thread::sleep(StdDuration::from_millis(2));
  }
  assert!(
    most <= MAX_RENDER_BATCHES_IN_FLIGHT,
    "at most {MAX_RENDER_BATCHES_IN_FLIGHT} frame batches in flight, saw {most}"
  );

  // a view added now gets its first frame within a few slow frames
  let (pe_b, _) = add_view(&ctx, scene_id, "Camera B", (48, 48), (1.0, 0.1, 1000.0), |_| {});
  let t1 = Instant::now();
  assert!(wait_until(30.0, || count(pe_b) >= 1), "B never presented a frame");
  let first_ms = t1.elapsed().as_millis() as u64;
  println!("a view added beside a slow one presented its first frame after {first_ms} ms");
  // one batch being rendered and one queued, each of two slow frames, plus scheduling slack
  let bound_ms = (MAX_RENDER_BATCHES_IN_FLIGHT as u64 + 2) * 2 * SLOW_FRAME_MS + 500;
  assert!(first_ms <= bound_ms, "B's first frame came after {first_ms} ms (bound {bound_ms} ms)");
  let a0 = count(pe_a);
  assert!(wait_until(30.0, || count(pe_a) >= a0 + 3), "A must keep presenting beside B");
  shutdown(&ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// pause
// ─────────────────────────────────────────────────────────────────────────────

/// A paused engine gets no frame: the render thread records none and presents none for it while
/// the scene's other engine keeps rendering; resumed, it renders again. Pausing an unknown engine
/// or scene is refused.
#[test]
fn paused_viewport_builds_no_frames_and_resumes() {
  let Some(ctx) = startup() else {
    return;
  };
  let scene_id = empty_scene(&ctx);
  install_probe(&ctx, scene_id);
  let (pe_a, _) = add_view(&ctx, scene_id, "Camera A", (64, 64), (1.0, 0.1, 1000.0), |_| {});
  let (pe_b, _) = add_view(&ctx, scene_id, "Camera B", (64, 64), (1.0, 0.1, 1000.0), |_| {});
  play(&ctx, scene_id);
  let count = |pe| ctx.viewport_frame_count(pe);
  assert!(wait_until(60.0, || count(pe_a) >= 5 && count(pe_b) >= 5), "both engines must render first");
  assert_eq!(ctx.viewport_active(scene_id, pe_b), Some(true));

  assert!(!ctx.set_viewport_active(scene_id, PresentationEngineHandle(u64::MAX - 1), false));
  assert!(!ctx.set_viewport_active(scene_id + 999, pe_b, false));
  assert_eq!(ctx.viewport_active(scene_id, PresentationEngineHandle(u64::MAX - 1)), None);

  for round in 0..3 {
    assert!(ctx.set_viewport_active(scene_id, pe_b, false));
    assert!(ctx.set_viewport_active(scene_id, pe_b, false), "pausing twice is a pause");
    assert_eq!(ctx.viewport_active(scene_id, pe_b), Some(false));
    // a frame being recorded when the pause landed completes; the queued ones are dropped
    std::thread::sleep(StdDuration::from_millis(500));
    let (b_presented, b_recorded) = (count(pe_b), probe(pe_b).frames);
    let a0 = count(pe_a);
    assert!(
      wait_until(60.0, || count(pe_a) >= a0 + 20),
      "round {round}: A must keep rendering while B is paused: {} → {}",
      a0,
      count(pe_a)
    );
    assert_eq!(count(pe_b), b_presented, "round {round}: a paused engine presented a frame");
    assert_eq!(probe(pe_b).frames, b_recorded, "round {round}: a paused engine recorded a frame");

    assert!(ctx.set_viewport_active(scene_id, pe_b, true));
    assert_eq!(ctx.viewport_active(scene_id, pe_b), Some(true));
    let a1 = count(pe_a);
    assert!(
      wait_until(60.0, || count(pe_b) >= b_presented + 10 && count(pe_a) >= a1 + 10),
      "round {round}: B must render again once resumed, beside A: B {} → {}, A {} → {}",
      b_presented,
      count(pe_b),
      a1,
      count(pe_a)
    );
    assert!(probe(pe_b).frames >= b_recorded + 10, "round {round}: resumed frames were not recorded");
  }
  shutdown(&ctx);
}

/// `PresentationEngineData::take_frame_slot`: without an interval every slot is due; with one, a
/// slot sooner than the interval after the last due slot is skipped (and does not move it), the
/// first slot after it is due again; clearing the interval makes every slot due.
#[test]
fn frame_slot_follows_the_min_interval() {
  let d = super::structs::PresentationEngineData::new(true);
  assert!(d.take_frame_slot(1_000) && d.take_frame_slot(1_016), "no interval: every slot");
  d.min_frame_interval_us.store(100_000, core::sync::atomic::Ordering::Release);
  assert!(!d.take_frame_slot(1_016 + 99_999), "sooner than the interval after the last frame");
  assert!(!d.take_frame_slot(1_016 + 50_000), "a skipped slot does not move the last frame");
  assert!(d.take_frame_slot(1_016 + 100_000), "due at the interval");
  assert!(!d.take_frame_slot(1_016 + 100_016));
  d.min_frame_interval_us.store(0, core::sync::atomic::Ordering::Release);
  assert!(d.take_frame_slot(1_016 + 100_032), "cleared: every slot");
  // a fresh view's first slot is always due, whatever the clock
  let e = super::structs::PresentationEngineData::new(true);
  e.min_frame_interval_us.store(100_000, core::sync::atomic::Ordering::Release);
  assert!(e.take_frame_slot(0) && !e.take_frame_slot(50_000) && e.take_frame_slot(100_001));
}

/// `SimulationContext::set_viewport_min_frame_interval`: a view capped at 1 s presents at most
/// one frame a second (counted over 4 s, ≤ 4 + 1) while the uncapped view of the same scene keeps
/// its rate (more than twice the capped view's frames over the same 4 s); unknown engines and
/// scenes are refused; clearing the cap brings the view back to the other's pace. The cap is 1 s,
/// not the mini-view's 100 ms: windowless views in this harness render at only a few frames a
/// second, below a 10 fps cap.
#[test]
fn capped_viewport_renders_at_its_interval_beside_an_uncapped_one() {
  let Some(ctx) = startup() else {
    return;
  };
  let scene_id = empty_scene(&ctx);
  install_probe(&ctx, scene_id);
  let (pe_a, _) = add_view(&ctx, scene_id, "Camera A", (64, 64), (1.0, 0.1, 1000.0), |_| {});
  let (pe_b, _) = add_view(&ctx, scene_id, "Camera B", (64, 64), (1.0, 0.1, 1000.0), |_| {});
  play(&ctx, scene_id);
  let count = |pe| ctx.viewport_frame_count(pe);
  assert!(wait_until(60.0, || count(pe_a) >= 5 && count(pe_b) >= 5), "both engines must render first");

  assert!(!ctx.set_viewport_min_frame_interval(scene_id, PresentationEngineHandle(u64::MAX - 1), 100_000));
  assert!(!ctx.set_viewport_min_frame_interval(scene_id + 999, pe_b, 100_000));
  assert!(ctx.set_viewport_min_frame_interval(scene_id, pe_b, 1_000_000));
  std::thread::sleep(StdDuration::from_millis(500)); // frames queued before the cap drain
  let (a0, b0) = (count(pe_a), count(pe_b));
  std::thread::sleep(StdDuration::from_secs(4));
  let (da, db) = (count(pe_a) - a0, count(pe_b) - b0);
  assert!(db >= 2 && db <= 5, "the capped view presented {db} frames in 4 s (expected ~4, at most 5)");
  assert!(da > 2 * db, "the uncapped view must keep its rate: {da} frames vs the capped view's {db}");

  assert!(ctx.set_viewport_min_frame_interval(scene_id, pe_b, 0));
  std::thread::sleep(StdDuration::from_millis(300));
  let (a1, b1) = (count(pe_a), count(pe_b));
  std::thread::sleep(StdDuration::from_secs(4));
  let (da, db) = (count(pe_a) - a1, count(pe_b) - b1);
  assert!(db * 10 >= da * 8, "uncapped again, B must keep A's pace: B {db}, A {da} frames in 4 s");
  shutdown(&ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// no UI layer
// ─────────────────────────────────────────────────────────────────────────────

/// A camera with `CAMERA_VIEW_NO_UI_LAYER` uploads neither the UI batch nor the text batch and
/// its frames are drawn with the UI layer off, while the other camera of the same scene, in the
/// same frames, uploads and draws the scene's UI. Clearing the flag brings the UI back.
#[test]
fn no_ui_layer_camera_draws_no_labels() {
  use crate::scene::ui::{Transform2DComponent, UiComponent};
  let Some(ctx) = startup() else {
    return;
  };
  let scene_id = empty_scene(&ctx);
  // a UI element of the scene itself: every view with the UI layer uploads it
  {
    let scene = ctx.get_scene(scene_id).unwrap();
    let g = scene.read();
    let panel = g.scene.spawn_entity("test panel");
    let mut t2d = Transform2DComponent::default();
    t2d.local_position = [8.0, 8.0];
    t2d.size = [24.0, 16.0];
    g.scene.add_component(panel, t2d).unwrap();
    g.scene.add_component(panel, UiComponent::default()).unwrap();
  }
  install_probe(&ctx, scene_id);
  let (pe_ui, _) = add_view(&ctx, scene_id, "Camera UI", (96, 96), (1.0, 0.1, 1000.0), |_| {});
  let mut bare = None;
  let (pe_bare, _) = add_view(&ctx, scene_id, "Camera bare", (96, 96), (1.0, 0.1, 1000.0), |cam| {
    assert!(ctx.set_camera_view_flags(scene_id, cam.as_ffi(), CAMERA_VIEW_NO_UI_LAYER, true));
    bare = Some(cam);
  });
  let bare = bare.unwrap();
  // not a camera / not a scene
  let root = ctx.get_scene(scene_id).unwrap().read().root_entity;
  assert!(!ctx.set_camera_view_flags(scene_id, root.as_ffi(), CAMERA_VIEW_NO_UI_LAYER, true));
  assert!(!ctx.set_camera_view_flags(scene_id + 999, bare.as_ffi(), CAMERA_VIEW_NO_UI_LAYER, true));
  play(&ctx, scene_id);

  assert!(
    wait_until(60.0, || probe(pe_ui).frames >= 20 && probe(pe_bare).frames >= 20),
    "both views must render: {:?} {:?}",
    probe(pe_ui),
    probe(pe_bare)
  );
  let (with_ui, without) = (probe(pe_ui), probe(pe_bare));
  assert!(with_ui.ui_batches > 0, "the plain view must upload the scene's UI: {with_ui:?}");
  assert_eq!(with_ui.ui_hidden, 0, "the plain view draws its UI layer: {with_ui:?}");
  assert_eq!(without.ui_batches, 0, "a camera without the UI layer uploaded a UI batch: {without:?}");
  assert_eq!(without.text_batches, 0, "a camera without the UI layer uploaded a text batch: {without:?}");
  assert_eq!(without.ui_hidden, without.frames, "its frames must be drawn with the UI layer off: {without:?}");

  // the flag is the cause: cleared, the same camera uploads the UI again
  assert!(ctx.set_camera_view_flags(scene_id, bare.as_ffi(), CAMERA_VIEW_NO_UI_LAYER, false));
  assert!(
    wait_until(60.0, || probe(pe_bare).ui_batches >= 10),
    "the UI must come back once the flag is cleared: {:?}",
    probe(pe_bare)
  );
  shutdown(&ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// no dust
// ─────────────────────────────────────────────────────────────────────────────

/// Two cameras look at the same comet from the same place; one has `CAMERA_VIEW_NO_DUST`. Its
/// engine never gets a dust draw call, a dust pyramid, a dust view on the device or an entry in
/// the dust frame ledger, while the other engine has all of them. Clearing the flag allocates the
/// view: the flag is the cause.
#[test]
fn camera_without_dust_allocates_no_dust_view() {
  let spk = aethervk_test_assets::test_asset!(SPK_COMMITTED).to_str().unwrap().to_string();
  let Some(ctx) = startup() else {
    return;
  };
  load_almanac_files(&ctx, &spk);
  let (scene_id, comet) = dust_scene(&ctx);
  install_probe(&ctx, scene_id);
  let rel = dust_view_offset(0);
  let up = [0.0, 1.0, 0.0];
  let lens = (DUST_VIEW_FOV_RAD, 1e-5, 20.0);
  let (pe_dust, _) = add_view(&ctx, scene_id, "dust view", (160, 120), lens, |cam| {
    look_at_comet(&ctx, scene_id, cam, comet, rel, up);
  });
  let mut bare = None;
  let (pe_bare, _) = add_view(&ctx, scene_id, "no-dust view", (160, 120), lens, |cam| {
    assert!(ctx.set_camera_view_flags(scene_id, cam.as_ffi(), CAMERA_VIEW_NO_DUST, true));
    look_at_comet(&ctx, scene_id, cam, comet, rel, up);
    bare = Some(cam);
  });
  let bare = bare.unwrap();

  assert!(
    wait_until(180.0, || probe(pe_dust).dust_pyramids >= 3 && probe(pe_bare).frames >= 10),
    "the dust view must draw dust and the other view must render: {:?} {:?}",
    probe(pe_dust),
    probe(pe_bare)
  );
  let (dust, none) = (probe(pe_dust), probe(pe_bare));
  println!("dust view {dust:?}\nno-dust view {none:?}");
  assert!(dust.dust_calls > 0 && dust.no_dust == 0, "the plain view must have dust calls: {dust:?}");
  assert!(has_dust_view(&ctx, pe_dust), "the plain view must have a dust view on the device");
  assert!(!ctx.dust_frame_ledger(pe_dust, 0).is_empty(), "the plain view must be in the dust ledger");

  assert_eq!(none.dust_calls, 0, "a camera without dust got dust draw calls: {none:?}");
  assert_eq!(none.dust_pyramids, 0, "a camera without dust got a dust pyramid: {none:?}");
  assert_eq!(none.no_dust, none.frames, "every frame of it must be flagged: {none:?}");
  assert!(!has_dust_view(&ctx, pe_bare), "a camera without dust allocated a dust view");
  assert!(
    ctx.dust_frame_ledger(pe_bare, 0).is_empty(),
    "a camera without dust recorded dust frames: {:?}",
    ctx.dust_frame_ledger(pe_bare, 0)
  );

  // the same camera with the flag cleared: now it has one
  assert!(ctx.set_camera_view_flags(scene_id, bare.as_ffi(), CAMERA_VIEW_NO_DUST, false));
  assert!(
    wait_until(180.0, || has_dust_view(&ctx, pe_bare) && probe(pe_bare).dust_pyramids >= 1),
    "the dust view must be allocated once the flag is cleared: {:?}",
    probe(pe_bare)
  );
  shutdown(&ctx);
}

// ─────────────────────────────────────────────────────────────────────────────
// the dust view's bytes beside a second view
// ─────────────────────────────────────────────────────────────────────────────

/// `(task id, deliveries)` of the last frame delivered per windowless engine
/// (`SimulationContext::set_render_callback`; the task id is `u64::MAX` for a failed frame).
static DELIVERED: Mutex<BTreeMap<u64, (u64, u64)>> = Mutex::new(BTreeMap::new());

extern "C" fn on_render(_scene: u64, pe: u64, task: u64) {
  let mut delivered = DELIVERED.lock().unwrap_or_else(|e| e.into_inner());
  let entry = delivered.entry(pe).or_insert((0, 0));
  *entry = (task, entry.1 + 1);
}

fn delivered(pe: PresentationEngineHandle) -> (u64, u64) {
  DELIVERED.lock().unwrap_or_else(|e| e.into_inner()).get(&pe.0).copied().unwrap_or((0, 0))
}

/// Releases exactly one frame of the held engine `pe` and returns its image (the engine's native
/// 4 bytes a pixel), once the GPU has finished it.
fn release_one_frame(
  ctx: &SimulationContext,
  scene_id: u64,
  pe: PresentationEngineHandle,
  (width, height): (u32, u32),
  what: &str,
) -> Vec<u8> {
  let (_, before) = delivered(pe);
  assert!(ctx.set_render_budget(scene_id, pe, Some(1)));
  assert!(
    wait_until(600.0, || delivered(pe).1 > before),
    "{what}: the released frame never arrived"
  );
  let (task, after) = delivered(pe);
  assert_eq!(after, before + 1, "{what}: one frame was released, {} arrived", after - before);
  assert!(task != 0 && task != u64::MAX, "{what}: the released frame failed");
  assert_eq!(ctx.render_budget(scene_id, pe), Some(0), "{what}: the view must be held again");
  assert!(
    wait_until(120.0, || !matches!(ctx.get_task_status(task), TaskStatusCode::Pending)),
    "{what}: the frame's task never completed"
  );
  let mut image = vec![0u8; (width * height * 4) as usize];
  assert!(
    unsafe { ctx.download_image(task, image.as_mut_ptr(), image.len()) },
    "{what}: download failed"
  );
  image
}

/// What stands beside the dust view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SecondView {
  /// nothing: the dust view alone
  None,
  /// a no-dust, no-UI nucleus view rendering every frame
  Running,
  /// the same view, paused before the dust view's first frame
  Paused,
}

const MAIN_SIZE: (u32, u32) = (320, 240);
const MINI_SIZE: (u32, u32) = (96, 96);
/// frames at the first pose: the cold frame and the ones the work feedback settles over
const SETTLE_FRAMES: u32 = 12;
/// frames along the orbit after them
const MOVING_FRAMES: u32 = 32;

struct DustViewRun {
  /// the dust view's images, `SETTLE_FRAMES` at the first pose then `MOVING_FRAMES` along the orbit
  frames: Vec<Vec<u8>>,
  /// dust frames of the dust view that were drawn (not reused)
  drawn: usize,
}

/// One fresh context: the dust scene, the dust view (held, released one frame at a time) and the
/// second view `second`; the dust view's images over the whole sequence.
fn run_dust_view(spk: &str, second: SecondView) -> Option<DustViewRun> {
  let ctx = startup()?;
  load_almanac_files(&ctx, spk);
  let (scene_id, comet) = dust_scene(&ctx);
  let up = [0.0, 1.0, 0.0];

  // the dust view: created held, never released except by `release_one_frame`. Without the sky
  // and the orbit lines: the star field is turned by a parallax filter that integrates the
  // wall-clock frame time, which no two runs share.
  let main_pe = ctx
    .create_presentation_engine(scene_id, MAIN_SIZE.0, MAIN_SIZE.1)
    .expect("dust view engine");
  assert!(ctx.set_render_budget(scene_id, main_pe, Some(0)));
  let main_cam = ctx
    .add_perspective_camera(scene_id, main_pe, "dust view", DUST_VIEW_FOV_RAD, 1e-5, 20.0)
    .expect("dust view camera");
  let main_cam = EntityId::from_ffi(main_cam.get());
  assert!(ctx.set_camera_render_mask(scene_id, main_cam.as_ffi(), RENDER_LAYER_DEFAULT));
  look_at_comet(&ctx, scene_id, main_cam, comet, dust_view_offset(0), up);

  // the second view: 100 km from the nucleus, no dust, no UI layer
  let mini = (second != SecondView::None).then(|| {
    let (pe, _) = add_view(&ctx, scene_id, "nucleus view", MINI_SIZE, (0.5, 1e-9, 1.0), |cam| {
      assert!(ctx.set_camera_view_flags(
        scene_id,
        cam.as_ffi(),
        CAMERA_VIEW_NO_DUST | CAMERA_VIEW_NO_UI_LAYER,
        true
      ));
      look_at_comet(&ctx, scene_id, cam, comet, [0.0, 100.0 / AU_KM, 0.0], [0.0, 0.0, 1.0]);
    });
    assert!(
      wait_until(120.0, || ctx.viewport_frame_count(pe) >= 5),
      "{second:?}: the second view must render before the dust view starts"
    );
    pe
  });
  let mini_at_start = match (second, mini) {
    (SecondView::Paused, Some(pe)) => {
      assert!(ctx.set_viewport_active(scene_id, pe, false));
      std::thread::sleep(StdDuration::from_millis(500));
      ctx.viewport_frame_count(pe)
    }
    (_, Some(pe)) => ctx.viewport_frame_count(pe),
    _ => 0,
  };

  // every variant starts its dust frames settled: the seek's preview hold (wall clock) keeps the
  // play quality on for a while, and a run whose first frame falls inside it draws that frame
  // pinned to the play set; a run that waited for a second view would not
  let scene_arc = ctx.scenes.read().get_scene(scene_id).expect("dust scene");
  assert!(
    wait_until(120.0, || !scene_arc.read().dust_in_preview()),
    "{second:?}: the seek's preview hold never ended"
  );
  drop(scene_arc);

  let mut frames = Vec::with_capacity((SETTLE_FRAMES + MOVING_FRAMES) as usize);
  for k in 0..SETTLE_FRAMES + MOVING_FRAMES {
    let step = k.saturating_sub(SETTLE_FRAMES - 1);
    look_at_comet(&ctx, scene_id, main_cam, comet, dust_view_offset(step), up);
    frames.push(release_one_frame(
      &ctx,
      scene_id,
      main_pe,
      MAIN_SIZE,
      &format!("{second:?}, frame {k}"),
    ));
  }

  let ledger = ctx.dust_frame_ledger(main_pe, 0);
  let drawn = ledger.iter().filter(|e| !e.reused).count();
  assert!(has_dust_view(&ctx, main_pe), "{second:?}: the dust view has no dust view on the device");
  if let Some(pe) = mini {
    let now = ctx.viewport_frame_count(pe);
    match second {
      SecondView::Running => assert!(
        now >= mini_at_start + 8,
        "{second:?}: the second view must have rendered beside the dust view ({mini_at_start} → {now})"
      ),
      _ => assert_eq!(now, mini_at_start, "{second:?}: the paused second view rendered"),
    }
    assert!(!has_dust_view(&ctx, pe), "{second:?}: the second view allocated a dust view");
    assert!(ctx.dust_frame_ledger(pe, 0).is_empty(), "{second:?}: the second view recorded dust frames");
  }
  shutdown(&ctx);
  Some(DustViewRun { frames, drawn })
}

/// `(differing bytes, largest difference)` of two images of the same size.
fn byte_difference(a: &[u8], b: &[u8]) -> (usize, u8) {
  assert_eq!(a.len(), b.len());
  a.iter().zip(b).fold((0, 0), |(n, max), (x, y)| {
    let d = x.abs_diff(*y);
    (n + (d != 0) as usize, max.max(d))
  })
}

/// The dust view's read-back image, frame by frame (12 at the first pose from a cold start, then
/// 32 along a slow orbit), is byte-identical in three set-ups: alone, beside a running no-dust
/// view, beside that view paused. Each set-up runs in a fresh context on the same scene, epoch and
/// camera path, the dust view released one frame at a time.
///
/// The reference run is made twice. A frame the two reference runs disagree on is a frame this
/// harness cannot reproduce by itself (a cold frame's in-frame work gates depend on the dispatch
/// order): it is reported and left out. Every frame along the orbit must be reproducible, and on
/// every reproducible frame the two set-ups with a second view must match the reference exactly.
///
/// `AETHERVK_DUST_REFINE=0`: the refinement of a static view is paced by the measured frame time.
#[test]
fn main_view_dust_bytes_identical_with_and_without_a_mini_view() {
  let spk = aethervk_test_assets::test_asset!(SPK_COMMITTED).to_str().unwrap().to_string();
  let _guard = crate::simulation_api::lock_render_callback_tests();
  // read by the engine per frame; the test owns its process under nextest
  unsafe { std::env::set_var("AETHERVK_DUST_REFINE", "0") };
  SimulationContext::set_render_callback(Some(on_render));
  struct ClearCallback;
  impl Drop for ClearCallback {
    fn drop(&mut self) {
      SimulationContext::set_render_callback(None);
    }
  }
  let _clear = ClearCallback;

  let Some(alone) = run_dust_view(&spk, SecondView::None) else {
    return;
  };
  let again = run_dust_view(&spk, SecondView::None).expect("the device of the first run");
  let running = run_dust_view(&spk, SecondView::Running).expect("the device of the first run");
  let paused = run_dust_view(&spk, SecondView::Paused).expect("the device of the first run");

  let total = (SETTLE_FRAMES + MOVING_FRAMES) as usize;
  for run in [&alone, &again, &running, &paused] {
    assert_eq!(run.frames.len(), total);
  }
  println!(
    "dust frames drawn: alone {}, alone again {}, running second view {}, paused second view {}",
    alone.drawn, again.drawn, running.drawn, paused.drawn
  );
  assert!(alone.drawn > 0, "the dust view never drew dust: the comparison would be empty");
  // the sequence shows something that changes: the orbit's frames are not one repeated image
  assert!(
    alone.frames[total - 1] != alone.frames[SETTLE_FRAMES as usize - 1],
    "the dust view's image never changed along the orbit"
  );

  let mut reproducible = 0usize;
  let mut problems: Vec<String> = Vec::new();
  for k in 0..total {
    let moving = k >= SETTLE_FRAMES as usize;
    let (n, max) = byte_difference(&alone.frames[k], &again.frames[k]);
    if n != 0 {
      println!(
        "frame {k} ({}): the two runs without a second view differ in {n} bytes (max {max}): not compared",
        if moving { "orbit" } else { "first pose" }
      );
      if moving {
        problems.push(format!(
          "frame {k} (orbit): not reproducible without a second view ({n} bytes, max {max}): the harness, not the second view"
        ));
      }
      continue;
    }
    reproducible += 1;
    for (name, run) in [("a running second view", &running), ("a paused second view", &paused)] {
      let (n, max) = byte_difference(&alone.frames[k], &run.frames[k]);
      if n != 0 {
        problems.push(format!(
          "frame {k}: the dust view beside {name} differs in {n} bytes (max {max})"
        ));
      }
    }
  }
  println!("{reproducible} of {total} frames reproducible without a second view and compared");
  assert!(problems.is_empty(), "{}", problems.join("\n"));
  assert!(
    reproducible >= MOVING_FRAMES as usize,
    "only {reproducible} frames were compared, at least {MOVING_FRAMES} are required"
  );
}
