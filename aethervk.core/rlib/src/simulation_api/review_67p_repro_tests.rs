//! Headless reproductions of the reviewer's 67P save (`test_assets/saves/review_67p_2021-12`, the
//! reviewer's mail of 2026-10-29): the SPK coverage, the four jets (the −67° one never reached the
//! simulation), the sunlight of each jet site and the dust age at the committed start.
//!
//! The facts below are copied from `aethervk_index.json` (the committed timeline, the committed
//! SPK's rotation, the model and the `jets` list, in save order).

use crate::{
  scene::{
    AlmanacPlanet, BodyRotationalModel, JetDirectionLineComponent, ParticleSystemDTO,
    TransformComponent,
    dust::{DustTierStats, JetState},
    particles::{
      EMISSION_START_NOW, ParticleSystemComponent, ParticleSystemDrawParams,
      ParticleSystemEmitParams, emission_start_us_from_epoch, resolve_emission_start_us,
    },
  },
  simulation::almanac::AlmanacPackedData,
  simulation_api::{
    SimulationContext,
    structs::{KeplerianElements, LogicCommand, ReferenceOrbitMode, SceneEntityId},
  },
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
};
use anise::prelude::NAIFSummaryRecord;
use hifitime::{Duration, Epoch, TimeScale};

const SPK_COMMITTED: &str =
  "../../test_assets/saves/review_67p_2021-12/spk/spk_67P_90000703_2021-12-12_2021-12-28.bsp";
const SPK_M30: &str =
  "../../test_assets/saves/review_67p_2021-12/spk/spk_m30_90000703_2021-12-15_2025-11-11.bsp";
const NAIF_67P: i32 = 1000012;
/// `avkSimulationContext_addParticleSystem`'s `JET_TTL_US` (30 d)
const JET_TTL_US: i64 = 30 * 86_400 * 1_000_000i64;
/// `model.manual_nucleus_radius_km` (the SPK entry says 1.7)
const NUCLEUS_RADIUS_KM: f32 = 2.0;

/// committed range, TDB `(centuries, ns)` since J2000
fn committed_start() -> Epoch {
  Epoch::from_tdb_duration(Duration::from_parts(0, 692_539_200_000_000_000))
}
fn committed_end() -> Epoch {
  Epoch::from_tdb_duration(Duration::from_parts(0, 693_921_600_000_000_000))
}

/// the committed SPK entry's rotation (`spk[1].rotation`, `committed: true`)
fn save_rotation() -> BodyRotationalModel {
  BodyRotationalModel {
    pole_ra: 69.54,
    pole_dec: 64.11,
    prime_meridian: 0.0,
    pole_ra_rate: 0.0,
    pole_dec_rate: 0.0,
    rotation_rate: 696.2127316680096,
    body_fixed_orientation: true,
  }
}

/// the save's `sbdb` block (`reference_orbit_mode: osculating_at_start` re-osculates it)
fn save_elements() -> KeplerianElements {
  KeplerianElements {
    eccentricity: 0.6409081308996354,
    perihelion_distance_au: 1.243265640702404,
    inclination_deg: 7.040294937543767,
    longitude_of_ascending_node_deg: 50.13557377155012,
    argument_of_perihelion_deg: 12.79824970228189,
    time_of_perihelion_jd_tdb: 2457247.588657812,
  }
}

/// One `jets[]` record of the save.
#[derive(Clone, Copy, Debug)]
struct SaveJet {
  lat_deg: f32,
  lon_deg: f32,
  aperture_deg: f32,
  v_mean: f32,
  v_std: f32,
  seed: u32,
  color: [f32; 4],
}

/// `jets[]` in save order
const SAVE_JETS: [SaveJet; 4] = [
  SaveJet {
    lat_deg: -54.0,
    lon_deg: 0.0,
    aperture_deg: 0.50000376,
    v_mean: 70.0,
    v_std: 6.0,
    seed: 12716195,
    color: [0.7019608, 0.4745098, 0.92156863, 1.0],
  },
  SaveJet {
    lat_deg: -67.0,
    lon_deg: 0.0,
    aperture_deg: 0.2,
    v_mean: 70.0,
    v_std: 0.30233926,
    seed: 633654374,
    color: [0.03529412, 0.43137255, 0.92156863, 1.0],
  },
  SaveJet {
    lat_deg: -10.0,
    lon_deg: 0.0,
    aperture_deg: 1.0,
    v_mean: 50.0,
    v_std: 0.7800244,
    seed: 1892684929,
    color: [0.84313726, 0.92156863, 0.14117648, 1.0],
  },
  SaveJet {
    lat_deg: 40.0,
    lon_deg: 0.0,
    aperture_deg: 0.2,
    v_mean: 8.0,
    v_std: 0.35233048,
    seed: 1273352477,
    color: [0.5882353, 0.92156863, 0.45490196, 1.0],
  },
];

/// The DTO the UI sends for `jet` (the model's grain / activity values copied into the record).
fn save_dto(jet: &SaveJet, emission_start_us: i64) -> ParticleSystemDTO {
  let mut dto = ParticleSystemDTO {
    diametre_um: 30.0,
    density_gcm3: 0.8,
    scattering_efficiency: 1.0,
    geometric_albedo: 0.04,
    size_q: 2.1,
    size_range: 10.0,
    dust_rate_1au_kgs: 0.0,
    dust_rate_power: 2.0,
    dust_rate_cutoff_au: 15.0,
    dust_rate_max_kgs: 0.0,
    latitude_rad: jet.lat_deg.to_radians(),
    longitude_rad: jet.lon_deg.to_radians(),
    aperture_rad: jet.aperture_deg.to_radians(),
    start_velocity_mean: jet.v_mean,
    start_velocity_std: jet.v_std,
    stream_color: jet.color,
    nucleus_radius_km: NUCLEUS_RADIUS_KM,
    seed: jet.seed,
    _pad: 0,
    emission_start_us,
  };
  // the activity the save held as an Afρ of 100.0 cm (max 100_000.0 cm), as a dust rate (v9)
  let p = crate::scene::particles::ParticleSystemEmitParams::from_dto(&dto);
  dto.dust_rate_1au_kgs = p.dust_rate_from_afrho_kgs(100.0);
  dto.dust_rate_max_kgs = p.dust_rate_from_afrho_kgs(100_000.0);
  dto
}

fn fmt_epoch(e: Epoch) -> String {
  format!(
    "{} UTC | {} TDB",
    e.to_time_scale(TimeScale::UTC),
    e.to_time_scale(TimeScale::TDB)
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. SPK coverage
// ─────────────────────────────────────────────────────────────────────────────

/// The real segment epochs of 67P in both SPKs of the save. Answers: does either file cover
/// 30 d before the committed start (2021-11-12)?
#[test]
fn review_67p_spk_segment_bounds() {
  let (start, end) = (committed_start(), committed_end());
  let before = start - Duration::from_days(30.0);
  println!("committed start {}", fmt_epoch(start));
  println!("committed end   {}", fmt_epoch(end));
  println!("start − 30 d    {}", fmt_epoch(before));
  let mut domains = Vec::new();
  for path in [SPK_COMMITTED, SPK_M30] {
    let bytes = bytes::Bytes::from(std::fs::read(path).expect(path));
    let mut almanac = anise::almanac::Almanac::default();
    almanac.load_from_bytes_mut(bytes, path).expect(path);
    let summaries = almanac.spk_summaries(NAIF_67P).expect("no 67P segment");
    println!("{path}: {} segment(s) of {NAIF_67P}", summaries.len());
    for (k, s) in summaries.iter().enumerate() {
      println!(
        "  segment {k}: centre {} frame {} type {}\n    start {}\n    end   {}",
        s.center_id,
        s.frame_id,
        s.data_type_i,
        fmt_epoch(s.start_epoch()),
        fmt_epoch(s.end_epoch())
      );
    }
    let (ds, de) = almanac.spk_domain(NAIF_67P).expect("no 67P domain");
    println!(
      "  domain: {} → {}\n  covers start − 30 d: {}",
      fmt_epoch(ds),
      fmt_epoch(de),
      ds <= before
    );
    // the engine's own probe (2-day tolerance), for the record
    let (covers, probe_dom, id) =
      AlmanacPackedData::probe_spk_file_with_domain(path, NAIF_67P, start, end);
    println!("  probe_spk_file_with_domain: covers {covers} domain {probe_dom:?} id {id}");
    domains.push((path, ds, de));
  }
  let (_, ds, de) = domains[0];
  assert!(
    ds <= start && de >= end,
    "the committed SPK does not cover the committed range: {} → {}",
    fmt_epoch(ds),
    fmt_epoch(de)
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// shared GPU scene recipe
// ─────────────────────────────────────────────────────────────────────────────

fn load_almanac_files(ctx: &SimulationContext) {
  let mut logic_state = ctx.logic_state.write();
  for f in [
    "../../assets/planets/pck00011.pca",
    "../../assets/planets/gm_de431.pca",
    "../../assets/planets/de442.bsp",
    SPK_COMMITTED,
  ] {
    logic_state.almanac_data.load_almanac(f).expect(f);
  }
}

/// The committed scene: comet from the committed SPK, the save's rotation on the body and in the
/// cartesian cache (as the template does). Waits for the comet commit by polling the scene (no
/// global external-state callback: it is shared with the other test modules). Returns
/// `(scene_id, comet_body)`.
fn committed_scene(ctx: &SimulationContext) -> (u64, crate::scene::EntityId) {
  let (start, end) = (committed_start(), committed_end());
  let scene_id = ctx
    .create_empty_scene2(false, start, end)
    .expect("Failed to create empty scene")
    .scene_id;
  ctx
    .threads
    .logic_thread
    .tx()
    .try_send(LogicCommand::TryInitComet {
      scene_id,
      spk_id: NAIF_67P,
      proposed_start: start,
      proposed_end: end,
      keplerian_elements: save_elements(),
      reference_mode: ReferenceOrbitMode::OsculatingAtStart,
    })
    .unwrap();
  let comet_body = {
    let scene_ctx = ctx.scenes.read().get(&scene_id).cloned().unwrap();
    let body = scene_ctx.read().comet.unwrap().body;
    body
  };
  // `BuildCometTrajectory` commits the AlmanacPlanet after the reference elements
  let t0 = std::time::Instant::now();
  loop {
    let done = {
      let scenes = ctx.scenes.read();
      let g = scenes.get_scene(scene_id).unwrap();
      let g = g.read();
      g.comet_reference_elements.is_some()
        && g.scene.with_component(comet_body, |_: &AlmanacPlanet| ()).is_some()
    };
    if done {
      break;
    }
    assert!(
      t0.elapsed().as_secs_f64() < 30.0,
      "TryInitComet never committed the comet (see the [TryInitComet] breadcrumbs)"
    );
    std::thread::sleep(std::time::Duration::from_millis(20));
  }
  // let the commit finish (reposition, observer re-aim)
  std::thread::sleep(std::time::Duration::from_millis(500));

  let model = save_rotation();
  {
    let scenes = ctx.scenes.read();
    let scene_arc = scenes.get_scene(scene_id).unwrap();
    let g = scene_arc.read();
    if g
      .scene
      .with_component_mut(comet_body, |m: &mut BodyRotationalModel| *m = model)
      .is_none()
    {
      let _ = g.scene.add_component(comet_body, model);
    }
    let key = SceneEntityId::new(scene_id, comet_body);
    if let Some(mut state) = scenes.cartesian_state_cache.get_mut(&key) {
      if let Some(ref mut cs) = state.comet_state {
        cs.body_rotational_model = Some(model);
      }
    }
  }
  (scene_id, comet_body)
}

/// `avkSimulationContext_addParticleSystem` (cdylib `ffi.rs`), line by line: the jet entity
/// under the comet body, its transform, the direction line, `from_dto`, the 30 d TTL and the
/// ignition resolved against the scene's clock. Returns the jet entity or the error verbatim.
fn add_particle_system(
  ctx: &SimulationContext,
  scene_id: u64,
  dto: &ParticleSystemDTO,
) -> Result<crate::scene::EntityId, String> {
  let scenes = ctx.scenes.read();
  let scene_arc = scenes.get_scene(scene_id).ok_or("scene not found")?;
  let scene_guard = scene_arc.read();
  let comet = scene_guard.comet.ok_or("comet not found")?;
  let (lat, lon) = (dto.latitude_rad, dto.longitude_rad);
  let parent_scale = scene_guard
    .scene
    .with_component(comet.body, |t: &TransformComponent| t.scale.x())
    .unwrap_or(1.0);
  let local_r = dto.nucleus_radius_km / parent_scale;
  let desired_global_radius = dto.nucleus_radius_km / 50.0;
  let local_scale = desired_global_radius / parent_scale;
  let pos = Vec3f32::from_components(
    local_r * lat.cos() * lon.cos(),
    local_r * lat.cos() * lon.sin(),
    local_r * lat.sin(),
  );
  let jet_entity = scene_guard.scene.spawn_entity("jet");
  scene_guard.scene.set_parent(jet_entity, Some(comet.body));
  let _ = scene_guard.scene.add_component(
    jet_entity,
    TransformComponent {
      position: pos,
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(local_scale, local_scale, local_scale),
    },
  );
  let _ = scene_guard.scene.add_component(
    jet_entity,
    JetDirectionLineComponent {
      color: dto.stream_color,
      is_visible: true,
      show_angles: false,
    },
  );
  let emit_params = ParticleSystemEmitParams::from_dto(dto);
  let draw_params = ParticleSystemDrawParams {
    stream_color: dto.stream_color,
  };
  let Some(render_frontend) = ctx.render_frontend() else {
    scene_guard.scene.remove_entity(jet_entity);
    return Err("render frontend not found".into());
  };
  let render_device_handle = ctx.render_device_handle();
  let clock = scenes.time_managers.get(&scene_id).map(|t| t.current_epoch());
  let Some(emission_start_us) = resolve_emission_start_us(dto.emission_start_us, clock) else {
    scene_guard.scene.remove_entity(jet_entity);
    return Err("scene has no clock".into());
  };
  match ParticleSystemComponent::new(
    render_frontend,
    render_device_handle,
    jet_entity,
    emit_params,
    draw_params,
    JET_TTL_US,
    emission_start_us,
  ) {
    Ok(c) => {
      let _ = scene_guard.scene.add_component(jet_entity, c);
      Ok(jet_entity)
    }
    Err(e) => {
      scene_guard.scene.remove_entity(jet_entity);
      Err(format!(
        "failed to create particle system component: {e}"
      ))
    }
  }
}

fn particle_system_count(ctx: &SimulationContext, scene_id: u64) -> usize {
  let scenes = ctx.scenes.read();
  let g = scenes.get_scene(scene_id).unwrap();
  let g = g.read();
  let mut n = 0usize;
  g.scene.query1(|_, _: &ParticleSystemComponent| n += 1);
  n
}

fn wait_caught_up(ctx: &SimulationContext, scene_id: u64) -> Vec<DustTierStats> {
  let t0 = std::time::Instant::now();
  loop {
    let busy = ctx
      .get_scene(scene_id)
      .map(|s| s.read().active_physics_task.load(core::sync::atomic::Ordering::Acquire))
      .unwrap_or(false);
    let tiers = ctx.dust_stats(scene_id);
    if !busy && !tiers.is_empty() && tiers.iter().all(|t| t.caught_up) {
      return tiers;
    }
    assert!(
      t0.elapsed().as_secs_f64() < 120.0,
      "dust never caught up: {tiers:?}"
    );
    std::thread::sleep(std::time::Duration::from_millis(50));
  }
}

fn window_s(t: &DustTierStats) -> f64 {
  ((t.band_max_s - t.band_min_s) / 256.0).max(1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. the four jets
// ─────────────────────────────────────────────────────────────────────────────

/// All four jets of the save through the FFI recipe, in save order: every
/// `ParticleSystemComponent::new` must succeed (the −67° one never reached the simulation).
#[test]
fn review_67p_four_jets_all_create() {
  let ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");
  if ctx.render_frontend().is_none() {
    println!("Skipping: no GPU");
    return;
  }
  load_almanac_files(&ctx);
  let (scene_id, _comet_body) = committed_scene(&ctx);
  let mut jets = Vec::new();
  let mut errors = Vec::new();
  for (k, jet) in SAVE_JETS.iter().enumerate() {
    match add_particle_system(&ctx, scene_id, &save_dto(jet, EMISSION_START_NOW)) {
      Ok(id) => {
        println!("jet {k} (lat {}°): entity {}", jet.lat_deg, id.as_ffi());
        jets.push((k, id));
      }
      Err(e) => {
        println!("jet {k} (lat {}°): ERROR {e}", jet.lat_deg);
        errors.push((k, e));
      }
    }
  }
  let n = particle_system_count(&ctx, scene_id);
  println!("particle systems in the scene: {n}");

  // for the record: what each jet holds two days in (no assertion: the −67° site's light is
  // `review_67p_jet_sites_lit_fraction`'s question)
  let t_probe = committed_start() + Duration::from_days(2.0);
  if ctx.seek_epoch_sync(scene_id, t_probe) {
    let _ = wait_caught_up(&ctx, scene_id);
    let scenes = ctx.scenes.read();
    let g = scenes.get_scene(scene_id).unwrap();
    let g = g.read();
    for (k, id) in &jets {
      let stats = g
        .scene
        .with_component(*id, |ps: &ParticleSystemComponent| ps.dust.lock().stats());
      println!("jet {k} (lat {}°) at start + 2 d: {stats:?}", SAVE_JETS[*k].lat_deg);
    }
  } else {
    println!("seek to start + 2 d failed");
  }
  ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();

  assert!(errors.is_empty(), "jets refused: {errors:?}");
  assert_eq!(n, 4, "particle systems in the scene");
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. sunlight of each jet site
// ─────────────────────────────────────────────────────────────────────────────

/// The emission's jet state (`logic_thread::utils::jet_state_at`, private there, mirrored here
/// for the parts the light depends on: the body rotation, the site normal, the model spin and
/// the heliocentric site position) at `epoch`, on the SPK path (`t ≥ 0`).
fn jet_state(
  almanac: &AlmanacPackedData,
  planet: &AlmanacPlanet,
  model: &BodyRotationalModel,
  jet_local: &TransformComponent,
  t_s: f64,
  epoch: Epoch,
) -> Option<JetState> {
  let (pos_km, vel_kms, body_rot) = planet.step_with_velocity(epoch, almanac, Some(model)).ok()?;
  let pole = body_rot.rotate_vector(Vec3f32::from_components(0.0, 0.0, 1.0));
  let omega = model.rotation_rate * core::f64::consts::PI / (180.0 * 86400.0);
  let sgn = if omega < 0.0 { -1.0 } else { 1.0 };
  let spin = [
    pole.x() as f64 * sgn,
    pole.y() as f64 * sgn,
    pole.z() as f64 * sgn,
    omega.abs(),
  ];
  let off_km = body_rot.rotate_vector(jet_local.position);
  let site_normal = {
    let n = [off_km.x() as f64, off_km.y() as f64, off_km.z() as f64];
    let l = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
    if l > 0.0 {
      [n[0] / l, n[1] / l, n[2] / l]
    } else {
      [0.0, 0.0, 1.0]
    }
  };
  let rot = (body_rot * jet_local.rotation).0;
  let site_offset_m = {
    let o = jet_local.rotation.conjugate().rotate_vector(jet_local.position);
    [o.x() * 1000.0, o.y() * 1000.0, o.z() * 1000.0]
  };
  Some(JetState {
    t_s,
    r_m: [
      (pos_km.x() + off_km.x() as f64) * 1000.0,
      (pos_km.y() + off_km.y() as f64) * 1000.0,
      (pos_km.z() + off_km.z() as f64) * 1000.0,
    ],
    v_ms: [vel_kms.x() * 1000.0, vel_kms.y() * 1000.0, vel_kms.z() * 1000.0],
    rot: [rot.x(), rot.y(), rot.z(), rot.w()],
    site_normal,
    spin: Some(spin),
    site_offset_m,
  })
}

/// For each site of the save, the lit fraction over the committed range with the emission's
/// light model (`JetState::lit_window`): 10-minute windows, plus the instantaneous sign of
/// `ŝ·n` at each sample and the tier-0 emission windows (TTL / 256) that would emit nothing.
#[test]
fn review_67p_jet_sites_lit_fraction() {
  let mut almanac = AlmanacPackedData::default();
  for f in [
    "../../assets/planets/pck00011.pca",
    "../../assets/planets/gm_de431.pca",
    "../../assets/planets/de442.bsp",
    SPK_COMMITTED,
  ] {
    almanac.load_almanac(f).expect(f);
  }
  let planet = AlmanacPlanet::new(NAIF_67P);
  let model = save_rotation();
  let (start, end) = (committed_start(), committed_end());
  let span_s = (end - start).to_seconds();
  const STEP_S: f64 = 600.0;
  // the youngest tier's emission window (`DustHostState::window_len_s` of a 30 d band)
  let tier0_window_s = JET_TTL_US as f64 * 1e-6 / crate::scene::dust::WINDOWS_PER_TTL;
  let mut fractions = Vec::new();
  for jet in SAVE_JETS.iter() {
    let (lat, lon) = (jet.lat_deg.to_radians(), jet.lon_deg.to_radians());
    // unit comet scale: the site direction is all the light depends on
    let local_r = NUCLEUS_RADIUS_KM;
    let jet_local = TransformComponent {
      position: Vec3f32::from_components(
        local_r * lat.cos() * lon.cos(),
        local_r * lat.cos() * lon.sin(),
        local_r * lat.sin(),
      ),
      rotation: Quat::identity(),
      scale: Vec3f32::from_components(0.04, 0.04, 0.04),
    };
    let (mut lit_s, mut total_s, mut lit_samples, mut samples, mut ever_lit) =
      (0.0f64, 0.0f64, 0usize, 0usize, false);
    let mut max_cos = f64::NEG_INFINITY;
    let mut t = 0.0f64;
    while t < span_s {
      let dur = STEP_S.min(span_s - t);
      let epoch = start + Duration::from_seconds(t);
      let js = jet_state(&almanac, &planet, &model, &jet_local, t, epoch)
        .unwrap_or_else(|| panic!("no comet state at {}", fmt_epoch(epoch)));
      let lw = js.lit_window(dur);
      lit_s += lw.lit_time_s;
      total_s += dur;
      let s = js.sun_dir();
      let c = s[0] * js.site_normal[0] + s[1] * js.site_normal[1] + s[2] * js.site_normal[2];
      max_cos = max_cos.max(c);
      if c > 0.0 {
        lit_samples += 1;
      }
      ever_lit |= lw.lit_time_s > 0.0;
      samples += 1;
      t += STEP_S;
    }
    // tier-0 windows on the emission grid from the committed start
    let (mut dark_windows, mut windows) = (0usize, 0usize);
    let mut t = 0.0f64;
    while t + tier0_window_s <= span_s {
      let epoch = start + Duration::from_seconds(t);
      let js = jet_state(&almanac, &planet, &model, &jet_local, t, epoch).unwrap();
      if !(js.lit_window(tier0_window_s).lit_time_s > 0.0) {
        dark_windows += 1;
      }
      windows += 1;
      t += tier0_window_s;
    }
    let frac = lit_s / total_s;
    println!(
      "site lat {:>4}°: lit fraction {:.4} (lit_window, 10 min), instantaneous {}/{} = {:.4}, \
       max cos(sun, normal) {:.4}, ever lit {}, tier-0 windows ({:.0} s) without light {}/{}",
      jet.lat_deg,
      frac,
      lit_samples,
      samples,
      lit_samples as f64 / samples as f64,
      max_cos,
      ever_lit,
      tier0_window_s,
      dark_windows,
      windows
    );
    fractions.push((jet.lat_deg, frac));
  }
  for (lat, frac) in &fractions {
    if *lat == 40.0 || *lat == -10.0 {
      assert!(*frac > 0.0, "site lat {lat}° is never lit over the committed range");
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. dust age at the committed start
// ─────────────────────────────────────────────────────────────────────────────

/// The −10° jet ignited at `ignition` (TDB µs, or `EMISSION_START_NOW` resolved at the clock),
/// evaluated at the committed start through the real seek path (one day in, then back).
fn dust_at_committed_start(
  ctx: &SimulationContext,
  emission_start_us: i64,
) -> (i64, Vec<DustTierStats>) {
  let (scene_id, _) = committed_scene(ctx);
  let jet = add_particle_system(ctx, scene_id, &save_dto(&SAVE_JETS[2], emission_start_us))
    .unwrap_or_else(|e| panic!("jet refused: {e}"));
  let resolved = {
    let scenes = ctx.scenes.read();
    let g = scenes.get_scene(scene_id).unwrap();
    let g = g.read();
    g.scene
      .with_component(jet, |ps: &ParticleSystemComponent| ps.emission_start_us)
      .unwrap()
  };
  let start = committed_start();
  assert!(
    ctx.seek_epoch_sync(scene_id, start + Duration::from_days(1.0)),
    "seek start + 1 d"
  );
  let at1 = wait_caught_up(ctx, scene_id);
  println!("  at start + 1 d: {at1:?}");
  assert!(ctx.seek_epoch_sync(scene_id, start), "seek start");
  let at0 = wait_caught_up(ctx, scene_id);
  (resolved, at0)
}

/// The app ignites each jet "now" (the clock at the add): at the committed start that leaves no
/// dust at all. Ignited 30 d earlier (one TTL), the start shows a 30 d tail; the pre-start
/// history comes from the re-osculated reference orbit (`jet_state_at`, `t < 0`) since neither
/// SPK of the save covers it (`review_67p_spk_segment_bounds`).
#[test]
fn review_67p_dust_age_at_committed_start() {
  let ctx = SimulationContext::startup(None).expect("Failed to create SimulationContext");
  if ctx.render_frontend().is_none() {
    println!("Skipping: no GPU");
    return;
  }
  load_almanac_files(&ctx);
  let start = committed_start();

  // what the app does: EMISSION_START_NOW resolved while the clock is at the committed start
  println!("ignition = EMISSION_START_NOW (clock at the committed start {})", fmt_epoch(start));
  let (now_us, at_now) = dust_at_committed_start(&ctx, EMISSION_START_NOW);
  println!("  resolved ignition {now_us} µs (start = {} µs)", emission_start_us_from_epoch(start));
  println!("  at start: {at_now:?}");
  for (k, t) in at_now.iter().enumerate() {
    println!(
      "  tier {k}: live {} age span {:.3} d (youngest {:.3} d, oldest {:.3} d)",
      t.live_clusters,
      (t.oldest_age_s - t.youngest_age_s) / 86_400.0,
      t.youngest_age_s / 86_400.0,
      t.oldest_age_s / 86_400.0
    );
  }

  // one TTL before the committed start
  let early = start - Duration::from_days(30.0);
  println!("ignition = committed start − 30 d ({})", fmt_epoch(early));
  let (early_us, at_early) = dust_at_committed_start(&ctx, emission_start_us_from_epoch(early));
  println!("  resolved ignition {early_us} µs");
  println!("  at start: {at_early:?}");
  for (k, t) in at_early.iter().enumerate() {
    println!(
      "  tier {k}: live {} age span {:.3} d (youngest {:.3} d, oldest {:.3} d), unlit windows {}",
      t.live_clusters,
      (t.oldest_age_s - t.youngest_age_s) / 86_400.0,
      t.youngest_age_s / 86_400.0,
      t.oldest_age_s / 86_400.0,
      t.unlit_windows
    );
  }
  ctx.threads.logic_thread.tx().try_send(LogicCommand::Shutdown).unwrap();

  assert_eq!(
    now_us,
    emission_start_us_from_epoch(start),
    "EMISSION_START_NOW must resolve to the clock at the committed start"
  );
  for (k, t) in at_now.iter().enumerate() {
    if t.live_clusters > 0 {
      assert!(
        t.oldest_age_s - t.youngest_age_s <= window_s(t),
        "tier {k}: age span {:.3} d at the ignition epoch",
        (t.oldest_age_s - t.youngest_age_s) / 86_400.0
      );
    }
  }
}
