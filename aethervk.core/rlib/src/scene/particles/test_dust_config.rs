//! Dust configuration invariant tests.
//!
//! These tests operate on a bare [`Scene`] (no GPU required) to verify the
//! structural invariants of jet entities:
//!
//! 1. A jet entity spawned as a child of the comet carries a [`TransformComponent`] and a
//!    [`JetDirectionLineComponent`] (its direction line), no [`StaticMeshComponent`] (the old
//!    sphere marker) — this mirrors what `avkSimulationContext_addParticleSystem` does in `ffi.rs`.
//! 2. A jet entity must be a child of the comet body entity.
//! 3. Grain and activity parameters are per jet: modifying one jet (its
//!    [`ParticleSystemEmitParams`] rebuilt from the DTO, as `avkSimulationContext_modifyParticleSystem`
//!    does) leaves every sibling jet's parameters as they were.
//! 4. `JetDirectionLineComponent::color` is updated from `draw_params.stream_color`
//!    on modify (mirrors `avkSimulationContext_modifyParticleSystem`).

extern crate std;

use alloc::sync::Arc;
use bytemuck::Zeroable;
use parking_lot::RwLock;

use crate::scene::{
  HasComponentResultEnum, JetDirectionLineComponent, Scene, StaticMeshComponent, TransformComponent,
  particles::v2::ParticleSystemEmitParams,
};
use aethervk_oshal_rlib::math::{
  quaternion::Quaternion,
  vector::{Vector3, vec3::Vec3f32, vec4::Quat},
};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_scene() -> Scene {
  let texture_cache = Arc::new(RwLock::new(
    crate::simulation::texture_cache::TextureCache::new("test"),
  ));
  let scene = Scene::new(texture_cache);
  scene.register_component::<TransformComponent>(&[]);
  scene.register_component::<StaticMeshComponent>(&[core::any::TypeId::of::<TransformComponent>()]);
  scene.register_component::<JetDirectionLineComponent>(&[core::any::TypeId::of::<TransformComponent>()]);
  scene
}

fn make_transform() -> TransformComponent {
  TransformComponent {
    position: Vec3f32::from_components(1.0, 0.0, 0.0),
    rotation: Quat::identity(),
    scale: Vec3f32::from_components(0.05, 0.05, 0.05),
  }
}

fn make_jet_line(color: [f32; 4]) -> JetDirectionLineComponent {
  JetDirectionLineComponent {
    color,
    is_visible: true,
    show_angles: false,
  }
}

// ── tests ─────────────────────────────────────────────────────────────────────

/// Invariant 1: a jet entity spawned as a comet child carries its transform and its direction
/// line, and no marker mesh (the sphere on the site is gone). Mirrors the spawn sequence in
/// `avkSimulationContext_addParticleSystem`.
#[test]
fn jet_entity_has_a_direction_line_and_no_marker_mesh() {
  let scene = make_scene();
  let comet = scene.spawn_entity("comet");
  let jet = scene.spawn_entity("jet");
  scene.set_parent(jet, Some(comet));

  let _ = scene.add_component(jet, make_transform());
  let _ = scene.add_component(jet, make_jet_line([0.4, 0.8, 1.0, 1.0]));

  assert_eq!(
    scene.has_component::<TransformComponent>(jet),
    HasComponentResultEnum::EntityHasComponent,
    "jet entity must have TransformComponent"
  );
  assert_eq!(
    scene.has_component::<JetDirectionLineComponent>(jet),
    HasComponentResultEnum::EntityHasComponent,
    "jet entity must have its direction line"
  );
  assert_ne!(
    scene.has_component::<StaticMeshComponent>(jet),
    HasComponentResultEnum::EntityHasComponent,
    "no marker mesh on the jet site"
  );
  // ffi.rs spawns the line visible, angles off (the C# side turns them on for the selected jet)
  let line = scene.with_component(jet, |l: &JetDirectionLineComponent| l.clone()).unwrap();
  assert!(line.is_visible && !line.show_angles);
}

/// Invariant 1: the jet entity must appear in the comet's children list.
#[test]
fn jet_entity_is_child_of_comet() {
  let scene = make_scene();
  let comet = scene.spawn_entity("comet");
  let jet = scene.spawn_entity("jet");
  scene.set_parent(jet, Some(comet));

  let children = scene.get_children(comet).expect("comet must have children after set_parent");
  assert!(
    children.contains(&jet),
    "jet must appear in comet's children list"
  );
}

/// Stand-in for `ParticleSystemComponent` (which needs a GPU device): it holds the same
/// [`ParticleSystemEmitParams`] the FFI add / modify write.
#[derive(Debug, Clone)]
struct EmitParamsOnly(ParticleSystemEmitParams);
impl crate::scene::Component for EmitParamsOnly {}

fn jet_dto(diametre_um: f32, afrho_cm: f32, latitude_rad: f32) -> crate::scene::ParticleSystemDTO {
  let mut dto = crate::scene::ParticleSystemDTO::zeroed();
  dto.diametre_um = diametre_um;
  dto.density_gcm3 = 0.533;
  dto.scattering_efficiency = 1.0;
  dto.geometric_albedo = 0.04;
  dto.size_q = 3.5;
  dto.size_range = 10.0;
  dto.dust_rate_power = 2.0;
  dto.dust_rate_cutoff_au = 5.0;
  dto.latitude_rad = latitude_rad;
  dto.start_velocity_mean = 60.0;
  dto.start_velocity_std = 5.0;
  // the activity as a dust rate: `afrho_cm` (max 100 000 cm) converted with the jet's grain
  let p = ParticleSystemEmitParams::from_dto(&dto);
  dto.dust_rate_1au_kgs = p.dust_rate_from_afrho_kgs(afrho_cm);
  dto.dust_rate_max_kgs = p.dust_rate_from_afrho_kgs(100_000.0);
  dto
}

/// Invariant 3: two jets of one comet keep distinct grain and activity parameters after one of
/// them is modified. Mirrors `avkSimulationContext_modifyParticleSystem`: only the modified
/// entity's emission parameters are rebuilt from the DTO; nothing is copied to the siblings.
#[test]
fn two_systems_keep_distinct_emit_params_after_modify() {
  let scene = make_scene();
  scene.register_component::<EmitParamsOnly>(&[]);
  let comet = scene.spawn_entity("comet");
  let jet_a = scene.spawn_entity("jet a");
  let jet_b = scene.spawn_entity("jet b");
  scene.set_parent(jet_a, Some(comet));
  scene.set_parent(jet_b, Some(comet));
  let _ = scene.add_component(
    jet_a,
    EmitParamsOnly(ParticleSystemEmitParams::from_dto(&jet_dto(100.0, 500.0, -1.08))),
  );
  let _ = scene.add_component(
    jet_b,
    EmitParamsOnly(ParticleSystemEmitParams::from_dto(&jet_dto(10.0, 50.0, -0.91))),
  );

  // modify jet a: a 300 µm, denser, brighter grain with a different dust rate law
  let mut modified = jet_dto(300.0, 1200.0, -1.08);
  modified.density_gcm3 = 1.0;
  modified.scattering_efficiency = 1.5;
  modified.geometric_albedo = 0.06;
  modified.size_q = 3.0;
  modified.size_range = 30.0;
  modified.dust_rate_power = 3.0;
  scene.with_component_mut(jet_a, |p: &mut EmitParamsOnly| {
    p.0 = ParticleSystemEmitParams::from_dto(&modified);
  });

  let a = scene.with_component(jet_a, |p: &EmitParamsOnly| p.0).unwrap();
  let b = scene.with_component(jet_b, |p: &EmitParamsOnly| p.0).unwrap();
  assert_eq!(a, ParticleSystemEmitParams::from_dto(&modified));
  assert_eq!(
    b,
    ParticleSystemEmitParams::from_dto(&jet_dto(10.0, 50.0, -0.91)),
    "the sibling jet keeps its own grain and activity parameters"
  );
  // the two jets emit different grains: their β and dust production differ
  assert!(a.beta() < b.beta(), "β a {} b {}", a.beta(), b.beta());
  assert_ne!(a.dust_production_rate_kgs(1.0), b.dust_production_rate_kgs(1.0));
  let (cfg_a, cfg_b) = (a.dust_emit_config(1.0, 1), b.dust_emit_config(1.0, 1));
  assert_eq!((cfg_a.diameter_um, cfg_b.diameter_um), (300.0, 10.0));
  assert_eq!((cfg_a.albedo, cfg_b.albedo), (0.06, 0.04));
}

/// Invariant 4: the direction line's colour follows the jet's stream colour (mirrors the update
/// in `avkSimulationContext_modifyParticleSystem`).
#[test]
fn jet_line_color_matches_stream_color() {
  let scene = make_scene();
  let comet = scene.spawn_entity("comet");
  let jet = scene.spawn_entity("jet");
  scene.set_parent(jet, Some(comet));

  let initial_color = [0.4_f32, 0.8, 1.0, 1.0];
  let _ = scene.add_component(jet, make_transform());
  let _ = scene.add_component(jet, make_jet_line(initial_color));
  assert_eq!(
    scene.with_component(jet, |l: &JetDirectionLineComponent| l.color),
    Some(initial_color)
  );

  let new_color = [1.0_f32, 0.5, 0.0, 1.0];
  scene.with_component_mut(jet, |l: &mut JetDirectionLineComponent| {
    l.color = new_color;
  });
  assert_eq!(
    scene.with_component(jet, |l: &JetDirectionLineComponent| l.color),
    Some(new_color),
    "the line colour must update to the new stream_color on modify"
  );
}

/// Invariant 5: Decommit cleanup removes all jet entities completely.
///
/// Mirrors the logic in `CleanupComet` where children of the comet body are deleted.
#[test]
fn decommit_cleanup_removes_jets() {
  let scene = make_scene();
  let comet = scene.spawn_entity("comet");
  let jet1 = scene.spawn_entity("jet1");
  let jet2 = scene.spawn_entity("jet2");
  scene.set_parent(jet1, Some(comet));
  scene.set_parent(jet2, Some(comet));

  assert_eq!(scene.get_children(comet).unwrap().len(), 2);

  // Mimic CleanupComet logic:
  if let Some(children) = scene.get_children(comet) {
    for child in children {
      scene.remove_entity(child);
    }
  }

  if let Some(children) = scene.get_children(comet) {
    assert!(
      children.is_empty(),
      "comet should have no children after jets are removed"
    );
  }

  assert!(
    scene.get_name(jet1).is_none(),
    "jet1 entity should be destroyed"
  );
  assert!(
    scene.get_name(jet2).is_none(),
    "jet2 entity should be destroyed"
  );
}

/// The dynamic-model fields of the FFI record (insolation weighting, orbit size law, pre-start
/// source, lifetime, q table) round-trip through [`ParticleSystemEmitParams::from_dto`] and
/// [`ParticleSystemEmitParams::write_dto`], and the record has the layout the C# side mirrors.
#[test]
fn dynamic_model_fields_round_trip_through_the_dto() {
  use crate::scene::particles::{DEFAULT_TTL_DAYS, SIZE_Q_TABLE_MAX};
  assert_eq!(core::mem::size_of::<crate::scene::ParticleSystemDTO>(), 184);
  let mut dto = jet_dto(100.0, 1000.0, 0.3);
  dto.insolation_weighted = 1;
  dto.orbit_size_law = 1;
  dto.spk_before_start = 1;
  dto.ttl_days = 120.0;
  dto.size_q_table_len = 3;
  dto.size_q_table[0..6].copy_from_slice(&[-200.0, 2.5, 0.0, 3.8, 40.0, 4.2]);
  let p = ParticleSystemEmitParams::from_dto(&dto);
  assert!(p.insolation_weighted() && p.orbit_size_law() && p.spk_before_start());
  assert_eq!(p.ttl_days(), 120.0);
  assert_eq!(p.ttl_us(), 120 * 86_400 * 1_000_000);
  assert_eq!(p.size_q_table(), &[[-200.0, 2.5], [0.0, 3.8], [40.0, 4.2]]);
  let mut back = crate::scene::ParticleSystemDTO::zeroed();
  p.write_dto(&mut back);
  assert_eq!(
    (back.insolation_weighted, back.orbit_size_law, back.spk_before_start, back.ttl_days, back.size_q_table_len),
    (1, 1, 1, 120.0, 3)
  );
  assert_eq!(back.size_q_table, dto.size_q_table);
  assert_eq!(ParticleSystemEmitParams::from_dto(&back), p);
  // a longer table is cut to the capacity
  dto.size_q_table_len = 50;
  assert_eq!(ParticleSystemEmitParams::from_dto(&dto).size_q_table().len(), SIZE_Q_TABLE_MAX);
  let _ = DEFAULT_TTL_DAYS;
}

/// An older record (zeroed new fields) reads as the toggles off and the 30-day lifetime that was
/// the hard-coded constant; an out-of-range lifetime falls back to it too.
#[test]
fn zeroed_dynamic_model_fields_are_the_old_behaviour() {
  use crate::scene::particles::{DEFAULT_TTL_DAYS, TTL_DAYS_RANGE};
  let p = ParticleSystemEmitParams::from_dto(&jet_dto(100.0, 1000.0, 0.3));
  assert!(!p.insolation_weighted() && !p.orbit_size_law() && !p.spk_before_start());
  assert_eq!(p.ttl_days(), DEFAULT_TTL_DAYS);
  assert_eq!(p.ttl_s(), 30.0 * 86_400.0);
  // the previous `JET_TTL_US` constant: 30 days in µs
  assert_eq!(p.ttl_us(), 2_592_000_000_000);
  assert!(p.size_q_table().is_empty());
  assert_eq!(p.size_q_at(12.0), p.size_q());
  for bad in [0.5f32, 400.0, -3.0, f32::NAN, f32::INFINITY] {
    let mut dto = jet_dto(100.0, 1000.0, 0.3);
    dto.ttl_days = bad;
    assert_eq!(ParticleSystemEmitParams::from_dto(&dto).ttl_days(), DEFAULT_TTL_DAYS, "{bad}");
  }
  for ok in [*TTL_DAYS_RANGE.start(), 7.5, *TTL_DAYS_RANGE.end()] {
    let mut dto = jet_dto(100.0, 1000.0, 0.3);
    dto.ttl_days = ok;
    assert_eq!(ParticleSystemEmitParams::from_dto(&dto).ttl_days(), ok);
  }
}

/// The q table is interpolated linearly in time, held at the end rows outside, clamped to the
/// accepted range; one row is a constant; a NaN row falls back to the jet's q.
#[test]
fn size_q_table_interpolates_and_holds() {
  let mut dto = jet_dto(100.0, 1000.0, 0.3);
  dto.size_q = 3.5;
  dto.size_q_table_len = 1;
  dto.size_q_table[0..2].copy_from_slice(&[0.0, 2.0]);
  let one = ParticleSystemEmitParams::from_dto(&dto);
  assert_eq!((one.size_q_at(-1e4), one.size_q_at(1e4)), (2.0, 2.0));
  dto.size_q_table_len = 2;
  dto.size_q_table[0..4].copy_from_slice(&[-100.0, 2.0, 100.0, 4.0]);
  let two = ParticleSystemEmitParams::from_dto(&dto);
  assert_eq!(two.size_q_at(-500.0), 2.0);
  assert!((two.size_q_at(0.0) - 3.0).abs() < 1e-6);
  assert!((two.size_q_at(50.0) - 3.5).abs() < 1e-6);
  assert_eq!(two.size_q_at(500.0), 4.0);
  assert_eq!(two.size_q_at(f64::NAN), 3.5);
  dto.size_q_table[1] = 9.0;
  assert_eq!(ParticleSystemEmitParams::from_dto(&dto).size_q_at(-500.0), 6.0);
  dto.size_q_table[0] = f32::NAN;
  assert_eq!(ParticleSystemEmitParams::from_dto(&dto).size_q_at(0.0), 3.5);
}
