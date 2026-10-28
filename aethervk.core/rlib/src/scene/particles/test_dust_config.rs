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

fn jet_dto(diametre_um: f32, afrho_0_cm: f32, latitude_rad: f32) -> crate::scene::ParticleSystemDTO {
  let mut dto = crate::scene::ParticleSystemDTO::zeroed();
  dto.diametre_um = diametre_um;
  dto.density_gcm3 = 0.533;
  dto.scattering_efficiency = 1.0;
  dto.geometric_albedo = 0.04;
  dto.size_q = 3.5;
  dto.size_range = 10.0;
  dto.afrho_0_cm = afrho_0_cm;
  dto.afrho_power = 2.0;
  dto.afrho_cutoff_au = 5.0;
  dto.afrho_max_value_cm = 100_000.0;
  dto.latitude_rad = latitude_rad;
  dto.start_velocity_mean = 60.0;
  dto.start_velocity_std = 5.0;
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

  // modify jet a: a 300 µm, denser, brighter grain with a different Afρ law
  let mut modified = jet_dto(300.0, 1200.0, -1.08);
  modified.density_gcm3 = 1.0;
  modified.scattering_efficiency = 1.5;
  modified.geometric_albedo = 0.06;
  modified.size_q = 3.0;
  modified.size_range = 30.0;
  modified.afrho_power = 3.0;
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
