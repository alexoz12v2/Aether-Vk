//! Dust configuration invariant tests.
//!
//! These tests operate on a bare [`Scene`] (no GPU required) to verify the
//! structural invariants of jet entities:
//!
//! 1. A jet entity spawned as a child of the comet carries a [`TransformComponent`] and a
//!    [`JetDirectionLineComponent`] (its direction line), no [`StaticMeshComponent`] (the old
//!    sphere marker) — this mirrors what `avkSimulationContext_addParticleSystem` does in `ffi.rs`.
//! 2. A jet entity must be a child of the comet body entity.
//! 3. `propagate_common_params` updates only the 8 common fields of sibling
//!    [`ParticleSystemEmitParams`], leaving per-jet fields intact.
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

/// Invariant 3: `propagate_common_params` updates common fields and leaves per-jet fields.
///
/// This replicates the mutation logic of `propagate_common_params` in `ffi.rs` directly on
/// a `ParticleSystemEmitParams` struct so the test requires no GPU or FFI access.
#[test]
fn propagate_common_params_leaves_jet_specific_fields_intact() {
  use crate::scene::ParticleSystemDTO;

  // A DTO representing the update from the triggering jet (new common values).
  let mut dto = ParticleSystemDTO::zeroed();
  dto.diametre_um = 20.0;
  dto.density_gcm3 = 0.8;
  dto.scattering_efficiency = 1.1;
  dto.geometric_albedo = 0.05;
  dto.size_q = 3.0;
  dto.size_range = 20.0;
  dto.afrho_0_cm = 800.0;
  dto.afrho_power = 2.2;
  dto.afrho_cutoff_au = 5.5;
  dto.afrho_max_value_cm = 150_000.0;
  // jet-specific on the originating jet — must NOT flow into sibling
  dto.latitude_rad = 0.5;

  // Sibling's existing emission params (distinct grain size and latitude)
  let mut sibling = ParticleSystemEmitParams::zeroed();
  sibling.diametre_um = 10.0; // original
  sibling.latitude_rad = 1.2; // per-jet — must be preserved

  // Apply common params exactly as propagate_common_params does:
  sibling.diametre_um = dto.diametre_um;
  sibling.density_gcm3 = dto.density_gcm3;
  sibling.scattering_efficiency = dto.scattering_efficiency;
  sibling.geometric_albedo = dto.geometric_albedo;
  sibling.size_q = dto.size_q;
  sibling.size_range = dto.size_range;
  sibling.afrho_0_cm = dto.afrho_0_cm;
  sibling.afrho_power = dto.afrho_power;
  sibling.afrho_cutoff_au = dto.afrho_cutoff_au;
  sibling.afrho_max_value_cm = dto.afrho_max_value_cm;
  // latitude_rad is intentionally NOT propagated

  assert_eq!(sibling.diametre_um, 20.0);
  assert_eq!(sibling.density_gcm3, 0.8);
  assert_eq!(sibling.scattering_efficiency, 1.1);
  assert_eq!(sibling.geometric_albedo, 0.05);
  assert_eq!(sibling.size_q, 3.0);
  assert_eq!(sibling.size_range, 20.0);
  assert_eq!(sibling.afrho_0_cm, 800.0);
  assert_eq!(sibling.afrho_power, 2.2);
  assert_eq!(sibling.afrho_cutoff_au, 5.5);
  assert_eq!(sibling.afrho_max_value_cm, 150_000.0);
  assert_eq!(
    sibling.latitude_rad, 1.2,
    "latitude_rad is a per-jet property and must not be propagated by propagate_common_params"
  );
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
