//! Scene serialization / deserialization helpers.
//!
//! Called from `LogicCommand::DumpScene` and `LogicCommand::RestoreSceneDump` in the logic thread.
//! Contains no Vulkan dependencies — GPU-side operations (particle snapshot/restore, mesh resource
//! cleanup) are handled directly in `logic_thread.rs`.

use crate::{
  scene::{
    AlmanacPlanet, CameraComponent, CameraProjection, CometMarkerComponent, GridComponent,
    HighResTransformComponent, Scene, SkyComponent, StaticMeshComponent, SunComponent,
    TransformComponent,
    particles::v2::{ParticleSystemComponent, ParticleSystemDrawParams},
  },
  simulation::comet::{Comet, Texture, Vertex, next_comet_id},
  simulation_api::structs::{
    SerializedAlmanacPlanet, SerializedCamera, SerializedComponent, SerializedEntity,
    SerializedHighResTransform, SerializedParticleSystemConfig, SerializedStaticMesh,
    SerializedTexture, SerializedTransform, SerializedVertex,
  },
};

/// Result of [`deserialize_scene`], used by the logic thread to evict stale GPU resources.
pub struct DeserializeResult {
  /// Content hashes (`Comet::id`) of every `StaticMeshComponent` in the restored scene.
  pub new_mesh_hashes: alloc::collections::BTreeSet<u64>,
  /// Entity IDs of sun entities in the restored scene (for `sun_resources` DashMap cleanup).
  pub new_sun_entity_ids: alloc::vec::Vec<crate::scene::EntityId>,
}

// ─── Serialize ────────────────────────────────────────────────────────────────

/// Walk the entire ECS and produce a `Vec<SerializedEntity>` for the `SceneDump`.
///
/// `ParticleSystemComponent` is serialized as config-only (GPU particle bytes are captured
/// separately by `vulkan_device.snapshot_particles()`).
pub fn serialize_scene(scene: &Scene) -> alloc::vec::Vec<SerializedEntity> {
  // Collect all live entity IDs across every queryable component type.
  let mut entity_ids: alloc::collections::BTreeSet<crate::scene::EntityId> =
    alloc::collections::BTreeSet::new();

  macro_rules! collect_entities {
    ($($ty:ty),+ $(,)?) => {
      $(scene.query1(|e, _: &$ty| { entity_ids.insert(e); });)+
    };
  }
  collect_entities!(
    TransformComponent,
    HighResTransformComponent,
    StaticMeshComponent,
    CameraComponent,
    SunComponent,
    SkyComponent,
    GridComponent,
    CometMarkerComponent,
    AlmanacPlanet,
    ParticleSystemComponent,
  );

  let mut result = alloc::vec::Vec::with_capacity(entity_ids.len());

  for entity in entity_ids {
    let mut components = alloc::vec::Vec::new();

    scene.with_component(entity, |t: &HighResTransformComponent| {
      components.push(SerializedComponent::HighResTransform(hrt_to_serial(t)));
    });
    scene.with_component(entity, |t: &TransformComponent| {
      components.push(SerializedComponent::Transform(tr_to_serial(t)));
    });
    scene.with_component(entity, |c: &CameraComponent| {
      components.push(SerializedComponent::Camera(cam_to_serial(c)));
    });
    scene.with_component(entity, |m: &StaticMeshComponent| {
      components.push(SerializedComponent::StaticMesh(mesh_to_serial(m)));
    });
    scene.with_component(entity, |ps: &ParticleSystemComponent| {
      components.push(SerializedComponent::ParticleSystemConfig(ps_to_serial(
        entity, ps,
      )));
    });
    scene.with_component(entity, |_: &CometMarkerComponent| {
      components.push(SerializedComponent::CometMarker);
    });
    scene.with_component(entity, |_: &SunComponent| {
      components.push(SerializedComponent::SunMarker);
    });
    scene.with_component(entity, |_: &SkyComponent| {
      components.push(SerializedComponent::SkyMarker);
    });
    scene.with_component(entity, |_: &GridComponent| {
      components.push(SerializedComponent::GridMarker);
    });
    scene.with_component(entity, |p: &AlmanacPlanet| {
      components.push(SerializedComponent::AlmanacPlanet(
        SerializedAlmanacPlanet {
          naif_id: p.naif_id,
          mass_kg: 0.0, // AlmanacPlanet stores no mass; placeholder for future extension
        },
      ));
    });

    if components.is_empty() {
      continue;
    }

    result.push(SerializedEntity {
      ffi_id: entity.as_ffi(),
      name: scene.get_name(entity).unwrap_or_default(),
      parent_ffi_id: scene.get_parent(entity).map(|p| p.as_ffi()),
      components,
    });
  }

  result
}

fn mesh_to_serial(m: &StaticMeshComponent) -> SerializedStaticMesh {
  let comet: &Comet = &*m.mesh;
  let (vertices, indices) = if m.asset_path.is_empty() {
    // Procedural mesh — inline geometry required for round-trip
    let verts = comet
      .vertices
      .iter()
      .map(|v| SerializedVertex {
        position: v.position,
        normal: v.normal,
        uv: v.uv,
        tangent: v.tangent,
      })
      .collect();
    (Some(verts), Some(comet.indices.clone()))
  } else {
    // Asset-backed — only the path key is needed; geometry is re-loaded from cache on restore
    (None, None)
  };

  SerializedStaticMesh {
    asset_path: m.asset_path.clone(),
    vertices,
    indices,
    albedo_map: comet.albedo_map.as_ref().map(texture_to_serial),
    normal_map: comet.normal_map.as_ref().map(texture_to_serial),
    roughness_map: comet.roughness_map.as_ref().map(texture_to_serial),
    ao_map: comet.ao_map.as_ref().map(texture_to_serial),
    emissive_color: m.emissive_color,
    is_visible: m.is_visible,
  }
}

pub(crate) fn texture_to_serial(t: &Texture) -> SerializedTexture {
  SerializedTexture {
    width: t.width,
    height: t.height,
    format: t.format,
    has_mipmaps: t.has_mipmaps,
    data: t.data.to_vec(),
  }
}

fn ps_to_serial(
  entity: crate::scene::EntityId,
  ps: &ParticleSystemComponent,
) -> SerializedParticleSystemConfig {
  use core::sync::atomic::Ordering;
  SerializedParticleSystemConfig {
    entity_ffi_id: entity.as_ffi(),
    emission_params: ps.emission_params,
    stream_color: ps.draw_params.stream_color,
    ttl_us: ps.ttl_us,
    last_emission: ps.last_emission.load(Ordering::Relaxed),
    last_compaction: ps.last_compaction.load(Ordering::Relaxed),
  }
}

fn hrt_to_serial(t: &HighResTransformComponent) -> SerializedHighResTransform {
  use aethervk_oshal_rlib::math::vector::{Vector3, Vector4};
  SerializedHighResTransform {
    position: [t.position.x(), t.position.y(), t.position.z()],
    rotation: [
      t.rotation.0.x(),
      t.rotation.0.y(),
      t.rotation.0.z(),
      t.rotation.0.w(),
    ],
    scale: [t.scale.x(), t.scale.y(), t.scale.z()],
  }
}

fn tr_to_serial(t: &TransformComponent) -> SerializedTransform {
  use aethervk_oshal_rlib::math::vector::{Vector3, Vector4};
  SerializedTransform {
    position: [t.position.x(), t.position.y(), t.position.z()],
    rotation: [
      t.rotation.0.x(),
      t.rotation.0.y(),
      t.rotation.0.z(),
      t.rotation.0.w(),
    ],
    scale: [t.scale.x(), t.scale.y(), t.scale.z()],
  }
}

fn cam_to_serial(c: &CameraComponent) -> SerializedCamera {
  match &c.projection {
    CameraProjection::Perspective { fov, near, far, .. } => SerializedCamera {
      fov_y_rad: *fov,
      near: *near,
      far: *far,
      is_perspective: true,
    },
    CameraProjection::Orthographic { near, far, .. } => SerializedCamera {
      fov_y_rad: 0.0,
      near: *near,
      far: *far,
      is_perspective: false,
    },
  }
}

//// ─── Deserialize ──────────────────────────────────────────────────────────────

/// Overwrite the current scene ECS in-place with the deserialized component data.
///
/// Uses `add_component` to write each component — this means it works correctly both:
/// - **After `clone_structure_only`** where all slots are `None` (insert path), and
/// - **In a live running scene** where slots may already be `Some` (`add_component` overwrites).
///
/// Mesh resolution strategy (overwrite-merge by `asset_path`):
/// - Non-empty `asset_path` already in `mesh_cache` → reuse existing `Arc<Comet>` (no duplicate).
/// - Non-empty `asset_path` absent from `mesh_cache` → reconstruct from inline data and insert.
/// - Empty `asset_path` (procedural) → reconstruct from inline data, not cached.
///
/// `ParticleSystemConfig` is the only exception: it uses `with_component_mut` because it cannot
/// be freshly constructed from serialized data alone (the GPU `device_data` must already exist).
///
/// Returns [`DeserializeResult`] for GPU resource cleanup.
pub fn deserialize_scene(
  scene: &mut Scene,
  dump_entities: &[SerializedEntity],
  mesh_cache: &crate::scene::AssetCache<Comet>,
) -> DeserializeResult {
  let mut new_mesh_hashes = alloc::collections::BTreeSet::new();
  let mut new_sun_entity_ids = alloc::vec::Vec::new();

  for se in dump_entities {
    let entity = crate::scene::EntityId::from(slotmap::KeyData::from_ffi(se.ffi_id));

    for comp in &se.components {
      match comp {
        SerializedComponent::HighResTransform(h) => {
          use aethervk_oshal_rlib::math::vector::{
            Vector3, vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat,
          };
          let new_t = HighResTransformComponent {
            position: Vec3f64::from_components(h.position[0], h.position[1], h.position[2]),
            rotation: Quat::from_components(
              h.rotation[3],
              h.rotation[0],
              h.rotation[1],
              h.rotation[2],
            ),
            scale: Vec3f32::from_components(h.scale[0], h.scale[1], h.scale[2]),
          };
          let _ = scene.add_component(entity, new_t);
        }

        SerializedComponent::Transform(tr) => {
          use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec4::Quat};
          let new_t = TransformComponent {
            position: Vec3f32::from_components(tr.position[0], tr.position[1], tr.position[2]),
            rotation: Quat::from_components(
              tr.rotation[3],
              tr.rotation[0],
              tr.rotation[1],
              tr.rotation[2],
            ),
            scale: Vec3f32::from_components(tr.scale[0], tr.scale[1], tr.scale[2]),
          };
          let _ = scene.add_component(entity, new_t);
        }

        SerializedComponent::Camera(sc) => {
          let projection = if sc.is_perspective {
            CameraProjection::Perspective {
              fov: sc.fov_y_rad,
              // aspect_ratio is determined by the window at runtime; restore a safe default
              aspect_ratio: 16.0 / 9.0,
              near: sc.near,
              far: sc.far,
            }
          } else {
            CameraProjection::Orthographic {
              left: -1.0,
              right: 1.0,
              bottom: -1.0,
              top: 1.0,
              near: sc.near,
              far: sc.far,
            }
          };
          let _ = scene.add_component(entity, CameraComponent { projection, focus_distance: 10.0 });
        }

        SerializedComponent::StaticMesh(sm) => {
          let arc = resolve_or_create_mesh(sm, mesh_cache);
          new_mesh_hashes.insert(arc.id);
          let _ = scene.add_component(
            entity,
            StaticMeshComponent {
              asset_path: sm.asset_path.clone(),
              mesh: arc,
              emissive_color: sm.emissive_color,
              is_visible: sm.is_visible,
            },
          );
        }

        SerializedComponent::ParticleSystemConfig(psc) => {
          // Cannot freshly construct a ParticleSystemComponent from serialized data alone
          // (GPU `device_data` must already exist). Use with_component_mut to patch config fields.
          use core::sync::atomic::Ordering;
          scene.with_component_mut(entity, |ps: &mut ParticleSystemComponent| {
            ps.emission_params = psc.emission_params;
            ps.draw_params = ParticleSystemDrawParams { stream_color: psc.stream_color };
            ps.ttl_us = psc.ttl_us;
            ps.last_emission.store(psc.last_emission, Ordering::Relaxed);
            ps.last_compaction.store(psc.last_compaction, Ordering::Relaxed);
          });
        }

        SerializedComponent::AlmanacPlanet(sp) => {
          let _ = scene.add_component(entity, AlmanacPlanet::new(sp.naif_id));
        }

        // Marker components carry no data — their presence is already baked into the scene
        // hierarchy from `create_empty_scene2`. We only need to track sun entity IDs for cleanup.
        SerializedComponent::SunMarker => {
          new_sun_entity_ids.push(entity);
          // Restore a zero-value SunComponent; radius/resolution are typically set
          // via SetSunParameters after restore if needed.
          let _ = scene.add_component(entity, SunComponent { radius_km: 0.0, resolution: (0, 0, 0) });
        }
        SerializedComponent::SkyMarker => {
          let _ = scene.add_component(entity, SkyComponent {});
        }
        SerializedComponent::GridMarker => {
          let _ = scene.add_component(entity, GridComponent {});
        }
        SerializedComponent::CometMarker => {
          let _ = scene.add_component(entity, CometMarkerComponent {});
        }
      }
    }
  }

  DeserializeResult { new_mesh_hashes, new_sun_entity_ids }
}

/// Return an existing `Arc<Comet>` from the cache (lookup by `asset_path`), or construct a
/// new one from inline serialized data and optionally insert it under `asset_path`.
fn resolve_or_create_mesh(
  sm: &SerializedStaticMesh,
  mesh_cache: &crate::scene::AssetCache<Comet>,
) -> alloc::sync::Arc<Comet> {
  // Overwrite-merge: if the same asset_path is already cached, reuse it — no duplicate upload.
  if !sm.asset_path.is_empty()
    && let Some(existing) = mesh_cache.get(&sm.asset_path)
  {
    return existing;
  }

  // Reconstruct Comet from inline vertex/index data.
  let comet = Comet {
    id: next_comet_id(),
    vertices: sm
      .vertices
      .as_ref()
      .map(|vs| {
        vs.iter()
          .map(|v| Vertex {
            position: v.position,
            normal: v.normal,
            uv: v.uv,
            tangent: v.tangent,
          })
          .collect()
      })
      .unwrap_or_default(),
    indices: sm.indices.clone().unwrap_or_default(),
    albedo_map: sm.albedo_map.as_ref().map(serial_to_texture),
    normal_map: sm.normal_map.as_ref().map(serial_to_texture),
    roughness_map: sm.roughness_map.as_ref().map(serial_to_texture),
    ao_map: sm.ao_map.as_ref().map(serial_to_texture),
  };

  if !sm.asset_path.is_empty() {
    mesh_cache.insert(sm.asset_path.clone(), comet)
  } else {
    alloc::sync::Arc::new(comet)
  }
}

pub(crate) fn serial_to_texture(st: &SerializedTexture) -> Texture {
  Texture {
    data: bytes::Bytes::copy_from_slice(&st.data),
    format: st.format,
    width: st.width,
    height: st.height,
    has_mipmaps: st.has_mipmaps,
  }
}