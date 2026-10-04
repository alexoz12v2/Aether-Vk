//! Scene serialization / deserialization helpers.
//!
//! Called from `LogicCommand::DumpScene` and `LogicCommand::RestoreSceneDump` in the logic thread.
//! Contains no Vulkan dependencies — GPU-side operations (particle snapshot/restore, mesh resource
//! cleanup) are handled directly in `logic_thread.rs`.

use crate::{
  scene::{
    AlmanacPlanet, BodyRotationalModel, CameraComponent, CameraProjection, CometMarkerComponent,
    GridComponent, HighResTransformComponent, ReferenceFrameComponent, ReferenceFrameType, Scene,
    SkyComponent, StaticMeshComponent, SunComponent, TransformComponent,
    particles::v2::{ParticleSystemComponent, ParticleSystemDrawParams},
    trajectory::TrajectoryComponent,
  },
  simulation::comet::{Comet, Texture, Vertex, next_comet_id},
  simulation_api::structs::{
    SerializedAlmanacPlanet, SerializedCamera, SerializedComponent, SerializedEntity,
    SerializedHighResTransform, SerializedParticleSystemConfig, SerializedReferenceFrame,
    SerializedRotationalModel, SerializedStaticMesh, SerializedSun, SerializedTexture,
    SerializedTrajectory, SerializedTransform, SerializedVertex,
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
    BodyRotationalModel,
    TrajectoryComponent,
    ReferenceFrameComponent,
  );

  let mut result = alloc::vec::Vec::with_capacity(entity_ids.len());

  for entity in entity_ids {
    // The comet visual is aesthetic (see `comet_appearance`): a restore must not roll back the
    // user's display mode / mesh wiring, so it never enters a dump.
    if scene
      .with_component(entity, |_: &crate::simulation_api::comet_appearance::CometVisualComponent| ())
      .is_some()
    {
      continue;
    }
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
    scene.with_component(entity, |s: &SunComponent| {
      components.push(SerializedComponent::Sun(SerializedSun {
        radius_km: s.radius_km,
        resolution: s.resolution,
      }));
    });
    scene.with_component(entity, |m: &BodyRotationalModel| {
      components.push(SerializedComponent::BodyRotationalModel(
        SerializedRotationalModel {
          pole_ra: m.pole_ra,
          pole_dec: m.pole_dec,
          prime_meridian: m.prime_meridian,
          pole_ra_rate: m.pole_ra_rate,
          pole_dec_rate: m.pole_dec_rate,
          rotation_rate: m.rotation_rate,
          body_fixed_orientation: m.body_fixed_orientation,
        },
      ));
    });
    scene.with_component(entity, |t: &TrajectoryComponent| {
      components.push(SerializedComponent::Trajectory(SerializedTrajectory {
        control_points: t.control_points.clone(),
        control_points_f64: t.control_points_f64.clone(),
        color: t.color,
        line_width: t.line_width,
        texture_id: t.texture_id,
        subdivisions_per_segment: t.subdivisions_per_segment,
      }));
    });
    scene.with_component(entity, |f: &ReferenceFrameComponent| {
      components.push(SerializedComponent::ReferenceFrame(
        SerializedReferenceFrame {
          frame_type: f.frame_type as u8,
          scale: f.scale,
          soi_radius: f.soi_radius,
          depth_layer: f.depth_layer,
        },
      ));
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
  SerializedParticleSystemConfig {
    entity_ffi_id: entity.as_ffi(),
    emission_params: ps.emission_params,
    stream_color: ps.draw_params.stream_color,
    ttl_us: ps.ttl_us,
    // legacy field: emission is on a scaled-time grid now (deterministic, nothing to save)
    last_emission: 0,
    // dust v3 has no compaction; kept for dump format compatibility
    last_compaction: 0,
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
            // serialized xyzw, `Quat64::from_components` takes xyzw too (this used to pass wxyz)
            rotation: aethervk_oshal_rlib::math::vector::vec4f64::Quat64::from_components(
              h.rotation[0],
              h.rotation[1],
              h.rotation[2],
              h.rotation[3],
            ),
            scale: Vec3f32::from_components(h.scale[0], h.scale[1], h.scale[2]),
          };
          let _ = scene.add_component(entity, new_t);
        }

        SerializedComponent::Transform(tr) => {
          use aethervk_oshal_rlib::math::vector::{Vector3, vec3::Vec3f32, vec4::Quat};
          let new_t = TransformComponent {
            position: Vec3f32::from_components(tr.position[0], tr.position[1], tr.position[2]),
            // serialized xyzw, `Quat::from_components` takes xyzw too (this used to pass wxyz)
            rotation: Quat::from_components(
              tr.rotation[0],
              tr.rotation[1],
              tr.rotation[2],
              tr.rotation[3],
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
          let _ = scene.add_component(
            entity,
            CameraComponent {
              projection,
              focus_distance: 10.0,
            },
          );
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
          // Dust clusters are not serialized: the restored system starts empty and regrows.
          scene.with_component_mut(entity, |ps: &mut ParticleSystemComponent| {
            ps.emission_params = psc.emission_params;
            ps.draw_params = ParticleSystemDrawParams {
              stream_color: psc.stream_color,
            };
            ps.ttl_us = psc.ttl_us;
            ps.dust.get_mut().reset();
          });
        }

        SerializedComponent::AlmanacPlanet(sp) => {
          let _ = scene.add_component(entity, AlmanacPlanet::new(sp.naif_id));
        }

        // Marker components carry no data — their presence is already baked into the scene
        // hierarchy from `create_empty_scene2`. We only need to track sun entity IDs for cleanup.
        SerializedComponent::SunMarker => {
          // legacy (v1) dumps: keep the live SunComponent, a zeroed one made the Sun vanish
          new_sun_entity_ids.push(entity);
        }
        SerializedComponent::Sun(s) => {
          new_sun_entity_ids.push(entity);
          let _ = scene.add_component(
            entity,
            SunComponent {
              radius_km: s.radius_km,
              resolution: s.resolution,
            },
          );
        }
        SerializedComponent::BodyRotationalModel(m) => {
          let _ = scene.add_component(
            entity,
            BodyRotationalModel {
              pole_ra: m.pole_ra,
              pole_dec: m.pole_dec,
              prime_meridian: m.prime_meridian,
              pole_ra_rate: m.pole_ra_rate,
              pole_dec_rate: m.pole_dec_rate,
              rotation_rate: m.rotation_rate,
              body_fixed_orientation: m.body_fixed_orientation,
            },
          );
        }
        SerializedComponent::Trajectory(t) => {
          let mut c = TrajectoryComponent::new(
            t.control_points.clone(),
            t.color,
            t.line_width,
            t.texture_id,
            t.subdivisions_per_segment,
          );
          c.control_points_f64 = t.control_points_f64.clone();
          let _ = scene.add_component(entity, c);
        }
        SerializedComponent::ReferenceFrame(f) => {
          let _ = scene.add_component(
            entity,
            ReferenceFrameComponent {
              frame_type: if f.frame_type == 0 {
                ReferenceFrameType::Macro
              } else {
                ReferenceFrameType::Micro
              },
              scale: f.scale,
              soi_radius: f.soi_radius,
              depth_layer: f.depth_layer,
            },
          );
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

  DeserializeResult {
    new_mesh_hashes,
    new_sun_entity_ids,
  }
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

/// Compatibility "hash" of a scene: a deterministic JSON object of everything that shapes the
/// simulation, so a dump is only restored into a scene configured the same way (and, later, so a
/// cached state can be looked up by configuration). Floats use Rust's shortest round-trip
/// formatting; jets are ordered by entity id. Drawing options (stream colour, labels, camera) are
/// not part of it.
pub fn compatibility_json(
  scene_ctx: &crate::simulation_api::structs::SceneContext,
  start: hifitime::Epoch,
  end: hifitime::Epoch,
) -> alloc::string::String {
  use alloc::{format, string::String};
  let scene = &scene_ctx.scene;
  let parts = |e: hifitime::Epoch| e.to_tdb_duration().to_parts();
  let (sc, sn) = parts(start);
  let (ec, en) = parts(end);
  let comet = scene_ctx.comet;
  let naif = comet
    .and_then(|c| scene.with_component(c.body, |p: &AlmanacPlanet| p.naif_id))
    .map_or(String::from("null"), |id| format!("{id}"));
  let radius = comet
    .and_then(|c| {
      scene.with_component(c.body, |g: &crate::scene::SphereGizmoComponent| {
        g.radius as f64 * 0.5
      })
    })
    .map_or(String::from("null"), |r| format!("{r:?}"));
  let rotation = comet
    .and_then(|c| scene.with_component(c.body, |m: &BodyRotationalModel| *m))
    .map_or(String::from("null"), |m| {
      format!(
        "[{:?},{:?},{:?},{:?},{:?},{:?},{}]",
        m.pole_ra,
        m.pole_dec,
        m.prime_meridian,
        m.pole_ra_rate,
        m.pole_dec_rate,
        m.rotation_rate,
        m.body_fixed_orientation
      )
    });
  let elements = scene_ctx.comet_reference_elements.map_or(String::from("null"), |e| {
    format!(
      "[{:?},{:?},{:?},{:?},{:?},{:?}]",
      e.eccentricity,
      e.perihelion_distance_au,
      e.inclination_deg,
      e.longitude_of_ascending_node_deg,
      e.argument_of_perihelion_deg,
      // NaN is not JSON: unknown time of perihelion -> null
      e.time_of_perihelion_jd_tdb
    )
    .replace("NaN", "null")
  });
  let mut jets = alloc::vec::Vec::new();
  scene.query1(|id, ps: &ParticleSystemComponent| {
    jets.push((id.as_ffi(), format!("{:?}", ps.emission_params), ps.ttl_us));
  });
  jets.sort_by_key(|j| j.0);
  let jets: alloc::vec::Vec<String> = jets
    .into_iter()
    .map(|(id, p, ttl)| {
      format!(
        "{{\"id\":{id},\"ttl_us\":{ttl},\"params\":\"{}\"}}",
        p.replace('"', "'")
      )
    })
    .collect();
  format!(
    "{{\"version\":{},\"start\":[{sc},{sn}],\"end\":[{ec},{en}],\"comet_naif\":{naif},\"nucleus_radius_km\":{radius},\"rotation\":{rotation},\"reference_elements\":{elements},\"dust_tiers\":{},\"jets\":[{}]}}",
    crate::simulation_api::structs::SceneDump::CURRENT_VERSION,
    // the age tiers decide which dust history a restore reproduces
    crate::scene::dust::dust_tier_count(),
    jets.join(",")
  )
}
