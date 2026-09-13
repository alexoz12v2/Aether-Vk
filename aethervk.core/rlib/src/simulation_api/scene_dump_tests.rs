//! Unit tests for `simulation_api::scene_dump`.
//!
//! All tests are purely CPU-side — no Vulkan, no GPU, no logic thread.

#[cfg(test)]
mod scene_dump_tests {
  use alloc::sync::Arc;

  use bytes::Bytes;

  use crate::{
    scene::{
      AssetCache, HighResTransformComponent, Scene, StaticMeshComponent, TransformComponent,
    },
    simulation::comet::{Comet, TexelFormat, Texture, Vertex, next_comet_id},
    simulation_api::{
      scene_dump::{deserialize_scene, serial_to_texture, serialize_scene, texture_to_serial},
      structs::{SceneDump, SerializedComponent, SerializedEntity, SerializedTransform},
    },
  };

  /// Create a `Scene` with all crate components registered, backed by a fresh texture cache.
  fn make_test_scene() -> Scene {
    use crate::simulation::texture_cache::TextureCache;
    use parking_lot::RwLock;

    let texture_cache = Arc::new(RwLock::new(TextureCache::new("")));
    let scene = Scene::new(texture_cache);
    scene.register_all_crate_components();
    scene
  }

  // ── Test 1 ─────────────────────────────────────────────────────────────────

  /// A procedural mesh (empty `asset_path`) fully round-trips: vertices, indices,
  /// tangent (4-component!), `emissive_color`, and `is_visible` are all preserved.
  #[test]
  fn test_serialize_procedural_mesh_round_trip() {
    let mut scene = make_test_scene();
    let entity = scene.spawn_entity("comet");

    let comet = Arc::new(Comet {
      id: next_comet_id(),
      vertices: vec![
        Vertex {
          position: [1.0, 2.0, 3.0],
          normal: [0.0, 1.0, 0.0],
          uv: [0.5, 0.5],
          tangent: [1.0, 0.0, 0.0, 1.0],
        },
        Vertex {
          position: [4.0, 5.0, 6.0],
          normal: [0.0, 0.0, 1.0],
          uv: [0.0, 1.0],
          tangent: [0.0, 1.0, 0.0, -1.0],
        },
      ],
      indices: vec![0, 1, 0],
      albedo_map: None,
      normal_map: None,
      roughness_map: None,
      ao_map: None,
    });

    scene
      .add_component(
        entity,
        StaticMeshComponent {
          asset_path: String::new(),
          mesh: comet,
          emissive_color: [1.0, 0.5, 0.0, 1.0],
          is_visible: true,
        },
      )
      .expect("add StaticMeshComponent");

    let serialized = serialize_scene(&scene);
    assert_eq!(serialized.len(), 1, "one entity expected");

    // Inline geometry must be present for procedural meshes.
    let sm = serialized[0]
      .components
      .iter()
      .find_map(|c| {
        if let SerializedComponent::StaticMesh(m) = c {
          Some(m)
        } else {
          None
        }
      })
      .expect("StaticMesh component");
    assert!(
      sm.vertices.is_some(),
      "procedural mesh must inline geometry"
    );
    let verts = sm.vertices.as_ref().unwrap();
    assert_eq!(verts.len(), 2);
    assert_eq!(verts[0].position, [1.0_f32, 2.0, 3.0]);
    assert_eq!(verts[0].tangent, [1.0_f32, 0.0, 0.0, 1.0]);
    assert_eq!(verts[1].tangent, [0.0_f32, 1.0, 0.0, -1.0]);

    // Deserialize into a structure-only clone.
    let mut restored = scene.clone_structure_only();
    let cache = AssetCache::new();
    let result = deserialize_scene(&mut restored, &serialized, &cache);

    assert!(
      !result.new_mesh_hashes.is_empty(),
      "at least one mesh hash expected"
    );

    restored
      .with_component(entity, |m: &StaticMeshComponent| {
        assert!(m.asset_path.is_empty());
        assert_eq!(m.mesh.vertices.len(), 2);
        assert_eq!(m.mesh.vertices[0].position, [1.0, 2.0, 3.0]);
        assert_eq!(m.mesh.vertices[0].tangent, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(m.mesh.indices, vec![0, 1, 0]);
        assert_eq!(m.emissive_color, [1.0, 0.5, 0.0, 1.0]);
        assert!(m.is_visible);
      })
      .expect("StaticMeshComponent must exist after restore");
  }

  // ── Test 2 ─────────────────────────────────────────────────────────────────

  /// Asset-backed mesh (non-empty `asset_path` present in cache): no inline geometry is
  /// serialized, and deserialization returns the **same** Arc (same `Comet::id`).
  #[test]
  fn test_serialize_asset_backed_mesh_reuses_cache() {
    let mut scene = make_test_scene();
    let entity = scene.spawn_entity("planet");

    let original_id = next_comet_id();
    let comet = Comet {
      id: original_id,
      vertices: vec![Vertex {
        position: [0.0; 3],
        normal: [0.0, 1.0, 0.0],
        uv: [0.0; 2],
        tangent: [1.0, 0.0, 0.0, 1.0],
      }],
      indices: vec![0],
      albedo_map: None,
      normal_map: None,
      roughness_map: None,
      ao_map: None,
    };

    let cache = AssetCache::new();
    let cached_arc = cache.insert("__default_comet__".to_string(), comet);

    scene
      .add_component(
        entity,
        StaticMeshComponent {
          asset_path: "__default_comet__".to_string(),
          mesh: cached_arc,
          emissive_color: [0.0; 4],
          is_visible: true,
        },
      )
      .expect("add StaticMeshComponent");

    let serialized = serialize_scene(&scene);

    // Asset-backed mesh must NOT inline geometry.
    let sm = serialized[0]
      .components
      .iter()
      .find_map(|c| {
        if let SerializedComponent::StaticMesh(m) = c {
          Some(m)
        } else {
          None
        }
      })
      .expect("StaticMesh component");
    assert!(
      sm.vertices.is_none(),
      "asset-backed mesh must NOT inline geometry"
    );
    assert!(sm.indices.is_none());
    assert_eq!(sm.asset_path, "__default_comet__");

    // Deserialize: should reuse the cached Arc (same id, no new allocation).
    let mut restored = scene.clone_structure_only();
    let result = deserialize_scene(&mut restored, &serialized, &cache);

    // Original content hash must be reported (cache hit).
    assert!(
      result.new_mesh_hashes.contains(&original_id),
      "original hash must be tracked"
    );

    restored
      .with_component(entity, |m: &StaticMeshComponent| {
        assert_eq!(
          m.mesh.id, original_id,
          "must reuse cached Arc, not allocate a new id"
        );
      })
      .expect("StaticMeshComponent must exist after restore");
  }

  // ── Test 3 ─────────────────────────────────────────────────────────────────

  /// `format` and `has_mipmaps` survive `texture_to_serial` → `serial_to_texture`.
  /// Tests UNORM, BC7 compressed with mipmaps, and `Unsupported(u32)` variants.
  #[test]
  fn test_serialize_texture_format_round_trip() {
    // ── R8G8B8A8_UNORM, no mips
    let rgba = Texture {
      data: Bytes::from(vec![255u8; 16]),
      format: TexelFormat::R8G8B8A8_UNORM,
      width: 2,
      height: 2,
      has_mipmaps: false,
    };
    let serial_rgba = texture_to_serial(&rgba);
    assert_eq!(serial_rgba.format, TexelFormat::R8G8B8A8_UNORM);
    assert!(!serial_rgba.has_mipmaps);
    assert_eq!(serial_rgba.width, 2);
    assert_eq!(serial_rgba.height, 2);

    let back_rgba = serial_to_texture(&serial_rgba);
    assert_eq!(back_rgba.format, TexelFormat::R8G8B8A8_UNORM);
    assert_eq!(back_rgba.width, 2);
    assert_eq!(back_rgba.height, 2);
    assert!(!back_rgba.has_mipmaps);
    assert_eq!(back_rgba.data.as_ref(), &[255u8; 16]);

    // ── BC7 compressed with mipmaps
    let bc7 = Texture {
      data: Bytes::from(vec![0u8; 16]),
      format: TexelFormat::BC7_UNORM_BLOCK,
      width: 4,
      height: 4,
      has_mipmaps: true,
    };
    let serial_bc7 = texture_to_serial(&bc7);
    assert_eq!(serial_bc7.format, TexelFormat::BC7_UNORM_BLOCK);
    assert!(serial_bc7.has_mipmaps);

    let back_bc7 = serial_to_texture(&serial_bc7);
    assert_eq!(back_bc7.format, TexelFormat::BC7_UNORM_BLOCK);
    assert!(back_bc7.has_mipmaps);

    // ── Unsupported — raw Vulkan format value must survive round-trip
    let unsup = Texture {
      data: Bytes::from(vec![0u8; 4]),
      format: TexelFormat::Unsupported(1337),
      width: 1,
      height: 1,
      has_mipmaps: false,
    };
    let serial_unsup = texture_to_serial(&unsup);
    assert_eq!(serial_unsup.format, TexelFormat::Unsupported(1337));

    let back_unsup = serial_to_texture(&serial_unsup);
    assert_eq!(back_unsup.format, TexelFormat::Unsupported(1337));
  }

  // ── Test 4 ─────────────────────────────────────────────────────────────────

  /// `HighResTransformComponent` (f64 AU-scale) and `TransformComponent` (f32)
  /// positions survive the full serialize / deserialize cycle without precision loss.
  #[test]
  fn test_serialize_transforms_round_trip() {
    use aethervk_oshal_rlib::math::quaternion::Quaternion;
    use aethervk_oshal_rlib::math::vector::{
      Vector, Vector3, vec3::Vec3f32, vec3f64::Vec3f64, vec4::Quat,
    };

    let mut scene = make_test_scene();
    let entity = scene.spawn_entity("body");

    scene
      .add_component(
        entity,
        HighResTransformComponent {
          position: Vec3f64::from_components(1.496e11_f64, -2.3e10_f64, 5.7e9_f64),
          rotation: Quat::identity(),
          scale: Vec3f32::one(),
        },
      )
      .expect("add HighResTransformComponent");

    scene
      .add_component(
        entity,
        TransformComponent {
          position: Vec3f32::from_components(1.0, -2.0, 3.5),
          rotation: Quat::identity(),
          scale: Vec3f32::one(),
        },
      )
      .expect("add TransformComponent");

    let serialized = serialize_scene(&scene);
    let cache = AssetCache::new();
    let mut restored = scene.clone_structure_only();
    deserialize_scene(&mut restored, &serialized, &cache);

    restored
      .with_component(entity, |t: &HighResTransformComponent| {
        assert_eq!(t.position.x(), 1.496e11_f64, "x AU position");
        assert_eq!(t.position.y(), -2.3e10_f64, "y AU position");
        assert_eq!(t.position.z(), 5.7e9_f64, "z AU position");
      })
      .expect("HighResTransformComponent must exist after restore");

    restored
      .with_component(entity, |t: &TransformComponent| {
        assert_eq!(t.position.x(), 1.0_f32);
        assert_eq!(t.position.y(), -2.0_f32);
        assert_eq!(t.position.z(), 3.5_f32);
      })
      .expect("TransformComponent must exist after restore");
  }

  // ── Test 5 ─────────────────────────────────────────────────────────────────

  /// Full bincode encode → decode cycle for `SceneDump` (file format regression test).
  /// Verifies all top-level fields survive serialization without loss.
  #[test]
  fn test_bincode_scene_dump_encode_decode() {
    let dump = SceneDump {
      version: SceneDump::CURRENT_VERSION,
      scene_id: 42,
      start_epoch_parts: (0, 0),
      end_epoch_parts: (0, 86_400_000_000_000),
      entities: vec![SerializedEntity {
        ffi_id: 999,
        name: "earth".to_string(),
        parent_ffi_id: None,
        components: vec![SerializedComponent::Transform(SerializedTransform {
          position: [1.0, 2.0, 3.0],
          rotation: [0.0, 0.0, 0.0, 1.0],
          scale: [1.0, 1.0, 1.0],
        })],
      }],
      particle_snapshot: None,
    };

    let bytes = bincode::serde::encode_to_vec(&dump, bincode::config::standard()).expect("encode");
    assert!(!bytes.is_empty(), "encoded bytes must not be empty");

    let (decoded, consumed): (SceneDump, usize) =
      bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).expect("decode");

    assert_eq!(consumed, bytes.len(), "all bytes must be consumed");
    assert_eq!(decoded.version, SceneDump::CURRENT_VERSION);
    assert_eq!(decoded.scene_id, 42);
    assert_eq!(decoded.start_epoch_parts, (0, 0));
    assert_eq!(decoded.end_epoch_parts, (0, 86_400_000_000_000));
    assert_eq!(decoded.entities.len(), 1);
    assert_eq!(decoded.entities[0].ffi_id, 999);
    assert_eq!(decoded.entities[0].name, "earth");
    assert!(decoded.entities[0].parent_ffi_id.is_none());
    assert!(decoded.particle_snapshot.is_none());

    match &decoded.entities[0].components[0] {
      SerializedComponent::Transform(t) => {
        assert_eq!(t.position, [1.0, 2.0, 3.0]);
        assert_eq!(t.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(t.scale, [1.0, 1.0, 1.0]);
      }
      other => panic!("unexpected component: {:?}", other),
    }
  }
}
