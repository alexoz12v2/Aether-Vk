use super::*;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;

#[test]
fn test_uv_sphere_generation() {
  let sphere = generate_uv_sphere(2.0, 10, 10, 1.0, false);
  let expected_indices = 6 * 10 * (10 - 1);

  // Check if the number of vertices and indices is correct
  assert_eq!(sphere.vertices.len(), (10 + 1) * (10 + 1));
  assert_eq!(sphere.indices.len(), expected_indices);

  // Check that vertices are roughly distance 2.0 from origin
  for v in sphere.vertices.iter() {
    let pos = Vec3f32::from_components(v.position[0], v.position[1], v.position[2]);
    assert!((pos.length() - 2.0).abs() < 1e-5);
  }
}

#[test]
fn test_comet_glb_loading() {
  crate::gpu::set_asset_dir_for_tests();
  let assets_dir = std::path::PathBuf::from(crate::gpu::ASSET_DIR.read().as_ref().unwrap());
  let model_dir = assets_dir.join("Comet.glb");
  if model_dir.is_file() {
    let comet =
      load_comet_from_gltf(model_dir.to_str().unwrap(), false, None).expect("Failed to load comet");
    assert!(comet.vertices.len() > 0);
    assert!(comet.indices.len() > 0);
  }
}

#[test]
fn test_comet_obj_loading() {
  let obj_content = "
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.0 1.0 0.0
v 0.0 0.0 1.0
f 1 3 2
f 1 2 4
f 1 4 3
f 2 3 4
";
  let tmp_path = std::env::temp_dir().join("test.obj");
  std::fs::write(&tmp_path, obj_content).unwrap();
  let comet = load_comet_from_obj(tmp_path.to_str().unwrap(), false, None).unwrap();
  assert_eq!(comet.vertices.len(), 4);
  assert_eq!(comet.indices.len(), 12);
}

#[test]
fn test_comet_ply_loading() {
  let ply_content = "ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 4
property list uchar uint vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
3 0 2 1
3 0 1 3
3 0 3 2
3 1 2 3
";
  let tmp_path = std::env::temp_dir().join("test.ply");
  std::fs::write(&tmp_path, ply_content).unwrap();
  let comet = load_comet_from_ply(tmp_path.to_str().unwrap(), false, None).unwrap();
  assert_eq!(comet.vertices.len(), 4);
  assert_eq!(comet.indices.len(), 12);
}
