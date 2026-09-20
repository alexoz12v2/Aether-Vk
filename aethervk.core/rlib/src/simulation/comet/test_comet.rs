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

// ---------------------------------------------------------------------------
// Helper: compute the face normal of triangle (A, B, C) as cross(B−A, C−A).
// Returns (nx, ny, nz) – NOT normalised, magnitude = 2 × triangle area.
// ---------------------------------------------------------------------------
fn face_normal(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [f32; 3] {
  let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  [
    ab[1] * ac[2] - ab[2] * ac[1],
    ab[2] * ac[0] - ab[0] * ac[2],
    ab[0] * ac[1] - ab[1] * ac[0],
  ]
}

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn len3(v: [f32; 3]) -> f32 {
  dot3(v, v).sqrt()
}

/// For a UV sphere centred at the origin the centroid of every triangle lies
/// at some positive radius, so `dot(face_normal, centroid)` should be > 0
/// (outward-pointing normal).  Returns a list of (triangle_index, dot_value)
/// for any triangle that fails.
fn check_uv_sphere_winding(mesh: &Comet) -> Vec<(usize, f32)> {
  let mut failures = Vec::new();
  for (tri_idx, chunk) in mesh.indices.chunks_exact(3).enumerate() {
    let a = mesh.vertices[chunk[0] as usize].position;
    let b = mesh.vertices[chunk[1] as usize].position;
    let c = mesh.vertices[chunk[2] as usize].position;

    let fn_ = face_normal(a, b, c);

    // Triangle centroid
    let centroid = [
      (a[0] + b[0] + c[0]) / 3.0,
      (a[1] + b[1] + c[1]) / 3.0,
      (a[2] + b[2] + c[2]) / 3.0,
    ];

    // Skip degenerate triangles (pole caps can be degenerate due to seam)
    if len3(fn_) < 1e-6 {
      continue;
    }

    let d = dot3(fn_, centroid);
    if d <= 0.0 {
      failures.push((tri_idx, d));
    }
  }
  failures
}

// ---------------------------------------------------------------------------
// Test 1: winding order – all triangles face outward (normal·centroid > 0).
// ---------------------------------------------------------------------------
#[test]
fn test_uv_sphere_winding_is_consistently_outward() {
  for &(lat, lon) in &[
    (4u32, 4u32),
    (8, 8),
    (16, 16),
    (36, 36), // matches the render-time subdivision used in production
    (2, 4),
    (4, 8),
  ] {
    let mesh = generate_uv_sphere(1.0, lat, lon, 1.0, false);
    let failures = check_uv_sphere_winding(&mesh);
    assert!(
      failures.is_empty(),
      "generate_uv_sphere(lat={lat}, lon={lon}): \
       {count} triangle(s) have inward-pointing normals. \
       First failing triangles (index, dot): {first:?}",
      count = failures.len(),
      first = &failures[..failures.len().min(5)],
    );
  }
}

// ---------------------------------------------------------------------------
// Test 2: flip_winding=true inverts every triangle relative to flip_winding=false.
// ---------------------------------------------------------------------------
#[test]
fn test_uv_sphere_flip_winding_inverts_all_triangles() {
  let normal_mesh = generate_uv_sphere(1.0, 8, 8, 1.0, false);
  let flipped_mesh = generate_uv_sphere(1.0, 8, 8, 1.0, true);

  assert_eq!(
    normal_mesh.indices.len(),
    flipped_mesh.indices.len(),
    "flip_winding must not change the triangle count"
  );

  for (tri_idx, (n_chunk, f_chunk)) in normal_mesh
    .indices
    .chunks_exact(3)
    .zip(flipped_mesh.indices.chunks_exact(3))
    .enumerate()
  {
    // Flipped winding = reversed vertex order for the same triangle,
    // so flipped[0]==normal[0], flipped[1]==normal[2], flipped[2]==normal[1]
    // OR any rotation thereof that reverses orientation.
    // The simplest invariant: face normals should be opposite sign.
    let pos = |chunk: &[u32], i: usize| normal_mesh.vertices[chunk[i] as usize].position;

    let fn_normal = face_normal(pos(n_chunk, 0), pos(n_chunk, 1), pos(n_chunk, 2));
    let fn_flipped = face_normal(pos(f_chunk, 0), pos(f_chunk, 1), pos(f_chunk, 2));

    // Skip degenerate triangles
    if len3(fn_normal) < 1e-6 || len3(fn_flipped) < 1e-6 {
      continue;
    }

    let d = dot3(fn_normal, fn_flipped);
    assert!(
      d < 0.0,
      "Triangle {tri_idx}: flip_winding=true should invert the face normal, \
       but dot(normal_face_normal, flipped_face_normal) = {d:.4} (expected < 0)"
    );
  }
}

// ---------------------------------------------------------------------------
// Test 3: vertex normals (stored per-vertex) agree with the face normal of
//         every triangle they belong to (all should have positive dot product).
// ---------------------------------------------------------------------------
#[test]
fn test_uv_sphere_vertex_normals_agree_with_face_normals() {
  let mesh = generate_uv_sphere(1.0, 16, 16, 1.0, false);

  for (tri_idx, chunk) in mesh.indices.chunks_exact(3).enumerate() {
    let va = mesh.vertices[chunk[0] as usize];
    let vb = mesh.vertices[chunk[1] as usize];
    let vc = mesh.vertices[chunk[2] as usize];

    let fn_ = face_normal(va.position, vb.position, vc.position);
    if len3(fn_) < 1e-6 {
      continue; // degenerate
    }

    // Average stored vertex normal over the triangle
    let avg_stored_n = [
      (va.normal[0] + vb.normal[0] + vc.normal[0]) / 3.0,
      (va.normal[1] + vb.normal[1] + vc.normal[1]) / 3.0,
      (va.normal[2] + vb.normal[2] + vc.normal[2]) / 3.0,
    ];

    let d = dot3(fn_, avg_stored_n);
    assert!(
      d > 0.0,
      "Triangle {tri_idx}: stored vertex normals disagree with face normal \
       (dot = {d:.4}). Winding and/or stored normals are inconsistent.",
    );
  }
}

// ---------------------------------------------------------------------------
// Test 4: index count matches the expected formula.
//   Triangles = 2 * lat_segments * lon_segments  - 2 * lon_segments
//             = 2 * lon_segments * (lat_segments - 1)
//   Indices   = 3 × triangles
// ---------------------------------------------------------------------------
#[test]
fn test_uv_sphere_index_count() {
  for &(lat, lon) in &[(4u32, 4u32), (8, 8), (16, 16), (36, 36)] {
    let mesh = generate_uv_sphere(1.0, lat, lon, 1.0, false);
    let expected = 3 * 2 * lon * (lat - 1);
    assert_eq!(
      mesh.indices.len() as u32,
      expected,
      "generate_uv_sphere(lat={lat}, lon={lon}): \
       expected {expected} indices, got {}",
      mesh.indices.len()
    );
  }
}