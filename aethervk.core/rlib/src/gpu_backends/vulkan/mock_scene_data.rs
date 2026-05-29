use rand::Rng;
use std::vec::Vec;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SparseCollisionData {
  pub valid: u32,
  pub prim_a: u32,
  pub prim_b: u32,
  pub toi: f32,
  pub contact_normal: [f32; 3],
  pub _pad0: f32,
  pub contact_point: [f32; 3],
  pub penetration_depth: f32,
}

pub fn generate_sparse_collisions(count: usize, valid_ratio: f32) -> Vec<SparseCollisionData> {
  let mut rng = rand::thread_rng();
  let mut data = Vec::with_capacity(count);
  for _ in 0..count {
    let valid = if rng.r#gen::<f32>() < valid_ratio {
      1
    } else {
      0
    };
    data.push(SparseCollisionData {
      valid,
      prim_a: rng.r#gen::<u32>() % 1000,
      prim_b: rng.r#gen::<u32>() % 1000,
      toi: rng.r#gen::<f32>(),
      contact_normal: [0.0, 1.0, 0.0],
      _pad0: 0.0,
      contact_point: [0.0, 0.0, 0.0],
      penetration_depth: rng.r#gen::<f32>(),
    });
  }
  data
}

pub fn generate_mock_particles(count: usize) -> Vec<f32> {
  let mut data = vec![0.0; count * 10];
  let mut rng = rand::thread_rng();
  for i in 0..count {
    let base = i * 10;
    data[base + 0] = rng.gen_range(-100.0..100.0); // x
    data[base + 1] = rng.gen_range(-100.0..100.0); // y
    data[base + 2] = rng.gen_range(-100.0..100.0); // z
  }
  data
}
