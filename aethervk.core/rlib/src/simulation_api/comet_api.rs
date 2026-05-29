use crate::{
  expect_scene, expect_scene_and_entity,
  scene::{
    PhysicalMeshComponent, SphereGizmoComponent, TransformComponent, particles::JetComponent,
  },
  simulation_api::SimulationContext,
  types::{EngineError, EngineResult},
};
use aethervk_oshal_rlib as oshal;
use oshal::math::{
  matrix::{Matrix, SquareMatrix, mat3::Mat3f32, mat4::Mat4x4f32},
  quaternion::Quaternion,
  vector::{Vector, Vector3, vec3::Vec3f32, vec4::Quat},
};

pub trait CometApi {
  #[allow(clippy::too_many_arguments)]
  fn add_jet(
    &self,
    scene_id: u64,
    comet_id: u64,
    radius_km: f32,
    lat: f32,
    lon: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    mass: f32,
    particles_per_tick: u32,
    ttl: f32,
    mean_velocity: f32,
  ) -> EngineResult<u64>;
}

impl CometApi for SimulationContext {
  /// TODO: Document this item
  fn add_jet(
    &self,
    scene_id: u64,
    comet_id: u64,
    radius_km: f32,
    lat: f32,
    lon: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    mass: f32,
    particles_per_tick: u32,
    ttl: f32,
    mean_velocity: f32,
  ) -> EngineResult<u64> {
    let scenes = self.scenes.read();
    let (scene, entity) =
      expect_scene_and_entity!(scenes.get_scene(scene_id), comet_id, "comet_api:add_jet");
    let mut write_scene = scene.write();

    let mesh_arc = write_scene
      .scene
      .with_component(entity, |p: &PhysicalMeshComponent| p.mesh.clone())
      .ok_or(EngineError::InvalidOperation(
        "comet_api:add_jet | comet entity missing PhysicalMeshComponent",
      ))?;

    let bounding_sphere = mesh_arc
      .vertices
      .iter()
      .map(|v| {
        v.position[0] * v.position[0]
          + v.position[1] * v.position[1]
          + v.position[2] * v.position[2]
      })
      .fold(0.0_f32, f32::max)
      .sqrt();

    // Direction in User Local Frame
    let dir_z = lat.sin();
    let dir_x = lat.cos() * lon.cos();
    let dir_y = lat.cos() * lon.sin();
    let dir_user = Vec3f32::from_components(dir_x, dir_y, dir_z).normalize();

    let pa_basis_bf = mesh_arc.pa_basis_bf.unwrap_or_else(|| Mat3f32 {
      x: Vec3f32::from_components(1.0, 0.0, 0.0),
      y: Vec3f32::from_components(0.0, 1.0, 0.0),
      z: Vec3f32::from_components(0.0, 0.0, 1.0),
    });

    let dir_sim =
      (pa_basis_bf.x * dir_user.x() + pa_basis_bf.y * dir_user.y() + pa_basis_bf.z * dir_user.z())
        .normalize();

    let ray_orig = dir_sim * (bounding_sphere * 2.0);
    let ray_dir = Vec3f32::from_components(-dir_sim.x(), -dir_sim.y(), -dir_sim.z());

    let mut cached_emission_points = alloc::vec::Vec::new();
    let num_triangles = mesh_arc.indices.len() / 3;
    for i in 0..num_triangles {
      let i0 = mesh_arc.indices[i * 3] as usize;
      let i1 = mesh_arc.indices[i * 3 + 1] as usize;
      let i2 = mesh_arc.indices[i * 3 + 2] as usize;
      let v0 = Vec3f32::from_array(mesh_arc.vertices[i0].position);
      let v1 = Vec3f32::from_array(mesh_arc.vertices[i1].position);
      let v2 = Vec3f32::from_array(mesh_arc.vertices[i2].position);

      let edge1 = v1 - v0;
      let edge2 = v2 - v0;
      let h = ray_dir.cross(edge2);
      let a = edge1.dot(h);
      if a.abs() < 1e-6 {
        continue;
      }
      let f = 1.0 / a;
      let s = ray_orig - v0;
      let u = f * s.dot(h);
      if u < 0.0 || u > 1.0 {
        continue;
      }
      let q = s.cross(edge1);
      let v = f * ray_dir.dot(q);
      if v < 0.0 || u + v > 1.0 {
        continue;
      }
      let t = f * edge2.dot(q);

      if t > 1e-6 {
        cached_emission_points.push(ray_orig + ray_dir * t);
      }
    }

    let jet_entity_id = write_scene.scene.spawn_entity("jet");

    write_scene
      .scene
      .add_component(
        jet_entity_id,
        SphereGizmoComponent {
          radius: radius_km,
          subdivisions: 4.0,
          local_frame: Mat4x4f32::identity(),
          is_visible: true,
        },
      )
      .map_err(|e| EngineError::from(e))?;

    write_scene
      .scene
      .add_component(
        jet_entity_id,
        JetComponent {
          radius: radius_km,
          lat,
          lon,
          color: [color_r, color_g, color_b, 1.0],
          mass,
          particles_per_tick,
          ttl,
          mean_velocity,
          cached_emission_points,
        },
      )
      .map_err(|e| EngineError::from(e))?;

    write_scene
      .scene
      .add_component(
        jet_entity_id,
        TransformComponent {
          position: Vec3f32::from_components(0.0, 0.0, 0.0),
          rotation: Quat::identity(),
          scale: Vec3f32::from_components(1.0, 1.0, 1.0),
        },
      )
      .map_err(|e| EngineError::from(e))?;

    write_scene.scene.set_parent(jet_entity_id, Some(entity));
    let jet_ext_id = write_scene.register_entity(jet_entity_id);

    Ok(jet_ext_id)
  }
}
