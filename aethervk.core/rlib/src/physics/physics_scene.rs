//! physics_scene module.

use crate::{
  math::collision::{
    bounds::AABB,
    linear_bvh::{LinearBVH, LinearBVHHeader, LinearBVHNode, LinearBound},
  },
  scene::{EntityId, Scene, TransformComponent},
};
use aethervk_oshal_rlib::math::{
  matrix::{Matrix4, mat4::Mat4x4f32},
  vector::{Vector, Vector3, vec3::Vec3f32},
};
use alloc::vec::Vec;

pub mod math;

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuBvhNode {
  pub aabb_min: [f32; 3],
  pub left_child_or_prim: u32,
  pub aabb_max: [f32; 3],
  pub right_child_offset: u32,
  pub prim_count: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuReferenceFrame {
  pub center_pos: [f32; 3],
  pub scale: f32,
  pub center_vel: [f32; 3],
  pub soi_radius: f32,
  pub frame_type: u32,
  pub parent_frame_idx: u32,
  pub bvh_root_index: u32,
  pub _pad0: u32,
  pub entity_id_raw: u64,
  pub frame_bda: u64,
}

// Metadata helpers for TLAS
pub const BVH_FRAME_MACRO: u32 = 0;
pub const BVH_FRAME_MICRO: u32 = 1;
pub const BVH_SHAPE_AABB: u32 = 0;
pub const BVH_SHAPE_OBB: u32 = 1;
pub const BVH_SHAPE_SPHERE: u32 = 2;
pub const BVH_SHAPE_SUB_TLAS: u32 = 3;

pub fn pack_meta(is_leaf: bool, frame: u32, shape: u32, index: u32) -> u32 {
  let mut m = index & 0x07FF_FFFF;
  m |= (shape & 0x3) << 27;
  m |= (frame & 0x3) << 29;
  if is_leaf {
    m |= 0x8000_0000;
  }
  m
}

#[derive(Debug)]
pub struct RbNode {
  pub min: Vec3f32,
  pub max: Vec3f32,
  pub left: Option<u32>,
  pub right: Option<u32>,
  pub leaf_meta: Option<u32>,
  pub leaf_child_idx: u32,
}

#[derive(Debug)]
pub struct RootBoundsBvh {
  pub nodes: Vec<RbNode>,
}

impl RootBoundsBvh {
  pub fn build(leaves: &[(u32, Vec3f32, Vec3f32, u32, u32)]) -> Self {
    let mut nodes = Vec::new();
    if !leaves.is_empty() {
      let mut items: Vec<usize> = (0..leaves.len()).collect();
      Self::build_recursive(&mut items, leaves, &mut nodes);
    }
    Self { nodes }
  }

  pub fn build_recursive(
    items: &mut [usize],
    leaves: &[(u32, Vec3f32, Vec3f32, u32, u32)],
    nodes: &mut Vec<RbNode>,
  ) -> u32 {
    let node_idx = nodes.len() as u32;
    nodes.push(RbNode {
      min: Vec3f32::zero(),
      max: Vec3f32::zero(),
      left: None,
      right: None,
      leaf_meta: None,
      leaf_child_idx: 0,
    });

    let mut mn = Vec3f32::from_array([f32::INFINITY; 3]);
    let mut mx = Vec3f32::from_array([f32::NEG_INFINITY; 3]);
    for &i in items.iter() {
      mn = mn.min(leaves[i].1);
      mx = mx.max(leaves[i].2);
    }

    if items.len() == 1 {
      let &(root_idx, _, _, shape, frame) = &leaves[items[0]];
      let meta = pack_meta(true, frame, shape, root_idx);
      nodes[node_idx as usize] = RbNode {
        min: mn,
        max: mx,
        left: None,
        right: None,
        leaf_meta: Some(meta),
        leaf_child_idx: root_idx,
      };
      return node_idx;
    }

    let ext = mx - mn;
    let axis = if ext.x() > ext.y() && ext.x() > ext.z() {
      0
    } else if ext.y() > ext.z() {
      1
    } else {
      2
    };
    items.sort_by(|&a, &b| {
      let ca = (leaves[a].1 + leaves[a].2) * 0.5;
      let cb = (leaves[b].1 + leaves[b].2) * 0.5;
      ca[axis].partial_cmp(&cb[axis]).unwrap_or(core::cmp::Ordering::Equal)
    });
    let mid = items.len() / 2;
    let (left_items, right_items) = items.split_at_mut(mid);
    let left = Self::build_recursive(left_items, leaves, nodes);
    let right = Self::build_recursive(right_items, leaves, nodes);

    nodes[node_idx as usize] = RbNode {
      min: mn,
      max: mx,
      left: Some(left),
      right: Some(right),
      leaf_meta: None,
      leaf_child_idx: 0,
    };
    node_idx
  }
}

#[derive(Debug, Clone)]
pub struct CollisionEvent {
  pub entity_a_id: u32,
  pub entity_a_name: alloc::string::String,
  pub entity_b_id: u32,
  pub entity_b_name: alloc::string::String,
  pub contact_point: [f32; 3],
  pub contact_normal: [f32; 3],
  pub penetration_depth: f32,
  pub frame_id: u32,
  pub is_lca: bool,
  pub particle_path_a: Option<alloc::vec::Vec<u32>>,
  pub particle_path_b: Option<alloc::vec::Vec<u32>>,
}

pub struct PhysicsScene {
  pub gpu_frames: Vec<GpuReferenceFrame>,
  pub macro_tlas: RootBoundsBvh,
  pub micro_tlases: hashbrown::HashMap<u32, RootBoundsBvh>,
  pub mesh_blases: Vec<Option<LinearBVH<f32>>>,
  pub particle_blases: Vec<Option<LinearBVH<f32>>>,
  pub mesh_entity_map: Vec<u32>,
  pub particle_entity_map: Vec<u32>,
  pub dt_s: f32,
  pub recent_collisions: Vec<CollisionEvent>,
}

impl PhysicsScene {
  pub fn build_from_scene(scene: &Scene, dt_s: f32) -> Self {
    use crate::scene::{
      KinematicComponent, PhysicalMeshComponent, ReferenceFrameComponent,
      particles::ParticleSystemComponent,
    };
    use slotmap::Key;
    let mut frame_map: hashbrown::HashMap<EntityId, u32> = hashbrown::HashMap::new();
    let mut gpu_frames = Vec::new();

    scene.query2::<TransformComponent, ReferenceFrameComponent, _>(|e, t, f| {
      let vel = scene
        .with_component(e, |k: &KinematicComponent| k.velocity)
        .unwrap_or(Vec3f32::zero());
      let gpu_frame = GpuReferenceFrame {
        center_pos: [t.position.x(), t.position.y(), t.position.z()],
        scale: f.scale,
        center_vel: [vel.x(), vel.y(), vel.z()],
        soi_radius: f.soi_radius,
        frame_type: f.frame_type as u32,
        parent_frame_idx: u32::MAX,
        bvh_root_index: 0,
        _pad0: 0,
        entity_id_raw: slotmap::Key::data(&e).as_ffi(),
        frame_bda: 0,
      };
      frame_map.insert(e, gpu_frames.len() as u32);
      gpu_frames.push(gpu_frame);
    });

    for (entity, idx) in frame_map.iter() {
      let mut curr = *entity;
      while let Some(parent_id) = scene.get_parent(curr) {
        if let Some(&parent_idx) = frame_map.get(&parent_id) {
          gpu_frames[*idx as usize].parent_frame_idx = parent_idx;
          break;
        }
        curr = parent_id;
      }
    }

    if gpu_frames.is_empty() {
      gpu_frames.push(GpuReferenceFrame {
        center_pos: [0.0, 0.0, 0.0],
        scale: 1.0,
        center_vel: [0.0, 0.0, 0.0],
        soi_radius: f32::MAX,
        frame_type: 0,
        parent_frame_idx: u32::MAX,
        bvh_root_index: u32::MAX,
        _pad0: 0,
        entity_id_raw: 0,
        frame_bda: 0,
      });
    }

    let macro_frame_idx = gpu_frames.iter().position(|f| f.frame_type == 0).unwrap_or(0) as u32;

    let mut mesh_blases = Vec::new();
    let mut mesh_entity_map = Vec::new();
    let mut frame_leaves: hashbrown::HashMap<u32, Vec<(u32, Vec3f32, Vec3f32, u32, u32)>> =
      hashbrown::HashMap::new();

    let mut dense_mesh_idx = 0;
    scene.query2_without::<TransformComponent, KinematicComponent, crate::scene::particles::ParticleSystemComponent, _>(|entity, transform, kinematic| {
      let mesh_bvh = scene.with_component(entity, |mesh: &PhysicalMeshComponent| {
        mesh.mesh.bvh.clone()
      }).flatten();

      let lca_idx = find_lca_frame_idx(entity, scene, &gpu_frames).unwrap_or(macro_frame_idx);
      let frame_bits = if lca_idx == macro_frame_idx { BVH_FRAME_MACRO } else { BVH_FRAME_MICRO };

      let motion_blas = if let Some(bvh) = mesh_bvh {
        let b = build_motion_blas(&bvh, transform, kinematic.velocity, dt_s);
        let mut min_bound = Vec3f32::from_array([f32::INFINITY; 3]);
        let mut max_bound = Vec3f32::from_array([f32::NEG_INFINITY; 3]);
        for node in &b.nodes {
          match &node.bound {
            LinearBound::AABB(aabb) => {
              min_bound = min_bound.min(aabb.min());
              max_bound = max_bound.max(aabb.max());
            }
            LinearBound::OBB(_) => {}
          }
        }
        mesh_entity_map.push(entity.data().as_ffi() as u32);
        frame_leaves.entry(lca_idx).or_default().push((entity.data().as_ffi() as u32, min_bound, max_bound, BVH_SHAPE_AABB, frame_bits));
        Some(b)
      } else {
        // Fallback to primitive shape if it's a rigid body
        let collider = scene.with_component(entity, |c: &crate::scene::ColliderComponent| c.clone());
        if let Some(c) = collider {
          let p = transform.position;
          let (shape_type, extents) = match c.shape {
            crate::scene::ColliderShape::Sphere { radius } => (BVH_SHAPE_SPHERE, Vec3f32::from_components(radius, radius, radius)),
            crate::scene::ColliderShape::OBB { half_extents } => (BVH_SHAPE_OBB, half_extents),
          };
          let mut mn = Vec3f32::from_components(p.x() - extents.x(), p.y() - extents.y(), p.z() - extents.z());
          let mut mx = Vec3f32::from_components(p.x() + extents.x(), p.y() + extents.y(), p.z() + extents.z());

          let sweep = kinematic.velocity * dt_s;
          mn = mn.min(mn + sweep);
          mx = mx.max(mx + sweep);

          frame_leaves.entry(lca_idx).or_default().push((dense_mesh_idx, mn, mx, shape_type, frame_bits));
        }
        None
      };
      mesh_blases.push(motion_blas);
      dense_mesh_idx += 1;
    });

    let mut particle_blases = Vec::new();
    let mut particle_entity_map = Vec::new();
    let mut dense_particle_idx = 0;
    scene.query1::<ParticleSystemComponent, _>(|entity, sys| {
      let ps = sys.particles.read();
      let motion_blas = if !ps.is_empty() {
        let transform = scene.global_transform(entity).unwrap_or_default();
        use crate::math::{particles_edu::build_motion_particle_lbvh, physics::Particle};
        let pars: Vec<Particle> = ps
          .iter()
          .map(|p| Particle {
            position: Vec3f32::from_array(p.position),
            velocity: Vec3f32::from_array(p.velocity),
            mass: p.mass,
            accumulated_force: Vec3f32::zero(),
          })
          .collect();
        let vels: Vec<Vec3f32> = ps.iter().map(|p| Vec3f32::from_array(p.velocity)).collect();

        if let Some(lbvh) = build_motion_particle_lbvh(&pars, &vels, 1.0, dt_s) {
          let lca_idx = find_lca_frame_idx(entity, scene, &gpu_frames).unwrap_or(macro_frame_idx);
          let frame_bits = if lca_idx == macro_frame_idx {
            BVH_FRAME_MACRO
          } else {
            BVH_FRAME_MICRO
          };
          let mut min_bound = Vec3f32::from_array([f32::INFINITY; 3]);
          let mut max_bound = Vec3f32::from_array([f32::NEG_INFINITY; 3]);
          for node in &lbvh.nodes {
            match &node.bound {
              LinearBound::AABB(aabb) => {
                min_bound = min_bound.min(aabb.min());
                max_bound = max_bound.max(aabb.max());
              }
              LinearBound::OBB(_) => {}
            }
          }
          particle_entity_map.push(entity.data().as_ffi() as u32);
          frame_leaves.entry(lca_idx).or_default().push((
            entity.data().as_ffi() as u32,
            min_bound,
            max_bound,
            BVH_SHAPE_SPHERE,
            frame_bits,
          ));
          Some(lbvh)
        } else {
          None
        }
      } else {
        None
      };
      particle_blases.push(motion_blas);
      dense_particle_idx += 1;
    });

    let mut micro_tlases = hashbrown::HashMap::new();
    let mut macro_leaves = Vec::new();

    if let Some(orphans) = frame_leaves.remove(&macro_frame_idx) {
      macro_leaves.extend(orphans);
    }

    for (frame_idx, leaves) in frame_leaves {
      if leaves.is_empty() {
        continue;
      }
      let rbvh = RootBoundsBvh::build(&leaves);

      let mut mn = Vec3f32::from_array([f32::INFINITY; 3]);
      let mut mx = Vec3f32::from_array([f32::NEG_INFINITY; 3]);
      if !rbvh.nodes.is_empty() {
        let frame = &gpu_frames[frame_idx as usize];
        let scale = frame.scale;
        let pos = Vec3f32::from_array(frame.center_pos);
        mn = pos + (rbvh.nodes[0].min * scale);
        mx = pos + (rbvh.nodes[0].max * scale);
      }
      macro_leaves.push((frame_idx, mn, mx, BVH_SHAPE_SUB_TLAS, BVH_FRAME_MACRO));
      micro_tlases.insert(frame_idx, rbvh);
    }

    let macro_tlas = RootBoundsBvh::build(&macro_leaves);

    Self {
      gpu_frames,
      macro_tlas,
      micro_tlases,
      mesh_blases,
      particle_blases,
      mesh_entity_map,
      particle_entity_map,
      dt_s,
      recent_collisions: alloc::vec::Vec::new(),
    }
  }
}

pub fn build_motion_blas(
  mesh_bvh: &LinearBVH<f32>,
  transform: &TransformComponent,
  velocity_lca: Vec3f32,
  dt_s: f32,
) -> LinearBVH<f32> {
  let trs = Mat4x4f32::translation(transform.position)
    * <Mat4x4f32 as Matrix4>::from_quat_custom_frame(transform.rotation)
    * Mat4x4f32::from_scale(transform.scale);
  let disp = velocity_lca * dt_s;

  let nodes: Vec<LinearBVHNode<f32>> = mesh_bvh
    .nodes
    .iter()
    .map(|n| {
      let swept_bound = match &n.bound {
        LinearBound::AABB(aabb) => {
          let world = aabb.transform_f32(&trs);
          let swept_min = world.min::<Vec3f32>().min(world.min::<Vec3f32>() + disp);
          let swept_max = world.max::<Vec3f32>().max(world.max::<Vec3f32>() + disp);
          LinearBound::AABB(AABB::new(swept_min, swept_max))
        }
        LinearBound::OBB(obb) => {
          let world = obb.transform_f32(&trs).to_aabb::<Vec3f32>();
          let swept_min = world.min::<Vec3f32>().min(world.min::<Vec3f32>() + disp);
          let swept_max = world.max::<Vec3f32>().max(world.max::<Vec3f32>() + disp);
          LinearBound::AABB(AABB::new(swept_min, swept_max))
        }
      };
      LinearBVHNode {
        bound: swept_bound,
        left_child_or_primitive_offset: n.left_child_or_primitive_offset,
        right_child_offset: n.right_child_offset,
        primitive_count: n.primitive_count,
        mass: n.mass,
        center_of_mass: n.center_of_mass,
      }
    })
    .collect();

  LinearBVH {
    header: LinearBVHHeader {
      preciseness: 0,
      node_count: nodes.len() as u32,
      primitive_count: mesh_bvh.header.primitive_count,
    },
    nodes,
    primitives: mesh_bvh.primitives.clone(),
  }
}

fn find_lca_frame_idx(
  entity: EntityId,
  scene: &Scene,
  gpu_frames: &[GpuReferenceFrame],
) -> Option<u32> {
  let mut curr = entity;
  loop {
    let entity_raw = slotmap::Key::data(&curr).as_ffi();
    if let Some(idx) = gpu_frames.iter().position(|f| f.entity_id_raw == entity_raw) {
      return Some(idx as u32);
    }
    curr = scene.get_parent(curr)?;
  }
}

impl crate::math::collision::multi_bvh::BinaryBvh for RootBoundsBvh {
  type Bound = AABB<f32>;
  type Primitive = u32;

  fn root(&self) -> Option<u32> {
    if self.nodes.is_empty() { None } else { Some(0) }
  }

  fn bound(&self, idx: u32) -> AABB<f32> {
    let n = &self.nodes[idx as usize];
    AABB::new(n.min, n.max)
  }

  fn is_leaf(&self, idx: u32) -> bool {
    self.nodes[idx as usize].leaf_meta.is_some()
  }

  fn leaf_meta(&self, idx: u32) -> Option<u32> {
    self.nodes[idx as usize].leaf_meta
  }

  fn children(&self, idx: u32) -> (Option<u32>, Option<u32>) {
    let n = &self.nodes[idx as usize];
    (n.left, n.right)
  }

  fn extract_primitives(&self, idx: u32, out: &mut Vec<u32>) -> u32 {
    let n = &self.nodes[idx as usize];
    if let Some(meta) = n.leaf_meta {
      out.push(meta);
      1
    } else {
      0
    }
  }
}

impl core::fmt::Debug for PhysicsScene {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("PhysicsScene")
      .field("gpu_frames", &self.gpu_frames)
      .field("dt_s", &self.dt_s)
      .finish()
  }
}

#[test]
fn test_root_bounds_bvh_2_leaves() {
  let leaves = alloc::vec![
    (
      0,
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([0.0, 0.0, 0.0]),
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([1.0, 1.0, 1.0]),
      0,
      0
    ),
    (
      1,
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([2.0, 2.0, 2.0]),
      aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array([3.0, 3.0, 3.0]),
      1,
      0
    ),
  ];
  let bvh = RootBoundsBvh::build(&leaves);
  assert_eq!(bvh.nodes.len(), 3);
}