use aethervk_oshal_rlib::{math::matrix::SquareMatrix, math::matrix::mat4::Mat4x4f32};

/// Resolves the transformation between two entities possibly in different reference frames
/// by finding their Least Common Ancestor (LCA).
pub fn resolve_lca_transform(
  entity_a_frame: u32,
  entity_a_transform: Mat4x4f32,
  entity_b_frame: u32,
  entity_b_transform: Mat4x4f32,
  frames: &hashbrown::HashMap<u32, (u32, Mat4x4f32)>, // map from frame_id -> (parent_frame_id, transform)
) -> Option<Mat4x4f32> {
  if entity_a_frame == entity_b_frame {
    return entity_b_transform.inverse().map(|inv| inv * entity_a_transform);
  }

  // Traverse up to find LCA
  let mut a_ancestors = alloc::vec![entity_a_frame];
  let mut curr = entity_a_frame;
  while let Some(&(parent, _)) = frames.get(&curr) {
    if parent == curr {
      break;
    }
    a_ancestors.push(parent);
    if parent == 0 {
      break;
    }
    curr = parent;
  }

  let mut b_ancestors = alloc::vec![entity_b_frame];
  curr = entity_b_frame;
  while let Some(&(parent, _)) = frames.get(&curr) {
    if parent == curr {
      break;
    }
    b_ancestors.push(parent);
    if parent == 0 {
      break;
    }
    curr = parent;
  }

  // Find LCA
  let mut lca = None;
  for a in &a_ancestors {
    if b_ancestors.contains(a) {
      lca = Some(*a);
      break;
    }
  }

  let lca = lca?;

  // Compute M_{A -> LCA}
  let mut m_a_lca = entity_a_transform;
  curr = entity_a_frame;
  while curr != lca {
    let &(parent, ref transform) = frames.get(&curr)?;
    m_a_lca = *transform * m_a_lca;
    curr = parent;
  }

  // Compute M_{B -> LCA}
  let mut m_b_lca = entity_b_transform;
  curr = entity_b_frame;
  while curr != lca {
    let &(parent, ref transform) = frames.get(&curr)?;
    m_b_lca = *transform * m_b_lca;
    curr = parent;
  }

  m_b_lca.inverse().map(|inv| inv * m_a_lca)
}
