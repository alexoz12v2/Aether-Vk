//! Gilbert-Johnson-Keerthi (GJK) Algorithm for distance between convex objects.

use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::{math::vector::Vector, math::vector::vec3::Vec3f32};
use alloc::vec::Vec;

/// A trait for shapes that can be queried for their furthest point in a given direction.
pub trait Support {
  fn support(&self, dir: Vec3f32) -> Vec3f32;
}

use crate::math::collision::bounds::{AABB, OBB};

impl Support for AABB<f32> {
  fn support(&self, dir: Vec3f32) -> Vec3f32 {
    let min = self.min::<Vec3f32>();
    let max = self.max::<Vec3f32>();
    Vec3f32::from_array([
      if dir.x() > 0.0 { max.x() } else { min.x() },
      if dir.y() > 0.0 { max.y() } else { min.y() },
      if dir.z() > 0.0 { max.z() } else { min.z() },
    ])
  }
}

impl Support for OBB<f32> {
  fn support(&self, dir: Vec3f32) -> Vec3f32 {
    let origin: Vec3f32 = self.translation();
    let [x_axis, y_axis, z_axis] = self.axes();
    let extents: Vec3f32 = self.half_extents();

    let mut result = origin;
    result += x_axis
      * if Vector::dot(x_axis, dir) > 0.0 {
        extents.x()
      } else {
        -extents.x()
      };
    result += y_axis
      * if Vector::dot(y_axis, dir) > 0.0 {
        extents.y()
      } else {
        -extents.y()
      };
    result += z_axis
      * if Vector::dot(z_axis, dir) > 0.0 {
        extents.z()
      } else {
        -extents.z()
      };
    result
  }
}

pub struct GjkSphere {
  pub center: Vec3f32,
  pub radius: f32,
}

impl Support for GjkSphere {
  fn support(&self, dir: Vec3f32) -> Vec3f32 {
    let dir_normalized = if dir.length_squared() > 1e-6 {
      dir.normalize()
    } else {
      Vec3f32::from_array([1.0, 0.0, 0.0])
    };
    self.center + dir_normalized * self.radius
  }
}

/// Represents a point in the Minkowski Difference
#[derive(Clone, Copy, Debug)]
struct MinkowskiPoint {
  /// The point in the Minkowski difference (p_a - p_b)
  point: Vec3f32,
  /// The support point from shape A
  point_a: Vec3f32,
  /// The support point from shape B
  point_b: Vec3f32,
}

impl MinkowskiPoint {
  fn new(point_a: Vec3f32, point_b: Vec3f32) -> Self {
    Self {
      point: point_a - point_b,
      point_a,
      point_b,
    }
  }
}

/// Computes the distance between two convex shapes using the GJK algorithm.
/// Returns a tuple containing:
/// 1. The distance between the shapes (0.0 if intersecting).
/// 2. The closest point on shape 1.
/// 3. The closest point on shape 2.
pub fn gjk_distance<S1: Support, S2: Support>(shape1: &S1, shape2: &S2) -> (f32, Vec3f32, Vec3f32) {
  let mut dir = Vec3f32::from_array([1.0, 0.0, 0.0]); // Initial arbitrary direction

  let support_a = shape1.support(dir);
  let support_b = shape2.support(-dir);
  let mut simplex = alloc::vec![MinkowskiPoint::new(support_a, support_b)];
  dir = -simplex[0].point;

  const MAX_ITERATIONS: usize = 64;
  for _ in 0..MAX_ITERATIONS {
    if dir.length_squared() < 1e-6 {
      break; // Origin is enclosed or very close
    }

    let p_a = shape1.support(dir);
    let p_b = shape2.support(-dir);
    let new_pt = MinkowskiPoint::new(p_a, p_b);

    // If the new point is not further along the search direction than the current closest point,
    // we can't enclose the origin.
    if Vector::dot(new_pt.point, dir) < 0.0 {
      break;
    }

    simplex.push(new_pt);
    if do_simplex(&mut simplex, &mut dir) {
      // Intersecting
      return epa_distance(&simplex, shape1, shape2);
    }
  }

  // If not intersecting, compute the closest points
  let (closest_a, closest_b) = compute_closest_points(&simplex);
  let distance = (closest_a - closest_b).length();
  (distance, closest_a, closest_b)
}

#[derive(Clone, Debug)]
struct Face {
  a: usize,
  b: usize,
  c: usize,
  normal: Vec3f32,
  distance: f32,
}

fn epa_distance<S1: Support, S2: Support>(
  simplex: &Vec<MinkowskiPoint>,
  shape1: &S1,
  shape2: &S2,
) -> (f32, Vec3f32, Vec3f32) {
  let mut polytope = simplex.clone();
  let mut faces = Vec::new();

  let add_face = |polytope: &[MinkowskiPoint], a: usize, b: usize, c: usize| -> Face {
    let ab = polytope[b].point - polytope[a].point;
    let ac = polytope[c].point - polytope[a].point;
    let n = Vector3::cross(ab, ac);

    // Fallback if degenerate
    let mut normal = if n.length_squared() > 1e-8 {
      n.normalize()
    } else {
      Vec3f32::from_array([1.0, 0.0, 0.0])
    };
    let mut distance = Vector::dot(normal, polytope[a].point);

    if distance < 0.0 {
      normal = -normal;
      distance = -distance;
    }
    Face {
      a,
      b,
      c,
      normal,
      distance,
    }
  };

  if polytope.len() == 4 {
    faces.push(add_face(&polytope, 0, 1, 2));
    faces.push(add_face(&polytope, 0, 3, 1));
    faces.push(add_face(&polytope, 0, 2, 3));
    faces.push(add_face(&polytope, 1, 3, 2));
  } else {
    // Should not happen theoretically since do_simplex only returns true on 4-simplex
    return (0.0, Vec3f32::zero(), Vec3f32::zero());
  }

  const MAX_EPA_ITERATIONS: usize = 32;
  for _ in 0..MAX_EPA_ITERATIONS {
    let mut closest_face_idx = 0;
    let mut min_dist = faces[0].distance;
    for (i, face) in faces.iter().enumerate().skip(1) {
      if face.distance < min_dist {
        min_dist = face.distance;
        closest_face_idx = i;
      }
    }

    let closest_face = faces[closest_face_idx].clone();
    let search_dir = closest_face.normal;
    let p_a = shape1.support(search_dir);
    let p_b = shape2.support(-search_dir);
    let new_pt = MinkowskiPoint::new(p_a, p_b);

    let dist = Vector::dot(new_pt.point, search_dir);
    if dist - min_dist < 1e-4 {
      let a = polytope[closest_face.a];
      let b = polytope[closest_face.b];
      let c = polytope[closest_face.c];

      let n = closest_face.normal;
      let p = n * min_dist;

      let v0 = b.point - a.point;
      let v1 = c.point - a.point;
      let v2 = p - a.point;

      let d00 = Vector::dot(v0, v0);
      let d01 = Vector::dot(v0, v1);
      let d11 = Vector::dot(v1, v1);
      let d20 = Vector::dot(v2, v0);
      let d21 = Vector::dot(v2, v1);

      let denom = d00 * d11 - d01 * d01;
      let (v, w) = if denom.abs() < 1e-6 {
        (0.333, 0.333)
      } else {
        let inv_denom = 1.0 / denom;
        (
          (d11 * d20 - d01 * d21) * inv_denom,
          (d00 * d21 - d01 * d20) * inv_denom,
        )
      };
      let u = 1.0 - v - w;

      let contact_a = a.point_a * u + b.point_a * v + c.point_a * w;
      let contact_b = a.point_b * u + b.point_b * v + c.point_b * w;

      return (-min_dist, contact_a, contact_b);
    }

    let new_idx = polytope.len();
    polytope.push(new_pt);

    let mut edges = Vec::new();
    let mut i = 0;
    while i < faces.len() {
      if Vector::dot(faces[i].normal, new_pt.point - polytope[faces[i].a].point) > 0.0 {
        let f = faces.remove(i);
        let mut add_edge = |a: usize, b: usize| {
          if let Some(pos) = edges.iter().position(|&(ea, eb)| ea == b && eb == a) {
            edges.remove(pos);
          } else {
            edges.push((a, b));
          }
        };
        add_edge(f.a, f.b);
        add_edge(f.b, f.c);
        add_edge(f.c, f.a);
      } else {
        i += 1;
      }
    }

    if edges.is_empty() {
      break;
    }

    for (ea, eb) in edges {
      faces.push(add_face(&polytope, ea, eb, new_idx));
    }
  }

  let closest_face = match faces.first() {
    Some(face) => face,
    None => return (0.0, Vec3f32::zero(), Vec3f32::zero()),
  };
  let a = polytope[closest_face.a];
  let b = polytope[closest_face.b];
  let c = polytope[closest_face.c];
  let u = 0.3333;
  let v = 0.3333;
  let w = 0.3334;
  (
    -closest_face.distance,
    a.point_a * u + b.point_a * v + c.point_a * w,
    a.point_b * u + b.point_b * v + c.point_b * w,
  )
}

/// Updates the simplex and the search direction.
/// Returns true if the origin is enclosed by the simplex.
fn do_simplex(simplex: &mut Vec<MinkowskiPoint>, dir: &mut Vec3f32) -> bool {
  match simplex.len() {
    2 => {
      let a = simplex[1];
      let b = simplex[0];
      let ab = b.point - a.point;
      let ao = -a.point;

      if Vector::dot(ab, ao) > 0.0 {
        // Origin is along AB
        *dir = ab.cross(ao).cross(ab);
      } else {
        // Origin is behind A
        simplex.remove(0); // Remove B
        *dir = ao;
      }
      false
    }
    3 => {
      let a = simplex[2];
      let b = simplex[1];
      let c = simplex[0];

      let ab = b.point - a.point;
      let ac = c.point - a.point;
      let ao = -a.point;

      let abc = ab.cross(ac);

      if abc.cross(ac).dot(ao) > 0.0 {
        if Vector::dot(ac, ao) > 0.0 {
          simplex.remove(1); // Remove B
          *dir = ac.cross(ao).cross(ac);
        } else {
          if Vector::dot(ab, ao) > 0.0 {
            simplex.remove(0); // Remove C
            *dir = ab.cross(ao).cross(ab);
          } else {
            simplex.remove(0); // Remove C
            simplex.remove(0); // Remove B
            *dir = ao;
          }
        }
      } else {
        if Vector::dot(ab.cross(abc), ao) > 0.0 {
          if Vector::dot(ab, ao) > 0.0 {
            simplex.remove(0); // Remove C
            *dir = ab.cross(ao).cross(ab);
          } else {
            simplex.remove(0); // Remove C
            simplex.remove(0); // Remove B
            *dir = ao;
          }
        } else {
          if Vector::dot(abc, ao) > 0.0 {
            *dir = abc;
          } else {
            simplex.swap(0, 1); // Swap B and C to maintain winding
            *dir = -abc;
          }
        }
      }
      false
    }
    4 => {
      let a = simplex[3];
      let b = simplex[2];
      let c = simplex[1];
      let d = simplex[0];

      let ab = b.point - a.point;
      let ac = c.point - a.point;
      let ad = d.point - a.point;
      let ao = -a.point;

      let abc = ab.cross(ac);
      let acd = ac.cross(ad);
      let adb = ad.cross(ab);

      if Vector::dot(abc, ao) > 0.0 {
        simplex.remove(0); // Remove D
        *dir = abc;
        do_simplex(simplex, dir)
      } else if Vector::dot(acd, ao) > 0.0 {
        simplex.remove(1); // Remove B
        *dir = acd;
        do_simplex(simplex, dir)
      } else if Vector::dot(adb, ao) > 0.0 {
        simplex.remove(2); // Remove C
        *dir = adb;
        do_simplex(simplex, dir)
      } else {
        true // Origin is inside the tetrahedron
      }
    }
    _ => unreachable!(),
  }
}

/// Computes the closest points on shapes A and B from the final simplex.
fn compute_closest_points(simplex: &[MinkowskiPoint]) -> (Vec3f32, Vec3f32) {
  match simplex.len() {
    1 => (simplex[0].point_a, simplex[0].point_b),
    2 => {
      let a = simplex[1];
      let b = simplex[0];

      let ab = b.point - a.point;
      let t = Vector::dot(-a.point, ab) / ab.length_squared();
      let t = t.clamp(0.0, 1.0);

      let closest_a = a.point_a + (b.point_a - a.point_a) * t;
      let closest_b = a.point_b + (b.point_b - a.point_b) * t;

      (closest_a, closest_b)
    }
    3 => {
      let a = simplex[2];
      let b = simplex[1];
      let c = simplex[0];

      let ab = b.point - a.point;
      let ac = c.point - a.point;

      let n = Vector3::cross(ab, ac);
      let n_len_sq = n.length_squared();

      if n_len_sq < 1e-6 {
        return (a.point_a, a.point_b); // Degenerate case
      }

      let ao = -a.point;

      let u = Vector::dot(Vector3::cross(ac, n), ao) / n_len_sq;
      let v = Vector::dot(Vector3::cross(n, ab), ao) / n_len_sq;
      let w = 1.0 - u - v;

      let closest_a = a.point_a * w + b.point_a * u + c.point_a * v;
      let closest_b = a.point_b * w + b.point_b * u + c.point_b * v;

      (closest_a, closest_b)
    }
    4 => {
      // Origin is inside, so distance is 0.
      // Should not be called if intersecting, but fallback.
      (Vec3f32::zero(), Vec3f32::zero())
    }
    _ => unreachable!(),
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  struct Sphere {
    center: Vec3f32,
    radius: f32,
  }

  impl Support for Sphere {
    fn support(&self, dir: Vec3f32) -> Vec3f32 {
      let dir_normalized = if dir.length_squared() > 1e-6 {
        dir.normalize()
      } else {
        Vec3f32::from_array([1.0, 0.0, 0.0])
      };
      self.center + dir_normalized * self.radius
    }
  }

  #[test]
  fn test_gjk_non_intersecting_spheres() {
    let s1 = Sphere {
      center: Vec3f32::from_array([0.0, 0.0, 0.0]),
      radius: 1.0,
    };
    let s2 = Sphere {
      center: Vec3f32::from_array([4.0, 0.0, 0.0]),
      radius: 1.0,
    };

    let (dist, p_a, p_b) = gjk_distance(&s1, &s2);
    println!("dist: {}, p_a: {:?}, p_b: {:?}", dist, p_a, p_b);

    approx::assert_abs_diff_eq!(dist, 2.0, epsilon = 1e-4);
    approx::assert_abs_diff_eq!(p_a.x(), 1.0, epsilon = 1e-4);
    approx::assert_abs_diff_eq!(p_a.y(), 0.0, epsilon = 1e-4);
    approx::assert_abs_diff_eq!(p_a.z(), 0.0, epsilon = 1e-4);
    approx::assert_abs_diff_eq!(p_b.x(), 3.0, epsilon = 1e-4);
    approx::assert_abs_diff_eq!(p_b.y(), 0.0, epsilon = 1e-4);
    approx::assert_abs_diff_eq!(p_b.z(), 0.0, epsilon = 1e-4);
  }

  #[test]
  fn test_gjk_intersecting_spheres() {
    let s1 = Sphere {
      center: Vec3f32::from_array([0.0, 0.0, 0.0]),
      radius: 1.0,
    };
    let s2 = Sphere {
      center: Vec3f32::from_array([1.5, 0.0, 0.0]),
      radius: 1.0,
    };

    let (dist, _p_a, _p_b) = gjk_distance(&s1, &s2);

    assert_eq!(dist, 0.0);
  }
}
