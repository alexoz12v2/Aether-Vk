//! Gilbert-Johnson-Keerthi (GJK) Algorithm for distance between convex objects.

use aethervk_oshal_rlib::math::vector::Vector3;
use aethervk_oshal_rlib::{math::vector::Vector, math::vector::vec3::Vec3f32};
use alloc::vec::{ Vec };

/// A trait for shapes that can be queried for their furthest point in a given direction.
pub trait Support {
  fn support(&self, dir: Vec3f32) -> Vec3f32;
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
      return (0.0, Vec3f32::zero(), Vec3f32::zero()); // TODO: EPA for penetration depth if needed
    }
  }

  // If not intersecting, compute the closest points
  let (closest_a, closest_b) = compute_closest_points(&simplex);
  let distance = (closest_a - closest_b).length();
  (distance, closest_a, closest_b)
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
