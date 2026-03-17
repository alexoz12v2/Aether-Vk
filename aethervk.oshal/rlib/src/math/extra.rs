use crate::math::{
  Scalar,
  vector::{Vector, Vector3},
};

pub trait OrthonormalBasis: Vector3 {
  // TODO (blender or OLD)
  fn coordinate_system(self) -> (Self, Self);
}

pub trait Ray: Copy + Sized + PartialEq + Eq {
  type Scalar: Scalar;
  type Vector3: Vector3<Scalar = Self::Scalar>;

  fn origin(&self) -> Self::Vector3;
  fn direction(&self) -> Self::Vector3;
  fn inv_direction(&self) -> Self::Vector3;

  fn at(self, t: Self::Scalar) -> Self::Vector3 {
    self.origin() + self.direction() * t
  }
}
