use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_core_rlib::physics::physics_scene::{RootBoundsBvh, RbNode};

fn main() {
    for len in 0..10 {
        let mut leaves = Vec::new();
        for i in 0..len {
            leaves.push((i, Vec3f32::from_array([0.0; 3]), Vec3f32::from_array([1.0; 3]), 0, 0));
        }
        let bvh = RootBoundsBvh::build(&leaves);
        println!("leaves: {}, bvh.nodes.len(): {}", len, bvh.nodes.len());
    }
}
