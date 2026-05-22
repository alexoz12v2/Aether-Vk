use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_core_rlib::physics::physics_scene::{RootBoundsBvh, RbNode};

fn main() {
    let leaves = vec![
        (0, Vec3f32::from_array([0.0, 0.0, 0.0]), Vec3f32::from_array([1.0, 1.0, 1.0]), 0, 0),
        (1, Vec3f32::from_array([2.0, 2.0, 2.0]), Vec3f32::from_array([3.0, 3.0, 3.0]), 1, 0),
    ];
    let bvh = RootBoundsBvh::build(&leaves);
    println!("Nodes length: {}", bvh.nodes.len());
    for (i, node) in bvh.nodes.iter().enumerate() {
        println!("{}: leaf_meta={:?}, left={:?}, right={:?}", i, node.leaf_meta, node.left, node.right);
    }
}
