use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_core_rlib::physics::physics_scene::{RootBoundsBvh, RbNode};
use aethervk_core_rlib::math::collision::multi_bvh::convert_binary_to_multi_bvh;

fn main() {
    let leaves = alloc::vec![
        (0, Vec3f32::from_array([0.0, 0.0, 0.0]), Vec3f32::from_array([1.0, 1.0, 1.0]), 0, 0),
        (1, Vec3f32::from_array([2.0, 2.0, 2.0]), Vec3f32::from_array([3.0, 3.0, 3.0]), 1, 0),
    ];
    let bvh = RootBoundsBvh::build(&leaves);
    let mut multi_nodes = convert_binary_to_multi_bvh::<32, RootBoundsBvh>(&bvh);

    for node in multi_nodes.iter_mut() {
        for i in 0..32 {
            let meta = node.metadata[i];
            if meta != 0 && (meta & 0x8000_0000) != 0 {
                let binary_node_id = (meta & 0x7FFF_FFFF) as usize;
                let index = bvh.nodes[binary_node_id].leaf_child_idx as usize;
                let entity_id = leaves[index].3;
                println!("Success! index: {}", index);
            }
        }
    }
}
