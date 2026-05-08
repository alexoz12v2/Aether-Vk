import json
import argparse
import os
import struct

def uint_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def generate_test_data(out_path):
    # Construct a simple BVH with 4 leaves (particles)
    # Total nodes = 2*4 - 1 = 7
    # 0: Root (L=1, R=2)
    # 1: Internal (L=3, R=4)
    # 2: Internal (L=5, R=6)
    # Leaves: 3, 4, 5, 6 for prims 0, 1, 2, 3

    def make_node(min_b, max_b, left=0, right=0, prim_count=0, prim_offset=0):
        return [
            min_b[0], min_b[1], min_b[2], # bound.min
            max_b[0], max_b[1], max_b[2], # bound.max
            uint_to_float(left if prim_count == 0 else prim_offset), # left_child_or_primitive_offset
            uint_to_float(right), # right_child_offset
            uint_to_float(prim_count),
            uint_to_float(0), # node_type
            uint_to_float(0), # parent_idx
            0.0, # mass
            0.0, 0.0, 0.0 # center_of_mass
        ]

    nodes = [None] * 7

    # Prims:
    # 0: [0,0,0] - [1,1,1]
    # 1: [0.5,0,0] - [1.5,1,1] (overlaps 0)
    # 2: [3,0,0] - [4,1,1]
    # 3: [3.5,0,0] - [4.5,1,1] (overlaps 2)

    nodes[3] = make_node([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], prim_count=1, prim_offset=0)
    nodes[4] = make_node([0.5, 0.0, 0.0], [1.5, 1.0, 1.0], prim_count=1, prim_offset=1)
    nodes[1] = make_node([0.0, 0.0, 0.0], [1.5, 1.0, 1.0], left=3, right=4)

    nodes[5] = make_node([3.0, 0.0, 0.0], [4.0, 1.0, 1.0], prim_count=1, prim_offset=2)
    nodes[6] = make_node([3.5, 0.0, 0.0], [4.5, 1.0, 1.0], prim_count=1, prim_offset=3)
    nodes[2] = make_node([3.0, 0.0, 0.0], [4.5, 1.0, 1.0], left=5, right=6)

    nodes[0] = make_node([0.0, 0.0, 0.0], [4.5, 1.0, 1.0], left=1, right=2)

    flat_nodes = []
    for n in nodes:
        flat_nodes.extend(n)

    out = {
        "total_particles": 4,
        "root_index": 0,
        "bvh_nodes": flat_nodes,
        "expected_count": 2,
        "expected_pairs": [0, 1, 2, 3] # pairs (0,1) and (2,3)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/ccd.json")
    args = parser.parse_args()
    generate_test_data(args.out)
