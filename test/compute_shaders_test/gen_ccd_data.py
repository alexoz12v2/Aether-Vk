import json
import argparse
import os
import struct

def uint_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def generate_test_data(out_path):
    def make_node(min_b, max_b, left=0, right=0, prim_count=0, prim_offset=0, com=[0.0, 0.0, 0.0]):
        return [
            min_b[0], min_b[1], min_b[2], # bound.min
            max_b[0], max_b[1], max_b[2], # bound.max
            uint_to_float(left if prim_count == 0 else prim_offset), # left_child_or_primitive_offset
            uint_to_float(right), # right_child_offset
            uint_to_float(prim_count),
            uint_to_float(0), # parent_idx
            uint_to_float(0), # node_type
            1.0, # mass
            com[0], com[1], com[2], # center_of_mass
            0.0 # _pad
        ]

    nodes = [None] * 7

    # Prims:
    nodes[3] = make_node([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], prim_count=1, prim_offset=0, com=[0.5, 0.5, 0.5])
    nodes[4] = make_node([0.5, 0.0, 0.0], [1.5, 1.0, 1.0], prim_count=1, prim_offset=1, com=[1.0, 0.5, 0.5])
    nodes[1] = make_node([0.0, 0.0, 0.0], [1.5, 1.0, 1.0], left=3, right=4)

    nodes[5] = make_node([3.0, 0.0, 0.0], [4.0, 1.0, 1.0], prim_count=1, prim_offset=2, com=[3.5, 0.5, 0.5])
    nodes[6] = make_node([3.5, 0.0, 0.0], [4.5, 1.0, 1.0], prim_count=1, prim_offset=3, com=[4.0, 0.5, 0.5])
    nodes[2] = make_node([3.0, 0.0, 0.0], [4.5, 1.0, 1.0], left=5, right=6)

    nodes[0] = make_node([0.0, 0.0, 0.0], [4.5, 1.0, 1.0], left=1, right=2)

    flat_nodes = []
    for n in nodes:
        flat_nodes.extend(n)

    sg_size = 32
    particles = [0.0] * (10 * sg_size)
    
    # Particle 0
    particles[0 * sg_size + 0] = 0.5 # posX
    particles[1 * sg_size + 0] = 0.5 # posY
    particles[2 * sg_size + 0] = 0.5 # posZ
    particles[3 * sg_size + 0] = 5.0 # velX
    particles[4 * sg_size + 0] = 0.0 # velY
    particles[5 * sg_size + 0] = 0.0 # velZ
    particles[6 * sg_size + 0] = 1.0 # mass

    # Particle 1
    particles[0 * sg_size + 1] = 1.0
    particles[1 * sg_size + 1] = 0.5
    particles[2 * sg_size + 1] = 0.5
    particles[3 * sg_size + 1] = -5.0
    particles[4 * sg_size + 1] = 0.0
    particles[5 * sg_size + 1] = 0.0
    particles[6 * sg_size + 1] = 1.0

    # Particle 2
    particles[0 * sg_size + 2] = 3.5
    particles[1 * sg_size + 2] = 0.5
    particles[2 * sg_size + 2] = 0.5
    particles[3 * sg_size + 2] = 5.0
    particles[4 * sg_size + 2] = 0.0
    particles[5 * sg_size + 2] = 0.0
    particles[6 * sg_size + 2] = 1.0

    # Particle 3
    particles[0 * sg_size + 3] = 4.0
    particles[1 * sg_size + 3] = 0.5
    particles[2 * sg_size + 3] = 0.5
    particles[3 * sg_size + 3] = -5.0
    particles[4 * sg_size + 3] = 0.0
    particles[5 * sg_size + 3] = 0.0
    particles[6 * sg_size + 3] = 1.0

    out = {
        "total_particles": 4,
        "root_index": 0,
        "bvh_nodes": flat_nodes,
        "particles": particles,
        "particle_radius": 0.5,
        "dt": 0.1,
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