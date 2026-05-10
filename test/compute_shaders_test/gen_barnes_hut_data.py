import json
import argparse
import os
import struct
import math

def uint_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def float_to_uint(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def generate_test_data(out_path):
    subgroup_size = 32
    num_particles = 4
    
    # 4 particles
    # P0: (0, 0, 0), mass 10
    # P1: (1, 0, 0), mass 10
    # P2: (100, 0, 0), mass 20
    # P3: (101, 0, 0), mass 20
    
    particles = [0.0] * (10 * subgroup_size)
    
    def set_p(i, x, y, z, mass):
        particles[0 * subgroup_size + i] = x
        particles[1 * subgroup_size + i] = y
        particles[2 * subgroup_size + i] = z
        particles[6 * subgroup_size + i] = mass

    set_p(0, 0.0, 0.0, 0.0, 10.0)
    set_p(1, 1.0, 0.0, 0.0, 10.0)
    set_p(2, 100.0, 0.0, 0.0, 20.0)
    set_p(3, 101.0, 0.0, 0.0, 20.0)
    
    def make_node(min_b, max_b, left=0, right=0, prim_count=0, prim_offset=0, mass=0.0, com=[0,0,0]):
        return [
            min_b[0], min_b[1], min_b[2], # bound.min
            max_b[0], max_b[1], max_b[2], # bound.max
            uint_to_float(left if prim_count == 0 else prim_offset),
            uint_to_float(right),
            uint_to_float(prim_count),
            uint_to_float(0), # node_type
            uint_to_float(0), # parent_idx
            mass,
            com[0], com[1], com[2]
        ]
        
    nodes = [None] * 7
    # leaves at N-1 = 3,4,5,6
    nodes[3] = make_node([0,0,0], [0,0,0], prim_count=1, prim_offset=0, mass=10.0, com=[0,0,0])
    nodes[4] = make_node([1,0,0], [1,0,0], prim_count=1, prim_offset=1, mass=10.0, com=[1,0,0])
    nodes[5] = make_node([100,0,0], [100,0,0], prim_count=1, prim_offset=2, mass=20.0, com=[100,0,0])
    nodes[6] = make_node([101,0,0], [101,0,0], prim_count=1, prim_offset=3, mass=20.0, com=[101,0,0])
    
    # Internal 1 (P0, P1)
    nodes[1] = make_node([0,0,0], [1,0,0], left=3, right=4, mass=20.0, com=[0.5,0,0])
    # Internal 2 (P2, P3)
    nodes[2] = make_node([100,0,0], [101,0,0], left=5, right=6, mass=40.0, com=[100.5,0,0])
    # Root (P0..P3)
    nodes[0] = make_node([0,0,0], [101,0,0], left=1, right=2, mass=60.0, com=[67.166666,0,0])
    
    theta = 0.5
    G = 1.0
    
    expected_particles = list(particles)
    
    # Traverse tree as in shader
    for i in range(num_particles):
        my_p_id = i
        my_pos = [particles[0*subgroup_size+i], particles[1*subgroup_size+i], particles[2*subgroup_size+i]]
        my_mass = particles[6*subgroup_size+i]
        
        total_force = [0.0, 0.0, 0.0]
        
        stack = [0]
        while len(stack) > 0:
            node_idx = stack.pop()
            node = nodes[node_idx]
            is_leaf = struct.unpack('<I', struct.pack('<f', node[8]))[0] > 0
            
            if is_leaf:
                other_p_id = struct.unpack('<I', struct.pack('<f', node[6]))[0]
                if my_p_id != other_p_id:
                    com = [node[12], node[13], node[14]]
                    r = [com[0]-my_pos[0], com[1]-my_pos[1], com[2]-my_pos[2]]
                    dist_sq = r[0]**2 + r[1]**2 + r[2]**2
                    if dist_sq > 1e-6:
                        dist = math.sqrt(dist_sq)
                        f_mag = G * my_mass * node[11] / (dist_sq * dist)
                        total_force[0] += r[0] * f_mag
                        total_force[1] += r[1] * f_mag
                        total_force[2] += r[2] * f_mag
            else:
                com = [node[12], node[13], node[14]]
                r = [com[0]-my_pos[0], com[1]-my_pos[1], com[2]-my_pos[2]]
                dist_sq = r[0]**2 + r[1]**2 + r[2]**2
                dist = math.sqrt(max(dist_sq, 1e-6))
                
                extents = [node[3]-node[0], node[4]-node[1], node[5]-node[2]]
                size = max(extents[0], max(extents[1], extents[2]))
                
                if size / dist < theta:
                    f_mag = G * my_mass * node[11] / (dist_sq * dist)
                    total_force[0] += r[0] * f_mag
                    total_force[1] += r[1] * f_mag
                    total_force[2] += r[2] * f_mag
                else:
                    stack.append(struct.unpack('<I', struct.pack('<f', node[6]))[0]) # left
                    stack.append(struct.unpack('<I', struct.pack('<f', node[7]))[0]) # right
                    
        expected_particles[7*subgroup_size+i] = total_force[0]
        expected_particles[8*subgroup_size+i] = total_force[1]
        expected_particles[9*subgroup_size+i] = total_force[2]

    flat_nodes = []
    for n in nodes:
        flat_nodes.extend(n)

    out = {
        "total_particles": num_particles,
        "root_index": 0,
        "particles": particles,
        "bvh_nodes": flat_nodes,
        "expected_particles": expected_particles,
        "theta": theta,
        "G": G
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/barnes_hut.json")
    args = parser.parse_args()
    generate_test_data(args.out)
