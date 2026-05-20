import json
import argparse
import os
import math
import struct

def float_to_uint(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

def uint_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def generate_test_data(num_pairs, out_path):
    subgroup_size = 32
    num_particles = num_pairs * 2
    particles = [0.0] * (10 * subgroup_size * ((num_particles + subgroup_size - 1) // subgroup_size))
    
    dt = 0.01
    particle_radius = 1.0

    pairs = []
    
    min_tc = dt

    for i in range(num_pairs):
        pA_id = i * 2
        pB_id = i * 2 + 1
        
        pairs.extend([0, pA_id, 0, pB_id, 0, 0, 0, 0, 0, 0, 0, 0])
        
        # Set pos
        def set_particle(p_id, pos, vel):
            block_idx = p_id // subgroup_size
            local_idx = p_id % subgroup_size
            base = block_idx * 10 * subgroup_size + local_idx
            particles[base + 0 * subgroup_size] = pos[0]
            particles[base + 1 * subgroup_size] = pos[1]
            particles[base + 2 * subgroup_size] = pos[2]
            particles[base + 3 * subgroup_size] = vel[0]
            particles[base + 4 * subgroup_size] = vel[1]
            particles[base + 5 * subgroup_size] = vel[2]

        if i == 0:
            set_particle(pA_id, [0.0, 0.0, 0.0], [400.0, 0.0, 0.0])
            set_particle(pB_id, [4.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            tc = 0.005
        elif i == 1:
            set_particle(pA_id, [0.0, 0.0, 0.0], [1000.0, 0.0, 0.0])
            set_particle(pB_id, [4.0, 0.0, 0.0], [0.0, 0.0, 0.0])
            tc = 0.002
        elif i == 2:
            set_particle(pA_id, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])
            set_particle(pB_id, [2.0, 0.0, 0.0], [-1.0, 0.0, 0.0])
            # dp = -2, dv = 2. a = 4, b = -4, c = 4 - 4 = 0
            # discriminant = 16 - 0 = 16. t1 = (4 - 4) / 4 = 0.0
            tc = 0.0
        else:
            set_particle(pA_id, [0.0, i*10.0, 0.0], [0.0, 0.0, 0.0])
            set_particle(pB_id, [100.0, i*10.0, 0.0], [0.0, 0.0, 0.0])
            tc = dt

        min_tc = min(min_tc, tc)

    out = {
        "dt": dt,
        "particle_radius": particle_radius,
        "total_pairs": num_pairs,
        "particles": particles,
        "pairs": pairs,
        "expected_min_tc": min_tc,
        "expected_min_tc_uint": float_to_uint(min_tc)
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path} with {num_pairs} pairs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/reduce_toi.json")
    
    args = parser.parse_args()
    generate_test_data(args.num_pairs, args.out)
