import json
import argparse
import os
import math

def generate_test_data(out_path):
    subgroup_size = 32
    num_particles = 64
    dt = 0.01
    half_dt = dt * 0.5
    num_emitters = 1

    particles = [0.0] * (10 * subgroup_size * ((num_particles + subgroup_size - 1) // subgroup_size))
    
    emitters = [0.0] * 12
    emitters[0] = 0.0 # pos.x
    emitters[1] = 0.0 # pos.y
    emitters[2] = 0.0 # pos.z
    emitters[3] = 100000.0 # mu
    emitters[4] = 0.0 # normal.x
    emitters[5] = 0.0 # normal.y
    emitters[6] = 0.0 # normal.z
    import struct
    emitters[7] = struct.unpack('<f', struct.pack('<I', 0))[0] # type_id (0 = Gravity)
    emitters[8] = 0.0 # trunc_distance
    emitters[9] = 1.0 # scale_factor
    emitters[10] = 0.0 # pad0
    emitters[11] = 0.0 # pad1

    for i in range(num_particles):
        block_idx = i // subgroup_size
        local_idx = i % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx
        
        # Position (q_mid)
        particles[base + 0 * subgroup_size] = float(i+1) * 10.0
        particles[base + 1 * subgroup_size] = 0.0
        particles[base + 2 * subgroup_size] = 0.0
        
        # Velocity (v_{n+1/2})
        particles[base + 3 * subgroup_size] = 0.0
        particles[base + 4 * subgroup_size] = float(i)
        particles[base + 5 * subgroup_size] = 0.0
        
        # Mass
        particles[base + 6 * subgroup_size] = 2.0

    expected_particles = list(particles)
    
    for i in range(num_particles):
        block_idx = i // subgroup_size
        local_idx = i % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx
        
        mass = 2.0
        q_mid = [float(i+1) * 10.0, 0.0, 0.0]
        v_half = [0.0, float(i), 0.0]
        
        q_next = [q_mid[0] + v_half[0] * half_dt, q_mid[1] + v_half[1] * half_dt, q_mid[2] + v_half[2] * half_dt]
        
        # Evaluate force at q_next
        r = [emitters[0] - q_next[0], emitters[1] - q_next[1], emitters[2] - q_next[2]]
        dist_sq = r[0]**2 + r[1]**2 + r[2]**2
        f_next = [0.0, 0.0, 0.0]
        if dist_sq > 1e-6:
            dist = math.sqrt(dist_sq)
            coeff = emitters[3] * mass / (dist_sq * dist)
            f_next[0] = r[0] * coeff
            f_next[1] = r[1] * coeff
            f_next[2] = r[2] * coeff
            
        v_next = [
            v_half[0] + f_next[0] / mass * half_dt,
            v_half[1] + f_next[1] / mass * half_dt,
            v_half[2] + f_next[2] / mass * half_dt
        ]
        
        expected_particles[base + 0 * subgroup_size] = q_next[0]
        expected_particles[base + 1 * subgroup_size] = q_next[1]
        expected_particles[base + 2 * subgroup_size] = q_next[2]
        
        expected_particles[base + 3 * subgroup_size] = v_next[0]
        expected_particles[base + 4 * subgroup_size] = v_next[1]
        expected_particles[base + 5 * subgroup_size] = v_next[2]
        
        expected_particles[base + 7 * subgroup_size] = f_next[0]
        expected_particles[base + 8 * subgroup_size] = f_next[1]
        expected_particles[base + 9 * subgroup_size] = f_next[2]
        
    out = {
        "num_particles": num_particles,
        "num_emitters": num_emitters,
        "dt": dt,
        "particles": particles,
        "emitters": emitters,
        "expected_particles": expected_particles
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/p5.json")
    args = parser.parse_args()
    generate_test_data(args.out)
