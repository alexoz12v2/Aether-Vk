import json
import argparse
import sys
import os

def generate_test_data(num_particles, dt, subgroup_size, out_path):
    particles = [0.0] * (10 * subgroup_size * ((num_particles + subgroup_size - 1) // subgroup_size))

    # Initialize input particles
    for i in range(num_particles):
        block_idx = i // subgroup_size
        local_idx = i % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx

        # px, py, pz
        particles[base + 0 * subgroup_size] = float(i)
        particles[base + 1 * subgroup_size] = float(i) * 2.0
        particles[base + 2 * subgroup_size] = float(i) * 3.0

        # vx, vy, vz
        particles[base + 3 * subgroup_size] = 0.1
        particles[base + 4 * subgroup_size] = 0.2
        particles[base + 5 * subgroup_size] = 0.3

        # mass
        particles[base + 6 * subgroup_size] = 10.0

        # fx, fy, fz
        particles[base + 7 * subgroup_size] = 1.0
        particles[base + 8 * subgroup_size] = 2.0
        particles[base + 9 * subgroup_size] = 3.0

    half_dt = dt * 0.5
    expected_particles = list(particles)

    # Compute expected results
    for i in range(num_particles):
        block_idx = i // subgroup_size
        local_idx = i % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx
        
        inv_mass = 1.0 / 10.0
        
        vx = 0.1 + (1.0 * inv_mass) * half_dt
        vy = 0.2 + (2.0 * inv_mass) * half_dt
        vz = 0.3 + (3.0 * inv_mass) * half_dt

        expected_particles[base + 3 * subgroup_size] = vx
        expected_particles[base + 4 * subgroup_size] = vy
        expected_particles[base + 5 * subgroup_size] = vz

        qx = float(i) + vx * half_dt
        qy = float(i) * 2.0 + vy * half_dt
        qz = float(i) * 3.0 + vz * half_dt

        expected_particles[base + 0 * subgroup_size] = qx
        expected_particles[base + 1 * subgroup_size] = qy
        expected_particles[base + 2 * subgroup_size] = qz

    out = {
        "dt": dt,
        "total_particles": num_particles,
        "input_particles": particles,
        "expected_particles": expected_particles
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path} with {num_particles} particles and dt={dt}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-particles", type=int, default=64)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--subgroup-size", type=int, default=32)
    parser.add_argument("--out", type=str, default="test_data/p1_2.json")
    
    args = parser.parse_args()
    generate_test_data(args.num_particles, args.dt, args.subgroup_size, args.out)
