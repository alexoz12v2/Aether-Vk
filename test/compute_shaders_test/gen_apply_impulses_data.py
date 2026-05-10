import json
import argparse
import os

def generate_test_data(out_path):
    subgroup_size = 32
    num_particles = 64
    particles = [0.0] * (10 * subgroup_size * ((num_particles + subgroup_size - 1) // subgroup_size))
    
    # 3 dummy zeros for dispatch_x, y, z, then 2 collisions
    collisions = [0, 0, 0, 2, 0, 1, 2, 3] # dispatch(3), count = 2, pair 0 (0,1), pair 1 (2,3)
    impulses = [10.0, 20.0, 30.0, -5.0, -10.0, -15.0]

    for i in range(num_particles):
        block_idx = i // subgroup_size
        local_idx = i % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx
        
        # velocity = 0
        particles[base + 3 * subgroup_size] = 0.0
        particles[base + 4 * subgroup_size] = 0.0
        particles[base + 5 * subgroup_size] = 0.0
        
        # mass
        particles[base + 6 * subgroup_size] = 2.0

    expected_particles = list(particles)
    
    # Apply collision 0
    pA = 0
    pB = 1
    imp = (10.0, 20.0, 30.0)
    for p, sign in [(pA, 1.0), (pB, -1.0)]:
        block_idx = p // subgroup_size
        local_idx = p % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx
        mass = 2.0
        dvx = sign * imp[0] / mass
        dvy = sign * imp[1] / mass
        dvz = sign * imp[2] / mass
        expected_particles[base + 3 * subgroup_size] += dvx
        expected_particles[base + 4 * subgroup_size] += dvy
        expected_particles[base + 5 * subgroup_size] += dvz

    # Apply collision 1
    pA = 2
    pB = 3
    imp = (-5.0, -10.0, -15.0)
    for p, sign in [(pA, 1.0), (pB, -1.0)]:
        block_idx = p // subgroup_size
        local_idx = p % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx
        mass = 2.0
        dvx = sign * imp[0] / mass
        dvy = sign * imp[1] / mass
        dvz = sign * imp[2] / mass
        expected_particles[base + 3 * subgroup_size] += dvx
        expected_particles[base + 4 * subgroup_size] += dvy
        expected_particles[base + 5 * subgroup_size] += dvz

    out = {
        "num_particles": num_particles,
        "particles": particles,
        "collisions": collisions,
        "impulses": impulses,
        "expected_particles": expected_particles
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/apply_impulses.json")
    args = parser.parse_args()
    generate_test_data(args.out)
