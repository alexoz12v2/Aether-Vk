import json
import argparse
import os
import struct

def uint_to_float(i):
    return struct.unpack('<f', struct.pack('<I', i))[0]

def float_to_uint(f):
    return struct.unpack('<I', struct.pack('<f', f))[0]

# 3D Morton code (10 bits per dimension)
def expand_bits(v):
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
    return v

def morton_3d(x, y, z):
    x = min(max(x * 1024.0, 0.0), 1023.0)
    y = min(max(y * 1024.0, 0.0), 1023.0)
    z = min(max(z * 1024.0, 0.0), 1023.0)
    xx = expand_bits(int(x))
    yy = expand_bits(int(y))
    zz = expand_bits(int(z))
    return (xx << 2) | (yy << 1) | zz

def generate_test_data(out_path):
    subgroup_size = 32
    num_particles = 128
    dt = 0.01

    # Emitters: Earth and Moon at J2000 (roughly from SPICE)
    # Earth mass = 398600.4418
    # Moon mass = 4902.8000
    # In Macro frame, let's scale these or just use raw for testing.
    # The integration test checks if gravity applies.
    earth_pos = [-0.17713544, 0.88743025, 0.38474363]
    moon_pos  = [-0.17908473, 0.88564736, 0.38423494]
    
    emitters = [0.0] * 8
    emitters[0] = earth_pos[0]; emitters[1] = earth_pos[1]; emitters[2] = earth_pos[2]
    emitters[3] = 398600.4418
    emitters[4] = moon_pos[0]; emitters[5] = moon_pos[1]; emitters[6] = moon_pos[2]
    emitters[7] = 4902.8000
    
    particles = [0.0] * (10 * subgroup_size * ((num_particles + subgroup_size - 1) // subgroup_size))
    
    morton_list = []
    
    # Bounding box for morton coding
    min_b = [-1.0, -1.0, -1.0]
    max_b = [1.0, 1.0, 1.0]
    
    for i in range(num_particles):
        block_idx = i // subgroup_size
        local_idx = i % subgroup_size
        base = block_idx * 10 * subgroup_size + local_idx
        
        # Position: distribute randomly or in clusters around Earth and Moon
        # Let's put 64 around Earth, 64 around Moon
        if i < 64:
            px = earth_pos[0] + (i * 0.001)
            py = earth_pos[1] + ((i % 8) * 0.001)
            pz = earth_pos[2]
            mass = 1.0
        else:
            px = moon_pos[0] + ((i - 64) * 0.001)
            py = moon_pos[1] + (((i - 64) % 8) * 0.001)
            pz = moon_pos[2]
            mass = 1.0

        particles[base + 0 * subgroup_size] = px
        particles[base + 1 * subgroup_size] = py
        particles[base + 2 * subgroup_size] = pz
        
        # Initial velocity
        particles[base + 3 * subgroup_size] = 0.0
        particles[base + 4 * subgroup_size] = 0.0
        particles[base + 5 * subgroup_size] = 0.0
        
        # Mass
        particles[base + 6 * subgroup_size] = mass
        
        # Normalized coordinates for morton
        nx = (px - min_b[0]) / (max_b[0] - min_b[0])
        ny = (py - min_b[1]) / (max_b[1] - min_b[1])
        nz = (pz - min_b[2]) / (max_b[2] - min_b[2])
        m_code = morton_3d(nx, ny, nz)
        morton_list.append((m_code, i))

    # Sort by morton code
    morton_list.sort(key=lambda x: x[0])
    
    sorted_morton = []
    for m in morton_list:
        sorted_morton.extend([m[0], m[1]])
        
    out = {
        "num_particles": num_particles,
        "num_emitters": 2,
        "dt": dt,
        "particles": particles,
        "emitters": emitters,
        "sorted_morton": sorted_morton
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/imex_integration.json")
    args = parser.parse_args()
    generate_test_data(args.out)
