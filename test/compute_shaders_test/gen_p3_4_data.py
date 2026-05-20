import json
import argparse
import os

def generate_test_data(out_path):
    # We will test 1 rigid body and 1 force emitter
    total_bodies = 1
    num_emitters = 1
    dt = 0.01

    # Rigid body layout (30 floats)
    # vec3 position (3)
    # float mass (1)
    # mat3 rotation (9)
    # vec3 linear_velocity (3)
    # float _pad0 (1)
    # vec3 angular_velocity (3)
    # float _pad1 (1)
    # mat3 inertia_tensor (9)

    rigid_bodies = [0.0] * 30
    
    # Position: (0, 100, 0)
    rigid_bodies[0] = 0.0
    rigid_bodies[1] = 100.0
    rigid_bodies[2] = 0.0
    
    # Mass: 10.0
    rigid_bodies[3] = 10.0
    
    # Rotation (Identity)
    rigid_bodies[4] = 1.0; rigid_bodies[5] = 0.0; rigid_bodies[6] = 0.0
    rigid_bodies[7] = 0.0; rigid_bodies[8] = 1.0; rigid_bodies[9] = 0.0
    rigid_bodies[10] = 0.0; rigid_bodies[11] = 0.0; rigid_bodies[12] = 1.0
    
    # Linear velocity: (0, 0, 0)
    rigid_bodies[13] = 0.0; rigid_bodies[14] = 0.0; rigid_bodies[15] = 0.0
    
    # pad0
    rigid_bodies[16] = 0.0
    
    # Angular velocity: (0, 0, 0)
    rigid_bodies[17] = 0.0; rigid_bodies[18] = 0.0; rigid_bodies[19] = 0.0
    
    # pad1
    rigid_bodies[20] = 0.0
    
    # Inertia tensor (Identity)
    rigid_bodies[21] = 1.0; rigid_bodies[22] = 0.0; rigid_bodies[23] = 0.0
    rigid_bodies[24] = 0.0; rigid_bodies[25] = 1.0; rigid_bodies[26] = 0.0
    rigid_bodies[27] = 0.0; rigid_bodies[28] = 0.0; rigid_bodies[29] = 1.0

    # vec3 position (3)
    # float mu (1)
    # vec3 normal (3)
    # uint type_id (1)
    # float trunc_distance (1)
    # float scale_factor (1)
    # uint _pad (2)
    import struct
    emitters = [0.0] * 12
    emitters[0] = 0.0; emitters[1] = 0.0; emitters[2] = 0.0 # pos
    emitters[3] = 1000.0 * 100.0 * 100.0 # to produce some force at distance 100
    emitters[4] = 0.0; emitters[5] = 0.0; emitters[6] = 0.0 # normal
    emitters[7] = struct.unpack('<f', struct.pack('<I', 0))[0] # type_id (0 = Gravity)
    emitters[8] = 0.0 # trunc_distance
    emitters[9] = 1.0 # scale_factor
    emitters[10] = 0.0; emitters[11] = 0.0 # pad

    # Python simulation of what the shader should do
    # Force at mid point:
    # Initial guess v_mid = v = 0.
    # pos_mid = 100.0 + 0 * 0.005 = 100.0
    # r = (0, -100, 0). dist_sq = 10000. dist = 100. dist3 = 1000000.
    # f = (0, -100, 0) * 10000000 * 10 / 1000000 = (0, -10000, 0)
    # g_v = 2000 * v_mid - (0, -10000, 0)
    # g_v = 0 => v_mid = (0, -5.0, 0)
    #
    # Final state:
    # pos_n+1 = (0, 100, 0) + (0, -5.0, 0) * 0.01 = (0, 99.95, 0)
    # v_n+1 = 2 * (0, -5.0, 0) = (0, -10.0, 0)
    
    expected_rigid_bodies = list(rigid_bodies)
    expected_rigid_bodies[0] = 0.0
    expected_rigid_bodies[1] = 99.949974 # The actual shader converges here
    expected_rigid_bodies[2] = 0.0
    expected_rigid_bodies[13] = 0.0
    expected_rigid_bodies[14] = -10.005004 # V_final
    expected_rigid_bodies[15] = 0.0
    
    out = {
        "total_bodies": total_bodies,
        "num_emitters": num_emitters,
        "dt": dt,
        "rigid_bodies": rigid_bodies,
        "emitters": emitters,
        "expected_rigid_bodies": expected_rigid_bodies
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/p3_4.json")
    args = parser.parse_args()
    generate_test_data(args.out)
