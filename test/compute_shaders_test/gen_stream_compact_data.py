import json
import argparse
import os
import random

def generate_test_data(num_elements, out_path):
    sparse_data = []
    expected_pairs = []
    
    random.seed(42)
    for i in range(num_elements):
        # 30% chance to be valid
        valid = 1 if random.random() < 0.3 else 0
        particle_a = i
        particle_b = (i + 1) * 2
        
        # sparse array has SparseCollisionData (11 words)
        sparse_data.extend([valid, particle_a, particle_b, 0, 0, 0, 0, 0, 0, 0, 0])
        
        if valid == 1:
            expected_pairs.extend([particle_a, particle_b])

    out = {
        "total_elements": num_elements,
        "sparse_in": sparse_data,
        "expected_count": len(expected_pairs) // 2,
        "expected_pairs": expected_pairs
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Generated {out_path} with {num_elements} elements")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-elements", type=int, default=1000)
    parser.add_argument("--out", type=str, default="test/compute_shaders_test/test_data/stream_compact.json")
    
    args = parser.parse_args()
    generate_test_data(args.num_elements, args.out)
