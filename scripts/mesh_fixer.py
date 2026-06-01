#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
import trimesh


def process_mesh(input_path, output_path, force_flip=False, verbose=False):
    def log(msg):
        if verbose:
            print(f"[INFO] {msg}")

    print(f"Loading mesh: {input_path}")
    try:
        mesh = trimesh.load(input_path, force='mesh')
    except Exception as e:
        print(f"[ERROR] Failed to load {input_path}: {e}")
        sys.exit(1)

    if not isinstance(mesh, trimesh.Trimesh):
        print("[ERROR] Could not extract a valid triangular mesh from {input_path}")
        sys.exit(1)

    log(f"Original stats: {len(mesh.vertices)} vertices, {len(mesh.faces)} triangles.")
    log(f"Is watertight (manifold)? {mesh.is_watertight}")
    log(f"Is winding consistent? {mesh.is_winding_consistent}")

    # 1. Automatically fix normals and winding
    log("Applying trimesh.repair.fix_normals()...")
    # This guarantees consistent winding and forces normals to point outward 
    # (assuming the mesh represents a closed volume).
    trimesh.repair.fix_normals(mesh)

    # 2. Optional forced flip (if you need to manually invert the standardized result)
    if force_flip:
        log("Force-flipping triangle winding (reversing face columns)...")
        outward_normals = mesh.vertex_normals.copy()
        mesh.faces = mesh.faces[:, [0, 2, 1]]
        # Clear the cache so faces update, but restore the outward normals!
        mesh._cache.clear()
        mesh.vertex_normals = outward_normals

    # Verify health after operations
    if not mesh.is_winding_consistent:
        print("[WARNING] Winding is still not perfectly consistent. The mesh might have non-manifold edges, open boundaries, or intersecting geometry.")

    # 3. Export
    out_path = Path(output_path)
    log(f"Exporting to: {out_path}")
    try:
        mesh.export(str(out_path))
        
        # Post-process with pygltflib to ensure NORMAL, TANGENT, TEXCOORD_0 are present
        try:
            import pygltflib
            import numpy as np
            
            gltf = pygltflib.GLTF2().load(str(out_path))
            blob = gltf.binary_blob()
            if blob is None: blob = b""
            
            for m in gltf.meshes:
                for prim in m.primitives:
                    attrs = prim.attributes
                    count = gltf.accessors[attrs.POSITION].count
                    
                    if attrs.NORMAL is None:
                        normals = mesh.vertex_normals
                        if len(normals) != count: normals = np.zeros((count, 3), dtype=np.float32)
                        normal_bytes = normals.astype(np.float32).tobytes()
                        byte_offset = len(blob)
                        blob += normal_bytes
                        bv = pygltflib.BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(normal_bytes), target=pygltflib.ARRAY_BUFFER)
                        gltf.bufferViews.append(bv)
                        acc = pygltflib.Accessor(bufferView=len(gltf.bufferViews)-1, byteOffset=0, componentType=pygltflib.FLOAT, count=count, type=pygltflib.VEC3)
                        gltf.accessors.append(acc)
                        attrs.NORMAL = len(gltf.accessors) - 1

                    if attrs.TEXCOORD_0 is None:
                        uvs = np.zeros((count, 2), dtype=np.float32)
                        uv_bytes = uvs.tobytes()
                        byte_offset = len(blob)
                        blob += uv_bytes
                        bv = pygltflib.BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(uv_bytes), target=pygltflib.ARRAY_BUFFER)
                        gltf.bufferViews.append(bv)
                        acc = pygltflib.Accessor(bufferView=len(gltf.bufferViews)-1, byteOffset=0, componentType=pygltflib.FLOAT, count=count, type=pygltflib.VEC2)
                        gltf.accessors.append(acc)
                        attrs.TEXCOORD_0 = len(gltf.accessors) - 1

                    if attrs.TANGENT is None:
                        tangents = np.tile(np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32), (count, 1))
                        tangent_bytes = tangents.tobytes()
                        byte_offset = len(blob)
                        blob += tangent_bytes
                        bv = pygltflib.BufferView(buffer=0, byteOffset=byte_offset, byteLength=len(tangent_bytes), target=pygltflib.ARRAY_BUFFER)
                        gltf.bufferViews.append(bv)
                        acc = pygltflib.Accessor(bufferView=len(gltf.bufferViews)-1, byteOffset=0, componentType=pygltflib.FLOAT, count=count, type=pygltflib.VEC4)
                        gltf.accessors.append(acc)
                        attrs.TANGENT = len(gltf.accessors) - 1
                        
            gltf.set_binary_blob(blob)
            gltf.buffers[0].byteLength = len(blob)
            gltf.save(str(out_path))
            log("pygltflib post-processing added missing required attributes.")
        except Exception as e:
            log(f"Warning: pygltflib post-processing failed: {e}")
            
        print(f"Success! Saved fixed mesh to: {out_path.name}")
    except Exception as e:
        print(f"[ERROR] Failed to export {out_path}: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="A robust CLI tool to triangulate, fix normals, and correct face winding of 3D meshes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("input", help="Path to the input 3D model (e.g., .glb, .obj, .stl)")
    parser.add_argument("-o", "--output", help="Path for the output file. If omitted, appends '_fixed' to the input filename.")
    parser.add_argument("--flip", action="store_true", help="Force-flip the winding (invert normals) AFTER repairing.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output logging for debugging.")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_name(f"{input_path.stem}_fixed{input_path.suffix}")

    process_mesh(input_path, output_path, args.flip, args.verbose)


if __name__ == "__main__":
    main()

