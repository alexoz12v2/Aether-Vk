#if defined(KERNEL_stream_compact)

struct PushConstants_stream_compact {
    device void* sparse_in;
    device void* packed_out;
    uint total_elements;
};

[[kernel]]
void stream_compact(
    constant PushConstants_stream_compact& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
#ifdef DEBUG_SHADERS
    if (thread_position_in_grid.x == 0 && thread_position_in_grid.y == 0 && thread_position_in_grid.z == 0) {
        // MSL doesn't typically support debugPrintfEXT natively, but we can log or ignore
    }
#endif

    uint id = thread_position_in_grid.x;
    
    device uint* sparse_in_count = (device uint*)pc.sparse_in;
    uint in_count = *sparse_in_count;

    if (id == 0) {
        device uint* packed_out_dispatch = (device uint*)pc.packed_out;
        packed_out_dispatch[3] = in_count; // count at offset 12
        uint blocks = (in_count + 127) / 128;
        packed_out_dispatch[0] = blocks;   // dispatch_x
        packed_out_dispatch[1] = 1;        // dispatch_y
        packed_out_dispatch[2] = 1;        // dispatch_z
    }

    if (id < in_count) {
        device SparseCollisionData* sparse_pairs = (device SparseCollisionData*)((device char*)pc.sparse_in + 16);
        device PackedPair* packed_pairs = (device PackedPair*)((device char*)pc.packed_out + 16);
        
        SparseCollisionData in_data = sparse_pairs[id];
        
        packed_pairs[id].a.entity_id = in_data.entity_a;
        packed_pairs[id].a.primitive_index = in_data.prim_a;
        packed_pairs[id].b.entity_id = in_data.entity_b;
        packed_pairs[id].b.primitive_index = in_data.prim_b;
        packed_pairs[id].toi = in_data.toi;
        packed_pairs[id].contact_normal = in_data.contact_normal;
        packed_pairs[id].contact_point = in_data.contact_point;
        packed_pairs[id].penetration_depth = in_data.penetration_depth;
    }
}

#endif // KERNEL_stream_compact