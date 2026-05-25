#include <metal_stdlib>
using namespace metal;


#ifndef TYPE_PARTICLE_SYSTEM
#define TYPE_PARTICLE_SYSTEM 0
#define TYPE_RIGID_BODY      1
#define TYPE_MICRO_LCA       2
#endif

struct PairBuffer {
    atomic_uint count;
    uint pad;
    uint2 pairs[1];
};

struct TLASLeaf {
    packed_float3 min_bound;
    uint entity_idx;
    packed_float3 max_bound;
    uint metadata;
    uint64_t bda;
    uint pad[2];
};

struct LeafBuffer {
    TLASLeaf leaves[1];
};

struct EntityHeader {
    uint ty;
    uint pad[3];
};

struct PushConstants {
    uint64_t raw_pairs;
    uint2 out_rb_rb;
    uint2 out_rb_ps;
    uint2 out_ps_ps;
    uint64_t tlas_leaves;
    uint max_pairs;
    uint num_rigid_bodies;
};

[[kernel]]
void bp_classify(
    constant PushConstants& pc [[buffer(0)]],
    uint id [[thread_position_in_grid]]
) {
    device PairBuffer* raw_pairs = (device PairBuffer*)(pc.raw_pairs);
    uint count = atomic_load_explicit(&raw_pairs->count, memory_order_relaxed);
    if (id >= count) return;

    uint2 pair = raw_pairs->pairs[id];
    uint ent_A = pair.x;
    uint ent_B = pair.y;

    device LeafBuffer* tlas_leaves = (device LeafBuffer*)(pc.tlas_leaves);
    uint64_t bda_A = tlas_leaves->leaves[ent_A].bda;
    uint64_t bda_B = tlas_leaves->leaves[ent_B].bda;

    device EntityHeader* header_A = (device EntityHeader*)(bda_A);
    device EntityHeader* header_B = (device EntityHeader*)(bda_B);

    uint type_A = header_A->ty;
    uint type_B = header_B->ty;

    if (type_A > type_B) {
        uint temp = ent_A; ent_A = ent_B; ent_B = temp;
        temp = type_A; type_A = type_B; type_B = temp;
    }

    if (type_A == TYPE_PARTICLE_SYSTEM && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_ps_ps.x != 0 || pc.out_ps_ps.y != 0) {
            uint64_t ptr_val = ((uint64_t)pc.out_ps_ps.y << 32) | pc.out_ps_ps.x;
            device PairBuffer* buf = (device PairBuffer*)(ptr_val);
            uint out_idx = atomic_fetch_add_explicit(&buf->count, 1, memory_order_relaxed);
            if (out_idx < pc.max_pairs) {
                buf->pairs[out_idx] = uint2(ent_A, ent_B);
            }
        }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_PARTICLE_SYSTEM) {
        if (pc.out_rb_ps.x != 0 || pc.out_rb_ps.y != 0) {
            uint64_t ptr_val = ((uint64_t)pc.out_rb_ps.y << 32) | pc.out_rb_ps.x;
            device PairBuffer* buf = (device PairBuffer*)(ptr_val);
            uint out_idx = atomic_fetch_add_explicit(&buf->count, 1, memory_order_relaxed);
            if (out_idx < pc.max_pairs) {
                buf->pairs[out_idx] = uint2(ent_A, ent_B);
            }
        }
    } else if (type_A == TYPE_RIGID_BODY && type_B == TYPE_RIGID_BODY) {
        if (pc.out_rb_rb.x != 0 || pc.out_rb_rb.y != 0) {
            uint64_t ptr_val = ((uint64_t)pc.out_rb_rb.y << 32) | pc.out_rb_rb.x;
            device PairBuffer* buf = (device PairBuffer*)(ptr_val);
            uint out_idx = atomic_fetch_add_explicit(&buf->count, 1, memory_order_relaxed);
            if (out_idx < pc.max_pairs) {
                buf->pairs[out_idx] = uint2(ent_A, ent_B);
            }
        }
    }
}
