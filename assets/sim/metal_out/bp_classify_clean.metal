#include <metal_stdlib>
#include <metal_atomic>
#include <metal_simdgroup>
using namespace metal;

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

// Shared macros
#define P_READ(addr, offset) as_type<float>((addr)[(offset)])
#define P_WRITE(addr, offset, val) ((addr)[(offset)] = as_type<uint>(val))

// Typedefs
typedef device uint* ParticleData;

// Structs
struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE]; float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE]; float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE]; float max_z[SUBGROUP_SIZE];
    uint  child_indices[SUBGROUP_SIZE]; uint metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE]; float com_y[SUBGROUP_SIZE]; float com_z[SUBGROUP_SIZE];
    uint  particle_start[SUBGROUP_SIZE]; uint particle_count[SUBGROUP_SIZE];
    uint2 valid_mask;
    uint  parent_idx;
    uint  pad;
    uint  permutations[8][SUBGROUP_SIZE];
};

struct DepthIndices {
    uint indices[1];
};

struct Wrench { uint force_x; uint force_y; uint force_z; uint torque_x; uint torque_y; uint torque_z; };

// Atomics
inline void AtomicAddFloatBDA(device uint* addr, uint offset, float val) {
    device atomic_uint* a = (device atomic_uint*)(addr + offset);
    uint e = atomic_load_explicit(a, memory_order_relaxed);
    while (!atomic_compare_exchange_weak_explicit(a, &e, as_type<uint>(as_type<float>(e) + val), memory_order_relaxed, memory_order_relaxed));
}

// Bvh helpers
inline bool bvh_node_is_valid(uint2 mask, uint lane_id) {
    if (lane_id < 32) return (mask.x & (1u << lane_id)) != 0;
    return (mask.y & (1u << (lane_id - 32))) != 0;
}

inline bool bvh_is_leaf(uint metadata) {
    return (metadata & 0x80000000) != 0;
}

inline uint bvh_leaf_count(uint metadata) {
    return metadata & 0x7FFFFFFF;
}
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
