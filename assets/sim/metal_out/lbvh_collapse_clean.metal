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

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif

#define BVH_FRAME_MACRO  0u
#define BVH_FRAME_MICRO  1u
#define BVH_SHAPE_AABB   0u
#define BVH_SHAPE_OBB    1u
#define BVH_SHAPE_SPHERE 2u

template<typename T>
inline T spvFindUMSB(T x) {
    return select(clz(T(0)) - (clz(x) + T(1)), T(-1), x == T(0));
}

struct MultiBvhNode {
    float min_x[SUBGROUP_SIZE]; float max_x[SUBGROUP_SIZE];
    float min_y[SUBGROUP_SIZE]; float max_y[SUBGROUP_SIZE];
    float min_z[SUBGROUP_SIZE]; float max_z[SUBGROUP_SIZE];
    uint child_indices[SUBGROUP_SIZE]; uint metadata[SUBGROUP_SIZE];
    float masses[SUBGROUP_SIZE];
    float com_x[SUBGROUP_SIZE]; float com_y[SUBGROUP_SIZE]; float com_z[SUBGROUP_SIZE];
    uint particle_start[SUBGROUP_SIZE]; uint particle_count[SUBGROUP_SIZE];
    uint2 valid_mask;
    uint parent_idx;
    uint pad;
    uint permutations[8][SUBGROUP_SIZE];
};

struct MultiBvhBuffer {
    MultiBvhNode nodes[1];
};

struct CollapseMapBuffer {
    uint binary_roots[1];
};

struct PushConstants {
    device MultiBvhBuffer* binary_bvh;
    device MultiBvhBuffer* multi_bvh;
    device CollapseMapBuffer* collapse_map;
    uint num_multi_nodes;
};

inline bool bvh_is_leaf(uint meta) { return (meta & 0x80000000u) != 0u; }
inline uint bvh_get_index(uint meta) { return meta & 0x07FFFFFFu; }
inline uint bvh_pack_metadata(bool is_leaf, uint frame, uint shape, uint index) {
    uint meta = index & 0x07FFFFFFu;
    meta |= (shape & 3u) << 27;
    meta |= (frame & 3u) << 29;
    if (is_leaf) meta |= 0x80000000u;
    return meta;
}

[[kernel]]
void lbvh_collapse(
    constant PushConstants& pc [[buffer(0)]],
    uint3 gl_WorkGroupID [[threadgroup_position_in_grid]],
    uint gl_SubgroupInvocationID [[thread_index_in_simdgroup]]
) {
    uint multi_node_idx = gl_WorkGroupID.x;
    if (multi_node_idx >= pc.num_multi_nodes) return;

    uint lane = gl_SubgroupInvocationID;
    uint binary_idx = pc.collapse_map->binary_roots[multi_node_idx];
    
    bool is_leaf = false;
    uint payload = 0;
    uint f_parent = 0;
    uint f_dir = 0;

    int depth = int(spvFindUMSB(SUBGROUP_SIZE)) - 1;
    for (int d = depth; d >= 0; d--) {
        uint dir = (lane >> uint(d)) & 1u;
        uint meta = pc.binary_bvh->nodes[binary_idx].metadata[dir];
        
        is_leaf = bvh_is_leaf(meta);
        uint next_idx = bvh_get_index(meta);

        f_parent = binary_idx;
        f_dir = dir;
        if (is_leaf) { payload = next_idx; break; }
        binary_idx = next_idx;
    }

    if (!is_leaf) {
        payload = binary_idx;
        f_parent = pc.binary_bvh->nodes[binary_idx].parent_idx;
        f_dir = (pc.binary_bvh->nodes[f_parent].child_indices[1] == binary_idx) ? 1u : 0u;
    }

    pc.multi_bvh->nodes[multi_node_idx].min_x[lane] = pc.binary_bvh->nodes[f_parent].min_x[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].max_x[lane] = pc.binary_bvh->nodes[f_parent].max_x[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].min_y[lane] = pc.binary_bvh->nodes[f_parent].min_y[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].max_y[lane] = pc.binary_bvh->nodes[f_parent].max_y[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].min_z[lane] = pc.binary_bvh->nodes[f_parent].min_z[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].max_z[lane] = pc.binary_bvh->nodes[f_parent].max_z[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].child_indices[lane] = payload;
    
    pc.multi_bvh->nodes[multi_node_idx].metadata[lane] = bvh_pack_metadata(is_leaf, BVH_FRAME_MICRO, BVH_SHAPE_AABB, payload);
    
    pc.multi_bvh->nodes[multi_node_idx].masses[lane] = pc.binary_bvh->nodes[f_parent].masses[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].com_x[lane] = pc.binary_bvh->nodes[f_parent].com_x[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].com_y[lane] = pc.binary_bvh->nodes[f_parent].com_y[f_dir];
    pc.multi_bvh->nodes[multi_node_idx].com_z[lane] = pc.binary_bvh->nodes[f_parent].com_z[f_dir];

    if (lane == 0u) {
        uint mask_x = (SUBGROUP_SIZE >= 32u) ? 0xFFFFFFFFu : ((1u << SUBGROUP_SIZE) - 1u);
        uint mask_y = 0u;
        if (SUBGROUP_SIZE > 32u) mask_y = (SUBGROUP_SIZE >= 64u) ? 0xFFFFFFFFu : ((1u << (SUBGROUP_SIZE - 32u)) - 1u);
        
        pc.multi_bvh->nodes[multi_node_idx].valid_mask = uint2(mask_x, mask_y);
        for (uint i = 0u; i < 8u; ++i) {
            for (uint j = 0u; j < SUBGROUP_SIZE; ++j) {
                pc.multi_bvh->nodes[multi_node_idx].permutations[i][j] = j;
            }
        }
    }
}
