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

struct PushConstants_ccd {
    device struct MultiBvhBuffer* particle_bvh;
    device struct SparseCollisions* output_list;
    device struct ParticleData* particles;
    uint root_index;
    uint total_particles;
    float particle_radius;
    float dt;
};

[[kernel]]
void ccd(constant PushConstants_ccd& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint idx = thread_position_in_grid.x; 
    if (idx >= pc.total_particles) return;

    uint my_prim_id = idx;
    uint baseA = (my_prim_id / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (my_prim_id % SUBGROUP_SIZE);
    
    float3 my_center = float3(pc.particles->data[baseA+0], pc.particles->data[baseA+1*SUBGROUP_SIZE], pc.particles->data[baseA+2*SUBGROUP_SIZE]);
    float3 my_vel = float3(pc.particles->data[baseA+3*SUBGROUP_SIZE], pc.particles->data[baseA+4*SUBGROUP_SIZE], pc.particles->data[baseA+5*SUBGROUP_SIZE]);
    float3 p1 = my_center + my_vel * pc.dt;

    AABB my_aabb;
    my_aabb.minBounds = min(my_center - float3(pc.particle_radius), p1 - float3(pc.particle_radius));
    my_aabb.maxBounds = max(my_center + float3(pc.particle_radius), p1 + float3(pc.particle_radius));

    uint stack[64]; 
    int stackPtr = 0; 
    if (pc.root_index != 0xFFFFFFFFu) stack[stackPtr++] = pc.root_index;
    
    uint collisions_found = 0;

    while (stackPtr > 0) {
        uint node_idx = stack[--stackPtr];
        
        for (uint i = 0; i < SUBGROUP_SIZE; ++i) {
            if (!bvh_node_is_valid(pc.particle_bvh->nodes[node_idx].valid_mask, i)) continue;

            AABB bound;
            bound.minBounds = float3(pc.particle_bvh->nodes[node_idx].min_x[i], pc.particle_bvh->nodes[node_idx].min_y[i], pc.particle_bvh->nodes[node_idx].min_z[i]);
            bound.maxBounds = float3(pc.particle_bvh->nodes[node_idx].max_x[i], pc.particle_bvh->nodes[node_idx].max_y[i], pc.particle_bvh->nodes[node_idx].max_z[i]);

            if (intersectAABB(my_aabb, bound)) {
                uint meta = pc.particle_bvh->nodes[node_idx].metadata[i];
                uint offset = bvh_get_index(meta);

                if (bvh_is_leaf(meta)) {
                    if (my_prim_id < offset) {
                        float toi = 0.0, depth = 0.0; 
                        float3 normal = float3(0.0);
                        float3 point = float3(0.0);
                        
                        uint baseB = (offset / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (offset % SUBGROUP_SIZE);
                        float3 other_vel = float3(pc.particles->data[baseB+3*SUBGROUP_SIZE], pc.particles->data[baseB+4*SUBGROUP_SIZE], pc.particles->data[baseB+5*SUBGROUP_SIZE]) * pc.dt;
                        
                        float4x4 transA = float4x4(1.0); 
                        transA[3].xyz = my_center;
                        
                        float4x4 transB = float4x4(1.0); 
                        transB[3].xyz = float3(pc.particle_bvh->nodes[node_idx].com_x[i], pc.particle_bvh->nodes[node_idx].com_y[i], pc.particle_bvh->nodes[node_idx].com_z[i]);

                        if (compute_toi_generic(0, float3(pc.particle_radius,0,0), transA, my_vel * pc.dt, 0, float3(pc.particle_radius,0,0), transB, other_vel, 1e-3, 10, toi, normal, point, depth)) {
                            if (collisions_found < 16) {
                                uint outIdx = idx * 16 + collisions_found++;
                                pc.output_list->pairs[outIdx].valid = 1; 
                                pc.output_list->pairs[outIdx].prim_a = my_prim_id; 
                                pc.output_list->pairs[outIdx].prim_b = offset;
                                pc.output_list->pairs[outIdx].toi = toi; 
                                pc.output_list->pairs[outIdx].contact_normal = float4(normal, 0.0);
                                pc.output_list->pairs[outIdx].contact_point = float4(point, 1.0); 
                                pc.output_list->pairs[outIdx].penetration_depth = depth;
                            }
                        }
                    }
                } else if (offset != 0xFFFFFFFFu) {
                    stack[stackPtr++] = offset;
                }
            }
        }
    }
}
