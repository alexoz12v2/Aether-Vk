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
// @assets/sim/radix_sort.comp

struct PushConstants_radix_sort {
    device uint2* input_keys;
    device uint2* output_keys;
    device uint* histograms;
    uint num_particles;
    uint shift;
    uint stage;
    uint num_blocks;
};

#define STAGE_COUNT   0
#define STAGE_SCAN    1
#define STAGE_SCATTER 2

#define RADIX 16
#define ELEMENTS_PER_BLOCK 4096

[[kernel]]
void radix_sort(
    constant PushConstants_radix_sort& pc [[buffer(0)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroups_per_threadgroup [[simdgroups_per_threadgroup]]
) {
    uint lid = thread_position_in_threadgroup.x;
    uint wid = threadgroup_position_in_grid.x;
    uint sg_id = thread_index_in_simdgroup;
    uint sg_group_id = simdgroup_index_in_threadgroup;
    
    threadgroup atomic_uint s_counts[RADIX];
    threadgroup uint s_offsets[RADIX];
    threadgroup uint s_sg_counts[64];
    threadgroup uint s_bin_sums[RADIX];

    if (pc.stage == STAGE_COUNT) {
        if (lid < RADIX) atomic_store_explicit(&s_counts[lid], 0, memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint block_start = wid * ELEMENTS_PER_BLOCK;
        uint block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);

        for (uint i = block_start + lid; i < block_end; i += 256) {
            uint key = (pc.input_keys[i].x >> pc.shift) & 0xFu;
            atomic_fetch_add_explicit(&s_counts[key], 1, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < RADIX) {
            pc.histograms[lid * pc.num_blocks + wid] = atomic_load_explicit(&s_counts[lid], memory_order_relaxed);
        }
    }
    else if (pc.stage == STAGE_SCAN) {
        if (lid < RADIX) {
            uint bin_sum = 0;
            for (uint w = 0; w < pc.num_blocks; ++w) {
                bin_sum += pc.histograms[lid * pc.num_blocks + w];
            }
            s_bin_sums[lid] = bin_sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid == 0) {
            uint global_offset = 0;
            for (uint i = 0; i < RADIX; ++i) {
                uint val = s_bin_sums[i];
                s_bin_sums[i] = global_offset;
                global_offset += val;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (lid < RADIX) {
            uint running_offset = s_bin_sums[lid];
            for (uint w = 0; w < pc.num_blocks; ++w) {
                uint val = pc.histograms[lid * pc.num_blocks + w];
                pc.histograms[lid * pc.num_blocks + w] = running_offset;
                running_offset += val;
            }
        }
    }
    else if (pc.stage == STAGE_SCATTER) {
        if (lid < RADIX) {
            s_offsets[lid] = pc.histograms[lid * pc.num_blocks + wid];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint block_start = wid * ELEMENTS_PER_BLOCK;
        uint block_end = min(block_start + ELEMENTS_PER_BLOCK, pc.num_particles);

        for (uint chunk_start = block_start; chunk_start < block_end; chunk_start += 256) {
            uint i = chunk_start + lid;
            bool valid = (i < block_end);
            
            uint2 my_entry = uint2(0, 0);
            if (valid) {
                my_entry = pc.input_keys[i];
            }
            uint my_key = valid ? ((my_entry.x >> pc.shift) & 0xFu) : 0xFFFFFFFFu;

            uint local_offset = 0;
            uint my_global_base = 0;

            for (uint b = 0; b < RADIX; ++b) {
                bool match = (my_key == b);

                uint sg_match_count = simd_sum(match ? 1 : 0);
                uint my_sg_offset = simd_prefix_exclusive_sum(match ? 1 : 0);

                if (sg_id == 0) {
                    s_sg_counts[sg_group_id] = sg_match_count;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (lid == 0) {
                    uint sum = 0;
                    for (uint sg = 0; sg < simdgroups_per_threadgroup; ++sg) {
                        uint c = s_sg_counts[sg];
                        s_sg_counts[sg] = sum;
                        sum += c;
                    }
                    atomic_store_explicit(&s_counts[b], sum, memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);

                if (match) {
                    local_offset = s_sg_counts[sg_group_id] + my_sg_offset;
                    my_global_base = s_offsets[b];
                }

                if (lid == 0) {
                    s_offsets[b] += atomic_load_explicit(&s_counts[b], memory_order_relaxed);
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (valid) {
                uint dest = my_global_base + local_offset;
                pc.output_keys[dest] = my_entry;
            }
        }
    }
}
