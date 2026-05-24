#include <metal_stdlib>
using namespace metal;

#include "../debug_utils.metal"

struct ParticleData {
    uint id_low;
    uint id_high;
    uint age_low;
    uint age_high;
    packed_float3 position;
    float mass;
    packed_float3 velocity;
    uint is_active;
};

struct DrawIndirectCommand {
    uint vertexCount;
    uint instanceCount;
    uint firstVertex;
    uint firstInstance;
};

struct PushConstants {
    device float* aosoa_particles;
    device ParticleData* mega_particles;
    device DrawIndirectCommand* mega_indirect;
    device atomic_uint* atomic_counters;
    uint mega_indirect_index;
    uint mega_particle_offset;
};

constant uint SUBGROUP_SIZE = 32;

[[kernel]]
void convert_particles(
    uint thread_position_in_grid [[thread_position_in_grid]],
    constant PushConstants& pc [[buffer(0)]]
) {
    uint total_particles = atomic_load_explicit(&pc.atomic_counters[0], memory_order_relaxed);

    // Only thread 0 writes the indirect command
    if (thread_position_in_grid == 0) {
        pc.mega_indirect[pc.mega_indirect_index].vertexCount = 4;
        pc.mega_indirect[pc.mega_indirect_index].instanceCount = total_particles;
        pc.mega_indirect[pc.mega_indirect_index].firstVertex = 0;
        pc.mega_indirect[pc.mega_indirect_index].firstInstance = pc.mega_particle_offset;
    }

    uint idx = thread_position_in_grid;
    if (idx >= total_particles) {
        return;
    }

    uint in_block = idx / SUBGROUP_SIZE;
    uint in_lane  = idx % SUBGROUP_SIZE;
    uint in_base  = in_block * 10 * SUBGROUP_SIZE + in_lane;

    float3 pos;
    pos.x = pc.aosoa_particles[in_base + 0 * SUBGROUP_SIZE];
    pos.y = pc.aosoa_particles[in_base + 1 * SUBGROUP_SIZE];
    pos.z = pc.aosoa_particles[in_base + 2 * SUBGROUP_SIZE];

    float3 vel;
    vel.x = pc.aosoa_particles[in_base + 3 * SUBGROUP_SIZE];
    vel.y = pc.aosoa_particles[in_base + 4 * SUBGROUP_SIZE];
    vel.z = pc.aosoa_particles[in_base + 5 * SUBGROUP_SIZE];

    float mass = pc.aosoa_particles[in_base + 6 * SUBGROUP_SIZE];

    uint out_idx = pc.mega_particle_offset + idx;

    // We do not have IDs or Age from the physics simulation right now.
    // They could be added in emit_particles.comp in the future.
    pc.mega_particles[out_idx].id_low = 0;
    pc.mega_particles[out_idx].id_high = 0;
    pc.mega_particles[out_idx].age_low = 0;
    pc.mega_particles[out_idx].age_high = 0;
    pc.mega_particles[out_idx].position = pos;
    pc.mega_particles[out_idx].mass = mass;
    pc.mega_particles[out_idx].velocity = vel;
    pc.mega_particles[out_idx].is_active = 1;
}