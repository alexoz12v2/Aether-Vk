// @assets/sim/bp_bounds_gen.comp

struct PushConstants_bp_bounds_gen {
    device RigidBody* scene_entities;
    device uint* particles;
    device TLASLeaf* tlas_leaves;
    uint2    dt_us;
    uint     total_entities;
    uint     num_rigid_bodies;
    float    particle_radius;
};

[[kernel]]
void bp_bounds_gen(
    constant PushConstants_bp_bounds_gen& pc [[buffer(0)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]
) {
    uint id = thread_position_in_grid.x;
    if (id >= pc.total_entities) return;

    float dt = dt_to_seconds(pc.dt_us);
    float3 center, extents, vel;
    uint shape_type;
    uint64_t bda;

    if (id < pc.num_rigid_bodies) {
        device RigidBody& body = pc.scene_entities[id];
        center = body.position_mass.xyz;
        extents = body.shape_extents;
        vel = body.linear_vel_drag.xyz;
        shape_type = body.shape_type;
        bda = (uint64_t)&body;
    } else {
        uint particle_system_idx = id - pc.num_rigid_bodies;
        // The bounds of a particle system should ideally be computed over all its particles.
        // For now, since particles are grouped into entities (32 particles per entity),
        // we approximate the bounds using the center of the first particle in the group.
        uint base = particle_system_idx * (10 * SUBGROUP_SIZE);
        
        center = float3(
            as_type<float>(pc.particles[base + 0]),
            as_type<float>(pc.particles[base + 1 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 2 * SUBGROUP_SIZE])
        );
        extents = float3(pc.particle_radius * 16.0); // Rough approximation for 32 particles
        vel = float3(
            as_type<float>(pc.particles[base + 3 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 4 * SUBGROUP_SIZE]),
            as_type<float>(pc.particles[base + 5 * SUBGROUP_SIZE])
        );
        shape_type = BVH_SHAPE_SPHERE;
        bda = (uint64_t)&pc.particles[base]; // Address of this chunk of 32 particles
    }

    float3 static_min = center - extents;
    float3 static_max = center + extents;
    float3 sweep = vel * dt;

    device TLASLeaf& leaf = pc.tlas_leaves[id];
    leaf.min_bound = min(static_min, static_min + sweep);
    leaf.max_bound = max(static_max, static_max + sweep);
    leaf.entity_idx = id;
    leaf.metadata = bvh_pack_metadata(true, BVH_FRAME_MACRO, shape_type, id);
    leaf.bda = bda;
}
