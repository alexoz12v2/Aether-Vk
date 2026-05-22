#ifndef PHYSICS_TYPES_GLSL
#define PHYSICS_TYPES_GLSL

struct RigidBody {
    vec4 position_mass;       // xyz: world pos, w: mass
    vec4 orientation;         // quaternion (x,y,z,w)
    vec4 linear_vel_drag;     // xyz: linear velocity, w: linear damping coeff
    vec4 angular_vel_drag;    // xyz: angular velocity, w: angular damping coeff
    vec4 inertia_tensor_inv;  // xyz: diagonal local inverse inertia
    uint wrench_idx;          // Pointer to the associated accumulated Wrench
    uint leaf_start_idx;      // Mapping to the first leaf Wrench (for rb_force_assign)
    uint leaf_count;          // Number of leaves
    uint shape_type;          // BVH_SHAPE_*
    vec3 shape_extents;       // Dimensions
    uint pad2;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) buffer RigidBodyArray {
    RigidBody bodies[];
};

struct ForceEmitter {
    vec3 position;
    float mu; // type_id = 0 -> G * M, type_id = 1 -> base force
    vec3 normal; // type_id = 0 -> unused, type_id = 1 -> plane normal
    uint type_id; // 0 = Gravity, 1 = Planar
    float trunc_distance; // type_id = 0 -> unused, type_id = 1 -> max distance point - plane over which force is applied
    float scale_factor;
    uint _pad[2];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer EmitterArray {
    ForceEmitter emitters[];
};

struct KinematicBody {
    uint entity_id_low;
    uint entity_id_high;
    vec3 position;
    vec4 rotation;
    vec3 scale_vec;
    vec3 velocity;
    uint parent_frame_id;
    float mu;
    uint own_frame_id;
    uint frame_type;
    float scale;
    uint shape_type;
    float shape_data[3];
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer KinematicArray {
    KinematicBody bodies[];
};

#endif // PHYSICS_TYPES_GLSL
