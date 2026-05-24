// @assets/sim/morton_encode.comp
//
// Calculates a 30-bit Morton Code for each particle to be used for radix sorting
//
// Target: MSL Metal 3.0

struct PushConstants_morton_encode {
    MortonArray morton_out;
    ParticleData particles;
    uint num_particles;
    float3 scene_min;
    float3 scene_max;
};

// Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
inline uint morton_encode_expandBits(uint v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline uint morton_encode_morton3D(float3 norm_pos) {
    norm_pos = clamp(norm_pos, 0.0f, 1.0f);
    uint x = uint(norm_pos.x * 1023.0f);
    uint y = uint(norm_pos.y * 1023.0f);
    uint z = uint(norm_pos.z * 1023.0f);
    return (morton_encode_expandBits(x) << 2) | (morton_encode_expandBits(y) << 1) | morton_encode_expandBits(z);
}

[[kernel]]
void morton_encode(constant PushConstants_morton_encode& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint idx = thread_position_in_grid.x;
    if (idx >= pc.num_particles) return;

    // AOSOA unpacking matching your particle structure
    uint block_idx = idx / SUBGROUP_SIZE;
    uint local_idx = idx % SUBGROUP_SIZE;
    uint base = block_idx * (10 * SUBGROUP_SIZE) + local_idx;

    float3 pos = float3(
        P_READ(pc.particles, base + 0 * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 1 * SUBGROUP_SIZE),
        P_READ(pc.particles, base + 2 * SUBGROUP_SIZE)
    );

    // Normalize relative to scene bounds
    float3 extents = pc.scene_max - pc.scene_min;
    float3 norm_pos = (pos - pc.scene_min) / max(extents, float3(1e-5f));

    uint m_code = morton_encode_morton3D(norm_pos);

    pc.morton_out.entries[idx] = uint2(m_code, idx);
}