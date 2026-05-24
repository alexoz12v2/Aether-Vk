struct PushConstants_lbvh_build {
    MultiBvhBuffer bvh;
    MortonArray sorted_morton;
    AtomicCounters counters;
    ParticleData particles;
    uint num_primitives;
    float particle_radius;
    float dt;
};

int common_prefix(constant PushConstants_lbvh_build& pc, uint n, int i, int j) {
    if (j < 0 || j >= (int)n) return -1;
    uint key1 = pc.sorted_morton.entries[i].x; uint key2 = pc.sorted_morton.entries[j].x;
    if (key1 == key2) {
        uint idx1 = pc.sorted_morton.entries[i].y; uint idx2 = pc.sorted_morton.entries[j].y;
        return 32 + (31 - clz(idx1 ^ idx2));
    }
    return 31 - clz(key1 ^ key2);
}

float2 determine_range(constant PushConstants_lbvh_build& pc, uint n, int i) {
    int d = sign((float)(common_prefix(pc, n, i, i + 1) - common_prefix(pc, n, i, i - 1)));
    int min_p = common_prefix(pc, n, i, i - d), l_max = 2;
    while (common_prefix(pc, n, i, i + l_max * d) > min_p) l_max *= 2;
    int l = 0, t = l_max / 2;
    while (t >= 1) { if (common_prefix(pc, n, i, i + (l + t) * d) > min_p) l += t; t /= 2; }
    return float2(min(i, i + l * d), max(i, i + l * d));
}

int find_split(constant PushConstants_lbvh_build& pc, uint n, int first, int last) {
    int common_node = common_prefix(pc, n, first, last), split = first, step = last - first;
    do {
        step = (step + 1) >> 1; int new_split = split + step;
        if (new_split < last && common_prefix(pc, n, first, new_split) > common_node) split = new_split;
    } while (step > 1);
    return split;
}

[[kernel]]
void lbvh_build(constant PushConstants_lbvh_build& pc [[buffer(0)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
    uint idx = thread_position_in_grid.x, n = pc.num_primitives;
    if (idx >= n) return;
    uint num_internal_nodes = n - 1;

    if (idx < num_internal_nodes) {
        float2 range = determine_range(pc, n, int(idx));
        int split = find_split(pc, n, int(range.x), int(range.y));
        uint left_child = (split == int(range.x)) ? (num_internal_nodes + split) : uint(split);
        uint right_child = (split + 1 == int(range.y)) ? (num_internal_nodes + split + 1) : uint(split + 1);

        pc.bvh.nodes[idx].child_indices[0] = left_child;
        pc.bvh.nodes[idx].child_indices[1] = right_child;
        pc.bvh.nodes[idx].valid_mask = uint2(3u, 0u);
        pc.bvh.nodes[left_child].parent_idx = idx;
        pc.bvh.nodes[right_child].parent_idx = idx;
    }

    uint leaf_idx = num_internal_nodes + idx, p_id = pc.sorted_morton.entries[idx].y;
    uint base = (p_id / SUBGROUP_SIZE) * (10 * SUBGROUP_SIZE) + (p_id % SUBGROUP_SIZE);

    float3 pos = float3(P_READ(pc.particles, base+0), P_READ(pc.particles, base+1*SUBGROUP_SIZE), P_READ(pc.particles, base+2*SUBGROUP_SIZE));
    float3 vel = float3(P_READ(pc.particles, base+3*SUBGROUP_SIZE), P_READ(pc.particles, base+4*SUBGROUP_SIZE), P_READ(pc.particles, base+5*SUBGROUP_SIZE));
    float mass = P_READ(pc.particles, base+6*SUBGROUP_SIZE), r = pc.particle_radius;

    float3 p1 = pos + vel * pc.dt;
    float3 l_min = min(pos - float3(r), p1 - float3(r)), l_max = max(pos + float3(r), p1 + float3(r));

    uint current = pc.bvh.nodes[leaf_idx].parent_idx;
    uint is_right = (pc.bvh.nodes[current].child_indices[1] == leaf_idx) ? 1 : 0;

    pc.bvh.nodes[current].min_x[is_right] = l_min.x; pc.bvh.nodes[current].max_x[is_right] = l_max.x;
    pc.bvh.nodes[current].min_y[is_right] = l_min.y; pc.bvh.nodes[current].max_y[is_right] = l_max.y;
    pc.bvh.nodes[current].min_z[is_right] = l_min.z; pc.bvh.nodes[current].max_z[is_right] = l_max.z;
    pc.bvh.nodes[current].masses[is_right] = mass;
    pc.bvh.nodes[current].com_x[is_right] = pos.x; pc.bvh.nodes[current].com_y[is_right] = pos.y; pc.bvh.nodes[current].com_z[is_right] = pos.z;
    pc.bvh.nodes[current].metadata[is_right] = bvh_pack_metadata(true, BVH_FRAME_MICRO, BVH_SHAPE_AABB, p_id);

    threadgroup_barrier(mem_flags::mem_device);

    while (current != 0xFFFFFFFFu) {
        if (atomic_fetch_add_explicit((device atomic_uint*)&pc.counters.counts[current], 1, memory_order_relaxed) == 0) break;

        float3 c_l_min = float3(pc.bvh.nodes[current].min_x[0], pc.bvh.nodes[current].min_y[0], pc.bvh.nodes[current].min_z[0]);
        float3 c_l_max = float3(pc.bvh.nodes[current].max_x[0], pc.bvh.nodes[current].max_y[0], pc.bvh.nodes[current].max_z[0]);
        float l_m = pc.bvh.nodes[current].masses[0];
        float3 l_com = float3(pc.bvh.nodes[current].com_x[0], pc.bvh.nodes[current].com_y[0], pc.bvh.nodes[current].com_z[0]);

        float3 c_r_min = float3(pc.bvh.nodes[current].min_x[1], pc.bvh.nodes[current].min_y[1], pc.bvh.nodes[current].min_z[1]);
        float3 c_r_max = float3(pc.bvh.nodes[current].max_x[1], pc.bvh.nodes[current].max_y[1], pc.bvh.nodes[current].max_z[1]);
        float r_m = pc.bvh.nodes[current].masses[1];
        float3 r_com = float3(pc.bvh.nodes[current].com_x[1], pc.bvh.nodes[current].com_y[1], pc.bvh.nodes[current].com_z[1]);

        float3 c_min = min(c_l_min, c_r_min), c_max = max(c_l_max, c_r_max);
        float c_mass = l_m + r_m;
        float3 c_com = c_mass > 0.0 ? (l_com * l_m + r_com * r_m) / c_mass : (l_com + r_com) * 0.5;

        uint parent = pc.bvh.nodes[current].parent_idx;
        if (parent != 0xFFFFFFFFu) {
            uint is_r = (pc.bvh.nodes[parent].child_indices[1] == current) ? 1 : 0;
            pc.bvh.nodes[parent].min_x[is_r] = c_min.x; pc.bvh.nodes[parent].max_x[is_r] = c_max.x;
            pc.bvh.nodes[parent].min_y[is_r] = c_min.y; pc.bvh.nodes[parent].max_y[is_r] = c_max.y;
            pc.bvh.nodes[parent].min_z[is_r] = c_min.z; pc.bvh.nodes[parent].max_z[is_r] = c_max.z;
            pc.bvh.nodes[parent].masses[is_r] = c_mass;
            pc.bvh.nodes[parent].com_x[is_r] = c_com.x; pc.bvh.nodes[parent].com_y[is_r] = c_com.y; pc.bvh.nodes[parent].com_z[is_r] = c_com.z;
            pc.bvh.nodes[parent].metadata[is_r] = bvh_pack_metadata(false, BVH_FRAME_MICRO, BVH_SHAPE_AABB, current);
        }
        threadgroup_barrier(mem_flags::mem_device);
        current = parent;
    }
}
