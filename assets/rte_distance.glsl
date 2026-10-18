#ifndef RTE_DISTANCE_GLSL
#define RTE_DISTANCE_GLSL

// Shared by common.glsl (physical_mesh2) and trajectory.frag; never compiled on its own
// (compile_shaders.sh/.ps1 compile assets/*.vert|*.frag only).

// Euclidean distance |c + u| - D_ref of the point at offset u (f32, small: the local vertex
// offset mat3(model) * v) from an object centre c, without forming any large value in f32: the
// host sends (c_hat, |c| - D_ref) = centerDist and k = 1 / |c| = invCenterLen (f64 -> f32 once).
// Exact identity, no Taylor truncation (valid for |u| >= |c| too, i.e. a camera at the surface):
//   |c + u| - |c| = |c| (sqrt(1 + k n) - 1) = n / (1 + sqrt(1 + k n)),  n = 2 c_hat.u + k |u|^2.
// Every term is O(|u|), so the f32 error is a few ulp(|u|) whatever |c| is. The one exception is
// a point much nearer the camera than the centre: 1 + k n = |c+u|^2 / |c|^2 -> 0 cancels, the
// error grows as ulp(|c|) |c| / |c+u|. Below |c+u| = |c| / 2 the direct form length(c + u) (c
// rebuilt in f32, error ~ulp(|c|), as length(inWorldPos) had) is the better one and is used.
// Bound of both branches: a few ulp(|c| + |u|) absolute, and a few ulp(|u|) in the far field.
// Mirror: gpu::scene_conversion::trajectory_rte::rte_distance_offset_f32.
float rteDistanceOffset(vec4 centerDist, float invCenterLen, vec3 u) {
    if (invCenterLen == 0.0) return centerDist.w + length(u);
    float n = 2.0 * dot(centerDist.xyz, u) + dot(u, u) * invCenterLen;
    float s2 = 1.0 + invCenterLen * n; // |c + u|^2 / |c|^2
    if (s2 < 0.25) {
        float cLen = 1.0 / invCenterLen;
        return centerDist.w + (length(centerDist.xyz * cLen + u) - cLen);
    }
    return centerDist.w + n / (1.0 + sqrt(s2));
}

#endif
