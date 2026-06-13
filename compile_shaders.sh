#!/usr/bin/env bash
# compile_shaders.sh
# ──────────────────────────────────────────────────────────────────────────────
# Compiles every GLSL shader in assets/ and assets/sim/ to SPIR-V.
#
# For shaders that use LOCAL_SIZE_X (the "wg-variant" family) we produce six
# SPIR-V blobs, one per candidate workgroup / subgroup size:
#
#   <name>.comp.wg4.spv   (local_size_x = 4)
#   <name>.comp.wg8.spv   (local_size_x = 8)
#   <name>.comp.wg16.spv  (local_size_x = 16)
#   <name>.comp.wg32.spv  (local_size_x = 32)
#   <name>.comp.wg64.spv  (local_size_x = 64)
#   <name>.comp.wg128.spv (local_size_x = 128)
#
# The Rust runtime (PhysicsPipelines::new) picks the variant that matches the
# device's hardware subgroup size (VkPhysicalDeviceSubgroupProperties), so the
# shader's inner loop always occupies exactly one native SIMD batch — avoiding
# the Lavapipe ARM64 JIT register-aliasing crash (ld1r [x30] where x30 holds
# the loop counter instead of the constant-pool base).
#
# Shaders that already use a specialization constant for local_size_x
# (lbvh_collapse, morton_encode) or that are intentionally single-threaded
# (bp_clear) are compiled once as <name>.comp.spv.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

if [ -z "${VULKAN_SDK:-}" ]; then
    echo "ERROR: VULKAN_SDK environment variable is not set." >&2
    exit 1
fi

GLSLC="$VULKAN_SDK/bin/glslc"
if [ ! -x "$GLSLC" ]; then
    echo "ERROR: glslc not found or not executable at $GLSLC" >&2
    exit 1
fi

SPIRV_VAL="$VULKAN_SDK/bin/spirv-val"
if [ ! -x "$SPIRV_VAL" ]; then
    echo "ERROR: spirv-val not found or not executable at $SPIRV_VAL" >&2
    exit 1
fi

COMMON_FLAGS="-x glsl --target-env=vulkan1.1 --target-spv=spv1.4 -std=450core"
WG_SIZES=(4 8 16 32 64 128 256)

# ── Shaders that receive wg-variant SPIR-V blobs ──────────────────────────────
# Every .comp with a fixed local_size_x > 1 that uses the LOCAL_SIZE_X macro.
# (Shaders using local_size_x_id or local_size_x=1 are NOT in this list.)
WG_VARIANT_SHADERS=(
    # IMEX integrators
    integrate_bodies_p3.comp
    rb_force_assign.comp
    integrate_particles_p1_p2.comp
    integrate_particles_p4_5.comp
    # Particle systems
    accumulate_bvh_forces_to_particles.comp
    apply_emitters_to_particles.comp
    apply_emitters_direct.comp
    permute_particles.comp
    apply_impulses.comp
    emit_particles.comp
    convert_particles.comp
    # BVH builders
    lbvh_build.comp
    lbvh_build_bottomup.comp
    lbvh_prepass.comp
    lbvh_collapse.comp
    motion_bounds.comp
    motion_refit.comp
    # Broad-phase
    bp_bounds_gen.comp
    bp_classify.comp
    bp_cross_lca.comp
    bp_particle_self.comp
    bp_scene.comp
    bp_clear.comp
    # Narrow-phase / CCD
    ccd.comp
    narrow_ccd.comp
    narrow_ccd_cross_lca.comp
    reduce_toi.comp
    stream_compact.comp
    # Collision pipeline
    graph_coloring.comp
    lcp_solver.comp
    # Gravity / sorting
    barnes_hut.comp
    radix_sort.comp
)

is_wg_variant() {
    local base="$1"
    for s in "${WG_VARIANT_SHADERS[@]}"; do
        [ "$s" = "$base" ] && return 0
    done
    return 1
}

compile_one() {
    local file="$1" stage="$2" extra_flags="${3:-}" out="$4"
    echo "  glslc $extra_flags -> $(basename "$out")"
    # shellcheck disable=SC2086
    "$GLSLC" $COMMON_FLAGS -fshader-stage="$stage" $extra_flags -o "$out" "$file"
    "$SPIRV_VAL" "$out"
}

# ── Vertex / fragment shaders (always single-variant) ────────────────────────
echo "=== Compiling vertex / fragment shaders ==="
for file in assets/*.vert assets/*.frag; do
    [ -e "$file" ] || continue
    ext="${file##*.}"
    compile_one "$file" "$ext" "" "${file}.spv"
done

# ── Compute shaders ───────────────────────────────────────────────────────────
echo ""
echo "=== Compiling compute shaders ==="
for file in assets/*.comp assets/sim/*.comp; do
    [ -e "$file" ] || continue
    base="$(basename "$file")"
    echo "Shader: $file"

    if is_wg_variant "$base"; then
        # Also produce the natural .comp.spv (no -D override) so mk!() still works
        # for shaders that haven't been converted to mk_wg!() yet.
        compile_one "$file" comp "" "${file}.spv"
        compile_one "$file" comp "-DDEBUG_SHADERS" "${file%.comp}.comp.d.spv"
        # Produce one SPIR-V per candidate workgroup size for mk_wg!() shaders.
        for wg in "${WG_SIZES[@]}"; do
            out="${file%.comp}.comp.wg${wg}.spv"
            compile_one "$file" comp "-DLOCAL_SIZE_X=$wg" "$out"
            out_d="${file%.comp}.comp.wg${wg}.d.spv"
            compile_one "$file" comp "-DLOCAL_SIZE_X=$wg -DDEBUG_SHADERS" "$out_d"
        done
    else
        # Single-variant (specialization-constant or always-single-thread)
        compile_one "$file" comp "" "${file}.spv"
        compile_one "$file" comp "-DDEBUG_SHADERS" "${file%.comp}.comp.d.spv"
    fi
done

echo ""
echo "All shaders compiled successfully."