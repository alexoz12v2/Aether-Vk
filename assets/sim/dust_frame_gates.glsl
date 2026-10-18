// @assets/sim/dust_frame_gates.glsl
//
// The frame words the leaf path (dust_leaf.glsl, in dust_split.comp) and the volumetric draw
// (dust_volume_draw.glsl, in dust_draw_volume.comp) both read: the in-frame gates' switch, the
// timing probes, the gate level. Moved verbatim out of dust_volume_draw.glsl so the splitter
// needs no volumetric draw code. Include after the push-constant block `pc` and dust_scatter.glsl.
#ifndef DUST_FRAME_GATES_GLSL
#define DUST_FRAME_GATES_GLSL

// whether this frame's in-frame gates are on (bit 22 of DUST_PYR_SPLIT_DEPTH clear; dust::frame_gates_on)
bool dust_frame_gates_on() {
    return ((pc.pyramid.v[DUST_PYR_SPLIT_DEPTH] >> 22u) & 1u) == 0u;
}

// Timing probes (bits 30-31 of DUST_PYR_SPLIT_DEPTH, host `AETHERVK_DUST_TIMING_PROBE`;
// diagnostics only, never set in production, no CPU mirror): 1 = plain_add (dust_atomic_add_sat),
// 2 = no_volume_writes (the volumetric texels are evaluated but not written: one add per packet
// into its level's first texel), 3 = no_volume (volumetric packets are dropped)
uint dust_probe() { return (dust_pyr_split_word >> 30u) & 3u; }

// the level a packet past an in-frame gate is drawn at least at (dust::gate_level)
uint dust_gate_level(uint levels) {
    return levels >= 3u ? levels - 3u : 0u;
}

#endif
