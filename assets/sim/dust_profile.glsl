// @assets/sim/dust_profile.glsl
//
// The per-stage shader-clock profiler of dust_splat.comp and dust_draw_queue.comp (diagnostics:
// host `AETHERVK_DUST_PROFILE=1`, the `.prof` blobs compiled with -DDUST_PROFILE=1; the
// counters' layout is mirrored by dust::DUST_PROF_*). Without DUST_PROFILE every macro here is
// empty and the production blobs are unchanged; no CPU mirror.
//
// A stage's cycles (clock2x32ARB: the subgroup's clock, so a stall while another warp runs is
// counted — it is wall time on the SM, not issue slots) are summed per thread in a scalar global
// between DUST_PROF_BEGIN(s) and DUST_PROF_END(s), and flushed once per thread exit
// (DUST_PROF_FLUSH_*): each sum is reduced over the subgroup as two 16-bit halves and one elected
// lane adds the (lo, hi) pair into the 64-bit counter in the queue header (dust_queue_base +
// DUST_PROF_STAGE0 + 2 k), with a carry; so ~2 atomics per stage per subgroup, not per thread
// (same-address atomics serialise). The timers nest (the observer derives the exclusive times);
// a thread's whole life goes to a max and a log2 histogram as well (the tail).
// Include after dust_common.glsl and before dust_scatter.glsl (whose row loop carries probes).

// ── the counter block (words from the queue base; dust::DUST_PROF_*) ──────────────────────────
const uint DUST_PROF_STAGE0        = 8u;   // 16 x (lo, hi): the splat's stages, 8..39
const uint DUST_PROF_STAGES        = 16u;  // thread setup subpk packet cone split project scprep qput ewa vplan vdraw tracer vemit count(splitter) ecount(enumerator)
const uint DUST_PROF_MAX_THREAD    = 40u;  // the slowest thread's cycles
const uint DUST_PROF_THREADS       = 41u;  // threads below liveCount
const uint DUST_PROF_PACKETS       = 42u;  // dust_project_split_scatter_one calls
const uint DUST_PROF_CONE          = 43u;  // of those, drawn as cone / truncation slabs
const uint DUST_PROF_LEAVES        = 44u;  // EWA leaves projected on screen
const uint DUST_PROF_RECORDS       = 45u;  // queue records appended
const uint DUST_PROF_VOLUME        = 46u;  // dust_volume_scatter calls
const uint DUST_PROF_NODES         = 47u;  // split stack nodes popped
const uint DUST_PROF_EWA_TEXELS    = 48u;  // (lo, hi) texels of the in-thread EWA rows
const uint DUST_PROF_VOLUME_TEXELS = 50u;  // (lo, hi) texels of the volume segments
const uint DUST_PROF_HIST          = 52u;  // 32 log2 buckets of the threads' cycles, 52..83
const uint DUST_PROF_QSTAGE0       = 84u;  // 4 x (lo, hi): the queue draw's qload qrows qseg qthread, 84..91
const uint DUST_PROF_QSTAGES       = 4u;
const uint DUST_PROF_QRECORDS      = 92u;  // records drawn
const uint DUST_PROF_QTEXELS       = 93u;  // (lo, hi) texels drawn by the queue
const uint DUST_PROF_QMAX_WG       = 95u;  // the slowest workgroup's cycles
const uint DUST_PROF_CTHREAD       = 96u;  // (lo, hi) the volume consumer's lane cycles (dust_draw_volume.comp)
const uint DUST_PROF_CVPLAN        = 98u;  // (lo, hi) its vplan
const uint DUST_PROF_CVDRAW        = 100u; // (lo, hi) its vdraw
const uint DUST_PROF_CPACKETS      = 102u; // records it drew
const uint DUST_PROF_CTEXELS       = 103u; // (lo, hi) its texels
const uint DUST_PROF_CMAX          = 105u; // its slowest lane's cycles
const uint DUST_PROF_STHREAD       = 106u; // (lo, hi) the splitter's lane cycles (dust_split.comp)
const uint DUST_PROF_SLANES        = 108u; // its lanes with a record
const uint DUST_PROF_SMAX          = 109u; // its slowest lane's cycles
const uint DUST_PROF_K1C0          = 110u; // 12 x (lo, hi), K1c's parts of vplan / vdraw (every volumetric draw, in-thread and consumer): vclass (pass A's per-class sums), vsmallpts (a small piece's depth points), vsmall (its texels' rectangle integrals), vsubline (sub-ray and line texels), vcentre (the centre class's rows), 110..119; vsmallpts' parts: vsmprep (its prelude: the depth axis, the lateral axes' dots, the mass), vsmfast (the centre slice by one projection and the edge decision, every piece; the other slices of an interior piece), vsmedge (the slices of an edge piece by 4 sigma points each), 120..125; the consumer's pass B parts (round 8 step E): vhead (the piece loop's head: the control word, the record's words, the covariance slots), vbuild (dust_volume_piece_at), vdecide (the level table, the range, the class and order under K), vdeposit (dust_volume_deposit: the counts' atomics and the depth), 126..133
const uint DUST_PROF_SMALL_PATHS   = 134u; // 2 words (add32): the small pieces drawn by one projection a slice (126) and by 4 sigma points a slice (127) (dust::DUST_PROF_SMALL_PATHS)
const uint DUST_PROF_WORDS_END     = 144u; // (DUST_QUEUE_HEADER_WORDS = 144)

#ifdef DUST_PROFILE
#define DUST_PROF_BEGIN_AS(t)   uint dust_prof_t0_##t = clock2x32ARB().x;
#define DUST_PROF_END_AS(s, t)  if (!dust_counting) dust_prof_##s += clock2x32ARB().x - dust_prof_t0_##t;
#define DUST_PROF_BEGIN(s)      DUST_PROF_BEGIN_AS(s)
#define DUST_PROF_END(s)        DUST_PROF_END_AS(s, s)
#define DUST_PROF_COUNT(c, n)   if (!dust_counting) dust_prof_n_##c += uint(n);
#define DUST_PROF_FLUSH_SPLAT(pyr, base, ran) dust_prof_flush_splat(pyr, base, ran);
#define DUST_PROF_FLUSH_QUEUE(pyr, base)      dust_prof_flush_queue(pyr, base);
#define DUST_PROF_FLUSH_VOLUME(pyr, base, ran) dust_prof_flush_volume(pyr, base, ran);
#define DUST_PROF_FLUSH_SPLIT(pyr, base, ran)  dust_prof_flush_split(pyr, base, ran);
#define DUST_PROF_FLUSH_EDGE(pyr, base)        dust_prof_flush_edge(pyr, base);

// the splat's stage sums (cycles) and counts, per thread
#ifdef DUST_PROF_SHARED
shared uint dust_prof_thread_s[DUST_PROF_SHARED];
#define dust_prof_thread dust_prof_thread_s[gl_LocalInvocationID.x]
shared uint dust_prof_setup_s[DUST_PROF_SHARED];
#define dust_prof_setup dust_prof_setup_s[gl_LocalInvocationID.x]
shared uint dust_prof_subpk_s[DUST_PROF_SHARED];
#define dust_prof_subpk dust_prof_subpk_s[gl_LocalInvocationID.x]
shared uint dust_prof_packet_s[DUST_PROF_SHARED];
#define dust_prof_packet dust_prof_packet_s[gl_LocalInvocationID.x]
shared uint dust_prof_cone_s[DUST_PROF_SHARED];
#define dust_prof_cone dust_prof_cone_s[gl_LocalInvocationID.x]
shared uint dust_prof_split_s[DUST_PROF_SHARED];
#define dust_prof_split dust_prof_split_s[gl_LocalInvocationID.x]
shared uint dust_prof_project_s[DUST_PROF_SHARED];
#define dust_prof_project dust_prof_project_s[gl_LocalInvocationID.x]
shared uint dust_prof_scprep_s[DUST_PROF_SHARED];
#define dust_prof_scprep dust_prof_scprep_s[gl_LocalInvocationID.x]
shared uint dust_prof_qput_s[DUST_PROF_SHARED];
#define dust_prof_qput dust_prof_qput_s[gl_LocalInvocationID.x]
shared uint dust_prof_ewa_s[DUST_PROF_SHARED];
#define dust_prof_ewa dust_prof_ewa_s[gl_LocalInvocationID.x]
shared uint dust_prof_vplan_s[DUST_PROF_SHARED];
#define dust_prof_vplan dust_prof_vplan_s[gl_LocalInvocationID.x]
shared uint dust_prof_vdraw_s[DUST_PROF_SHARED];
#define dust_prof_vdraw dust_prof_vdraw_s[gl_LocalInvocationID.x]
shared uint dust_prof_tracer_s[DUST_PROF_SHARED];
#define dust_prof_tracer dust_prof_tracer_s[gl_LocalInvocationID.x]
shared uint dust_prof_vemit_s[DUST_PROF_SHARED];
#define dust_prof_vemit dust_prof_vemit_s[gl_LocalInvocationID.x]
shared uint dust_prof_count_s[DUST_PROF_SHARED];
#define dust_prof_count dust_prof_count_s[gl_LocalInvocationID.x]
shared uint dust_prof_t0_thread_s[DUST_PROF_SHARED], dust_prof_t0_subpk_s[DUST_PROF_SHARED];
#define dust_prof_t0_thread dust_prof_t0_thread_s[gl_LocalInvocationID.x]
#define dust_prof_t0_subpk dust_prof_t0_subpk_s[gl_LocalInvocationID.x]
#define DUST_PROF_BEGIN_SH(t) dust_prof_t0_##t = clock2x32ARB().x;
void dust_prof_zero() { dust_prof_thread = 0u; dust_prof_setup = 0u; dust_prof_subpk = 0u; dust_prof_packet = 0u; dust_prof_cone = 0u; dust_prof_split = 0u; dust_prof_project = 0u; dust_prof_scprep = 0u; dust_prof_qput = 0u; dust_prof_ewa = 0u; dust_prof_vplan = 0u; dust_prof_vdraw = 0u; dust_prof_tracer = 0u; dust_prof_vemit = 0u; dust_prof_count = 0u; }
#else
uint dust_prof_thread = 0u, dust_prof_setup = 0u, dust_prof_subpk = 0u, dust_prof_packet = 0u,
     dust_prof_cone = 0u, dust_prof_split = 0u, dust_prof_project = 0u, dust_prof_scprep = 0u,
     dust_prof_qput = 0u, dust_prof_ewa = 0u, dust_prof_vplan = 0u, dust_prof_vdraw = 0u,
     dust_prof_tracer = 0u, dust_prof_vemit = 0u, dust_prof_count = 0u;
void dust_prof_zero() {}
#define DUST_PROF_BEGIN_SH(t) DUST_PROF_BEGIN(t)
#endif
// K1c's parts (nested in vplan / vdraw)
uint dust_prof_vclass = 0u, dust_prof_vsmallpts = 0u, dust_prof_vsmall = 0u, dust_prof_vsubline = 0u,
     dust_prof_vcentre = 0u, dust_prof_vsmprep = 0u, dust_prof_vsmfast = 0u, dust_prof_vsmedge = 0u,
     dust_prof_vhead = 0u, dust_prof_vbuild = 0u, dust_prof_vdecide = 0u, dust_prof_vdeposit = 0u;
uint dust_prof_n_small_fast = 0u, dust_prof_n_small_edge = 0u;
void dust_prof_flush_k1c(DustUintBuffer pyr, uint base);
uint dust_prof_n_packets = 0u, dust_prof_n_cone = 0u, dust_prof_n_leaves = 0u,
     dust_prof_n_records = 0u, dust_prof_n_volume = 0u, dust_prof_n_nodes = 0u,
     dust_prof_n_ewa_texels = 0u, dust_prof_n_volume_texels = 0u;
// the queue draw's
uint dust_prof_qload = 0u, dust_prof_qrows = 0u, dust_prof_qseg = 0u, dust_prof_qthread = 0u;
uint dust_prof_n_qtexels = 0u;

// v summed over the subgroup into the 64-bit counter at (w, w + 1): the sum of the 16-bit halves
// (each < 2^21 for 32 lanes) recombined with carries
void dust_prof_add64(DustUintBuffer pyr, uint w, uint v) {
    uint hi = subgroupAdd(v >> 16u), lo = subgroupAdd(v & 0xFFFFu);
    if (subgroupElect()) {
        uint sumLo = (hi << 16u) + lo;
        uint sumHi = (hi >> 16u) + (((hi << 16u) > 0xFFFFFFFFu - lo) ? 1u : 0u);
        uint old = atomicAdd(pyr.v[w], sumLo);
        sumHi += (old > 0xFFFFFFFFu - sumLo) ? 1u : 0u;
        if (sumHi > 0u) atomicAdd(pyr.v[w + 1u], sumHi);
    }
}
void dust_prof_add32(DustUintBuffer pyr, uint w, uint v) {
    uint s = subgroupAdd(v);
    if (subgroupElect() && s > 0u) atomicAdd(pyr.v[w], s);
}
void dust_prof_max32(DustUintBuffer pyr, uint w, uint v) {
    uint m = subgroupMax(v);
    if (subgroupElect() && m > 0u) atomicMax(pyr.v[w], m);
}
// every lane of the subgroup must reach these (no early return before them)
void dust_prof_flush_splat(DustUintBuffer pyr, uint base, bool ran) {
    uint s0 = base + DUST_PROF_STAGE0;
    dust_prof_add64(pyr, s0 + 0u,  dust_prof_thread);
    dust_prof_add64(pyr, s0 + 2u,  dust_prof_setup);
    dust_prof_add64(pyr, s0 + 4u,  dust_prof_subpk);
    dust_prof_add64(pyr, s0 + 6u,  dust_prof_packet);
    dust_prof_add64(pyr, s0 + 8u,  dust_prof_cone);
    dust_prof_add64(pyr, s0 + 10u, dust_prof_split);
    dust_prof_add64(pyr, s0 + 12u, dust_prof_project);
    dust_prof_add64(pyr, s0 + 14u, dust_prof_scprep);
    dust_prof_add64(pyr, s0 + 16u, dust_prof_qput);
    dust_prof_add64(pyr, s0 + 18u, dust_prof_ewa);
    dust_prof_add64(pyr, s0 + 20u, dust_prof_vplan);
    dust_prof_add64(pyr, s0 + 22u, dust_prof_vdraw);
    dust_prof_add64(pyr, s0 + 24u, dust_prof_tracer);
    dust_prof_add64(pyr, s0 + 26u, dust_prof_vemit);
    dust_prof_add64(pyr, s0 + 30u, dust_prof_count);
    dust_prof_max32(pyr, base + DUST_PROF_MAX_THREAD, ran ? dust_prof_thread : 0u);
    dust_prof_add32(pyr, base + DUST_PROF_THREADS, ran ? 1u : 0u);
    dust_prof_add32(pyr, base + DUST_PROF_PACKETS, dust_prof_n_packets);
    dust_prof_add32(pyr, base + DUST_PROF_CONE, dust_prof_n_cone);
    dust_prof_add32(pyr, base + DUST_PROF_LEAVES, dust_prof_n_leaves);
    dust_prof_add32(pyr, base + DUST_PROF_RECORDS, dust_prof_n_records);
    dust_prof_add32(pyr, base + DUST_PROF_VOLUME, dust_prof_n_volume);
    dust_prof_add32(pyr, base + DUST_PROF_NODES, dust_prof_n_nodes);
    dust_prof_add64(pyr, base + DUST_PROF_EWA_TEXELS, dust_prof_n_ewa_texels);
    dust_prof_add64(pyr, base + DUST_PROF_VOLUME_TEXELS, dust_prof_n_volume_texels);
    dust_prof_flush_k1c(pyr, base);
    if (ran) {
        uint bucket = uint(clamp(findMSB(max(dust_prof_thread, 1u)), 0, 31));
        atomicAdd(pyr.v[base + DUST_PROF_HIST + bucket], 1u);
    }
}
// K1c's parts of the volumetric draw (DUST_PROF_K1C0: vclass vsmallpts vsmall vsubline vcentre
// vsmprep vsmfast vsmedge, in dust::DUST_PROF_K1C_NAMES' order) and the small pieces by path
void dust_prof_flush_k1c(DustUintBuffer pyr, uint base) {
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 0u, dust_prof_vclass);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 2u, dust_prof_vsmallpts);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 4u, dust_prof_vsmall);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 6u, dust_prof_vsubline);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 8u, dust_prof_vcentre);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 10u, dust_prof_vsmprep);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 12u, dust_prof_vsmfast);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 14u, dust_prof_vsmedge);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 16u, dust_prof_vhead);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 18u, dust_prof_vbuild);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 20u, dust_prof_vdecide);
    dust_prof_add64(pyr, base + DUST_PROF_K1C0 + 22u, dust_prof_vdeposit);
    dust_prof_add32(pyr, base + DUST_PROF_SMALL_PATHS + 0u, dust_prof_n_small_fast);
    dust_prof_add32(pyr, base + DUST_PROF_SMALL_PATHS + 1u, dust_prof_n_small_edge);
}
void dust_prof_flush_queue(DustUintBuffer pyr, uint base) {
    uint s0 = base + DUST_PROF_QSTAGE0;
    dust_prof_add64(pyr, s0 + 0u, dust_prof_qload);
    dust_prof_add64(pyr, s0 + 2u, dust_prof_qrows);
    dust_prof_add64(pyr, s0 + 4u, dust_prof_qseg);
    dust_prof_add64(pyr, s0 + 6u, dust_prof_qthread);
    dust_prof_add32(pyr, base + DUST_PROF_QRECORDS, gl_LocalInvocationID.x == 0u ? 1u : 0u);
    dust_prof_add64(pyr, base + DUST_PROF_QTEXELS, dust_prof_n_qtexels);
    dust_prof_max32(pyr, base + DUST_PROF_QMAX_WG, dust_prof_qthread);
}
// the splitter (dust_split.comp): the leaf stages (packet cone split project scprep qput ewa
// vplan vdraw vemit count) into the stage words, its own lane cycles apart; every lane of the
// subgroup must reach this
void dust_prof_flush_split(DustUintBuffer pyr, uint base, bool ran) {
    uint s0 = base + DUST_PROF_STAGE0;
    dust_prof_add64(pyr, s0 + 6u,  dust_prof_packet);
    dust_prof_add64(pyr, s0 + 8u,  dust_prof_cone);
    dust_prof_add64(pyr, s0 + 10u, dust_prof_split);
    dust_prof_add64(pyr, s0 + 12u, dust_prof_project);
    dust_prof_add64(pyr, s0 + 14u, dust_prof_scprep);
    dust_prof_add64(pyr, s0 + 16u, dust_prof_qput);
    dust_prof_add64(pyr, s0 + 18u, dust_prof_ewa);
    dust_prof_add64(pyr, s0 + 20u, dust_prof_vplan);
    dust_prof_add64(pyr, s0 + 22u, dust_prof_vdraw);
    dust_prof_add64(pyr, s0 + 26u, dust_prof_vemit);
    dust_prof_add64(pyr, s0 + 28u, dust_prof_count);
    dust_prof_add32(pyr, base + DUST_PROF_PACKETS, dust_prof_n_packets);
    dust_prof_add32(pyr, base + DUST_PROF_CONE, dust_prof_n_cone);
    dust_prof_add32(pyr, base + DUST_PROF_LEAVES, dust_prof_n_leaves);
    dust_prof_add32(pyr, base + DUST_PROF_RECORDS, dust_prof_n_records);
    dust_prof_add32(pyr, base + DUST_PROF_VOLUME, dust_prof_n_volume);
    dust_prof_add32(pyr, base + DUST_PROF_NODES, dust_prof_n_nodes);
    dust_prof_add64(pyr, base + DUST_PROF_EWA_TEXELS, dust_prof_n_ewa_texels);
    dust_prof_add64(pyr, base + DUST_PROF_VOLUME_TEXELS, dust_prof_n_volume_texels);
    dust_prof_flush_k1c(pyr, base);
    dust_prof_add64(pyr, base + DUST_PROF_STHREAD, dust_prof_thread);
    dust_prof_add32(pyr, base + DUST_PROF_SLANES, ran ? 1u : 0u);
    dust_prof_max32(pyr, base + DUST_PROF_SMAX, ran ? dust_prof_thread : 0u);
}
// the volume consumer (dust_draw_volume.comp): its lanes' vplan / vdraw and texels go to their
// own words; every lane of the subgroup must reach this
void dust_prof_flush_volume(DustUintBuffer pyr, uint base, bool ran) {
    dust_prof_add64(pyr, base + DUST_PROF_CTHREAD, dust_prof_thread);
    dust_prof_add64(pyr, base + DUST_PROF_CVPLAN, dust_prof_vplan);
    dust_prof_add64(pyr, base + DUST_PROF_CVDRAW, dust_prof_vdraw);
    dust_prof_add32(pyr, base + DUST_PROF_CPACKETS, ran ? 1u : 0u);
    dust_prof_add64(pyr, base + DUST_PROF_CTEXELS, dust_prof_n_volume_texels);
    dust_prof_flush_k1c(pyr, base);
    dust_prof_max32(pyr, base + DUST_PROF_CMAX, ran ? dust_prof_thread : 0u);
}
// track 9A (t9a2d): the edge pass (dust_draw_small_edge.comp): its pieces' K1c parts and paths and
// their texels with the consumer's (its lanes' own cycles in the K1c words only); every lane of the
// subgroup must reach this
void dust_prof_flush_edge(DustUintBuffer pyr, uint base) {
    dust_prof_add64(pyr, base + DUST_PROF_CTEXELS, dust_prof_n_volume_texels);
    dust_prof_flush_k1c(pyr, base);
}
#else
#define DUST_PROF_BEGIN_AS(t)
#define DUST_PROF_END_AS(s, t)
#define DUST_PROF_BEGIN(s)
#define DUST_PROF_BEGIN_SH(t)
#define DUST_PROF_END(s)
#define DUST_PROF_COUNT(c, n)
#define DUST_PROF_FLUSH_SPLAT(pyr, base, ran)
#define DUST_PROF_FLUSH_QUEUE(pyr, base)
#define DUST_PROF_FLUSH_VOLUME(pyr, base, ran)
#define DUST_PROF_FLUSH_SPLIT(pyr, base, ran)
#define DUST_PROF_FLUSH_EDGE(pyr, base)
#endif
