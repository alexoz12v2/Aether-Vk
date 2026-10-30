//! Vulkan Backend Integration for the IMEX / LCP Physics Engine
//!
//! This module scaffolds the execution of the massive compute-shader pipeline.
//! It assumes Vulkan 1.1 with `VK_KHR_buffer_device_address` and `VK_KHR_shader_subgroup_basic`.

use crate::{
  gpu_backends::vulkan::device::{LogicalDevice, VulkanDebugNameExt, resources},
  types::{GpuError, GpuResult},
};
use ash::vk;

// Disabled by default: Enabling PRINTF shaders under Lavapipe (ARM64) dramatically increases
// register pressure in the llvmpipe JIT compiler. This leads to register spilling bugs that
// overwrite the stack-saved link register (x30), causing a SIGSEGV upon kernel return.
/// The dust profiler request (`AETHERVK_DUST_PROFILE=1`, or this override set before the device
/// is created: tests): the device enables `VK_KHR_shader_clock` and
/// `VK_KHR_pipeline_executable_properties` when it has them, the `.prof` variants of the dust
/// splat and queue shaders are built (`dust_profile.glsl`) and used while the request is on
/// (`dust_profile_requested`, read per dispatch), and the dust pipelines' compiled statistics
/// are logged once (`log_pipeline_statistics`).
pub static DUST_PROFILE_OVERRIDE: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);
pub fn dust_profile_requested() -> bool {
  DUST_PROFILE_OVERRIDE.load(core::sync::atomic::Ordering::Relaxed)
    || aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PROFILE").is_some_and(|v| v.trim() == "1")
}

#[cfg(all(
  any(debug_assertions, test, feature = "shader_debug_sync"),
  not(target_vendor = "apple")
))]
pub static USE_PRINTF_SHADERS: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub static READBACK_DIAGNOSTICS: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

/// When true, dust emission and evaluation run on the CPU with the reference implementation
/// ([`crate::scene::dust`]) instead of `dust_emit.comp` / `dust_propagate.comp`; only drawing
/// stays on the GPU. Activated by setting `AETHERVK_PARTICLES_CPU=1` at startup.
pub static USE_CPU_PARTICLES: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

/// Activates the CPU particle simulation path. Call once at startup.
pub fn enable_cpu_particles() {
  USE_CPU_PARTICLES.store(true, core::sync::atomic::Ordering::Relaxed);
  aethervk_oshal_rlib::log!("[Physics] CPU particle simulation mode ENABLED.");
}

/// Returns `true` when `AETHERVK_PARTICLES_CPU=1` was set at startup.
#[inline]
pub fn is_cpu_particles_mode() -> bool {
  USE_CPU_PARTICLES.load(core::sync::atomic::Ordering::Relaxed)
}

/// Configuration parameters for the physics pipeline
pub struct PhysicsPipelineConfig {
  pub max_particles: u32,
  pub hardware_subgroup_size: u32,
}

/// The devices whose shader compiler rejects the dust monoliths (Qualcomm's
/// Adreno 619 driver: the propagate / emit solves and the speck walk as one program; AMD's
/// amdvlk (0x1002, 31.0.21924.61 on a Radeon iGPU): `vkCreateComputePipelines` never returns on
/// `dust_splat_mono.comp` and `dust_draw_volume_mono.comp`) load the staged blob sets instead
/// (`dust_propagate.comp.pass*`, `dust_emit.comp.pass*`) and build none of the `*_mono.comp`
/// pipelines; the others keep the monoliths. `AETHERVK_DUST_STAGED=1` / `0` overrides the vendor
/// rule (the 1070's byte gate runs both).
pub fn dust_staged_passes(vendor_id: u32) -> bool {
  match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_STAGED").as_deref().map(str::trim) {
    Some("1") => true,
    Some("0") => false,
    _ => matches!(vendor_id, 0x5143 | 0x1002),
  }
}

/// The staged volume consumer (`dust_draw_volume_stages`): built on the Qualcomm
/// devices (vendor 0x5143, the devices it was gated on) or with `AETHERVK_DUST_VOLUME_STAGES=1`;
/// `AETHERVK_DUST_VOLUME_STAGES=0` turns it off everywhere
pub fn dust_volume_stages_wanted(vendor_id: u32) -> bool {
  match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_VOLUME_STAGES").as_deref().map(str::trim) {
    Some("1") => true,
    Some("0") => false,
    _ => vendor_id == 0x5143,
  }
}

/// The splat's path. The stage chain (`dust_splat.comp.stN`, the zero-spill
/// design of the phones) on Qualcomm (0x5143) and ARM / Mali (0x13B5); the monolithic enumerator
/// (`dust_splat_mono.comp`, the monolithic blob: 233 vs 300 ms on the GTX 1070, its byte
/// reference) everywhere else. `AETHERVK_DUST_SPLAT_STAGED=1` / `0` overrides the vendor rule;
/// `Device::dust_set_splat_staged` overrides both (the parity tests)
pub fn dust_splat_staged_default(vendor_id: u32) -> bool {
  match aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPLAT_STAGED").as_deref().map(str::trim) {
    Some("1") => true,
    Some("0") => false,
    _ => matches!(vendor_id, 0x5143 | 0x13B5),
  }
}

/// Whether the monolithic enumerator's pipeline is built: not on the devices whose
/// compiler rejects the dust monoliths (`staged_dust`, [`dust_staged_passes`]) unless
/// `AETHERVK_DUST_SPLAT_STAGED=0` asks for it; without it the splat runs the chain
pub fn dust_splat_mono_wanted(staged_dust: bool) -> bool {
  !staged_dust
    || aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_SPLAT_STAGED").as_deref().map(str::trim) == Some("0")
}

/// Struct holding all [`ash::vk::Pipeline`] handles to pipelines to simulate particles on Vulkan
/// device. Note that all of them share the same [`ash::vk::PipelineLayout`] because it declares no
/// descriptor sets and permits usage of the max allowed by minimum requirements, 128 Bytes, for
/// push constant
pub struct PhysicsPipelines {
  /// Pipeline Layout shared by all pipelines
  pub pipeline_layout: vk::PipelineLayout,

  // ── Dust v3 (see `crate::scene::dust`) ────────────────────────────────────
  /// writes one batch of immutable cluster records into a ring (`dust_emit.comp`, wg 64)
  pub dust_emit: vk::Pipeline,
  /// per-frame closed-form Kepler evaluation of the live ring range (`dust_propagate.comp`, wg 64)
  pub dust_propagate: vk::Pipeline,
  /// per-frame analytic splat of one tier into the view's pyramid (`dust_splat.comp`, wg 64):
  /// a chain of stage pipelines (`dust_splat.comp.stN.spv`, stage ids and lanes
  /// in `scene::dust::DUST_SPLAT_STAGE_*`); `dust_splat` aliases stage 0 (the push-constant size
  /// assert, the statistics)
  pub dust_splat: vk::Pipeline,
  pub dust_splat_stages: [vk::Pipeline; crate::scene::dust::DUST_SPLAT_STAGES],
  pub dust_splat_prof_stages: Option<[vk::Pipeline; crate::scene::dust::DUST_SPLAT_STAGES]>,
  /// the monolithic enumerator (`dust_splat_mono.comp`, wg 64, `DUST_SPLAT_LANES` lanes
  /// per cluster) and its profiler variant; `None` where it is not built ([`dust_splat_mono_wanted`])
  pub dust_splat_mono: Option<vk::Pipeline>,
  pub dust_splat_mono_prof: Option<vk::Pipeline>,
  /// the splat path's override (0 the rule [`dust_splat_staged_default`], 1 the chain,
  /// 2 the monolith: `Device::dust_set_splat_staged`) and the path last logged (0 none, 1, 2)
  pub dust_splat_mode: core::sync::atomic::AtomicU8,
  pub dust_splat_logged: core::sync::atomic::AtomicU8,
  /// per-frame white-point measurement grid from the finished pyramid (`dust_measure.comp`, wg 64)
  pub dust_measure: vk::Pipeline,
  /// per-frame packet queue: a workgroup per queued large packet (`dust_draw_queue.comp`, wg 64,
  /// dispatched indirectly)
  pub dust_draw_queue: vk::Pipeline,
  /// per-frame physical specks of one tier into the view's pyramid (`dust_specks.comp`, wg 64,
  /// one thread per cluster × size bin: the gate pass, the density at the eye and the walk mask)
  pub dust_specks: vk::Pipeline,
  /// the walk's step pass (`dust_specks.comp.walk.spv`, `-DDUST_SPECKS_WALK=1`:
  /// breadth first): 8 lanes per frontier node, once per tree level, dispatched indirectly from
  /// the walk region's per-step args (push word 24 = the step)
  pub dust_specks_walk: vk::Pipeline,
  /// the walk's prologue: setup1 (a thread per marked (cluster, bin): the record), setup2a / 2b /
  /// 3 / 4 / 5 (a thread per record, indirect: the map, the class, the start nodes, frontier 0)
  pub dust_specks_setup: [vk::Pipeline; 6],
  /// the many-speck leaves (a workgroup per leaf), the single-speck leaves (a lane per leaf),
  /// the trace fill (the test hook)
  pub dust_specks_leaves: vk::Pipeline,
  pub dust_specks_leaves1: vk::Pipeline,
  pub dust_specks_trace: vk::Pipeline,
  /// the propagate's moments stage (`dust_propagate.comp.moments.spv`): after the solve stage,
  /// the same workgroups over the same chunk
  pub dust_propagate_moments: vk::Pipeline,
  /// the staged solves ([`dust_staged_passes`]): propagate passes 1, 2, 3, 6, 4, 5m, 5s (recorded
  /// 1, (2, 3) × `DUST_KEPLER_ITERS_DF`, 6, 4, 5m, 5s) and emit passes 1, 2, 3, 6, 4, 5r, 5v, 71,
  /// 72 (1, (2, 3) × iters, 6, 4, 5r, 5v, 71, 72); `None` = the monoliths `dust_propagate` /
  /// `dust_emit` (which then alias pass 1 for the size asserts)
  pub dust_propagate_passes: Option<[vk::Pipeline; 7]>,
  pub dust_emit_passes: Option<[vk::Pipeline; 9]>,
  /// the monolithic propagate and emission (`dust_propagate_mono.comp`,
  /// `dust_emit_mono.comp`: one thread per cluster, the df64 solve in registers), dispatched when
  /// `Device::dust_prop_staged` is false (the GTX 1070 default); `None` where the monoliths are not
  /// loaded ([`dust_staged_passes`])
  pub dust_propagate_mono: Option<vk::Pipeline>,
  pub dust_emit_mono: Option<vk::Pipeline>,
  /// the propagate / emit family's mode (`Device::dust_prop_staged` /
  /// `dust_set_prop_staged`): 0 not resolved yet, 1 mono / 2 package (the test hook's), 3 mono /
  /// 4 package (the vendor rule's)
  pub dust_prop_mode: core::sync::atomic::AtomicU8,
  /// the shader-clock profiler variants of the splat and the queue draw (`dust_profile.glsl`;
  /// `None` without `VK_KHR_shader_clock` or the request, [`dust_profile_requested`])
  pub dust_splat_prof: Option<vk::Pipeline>,
  pub dust_draw_queue_prof: Option<vk::Pipeline>,
  /// per-chunk volume packet queue consumer: a lane per queued volumetric packet
  /// (`dust_draw_volume.comp`, wg 64, dispatched indirectly after each producer chunk)
  pub dust_draw_volume: vk::Pipeline,
  pub dust_draw_volume_prof: Option<vk::Pipeline>,
  /// the plain blob while the `.sg` one is in use (the baseline devices' path,
  /// `Device::dust_set_volume_plain` runs it on a device that has both: the test)
  pub dust_draw_volume_plain: Option<vk::Pipeline>,
  /// the staged consumer's seven stages in dispatch order s1, s2, s3a, s3b, s4a, s4b,
  /// s5 (`dust_draw_volume.comp.s<k>[.sg][.vmm].spv`; each stage a main drain then a spill drain,
  /// `Device::cmd_dust_splat`); `None` unless [`dust_volume_stages_wanted`] (and never with a profiler)
  pub dust_draw_volume_stages: Option<[vk::Pipeline; 7]>,
  /// the edge pass: a lane per small piece the consumer deferred (its centre
  /// slice's 3σ crosses the frame; `dust_draw_small_edge.comp`, wg 64, dispatched indirectly beside
  /// the next chunk's producer): the demotion or the sigma points, then its texels
  pub dust_draw_small_edge: vk::Pipeline,
  pub dust_draw_small_edge_prof: Option<vk::Pipeline>,
  /// the mono draw family, the monolithic blobs (`dust_draw_volume_mono.comp`: `.sg` as
  /// `dust_draw_volume`, `_plain` the plain blob beside it, `.prof`; `dust_draw_small_edge_mono.comp`
  /// and `dust_draw_queue_mono.comp` with their `.prof`), dispatched when `Device::dust_draw_staged`
  /// is false (the GTX 1070 default); `None` where the monoliths are not built
  /// ([`dust_staged_passes`])
  pub dust_draw_volume_mono: Option<vk::Pipeline>,
  pub dust_draw_volume_mono_prof: Option<vk::Pipeline>,
  pub dust_draw_volume_mono_plain: Option<vk::Pipeline>,
  pub dust_draw_small_edge_mono: Option<vk::Pipeline>,
  pub dust_draw_small_edge_mono_prof: Option<vk::Pipeline>,
  pub dust_draw_queue_mono: Option<vk::Pipeline>,
  pub dust_draw_queue_mono_prof: Option<vk::Pipeline>,
  /// the draw family's mode (`Device::dust_draw_staged` / `dust_set_draw_staged`): 0 not
  /// resolved yet, 1 mono / 2 staged (the test hook's), 3 mono / 4 staged (the vendor rule's)
  pub dust_draw_mode: core::sync::atomic::AtomicU8,
  /// per-chunk sub-packet queue splitter: a workgroup per cluster over its records
  /// (`dust_split.comp`, wg 64; the leaf path of the former per-cluster splat thread)
  pub dust_split: vk::Pipeline,
  pub dust_split_prof: Option<vk::Pipeline>,
  /// the pre-split splitter (`dust_split_mono.comp`: the old dust_split.comp and
  /// dust_leaf.glsl, one dispatch per chunk, a workgroup per cluster) and its profiler variant,
  /// dispatched when `Device::dust_split_staged` is false (the GTX 1070 default); `None` where
  /// the monoliths are not built ([`dust_staged_passes`])
  pub dust_split_mono: Option<vk::Pipeline>,
  pub dust_split_mono_prof: Option<vk::Pipeline>,
  /// the split family's mode (`Device::dust_split_staged` / `dust_set_split_staged`): 0
  /// not resolved yet, 1 mono / 2 staged (the test hook's), 3 mono / 4 staged (the vendor rule's)
  pub dust_split_mode: core::sync::atomic::AtomicU8,
  /// the splitter family (`dust_split.comp` stage blobs, the same push
  /// constants): the prep stage (`.prep`: a lane per record, the cone words), the hard-axis
  /// diagnostic walk (`.hard`, in place of the walk under `AETHERVK_DUST_HARD_AXIS`), the SEQ
  /// order stage (`.seq`: a workgroup per cluster, the candidates' ordinals) and the SEQ draw stage
  /// (`.seqd`: a lane per candidate, indirect from the candidate header)
  pub dust_split_prep: vk::Pipeline,
  pub dust_split_prep_prof: Option<vk::Pipeline>,
  pub dust_split_hard: vk::Pipeline,
  pub dust_split_hard_prof: Option<vk::Pipeline>,
  pub dust_split_seq: vk::Pipeline,
  pub dust_split_seq_prof: Option<vk::Pipeline>,
  pub dust_split_seq_draw: vk::Pipeline,
  pub dust_split_seq_draw_prof: Option<vk::Pipeline>,
  /// the project stage in three passes (`dust_project_leaf.comp.proj1/2/3`: a lane per candidate,
  /// indirect from the candidate header; phase A, phase B to kx/ky, the rest of B and F)
  pub dust_project_leaf_p1: vk::Pipeline,
  pub dust_project_leaf_p2: vk::Pipeline,
  pub dust_project_leaf_p3: vk::Pipeline,
  pub dust_project_leaf_p1_prof: Option<vk::Pipeline>,
  pub dust_project_leaf_p2_prof: Option<vk::Pipeline>,
  pub dust_project_leaf_p3_prof: Option<vk::Pipeline>,
  /// the leaf consumers (`dust_draw_leaf.comp`): a lane per leaf record (the leaf queue, then the
  /// leaf spill), a workgroup per large leaf (`.wg`, the list)
  pub dust_draw_leaf: vk::Pipeline,
  pub dust_draw_leaf_prof: Option<vk::Pipeline>,
  pub dust_draw_leaf_wg: vk::Pipeline,
  pub dust_draw_leaf_wg_prof: Option<vk::Pipeline>,
  /// the cadence's per-slot passes (`dust_classify.comp`: classify, transitions, marks, chunks)
  pub dust_classify: vk::Pipeline,
  /// the cadence's list compaction and control pass (`dust_compact.comp`)
  pub dust_compact: vk::Pipeline,
  /// the reprojection of the shown class images into the frame (`dust_reproject.comp`)
  pub dust_reproject: vk::Pipeline,

  /// SPIR-V-reflected push constant block size per pipeline.
  /// Used by `debug_assert!` in dispatch helpers to catch size mismatches
  /// before they become cryptic Metal validation errors.
  pub pc_sizes: hashbrown::HashMap<u64, u32>,
  pub wg_sizes: hashbrown::HashMap<u64, [u32; 3]>,
  /// Hardware subgroup size (SIMD width), used for AOSOA packing.
  pub subgroup_size: u32,
  /// True when running on a CPU Vulkan device (Lavapipe / llvmpipe).
  /// Enables CPU-optimised SPIR-V variants and reduced workgroup sizes.
  pub is_lavapipe: bool,
  /// True when the device supports shaderFloat16 (VK_KHR_shader_float16_int8 feature bit).
  /// False on Pascal (GTX 10xx) and older NVIDIA GPUs — selects .nofp16 SPIR-V variants.
  pub has_native_float16: bool,
}

impl PhysicsPipelines {
  /// TODO: Document this item
  pub fn new(
    device: &LogicalDevice,
    debug_shaders: bool,
    subgroup_size: u32,
    is_cpu: bool,
    has_native_float16: bool,
    has_vulkan_memory_model: bool,
    has_shader_clock: bool,
    capture_statistics: bool,
    has_subgroup_arithmetic: bool,
    staged_dust: bool,
    volume_stages: bool,
  ) -> GpuResult<Self> {
    if has_native_float16 {
      aethervk_oshal_rlib::log!(
        "[Physics] Native float16 arithmetic is SUPPORTED (VK_KHR_shader_float16_int8:shaderFloat16)."
      );
    } else {
      aethervk_oshal_rlib::log!(
        "[Physics] Native float16 arithmetic is MISSING. Using fallback FP32 shaders with 16-bit storage."
      );
    }

    if subgroup_size <= 16 && is_cpu {
      // For Lavapipe, if subgroup_size <= 16, workgroup size will be reduced to subgroup_size.
      // So subgroup_size remains the hardware subgroup_size.
    }

    let push_constant_range = vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::COMPUTE)
      .offset(0)
      .size(128); // Max push constant size

    let layout_info = vk::PipelineLayoutCreateInfo::default()
      .push_constant_ranges(core::slice::from_ref(&push_constant_range));

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }
      .with_name(device, "VkPipelineLayout_Physics")
      .map_err(|e| {
        GpuError::BackendSpecific(alloc::format!("Failed to create pipeline layout: {:?}", e))
      })?;

    let mut created_pipelines = alloc::vec::Vec::new();
    // the executable statistics (`log_pipeline_statistics`): only with the feature on, else a
    // validation error
    let pipeline_flags = if capture_statistics {
      vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR
        | vk::PipelineCreateFlags::CAPTURE_INTERNAL_REPRESENTATIONS_KHR
    } else {
      vk::PipelineCreateFlags::empty()
    };

    let mut create_pipeline = |spv_path: &str| -> GpuResult<(vk::Pipeline, u32, [u32; 3])> {
      let mut spv_code = aethervk_oshal_rlib::os::fs::read(spv_path)
        .map_err(|_| GpuError::BackendSpecific(alloc::format!("Failed to read {}", spv_path)))?;

      if cfg!(debug_assertions) || cfg!(test) {
        let (_, code_mut, _) = unsafe { spv_code.align_to_mut::<u32>() };
        crate::gpu_backends::vulkan::device::shader_manager::patch_spirv_prevent_inlining(code_mut);
        // TODO: re-enable once fixed issues .NET Side
        // crate::gpu_backends::vulkan::device::shader_manager::disassemble_and_log_spirv(
        //   spv_code.as_slice(),
        // );
      }

      let (prefix, code, suffix) = unsafe { spv_code.align_to::<u32>() };
      assert!(prefix.is_empty() && suffix.is_empty());

      // ── SPIR-V reflection: extract push constant block size and wg size ─────────────
      let (reflected_pc_size, workgroup_size) = {
        let spv_module = spirv_reflect::create_shader_module(&spv_code).map_err(|_| {
          GpuError::BackendSpecific(alloc::format!("spirv-reflect failed for {}", spv_path))
        })?;
        let pcs = spv_module.enumerate_push_constant_blocks(None).map_err(|_| {
          GpuError::BackendSpecific(alloc::format!(
            "spirv-reflect PC enum failed for {}",
            spv_path
          ))
        })?;

        let pc_size = if let Some(pc_block) = pcs.first() {
          pc_block.size
        } else {
          0 // shader has no push constants
        };

        let local_size = spv_module
          .enumerate_entry_points()
          .unwrap()
          .iter()
          .find(|ep| ep.name == "main")
          .map(|ep| [ep.local_size.x, ep.local_size.y, ep.local_size.z])
          .unwrap();

        (pc_size, local_size)
      };
      aethervk_oshal_rlib::log!(
        "[SPIR-V] {} -> push_constant_size = {} bytes",
        spv_path,
        reflected_pc_size
      );

      let shader_info = vk::ShaderModuleCreateInfo::default().code(code);
      let shader_module =
        unsafe { device.create_shader_module(&shader_info, None) }.map_err(|e| {
          GpuError::BackendSpecific(alloc::format!("Failed to create shader module: {:?}", e))
        })?;

      let main_name = alloc::ffi::CString::new("main").unwrap();

      let mut spec_map_entries = alloc::vec::Vec::new();
      let mut spec_data = alloc::vec::Vec::new();
      let sg_size = subgroup_size;
      spec_map_entries.push(vk::SpecializationMapEntry {
        constant_id: 0,
        offset: 0,
        size: 4,
      });
      spec_data.extend_from_slice(&sg_size.to_le_bytes());

      let debug_shaders_val = if debug_shaders { 1u32 } else { 0u32 };
      spec_map_entries.push(vk::SpecializationMapEntry {
        constant_id: 10,
        offset: 4,
        size: 4,
      });
      spec_data.extend_from_slice(&debug_shaders_val.to_le_bytes());

      // BVH traversal stack depths — computed to keep shared memory ≤ 16 KB.
      // With WG=256: subgroups_per_wg = 256/sg, budget_per_sg = 16384/(subgroups_per_wg*4) uints.
      let subgroups_per_wg = 256u32 / sg_size;
      let budget_per_sg = 16384u32 / (subgroups_per_wg * 4); // in uints
      let bvh_stack_depth = budget_per_sg.saturating_sub(1).min(128);
      let bvh_stack_depth_short = budget_per_sg.saturating_sub(1).min(64);
      spec_map_entries.push(vk::SpecializationMapEntry {
        constant_id: 2,
        offset: 8,
        size: 4,
      });
      spec_data.extend_from_slice(&bvh_stack_depth.to_le_bytes());
      spec_map_entries.push(vk::SpecializationMapEntry {
        constant_id: 3,
        offset: 12,
        size: 4,
      });
      spec_data.extend_from_slice(&bvh_stack_depth_short.to_le_bytes());

      let spec_info = vk::SpecializationInfo::default()
        .map_entries(&spec_map_entries)
        .data(&spec_data);

      let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&main_name)
        .specialization_info(&spec_info);

      let compute_info = vk::ComputePipelineCreateInfo::default()
        .flags(pipeline_flags)
        .stage(stage_info)
        .layout(pipeline_layout);

      use crate::gpu_backends::vulkan::device::VulkanDebugNameExt;
      let pipeline = unsafe {
        let mut p = vk::Pipeline::null();
        (device.handle.fp_v1_0().create_compute_pipelines)(
          device.handle.handle(),
          vk::PipelineCache::null(),
          1u32,
          core::ptr::from_ref(&compute_info),
          core::ptr::null(),
          core::ptr::from_mut(&mut p),
        )
        .result_with_success(p)
        .with_name(device, last_segment(spv_path))
      }
      .map_err(|e| {
        GpuError::BackendSpecific(alloc::format!(
          "Failed to create compute pipeline ({}): {:?}",
          spv_path,
          e
        ))
      })?;
      crate::gpu_backends::vulkan::utils::dump_pipeline_cache(
        &device.handle,
        last_segment(spv_path),
        |cache| unsafe {
          let mut p = vk::Pipeline::null();
          (device.handle.fp_v1_0().create_compute_pipelines)(
            device.handle.handle(),
            cache,
            1u32,
            core::ptr::from_ref(&compute_info),
            core::ptr::null(),
            core::ptr::from_mut(&mut p),
          )
          .result_with_success(p)
        },
      );

      unsafe {
        device.destroy_shader_module(shader_module, None);
      }
      created_pipelines.push(pipeline);

      Ok((pipeline, reflected_pc_size, workgroup_size))
    };

    // Need to adjust path depending on where the test runs from.
    let dir_lock = crate::gpu::ASSET_DIR.read();
    let base_dir = dir_lock.as_ref().unwrap();
    let sim_dir = if base_dir.ends_with("sim") {
      base_dir.clone()
    } else {
      alloc::format!("{}/sim", base_dir)
    };

    #[cfg(all(
      not(test),
      any(debug_assertions, feature = "shader_debug_sync"),
      not(target_vendor = "apple")
    ))]
    let use_debug = USE_PRINTF_SHADERS.load(core::sync::atomic::Ordering::Relaxed);

    #[cfg(not(all(
      not(test),
      any(debug_assertions, feature = "shader_debug_sync"),
      not(target_vendor = "apple")
    )))]
    let use_debug = false;

    // Helper to unwrap (Pipeline, pc_size) — stores pc_size, returns pipeline
    let mut pc_sizes = hashbrown::HashMap::<u64, u32>::new();
    // Helper to unwrap (Pipeline, workgroup_size) — stores pc_size, returns pipeline
    let mut wg_sizes = hashbrown::HashMap::<u64, [u32; 3]>::new();
    let vmm_infix = if has_vulkan_memory_model { ".vmm" } else { "" };

    // Create all pipelines using a helper that extracts and stores reflected PC sizes
    macro_rules! mk {
      ($path:expr) => {{
        let mut final_path = alloc::format!("{}/{}", sim_dir, $path);
        final_path = final_path.replace(".spv", &alloc::format!("{}.spv", vmm_infix));
        if use_debug {
          final_path = final_path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size, wg_size) = create_pipeline(&final_path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
        wg_sizes.insert(ash::vk::Handle::as_raw(pipeline), wg_size);
        pipeline
      }};
    }

    // mk_wg! — Lavapipe-aware workgroup-size variant selector.
    //
    // Applies to shaders whose local_size_x was 32 (integrate_bodies_p3,
    // rb_force_assign).  On Lavapipe (CPU device, subgroup_size ≤ 8) the
    // Lavapipe ARM64 JIT miscompiles the inner SIMD loop when local_size_x > sg:
    // it clobbers x30 (link register, reused as a data register in a leaf
    // function) with the loop counter and then executes `ld1r {v11.4s}, [x30]`
    // with x30 = 12 (loop counter value), causing SIGSEGV.
    //
    // Fix: on Lavapipe pick the wgN.spv variant where N = subgroup_size, so
    // each workgroup is exactly ONE SIMD batch — no inner loop, no aliasing.
    //
    // On every other backend (NVIDIA warp=32, AMD wave=64, Apple sg=32, …) the
    // wg32.spv variant is loaded, which is byte-for-byte the original shader.
    //
    // Path format: <stem>.comp.wgN.spv  (e.g. integrate_bodies_p3.comp.wg4.spv)
    //
    // CPU SIMD width → subgroup_size mapping (Lavapipe / llvmpipe):
    //   NEON (ARM64) / SSE2  →  4
    //   AVX / AVX2           →  8
    //   AVX-512              → 16
    //   (manual override via MESA_NO_AVX* env vars may give 1 or 2 — clamp to 4)
    //
    // mk_wg!  → throughput shaders: CPU picks wg{sg}, GPU uses the bare .spv (LOCAL_SIZE_X=128).
    // mk_wg_sg! → one-subgroup-per-WG shaders (BVH builders, gravity): always picks wg{sg}
    //             so LOCAL_SIZE_X == SUBGROUP_SIZE on every platform.
    #[allow(unused_macros)]
    macro_rules! mk_wg {
      ($stem:expr) => {{
        let mut path;
        if is_cpu && subgroup_size <= 16 {
          let wg_suffix = match subgroup_size {
            1..=4 => "wg4",
            5..=8 => "wg8",
            _ => "wg16",
          };
          path = alloc::format!("{}/{}{}.{}.spv", sim_dir, $stem, vmm_infix, wg_suffix);
        } else {
          path = alloc::format!("{}/{}{}.spv", sim_dir, $stem, vmm_infix);
        };
        if use_debug {
          path = path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size, wg_size) = create_pipeline(&path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
        wg_sizes.insert(ash::vk::Handle::as_raw(pipeline), wg_size);
        pipeline
      }};
      ($stem:expr, $wg:expr) => {{
        let mut path;
        if is_cpu && subgroup_size <= 16 {
          let wg_suffix = match subgroup_size {
            1..=4 => "wg4",
            5..=8 => "wg8",
            _ => "wg16",
          };
          path = alloc::format!("{}/{}{}.{}.spv", sim_dir, $stem, vmm_infix, wg_suffix);
        } else {
          path = alloc::format!("{}/{}{}.{}.spv", sim_dir, $stem, vmm_infix, $wg);
        };
        if use_debug {
          path = path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size, wg_size) = create_pipeline(&path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
        wg_sizes.insert(ash::vk::Handle::as_raw(pipeline), wg_size);
        pipeline
      }};
    }
    // For shaders that need LOCAL_SIZE_X == SUBGROUP_SIZE (one subgroup per WG):
    // BVH builders use gl_SubgroupID == 0 and subgroup ops over the full WG.
    #[allow(unused_macros)]
    macro_rules! mk_wg_sg {
      ($stem:expr) => {{
        let wg_suffix = match subgroup_size {
          1..=4 => "wg4",
          5..=8 => "wg8",
          9..=16 => "wg16",
          17..=32 => "wg32",
          33..=64 => "wg64",
          65..=128 => "wg128",
          _ => "wg256",
        };
        let mut path = alloc::format!("{}/{}{}.{}.spv", sim_dir, $stem, vmm_infix, wg_suffix);
        if use_debug {
          path = path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size, wg_size) = create_pipeline(&path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
        wg_sizes.insert(ash::vk::Handle::as_raw(pipeline), wg_size);
        pipeline
      }};
    }

    // mk_wg_fp! — like mk_wg! but also selects the NATIVE_FLOAT16 SPIR-V variant.
    // Used for the five particle shaders that use float16_t / f16vec4 arithmetic.
    //   NATIVE_FLOAT16=1 (capable hardware): loads the bare .comp[.wgN].spv
    //   NATIVE_FLOAT16=0 (Pascal / GTX10xx): loads .comp.nofp16[.wgN].spv
    #[allow(unused_macros)]
    macro_rules! mk_wg_fp {
      ($stem:expr) => {{
        let fp16_infix = if has_native_float16 { "" } else { ".nofp16" };
        let mut path;
        if is_cpu && subgroup_size <= 16 {
          let wg_suffix = match subgroup_size {
            1..=4 => "wg4",
            5..=8 => "wg8",
            _ => "wg16",
          };
          path = alloc::format!(
            "{}/{}{}{}.{}.spv",
            sim_dir,
            $stem,
            vmm_infix,
            fp16_infix,
            wg_suffix
          );
        } else {
          path = alloc::format!("{}/{}{}{}.spv", sim_dir, $stem, vmm_infix, fp16_infix);
        };
        if use_debug {
          path = path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size, wg_size) = create_pipeline(&path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
        wg_sizes.insert(ash::vk::Handle::as_raw(pipeline), wg_size);
        pipeline
      }};
      ($stem:expr, $wg:expr) => {{
        let fp16_infix = if has_native_float16 { "" } else { ".nofp16" };
        let mut path;
        if is_cpu && subgroup_size <= 16 {
          let wg_suffix = match subgroup_size {
            1..=4 => "wg4",
            5..=8 => "wg8",
            _ => "wg16",
          };
          path = alloc::format!(
            "{}/{}{}{}.{}.spv",
            sim_dir,
            $stem,
            vmm_infix,
            fp16_infix,
            wg_suffix
          );
        } else {
          path = alloc::format!(
            "{}/{}{}{}.{}.spv",
            sim_dir,
            $stem,
            vmm_infix,
            fp16_infix,
            $wg
          );
        };
        if use_debug {
          path = path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size, wg_size) = create_pipeline(&path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
        wg_sizes.insert(ash::vk::Handle::as_raw(pipeline), wg_size);
        pipeline
      }};
    }

    let res: GpuResult<Self> = (|| {
      // the staged blob sets where the monoliths do not compile
      // (`dust_staged_passes`); the monoliths are never loaded there
      let dust_propagate_passes = if staged_dust {
        Some([
          mk!("dust_propagate.comp.pass1.spv"),
          mk!("dust_propagate.comp.pass2.spv"),
          mk!("dust_propagate.comp.pass3.spv"),
          mk!("dust_propagate.comp.pass6.spv"),
          mk!("dust_propagate.comp.pass4.spv"),
          mk!("dust_propagate.comp.pass5m.spv"),
          mk!("dust_propagate.comp.pass5s.spv"),
        ])
      } else {
        None
      };
      let dust_emit_passes = if staged_dust {
        Some([
          mk!("dust_emit.comp.pass1.spv"),
          mk!("dust_emit.comp.pass2.spv"),
          mk!("dust_emit.comp.pass3.spv"),
          mk!("dust_emit.comp.pass6.spv"),
          mk!("dust_emit.comp.pass4.spv"),
          mk!("dust_emit.comp.pass5r.spv"),
          mk!("dust_emit.comp.pass5v.spv"),
          mk!("dust_emit.comp.pass71.spv"),
          mk!("dust_emit.comp.pass72.spv"),
        ])
      } else {
        None
      };
      let dust_emit = match dust_emit_passes {
        Some(p) => p[0],
        None => mk!("dust_emit.comp.spv"),
      };
      let dust_propagate = match dust_propagate_passes {
        Some(p) => p[0],
        None => mk!("dust_propagate.comp.spv"),
      };
      // the splat's stage chain: a blob per stage, the profiler variants too
      let dust_splat_stages = [
        mk!("dust_splat.comp.st0.spv"),
        mk!("dust_splat.comp.st1.spv"),
        mk!("dust_splat.comp.st2.spv"),
        mk!("dust_splat.comp.st3.spv"),
        mk!("dust_splat.comp.st4.spv"),
        mk!("dust_splat.comp.st5.spv"),
        mk!("dust_splat.comp.st6.spv"),
        mk!("dust_splat.comp.st7.spv"),
        mk!("dust_splat.comp.st8.spv"),
        mk!("dust_splat.comp.st9.spv"),
        mk!("dust_splat.comp.st10.spv"),
        mk!("dust_splat.comp.st11.spv"),
      ];
      let dust_splat_prof_stages = if has_shader_clock && dust_profile_requested() {
        Some([
          mk!("dust_splat.comp.prof.st0.spv"),
          mk!("dust_splat.comp.prof.st1.spv"),
          mk!("dust_splat.comp.prof.st2.spv"),
          mk!("dust_splat.comp.prof.st3.spv"),
          mk!("dust_splat.comp.prof.st4.spv"),
          mk!("dust_splat.comp.prof.st5.spv"),
          mk!("dust_splat.comp.prof.st6.spv"),
          mk!("dust_splat.comp.prof.st7.spv"),
          mk!("dust_splat.comp.prof.st8.spv"),
          mk!("dust_splat.comp.prof.st9.spv"),
          mk!("dust_splat.comp.prof.st10.spv"),
          mk!("dust_splat.comp.prof.st11.spv"),
        ])
      } else {
        None
      };
      // the monolithic enumerator ([`dust_splat_mono_wanted`]) and its profiler variant
      let dust_splat_mono = if dust_splat_mono_wanted(staged_dust) {
        Some(mk!("dust_splat_mono.comp.spv"))
      } else {
        None
      };
      let dust_splat_mono_prof = if dust_splat_mono.is_some() && has_shader_clock && dust_profile_requested() {
        Some(mk!("dust_splat_mono.comp.prof.spv"))
      } else {
        None
      };
      Ok(Self {
        pipeline_layout,
        // ── Dust v3 ──────────────────────────────────────────────────────────────
        dust_emit,
        dust_propagate,
        dust_propagate_passes,
        dust_emit_passes,
        dust_propagate_moments: mk!("dust_propagate.comp.moments.spv"),
        // the propagate / emit monoliths, where monoliths load
        dust_propagate_mono: if staged_dust { None } else { Some(mk!("dust_propagate_mono.comp.spv")) },
        dust_emit_mono: if staged_dust { None } else { Some(mk!("dust_emit_mono.comp.spv")) },
        dust_prop_mode: core::sync::atomic::AtomicU8::new(0),
        dust_splat: dust_splat_stages[0],
        dust_splat_stages,
        dust_splat_prof_stages,
        dust_splat_mono,
        dust_splat_mono_prof,
        dust_splat_mode: core::sync::atomic::AtomicU8::new(0),
        dust_splat_logged: core::sync::atomic::AtomicU8::new(0),
        dust_measure: mk!("dust_measure.comp.spv"),
        dust_draw_queue: mk!("dust_draw_queue.comp.spv"),
        dust_specks: mk!("dust_specks.comp.spv"),
        dust_specks_walk: mk!("dust_specks.comp.walk.spv"),
        dust_specks_setup: [
          mk!("dust_specks.comp.setup1.spv"),
          mk!("dust_specks.comp.setup2a.spv"),
          mk!("dust_specks.comp.setup2b.spv"),
          mk!("dust_specks.comp.setup3.spv"),
          mk!("dust_specks.comp.setup4.spv"),
          mk!("dust_specks.comp.setup5.spv"),
        ],
        dust_specks_leaves: mk!("dust_specks.comp.leaves.spv"),
        dust_specks_leaves1: mk!("dust_specks.comp.leaves1.spv"),
        dust_specks_trace: mk!("dust_specks.comp.trace.spv"),
        dust_splat_prof: dust_splat_prof_stages.map(|p| p[0]),
        dust_draw_queue_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_queue.comp.prof.spv"))
        } else {
          None
        },
        // the `.sg` blob reduces the top-level deposits over the subgroup
        // (compute subgroup arithmetic, optional on the baseline devices); else the plain one
        dust_draw_volume: if has_subgroup_arithmetic {
          mk!("dust_draw_volume.comp.sg.spv")
        } else {
          mk!("dust_draw_volume.comp.spv")
        },
        dust_draw_volume_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_volume.comp.prof.spv"))
        } else {
          None
        },
        dust_draw_volume_plain: if has_subgroup_arithmetic {
          Some(mk!("dust_draw_volume.comp.spv"))
        } else {
          None
        },
        // the staged consumer: seven stage blobs, `.sg` as dust_draw_volume
        dust_draw_volume_stages: if volume_stages {
          Some(if has_subgroup_arithmetic {
            [
              mk!("dust_draw_volume.comp.s1.sg.spv"),
              mk!("dust_draw_volume.comp.s2.sg.spv"),
              mk!("dust_draw_volume.comp.s3a.sg.spv"),
              mk!("dust_draw_volume.comp.s3b.sg.spv"),
              mk!("dust_draw_volume.comp.s4a.sg.spv"),
              mk!("dust_draw_volume.comp.s4b.sg.spv"),
              mk!("dust_draw_volume.comp.s5.sg.spv"),
            ]
          } else {
            [
              mk!("dust_draw_volume.comp.s1.spv"),
              mk!("dust_draw_volume.comp.s2.spv"),
              mk!("dust_draw_volume.comp.s3a.spv"),
              mk!("dust_draw_volume.comp.s3b.spv"),
              mk!("dust_draw_volume.comp.s4a.spv"),
              mk!("dust_draw_volume.comp.s4b.spv"),
              mk!("dust_draw_volume.comp.s5.spv"),
            ]
          })
        } else {
          None
        },
        dust_draw_small_edge: mk!("dust_draw_small_edge.comp.spv"),
        // the mono draw family: the monolithic blobs, chosen as dust_draw_volume's (`.sg` / plain);
        // none where the monoliths are not built (`staged_dust`)
        dust_draw_volume_mono: match (staged_dust, has_subgroup_arithmetic) {
          (true, _) => None,
          (false, true) => Some(mk!("dust_draw_volume_mono.comp.sg.spv")),
          (false, false) => Some(mk!("dust_draw_volume_mono.comp.spv")),
        },
        dust_draw_volume_mono_prof: if !staged_dust && has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_volume_mono.comp.prof.spv"))
        } else {
          None
        },
        dust_draw_volume_mono_plain: if !staged_dust && has_subgroup_arithmetic {
          Some(mk!("dust_draw_volume_mono.comp.spv"))
        } else {
          None
        },
        dust_draw_small_edge_mono: if staged_dust { None } else { Some(mk!("dust_draw_small_edge_mono.comp.spv")) },
        dust_draw_small_edge_mono_prof: if !staged_dust && has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_small_edge_mono.comp.prof.spv"))
        } else {
          None
        },
        dust_draw_queue_mono: if staged_dust { None } else { Some(mk!("dust_draw_queue_mono.comp.spv")) },
        dust_draw_queue_mono_prof: if !staged_dust && has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_queue_mono.comp.prof.spv"))
        } else {
          None
        },
        dust_draw_mode: core::sync::atomic::AtomicU8::new(0),
        dust_draw_small_edge_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_small_edge.comp.prof.spv"))
        } else {
          None
        },
        dust_split: mk!("dust_split.comp.spv"),
        // the pre-split splitter; none where the monoliths are not built
        dust_split_mono: if staged_dust { None } else { Some(mk!("dust_split_mono.comp.spv")) },
        dust_split_mono_prof: if !staged_dust && has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_split_mono.comp.prof.spv"))
        } else {
          None
        },
        dust_split_mode: core::sync::atomic::AtomicU8::new(0),
        dust_classify: mk!("dust_classify.comp.spv"),
        dust_compact: mk!("dust_compact.comp.spv"),
        dust_reproject: mk!("dust_reproject.comp.spv"),
        dust_split_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_split.comp.prof.spv"))
        } else {
          None
        },
        dust_split_prep: mk!("dust_split.comp.prep.spv"),
        dust_split_prep_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_split.comp.prof.prep.spv"))
        } else {
          None
        },
        dust_split_hard: mk!("dust_split.comp.hard.spv"),
        dust_split_hard_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_split.comp.prof.hard.spv"))
        } else {
          None
        },
        dust_split_seq: mk!("dust_split.comp.seq.spv"),
        dust_split_seq_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_split.comp.prof.seq.spv"))
        } else {
          None
        },
        dust_split_seq_draw: mk!("dust_split.comp.seqd.spv"),
        dust_split_seq_draw_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_split.comp.prof.seqd.spv"))
        } else {
          None
        },
        dust_project_leaf_p1: mk!("dust_project_leaf.comp.proj1.spv"),
        dust_project_leaf_p2: mk!("dust_project_leaf.comp.proj2.spv"),
        dust_project_leaf_p3: mk!("dust_project_leaf.comp.proj3.spv"),
        dust_project_leaf_p1_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_project_leaf.comp.prof.proj1.spv"))
        } else {
          None
        },
        dust_project_leaf_p2_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_project_leaf.comp.prof.proj2.spv"))
        } else {
          None
        },
        dust_project_leaf_p3_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_project_leaf.comp.prof.proj3.spv"))
        } else {
          None
        },
        dust_draw_leaf: mk!("dust_draw_leaf.comp.spv"),
        dust_draw_leaf_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_leaf.comp.prof.spv"))
        } else {
          None
        },
        dust_draw_leaf_wg: mk!("dust_draw_leaf.comp.wg.spv"),
        dust_draw_leaf_wg_prof: if has_shader_clock && dust_profile_requested() {
          Some(mk!("dust_draw_leaf.comp.prof.wg.spv"))
        } else {
          None
        },
        pc_sizes,
        wg_sizes,
        subgroup_size,
        is_lavapipe: is_cpu,
        has_native_float16,
      })
    })();
    match res {
      Ok(s) => Ok(s),
      Err(e) => {
        unsafe {
          for p in created_pipelines {
            device.destroy_pipeline(p, None);
          }
          device.destroy_pipeline_layout(pipeline_layout, None);
        }
        Err(e)
      }
    }
  }

  /// Debug assertion: verifies that the Rust push constant struct size matches
  /// the SPIR-V reflection for the given pipeline. Panics in debug builds on mismatch.
  #[inline]
  pub fn assert_pc_size(&self, pipeline: vk::Pipeline, rust_size: usize) {
    if let Some(&spv_size) = self.pc_sizes.get(&ash::vk::Handle::as_raw(pipeline)) {
      debug_assert_eq!(
        rust_size as u32, spv_size,
        "Push constant size mismatch! Rust struct = {} bytes, SPIR-V reflection = {} bytes (pipeline {:?})",
        rust_size, spv_size, pipeline
      );
    }
  }

  pub fn discard(&mut self, discard_pool: &resources::DiscardPool, timeline: u64) {
    // (emit and propagate: discarded below, as the staged passes or the monoliths)
    // (the monolithic enumerator and its profiler variant, when built)
    for p in [self.dust_splat_mono, self.dust_splat_mono_prof].into_iter().flatten() {
      discard_pool.discard_pipeline(p, timeline);
    }
    // (`dust_splat` / `dust_splat_prof` alias stage 0)
    for p in self.dust_splat_stages {
      discard_pool.discard_pipeline(p, timeline);
    }
    if let Some(ps) = self.dust_splat_prof_stages {
      for p in ps {
        discard_pool.discard_pipeline(p, timeline);
      }
    }
    discard_pool.discard_pipeline(self.dust_measure, timeline);
    discard_pool.discard_pipeline(self.dust_draw_queue, timeline);
    discard_pool.discard_pipeline(self.dust_specks, timeline);
    discard_pool.discard_pipeline(self.dust_specks_walk, timeline);
    for p in self.dust_specks_setup {
      discard_pool.discard_pipeline(p, timeline);
    }
    discard_pool.discard_pipeline(self.dust_specks_leaves, timeline);
    discard_pool.discard_pipeline(self.dust_specks_leaves1, timeline);
    discard_pool.discard_pipeline(self.dust_specks_trace, timeline);
    discard_pool.discard_pipeline(self.dust_propagate_moments, timeline);
    // the propagate / emit monoliths
    for p in [self.dust_propagate_mono, self.dust_emit_mono].into_iter().flatten() {
      discard_pool.discard_pipeline(p, timeline);
    }
    // the monoliths alias pass 1 when staged: discard the passes, else the monoliths
    match self.dust_propagate_passes {
      Some(ps) => {
        for p in ps {
          discard_pool.discard_pipeline(p, timeline);
        }
      }
      None => discard_pool.discard_pipeline(self.dust_propagate, timeline),
    }
    match self.dust_emit_passes {
      Some(ps) => {
        for p in ps {
          discard_pool.discard_pipeline(p, timeline);
        }
      }
      None => discard_pool.discard_pipeline(self.dust_emit, timeline),
    }
    // (dust_splat_prof: the stages above)
    if let Some(p) = self.dust_draw_queue_prof {
      discard_pool.discard_pipeline(p, timeline);
    }
    discard_pool.discard_pipeline(self.dust_draw_volume, timeline);
    if let Some(p) = self.dust_draw_volume_prof {
      discard_pool.discard_pipeline(p, timeline);
    }
    if let Some(p) = self.dust_draw_volume_plain {
      discard_pool.discard_pipeline(p, timeline);
    }
    if let Some(stages) = self.dust_draw_volume_stages {
      for p in stages {
        discard_pool.discard_pipeline(p, timeline);
      }
    }
    discard_pool.discard_pipeline(self.dust_draw_small_edge, timeline);
    // the mono draw family
    for p in [
      self.dust_draw_volume_mono,
      self.dust_draw_small_edge_mono,
      self.dust_draw_queue_mono,
      self.dust_draw_volume_mono_prof,
      self.dust_draw_volume_mono_plain,
      self.dust_draw_small_edge_mono_prof,
      self.dust_draw_queue_mono_prof,
    ]
    .into_iter()
    .flatten()
    {
      discard_pool.discard_pipeline(p, timeline);
    }
    if let Some(p) = self.dust_draw_small_edge_prof {
      discard_pool.discard_pipeline(p, timeline);
    }
    discard_pool.discard_pipeline(self.dust_split, timeline);
    // the pre-split splitter
    if let Some(p) = self.dust_split_mono {
      discard_pool.discard_pipeline(p, timeline);
    }
    if let Some(p) = self.dust_split_mono_prof {
      discard_pool.discard_pipeline(p, timeline);
    }
    discard_pool.discard_pipeline(self.dust_classify, timeline);
    discard_pool.discard_pipeline(self.dust_compact, timeline);
    discard_pool.discard_pipeline(self.dust_reproject, timeline);
    if let Some(p) = self.dust_split_prof {
      discard_pool.discard_pipeline(p, timeline);
    }
    // the splitter family
    for p in [
      self.dust_split_prep,
      self.dust_split_hard,
      self.dust_split_seq,
      self.dust_split_seq_draw,
      self.dust_project_leaf_p1,
      self.dust_project_leaf_p2,
      self.dust_project_leaf_p3,
      self.dust_draw_leaf,
      self.dust_draw_leaf_wg,
    ] {
      discard_pool.discard_pipeline(p, timeline);
    }
    for p in [
      self.dust_split_prep_prof,
      self.dust_split_hard_prof,
      self.dust_split_seq_prof,
      self.dust_split_seq_draw_prof,
      self.dust_project_leaf_p1_prof,
      self.dust_project_leaf_p2_prof,
      self.dust_project_leaf_p3_prof,
      self.dust_draw_leaf_prof,
      self.dust_draw_leaf_wg_prof,
    ]
    .into_iter()
    .flatten()
    {
      discard_pool.discard_pipeline(p, timeline);
    }

    discard_pool.discard_pipeline_layout(self.pipeline_layout, timeline);
  }

  /// The dust pipelines' compiled statistics (`VK_KHR_pipeline_executable_properties`:
  /// registers, local memory and whatever else the driver reports, by the driver's names), logged
  /// once at device creation under `AETHERVK_DUST_PROFILE=1`; the internal representations
  /// (NVIDIA: the SASS text) are written to `AETHERVK_DUST_PROFILE_DIR` when it is set, and a
  /// text one's local-memory loads and stores (`LDL` / `STL`) are counted.
  pub fn log_pipeline_statistics(&self, instance: &ash::Instance, device: &ash::Device) {
    let ext = ash::khr::pipeline_executable_properties::Device::new(instance, device);
    let dir = aethervk_oshal_rlib::os::env::var("AETHERVK_DUST_PROFILE_DIR");
    let mut named: alloc::vec::Vec<(alloc::string::String, Option<vk::Pipeline>)> = alloc::vec![
      ("dust_reproject", Some(self.dust_reproject)),
      ("dust_classify", Some(self.dust_classify)),
      ("dust_compact", Some(self.dust_compact)),
      ("dust_splat", Some(self.dust_splat)),
      ("dust_splat_prof", self.dust_splat_prof),
      ("dust_split", Some(self.dust_split)),
      ("dust_split_prof", self.dust_split_prof),
      ("dust_split_mono", self.dust_split_mono),
      ("dust_split_mono_prof", self.dust_split_mono_prof),
      ("dust_draw_queue", Some(self.dust_draw_queue)),
      ("dust_draw_queue_prof", self.dust_draw_queue_prof),
      ("dust_draw_volume", Some(self.dust_draw_volume)),
      ("dust_draw_volume_prof", self.dust_draw_volume_prof),
      ("dust_draw_volume_plain", self.dust_draw_volume_plain),
      ("dust_draw_volume_s1", self.dust_draw_volume_stages.map(|s| s[0])),
      ("dust_draw_volume_s2", self.dust_draw_volume_stages.map(|s| s[1])),
      ("dust_draw_volume_s3a", self.dust_draw_volume_stages.map(|s| s[2])),
      ("dust_draw_volume_s3b", self.dust_draw_volume_stages.map(|s| s[3])),
      ("dust_draw_volume_s4a", self.dust_draw_volume_stages.map(|s| s[4])),
      ("dust_draw_volume_s4b", self.dust_draw_volume_stages.map(|s| s[5])),
      ("dust_draw_volume_s5", self.dust_draw_volume_stages.map(|s| s[6])),
      ("dust_draw_small_edge", Some(self.dust_draw_small_edge)),
      ("dust_draw_small_edge_prof", self.dust_draw_small_edge_prof),
      ("dust_draw_volume_mono", self.dust_draw_volume_mono),
      ("dust_draw_volume_mono_prof", self.dust_draw_volume_mono_prof),
      ("dust_draw_volume_mono_plain", self.dust_draw_volume_mono_plain),
      ("dust_draw_small_edge_mono", self.dust_draw_small_edge_mono),
      ("dust_draw_small_edge_mono_prof", self.dust_draw_small_edge_mono_prof),
      ("dust_draw_queue_mono", self.dust_draw_queue_mono),
      ("dust_draw_queue_mono_prof", self.dust_draw_queue_mono_prof),
      ("dust_specks", Some(self.dust_specks)),
      ("dust_specks_walk", Some(self.dust_specks_walk)),
      ("dust_propagate", Some(self.dust_propagate)),
      ("dust_emit", Some(self.dust_emit)),
      ("dust_measure", Some(self.dust_measure)),
      ("dust_propagate_moments", Some(self.dust_propagate_moments)),
      ("dust_propagate_mono", self.dust_propagate_mono),
      ("dust_emit_mono", self.dust_emit_mono),
      ("dust_specks_setup1", Some(self.dust_specks_setup[0])),
      ("dust_specks_setup2a", Some(self.dust_specks_setup[1])),
      ("dust_specks_setup2b", Some(self.dust_specks_setup[2])),
      ("dust_specks_setup3", Some(self.dust_specks_setup[3])),
      ("dust_specks_setup4", Some(self.dust_specks_setup[4])),
      ("dust_specks_setup5", Some(self.dust_specks_setup[5])),
      ("dust_specks_leaves", Some(self.dust_specks_leaves)),
      ("dust_specks_leaves1", Some(self.dust_specks_leaves1)),
      ("dust_specks_trace", Some(self.dust_specks_trace)),
    ]
    .into_iter()
    .map(|(n, p)| (alloc::string::String::from(n), p))
    .collect();
    named.push((alloc::string::String::from("dust_splat_mono"), self.dust_splat_mono));
    named.push((alloc::string::String::from("dust_splat_mono_prof"), self.dust_splat_mono_prof));
    for (k, p) in self.dust_splat_stages.iter().enumerate() {
      named.push((alloc::format!("dust_splat_st{k}"), Some(*p)));
    }
    if let Some(ps) = &self.dust_splat_prof_stages {
      for (k, p) in ps.iter().enumerate() {
        named.push((alloc::format!("dust_splat_prof_st{k}"), Some(*p)));
      }
    }
    // the splitter family after the list
    let named: alloc::vec::Vec<(alloc::string::String, Option<vk::Pipeline>)> = named
      .into_iter()
      .map(|(n, p)| (alloc::string::String::from(n), p))
      .chain(
        [
          ("dust_split_prep", Some(self.dust_split_prep)),
          ("dust_split_prep_prof", self.dust_split_prep_prof),
          ("dust_split_hard", Some(self.dust_split_hard)),
          ("dust_split_hard_prof", self.dust_split_hard_prof),
          ("dust_split_seq", Some(self.dust_split_seq)),
          ("dust_split_seq_prof", self.dust_split_seq_prof),
          ("dust_split_seq_draw", Some(self.dust_split_seq_draw)),
          ("dust_split_seq_draw_prof", self.dust_split_seq_draw_prof),
          ("dust_project_leaf_p1", Some(self.dust_project_leaf_p1)),
          ("dust_project_leaf_p2", Some(self.dust_project_leaf_p2)),
          ("dust_project_leaf_p3", Some(self.dust_project_leaf_p3)),
          ("dust_project_leaf_p1_prof", self.dust_project_leaf_p1_prof),
          ("dust_project_leaf_p2_prof", self.dust_project_leaf_p2_prof),
          ("dust_project_leaf_p3_prof", self.dust_project_leaf_p3_prof),
          ("dust_draw_leaf", Some(self.dust_draw_leaf)),
          ("dust_draw_leaf_prof", self.dust_draw_leaf_prof),
          ("dust_draw_leaf_wg", Some(self.dust_draw_leaf_wg)),
          ("dust_draw_leaf_wg_prof", self.dust_draw_leaf_wg_prof),
        ]
        .into_iter()
        .map(|(n, p)| (alloc::string::String::from(n), p)),
      )
      .collect();
    let cstr = |a: &[core::ffi::c_char]| -> alloc::string::String {
      unsafe { core::ffi::CStr::from_ptr(a.as_ptr()) }.to_string_lossy().into_owned()
    };
    for (name, pipeline) in named {
      let Some(pipeline) = pipeline else { continue };
      let info = vk::PipelineInfoKHR::default().pipeline(pipeline);
      let props = match unsafe { ext.get_pipeline_executable_properties(&info) } {
        Ok(p) => p,
        Err(e) => {
          aethervk_oshal_rlib::log!("[dust profile] {name}: executable properties failed: {e:?}");
          continue;
        }
      };
      for (k, prop) in props.iter().enumerate() {
        let exec = vk::PipelineExecutableInfoKHR::default()
          .pipeline(pipeline)
          .executable_index(k as u32);
        let mut line = alloc::format!(
          "[dust profile] {name}[{k}] {} ({}; subgroup {}):",
          cstr(&prop.name),
          cstr(&prop.description),
          prop.subgroup_size
        );
        if let Ok(stats) = unsafe { ext.get_pipeline_executable_statistics(&exec) } {
          for s in &stats {
            let v = match s.format {
              vk::PipelineExecutableStatisticFormatKHR::BOOL32 => {
                alloc::format!("{}", unsafe { s.value.b32 } == vk::TRUE)
              }
              vk::PipelineExecutableStatisticFormatKHR::INT64 => {
                alloc::format!("{}", unsafe { s.value.i64 })
              }
              vk::PipelineExecutableStatisticFormatKHR::UINT64 => {
                alloc::format!("{}", unsafe { s.value.u64 })
              }
              vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => {
                alloc::format!("{}", unsafe { s.value.f64 })
              }
              _ => alloc::string::String::from("?"),
            };
            line.push_str(&alloc::format!(" {} = {v};", cstr(&s.name)));
          }
        }
        aethervk_oshal_rlib::log!("{line}");
        // the internal representations: ash's two-call wrapper reports the sizes (p_data null);
        // a third call with buffers fetches the data
        let Ok(mut reprs) = (unsafe { ext.get_pipeline_executable_internal_representations(&exec) })
        else {
          continue;
        };
        let mut buffers: alloc::vec::Vec<alloc::vec::Vec<u8>> =
          reprs.iter().map(|r| alloc::vec![0u8; r.data_size]).collect();
        for (r, b) in reprs.iter_mut().zip(buffers.iter_mut()) {
          r.p_data = b.as_mut_ptr().cast();
        }
        let mut count = reprs.len() as u32;
        let res = unsafe {
          (ext.fp().get_pipeline_executable_internal_representations_khr)(
            device.handle(),
            &exec,
            &mut count,
            reprs.as_mut_ptr(),
          )
        };
        if res != vk::Result::SUCCESS && res != vk::Result::INCOMPLETE {
          continue;
        }
        for (r, b) in reprs.iter().zip(buffers.iter()) {
          let rname = cstr(&r.name);
          let mut note = alloc::format!(
            "[dust profile] {name}[{k}] representation '{rname}' ({}): {} bytes{}",
            cstr(&r.description),
            r.data_size,
            if r.is_text == vk::TRUE { ", text" } else { ", binary" }
          );
          if r.is_text == vk::TRUE {
            let text = alloc::string::String::from_utf8_lossy(b);
            let count_op = |op: &str| text.lines().filter(|l| l.split_whitespace().any(|w| w == op || w.starts_with(&alloc::format!("{op}.")))).count();
            note.push_str(&alloc::format!(
              "; LDL {} STL {} (local-memory loads / stores in the text)",
              count_op("LDL"),
              count_op("STL")
            ));
          }
          #[cfg(any(test, feature = "std"))]
          if let Some(dir) = dir.as_deref() {
            let safe: alloc::string::String =
              rname.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
            let ext_ = if r.is_text == vk::TRUE { "txt" } else { "bin" };
            let path = alloc::format!("{dir}/{name}_{k}_{safe}.{ext_}");
            match std::fs::write(&path, b) {
              Ok(()) => note.push_str(&alloc::format!("; written to {path}")),
              Err(e) => note.push_str(&alloc::format!("; not written to {path}: {e}")),
            }
          }
          aethervk_oshal_rlib::log!("{note}");
        }
      }
    }
  }
}

pub struct VulkanComputeKernels {
  pub pipelines: PhysicsPipelines,
  pub timeline: vk::Semaphore,
  /// Need to keep track of current compute timeline value for discard pool
  pub next_submit_value: core::sync::atomic::AtomicU64,
  pub next_cmd_id: core::sync::atomic::AtomicU64,
  pub discard_pool: crate::gpu_backends::vulkan::device::resources::DiscardPool,
}

impl VulkanComputeKernels {
  pub fn new(
    device: &LogicalDevice,
    debug_shaders: bool,
    subgroup_size: u32,
    is_cpu: bool,
    has_native_float16: bool,
    has_vulkan_memory_model: bool,
    has_shader_clock: bool,
    capture_statistics: bool,
    has_subgroup_arithmetic: bool,
    staged_dust: bool,
    volume_stages: bool,
  ) -> GpuResult<Self> {
    let pipelines = PhysicsPipelines::new(
      device,
      debug_shaders,
      subgroup_size,
      is_cpu,
      has_native_float16,
      has_vulkan_memory_model,
      has_shader_clock,
      capture_statistics,
      has_subgroup_arithmetic,
      staged_dust,
      volume_stages,
    )?;

    let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
      .initial_value(0)
      .semaphore_type(vk::SemaphoreType::TIMELINE);
    let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
    let timeline = unsafe { device.create_semaphore(&sem_info, None) }?;

    let discard_pool =
      unsafe { crate::gpu_backends::vulkan::device::resources::DiscardPool::new(1024) };

    Ok(Self {
      pipelines,
      timeline,
      next_submit_value: core::sync::atomic::AtomicU64::new(1), // Timeline starts at 0, first signal is 1
      next_cmd_id: core::sync::atomic::AtomicU64::new(1),
      discard_pool,
    })
  }

  /// Returns the effective workgroup size to use when calculating dispatch groups.
  ///
  /// On CPU/Lavapipe we loaded a `wgN.spv` variant whose `local_size_x = N` equals
  /// the hardware subgroup size (4 for NEON/SSE, 8 for AVX, 16 for AVX-512).
  /// Dispatching `ceil(count / N)` groups means every CPU thread processes exactly
  /// one SIMD lane — no idle lanes, no wasted context switches.
  ///
  /// On GPU we keep the caller-supplied `gpu_target` (128 or 256) which is what
  /// the wg32.spv shader's `local_size_x` was compiled for.  The SPIR-V local size
  /// and the dispatch group count must always agree.
  #[inline]
  pub fn effective_wg(&self, gpu_target: u32) -> u32 {
    // CPU (Lavapipe): workgroup size == subgroup size (SIMD width).
    // GPU: throughput shaders load wg128 (bare .spv), so dispatch with gpu_target.
    // BVH shaders loaded with mk_wg_sg! use subgroup_size directly — they never
    // call effective_wg(); they compute dispatch groups as ceil(N / subgroup_size).
    if self.pipelines.is_lavapipe {
      self.pipelines.subgroup_size
    } else {
      gpu_target
    }
  }

  /// Function called on [`crate::gpu_backends::vulkan::device::Device::drop`]
  pub fn cleanup(&mut self, device: &LogicalDevice, allocator: vk_mem::AllocatorView) {
    self.pipelines.discard(&self.discard_pool, u64::MAX);
    self.discard_pool.destroy_discarded_resources_all(device);

    // Drain the VMA event ring one last time so all alloc/free events that
    // fired during cleanup are flushed into GPU_ALLOCATIONS before we destroy
    // the allocator. This ensures report_leaked_gpu_allocations() sees the
    // full picture. Must be called BEFORE destroying the VMA allocator.
    #[cfg(all(debug_assertions, any(feature = "debug_gpu", test)))]
    aethervk_oshal_rlib::os::memory::tracking::drain_vma_events();

    // since this is called after `vkDeviceWaitIdle`, we are safe
    unsafe { device.destroy_semaphore(self.timeline, None) };
  }
}

fn last_segment(path: &str) -> &str {
  // rfind looks from the end of the string backwards.
  // An array of chars implements the Pattern trait, matching any char in the array.
  if let Some(idx) = path.rfind(['/', '\\']) {
    // Safe to add 1 because both '/' and '\' are exactly 1 byte long in UTF-8.
    &path[idx + 1..]
  } else {
    // If no slashes are found, the whole string is the last segment.
    path
  }
}

#[cfg(test)]
#[path = "mock_physics_tests.rs"]
mod physics_tests;
