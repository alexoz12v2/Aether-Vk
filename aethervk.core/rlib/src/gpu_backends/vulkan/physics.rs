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
#[cfg(all(
  any(debug_assertions, test, feature = "shader_debug_sync"),
  not(target_vendor = "apple")
))]
pub static USE_PRINTF_SHADERS: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub static READBACK_DIAGNOSTICS: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

/// Particle System v2: boolean to switch from v1 to v2. All
/// ['crate::gpu_backends::vulkan::physics::VulkanComputeKernels'] instantiated when this is true
/// will use the new particle system simulation path, when combined with the new simulation
/// function
pub static USE_PARTICLE_SYSTEM_V2: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

/// Configuration parameters for the physics pipeline
pub struct PhysicsPipelineConfig {
  pub max_particles: u32,
  pub hardware_subgroup_size: u32,
}

/// Struct holding all [`ash::vk::Pipeline`] handles to pipelines to simulate particles on Vulkan
/// device. Note that all of them share the same [`ash::vk::PipelineLayout`] because it declares no
/// descriptor sets and permits usage of the max allowed by minimum requirements, 128 Bytes, for
/// push constant
pub struct PhysicsPipelines {
  /// Pipeline Layout shared by all pipelines
  pub pipeline_layout: vk::PipelineLayout,

  // ── New Particle System ───────────────────────────────────────────────────
  pub new_particles_compact_reset: vk::Pipeline,
  /// Constrained to a workgroup size of 64, because Since `PCHUNK_VEC4_SIZE` is 64, every thread
  /// processes one `vec4`. It's a perfect 1 to 1 mapping. It still works on smaller or bigger
  /// sizes, but with smaller, the workgroup does a stride loop, while on bigger excess threads
  /// sleep on the `barrier()`
  pub new_particles_emit: vk::Pipeline,
  pub new_particles_compact: vk::Pipeline,
  pub apply_emitters_direct_new: vk::Pipeline,
  pub integrate_particles_p1_p2_new: vk::Pipeline,
  pub integrate_particles_p4_5_new: vk::Pipeline,
  pub new_particles_offset_particles: vk::Pipeline,
  pub reset_particles: vk::Pipeline,

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

    let mut create_pipeline = |spv_path: &str| -> GpuResult<(vk::Pipeline, u32, [u32; 3])> {
      let mut spv_code = aethervk_oshal_rlib::os::fs::read(spv_path)
        .map_err(|_| GpuError::BackendSpecific(alloc::format!("Failed to read {}", spv_path)))?;

      if cfg!(debug_assertions) || cfg!(test) {
        let (_, code_mut, _) = unsafe { spv_code.align_to_mut::<u32>() };
        crate::gpu_backends::vulkan::device::shader_manager::patch_spirv_prevent_inlining(code_mut);
        crate::gpu_backends::vulkan::device::shader_manager::disassemble_and_log_spirv(
          spv_code.as_slice(),
        );
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

    #[cfg(all(test, not(target_vendor = "apple")))]
    let use_debug = USE_PRINTF_SHADERS.load(core::sync::atomic::Ordering::Relaxed);
    #[cfg(not(all(test, not(target_vendor = "apple"))))]
    let use_debug = false;

    // Helper to unwrap (Pipeline, pc_size) — stores pc_size, returns pipeline
    let mut pc_sizes = hashbrown::HashMap::<u64, u32>::new();
    // Helper to unwrap (Pipeline, workgroup_size) — stores pc_size, returns pipeline
    let mut wg_sizes = hashbrown::HashMap::<u64, [u32; 3]>::new();

    // Create all pipelines using a helper that extracts and stores reflected PC sizes
    macro_rules! mk {
      ($path:expr) => {{
        let mut final_path = alloc::format!("{}/{}", sim_dir, $path);
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
    macro_rules! mk_wg {
      ($stem:expr) => {{
        let mut path;
        if is_cpu && subgroup_size <= 16 {
          let wg_suffix = match subgroup_size {
            1..=4 => "wg4",
            5..=8 => "wg8",
            _ => "wg16",
          };
          path = alloc::format!("{}/{}.{}.spv", sim_dir, $stem, wg_suffix);
        } else {
          path = alloc::format!("{}/{}.spv", sim_dir, $stem);
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
          path = alloc::format!("{}/{}.{}.spv", sim_dir, $stem, wg_suffix);
        } else {
          path = alloc::format!("{}/{}.{}.spv", sim_dir, $stem, $wg);
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
        let mut path = alloc::format!("{}/{}.{}.spv", sim_dir, $stem, wg_suffix);
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
          path = alloc::format!("{}/{}{}.{}.spv", sim_dir, $stem, fp16_infix, wg_suffix);
        } else {
          path = alloc::format!("{}/{}{}.spv", sim_dir, $stem, fp16_infix);
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
          path = alloc::format!("{}/{}{}.{}.spv", sim_dir, $stem, fp16_infix, wg_suffix);
        } else {
          path = alloc::format!("{}/{}{}.{}.spv", sim_dir, $stem, fp16_infix, $wg);
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
      Ok(Self {
        pipeline_layout,
        // ── New Particle System ──────────────────────────────────────────────────────────────────────────────
        new_particles_compact_reset: mk_wg!("new_particles_compact_reset.comp"),
        new_particles_emit: mk_wg_fp!("new_particles_emit.comp", "wg64"),
        new_particles_compact: mk_wg!("new_particles_compact.comp", "wg64"),
        apply_emitters_direct_new: mk_wg_fp!("apply_emitters_direct_new.comp"),
        integrate_particles_p1_p2_new: mk_wg_fp!("integrate_particles_p1_p2_new.comp"),
        integrate_particles_p4_5_new: mk_wg_fp!("integrate_particles_p4_5_new.comp"),
        new_particles_offset_particles: mk_wg_fp!("new_particles_offset_particles.comp"),
        reset_particles: mk_wg!("reset_particles.comp"),
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
    // ── New Particle System ───────────────────────────────────────────────────
    discard_pool.discard_pipeline(self.new_particles_compact_reset, timeline);
    discard_pool.discard_pipeline(self.new_particles_emit, timeline);
    discard_pool.discard_pipeline(self.new_particles_compact, timeline);
    discard_pool.discard_pipeline(self.apply_emitters_direct_new, timeline);
    discard_pool.discard_pipeline(self.integrate_particles_p1_p2_new, timeline);
    discard_pool.discard_pipeline(self.integrate_particles_p4_5_new, timeline);
    discard_pool.discard_pipeline(self.new_particles_offset_particles, timeline);
    discard_pool.discard_pipeline(self.reset_particles, timeline);

    discard_pool.discard_pipeline_layout(self.pipeline_layout, timeline);
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
  ) -> GpuResult<Self> {
    let pipelines = PhysicsPipelines::new(
      device,
      debug_shaders,
      subgroup_size,
      is_cpu,
      has_native_float16,
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