//! Vulkan Backend Integration for the IMEX / LCP Physics Engine
//!
//! This module scaffolds the execution of the massive compute-shader pipeline.
//! It assumes Vulkan 1.1 with `VK_KHR_buffer_device_address` and `VK_KHR_shader_subgroup_basic`.

use crate::{
  gpu::{
    self, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, Kernels, KinematicBody, WaitHandle,
    compute_push_constants::*,
    vulkan::{
      device::{self, Device, LogicalDevice, commands, resources},
      utils,
    },
  },
  gpu_err,
  physics::physics_scene::{GpuReferenceFrame, PhysicsScene},
  scene::Scene,
  types::{EngineError, EngineResult, GpuError, GpuResult},
};
use aethervk_oshal_rlib::{
  math::{
    matrix::Matrix4,
    vector::{Vector, Vector3, Vector4, vec3::Vec3f32, vec4::Quat},
  },
  os::time::timeus_t,
};
use alloc::{format, vec::Vec};
use ash::vk;

// Disabled by default: Enabling PRINTF shaders under Lavapipe (ARM64) dramatically increases
// register pressure in the llvmpipe JIT compiler. This leads to register spilling bugs that
// overwrite the stack-saved link register (x30), causing a SIGSEGV upon kernel return.
#[cfg(all(any(debug_assertions, test), not(target_vendor = "apple")))]
pub static USE_PRINTF_SHADERS: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

#[cfg(test)]
pub static READBACK_DIAGNOSTICS: core::sync::atomic::AtomicBool =
  core::sync::atomic::AtomicBool::new(false);

use vk_mem::{Alloc, AllocatorView, AsAllocatorView};

/// Configuration parameters for the physics pipeline
pub struct PhysicsPipelineConfig {
  pub max_particles: u32,
  pub hardware_subgroup_size: u32,
}

/// TODO: remove this in favour of a flexible buffer class
#[deprecated]
#[derive(Default)]
pub struct PhysicsDeviceAddresses {
  pub particle_data: u64,
  pub rigid_body_data: u64,
  pub sorted_morton: u64,
  pub bvh_nodes: u64,
  pub atomic_counters: u64,
  pub ccd_candidates: u64,
  pub packed_collisions: u64,
  pub reduce_toi: u64,
  pub impulses: u64,
  pub emitters: u64,
}

// ─────────────────────────────────────────────────────────────────────────────

/// TODO: Document this item
pub struct PhysicsPipelines {
  pub pipeline_layout: vk::PipelineLayout,
  // ── Legacy IMEX pipelines (kept for backward compatibility) ───────────────
  pub emit_particles: vk::Pipeline,
  pub lbvh_prepass: vk::Pipeline,
  pub lbvh_build: vk::Pipeline,
  pub lbvh_build_bottomup: vk::Pipeline,
  pub motion_bounds: vk::Pipeline,
  pub motion_refit: vk::Pipeline,

  #[cfg(any(test, feature = "collisions"))]
  pub stream_compact: vk::Pipeline,
  #[cfg(any(test, feature = "collisions"))]
  pub reduce_toi: vk::Pipeline,
  #[cfg(any(test, feature = "collisions"))]
  pub lcp_solver: vk::Pipeline,
  #[cfg(any(test, feature = "collisions"))]
  pub apply_impulses: vk::Pipeline,
  pub barnes_hut: vk::Pipeline,
  pub radix_sort: vk::Pipeline,
  pub morton_encode: vk::Pipeline,
  pub permute_particles: vk::Pipeline,
  pub convert_particles: vk::Pipeline,
  #[cfg(any(test, feature = "collisions"))]
  pub graph_coloring: vk::Pipeline,
  #[cfg(any(test, feature = "collisions"))]
  pub lbvh_collapse: vk::Pipeline,
  // ── New Symmetric Strang-Split IMEX integrators ───────────────────────────
  /// VV predictor: x_n → x_{n+1}, v_{n+½}; clears force buffer
  pub integrate_particles_p1_p2: vk::Pipeline,
  /// RB Implicit Midpoint Rule + Picard gyro-stabilisation; clears wrench
  pub integrate_bodies_p3: vk::Pipeline,
  /// VV corrector: v_{n+½} → v_{n+1}; advances 64-bit engine clock
  pub integrate_particles_p4_5: vk::Pipeline,
  /// External gravity emitters → particle force accumulation (macro→micro transform)
  pub apply_emitters_to_particles: vk::Pipeline,
  // ── Narrow Phase ──────────────────────────────────────────────────────────
  #[cfg(any(test, feature = "collisions"))]
  pub narrow_ccd: vk::Pipeline,

  #[cfg(any(test, feature = "collisions"))]
  pub narrow_ccd_cross_lca: vk::Pipeline,
  // ── Force aggregation ─────────────────────────────────────────────────────
  /// Leaf-wrench → CoM-wrench reduction (one WG per RB)
  pub rb_force_assign: vk::Pipeline,
  // ── Broad-phase suite ─────────────────────────────────────────────────────
  #[cfg(any(test, feature = "collisions"))]
  pub bp_clear: vk::Pipeline,

  #[cfg(any(test, feature = "collisions"))]
  pub bp_bounds_gen: vk::Pipeline,

  #[cfg(any(test, feature = "collisions"))]
  pub bp_scene: vk::Pipeline,

  #[cfg(any(test, feature = "collisions"))]
  pub bp_classify: vk::Pipeline,

  #[cfg(any(test, feature = "collisions"))]
  pub bp_cross_lca: vk::Pipeline,

  #[cfg(any(test, feature = "collisions"))]
  pub bp_particle_self: vk::Pipeline,

  /// SPIR-V-reflected push constant block size per pipeline.
  /// Used by `debug_assert!` in dispatch helpers to catch size mismatches
  /// before they become cryptic Metal validation errors.
  pub pc_sizes: hashbrown::HashMap<u64, u32>,
  /// Hardware subgroup size (SIMD width), used for AOSOA packing.
  pub subgroup_size: u32,
  /// True when running on a CPU Vulkan device (Lavapipe / llvmpipe).
  /// Enables CPU-optimised SPIR-V variants and reduced workgroup sizes.
  pub is_lavapipe: bool,
}

impl PhysicsPipelines {
  /// TODO: Document this item
  pub fn new(
    device: &LogicalDevice,
    debug_shaders: bool,
    mut subgroup_size: u32,
    is_cpu: bool,
  ) -> GpuResult<Self> {
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

    let pipeline_layout =
      unsafe { device.create_pipeline_layout(&layout_info, None) }.map_err(|e| {
        GpuError::BackendSpecific(alloc::format!("Failed to create pipeline layout: {:?}", e))
      })?;

    let mut created_pipelines = alloc::vec::Vec::new();

    let mut create_pipeline = |spv_path: &str| -> GpuResult<(vk::Pipeline, u32)> {
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

      // ── SPIR-V reflection: extract push constant block size ─────────────
      let reflected_pc_size = {
        let spv_module = spirv_reflect::create_shader_module(&spv_code).map_err(|_| {
          GpuError::BackendSpecific(alloc::format!("spirv-reflect failed for {}", spv_path))
        })?;
        let pcs = spv_module.enumerate_push_constant_blocks(None).map_err(|_| {
          GpuError::BackendSpecific(alloc::format!(
            "spirv-reflect PC enum failed for {}",
            spv_path
          ))
        })?;
        if let Some(pc_block) = pcs.first() {
          pc_block.size
        } else {
          0 // shader has no push constants
        }
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

      let pipeline = unsafe {
        device.create_compute_pipelines(
          vk::PipelineCache::null(),
          core::slice::from_ref(&compute_info),
          None,
        )
      }
      .map_err(|(_pipelines, e)| {
        GpuError::BackendSpecific(alloc::format!(
          "Failed to create compute pipeline ({}): {:?}",
          spv_path,
          e
        ))
      })?[0];

      unsafe {
        device.destroy_shader_module(shader_module, None);
      }
      created_pipelines.push(pipeline);

      Ok((pipeline, reflected_pc_size))
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

    // Create all pipelines using a helper that extracts and stores reflected PC sizes
    macro_rules! mk {
      ($path:expr) => {{
        let mut final_path = alloc::format!("{}/{}", sim_dir, $path);
        if use_debug {
          final_path = final_path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size) = create_pipeline(&final_path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
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
    macro_rules! mk_wg {
      ($stem:expr) => {{
        let mut path;
        if is_cpu && subgroup_size <= 16 {
          let wg_suffix = match subgroup_size {
            1..=4 => "wg4", // SSE2/NEON or manual throttle → smallest
            5..=8 => "wg8", // AVX / AVX2
            _ => "wg16",    // AVX-512
          };
          path = alloc::format!("{}/{}.{}.spv", sim_dir, $stem, wg_suffix);
        } else {
          path = alloc::format!("{}/{}.spv", sim_dir, $stem);
        };
        if use_debug {
          path = path.replace(".spv", ".d.spv");
        }
        let (pipeline, pc_size) = create_pipeline(&path)?;
        pc_sizes.insert(ash::vk::Handle::as_raw(pipeline), pc_size);
        pipeline
      }};
    }

    let res: GpuResult<Self> = (|| {
      Ok(Self {
        pipeline_layout,
        // ── Particle integrators ──────────────────────────────────────────
        emit_particles: mk_wg!("emit_particles.comp"),
        integrate_particles_p1_p2: mk_wg!("integrate_particles_p1_p2.comp"),
        integrate_bodies_p3: mk_wg!("integrate_bodies_p3.comp"),
        integrate_particles_p4_5: mk_wg!("integrate_particles_p4_5.comp"),
        apply_emitters_to_particles: mk_wg!("apply_emitters_to_particles.comp"),
        rb_force_assign: mk_wg!("rb_force_assign.comp"),
        convert_particles: mk_wg!("convert_particles.comp"),
        // ── BVH builders ─────────────────────────────────────────────────
        lbvh_prepass: mk_wg!("lbvh_prepass.comp"),
        lbvh_build: mk_wg!("lbvh_build.comp"),
        lbvh_build_bottomup: mk_wg!("lbvh_build_bottomup.comp"),
        motion_bounds: mk_wg!("motion_bounds.comp"),
        motion_refit: mk_wg!("motion_refit.comp"),
        // ── Gravity / sorting ────────────────────────────────────────────
        barnes_hut: mk_wg!("barnes_hut.comp"),
        radix_sort: mk_wg!("radix_sort.comp"),
        // ── Single-variant (specialisation constant or single-threaded) ──
        morton_encode: mk!("morton_encode.comp.spv"),
        permute_particles: mk_wg!("permute_particles.comp"),
        #[cfg(any(test, feature = "collisions"))]
        lbvh_collapse: mk!("lbvh_collapse.comp.spv"),
        // ── Broad-phase ──────────────────────────────────────────────────
        #[cfg(any(test, feature = "collisions"))]
        bp_bounds_gen: mk_wg!("bp_bounds_gen.comp"),
        #[cfg(any(test, feature = "collisions"))]
        bp_scene: mk_wg!("bp_scene.comp"),
        #[cfg(any(test, feature = "collisions"))]
        bp_classify: mk_wg!("bp_classify.comp"),
        #[cfg(any(test, feature = "collisions"))]
        bp_cross_lca: mk_wg!("bp_cross_lca.comp"),
        #[cfg(any(test, feature = "collisions"))]
        bp_particle_self: mk_wg!("bp_particle_self.comp"),
        #[cfg(any(test, feature = "collisions"))]
        bp_clear: mk_wg!("bp_clear.comp"), // Must match subgroup size for MultiBvhNode layout
        // ── Narrow-phase / CCD ───────────────────────────────────────────
        #[cfg(any(test, feature = "collisions"))]
        narrow_ccd: mk_wg!("narrow_ccd.comp"),
        #[cfg(any(test, feature = "collisions"))]
        narrow_ccd_cross_lca: mk_wg!("narrow_ccd_cross_lca.comp"),
        #[cfg(any(test, feature = "collisions"))]
        reduce_toi: mk_wg!("reduce_toi.comp"),
        #[cfg(any(test, feature = "collisions"))]
        stream_compact: mk_wg!("stream_compact.comp"),
        // ── Collision resolution ──────────────────────────────────────────
        #[cfg(any(test, feature = "collisions"))]
        graph_coloring: mk_wg!("graph_coloring.comp"),
        #[cfg(any(test, feature = "collisions"))]
        lcp_solver: mk_wg!("lcp_solver.comp"),
        #[cfg(any(test, feature = "collisions"))]
        apply_impulses: mk_wg!("apply_impulses.comp"),
        pc_sizes,
        subgroup_size,
        is_lavapipe: is_cpu,
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
    // Layout must be last — it backs all pipelines
    discard_pool.discard_pipeline(self.emit_particles, timeline);
    discard_pool.discard_pipeline(self.lbvh_prepass, timeline);
    discard_pool.discard_pipeline(self.lbvh_build, timeline);
    discard_pool.discard_pipeline(self.lbvh_build_bottomup, timeline);
    discard_pool.discard_pipeline(self.motion_bounds, timeline);
    discard_pool.discard_pipeline(self.motion_refit, timeline);
    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.stream_compact, timeline);
    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.reduce_toi, timeline);
    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.lcp_solver, timeline);
    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.apply_impulses, timeline);
    discard_pool.discard_pipeline(self.barnes_hut, timeline);
    discard_pool.discard_pipeline(self.radix_sort, timeline);
    discard_pool.discard_pipeline(self.morton_encode, timeline);
    discard_pool.discard_pipeline(self.permute_particles, timeline);
    discard_pool.discard_pipeline(self.convert_particles, timeline);
    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.graph_coloring, timeline);
    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.lbvh_collapse, timeline);
    // New IMEX integrators
    discard_pool.discard_pipeline(self.integrate_particles_p1_p2, timeline);
    discard_pool.discard_pipeline(self.integrate_bodies_p3, timeline);
    discard_pool.discard_pipeline(self.integrate_particles_p4_5, timeline);
    // BUG FIX (2025-05): apply_emitters_to_particles was created in PhysicsPipelines::new()
    // but was accidentally omitted from this discard list.  At vkDestroyDevice the
    // validation layer reported "VkPipeline 0x... has not been destroyed"
    // (VUID-vkDestroyDevice-device-05137).  Added here to close the leak.
    discard_pool.discard_pipeline(self.apply_emitters_to_particles, timeline);

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.narrow_ccd, timeline);

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.narrow_ccd_cross_lca, timeline);
    // Force aggregation
    discard_pool.discard_pipeline(self.rb_force_assign, timeline);
    // Broad-phase suite

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.bp_clear, timeline);

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.bp_bounds_gen, timeline);

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.bp_scene, timeline);

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.bp_classify, timeline);

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.bp_cross_lca, timeline);

    #[cfg(any(test, feature = "collisions"))]
    discard_pool.discard_pipeline(self.bp_particle_self, timeline);
    discard_pool.discard_pipeline_layout(self.pipeline_layout, timeline);
  }
}

/// TODO: Document this item
pub struct VulkanCommandBuffer {
  pub cmd: vk::CommandBuffer,
  pub queue: device::Queue,
  pub command_pools: alloc::sync::Arc<crate::gpu_backends::vulkan::device::commands::CommandPools>,
  pub id: crate::gpu_backends::vulkan::device::commands::CommandBufferId,
  pub tid: aethervk_oshal_rlib::os::native::ThreadId,
  pub device_ptr: core::ptr::NonNull<LogicalDevice>,
  pub discard_pool_ptr:
    core::ptr::NonNull<crate::gpu_backends::vulkan::device::resources::DiscardPool>,
  pub timeline_value: u64,
  pub timeline_sem: vk::Semaphore,
  pub next_submit_value_ptr: core::ptr::NonNull<core::sync::atomic::AtomicU64>,
  pub assigned_timeline_value: alloc::sync::Arc<core::sync::atomic::AtomicU64>,
  /// Fence signalled on submit — used by VulkanReadHandle::wait() to guarantee
  /// Required for MoltenVK where timeline semaphore emulation can have sync gaps.
  pub throwaway_sem: vk::Semaphore,
}

unsafe impl Send for VulkanCommandBuffer {}
unsafe impl Sync for VulkanCommandBuffer {}

impl VulkanCommandBuffer {
  // TODO cleanup function
}
impl CommandBuffer for VulkanCommandBuffer {
  fn submit(&mut self) -> EngineResult<Option<crate::gpu::CommandBufferSyncInfo>> {
    unsafe {
      let device = self.device_ptr.as_ref();
      let discard_pool = self.discard_pool_ptr.as_ref();
      let next_submit_value = self.next_submit_value_ptr.as_ref();

      device.end_command_buffer(self.cmd).map_err(|e| GpuError::from(e))?;

      // Create a throwaway timeline semaphore for this submission
      let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
      let sem_ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
      self.throwaway_sem =
        device.handle.create_semaphore(&sem_ci, None).map_err(|e| GpuError::from(e))?;

      let command_buffers = [self.cmd];
      let signal_semaphores = [self.timeline_sem, self.throwaway_sem];

      // TAKE SUBMISSION LOCK BEFORE ALLOCATING TIMELINE!
      // This ensures that the order we get timeline values exactly matches the order we submit to the queue.
      let _guard = device.submission_lock.lock();
      self.timeline_value = next_submit_value.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
      let signal_values = [self.timeline_value, 1];

      let mut timeline_info =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

      let submit_info = vk::SubmitInfo::default()
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores)
        .push_next(&mut timeline_info);

      aethervk_oshal_rlib::log!("cmd.submit(): before queue_submit");
      device
        .handle
        .queue_submit(self.queue.handle, &[submit_info], vk::Fence::null())
        .map_err(|e| GpuError::from(e))?;
      aethervk_oshal_rlib::log!("cmd.submit(): after queue_submit");
      drop(_guard);

      self
        .assigned_timeline_value
        .store(self.timeline_value, core::sync::atomic::Ordering::Release);

      discard_pool.discard_command_buffer(
        self.tid,
        self.id,
        self.cmd,
        self.command_pools.clone(),
        self.timeline_value,
      );
      discard_pool.discard_semaphore(self.throwaway_sem, self.timeline_value);
    }

    Ok(Some(crate::gpu::CommandBufferSyncInfo {
      timeline_semaphore: ash::vk::Handle::as_raw(self.timeline_sem),
      timeline_value: self.timeline_value,
    }))
  }
}

/// TODO: Document this item
pub struct VulkanWaitHandle<T> {
  pub data: T,
}

impl<T: Send + Sync> WaitHandle<T> for VulkanWaitHandle<T> {
  fn wait(self) -> EngineResult<T> {
    Ok(self.data)
  }
}

pub struct VulkanReadHandle<T> {
  pub device: core::ptr::NonNull<LogicalDevice>,
  pub allocator: vk_mem::AllocatorView,
  pub staging_buffer: ash::vk::Buffer,
  pub staging_allocation: Option<vk_mem::Allocation>,
  pub is_list: bool,
  pub capacity: usize,
  pub timeline_sem: ash::vk::Semaphore,
  pub assigned_timeline_value: alloc::sync::Arc<core::sync::atomic::AtomicU64>,
  /// signal on MoltenVK where timeline semaphore emulation may have sync gaps.
  pub throwaway_sem: vk::Semaphore,
  pub _marker: core::marker::PhantomData<T>,
}

unsafe impl<T> Send for VulkanReadHandle<T> {}
unsafe impl<T> Sync for VulkanReadHandle<T> {}

impl<T> Drop for VulkanReadHandle<T> {
  fn drop(&mut self) {
    if let Some(mut alloc) = self.staging_allocation.take() {
      aethervk_oshal_rlib::log!(
        "WARN: VulkanReadHandle<{}> dropped without being awaited! Staging buffer freed immediately.",
        core::any::type_name::<T>()
      );
      unsafe {
        self.allocator.destroy_buffer(self.staging_buffer, &mut alloc);
      }
    }
  }
}

impl<T: Copy + Send + Sync> WaitHandle<Vec<T>> for VulkanReadHandle<T> {
  #[function_name::named]
  fn wait(mut self) -> EngineResult<Vec<T>> {
    let mut target_value = self.assigned_timeline_value.load(core::sync::atomic::Ordering::Acquire);

    // Safety spin: Ensure the submit() actually ran and allocated a timeline value
    while target_value == 0 {
      core::hint::spin_loop();
      target_value = self.assigned_timeline_value.load(core::sync::atomic::Ordering::Acquire);
    }

    let device = unsafe { self.device.as_ref() };

    // Wait on the throwaway timeline semaphore first
    if self.throwaway_sem != vk::Semaphore::null() {
      device.wait_for_semaphore_value(self.throwaway_sem, 1, u64::MAX).map_err(|e| {
        crate::types::EngineError::Gpu(crate::gpu_err!("Throwaway sem wait failed: {:?}", e))
      })?;
    }

    // Also wait on timeline semaphore (belt-and-suspenders)
    device
      .wait_for_semaphore_value(self.timeline_sem, target_value, u64::MAX)
      .map_err(|e| crate::types::EngineError::Gpu(crate::gpu_err!("Wait failed: {:?}", e)))?;

    let mut alloc_mut = self.staging_allocation.take().unwrap();
    let info = self.allocator.get_allocation_info(&alloc_mut);

    self
      .allocator
      .invalidate_allocation(&alloc_mut, 0, ash::vk::WHOLE_SIZE)
      .map_err(|e| crate::types::EngineError::Gpu(crate::gpu_err!("{}", e)))?;

    let mut data = alloc::vec::Vec::with_capacity(self.capacity);
    unsafe {
      if !info.mapped_data.is_null() {
        let count = if self.is_list {
          let c = unsafe { *((info.mapped_data as *const u32).add(3)) } as usize;
          c.min(self.capacity)
        } else {
          self.capacity
        };

        let offset = if self.is_list { 16 } else { 0 };
        let mapped_ptr = (info.mapped_data as *const u8).add(offset);
        core::ptr::copy_nonoverlapping(mapped_ptr as *const T, data.as_mut_ptr(), count);
        data.set_len(count);
      }

      // Cleanup staging buffer safely — both fence and timeline semaphore
      // have been waited on, GPU is guaranteed to be finished.
      self.allocator.destroy_buffer(self.staging_buffer, &mut alloc_mut);
    }
    Ok(data)
  }
}

/// TODO: Document this item
pub struct VulkanBuffer<T> {
  pub buffer: vk::Buffer,
  pub address: u64,
  pub capacity: usize,
  pub allocation: vk_mem::Allocation,
  pub allocator: vk_mem::AllocatorView,
  pub is_list: bool,
  pub usage: ash::vk::BufferUsageFlags,
  /// Set to true after `discard()` to prevent the Drop impl from double-freeing.
  discarded: bool,
  pub _marker: core::marker::PhantomData<T>,
}

impl<T> VulkanBuffer<T> {
  pub fn cast<U>(mut self) -> VulkanBuffer<U> {
    self.discarded = true;
    VulkanBuffer {
      buffer: self.buffer,
      address: self.address,
      capacity: self.capacity,
      allocation: self.allocation,
      allocator: self.allocator,
      is_list: self.is_list,
      usage: self.usage,
      discarded: false,
      _marker: core::marker::PhantomData,
    }
  }

  pub fn discard(
    &mut self,
    discard_pool: &crate::gpu_backends::vulkan::device::resources::DiscardPool,
    timeline: u64,
  ) {
    self.discarded = true;
    discard_pool.discard_buffer(
      self.allocator.get_raw(),
      self.buffer,
      self.allocation,
      timeline,
    );
  }

  /// Read buffer contents as a slice via its persistently-mapped pointer.
  ///
  /// # Safety
  ///
  /// The GPU must have finished writing to this buffer (timeline semaphore waited) before
  /// calling this. The buffer must have been allocated with `HOST_VISIBLE + HOST_COHERENT +
  /// MAPPED` flags (all buffers created by `allocate_device_buffer` satisfy this on Lavapipe).
  ///
  /// The returned slice is only valid as long as `self` lives.
  pub unsafe fn mapped_slice(&self) -> Option<&[T]>
  where
    T: Copy,
  {
    let info = self.allocator.get_allocation_info(&self.allocation);
    if info.mapped_data.is_null() {
      return None;
    }
    unsafe {
      let count = if self.is_list {
        // Lists have a 16-byte header. The element count is stored at the 4th `u32` (offset 12).
        let count_ptr = info.mapped_data as *const u32;
        (*count_ptr.add(3)) as usize
      } else {
        self.capacity
      };
      let data_ptr = if self.is_list {
        (info.mapped_data as *const u8).add(16) as *const T
      } else {
        info.mapped_data as *const T
      };
      Some(core::slice::from_raw_parts(
        data_ptr,
        count.min(self.capacity),
      ))
    }
  }
}

/// Safety-net `Drop` for `VulkanBuffer`:
/// In debug builds, emit a warning when a buffer is dropped without an explicit `discard()` call.
/// We intentionally do NOT free the VMA allocation here because at Drop time the GPU may still
/// be executing commands that reference this buffer. Immediate destruction would cause
/// use-after-free and Vulkan validation errors.
/// The root fix is to ensure all code paths (including error returns via `?`) call
/// `kernels.discard_buffer()` or `kernels.discard_list()` before returning.
impl<T> Drop for VulkanBuffer<T> {
  fn drop(&mut self) {
    #[cfg(debug_assertions)]
    if !self.discarded {
      aethervk_oshal_rlib::log!(
        "WARN: VulkanBuffer<{}> (capacity={}) dropped without discard() — GPU memory leak. \
         Ensure all error paths call kernels.discard_buffer() / discard_list().",
        core::any::type_name::<T>(),
        self.capacity,
      );
    }
  }
}

#[cfg(test)]
impl<T> VulkanBuffer<T> {
  /// Assign a human-readable name to this buffer's VMA allocation.
  /// In test builds, this name appears in VMA corruption diagnostics.
  pub fn set_name(&mut self, name: &core::ffi::CStr) {
    unsafe {
      self.allocator.set_allocation_name(&mut self.allocation, name.as_ptr());
    }
  }
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for VulkanBuffer<T> {
  type Cmd = VulkanCommandBuffer;
  type ReadHandle<'a>
    = VulkanReadHandle<T>
  where
    Self: 'a,
    T: 'a;
  fn capacity(&self) -> usize {
    self.capacity
  }
  fn address(&self) -> u64 {
    self.address
  }
  unsafe fn mapped_slice(&self) -> Option<&[T]> {
    self.mapped_slice()
  }
  fn enqueue_read_to_cpu(&self, cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
    let payload_size = (self.capacity.max(1) * core::mem::size_of::<T>()) as u64;
    let total_size = payload_size + if self.is_list { 16 } else { 0 };

    let buffer_info = ash::vk::BufferCreateInfo::default()
      .size(total_size)
      .usage(ash::vk::BufferUsageFlags::TRANSFER_DST);

    let mut alloc_info = vk_mem::AllocationCreateInfo::default();
    alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
    // Prefer HOST_CACHED for lightning-fast sequential reads from the CPU
    alloc_info.flags =
      vk_mem::AllocationCreateFlags::HOST_ACCESS_RANDOM | vk_mem::AllocationCreateFlags::MAPPED;
    crate::apply_test_dedicated_alloc!(alloc_info);

    let (staging_buffer, staging_allocation) =
      unsafe { self.allocator.create_buffer(&buffer_info, &alloc_info) }.map_err(|e| {
        crate::types::EngineError::Gpu(crate::types::GpuError::BackendSpecific(alloc::format!(
          "Failed to create staging buffer: {:?}",
          e
        )))
      })?;

    unsafe {
      let device = cmd.device_ptr.as_ref();

      // Ensure compute writes to the original buffer are complete before the transfer
      let pre_barrier = ash::vk::BufferMemoryBarrier2::default()
        .src_stage_mask(
          ash::vk::PipelineStageFlags2::COMPUTE_SHADER | ash::vk::PipelineStageFlags2::TRANSFER,
        )
        .src_access_mask(
          ash::vk::AccessFlags2::SHADER_WRITE | ash::vk::AccessFlags2::TRANSFER_WRITE,
        )
        .dst_stage_mask(ash::vk::PipelineStageFlags2::COPY)
        .dst_access_mask(ash::vk::AccessFlags2::TRANSFER_READ)
        .buffer(self.buffer)
        .offset(0)
        .size(ash::vk::WHOLE_SIZE);

      let pre_dep = ash::vk::DependencyInfo::default()
        .buffer_memory_barriers(core::slice::from_ref(&pre_barrier));
      device.synchronization2.cmd_pipeline_barrier2(cmd.cmd, &pre_dep);

      // Copy to the host-visible staging buffer natively using DMA speeds
      let copy_region = ash::vk::BufferCopy::default().src_offset(0).dst_offset(0).size(total_size);
      device.cmd_copy_buffer(
        cmd.cmd,
        self.buffer,
        staging_buffer,
        core::slice::from_ref(&copy_region),
      );

      // Ensure transfer writes are complete before the CPU reads memory on Wait()
      let post_barrier = ash::vk::BufferMemoryBarrier2::default()
        .src_stage_mask(ash::vk::PipelineStageFlags2::COPY)
        .src_access_mask(ash::vk::AccessFlags2::TRANSFER_WRITE)
        .dst_stage_mask(ash::vk::PipelineStageFlags2::HOST)
        .dst_access_mask(ash::vk::AccessFlags2::HOST_READ)
        .buffer(staging_buffer)
        .offset(0)
        .size(ash::vk::WHOLE_SIZE);

      let post_dep = ash::vk::DependencyInfo::default()
        .buffer_memory_barriers(core::slice::from_ref(&post_barrier));
      device.synchronization2.cmd_pipeline_barrier2(cmd.cmd, &post_dep);
    }

    Ok(VulkanReadHandle {
      device: cmd.device_ptr,
      allocator: self.allocator,
      staging_buffer,
      staging_allocation: Some(staging_allocation),
      is_list: self.is_list,
      capacity: self.capacity,
      timeline_sem: cmd.timeline_sem,
      assigned_timeline_value: cmd.assigned_timeline_value.clone(),
      throwaway_sem: vk::Semaphore::null(), // set after cmd.submit()
      _marker: core::marker::PhantomData,
    })
  }
}

impl<T: Copy + Send + Sync> DeviceList<T> for VulkanBuffer<T> {
  fn clear(&mut self, _cmd: &mut Self::Cmd) -> EngineResult<()> {
    Ok(())
  }
}

impl DeviceBvh for VulkanBuffer<()> {
  type Cmd = VulkanCommandBuffer;
  fn address(&self) -> u64 {
    self.address
  }
}

/// TODO: Document this item

pub struct TransientBufferEntry {
  pub buffer: vk::Buffer,
  pub address: u64,
  pub capacity: usize,
  pub allocation: vk_mem::Allocation,
  pub item_size: usize,
  pub is_list: bool,
  pub timeline_freed: u64,
  pub usage: vk::BufferUsageFlags,
}

pub struct TransientBufferPool {
  pub entries: alloc::vec::Vec<TransientBufferEntry>,
}
impl TransientBufferPool {
  pub fn new() -> Self {
    Self {
      entries: alloc::vec::Vec::new(),
    }
  }
}

pub struct VulkanComputeKernels {
  pub pipelines: PhysicsPipelines,
  pub addresses: PhysicsDeviceAddresses,
  pub timeline: vk::Semaphore,
  /// Need to keep track of current compute timeline value for discard pool
  pub next_submit_value: core::sync::atomic::AtomicU64,
  pub next_cmd_id: core::sync::atomic::AtomicU64,
  pub discard_pool: crate::gpu_backends::vulkan::device::resources::DiscardPool,
  pub queue_sharing_info: crate::gpu::QueueSharingInfo,
  pub transient_pool: spin::Mutex<TransientBufferPool>,
  pub particle_self_gravity_enabled: core::sync::atomic::AtomicBool,
  #[cfg(test)]
  pub tracked_physical_allocations: spin::Mutex<alloc::vec::Vec<u64>>,
}

impl VulkanComputeKernels {
  pub fn new(
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    queue_sharing_info: crate::gpu::QueueSharingInfo,
    debug_shaders: bool,
    subgroup_size: u32,
    is_cpu: bool,
  ) -> GpuResult<Self> {
    let pipelines = PhysicsPipelines::new(device, debug_shaders, subgroup_size, is_cpu)?;
    let addresses = PhysicsDeviceAddresses::default();

    let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
      .initial_value(0)
      .semaphore_type(vk::SemaphoreType::TIMELINE);
    let sem_info = vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info);
    let timeline = unsafe { device.create_semaphore(&sem_info, None) }?;

    let discard_pool =
      unsafe { crate::gpu_backends::vulkan::device::resources::DiscardPool::new(1024) };

    Ok(Self {
      pipelines,
      addresses,
      timeline,
      next_submit_value: core::sync::atomic::AtomicU64::new(1), // Timeline starts at 0, first signal is 1
      next_cmd_id: core::sync::atomic::AtomicU64::new(1),
      discard_pool,
      queue_sharing_info,
      transient_pool: spin::Mutex::new(TransientBufferPool::new()),
      particle_self_gravity_enabled: core::sync::atomic::AtomicBool::new(false),
      #[cfg(test)]
      tracked_physical_allocations: spin::Mutex::new(alloc::vec::Vec::new()),
    })
  }

  // TODO: How do I know if there's a command in flight? Should It be externally synchronized?

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
    if self.pipelines.is_lavapipe && self.pipelines.subgroup_size <= 16 {
      self.pipelines.subgroup_size.max(4)
    } else {
      gpu_target
    }
  }

  /// Returns the BDA of the GPU-built particle LBVH, updated each tick by `build_motion_bvh`.
  /// Returns 0 if no particles have been built yet (early frames or CPU path).
  pub fn get_particle_lbvh_address(&self) -> u64 {
    self.addresses.bvh_nodes
  }

  pub fn cleanup(&mut self, device: &LogicalDevice, _allocator: vk_mem::AllocatorView) {
    let mut pool = self.transient_pool.lock();
    for entry in pool.entries.drain(..) {
      self.discard_pool.discard_buffer(
        _allocator.get_raw(),
        entry.buffer,
        entry.allocation,
        u64::MAX,
      );
    }
    drop(pool);

    self.pipelines.discard(&self.discard_pool, u64::MAX);
    self.discard_pool.destroy_discarded_resources_all(device);

    unsafe { device.destroy_semaphore(self.timeline, None) };
  }

  pub(crate) fn recycle_transient_buffer<T: Copy + Send + Sync>(&self, mut buf: VulkanBuffer<T>, timeline: u64) {
    buf.discarded = true; // Prevent Drop warning
    self.transient_pool.lock().entries.push(TransientBufferEntry {
      buffer: buf.buffer,
      address: buf.address,
      capacity: buf.capacity,
      allocation: buf.allocation,
      item_size: core::mem::size_of::<T>(),
      is_list: buf.is_list,
      timeline_freed: timeline,
      usage: buf.usage,
    });
  }
}

impl VulkanComputeKernels {
  #[function_name::named]
  pub(crate) fn allocate_and_upload<T: Copy + Send + Sync>(
    &self,
    device: &LogicalDevice,
    allocator: AllocatorView,
    data: &[T],
    usage: vk::BufferUsageFlags,
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<VulkanBuffer<T>> {
    let is_list = false;
    let mut size = (core::mem::size_of::<T>() * data.len().max(1)) as u64;
    // Pad to 256 bytes to prevent Lavapipe LLVM JIT speculative out-of-bounds writes
    // from corrupting VMA block sentinels.
    if size % 256 != 0 {
      size += 256 - (size % 256);
    }

    let sharing_mode = if self.queue_sharing_info.mode == crate::gpu::SharingMode::Concurrent {
      vk::SharingMode::CONCURRENT
    } else {
      vk::SharingMode::EXCLUSIVE
    };

    let mut buffer_info = vk::BufferCreateInfo::default()
      .size(size)
      .usage(
        usage
          | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
          | vk::BufferUsageFlags::TRANSFER_SRC
          | vk::BufferUsageFlags::TRANSFER_DST,
      )
      .sharing_mode(sharing_mode);

    if sharing_mode == vk::SharingMode::CONCURRENT {
      buffer_info = buffer_info.queue_family_indices(&self.queue_sharing_info.queue_family_indices);
    }

    let alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::AutoPreferDevice,
      flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED,
      required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE
        | vk::MemoryPropertyFlags::HOST_COHERENT,
      ..Default::default()
    };
    crate::apply_test_dedicated_alloc!(alloc_info);

    let (buffer, mut alloc, info) =
      unsafe { allocator.create_buffer_get_info(&buffer_info, &alloc_info) }?;
    aethervk_oshal_rlib::log!("physics alloc: {:?}", alloc.get_raw());
    #[cfg(test)]
    {
      use ash::vk::Handle;
      self.tracked_physical_allocations.lock().push(info.device_memory.as_raw());
    }
    rollback.defer(move |_device| unsafe {
      allocator.destroy_buffer(buffer, &mut alloc);
    });

    if !data.is_empty() {
      unsafe {
        core::ptr::copy_nonoverlapping(
          data.as_ptr() as *const u8,
          info.mapped_data as *mut u8,
          core::mem::size_of::<T>() * data.len(),
        );
      }
    }

    let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let address =
      unsafe { device.buffer_device_address.get_buffer_device_address(&device_address_info) };

    let mut buf = VulkanBuffer {
      buffer,
      address,
      capacity: data.len().max(1),
      allocation: alloc,
      allocator,
      is_list,
      usage,
      discarded: false,
      _marker: core::marker::PhantomData,
    };
    #[cfg(test)]
    {
      let name = alloc::format!(
        "upload<{}> cap={} size={}\0",
        core::any::type_name::<T>(),
        data.len(),
        size
      );
      buf.set_name(unsafe { core::ffi::CStr::from_ptr(name.as_ptr() as *const _) });
    }
    Ok(buf)
  }
  #[function_name::named]
  pub(crate) fn allocate_device_buffer<T: Copy + Send + Sync>(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    capacity: usize,
    usage: vk::BufferUsageFlags,
    is_list: bool,
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<VulkanBuffer<T>> {
    let current_timeline = self.next_submit_value.load(core::sync::atomic::Ordering::Relaxed) - 1;
    let mut pool = self.transient_pool.lock();

    // Garbage collection of old transient buffers
    pool.entries.retain(|entry| {
      if entry.timeline_freed + 10 < current_timeline {
        self.discard_pool.discard_buffer(
          allocator.get_raw(),
          entry.buffer,
          entry.allocation,
          current_timeline,
        );
        false
      } else {
        true
      }
    });

    for i in 0..pool.entries.len() {
      let entry = &pool.entries[i];
      if entry.item_size == core::mem::size_of::<T>()
        && entry.capacity >= capacity
        && entry.is_list == is_list
        && (entry.usage & usage) == usage
        && entry.timeline_freed <= current_timeline + 1
      {
        let entry = pool.entries.remove(i);
        return Ok(VulkanBuffer {
          buffer: entry.buffer,
          address: entry.address,
          capacity: entry.capacity,
          allocation: entry.allocation,
          allocator,
          is_list: entry.is_list,
          usage: entry.usage,
          discarded: false,
          _marker: core::marker::PhantomData,
        });
      }
    }
    drop(pool);

    // Pad capacity to a multiple of 256 to prevent Lavapipe's LLVM JIT from
    // speculatively reading/writing out-of-bounds and corrupting VMA sentinels.
    let mut padded_capacity = capacity.max(1);
    if padded_capacity % 256 != 0 {
      padded_capacity += 256 - (padded_capacity % 256);
    }

    let payload_size = (core::mem::size_of::<T>() * padded_capacity) as u64;
    let size = payload_size + if is_list { 16 } else { 0 };
    aethervk_oshal_rlib::log!(
      "VMA CREATE BUFFER T: {}, capacity: {}, is_list: {}, size: {}",
      core::any::type_name::<T>(),
      capacity,
      is_list,
      size
    );

    let sharing_mode = if self.queue_sharing_info.mode == crate::gpu::SharingMode::Concurrent {
      vk::SharingMode::CONCURRENT
    } else {
      vk::SharingMode::EXCLUSIVE
    };

    let mut buffer_info = vk::BufferCreateInfo::default()
      .size(size)
      .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
      .sharing_mode(sharing_mode);

    if sharing_mode == vk::SharingMode::CONCURRENT {
      buffer_info = buffer_info.queue_family_indices(&self.queue_sharing_info.queue_family_indices);
    }

    let alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::AutoPreferDevice,
      flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED,
      required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE
        | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
      ..Default::default()
    };
    crate::apply_test_dedicated_alloc!(alloc_info);

    let (buffer, mut alloc, info) =
      unsafe { allocator.create_buffer_get_info(&buffer_info, &alloc_info) }?;
    aethervk_oshal_rlib::log!("physics alloc: {:?}", alloc.get_raw());
    #[cfg(test)]
    {
      use ash::vk::Handle;
      self.tracked_physical_allocations.lock().push(info.device_memory.as_raw());
    }
    rollback.defer(move |_device| unsafe {
      allocator.destroy_buffer(buffer, &mut alloc);
    });

    unsafe {
      if !info.mapped_data.is_null() && is_list {
        core::ptr::write_bytes(info.mapped_data as *mut u8, 0, 16);
      }
    }

    let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let address =
      unsafe { device.buffer_device_address.get_buffer_device_address(&device_address_info) };

    let mut buf = VulkanBuffer {
      buffer,
      address,
      capacity: capacity.max(1),
      allocation: alloc,
      allocator,
      is_list,
      usage,
      discarded: false,
      _marker: core::marker::PhantomData,
    };
    #[cfg(test)]
    {
      let name = alloc::format!(
        "device<{}> cap={} size={} list={}\0",
        core::any::type_name::<T>(),
        capacity,
        size,
        is_list
      );
      buf.set_name(unsafe { core::ffi::CStr::from_ptr(name.as_ptr() as *const _) });
    }
    Ok(buf)
  }

  // -- Methods From Kernel Trait implementation --
  #[function_name::named]
  fn create_command_buffer(
    &self,
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    command_pools: alloc::sync::Arc<crate::gpu_backends::vulkan::device::commands::CommandPools>,
    rollback: &mut utils::RollbackContext<'_>,
    compute_queue: device::Queue,
  ) -> GpuResult<<Device as gpu::Kernels>::Cmd> {
    let tid = aethervk_oshal_rlib::os::native::this_thread::id();
    // Use `next_cmd_id` to generate ID without advancing the timeline value
    let id = crate::gpu_backends::vulkan::device::commands::CommandBufferId(
      self.next_cmd_id.fetch_add(1, core::sync::atomic::Ordering::SeqCst),
    );

    let cmd = command_pools.allocate_primary(device, tid, id)?;

    // We defer recycling the command buffer to the compute timeline
    let cp_clone = command_pools.clone();
    rollback.defer(move |_dev| {
      let _ = cp_clone.recycle(tid, id, cmd);
    });

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info)? };

    Ok(VulkanCommandBuffer {
      cmd,
      queue: compute_queue,
      command_pools,
      id,
      tid,
      device_ptr: core::ptr::NonNull::from(device),
      discard_pool_ptr: core::ptr::NonNull::from(&self.discard_pool),
      timeline_value: 0, // Assigned correctly at submission time
      timeline_sem: self.timeline,
      next_submit_value_ptr: core::ptr::NonNull::from(&self.next_submit_value),
      assigned_timeline_value: alloc::sync::Arc::new(core::sync::atomic::AtomicU64::new(0)),
      throwaway_sem: vk::Semaphore::null(), // Created in submit()
    })
  }

  fn build_kinematic_bodies(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> GpuResult<VulkanBuffer<KinematicBody>> {
    let mut bodies = Vec::new();

    let get_shape_info = |entity| {
      scene0
        .with_component(entity, |c: &crate::scene::ColliderComponent| {
          match c.shape {
            crate::scene::ColliderShape::Sphere { radius } => (2, [radius, 0.0, 0.0]),
            crate::scene::ColliderShape::OBB { half_extents } => {
              (1, [half_extents.x(), half_extents.y(), half_extents.z()])
            }
          }
        })
        .unwrap_or((0, [1.0, 0.0, 0.0]))
    };

    scene0.query2::<crate::scene::TransformComponent, crate::scene::AlmanacPlanet, _>(
      |entity, transform, planet| {
        let t = scene0.global_transform(entity).unwrap_or(transform.clone());
        let vel = scene0
          .with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
          .unwrap_or(Vec3f32::zero());
        let parent_id = scene0
          .get_parent(entity)
          .map(|id| slotmap::Key::data(&id).as_ffi() as u32)
          .unwrap_or(0);
        let own_id = slotmap::Key::data(&entity).as_ffi() as u32;
        let (frame_type, scale) = scene0
          .with_component(entity, |f: &crate::scene::ReferenceFrameComponent| {
            (f.frame_type as u32, f.scale)
          })
          .unwrap_or((crate::scene::ReferenceFrameType::Macro as u32, 1.0));
        let (shape_type, shape_data) = get_shape_info(entity);
        bodies.push(KinematicBody {
          entity_id: entity,
          transform: t.clone(),
          velocity: vel,
          parent_frame_id: parent_id,
          mu: planet.mu,
          own_frame_id: own_id,
          frame_type,
          scale: scale * t.scale.x(),
          shape_type,
          shape_data,
        });
      },
    );

    self.allocate_and_upload(
      device,
      allocator,
      &bodies,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )
  }

  fn build_rigid_bodies(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> GpuResult<VulkanBuffer<gpu::RigidBodyGpu>> {
    let mut bodies = Vec::new();
    scene0.query2_without::<crate::scene::TransformComponent, crate::scene::ColliderComponent, crate::scene::particles::ParticleSystemComponent, _>(
            |entity, transform, collider| {
                use aethervk_oshal_rlib::math::vector::Vector;
                use aethervk_oshal_rlib::math::matrix::Matrix;
                let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
                let velocity = scene0.with_component(entity, |k: &crate::scene::KinematicComponent| k.velocity)
                    .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero());
                let angular_velocity = scene0.with_component(entity, |k: &crate::scene::KinematicComponent| k.angular_velocity)
                    .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero());

                let mass = collider.mass;
                let (_, _, inertia_tensor) = match collider.shape {
                    crate::scene::ColliderShape::Sphere { radius } => {
                    let i = 0.4 * mass * radius * radius;
                    (2, [radius, 0.0, 0.0], [i, 0.0, 0.0, 0.0, i, 0.0, 0.0, 0.0, i])
                    }
                    crate::scene::ColliderShape::OBB { half_extents } => {
                    let dx = half_extents.x() * 2.0;
                    let dy = half_extents.y() * 2.0;
                    let dz = half_extents.z() * 2.0;
                    let ix = (1.0 / 12.0) * mass * (dy * dy + dz * dz);
                    let iy = (1.0 / 12.0) * mass * (dx * dx + dz * dz);
                    let iz = (1.0 / 12.0) * mass * (dx * dx + dy * dy);
                    (1, [half_extents.x(), half_extents.y(), half_extents.z()], [ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz])
                    }
                };

                let rot_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_quat_custom_frame(transform.rotation);
                let rot_arr = [
                    rot_mat.component(0).unwrap(), rot_mat.component(1).unwrap(), rot_mat.component(2).unwrap(),
                    rot_mat.component(4).unwrap(), rot_mat.component(5).unwrap(), rot_mat.component(6).unwrap(),
                    rot_mat.component(8).unwrap(), rot_mat.component(9).unwrap(), rot_mat.component(10).unwrap(),
                ];

                bodies.push(gpu::RigidBodyGpu {
                    position: [transform.position.x(), transform.position.y(), transform.position.z()],
                    mass,
                    rotation: rot_arr,
                    _pad_rot: [0.0; 3],
                    linear_velocity: [velocity.x(), velocity.y(), velocity.z()],
                    _pad0: 0.0,
                    angular_velocity: [angular_velocity.x(), angular_velocity.y(), angular_velocity.z()],
                    _pad1: 0.0,
                    inertia_tensor,
                    _pad_inertia: [0.0; 3],
                });
            }
        );

    self.allocate_and_upload(
      device,
      allocator,
      &bodies,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )
  }

  fn build_rigid_bodies_imex(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    _scene: &PhysicsScene,
    scene0: &Scene,
  ) -> GpuResult<(VulkanBuffer<RigidBodyImex>, VulkanBuffer<Wrench>, u32)> {
    let mut bodies: alloc::vec::Vec<RigidBodyImex> = alloc::vec::Vec::new();
    let mut wrench_idx: u32 = 0;

    scene0.query2::<crate::scene::TransformComponent, crate::scene::KinematicComponent, _>(
      |entity, transform, kinematic| {
        let parent_id = scene0
          .get_parent(entity)
          .map(|id| slotmap::Key::data(&id).as_ffi())
          .unwrap_or(0);
        let macro_frame_idx = _scene.gpu_frames.iter().position(|f| f.frame_type == 0).unwrap_or(0);
        let frame_idx = _scene
          .gpu_frames
          .iter()
          .position(|f| f.entity_id_raw == parent_id)
          .unwrap_or(macro_frame_idx) as u32;
        let (m, i_inv_diag, shape_type, shape_extents) = scene0
          .with_component(entity, |c: &crate::scene::ColliderComponent| {
            match c.shape {
              crate::scene::ColliderShape::Sphere { radius } => {
                let m = c.mass;
                let i = 0.4 * m * radius * radius;
                let i_inv = if i > 0.0 { 1.0 / i } else { 0.0 };
                (
                  m as f32,
                  [i_inv as f32, i_inv as f32, i_inv as f32, 0.0],
                  2u32,
                  [radius as f32, radius as f32, radius as f32],
                )
              }
              crate::scene::ColliderShape::OBB { half_extents } => {
                let x = half_extents.x();
                let y = half_extents.y();
                let z = half_extents.z();
                let m = c.mass;
                let ix = (1.0 / 3.0) * m * (y * y + z * z);
                let iy = (1.0 / 3.0) * m * (x * x + z * z);
                let iz = (1.0 / 3.0) * m * (x * x + y * y);
                (
                  m as f32,
                  [
                    if ix > 0.0 { 1.0 / ix as f32 } else { 0.0 },
                    if iy > 0.0 { 1.0 / iy as f32 } else { 0.0 },
                    if iz > 0.0 { 1.0 / iz as f32 } else { 0.0 },
                    0.0,
                  ],
                  1u32,
                  [x as f32, y as f32, z as f32],
                )
              }
            }
          })
          .unwrap_or((1.0_f32, [0.0, 0.0, 0.0, 0.0], 0u32, [1.0, 1.0, 1.0]));

        let q = transform.rotation;
        bodies.push(RigidBodyImex {
          position_mass: [
            transform.position.x(),
            transform.position.y(),
            transform.position.z(),
            m,
          ],
          orientation: [q.0.x(), q.0.y(), q.0.z(), q.0.w()],
          linear_vel_drag: [
            kinematic.velocity.x(),
            kinematic.velocity.y(),
            kinematic.velocity.z(),
            0.01_f32,
          ],
          angular_vel_drag: [
            kinematic.angular_velocity.x(),
            kinematic.angular_velocity.y(),
            kinematic.angular_velocity.z(),
            0.01_f32,
          ],
          inertia_inv_diag: i_inv_diag,
          wrench_idx,
          leaf_start_idx: 0,
          leaf_count: 0,
          shape_type,
          shape_extents,
          frame_idx,
        });
        wrench_idx += 1;
      },
    );

    // Pad bodies (and matching wrenches) to a multiple of the actual local_size_x
    // of the SPIR-V variant loaded by mk_wg!:
    //   • Lavapipe (subgroup_size ≤ 8): wgN.spv  → local_size_x = subgroup_size
    //   • All other backends:           wg32.spv → local_size_x = 32
    // Padding ensures every speculative lane access in the shader hits valid
    // (zeroed) memory even when the invocation id ≥ n_bodies.
    let wg_size = if self.pipelines.subgroup_size as usize <= 8 {
      self.pipelines.subgroup_size as usize // Lavapipe: local_size_x == sg
    } else {
      32usize // all other backends
    };
    let real_n = bodies.len().max(1);
    let padded = ((real_n + wg_size - 1) / wg_size) * wg_size;
    // Extend bodies/wrenches with zeroed dummy entries up to `padded`.
    bodies.resize(padded, RigidBodyImex::default());

    let rb_buf = self.allocate_and_upload::<RigidBodyImex>(
      device,
      allocator,
      &bodies,
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
      rollback,
    )?;

    let zeroed_wrenches: alloc::vec::Vec<Wrench> = alloc::vec![Wrench::default(); padded];
    let w_buf = self.allocate_and_upload::<Wrench>(
      device,
      allocator,
      &zeroed_wrenches,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
      rollback,
    )?;

    // Return the ACTUAL body count (not the padded size) so the shader only
    // processes real bodies.  Dummy entries are unreachable (id >= n_bodies).
    let len = real_n as u32;
    Ok((rb_buf, w_buf, len))
  }

  fn build_frames(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    scene: &PhysicsScene,
  ) -> GpuResult<VulkanBuffer<GpuReferenceFrame>> {
    let frames = if scene.gpu_frames.is_empty() {
      alloc::vec![GpuReferenceFrame::default()]
    } else {
      scene.gpu_frames.clone()
    };
    self.allocate_and_upload::<GpuReferenceFrame>(
      device,
      allocator,
      &frames,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )
  }

  fn build_list_inner<T: Copy + Send + Sync>(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    capacity: usize,
  ) -> GpuResult<VulkanBuffer<T>> {
    self.allocate_device_buffer::<T>(
      device,
      allocator,
      capacity,
      ash::vk::BufferUsageFlags::STORAGE_BUFFER
        | ash::vk::BufferUsageFlags::TRANSFER_DST
        | ash::vk::BufferUsageFlags::TRANSFER_SRC,
      true,
      rollback,
    )
  }

  fn build_particles(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    scene0: &Scene,
  ) -> GpuResult<(VulkanBuffer<f32>, alloc::vec::Vec<gpu::ParticleMetadata>)> {
    let mut flat_particles = Vec::new();
    let mut metadata = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
            |entity, _transform, sys| {
                let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
                let particles = sys.particles.read();
                // TODO faster, pure GPU backing
                for (i, p) in particles.iter().enumerate().filter(|(_, p)| p.active != 0) {
                    flat_particles.push(alloc::vec![
                        p.position[0], p.position[1], p.position[2],
                        p.velocity[0], p.velocity[1], p.velocity[2],
                        p.mass,
                        0.0, 0.0, 0.0, // force slots (cleared each frame)
                        sys.beta,      // slot 10: radiation pressure β
                    ]);
                    metadata.push(gpu::ParticleMetadata {
                        entity_id: entity,
                        parent_frame_id: parent_id,
                        original_index: i as u32,
                    });
                }
            }
        );

    let sg = self.pipelines.subgroup_size as usize;
    let packed = gpu::pack_particles_aosoa(&flat_particles, sg, gpu::PARTICLE_FIELDS);

    let buffer = self.allocate_and_upload(
      device,
      allocator,
      &packed,
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST,
      rollback,
    )?;
    Ok((buffer, metadata))
  }

  fn build_particle_frame_ids(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    particle_metadata: &[gpu::ParticleMetadata],
  ) -> GpuResult<VulkanBuffer<u32>> {
    // Extract parent_frame_id in AOSOA order (same order build_particles emits particles)
    let ids: alloc::vec::Vec<u32> = if particle_metadata.is_empty() {
      alloc::vec![0u32] // dummy — shader guards on total_particles
    } else {
      particle_metadata.iter().map(|m| m.parent_frame_id).collect()
    };
    self.allocate_and_upload::<u32>(
      device,
      allocator,
      &ids,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )
  }

  fn build_emitters(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    scene0: &Scene,
  ) -> GpuResult<(VulkanBuffer<gpu::ForceEmitter>, u32)> {
    let mut emitters = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::ForceEmitterComponent, _>(
      |_, t, emitter| match emitter {
        crate::scene::ForceEmitterComponent::Gravity { mu, beta } => {
          emitters.push(gpu::ForceEmitter {
            position: [t.position.x(), t.position.y(), t.position.z()],
            mu: *mu,
            normal: [0.0, 0.0, 0.0],
            type_id: 0,
            trunc_distance: 0.0,
            beta: *beta,
            _pad: [0, 0],
          });
        }
        crate::scene::ForceEmitterComponent::Planar {
          normal,
          base_force,
          trunc_distance,
        } => {
          emitters.push(gpu::ForceEmitter {
            position: [t.position.x(), t.position.y(), t.position.z()],
            mu: *base_force,
            normal: [normal.x(), normal.y(), normal.z()],
            type_id: 1,
            trunc_distance: *trunc_distance,
            beta: 0.0,
            _pad: [0, 0],
          });
        }
      },
    );

    let len = emitters.len() as u32;
    self
      .allocate_and_upload(
        device,
        allocator,
        &emitters,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        rollback,
      )
      .map(|buf| (buf, len))
  }

  fn build_emission_candidates(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    scene0: &Scene,
  ) -> GpuResult<VulkanBuffer<f32>> {
    let mut flat_candidates = Vec::new();
    scene0
      .query2::<crate::scene::TransformComponent, crate::scene::ParticleEmitterCirclesComponent, _>(
        |_, _t, emitter| {
          for circle in &emitter.circles {
            let pos = circle.cached_point.unwrap_or([0.0, 0.0, 0.0]);
            let vel = circle.cached_normal.unwrap_or([0.0, 0.0, 0.0]);
            let mass = circle.mass;
            for _ in 0..circle.particles_per_tick {
              flat_candidates.push(alloc::vec![
                pos[0],
                pos[1],
                pos[2],
                vel[0],
                vel[1],
                vel[2],
                mass,
                circle.mean_velocity,
                circle.velocity_std_dev,
                circle.ttl as f32,
              ]);
            }
          }
        },
      );

    let sg = self.pipelines.subgroup_size as usize;
    let packed = gpu::pack_particles_aosoa(&flat_candidates, sg, gpu::PARTICLE_FIELDS);

    self.allocate_and_upload(
      device,
      allocator,
      &packed,
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST,
      rollback,
    )
  }

  fn emit_particles(
    &self,
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    _rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<f32>,
    _physical_scene: &PhysicsScene,
    _scene: &Scene,
    sun_pos: Vec3f32,
    dt: timeus_t,
  ) -> GpuResult<()> {
    let candidates_buf =
      self.build_emission_candidates(device, _allocator, _rollback, cmd, _scene)?;
    let atomic_counter = self.allocate_and_upload::<u32>(
      device,
      _allocator,
      &[0],
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST,
      _rollback,
    )?;

    let sg = self.pipelines.subgroup_size;
    let stride = gpu::PARTICLE_FIELDS as u32 * sg;
    let max_particles = (particles.capacity() as u32 / stride) * sg;
    let wg_size = self.effective_wg(128);
    let dispatch_groups = (max_particles + wg_size - 1) / wg_size;
    let num_candidates = (candidates_buf.capacity() / gpu::PARTICLE_FIELDS) as u32;

    let pc = EmitParticlesPushConstants {
      particles: particles.address,
      candidates: candidates_buf.address,
      bvh: self.addresses.bvh_nodes,
      counter: atomic_counter.address,
      root_index: 0,
      num_candidates,
      _pad0: [0; 2],
      sun_pos: [sun_pos.x(), sun_pos.y(), sun_pos.z()],
      _pad1: 0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<EmitParticlesPushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.emit_particles,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[],
        &[],
      );
      let timeline = self.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
      self.discard_pool.discard_buffer(
        candidates_buf.allocator.get_raw(),
        candidates_buf.buffer,
        candidates_buf.allocation,
        timeline,
      );
      self.discard_pool.discard_buffer(
        atomic_counter.allocator.get_raw(),
        atomic_counter.buffer,
        atomic_counter.allocation,
        timeline,
      );

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(())
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // New Symmetric Strang-Split IMEX Dispatch Methods
  // ═══════════════════════════════════════════════════════════════════════════

  /// Dispatches `integrate_particles_p1_p2.comp`.
  ///
  /// VV predictor — half-kick + full position leap to x_{n+1}.
  /// Clears the particle force buffer so force generators start from zero.
  ///
  /// `particles_addr` — BDA to AOSOA particle buffer (float[]).
  pub fn imex_integrate_particles_p1_p2(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    particles_addr: u64,
    total_particles: u32,
    dt: timeus_t,
  ) {
    if total_particles == 0 {
      return;
    }
    let dt_sec = dt as f32 / 1_000_000.0_f32;
    let wg_size = self.effective_wg(128);
    let groups = (total_particles + wg_size - 1) / wg_size;

    let pc = ImexParticlesP12PushConstants {
      particles: particles_addr,
      dt: dt_sec,
      total_particles,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.integrate_particles_p1_p2,
      );
      self
        .pipelines
        .assert_pc_size(self.pipelines.integrate_particles_p1_p2, bytes.len());
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      aethervk_oshal_rlib::log!(
        "P1P2 pc: address={}, total={}, dt={}",
        particles_addr,
        total_particles,
        dt_sec
      );
      if groups > 0 {
        device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      }
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `integrate_bodies_p3.comp`.
  ///
  /// Rigid Body Implicit Midpoint Rule + Picard gyroscopic stabilisation.
  /// Integrates all RBs from x_n to x_{n+1} and v_n to v_{n+1}.
  /// Clears the wrench buffer so force generators start from zero.
  ///
  /// `rigid_bodies_addr` — BDA to `RigidBodyArray` (imex_math.glsl layout: quaternion + wrench_idx).
  /// `wrenches_addr`     — BDA to `WrenchArray` (6-float `Wrench` per entry).
  /// `frames_addr`       — BDA to `GpuReferenceFrameArray` (for macro→micro transform).
  /// `n_iterations`      — Picard iteration count; 4 suffices for most scenes.
  pub fn imex_integrate_bodies_p3(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies_addr: u64,
    wrenches_addr: u64,
    emitters_addr: u64,
    frames_addr: u64,
    n_bodies: u32,
    num_emitters: u32,
    dt: timeus_t,
    n_iterations: u32,
  ) {
    if n_bodies == 0 {
      return;
    }
    let dt_sec = dt as f32 / 1_000_000.0_f32;
    let wg_size = if self.pipelines.subgroup_size <= 8 {
      self.pipelines.subgroup_size // Lavapipe: wg == sg, one SIMD batch per workgroup
    } else {
      32u32 // all other backends: natural local_size_x for this shader
    };
    let groups = (n_bodies + wg_size - 1) / wg_size;

    let pc = ImexBodiesP3PushConstants {
      rigid_bodies: rigid_bodies_addr,
      wrenches: wrenches_addr,
      emitters: emitters_addr,
      frames: frames_addr,
      dt: dt_sec,
      n_bodies,
      n_iterations,
      num_emitters,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.integrate_bodies_p3,
      );
      self.pipelines.assert_pc_size(self.pipelines.integrate_bodies_p3, bytes.len());
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if groups > 0 {
        device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      }
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `integrate_particles_p4_5.comp`.
  ///
  /// VV corrector — completes v_{n+½} → v_{n+1} using the freshly computed
  /// F(x_{n+1}).  Thread 0 simultaneously advances the 64-bit engine clock.
  /// Force buffer is intentionally **not** cleared (persists for next frame).
  ///
  /// `particles_addr`   — BDA to AOSOA particle buffer.
  /// `clock_addr`       — BDA to `ClockBuffer` (`uvec2 global_time_us`).
  /// `current_time_us`  — t_n as a 64-bit microsecond value.
  pub fn imex_integrate_particles_p4_5(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    particles_addr: u64,
    clock_addr: u64,
    total_particles: u32,
    dt: timeus_t,
    current_time_us: timeus_t,
  ) {
    // We must NOT return early if total_particles == 0 because Thread 0 is responsible
    // for advancing the global 64-bit engine clock! Even with 0 particles, we must dispatch at least 1 group.
    let dt_sec = dt as f32 / 1_000_000.0_f32;
    let wg_size = self.effective_wg(128);
    let groups = (total_particles.max(1) + wg_size - 1) / wg_size;

    let pc = ImexParticlesP45PushConstants {
      particles: particles_addr,
      clock: clock_addr,
      dt: dt_sec,
      total_particles,
      dt_us_lo: dt as u32,
      dt_us_hi: (dt >> 32) as u32,
      current_time_lo: current_time_us as u32,
      current_time_hi: (current_time_us >> 32) as u32,
      _pad_align16: [0; 2],
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.integrate_particles_p4_5,
      );
      self
        .pipelines
        .assert_pc_size(self.pipelines.integrate_particles_p4_5, bytes.len());
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if groups > 0 {
        device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      }
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `apply_emitters_to_particles.comp`.
  ///
  /// Applies macro-frame gravity emitters (e.g. the Sun) to microframe particles.
  /// Must run **after** Barnes-Hut (self-gravity) and before P4_5 (VV corrector)
  /// so that external forces are included in F(x_{n+1}).
  ///
  /// `particles_addr`         — BDA to AOSOA particle float buffer.
  /// `emitters_addr`          — BDA to EmitterArray.
  /// `frames_addr`            — BDA to GpuReferenceFrameArray.
  /// `particle_frame_ids_addr`— BDA to u32[]; one frame index per particle (AOSOA order).
  pub fn apply_emitters_to_particles(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    particles_addr: u64,
    emitters_addr: u64,
    frames_addr: u64,
    particle_frame_ids_addr: u64,
    num_emitters: u32,
    total_particles: u32,
  ) {
    if num_emitters == 0 || total_particles == 0 {
      return;
    }
    let wg_size = self.effective_wg(128);
    let groups = (total_particles + wg_size - 1) / wg_size;

    let pc = ApplyEmittersPushConstants {
      particles: particles_addr,
      emitters: emitters_addr,
      frames: frames_addr,
      particle_frame_ids: particle_frame_ids_addr,
      bvh: 0,
      num_emitters,
      total_particles,
      root_node_idx: 0,
      _pad: [0; 3],
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.apply_emitters_to_particles,
      );
      self
        .pipelines
        .assert_pc_size(self.pipelines.apply_emitters_to_particles, bytes.len());
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if groups > 0 {
        device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      }
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `rb_force_assign.comp`.
  ///
  /// Reduces per-leaf wrenches into each rigid body's CoM wrench.
  /// Dispatched as one workgroup per rigid body.
  ///
  /// Both leaf wrenches and CoM wrenches live in the same `WrenchArray` buffer;
  /// `body.wrench_idx` points to the CoM slot, `body.leaf_start_idx` to the
  /// first leaf slot.
  pub fn imex_rb_force_assign(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies_addr: u64,
    wrenches_addr: u64,
    n_bodies: u32,
  ) {
    if n_bodies == 0 {
      return;
    }
    let pc = RbForceAssignPushConstants {
      rigid_bodies: rigid_bodies_addr,
      wrenches: wrenches_addr,
      n_bodies,
      _pad: 0,
      _pad_align16: [0; 2],
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.rb_force_assign,
      );
      self.pipelines.assert_pc_size(self.pipelines.rb_force_assign, bytes.len());
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      // One WG per body
      if n_bodies > 0 {
        device.cmd_dispatch(cmd.cmd, n_bodies, 1, 1);
      }
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `bp_clear.comp` (single thread, clears all four pair queues).
  #[cfg(any(test, feature = "collisions"))]
  pub fn bp_clear(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    raw_scene_pairs: u64,
    out_rb_rb: u64,
    out_rb_ps: u64,
    out_rb_lca: u64,
    internal_pairs: u64,
    out_sparse: u64,
  ) {
    let pc = BpClearPushConstants {
      raw_scene_pairs,
      out_rb_rb,
      out_rb_ps,
      out_rb_lca,
      out_internal: internal_pairs,
      out_sparse,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.bp_clear,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, 1, 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `bp_bounds_gen.comp`.
  ///
  /// Generates one swept `TLASLeaf` per entity (n_particles + n_bodies threads).
  ///
  /// `scene_entities_addr` — BDA to `EntityArray` (rigid bodies; one `RigidBody` per entry).
  /// `tlas_leaves_addr`    — BDA to output `LeafBuffer`.
  #[cfg(any(test, feature = "collisions"))]
  pub fn bp_bounds_gen(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    scene_entities_addr: u64,
    tlas_leaves_addr: u64,
    lca_entities_addr: u64,
    total_entities: u32,
    dt: timeus_t,
  ) {
    let wg_size = self.effective_wg(128);
    let groups = (total_entities + wg_size - 1) / wg_size;

    let pc = BpBoundsGenPushConstants {
      scene_entities: scene_entities_addr,
      tlas_leaves: tlas_leaves_addr,
      lca_entities: lca_entities_addr,
      dt_us_lo: dt as u32,
      dt_us_hi: (dt >> 32) as u32,
      total_entities,
      _pad: 0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.bp_bounds_gen,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if groups > 0 {
        device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      }
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `bp_scene.comp`.
  ///
  /// Subgroup-cooperative TLAS traversal — each subgroup queries one leaf against
  /// the macro TLAS and outputs raw overlapping pairs.
  ///
  /// `tlas_bvh_addr`        — BDA to `MultiBvhBuffer`.
  /// `query_leaves_addr`    — BDA to swept `LeafBuffer` produced by `bp_bounds_gen`.
  /// `overlapping_pairs_addr` — BDA to output `PairBuffer` (`uint count; uvec2 pairs[]`).
  #[cfg(any(test, feature = "collisions"))]
  pub fn bp_scene(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    tlas_bvh_addr: u64,
    query_leaves_addr: u64,
    overlapping_pairs_addr: u64,
    tlas_root_index: u32,
    total_queries: u32,
  ) {
    // One subgroup per query — dispatch groups = ceil(queries / SUBGROUPS_PER_WG)
    // WG size = 256 for GPU (8 subgroups × 32-wide); on CPU use effective_wg().
    let wg_size = self.effective_wg(128);
    let subgroups_per_wg = wg_size / self.pipelines.subgroup_size.max(1);
    let groups = (total_queries + subgroups_per_wg - 1) / subgroups_per_wg;

    let pc = BpScenePushConstants {
      tlas_bvh: tlas_bvh_addr,
      query_leaves: query_leaves_addr,
      overlapping_pairs: overlapping_pairs_addr,
      tlas_root_index,
      total_queries,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.bp_scene,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, groups.max(1), 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
    let _ = wg_size; // suppress unused warning
  }

  /// Dispatches `bp_classify.comp`.
  ///
  /// Bins raw `overlapping_pairs` into RB-RB, RB-PS, and cross-LCA typed queues.
  #[cfg(any(test, feature = "collisions"))]
  pub fn bp_classify(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies_addr: u64,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_macro_lca_addr: u64,
    out_lca_lca_addr: u64,
    total_raw_pairs: u32,
    num_rigid_bodies: u32,
  ) {
    let wg_size = self.effective_wg(128);
    let groups = (total_raw_pairs + wg_size - 1) / wg_size;

    let pc = BpClassifyPushConstants {
      raw_pairs: raw_pairs_addr,
      out_rb_rb: out_rb_rb_addr,
      out_rb_ps: out_rb_ps_addr,
      out_ps_ps: out_ps_ps_addr,
      out_macro_lca: out_macro_lca_addr,
      out_lca_lca: out_lca_lca_addr,
      max_pairs: 4000, // Matches the allocation capacity in gpu_backends.rs
      num_rigid_bodies,
      rigid_bodies: rigid_bodies_addr,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.bp_classify,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, groups.max(1), 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `bp_cross_lca.comp`.
  ///
  /// Takes the cross-LCA pairs from `bp_classify` and refines them:
  /// transforms the macro-frame rigid body's AABB into the micro-frame using
  /// `InstanceDescriptor.inv_transform`, then traverses the micro-frame BVH
  /// to produce refined narrow-phase candidate pairs.
  #[cfg(any(test, feature = "collisions"))]
  pub fn bp_cross_lca(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    tlas_bvh_addr: u64,
    lca_entities_addr: u64,
    macro_leaves_addr: u64,
    entity_headers_addr: u64,
    lca_query_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_cross_pairs_addr: u64,
    total_queries: u32,
    max_pairs: u32,
    num_rigid_bodies: u32,
  ) {
    let _wg_size = self.effective_wg(128);
    let subgroups_per_wg = _wg_size / self.pipelines.subgroup_size.max(1);
    let groups = (total_queries + subgroups_per_wg - 1) / subgroups_per_wg;

    let pc = BpCrossLcaPushConstants {
      tlas_bvh_addr,
      lca_entities: lca_entities_addr,
      macro_leaves: macro_leaves_addr,
      rigid_bodies: entity_headers_addr,
      lca_query_pairs: lca_query_pairs_addr,
      out_rb_rb: out_rb_rb_addr,
      out_rb_ps: out_rb_ps_addr,
      out_ps_ps: out_ps_ps_addr,
      out_cross_pairs: out_cross_pairs_addr,
      total_queries,
      max_pairs,
      num_rigid_bodies,
      _pad: 0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.bp_cross_lca,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, groups.max(1), 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `bp_particle_self.comp`.
  ///
  /// Subgroup-cooperative LBVH traversal for particle–particle self-collision.
  /// For each overlapping pair, computes a Hookean spring repulsive force and
  /// atomicAdds it directly into the particle AOSOA force slots (indices 7/8/9).
  /// Bypasses the pair-list pipeline entirely.
  ///
  /// `bvh_addr`      — BDA to particle LBVH (`MultiBvhBuffer`).
  /// `particles_addr` — BDA to AOSOA particle buffer.
  /// `wrench_buffer` — BDA to the same particle buffer (used as float[] for atomicAdd).
  /// `stiffness`     — Spring constant k for the linear penalty force.
  #[cfg(any(test, feature = "collisions"))]
  pub fn bp_particle_self(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    bvh_addr: u64,
    particles_addr: u64,
    wrench_buffer_addr: u64,
    total_particles: u32,
    root_index: u32,
    particle_radius: f32,
    stiffness: f32,
  ) {
    let _wg_size = self.effective_wg(128);
    let subgroups_per_wg = _wg_size / self.pipelines.subgroup_size.max(1);
    let groups = (total_particles + subgroups_per_wg - 1) / subgroups_per_wg;

    let pc = BpParticleSelfPushConstants {
      bvh: bvh_addr,
      particles: particles_addr,
      wrench_buffer: wrench_buffer_addr,
      root_index,
      total_particles,
      particle_radius,
      stiffness,
      _pad_align16: [0; 2],
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.bp_particle_self,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, groups.max(1), 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  // Legacy dispatch methods (unchanged)
  // ═══════════════════════════════════════════════════════════════════════════

  fn step_ode_p1_p2(
    &self,
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    _rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<f32>,
    dt: timeus_t,
  ) -> GpuResult<()> {
    let wg_size = self.effective_wg(128);
    let total_particles = {
      let sg = self.pipelines.subgroup_size;
      let stride = gpu::PARTICLE_FIELDS as u32 * sg;
      (particles.capacity() as u32 / stride) * sg
    };
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;
    let dt_sec = dt as f32 / 1_000_000.0;

    let pc = P12PushConstants {
      particles: particles.address,
      dt: dt_sec,
      total_particles,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<P12PushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.integrate_particles_p1_p2,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(())
  }

  fn step_ode_p3_p4(
    &self,
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    _rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    kinematics: &VulkanBuffer<gpu::KinematicBody>,
    rigid_bodies: &VulkanBuffer<gpu::RigidBodyGpu>,
    _emitters: &VulkanBuffer<gpu::ForceEmitter>,
    dt: timeus_t,
  ) -> GpuResult<()> {
    let wg_size = self.effective_wg(128);
    let total_rigid_bodies = rigid_bodies.capacity() as u32;
    let dispatch_groups = (total_rigid_bodies + wg_size - 1) / wg_size;
    let dt_sec = dt as f32 / 1_000_000.0;

    let pc = P34PushConstants {
      rigid_bodies: self.addresses.rigid_body_data,
      emitters: self.addresses.emitters,
      kinematics: kinematics.address, // Using kinematics device address
      dt: dt_sec,
      total_rigid_bodies,
      num_emitters: 1,   // TODO dynamic
      num_kinematics: 0, // TODO dynamic
      _pad_align16: [0; 2],
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<P34PushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.integrate_bodies_p3,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(())
  }
  fn compute_self_gravity(
    &self,
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    _rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    bvh: &VulkanBuffer<()>,
    particles: &mut VulkanBuffer<f32>,
  ) -> GpuResult<()> {
    let total_particles = {
      let sg = self.pipelines.subgroup_size;
      let stride = gpu::PARTICLE_FIELDS as u32 * sg;
      (particles.capacity() as u32 / stride) * sg
    };
    let wg_size = self.effective_wg(128);
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;

    let pc_bh = BarnesHutPushConstants {
      particles: particles.address,
      bvh: bvh.address,
      cluster_list: 0,               // Fallback to sequential subgroup iteration if 0
      wrenches: 0,                   // Not used by particles
      num_clusters: total_particles, // cluster count ≈ particle count for now
      dt: 0.0,
      theta: 0.5,
      g: 1.0,
      softening_sq: 1e-6,
      root_node_idx: 0,
      cluster_threshold: 32,
      _pad: 0,
    };
    let bytes_bh = unsafe {
      core::slice::from_raw_parts(
        &pc_bh as *const _ as *const u8,
        core::mem::size_of::<BarnesHutPushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.barnes_hut,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes_bh,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      // TODO swittch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
    Ok(())
  }

  fn step_ode_p5(
    &self,
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    _rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    kinematics: &VulkanBuffer<KinematicBody>,
    particles: &mut VulkanBuffer<f32>,
    _emitters: &VulkanBuffer<gpu::ForceEmitter>,
    dt: timeus_t,
  ) -> GpuResult<()> {
    let wg_size = self.effective_wg(128);
    let total_particles = {
      let sg = self.pipelines.subgroup_size;
      let stride = gpu::PARTICLE_FIELDS as u32 * sg;
      (particles.capacity() as u32 / stride) * sg
    };
    let num_kinematics = kinematics.capacity() as u32;
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;
    let dt_sec = dt as f32 / 1_000_000.0;

    let pc = P5PushConstants {
      particles: particles.address,
      emitters: self.addresses.emitters,
      kinematics: kinematics.address,
      dt: dt_sec,
      total_particles,
      num_emitters: 1, // TODO dynamic -> VulkanBuffer
      num_kinematics,
      _pad_align16: [0; 2],
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<P5PushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.integrate_particles_p4_5,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(())
  }
  fn build_motion_bvh(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    _kinematics: &VulkanBuffer<gpu::KinematicBody>,
    _rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &mut VulkanBuffer<f32>,
    particle_frame_ids: &mut VulkanBuffer<u32>,
    _dt: timeus_t,
  ) -> GpuResult<VulkanBuffer<()>> {
    let total_particles = {
      let sg = self.pipelines.subgroup_size;
      let stride = gpu::PARTICLE_FIELDS as u32 * sg;
      (particles.capacity() as u32 / stride) * sg
    };
    let wg_size = self.effective_wg(128);
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;

    let num_nodes = (total_particles * 2).max(1) as usize;

    // Dispatch to the concrete MultiBvhNodeWideGpu<N> based on hardware subgroup size.
    macro_rules! alloc_bvh_buf {
      ($sg:literal) => {{
        type Node = crate::gpu::compute_push_constants::MultiBvhNodeWideGpu<$sg>;
        aethervk_oshal_rlib::log!(
          "build_motion_bvh: allocating bvh_buffer with num_nodes={}, capacity={}, size={}",
          num_nodes,
          particles.capacity(),
          num_nodes * core::mem::size_of::<Node>()
        );
        self
          .allocate_device_buffer::<Node>(
            device,
            allocator,
            num_nodes,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            false,
            rollback,
          )
          .map(|b| b.cast::<()>())
      }};
    }

    let bvh_buffer_result = match self.pipelines.subgroup_size {
      128 => alloc_bvh_buf!(128),
      64 => alloc_bvh_buf!(64),
      32 => alloc_bvh_buf!(32),
      16 => alloc_bvh_buf!(16),
      8 => alloc_bvh_buf!(8),
      4 => alloc_bvh_buf!(4),
      _ => alloc_bvh_buf!(32),
    };
    if let Err(e) = &bvh_buffer_result {
      aethervk_oshal_rlib::log!("build_motion_bvh failed to allocate bvh_buffer: {:?}", e);
    }
    let bvh_buffer = bvh_buffer_result?;
    aethervk_oshal_rlib::log!("build_motion_bvh: allocated bvh_buffer successfully");

    let mut sorted_morton = self.allocate_device_buffer::<[u32; 2]>(
      device,
      allocator,
      total_particles as usize,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
      false,
      rollback,
    )?;

    // 1. Morton Encode
    let mut pc_encode = crate::gpu::compute_push_constants::MortonEncodePushConstants {
      morton_out: sorted_morton.address,
      particles: particles.address,
      num_particles: total_particles,
      _pad0: 0,
      scene_min: [-1e9, -1e9, -1e9], // Arbitrary bounds for now
      _pad1: 0,
      scene_max: [1e9, 1e9, 1e9],
      _pad2: 0,
    };
    let bytes_encode = unsafe {
      core::slice::from_raw_parts(
        &pc_encode as *const _ as *const u8,
        core::mem::size_of::<crate::gpu::compute_push_constants::MortonEncodePushConstants>(),
      )
    };
    unsafe {
      device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.morton_encode);
      device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_encode);
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }
      let barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(sorted_morton.buffer)
        .size(vk::WHOLE_SIZE);
      device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);
    }

    // 2. Radix Sort
    let mut sorted_morton_alt = self.allocate_device_buffer::<[u32; 2]>(
      device,
      allocator,
      total_particles as usize,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::TRANSFER_SRC,
      false,
      rollback,
    )?;

    let num_blocks = (total_particles + 4095) / 4096;
    let histograms = self.allocate_device_buffer::<u32>(
      device,
      allocator,
      (16 * num_blocks) as usize,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      false,
      rollback,
    )?;

    unsafe {
      device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.radix_sort);

      let mut in_keys = sorted_morton.address;
      let mut out_keys = sorted_morton_alt.address;

      for shift in (0..30).step_by(4) {
        // Stage 0: Count
        let pc_count = crate::gpu::compute_push_constants::RadixSortPushConstants {
          input_keys: in_keys,
          output_keys: out_keys,
          histograms: histograms.address,
          num_particles: total_particles,
          shift,
          stage: 0,
          num_blocks,
        };
        device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, core::slice::from_raw_parts(&pc_count as *const _ as *const u8, core::mem::size_of_val(&pc_count)));
        if num_blocks > 0 {
          device.cmd_dispatch(cmd.cmd, num_blocks, 1, 1);
        }

        let barrier = vk::BufferMemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ).buffer(histograms.buffer).size(vk::WHOLE_SIZE);
        device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        // Stage 1: Scan
        let pc_scan = crate::gpu::compute_push_constants::RadixSortPushConstants {
          stage: 1,
          ..pc_count
        };
        device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, core::slice::from_raw_parts(&pc_scan as *const _ as *const u8, core::mem::size_of_val(&pc_scan)));
        device.cmd_dispatch(cmd.cmd, 1, 1, 1);

        device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), &[], &[barrier], &[]);

        // Stage 2: Scatter
        let pc_scatter = crate::gpu::compute_push_constants::RadixSortPushConstants {
          stage: 2,
          ..pc_count
        };
        device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, core::slice::from_raw_parts(&pc_scatter as *const _ as *const u8, core::mem::size_of_val(&pc_scatter)));
        if num_blocks > 0 {
          device.cmd_dispatch(cmd.cmd, num_blocks, 1, 1);
        }

        let barrier_keys = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
        device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier_keys), &[], &[]);

        core::mem::swap(&mut in_keys, &mut out_keys);
      }

      if in_keys != sorted_morton.address {
        core::mem::swap(&mut sorted_morton.buffer, &mut sorted_morton_alt.buffer);
        core::mem::swap(&mut sorted_morton.address, &mut sorted_morton_alt.address);
        core::mem::swap(&mut sorted_morton.allocation, &mut sorted_morton_alt.allocation);
      }
    }

    // 3. Permute Particles and Frame IDs
    let mut particles_out = self.allocate_device_buffer::<f32>(
      device,
      allocator,
      particles.capacity,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
      false,
      rollback,
    )?;
    particles_out.is_list = particles.is_list;

    let mut frame_ids_out = self.allocate_device_buffer::<u32>(
      device,
      allocator,
      particle_frame_ids.capacity,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
      false,
      rollback,
    )?;
    frame_ids_out.is_list = particle_frame_ids.is_list;

    let pc_permute = crate::gpu::compute_push_constants::PermuteParticlesPushConstants {
      particles_in: particles.address,
      particles_out: particles_out.address,
      frame_ids_in: particle_frame_ids.address,
      frame_ids_out: frame_ids_out.address,
      sorted_morton: sorted_morton.address,
      num_particles: total_particles,
      _pad: [0; 1],
    };
    unsafe {
      device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.permute_particles);
      device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, core::slice::from_raw_parts(&pc_permute as *const _ as *const u8, core::mem::size_of_val(&pc_permute)));
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }
      let barrier = vk::MemoryBarrier::default().src_access_mask(vk::AccessFlags::SHADER_WRITE).dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
    }

    core::mem::swap(&mut particles.buffer, &mut particles_out.buffer);
    core::mem::swap(&mut particles.address, &mut particles_out.address);
    core::mem::swap(&mut particles.allocation, &mut particles_out.allocation);

    core::mem::swap(&mut particle_frame_ids.buffer, &mut frame_ids_out.buffer);
    core::mem::swap(&mut particle_frame_ids.address, &mut frame_ids_out.address);
    core::mem::swap(&mut particle_frame_ids.allocation, &mut frame_ids_out.allocation);

    let atomic_counters = self.allocate_device_buffer::<u32>(
      device,
      allocator,
      num_nodes,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
      true,
      rollback,
    )?;

    let pc = LbvhPushConstants {
      bvh: bvh_buffer.address, // self.addresses.bvh_nodes,
      sorted_morton: 0,
      counters: atomic_counters.address,
      particles: particles.address, // self.addresses.particle_data,
      num_primitives: total_particles,
      particle_radius: 1.0, // TODO
      dt: _dt as f32 / 1_000_000.0,
      _pad: 0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<LbvhPushConstants>(),
      )
    };

    unsafe {
      let pc_prepass = crate::gpu::compute_push_constants::LbvhPrepassPushConstants {
        bvh: bvh_buffer.address,
        counters_addr: atomic_counters.address,
        num_internal_nodes: (total_particles.saturating_sub(1)).max(1) as u32,
        _pad: 0,
        _pad2: 0,
      };
      let bytes_prepass = core::slice::from_raw_parts(
        &pc_prepass as *const _ as *const u8,
        core::mem::size_of::<crate::gpu::compute_push_constants::LbvhPrepassPushConstants>(),
      );

      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.lbvh_prepass,
      );
      self.pipelines.assert_pc_size(self.pipelines.lbvh_prepass, bytes_prepass.len());
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes_prepass,
      );
      let prepass_wg = self.effective_wg(128);
      let prepass_groups = (total_particles + prepass_wg - 1) / prepass_wg;
      if prepass_groups > 0 {
        device.cmd_dispatch(cmd.cmd, prepass_groups, 1, 1);
      }

      let fill_barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .buffer(atomic_counters.buffer)
        .size(vk::WHOLE_SIZE);

      let bvh_barrier = vk::BufferMemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .buffer(bvh_buffer.buffer)
        .size(vk::WHOLE_SIZE);

      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[fill_barrier, bvh_barrier],
        &[],
      );

      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.lbvh_build,
      );
      self.pipelines.assert_pc_size(self.pipelines.lbvh_build, bytes.len());
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );

      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.lbvh_build_bottomup,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    let timeline = self.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
    self.recycle_transient_buffer(atomic_counters, timeline);
    self.recycle_transient_buffer(particles_out, timeline);
    self.recycle_transient_buffer(frame_ids_out, timeline);
    self.recycle_transient_buffer(sorted_morton, timeline);
    self.recycle_transient_buffer(sorted_morton_alt, timeline);
    self.recycle_transient_buffer(histograms, timeline);

    Ok(bvh_buffer)
  }

  #[function_name::named]
  pub fn refit_motion_blas(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    bvh_addr: u64,
    depth_indices_addr: u64,
    total_nodes: u32,
  ) -> GpuResult<()> {
    let pc = gpu::compute_push_constants::MotionRefitPushConstants {
      bvh: bvh_addr,
      depth_indices_addr,
      total_nodes_at_depth: total_nodes,
      _pad: 0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<gpu::compute_push_constants::MotionRefitPushConstants>(),
      )
    };

    unsafe {
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.motion_refit,
      );
      let wg_size = self.effective_wg(128);
      let dispatch_groups = (total_nodes + wg_size - 1) / wg_size;
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }
    }
    Ok(())
  }

  #[function_name::named]
  #[cfg(any(test, feature = "collisions"))]
  fn self_intersect_scene(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    _bvh: &VulkanBuffer<()>,
  ) -> GpuResult<VulkanBuffer<gpu::CollisionPair>> {
    // We'll pass total_entities via some state, hardcoded to some value here or assume we have it
    let total_entities = 1000; // Placeholder
    let wg_size = 32; // TODO: one subgroup (=warp) per BVH traversal stack
    let dispatch_groups = (total_entities + wg_size - 1) / wg_size;

    let max_candidates = 10000; // Placeholder TODO parameter of kernels?
    let candidates_buffer = self.allocate_device_buffer::<gpu::CollisionPair>(
      device,
      allocator,
      max_candidates,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
      true,
      rollback,
    )?;

    let pc = BpScenePushConstants {
      tlas_bvh: _bvh.address,
      query_leaves: self.addresses.particle_data, // Placeholder
      overlapping_pairs: candidates_buffer.address, // self.addresses.ccd_candidates,
      tlas_root_index: 0,
      total_queries: total_entities,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<BpScenePushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.bp_scene,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(candidates_buffer)
  }

  #[cfg(any(test, feature = "collisions"))]
  #[deprecated]
  fn intersect_instances(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    potentials: &VulkanBuffer<gpu::CollisionPair>,
    _kinematics: &VulkanBuffer<gpu::KinematicBody>,
    _rigid_bodies: &VulkanBuffer<gpu::RigidBodyImex>,
    _particles: &VulkanBuffer<f32>,
  ) -> GpuResult<VulkanBuffer<gpu::CollisionPair>> {
    let max_contacts = 10000; // Placeholder
    let output_list = self.allocate_device_buffer::<gpu::CollisionPair>(
      device,
      allocator,
      max_contacts,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
      true,
      rollback,
    )?;

    let pc = CcdPushConstants {
      particle_bvh: self.addresses.bvh_nodes,
      output_list: output_list.address, // self.addresses.ccd_candidates,
      root_index: 0,
      total_particles: 10000, // Should be passed dynamically
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<CcdPushConstants>(),
      )
    };

    unsafe {
      // Note: Pipeline is not there anymore. This method is marked for deletion
      // device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.ccd);
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );

      // Dispatch indirect using the potentials buffer
      device.cmd_dispatch_indirect(cmd.cmd, potentials.buffer, 0);

      // TODO synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    // TODO count?
    Ok(output_list)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn compact_collisions(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    globals: &VulkanBuffer<gpu::CollisionPair>,
    _time_delta: timeus_t,
  ) -> GpuResult<VulkanBuffer<gpu::CollisionPair>> {
    let total_elements = globals.capacity() as u32;
    let wg_size = self.effective_wg(128);
    let dispatch_groups = (total_elements + wg_size - 1) / wg_size;

    let max_packed = total_elements as usize; // Max possible is all valid
    let packed_out = self.allocate_device_buffer::<gpu::CollisionPair>(
      device,
      allocator,
      max_packed,
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::INDIRECT_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC,
      true,
      rollback,
    )?;

    let pc = StreamCompactPushConstants {
      sparse_in: globals.address,     // self.addresses.ccd_candidates,
      packed_out: packed_out.address, // self.addresses.packed_collisions,
      total_elements,
      _pad: 0,
      _pad_align16: [0; 2],
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<StreamCompactPushConstants>(),
      )
    };

    unsafe {
      device.cmd_fill_buffer(cmd.cmd, packed_out.buffer, 0, 16, 0);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );

      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.stream_compact,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      if dispatch_groups > 0 {
        device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::INDIRECT_COMMAND_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(packed_out)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn find_earliest_collision(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    compacted: &VulkanBuffer<gpu::CollisionPair>,
    dt: f32,
  ) -> GpuResult<VulkanBuffer<u32>> {
    let out_toi = self.allocate_device_buffer::<u32>(
      device,
      allocator,
      1,
      // fix: Added TRANSFER_SRC for CPU download
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC,
      false,
      rollback,
    )?;

    unsafe {
      device.cmd_fill_buffer(cmd.cmd, out_toi.buffer, 0, 4, 0xFFFFFFFF);

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    let pc = ReduceToiPushConstants {
      particles: self.addresses.particle_data,
      collisions: compacted.address, // self.addresses.packed_collisions,
      out_toi: out_toi.address,      // self.addresses.reduce_toi,
      particle_radius: 1.0,
      dt,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<ReduceToiPushConstants>(),
      )
    };

    unsafe {
      device.cmd_fill_buffer(cmd.cmd, out_toi.buffer, 0, 4, 0xFFFFFFFF);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );

      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.reduce_toi,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );

      device.cmd_dispatch_indirect(cmd.cmd, compacted.buffer, 0);

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::TRANSFER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(out_toi)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn apply_collision_responses(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    _kinematics: &VulkanBuffer<gpu::KinematicBody>,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &mut VulkanBuffer<f32>,
    collisions: &VulkanBuffer<gpu::CollisionPair>,
    _lca_entities_addr: u64,
    force_inelastic: bool,
  ) -> GpuResult<()> {
    let max_contacts = collisions.capacity() as usize;
    // TODO where are contact forces? How do you know to which body/particle they are applied to?
    // TODO how do you know the frame of reference of these forces (not only that, Macro or micro
    // frame?)
    let mut impulses_buffer = self.allocate_device_buffer::<[f32; 3]>(
      device,
      allocator,
      max_contacts,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      false,
      rollback,
    )?;

    let restitution_val = if force_inelastic { 0.0 } else { 0.5 };

    // LCP Solver
    let total_clusters = collisions.capacity() as u32;
    let pc_lcp = LcpPushConstants {
      particles: particles.address,     // self.addresses.particle_data,
      collisions: collisions.address,   // self.addresses.packed_collisions,
      outputs: impulses_buffer.address, // self.addresses.impulses,
      total_clusters,
      num_rigid_bodies: rigid_bodies.capacity() as u32,
      restitution: restitution_val,
      rigid_bodies: rigid_bodies.address,
      // dt for Baumgarte stabilization: use the standard physics frame dt.
      // The Baumgarte term is β/dt · max(depth - slop, 0). With dt=0.001,
      // the coefficient was 200x the depth, causing massive energy injection.
      // With dt=1/60, it's ~12x, which is the standard stabilization rate.
      dt: 1.0_f32 / 60.0_f32,
      lca_entities: _lca_entities_addr,
    };
    let bytes_lcp = unsafe {
      core::slice::from_raw_parts(
        &pc_lcp as *const _ as *const u8,
        core::mem::size_of::<LcpPushConstants>(),
      )
    };

    let pc_apply = ApplyImpulsesPushConstants {
      particles_addr: particles.address,
      collisions_addr: collisions.address,
      impulses_addr: impulses_buffer.address,
      rigid_bodies_addr: rigid_bodies.address,
      lca_entities: _lca_entities_addr,
      num_rigid_bodies: rigid_bodies.capacity() as u32,
      _pad: 0,
    };
    let bytes_apply = unsafe {
      core::slice::from_raw_parts(
        &pc_apply as *const _ as *const u8,
        core::mem::size_of::<ApplyImpulsesPushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.lcp_solver,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes_lcp,
      );
      device.cmd_dispatch_indirect(cmd.cmd, collisions.buffer, 0);

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    // Apply Impulses
    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.apply_impulses,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes_apply,
      );
      device.cmd_dispatch_indirect(cmd.cmd, collisions.buffer, 0);

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    // Deferred-free: GPU commands referencing `impulses_buffer` have been recorded,
    // so it is safe to release once the compute timeline reaches `next_submit_value`.
    self.recycle_transient_buffer(impulses_buffer, self.next_submit_value.load(core::sync::atomic::Ordering::Relaxed));

    Ok(())
  }

  #[cfg(any(test, feature = "collisions"))]
  fn snapshot_dynamics(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &VulkanBuffer<f32>,
  ) -> GpuResult<(VulkanBuffer<RigidBodyImex>, VulkanBuffer<f32>)> {
    let rb_snap = self.allocate_device_buffer::<RigidBodyImex>(
      device,
      allocator,
      rigid_bodies.capacity(),
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC,
      false,
      rollback,
    )?;
    let p_snap = self.allocate_device_buffer::<f32>(
      device,
      allocator,
      particles.capacity(),
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_DST
        | vk::BufferUsageFlags::TRANSFER_SRC,
      false,
      rollback,
    )?;

    let rb_copy = vk::BufferCopy::default()
      .size((rigid_bodies.capacity().max(1) * core::mem::size_of::<RigidBodyImex>()) as u64);
    let p_copy = vk::BufferCopy::default()
      .size((particles.capacity().max(1) * core::mem::size_of::<f32>()) as u64);

    unsafe {
      if rigid_bodies.capacity() > 0 {
        // device.cmd_copy_buffer(
        //   cmd.cmd,
        //   rigid_bodies.buffer,
        //   rb_snap.buffer,
        //   core::slice::from_ref(&rb_copy),
        // );
      }
      if particles.capacity() > 0 {
        // device.cmd_copy_buffer(
        //   cmd.cmd,
        //   particles.buffer,
        //   p_snap.buffer,
        //   core::slice::from_ref(&p_copy),
        // );
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok((rb_snap, p_snap))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn restore_dynamics(
    &self,
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    _rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &mut VulkanBuffer<RigidBodyImex>,
    particles: &mut VulkanBuffer<f32>,
    snapshot: &(VulkanBuffer<RigidBodyImex>, VulkanBuffer<f32>),
  ) -> GpuResult<()> {
    let rb_copy = vk::BufferCopy::default()
      .size((rigid_bodies.capacity().max(1) * core::mem::size_of::<RigidBodyImex>()) as u64);
    let p_copy = vk::BufferCopy::default()
      .size((particles.capacity().max(1) * core::mem::size_of::<f32>()) as u64);

    unsafe {
      if rigid_bodies.capacity() > 0 {
        // device.cmd_copy_buffer(
        //   cmd.cmd,
        //   snapshot.0.buffer,
        //   rigid_bodies.buffer,
        //   core::slice::from_ref(&rb_copy),
        // );
      }
      if particles.capacity() > 0 {
        // device.cmd_copy_buffer(
        //   cmd.cmd,
        //   snapshot.1.buffer,
        //   particles.buffer,
        //   core::slice::from_ref(&p_copy),
        // );
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(())
  }

  #[function_name::named]
  fn write_back_to_scene(
    &self,
    _device: &LogicalDevice,
    _discard_pool: &crate::gpu_backends::vulkan::device::resources::DiscardPool,
    _timeline_value: u64,
    _allocator: vk_mem::AllocatorView,
    _rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &VulkanBuffer<RigidBodyImex>,
    particles: &VulkanBuffer<f32>,
    particle_metadata: &[gpu::ParticleMetadata],
    _physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> GpuResult<Option<crate::gpu::CommandBufferSyncInfo>> {
    // All device buffers are allocated with HOST_VISIBLE + HOST_COHERENT + MAPPED.
    // The GPU has already been waited on (timeline semaphore) by the caller before this
    // function runs, so the mapped memory is safe to read directly. This avoids creating
    // staging buffers (VMA allocations) per frame, which corrupt Lavapipe's TLSF allocator
    // after many alloc/free cycles.
    let rb_data: alloc::vec::Vec<RigidBodyImex> =
      unsafe { rigid_bodies.mapped_slice().unwrap_or(&[]).to_vec() };
    let p_data: alloc::vec::Vec<f32> = unsafe { particles.mapped_slice().unwrap_or(&[]).to_vec() };

    // No GPU work needed; don't submit an empty command buffer.
    let _ = cmd; // cmd is unused but kept as parameter for API compatibility

    let sg = self.pipelines.subgroup_size as usize;
    let unpacked_particles =
      gpu::unpack_particles_aosoa(&p_data, sg, gpu::PARTICLE_FIELDS, particle_metadata.len());

    scene.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>( |entity, _transform, sys| {
      let mut sys_particles = sys.particles.write();
      for (i, p_gpu) in unpacked_particles.iter().enumerate() {
        let meta = &particle_metadata[i];
        if meta.entity_id == entity {
          if (meta.original_index as usize) < sys_particles.len() {
            sys_particles[meta.original_index as usize].position = [p_gpu[0], p_gpu[1], p_gpu[2]];
            sys_particles[meta.original_index as usize].velocity = [p_gpu[3], p_gpu[4], p_gpu[5]];
          }
        }
      }
    });

    let mut rb_idx = 0usize;
    scene.query3_mut::<crate::scene::TransformComponent, crate::scene::ColliderComponent, crate::scene::KinematicComponent, _>(
      |_entity,
       trans: &mut crate::scene::TransformComponent,
       _coll: &mut crate::scene::ColliderComponent,
       kin: &mut crate::scene::KinematicComponent| {
        if let Some(rb) = rb_data.get(rb_idx) {
          trans.position = Vec3f32::from_components(
            rb.position_mass[0],
            rb.position_mass[1],
            rb.position_mass[2],
          );
          trans.rotation = Quat::from_components(
            rb.orientation[0],
            rb.orientation[1],
            rb.orientation[2],
            rb.orientation[3],
          );
          kin.velocity = Vec3f32::from_components(
            rb.linear_vel_drag[0],
            rb.linear_vel_drag[1],
            rb.linear_vel_drag[2],
          );
          kin.angular_velocity = Vec3f32::from_components(
            rb.angular_vel_drag[0],
            rb.angular_vel_drag[1],
            rb.angular_vel_drag[2],
          );
          aethervk_oshal_rlib::log!("COMET POS_Y: {}", rb.position_mass[1]);
          rb_idx += 1;
        }
      },
    );

    Ok(None) // No GPU submission; GPU was already waited by caller.
  }
}

impl Kernels for Device {
  fn toggle_particle_self_gravity(&self, enable: bool) {
    self.kernels.particle_self_gravity_enabled.store(enable, core::sync::atomic::Ordering::Relaxed);
  }

  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd(
    &self,
    cmd: &mut Self::Cmd,
    broadphase_pairs: &Self::List<crate::gpu::CollisionPair>,
    rigid_bodies: &Self::Buffer<crate::gpu::RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    lca_entities: u64,
    space_type: u32,
    dt: f32,
    output_list: &Self::List<crate::gpu::CollisionPair>,
  ) -> EngineResult<()> {
    self.narrow_ccd(
      cmd,
      broadphase_pairs,
      rigid_bodies,
      particles,
      lca_entities,
      space_type,
      dt,
      output_list,
    )
  }

  #[cfg(any(test, feature = "collisions"))]
  fn narrow_ccd_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    broadphase_pairs: &Self::List<crate::gpu::CrossPair>,
    rigid_bodies: &Self::Buffer<crate::gpu::RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    lca_entities: u64,
    space_type: u32,
    dt: f32,
    output_list: &Self::List<crate::gpu::CollisionPair>,
  ) -> EngineResult<()> {
    self.narrow_ccd_cross_lca(
      cmd,
      broadphase_pairs,
      rigid_bodies,
      particles,
      lca_entities,
      space_type,
      dt,
      output_list,
    )
  }

  type Cmd = VulkanCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = VulkanBuffer<T>;
  type List<T: Copy + Send + Sync> = VulkanBuffer<T>;
  type MotionBvh = VulkanBuffer<()>;
  type MotionTlas = VulkanBuffer<()>;

  fn build_leaves(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::Buffer<[u32; 8]>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.allocate_device_buffer::<[u32; 8]>(
          &self.device,
          allocator,
          capacity,
          ash::vk::BufferUsageFlags::STORAGE_BUFFER,
          false, // Not a list, no header
          rollback,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn discard_buffer<T: Copy + Send + Sync>(&self, mut buffer: Self::Buffer<T>) {
    let timeline = self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
    buffer.discarded = true; // prevent drop warning
    self.kernels.transient_pool.lock().entries.push(TransientBufferEntry {
      buffer: buffer.buffer,
      address: buffer.address,
      capacity: buffer.capacity,
      allocation: buffer.allocation,
      item_size: core::mem::size_of::<T>(),
      is_list: buffer.is_list,
      timeline_freed: timeline,
      usage: buffer.usage,
    });
  }

  fn discard_list<T: Copy + Send + Sync>(&self, mut list: Self::List<T>) {
    let timeline = self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
    list.discarded = true;
    self.kernels.transient_pool.lock().entries.push(TransientBufferEntry {
      buffer: list.buffer,
      address: list.address,
      capacity: list.capacity,
      allocation: list.allocation,
      item_size: core::mem::size_of::<T>(),
      is_list: list.is_list,
      timeline_freed: timeline,
      usage: list.usage,
    });
  }

  fn discard_bvh(&self, mut bvh: Self::MotionBvh) {
    let timeline = self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
    bvh.discarded = true;
    self.kernels.transient_pool.lock().entries.push(TransientBufferEntry {
      buffer: bvh.buffer,
      address: bvh.address,
      capacity: bvh.capacity,
      allocation: bvh.allocation,
      item_size: 0,
      is_list: bvh.is_list,
      timeline_freed: timeline,
      usage: bvh.usage,
    });
  }

  fn discard_tlas(&self, mut tlas: Self::MotionTlas) {
    let timeline = self.kernels.next_submit_value.load(core::sync::atomic::Ordering::Relaxed);
    tlas.discarded = true;
    self.kernels.transient_pool.lock().entries.push(TransientBufferEntry {
      buffer: tlas.buffer,
      address: tlas.address,
      capacity: tlas.capacity,
      allocation: tlas.allocation,
      item_size: 0,
      is_list: tlas.is_list,
      timeline_freed: timeline,
      usage: tlas.usage,
    });
  }

  fn subgroup_size(&self) -> Option<crate::gpu::SubgroupSize> {
    use crate::gpu::SubgroupSize;
    Some(match self.query_result.subgroup_size {
      s if s >= 128 => SubgroupSize::Size128,
      s if s >= 64 => SubgroupSize::Size64,
      s if s >= 32 => SubgroupSize::Size32,
      s if s >= 16 => SubgroupSize::Size16,
      s if s >= 8 => SubgroupSize::Size8,
      _ => SubgroupSize::Size4,
    })
  }

  fn wait_idle(&self) -> EngineResult<()> {
    unsafe { self.device.device_wait_idle() }.map_err(|e| {
      crate::types::EngineError::Gpu(crate::types::GpuError::BackendSpecific(alloc::format!(
        "{:?}", e
      )))
    })?;
    Ok(())
  }

  fn is_cpu_device(&self) -> bool {
    crate::gpu::RenderDevice::is_cpu_device(self)
  }

  fn wait_sync(&self, sync: &crate::gpu::CommandBufferSyncInfo) -> EngineResult<()> {
    use ash::vk::Handle;
    let sem = ash::vk::Semaphore::from_raw(sync.timeline_semaphore);
    self
      .device
      .wait_for_semaphore_value(sem, sync.timeline_value, u64::MAX)
      .map_err(|e| {
        crate::types::EngineError::Gpu(crate::types::GpuError::BackendSpecific(alloc::format!(
          "{:?}", e
        )))
      })?;

    // WORKAROUND: Lavapipe's timeline semaphores return early. Wait idle to ensure JIT threads finish.
    if self.is_cpu_device() {
      let _ = unsafe { self.device.device_wait_idle() };
    }

    // The integration tests do not advance the frame manager, so we must clean up the DiscardPool manually here to avoid exhausting memory/resources.
    let items = self.kernels.discard_pool.pop_ready_items(sync.timeline_value);
    crate::gpu_backends::vulkan::device::resources::DiscardPool::destroy_items_lock_free(
      &self.device.handle,
      items,
    );

    Ok(())
  }

  fn upload_motion_tlas(
    &self,
    _cmd: &mut Self::Cmd,
    node_bytes: &[u8],
  ) -> EngineResult<Self::MotionTlas> {
    use crate::physics::tlas_builder::PARTICLE_BLAS_SENTINEL;
    use core::sync::atomic::Ordering;

    if node_bytes.is_empty() {
      // No entities: upload a single zeroed node so BDA is valid but TLAS is empty.
      // Must use the same node size the GPU shader expects (SUBGROUP_SIZE-dependent).
      use crate::math::collision::multi_bvh::TlasMultiNode;
      let node_size = match self.kernels.pipelines.subgroup_size {
        128 => core::mem::size_of::<TlasMultiNode<128>>(),
        64 => core::mem::size_of::<TlasMultiNode<64>>(),
        32 => core::mem::size_of::<TlasMultiNode<32>>(),
        16 => core::mem::size_of::<TlasMultiNode<16>>(),
        8 => core::mem::size_of::<TlasMultiNode<8>>(),
        4 => core::mem::size_of::<TlasMultiNode<4>>(),
        _ => core::mem::size_of::<TlasMultiNode<32>>(),
      };
      let zero = alloc::vec![0u8; node_size];
      return self.upload_motion_tlas(_cmd, &zero);
    }

    // ── Allocate a device-visible (HOST_VISIBLE | HOST_COHERENT) mapped buffer.
    // Using AutoPreferDevice + HOST_ACCESS_SEQUENTIAL_WRITE so vma gives us
    // a persistently-mapped ReBAR or host-coherent buffer — same as every other
    // build_* helper in this file.
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        let size = node_bytes.len().max(1) as u64;
        let current_timeline = self
          .kernels
          .next_submit_value
          .load(core::sync::atomic::Ordering::Relaxed)
          .saturating_sub(1);
        let mut pool = self.kernels.transient_pool.lock();
        for i in 0..pool.entries.len() {
          let entry = &pool.entries[i];
          if entry.item_size == 0
            && (entry.capacity as u64) * 4 >= size
            && !entry.is_list
            && entry.timeline_freed <= current_timeline
          {
            let entry = pool.entries.remove(i);
            drop(pool);

            // Write the new bytes to the recycled buffer
            let info = allocator.get_allocation_info(&entry.allocation);
            let mapped = info.mapped_data as *mut u8;
            assert!(!mapped.is_null(), "TLAS buffer not persistently mapped");
            unsafe {
              core::ptr::copy_nonoverlapping(node_bytes.as_ptr(), mapped, node_bytes.len());
            }

            return Ok(VulkanBuffer::<()> {
              buffer: entry.buffer,
              address: entry.address,
              capacity: entry.capacity,
              allocation: entry.allocation,
              allocator,
              is_list: false,
              usage: entry.usage,
              discarded: false,
              _marker: core::marker::PhantomData,
            });
          }
        }
        drop(pool);

        let sharing_mode =
          if self.kernels.queue_sharing_info.mode == crate::gpu::SharingMode::Concurrent {
            ash::vk::SharingMode::CONCURRENT
          } else {
            ash::vk::SharingMode::EXCLUSIVE
          };
        let mut buf_info = ash::vk::BufferCreateInfo::default()
          .size(size)
          .usage(
            ash::vk::BufferUsageFlags::STORAGE_BUFFER
              | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
          )
          .sharing_mode(sharing_mode);
        if sharing_mode == ash::vk::SharingMode::CONCURRENT {
          buf_info =
            buf_info.queue_family_indices(&self.kernels.queue_sharing_info.queue_family_indices);
        }
        let alloc_info = vk_mem::AllocationCreateInfo {
          usage: vk_mem::MemoryUsage::AutoPreferDevice,
          flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
            | vk_mem::AllocationCreateFlags::MAPPED,
          required_flags: ash::vk::MemoryPropertyFlags::HOST_VISIBLE
            | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
          ..Default::default()
        };
        crate::apply_test_dedicated_alloc!(alloc_info);
        let (buffer, mut alloc, info) =
          unsafe { allocator.create_buffer_get_info(&buf_info, &alloc_info) }?;
        aethervk_oshal_rlib::log!("physics alloc: {:?}", alloc.get_raw());
        #[cfg(test)]
        {
          use ash::vk::Handle;
          self
            .kernels
            .tracked_physical_allocations
            .lock()
            .push(info.device_memory.as_raw());
        }
        aethervk_oshal_rlib::log!("upload_motion_tlas alloc: {:?}", alloc.get_raw());
        #[cfg(test)]
        {
          let name = alloc::format!("motion_tlas size={}\0", size);
          unsafe {
            allocator.set_allocation_name(
              &mut alloc,
              core::ffi::CStr::from_ptr(name.as_ptr() as *const _).as_ptr(),
            );
          }
        }
        rollback.defer(move |_| unsafe { allocator.destroy_buffer(buffer, &mut alloc) });

        // ── Write TLAS node bytes into the mapped region.
        let mapped = info.mapped_data as *mut u8;
        assert!(!mapped.is_null(), "TLAS buffer not persistently mapped");
        unsafe {
          core::ptr::copy_nonoverlapping(node_bytes.as_ptr(), mapped, node_bytes.len());
        }

        // Sentinel patching is no longer needed because shaders extract the EntityID
        // from the leaf metadata and fetch the 64-bit BDA directly from the EntityArray.

        let addr_info = ash::vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address =
          unsafe { self.device.buffer_device_address.get_buffer_device_address(&addr_info) };

        Ok::<_, GpuError>(VulkanBuffer::<()> {
          buffer,
          address,
          capacity: (size / 4).max(1) as usize,
          allocation: alloc,
          allocator,
          is_list: false,
          usage: ash::vk::BufferUsageFlags::STORAGE_BUFFER
            | ash::vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
          discarded: false,
          _marker: core::marker::PhantomData,
        })
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    utils::NestedVulkanTransaction::new(
      &*self.res,
      &self.device,
      |res: &device::DeviceResources| &res.command_pools,
    )
    .prepare_read(
      self.get_compute_queue(),
      |res_guard, command_pools, compute_queue| {
        let opt_opt = command_pools.get(compute_queue.family_index as usize);
        let opt_pos: Option<&alloc::sync::Arc<commands::CommandPools>> = opt_opt.unwrap().as_ref();
        let command_pool_arc = alloc::sync::Arc::clone(opt_pos.ok_or(GpuError::NotFound)?);
        Ok::<_, GpuError>((
          command_pool_arc,
          res_guard.allocator.allocator.as_allocator_view(),
        ))
      },
    )?
    .execute(|(command_pool_arc, allocator), rollback| {
      self.kernels.create_command_buffer(
        &self.device,
        allocator,
        command_pool_arc,
        rollback,
        self.get_compute_queue(),
      )
    })
    .commit_read(|_res_guard, _command_pools, result| result)
    .map_err(EngineError::from)
  }

  #[cfg(feature = "shader_debug_sync")]
  fn debug_sync_barrier(&self, mut cmd: Self::Cmd) -> EngineResult<Self::Cmd> {
    unsafe {
      let device = cmd.device_ptr.as_ref();
      let next_submit_value = cmd.next_submit_value_ptr.as_ref();

      device
        .end_command_buffer(cmd.cmd)
        .map_err(|e| EngineError::from(GpuError::from(e)))?;

      let mut type_info = vk::SemaphoreTypeCreateInfo::default()
        .semaphore_type(vk::SemaphoreType::TIMELINE)
        .initial_value(0);
      let sem_ci = vk::SemaphoreCreateInfo::default().push_next(&mut type_info);
      cmd.throwaway_sem = device
        .handle
        .create_semaphore(&sem_ci, None)
        .map_err(|e| EngineError::from(GpuError::from(e)))?;

      let command_buffers = [cmd.cmd];
      let signal_semaphores = [cmd.timeline_sem, cmd.throwaway_sem];

      let _guard = device.submission_lock.lock();
      cmd.timeline_value = next_submit_value.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
      let signal_values = [cmd.timeline_value, 1];

      let mut timeline_info =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

      let submit_info = vk::SubmitInfo::default()
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores)
        .push_next(&mut timeline_info);

      aethervk_oshal_rlib::log!("debug_sync_barrier: before queue_submit");
      device
        .handle
        .queue_submit(cmd.queue.handle, &[submit_info], vk::Fence::null())
        .map_err(|e| EngineError::from(GpuError::from(e)))?;
      drop(_guard);

      cmd
        .assigned_timeline_value
        .store(cmd.timeline_value, core::sync::atomic::Ordering::Release);

      aethervk_oshal_rlib::log!("debug_sync_barrier: before wait_for_semaphore_value");
      self
        .device
        .wait_for_semaphore_value(cmd.throwaway_sem, 1, 60_000_000_000) // 60s timeout
        .map_err(|e| {
          EngineError::Gpu(crate::gpu_err!(
            "debug_sync_barrier throwaway sem wait failed: {:?}",
            e
          ))
        })?;
      aethervk_oshal_rlib::log!("debug_sync_barrier: after wait_for_semaphore_value");

      // Manually clean up since we waited synchronously!
      device.handle.destroy_semaphore(cmd.throwaway_sem, None);
      let _ = cmd.command_pools.recycle(cmd.tid, cmd.id, cmd.cmd);
      cmd.throwaway_sem = vk::Semaphore::null();
    }

    // Allocate a fresh command buffer for subsequent dispatches
    self.create_command_buffer()
  }

  #[cfg(feature = "shader_debug_sync")]
  fn check_corruption(&self, label: &str) -> EngineResult<()> {
    let allocator = utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, _rollback| Ok(allocator))
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)?;

    let result = unsafe {
      allocator.check_corruption(
        ash::vk::MemoryPropertyFlags::HOST_VISIBLE | ash::vk::MemoryPropertyFlags::HOST_COHERENT,
      )
    };
    match result {
      Ok(()) | Err(ash::vk::Result::ERROR_FEATURE_NOT_PRESENT) => Ok(()),
      Err(e) => {
        aethervk_oshal_rlib::log!(
          "[CORRUPTION-CHECK] *** CORRUPTION DETECTED after '{}': {:?} ***",
          label,
          e
        );
        Err(EngineError::Gpu(crate::gpu_err!(
          "VMA corruption detected after '{}'",
          label
        )))
      }
    }
  }

  fn build_kinematic_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<gpu::KinematicBody>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self
          .kernels
          .build_kinematic_bodies(&self.device, allocator, rollback, cmd, scene, scene0)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>, u32)> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self
          .kernels
          .build_rigid_bodies_imex(&self.device, allocator, rollback, cmd, scene, scene0)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_particles(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<(Self::Buffer<f32>, alloc::vec::Vec<gpu::ParticleMetadata>)> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_particles(&self.device, allocator, rollback, cmd, scene)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_particle_frame_ids(
    &self,
    cmd: &mut Self::Cmd,
    particle_metadata: &[gpu::ParticleMetadata],
  ) -> EngineResult<Self::Buffer<u32>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_particle_frame_ids(
          &self.device,
          allocator,
          rollback,
          cmd,
          particle_metadata,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_emitters(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<(Self::Buffer<gpu::ForceEmitter>, u32)> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_emitters(&self.device, allocator, rollback, cmd, scene)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_emission_candidates(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<f32>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self
          .kernels
          .build_emission_candidates(&self.device, allocator, rollback, cmd, scene)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn emit_particles(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    physical_scene: &PhysicsScene,
    scene: &Scene,
    sun_pos: Vec3f32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.emit_particles(
          &self.device,
          allocator,
          rollback,
          cmd,
          particles,
          physical_scene,
          scene,
          sun_pos,
          dt,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn step_ode_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self
          .kernels
          .step_ode_p1_p2(&self.device, allocator, rollback, cmd, particles, dt)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn step_ode_p3_p4(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &mut Self::Buffer<gpu::RigidBodyGpu>,
    emitters: &Self::Buffer<gpu::ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.step_ode_p3_p4(
          &self.device,
          allocator,
          rollback,
          cmd,
          kinematics,
          rigid_bodies,
          emitters,
          dt,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn compute_self_gravity(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    particles: &mut Self::Buffer<f32>,
  ) -> EngineResult<()> {
    if !self.kernels.particle_self_gravity_enabled.load(core::sync::atomic::Ordering::Relaxed) {
      return Ok(());
    }

    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self
          .kernels
          .compute_self_gravity(&self.device, allocator, rollback, cmd, bvh, particles)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn step_ode_p5(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    particles: &mut Self::Buffer<f32>,
    emitters: &Self::Buffer<gpu::ForceEmitter>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.step_ode_p5(
          &self.device,
          allocator,
          rollback,
          cmd,
          kinematics,
          particles,
          emitters,
          dt,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_motion_bvh(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    particle_frame_ids: &mut Self::Buffer<u32>,
    dt: timeus_t,
  ) -> EngineResult<Self::MotionBvh> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_motion_bvh(
          &self.device,
          allocator,
          rollback,
          cmd,
          kinematics,
          rigid_bodies,
          particles,
          particle_frame_ids,
          dt,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn refit_motion_blas(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    depth_indices: &Self::Buffer<u32>,
    total_nodes: u32,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.refit_motion_blas(
          &self.device,
          cmd,
          bvh.address,
          depth_indices.address,
          total_nodes,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn self_intersect_scene(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
  ) -> EngineResult<Self::List<gpu::CollisionPair>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.self_intersect_scene(&self.device, allocator, rollback, cmd, bvh)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<gpu::CollisionPair>,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &Self::Buffer<gpu::RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<Self::List<gpu::CollisionPair>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.intersect_instances(
          &self.device,
          allocator,
          rollback,
          cmd,
          potentials as &VulkanBuffer<_>,
          kinematics as &VulkanBuffer<_>,
          rigid_bodies as &VulkanBuffer<_>,
          particles as &VulkanBuffer<_>,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn compact_collisions(
    &self,
    cmd: &mut Self::Cmd,
    globals: &Self::List<gpu::CollisionPair>,
    time_delta: timeus_t,
  ) -> EngineResult<Self::List<gpu::CollisionPair>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self
          .kernels
          .compact_collisions(&self.device, allocator, rollback, cmd, globals, time_delta)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<gpu::CollisionPair>,
    dt: f32,
  ) -> EngineResult<Self::Buffer<u32>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self
          .kernels
          .find_earliest_collision(&self.device, allocator, rollback, cmd, compacted, dt)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn read_buffer_u32_first(&self, buf: &Self::Buffer<u32>) -> EngineResult<u32> {
    // All device buffers are HOST_VISIBLE + HOST_COHERENT + MAPPED.
    // After the GPU wait, reading from the mapped pointer is safe and avoids
    // creating a staging buffer allocation (which corrupts Lavapipe's TLSF).
    Ok(unsafe { buf.mapped_slice().and_then(|s| s.first().copied()).unwrap_or(0xFFFF_FFFF) })
  }

  #[cfg(any(test, feature = "collisions"))]
  fn apply_collision_responses(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    collisions: &Self::List<gpu::CollisionPair>,
    lca_entities_addr: u64,
    force_inelastic: bool,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.apply_collision_responses(
          &self.device,
          allocator,
          rollback,
          cmd,
          kinematics,
          rigid_bodies,
          particles,
          collisions,
          lca_entities_addr,
          force_inelastic,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>)> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.snapshot_dynamics(
          &self.device,
          allocator,
          rollback,
          cmd,
          rigid_bodies,
          particles,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  #[cfg(any(test, feature = "collisions"))]
  fn restore_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    snapshot: &(Self::Buffer<RigidBodyImex>, Self::Buffer<f32>),
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.restore_dynamics(
          &self.device,
          allocator,
          rollback,
          cmd,
          rigid_bodies,
          particles,
          snapshot,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn write_back_to_scene(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<RigidBodyImex>,
    particles: &Self::Buffer<f32>,
    particle_metadata: &[gpu::ParticleMetadata],
    physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<Option<crate::gpu::CommandBufferSyncInfo>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        let timeline_value = self
          .kernels
          .next_submit_value
          .fetch_add(1, core::sync::atomic::Ordering::SeqCst);
        self.kernels.write_back_to_scene(
          &self.device,
          &self.kernels.discard_pool,
          timeline_value,
          allocator,
          rollback,
          cmd,
          rigid_bodies,
          particles,
          particle_metadata,
          physical_scene,
          scene,
        )
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_frames(
    &self,
    _cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
  ) -> EngineResult<Self::Buffer<GpuReferenceFrame>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_frames(&self.device, allocator, rollback, scene)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  fn build_list<T: Copy + Send + Sync>(
    &self,
    cmd: &mut Self::Cmd,
    capacity: usize,
  ) -> EngineResult<Self::List<T>> {
    let list = utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_list_inner::<T>(&self.device, allocator, rollback, capacity)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)?;

    unsafe {
      // Zero the entire 16-byte list header (count=0, capacity=0, and first 8 payload bytes).
      self.device.cmd_fill_buffer(cmd.cmd, list.buffer, 0, 16, 0);
      // Write the element capacity at offset 4.  Every GLSL list type used here has the layout
      //   { uint count; uint capacity; T pairs[]; }
      // so bytes 4-7 are the capacity field that narrow_ccd{,_cross_lca} use as an upper bound
      // before atomically claiming a slot.  Without this write the field stays zero (from the
      // cmd_fill_buffer above) and every bounds-check `count < capacity` fails, silently
      // discarding all collision pairs.
      self.device.cmd_fill_buffer(cmd.cmd, list.buffer, 4, 4, capacity as u32);
      let barrier = ash::vk::MemoryBarrier::default()
        .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::SHADER_WRITE);
      self.device.cmd_pipeline_barrier(
        cmd.cmd,
        ash::vk::PipelineStageFlags::TRANSFER,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        ash::vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
    Ok(list)
  }

  fn imex_integrate_particles_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.imex_integrate_particles_p1_p2(
          &self.device,
          cmd,
          particles.address,
          {
            let sg = self.kernels.pipelines.subgroup_size;
            let stride = gpu::PARTICLE_FIELDS as u32 * sg;
            (particles.capacity() as u32 / stride) * sg
          },
          dt,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  fn imex_integrate_bodies_p3(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &mut Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
    emitters: &Self::Buffer<crate::gpu::ForceEmitter>,
    frames: &Self::Buffer<crate::physics::physics_scene::GpuReferenceFrame>,
    n_bodies: u32,
    n_emitters: u32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.imex_integrate_bodies_p3(
          &self.device,
          cmd,
          bodies.address,
          wrenches.address,
          emitters.address,
          frames.address,
          n_bodies,
          n_emitters,
          dt,
          4,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  fn imex_rb_force_assign(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    wrenches: &mut Self::Buffer<Wrench>,
    n_bodies: u32,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.imex_rb_force_assign(
          &self.device,
          cmd,
          bodies.address,
          wrenches.address,
          n_bodies,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  fn imex_integrate_particles_p4_5(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    dt: timeus_t,
    current_time_us: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        // clock_addr = 0 — no clock buffer in this integration pass
        self.kernels.imex_integrate_particles_p4_5(
          &self.device,
          cmd,
          particles.address,
          0u64,
          {
            let sg = self.kernels.pipelines.subgroup_size;
            let stride = gpu::PARTICLE_FIELDS as u32 * sg;
            (particles.capacity() as u32 / stride) * sg
          },
          dt,
          current_time_us,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  fn apply_emitters_to_particles(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<f32>,
    emitters: &Self::Buffer<crate::gpu::ForceEmitter>,
    frames: &Self::Buffer<crate::physics::physics_scene::GpuReferenceFrame>,
    particle_frame_ids: &Self::Buffer<u32>,
    num_emitters: u32,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        let total_particles = particle_frame_ids.capacity() as u32;
        self.kernels.apply_emitters_to_particles(
          &self.device,
          cmd,
          particles.address,
          emitters.address,
          frames.address,
          particle_frame_ids.address,
          num_emitters,
          total_particles,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn bp_clear(
    &self,
    cmd: &mut Self::Cmd,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_rb_lca_addr: u64,
    internal_pairs_addr: u64,
    out_sparse_addr: u64,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.bp_clear(
          &self.device,
          cmd,
          raw_pairs_addr,
          out_rb_rb_addr,
          out_rb_ps_addr,
          out_rb_lca_addr,
          internal_pairs_addr,
          out_sparse_addr,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn bp_bounds_gen(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    leaves_addr: u64,
    lca_entities_addr: u64,
    total_entities: u32,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.bp_bounds_gen(
          &self.device,
          cmd,
          bodies.address,
          leaves_addr,
          lca_entities_addr,
          total_entities,
          dt,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn bp_scene(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    query_leaves_addr: u64,
    overlapping_pairs_addr: u64,
    tlas_root_index: u32,
    total_queries: u32,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.bp_scene(
          &self.device,
          cmd,
          tlas_bvh_addr,
          query_leaves_addr,
          overlapping_pairs_addr,
          tlas_root_index,
          total_queries,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn bp_classify(
    &self,
    cmd: &mut Self::Cmd,
    bodies: &Self::Buffer<RigidBodyImex>,
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_macro_lca_addr: u64,
    out_lca_lca_addr: u64,
    total_raw_pairs: u32,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.bp_classify(
          &self.device,
          cmd,
          bodies.address,
          raw_pairs_addr,
          out_rb_rb_addr,
          out_rb_ps_addr,
          out_ps_ps_addr,
          out_macro_lca_addr,
          out_lca_lca_addr,
          total_raw_pairs,
          bodies.capacity() as u32, // Extract rigid body count
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn bp_cross_lca(
    &self,
    cmd: &mut Self::Cmd,
    tlas_bvh_addr: u64,
    lca_entities_addr: u64,
    macro_leaves_addr: u64,
    entity_headers_addr: u64,
    lca_query_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    out_cross_pairs_addr: u64,
    total_queries: u32,
    max_pairs: u32,
    num_rigid_bodies: u32,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.bp_cross_lca(
          &self.device,
          cmd,
          tlas_bvh_addr,
          lca_entities_addr,
          macro_leaves_addr,
          entity_headers_addr,
          lca_query_pairs_addr,
          out_rb_rb_addr,
          out_rb_ps_addr,
          out_ps_ps_addr,
          out_cross_pairs_addr,
          total_queries,
          max_pairs,
          num_rigid_bodies,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }

  #[cfg(any(test, feature = "collisions"))]
  fn bp_particle_self(
    &self,
    cmd: &mut Self::Cmd,
    bvh_addr: u64,
    particles: &mut Self::Buffer<f32>,
    wrench_buffer_addr: u64,
    total_particles: u32,
    root_index: u32,
    particle_radius: f32,
    stiffness: f32,
  ) -> EngineResult<()> {
    if !self.kernels.particle_self_gravity_enabled.load(core::sync::atomic::Ordering::Relaxed) {
      return Ok(());
    }

    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.bp_particle_self(
          &self.device,
          cmd,
          bvh_addr,
          particles.address,
          wrench_buffer_addr,
          total_particles,
          root_index,
          particle_radius,
          stiffness,
        );
        Ok(())
      })
      .commit_read(|_res_guard, result| result)
      .map_err(EngineError::from)
  }
}

impl Device {
  #[cfg(any(test, feature = "collisions"))]
  pub fn narrow_ccd_cross_lca(
    &self,
    cmd: &mut VulkanCommandBuffer,
    broadphase_pairs: &VulkanBuffer<crate::gpu::CrossPair>,
    rigid_bodies: &VulkanBuffer<crate::gpu::RigidBodyImex>,
    particles: &VulkanBuffer<f32>,
    lca_entities: u64,
    space_type: u32,
    dt: f32,
    output_list: &VulkanBuffer<crate::gpu::CollisionPair>,
  ) -> crate::types::EngineResult<()> {
    let pc = NarrowCcdCrossLcaPushConstants {
      scene_entities: rigid_bodies.address,
      cross_output_list: output_list.address,
      particles: particles.address,
      cross_pair_buffer: broadphase_pairs.address,
      dt,
      particle_radius: 0.5,
      lca_entities,
      space_type,
      _pad: 0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };

    let dispatch_groups = (broadphase_pairs.capacity as u32 + self.kernels.effective_wg(128) - 1)
      / self.kernels.effective_wg(128);

    unsafe {
      self.device.cmd_bind_pipeline(
        cmd.cmd,
        ash::vk::PipelineBindPoint::COMPUTE,
        self.kernels.pipelines.narrow_ccd_cross_lca,
      );

      self.device.cmd_push_constants(
        cmd.cmd,
        self.kernels.pipelines.pipeline_layout,
        ash::vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );

      if dispatch_groups > 0 {
        self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
      }

      // Barrier: ensure narrow_ccd_cross_lca writes to sparse_collisions are
      // visible to the subsequent compact_collisions (stream_compact) dispatch.
      let barrier = ash::vk::MemoryBarrier::default()
        .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(ash::vk::AccessFlags::SHADER_READ);
      self.device.cmd_pipeline_barrier(
        cmd.cmd,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
        ash::vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
    Ok(())
  }
  #[cfg(any(test, feature = "collisions"))]
  pub fn narrow_ccd(
    &self,
    cmd: &mut VulkanCommandBuffer,
    broadphase_pairs: &VulkanBuffer<crate::gpu::CollisionPair>,
    rigid_bodies: &VulkanBuffer<crate::gpu::RigidBodyImex>,
    particles: &VulkanBuffer<f32>,
    lca_entities: u64,
    space_type: u32,
    dt: f32,
    output_list: &VulkanBuffer<crate::gpu::CollisionPair>,
  ) -> crate::types::EngineResult<()> {
    let pc = NarrowCcdPushConstants {
      scene_entities: rigid_bodies.address,
      output_list: output_list.address,
      particles: particles.address,
      pair_buffer: broadphase_pairs.address,
      dt,
      particle_radius: 0.5,
      lca_entities,
      space_type,
      _pad: 0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of_val(&pc))
    };

    let dispatch_groups = (broadphase_pairs.capacity as u32 + self.kernels.effective_wg(128) - 1)
      / self.kernels.effective_wg(128);

    unsafe {
      let pipeline = if space_type == 1 {
        self.kernels.pipelines.narrow_ccd_cross_lca
      } else {
        self.kernels.pipelines.narrow_ccd
      };

      self
        .device
        .cmd_bind_pipeline(cmd.cmd, ash::vk::PipelineBindPoint::COMPUTE, pipeline);
      self.device.cmd_push_constants(
        cmd.cmd,
        self.kernels.pipelines.pipeline_layout,
        ash::vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      self.device.cmd_dispatch(cmd.cmd, dispatch_groups.max(1), 1, 1);

      let barrier = ash::vk::MemoryBarrier::default()
        .src_access_mask(ash::vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(ash::vk::AccessFlags::SHADER_READ);
      self.device.cmd_pipeline_barrier(
        cmd.cmd,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::DRAW_INDIRECT,
        ash::vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );

      // TODO: Particle–particle narrow CCD.
      //
      // This dispatch is intentionally disabled until the push-constant layout is
      // corrected.  Two bugs block it:
      //
      // 1. `NarrowCcdParticlesPushConstants` is 40 bytes but the `narrow_ccd.comp`
      //    pipeline declares a 64-byte push-constant range (`NarrowCcdPushConstants`).
      //    Pushing 40 bytes leaves `lca_entities` / `frame_bda` / `space_type` (the
      //    upper 24 bytes) with **stale / garbage values** from the previous dispatch.
      //    On MoltenVK/macOS those garbage BDAs happen to be harmless; on Lavapipe
      //    with GPU-Assisted Validation enabled they cause an immediate SIGSEGV.
      //
      // 2. The `particles` buffer passed here is the IMEX rigid-body float buffer
      //    (AoSoA stride = PARTICLE_FIELDS × subgroup_size = 11 × sg ≠ 32), so
      //    `capacity / 32` yields a wrong particle count on every subgroup size.
      //    A dedicated particle-system BDA should be passed instead.
      //
      // Once both issues are resolved, re-enable this block with the correct
      // `NarrowCcdPushConstants` (or a new 64-byte particle-CCD struct) and the
      // proper particle-system buffer.
      let _ = NarrowCcdParticlesPushConstants {
        scene_entities: 0,
        output_list: 0,
        particles: 0,
        num_rigid_bodies: 0,
        num_particles: 0,
        dt: 0.0,
        particle_radius: 0.0,
      }; // keep the struct reachable to avoid dead-code warnings during transition
    }

    Ok(())
  }
}

#[cfg(test)]
#[path = "mock_physics_tests.rs"]
mod physics_tests;