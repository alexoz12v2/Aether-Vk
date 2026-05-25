//! Vulkan Backend Integration for the IMEX / LCP Physics Engine
//!
//! This module scaffolds the execution of the massive compute-shader pipeline.
//! It assumes Vulkan 1.1 with `VK_KHR_buffer_device_address` and `VK_KHR_shader_subgroup_basic`.

use crate::gpu::compute_push_constants::ApplyImpulsesPushConstants;
use crate::gpu::compute_push_constants::{RigidBodyImex, Wrench};
use crate::gpu::vulkan::device::{self, Device, LogicalDevice, commands, resources};
use crate::gpu::vulkan::utils;
use crate::gpu::{
  self, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, Kernels, KinematicBody, WaitHandle,
};
use crate::gpu_err;
use crate::physics::physics_scene::GpuReferenceFrame;
use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;
use crate::types::{EngineError, EngineResult, GpuError, GpuResult};
use aethervk_oshal_rlib::math::matrix::Matrix4;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::Quat;
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4};
use aethervk_oshal_rlib::os::time::timeus_t;
use alloc::format;
use alloc::vec::Vec;
use ash::vk;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NarrowCcdPushConstants {
  pub scene_entities: u64,
  pub particles: u64,
  pub output_list: u64,
  pub pair_buffer: u64,
  pub dt: f32,
  pub particle_radius: f32,
  pub lca_entities: u64,
  pub space_type: u32,
  pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NarrowCcdCrossLcaPushConstants {
  pub scene_entities: u64,
  pub particles: u64,
  pub cross_output_list: u64,
  pub cross_pair_buffer: u64,
  pub dt: f32,
  pub particle_radius: f32,
  pub lca_entities: u64,
  pub space_type: u32,
  pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct NarrowCcdParticlesPushConstants {
  pub scene_entities: u64,
  pub output_list: u64,
  pub particles: u64,
  pub num_rigid_bodies: u32,
  pub num_particles: u32,
  pub dt: f32,
  pub particle_radius: f32,
}

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

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct P12PushConstants {
  pub particles: u64,
  pub dt: f32,
  pub total_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct LbvhPushConstants {
  pub bvh: u64,
  pub sorted_morton: u64,
  pub counters: u64,
  pub particles: u64,
  pub num_primitives: u32,
  pub particle_radius: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct CcdPushConstants {
  pub particle_bvh: u64,
  pub output_list: u64,
  pub root_index: u32,
  pub total_particles: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct StreamCompactPushConstants {
  pub sparse_in: u64,
  pub packed_out: u64,
  pub total_elements: u32,
  pub _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct ReduceToiPushConstants {
  pub particles: u64,
  pub collisions: u64,
  pub out_toi: u64,
  pub particle_radius: f32,
  pub dt: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct LcpPushConstants {
  pub particles: u64,
  pub collisions: u64,
  pub outputs: u64,
  pub total_clusters: u32,
  pub rigid_bodies: u64,
  pub dt: f32,
  pub restitution: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct BarnesHutPushConstants {
  pub particles: u64,
  pub bvh: u64,
  pub root_index: u32,
  pub total_particles: u32,
  pub theta: f32,
  pub g: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// TODO: Document this item
pub struct P5PushConstants {
  pub particles: u64,
  pub emitters: u64,
  pub kinematics: u64,
  pub dt: f32,
  pub total_particles: u32,
  pub num_emitters: u32,
  pub num_kinematics: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct P34PushConstants {
  pub rigid_bodies: u64,
  pub emitters: u64,
  pub kinematics: u64,
  pub dt: f32,
  pub total_rigid_bodies: u32,
  pub num_emitters: u32,
  pub num_kinematics: u32,
}

// ─────────────────────────────────────────────────────────────────────────────
// IMEX Pipeline Push-Constant Structs
// All sizes verified against shader `layout(push_constant, scalar)` blocks.
// ─────────────────────────────────────────────────────────────────────────────

/// `integrate_particles_p1_p2.comp` — 16 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImexParticlesP12PushConstants {
  /// BDA to AOSOA particle data (float[])
  pub particles: u64,
  /// Physical dt in seconds
  pub dt: f32,
  pub total_particles: u32,
}

/// `integrate_bodies_p3.comp` — 32 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImexBodiesP3PushConstants {
  /// BDA to RigidBodyArray (imex_math.glsl layout: quaternion + wrench_idx)
  pub rigid_bodies: u64,
  /// BDA to WrenchArray (6-float Wrench per entry)
  pub wrenches: u64,
  pub emitters: u64,
  /// Physical dt in seconds
  pub dt: f32,
  pub n_bodies: u32,
  /// Picard iteration count (4 is sufficient for most scenes; 8–10 for high-spin)
  pub n_iterations: u32,
  pub num_emitters: u32,
}

/// `integrate_particles_p4_5.comp` — 40 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ImexParticlesP45PushConstants {
  /// BDA to AOSOA particle data
  pub particles: u64,
  /// BDA to ClockBuffer (uvec2 global_time_us)
  pub clock: u64,
  /// Physical dt in seconds
  pub dt: f32,
  pub total_particles: u32,
  /// dt in microseconds — low 32 bits
  pub dt_us_lo: u32,
  /// dt in microseconds — high 32 bits
  pub dt_us_hi: u32,
  /// t_n (current frame start) in microseconds — low 32 bits
  pub current_time_lo: u32,
  /// t_n (current frame start) in microseconds — high 32 bits
  pub current_time_hi: u32,
}

/// `rb_force_assign.comp` — 24 bytes
#[repr(C)]
#[derive(Clone, Copy)]
pub struct RbForceAssignPushConstants {
  /// BDA to RigidBodyArray (read-only; only leaf_start_idx / leaf_count / wrench_idx used)
  pub rigid_bodies: u64,
  /// BDA to WrenchArray (leaf wrenches AND CoM wrench; both in same buffer)
  pub wrenches: u64,
  pub n_bodies: u32,
  pub _pad: u32,
}

/// `bp_clear.comp` — 32 bytes  (4 × 8-byte BDAs)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BpClearPushConstants {
  pub raw_scene_pairs: u64,
  pub out_rb_rb: u64,
  pub out_rb_ps: u64,
  pub out_rb_lca: u64,
  pub out_internal: u64,
  pub out_sparse: u64,
}

/// `bp_bounds_gen.comp` — 28 bytes
///
/// Matches the existing `bp_bounds_gen.comp` push-constant block:
/// ```glsl
/// EntityArray scene_entities;  // BDA
/// LeafBuffer  tlas_leaves;     // BDA
/// uvec2       dt_us;
/// uint        total_entities;
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BpBoundsGenPushConstants {
  pub scene_entities: u64,
  pub tlas_leaves: u64,
  pub dt_us_lo: u32,
  pub dt_us_hi: u32,
  pub total_entities: u32,
  pub _pad: u32,
}

/// `bp_scene.comp` — 40 bytes
///
/// Matches:
/// ```glsl
/// MultiBvhBuffer tlas_bvh;          // BDA (8 bytes — treated as u64 opaque)
/// LeafBuffer     query_leaves;       // BDA
/// PairBuffer     overlapping_pairs;  // BDA
/// uint           tlas_root_index;
/// uint           total_queries;
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BpScenePushConstants {
  pub tlas_bvh: u64,
  pub query_leaves: u64,
  pub overlapping_pairs: u64,
  pub tlas_root_index: u32,
  pub total_queries: u32,
}

/// `bp_classify.comp` — 40 bytes
///
/// ```glsl
/// EntityArray scene_entities;  // BDA
/// RawPairs    raw_pairs;       // BDA
/// QueueBuf    out_rb_rb;       // BDA
/// QueueBuf    out_rb_ps;       // BDA
/// QueueBuf    out_rb_lca;      // BDA
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BpClassifyPushConstants {
  pub raw_pairs: u64,
  pub out_rb_rb: u64,
  pub out_rb_ps: u64,
  pub out_ps_ps: u64,
  pub max_pairs: u32,
  pub num_rigid_bodies: u32,
}

/// `bp_cross_lca.comp` — 72 bytes
///
/// ```glsl
/// LcaEntityArray lca_entities;
/// LeafBuffer macro_leaves;
/// EntityHeaderArray entity_headers;
/// PairBuffer lca_query_pairs;
/// PairBuffer out_rb_rb;
/// PairBuffer out_rb_ps;
/// PairBuffer out_ps_ps;
/// CrossPairBuffer out_cross_pairs;
/// uint total_queries;
/// uint max_pairs;
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BpCrossLcaPushConstants {
  pub tlas_bvh_addr: u64,
  pub lca_entities: u64,
  pub macro_leaves: u64,
  pub entity_headers: u64,
  pub lca_query_pairs: u64,
  pub out_rb_rb: u64,
  pub out_rb_ps: u64,
  pub out_ps_ps: u64,
  pub out_cross_pairs: u64,
  pub total_queries: u32,
  pub max_pairs: u32,
}

/// `bp_particle_self.comp` — 40 bytes
///
/// ```glsl
/// MultiBvhBuffer bvh;            // BDA
/// ParticleData   particles;      // BDA
/// WrenchArray    wrench_buffer;  // BDA (writes into AOSOA force slots via atomicAdd)
/// uint           root_index;
/// uint           total_particles;
/// float          particle_radius;
/// float          stiffness;
/// ```
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BpParticleSelfPushConstants {
  pub bvh: u64,
  pub particles: u64,
  pub wrench_buffer: u64,
  pub root_index: u32,
  pub total_particles: u32,
  pub particle_radius: f32,
  pub stiffness: f32,
}

// ─────────────────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmitParticlesPushConstants {
  pub particles: u64,
  pub emitters: u64,
  pub bvh_nodes: u64,
  pub _pad0: u64,
  pub sun_pos: [f32; 3],
  pub dt: f32,
  pub max_particles: u32,
  pub num_emitters: u32,
}

/// TODO: Document this item
pub struct PhysicsPipelines {
  pub pipeline_layout: vk::PipelineLayout,
  // ── Legacy IMEX pipelines (kept for backward compatibility) ───────────────
  pub emit_particles: vk::Pipeline,
  pub lbvh_prepass: vk::Pipeline,
  pub lbvh_build: vk::Pipeline,
  pub motion_bounds: vk::Pipeline,
  pub motion_refit: vk::Pipeline,
  pub ccd: vk::Pipeline,
  pub stream_compact: vk::Pipeline,
  pub reduce_toi: vk::Pipeline,
  pub lcp_solver: vk::Pipeline,
  pub apply_impulses: vk::Pipeline,
  pub barnes_hut: vk::Pipeline,
  pub radix_sort: vk::Pipeline,
  pub morton_encode: vk::Pipeline,
  pub convert_particles: vk::Pipeline,
  pub graph_coloring: vk::Pipeline,
  pub lbvh_collapse: vk::Pipeline,
  // ── New Symmetric Strang-Split IMEX integrators ───────────────────────────
  /// VV predictor: x_n → x_{n+1}, v_{n+½}; clears force buffer
  pub integrate_particles_p1_p2: vk::Pipeline,
  /// RB Implicit Midpoint Rule + Picard gyro-stabilisation; clears wrench
  pub integrate_bodies_p3: vk::Pipeline,
  /// VV corrector: v_{n+½} → v_{n+1}; advances 64-bit engine clock
  pub integrate_particles_p4_5: vk::Pipeline,
  // ── Narrow Phase ──────────────────────────────────────────────────────────
  
  pub narrow_ccd: vk::Pipeline,
  
  pub narrow_ccd_cross_lca: vk::Pipeline,
  // ── Force aggregation ─────────────────────────────────────────────────────
  /// Leaf-wrench → CoM-wrench reduction (one WG per RB)
  pub rb_force_assign: vk::Pipeline,
  // ── Broad-phase suite ─────────────────────────────────────────────────────
  
  pub bp_clear: vk::Pipeline,
  
  pub bp_bounds_gen: vk::Pipeline,
  
  pub bp_scene: vk::Pipeline,
  
  pub bp_classify: vk::Pipeline,
  
  pub bp_cross_lca: vk::Pipeline,
  
  pub bp_particle_self: vk::Pipeline,
}

impl PhysicsPipelines {
  /// TODO: Document this item
  pub fn new(device: &LogicalDevice, debug_shaders: bool) -> GpuResult<Self> {
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
    let mut create_pipeline = |spv_path: &str| -> GpuResult<vk::Pipeline> {
      let spv_code = aethervk_oshal_rlib::os::fs::read(spv_path)
        .map_err(|_| GpuError::BackendSpecific(alloc::format!("Failed to read {}", spv_path)))?;
      let (prefix, code, suffix) = unsafe { spv_code.align_to::<u32>() };
      assert!(prefix.is_empty() && suffix.is_empty());

      let shader_info = vk::ShaderModuleCreateInfo::default().code(code);
      let shader_module =
        unsafe { device.create_shader_module(&shader_info, None) }.map_err(|e| {
          GpuError::BackendSpecific(alloc::format!("Failed to create shader module: {:?}", e))
        })?;

      let main_name = alloc::ffi::CString::new("main").unwrap();

      let mut spec_map_entries = alloc::vec::Vec::new();
      let mut spec_data = alloc::vec::Vec::new();
      let sg_size = 32u32;
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

      let spec_info =
        vk::SpecializationInfo::default().map_entries(&spec_map_entries).data(&spec_data);

      let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&main_name)
        .specialization_info(&spec_info);

      let compute_info =
        vk::ComputePipelineCreateInfo::default().stage(stage_info).layout(pipeline_layout);

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
      Ok(pipeline)
    };

    // Need to adjust path depending on where the test runs from.
    // Assuming root of workspace or test dir. We'll use absolute-ish or relative to workspace.
    // For safety, let's use a known path relative to the crate root or check multiple.
    let dir_lock = crate::gpu::ASSET_DIR.read();
    let base_dir = dir_lock.as_ref().unwrap();
    let sim_dir = if base_dir.ends_with("sim") {
      base_dir.clone()
    } else {
      alloc::format!("{}/sim", base_dir)
    };
    let res = (|| -> GpuResult<Self> {
      Ok(Self {
        pipeline_layout,
        // ── Legacy pipelines ────────────────────────────────────────────────
        emit_particles: create_pipeline(&alloc::format!("{}/emit_particles.comp.spv", sim_dir))?,
        lbvh_prepass: create_pipeline(&alloc::format!("{}/lbvh_prepass.comp.spv", sim_dir))?,
        lbvh_build: create_pipeline(&alloc::format!("{}/lbvh_build.comp.spv", sim_dir))?,
        motion_bounds: create_pipeline(&alloc::format!("{}/motion_bounds.comp.spv", sim_dir))?,
        motion_refit: create_pipeline(&alloc::format!("{}/motion_refit.comp.spv", sim_dir))?,
        ccd: create_pipeline(&alloc::format!("{}/ccd.comp.spv", sim_dir))?,
        stream_compact: create_pipeline(&alloc::format!("{}/stream_compact.comp.spv", sim_dir))?,
        reduce_toi: create_pipeline(&alloc::format!("{}/reduce_toi.comp.spv", sim_dir))?,
        lcp_solver: create_pipeline(&alloc::format!("{}/lcp_solver.comp.spv", sim_dir))?,
        apply_impulses: create_pipeline(&alloc::format!("{}/apply_impulses.comp.spv", sim_dir))?,
        barnes_hut: create_pipeline(&alloc::format!("{}/barnes_hut.comp.spv", sim_dir))?,
        radix_sort: create_pipeline(&alloc::format!("{}/radix_sort.comp.spv", sim_dir))?,
        morton_encode: create_pipeline(&alloc::format!("{}/morton_encode.comp.spv", sim_dir))?,
        convert_particles: create_pipeline(&alloc::format!(
          "{}/convert_particles.comp.spv",
          sim_dir
        ))?,
        graph_coloring: create_pipeline(&alloc::format!("{}/graph_coloring.comp.spv", sim_dir))?,
        lbvh_collapse: create_pipeline(&alloc::format!("{}/lbvh_collapse.comp.spv", sim_dir))?,
        // ── New IMEX integrators ────────────────────────────────────────────
        integrate_particles_p1_p2: create_pipeline(&alloc::format!(
          "{}/integrate_particles_p1_p2.comp.spv",
          sim_dir
        ))?,
        integrate_bodies_p3: create_pipeline(&alloc::format!(
          "{}/integrate_bodies_p3.comp.spv",
          sim_dir
        ))?,
        integrate_particles_p4_5: create_pipeline(&alloc::format!(
          "{}/integrate_particles_p4_5.comp.spv",
          sim_dir
        ))?,
        
        narrow_ccd: create_pipeline(&alloc::format!("{}/narrow_ccd.comp.spv", sim_dir))?,
        
        narrow_ccd_cross_lca: create_pipeline(&alloc::format!("{}/narrow_ccd_cross_lca.comp.spv", sim_dir))?,
        // ── Force aggregation ───────────────────────────────────────────────
        rb_force_assign: create_pipeline(&alloc::format!("{}/rb_force_assign.comp.spv", sim_dir))?,
        // ── Broad-phase suite ───────────────────────────────────────────────
        
        bp_clear: create_pipeline(&alloc::format!("{}/bp_clear.comp.spv", sim_dir))?,
        
        bp_bounds_gen: create_pipeline(&alloc::format!("{}/bp_bounds_gen.comp.spv", sim_dir))?,
        
        bp_scene: create_pipeline(&alloc::format!("{}/bp_scene.comp.spv", sim_dir))?,
        
        bp_classify: create_pipeline(&alloc::format!("{}/bp_classify.comp.spv", sim_dir))?,
        
        bp_cross_lca: create_pipeline(&alloc::format!("{}/bp_cross_lca.comp.spv", sim_dir))?,
        
        bp_particle_self: create_pipeline(&alloc::format!(
          "{}/bp_particle_self.comp.spv",
          sim_dir
        ))?,
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

  pub fn discard(&mut self, discard_pool: &resources::DiscardPool, timeline: u64) {
    // Layout must be last — it backs all pipelines
    discard_pool.discard_pipeline(self.emit_particles, timeline);
    discard_pool.discard_pipeline(self.lbvh_prepass, timeline);
    discard_pool.discard_pipeline(self.lbvh_build, timeline);
    discard_pool.discard_pipeline(self.motion_bounds, timeline);
    discard_pool.discard_pipeline(self.motion_refit, timeline);
    discard_pool.discard_pipeline(self.ccd, timeline);
    discard_pool.discard_pipeline(self.stream_compact, timeline);
    discard_pool.discard_pipeline(self.reduce_toi, timeline);
    discard_pool.discard_pipeline(self.lcp_solver, timeline);
    discard_pool.discard_pipeline(self.apply_impulses, timeline);
    discard_pool.discard_pipeline(self.barnes_hut, timeline);
    discard_pool.discard_pipeline(self.radix_sort, timeline);
    discard_pool.discard_pipeline(self.morton_encode, timeline);
    discard_pool.discard_pipeline(self.convert_particles, timeline);
    discard_pool.discard_pipeline(self.graph_coloring, timeline);
    discard_pool.discard_pipeline(self.lbvh_collapse, timeline);
    // New IMEX integrators
    discard_pool.discard_pipeline(self.integrate_particles_p1_p2, timeline);
    discard_pool.discard_pipeline(self.integrate_bodies_p3, timeline);
    discard_pool.discard_pipeline(self.integrate_particles_p4_5, timeline);
    
    discard_pool.discard_pipeline(self.narrow_ccd, timeline);
    
    discard_pool.discard_pipeline(self.narrow_ccd_cross_lca, timeline);
    // Force aggregation
    discard_pool.discard_pipeline(self.rb_force_assign, timeline);
    // Broad-phase suite
    
    discard_pool.discard_pipeline(self.bp_clear, timeline);
    
    discard_pool.discard_pipeline(self.bp_bounds_gen, timeline);
    
    discard_pool.discard_pipeline(self.bp_scene, timeline);
    
    discard_pool.discard_pipeline(self.bp_classify, timeline);
    
    discard_pool.discard_pipeline(self.bp_cross_lca, timeline);
    
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

      let command_buffers = [self.cmd];
      let signal_semaphores = [self.timeline_sem];

      // TAKE SUBMISSION LOCK BEFORE ALLOCATING TIMELINE!
      // This ensures that the order we get timeline values exactly matches the order we submit to the queue.
      let _guard = device.submission_lock.lock();
      self.timeline_value = next_submit_value.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
      let signal_values = [self.timeline_value];

      let mut timeline_info =
        vk::TimelineSemaphoreSubmitInfo::default().signal_semaphore_values(&signal_values);

      let submit_info = vk::SubmitInfo::default()
        .command_buffers(&command_buffers)
        .signal_semaphores(&signal_semaphores)
        .push_next(&mut timeline_info);

      device
        .handle
        .queue_submit(
          self.queue.handle,
          &[submit_info],
          vk::Fence::null(), // TODO fence?
        )
        .map_err(|e| GpuError::from(e))?;
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

    // Explicit CPU block for the buffer transfer to complete!
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
        let raw_u32 = core::slice::from_raw_parts(info.mapped_data as *const u32, 16);
        aethervk_oshal_rlib::log!("RAW STAGING BUFFER: {:?}", raw_u32);

        let offset = if self.is_list { 16 } else { 0 };
        let mapped_ptr = (info.mapped_data as *const u8).add(offset);
        core::ptr::copy_nonoverlapping(mapped_ptr as *const T, data.as_mut_ptr(), self.capacity);
        data.set_len(self.capacity);
      }

      // Cleanup staging buffer safely and immediately. Because `wait()` is invoked
      // strictly after `kernels.wait_sync(sync)`, we are 100% sure the GPU is finished!
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
  #[cfg(test)]
  pub tracked_physical_allocations: spin::Mutex<alloc::vec::Vec<u64>>,
}

impl VulkanComputeKernels {
  pub fn new(
    device: &LogicalDevice,
    _allocator: vk_mem::AllocatorView,
    queue_sharing_info: crate::gpu::QueueSharingInfo,
    debug_shaders: bool,
  ) -> GpuResult<Self> {
    let pipelines = PhysicsPipelines::new(device, debug_shaders)?;
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
      #[cfg(test)]
      tracked_physical_allocations: spin::Mutex::new(alloc::vec::Vec::new()),
    })
  }

  // TODO: How do I know if there's a command in flight? Should It be externally synchronized?
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
}

impl VulkanComputeKernels {
  #[function_name::named]
  fn allocate_and_upload<T: Copy + Send + Sync>(
    &self,
    device: &LogicalDevice,
    allocator: AllocatorView,
    data: &[T],
    usage: vk::BufferUsageFlags,
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<VulkanBuffer<T>> {
    let is_list = false;
    let size = (core::mem::size_of::<T>() * data.len().max(1)) as u64;

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
          size as usize,
        );
      }
    }

    let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let address =
      unsafe { device.buffer_device_address.get_buffer_device_address(&device_address_info) };

    Ok(VulkanBuffer {
      buffer,
      address,
      capacity: data.len().max(1),
      allocation: alloc,
      allocator,
      is_list,
      usage: ash::vk::BufferUsageFlags::empty(),
      discarded: false,
      _marker: core::marker::PhantomData,
    })
  }
  #[function_name::named]
  fn allocate_device_buffer<T: Copy + Send + Sync>(
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
    for i in 0..pool.entries.len() {
      let entry = &pool.entries[i];
      if entry.item_size == core::mem::size_of::<T>()
        && entry.capacity >= capacity
        && entry.is_list == is_list
        && (entry.usage & usage) == usage
        && entry.timeline_freed <= current_timeline
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

    let payload_size = (core::mem::size_of::<T>() * capacity.max(1)) as u64;
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

    Ok(VulkanBuffer {
      buffer,
      address,
      capacity: capacity.max(1),
      allocation: alloc,
      allocator,
      is_list,
      usage: ash::vk::BufferUsageFlags::empty(),
      discarded: false,
      _marker: core::marker::PhantomData,
    })
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
            crate::scene::ColliderShape::Sphere { radius } => (0, [radius, 0.0, 0.0]),
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
        let parent_id =
          scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
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
                let (shape_type, shape_data, inertia_tensor) = match collider.shape {
                    crate::scene::ColliderShape::Sphere { radius } => {
                    let i = 0.4 * mass * radius * radius;
                    (0, [radius, 0.0, 0.0], [[i, 0.0, 0.0], [0.0, i, 0.0], [0.0, 0.0, i]])
                    }
                    crate::scene::ColliderShape::OBB { half_extents } => {
                    let dx = half_extents.x() * 2.0;
                    let dy = half_extents.y() * 2.0;
                    let dz = half_extents.z() * 2.0;
                    let ix = (1.0 / 12.0) * mass * (dy * dy + dz * dz);
                    let iy = (1.0 / 12.0) * mass * (dx * dx + dz * dz);
                    let iz = (1.0 / 12.0) * mass * (dx * dx + dy * dy);
                    (1, [half_extents.x(), half_extents.y(), half_extents.z()], [[ix, 0.0, 0.0], [0.0, iy, 0.0], [0.0, 0.0, iz]])
                    }
                };

                let rot_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_quat_custom_frame(transform.rotation);
                let rot_arr = [
                    [rot_mat.component(0).unwrap(), rot_mat.component(1).unwrap(), rot_mat.component(2).unwrap()],
                    [rot_mat.component(4).unwrap(), rot_mat.component(5).unwrap(), rot_mat.component(6).unwrap()],
                    [rot_mat.component(8).unwrap(), rot_mat.component(9).unwrap(), rot_mat.component(10).unwrap()],
                ];

                bodies.push(gpu::RigidBodyGpu {
                    position: [transform.position.x(), transform.position.y(), transform.position.z()],
                    mass,
                    rotation: rot_arr,
                    linear_velocity: [velocity.x(), velocity.y(), velocity.z()],
                    _pad0: 0.0,
                    angular_velocity: [angular_velocity.x(), angular_velocity.y(), angular_velocity.z()],
                    _pad1: 0.0,
                    inertia_tensor,
                    force: [0.0, 0.0, 0.0],
                    torque: [0.0, 0.0, 0.0],
                    entity_id: entity,
                    parent_frame_id: parent_id,
                    shape_type,
                    shape_data,
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
  ) -> GpuResult<(VulkanBuffer<RigidBodyImex>, VulkanBuffer<Wrench>)> {
    let mut bodies: alloc::vec::Vec<RigidBodyImex> = alloc::vec::Vec::new();
    let mut wrench_idx: u32 = 0;

    scene0.query2::<crate::scene::TransformComponent, crate::scene::KinematicComponent, _>(
      |entity, transform, kinematic| {
        let mass = scene0
          .with_component(entity, |c: &crate::scene::ColliderComponent| {
            match c.shape {
              crate::scene::ColliderShape::Sphere { radius } => (c.mass, radius),
              crate::scene::ColliderShape::OBB { .. } => (c.mass, 0.5),
            }
          })
          .unwrap_or((1.0_f32, 0.5_f32));
        let (m, r) = mass;
        let i_inv = if r > 0.0 {
          1.0_f32 / (0.4_f32 * m * r * r)
        } else {
          1.0_f32
        };

        let (shape_type, shape_extents) = scene0
          .with_component(entity, |c: &crate::scene::ColliderComponent| {
            match c.shape {
              crate::scene::ColliderShape::Sphere { radius } => (2u32, [radius, 0.0, 0.0]),
              crate::scene::ColliderShape::OBB { half_extents } => {
                (1u32, [half_extents.x(), half_extents.y(), half_extents.z()])
              }
            }
          })
          .unwrap_or((0u32, [0.0, 0.0, 0.0]));

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
          inertia_inv_diag: [i_inv, i_inv, i_inv, 0.0],
          wrench_idx,
          leaf_start_idx: 0,
          leaf_count: 0,
          shape_type,
          shape_extents,
          _pad: 0,
        });
        wrench_idx += 1;
      },
    );

    let n = bodies.len().max(1);
    let rb_buf = self.allocate_and_upload::<RigidBodyImex>(
      device,
      allocator,
      &bodies,
      vk::BufferUsageFlags::STORAGE_BUFFER
        | vk::BufferUsageFlags::TRANSFER_SRC
        | vk::BufferUsageFlags::TRANSFER_DST,
      rollback,
    )?;

    let zeroed_wrenches: alloc::vec::Vec<Wrench> = alloc::vec![Wrench::default(); n];
    let w_buf = self.allocate_and_upload::<Wrench>(
      device,
      allocator,
      &zeroed_wrenches,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )?;

    Ok((rb_buf, w_buf))
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
      ash::vk::BufferUsageFlags::STORAGE_BUFFER | ash::vk::BufferUsageFlags::TRANSFER_DST,
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
                for (i, p) in particles.iter().enumerate().filter(|(_, p)| p.active != 0) {
                    flat_particles.push([
                        p.position[0], p.position[1], p.position[2],
                        p.velocity[0], p.velocity[1], p.velocity[2],
                        p.mass,
                        0.0, 0.0, 0.0
                    ]);
                    metadata.push(gpu::ParticleMetadata {
                        entity_id: entity,
                        parent_frame_id: parent_id,
                        original_index: i as u32,
                    });
                }
            }
        );

    let packed = gpu::pack_particles_aosoa(&flat_particles, 32);

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

  fn build_emitters(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    scene0: &Scene,
  ) -> GpuResult<VulkanBuffer<gpu::ForceEmitter>> {
    let mut emitters = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::ForceEmitterComponent, _>(
      |_, t, emitter| match emitter {
        crate::scene::ForceEmitterComponent::Gravity { mu } => {
          emitters.push(gpu::ForceEmitter {
            position: [t.position.x(), t.position.y(), t.position.z()],
            mu: *mu,
            normal: [0.0, 0.0, 0.0],
            type_id: 0,
            trunc_distance: 0.0,
            scale_factor: 1.0,
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
            scale_factor: 1.0,
            _pad: [0, 0],
          });
        }
      },
    );

    self.allocate_and_upload(
      device,
      allocator,
      &emitters,
      vk::BufferUsageFlags::STORAGE_BUFFER,
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
    let max_particles = particles.capacity() as u32;
    let wg_size = 128;
    let num_emitters = 1; // TODO Passed dynamically in reality
    let dispatch_groups = (max_particles + wg_size - 1) / wg_size;
    let dt_sec = dt as f32 / 1_000_000.0;

    let pc = EmitParticlesPushConstants {
      particles: particles.address,
      emitters: self.addresses.emitters,
      bvh_nodes: self.addresses.bvh_nodes,
      _pad0: 0,
      sun_pos: [sun_pos.x(), sun_pos.y(), sun_pos.z()],
      dt: dt_sec,
      max_particles,
      num_emitters,
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
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let dt_sec = dt as f32 / 1_000_000.0_f32;
    let wg_size = 128u32;
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
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
  /// `n_iterations`      — Picard iteration count; 4 suffices for most scenes.
  pub fn imex_integrate_bodies_p3(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies_addr: u64,
    wrenches_addr: u64,
    emitters_addr: u64,
    n_bodies: u32,
    num_emitters: u32,
    dt: timeus_t,
    n_iterations: u32,
  ) {
    let dt_sec = dt as f32 / 1_000_000.0_f32;
    let wg_size = 32u32;
    let groups = (n_bodies + wg_size - 1) / wg_size;

    let pc = ImexBodiesP3PushConstants {
      rigid_bodies: rigid_bodies_addr,
      wrenches: wrenches_addr,
      emitters: emitters_addr,
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
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let dt_sec = dt as f32 / 1_000_000.0_f32;
    let wg_size = 128u32;
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
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let pc = RbForceAssignPushConstants {
      rigid_bodies: rigid_bodies_addr,
      wrenches: wrenches_addr,
      n_bodies,
      _pad: 0,
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
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      // One WG per body
      device.cmd_dispatch(cmd.cmd, n_bodies, 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }
  }

  /// Dispatches `bp_clear.comp` (single thread, clears all four pair queues).
  
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
  
  pub fn bp_bounds_gen(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    scene_entities_addr: u64,
    tlas_leaves_addr: u64,
    total_entities: u32,
    dt: timeus_t,
  ) {
    let wg_size = 256u32;
    let groups = (total_entities + wg_size - 1) / wg_size;

    let pc = BpBoundsGenPushConstants {
      scene_entities: scene_entities_addr,
      tlas_leaves: tlas_leaves_addr,
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
      device.cmd_dispatch(cmd.cmd, groups, 1, 1);
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    // WG size = 256, assume SUBGROUP_SIZE = 32 → 8 subgroups/WG.  Conservatively 1 WG = 1 query.
    let wg_size = 256u32;
    let subgroups_per_wg = 256u32 / 32u32; // conservative for dispatch sizing
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
  
  pub fn bp_classify(
    &self,
    device: &LogicalDevice,
    cmd: &mut VulkanCommandBuffer,
    _scene_entities_addr: u64, // Ignored, no longer used by shader
    raw_pairs_addr: u64,
    out_rb_rb_addr: u64,
    out_rb_ps_addr: u64,
    out_ps_ps_addr: u64,
    _out_macro_lca_addr: u64, // Ignored (probably TODO?)
    _out_lca_lca_addr: u64,   // Ignored (probably TODO?)
    total_raw_pairs: u32,
    num_rigid_bodies: u32,
  ) {
    let wg_size = 256u32;
    let groups = (total_raw_pairs + wg_size - 1) / wg_size;

    let pc = BpClassifyPushConstants {
      raw_pairs: raw_pairs_addr,
      out_rb_rb: out_rb_rb_addr,
      out_rb_ps: out_rb_ps_addr,
      out_ps_ps: out_ps_ps_addr,
      max_pairs: 4000, // Matches the allocation capacity in gpu_backends.rs
      num_rigid_bodies,
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
  ) {
    let _wg_size = 256u32;
    let subgroups_per_wg = 256u32 / 32u32;
    let groups = (total_queries + subgroups_per_wg - 1) / subgroups_per_wg;

    let pc = BpCrossLcaPushConstants {
      tlas_bvh_addr,
      lca_entities: lca_entities_addr,
      macro_leaves: macro_leaves_addr,
      entity_headers: entity_headers_addr,
      lca_query_pairs: lca_query_pairs_addr,
      out_rb_rb: out_rb_rb_addr,
      out_rb_ps: out_rb_ps_addr,
      out_ps_ps: out_ps_ps_addr,
      out_cross_pairs: out_cross_pairs_addr,
      total_queries,
      max_pairs,
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let _wg_size = 256u32;
    let subgroups_per_wg = 256u32 / 32u32; // conservative; shader uses specialization const
    let groups = (total_particles + subgroups_per_wg - 1) / subgroups_per_wg;

    let pc = BpParticleSelfPushConstants {
      bvh: bvh_addr,
      particles: particles_addr,
      wrench_buffer: wrench_buffer_addr,
      root_index,
      total_particles,
      particle_radius,
      stiffness,
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let wg_size = 128;
    let total_particles = particles.capacity() as u32;
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;
    let dt_sec = dt as f32 / 1_000_000.0;

    let pc = P12PushConstants {
      particles: self.addresses.particle_data,
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
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let wg_size = 128;
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
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    _bvh: &VulkanBuffer<()>,
    particles: &mut VulkanBuffer<f32>,
  ) -> GpuResult<()> {
    let total_particles = particles.capacity() as u32;
    let dispatch_groups = (total_particles + 127) / 128;

    let pc_bh = BarnesHutPushConstants {
      particles: self.addresses.particle_data,
      bvh: self.addresses.bvh_nodes,
      root_index: 0,
      total_particles,
      theta: 0.5,
      // TODO switch to mu (G * M_Sun or whatever field)
      g: 1.0,
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
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      // TODO swittch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let wg_size = 128;
    let total_particles = particles.capacity() as u32;
    let num_kinematics = kinematics.capacity() as u32;
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;
    let dt_sec = dt as f32 / 1_000_000.0;

    let pc = P5PushConstants {
      particles: self.addresses.particle_data,
      emitters: self.addresses.emitters,
      kinematics: kinematics.address,
      dt: dt_sec,
      total_particles,
      num_emitters: 1, // TODO dynamic -> VulkanBuffer
      num_kinematics,
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
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    particles: &VulkanBuffer<f32>,
    _dt: timeus_t,
  ) -> GpuResult<VulkanBuffer<()>> {
    let total_particles = particles.capacity() as u32;
    let wg_size = 128;
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;

    let num_nodes = (total_particles * 2).max(1) as usize;
    aethervk_oshal_rlib::log!(
      "build_motion_bvh: allocating bvh_buffer with num_nodes={}, capacity={}, size={}",
      num_nodes,
      particles.capacity(),
      num_nodes * core::mem::size_of::<crate::gpu::compute_push_constants::MultiBvhNodeGpu>()
    );
    let bvh_buffer_result = self
      .allocate_device_buffer::<crate::gpu::compute_push_constants::MultiBvhNodeGpu>(
        device,
        allocator,
        num_nodes,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        false,
        rollback,
      );
    if let Err(e) = &bvh_buffer_result {
      aethervk_oshal_rlib::log!("build_motion_bvh failed to allocate bvh_buffer: {:?}", e);
    }
    let bvh_buffer = bvh_buffer_result?;
    aethervk_oshal_rlib::log!("build_motion_bvh: allocated bvh_buffer successfully");

    let pc = LbvhPushConstants {
      bvh: bvh_buffer.address, // self.addresses.bvh_nodes,
      sorted_morton: self.addresses.sorted_morton,
      counters: self.addresses.atomic_counters,
      particles: particles.address, // self.addresses.particle_data,
      num_primitives: total_particles,
      particle_radius: 1.0,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<LbvhPushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.lbvh_build,
      );
      device.cmd_push_constants(
        cmd.cmd,
        self.pipelines.pipeline_layout,
        vk::ShaderStageFlags::COMPUTE,
        0,
        bytes,
      );
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(bvh_buffer.cast())
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
      let wg_size = 256;
      let dispatch_groups = (total_nodes + wg_size - 1) / wg_size;
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
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
    let wg_size = 32; // TODO we are approximating a warp. rework the bp_scene shader for 128
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
      tlas_bvh: self.addresses.bvh_nodes,
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
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    Ok(candidates_buffer)
  }

  
  #[cfg(any(test, feature = "collisions"))]
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
      device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.ccd);
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let wg_size = 128;
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
      device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
      dt: 0.0,
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
      restitution: restitution_val,
      rigid_bodies: rigid_bodies.address,
      dt: 0.001_f32, // used only in Baumgarte stabilization, so don't care
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );
    }

    // Deferred-free: GPU commands referencing `impulses_buffer` have been recorded,
    // so it is safe to release once the compute timeline reaches `next_submit_value`.
    impulses_buffer.discard(
      &self.discard_pool,
      self.next_submit_value.load(core::sync::atomic::Ordering::Relaxed),
    );

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
        device.cmd_copy_buffer(
          cmd.cmd,
          rigid_bodies.buffer,
          rb_snap.buffer,
          core::slice::from_ref(&rb_copy),
        );
      }
      if particles.capacity() > 0 {
        device.cmd_copy_buffer(
          cmd.cmd,
          particles.buffer,
          p_snap.buffer,
          core::slice::from_ref(&p_copy),
        );
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
        device.cmd_copy_buffer(
          cmd.cmd,
          snapshot.0.buffer,
          rigid_bodies.buffer,
          core::slice::from_ref(&rb_copy),
        );
      }
      if particles.capacity() > 0 {
        device.cmd_copy_buffer(
          cmd.cmd,
          snapshot.1.buffer,
          particles.buffer,
          core::slice::from_ref(&p_copy),
        );
      }

      // TODO switch to synchronization2
      let barrier = vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
      device.cmd_pipeline_barrier(
        cmd.cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
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
    let rb_handle = rigid_bodies.enqueue_read_to_cpu(cmd).map_err(|e| gpu_err!("{}", e))?;
    let p_handle = particles.enqueue_read_to_cpu(cmd).map_err(|e| gpu_err!("{}", e))?;

    let sync_info = cmd.submit().map_err(|e| gpu_err!("{}", e))?;

    let rb_data = rb_handle.wait().map_err(|e| gpu_err!("{}", e))?;
    let p_data = p_handle.wait().map_err(|e| gpu_err!("{}", e))?;

    let unpacked_particles = gpu::unpack_particles_aosoa(&p_data, 32, particle_metadata.len());

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
    scene.query2_mut::<crate::scene::TransformComponent, crate::scene::KinematicComponent, _>(
      |_entity,
       trans: &mut crate::scene::TransformComponent,
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

    Ok(sync_info)
  }
}

impl Kernels for Device {
  
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
    let s = self.query_result.subgroup_size;
    if s >= 64 {
      Some(crate::gpu::SubgroupSize::Size64)
    } else if s <= 16 {
      Some(crate::gpu::SubgroupSize::Size16)
    } else {
      Some(crate::gpu::SubgroupSize::Size32)
    }
  }

  fn wait_sync(&self, sync: &crate::gpu::CommandBufferSyncInfo) -> EngineResult<()> {
    use ash::vk::Handle;
    let sem = ash::vk::Semaphore::from_raw(sync.timeline_semaphore);
    self.device.wait_for_semaphore_value(sem, sync.timeline_value, u64::MAX).map_err(|e| {
      crate::types::EngineError::Gpu(crate::types::GpuError::BackendSpecific(alloc::format!(
        "{:?}", e
      )))
    })
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
      let zero = alloc::vec![0u8; core::mem::size_of::<crate::math::collision::multi_bvh::TlasMultiNode<32>>()];
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
          self.kernels.tracked_physical_allocations.lock().push(info.device_memory.as_raw());
        }
        aethervk_oshal_rlib::log!("upload_motion_tlas alloc: {:?}", alloc.get_raw());
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
          usage: ash::vk::BufferUsageFlags::empty(),
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
        self.kernels.build_kinematic_bodies(&self.device, allocator, rollback, cmd, scene, scene0)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<(Self::Buffer<RigidBodyImex>, Self::Buffer<Wrench>)> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_rigid_bodies_imex(&self.device, allocator, rollback, cmd, scene, scene0)
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

  fn build_emitters(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<gpu::ForceEmitter>> {
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
        self.kernels.step_ode_p1_p2(&self.device, allocator, rollback, cmd, particles, dt)
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
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.compute_self_gravity(&self.device, allocator, rollback, cmd, bvh, particles)
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
    particles: &Self::Buffer<f32>,
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
        self.kernels.compact_collisions(&self.device, allocator, rollback, cmd, globals, time_delta)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  
  #[cfg(any(test, feature = "collisions"))]
  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<gpu::CollisionPair>,
  ) -> EngineResult<Self::Buffer<u32>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.find_earliest_collision(&self.device, allocator, rollback, cmd, compacted)
      })
      .commit_read(|_res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  
  #[cfg(any(test, feature = "collisions"))]
  fn apply_collision_responses(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &mut Self::Buffer<RigidBodyImex>,
    particles: &mut Self::Buffer<f32>,
    collisions: &Self::List<gpu::CollisionPair>,
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
        let timeline_value =
          self.kernels.next_submit_value.fetch_add(1, core::sync::atomic::Ordering::SeqCst);
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
      self.device.cmd_fill_buffer(cmd.cmd, list.buffer, 0, 16, 0);
      let barrier = ash::vk::MemoryBarrier::default()
        .src_access_mask(ash::vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(ash::vk::AccessFlags::SHADER_READ | ash::vk::AccessFlags::SHADER_WRITE);
      self.device.cmd_pipeline_barrier(
        cmd.cmd,
        ash::vk::PipelineStageFlags::TRANSFER,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
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
          particles.capacity() as u32,
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
          bodies.capacity() as u32,
          emitters.capacity() as u32,
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
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |_res_guard, _| Ok::<_, GpuError>(()))?
      .execute(|(), _rollback| {
        self.kernels.imex_rb_force_assign(
          &self.device,
          cmd,
          bodies.address,
          wrenches.address,
          bodies.capacity() as u32,
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
          particles.capacity() as u32,
          dt,
          current_time_us,
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

    let dispatch_groups = (broadphase_pairs.capacity as u32 + 255) / 256;

    unsafe {
      let pipeline = if space_type == 1 {
        self.kernels.pipelines.narrow_ccd_cross_lca
      } else {
        self.kernels.pipelines.narrow_ccd
      };

      self.device.cmd_bind_pipeline(
        cmd.cmd,
        ash::vk::PipelineBindPoint::COMPUTE,
        pipeline,
      );
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
        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
        ash::vk::PipelineStageFlags::COMPUTE_SHADER,
        ash::vk::DependencyFlags::empty(),
        core::slice::from_ref(&barrier),
        &[],
        &[],
      );

      let pc_part = NarrowCcdParticlesPushConstants {
        scene_entities: rigid_bodies.address,
        output_list: output_list.address,
        particles: particles.address,
        num_rigid_bodies: rigid_bodies.capacity as u32,
        num_particles: particles.capacity as u32 / 32, // Particles have 32 floats
        dt: 0.0,
        particle_radius: 0.5,
      };
      let bytes_part = core::slice::from_raw_parts(
        &pc_part as *const _ as *const u8,
        core::mem::size_of_val(&pc_part),
      );
      let dispatch_groups_part = (pc_part.num_particles + 255) / 256;

      self.device.cmd_bind_pipeline(
        cmd.cmd,
        ash::vk::PipelineBindPoint::COMPUTE,
        self.kernels.pipelines.narrow_ccd,
      );
      self.device.cmd_push_constants(
        cmd.cmd,
        self.kernels.pipelines.pipeline_layout,
        ash::vk::ShaderStageFlags::COMPUTE,
        0,
        bytes_part,
      );
      self.device.cmd_dispatch(cmd.cmd, dispatch_groups_part.max(1), 1, 1);

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
}

#[cfg(test)]
#[path = "mock_physics_tests.rs"]
mod physics_tests;
