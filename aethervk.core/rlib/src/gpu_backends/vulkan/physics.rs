//! Vulkan Backend Integration for the IMEX / LCP Physics Engine
//!
//! This module scaffolds the execution of the massive compute-shader pipeline.
//! It assumes Vulkan 1.1 with `VK_KHR_buffer_device_address` and `VK_KHR_shader_subgroup_basic`.

use alloc::vec::Vec;
use crate::gpu::{CommandBuffer, DeviceBuffer, WaitHandle, DeviceList, DeviceBvh, Kernels, KinematicBody, DynamicBody, CollisionPair};
use crate::types::{EngineResult, EngineError};
use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;
use aethervk_oshal_rlib::os::time::timeus_t;
use ash::vk;

/// Configuration parameters for the physics pipeline
pub struct PhysicsPipelineConfig {
  pub max_particles: u32,
  pub hardware_subgroup_size: u32,
}

/// TODO: Document this item
pub struct PhysicsDeviceAddresses {
  pub particle_data: u64,
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
    pub dt: f32,
    pub total_particles: u32,
    pub num_emitters: u32,
    pub _pad: u32,
}

/// TODO: Document this item
pub struct PhysicsPipelines {
  pub pipeline_layout: vk::PipelineLayout,
  pub p1_2_imex: vk::Pipeline,
  pub lbvh_build: vk::Pipeline,
  pub ccd: vk::Pipeline,
  pub ccd_rigidbody: vk::Pipeline,
  pub stream_compact: vk::Pipeline,
  pub reduce_toi: vk::Pipeline,
  pub lcp_solver: vk::Pipeline,
  pub apply_impulses: vk::Pipeline,
  pub barnes_hut: vk::Pipeline,
  pub p5_imex: vk::Pipeline,
}

impl PhysicsPipelines {
  /// TODO: Document this item
  pub fn new(_config: &PhysicsPipelineConfig) -> Self {
    // Placeholder, in reality this would compile shaders and create pipelines
    Self {
      pipeline_layout: vk::PipelineLayout::null(),
      p1_2_imex: vk::Pipeline::null(),
      lbvh_build: vk::Pipeline::null(),
      ccd: vk::Pipeline::null(),
      ccd_rigidbody: vk::Pipeline::null(),
      stream_compact: vk::Pipeline::null(),
      reduce_toi: vk::Pipeline::null(),
      lcp_solver: vk::Pipeline::null(),
      apply_impulses: vk::Pipeline::null(),
      barnes_hut: vk::Pipeline::null(),
      p5_imex: vk::Pipeline::null(),
    }
  }
}

/// TODO: Document this item
pub struct VulkanCommandBuffer {
    pub cmd: vk::CommandBuffer,
}

impl CommandBuffer for VulkanCommandBuffer {
    fn submit(&mut self) -> EngineResult<()> {
        // Handled by RenderDevice/CommandPools in practice
        Ok(())
    }
}

/// TODO: Document this item
pub struct VulkanWaitHandle<T> {
    _marker: core::marker::PhantomData<T>,
}

impl<T: Send + Sync> WaitHandle<T> for VulkanWaitHandle<T> {
    fn wait(self) -> EngineResult<T> {
        Err(EngineError::InvalidState("Not fully implemented yet"))
    }
}

/// TODO: Document this item
pub struct VulkanBuffer<T> {
    pub buffer: vk::Buffer,
    pub address: u64,
    pub capacity: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for VulkanBuffer<T> {
    type Cmd = VulkanCommandBuffer;
    type ReadHandle<'a> = VulkanWaitHandle<Vec<T>> where Self: 'a, T: 'a;

    fn capacity(&self) -> usize { self.capacity }

    fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>> {
        Ok(VulkanWaitHandle { _marker: core::marker::PhantomData })
    }
}

/// TODO: Document this item
pub struct VulkanList<T> {
    pub buffer: vk::Buffer,
    pub address: u64,
    pub capacity: usize,
    _marker: core::marker::PhantomData<T>,
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for VulkanList<T> {
    type Cmd = VulkanCommandBuffer;
    type ReadHandle<'a> = VulkanWaitHandle<Vec<T>> where Self: 'a, T: 'a;

    fn capacity(&self) -> usize { self.capacity }

    fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>> {
        Ok(VulkanWaitHandle { _marker: core::marker::PhantomData })
    }
}

impl<T: Copy + Send + Sync> DeviceList<T> for VulkanList<T> {
    fn clear(&mut self, _cmd: &mut Self::Cmd) -> EngineResult<()> {
        Ok(())
    }
}

/// TODO: Document this item
pub struct VulkanMotionBvh {
    pub buffer: vk::Buffer,
}

impl DeviceBvh for VulkanMotionBvh {
    type Cmd = VulkanCommandBuffer;
}

/// TODO: Document this item
pub struct VulkanComputeKernels {
    pub device: ash::Device,
    pub pipelines: PhysicsPipelines,
    pub addresses: PhysicsDeviceAddresses,
}

impl Kernels for VulkanComputeKernels {
    type Cmd = VulkanCommandBuffer;
    type Buffer<T: Copy + Send + Sync> = VulkanBuffer<T>;
    type List<T: Copy + Send + Sync> = VulkanList<T>;
    type MotionBvh = VulkanMotionBvh;

    fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
        Ok(VulkanCommandBuffer { cmd: vk::CommandBuffer::null() })
    }

    fn build_kinematic_bodies(
        &self,
        _cmd: &mut Self::Cmd,
        _scene: &PhysicsScene,
        _scene0: &Scene,
    ) -> EngineResult<Self::Buffer<KinematicBody>> {
        Ok(VulkanBuffer { buffer: vk::Buffer::null(), address: 0, capacity: 0, _marker: core::marker::PhantomData })
    }

    fn build_dynamic_bodies(
        &self,
        _cmd: &mut Self::Cmd,
        _scene: &PhysicsScene,
        _scene0: &Scene,
    ) -> EngineResult<Self::Buffer<DynamicBody>> {
        Ok(VulkanBuffer { buffer: vk::Buffer::null(), address: 0, capacity: 0, _marker: core::marker::PhantomData })
    }

    fn step_ode_p1_p2(
        &self,
        cmd: &mut Self::Cmd,
        dynamics: &mut Self::Buffer<DynamicBody>,
        dt: timeus_t,
    ) -> EngineResult<()> {
        let wg_size = 256;
        let total_particles = dynamics.capacity() as u32;
        let dispatch_groups = (total_particles + wg_size - 1) / wg_size;
        let dt_sec = dt as f32 / 1_000_000.0;
        
        let pc = P12PushConstants {
            particles: self.addresses.particle_data,
            dt: dt_sec,
            total_particles,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<P12PushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.p1_2_imex);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        
        Ok(())
    }

    fn step_ode_p5(
        &self,
        cmd: &mut Self::Cmd,
        _kinematics: &Self::Buffer<KinematicBody>,
        dynamics: &mut Self::Buffer<DynamicBody>,
        _bvh: &Self::MotionBvh,
        dt: timeus_t,
    ) -> EngineResult<()> {
        let wg_size = 256;
        let total_particles = dynamics.capacity() as u32;
        let dispatch_groups = (total_particles + wg_size - 1) / wg_size;
        let dt_sec = dt as f32 / 1_000_000.0;
        
        let pc = P5PushConstants {
            particles: self.addresses.particle_data,
            emitters: self.addresses.emitters,
            dt: dt_sec,
            total_particles,
            num_emitters: 1, // TODO dynamic
            _pad: 0,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<P5PushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.p5_imex);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        Ok(())
    }

    fn build_motion_bvh(
        &self,
        _cmd: &mut Self::Cmd,
        _kinematics: &Self::Buffer<KinematicBody>,
        _dynamics: &Self::Buffer<DynamicBody>,
    ) -> EngineResult<Self::MotionBvh> {
        let total_particles = dynamics.capacity() as u32;
        let wg_size = 256;
        let dispatch_groups = (total_particles + wg_size - 1) / wg_size;

        let pc = LbvhPushConstants {
            bvh: self.addresses.bvh_nodes,
            sorted_morton: self.addresses.sorted_morton,
            counters: self.addresses.atomic_counters,
            particles: self.addresses.particle_data,
            num_primitives: total_particles,
            particle_radius: 1.0, 
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<LbvhPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.lbvh_build);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        Ok(VulkanMotionBvh { buffer: vk::Buffer::null() })
    }

    fn self_intersect_scene(
        &self,
        cmd: &mut Self::Cmd,
        _bvh: &Self::MotionBvh,
    ) -> EngineResult<Self::List<CollisionPair>> {
        // We'll pass total_entities via some state, hardcoded to some value here or assume we have it
        let total_entities = 1000; // Placeholder
        let wg_size = 32;
        let dispatch_groups = (total_entities + wg_size - 1) / wg_size;

        let pc = crate::gpu::compute_push_constants::BroadPhasePushConstants {
            tlas_bvh_addr: self.addresses.bvh_nodes,
            scene_entities_addr: self.addresses.particle_data, // Placeholder
            overlapping_pairs_addr: self.addresses.ccd_candidates,
            tlas_root_index: 0,
            total_entities,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<crate::gpu::compute_push_constants::BroadPhasePushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.broad_phase);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        Ok(VulkanList { buffer: vk::Buffer::null(), address: 0, capacity: 0, _marker: core::marker::PhantomData })
    }

    fn intersect_instances(
        &self,
        _cmd: &mut Self::Cmd,
        _potentials: &Self::List<CollisionPair>,
    ) -> EngineResult<Self::List<CollisionPair>> {
        Ok(VulkanList { buffer: vk::Buffer::null(), address: 0, capacity: 0, _marker: core::marker::PhantomData })
    }

    fn compact_collisions(
        &self,
        cmd: &mut Self::Cmd,
        _globals: &Self::List<CollisionPair>,
        _time_delta: timeus_t,
    ) -> EngineResult<Self::List<CollisionPair>> {
        let total_elements = 10000; // Max Candidates placeholder
        let wg_size = 256;
        let dispatch_groups = (total_elements + wg_size - 1) / wg_size;

        let pc = StreamCompactPushConstants {
            sparse_in: self.addresses.ccd_candidates,
            packed_out: self.addresses.packed_collisions,
            total_elements,
            _pad: 0,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<StreamCompactPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.stream_compact);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        Ok(VulkanList { buffer: vk::Buffer::null(), address: 0, capacity: 0, _marker: core::marker::PhantomData })
    }

    fn find_earliest_collision(
        &self,
        cmd: &mut Self::Cmd,
        _compacted: &Self::List<CollisionPair>,
    ) -> EngineResult<Self::Buffer<timeus_t>> {
        let pc = ReduceToiPushConstants {
            particles: self.addresses.particle_data,
            collisions: self.addresses.packed_collisions,
            out_toi: self.addresses.reduce_toi,
            particle_radius: 1.0,
            dt: 0.016,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<ReduceToiPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.reduce_toi);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            // Packed collisions buffer is passed in via the struct or globals buffer somehow.
            // Since `packed_collisions` is just an address, we can't `cmd_dispatch_indirect` off of `u64` directly.
            // We need the actual `vk::Buffer`. 
            // Wait, we can't directly get the buffer from the address here without it being passed.
            // But we can just use the `compacted.buffer`.
            self.device.cmd_dispatch_indirect(cmd.cmd, _compacted.buffer, 0);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        Ok(VulkanBuffer { buffer: vk::Buffer::null(), address: 0, capacity: 0, _marker: core::marker::PhantomData })
    }

    fn apply_collision_responses(
        &self,
        cmd: &mut Self::Cmd,
        _dynamics: &mut Self::Buffer<DynamicBody>,
        collisions: &Self::List<CollisionPair>,
        _force_inelastic: bool,
    ) -> EngineResult<()> {
        // LCP Solver
        let total_clusters = 100; // Unused when using indirect
        let pc_lcp = LcpPushConstants {
            particles: self.addresses.particle_data,
            collisions: self.addresses.packed_collisions,
            outputs: self.addresses.impulses,
            total_clusters,
        };
        let bytes_lcp = unsafe { core::slice::from_raw_parts(&pc_lcp as *const _ as *const u8, core::mem::size_of::<LcpPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.lcp_solver);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_lcp);
            self.device.cmd_dispatch_indirect(cmd.cmd, collisions.buffer, 0);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        // Apply Impulses
        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.apply_impulses);
            self.device.cmd_dispatch_indirect(cmd.cmd, collisions.buffer, 0);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        let total_particles = _dynamics.capacity() as u32;
        let dispatch_groups = (total_particles + 127) / 128;

        // Barnes Hut Self Gravity
        let pc_bh = BarnesHutPushConstants {
            particles: self.addresses.particle_data,
            bvh: self.addresses.bvh_nodes,
            root_index: 0,
            total_particles,
            theta: 0.5,
            g: 6.67430e-11,
        };
        let bytes_bh = unsafe { core::slice::from_raw_parts(&pc_bh as *const _ as *const u8, core::mem::size_of::<BarnesHutPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.barnes_hut);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_bh);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        // Phase 5 Drift & Kick
        let pc_p5 = P5PushConstants {
            particles: self.addresses.particle_data,
            emitters: self.addresses.emitters,
            dt: 0.016,
            total_particles,
            num_emitters: 1,
            _pad: 0,
        };
        let bytes_p5 = unsafe { core::slice::from_raw_parts(&pc_p5 as *const _ as *const u8, core::mem::size_of::<P5PushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.p5_imex);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_p5);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        Ok(())
    }

    fn snapshot_dynamics(
        &self,
        _cmd: &mut Self::Cmd,
        _dynamics: &Self::Buffer<DynamicBody>,
    ) -> EngineResult<Self::Buffer<DynamicBody>> {
        Ok(VulkanBuffer { buffer: vk::Buffer::null(), address: 0, capacity: 0, _marker: core::marker::PhantomData })
    }

    fn restore_dynamics(
        &self,
        _cmd: &mut Self::Cmd,
        _dynamics: &mut Self::Buffer<DynamicBody>,
        _snapshot: &Self::Buffer<DynamicBody>,
    ) -> EngineResult<()> {
        Ok(())
    }

    fn write_back_to_scene(
        &self,
        _cmd: &mut Self::Cmd,
        _dynamics: &Self::Buffer<DynamicBody>,
        _physical_scene: &mut PhysicsScene,
        _scene: &Scene,
    ) -> EngineResult<()> {
        Ok(())
    }
}
