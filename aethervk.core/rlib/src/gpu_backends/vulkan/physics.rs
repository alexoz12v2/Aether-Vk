use alloc::string::ToString;
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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EmitParticlesPushConstants {
    pub particles: u64,
    pub emitters: u64,
    pub bvh_nodes: u64,
    pub sun_pos: [f32; 3],
    pub dt: f32,
    pub max_particles: u32,
    pub num_emitters: u32,
}

/// TODO: Document this item
pub struct PhysicsPipelines {
  pub pipeline_layout: vk::PipelineLayout,
  pub emit_particles: vk::Pipeline,
  pub p1_2_imex: vk::Pipeline,
  pub p3_4_imex: vk::Pipeline,
  pub lbvh_build: vk::Pipeline,
  pub ccd: vk::Pipeline,
  pub ccd_rigidbody: vk::Pipeline,
  pub stream_compact: vk::Pipeline,
  pub reduce_toi: vk::Pipeline,
  pub lcp_solver: vk::Pipeline,
  pub apply_impulses: vk::Pipeline,
  pub barnes_hut: vk::Pipeline,
  pub p5_imex: vk::Pipeline,
  pub broad_phase: vk::Pipeline,
}

impl PhysicsPipelines {
  /// TODO: Document this item
  pub fn new(device: &ash::Device, _config: &PhysicsPipelineConfig) -> Self {
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(128); // Max typical push constant size

    let layout_info = vk::PipelineLayoutCreateInfo::default()
        .push_constant_ranges(core::slice::from_ref(&push_constant_range));

    let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }.unwrap();

    let create_pipeline = |spv_path: &str| -> vk::Pipeline {
        let spv_code = std::fs::read(spv_path).unwrap_or_else(|_| panic!("Failed to read {}", spv_path));
        let (prefix, code, suffix) = unsafe { spv_code.align_to::<u32>() };
        assert!(prefix.is_empty() && suffix.is_empty());

        let shader_info = vk::ShaderModuleCreateInfo::default().code(code);
        let shader_module = unsafe { device.create_shader_module(&shader_info, None) }.unwrap();

        let main_name = std::ffi::CString::new("main").unwrap();

        let mut spec_map_entries = alloc::vec::Vec::new();
        let mut spec_data = alloc::vec::Vec::new();
        let sg_size = 32u32;
        spec_map_entries.push(vk::SpecializationMapEntry {
            constant_id: 0,
            offset: 0,
            size: 4,
        });
        spec_data.extend_from_slice(&sg_size.to_le_bytes());

        let debug_shaders = 1u32;
        spec_map_entries.push(vk::SpecializationMapEntry {
            constant_id: 10,
            offset: 4,
            size: 4,
        });
        spec_data.extend_from_slice(&debug_shaders.to_le_bytes());

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
        }.unwrap()[0];

        unsafe { device.destroy_shader_module(shader_module, None); }
        pipeline
    };

    // Need to adjust path depending on where the test runs from. 
    // Assuming root of workspace or test dir. We'll use absolute-ish or relative to workspace.
    // For safety, let's use a known path relative to the crate root or check multiple.
    let base_dir = if std::path::Path::new("assets/sim/p1-2_imex_particles.comp.spv").exists() {
        "assets/sim"
    } else if std::path::Path::new("../../assets/sim/p1-2_imex_particles.comp.spv").exists() {
        "../../assets/sim"
    } else {
        panic!("Could not find shaders directory!");
    };

    Self {
        pipeline_layout,
        emit_particles: create_pipeline(&format!("{}/emit_particles.comp.spv", base_dir)),
        p1_2_imex: create_pipeline(&format!("{}/p1-2_imex_particles.comp.spv", base_dir)),
        p3_4_imex: create_pipeline(&format!("{}/p3-4_imex_rigidbody_imr.comp.spv", base_dir)),
        lbvh_prepass: create_pipeline(&format!("{}/lbvh_prepass.comp.spv", base_dir)),
        lbvh_build: create_pipeline(&format!("{}/lbvh_build.comp.spv", base_dir)),
        ccd: create_pipeline(&format!("{}/ccd.comp.spv", base_dir)),
        ccd_rigidbody: create_pipeline(&format!("{}/narrow_ccd_rigidbody.comp.spv", base_dir)),
        stream_compact: create_pipeline(&format!("{}/stream_compact.comp.spv", base_dir)),
        reduce_toi: create_pipeline(&format!("{}/reduce_toi.comp.spv", base_dir)),
        lcp_solver: create_pipeline(&format!("{}/lcp_solver.comp.spv", base_dir)),
        apply_impulses: create_pipeline(&format!("{}/apply_impulses.comp.spv", base_dir)),
        barnes_hut: create_pipeline(&format!("{}/barnes_hut.comp.spv", base_dir)),
        p5_imex: create_pipeline(&format!("{}/p5_imex_particles.comp.spv", base_dir)),
        broad_phase: create_pipeline(&format!("{}/broad_phase.comp.spv", base_dir)),
    }
  }
  }
}

/// TODO: Document this item
pub struct VulkanCommandBuffer {
    pub cmd: vk::CommandBuffer,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
}

impl Drop for VulkanCommandBuffer {
    fn drop(&mut self) {
        unsafe {
            self.device.free_command_buffers(self.command_pool, core::slice::from_ref(&self.cmd));
        }
    }
}

impl CommandBuffer for VulkanCommandBuffer {
    fn submit(&mut self) -> EngineResult<()> {
        unsafe {
            self.device.end_command_buffer(self.cmd).map_err(|_| EngineError::Gpu(GpuError::InvalidState("Failed to end cmd".to_string())))?;
            let submit_info = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&self.cmd));
            let fence = self.device.create_fence(&vk::FenceCreateInfo::default(), None).map_err(|_| EngineError::Gpu(GpuError::OutOfMemory))?;
            self.device.queue_submit(self.queue, core::slice::from_ref(&submit_info), fence).map_err(|_| EngineError::Gpu(GpuError::InvalidState("Failed queue submit".to_string())))?;
            self.device.wait_for_fences(core::slice::from_ref(&fence), true, u64::MAX).map_err(|_| EngineError::Gpu(GpuError::InvalidState("Fence wait error".to_string())))?;
            self.device.destroy_fence(fence, None);
            
            // Re-begin for further commands
            self.device.reset_command_buffer(self.cmd, vk::CommandBufferResetFlags::empty()).map_err(|_| EngineError::Gpu(GpuError::InvalidState("Reset CMD error".to_string())))?;
            let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(self.cmd, &begin_info).map_err(|_| EngineError::Gpu(GpuError::InvalidState("Begin CMD error".to_string())))?;
        }
        Ok(())
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

/// TODO: Document this item
pub struct VulkanBuffer<T> {
    pub buffer: vk::Buffer,
    pub address: u64,
    pub capacity: usize,
    pub allocation: Option<vk_mem::ffi::VmaAllocation>,
    pub allocator: Option<vk_mem::ffi::VmaAllocator>,
    _marker: core::marker::PhantomData<T>,
}

impl<T> Drop for VulkanBuffer<T> {
    fn drop(&mut self) {
        if let (Some(alloc), Some(allocator)) = (self.allocation, self.allocator) {
            let safe_allocator = unsafe { vk_mem::AllocatorView::from_raw(allocator) };
            unsafe {
                // To properly drop, we'd need to convert to vk_mem::Allocation and destroy it.
                // Assuming the wrapper works like this:
                let allocation = vk_mem::Allocation::from_raw(alloc);
                safe_allocator.destroy_buffer(self.buffer, &allocation);
                core::mem::forget(safe_allocator); // Don't destroy the main allocator
            }
        }
    }
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for VulkanBuffer<T> {
    type Cmd = VulkanCommandBuffer;
    type ReadHandle<'a> = VulkanWaitHandle<Vec<T>> where Self: 'a, T: 'a;

    fn capacity(&self) -> usize { self.capacity }

    fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>> {
        let allocator = self.allocator.ok_or(EngineError::InvalidState("Missing VMA allocator"))?;
        let safe_allocator = unsafe { vk_mem::AllocatorView::from_raw(allocator) };
        let alloc = self.allocation.ok_or(EngineError::InvalidState("Missing allocation"))?;
        let allocation = unsafe { vk_mem::Allocation::from_raw(alloc) };
        
        let info = safe_allocator.get_allocation_info(&allocation);
        let mut data = Vec::with_capacity(self.capacity);
        unsafe {
            if !info.mapped_data.is_null() {
                core::ptr::copy_nonoverlapping(
                    info.mapped_data as *const T,
                    data.as_mut_ptr(),
                    self.capacity,
                );
                data.set_len(self.capacity);
            }
        }
        core::mem::forget(safe_allocator);
        
        Ok(VulkanWaitHandle { data })
    }
}

/// TODO: Document this item
pub struct VulkanList<T> {
    pub buffer: vk::Buffer,
    pub address: u64,
    pub capacity: usize,
    pub allocation: Option<vk_mem::ffi::VmaAllocation>,
    pub allocator: Option<vk_mem::ffi::VmaAllocator>,
    _marker: core::marker::PhantomData<T>,
}

impl<T> Drop for VulkanList<T> {
    fn drop(&mut self) {
        if let (Some(alloc), Some(allocator)) = (self.allocation, self.allocator) {
            let safe_allocator = unsafe { vk_mem::AllocatorView::from_raw(allocator) };
            unsafe {
                let allocation = vk_mem::Allocation::from_raw(alloc);
                safe_allocator.destroy_buffer(self.buffer, &allocation);
                core::mem::forget(safe_allocator);
            }
        }
    }
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for VulkanList<T> {
    type Cmd = VulkanCommandBuffer;
    type ReadHandle<'a> = VulkanWaitHandle<Vec<T>> where Self: 'a, T: 'a;

    fn capacity(&self) -> usize { self.capacity }

    fn enqueue_read_to_cpu<'a>(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'a>> {
        let allocator = self.allocator.ok_or(EngineError::InvalidState("Missing VMA allocator"))?;
        let safe_allocator = unsafe { vk_mem::AllocatorView::from_raw(allocator) };
        let alloc = self.allocation.ok_or(EngineError::InvalidState("Missing allocation"))?;
        let allocation = unsafe { vk_mem::Allocation::from_raw(alloc) };
        
        let info = safe_allocator.get_allocation_info(&allocation);
        let mut data = Vec::with_capacity(self.capacity);
        unsafe {
            if !info.mapped_data.is_null() {
                core::ptr::copy_nonoverlapping(
                    info.mapped_data as *const T,
                    data.as_mut_ptr(),
                    self.capacity,
                );
                data.set_len(self.capacity);
            }
        }
        core::mem::forget(safe_allocator);
        
        Ok(VulkanWaitHandle { data })
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
    pub allocation: Option<vk_mem::ffi::VmaAllocation>,
    pub allocator: Option<vk_mem::ffi::VmaAllocator>,
}

impl Drop for VulkanMotionBvh {
    fn drop(&mut self) {
        if let (Some(alloc), Some(allocator)) = (self.allocation, self.allocator) {
            let safe_allocator = unsafe { vk_mem::AllocatorView::from_raw(allocator) };
            unsafe {
                let allocation = vk_mem::Allocation::from_raw(alloc);
                safe_allocator.destroy_buffer(self.buffer, &allocation);
                core::mem::forget(safe_allocator);
            }
        }
    }
}

impl DeviceBvh for VulkanMotionBvh {
    type Cmd = VulkanCommandBuffer;
}

/// TODO: Document this item
pub struct VulkanComputeKernels {
    pub device: ash::Device,
    pub pipelines: PhysicsPipelines,
    pub addresses: PhysicsDeviceAddresses,
    pub allocator: Option<vk_mem::ffi::VmaAllocator>,
    pub thread_pool: alloc::sync::Arc<aethervk_oshal_rlib::os::pool::ThreadPool>,
}

impl VulkanComputeKernels {
    fn allocate_and_upload<T: Copy + Send + Sync>(&self, data: &[T], usage: vk::BufferUsageFlags) -> EngineResult<VulkanBuffer<T>> {
        let allocator = if let Some(alloc) = self.allocator {
            unsafe { vk_mem::AllocatorView::from_raw(alloc) }
        } else {
            return Err(EngineError::InvalidState("Missing VMA allocator"));
        };

        let size = (core::mem::size_of::<T>() * data.len().max(1)) as u64;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        let mut alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            flags: vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE | vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };
        crate::apply_test_dedicated_alloc!(alloc_info);

        let (buffer, alloc, info) = unsafe { allocator.create_buffer_get_info(&buffer_info, &alloc_info) }
            .map_err(|_| EngineError::Gpu(GpuError::OutOfMemory))?;

        if !data.is_empty() {
            unsafe {
                core::ptr::copy_nonoverlapping(
                    data.as_ptr() as *const u8,
                    info.mapped_data as *mut u8,
                    size as usize,
                );
            }
        }

        core::mem::forget(allocator);

        let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe { self.device.get_buffer_device_address(&device_address_info) };

        Ok(VulkanBuffer {
            buffer,
            address,
            capacity: data.len().max(1),
            allocation: Some(alloc.as_raw()),
            allocator: self.allocator,
            _marker: core::marker::PhantomData,
        })
    }
    fn allocate_device_buffer<T: Copy + Send + Sync>(&self, capacity: usize, usage: vk::BufferUsageFlags) -> EngineResult<VulkanBuffer<T>> {
        let allocator = if let Some(alloc) = self.allocator {
            unsafe { vk_mem::AllocatorView::from_raw(alloc) }
        } else {
            return Err(EngineError::InvalidState("Missing VMA allocator"));
        };

        let size = (core::mem::size_of::<T>() * capacity.max(1)) as u64;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

        let mut alloc_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::AutoPreferDevice,
            ..Default::default()
        };
        crate::apply_test_dedicated_alloc!(alloc_info);

        let (buffer, alloc, _) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }
            .map_err(|_| EngineError::Gpu(GpuError::OutOfMemory))?;

        core::mem::forget(allocator);

        let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
        let address = unsafe { self.device.get_buffer_device_address(&device_address_info) };

        Ok(VulkanBuffer {
            buffer,
            address,
            capacity: capacity.max(1),
            allocation: Some(alloc.as_raw()),
            allocator: self.allocator,
            _marker: core::marker::PhantomData,
        })
    }
}

impl Kernels for VulkanComputeKernels {
    type Cmd = VulkanCommandBuffer;
    type Buffer<T: Copy + Send + Sync> = VulkanBuffer<T>;
    type List<T: Copy + Send + Sync> = VulkanList<T>;
    type MotionBvh = VulkanMotionBvh;

    fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(self.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let cmd = unsafe { self.device.allocate_command_buffers(&alloc_info) }
            .map_err(|_| EngineError::Gpu(GpuError::OutOfMemory))?[0];

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { self.device.begin_command_buffer(cmd, &begin_info).map_err(|_| EngineError::Gpu(GpuError::InvalidState("Begin CMD error".to_string())))? };

        Ok(VulkanCommandBuffer { 
            cmd,
            device: self.device.clone(),
            queue: self.queue,
            command_pool: self.command_pool,
        })
    }

    fn build_kinematic_bodies(
        &self,
        _cmd: &mut Self::Cmd,
        _scene: &PhysicsScene,
        scene0: &Scene,
    ) -> EngineResult<Self::Buffer<KinematicBody>> {
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
                .unwrap_or(aethervk_oshal_rlib::math::vector::vec3::Vec3f32::zero());
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

        self.allocate_and_upload(&bodies, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    fn build_rigid_bodies(
        &self,
        _cmd: &mut Self::Cmd,
        _scene: &PhysicsScene,
        scene0: &Scene,
    ) -> EngineResult<Self::Buffer<RigidBodyGpu>> {
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

                bodies.push(RigidBodyGpu {
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

        self.allocate_and_upload(&bodies, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    fn build_particles(
        &self,
        _cmd: &mut Self::Cmd,
        scene0: &Scene,
    ) -> EngineResult<Self::Buffer<ParticleGpu>> {
        let mut bodies = Vec::new();
        scene0.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
            |entity, _transform, sys| {
                use aethervk_oshal_rlib::math::vector::Vector;
                let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
                let particles = sys.particles.read();
                for (i, p) in particles.iter().enumerate().filter(|(_, p)| p.active != 0) {
                    bodies.push(ParticleGpu {
                        position: p.position,
                        velocity: p.velocity,
                        mass: p.mass,
                        force: [0.0, 0.0, 0.0],
                        entity_id: entity,
                        parent_frame_id: parent_id,
                        original_index: i as u32,
                    });
                }
            }
        );

        self.allocate_and_upload(&bodies, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    fn build_emitters(
        &self,
        _cmd: &mut Self::Cmd,
        scene0: &Scene,
    ) -> EngineResult<Self::Buffer<ForceEmitter>> {
        let mut emitters = Vec::new();
        scene0.query2::<crate::scene::TransformComponent, crate::scene::ForceEmitterComponent, _>(
        |_, t, emitter| match emitter {
            crate::scene::ForceEmitterComponent::Gravity { mu } => {
            emitters.push(ForceEmitter {
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
            use aethervk_oshal_rlib::math::vector::Vector;
            emitters.push(ForceEmitter {
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

        self.allocate_and_upload(&emitters, vk::BufferUsageFlags::STORAGE_BUFFER)
    }

    fn emit_particles(
        &self,
        cmd: &mut Self::Cmd,
        particles: &mut Self::Buffer<ParticleGpu>,
        _physical_scene: &PhysicsScene,
        _scene: &Scene,
        sun_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
        dt: timeus_t,
    ) -> EngineResult<()> {
        let max_particles = particles.capacity() as u32;
        let wg_size = 256;
        let num_emitters = 1; // Passed dynamically in reality
        let dispatch_groups = (max_particles + wg_size - 1) / wg_size;
        let dt_sec = dt as f32 / 1_000_000.0;
        
        let pc = EmitParticlesPushConstants {
            particles: self.addresses.particle_data,
            emitters: self.addresses.emitters,
            bvh_nodes: self.addresses.bvh_nodes,
            sun_pos: [sun_pos.x(), sun_pos.y(), sun_pos.z()],
            dt: dt_sec,
            max_particles,
            num_emitters,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<EmitParticlesPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.emit_particles);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        
        Ok(())
    }

    fn step_ode_p1_p2(
        &self,
        cmd: &mut Self::Cmd,
        particles: &mut Self::Buffer<ParticleGpu>,
        dt: timeus_t,
    ) -> EngineResult<()> {
        let wg_size = 256;
        let total_particles = particles.capacity() as u32;
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

    fn step_ode_p3_p4(
        &self,
        cmd: &mut Self::Cmd,
        kinematics: &Self::Buffer<KinematicBody>,
        rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
        _emitters: &Self::Buffer<ForceEmitter>,
        dt: timeus_t,
    ) -> EngineResult<()> {
        let wg_size = 256;
        let total_rigid_bodies = rigid_bodies.capacity() as u32;
        let dispatch_groups = (total_rigid_bodies + wg_size - 1) / wg_size;
        let dt_sec = dt as f32 / 1_000_000.0;

        let pc = P34PushConstants {
            rigid_bodies: self.addresses.rigid_body_data,
            emitters: self.addresses.emitters,
            kinematics: self.addresses.particle_data, // Using a dummy address here until kinematics is tracked in pipeline properly, or we can use kinematics.address if it was passed.
            dt: dt_sec,
            total_rigid_bodies,
            num_emitters: 1, // TODO dynamic
            num_kinematics: 0,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<P34PushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.p3_4_imex);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            self.device.cmd_dispatch(cmd.cmd, dispatch_groups, 1, 1);

            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        Ok(())
    }
    fn compute_self_gravity(
        &self,
        cmd: &mut Self::Cmd,
        _bvh: &Self::MotionBvh,
        particles: &mut Self::Buffer<ParticleGpu>,
    ) -> EngineResult<()> {
        let total_particles = particles.capacity() as u32;
        let dispatch_groups = (total_particles + 127) / 128;

        let pc_bh = BarnesHutPushConstants {
            particles: self.addresses.particle_data,
            bvh: self.addresses.bvh_nodes,
            root_index: 0,
            total_particles,
            theta: 0.5,
            g: crate::simulation::constants::G as f32,
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
        Ok(())
    }

    fn step_ode_p5(
        &self,
        cmd: &mut Self::Cmd,
        kinematics: &Self::Buffer<KinematicBody>,
        particles: &mut Self::Buffer<ParticleGpu>,
        _emitters: &Self::Buffer<ForceEmitter>,
        dt: timeus_t,
    ) -> EngineResult<()> {
        let wg_size = 256;
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
            num_emitters: 1, // TODO dynamic
            num_kinematics,
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
        cmd: &mut Self::Cmd,
        _kinematics: &Self::Buffer<KinematicBody>,
        _rigid_bodies: &Self::Buffer<RigidBodyGpu>,
        particles: &Self::Buffer<ParticleGpu>,
        dt: timeus_t,
    ) -> EngineResult<Self::MotionBvh> {
        let total_particles = particles.capacity() as u32;
        let wg_size = 256;
        let dispatch_groups = (total_particles + wg_size - 1) / wg_size;

        let num_nodes = (total_particles * 2).max(1) as usize;
        let mut bvh_buffer = self.allocate_device_buffer::<crate::gpu::BvhNodeAABBGpu>(num_nodes, vk::BufferUsageFlags::STORAGE_BUFFER)?;

        let pc = LbvhPushConstants {
            bvh_addr: bvh_buffer.address, // self.addresses.bvh_nodes,
            sorted_morton_addr: self.addresses.sorted_morton,
            counters_addr: self.addresses.atomic_counters,
            particles_addr: particles.address, // self.addresses.particle_data,
            num_primitives: total_particles,
            particle_radius: 1.0, 
            dt: dt.as_seconds_f32(),
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

        let alloc = bvh_buffer.allocation.take();
        Ok(VulkanMotionBvh { buffer: bvh_buffer.buffer, allocation: alloc, allocator: self.allocator })
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

        let max_candidates = 10000; // Placeholder
        let mut candidates_buffer = self.allocate_device_buffer::<CollisionPair>(max_candidates, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER)?;

        let pc = crate::gpu::compute_push_constants::BroadPhasePushConstants {
            tlas_bvh_addr: self.addresses.bvh_nodes,
            scene_entities_addr: self.addresses.particle_data, // Placeholder
            overlapping_pairs_addr: candidates_buffer.address, // self.addresses.ccd_candidates,
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
        
        let alloc = candidates_buffer.allocation.take();
        Ok(VulkanList { buffer: candidates_buffer.buffer, address: candidates_buffer.address, capacity: candidates_buffer.capacity, allocation: alloc, allocator: self.allocator, _marker: core::marker::PhantomData })
    }

    fn intersect_instances(
      &self,
      cmd: &mut Self::Cmd,
      potentials: &Self::List<CollisionPair>,
      _kinematics: &Self::Buffer<KinematicBody>,
      _rigid_bodies: &Self::Buffer<RigidBodyGpu>,
      _particles: &Self::Buffer<ParticleGpu>,
    ) -> EngineResult<Self::List<CollisionPair>> {
        let max_contacts = 10000; // Placeholder
        let mut output_list = self.allocate_device_buffer::<CollisionPair>(max_contacts, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER)?;

        let pc = CcdPushConstants {
            particle_bvh: self.addresses.bvh_nodes,
            output_list: output_list.address, // self.addresses.ccd_candidates,
            root_index: 0,
            total_particles: 10000, // Should be passed dynamically
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<CcdPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.ccd);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            
            // Dispatch indirect using the potentials buffer
            self.device.cmd_dispatch_indirect(cmd.cmd, potentials.buffer, 0);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        
        let alloc = output_list.allocation.take();
        Ok(VulkanList { buffer: output_list.buffer, address: output_list.address, capacity: output_list.capacity, allocation: alloc, allocator: self.allocator, _marker: core::marker::PhantomData })
    }

    fn compact_collisions(
        &self,
        cmd: &mut Self::Cmd,
        globals: &Self::List<CollisionPair>,
        _time_delta: timeus_t,
    ) -> EngineResult<Self::List<CollisionPair>> {
        let total_elements = globals.capacity() as u32;
        let wg_size = 256;
        let dispatch_groups = (total_elements + wg_size - 1) / wg_size;

        let max_packed = total_elements as usize; // Max possible is all valid
        let mut packed_out = self.allocate_device_buffer::<CollisionPair>(max_packed, vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER)?;

        let pc = StreamCompactPushConstants {
            sparse_in: globals.address, // self.addresses.ccd_candidates,
            packed_out: packed_out.address, // self.addresses.packed_collisions,
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
        
        let alloc = packed_out.allocation.take();
        Ok(VulkanList { buffer: packed_out.buffer, address: packed_out.address, capacity: packed_out.capacity, allocation: alloc, allocator: self.allocator, _marker: core::marker::PhantomData })
    }

    fn find_earliest_collision(
        &self,
        cmd: &mut Self::Cmd,
        compacted: &Self::List<CollisionPair>,
    ) -> EngineResult<Self::Buffer<timeus_t>> {
        let mut out_toi = self.allocate_device_buffer::<timeus_t>(1, vk::BufferUsageFlags::STORAGE_BUFFER)?;

        let pc = ReduceToiPushConstants {
            particles: self.addresses.particle_data,
            collisions: compacted.address, // self.addresses.packed_collisions,
            out_toi: out_toi.address, // self.addresses.reduce_toi,
            particle_radius: 1.0,
            dt: 0.016,
        };
        let bytes = unsafe { core::slice::from_raw_parts(&pc as *const _ as *const u8, core::mem::size_of::<ReduceToiPushConstants>()) };

        unsafe {
            self.device.cmd_bind_pipeline(cmd.cmd, vk::PipelineBindPoint::COMPUTE, self.pipelines.reduce_toi);
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes);
            
            self.device.cmd_dispatch_indirect(cmd.cmd, compacted.buffer, 0);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        
        let alloc = out_toi.allocation.take();
        Ok(VulkanBuffer { buffer: out_toi.buffer, address: out_toi.address, capacity: out_toi.capacity, allocation: alloc, allocator: self.allocator, _marker: core::marker::PhantomData })
    }

    fn apply_collision_responses(
      &self,
      cmd: &mut Self::Cmd,
      _kinematics: &Self::Buffer<KinematicBody>,
      _rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
      particles: &mut Self::Buffer<ParticleGpu>,
      collisions: &Self::List<CollisionPair>,
      force_inelastic: bool,
    ) -> EngineResult<()> {
        let max_contacts = collisions.capacity() as usize;
        let impulses_buffer = self.allocate_device_buffer::<[f32; 3]>(max_contacts, vk::BufferUsageFlags::STORAGE_BUFFER)?;

        let restitution_val = if force_inelastic { 0.0 } else { 0.5 };

        // LCP Solver
        let total_clusters = 100; // Unused when using indirect
        let pc_lcp = LcpPushConstants {
            particles: particles.address, // self.addresses.particle_data,
            collisions: collisions.address, // self.addresses.packed_collisions,
            outputs: impulses_buffer.address, // self.addresses.impulses,
            total_clusters,
            restitution: restitution_val,
        };
        let bytes_lcp = unsafe { core::slice::from_raw_parts(&pc_lcp as *const _ as *const u8, core::mem::size_of::<LcpPushConstants>()) };

        let pc_apply = ApplyImpulsesPushConstants {
            particles_addr: particles.address,
            collisions_addr: collisions.address,
            impulses_addr: impulses_buffer.address,
        };
        let bytes_apply = unsafe { core::slice::from_raw_parts(&pc_apply as *const _ as *const u8, core::mem::size_of::<ApplyImpulsesPushConstants>()) };

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
            self.device.cmd_push_constants(cmd.cmd, self.pipelines.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, bytes_apply);
            self.device.cmd_dispatch_indirect(cmd.cmd, collisions.buffer, 0);
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::COMPUTE_SHADER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }

        Ok(())
    }

    fn snapshot_dynamics(
        &self,
        cmd: &mut Self::Cmd,
        rigid_bodies: &Self::Buffer<RigidBodyGpu>,
        particles: &Self::Buffer<ParticleGpu>,
    ) -> EngineResult<(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>)> {
        let rb_snap = self.allocate_device_buffer::<RigidBodyGpu>(rigid_bodies.capacity(), vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
        let p_snap = self.allocate_device_buffer::<ParticleGpu>(particles.capacity(), vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)?;
        
        let rb_copy = vk::BufferCopy::default().size((rigid_bodies.capacity().max(1) * core::mem::size_of::<RigidBodyGpu>()) as u64);
        let p_copy = vk::BufferCopy::default().size((particles.capacity().max(1) * core::mem::size_of::<ParticleGpu>()) as u64);
        
        unsafe {
            if rigid_bodies.capacity() > 0 {
                self.device.cmd_copy_buffer(cmd.cmd, rigid_bodies.buffer, rb_snap.buffer, core::slice::from_ref(&rb_copy));
            }
            if particles.capacity() > 0 {
                self.device.cmd_copy_buffer(cmd.cmd, particles.buffer, p_snap.buffer, core::slice::from_ref(&p_copy));
            }
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        
        Ok((rb_snap, p_snap))
    }

    fn restore_dynamics(
        &self,
        cmd: &mut Self::Cmd,
        rigid_bodies: &mut Self::Buffer<RigidBodyGpu>,
        particles: &mut Self::Buffer<ParticleGpu>,
        snapshot: &(Self::Buffer<RigidBodyGpu>, Self::Buffer<ParticleGpu>),
    ) -> EngineResult<()> {
        let rb_copy = vk::BufferCopy::default().size((rigid_bodies.capacity().max(1) * core::mem::size_of::<RigidBodyGpu>()) as u64);
        let p_copy = vk::BufferCopy::default().size((particles.capacity().max(1) * core::mem::size_of::<ParticleGpu>()) as u64);
        
        unsafe {
            if rigid_bodies.capacity() > 0 {
                self.device.cmd_copy_buffer(cmd.cmd, snapshot.0.buffer, rigid_bodies.buffer, core::slice::from_ref(&rb_copy));
            }
            if particles.capacity() > 0 {
                self.device.cmd_copy_buffer(cmd.cmd, snapshot.1.buffer, particles.buffer, core::slice::from_ref(&p_copy));
            }
            
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
            self.device.cmd_pipeline_barrier(cmd.cmd, vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::COMPUTE_SHADER, vk::DependencyFlags::empty(), core::slice::from_ref(&barrier), &[], &[]);
        }
        
        Ok(())
    }

    fn write_back_to_scene(
        &self,
        cmd: &mut Self::Cmd,
        rigid_bodies: &Self::Buffer<RigidBodyGpu>,
        particles: &Self::Buffer<ParticleGpu>,
        _physical_scene: &mut PhysicsScene,
        scene: &Scene,
    ) -> EngineResult<()> {
        let rb_handle = rigid_bodies.enqueue_read_to_cpu(cmd)?;
        let p_handle = particles.enqueue_read_to_cpu(cmd)?;
        
        cmd.submit()?;
        
        let rb_data = rb_handle.wait()?;
        let p_data = p_handle.wait()?;

        scene.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
            |entity, _transform, sys| {
                let mut sys_particles = sys.particles.write();
                for p_gpu in p_data.iter() {
                    if p_gpu.entity_id == entity {
                        if (p_gpu.original_index as usize) < sys_particles.len() {
                            sys_particles[p_gpu.original_index as usize].position = p_gpu.position;
                            sys_particles[p_gpu.original_index as usize].velocity = p_gpu.velocity;
                        }
                    }
                }
            }
        );

        for rb in rb_data.iter() {
            let _ = scene.with_component_mut(rb.entity_id, |trans: &mut crate::scene::TransformComponent| {
                trans.position = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.position);
                let mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_columns(
                    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(rb.rotation[0][0], rb.rotation[0][1], rb.rotation[0][2], 0.0),
                    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(rb.rotation[1][0], rb.rotation[1][1], rb.rotation[1][2], 0.0),
                    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(rb.rotation[2][0], rb.rotation[2][1], rb.rotation[2][2], 0.0),
                    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
                );
                trans.rotation = aethervk_oshal_rlib::math::vector::vec4::Quat::from_mat4(&mat);
            });
            let _ = scene.with_component_mut(rb.entity_id, |kin: &mut crate::scene::KinematicComponent| {
                kin.velocity = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.linear_velocity);
                kin.angular_velocity = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.angular_velocity);
            });
        }
        
        Ok(())
    }
}
omponents(rb.rotation[2][0], rb.rotation[2][1], rb.rotation[2][2], 0.0),
                    aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
                );
                trans.rotation = aethervk_oshal_rlib::math::vector::vec4::Quat::from_mat4(&mat);
            });
            let _ = scene.with_component_mut(rb.entity_id, |kin: &mut crate::scene::KinematicComponent| {
                kin.velocity = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.linear_velocity);
                kin.angular_velocity = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_array(rb.angular_velocity);
            });
        }
        
        Ok(())
    }
}
