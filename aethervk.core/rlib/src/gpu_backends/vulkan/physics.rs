//! Vulkan Backend Integration for the IMEX / LCP Physics Engine
//!
//! This module scaffolds the execution of the massive compute-shader pipeline.
//! It assumes Vulkan 1.1 with `VK_KHR_buffer_device_address` and `VK_KHR_shader_subgroup_basic`.

use crate::gpu::compute_push_constants::ApplyImpulsesPushConstants;
use crate::gpu::vulkan::device::{self, Device, LogicalDevice};
use crate::gpu::vulkan::utils;
use crate::gpu::{
  self, CollisionPair, CommandBuffer, DeviceBuffer, DeviceBvh, DeviceList, DynamicBody, Kernels,
  KinematicBody, WaitHandle,
};
use crate::gpu_err;
use crate::physics::physics_scene::PhysicsScene;
use crate::scene::Scene;
use crate::types::{EngineError, EngineResult, GpuError, GpuResult};
use aethervk_oshal_rlib::math::matrix::Matrix4;
use aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32;
use aethervk_oshal_rlib::math::vector::vec3::Vec3f32;
use aethervk_oshal_rlib::math::vector::vec4::{Quat, Vec4f32};
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4};
use aethervk_oshal_rlib::os;
use aethervk_oshal_rlib::os::time::timeus_t;
use alloc::string::ToString;
use alloc::vec::Vec;
use ash::vk;
use vk_mem::{Alloc, AllocatorView, AsAllocatorView};
use alloc::format;

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
  pub lbvh_prepass: vk::Pipeline,
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
      let spv_code =
        std::fs::read(spv_path).unwrap_or_else(|_| panic!("Failed to read {}", spv_path));
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
      .unwrap()[0];

      unsafe {
        device.destroy_shader_module(shader_module, None);
      }
      pipeline
    };

    // Need to adjust path depending on where the test runs from.
    // Assuming root of workspace or test dir. We'll use absolute-ish or relative to workspace.
    // For safety, let's use a known path relative to the crate root or check multiple.
    let dir_lock = crate::gpu::ASSET_DIR.read();
    let base_dir = dir_lock.as_ref().unwrap();
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

/// TODO: Document this item
pub struct VulkanCommandBuffer {
  pub cmd: vk::CommandBuffer,
  pub queue: device::Queue,
}

pub struct VulkanCommandBufferContext<'a> {
  pub device: &'a LogicalDevice,
  pub rollback: &'a mut utils::RollbackContext<'a>,
}

impl VulkanCommandBuffer {
  // TODO cleanup function
}

impl CommandBuffer for VulkanCommandBuffer {
  type Context<'a> = VulkanCommandBufferContext<'a>;

  fn submit(&mut self, ctx: &VulkanCommandBufferContext<'_>) -> EngineResult<()> {
    unsafe {
      ctx.device.end_command_buffer(self.cmd).map_err(|e| GpuError::from(e))?;
      // TODO insert fences for synchronization between multiple iterations of commands (eg. double
      // buffering with 2 semaphore. Fence synchronization only when interacting with CPU, if a CPU
      // download operation has been issued)
      let submit_info = vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&self.cmd));
      ctx
        .device
        .queue_submit(
          self.queue.handle,
          core::slice::from_ref(&submit_info),
          vk::Fence::null(), // TODO
        )
        .map_err(|e| GpuError::from(e))?;
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
  pub allocation: Option<vk_mem::Allocation>,
  pub allocator: Option<vk_mem::AllocatorView>,
  _marker: core::marker::PhantomData<T>,
}

impl<T> Drop for VulkanBuffer<T> {
  fn drop(&mut self) {
    if let (Some(mut alloc), Some(allocator)) = (self.allocation, self.allocator) {
      unsafe {
        allocator.destroy_buffer(self.buffer, &mut alloc);
      }
    }
  }
}

impl<T: Copy + Send + Sync> DeviceBuffer<T> for VulkanBuffer<T> {
  type Cmd = VulkanCommandBuffer;
  type ReadHandle<'a>
    = VulkanWaitHandle<Vec<T>>
  where
    Self: 'a,
    T: 'a;

  fn capacity(&self) -> usize {
    self.capacity
  }

  #[function_name::named]
  fn enqueue_read_to_cpu(&self, _cmd: &mut Self::Cmd) -> EngineResult<Self::ReadHandle<'_>> {
    let allocator = self.allocator.ok_or(gpu_err!("Missing VMA allocator"))?;
    let alloc = self.allocation.ok_or(gpu_err!("Missing allocation"))?;

    let info = allocator.get_allocation_info(&alloc);
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

    // TODO: enqueue read should create a fence onto the Cmd
    Ok(VulkanWaitHandle { data })
  }
}

impl<T: Copy + Send + Sync> DeviceList<T> for VulkanBuffer<T> {
  fn clear(&mut self, _cmd: &mut Self::Cmd) -> EngineResult<()> {
    Ok(())
  }
}

impl DeviceBvh for VulkanBuffer<()> {
  type Cmd = VulkanCommandBuffer;
}

/// TODO: Document this item
pub struct VulkanComputeKernels {
  pub pipelines: PhysicsPipelines,
  pub addresses: PhysicsDeviceAddresses,
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
    let size = (core::mem::size_of::<T>() * data.len().max(1)) as u64;
    let buffer_info = vk::BufferCreateInfo::default()
      .size(size)
      .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

    let mut alloc_info = vk_mem::AllocationCreateInfo {
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
    rollback.defer(move |device| unsafe {
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
      allocation: Some(alloc),
      allocator: Some(allocator),
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
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<VulkanBuffer<T>> {
    let size = (core::mem::size_of::<T>() * capacity.max(1)) as u64;
    let buffer_info = vk::BufferCreateInfo::default()
      .size(size)
      .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);

    let mut alloc_info = vk_mem::AllocationCreateInfo {
      usage: vk_mem::MemoryUsage::AutoPreferDevice,
      ..Default::default()
    };
    crate::apply_test_dedicated_alloc!(alloc_info);

    let (buffer, mut alloc) = unsafe { allocator.create_buffer(&buffer_info, &alloc_info) }?;
    rollback.defer(move |device| unsafe {
      allocator.destroy_buffer(buffer, &mut alloc);
    });

    let device_address_info = vk::BufferDeviceAddressInfo::default().buffer(buffer);
    let address =
      unsafe { device.buffer_device_address.get_buffer_device_address(&device_address_info) };

    Ok(VulkanBuffer {
      buffer,
      address,
      capacity: capacity.max(1),
      allocation: Some(alloc),
      allocator: Some(allocator),
      _marker: core::marker::PhantomData,
    })
  }

  // -- Methods From Kernel Trait implementation --

  #[function_name::named]
  fn create_command_buffer(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    compute_queue: device::Queue,
  ) -> GpuResult<<Device as gpu::Kernels>::Cmd> {
    // TODO transition to commands
    let alloc_info = vk::CommandBufferAllocateInfo::default()
      // .command_pool(self.command_pool)
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_buffer_count(1);

    use super::device::VulkanDebugNameExt;
    let cmd = unsafe { device.allocate_command_buffers(&alloc_info) }
      .map(|vec| vec[0])
      .with_name(&device, &alloc::format!("compute buffer"))?;
    // TODO: rollback defer with commands

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info)? };

    Ok(VulkanCommandBuffer {
      cmd,
      queue: compute_queue,
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

  fn build_particles(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    _cmd: &mut VulkanCommandBuffer,
    scene0: &Scene,
  ) -> GpuResult<VulkanBuffer<gpu::ParticleGpu>> {
    let mut bodies = Vec::new();
    scene0.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>(
            |entity, _transform, sys| {
                let parent_id = scene0.get_parent(entity).map(|id| slotmap::Key::data(&id).as_ffi() as u32).unwrap_or(0);
                let particles = sys.particles.read();
                for (i, p) in particles.iter().enumerate().filter(|(_, p)| p.active != 0) {
                    bodies.push(gpu::ParticleGpu {
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

    self.allocate_and_upload(
      device,
      allocator,
      &bodies,
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
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<gpu::ParticleGpu>,
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
      particles: self.addresses.particle_data,
      emitters: self.addresses.emitters,
      bvh_nodes: self.addresses.bvh_nodes,
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

  fn step_ode_p1_p2(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    particles: &mut VulkanBuffer<gpu::ParticleGpu>,
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
        self.pipelines.p1_2_imex,
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
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
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
      kinematics: self.addresses.particle_data, // Using a dummy address here until kinematics is tracked in pipeline properly, or we can use kinematics.address if it was passed.
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
        self.pipelines.p3_4_imex,
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
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    _bvh: &VulkanBuffer<()>,
    particles: &mut VulkanBuffer<gpu::ParticleGpu>,
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
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    kinematics: &VulkanBuffer<KinematicBody>,
    particles: &mut VulkanBuffer<gpu::ParticleGpu>,
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
        self.pipelines.p5_imex,
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
    _rigid_bodies: &VulkanBuffer<gpu::RigidBodyGpu>,
    particles: &VulkanBuffer<gpu::ParticleGpu>,
    dt: timeus_t,
  ) -> GpuResult<VulkanBuffer<()>> {
    let total_particles = particles.capacity() as u32;
    let wg_size = 128;
    let dispatch_groups = (total_particles + wg_size - 1) / wg_size;

    let num_nodes = (total_particles * 2).max(1) as usize;
    let bvh_buffer = self.allocate_device_buffer::<()>(
      device,
      allocator,
      num_nodes,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )?;

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

    Ok(bvh_buffer)
  }

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
    let wg_size = 32; // TODO we are approximating a warp. rework the broad_phase shader for 128
    let dispatch_groups = (total_entities + wg_size - 1) / wg_size;

    let max_candidates = 10000; // Placeholder TODO parameter of kernels?
    let mut candidates_buffer = self.allocate_device_buffer::<gpu::CollisionPair>(
      device,
      allocator,
      max_candidates,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
      rollback,
    )?;

    let pc = crate::gpu::compute_push_constants::BroadPhasePushConstants {
      tlas_bvh_addr: self.addresses.bvh_nodes,
      scene_entities_addr: self.addresses.particle_data, // Placeholder
      overlapping_pairs_addr: candidates_buffer.address, // self.addresses.ccd_candidates,
      tlas_root_index: 0,
      total_entities,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<crate::gpu::compute_push_constants::BroadPhasePushConstants>(),
      )
    };

    unsafe {
      device.cmd_bind_pipeline(
        cmd.cmd,
        vk::PipelineBindPoint::COMPUTE,
        self.pipelines.broad_phase,
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

  fn intersect_instances(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    potentials: &VulkanBuffer<gpu::CollisionPair>,
    _kinematics: &VulkanBuffer<gpu::KinematicBody>,
    _rigid_bodies: &VulkanBuffer<gpu::RigidBodyGpu>,
    _particles: &VulkanBuffer<gpu::ParticleGpu>,
  ) -> GpuResult<VulkanBuffer<gpu::CollisionPair>> {
    let max_contacts = 10000; // Placeholder
    let mut output_list = self.allocate_device_buffer::<gpu::CollisionPair>(
      device,
      allocator,
      max_contacts,
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
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
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
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

  fn find_earliest_collision(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    compacted: &VulkanBuffer<gpu::CollisionPair>,
  ) -> GpuResult<VulkanBuffer<timeus_t>> {
    let mut out_toi = self.allocate_device_buffer::<timeus_t>(
      device,
      allocator,
      1,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )?;

    let pc = ReduceToiPushConstants {
      particles: self.addresses.particle_data,
      collisions: compacted.address, // self.addresses.packed_collisions,
      out_toi: out_toi.address,      // self.addresses.reduce_toi,
      particle_radius: 1.0,
      dt: 0.016,
    };
    let bytes = unsafe {
      core::slice::from_raw_parts(
        &pc as *const _ as *const u8,
        core::mem::size_of::<ReduceToiPushConstants>(),
      )
    };

    unsafe {
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

    Ok(out_toi)
  }

  fn apply_collision_responses(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    _kinematics: &VulkanBuffer<gpu::KinematicBody>,
    rigid_bodies: &VulkanBuffer<gpu::RigidBodyGpu>,
    particles: &mut VulkanBuffer<gpu::ParticleGpu>,
    collisions: &VulkanBuffer<gpu::CollisionPair>,
    force_inelastic: bool,
  ) -> GpuResult<()> {
    let max_contacts = collisions.capacity() as usize;
    // TODO where are contact forces? How do you know to which body/particle they are applied to?
    // TODO how do you know the frame of reference of these forces (not only that, Macro or micro
    // frame?)
    let impulses_buffer = self.allocate_device_buffer::<[f32; 3]>(
      device,
      allocator,
      max_contacts,
      vk::BufferUsageFlags::STORAGE_BUFFER,
      rollback,
    )?;

    let restitution_val = if force_inelastic { 0.0 } else { 0.5 };

    // LCP Solver
    let total_clusters = 100; // Unused when using indirect
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

    Ok(())
  }

  fn snapshot_dynamics(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &VulkanBuffer<gpu::RigidBodyGpu>,
    particles: &VulkanBuffer<gpu::ParticleGpu>,
  ) -> GpuResult<(
    VulkanBuffer<gpu::RigidBodyGpu>,
    VulkanBuffer<gpu::ParticleGpu>,
  )> {
    let rb_snap = self.allocate_device_buffer::<gpu::RigidBodyGpu>(
      device,
      allocator,
      rigid_bodies.capacity(),
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
      rollback,
    )?;
    let p_snap = self.allocate_device_buffer::<gpu::ParticleGpu>(
      device,
      allocator,
      particles.capacity(),
      vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
      rollback,
    )?;

    let rb_copy = vk::BufferCopy::default()
      .size((rigid_bodies.capacity().max(1) * core::mem::size_of::<gpu::RigidBodyGpu>()) as u64);
    let p_copy = vk::BufferCopy::default()
      .size((particles.capacity().max(1) * core::mem::size_of::<gpu::ParticleGpu>()) as u64);

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

  fn restore_dynamics(
    &self,
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &mut VulkanBuffer<gpu::RigidBodyGpu>,
    particles: &mut VulkanBuffer<gpu::ParticleGpu>,
    snapshot: &(
      VulkanBuffer<gpu::RigidBodyGpu>,
      VulkanBuffer<gpu::ParticleGpu>,
    ),
  ) -> GpuResult<()> {
    let rb_copy = vk::BufferCopy::default()
      .size((rigid_bodies.capacity().max(1) * core::mem::size_of::<gpu::RigidBodyGpu>()) as u64);
    let p_copy = vk::BufferCopy::default()
      .size((particles.capacity().max(1) * core::mem::size_of::<gpu::ParticleGpu>()) as u64);

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
    device: &LogicalDevice,
    allocator: vk_mem::AllocatorView,
    rollback: &mut utils::RollbackContext<'_>,
    cmd: &mut VulkanCommandBuffer,
    rigid_bodies: &VulkanBuffer<gpu::RigidBodyGpu>,
    particles: &VulkanBuffer<gpu::ParticleGpu>,
    _physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> GpuResult<()> {
    let rb_handle = rigid_bodies.enqueue_read_to_cpu(cmd).map_err(|e| gpu_err!("{}", e))?;
    let p_handle = particles.enqueue_read_to_cpu(cmd).map_err(|e| gpu_err!("{}", e))?;

    cmd.submit(&VulkanCommandBufferContext { device, rollback }).map_err(|e| gpu_err!("{}", e))?;

    let rb_data = rb_handle.wait().map_err(|e| gpu_err!("{}", e))?;
    let p_data = p_handle.wait().map_err(|e| gpu_err!("{}", e))?;

    scene.query2::<crate::scene::TransformComponent, crate::scene::particles::ParticleSystemComponent, _>( |entity, _transform, sys| {
      let mut sys_particles = sys.particles.write();
      for p_gpu in p_data.iter() {
        if p_gpu.entity_id == entity {
          if (p_gpu.original_index as usize) < sys_particles.len() {
            sys_particles[p_gpu.original_index as usize].position = p_gpu.position;
            sys_particles[p_gpu.original_index as usize].velocity = p_gpu.velocity;
          }
        }
      }
    });

    for rb in rb_data.iter() {
      let _ = scene.with_component_mut(
        rb.entity_id,
        |trans: &mut crate::scene::TransformComponent| {
          trans.position = Vec3f32::from_array(rb.position);
          let mat = Mat4x4f32::from_columns(
            Vec4f32::from_components(rb.rotation[0][0], rb.rotation[0][1], rb.rotation[0][2], 0.0),
            Vec4f32::from_components(rb.rotation[1][0], rb.rotation[1][1], rb.rotation[1][2], 0.0),
            Vec4f32::from_components(rb.rotation[2][0], rb.rotation[2][1], rb.rotation[2][2], 0.0),
            Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
          );
          trans.rotation = Quat::from_mat4(&mat);
        },
      );
      let _ = scene.with_component_mut(
        rb.entity_id,
        |kin: &mut crate::scene::KinematicComponent| {
          kin.velocity = Vec3f32::from_array(rb.linear_velocity);
          kin.angular_velocity = Vec3f32::from_array(rb.angular_velocity);
        },
      );
    }

    Ok(())
  }
}

impl Kernels for Device {
  type Cmd = VulkanCommandBuffer;
  type Buffer<T: Copy + Send + Sync> = VulkanBuffer<T>;
  type List<T: Copy + Send + Sync> = VulkanBuffer<T>;
  type MotionBvh = VulkanBuffer<()>;

  fn create_command_buffer(&self) -> EngineResult<Self::Cmd> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.create_command_buffer(
          &self.device,
          allocator,
          rollback,
          self.get_compute_queue(),
        )
      })
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_rigid_bodies(
    &self,
    cmd: &mut Self::Cmd,
    scene: &PhysicsScene,
    scene0: &Scene,
  ) -> EngineResult<Self::Buffer<gpu::RigidBodyGpu>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_rigid_bodies(&self.device, allocator, rollback, cmd, scene, scene0)
      })
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_particles(
    &self,
    cmd: &mut Self::Cmd,
    scene: &Scene,
  ) -> EngineResult<Self::Buffer<gpu::ParticleGpu>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.build_particles(&self.device, allocator, rollback, cmd, scene)
      })
      .commit_read(|res_guard, result| result)
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn emit_particles(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<gpu::ParticleGpu>,
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn step_ode_p1_p2(
    &self,
    cmd: &mut Self::Cmd,
    particles: &mut Self::Buffer<gpu::ParticleGpu>,
    dt: timeus_t,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.step_ode_p1_p2(&self.device, allocator, rollback, cmd, particles, dt)
      })
      .commit_read(|res_guard, result| result)
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn compute_self_gravity(
    &self,
    cmd: &mut Self::Cmd,
    bvh: &Self::MotionBvh,
    particles: &mut Self::Buffer<gpu::ParticleGpu>,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.compute_self_gravity(&self.device, allocator, rollback, cmd, bvh, particles)
      })
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn step_ode_p5(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    particles: &mut Self::Buffer<gpu::ParticleGpu>,
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn build_motion_bvh(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &Self::Buffer<gpu::RigidBodyGpu>,
    particles: &Self::Buffer<gpu::ParticleGpu>,
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn intersect_instances(
    &self,
    cmd: &mut Self::Cmd,
    potentials: &Self::List<gpu::CollisionPair>,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &Self::Buffer<gpu::RigidBodyGpu>,
    particles: &Self::Buffer<gpu::ParticleGpu>,
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
          potentials,
          kinematics,
          rigid_bodies,
          particles,
        )
      })
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn find_earliest_collision(
    &self,
    cmd: &mut Self::Cmd,
    compacted: &Self::List<gpu::CollisionPair>,
  ) -> EngineResult<Self::Buffer<timeus_t>> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.find_earliest_collision(&self.device, allocator, rollback, cmd, compacted)
      })
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn apply_collision_responses(
    &self,
    cmd: &mut Self::Cmd,
    kinematics: &Self::Buffer<gpu::KinematicBody>,
    rigid_bodies: &mut Self::Buffer<gpu::RigidBodyGpu>,
    particles: &mut Self::Buffer<gpu::ParticleGpu>,
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn snapshot_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<gpu::RigidBodyGpu>,
    particles: &Self::Buffer<gpu::ParticleGpu>,
  ) -> EngineResult<(
    Self::Buffer<gpu::RigidBodyGpu>,
    Self::Buffer<gpu::ParticleGpu>,
  )> {
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn restore_dynamics(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &mut Self::Buffer<gpu::RigidBodyGpu>,
    particles: &mut Self::Buffer<gpu::ParticleGpu>,
    snapshot: &(
      Self::Buffer<gpu::RigidBodyGpu>,
      Self::Buffer<gpu::ParticleGpu>,
    ),
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
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }

  fn write_back_to_scene(
    &self,
    cmd: &mut Self::Cmd,
    rigid_bodies: &Self::Buffer<gpu::RigidBodyGpu>,
    particles: &Self::Buffer<gpu::ParticleGpu>,
    physical_scene: &mut PhysicsScene,
    scene: &Scene,
  ) -> EngineResult<()> {
    utils::VulkanTransaction::new(&*self.res, &self.device)
      .prepare_read((), |res_guard, _| {
        Ok::<_, GpuError>(res_guard.allocator.allocator.as_allocator_view())
      })?
      .execute(|allocator, rollback| {
        self.kernels.write_back_to_scene(
          &self.device,
          allocator,
          rollback,
          cmd,
          rigid_bodies,
          particles,
          physical_scene,
          scene,
        )
      })
      .commit_read(|res_guard, result| result)
      .map_err(|e| EngineError::from(e))
  }
}
