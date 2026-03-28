use core::{
  cell::{Cell, RefCell},
  f32::consts::E,
  hash::{Hash, Hasher},
  num::NonZero,
  ops::Index,
  ptr::{self, NonNull},
  sync::atomic::{AtomicU64, Ordering},
};
use aethervk_oshal_rlib::{
  hash::FnvHasher,
  os::{
    fs::{self, Path, PathBuf},
    memory::{MaxAlignedStorage, StackAllocator},
    native::this_thread,
  },
};
use alloc::{format, string::ToString, sync::Arc, vec::Vec};

use crate::{
  gpu::{
    AcquireResult, CommandBufferHandle, GpuResourceHandle, PipelineKeyable,
    PresentationEngineHandle, RenderDevice, RenderableInstanceId, frame::ResourceUploadResult,
  },
  gpu_backends::vulkan::{
    self,
    device::{
      commands::CommandBufferId,
      memory::GlobalDeviceAllocator,
      pipelines::{
        FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization,
        StencilCompareOp, StencilLogicOp, VertexIn,
      },
      renderpasses::RenderPassSpecification,
      resources::{
        DiscardableResource, ForwardMeshRenderResource, ForwardMeshRenderResourceArchetype, Image,
        TextureFlags,
      },
      shader_manager::ShaderKey,
    },
    instance,
    utils::{self, NonZeroHandle},
  },
  scene::{EntityId, PhysicalMeshComponent, TransformComponent},
  simulation::comet::{
    Comet, NORMAL_COMPONENTS, POSITION_COMPONENTS, PushConstants, UV_COMPONENTS, Vertex,
  },
  types::{GpuError, GpuResult},
};

use ash::vk;
use heapless::{index_map::FnvIndexMap};

// companion classes inside Device. Each of these structs implement a given api
// taking as parameters devices and instances, and export a trait which reiterates
// the same interface without device and instance, implemented by `Device`
mod commands;
mod descriptors;
mod memory;
mod pipelines;
mod renderpasses;
mod resources;
mod shader_manager;
mod swapchain;

trait DeviceResource {
  /// Cleanup function to facilitate hierarchical manual Drop of resources
  /// without having to propagate through `Arc` or other means a reference
  /// to device handle and its function pointers
  /// Note: This function is not responsible to setup the proper state for cleanup (eg synchronization)
  fn cleanup(&mut self, device: &ash::Device);
}

struct FunctionalDeviceResource<H: ash::vk::Handle + Copy, F: FnOnce(H, &ash::Device)> {
  handle: H,
  cleanup: Option<F>,
}

impl<H: ash::vk::Handle + Copy, F: FnOnce(H, &ash::Device)> FunctionalDeviceResource<H, F> {
  fn new(handle: H, cleanup: F) -> Self {
    Self {
      handle,
      cleanup: Some(cleanup),
    }
  }
}

impl<H: ash::vk::Handle + Copy, F: FnOnce(H, &ash::Device)> DeviceResource
  for FunctionalDeviceResource<H, F>
{
  fn cleanup(&mut self, device: &ash::Device) {
    let h = self.handle;
    if let Some(cleanup) = self.cleanup.take() {
      cleanup(h, device);
    }
  }
}

struct DeviceResourceJanitor<'a, const N: usize> {
  device: &'a ash::Device,
  resources: heapless::Vec<NonNull<dyn DeviceResource + 'a>, N>,
  allocator: StackAllocator,
  storage: MaxAlignedStorage<N>,
}

impl<'a, const N: usize> DeviceResourceJanitor<'a, N> {
  fn new(device: &'a ash::Device) -> Self {
    Self {
      device,
      allocator: StackAllocator::new(),
      resources: heapless::Vec::new(),
      storage: MaxAlignedStorage([0; N]),
    }
  }

  pub fn clear(&mut self) {
    self.resources.clear();
  }

  pub fn push<T: DeviceResource + 'a>(&mut self, resource: T) -> Result<(), &'static str> {
    unsafe {
      let base_ptr = self.storage.0.as_mut_ptr();
      // 1. allocate within local storage array
      let ptr: *mut T = self.allocator.allocate(base_ptr, N, resource)?;

      // 2. Coerce to a fat pointer (*mut dyn Trait)
      let dyn_ptr: *mut (dyn DeviceResource + 'a) = ptr;

      // 3. wrap in NonNull
      let non_null = NonNull::new_unchecked(dyn_ptr);

      // 4. store fat pointer in vec
      self
        .resources
        .push(non_null)
        .map_err(|_| "Janitor capacity exceeded")?;
    }

    Ok(())
  }
}

impl<'a, const N: usize> Drop for DeviceResourceJanitor<'a, N> {
  fn drop(&mut self) {
    // destroy most recently allocated resources first
    for resource in self.resources.iter_mut().rev() {
      unsafe {
        let resource = resource.as_mut();
        resource.cleanup(self.device);

        core::ptr::drop_in_place(ptr::from_mut(resource));
      }
    }
  }
}

/// Safety: [`ForwardMeshRenderResourceArchetype`] should contain [`crate::gpu::PipelineKey`]
unsafe fn physical_mesh_resource_backend_to_frontend(
  value: &ForwardMeshRenderResource,
  archetype: &ForwardMeshRenderResourceArchetype,
) -> ResourceUploadResult {
  ResourceUploadResult {
    pipeline: unsafe { archetype.pipeline_key.unwrap_unchecked() },
    buffers: GpuResourceHandle(value.buffers_hash()),
  }
}

/// Device Resources. Each member here implements `DeviceResources` trait and is either
/// - implementing `Sync` and `Send`
/// - Wrapped into a RwLock/Mutex
/// - Native Vulkan Handle, externally synchronized
struct DeviceResources {
  allocator: memory::GlobalDeviceAllocator,
  discard_pool: resources::DiscardPool,
  live_presentation_engines: spin::RwLock<
    hashbrown::HashMap<PresentationEngineHandle, spin::RwLock<swapchain::PresentationState>>,
  >,
  command_pools:
    heapless::Vec<Option<Arc<commands::CommandPools>>, { utils::MAX_QUEUE_FAMILY_COUNT }>,
  descriptor_pool: Option<Arc<descriptors::DescriptorPools>>,
  pipeline_pool: spin::RwLock<pipelines::PipelinePool>,
  renderpasses: renderpasses::RenderPasses,
  timeline_semaphore: NonZeroHandle<vk::Semaphore>,
  timeline_semaphore_cached_value: AtomicU64,

  linear_sampler: NonZeroHandle<vk::Sampler>,

  shader_manager: spin::RwLock<shader_manager::ShaderManager>,

  physical_mesh_render_archetype: Option<ForwardMeshRenderResourceArchetype>,
  /// FScene (almost, more like a registry of all known static meshes)
  physical_mesh_resources:
    spin::RwLock<Option<hashbrown::HashMap<RenderableInstanceId, ForwardMeshRenderResource>>>,
}

impl DeviceResource for DeviceResources {
  /// cleanup in reverse order of declaration in the struct
  fn cleanup(&mut self, device: &ash::Device) {
    // all discardable resources should have been already discarded
    if self.has_discardables() {
      self.clear_discardables(&device);
    }
    self.discard_pool.cleanup(device);

    unsafe { device.destroy_semaphore(self.timeline_semaphore.get(), None) };

    self.renderpasses.cleanup(device);

    self.shader_manager.write().destroy(device);

    // Safety: If this is a properly constructed `DeviceResources`, then `descriptor_pool = Some(_)`
    assert!(Arc::strong_count(unsafe { self.descriptor_pool.as_ref().unwrap_unchecked() }) == 1);
    let mut descriptor_pool: descriptors::DescriptorPools =
      Arc::try_unwrap(unsafe { self.descriptor_pool.take().unwrap_unchecked() }).unwrap();
    descriptor_pool.cleanup(device);

    self.pipeline_pool.write().cleanup(device);

    for command_pool in self.command_pools.iter_mut() {
      assert!(Arc::strong_count(unsafe { command_pool.as_mut().unwrap_unchecked() }) == 1);
      let mut command_pool =
        unsafe { Arc::try_unwrap(command_pool.take().unwrap()).unwrap_unchecked() };
      command_pool.cleanup(device);
    }

    for (_, presentation_state) in self.live_presentation_engines.write().drain() {
      presentation_state.write().cleanup(device);
    }

    // - Linear Sampler
    unsafe { device.destroy_sampler(self.linear_sampler.get(), None) };

    self.allocator.cleanup(device);
  }
}

impl DeviceResources {
  /// update [`pipelines::FragmentOut`] and [`vk::RenderPass`] inside [`pipelines::GraphicsInfo`]
  /// disard old and create updated graphics [`vk::Pipeline`]
  /// Note: Update is performed only if archetype initialized once
  fn update_physical_mesh_archetype_for_presentation_engine(
    &mut self,
    device: &ash::Device,
    presentation_engine_state: &swapchain::PresentationState,
    timeline: u64,
  ) -> GpuResult<()> {
    if self.physical_mesh_render_archetype.is_none() {
      return Err(GpuError::InvalidState);
    }
    let archetype = unsafe {
      self
        .physical_mesh_render_archetype
        .as_mut()
        .unwrap_unchecked()
    };
    if archetype.graphics_info.is_none() || archetype.pipeline_key.is_none() {
      return Err(GpuError::InvalidState);
    }
    let pipeline_key = *unsafe { archetype.pipeline_key.as_ref().unwrap_unchecked() };
    let mut write_pipeline = self.pipeline_pool.write();

    let graphics_info = unsafe { archetype.graphics_info.as_mut().unwrap_unchecked() };
    let depth_stencil_format = graphics_info
      .fragment_out
      .depth_attachment_format
      .unwrap_or(vk::Format::UNDEFINED);

    graphics_info.fragment_out.color_attachment_formats.clear();
    graphics_info
      .fragment_out
      .color_attachment_formats
      .push(presentation_engine_state.format());
    graphics_info.render_pass = self
      .renderpasses
      .get_or_create_render_pass(
        RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
        0,
        device,
        &self.allocator.allocator,
        &self.discard_pool,
        timeline,
      )?
      .0
      .get();
    // Note: don't care about viewport and scissor cause they are dynamic state
    write_pipeline.get_or_create_graphics_pipeline(device, graphics_info)?;
    write_pipeline.discard_graphics_pipeline_if_present(pipeline_key, &self.discard_pool, timeline);

    let pipeline_key = graphics_info.pipeline_key();
    archetype.pipeline_key = Some(pipeline_key);

    Ok(())
  }

  fn get_physical_mesh_archetype(&self) -> Option<&'_ ForwardMeshRenderResourceArchetype> {
    self.physical_mesh_render_archetype.as_ref()
  }

  fn create_physical_mesh_archetype(
    &mut self,
    device: &ash::Device,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    if self.physical_mesh_render_archetype.is_some() {
      return Err(GpuError::InvalidState);
    }

    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock = live_presentation_engines_lock
      .get(&handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine_lock.read();
    if self.descriptor_pool.is_none() {
      return Err(GpuError::InvalidState);
    }

    let shader_manager = self.shader_manager.read();
    let vertex_shader = shader_manager
      .get(vertex_shader_key)
      .ok_or(GpuError::InvalidShader)?;
    if vertex_shader.shader_stage != vk::ShaderStageFlags::VERTEX {
      return Err(GpuError::InvalidShader);
    }
    let fragment_shader = shader_manager
      .get(fragment_shader_key)
      .ok_or(GpuError::InvalidShader)?;
    if fragment_shader.shader_stage != vk::ShaderStageFlags::FRAGMENT {
      return Err(GpuError::InvalidShader);
    }

    // Create initial struct
    let res = unsafe {
      ForwardMeshRenderResourceArchetype::new(
        self.descriptor_pool.as_ref().unwrap_unchecked(),
        device,
        &self.discard_pool,
        &vertex_shader,
        &fragment_shader,
      )?
    };
    self.physical_mesh_render_archetype = Some(res);

    // then populate graphics info and pipeline key
    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .add_binding(
            0,
            POSITION_COMPONENTS * size_of::<f32>() as u32,
            vk::VertexInputRate::VERTEX,
          )
          .add_binding(1, 9 * size_of::<f32>() as u32, vk::VertexInputRate::VERTEX)
          .add_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0) // inPosition
          .add_attribute(1, 1, vk::Format::R32G32B32_SFLOAT, 0) // inNormal
          .add_attribute(
            1,
            2,
            vk::Format::R32G32_SFLOAT,
            NORMAL_COMPONENTS * size_of::<f32>() as u32,
          ) // inUV
          .add_attribute(
            1,
            2,
            vk::Format::R32G32B32A32_SFLOAT,
            (NORMAL_COMPONENTS + UV_COMPONENTS) * size_of::<f32>() as u32,
          ) // inTangent
          .clone(),
      )
      .with_pre_rasterization(
        PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(vk::Viewport {
            width: presentation_engine_state.extent().0 as _,
            height: presentation_engine_state.extent().1 as _,
            x: 0.0,
            y: 0.0,
            min_depth: 0.0,
            max_depth: 1.0,
          })
          .add_scissors(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
              width: presentation_engine_state.extent().0,
              height: presentation_engine_state.extent().1,
            },
          })
          .clone(),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(presentation_engine_state.format())
          .with_depth_attachment_format(depth_stencil_format)
          .with_stencil_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(
        unsafe {
          self
            .physical_mesh_render_archetype
            .as_ref()
            .unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(PipelineFlags::CULL_BACK | PipelineFlags::STENCIL_ENABLE)
      .with_render_pass(
        self
          .renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            &self.allocator.allocator,
            &self.discard_pool,
            timeline,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .with_stencil_compare_op(StencilCompareOp::None)
      .with_stencil_logic_op(StencilLogicOp::Replace)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(0)
      .with_stencil_write_mask(u32::MAX)
      .clone();
    self
      .pipeline_pool
      .write()
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    self.physical_mesh_render_archetype = Some(
      unsafe {
        self
          .physical_mesh_render_archetype
          .take()
          .unwrap_unchecked()
      }
      .with_graphics_info(pipeline_graphics_info),
    );

    Ok(())
  }

  fn has_discardables(&self) -> bool {
    self.physical_mesh_render_archetype.is_some() && {
      let resources = self.physical_mesh_resources.read();
      !resources.is_none() && !unsafe { resources.as_ref().unwrap_unchecked() }.is_empty()
    }
  }

  fn clear_discardables(&mut self, device: &ash::Device) {
    debug_assert!(self.has_discardables());
    if let Some(mut archetype) = self.physical_mesh_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut resources) = self.physical_mesh_resources.write().take() {
      for (_, mut resource) in resources.drain() {
        resource.discard(device, &self.discard_pool, u64::MAX);
      }
    }
    debug_assert!(!self.has_discardables());
  }

  fn new<'a>(
    instance: &instance::Instance,
    physical_device: vk::PhysicalDevice,
    device: &ash::Device,
    unique_family_indices_iter: impl Iterator<Item = &'a u32>,
  ) -> GpuResult<Self> {
    // - VMA Device Allocator
    let mut allocator = unsafe {
      GlobalDeviceAllocator::new(
        &instance.instance,
        &device,
        physical_device,
        instance.api_version(),
      )
    }?;
    // - Timeline Semaphore
    let mut sem_type_info = vk::SemaphoreTypeCreateInfo::default()
      .initial_value(0)
      .semaphore_type(vk::SemaphoreType::TIMELINE);
    let sem_create_info = vk::SemaphoreCreateInfo::default().push_next(&mut sem_type_info);
    let timeline_semaphore = match unsafe { device.create_semaphore(&sem_create_info, None) } {
      Ok(semaphore) => semaphore,
      Err(e) => {
        allocator.cleanup(device);
        return Err(e.into());
      }
    };

    // - Descriptor Pool
    let mut descriptor_pool = match descriptors::DescriptorPools::new(device, 256) {
      Ok(pool) => pool,
      Err(e) => {
        unsafe { device.destroy_semaphore(timeline_semaphore, None) };
        allocator.cleanup(device);
        return Err(e);
      }
    };

    let renderpasses =
      renderpasses::RenderPasses::new(&instance.instance, &device, &allocator.allocator);

    // - Pipeline Pool (TODO: cache data?)
    let pipeline_pool = match pipelines::PipelinePool::new(device, None) {
      Ok(pool) => spin::RwLock::new(pool),
      Err(e) => {
        let descriptor_pool = unsafe { Arc::get_mut(&mut descriptor_pool).unwrap_unchecked() };
        descriptor_pool.cleanup(device);
        unsafe { device.destroy_semaphore(timeline_semaphore, None) };
        allocator.cleanup(device);
        return Err(e);
      }
    };

    // - Discard Pool
    let discard_pool = unsafe { resources::DiscardPool::new(64) };
    // - Command Pools
    let mut command_pools = heapless::Vec::new();
    for &queue_family_index in unique_family_indices_iter {
      unsafe {
        command_pools.push_unchecked(Some(Arc::new(commands::CommandPools::new(
          queue_family_index,
        ))))
      };
    }
    // - Swapchain hashmap
    let live_presentation_engines = spin::RwLock::new(hashbrown::HashMap::new());
    // - linear sampler
    let sampler_info = vk::SamplerCreateInfo::default()
      .mag_filter(vk::Filter::LINEAR)
      .min_filter(vk::Filter::LINEAR)
      .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
      .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
      .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
      .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let linear_sampler = unsafe { device.create_sampler(&sampler_info, None) }?;
    Ok(Self {
      allocator,
      command_pools,
      discard_pool,
      live_presentation_engines,
      descriptor_pool: Some(descriptor_pool),
      pipeline_pool,
      renderpasses,
      shader_manager: spin::RwLock::new(shader_manager::ShaderManager::new()),
      linear_sampler: unsafe { NonZeroHandle::new_unchecked(linear_sampler) },
      timeline_semaphore: unsafe { NonZeroHandle::new_unchecked(timeline_semaphore) },
      timeline_semaphore_cached_value: AtomicU64::new(0),
      physical_mesh_render_archetype: None,
      physical_mesh_resources: spin::RwLock::new(None),
    })
  }

  fn get_timeline_semaphore_cached_value(&self) -> u64 {
    self.timeline_semaphore_cached_value.load(Ordering::Relaxed)
  }

  fn refresh_timeline_semaphore_cached_value(
    &self,
    device: &ash::Device,
  ) -> ash::prelude::VkResult<()> {
    self.timeline_semaphore_cached_value.store(
      unsafe { device.get_semaphore_counter_value(self.timeline_semaphore.get()) }?,
      Ordering::Relaxed,
    );
    Ok(())
  }
}

#[derive(Clone, Copy)]
struct RecordingCmdBufferDataPresentation {
  acquire_result: AcquireResult,
  presentation_engine: PresentationEngineHandle,
}

struct RecordingCmdBufferData {
  command_buffer: NonZeroHandle<vk::CommandBuffer>,
  bound_pipeline: Option<NonZeroHandle<vk::Pipeline>>,
  presentation: Option<RecordingCmdBufferDataPresentation>,
}

impl RecordingCmdBufferData {
  fn new(command_buffer: NonZeroHandle<vk::CommandBuffer>) -> Self {
    Self {
      command_buffer,
      bound_pipeline: None,
      presentation: None,
    }
  }

  fn has_begun_renderpass(&self) -> bool {
    self.presentation.is_none()
  }

  /// command buffer is automatically recycled by [`commands::CommandPools`]
  fn discard(
    &mut self,
    cmd_buf_id: CommandBufferId,
    discard_pool: &resources::DiscardPool,
    cmd_pools: Arc<commands::CommandPools>,
    timeline: u64,
  ) {
    let tid = this_thread::id();
    if let Some(pipeline) = self.bound_pipeline {
      discard_pool.discard_pipeline(pipeline.get(), timeline);
    }
    discard_pool.discard_command_buffer(
      tid,
      cmd_buf_id,
      self.command_buffer.get(),
      cmd_pools,
      timeline,
    );
  }
}

pub(super) struct Device<'a> {
  query_result: utils::PhysicalDeviceQueryResult,
  pub device: ash::Device,
  queues: Queues,
  instance: &'a instance::Instance,

  create_renderpass2: ash::khr::create_renderpass2::Device,

  res: spin::RwLock<DeviceResources>,

  // Some bookkeeping I don't know where to put
  depth_stencil_format: vk::Format,
  /// Recording command buffers
  recording_command_buffers:
    spin::RwLock<hashbrown::HashMap<(u64, CommandBufferHandle), RecordingCmdBufferData>>,
}

const MAX_QUEUE_COUNT: usize = 4;

/// internal queue indicator for `Queues` struct to reference a given queue. Metadata is still held by QueryResult
/// These values are used as shift amounts for bitmasks
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum QueueId {
  GRAPHICS = 1,
  COMPUTE = 2,
  TRANSFER = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Queue {
  handle: vk::Queue,
  index: u32,
  family_index: u32,
}

// ~28 bytes per queue. total for `MAX_QUEUE_COUNT` = 4 at 96 bytes
#[ouroboros::self_referencing]
struct Queues {
  queue_buffer: heapless::Vec<Queue, MAX_QUEUE_COUNT>,
  #[borrows(queue_buffer)]
  #[covariant]
  queue_ref_map: FnvIndexMap<QueueId, &'this Queue, MAX_QUEUE_COUNT>,
}

impl Queues {
  fn from_device(device: &ash::Device, query_result: &utils::PhysicalDeviceQueryResult) -> Self {
    let unique_queue_families = query_result.unique_family_indices_set();
    let mut queue_buffer: heapless::Vec<Queue, MAX_QUEUE_COUNT> = heapless::Vec::new();
    for &family_index in unique_queue_families.iter() {
      let queue_info = vk::DeviceQueueInfo2::default()
        .queue_family_index(family_index)
        .queue_index(0);
      let handle = unsafe { device.get_device_queue2(&queue_info) };
      unsafe {
        queue_buffer.push_unchecked(Queue {
          handle,
          index: 0,
          family_index,
        })
      };
    }

    QueuesBuilder {
      queue_buffer,
      queue_ref_map_builder: |queue_buffer: &heapless::Vec<_, _>| {
        let mut queue_ref_map: FnvIndexMap<QueueId, &Queue, MAX_QUEUE_COUNT> = FnvIndexMap::new();
        let mut queue_type_inserted: u32 = 0;
        for i in 0..queue_buffer.len() {
          if (queue_type_inserted & (1u32 << QueueId::GRAPHICS as u32)) == 0
            && query_result.graphics_queue_family_index as usize == i
          {
            queue_ref_map
              .insert(QueueId::GRAPHICS, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::GRAPHICS as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::COMPUTE as u32)) == 0
            && query_result.compute_queue_family_index as usize == i
          {
            queue_ref_map
              .insert(QueueId::COMPUTE, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::COMPUTE as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::TRANSFER as u32)) == 0
            && query_result.transfer_queue_family_index as usize == i
          {
            queue_ref_map
              .insert(QueueId::TRANSFER, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::TRANSFER as u32;
          }
        }

        queue_ref_map
      },
    }
    .build()
  }

  fn get_graphics_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::GRAPHICS).unwrap())
  }

  fn get_compute_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::COMPUTE).unwrap())
  }

  fn get_transfer_queue(&self) -> Queue {
    self.with_queue_ref_map(|queue_ref_map| **queue_ref_map.get(&QueueId::TRANSFER).unwrap())
  }
}

fn reflect_to_vulkan_descriptor_type(
  reflect_ty: spirv_reflect::types::ReflectDescriptorType,
) -> vk::DescriptorType {
  use spirv_reflect::types::ReflectDescriptorType as Rt;
  match reflect_ty {
    Rt::Sampler => vk::DescriptorType::SAMPLER,
    Rt::CombinedImageSampler => vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
    Rt::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
    Rt::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
    Rt::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
    Rt::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
    Rt::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
    Rt::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
    Rt::UniformBufferDynamic => vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
    Rt::StorageBufferDynamic => vk::DescriptorType::STORAGE_BUFFER_DYNAMIC,
    Rt::InputAttachment => vk::DescriptorType::INPUT_ATTACHMENT,
    Rt::AccelerationStructureKHR => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
    _ => vk::DescriptorType::UNIFORM_BUFFER, // Fallback
  }
}

impl<'a> Device<'a> {
  /// Initializes a Device directly into the provided memory location
  /// This avoids returning a Device by value (which would probably cause stack overflow)
  pub(super) unsafe fn init_at_ptr(
    dst: *mut Self,
    instance: &'a vulkan::instance::Instance,
    index: usize,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<()> {
    unsafe { ptr::write(dst, Self::new(instance, index, query_input)?) };
    Ok(())
  }

  pub(super) fn new(
    instance: &'a vulkan::instance::Instance,
    index: usize,
    query_input: &utils::PhysicalDeviceQueryInput,
  ) -> GpuResult<Self> {
    let eligible_physical_devices = instance.get_eligible_devices(query_input)?;

    let chosen_physical_device_query_result = match eligible_physical_devices.get(index) {
      Some(chosen_physical_device_query_result) => Ok(chosen_physical_device_query_result),
      None => Err(GpuError::BackendSpecific(format!(
        "There isn't a Vulkan capable device at index {}",
        index
      ))),
    }?;
    let physical_device = chosen_physical_device_query_result.physical_device;

    // 1. enable required and TODO optional features
    let mut required_features = utils::RequiredFeatures::new();
    required_features.populate();
    let mut features2 = required_features.as_features2();

    // 2. Setup queue create infos for necessary queues from query result
    let queue_priorities = [1f32];
    let queue_infos_len = chosen_physical_device_query_result.family_count();
    let queue_infos: Vec<_> = (0..queue_infos_len)
      .map(|i| {
        vk::DeviceQueueCreateInfo::default()
          .queue_family_index(i as _)
          .queue_priorities(&queue_priorities)
      })
      .collect();

    // 3. Device creation
    let enabled_extension_names: Vec<_> =
      chosen_physical_device_query_result.enabled_extension_names();
    let device_create_info = vk::DeviceCreateInfo::default()
      .enabled_extension_names(&enabled_extension_names)
      .push_next(&mut features2)
      .queue_create_infos(&queue_infos);

    let device = unsafe {
      instance
        .instance
        .create_device(physical_device, &device_create_info, None)
    }?;

    let queues = Queues::from_device(&device, chosen_physical_device_query_result);
    let res = DeviceResources::new(
      instance,
      physical_device,
      &device,
      chosen_physical_device_query_result
        .unique_family_indices_set()
        .iter(),
    )?;

    // bookkeeping data instantiation
    let depth_stencil_format: vk::Format = 'block: {
      // specification says that at least one of D24/S8 or D32/S8 must be supported
      let mut props = vk::FormatProperties2::default();
      for f in [
        vk::Format::D24_UNORM_S8_UINT,
        vk::Format::D32_SFLOAT_S8_UINT,
      ] {
        unsafe {
          instance
            .instance
            .get_physical_device_format_properties2(physical_device, f, &mut props)
        };
        if props
          .format_properties
          .optimal_tiling_features
          .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        {
          break 'block Ok(f);
        }
      }
      // never reached
      Err(GpuError::UnsupportedFeature)
    }?;

    let create_renderpass2 = ash::khr::create_renderpass2::Device::new(&instance.instance, &device);
    Ok(Self {
      query_result: *chosen_physical_device_query_result,
      device,
      create_renderpass2,
      queues,
      res: res.into(),
      instance,
      depth_stencil_format,
      recording_command_buffers: spin::RwLock::new(hashbrown::HashMap::new()),
    })
  }

  pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
    self.query_result.physical_device
  }

  pub(super) fn with_device_allocator<T>(&self, f: impl FnOnce(&GlobalDeviceAllocator) -> T) -> T {
    let dalloc = &self.res.read().allocator;
    f(dalloc)
  }

  pub(super) fn with_device_allocator_mut<T>(
    &mut self,
    f: impl FnOnce(&mut GlobalDeviceAllocator) -> T,
  ) -> T {
    let dalloc = &mut self.res.write().allocator;
    f(dalloc)
  }

  fn ensure_physical_mesh_shader_modules(&self) -> GpuResult<(ShaderKey, ShaderKey)> {
    // TODO: proper path management
    let vert_path = PathBuf::from("assets/physical_mesh.vert.spv");
    let frag_path = PathBuf::from("assets/physical_mesh.frag.spv");

    let res = self.res.read();
    let mut shader_manager = res.shader_manager.write();

    let vert_key = shader_manager.get_or_load(
      &self.device,
      &vert_path,
      "main",
      spirv::ExecutionModel::Vertex,
    )?;
    let frag_key = shader_manager.get_or_load(
      &self.device,
      &frag_path,
      "main",
      spirv::ExecutionModel::Fragment,
    )?;

    Ok((vert_key, frag_key))
  }
}

impl<'a> Drop for Device<'a> {
  fn drop(&mut self) {
    unsafe { self.device.device_wait_idle().unwrap_unchecked() };

    self.res.write().cleanup(&self.device);

    // in the end, destroy the device
    unsafe { self.device.destroy_device(None) };
  }
}

impl<'a> RenderDevice for Device<'a> {
  #[cfg(debug_assertions)]
  fn print_info(&self) -> alloc::string::String {
    use alloc::format;

    let props = &self.query_result.physical_device_properties;
    let device_name = props
      .device_name_as_c_str()
      .unwrap()
      .to_string_lossy()
      .into_owned();
    let device_type = match props.device_type {
      vk::PhysicalDeviceType::CPU => "CPU",
      vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated GPU",
      vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual GPU",
      vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete GPU",
      _ => "Other",
    };

    let api_major = vk::api_version_major(props.api_version);
    let api_minor = vk::api_version_minor(props.api_version);
    let api_patch = vk::api_version_patch(props.api_version);

    format!(
      "Vulkan Device Info\n\
       ------------------\n\
       Name: {}\n\
       Vendor ID: {:#X} ({})\n\
       Device ID: {:#X}\n\
       Type: {}\n\
       API Version: {}.{}.{}\n\
       Driver Version: {}\n\
       Queue Families: {}\n",
      device_name,
      props.vendor_id,
      match props.vendor_id {
        0x10DE => "NVIDIA",
        0x1002 | 0x1022 => "AMD",
        0x106B => "Apple",
        0x8086 => "Intel",
        0x13B5 => "ARM",
        0x5143 => "Qualcomm",
        0x1010 => "ImgTec",
        _ => "Unknown",
      },
      props.device_id,
      device_type,
      api_major,
      api_minor,
      api_patch,
      props.driver_version,
      self.query_result.family_count()
    )
  }

  fn context_id(&self) -> u64 {
    vulkan::VULKAN_RENDER_BACKEND.0
  }

  fn start_frame(&self) -> GpuResult<()> {
    self
      .res
      .write()
      .refresh_timeline_semaphore_cached_value(&self.device)
      .map_err(|e| e.into())
  }

  fn create_presentation_engine(
    &self,
    params: &crate::gpu::PresentationEngineParams,
  ) -> GpuResult<crate::gpu::PresentationEngineHandle> {
    let entry =
      self
        .instance
        .entry_wrapper
        .weak_entry()
        .upgrade()
        .ok_or(GpuError::BackendSpecific(
          "Vulkan Entry wasn't loaded".to_string(),
        ))?;
    let physical_device_handle = unsafe { NonZeroHandle::new_unchecked(self.physical_device()) };
    let presentation_state = swapchain::PresentationState::new(
      &entry,
      &self.instance.instance,
      &self.device,
      physical_device_handle,
      params,
    )?;

    static NEXT_HANDLE: core::sync::atomic::AtomicU64 = core::sync::atomic::AtomicU64::new(1);
    let handle =
      PresentationEngineHandle(NEXT_HANDLE.fetch_add(1, core::sync::atomic::Ordering::Relaxed));

    self
      .res
      .write()
      .live_presentation_engines
      .write()
      .insert(handle, spin::RwLock::new(presentation_state));

    Ok(handle)
  }

  fn resize_presentation_engine(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
    width: u32,
    height: u32,
  ) -> GpuResult<()> {
    let res = self.res.read();
    if let Some(engine) = res.live_presentation_engines.read().get(&handle) {
      let entry =
        self
          .instance
          .entry_wrapper
          .weak_entry()
          .upgrade()
          .ok_or(GpuError::BackendSpecific(
            "Vulkan Entry wasn't loaded".to_string(),
          ))?;
      let physical_device_handle = unsafe { NonZeroHandle::new_unchecked(self.physical_device()) };
      engine.write().resize(
        &entry,
        &self.instance.instance,
        &self.device,
        physical_device_handle,
        width,
        height,
      )
    } else {
      Err(GpuError::InvalidArgument)
    }
  }

  fn acquire_next_image(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
  ) -> GpuResult<crate::gpu::AcquireResult> {
    if let Some(engine) = self
      .res
      .read()
      .live_presentation_engines
      .read()
      .get(&handle)
    {
      engine.write().acquire_next_image(&self.device)
    } else {
      Err(GpuError::InvalidArgument)
    }
  }

  fn present(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
    image_index: usize,
    frame_index: usize,
  ) -> GpuResult<crate::gpu::SwapchainStatus> {
    if let Some(engine) = self
      .res
      .read()
      .live_presentation_engines
      .read()
      .get(&handle)
    {
      let graphics_queue = self.queues.get_graphics_queue().handle;
      unsafe {
        engine
          .write()
          .submit_image(graphics_queue, image_index as u32, frame_index as u32)
      }
    } else {
      Err(GpuError::InvalidArgument)
    }
  }

  /// Note: This function may have the following side effects
  /// - Creation of VkBuffer/VkMemory through VMA for vertex and index buffer associated with given mesh
  /// - Creation of VkImage/VkMemory + VkImageView through VMA for each texture associated with given mesh
  /// for each instance of physical mesh requested to render.
  /// The following resources are instead for every physical mesh, and hence lazily initialized when the first
  /// physical mesh is requested to be rendered
  /// - VkPipeline, VkPipelineLayout, VkPushConstantRange, VkDescriptorSets
  /// - VkRenderPass (and associated VkFramebuffer), which are linked to swapchain,
  ///   hence possibly refreshed each time the swapchain is resized
  /// What is not created by the following function
  /// - VkCommandBuffer, which is instead created through the `record_commands` function from render_path
  /// Note: it assumes that you are preparing for the next frame
  fn get_or_create_physical_mesh_resources(
    &self,
    entity_id: EntityId,
    component: &PhysicalMeshComponent,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    // WARNING: if state isn't properly maintained, this is a nightmare
    let next_frame_timeline = self.res.read().get_timeline_semaphore_cached_value() + 1;
    let current_frame_timeline = next_frame_timeline - 1;

    // ensure that the archetype for physical meshes exists
    let exists = self.res.read().physical_mesh_render_archetype.is_some();
    if !exists {
      let mut wres = self.res.write();
      let (vkey, fkey) = self.ensure_physical_mesh_shader_modules()?;
      wres.create_physical_mesh_archetype(
        &self.device,
        vkey,
        fkey,
        self.depth_stencil_format,
        handle,
        next_frame_timeline,
      )?;
    }

    let res = self.res.read();
    let archetype = unsafe { res.get_physical_mesh_archetype().unwrap_unchecked() };

    // Safety: Archetype, once properly constructed, has everything populated
    let pipeline_key = unsafe { archetype.pipeline_key.unwrap_unchecked() };

    // Get rendering system Internal Mesh Identifier
    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(entity_id, component);

    // Does the mesh already exist? If so, return cached resource
    let read_resouces = res.physical_mesh_resources.read();
    if let Some(resources) = read_resouces.as_ref() {
      if let Some(resource) = resources.get(&physical_mesh_id) {
        unsafe {
          return Ok(physical_mesh_resource_backend_to_frontend(
            &resource, &archetype,
          ));
        }
      }
    }
    drop(read_resouces);

    // Otherwise, create it inside the resources registry
    let mut write_resources = res.physical_mesh_resources.write();
    if write_resources.is_none() {
      *write_resources = Some(hashbrown::HashMap::new());
    }

    let command_buffer: vk::CommandBuffer = {
      let cmd_buffer_handle = cmd_id_from_timeline_and_thread_id(current_frame_timeline);
      if let Some(cmd_buf) = self
        .recording_command_buffers
        .read()
        .get(&(current_frame_timeline, cmd_buffer_handle))
        .map(|v| v.command_buffer.get())
      {
        Ok::<vk::CommandBuffer, GpuError>(cmd_buf)
      } else {
        let cmd = unsafe {
          res
            .command_pools
            .get_unchecked(self.queues.get_graphics_queue().index as usize)
            .as_ref()
            .unwrap_unchecked()
            .allocate_primary(&self.device, this_thread::id(), cmd_buffer_handle.into())
        }?;
        self.recording_command_buffers.write().insert(
          (current_frame_timeline, cmd_buffer_handle),
          RecordingCmdBufferData::new(unsafe { NonZeroHandle::new_unchecked(cmd) }),
        );

        Ok(cmd)
      }
    }?;

    let position_data = extract_position_data(&component.mesh);
    let attribute_data = extract_attribute_data(&component.mesh);
    // TODO: log on error
    let albedo_image = component.mesh.albedo_map.as_ref().and_then(|t| {
      Image::new_2d(
        &self.device,
        &res.allocator.allocator,
        command_buffer,
        &res.discard_pool,
        current_frame_timeline,
        &t,
        vk::ImageUsageFlags::SAMPLED,
      )
      .ok()
    });
    let normal_image = component.mesh.normal_map.as_ref().and_then(|t| {
      Image::new_2d(
        &self.device,
        &res.allocator.allocator,
        command_buffer,
        &res.discard_pool,
        current_frame_timeline,
        &t,
        vk::ImageUsageFlags::SAMPLED,
      )
      .ok()
    });
    let roughness_image = component.mesh.roughness_map.as_ref().and_then(|t| {
      Image::new_2d(
        &self.device,
        &res.allocator.allocator,
        command_buffer,
        &res.discard_pool,
        current_frame_timeline,
        &t,
        vk::ImageUsageFlags::SAMPLED,
      )
      .ok()
    });
    let ao_image = component.mesh.ao_map.as_ref().and_then(|t| {
      Image::new_2d(
        &self.device,
        &res.allocator.allocator,
        command_buffer,
        &res.discard_pool,
        current_frame_timeline,
        &t,
        vk::ImageUsageFlags::SAMPLED,
      )
      .ok()
    });

    let resource = ForwardMeshRenderResource::new(
      &self.device,
      &res.allocator.allocator,
      command_buffer,
      &res.discard_pool,
      current_frame_timeline,
      &position_data,
      &attribute_data,
      &component.mesh.indices,
      albedo_image,
      normal_image,
      roughness_image,
      ao_image,
    )?;

    let wres = self.res.write();
    let mut wresources = wres.physical_mesh_resources.write();
    // Safety: already checked for existance above
    unsafe {
      wresources
        .as_mut()
        .unwrap_unchecked()
        .insert_unique_unchecked(physical_mesh_id, resource)
    };

    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      buffers: physical_mesh_id.into(),
    })
  }

  fn get_command_buffer(&self) -> GpuResult<crate::gpu::CommandBufferHandle> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buf_id = cmd_id_from_timeline_and_thread_id(timeline);

    if !self
      .recording_command_buffers
      .read()
      .contains_key(&(timeline, cmd_buf_id))
    {
      let cmd = unsafe {
        res
          .command_pools
          .get_unchecked(self.queues.get_graphics_queue().index as usize)
          .as_ref()
          .unwrap_unchecked()
          .allocate_primary(&self.device, this_thread::id(), cmd_buf_id.into())
      }?;
      self.recording_command_buffers.write().insert(
        (timeline, cmd_buf_id),
        RecordingCmdBufferData::new(unsafe { NonZeroHandle::new_unchecked(cmd) }),
      );
    }

    Ok(cmd_buf_id)
  }

  fn begin_render_pass(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    presentation_engine: PresentationEngineHandle,
    acquire_result: &AcquireResult,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let presentation_engines = res.live_presentation_engines.read();
    let cmd_buffers = self.recording_command_buffers.read();
    if !cmd_buffers.contains_key(&(timeline, cmd_buffer)) {
      return Err(GpuError::InvalidArgument);
    }
    if !presentation_engines.contains_key(&presentation_engine) {
      return Err(GpuError::InvalidArgument);
    }
    let wpresentation_engine = unsafe {
      presentation_engines
        .get(&presentation_engine)
        .unwrap_unchecked()
    }
    .write();
    if acquire_result.status.needs_resize() {
      // The caller should handle the resize.
      return Err(GpuError::ResizeRequired);
    }

    let mut cmd_buffers = self.recording_command_buffers.write();
    let data = unsafe {
      cmd_buffers
        .get_mut(&(timeline, cmd_buffer))
        .unwrap_unchecked()
    };
    data.presentation = Some(RecordingCmdBufferDataPresentation {
      acquire_result: *acquire_result,
      presentation_engine: presentation_engine,
    });
    let (render_pass, framebuffer) = res.renderpasses.get_or_create_render_pass(
      RenderPassSpecification::single_pass(&wpresentation_engine, self.depth_stencil_format),
      acquire_result.frame_index as u32,
      &self.device,
      &res.allocator.allocator,
      &res.discard_pool,
      timeline,
    )?;

    let cmd = data.command_buffer.get();
    let black = vk::ClearValue::default();

    let render_pass_begin_info = vk::RenderPassBeginInfo::default()
      .render_pass(render_pass.get())
      .framebuffer(framebuffer.get())
      .render_area(vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
          width: wpresentation_engine.extent().0,
          height: wpresentation_engine.extent().1,
        },
      })
      .clear_values(core::slice::from_ref(&black));
    let subpass_begin_info = vk::SubpassBeginInfo::default().contents(vk::SubpassContents::INLINE);

    unsafe {
      self.create_renderpass2.cmd_begin_render_pass2(
        cmd,
        &render_pass_begin_info,
        &subpass_begin_info,
      )
    };

    Ok(())
  }

  fn bind_pipeline(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    pipeline_key: crate::gpu::PipelineKey,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    if !cmd_buffers.contains_key(&(timeline, cmd_buffer)) {
      return Err(GpuError::InvalidArgument);
    }

    let mut cmd_buffers = self.recording_command_buffers.write();
    let data = unsafe {
      cmd_buffers
        .get_mut(&(timeline, cmd_buffer))
        .unwrap_unchecked()
    };
    let cmd = data.command_buffer.get();

    let pipeline = res
      .pipeline_pool
      .read()
      .get_graphics_pipeline(pipeline_key)
      .ok_or(GpuError::InvalidState)?;

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
    }
    // ready to discard it if necessary (on resize)
    data.bound_pipeline = Some(pipeline);

    Ok(())
  }

  fn bind_buffers(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    _pipeline: crate::gpu::PipelineKey,
    buffers: GpuResourceHandle,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let physical_mesh_id = RenderableInstanceId(buffers.0);
    let physical_mesh_resources = res.physical_mesh_resources.read();
    let resource = physical_mesh_resources
      .as_ref()
      .and_then(|map| map.get(&physical_mesh_id))
      .ok_or(GpuError::InvalidArgument)?;

    let archetype = res
      .physical_mesh_render_archetype
      .as_ref()
      .ok_or(GpuError::InvalidState)?;

    let cmd = data.command_buffer.get();

    // Bind vertex buffers
    unsafe {
      self.device.cmd_bind_vertex_buffers(
        cmd,
        0,
        &[
          resource.position_vertex_buffer.buffer.get(),
          resource.attributes_vertex_buffer.buffer.get(),
        ],
        &[0, 0],
      );
    }

    // Bind index buffer
    unsafe {
      self.device.cmd_bind_index_buffer(
        cmd,
        resource.index_buffer.buffer.get(),
        0,
        vk::IndexType::UINT32,
      );
    }

    // Update and bind descriptor sets
    let mut image_infos = Vec::with_capacity(4);
    if let Some(image) = &resource.albedo_image {
      image_infos.push((0, image.to_descriptor_image_info(res.linear_sampler)));
    }
    if let Some(image) = &resource.normal_image {
      image_infos.push((1, image.to_descriptor_image_info(res.linear_sampler)));
    }
    if let Some(image) = &resource.roughness_image {
      image_infos.push((2, image.to_descriptor_image_info(res.linear_sampler)));
    }
    if let Some(image) = &resource.ao_image {
      image_infos.push((3, image.to_descriptor_image_info(res.linear_sampler)));
    }

    let write_descriptor_sets: Vec<_> = image_infos
      .iter()
      .map(|(binding, info)| {
        vk::WriteDescriptorSet::default()
          .dst_set(archetype.descriptor_sets[0].get())
          .dst_binding(*binding)
          .dst_array_element(0)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .image_info(core::slice::from_ref(info))
      })
      .collect();

    if !write_descriptor_sets.is_empty() {
      unsafe {
        self
          .device
          .update_descriptor_sets(&write_descriptor_sets, &[]);
      }
    }

    unsafe {
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        archetype.pipeline_layout.get(),
        0,
        &[archetype.descriptor_sets[0].get()],
        &[],
      );
    }

    Ok(())
  }

  fn push_constants(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    push_constants: &crate::simulation::comet::PushConstants,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let archetype = res
      .physical_mesh_render_archetype
      .as_ref()
      .ok_or(GpuError::InvalidState)?;

    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    for range in &archetype.push_contant_ranges {
      unsafe {
        let push_constants_bytes = core::slice::from_raw_parts(
          push_constants as *const _ as *const u8,
          core::mem::size_of::<PushConstants>(),
        );
        self.device.cmd_push_constants(
          cmd,
          layout,
          range.stage_flags,
          range.offset,
          &push_constants_bytes[range.offset as usize..(range.offset + range.size) as usize],
        );
      }
    }

    Ok(())
  }

  fn draw_indexed(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    index_count: u32,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let cmd = data.command_buffer.get();

    unsafe {
      self.device.cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
    }

    Ok(())
  }

  fn end_render_pass(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let cmd = data.command_buffer.get();
    let subpass_end_info = vk::SubpassEndInfo::default();

    unsafe {
      self
        .create_renderpass2
        .cmd_end_render_pass2(cmd, &subpass_end_info);
    }

    Ok(())
  }

  fn submit_command_buffer(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let mut cmd_buffers = self.recording_command_buffers.write();
    let mut data = cmd_buffers
      .remove(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    unsafe {
      self.device.end_command_buffer(data.command_buffer.get())?;
    }

    let presentation = data.presentation.ok_or(GpuError::InvalidState)?;
    let presentation_engines = res.live_presentation_engines.read();
    let presentation_engine = presentation_engines
      .get(&presentation.presentation_engine)
      .ok_or(GpuError::InvalidArgument)?;

    let rpresentation_engine = presentation_engine.read();
    let (wait_semaphore, submission_fence) = unsafe {
      rpresentation_engine.get_frame_resources(presentation.acquire_result.frame_index as usize)
    };
    let (_, _, signal_semaphore) = unsafe {
      rpresentation_engine.get_image_resources(presentation.acquire_result.image_index as usize)
    };
    let next_timeline_value = timeline + 1;

    let timeline_values = [0, next_timeline_value];
    let wait_semaphores = [wait_semaphore.get()];
    let signal_semaphores = [signal_semaphore.get(), res.timeline_semaphore.get()];
    let command_buffers = [data.command_buffer.get()];
    let mut timeline_info = vk::TimelineSemaphoreSubmitInfo::default()
      .wait_semaphore_values(&[0])
      .signal_semaphore_values(&timeline_values);

    let submit_info = vk::SubmitInfo::default()
      .wait_semaphores(&wait_semaphores)
      .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT])
      .command_buffers(&command_buffers)
      .signal_semaphores(&signal_semaphores)
      .push_next(&mut timeline_info);

    let graphics_queue = self.queues.get_graphics_queue();
    unsafe {
      self.device.queue_submit(
        graphics_queue.handle,
        &[submit_info],
        submission_fence.get(),
      )?;
    }

    self
      .res
      .read()
      .timeline_semaphore_cached_value
      .store(next_timeline_value, Ordering::Relaxed);

    let cmd_pools = res
      .command_pools
      .get(graphics_queue.index as usize)
      .and_then(|opt| opt.as_ref())
      .cloned()
      .ok_or(GpuError::InvalidState)?;

    data.discard(
      cmd_buffer.into(),
      &res.discard_pool,
      cmd_pools,
      next_timeline_value,
    );

    Ok(())
  }
}

fn cmd_id_from_timeline_and_thread_id(timeline: u64) -> CommandBufferHandle {
  let mut hasher = FnvHasher::new();
  timeline.hash(&mut hasher);
  this_thread::id().hash(&mut hasher);
  CommandBufferHandle(hasher.finish())
}

fn extract_position_data(comet: &Comet) -> Vec<f32> {
  let mut position_data = Vec::with_capacity(comet.vertices.len() * 3);
  for vertex in &comet.vertices {
    position_data.extend_from_slice(&vertex.position);
  }
  position_data
}

fn extract_attribute_data(comet: &Comet) -> Vec<f32> {
  let mut attribute_data = Vec::with_capacity(comet.vertices.len() * 9);
  for vertex in &comet.vertices {
    attribute_data.extend_from_slice(&vertex.normal);
    attribute_data.extend_from_slice(&vertex.uv);
    attribute_data.extend_from_slice(&vertex.tangent);
  }
  attribute_data
}
