use core::{
  ptr::{self, NonNull},
  sync::atomic::{AtomicU64, Ordering},
};
use aethervk_oshal_rlib::os::{
  fs::{self, PathBuf, Path},
  memory::{MaxAlignedStorage, StackAllocator},
};
use alloc::{format, string::ToString, sync::Arc, vec::Vec};

use crate::{
  gpu::{
    GpuResourceHandle, PipelineKeyable, PresentationEngineHandle, RenderDevice,
    RenderableInstanceId, frame::ResourceUploadResult,
  },
  gpu_backends::vulkan::{
    self,
    device::{
      memory::GlobalDeviceAllocator,
      pipelines::{
        FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization,
        StencilCompareOp, StencilLogicOp, VertexIn,
      },
      resources::{
        DiscardableResource, ForwardMeshRenderResource, ForwardMeshRenderResourceArchetype,
      },
      shader_manager::ShaderKey,
    },
    instance,
    utils::{self, NonZeroHandle},
  },
  scene::{EntityId, PhysicalMeshComponent, TransformComponent},
  types::{GpuError, GpuResult},
};

use ash::vk;
use hashbrown::HashMap;
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
  command_pools: heapless::Vec<commands::CommandPools, { utils::MAX_QUEUE_FAMILY_COUNT }>,
  descriptor_pool: Option<Arc<descriptors::DescriptorPools>>,
  pipeline_pool: spin::RwLock<pipelines::PipelinePool>,
  renderpasses: renderpasses::RenderPasses,
  timeline_semaphore: NonZeroHandle<vk::Semaphore>,
  timeline_semaphore_cached_value: AtomicU64,

  shader_manager: spin::RwLock<shader_manager::ShaderManager>,

  physical_mesh_render_archetype: Option<ForwardMeshRenderResourceArchetype>,
  /// why not slotmap? This is ephimeral. Each frame this is drained and repopulated
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
      command_pool.cleanup(device);
    }

    for (_, presentation_state) in self.live_presentation_engines.write().drain() {
      presentation_state.write().cleanup(device);
    }

    self.discard_pool.cleanup(device);

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
        renderpasses::RenderPassSpecification::ColorDepthSingleSubpass {
          color_format: presentation_engine_state.format(),
          depth_stencil_format: depth_stencil_format,
          swapchain: &presentation_engine_state,
        },
        device,
        &self.allocator.allocator,
        &self.discard_pool,
        timeline,
      )?
      .get();
    // Note: don't care about viewport and scissor cause they are dynamic state
    write_pipeline.get_or_create_graphics_pipeline(device, graphics_info)?;
    write_pipeline.discard_graphics_pipeline_if_present(pipeline_key, &self.discard_pool, timeline);

    let pipeline_key = graphics_info.pipeline_key();
    archetype.pipeline_key = Some(pipeline_key);

    Ok(())
  }

  fn get_or_create_physical_mesh_archetype(
    &mut self,
    device: &ash::Device,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    timeline: u64,
  ) -> GpuResult<&'_ ForwardMeshRenderResourceArchetype> {
    if self.physical_mesh_render_archetype.is_some() {
      return Ok(unsafe {
        self
          .physical_mesh_render_archetype
          .as_ref()
          .unwrap_unchecked()
      });
    }
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
          .add_binding(0, 3 * size_of::<f32>() as u32, vk::VertexInputRate::VERTEX)
          .add_binding(1, 9 * size_of::<f32>() as u32, vk::VertexInputRate::VERTEX)
          .add_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0) // inPosition
          .add_attribute(1, 1, vk::Format::R32G32B32_SFLOAT, 0) // inNormal
          .add_attribute(1, 2, vk::Format::R32G32_SFLOAT, 3 * size_of::<f32>() as u32) // inUV
          .add_attribute(1, 2, vk::Format::R32G32B32A32_SFLOAT, 20) // inTangent
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
            renderpasses::RenderPassSpecification::ColorDepthSingleSubpass {
              color_format: presentation_engine_state.format(),
              depth_stencil_format,
              swapchain: presentation_engine_state,
            },
            device,
            &self.allocator.allocator,
            &self.discard_pool,
            timeline,
          )?
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
    Ok(unsafe {
      self
        .physical_mesh_render_archetype
        .as_ref()
        .unwrap_unchecked()
    })
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
      unsafe { command_pools.push_unchecked(commands::CommandPools::new(queue_family_index)) };
    }
    // - Swapchain hashmap
    let live_presentation_engines = spin::RwLock::new(hashbrown::HashMap::new());
    Ok(Self {
      allocator,
      command_pools,
      discard_pool,
      live_presentation_engines,
      descriptor_pool: Some(descriptor_pool),
      pipeline_pool,
      renderpasses,
      shader_manager: spin::RwLock::new(shader_manager::ShaderManager::new()),
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

pub(super) struct Device<'a> {
  query_result: utils::PhysicalDeviceQueryResult,
  pub device: ash::Device,
  queues: Queues,
  instance: &'a instance::Instance,

  res: DeviceResources,

  // Some bookkeeping I don't know where to put
  depth_stencil_format: vk::Format,
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

    Ok(Self {
      query_result: *chosen_physical_device_query_result,
      device,
      queues,
      res,
      instance,
      depth_stencil_format,
    })
  }

  pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
    self.query_result.physical_device
  }

  pub(super) fn with_device_allocator<T>(&self, f: impl FnOnce(&GlobalDeviceAllocator) -> T) -> T {
    let dalloc = &self.res.allocator;
    f(dalloc)
  }

  pub(super) fn with_device_allocator_mut<T>(
    &mut self,
    f: impl FnOnce(&mut GlobalDeviceAllocator) -> T,
  ) -> T {
    let dalloc = &mut self.res.allocator;
    f(dalloc)
  }

  fn ensure_physical_mesh_shader_modules(&self) -> GpuResult<(ShaderKey, ShaderKey)> {
    // TODO: proper path management
    let vert_path = PathBuf::from("assets/physical_mesh.vert.spv");
    let frag_path = PathBuf::from("assets/physical_mesh.frag.spv");

    let mut shader_manager = self.res.shader_manager.write();

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

    self.res.cleanup(&self.device);

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
    if let Some(engine) = self.res.live_presentation_engines.read().get(&handle) {
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
    if let Some(engine) = self.res.live_presentation_engines.read().get(&handle) {
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
    if let Some(engine) = self.res.live_presentation_engines.read().get(&handle) {
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
    transform: &TransformComponent,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    // WARNING: if state isn't properly maintained, this is a nightmare
    let next_frame_timeline = self.res.get_timeline_semaphore_cached_value() + 1;

    // create frame for next 
    todo!();

    // Input check: - presentation engine exists
    let presentation_engines_read = self.res.live_presentation_engines.read();
    let presentation_engine = presentation_engines_read
      .get(&handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine.read();
    let (width, height) = presentation_engine_state.extent();

    // ensure that the archetype for physical meshes exists
    let archetype = {
      let (vkey, fkey) = self.ensure_physical_mesh_shader_modules()?;
      self.res.get_or_create_physical_mesh_archetype(
        &self.device,
        vkey,
        fkey,
        self.depth_stencil_format,
        &presentation_engine_state,
        next_frame_timeline,
      )?
    };

    // Get rendering system Internal Mesh Identifier
    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(entity_id, component);

    // Does the mesh already exist? If so, return cached resource
    let read_resouces = self.res.physical_mesh_resources.read();
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
    let mut write_resources = self.res.physical_mesh_resources.write();
    if write_resources.is_none() {
      *write_resources = Some(hashbrown::HashMap::new());
    }

    Ok((pipeline, vertex_buffer, index_buffer, uniforms))
  }

  fn get_command_buffer(&self, timeline: u64) -> GpuResult<crate::gpu::CommandBufferHandle> {
    todo!()
  }

  fn begin_render_pass(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    todo!()
  }

  fn bind_pipeline(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    pipeline: crate::gpu::PipelineKey,
  ) -> GpuResult<()> {
    todo!()
  }

  fn bind_buffers(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    pipeline: crate::gpu::PipelineKey,
    buffers: GpuResourceHandle,
  ) -> GpuResult<()> {
    todo!()
  }

  fn push_constants(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    push_constants: &crate::simulation::comet::PushConstants,
  ) -> GpuResult<()> {
    todo!()
  }

  fn draw_indexed(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    index_count: u32,
  ) -> GpuResult<()> {
    todo!()
  }

  fn end_render_pass(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    todo!()
  }

  fn submit_command_buffer(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    todo!()
  }
}
