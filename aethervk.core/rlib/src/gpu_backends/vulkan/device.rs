use core::{marker::PhantomData, ptr};
use alloc::{format, string::ToString, sync, vec::Vec};

use crate::{
  gpu::{GpuResourceHandle, PresentationEngineHandle, RenderDevice, RenderableInstanceId},
  gpu_backends::vulkan::{
    self,
    device::{
      memory::GlobalDeviceAllocator,
      pipelines::{
        FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization,
        StencilCompareOp, StencilLogicOp, VertexIn,
      },
    },
    instance,
    utils::NonZeroHandle,
  },
  scene::{EntityId, PhysicalMeshComponent, TransformComponent},
  types::{GpuError, GpuResult},
};
use super::utils;

use ash::vk;
use heapless::{index_map::FnvIndexMap};

// companion classes inside Device. Each of these structs implement a given api
// taking as parameters devices and instances, and export a trait which reiterates
// the same interface without device and instance, implemented by `Device`
mod commands;
mod descriptors;
mod memory;
mod pipelines;
mod resources;
mod shaders;
mod swapchain;

trait DeviceResource {
  /// Cleanup function to facilitate hierarchical manual Drop of resources
  /// without having to propagate through `Arc` or other means a reference
  /// to device handle and its function pointers
  /// Note: This function is not responsible to setup the proper state for cleanup (eg synchronization)
  fn cleanup(&mut self, device: &ash::Device);
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
  descriptor_pool: Option<sync::Arc<descriptors::DescriptorPools>>,
  pipeline_pool: spin::RwLock<pipelines::PipelinePool>,
  timeline_semaphore: NonZeroHandle<vk::Semaphore>,
}

impl DeviceResource for DeviceResources {
  /// cleanup in reverse order of declaration in the struct
  fn cleanup(&mut self, device: &ash::Device) {
    unsafe { device.destroy_semaphore(self.timeline_semaphore.get(), None) };

    self.pipeline_pool.write().cleanup(device);

    // Safety: If this is a properly constructed `DeviceResources`, then `descriptor_pool = Some(_)`
    assert!(
      sync::Arc::strong_count(unsafe { self.descriptor_pool.as_ref().unwrap_unchecked() }) == 1
    );
    let mut descriptor_pool: descriptors::DescriptorPools =
      sync::Arc::try_unwrap(unsafe { self.descriptor_pool.take().unwrap_unchecked() }).unwrap();
    descriptor_pool.cleanup(device);

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

    // - Pipeline Pool (TODO: cache data?)
    let pipeline_pool = match pipelines::PipelinePool::new(device, None) {
      Ok(pool) => spin::RwLock::new(pool),
      Err(e) => {
        let descriptor_pool =
          unsafe { sync::Arc::get_mut(&mut descriptor_pool).unwrap_unchecked() };
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
      timeline_semaphore: unsafe { NonZeroHandle::new_unchecked(timeline_semaphore) },
    })
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

  fn get_physical_mesh_shader_modules(
    &self,
    physical_mesh_id: RenderableInstanceId,
  ) -> GpuResult<(
    NonZeroHandle<vk::ShaderModule>,
    NonZeroHandle<vk::ShaderModule>,
  )> {
    todo!()
  }

  fn get_physical_mesh_attributes_bindings(
    &self,
    physical_mesh_id: RenderableInstanceId,
  ) -> GpuResult<VertexIn> {
    todo!()
  }

  fn get_physical_mesh_pipeline_layout(
    &self,
    physical_mesh_id: RenderableInstanceId,
  ) -> GpuResult<NonZeroHandle<vk::PipelineLayout>> {
    todo!()
  }

  fn get_physical_mesh_render_pass(
    &self,
    physical_mesh_id: RenderableInstanceId,
    color_format: vk::Format,
    depth_stencil_format: vk::Format,
  ) -> GpuResult<NonZeroHandle<vk::RenderPass>> {
    todo!()
  }

  fn get_or_create_physical_mesh_pipeline(
    &self,
    physical_mesh_id: RenderableInstanceId,
    pipeline_graphics_info: &GraphicsInfo,
  ) -> GpuResult<GpuResourceHandle> {
    // NonZeroHandle<vk::Pipeline> is stored in an internal mapping hashed together with RenderableInstanceId,
    // and yield frontend handle
    todo!()
  }

  /// Allocates the following resources in the Backend Device
  /// - Vertex Buffer and Index Buffer
  /// - Copies transform component matrix as model and model view proj in a struct corresponding to the Push struct in `comet.rs`
  /// and maps them to frontend handles. Returns only the handles needed for the interface
  fn get_or_allocate_device_memory(
    &self,
    physical_mesh_id: RenderableInstanceId,
    mesh_component: &PhysicalMeshComponent,
    transform: &TransformComponent,
  ) -> GpuResult<(GpuResourceHandle, GpuResourceHandle)> {
    todo!()
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

  /// TODO: expose as parameters the following specialization constants
  /// - layout(constant_id = 0) const float BASE_ALBEDO_R = 0.04;
  /// - layout(constant_id = 1) const float BASE_ALBEDO_G = 0.04;
  /// - layout(constant_id = 2) const float BASE_ALBEDO_B = 0.04;
  /// - layout(constant_id = 3) const float BASE_ROUGHNESS = 0.9;
  /// - layout(constant_id = 4) const float BASE_AO = 1.0;
  fn get_or_create_physical_mesh_resources(
    &self,
    entity_id: EntityId,
    component: &PhysicalMeshComponent,
    transform: &TransformComponent,
    handle: PresentationEngineHandle,
  ) -> GpuResult<(GpuResourceHandle, GpuResourceHandle, GpuResourceHandle)> {
    let presentation_engines_read = self.res.live_presentation_engines.read();
    let presentation_engine = presentation_engines_read
      .get(&handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine.read();
    let (width, height) = presentation_engine_state.extent();

    let physical_mesh_id = RenderableInstanceId::from_physical_mesh(entity_id, component);
    let (vertex_shader_module, fragment_shader_module) =
      self.get_physical_mesh_shader_modules(physical_mesh_id)?;

    // TODO add_specialization_constant_u32
    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(self.get_physical_mesh_attributes_bindings(physical_mesh_id)?)
      .with_pre_rasterization(
        PreRasterization::default()
          .with_vertex_module(vertex_shader_module.get())
          .clone(),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader_module.get())
          .add_viewport(vk::Viewport {
            width: width as _,
            height: height as _,
            x: 0.0,
            y: 0.0,
            min_depth: 0.0,
            max_depth: 1.0,
          })
          .add_scissors(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D { width, height },
          })
          .clone(),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(presentation_engine_state.format())
          .with_depth_attachment_format(self.depth_stencil_format)
          .with_stencil_attachment_format(self.depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(
        // This has the side effect to create descriptor sets and push constants
        self
          .get_physical_mesh_pipeline_layout(physical_mesh_id)?
          .get(),
      )
      .with_pipeline_flags(PipelineFlags::CULL_BACK | PipelineFlags::STENCIL_ENABLE)
      .with_render_pass(
        self
          .get_physical_mesh_render_pass(
            physical_mesh_id,
            presentation_engine_state.format(),
            self.depth_stencil_format,
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

    let pipeline: GpuResourceHandle =
      self.get_or_create_physical_mesh_pipeline(physical_mesh_id, &pipeline_graphics_info)?;
    let (vertex_buffer, index_buffer) =
      self.get_or_allocate_device_memory(physical_mesh_id, &component, &transform)?;

    Ok((pipeline, vertex_buffer, index_buffer))
  }
}
