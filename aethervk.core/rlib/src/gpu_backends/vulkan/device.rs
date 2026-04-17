use core::{
  hash::{Hash, Hasher},
  ptr::{self, NonNull},
  sync::atomic::{AtomicU64, Ordering},
};
use spin::Once;
use aethervk_oshal_rlib as oshal;
use oshal::{
  hash::FnvHasher,
  os::{
    fs::{self, PathBuf},
    memory::{MaxAlignedStorage, StackAllocator},
    native::this_thread,
  },
};
use alloc::{format, string::ToString, sync::Arc, vec::Vec, boxed::Box};
#[cfg(debug_assertions)]
use oshal::os::debug::{TrackedOption, DropTracker};
use vk_mem::Alloc;

use crate::{
  gpu::{
    AcquireResult, CommandBufferHandle, GpuResourceHandle, NativeGpuProperty, PipelineKeyable,
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
      renderpasses::{RenderPassSpecification, RenderPassType},
      resources::{
        DiscardableResource, ForwardMeshRenderResource, ForwardMeshRenderResourceArchetype, Image,
      },
      shader_manager::ShaderKey,
      swapchain::PresentationState,
    },
    instance,
    utils::{self, NonZeroHandle},
  },
  scene::{EntityId, PhysicalMeshComponent},
  simulation::comet::{
    Comet, NORMAL_COMPONENTS, POSITION_COMPONENTS, PushConstants, TextureFlags, UV_COMPONENTS,
  },
  types::{GpuError, GpuResult},
};

use ash::vk::{self, Pipeline};
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

// TODO: diminish push constant from 160 bytes to 128 bytes

use aethervk_oshal_rlib::math::matrix::{SquareMatrix, Matrix};
use aethervk_oshal_rlib::math::vector::{Vector, Vector3, Vector4};

#[cfg(debug_assertions)]
static ARCHETYPE_CREATED: spin::Once<spin::Mutex<bool>> = Once::new();

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
  heap_resources: Vec<Box<dyn DeviceResource + 'a>>,
  allocator: StackAllocator,
  storage: MaxAlignedStorage<N>,
}

impl<'a, const N: usize> DeviceResourceJanitor<'a, N> {
  fn new(device: &'a ash::Device) -> Self {
    Self {
      device,
      allocator: StackAllocator::new(),
      resources: heapless::Vec::new(),
      heap_resources: Vec::new(),
      storage: MaxAlignedStorage([0; N]),
    }
  }

  pub fn clear(&mut self) {
    // The `drop` implementation will handle cleanup of existing resources.
    // Here we just need to reset the state.
    self.allocator = StackAllocator::new();
    self.resources.clear();
    self.heap_resources.clear();
  }

  pub fn push<T: DeviceResource + 'a>(&mut self, resource: T) -> Result<(), &'static str> {
    // Check if there's space in the inline allocator
    let layout = core::alloc::Layout::new::<T>();
    let start = self.allocator.offset.get();
    let align_offset = unsafe {
      self
        .storage
        .0
        .as_ptr()
        .add(start)
        .align_offset(layout.align())
    };

    let aligned_start = match start.checked_add(align_offset) {
      Some(s) => s,
      None => {
        // overflow, definitely no space
        self.heap_resources.push(Box::new(resource));
        return Ok(());
      }
    };
    let end = match aligned_start.checked_add(layout.size()) {
      Some(e) => e,
      None => {
        // overflow
        self.heap_resources.push(Box::new(resource));
        return Ok(());
      }
    };

    if end > N || self.resources.is_full() {
      // Inline storage is full, or the pointer vec is full. Fallback to heap.
      self.heap_resources.push(Box::new(resource));
      return Ok(());
    }

    // There is space, so we can safely allocate.
    unsafe {
      let base_ptr = self.storage.0.as_mut_ptr();
      let ptr: *mut T = self.allocator.allocate(base_ptr, N, resource)?;

      let dyn_ptr: *mut (dyn DeviceResource + 'a) = ptr;
      let non_null = NonNull::new_unchecked(dyn_ptr);

      // This push should not fail because we checked is_full()
      self.resources.push(non_null).ok().unwrap();
    }

    Ok(())
  }
}

impl<'a, const N: usize> Drop for DeviceResourceJanitor<'a, N> {
  fn drop(&mut self) {
    // Destroy heap-allocated resources first, in reverse order of allocation
    for mut resource_box in self.heap_resources.drain(..).rev() {
      resource_box.cleanup(self.device);
    }

    // Destroy most recently allocated inline resources first
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
  handle: RenderableInstanceId,
  value: &ForwardMeshRenderResource,
  archetype: &ForwardMeshRenderResourceArchetype,
) -> ResourceUploadResult {
  ResourceUploadResult {
    pipeline: unsafe { archetype.pipeline_key.unwrap_unchecked() },
    buffers: handle.into(),
    texture_flags: value.frontend_texture_flags(),
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

  #[cfg(debug_assertions)]
  physical_mesh_render_archetype:
    DropTracker<TrackedOption<ForwardMeshRenderResourceArchetype, 0>, 0>,
  #[cfg(not(debug_assertions))]
  physical_mesh_render_archetype: Option<ForwardMeshRenderResourceArchetype>,
  /// FScene (almost, more like a registry of all known static meshes)
  physical_mesh_resources:
    spin::RwLock<Option<hashbrown::HashMap<RenderableInstanceId, ForwardMeshRenderResource>>>,

  sun_resources:
    spin::RwLock<Option<hashbrown::HashMap<crate::scene::EntityId, resources::SunRenderResource>>>,

  #[cfg(debug_assertions)]
  sun_render_archetype: DropTracker<TrackedOption<resources::SunRenderResourceArchetype, 2>, 2>,
  #[cfg(not(debug_assertions))]
  sun_render_archetype: Option<resources::SunRenderResourceArchetype>,

  #[cfg(debug_assertions)]
  cursor_render_archetype:
    DropTracker<TrackedOption<resources::CursorRenderResourceArchetype, 1>, 1>,
  #[cfg(not(debug_assertions))]
  cursor_render_archetype: Option<resources::CursorRenderResourceArchetype>,

  sky_image: Option<resources::Image>,

  #[cfg(debug_assertions)]
  sky_render_archetype: DropTracker<TrackedOption<resources::SkyRenderResourceArchetype, 3>, 3>,
  #[cfg(not(debug_assertions))]
  sky_render_archetype: Option<resources::SkyRenderResourceArchetype>,

  // not cleaned stuff
  timeline_sem_device: ash::khr::timeline_semaphore::Device,
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

    if let Some(img) = &self.sky_image {
      unsafe {
        vk_mem::ffi::vmaDestroyImage(
          self.allocator.allocator.get_raw(),
          img.image.get(),
          img.allocation.get_raw(),
        );
        device.destroy_image_view(img.image_view.get(), None);
      }
    }

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
    presentation_engine_handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let presentation_engines = self.live_presentation_engines.read();
    let presentation_engine_state_lock = presentation_engines
      .get(&presentation_engine_handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine_state_lock.read();
    if self.physical_mesh_render_archetype.is_none() {
      return Err(GpuError::InvalidState);
    }
    let archetype = unsafe {
      let mut_arch: Option<&mut _>;
      #[cfg(debug_assertions)]
      {
        mut_arch = self.physical_mesh_render_archetype.as_mut().as_mut()
      }
      #[cfg(not(debug_assertions))]
      {
        mut_arch = self.physical_mesh_render_archetype.as_mut();
      }

      mut_arch.unwrap_unchecked()
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

  fn update_cursor_archetype_for_presentation_engine(
    &mut self,
    device: &ash::Device,
    presentation_engine_handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let presentation_engines = self.live_presentation_engines.read();
    let presentation_engine_state_lock = presentation_engines
      .get(&presentation_engine_handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine_state_lock.read();
    if self.cursor_render_archetype.is_none() {
      return Err(GpuError::InvalidState);
    }
    let archetype = unsafe {
      let mut_arch: Option<&mut _>;
      #[cfg(debug_assertions)]
      {
        mut_arch = self.cursor_render_archetype.as_mut().as_mut()
      }
      #[cfg(not(debug_assertions))]
      {
        mut_arch = self.cursor_render_archetype.as_mut();
      }

      mut_arch.unwrap_unchecked()
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

  fn update_sun_archetype_for_presentation_engine(
    &mut self,
    device: &ash::Device,
    presentation_engine_handle: PresentationEngineHandle,
    timeline: u64,
  ) -> GpuResult<()> {
    let presentation_engines = self.live_presentation_engines.read();
    let presentation_engine_state_lock = presentation_engines
      .get(&presentation_engine_handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine_state_lock.read();
    if self.sun_render_archetype.is_none() {
      return Err(GpuError::InvalidState);
    }
    let archetype = unsafe {
      let mut_arch: Option<&mut _>;
      #[cfg(debug_assertions)]
      {
        mut_arch = self.sun_render_archetype.as_mut().as_mut()
      }
      #[cfg(not(debug_assertions))]
      {
        mut_arch = self.sun_render_archetype.as_mut();
      }

      mut_arch.unwrap_unchecked()
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
    #[cfg(debug_assertions)]
    {
      self.physical_mesh_render_archetype.as_ref().as_ref()
    }
    #[cfg(not(debug_assertions))]
    {
      self.physical_mesh_render_archetype.as_ref()
    }
  }

  fn create_physical_mesh_archetype(
    &mut self,
    device: &ash::Device,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    synchronization2: &ash::khr::synchronization2::Device,
    queue: &Queue,
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
        device,
        &vertex_shader,
        &fragment_shader,
        &synchronization2,
        &self.allocator.allocator,
        &self.discard_pool,
        &queue,
      )
    }?;
    #[cfg(not(debug_assertions))]
    {
      self.physical_mesh_render_archetype = Some(res);
    }
    #[cfg(debug_assertions)]
    {
      self.physical_mesh_render_archetype = DropTracker::new(TrackedOption::some(res));
    }

    // then populate graphics info and pipeline key
    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
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
            3,
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
            height: -(presentation_engine_state.extent().1 as f32), // Y axis points downwards in Vulkan, so flip it
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
          let ref_arch: Option<&_>;
          #[cfg(debug_assertions)]
          {
            ref_arch = self.physical_mesh_render_archetype.as_ref().as_ref();
          }
          #[cfg(not(debug_assertions))]
          {
            ref_arch = self.physical_mesh_render_archetype.as_ref();
          }

          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      // TODO remove inversion if mesh is proper
      .with_pipeline_flags(
        PipelineFlags::CULL_ALL | PipelineFlags::STENCIL_ENABLE | PipelineFlags::INVERT_FRONT_FACE,
      )
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

    {
      let val = unsafe {
        self
          .physical_mesh_render_archetype
          .take()
          .unwrap_unchecked()
      }
      .with_graphics_info(pipeline_graphics_info);

      #[cfg(not(debug_assertions))]
      {
        self.physical_mesh_render_archetype = Some(val);
      }
      #[cfg(debug_assertions)]
      {
        self.physical_mesh_render_archetype = DropTracker::new(TrackedOption::some(val));
      }
    }

    debug_assert!(self.physical_mesh_render_archetype.is_some());

    Ok(())
  }

  fn get_sun_archetype(&self) -> Option<&'_ resources::SunRenderResourceArchetype> {
    #[cfg(debug_assertions)]
    {
      self.sun_render_archetype.as_ref().as_ref()
    }
    #[cfg(not(debug_assertions))]
    {
      self.sun_render_archetype.as_ref()
    }
  }

  fn create_sun_archetype(
    &mut self,
    device: &ash::Device,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    if self.sun_render_archetype.is_some() {
      return Err(GpuError::InvalidState);
    }

    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock = live_presentation_engines_lock
      .get(&handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine_lock.read();

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
      resources::SunRenderResourceArchetype::new(device, &vertex_shader, &fragment_shader)
    }?;
    #[cfg(not(debug_assertions))]
    {
      self.sun_render_archetype = Some(res);
    }
    #[cfg(debug_assertions)]
    {
      self.sun_render_archetype = DropTracker::new(TrackedOption::some(res));
    }

    // then populate graphics info and pipeline key
    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
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
            height: -(presentation_engine_state.extent().1 as f32), // Y axis points downwards in Vulkan, so flip it
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
          let ref_arch: Option<&_>;
          #[cfg(debug_assertions)]
          {
            ref_arch = self.sun_render_archetype.as_ref().as_ref();
          }
          #[cfg(not(debug_assertions))]
          {
            ref_arch = self.sun_render_archetype.as_ref();
          }

          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_TEST | PipelineFlags::CULL_ALL) // No culling so we see it from inside and outside (yes, cull all means no culling)
      .with_render_pass(
        self
          .renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            &self.allocator.allocator,
            &self.discard_pool,
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();
    self
      .pipeline_pool
      .write()
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    {
      let val = unsafe { self.sun_render_archetype.take().unwrap_unchecked() }
        .with_graphics_info(pipeline_graphics_info);

      #[cfg(not(debug_assertions))]
      {
        self.sun_render_archetype = Some(val);
      }
      #[cfg(debug_assertions)]
      {
        self.sun_render_archetype = DropTracker::new(TrackedOption::some(val));
      }
    }

    debug_assert!(self.sun_render_archetype.is_some());

    Ok(())
  }

  fn get_sky_archetype(&self) -> Option<&'_ resources::SkyRenderResourceArchetype> {
    #[cfg(debug_assertions)]
    {
      self.sky_render_archetype.as_ref().as_ref()
    }
    #[cfg(not(debug_assertions))]
    {
      self.sky_render_archetype.as_ref()
    }
  }

  fn create_sky_archetype(
    &mut self,
    device: &ash::Device,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    if self.sky_render_archetype.is_some() {
      return Err(GpuError::InvalidState);
    }

    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock = live_presentation_engines_lock
      .get(&handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine_lock.read();

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

    let sky_image = self.sky_image.as_ref().ok_or(GpuError::InvalidState)?;

    // Create initial struct
    let res = unsafe {
      resources::SkyRenderResourceArchetype::new(
        device,
        sky_image.image_view.get(),
        self.linear_sampler.get(),
      )
    }?;
    #[cfg(not(debug_assertions))]
    {
      self.sky_render_archetype = Some(res);
    }
    #[cfg(debug_assertions)]
    {
      self.sky_render_archetype = DropTracker::new(TrackedOption::some(res));
    }

    // then populate graphics info and pipeline key
    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
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
            height: -(presentation_engine_state.extent().1 as f32), // Y axis points downwards in Vulkan, so flip it
            y: presentation_engine_state.extent().1 as f32, // and translate Y accordingly
            min_depth: 0.0,
            max_depth: 1.0,
            ..Default::default()
          })
          .add_scissors(vk::Rect2D {
            extent: vk::Extent2D {
              width: presentation_engine_state.extent().0,
              height: presentation_engine_state.extent().1,
            },
            ..Default::default()
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
          let ref_arch: Option<&_>;
          #[cfg(debug_assertions)]
          {
            ref_arch = self.sky_render_archetype.as_ref().as_ref();
          }
          #[cfg(not(debug_assertions))]
          {
            ref_arch = self.sky_render_archetype.as_ref();
          }

          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(PipelineFlags::CULL_ALL)
      .with_render_pass(
        self
          .renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            &self.allocator.allocator,
            &self.discard_pool,
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .clone();
    self
      .pipeline_pool
      .write()
      .get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

    {
      let val = unsafe { self.sky_render_archetype.take().unwrap_unchecked() }
        .with_graphics_info(pipeline_graphics_info);

      #[cfg(not(debug_assertions))]
      {
        self.sky_render_archetype = Some(val);
      }
      #[cfg(debug_assertions)]
      {
        self.sky_render_archetype = DropTracker::new(TrackedOption::some(val));
      }
    }

    debug_assert!(self.sky_render_archetype.is_some());

    Ok(())
  }

  fn get_cursor_archetype(&self) -> Option<&'_ resources::CursorRenderResourceArchetype> {
    #[cfg(debug_assertions)]
    {
      self.cursor_render_archetype.as_ref().as_ref()
    }
    #[cfg(not(debug_assertions))]
    {
      self.cursor_render_archetype.as_ref()
    }
  }

  fn create_cursor_archetype(
    &mut self,
    device: &ash::Device,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    if self.cursor_render_archetype.is_some() {
      return Err(GpuError::InvalidState);
    }

    let live_presentation_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_lock = live_presentation_engines_lock
      .get(&handle)
      .ok_or(GpuError::InvalidArgument)?;
    let presentation_engine_state = presentation_engine_lock.read();

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
      resources::CursorRenderResourceArchetype::new(device, &vertex_shader, &fragment_shader)
    }?;
    #[cfg(not(debug_assertions))]
    {
      self.cursor_render_archetype = Some(res);
    }
    #[cfg(debug_assertions)]
    {
      self.cursor_render_archetype = DropTracker::new(TrackedOption::some(res));
    }

    // then populate graphics info and pipeline key
    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
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
            height: -(presentation_engine_state.extent().1 as f32), // Y axis points downwards in Vulkan, so flip it
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
          let ref_arch: Option<&_>;
          #[cfg(debug_assertions)]
          {
            ref_arch = self.cursor_render_archetype.as_ref().as_ref();
          }
          #[cfg(not(debug_assertions))]
          {
            ref_arch = self.cursor_render_archetype.as_ref();
          }

          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_TEST | PipelineFlags::CULL_ALL) // NO Culling, NO Depth Test (Yes, cull all means no culling)
      .with_render_pass(
        self
          .renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            &self.allocator.allocator,
            &self.discard_pool,
            0,
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

    {
      let val = unsafe { self.cursor_render_archetype.take().unwrap_unchecked() }
        .with_graphics_info(pipeline_graphics_info);

      #[cfg(not(debug_assertions))]
      {
        self.cursor_render_archetype = Some(val);
      }
      #[cfg(debug_assertions)]
      {
        self.cursor_render_archetype = DropTracker::new(TrackedOption::some(val));
      }
    }

    debug_assert!(self.cursor_render_archetype.is_some());

    Ok(())
  }

  fn has_discardables(&self) -> bool {
    self.physical_mesh_render_archetype.is_some() && {
      let resources = self.physical_mesh_resources.read();
      !resources.is_none() && !unsafe { resources.as_ref().unwrap_unchecked() }.is_empty()
    } || self.sun_render_archetype.is_some() && {
      let resources = self.sun_resources.read();
      !resources.is_none() && !unsafe { resources.as_ref().unwrap_unchecked() }.is_empty()
    } || self.cursor_render_archetype.is_some()
      || self.sky_render_archetype.is_some()
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
    if let Some(mut archetype) = self.sun_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut resources) = self.sun_resources.write().take() {
      for (_, mut resource) in resources.drain() {
        if let Some(img) = resource.image {
          unsafe {
            vk_mem::ffi::vmaDestroyImage(
              self.allocator.allocator.get_raw(),
              img.image.get(),
              img.allocation.get_raw(),
            );
            device.destroy_image_view(img.image_view.get(), None);
          }
        }
      }
    }
    if let Some(mut archetype) = self.cursor_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.sky_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
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

    // timeline semaphore promoted to core after 1.2 (included)
    debug_assert!(instance.api_version() < vk::API_VERSION_1_2);
    let timeline_sem_device =
      ash::khr::timeline_semaphore::Device::new(&instance.instance, &device);

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
      physical_mesh_render_archetype: {
        #[cfg(debug_assertions)]
        {
          DropTracker::new(TrackedOption::none())
        }
        #[cfg(not(debug_assertions))]
        {
          None
        }
      },
      physical_mesh_resources: spin::RwLock::new(None),
      sun_resources: spin::RwLock::new(None),
      #[cfg(debug_assertions)]
      sun_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      sun_render_archetype: None,
      #[cfg(debug_assertions)]
      cursor_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      cursor_render_archetype: None,
      sky_image: None,
      #[cfg(debug_assertions)]
      sky_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      sky_render_archetype: None,
      timeline_sem_device,
    })
  }

  fn get_timeline_semaphore_cached_value(&self) -> u64 {
    self.timeline_semaphore_cached_value.load(Ordering::Relaxed)
  }

  fn refresh_timeline_semaphore_cached_value(
    &self,
    _device: &ash::Device,
  ) -> ash::prelude::VkResult<()> {
    // Note: if you are not using Vulkan 1.2 you are not allowed to use the core function. you need
    // the extension fetched function pointer
    let gpu_value = unsafe {
      self
        .timeline_sem_device
        .get_semaphore_counter_value(self.timeline_semaphore.get())
    }?;
    self
      .timeline_semaphore_cached_value
      .fetch_max(gpu_value, Ordering::Relaxed);
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
  has_begun: bool,
}

impl RecordingCmdBufferData {
  fn new(command_buffer: NonZeroHandle<vk::CommandBuffer>) -> Self {
    Self {
      command_buffer,
      bound_pipeline: None,
      presentation: None,
      has_begun: false,
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
    if self.has_begun {
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
    } else {
      // Not recorded, so just recycle it immediately.
      let _ = cmd_pools.recycle(tid, cmd_buf_id, self.command_buffer.get());
    }
  }
}

pub(super) struct Device<'a> {
  query_result: utils::PhysicalDeviceQueryResult,
  pub device: ash::Device,
  queues: Queues,
  instance: &'a instance::Instance,

  /// Note: Remove if API_VERSION_1_2
  create_renderpass2: ash::khr::create_renderpass2::Device,
  buffer_device_address: ash::khr::buffer_device_address::Device,
  /// Note: Remove if API_VERSION_1_3
  synchronization2: ash::khr::synchronization2::Device,
  #[cfg(target_vendor = "apple")]
  metal_objects: ash::ext::metal_objects::Device,

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
          let family_index = unsafe { queue_buffer.get_unchecked(i).family_index };
          if (queue_type_inserted & (1u32 << QueueId::GRAPHICS as u32)) == 0
            && query_result.graphics_queue_family_index == family_index
          {
            queue_ref_map
              .insert(QueueId::GRAPHICS, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::GRAPHICS as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::COMPUTE as u32)) == 0
            && query_result.compute_queue_family_index == family_index
          {
            queue_ref_map
              .insert(QueueId::COMPUTE, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::COMPUTE as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::TRANSFER as u32)) == 0
            && query_result.transfer_queue_family_index == family_index
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
    let synchronization2 = ash::khr::synchronization2::Device::new(&instance.instance, &device);
    let buffer_device_address =
      ash::khr::buffer_device_address::Device::new(&instance.instance, &device);
    #[cfg(target_vendor = "apple")]
    {
      let metal_objects = ash::ext::metal_objects::Device::new(&instance.instance, &device);
      Ok(Self {
        query_result: *chosen_physical_device_query_result,
        device,
        create_renderpass2,
        synchronization2,
        buffer_device_address,
        metal_objects,
        queues,
        res: res.into(),
        instance,
        depth_stencil_format,
        recording_command_buffers: spin::RwLock::new(hashbrown::HashMap::new()),
      })
    }
    #[cfg(not(target_vendor = "apple"))]
    {
      Ok(Self {
        query_result: *chosen_physical_device_query_result,
        device,
        create_renderpass2,
        synchronization2,
        buffer_device_address,
        queues,
        res: res.into(),
        instance,
        depth_stencil_format,
        recording_command_buffers: spin::RwLock::new(hashbrown::HashMap::new()),
      })
    }
  }

  pub(super) fn physical_device(&self) -> vk::PhysicalDevice {
    self.query_result.physical_device
  }

  fn ensure_physical_mesh_shader_modules(
    &self,
    res: &impl core::ops::Deref<Target = DeviceResources>,
  ) -> GpuResult<(ShaderKey, ShaderKey)> {
    let vert_path: PathBuf;
    let frag_path: PathBuf;

    // TODO: proper path management
    #[cfg(debug_assertions)]
    {
      let assets_dir: PathBuf = {
        use aethervk_oshal_rlib::os;
        use aethervk_oshal_rlib::os::fs::FileSystemObject;

        let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
        if args.len() > 1 {
          let p = PathBuf::from(&args[1]);
          if !p.is_dir() {
            return Err(GpuError::InvalidArgument);
          }

          p
        } else {
          let exe_path = fs::current_exe().map_err(|_| {
            GpuError::BackendSpecific(
              "Failed to get executable path for debug asset loading".into(),
            )
          })?;

          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;

          while let Some(p) = path {
            use aethervk_oshal_rlib::os::fs::FileSystemObject;

            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            path = p.parent();
          }

          let assets_dir = assets_dir
            .expect("Could not find assets directory when searching from executable path");

          assets_dir
        }
      };

      vert_path = assets_dir.join("physical_mesh.vert.spv");
      #[cfg(feature = "physical_mesh_debug_normals")]
      {
        frag_path = assets_dir.join("physical_mesh_debug_normals.frag.spv");
      }
      #[cfg(not(feature = "physical_mesh_debug_normals"))]
      {
        frag_path = assets_dir.join("physical_mesh.frag.spv");
      }
    }
    #[cfg(not(debug_assertions))]
    {
      todo!();
    }

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

  fn ensure_skygen_shader_module(
    &self,
    res: &impl core::ops::Deref<Target = DeviceResources>,
  ) -> GpuResult<ShaderKey> {
    let comp_path: PathBuf;

    // TODO: proper path management
    #[cfg(debug_assertions)]
    {
      let assets_dir: PathBuf = {
        use aethervk_oshal_rlib::os;
        use aethervk_oshal_rlib::os::fs::FileSystemObject;

        let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
        if args.len() > 1 {
          let p = PathBuf::from(&args[1]);
          if !p.is_dir() {
            return Err(GpuError::InvalidArgument);
          }
          p
        } else {
          let exe_path = os::fs::current_exe().map_err(|_| {
            GpuError::BackendSpecific(
              "Failed to get executable path for debug asset loading".into(),
            )
          })?;

          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;

          while let Some(p) = path {
            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            path = p.parent();
          }

          assets_dir.expect("Could not find assets directory when searching from executable path")
        }
      };
      comp_path = assets_dir.join("skygen.comp.spv");
    }
    #[cfg(not(debug_assertions))]
    {
      todo!();
    }

    let mut shader_manager = res.shader_manager.write();

    let comp_key = shader_manager.get_or_load(
      &self.device,
      &comp_path,
      "main",
      spirv::ExecutionModel::GLCompute,
    )?;

    Ok(comp_key)
  }

  fn ensure_sungen_shader_module(
    &self,
    res: &impl core::ops::Deref<Target = DeviceResources>,
  ) -> GpuResult<ShaderKey> {
    let comp_path: PathBuf;

    #[cfg(debug_assertions)]
    {
      let assets_dir: PathBuf = {
        use aethervk_oshal_rlib::os;
        use aethervk_oshal_rlib::os::fs::FileSystemObject;

        let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
        if args.len() > 1 {
          let p = PathBuf::from(&args[1]);
          if !p.is_dir() {
            return Err(GpuError::InvalidArgument);
          }
          p
        } else {
          let exe_path = os::fs::current_exe().map_err(|_| {
            GpuError::BackendSpecific(
              "Failed to get executable path for debug asset loading".into(),
            )
          })?;

          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;

          while let Some(p) = path {
            use aethervk_oshal_rlib::os::fs::FileSystemObject;

            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            path = p.parent();
          }

          assets_dir.expect("Could not find assets directory when searching from executable path")
        }
      };
      comp_path = assets_dir.join("sungen.comp.spv");
    }
    #[cfg(not(debug_assertions))]
    {
      todo!();
    }

    let mut shader_manager = res.shader_manager.write();
    let comp_key = shader_manager.get_or_load(
      &self.device,
      &comp_path,
      "main",
      spirv::ExecutionModel::GLCompute,
    )?;

    Ok(comp_key)
  }

  fn ensure_sun_shader_modules(
    &self,
    res: &impl core::ops::Deref<Target = DeviceResources>,
  ) -> GpuResult<(ShaderKey, ShaderKey)> {
    let vert_path: PathBuf;
    let frag_path: PathBuf;

    #[cfg(debug_assertions)]
    {
      let assets_dir: PathBuf = {
        use aethervk_oshal_rlib::os;
        use aethervk_oshal_rlib::os::fs::FileSystemObject;

        let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
        if args.len() > 1 {
          let p = PathBuf::from(&args[1]);
          if !p.is_dir() {
            return Err(GpuError::InvalidArgument);
          }

          p
        } else {
          let exe_path = os::fs::current_exe().map_err(|_| {
            GpuError::BackendSpecific(
              "Failed to get executable path for debug asset loading".into(),
            )
          })?;

          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;

          while let Some(p) = path {
            use aethervk_oshal_rlib::os::fs::FileSystemObject;

            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            path = p.parent();
          }

          let assets_dir = assets_dir
            .expect("Could not find assets directory when searching from executable path");

          assets_dir
        }
      };

      vert_path = assets_dir.join("sun_volume.vert.spv");
      frag_path = assets_dir.join("sun_volume.frag.spv");
    }
    #[cfg(not(debug_assertions))]
    {
      todo!();
    }

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

  fn ensure_sky_shader_modules(
    &self,
    res: &impl core::ops::Deref<Target = DeviceResources>,
  ) -> GpuResult<(ShaderKey, ShaderKey)> {
    let vert_path: PathBuf;
    let frag_path: PathBuf;

    #[cfg(debug_assertions)]
    {
      let assets_dir: PathBuf = {
        use aethervk_oshal_rlib::os;
        use aethervk_oshal_rlib::os::fs::FileSystemObject;

        let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
        if args.len() > 1 {
          let p = PathBuf::from(&args[1]);
          if !p.is_dir() {
            return Err(GpuError::InvalidArgument);
          }

          p
        } else {
          let exe_path = fs::current_exe().map_err(|_| {
            GpuError::BackendSpecific(
              "Failed to get executable path for debug asset loading".into(),
            )
          })?;

          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;

          while let Some(p) = path {
            use aethervk_oshal_rlib::os::fs::FileSystemObject;

            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            path = p.parent();
          }

          let assets_dir = assets_dir
            .expect("Could not find assets directory when searching from executable path");

          assets_dir
        }
      };

      vert_path = assets_dir.join("sky.vert.spv");
      frag_path = assets_dir.join("sky.frag.spv");
    }
    #[cfg(not(debug_assertions))]
    {
      todo!();
    }

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

  fn ensure_cursor_shader_modules(
    &self,
    res: &impl core::ops::Deref<Target = DeviceResources>,
  ) -> GpuResult<(ShaderKey, ShaderKey)> {
    let vert_path: PathBuf;
    let frag_path: PathBuf;

    #[cfg(debug_assertions)]
    {
      let assets_dir: PathBuf = {
        use aethervk_oshal_rlib::os;
        use aethervk_oshal_rlib::os::fs::FileSystemObject;

        let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
        if args.len() > 1 {
          let p = PathBuf::from(&args[1]);
          if !p.is_dir() {
            return Err(GpuError::InvalidArgument);
          }

          p
        } else {
          let exe_path = fs::current_exe().map_err(|_| {
            GpuError::BackendSpecific(
              "Failed to get executable path for debug asset loading".into(),
            )
          })?;

          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;

          while let Some(p) = path {
            use aethervk_oshal_rlib::os::fs::FileSystemObject;

            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            path = p.parent();
          }

          let assets_dir = assets_dir
            .expect("Could not find assets directory when searching from executable path");

          assets_dir
        }
      };

      vert_path = assets_dir.join("cursor.vert.spv");
      #[cfg(feature = "cursor_debug")]
      {
        frag_path = assets_dir.join("cursor_debug.frag.spv");
      }
      #[cfg(not(feature = "cursor_debug"))]
      {
        frag_path = assets_dir.join("cursor.frag.spv");
      }
    }
    #[cfg(not(debug_assertions))]
    {
      todo!();
    }

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
    let _ = unsafe { self.device.device_wait_idle() };

    self.res.write().cleanup(&self.device);

    // in the end, destroy the device
    unsafe { self.device.destroy_device(None) };
  }
}

impl<'a> RenderDevice for Device<'a> {
  fn download_windowless_image(
    &self,
    handle: PresentationEngineHandle,
    buffer: &mut [u8],
  ) -> GpuResult<()> {
    let mut res = self.res.write();
    let engine_lock = res.live_presentation_engines.read();
    let state_lock = engine_lock.get(&handle).ok_or(GpuError::InvalidState)?;
    let mut state = state_lock.write();

    if let swapchain::PresentationState::Windowless(windowless) = &mut *state {
      let image = windowless.get_last_submitted_image()?;
      let (width, height) = windowless.extent();

      let buffer_size = (width * height * 4) as vk::DeviceSize;
      if buffer.len() != buffer_size as usize {
        return Err(GpuError::InvalidArgument);
      }

      let buffer_info = vk::BufferCreateInfo::default()
        .size(buffer_size)
        .usage(vk::BufferUsageFlags::TRANSFER_DST);

      let mut alloc_info = vk_mem::AllocationCreateInfo::default();
      alloc_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
      alloc_info.flags = vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
        | vk_mem::AllocationCreateFlags::MAPPED;

      let (staging_buffer, alloc) = unsafe {
        res
          .allocator
          .allocator
          .create_buffer(&buffer_info, &alloc_info)
      }?;
      let alloc_info_res = res.allocator.allocator.get_allocation_info(&alloc);

      let graphics_queue = self.queues.get_graphics_queue();
      unsafe { self.device.queue_wait_idle(graphics_queue.handle) }?;

      let command_pool_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(graphics_queue.family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
      let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }?;

      let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
      let command_buffer =
        unsafe { self.device.allocate_command_buffers(&command_buffer_info) }?[0];

      let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
      unsafe {
        self
          .device
          .begin_command_buffer(command_buffer, &begin_info)
      }?;

      let image_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .image(image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );
      let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&image_barrier));
      unsafe {
        self
          .synchronization2
          .cmd_pipeline_barrier2(command_buffer, &dep_info)
      };

      let region = vk::BufferImageCopy::default()
        .image_subresource(
          vk::ImageSubresourceLayers::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1),
        )
        .image_extent(vk::Extent3D {
          width,
          height,
          depth: 1,
        });

      unsafe {
        self.device.cmd_copy_image_to_buffer(
          command_buffer,
          image.get(),
          vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
          staging_buffer,
          &[region],
        )
      };

      let image_barrier2 = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
        .src_access_mask(vk::AccessFlags2::TRANSFER_READ)
        .dst_stage_mask(vk::PipelineStageFlags2::NONE)
        .dst_access_mask(vk::AccessFlags2::NONE)
        .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
        .image(image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );
      let dep_info2 =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&image_barrier2));
      unsafe {
        self
          .synchronization2
          .cmd_pipeline_barrier2(command_buffer, &dep_info2)
      };

      unsafe { self.device.end_command_buffer(command_buffer) }?;

      let submit_info =
        vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
      unsafe {
        self
          .device
          .queue_submit(graphics_queue.handle, &[submit_info], vk::Fence::null())
      }?;
      unsafe { self.device.queue_wait_idle(graphics_queue.handle) }?;

      unsafe { self.device.destroy_command_pool(command_pool, None) };

      let mapped_ptr = alloc_info_res.mapped_data as *const u8;
      oshal::log!("mapped_ptr is null: {}", mapped_ptr.is_null());
      if !mapped_ptr.is_null() {
        let _ = unsafe {
          res
            .allocator
            .allocator
            .invalidate_allocation(&alloc, 0, vk::WHOLE_SIZE)
        };
        unsafe {
          core::ptr::copy_nonoverlapping(mapped_ptr, buffer.as_mut_ptr(), buffer_size as usize);
        }
      }

      unsafe {
        let mut mut_alloc = alloc;
        res
          .allocator
          .allocator
          .destroy_buffer(staging_buffer, &mut mut_alloc);
      }

      Ok(())
    } else {
      Err(GpuError::InvalidState)
    }
  }

  fn get_native_prop(&self, prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void> {
    #[cfg(target_vendor = "apple")]
    {
      if prop == NativeGpuProperty::VulkanMetalDeviceId {
        let mut metal_device_info = vk::ExportMetalDeviceInfoEXT::default();
        let mut metal_objects_info =
          vk::ExportMetalObjectsInfoEXT::default().push_next(&mut metal_device_info);
        unsafe {
          (self.metal_objects.fp().export_metal_objects_ext)(
            self.device.handle(),
            core::ptr::from_mut(&mut metal_objects_info),
          );
        };

        return Some(metal_device_info.mtl_device);
      }
    }

    None
  }

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

    // Acquire a single write lock to perform the entire resize operation atomically.
    // This prevents deadlocks and satisfies the borrow checker.
    let mut wres = self.res.write();
    let timeline = wres.get_timeline_semaphore_cached_value();

    // Get a mutable reference to the presentation engine to resize it.
    // borrow checker enforces us to reacquire the lock after we call a mutating method
    {
      let engine_lock = wres.live_presentation_engines.read();
      let engine = engine_lock.get(&handle).ok_or(GpuError::InvalidArgument)?;
      engine.write().resize(
        &self.instance.instance,
        &self.device,
        physical_device_handle,
        width,
        height,
      )?;
    }

    // After resizing, update dependent resources like pipelines/renderpasses.
    // `update_physical_mesh_archetype_for_presentation_engine` takes `&mut self` (for `wres`)
    // and an immutable `&PresentationState` (for `engine`). This is a valid borrow pattern.
    wres.update_physical_mesh_archetype_for_presentation_engine(&self.device, handle, timeline)?;
    let _ = wres.update_cursor_archetype_for_presentation_engine(&self.device, handle, timeline);
    let _ = wres.update_sun_archetype_for_presentation_engine(&self.device, handle, timeline);

    Ok(())
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
      let graphics_queue = self.queues.get_graphics_queue().handle;
      engine.write().acquire_next_image(&self.device, graphics_queue)
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
    let next_frame_timeline = self.res.read().get_timeline_semaphore_cached_value() + 1;
    let current_frame_timeline = next_frame_timeline - 1;

    // ensure that the archetype for physical meshes exists
    if self.res.read().physical_mesh_render_archetype.is_none() {
      let mut wres = self.res.write();
      // Re-check condition after acquiring write lock
      if wres.physical_mesh_render_archetype.is_none() {
        #[cfg(debug_assertions)]
        {
          let initialized = ARCHETYPE_CREATED.call_once(|| spin::Mutex::new(false));
          let mut guard = initialized.lock();
          assert!(
            !*guard,
            "physical_mesh_render_archetype created more than once!"
          );
          *guard = true;
        }
        let (vkey, fkey) = self.ensure_physical_mesh_shader_modules(&wres)?;
        wres.create_physical_mesh_archetype(
          &self.device,
          vkey,
          fkey,
          self.depth_stencil_format,
          &self.synchronization2,
          &self.queues.get_graphics_queue(),
          handle,
          next_frame_timeline,
        )?;
        #[cfg(debug_assertions)]
        {
          oshal::log!("Created Physical Mesh Archetype");
          oshal::os::debug::print_stacktrace();
        }
      }
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
            physical_mesh_id,
            &resource,
            &archetype,
          ));
        }
      }
    }
    drop(read_resouces);

    // Otherwise, create it inside the resources registry
    // Upload is a blocking operation, so we need to release the read lock on res
    // and acquire it later.
    let graphics_queue = self.queues.get_graphics_queue();
    // Inefficient creation of one-shot command pool. Hopefully, mesh update is infrequent
    let command_pool = {
      let create_info = vk::CommandPoolCreateInfo::default()
        .queue_family_index(graphics_queue.family_index)
        .flags(vk::CommandPoolCreateFlags::TRANSIENT);
      unsafe { self.device.create_command_pool(&create_info, None) }
    }?;

    let command_buffer = {
      let alloc_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
      unsafe { self.device.allocate_command_buffers(&alloc_info) }?[0]
    };

    let (resource, texture_flags) = 'resource_creation: {
      let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
      unsafe {
        self
          .device
          .begin_command_buffer(command_buffer, &begin_info)?
      };

      let res = self.res.read();
      let position_data = extract_position_data(&component.mesh);
      let attribute_data = extract_attribute_data(&component.mesh);
      let mut texture_flags: TextureFlags = TextureFlags::empty();
      let albedo_image = component.mesh.albedo_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::ALBEDO;
        Image::new_2d(
          &self.device,
          &self.synchronization2,
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
        texture_flags |= TextureFlags::NORMAL;
        Image::new_2d(
          &self.device,
          &self.synchronization2,
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
        texture_flags |= TextureFlags::ROUGHNESS;
        Image::new_2d(
          &self.device,
          &self.synchronization2,
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
        texture_flags |= TextureFlags::AO;
        Image::new_2d(
          &self.device,
          &self.synchronization2,
          &res.allocator.allocator,
          command_buffer,
          &res.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
        )
        .ok()
      });

      let resource = unsafe {
        let descriptor_set = archetype.create_descriptor_set_from_layout_at_index(
          &self.device,
          res.descriptor_pool.as_ref().unwrap_unchecked(),
          &res.discard_pool,
          0,
        )?;
        ForwardMeshRenderResource::new(
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
          res.sky_image.as_ref().map(|sky| resources::Image {
            image: sky.image,
            image_view: sky.image_view,
            allocation: sky.allocation, // Assuming Allocation implements Copy/Clone. It's a pointer.
          }),
          res.linear_sampler,
          descriptor_set,
          &archetype.dummy_texture_handle,
        )?
      };

      unsafe {
        self.device.end_command_buffer(command_buffer)?;
        let command_buffers = [command_buffer];
        let submits = [vk::SubmitInfo::default().command_buffers(&command_buffers)];
        self
          .device
          .queue_submit(graphics_queue.handle, &submits, vk::Fence::null())?;
        // Less efficient than using the transfer queue and synchronizing with timeline semaphore and
        // barriers, but much simpler
        // TODO: improve?
        self.device.queue_wait_idle(graphics_queue.handle)?;
      };

      break 'resource_creation (resource, texture_flags);
    };
    unsafe {
      self.device.destroy_command_pool(command_pool, None);
    }

    drop(res);
    let wres = self.res.write();
    let mut wresources = wres.physical_mesh_resources.write();
    if wresources.is_none() {
      *wresources = Some(hashbrown::HashMap::new());
    }
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
      texture_flags,
    })
  }

  fn generate_sky(&self) -> GpuResult<()> {
    let res = self.res.read();
    let comp_key = self.ensure_skygen_shader_module(&res)?;

    let shader_module = {
      let shader_manager = res.shader_manager.read();
      let shader = shader_manager
        .get(comp_key)
        .ok_or(GpuError::InvalidShader)?;
      shader.module.get()
    };

    let graphics_queue = self.queues.get_graphics_queue();
    let compute_queue = self.queues.get_compute_queue();
    let families_different = graphics_queue.family_index != compute_queue.family_index;
    oshal::log!(
      "GRAPHICS: {}, COMPUTE: {}",
      graphics_queue.family_index,
      compute_queue.family_index
    );

    // Create sky image 512x512
    let sky_image = resources::Image::new_storage_2d(
      &self.device,
      &res.allocator.allocator,
      512,
      512,
      vk::Format::R16G16B16A16_SFLOAT,
      graphics_queue.family_index,
      compute_queue.family_index,
    )?;

    // Create Descriptor Set Layout
    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
      .descriptor_count(1)
      .stage_flags(vk::ShaderStageFlags::COMPUTE)];

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }?;

    // Create Pipeline Layout
    let set_layouts = [set_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
    let pipeline_layout = unsafe {
      self
        .device
        .create_pipeline_layout(&pipeline_layout_info, None)
    }?;

    // Create Descriptor Pool and Set
    let pool_sizes = [vk::DescriptorPoolSize::default()
      .ty(vk::DescriptorType::STORAGE_IMAGE)
      .descriptor_count(1)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
      .pool_sizes(&pool_sizes)
      .max_sets(1);
    let descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }?;

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
      .descriptor_pool(descriptor_pool)
      .set_layouts(&set_layouts);
    let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?[0];

    // Write descriptor set
    let image_info = vk::DescriptorImageInfo::default()
      .image_layout(vk::ImageLayout::GENERAL)
      .image_view(sky_image.image_view.get());
    let write_descriptor_set = vk::WriteDescriptorSet::default()
      .dst_set(descriptor_set)
      .dst_binding(0)
      .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
      .image_info(core::slice::from_ref(&image_info));
    unsafe {
      self
        .device
        .update_descriptor_sets(&[write_descriptor_set], &[])
    };

    // Get or Create Compute Pipeline
    let mut compute_info = pipelines::ComputeInfo::default();
    compute_info.shader_module = shader_module;
    compute_info.pipeline_layout = pipeline_layout;
    compute_info.add_specialization_constant_u32(
      vk::SpecializationMapEntry::default()
        .constant_id(0)
        .offset(0)
        .size(4),
      16,
    );
    compute_info.add_specialization_constant_u32(
      vk::SpecializationMapEntry::default()
        .constant_id(1)
        .offset(4)
        .size(4),
      16,
    );

    let compute_pipeline = res
      .pipeline_pool
      .write()
      .get_or_create_compute_pipeline(&self.device, &compute_info)?;

    // Create Command Pool and Buffer for compute
    let command_pool_info = vk::CommandPoolCreateInfo::default()
      .queue_family_index(compute_queue.family_index)
      .flags(vk::CommandPoolCreateFlags::TRANSIENT);
    let command_pool = unsafe { self.device.create_command_pool(&command_pool_info, None) }?;

    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
      .command_pool(command_pool)
      .level(vk::CommandBufferLevel::PRIMARY)
      .command_buffer_count(1);
    let command_buffer = unsafe { self.device.allocate_command_buffers(&command_buffer_info) }?[0];

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
      self
        .device
        .begin_command_buffer(command_buffer, &begin_info)?;

      // Transition to GENERAL
      let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::NONE)
        .src_access_mask(vk::AccessFlags2::NONE)
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .image(sky_image.image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        )
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
      let dep_info =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier));
      self
        .synchronization2
        .cmd_pipeline_barrier2(command_buffer, &dep_info);

      // Dispatch
      self.device.cmd_bind_pipeline(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        compute_pipeline.get(),
      );
      self.device.cmd_bind_descriptor_sets(
        command_buffer,
        vk::PipelineBindPoint::COMPUTE,
        pipeline_layout,
        0,
        &[descriptor_set],
        &[],
      );
      self
        .device
        .cmd_dispatch(command_buffer, 512 / 16, 512 / 16, 1);

      // Transition to SHADER_READ_ONLY_OPTIMAL
      let barrier2 = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(sky_image.image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        )
        .src_queue_family_index(if families_different {
          compute_queue.family_index
        } else {
          vk::QUEUE_FAMILY_IGNORED
        })
        .dst_queue_family_index(if families_different {
          graphics_queue.family_index
        } else {
          vk::QUEUE_FAMILY_IGNORED
        });

      let dep_info2 =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier2));
      self
        .synchronization2
        .cmd_pipeline_barrier2(command_buffer, &dep_info2);

      self.device.end_command_buffer(command_buffer)?;

      // Submit
      let submit_info =
        vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
      self
        .device
        .queue_submit(compute_queue.handle, &[submit_info], vk::Fence::null())?;
      self.device.queue_wait_idle(compute_queue.handle)?;

      self.device.destroy_command_pool(command_pool, None);
      self.device.destroy_descriptor_pool(descriptor_pool, None);
      self.device.destroy_pipeline_layout(pipeline_layout, None);
      self.device.destroy_descriptor_set_layout(set_layout, None);
    }

    drop(res);
    let mut wres = self.res.write();
    // In case it already has an image, destroy it
    if let Some(img) = &wres.sky_image {
      unsafe {
        vk_mem::ffi::vmaDestroyImage(
          wres.allocator.allocator.get_raw(),
          img.image.get(),
          img.allocation.get_raw(),
        );
        self.device.destroy_image_view(img.image_view.get(), None);
      }
    }
    wres.sky_image = Some(sky_image);

    Ok(())
  }

  fn get_or_create_cursor_resources(
    &self,
    handle: PresentationEngineHandle,
  ) -> GpuResult<ResourceUploadResult> {
    let next_frame_timeline = self.res.read().get_timeline_semaphore_cached_value() + 1;

    // ensure that the archetype for cursors exists
    if self.res.read().cursor_render_archetype.is_none() {
      let mut wres = self.res.write();
      // Re-check condition after acquiring write lock
      if wres.cursor_render_archetype.is_none() {
        let (vkey, fkey) = self.ensure_cursor_shader_modules(&wres)?;
        wres.create_cursor_archetype(
          &self.device,
          vkey,
          fkey,
          self.depth_stencil_format,
          handle,
        )?;
      }
    }

    let res = self.res.read();
    let archetype = unsafe { res.get_cursor_archetype().unwrap_unchecked() };

    // Safety: Archetype, once properly constructed, has everything populated
    let pipeline_key = unsafe { archetype.pipeline_key.unwrap_unchecked() };

    // the cursor doesn't have descriptor sets or vertex/index buffers
    Ok(ResourceUploadResult {
      pipeline: pipeline_key,
      buffers: crate::gpu::NULL_GPU_RESOURCE, // no buffers
      texture_flags: TextureFlags::empty(),
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

  fn begin_command_buffer(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let mut cmd_buffers = self.recording_command_buffers.write();
    let data = cmd_buffers
      .get_mut(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    if data.has_begun {
      return Ok(());
    }

    let begin_info =
      vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
      self
        .device
        .begin_command_buffer(data.command_buffer.get(), &begin_info)?;
    }
    data.has_begun = true;

    Ok(())
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

    let data = unsafe { cmd_buffers.get(&(timeline, cmd_buffer)).unwrap_unchecked() };

    if !data.has_begun {
      return Err(GpuError::InvalidState);
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
    drop(cmd_buffers);

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
    let mut black = [vk::ClearValue::default(), vk::ClearValue::default()]; // 2 attachments
    res
      .renderpasses
      .get_clear_values_render_pass(RenderPassType::ColorDepthSingleSubpass, &mut black)?;

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
      .clear_values(&black);
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

  fn get_presentation_engine_extent(
    &self,
    handle: crate::gpu::PresentationEngineHandle,
  ) -> GpuResult<[u32; 2]> {
    if let Some(engine) = self
      .res
      .read()
      .live_presentation_engines
      .read()
      .get(&handle)
    {
      let e = engine.read().extent();
      Ok([e.0, e.1])
    } else {
      Err(GpuError::InvalidArgument)
    }
  }

  fn set_viewport(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    viewport: &crate::gpu::Viewport,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let cmd = data.command_buffer.get();
    let vk_viewport = vk::Viewport {
      x: viewport.x,
      y: viewport.y,
      width: viewport.width,
      height: viewport.height,
      min_depth: viewport.min_depth,
      max_depth: viewport.max_depth,
    };
    unsafe {
      self.device.cmd_set_viewport(cmd, 0, &[vk_viewport]);
    }

    Ok(())
  }

  fn set_scissor(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    scissor: &crate::gpu::Rect2D,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let cmd = data.command_buffer.get();
    let vk_scissor = vk::Rect2D {
      offset: vk::Offset2D {
        x: scissor.offset[0],
        y: scissor.offset[1],
      },
      extent: vk::Extent2D {
        width: scissor.extent[0],
        height: scissor.extent[1],
      },
    };
    unsafe {
      self.device.cmd_set_scissor(cmd, 0, &[vk_scissor]);
    }

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

    drop(cmd_buffers);
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

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.physical_mesh_render_archetype.as_ref().as_ref()
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.physical_mesh_render_archetype.as_ref()
    }
    let archetype = arch_ref.ok_or(GpuError::InvalidState)?;

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
    // Errata: modifying a descriptor set while the GPU is executing (or about to execute) commands that use that set is a severe data race. Since multiple meshes share that one archetype set, Mesh B will overwrite Mesh A's descriptors before the GPU even gets a chance to render Mesh A
    unsafe {
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        archetype.pipeline_layout.get(),
        0,
        &[resource.descriptor_set.get()],
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

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.physical_mesh_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.physical_mesh_render_archetype.as_ref();
    }
    let archetype = arch_ref.ok_or(GpuError::InvalidState)?;

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

  fn push_sun_constants(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    push_constants: &crate::gpu::SunPushConstants,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.sun_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.sun_render_archetype.as_ref();
    }
    let archetype = arch_ref.ok_or(GpuError::InvalidState)?;

    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    for range in &archetype.push_contant_ranges {
      unsafe {
        let push_constants_bytes = core::slice::from_raw_parts(
          push_constants as *const _ as *const u8,
          core::mem::size_of::<crate::gpu::SunPushConstants>(),
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

  fn push_cursor_constants(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    push_constants: &crate::gpu::CursorPushConstants,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.cursor_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.cursor_render_archetype.as_ref();
    }
    let archetype = arch_ref.ok_or(GpuError::InvalidState)?;

    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    for range in &archetype.push_contant_ranges {
      unsafe {
        let push_constants_bytes = core::slice::from_raw_parts(
          push_constants as *const _ as *const u8,
          core::mem::size_of::<crate::gpu::CursorPushConstants>(),
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

  fn draw(&self, cmd_buffer: crate::gpu::CommandBufferHandle, vertex_count: u32) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let cmd = data.command_buffer.get();

    unsafe {
      self.device.cmd_draw(cmd, vertex_count, 1, 0, 0);
    }

    Ok(())
  }

  fn render_sun(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    component: &crate::scene::SunComponent,
    transform: &crate::scene::TransformComponent,
    view: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<()> {
    let mut res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    // ensure that the archetype for sun exists
    if res.sun_render_archetype.is_none() {
      drop(res);
      let mut wres = self.res.write();
      // Re-check condition after acquiring write lock
      if wres.sun_render_archetype.is_none() {
        let (vkey, fkey) = self.ensure_sun_shader_modules(&wres)?;
        // We need a handle. The render_sun function doesn't take one. We can get it from recording cmd buffers or presentation engines
        // But sun_render_archetype uses depth_stencil_format which is global and presentation_engine_handle which gives us format.
        // Actually, the presentation format is needed for the renderpass.
        // For simplicity, let's just grab the first live presentation engine. In a multi-window setup this might be wrong,
        // but it's a start. Or we can just use the swapchain format if we know it.
        let handle = wres.live_presentation_engines.read().keys().next().copied();
        if let Some(h) = handle {
          wres.create_sun_archetype(&self.device, vkey, fkey, self.depth_stencil_format, h)?;
        }
      }
      drop(wres);
      res = self.res.read();
    }

    let mut sun_res_lock = res.sun_resources.write();
    if sun_res_lock.is_none() {
      *sun_res_lock = Some(hashbrown::HashMap::new());
    }
    let map = sun_res_lock.as_mut().unwrap();
    if !map.contains_key(&entity_id) {
      let graphics_queue = self.queues.get_graphics_queue();
      let compute_queue = self.queues.get_compute_queue();
      let image = resources::Image::new_storage_3d(
        &self.device,
        &res.allocator.allocator,
        component.resolution.0,
        component.resolution.1,
        component.resolution.2,
        vk::Format::R16G16B16A16_SFLOAT,
        graphics_queue.family_index,
        compute_queue.family_index,
      )?;

      // Now run sungen.comp
      let comp_key = self.ensure_sungen_shader_module(&res)?;
      let shader_module = {
        let shader_manager = res.shader_manager.read();
        let shader = shader_manager
          .get(comp_key)
          .ok_or(GpuError::InvalidShader)?;
        shader.module.get()
      };

      // Create Descriptor Set Layout
      let bindings = [vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)];

      let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
      let set_layout = unsafe { self.device.create_descriptor_set_layout(&layout_info, None) }?;

      // Create Pipeline Layout
      let push_constant_ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(8)]; // 64-bit pointer
      let set_layouts = [set_layout];
      let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&set_layouts)
        .push_constant_ranges(&push_constant_ranges);
      let pipeline_layout = unsafe {
        self
          .device
          .create_pipeline_layout(&pipeline_layout_info, None)
      }?;

      // Create Descriptor Pool and Set
      let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_IMAGE)
        .descriptor_count(1)];
      let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1);
      let descriptor_pool = unsafe { self.device.create_descriptor_pool(&pool_info, None) }?;

      let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&set_layouts);
      let descriptor_set = unsafe { self.device.allocate_descriptor_sets(&alloc_info) }?[0];

      // Write descriptor set
      let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::GENERAL)
        .image_view(image.image_view.get());
      let write_descriptor_set = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
        .image_info(core::slice::from_ref(&image_info));
      unsafe {
        self
          .device
          .update_descriptor_sets(&[write_descriptor_set], &[])
      };

      // Get or Create Compute Pipeline
      let mut compute_info = pipelines::ComputeInfo::default();
      compute_info.shader_module = shader_module;
      compute_info.pipeline_layout = pipeline_layout;

      compute_info.add_specialization_constant_u32(
        vk::SpecializationMapEntry {
          constant_id: 0,
          offset: 0,
          size: 4,
        },
        8,
      );
      compute_info.add_specialization_constant_u32(
        vk::SpecializationMapEntry {
          constant_id: 1,
          offset: 4,
          size: 4,
        },
        8,
      );
      compute_info.add_specialization_constant_u32(
        vk::SpecializationMapEntry {
          constant_id: 2,
          offset: 8,
          size: 4,
        },
        8,
      );

      let compute_pipeline = res
        .pipeline_pool
        .write()
        .get_or_create_compute_pipeline(&self.device, &compute_info)?;

      // Buffer for SunParams
      let params_size = core::mem::size_of::<[f32; 6]>() as u64; // 6 floats
      let mut allocation_create_info = vk_mem::AllocationCreateInfo::default();
      allocation_create_info.usage = vk_mem::MemoryUsage::AutoPreferHost;
      allocation_create_info.flags = vk_mem::AllocationCreateFlags::MAPPED
        | vk_mem::AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE;

      let buffer_info = vk::BufferCreateInfo::default()
        .size(params_size)
        .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
      let (params_buffer, params_alloc) = unsafe {
        res
          .allocator
          .allocator
          .create_buffer(&buffer_info, &allocation_create_info)?
      };

      let alloc_info = res.allocator.allocator.get_allocation_info(&params_alloc);

      unsafe {
        let ptr = alloc_info.mapped_data as *mut [f32; 6];
        *ptr = [
          0.0,       // time
          5778.0,    // photosphereTemp
          1000000.0, // coronaTemp
          0.6,       // radius
          0.05,      // scaleHeight
          15.0,      // noiseScale
        ];
      }

      let bda_info = vk::BufferDeviceAddressInfo::default().buffer(params_buffer);
      let buffer_address = unsafe {
        self
          .buffer_device_address
          .get_buffer_device_address(&bda_info)
      };

      unsafe {
        let command_pool_info = vk::CommandPoolCreateInfo::default()
          .queue_family_index(compute_queue.family_index)
          .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = self.device.create_command_pool(&command_pool_info, None)?;

        let command_buffer_info = vk::CommandBufferAllocateInfo::default()
          .command_pool(command_pool)
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(1);
        let command_buffer = self.device.allocate_command_buffers(&command_buffer_info)?[0];

        let begin_info =
          vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self
          .device
          .begin_command_buffer(command_buffer, &begin_info)?;

        let barrier = vk::ImageMemoryBarrier2::default()
          .src_stage_mask(vk::PipelineStageFlags2::NONE)
          .src_access_mask(vk::AccessFlags2::NONE)
          .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
          .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
          .old_layout(vk::ImageLayout::UNDEFINED)
          .new_layout(vk::ImageLayout::GENERAL)
          .image(image.image.get())
          .subresource_range(
            vk::ImageSubresourceRange::default()
              .aspect_mask(vk::ImageAspectFlags::COLOR)
              .base_mip_level(0)
              .level_count(1)
              .base_array_layer(0)
              .layer_count(1),
          );
        let dep_info =
          vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier));
        self
          .synchronization2
          .cmd_pipeline_barrier2(command_buffer, &dep_info);

        self.device.cmd_bind_pipeline(
          command_buffer,
          vk::PipelineBindPoint::COMPUTE,
          compute_pipeline.get(),
        );
        self.device.cmd_bind_descriptor_sets(
          command_buffer,
          vk::PipelineBindPoint::COMPUTE,
          pipeline_layout,
          0,
          &[descriptor_set],
          &[],
        );
        let push_constants_bytes =
          core::slice::from_raw_parts(&buffer_address as *const _ as *const u8, 8);
        self.device.cmd_push_constants(
          command_buffer,
          pipeline_layout,
          vk::ShaderStageFlags::COMPUTE,
          0,
          push_constants_bytes,
        );

        let group_count_x = (component.resolution.0 + 7) / 8;
        let group_count_y = (component.resolution.1 + 7) / 8;
        let group_count_z = (component.resolution.2 + 7) / 8;
        self
          .device
          .cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);

        let mut barrier2 = vk::ImageMemoryBarrier2::default()
          .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
          .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
          .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
          .dst_access_mask(vk::AccessFlags2::SHADER_READ)
          .old_layout(vk::ImageLayout::GENERAL)
          .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
          .image(image.image.get())
          .subresource_range(
            vk::ImageSubresourceRange::default()
              .aspect_mask(vk::ImageAspectFlags::COLOR)
              .base_mip_level(0)
              .level_count(1)
              .base_array_layer(0)
              .layer_count(1),
          );

        if compute_queue.family_index != graphics_queue.family_index {
          barrier2 = barrier2
            .src_queue_family_index(compute_queue.family_index)
            .dst_queue_family_index(graphics_queue.family_index);
        } else {
          barrier2 = barrier2
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
        }

        let dep_info2 =
          vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier2));
        self
          .synchronization2
          .cmd_pipeline_barrier2(command_buffer, &dep_info2);

        self.device.end_command_buffer(command_buffer)?;

        let submit_info =
          vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&command_buffer));
        self
          .device
          .queue_submit(compute_queue.handle, &[submit_info], vk::Fence::null())?;
        self.device.queue_wait_idle(compute_queue.handle)?;

        res
          .allocator
          .allocator
          .destroy_buffer(params_buffer, &mut { params_alloc });

        self.device.destroy_command_pool(command_pool, None);
        self.device.destroy_descriptor_pool(descriptor_pool, None);
        self.device.destroy_pipeline_layout(pipeline_layout, None);
        self.device.destroy_descriptor_set_layout(set_layout, None);
      }

      // If queues are different, we need a release/acquire barrier on graphics queue.
      // The release barrier was already dispatched in the compute queue.
      if compute_queue.family_index != graphics_queue.family_index {
        unsafe {
          let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(graphics_queue.family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
          let command_pool = self.device.create_command_pool(&command_pool_info, None)?;

          let command_buffer_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
          let transition_cmd = self.device.allocate_command_buffers(&command_buffer_info)?[0];

          let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
          self
            .device
            .begin_command_buffer(transition_cmd, &begin_info)?;

          let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::GENERAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image(image.image.get())
            .subresource_range(
              vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
            )
            .src_queue_family_index(compute_queue.family_index)
            .dst_queue_family_index(graphics_queue.family_index);

          let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier));
          self
            .synchronization2
            .cmd_pipeline_barrier2(transition_cmd, &dep_info);

          self.device.end_command_buffer(transition_cmd)?;

          let submit_info =
            vk::SubmitInfo::default().command_buffers(core::slice::from_ref(&transition_cmd));
          self
            .device
            .queue_submit(graphics_queue.handle, &[submit_info], vk::Fence::null())?;
          self.device.queue_wait_idle(graphics_queue.handle)?;

          self.device.destroy_command_pool(command_pool, None);
        }
      }

      let arch_ref: Option<&_>;
      #[cfg(debug_assertions)]
      {
        arch_ref = res.sun_render_archetype.as_ref().as_ref();
      }
      #[cfg(not(debug_assertions))]
      {
        arch_ref = res.sun_render_archetype.as_ref();
      }
      let archetype = arch_ref.ok_or(GpuError::InvalidState)?;

      let descriptor_set = res
        .descriptor_pool
        .as_ref()
        .unwrap()
        .allocate(
          &self.device,
          archetype.descriptor_set_layout.get(),
          &res.discard_pool,
          timeline,
        )?
        .get();

      // Write descriptor set
      let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) // assuming it will be transitioned or is already
        .image_view(image.image_view.get())
        .sampler(res.linear_sampler.get());
      let write_descriptor_set = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(core::slice::from_ref(&image_info));
      unsafe {
        self
          .device
          .update_descriptor_sets(&[write_descriptor_set], &[])
      };

      map.insert(
        entity_id,
        resources::SunRenderResource {
          resolution: component.resolution,
          image: Some(image),
          descriptor_set: Some(unsafe { NonZeroHandle::new_unchecked(descriptor_set) }),
          is_generated: true,
        },
      );
    }

    let sun_resource = map.get(&entity_id).unwrap();

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.sun_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.sun_render_archetype.as_ref();
    }
    let archetype = arch_ref.ok_or(GpuError::InvalidState)?;

    let pipeline_key = archetype.pipeline_key.ok_or(GpuError::InvalidState)?;
    let pipeline = res
      .pipeline_pool
      .read()
      .get_graphics_pipeline(pipeline_key)
      .ok_or(GpuError::InvalidState)?;

    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());

      if let Some(ds) = sun_resource.descriptor_set {
        self.device.cmd_bind_descriptor_sets(
          cmd,
          vk::PipelineBindPoint::GRAPHICS,
          layout,
          0,
          &[ds.get()],
          &[],
        );
      }

      let cam_col = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::inverse(view)
        .unwrap()
        .column(3)
        .unwrap();
      let camera_world_pos = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
        cam_col.x(),
        cam_col.y(),
        cam_col.z(),
      );
      let model_matrix = transform.to_mat4();
      let model_inv =
        aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::inverse(model_matrix).unwrap();
      let mvp: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 = view_proj * model_matrix;

      let push_constants = crate::gpu::SunPushConstants {
        model_view_proj: mvp.into(),
        model_inv: model_inv.into(),
        camera_world_pos: camera_world_pos.into(),
        _unused: 0,
      };

      for range in &archetype.push_contant_ranges {
        let push_constants_bytes = core::slice::from_raw_parts(
          &push_constants as *const _ as *const u8,
          core::mem::size_of::<crate::gpu::SunPushConstants>(),
        );
        self.device.cmd_push_constants(
          cmd,
          layout,
          range.stage_flags,
          range.offset,
          &push_constants_bytes[range.offset as usize..(range.offset + range.size) as usize],
        );
      }

      // Draw a cube. We could use vertex buffers, or just generate inside the shader. But the archetype has TRIANGLE_STRIP
      // Let's use TRIANGLE_LIST and draw a box?
      // Actually, wait, the archetype is TRIANGLE_STRIP. We can draw a cube with a triangle strip of 14 vertices.
      // But we have no vertex buffer bound. So the shader must generate vertices, or we just draw 14 vertices and the shader handles it.
      // sun_volume.vert says: `layout(location = 0) in vec3 inPosition; // Assuming a unit cube [-1, 1]`
      // so it expects a vertex buffer! We need a cube vertex buffer.
      self.device.cmd_draw(cmd, 14, 1, 0, 0);
    }

    // In a real implementation we would bind the raymarching pipeline,
    // bind the 3D texture, push constants for the sun raymarching and draw a cube.
    Ok(())
  }

  fn render_sky(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    inv_view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
  ) -> GpuResult<()> {
    let mut res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    if res.sky_render_archetype.is_none() {
      drop(res);
      let mut wres = self.res.write();
      if wres.sky_render_archetype.is_none() {
        let (vkey, fkey) = self.ensure_sky_shader_modules(&wres)?;
        let handle = wres.live_presentation_engines.read().keys().next().copied();
        if let Some(h) = handle {
          wres.create_sky_archetype(&self.device, vkey, fkey, self.depth_stencil_format, h)?;
        }
      }
      drop(wres);
      res = self.res.read();
    }

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.sky_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.sky_render_archetype.as_ref();
    }
    let archetype = arch_ref.ok_or(GpuError::InvalidState)?;

    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    let push_constants = crate::gpu::SkyPushConstants {
      inv_view_proj: inv_view_proj.into(),
    };

    unsafe {
      let push_constants_bytes = core::slice::from_raw_parts(
        &push_constants as *const _ as *const u8,
        core::mem::size_of::<crate::gpu::SkyPushConstants>(),
      );
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constants_bytes,
      );

      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        0,
        &[archetype.descriptor_set.get()],
        &[],
      );

      let pipeline = res
        .pipeline_pool
        .read()
        .get_graphics_pipeline(archetype.pipeline_key.unwrap())
        .ok_or(GpuError::InvalidState)?;

      self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
      self.device.cmd_draw(cmd, 3, 1, 0, 0); // draw a full-screen triangle
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

    let mut signal_semaphores = heapless::Vec::<vk::Semaphore, 2>::new();
    let mut timeline_values = heapless::Vec::<u64, 2>::new();

    if let Some(sem) = signal_semaphore {
      let _ = signal_semaphores.push(sem.get());
      let _ = timeline_values.push(0);
    }
    let _ = signal_semaphores.push(res.timeline_semaphore.get());
    let _ = timeline_values.push(next_timeline_value);

    let wait_semaphores = [wait_semaphore.get()];
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
      self
        .device
        .queue_submit(
          graphics_queue.handle,
          &[submit_info],
          submission_fence.get(),
        )
        .map_err(|e| {
          oshal::log!("queue_submit failed: {:?}", e);
          e
        })?;
    }

    res
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
