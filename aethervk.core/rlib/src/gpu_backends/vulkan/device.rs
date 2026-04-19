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
use aethervk_oshal_rlib::math::quaternion::Quaternion;
use aethervk_oshal_rlib::math::matrix::MatrixVectorMul;

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
  emissive_intensity: f32,
  emissive_color: [f32; 3],
) -> ResourceUploadResult {
  ResourceUploadResult {
    pipeline: unsafe { archetype.pipeline_key.unwrap_unchecked() },
    outline_pipeline: archetype.outline_pipeline_key,
    buffers: handle.into(),
    texture_flags: value.frontend_texture_flags(),
    emissive_intensity,
    emissive_color,
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

  #[cfg(debug_assertions)]
  sky_render_archetype: DropTracker<TrackedOption<resources::SkyRenderResourceArchetype, 3>, 3>,
  #[cfg(not(debug_assertions))]
  sky_render_archetype: Option<resources::SkyRenderResourceArchetype>,

  #[cfg(debug_assertions)]
  grid_render_archetype: DropTracker<TrackedOption<resources::GridRenderResourceArchetype, 4>, 4>,
  #[cfg(not(debug_assertions))]
  grid_render_archetype: Option<resources::GridRenderResourceArchetype>,

  #[cfg(debug_assertions)]
  minimap_render_archetype: DropTracker<TrackedOption<resources::MinimapRenderResourceArchetype, 5>, 5>,
  #[cfg(not(debug_assertions))]
  minimap_render_archetype: Option<resources::MinimapRenderResourceArchetype>,

  #[cfg(debug_assertions)]
  text_render_archetype: DropTracker<TrackedOption<resources::TextRenderResourceArchetype, 6>, 6>,
  #[cfg(not(debug_assertions))]
  text_render_archetype: Option<resources::TextRenderResourceArchetype>,

  #[cfg(debug_assertions)]
  bvh_render_archetype: DropTracker<TrackedOption<resources::BvhRenderResourceArchetype, 7>, 7>,
  #[cfg(not(debug_assertions))]
  bvh_render_archetype: Option<resources::BvhRenderResourceArchetype>,

  sky_image: Option<resources::Image>,

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

    let outline_graphics_info = graphics_info
      .clone()
      .with_pipeline_flags(PipelineFlags::CULL_BACK | PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::STENCIL_ENABLE)
      .with_stencil_compare_op(StencilCompareOp::NotEqual)
      .with_stencil_logic_op(StencilLogicOp::None)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(255)
      .with_stencil_write_mask(0)
      .clone();

    if let Some(outline_key) = archetype.outline_pipeline_key {
      write_pipeline.discard_graphics_pipeline_if_present(outline_key, &self.discard_pool, timeline);
    }
    let outline_pipeline_key = outline_graphics_info.pipeline_key();
    write_pipeline.get_or_create_graphics_pipeline(device, &outline_graphics_info)?;
    archetype.outline_pipeline_key = Some(outline_pipeline_key);

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
    device: &vulkan::device::LogicalDevice,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
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
            y: presentation_engine_state.extent().1 as f32,
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

    let outline_graphics_info = pipeline_graphics_info
      .clone()
      .with_pipeline_flags(PipelineFlags::CULL_BACK | PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::STENCIL_ENABLE)
      .with_stencil_compare_op(StencilCompareOp::NotEqual)
      .with_stencil_logic_op(StencilLogicOp::None)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(255)
      .with_stencil_write_mask(0)
      .clone();

    let outline_pipeline_key = outline_graphics_info.pipeline_key();
    self
      .pipeline_pool
      .write()
      .get_or_create_graphics_pipeline(&device, &outline_graphics_info)?;

    {
      let val = unsafe {
        self
          .physical_mesh_render_archetype
          .take()
          .unwrap_unchecked()
      }
      .with_graphics_info(pipeline_graphics_info)
      .with_outline_pipeline_key(outline_pipeline_key);

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
    device: &LogicalDevice,
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
    let res = unsafe { resources::SunRenderResourceArchetype::new(device) }?;
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
            y: presentation_engine_state.extent().1 as f32,
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
      .with_pipeline_flags(PipelineFlags::CULL_ALL | PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::NO_DEPTH_WRITE) // No culling so we see it from inside and outside (yes, cull all means no culling)
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
    let live_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_state = live_engines_lock.get(&handle).unwrap().read();
    let shader_manager = self.shader_manager.read();
    let vertex_shader = shader_manager.get(vertex_shader_key).unwrap();
    let fragment_shader = shader_manager.get(fragment_shader_key).unwrap();

    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(1)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
    let set_layouts = [set_layout];

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(64)];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(&set_layouts)
      .push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    let mut arch = resources::SkyRenderResourceArchetype {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      pipeline_key: None,
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(set_layout) },
      descriptor_set: None,
    };

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
            height: -(presentation_engine_state.extent().1 as f32),
            x: 0.0,
            y: presentation_engine_state.extent().1 as f32,
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
          .clone(),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(
        PipelineFlags::CULL_ALL | PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST | PipelineFlags::INVERT_FRONT_FACE,
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
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    self
      .pipeline_pool
      .write()
      .get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;
    arch.pipeline_key = Some(pipeline_key);

    #[cfg(not(debug_assertions))]
    {
      self.sky_render_archetype = Some(arch);
    }
    #[cfg(debug_assertions)]
    {
      self.sky_render_archetype = DropTracker::new(TrackedOption::some(arch));
    }

    Ok(())
  }

  fn create_grid_archetype(
    &mut self,
    device: &ash::Device,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    if self.grid_render_archetype.is_some() {
      return Err(GpuError::InvalidState);
    }
    let live_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_state = live_engines_lock.get(&handle).unwrap().read();
    let shader_manager = self.shader_manager.read();
    let vertex_shader = shader_manager.get(vertex_shader_key).unwrap();
    let fragment_shader = shader_manager.get(fragment_shader_key).unwrap();

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(256)];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    let mut arch = resources::GridRenderResourceArchetype {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      pipeline_key: None,
    };

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
            height: -(presentation_engine_state.extent().1 as f32),
            x: 0.0,
            y: presentation_engine_state.extent().1 as f32,
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
          .clone(),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(
        PipelineFlags::CULL_ALL | PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::INVERT_FRONT_FACE,
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
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    self
      .pipeline_pool
      .write()
      .get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;
    arch.pipeline_key = Some(pipeline_key);

    #[cfg(not(debug_assertions))]
    {
      self.grid_render_archetype = Some(arch);
    }
    #[cfg(debug_assertions)]
    {
      self.grid_render_archetype = DropTracker::new(TrackedOption::some(arch));
    }

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
    device: &LogicalDevice,
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
    let res = unsafe { resources::CursorRenderResourceArchetype::new(device) }?;
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
            y: presentation_engine_state.extent().1 as f32,
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
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_TEST | PipelineFlags::CULL_ALL | PipelineFlags::INVERT_FRONT_FACE) // NO Culling, NO Depth Test (Yes, cull all means no culling)
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

  fn ensure_text_shader_modules(
    &self,
    device: &ash::Device,
  ) -> GpuResult<(ShaderKey, ShaderKey)> {
    use aethervk_oshal_rlib::os::fs::FileSystemObject;
    let vert_path: PathBuf;
    let frag_path: PathBuf;

    #[cfg(debug_assertions)]
    {
      let mut exe_path = oshal::os::fs::current_exe().unwrap();
      while let Some(p) = exe_path.parent() {
        if p.join("assets").is_dir() {
          exe_path = p.join("assets");
          break;
        }
        exe_path = p;
      }
      vert_path = exe_path.join("text.vert.spv");
      frag_path = exe_path.join("text.frag.spv");
    }
    #[cfg(not(debug_assertions))]
    {
      vert_path = PathBuf::from("assets/text.vert.spv");
      frag_path = PathBuf::from("assets/text.frag.spv");
    }

    let mut shader_manager = self.shader_manager.write();
    let vkey = shader_manager.get_or_load(device, vert_path.as_ref(), "main", spirv::ExecutionModel::Vertex)?;
    let fkey = shader_manager.get_or_load(device, frag_path.as_ref(), "main", spirv::ExecutionModel::Fragment)?;

    Ok((vkey, fkey))
  }

  fn create_text_archetype(
    &mut self,
    device: &vulkan::device::LogicalDevice,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    timeline: u64,
    handle: PresentationEngineHandle,
  ) -> GpuResult<()> {
    if self.text_render_archetype.is_some() {
      return Err(GpuError::InvalidState);
    }
    let live_engines_lock = self.live_presentation_engines.read();
    let presentation_engine_state = live_engines_lock.get(&handle).unwrap().read();
    let shader_manager = self.shader_manager.read();
    let vertex_shader = shader_manager.get(vertex_shader_key).unwrap();
    let fragment_shader = shader_manager.get(fragment_shader_key).unwrap();

    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(1)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
    let set_layouts = [set_layout];

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(48)]; // vec2 + vec2 + vec4 + vec4 = 48 bytes
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(&set_layouts)
      .push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    let mut arch = resources::TextRenderResourceArchetype {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      pipeline_key: None,
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(set_layout) },
      descriptor_pool: None,
      descriptor_set: None,
      font_texture: None,
      font_sampler: None,
      font_atlas: None,
      allocator_raw: Some(self.allocator.allocator.get_raw()),
    };

    // Upload Font Atlas
    let font_path: aethervk_oshal_rlib::os::fs::PathBuf;
    #[cfg(debug_assertions)]
    {
      let assets_dir: aethervk_oshal_rlib::os::fs::PathBuf = {
        use aethervk_oshal_rlib::os;
        use aethervk_oshal_rlib::os::fs::FileSystemObject;
        let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
        if args.len() > 1 {
          aethervk_oshal_rlib::os::fs::PathBuf::from(&args[1])
        } else {
          let exe_path = os::fs::current_exe().map_err(|_| GpuError::BackendSpecific("Fail".into()))?;
          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;
          while let Some(p) = path {
            let test_path = p.join("assets");
            if test_path.is_dir() { assets_dir = Some(test_path); break; }
            path = p.parent();
          }
          assets_dir.expect("Fail")
        }
      };
      font_path = assets_dir.join("fonts/JetBrainsMono-Bold.ttf");
    }
    #[cfg(not(debug_assertions))]
    {
      font_path = aethervk_oshal_rlib::os::fs::PathBuf::from("assets/fonts/JetBrainsMono-Bold.ttf");
    }

    if let Ok(mapped_file) = aethervk_oshal_rlib::os::files::MappedFile::new(font_path) {
      let font_data = mapped_file.as_slice();
      if let Some(atlas) = crate::scene::text::create_ascii_atlas(font_data, 64.0) {
        let texture = crate::simulation::comet::Texture {
           data: atlas.image_data.clone(),
           format: crate::simulation::comet::TexelFormat::R8_UNORM,
           width: atlas.width,
           height: atlas.height,
           has_mipmaps: false,
        };

        let command_pool_info = vk::CommandPoolCreateInfo::default()
          .queue_family_index(queue.family_index)
          .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None) }?;

        let alloc_info = vk::CommandBufferAllocateInfo::default()
          .command_pool(command_pool)
          .level(vk::CommandBufferLevel::PRIMARY)
          .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }?[0];

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(command_buffer, &begin_info)? };

        if let Ok(image) = resources::Image::new_2d(device, &self.allocator.allocator, command_buffer, &self.discard_pool, timeline, &texture, vk::ImageUsageFlags::SAMPLED, "FontAtlas") {
          let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
          if let Ok(sampler) = unsafe { device.create_sampler(&sampler_info, None) } {
            let pool_sizes = [vk::DescriptorPoolSize::default().ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).descriptor_count(1)];
            let pool_info = vk::DescriptorPoolCreateInfo::default().pool_sizes(&pool_sizes).max_sets(1);
            if let Ok(pool) = unsafe { device.create_descriptor_pool(&pool_info, None) } {
              let set_layouts = [arch.descriptor_set_layout.get()];
              let alloc_info = vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts);
              if let Ok(sets) = unsafe { device.allocate_descriptor_sets(&alloc_info) } {
                let set = sets[0];
                let image_info = [vk::DescriptorImageInfo::default()
                  .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                  .image_view(image.image_view.get())
                  .sampler(sampler)];
                let write = vk::WriteDescriptorSet::default()
                  .dst_set(set)
                  .dst_binding(0)
                  .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                  .image_info(&image_info);
                unsafe { device.update_descriptor_sets(&[write], &[]) };
                
                arch.font_texture = Some(image);
                arch.font_sampler = Some(sampler);
                arch.descriptor_pool = Some(unsafe { NonZeroHandle::new_unchecked(pool) });
                arch.descriptor_set = Some(set);
                arch.font_atlas = Some(atlas);
              }
            }
          }
        }
        unsafe { device.end_command_buffer(command_buffer)? };

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
        
        unsafe {
          device.queue_submit(
            queue.handle,
            &[submit_info],
            vk::Fence::null(),
          )?;
          device.queue_wait_idle(queue.handle)?;
          device.destroy_command_pool(command_pool, None);
        }
      }
    }

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
            height: -(presentation_engine_state.extent().1 as f32),
            x: 0.0,
            y: presentation_engine_state.extent().1 as f32,
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
          .clone(),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(
        PipelineFlags::CULL_ALL | PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST | PipelineFlags::INVERT_FRONT_FACE,
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
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    self
      .pipeline_pool
      .write()
      .get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;
    arch.pipeline_key = Some(pipeline_key);

    #[cfg(not(debug_assertions))]
    {
      self.text_render_archetype = Some(arch);
    }
    #[cfg(debug_assertions)]
    {
      self.text_render_archetype = DropTracker::new(TrackedOption::some(arch));
    }

    Ok(())
  }

  fn has_discardables(&self) -> bool {
    self.physical_mesh_render_archetype.is_some()
      || self.physical_mesh_resources.read().is_some()
      || self.sun_render_archetype.is_some()
      || self.sun_resources.read().is_some()
      || self.cursor_render_archetype.is_some()
      || self.sky_render_archetype.is_some()
      || self.grid_render_archetype.is_some()
      || self.minimap_render_archetype.is_some()
      || self.text_render_archetype.is_some()
      || self.bvh_render_archetype.is_some()
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
        if let Some(buffer) = resource.params_buffer {
          unsafe {
            vk_mem::ffi::vmaDestroyBuffer(
              self.allocator.allocator.get_raw(),
              buffer,
              resource.params_alloc.unwrap().get_raw(),
            );
          }
        }
        if let Some(layout) = resource.compute_pipeline_layout {
          unsafe { device.destroy_pipeline_layout(layout, None) };
        }
        if let Some(pool) = resource.compute_descriptor_pool {
          unsafe { device.destroy_descriptor_pool(pool, None) };
        }
        if let Some(layout) = resource.compute_descriptor_set_layout {
          unsafe { device.destroy_descriptor_set_layout(layout, None) };
        }
      }
    }
    if let Some(mut archetype) = self.cursor_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.sky_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.grid_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.minimap_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.text_render_archetype.take() {
      archetype.discard(device, &self.discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.bvh_render_archetype.take() {
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
      #[cfg(debug_assertions)]
      sky_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      sky_render_archetype: None,
      #[cfg(debug_assertions)]
      grid_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      grid_render_archetype: None,
      #[cfg(debug_assertions)]
      minimap_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      minimap_render_archetype: None,
      #[cfg(debug_assertions)]
      text_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      text_render_archetype: None,
      #[cfg(debug_assertions)]
      bvh_render_archetype: DropTracker::new(TrackedOption::none()),
      #[cfg(not(debug_assertions))]
      bvh_render_archetype: None,
      sky_image: None,
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

// TODO Store api version and redirect methods from extensions to core if promoted
pub(super) struct LogicalDevice {
  pub handle: ash::Device,
  /// Note: Remove if API_VERSION_1_2
  pub create_renderpass2: ash::khr::create_renderpass2::Device,
  pub buffer_device_address: ash::khr::buffer_device_address::Device,
  pub timeline_semaphore: ash::khr::timeline_semaphore::Device,
  /// Note: Remove if API_VERSION_1_3
  pub synchronization2: ash::khr::synchronization2::Device,

  #[cfg(debug_assertions)]
  pub debug_utils: ash::ext::debug_utils::Device,

  #[cfg(target_vendor = "apple")]
  pub metal_objects: ash::ext::metal_objects::Device,
}

impl core::ops::Deref for LogicalDevice {
  type Target = ash::Device;

  fn deref(&self) -> &Self::Target {
    &self.handle
  }
}

impl LogicalDevice {
  #[cfg(debug_assertions)]
  pub fn set_debug_name<T: vk::Handle>(&self, object: T, name: &str) {
    use core::str::FromStr;

    if name.is_empty() {
      return;
    }
    let name_cstr = alloc::ffi::CString::from_str(name).unwrap();
    let name_info = vk::DebugUtilsObjectNameInfoEXT::default()
      .object_handle(object)
      .object_name(&name_cstr);

    unsafe {
      self
        .debug_utils
        .set_debug_utils_object_name(&name_info)
        .expect(&alloc::format!("failed to set name {}", name));
    }
  }

  #[cfg(not(debug_assertions))]
  #[inline]
  pub fn set_debug_name<T: vk::Handle>(&self, _object: T, _name: &str) {
    // This is a no-op in release builds, and should be optimized away.
  }
}

pub trait VulkanDebugNameExt: Sized {
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self;
}
pub trait VmaDebugNameExt: Sized {
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self;
}

// 2. Apply to Results containing Vulkan Handles
impl<T: vk::Handle + Copy> VulkanDebugNameExt for ash::prelude::VkResult<T> {
  #[inline]
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self {
    if let Ok(handle) = &self {
      device.set_debug_name(*handle, name);
    }
    self
  }
}

// Implements the trait for ANY `Result` returning a VMA Tuple (Buffer/Image + Allocation)
impl<T, A> VmaDebugNameExt for ash::prelude::VkResult<(T, A)>
where
  T: vk::Handle + Copy,
{
  #[inline]
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self {
    if let Ok((handle, _alloc)) = &self {
      // Apply the debug name to the Vulkan handle (VkBuffer / VkImage)
      // This guarantees it shows up properly in RenderDoc!
      device.set_debug_name(*handle, name);
    }

    // Pass the Result unmodified down the chain
    self
  }
}

impl<T, A1, A2> VmaDebugNameExt for ash::prelude::VkResult<(T, A1, A2)>
where
  T: vk::Handle + Copy,
{
  #[inline]
  fn with_name(self, device: &LogicalDevice, name: &str) -> Self {
    if let Ok((handle, _alloc, _alloc_info)) = &self {
      // Apply the debug name to the Vulkan handle (VkBuffer / VkImage)
      // This guarantees it shows up properly in RenderDoc!
      device.set_debug_name(*handle, name);
    }

    // Pass the Result unmodified down the chain
    self
  }
}

pub(super) struct Device<'a> {
  query_result: utils::PhysicalDeviceQueryResult,
  queues: Queues,
  instance: &'a instance::Instance,

  device: LogicalDevice,

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
            && query_result.graphics_queue_family_index == queue_buffer[i].family_index
          {
            queue_ref_map
              .insert(QueueId::GRAPHICS, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::GRAPHICS as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::COMPUTE as u32)) == 0
            && query_result.compute_queue_family_index == queue_buffer[i].family_index
          {
            queue_ref_map
              .insert(QueueId::COMPUTE, unsafe { queue_buffer.get_unchecked(i) })
              .unwrap();
            queue_type_inserted |= 1u32 << QueueId::COMPUTE as u32;
          }
          if (queue_type_inserted & (1u32 << QueueId::TRANSFER as u32)) == 0
            && query_result.transfer_queue_family_index == queue_buffer[i].family_index
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
    let timeline_semaphore = ash::khr::timeline_semaphore::Device::new(&instance.instance, &device);

    #[cfg(debug_assertions)]
    let debug_utils = ash::ext::debug_utils::Device::new(&instance.instance, &device);
    #[cfg(target_vendor = "apple")]
    let metal_objects = ash::ext::metal_objects::Device::new(&instance.instance, &device);

    Ok(Self {
      query_result: *chosen_physical_device_query_result,
      device: LogicalDevice {
        timeline_semaphore,
        handle: device,
        create_renderpass2,
        synchronization2,
        buffer_device_address,
        #[cfg(target_vendor = "apple")]
        metal_objects,
        #[cfg(debug_assertions)]
        debug_utils,
      },
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

  fn ensure_grid_shader_modules(
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
          let mut iter: i32 = 0;
          const MAX_ITER: i32 = 32;

          while let Some(p) = path {
            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            if iter >= MAX_ITER {
              break;
            }
            path = p.parent();
            iter += 1;
          }
          assets_dir.unwrap()
        }
      };
      vert_path = assets_dir.join("grid.vert.spv");
      frag_path = assets_dir.join("grid.frag.spv");
    }
    #[cfg(not(debug_assertions))]
    {
      todo!();
    }

    let mut shader_manager = res.shader_manager.write();
    let vkey = shader_manager.get_or_load(
      &self.device,
      &vert_path,
      "main",
      spirv::ExecutionModel::Vertex,
    )?;
    let fkey = shader_manager.get_or_load(
      &self.device,
      &frag_path,
      "main",
      spirv::ExecutionModel::Fragment,
    )?;

    Ok((vkey, fkey))
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
          let exe_path = os::fs::current_exe().map_err(|_| {
            GpuError::BackendSpecific(
              "Failed to get executable path for debug asset loading".into(),
            )
          })?;

          let mut path = exe_path.parent();
          let mut assets_dir: Option<PathBuf> = None;
          let mut iter: i32 = 0;
          const MAX_ITER: i32 = 32;

          while let Some(p) = path {
            let test_path = p.join("assets");
            if test_path.is_dir() {
              assets_dir = Some(test_path);
              break;
            }
            if iter >= MAX_ITER {
              break;
            }
            path = p.parent();
            iter += 1;
          }
          assets_dir.unwrap()
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
    let vkey = shader_manager.get_or_load(
      &self.device,
      &vert_path,
      "main",
      spirv::ExecutionModel::Vertex,
    )?;
    let fkey = shader_manager.get_or_load(
      &self.device,
      &frag_path,
      "main",
      spirv::ExecutionModel::Fragment,
    )?;

    Ok((vkey, fkey))
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
    aethervk_oshal_rlib::log!("Device::drop started. Waiting for device idle...");
    unsafe { self.device.device_wait_idle().unwrap_unchecked() };
    aethervk_oshal_rlib::log!("Device::drop device_wait_idle complete. Starting cleanup...");

    self.res.write().cleanup(&self.device);

    aethervk_oshal_rlib::log!("Device::drop cleanup complete. Destroying device...");
    // in the end, destroy the device
    unsafe { self.device.destroy_device(None) };
    aethervk_oshal_rlib::log!("Device::drop finished.");
  }
}


fn ensure_bvh_shader_modules(
  device: &LogicalDevice,
  res: &DeviceResources,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  #[cfg(debug_assertions)]
  {
    let assets_dir: aethervk_oshal_rlib::os::fs::PathBuf = {
      use aethervk_oshal_rlib::os;
      use aethervk_oshal_rlib::os::fs::FileSystemObject;

      let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
      if args.len() > 1 {
        aethervk_oshal_rlib::os::fs::PathBuf::from(&args[1])
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

    vert_path = assets_dir.join("bvh_debug.vert.spv");
    frag_path = assets_dir.join("bvh_debug.frag.spv");
  }
  #[cfg(not(debug_assertions))]
  {
    vert_path = PathBuf::from("assets/bvh_debug.vert.spv");
    frag_path = PathBuf::from("assets/bvh_debug.frag.spv");
  }

  let mut shader_manager = res.shader_manager.write();
  let vkey = shader_manager.get_or_load(device, vert_path.as_ref(), "main", spirv::ExecutionModel::Vertex)?;
  let fkey = shader_manager.get_or_load(device, frag_path.as_ref(), "main", spirv::ExecutionModel::Fragment)?;

  Ok((vkey, fkey))
}

fn ensure_minimap_shader_modules(
  device: &LogicalDevice,
  res: &DeviceResources,
) -> GpuResult<(shader_manager::ShaderKey, shader_manager::ShaderKey)> {
  let vert_path: aethervk_oshal_rlib::os::fs::PathBuf;
  let frag_path: aethervk_oshal_rlib::os::fs::PathBuf;

  #[cfg(debug_assertions)]
  {
    let assets_dir: aethervk_oshal_rlib::os::fs::PathBuf = {
      use aethervk_oshal_rlib::os;
      use aethervk_oshal_rlib::os::fs::FileSystemObject;

      let args = os::env::args().map_err(|_| GpuError::InvalidArgument)?;
      if args.len() > 1 {
        aethervk_oshal_rlib::os::fs::PathBuf::from(&args[1])
      } else {
        let exe_path = aethervk_oshal_rlib::os::fs::current_exe().unwrap();
        let mut path = exe_path.parent();
        let mut assets_dir: Option<aethervk_oshal_rlib::os::fs::PathBuf> = None;
        while let Some(p) = path {
          let test_path = p.join("assets");
          if test_path.is_dir() { assets_dir = Some(test_path); break; }
          path = p.parent();
        }
        assets_dir.unwrap()
      }
    };
    vert_path = assets_dir.join("minimap.vert.spv");
    frag_path = assets_dir.join("minimap.frag.spv");
  }
  #[cfg(not(debug_assertions))]
  { todo!(); }

  let mut shader_manager = res.shader_manager.write();
  let vert_key = shader_manager.get_or_load(device, &vert_path, "main", spirv::ExecutionModel::Vertex)?;
  let frag_key = shader_manager.get_or_load(device, &frag_path, "main", spirv::ExecutionModel::Fragment)?;
  Ok((vert_key, frag_key))
}

impl<'a> RenderDevice for Device<'a> {
  fn get_native_prop(&self, prop: NativeGpuProperty) -> Option<*mut core::ffi::c_void> {
    #[cfg(target_vendor = "apple")]
    {
      if prop == NativeGpuProperty::VulkanMetalDeviceId {
        let mut metal_device_info = vk::ExportMetalDeviceInfoEXT::default();
        let mut metal_objects_info =
          vk::ExportMetalObjectsInfoEXT::default().push_next(&mut metal_device_info);
        unsafe {
          (self.device.metal_objects.fp().export_metal_objects_ext)(
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
      "Vulkan Device Info
\
       ------------------
\
       Name: {}
\
       Vendor ID: {:#X} ({})
\
       Device ID: {:#X}
\
       Type: {}
\
       API Version: {}.{}.{}
\
       Driver Version: {}
\
       Queue Families: {}
",
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

  fn init_archetypes(&self, handle: crate::gpu::PresentationEngineHandle) -> GpuResult<()> {
    let mut wres = self.res.write();
    let timeline = wres.get_timeline_semaphore_cached_value() + 1;

    if wres.physical_mesh_render_archetype.is_none() {
      let (vkey, fkey) = self.ensure_physical_mesh_shader_modules(&wres)?;
      wres.create_physical_mesh_archetype(
        &self.device,
        vkey,
        fkey,
        self.depth_stencil_format,
        &self.queues.get_graphics_queue(),
        handle,
        timeline,
      )?;
    }

    if wres.cursor_render_archetype.is_none() {
      let (vkey, fkey) = self.ensure_cursor_shader_modules(&wres)?;
      wres.create_cursor_archetype(&self.device, vkey, fkey, self.depth_stencil_format, handle)?;
    }

    if wres.sun_render_archetype.is_none() {
      let (vkey, fkey) = self.ensure_sun_shader_modules(&wres)?;
      wres.create_sun_archetype(&self.device, vkey, fkey, self.depth_stencil_format, handle)?;
    }

    if wres.sky_render_archetype.is_none() {
      let (vkey, fkey) = self.ensure_sky_shader_modules(&wres)?;
      wres.create_sky_archetype(&self.device, vkey, fkey, self.depth_stencil_format, handle)?;
    }

    if wres.grid_render_archetype.is_none() {
      let (vkey, fkey) = self.ensure_grid_shader_modules(&wres)?;
      wres.create_grid_archetype(&self.device, vkey, fkey, self.depth_stencil_format, handle)?;
    }

    if wres.text_render_archetype.is_none() {
      let (vkey, fkey) = wres.ensure_text_shader_modules(&self.device)?;
      wres.create_text_archetype(&self.device, vkey, fkey, self.depth_stencil_format, &self.queues.get_graphics_queue(), timeline, handle)?;
    }

    Ok(())
  }

  fn set_line_width(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    width: f32,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();
    let cmd = data.command_buffer.get();
    unsafe {
      self.device.cmd_set_line_width(cmd, width);
    }
    Ok(())
  }

  fn render_frame(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    _viewports: &crate::gpu::viewport::ViewportQuadTree,
    render_scene: &crate::gpu::frame::RenderScene,
  ) -> GpuResult<()> {
    use aethervk_oshal_rlib::math::matrix::{Matrix4, MatrixVectorMul, SquareMatrix};
    let camera = &render_scene.camera;
      let view =
      <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as Matrix4>::from_columns(
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 1.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_quat_custom_frame(
        camera.0.rotation.conjugate(),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as Matrix4>::translation(camera.0.position * -1.0);
    let proj = camera.1.projection;
    let view_proj = proj * view;

    if let Some((sky_entity, sky_comp)) = &render_scene.sky {
      let sky_view = <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as Matrix4>::from_columns(
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(1.0, 0.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, -1.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 1.0, 0.0, 0.0),
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(0.0, 0.0, 0.0, 1.0),
      ) * <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::Matrix4>::from_quat_custom_frame(
        camera.0.rotation.conjugate(),
      );
      let sky_view_proj = proj * sky_view;
      self.render_sky(cmd_buffer, *sky_entity, sky_comp, sky_view_proj)?;
    }

    let sun_pos = render_scene.sun.as_ref().map(|(_, _, t)| t.position).unwrap_or_else(|| aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(0.0, 0.0, 0.0));

    if let Some((sun_entity, sun_comp, sun_transform)) = &render_scene.sun {
      self.render_sun(
        cmd_buffer,
        *sun_entity,
        sun_comp,
        sun_transform,
        view,
        view_proj,
      )?;
    }

    for draw_call in &render_scene.draw_calls {
      crate::gpu::frame::do_draw_call(self, view_proj, camera.0.position, sun_pos, [1.0, 1.0, 1.0, 1.0], cmd_buffer, draw_call)?;
    }

    if let Some((grid_entity, grid_comp)) = &render_scene.grid {
      self.render_grid(
        cmd_buffer,
        *grid_entity,
        grid_comp,
        view_proj,
        camera.0.position,
        camera.1.near_plane,
        camera.1.far_plane,
      )?;
    }

    for cursor_call in &render_scene.cursor_calls {
      crate::gpu::frame::do_draw_cursor(self, view, view_proj, cmd_buffer, cursor_call)?;
    }

    Ok(())
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
      engine.write().acquire_next_image(&self.device, self.queues.get_graphics_queue().handle)
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
    debug_name: &str,
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
            component.emissive_intensity,
            component.emissive_color,
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
          &res.allocator.allocator,
          command_buffer,
          &res.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureAlbedo_{}", debug_name),
        )
        .ok()
      });
      let normal_image = component.mesh.normal_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::NORMAL;
        Image::new_2d(
          &self.device,
          &res.allocator.allocator,
          command_buffer,
          &res.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureNormal_{}", debug_name),
        )
        .ok()
      });
      let roughness_image = component.mesh.roughness_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::ROUGHNESS;
        Image::new_2d(
          &self.device,
          &res.allocator.allocator,
          command_buffer,
          &res.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureRoughness_{}", debug_name),
        )
        .ok()
      });
      let ao_image = component.mesh.ao_map.as_ref().and_then(|t| {
        texture_flags |= TextureFlags::AO;
        Image::new_2d(
          &self.device,
          &res.allocator.allocator,
          command_buffer,
          &res.discard_pool,
          current_frame_timeline,
          &t,
          vk::ImageUsageFlags::SAMPLED,
          &alloc::format!("TextureAO_{}", debug_name),
        )
        .ok()
      });

      let resource = unsafe {
        let descriptor_set = archetype.create_descriptor_set_from_layout_at_index(
          &self.device,
          res.descriptor_pool.as_ref().unwrap_unchecked(),
          &res.discard_pool,
          0,
          debug_name,
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
          debug_name,
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

    let outline_pipeline_key = archetype.outline_pipeline_key;
    let pipeline_key = unsafe { archetype.pipeline_key.unwrap_unchecked() };

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
      outline_pipeline: outline_pipeline_key,
      buffers: physical_mesh_id.into(),
      texture_flags,
      emissive_intensity: component.emissive_intensity,
      emissive_color: component.emissive_color,
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

    // Create sky image 2048x2048
    let sky_image = resources::Image::new_storage_2d(
      &self.device,
      &res.allocator.allocator,
      2048,
      2048,
      vk::Format::R16G16B16A16_SFLOAT,
      graphics_queue.family_index,
      compute_queue.family_index,
      "Sky",
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
      vk::SpecializationMapEntry {
        constant_id: 0,
        offset: 0,
        size: 4,
      },
      16,
    );
    compute_info.add_specialization_constant_u32(
      vk::SpecializationMapEntry {
        constant_id: 1,
        offset: 4,
        size: 4,
      },
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
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(sky_image.image.get())
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
        .device
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
        .cmd_dispatch(command_buffer, 2048 / 16, 2048 / 16, 1);

      // Transition to SHADER_READ_ONLY_OPTIMAL
      let mut barrier2 = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
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
        );

      barrier2 = barrier2
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

      let dep_info2 =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier2));
      self
        .device
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
      outline_pipeline: None,
      buffers: crate::gpu::NULL_GPU_RESOURCE, // no buffers
      texture_flags: TextureFlags::empty(),
      emissive_intensity: 0.0,
      emissive_color: [0.0; 3],
    })
  }

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
          .device.synchronization2
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
          .device.synchronization2
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
      self.device.create_renderpass2.cmd_begin_render_pass2(
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

  fn update_sun(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    component: &crate::scene::SunComponent,
  ) -> GpuResult<()> {
    let mut res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    if res.sun_render_archetype.is_none() {
      drop(res);
      let mut wres = self.res.write();
      if wres.sun_render_archetype.is_none() {
        let (vkey, fkey) = self.ensure_sun_shader_modules(&wres)?;
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
        "Sun",
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
          timeline as f32 * 0.016, // time (TODO pass time)
          5778.0,                  // photosphereTemp
          1000000.0,               // coronaTemp
          0.6,                     // radius
          0.05,                    // scaleHeight
          15.0,                    // noiseScale
        ];
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

      let graphics_descriptor_set = res
        .descriptor_pool
        .as_ref()
        .unwrap()
        .allocate(
          &self.device,
          archetype.descriptor_set_layout.get(),
          &res.discard_pool,
          timeline,
          "Sun",
        )?
        .get();

      // Write graphics descriptor set
      let image_info = vk::DescriptorImageInfo::default()
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image_view(image.image_view.get())
        .sampler(res.linear_sampler.get());
      let write_descriptor_set = vk::WriteDescriptorSet::default()
        .dst_set(graphics_descriptor_set)
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
          descriptor_set: Some(unsafe { NonZeroHandle::new_unchecked(graphics_descriptor_set) }),
          is_generated: false,
          compute_descriptor_pool: Some(descriptor_pool),
          compute_descriptor_set_layout: Some(set_layout),
          compute_descriptor_set: Some(descriptor_set),
          compute_pipeline: Some(compute_pipeline),
          compute_pipeline_layout: Some(pipeline_layout),
          params_buffer: Some(params_buffer),
          params_alloc: Some(params_alloc),
        },
      );
    }

    let sun_resource = map.get_mut(&entity_id).unwrap();
    if sun_resource.compute_pipeline.is_none() {
      return Ok(());
    }

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();
    let cmd = data.command_buffer.get();

    unsafe {
      let alloc_info = res
        .allocator
        .allocator
        .get_allocation_info(sun_resource.params_alloc.as_ref().unwrap());
      let ptr = alloc_info.mapped_data as *mut f32;
      *ptr = timeline as f32 * 0.016;

      let _ = res.allocator.allocator.flush_allocation(
        sun_resource.params_alloc.as_ref().unwrap(),
        0,
        vk::WHOLE_SIZE as u64,
      );

      let bda_info =
        vk::BufferDeviceAddressInfo::default().buffer(sun_resource.params_buffer.unwrap());
      let buffer_address = self
        .device
        .buffer_device_address
        .get_buffer_device_address(&bda_info);

      let old_layout = if sun_resource.is_generated {
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
      } else {
        vk::ImageLayout::UNDEFINED
      };
      let barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(if sun_resource.is_generated {
          vk::PipelineStageFlags2::FRAGMENT_SHADER
        } else {
          vk::PipelineStageFlags2::NONE
        })
        .src_access_mask(if sun_resource.is_generated {
          vk::AccessFlags2::SHADER_READ
        } else {
          vk::AccessFlags2::NONE
        })
        .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .old_layout(old_layout)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(sun_resource.image.as_ref().unwrap().image.get())
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
        .device
        .synchronization2
        .cmd_pipeline_barrier2(cmd, &dep_info);

      self.device.cmd_bind_pipeline(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        sun_resource.compute_pipeline.unwrap().get(),
      );
      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        sun_resource.compute_pipeline_layout.unwrap(),
        0,
        &[sun_resource.compute_descriptor_set.unwrap()],
        &[],
      );
      let push_constants_bytes =
        core::slice::from_raw_parts(&buffer_address as *const _ as *const u8, 8);
      self.device.cmd_push_constants(
        cmd,
        sun_resource.compute_pipeline_layout.unwrap(),
        vk::ShaderStageFlags::COMPUTE,
        0,
        push_constants_bytes,
      );

      let group_count_x = (sun_resource.resolution.0 + 7) / 8;
      let group_count_y = (sun_resource.resolution.1 + 7) / 8;
      let group_count_z = (sun_resource.resolution.2 + 7) / 8;
      self
        .device
        .cmd_dispatch(cmd, group_count_x, group_count_y, group_count_z);

      let barrier2 = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
        .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(sun_resource.image.as_ref().unwrap().image.get())
        .subresource_range(
          vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1),
        );

      let dep_info2 =
        vk::DependencyInfo::default().image_memory_barriers(core::slice::from_ref(&barrier2));
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(cmd, &dep_info2);
    }

    sun_resource.is_generated = true;

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

    let map = res.sun_resources.read();
    let sun_opt = map.as_ref().and_then(|m| m.get(&entity_id));
    if sun_opt.is_none() {
      return Ok(());
    }
    let sun_resource = sun_opt.unwrap();

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

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();
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
        <aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 as aethervk_oshal_rlib::math::matrix::SquareMatrix>::inverse(model_matrix).unwrap();
      let mvp: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32 = view_proj * model_matrix;

      // Ensure camera position is in local space of the sun
      let local_camera_pos = model_inv.mul_vector(
        aethervk_oshal_rlib::math::vector::vec4::Vec4f32::from_components(
          camera_world_pos.x(),
          camera_world_pos.y(),
          camera_world_pos.z(),
          1.0,
        ),
      );
      let local_camera_pos_vec3 = aethervk_oshal_rlib::math::vector::vec3::Vec3f32::from_components(
        local_camera_pos.x() / local_camera_pos.w(),
        local_camera_pos.y() / local_camera_pos.w(),
        local_camera_pos.z() / local_camera_pos.w(),
      );

      let push_constants = crate::gpu::SunPushConstants {
        model_view_proj: mvp.into(),
        local_camera_pos: local_camera_pos_vec3.into(),
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
    entity_id: crate::scene::EntityId,
    component: &crate::scene::SkyComponent,
    view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
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

    if res.sky_image.is_none() {
      return Ok(());
    }

    let needs_desc = {
      let arch_ref: Option<&_>;
      #[cfg(debug_assertions)]
      {
        arch_ref = res.sky_render_archetype.as_ref().as_ref();
      }
      #[cfg(not(debug_assertions))]
      {
        arch_ref = res.sky_render_archetype.as_ref();
      }
      arch_ref.unwrap().descriptor_set.is_none()
    };

    if needs_desc {
      drop(res);
      let mut wres = self.res.write();

      let mut do_alloc = false;
      {
        let mut w_arch_ref = None;
        #[cfg(debug_assertions)]
        {
          w_arch_ref = wres.sky_render_archetype.as_mut().as_mut();
        }
        #[cfg(not(debug_assertions))]
        {
          w_arch_ref = wres.sky_render_archetype.as_mut();
        }
        if let Some(arch) = w_arch_ref {
          if arch.descriptor_set.is_none() {
            do_alloc = true;
          }
        }
      }

      if do_alloc {
        let layout = {
          let arch_ref: Option<&_>;
          #[cfg(debug_assertions)]
          {
            arch_ref = wres.sky_render_archetype.as_ref().as_ref();
          }
          #[cfg(not(debug_assertions))]
          {
            arch_ref = wres.sky_render_archetype.as_ref();
          }
          arch_ref.unwrap().descriptor_set_layout.get()
        };

        let new_set = wres
          .descriptor_pool
          .as_ref()
          .unwrap()
          .allocate(&self.device, layout, &wres.discard_pool, timeline, "Sky")?
          .get();

        let image_info = vk::DescriptorImageInfo::default()
          .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
          .image_view(wres.sky_image.as_ref().unwrap().image_view.get())
          .sampler(wres.linear_sampler.get());
        let write_descriptor_set = vk::WriteDescriptorSet::default()
          .dst_set(new_set)
          .dst_binding(0)
          .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
          .image_info(core::slice::from_ref(&image_info));
        unsafe {
          self
            .device
            .update_descriptor_sets(&[write_descriptor_set], &[])
        };

        let mut w_arch_ref = None;
        #[cfg(debug_assertions)]
        {
          w_arch_ref = wres.sky_render_archetype.as_mut().as_mut();
        }
        #[cfg(not(debug_assertions))]
        {
          w_arch_ref = wres.sky_render_archetype.as_mut();
        }
        w_arch_ref.unwrap().descriptor_set = Some(unsafe { NonZeroHandle::new_unchecked(new_set) });
      }

      drop(wres);
      res = self.res.read();
    }

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.sky_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.sky_render_archetype.as_ref();
    }
    let archetype = arch_ref.unwrap();

    let pipeline_key = archetype.pipeline_key.unwrap();
    let pipeline = res
      .pipeline_pool
      .read()
      .get_graphics_pipeline(pipeline_key)
      .unwrap();

    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());

      let descriptor_set = archetype.descriptor_set.unwrap().get();

      self.device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        layout,
        0,
        &[descriptor_set],
        &[],
      );

      let inv_view_proj_mat = view_proj.inverse().unwrap();
      let inv_view_proj_arr: [f32; 16] = inv_view_proj_mat.into();
      let push_constants_bytes =
        core::slice::from_raw_parts(&inv_view_proj_arr as *const _ as *const u8, 64);
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constants_bytes,
      );

      self.device.cmd_draw(cmd, 3, 1, 0, 0);
    }
    Ok(())
  }

  fn render_grid(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    entity_id: crate::scene::EntityId,
    component: &crate::scene::GridComponent,
    view_proj: aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32,
    camera_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    near_plane: f32,
    far_plane: f32,
  ) -> GpuResult<()> {
    let mut res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    if res.grid_render_archetype.is_none() {
      drop(res);
      let mut wres = self.res.write();
      if wres.grid_render_archetype.is_none() {
        let (vkey, fkey) = self.ensure_grid_shader_modules(&wres)?;
        let handle = wres.live_presentation_engines.read().keys().next().copied();
        if let Some(h) = handle {
          wres.create_grid_archetype(&self.device, vkey, fkey, self.depth_stencil_format, h)?;
        }
      }
      drop(wres);
      res = self.res.read();
    }

    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();

    let arch_ref: Option<&_>;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.grid_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.grid_render_archetype.as_ref();
    }
    let archetype = arch_ref.unwrap();

    let pipeline_key = archetype.pipeline_key.unwrap();
    let pipeline = res
      .pipeline_pool
      .read()
      .get_graphics_pipeline(pipeline_key)
      .unwrap();

    let cmd = data.command_buffer.get();
    let layout = archetype.pipeline_layout.get();

    #[repr(C)]
    struct GridPush {
      view_proj: [f32; 16],
      inv_view_proj: [f32; 16],
      camera_pos: [f32; 3],
      near_plane: f32,
      far_plane: f32,
      density: f32,
      _pad1: [f32; 2],
      grid_color: [f32; 3],
      _pad2: f32,
    }

    let inv_view_proj_mat = view_proj.inverse().unwrap();

    let push = GridPush {
      view_proj: view_proj.into(),
      inv_view_proj: inv_view_proj_mat.into(),
      camera_pos: camera_pos.into(),
      near_plane,
      far_plane,
      density: 0.01,
      _pad1: [0.0; 2],
      grid_color: [0.5, 0.5, 0.5],
      _pad2: 0.0,
    };

    unsafe {
      self
        .device
        .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
      let push_constants_bytes = core::slice::from_raw_parts(
        &push as *const _ as *const u8,
        core::mem::size_of::<GridPush>(),
      );
      self.device.cmd_push_constants(
        cmd,
        layout,
        vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
        0,
        push_constants_bytes,
      );

      self.device.cmd_draw(cmd, 4, 1, 0, 0);
    }
    Ok(())
  }

  fn render_minimap(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    player_pos: aethervk_oshal_rlib::math::vector::vec3::Vec3f32,
    max_distance: f32,
    planets: &[(aethervk_oshal_rlib::math::vector::vec3::Vec3f32, f32, [f32; 4])],
  ) -> GpuResult<()> {
    let mut res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    if res.minimap_render_archetype.is_none() {
      drop(res);
      let mut wres = self.res.write();
      if wres.minimap_render_archetype.is_none() {
        let (vkey, fkey) = ensure_minimap_shader_modules(&self.device, &wres)?;
        
        let mut arch = unsafe { resources::MinimapRenderResourceArchetype::new(&self.device)? };
        
        let shader_manager = wres.shader_manager.read();
        let vertex_shader = shader_manager.get(vkey).unwrap();
        let fragment_shader = shader_manager.get(fkey).unwrap();
        
        let handle = wres.live_presentation_engines.read().keys().next().copied();
        if let Some(h) = handle {
          let presentation_engine_lock = wres.live_presentation_engines.read();
          let pe = presentation_engine_lock.get(&h).unwrap().read();
          
          let pipeline_graphics_info = pipelines::GraphicsInfo::default()
            .with_vertex_in(pipelines::VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP).clone())
            .with_pre_rasterization(
              pipelines::PreRasterization::default()
                .with_vertex_module(vertex_shader.module.get())
                .clone()
            )
            .with_fragment_shader(
              pipelines::FragmentShader::default()
                .with_fragment_module(fragment_shader.module.get())
                .add_viewport(vk::Viewport {
                  width: pe.extent().0 as f32, height: -(pe.extent().1 as f32), x: 0.0, y: pe.extent().1 as f32, min_depth: 0.0, max_depth: 1.0
                })
                .add_scissors(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: pe.extent().0, height: pe.extent().1 } })
                .clone()
            )
            .with_fragment_out(
              pipelines::FragmentOut::default()
                .add_color_attachment_format(pe.format())
                .clone()
            )
            .with_pipeline_layout(arch.pipeline_layout.get())
            .with_pipeline_flags(pipelines::PipelineFlags::CULL_ALL | pipelines::PipelineFlags::NO_DEPTH_TEST | pipelines::PipelineFlags::NO_DEPTH_WRITE)
            .with_render_pass(
              wres.renderpasses.get_or_create_render_pass(
                renderpasses::RenderPassSpecification::single_pass(&pe, self.depth_stencil_format),
                0, &self.device, &wres.allocator.allocator, &wres.discard_pool, timeline
              )?.0.get()
            )
            .with_subpass(0)
            .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
            .clone();
            
          drop(pe);
          drop(presentation_engine_lock);
          drop(shader_manager);

          let pipeline_key = pipeline_graphics_info.pipeline_key();
          wres.pipeline_pool.write().get_or_create_graphics_pipeline(&self.device, &pipeline_graphics_info)?;
          arch.pipeline_key = Some(pipeline_key);
          
          #[cfg(not(debug_assertions))]
          {
            wres.minimap_render_archetype = Some(arch);
          }
          #[cfg(debug_assertions)]
          {
            wres.minimap_render_archetype = aethervk_oshal_rlib::os::debug::DropTracker::new(aethervk_oshal_rlib::os::debug::TrackedOption::some(arch));
          }
        }
      }
      drop(wres);
      res = self.res.read();
    }

    let arch_ref;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.minimap_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.minimap_render_archetype.as_ref();
    }

    if let Some(archetype) = arch_ref {
      let pipeline = res.pipeline_pool.read().get_graphics_pipeline(archetype.pipeline_key.unwrap()).unwrap();
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();
      let cmd = data.command_buffer.get();
      let layout = archetype.pipeline_layout.get();

      // Fetch aspect ratio from the active presentation engine (we assume there's at least one, or we default to 1.0)
      let live_engines_lock = res.live_presentation_engines.read();
      let aspect_ratio = if let Some(engine_state) = live_engines_lock.values().next() {
        let ext = engine_state.read().extent();
        if ext.1 > 0 {
          ext.0 as f32 / ext.1 as f32
        } else {
          1.0
        }
      } else {
        1.0
      };

      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
        
        let mut push_bytes = [0u8; 544];
        
        let offset = [0.7f32, 0.7f32];
        let size = [0.25f32, 0.25f32 * aspect_ratio];
        use aethervk_oshal_rlib::math::vector::Vector;
        let player_pos_arr = [player_pos.x(), player_pos.y()];
        let max_distance_f = max_distance;
        let num_planets = planets.len() as u32;
        
        core::ptr::copy_nonoverlapping(&offset as *const _ as *const u8, push_bytes.as_mut_ptr().add(0), 8);
        core::ptr::copy_nonoverlapping(&size as *const _ as *const u8, push_bytes.as_mut_ptr().add(8), 8);
        core::ptr::copy_nonoverlapping(&player_pos_arr as *const _ as *const u8, push_bytes.as_mut_ptr().add(16), 8);
        core::ptr::copy_nonoverlapping(&max_distance_f as *const _ as *const u8, push_bytes.as_mut_ptr().add(24), 4);
        core::ptr::copy_nonoverlapping(&num_planets as *const _ as *const u8, push_bytes.as_mut_ptr().add(28), 4);
        
        for (i, p) in planets.iter().enumerate().take(16) {
           let base = 32 + i * 32;
           let p_pos = [p.0.x(), p.0.y()];
           let p_size = p.1;
           let p_pad = 0.0f32;
           let p_color = p.2;
           
           core::ptr::copy_nonoverlapping(&p_pos as *const _ as *const u8, push_bytes.as_mut_ptr().add(base + 0), 8);
           core::ptr::copy_nonoverlapping(&p_size as *const _ as *const u8, push_bytes.as_mut_ptr().add(base + 8), 4);
           core::ptr::copy_nonoverlapping(&p_pad as *const _ as *const u8, push_bytes.as_mut_ptr().add(base + 12), 4);
           core::ptr::copy_nonoverlapping(&p_color as *const _ as *const u8, push_bytes.as_mut_ptr().add(base + 16), 16);
        }
        
        self.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &push_bytes);
        
        self.device.cmd_draw(cmd, 4, 1, 0, 0);
      }
    }
    Ok(())
  }

  fn render_bvh(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    nodes: &[(crate::math::collision::linear_bvh::LinearBound<f32, aethervk_oshal_rlib::math::vector::vec3::Vec3f32, aethervk_oshal_rlib::math::matrix::mat3::Mat3f32>, aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32)],
    view_proj: [f32; 16],
    presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    let mut res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    if res.bvh_render_archetype.is_none() {
      drop(res);
      let mut wres = self.res.write();
      if wres.bvh_render_archetype.is_none() {
        let (vkey, fkey) = ensure_bvh_shader_modules(&self.device, &wres)?;
        
        let mut arch = unsafe { resources::BvhRenderResourceArchetype::new(&self.device)? };
        
        let shader_manager = wres.shader_manager.read();
        let vertex_shader = shader_manager.get(vkey).unwrap();
        let fragment_shader = shader_manager.get(fkey).unwrap();
        
        let live_engines_lock = wres.live_presentation_engines.read();
        let pe = live_engines_lock.get(&presentation_engine).unwrap().read();
        
        let pipeline_graphics_info = pipelines::GraphicsInfo::default()
          .with_vertex_in(pipelines::VertexIn::default().with_topology(vk::PrimitiveTopology::LINE_LIST).clone())
          .with_pre_rasterization(
            pipelines::PreRasterization::default()
              .with_vertex_module(vertex_shader.module.get())
              .clone()
          )
          .with_fragment_shader(
            pipelines::FragmentShader::default()
              .with_fragment_module(fragment_shader.module.get())
              .add_viewport(vk::Viewport {
                width: pe.extent().0 as f32, height: -(pe.extent().1 as f32), x: 0.0, y: pe.extent().1 as f32, min_depth: 0.0, max_depth: 1.0
              })
              .add_scissors(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent: vk::Extent2D { width: pe.extent().0, height: pe.extent().1 } })
              .clone()
          )
          .with_fragment_out(
            pipelines::FragmentOut::default()
              .add_color_attachment_format(pe.format())
              .with_depth_attachment_format(self.depth_stencil_format)
              .clone()
          )
          .with_pipeline_layout(arch.pipeline_layout.get())
          .with_pipeline_flags(pipelines::PipelineFlags::NO_DEPTH_WRITE | pipelines::PipelineFlags::INVERT_FRONT_FACE)
          .with_render_pass(
            wres.renderpasses.get_or_create_render_pass(
              renderpasses::RenderPassSpecification::single_pass(&pe, self.depth_stencil_format),
              0, &self.device, &wres.allocator.allocator, &wres.discard_pool, timeline
            )?.0.get()
          )
          .with_subpass(0)
          .with_rasterization_polygon_mode(vk::PolygonMode::LINE)
          .clone();
          
        drop(pe);
        drop(live_engines_lock);
        drop(shader_manager);

        let pipeline_key = pipeline_graphics_info.pipeline_key();
        wres.pipeline_pool.write().get_or_create_graphics_pipeline(&self.device, &pipeline_graphics_info)?;
        arch.pipeline_key = Some(pipeline_key);
        
        #[cfg(not(debug_assertions))]
        {
          wres.bvh_render_archetype = Some(arch);
        }
        #[cfg(debug_assertions)]
        {
          wres.bvh_render_archetype = aethervk_oshal_rlib::os::debug::DropTracker::new(aethervk_oshal_rlib::os::debug::TrackedOption::some(arch));
        }
      }
      drop(wres);
      res = self.res.read();
    }

    let arch_ref;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.bvh_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.bvh_render_archetype.as_ref();
    }

    if let Some(archetype) = arch_ref {
      let pipeline = res.pipeline_pool.read().get_graphics_pipeline(archetype.pipeline_key.unwrap()).unwrap();
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();
      let cmd = data.command_buffer.get();
      let layout = archetype.pipeline_layout.get();

      // Fetch aspect ratio from the active presentation engine (we assume there's at least one, or we default to 1.0)
      let live_engines_lock = res.live_presentation_engines.read();
      let pe = live_engines_lock.get(&presentation_engine).unwrap().read();
      let extent = pe.extent();
      let aspect_ratio = if extent.1 > 0 { extent.0 as f32 / extent.1 as f32 } else { 1.0 };
      drop(pe);
      drop(live_engines_lock);

      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
        self.device.cmd_set_line_width(cmd, 1.0);
        
        for &(ref bound, ref model_matrix) in nodes {
          use crate::math::collision::linear_bvh::LinearBound;
          use aethervk_oshal_rlib::math::matrix::Matrix;
          
          let (center, type_val, extents, ax, ay, az) = match bound {
            LinearBound::AABB(aabb) => {
              let center = aabb.center();
              let he = aabb.half_extents();
              (center, 1.0f32, he, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
            }
            LinearBound::OBB(obb) => {
              let center = obb.center();
              let he = obb.half_extents();
              let axes = obb.axes();
              (center, 1.0f32, he, [axes[0].x(), axes[0].y(), axes[0].z()], 
               [axes[1].x(), axes[1].y(), axes[1].z()], 
               [axes[2].x(), axes[2].y(), axes[2].z()])
            }
          };

          let mut push_bytes = [0u8; 144];
          
          use aethervk_oshal_rlib::math::matrix::Matrix4;
          let view_proj_mat = aethervk_oshal_rlib::math::matrix::mat4::Mat4x4f32::from_array(&view_proj);
          let mvp = view_proj_mat * *model_matrix;
          let mvp_arr: [f32; 16] = mvp.into();

          core::ptr::copy_nonoverlapping(&mvp_arr as *const _ as *const u8, push_bytes.as_mut_ptr(), 64);
          
          use aethervk_oshal_rlib::math::vector::Vector;
          let center_type = [center.x(), center.y(), center.z(), type_val];
          let extents_arr = [extents.x(), extents.y(), extents.z(), 0.0];
          let axes_x = [ax[0], ax[1], ax[2], 0.0];
          let axes_y = [ay[0], ay[1], ay[2], 0.0];
          let axes_z = [az[0], az[1], az[2], 0.0];
          
          core::ptr::copy_nonoverlapping(&center_type as *const _ as *const u8, push_bytes.as_mut_ptr().add(64), 16);
          core::ptr::copy_nonoverlapping(&extents_arr as *const _ as *const u8, push_bytes.as_mut_ptr().add(80), 16);
          core::ptr::copy_nonoverlapping(&axes_x as *const _ as *const u8, push_bytes.as_mut_ptr().add(96), 16);
          core::ptr::copy_nonoverlapping(&axes_y as *const _ as *const u8, push_bytes.as_mut_ptr().add(112), 16);
          core::ptr::copy_nonoverlapping(&axes_z as *const _ as *const u8, push_bytes.as_mut_ptr().add(128), 16);
          
          self.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &push_bytes);
          
          let vert_count = if type_val == 1.0 { 24 } else { 216 };
          self.device.cmd_draw(cmd, vert_count, 1, 0, 0);
        }
      }
    }
    Ok(())
  }

  fn render_ui_rect(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    color: [f32; 4],
    position: [f32; 2],
    size: [f32; 2],
    presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    let arch_ref;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.text_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.text_render_archetype.as_ref();
    }

    if let Some(archetype) = arch_ref {
      let pipeline = res.pipeline_pool.read().get_graphics_pipeline(archetype.pipeline_key.unwrap()).unwrap();
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();
      let cmd = data.command_buffer.get();
      let layout = archetype.pipeline_layout.get();

      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());
        
        if let Some(set) = archetype.descriptor_set {
           self.device.cmd_bind_descriptor_sets(
             cmd,
             vk::PipelineBindPoint::GRAPHICS,
             layout,
             0,
             &[set],
             &[]
           );
        } else {
           return Ok(());
        }
        
        let mut push_bytes = [0u8; 48];
        let pos_arr = [position[0], position[1]];
        let scale_arr = [size[0], size[1]];
        let uv_bounds = [0.0f32, 0.0f32, -1.0f32, -1.0f32];
        
        core::ptr::copy_nonoverlapping(&pos_arr as *const _ as *const u8, push_bytes.as_mut_ptr(), 8);
        core::ptr::copy_nonoverlapping(&scale_arr as *const _ as *const u8, push_bytes.as_mut_ptr().add(8), 8);
        core::ptr::copy_nonoverlapping(&color as *const _ as *const u8, push_bytes.as_mut_ptr().add(16), 16);
        core::ptr::copy_nonoverlapping(&uv_bounds as *const _ as *const u8, push_bytes.as_mut_ptr().add(32), 16);
        
        self.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &push_bytes);
        self.device.cmd_draw(cmd, 4, 1, 0, 0);
      }
    }
    
    Ok(())
  }

  fn render_text(
    &self,
    cmd_buffer: crate::gpu::CommandBufferHandle,
    text: &str,
    font_path: &str,
    points: f32,
    color: [f32; 4],
    position: [f32; 2],
    presentation_engine: PresentationEngineHandle,
  ) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();

    let arch_ref;
    #[cfg(debug_assertions)]
    {
      arch_ref = res.text_render_archetype.as_ref().as_ref();
    }
    #[cfg(not(debug_assertions))]
    {
      arch_ref = res.text_render_archetype.as_ref();
    }

    if let Some(archetype) = arch_ref {
      let pipeline = res.pipeline_pool.read().get_graphics_pipeline(archetype.pipeline_key.unwrap()).unwrap();
      let cmd_buffers = self.recording_command_buffers.read();
      let data = cmd_buffers.get(&(timeline, cmd_buffer)).unwrap();
      let cmd = data.command_buffer.get();
      let layout = archetype.pipeline_layout.get();

      // Fetch screen size
      let live_engines_lock = res.live_presentation_engines.read();
      let (screen_width, screen_height) = if let Some(engine_state) = live_engines_lock.values().next() {
        let ext = engine_state.read().extent();
        if ext.0 > 0 && ext.1 > 0 { (ext.0 as f32, ext.1 as f32) } else { (800.0, 600.0) }
      } else { (800.0, 600.0) };

      unsafe {
        self.device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline.get());

        if let Some(set) = archetype.descriptor_set {
           self.device.cmd_bind_descriptor_sets(
             cmd,
             vk::PipelineBindPoint::GRAPHICS,
             layout,
             0,
             &[set],
             &[]
           );
        } else {
           return Ok(());
        }

        let mut cursor_x = position[0];
        let mut cursor_y = position[1];

        let scale_factor = points / 64.0; // Assuming atlas generated at 64pt
        let scale_x = scale_factor * 2.0 / screen_width;
        let scale_y = scale_factor * 2.0 / screen_height;
        for c in text.chars() {
          if c == '\n' {
             cursor_x = position[0];
             // Simple line height advance
             cursor_y += 64.0 * scale_y * 1.5;
             continue;
          }          
          let mut uv_bounds = [0.0f32, 0.0f32, 1.0f32, 1.0f32];
          let mut char_size = [1.0f32, 1.0f32];
          let mut char_offset = [0.0f32, 0.0f32];
          let mut advance = 0.5f32;
          
          if let Some(atlas) = &archetype.font_atlas {
            if let Some(glyph) = atlas.glyphs.get(&c) {
               uv_bounds = [glyph.uv_min[0], glyph.uv_min[1], glyph.uv_max[0], glyph.uv_max[1]];
               char_size = glyph.size;
               char_offset = glyph.offset;
               advance = glyph.advance;
            } else if let Some(glyph) = atlas.glyphs.get(&'█') {
               uv_bounds = [glyph.uv_min[0], glyph.uv_min[1], glyph.uv_max[0], glyph.uv_max[1]];
               char_size = glyph.size;
               char_offset = glyph.offset;
               advance = glyph.advance;
            }
          }
          
          let mut push_bytes = [0u8; 48];
          let cx = cursor_x + char_offset[0] * scale_x;
          let cy = cursor_y + char_offset[1] * scale_y;
          let pos_arr = [cx, cy];
          let scale_arr = [char_size[0] * scale_x, char_size[1] * scale_y];
          
          core::ptr::copy_nonoverlapping(&pos_arr as *const _ as *const u8, push_bytes.as_mut_ptr(), 8);
          core::ptr::copy_nonoverlapping(&scale_arr as *const _ as *const u8, push_bytes.as_mut_ptr().add(8), 8);
          core::ptr::copy_nonoverlapping(&color as *const _ as *const u8, push_bytes.as_mut_ptr().add(16), 16);
          core::ptr::copy_nonoverlapping(&uv_bounds as *const _ as *const u8, push_bytes.as_mut_ptr().add(32), 16);
          
          self.device.cmd_push_constants(cmd, layout, vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT, 0, &push_bytes);
          self.device.cmd_draw(cmd, 4, 1, 0, 0);
          
          cursor_x += advance * scale_x;
        }
      }
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
        .device
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

    let mut signal_semaphores = alloc::vec::Vec::new();
    let mut timeline_values = alloc::vec::Vec::new();

    if let Some(sem) = signal_semaphore {
      signal_semaphores.push(sem.get());
      timeline_values.push(0);
    }

    signal_semaphores.push(res.timeline_semaphore.get());
    timeline_values.push(next_timeline_value);

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

impl<'a> crate::gpu::Kernels for Device<'a> {
  fn dispatch_physics_step(
    &self,
    _cmd_buffer: crate::gpu::CommandBufferHandle,
    _physical_scene: &crate::gpu::PhysicalScene,
    _dt: f32,
  ) -> GpuResult<()> {
    // TODO: Bind compute pipeline for IMR interval arithmetic integration
    Ok(())
  }

  fn dispatch_particles(&self, _cmd_buffer: crate::gpu::CommandBufferHandle, _dt: f32) -> GpuResult<()> {
    // TODO: Bind particle compute pipeline
    Ok(())
  }
}

impl<'a> crate::gpu::KernelRenderBridge for Device<'a> {
  fn sync_compute_to_graphics(&self, cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    let res = self.res.read();
    let timeline = res.get_timeline_semaphore_cached_value();
    let cmd_buffers = self.recording_command_buffers.read();
    let data = cmd_buffers
      .get(&(timeline, cmd_buffer))
      .ok_or(GpuError::InvalidArgument)?;

    // Memory barrier to sync COMPUTE_SHADER writing to VERTEX_SHADER / FRAGMENT_SHADER reading
    let mem_barrier = vk::MemoryBarrier2::default()
      .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
      .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
      .dst_stage_mask(
        vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER,
      )
      .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::UNIFORM_READ);

    let dep_info =
      vk::DependencyInfo::default().memory_barriers(core::slice::from_ref(&mem_barrier));

    unsafe {
      self
        .device
        .synchronization2
        .cmd_pipeline_barrier2(data.command_buffer.get(), &dep_info);
    }
    Ok(())
  }

  fn sync_graphics_to_compute(&self, _cmd_buffer: crate::gpu::CommandBufferHandle) -> GpuResult<()> {
    // Counterpart memory barrier from Graphics output back to Compute reads
    Ok(())
  }
}
