use ash::vk;
use crate::gpu;
use crate::gpu::{vulkan, PipelineKeyable, PresentationEngineHandle};
use crate::gpu::vulkan::device::{renderpasses, resources, shader_manager, LogicalDevice, Queue};
use crate::gpu::vulkan::device::pipelines::{
  FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization, StencilCompareOp,
  StencilLogicOp, VertexIn,
};
use crate::gpu::vulkan::device::renderpasses::RenderPassSpecification;
use crate::gpu::vulkan::device::resources::{
  DiscardableResource, ForwardMeshRenderResourceArchetype, Image,
};
use crate::gpu::vulkan::device::shader_manager::ShaderKey;
use crate::gpu::vulkan::utils::NonZeroHandle;
use crate::gpu_backends::vulkan::device::{pipelines, swapchain};
use crate::simulation::comet::{NORMAL_COMPONENTS, POSITION_COMPONENTS, UV_COMPONENTS};
use crate::types::{GpuError, GpuResult};
use alloc::vec::Vec;

// TODO rewrite error messages

#[derive(Default)]
pub(super) struct Archetypes {
  pub sun_render_archetype: spin::RwLock<Option<resources::SunRenderResourceArchetype>>,
  pub physical_mesh_render_archetype: spin::RwLock<Option<ForwardMeshRenderResourceArchetype>>,
  pub billboard_render_archetype: spin::RwLock<Option<resources::BillboardRenderResourceArchetype>>,
  pub cursor_render_archetype: spin::RwLock<Option<resources::CursorRenderResourceArchetype>>,
  pub marker_render_archetype: spin::RwLock<Option<resources::MarkerRenderResourceArchetype>>,
  pub measurement_render_archetype:
    spin::RwLock<Option<resources::MeasurementRenderResourceArchetype>>,
  pub sky_render_archetype: spin::RwLock<Option<resources::SkyRenderResourceArchetype>>,
  pub grid_render_archetype: spin::RwLock<Option<resources::GridRenderResourceArchetype>>,
  pub minimap_render_archetype: spin::RwLock<Option<resources::MinimapRenderResourceArchetype>>,
  pub text_render_archetype: spin::RwLock<Option<resources::TextRenderResourceArchetype>>,
  pub bvh_render_archetype: spin::RwLock<Option<resources::BvhRenderResourceArchetype>>,
}

impl Archetypes {
  pub fn has_discardables(&self) -> bool {
    self.sun_render_archetype.read().is_some()
      || self.physical_mesh_render_archetype.read().is_some()
      || self.billboard_render_archetype.read().is_some()
      || self.cursor_render_archetype.read().is_some()
      || self.marker_render_archetype.read().is_some()
      || self.measurement_render_archetype.read().is_some()
      || self.sky_render_archetype.read().is_some()
      || self.grid_render_archetype.read().is_some()
      || self.minimap_render_archetype.read().is_some()
      || self.text_render_archetype.read().is_some()
      || self.bvh_render_archetype.read().is_some()
  }

  pub fn discard(&self, device: &ash::Device, discard_pool: &resources::DiscardPool) {
    if let Some(mut archetype) = self.sun_render_archetype.write().take() {
      archetype.discard(device, &discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.physical_mesh_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.billboard_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.cursor_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.marker_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.sky_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.grid_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.minimap_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.text_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.bvh_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.measurement_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
  }

  /// update [`pipelines::FragmentOut`] and [`vk::RenderPass`] inside [`pipelines::GraphicsInfo`]
  /// disard old and create updated graphics [`vk::Pipeline`]
  /// Note: Update is performed only if archetype initialized once
  pub fn update_physical_mesh_archetype_for_presentation_engine(
    &self,
    device: &LogicalDevice,
    presentation_engine: &swapchain::PresentationState,
    write_pipeline: &mut pipelines::PipelinePool,
    renderpasses: &renderpasses::RenderPasses,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut archetype_lock = self.physical_mesh_render_archetype.write();
    let archetype = unsafe {
      let mut_arch: Option<&mut _>;
      mut_arch = archetype_lock.as_mut();

      mut_arch.ok_or(GpuError::InvalidState("device.rs"))?
    };

    if archetype.graphics_info.is_none() || archetype.pipeline_key.is_none() {
      return Err(GpuError::InvalidState("device.rs"));
    }
    let pipeline_key = *unsafe { archetype.pipeline_key.as_ref().unwrap_unchecked() };
    let graphics_info = unsafe { archetype.graphics_info.as_mut().unwrap_unchecked() };
    let depth_stencil_format = graphics_info
      .fragment_out
      .depth_attachment_format
      .unwrap_or(vk::Format::UNDEFINED);

    graphics_info.fragment_out.color_attachment_formats.clear();
    graphics_info
      .fragment_out
      .color_attachment_formats
      .push(presentation_engine.format());
    graphics_info.render_pass = renderpasses
      .get_or_create_render_pass(
        renderpasses::RenderPassSpecification::single_pass(
          &presentation_engine,
          depth_stencil_format,
        ),
        0,
        device,
        allocator,
        discard_pool,
        timeline,
      )?
      .0
      .get();
    // Note: don't care about viewport and scissor cause they are dynamic state
    write_pipeline.get_or_create_graphics_pipeline(device, graphics_info)?;
    write_pipeline.discard_graphics_pipeline_if_present(pipeline_key, discard_pool, timeline);

    let pipeline_key = graphics_info.pipeline_key();
    archetype.pipeline_key = Some(pipeline_key);

    let outline_graphics_info = graphics_info
      .clone()
      .with_pipeline_flags(
        PipelineFlags::CULL_BACK | PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::STENCIL_ENABLE,
      )
      .with_stencil_compare_op(StencilCompareOp::NotEqual)
      .with_stencil_logic_op(StencilLogicOp::None)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(255)
      .with_stencil_write_mask(0)
      .clone();

    if let Some(outline_key) = archetype.outline_pipeline_key {
      write_pipeline.discard_graphics_pipeline_if_present(outline_key, discard_pool, timeline);
    }
    let outline_pipeline_key = outline_graphics_info.pipeline_key();
    write_pipeline.get_or_create_graphics_pipeline(device, &outline_graphics_info)?;
    archetype.outline_pipeline_key = Some(outline_pipeline_key);

    Ok(())
  }

  pub fn update_cursor_archetype_for_presentation_engine(
    &self,
    device: &LogicalDevice,
    presentation_engine: &swapchain::PresentationState,
    write_pipeline: &mut pipelines::PipelinePool,
    renderpasses: &renderpasses::RenderPasses,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut archetype_lock = self.cursor_render_archetype.write();
    let archetype = {
      let mut_arch: Option<&mut _> = archetype_lock.as_mut();

      mut_arch.ok_or(GpuError::InvalidState("device.rs"))?
    };

    if archetype.graphics_info.is_none() || archetype.pipeline_key.is_none() {
      return Err(GpuError::InvalidState("device.rs"));
    }
    let pipeline_key = *unsafe { archetype.pipeline_key.as_ref().unwrap_unchecked() };

    let graphics_info = unsafe { archetype.graphics_info.as_mut().unwrap_unchecked() };
    let depth_stencil_format = graphics_info
      .fragment_out
      .depth_attachment_format
      .unwrap_or(vk::Format::UNDEFINED);

    graphics_info.fragment_out.color_attachment_formats.clear();
    graphics_info
      .fragment_out
      .color_attachment_formats
      .push(presentation_engine.format());
    graphics_info.render_pass = renderpasses
      .get_or_create_render_pass(
        RenderPassSpecification::single_pass(&presentation_engine, depth_stencil_format),
        0,
        device,
        allocator,
        discard_pool,
        timeline,
      )?
      .0
      .get();
    // Note: don't care about viewport and scissor cause they are dynamic state
    write_pipeline.get_or_create_graphics_pipeline(device, graphics_info)?;
    write_pipeline.discard_graphics_pipeline_if_present(pipeline_key, discard_pool, timeline);

    let pipeline_key = graphics_info.pipeline_key();
    archetype.pipeline_key = Some(pipeline_key);

    Ok(())
  }

  pub fn update_sun_archetype_for_presentation_engine(
    &self,
    device: &LogicalDevice,
    presentation_engine: &swapchain::PresentationState,
    write_pipeline: &mut pipelines::PipelinePool,
    renderpasses: &renderpasses::RenderPasses,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut archetype_lock = self.sun_render_archetype.write();
    let archetype = unsafe {
      let mut_arch: Option<&mut _> = archetype_lock.as_mut();

      mut_arch.ok_or(GpuError::InvalidState("device.rs"))?
    };

    if archetype.graphics_info.is_none() || archetype.pipeline_key.is_none() {
      return Err(GpuError::InvalidState("device.rs"));
    }
    let pipeline_key = *unsafe { archetype.pipeline_key.as_ref().unwrap_unchecked() };

    let graphics_info = unsafe { archetype.graphics_info.as_mut().unwrap_unchecked() };
    let depth_stencil_format = graphics_info
      .fragment_out
      .depth_attachment_format
      .unwrap_or(vk::Format::UNDEFINED);

    graphics_info.fragment_out.color_attachment_formats.clear();
    graphics_info
      .fragment_out
      .color_attachment_formats
      .push(presentation_engine.format());
    graphics_info.render_pass = renderpasses
      .get_or_create_render_pass(
        RenderPassSpecification::single_pass(&presentation_engine, depth_stencil_format),
        0,
        device,
        allocator,
        discard_pool,
        timeline,
      )?
      .0
      .get();
    // Note: don't care about viewport and scissor cause they are dynamic state
    write_pipeline.get_or_create_graphics_pipeline(device, graphics_info)?;
    write_pipeline.discard_graphics_pipeline_if_present(pipeline_key, discard_pool, timeline);

    let pipeline_key = graphics_info.pipeline_key();
    archetype.pipeline_key = Some(pipeline_key);

    Ok(())
  }

  // ------------------------------------ Creation ------------------------------------

  pub fn create_physical_mesh_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    outline_vertex_shader_key: ShaderKey,
    outline_fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    if self.physical_mesh_render_archetype.read().is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }

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

    let outline_vertex_shader = shader_manager
      .get(outline_vertex_shader_key)
      .ok_or(GpuError::InvalidShader)?;
    if outline_vertex_shader.shader_stage != vk::ShaderStageFlags::VERTEX {
      return Err(GpuError::InvalidShader);
    }
    let outline_fragment_shader = shader_manager
      .get(outline_fragment_shader_key)
      .ok_or(GpuError::InvalidShader)?;
    if outline_fragment_shader.shader_stage != vk::ShaderStageFlags::FRAGMENT {
      return Err(GpuError::InvalidShader);
    }

    // Create initial struct
    let res = unsafe {
      ForwardMeshRenderResourceArchetype::new(
        device,
        &vertex_shader,
        &fragment_shader,
        allocator,
        discard_pool,
        &queue,
      )
    }?;

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
      .with_pipeline_layout(res.pipeline_layout.get())
      .with_pipeline_flags(PipelineFlags::CULL_BACK | PipelineFlags::STENCIL_ENABLE)
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
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

    pipeline_pool.get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    let outline_graphics_info = pipeline_graphics_info
      .clone()
      .with_pre_rasterization(
        PreRasterization::default()
          .with_vertex_module(outline_vertex_shader.module.get())
          .clone(),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(outline_fragment_shader.module.get())
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
      .with_pipeline_flags(
        PipelineFlags::CULL_BACK
          | PipelineFlags::INVERT_FRONT_FACE
          | PipelineFlags::STENCIL_ENABLE
          | PipelineFlags::NO_DEPTH_TEST,
      )
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .with_stencil_compare_op(StencilCompareOp::NotEqual)
      .with_stencil_logic_op(StencilLogicOp::None)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(255)
      .with_stencil_write_mask(0)
      .clone();

    let outline_pipeline_key = outline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(&device, &outline_graphics_info)?;

    let final_res = res
      .with_graphics_info(pipeline_graphics_info)
      .with_outline_pipeline_key(outline_pipeline_key);

    *self.physical_mesh_render_archetype.write() = Some(final_res);

    Ok(())
  }

  pub fn create_sun_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
  ) -> GpuResult<()> {
    let mut sun_render_archetype = self.sun_render_archetype.write();
    if sun_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }

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
    *sun_render_archetype = Some(res);

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
          let ref_arch: Option<&_> = sun_render_archetype.as_ref();
          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(
        PipelineFlags::CULL_ALL | PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::NO_DEPTH_WRITE,
      ) // No culling so we see it from inside and outside (yes, cull all means no culling) + No depth write for volume
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

 pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    let sun_render_archetype_mut = sun_render_archetype.as_mut().unwrap();
    sun_render_archetype_mut.graphics_info = Some(pipeline_graphics_info);
    sun_render_archetype_mut.pipeline_key = Some(pipeline_key);

    debug_assert!(sun_render_archetype.is_some());

    Ok(())
  }

  pub fn create_sky_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
  ) -> GpuResult<()> {
    let mut sky_render_archetype = self.sky_render_archetype.write();
    if sky_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }
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
        PipelineFlags::CULL_ALL
          | PipelineFlags::NO_DEPTH_WRITE
          | PipelineFlags::NO_DEPTH_TEST
          | PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;
    arch.pipeline_key = Some(pipeline_key);

    *sky_render_archetype = Some(arch);

    Ok(())
  }

  pub fn create_grid_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
  ) -> GpuResult<()> {
    let mut grid_render_archetype = self.grid_render_archetype.write();
    if grid_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }
    let vertex_shader = shader_manager.get(vertex_shader_key).unwrap();
    let fragment_shader = shader_manager.get(fragment_shader_key).unwrap();

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(128)];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    *grid_render_archetype = Some(resources::GridRenderResourceArchetype {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      pipeline_key: None,
    });

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
        PipelineFlags::CULL_ALL
          | PipelineFlags::NO_DEPTH_WRITE
          | PipelineFlags::NO_DEPTH_TEST
          | PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

    let arch = grid_render_archetype.as_mut().unwrap();
    arch.pipeline_key = Some(pipeline_key);

    Ok(())
  }

  pub fn create_minimap_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    pe: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut minimap_render_archetype = self.minimap_render_archetype.write();
    if minimap_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }
    *minimap_render_archetype =
      Some(unsafe { resources::MinimapRenderResourceArchetype::new(device)? });
    let arch_mut = minimap_render_archetype.as_mut().unwrap();

    let vertex_shader = shader_manager.get(vkey).unwrap();
    let fragment_shader = shader_manager.get(fkey).unwrap();

    let pipeline_graphics_info = pipelines::GraphicsInfo::default()
      .with_vertex_in(
        pipelines::VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP)
          .clone(),
      )
      .with_pre_rasterization(
        pipelines::PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
      )
      .with_fragment_shader(
        pipelines::FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(vk::Viewport {
            width: pe.extent().0 as f32,
            height: -(pe.extent().1 as f32),
            x: 0.0,
            y: pe.extent().1 as f32,
            min_depth: 0.0,
            max_depth: 1.0,
          })
          .add_scissors(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
              width: pe.extent().0,
              height: pe.extent().1,
            },
          })
          .clone(),
      )
      .with_fragment_out(
        pipelines::FragmentOut::default()
          .add_color_attachment_format(pe.format())
          .clone(),
      )
      .with_pipeline_layout(arch_mut.pipeline_layout.get())
      .with_pipeline_flags(
        pipelines::PipelineFlags::CULL_ALL
          | pipelines::PipelineFlags::NO_DEPTH_TEST
          | pipelines::PipelineFlags::NO_DEPTH_WRITE,
      )
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            renderpasses::RenderPassSpecification::single_pass(&pe, depth_stencil_format),
            0,
            &device,
            allocator,
            discard_pool,
            timeline,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;
    arch_mut.pipeline_key = Some(pipeline_key);

    Ok(())
  }

  pub fn create_measurement_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
  ) -> GpuResult<()> {
    let mut measurement_render_archetype = self.measurement_render_archetype.write();
    if measurement_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }

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

    let res = unsafe { resources::MeasurementRenderResourceArchetype::new(device) }?;
    *measurement_render_archetype = Some(res);

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::LINE_LIST)
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
          .with_stencil_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(
        unsafe {
          let ref_arch: Option<&_>;
          ref_arch = measurement_render_archetype.as_ref();
          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(
        PipelineFlags::CULL_ALL
          | PipelineFlags::INVERT_FRONT_FACE
          | PipelineFlags::NO_DEPTH_TEST
          | PipelineFlags::NO_DEPTH_WRITE,
      )
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
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

    pipeline_pool.get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    let pkey = pipeline_graphics_info.pipeline_key();
    measurement_render_archetype.as_mut().unwrap().pipeline_key = Some(pkey);
    measurement_render_archetype.as_mut().unwrap().graphics_info = Some(pipeline_graphics_info);

    Ok(())
  }

  pub fn create_marker_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
  ) -> GpuResult<()> {
    let mut marker_render_archetype = self.marker_render_archetype.write();
    if marker_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }

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

    let res = unsafe { resources::MarkerRenderResourceArchetype::new(device) }?;
    *marker_render_archetype = Some(res);

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
          .with_stencil_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(
        unsafe {
          let ref_arch: Option<&_>;
          ref_arch = marker_render_archetype.as_ref();
          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(PipelineFlags::CULL_ALL | PipelineFlags::INVERT_FRONT_FACE)
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
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

    pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    let pkey = pipeline_graphics_info.pipeline_key();
    marker_render_archetype.as_mut().unwrap().pipeline_key = Some(pkey);
    marker_render_archetype.as_mut().unwrap().graphics_info = Some(pipeline_graphics_info);

    Ok(())
  }

  pub fn create_billboard_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
  ) -> GpuResult<()> {
    let mut billboard_render_archetype = self.billboard_render_archetype.write();
    if billboard_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }

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
    let res = unsafe { resources::BillboardRenderResourceArchetype::new(device) }?;
    *billboard_render_archetype = Some(res);

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
          ref_arch = billboard_render_archetype.as_ref();
          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(
        PipelineFlags::NO_DEPTH_TEST | PipelineFlags::CULL_ALL | PipelineFlags::INVERT_FRONT_FACE,
      ) // NO Culling, NO Depth Test (Yes, cull all means no culling)
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
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
    pipeline_pool.get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    billboard_render_archetype.as_mut().unwrap().graphics_info = Some(pipeline_graphics_info);

    debug_assert!(billboard_render_archetype.is_some());

    Ok(())
  }

  pub fn create_cursor_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
  ) -> GpuResult<()> {
    let mut cursor_render_archetype = self.cursor_render_archetype.write();
    if cursor_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }

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
    *cursor_render_archetype = Some(res);

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
          ref_arch = cursor_render_archetype.as_ref();
          ref_arch.unwrap_unchecked()
        }
        .pipeline_layout
        .get(),
      )
      .with_pipeline_flags(
        PipelineFlags::NO_DEPTH_TEST | PipelineFlags::CULL_ALL | PipelineFlags::INVERT_FRONT_FACE,
      ) // NO Culling, NO Depth Test (Yes, cull all means no culling)
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
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
    pipeline_pool.get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    let cursor_render_archetype_mut = cursor_render_archetype.as_mut().unwrap();
    cursor_render_archetype_mut.graphics_info = Some(pipeline_graphics_info);
    cursor_render_archetype_mut.pipeline_key = Some(pipeline_key);

    debug_assert!(cursor_render_archetype.is_some());

    Ok(())
  }

  pub fn create_text_archetype(
    &self,
    device: &vulkan::device::LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    presentation_engine_state: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut text_render_archetype = self.text_render_archetype.write();
    if text_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }

    let vertex_shader = shader_manager.get(vertex_shader_key).unwrap();
    let fragment_shader = shader_manager.get(fragment_shader_key).unwrap();

    let max_fonts = 256; // Array limit

    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_fonts)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    // flags from descriptor_indexing
    let binding_flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
    let mut binding_flags_info = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    // inject flags allowing arrays with partial holes / after bind updates/writes
    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(&bindings)
        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
        .push_next(&mut binding_flags_info);
    let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
    let set_layouts = [set_layout];

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(core::mem::size_of::<gpu::TextPushConstants>() as _)];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
      .set_layouts(&set_layouts)
      .push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    let sampler_info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let font_sampler = unsafe { device.create_sampler(&sampler_info, None) }?;

    let pool_sizes = [vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(max_fonts)];
    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(&pool_sizes)
        .max_sets(1)
        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND);
    let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&set_layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?[0];

    let mut arch = resources::TextRenderResourceArchetype {
      pipeline_layout: unsafe { NonZeroHandle::new_unchecked(pipeline_layout) },
      pipeline_key: None,
      descriptor_set_layout: unsafe { NonZeroHandle::new_unchecked(set_layout) },
      descriptor_pool: Some(unsafe { NonZeroHandle::new_unchecked(pool) }),
      descriptor_set: Some(descriptor_set),
      font_sampler: Some(font_sampler),
      uploaded_fonts: hashbrown::HashMap::new(),
      free_descriptor_indices: Vec::new(),
      next_descriptor_index: 0,
      max_fonts,
      allocator_raw: Some(allocator.get_raw()),
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
        PipelineFlags::CULL_ALL
          | PipelineFlags::NO_DEPTH_WRITE
          | PipelineFlags::NO_DEPTH_TEST
          | PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            RenderPassSpecification::single_pass(&presentation_engine_state, depth_stencil_format),
            0,
            device,
            allocator,
            discard_pool,
            0,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

    arch.pipeline_key = Some(pipeline_key);

    *text_render_archetype = Some(arch);

    Ok(())
  }

  pub fn create_bvh_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    pe: &swapchain::PresentationState,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut bvh_render_archetype = self.bvh_render_archetype.write();
    if bvh_render_archetype.is_some() {
      return Err(GpuError::InvalidState("device.rs"));
    }
    *bvh_render_archetype = Some(unsafe { resources::BvhRenderResourceArchetype::new(device) }?);
    let archetype = bvh_render_archetype.as_mut().unwrap();

    let vertex_shader = shader_manager.get(vkey).unwrap();
    let fragment_shader = shader_manager.get(fkey).unwrap();

    let pipeline_graphics_info = pipelines::GraphicsInfo::default()
      .with_vertex_in(
        pipelines::VertexIn::default()
          .with_topology(vk::PrimitiveTopology::LINE_LIST)
          .clone(),
      )
      .with_pre_rasterization(
        pipelines::PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
      )
      .with_fragment_shader(
        pipelines::FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(vk::Viewport {
            width: pe.extent().0 as f32,
            height: -(pe.extent().1 as f32),
            x: 0.0,
            y: pe.extent().1 as f32,
            min_depth: 0.0,
            max_depth: 1.0,
          })
          .add_scissors(vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
              width: pe.extent().0,
              height: pe.extent().1,
            },
          })
          .clone(),
      )
      .with_fragment_out(
        pipelines::FragmentOut::default()
          .add_color_attachment_format(pe.format())
          .with_depth_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(archetype.pipeline_layout.get())
      .with_pipeline_flags(
        pipelines::PipelineFlags::NO_DEPTH_WRITE | pipelines::PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_render_pass(
        renderpasses
          .get_or_create_render_pass(
            renderpasses::RenderPassSpecification::single_pass(&pe, depth_stencil_format),
            0,
            &device,
            allocator,
            discard_pool,
            timeline,
          )?
          .0
          .get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::LINE)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)?;
    archetype.pipeline_key = Some(pipeline_key);

    Ok(())
  }
}
