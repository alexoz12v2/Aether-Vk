//! archetypes_struct module.

use crate::gpu;
use crate::gpu::vulkan::device::pipelines::{
  FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization, StencilCompareOp,
  StencilLogicOp, VertexIn,
};
use crate::gpu::vulkan::device::renderpasses::RenderPassSpecification;
use crate::gpu::vulkan::device::resources::{
  DiscardableResource, ForwardMeshRenderResourceArchetype, Image,
};
use crate::gpu::vulkan::device::shader_manager::ShaderKey;
use crate::gpu::vulkan::device::{LogicalDevice, Queue, renderpasses, resources, shader_manager};
use crate::gpu::vulkan::utils::NonZeroHandle;
use crate::gpu::{PipelineKeyable, PresentationEngineHandle, vulkan};
use crate::gpu_backends::vulkan::device::{pipelines, swapchain};
use crate::simulation::comet::{NORMAL_COMPONENTS, POSITION_COMPONENTS, UV_COMPONENTS};
use crate::types::{GpuError, GpuResult};
use alloc::vec::Vec;
use ash::vk;
use function_name::named;

// TODO rewrite error messages

#[named]
fn get_validated_shaders(
  shader_manager: &shader_manager::ShaderManager,
  vertex_shader_key: ShaderKey,
  fragment_shader_key: ShaderKey,
) -> GpuResult<(&shader_manager::Shader, &shader_manager::Shader)> {
  let vertex_shader = shader_manager.get(vertex_shader_key).ok_or(GpuError::InvalidShader)?;
  if vertex_shader.shader_stage != vk::ShaderStageFlags::VERTEX {
    return Err(GpuError::InvalidShader);
  }
  let fragment_shader = shader_manager.get(fragment_shader_key).ok_or(GpuError::InvalidShader)?;
  if fragment_shader.shader_stage != vk::ShaderStageFlags::FRAGMENT {
    return Err(GpuError::InvalidShader);
  }
  Ok((vertex_shader, fragment_shader))
}

#[derive(Default)]
/// TODO: Document this item
pub(super) struct Archetypes {
  pub sun_render_archetype: spin::RwLock<Option<resources::SunRenderResourceArchetype>>,
  pub physical_mesh_render_archetype: spin::RwLock<Option<ForwardMeshRenderResourceArchetype>>,
  pub physical_mesh2_render_archetype:
    spin::RwLock<Option<resources::ForwardMesh2RenderResourceArchetype>>,
  pub billboard_render_archetype: spin::RwLock<Option<resources::BillboardRenderResourceArchetype>>,
  pub particle_render_archetype: spin::RwLock<Option<resources::ParticleRenderResourceArchetype>>,
  pub cursor_render_archetype: spin::RwLock<Option<resources::CursorRenderResourceArchetype>>,
  pub marker_render_archetype: spin::RwLock<Option<resources::MarkerRenderResourceArchetype>>,
  pub measurement_render_archetype:
    spin::RwLock<Option<resources::MeasurementRenderResourceArchetype>>,
  pub sky_render_archetype: spin::RwLock<Option<resources::SkyRenderResourceArchetype>>,
  pub grid_render_archetype: spin::RwLock<Option<resources::GridRenderResourceArchetype>>,
  pub minimap_render_archetype: spin::RwLock<Option<resources::MinimapRenderResourceArchetype>>,
  pub text_render_archetype: spin::RwLock<Option<resources::TextRenderResourceArchetype>>,
  pub text2_render_archetype: spin::RwLock<Option<resources::Text2RenderResourceArchetype>>,
  pub bvh_render_archetype: spin::RwLock<Option<resources::BvhRenderResourceArchetype>>,
  pub bvhwire2_render_archetype: spin::RwLock<Option<resources::Bvhwire2RenderResourceArchetype>>,
  pub gizmo_render_archetype: spin::RwLock<Option<resources::GizmoRenderResourceArchetype>>,
  pub particle2_render_archetype: spin::RwLock<Option<resources::Particle2RenderResourceArchetype>>,
  pub trajectory_render_archetype:
    spin::RwLock<Option<resources::TrajectoryRenderResourceArchetype>>,
  pub ui_render_archetype: spin::RwLock<Option<resources::UiRenderResourceArchetype>>,
  pub background_render_archetype:
    spin::RwLock<Option<resources::BackgroundRenderResourceArchetype>>,
}

impl Archetypes {
  /// TODO: Document this item
  #[named]
  pub fn has_discardables(&self) -> bool {
    self.sun_render_archetype.read().is_some()
      || self.physical_mesh_render_archetype.read().is_some()
      || self.physical_mesh2_render_archetype.read().is_some()
      || self.billboard_render_archetype.read().is_some()
      || self.particle_render_archetype.read().is_some()
      || self.cursor_render_archetype.read().is_some()
      || self.marker_render_archetype.read().is_some()
      || self.measurement_render_archetype.read().is_some()
      || self.sky_render_archetype.read().is_some()
      || self.grid_render_archetype.read().is_some()
      || self.minimap_render_archetype.read().is_some()
      || self.text_render_archetype.read().is_some()
      || self.text2_render_archetype.read().is_some()
      || self.bvh_render_archetype.read().is_some()
      || self.bvhwire2_render_archetype.read().is_some()
      || self.ui_render_archetype.read().is_some()
      || self.gizmo_render_archetype.read().is_some()
      || self.trajectory_render_archetype.read().is_some()
      || self.background_render_archetype.read().is_some()
  }

  /// TODO: Document this item
  #[named]
  pub fn discard(&self, device: &ash::Device, discard_pool: &resources::DiscardPool) {
    if let Some(mut archetype) = self.sun_render_archetype.write().take() {
      archetype.discard(device, &discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.physical_mesh_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.physical_mesh2_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.billboard_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.particle_render_archetype.write().take() {
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
    if let Some(mut archetype) = self.text2_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.bvh_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.bvhwire2_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.measurement_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.gizmo_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.particle2_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.trajectory_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.ui_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
    if let Some(mut archetype) = self.background_render_archetype.write().take() {
      archetype.discard(device, discard_pool, u64::MAX);
    }
  }
}

macro_rules! impl_update_archetype {
  (
    $fn_name:ident,
    $archetype_field:ident
    $(, |$arch:ident, $dev:ident, $wp:ident, $dp:ident, $tl:ident, $gi:ident, $fmt:ident| $extra:block)?
  ) => {
    /// TODO: Document this item
    #[named]
    pub fn $fn_name(
      &self,
      device: &LogicalDevice,
      color_format: vk::Format,
      write_pipeline: &mut pipelines::PipelinePool,
      renderpasses: &renderpasses::RenderPasses,
      allocator: &vk_mem::Allocator,
      discard_pool: &resources::DiscardPool,
      timeline: u64,
    ) -> GpuResult<()> {
      let mut archetype_lock = self.$archetype_field.write();
      let archetype = match archetype_lock.as_mut() {
        Some(a) => a,
        None => return Ok(()), // if it's not there, there's nothing to update (TODO check if correct)
      };

      let mut graphics_info = archetype.get_any_graphics_info().ok_or(crate::gpu_err_device!())?;

      let format = color_format;

      let mut new_pipeline_key = None;
      if !archetype.has_format(format) {
        let depth_stencil_format = graphics_info
          .fragment_out
          .depth_attachment_format
          .unwrap_or(vk::Format::UNDEFINED);

        graphics_info.fragment_out.color_attachment_formats.clear();
        graphics_info
          .fragment_out
          .color_attachment_formats
          .push(format);
        graphics_info.render_pass = renderpasses
          .get_pipeline_render_pass(
            color_format,
            depth_stencil_format,
          )?
          .get();
        write_pipeline.get_or_create_graphics_pipeline(device, &graphics_info)?;
        let key = graphics_info.pipeline_key();
        new_pipeline_key = Some(key);
      }

      if let Some(key) = new_pipeline_key {
        archetype.insert_graphics_info(format, graphics_info.clone(), key);
      }

      $(
        let $arch = &mut *archetype;
        let $dev = device;
        let $wp = &mut *write_pipeline;
        let $dp = discard_pool;
        let $tl = timeline;
        let $gi = &graphics_info;
        let $fmt = format;
        $extra
      )?

      Ok(())
    }
  };
}

macro_rules! impl_create_archetype {
  // 1. Match WITH the `ref_alloc` keyword (Passes `allocator` directly)
  (
    $fn_name:ident,
    $archetype_field:ident,
    $resource_struct:ident,
    ref_alloc
    $(, |$gi:ident| $extra:block)?
  ) => {
    /// TODO: Document this item
    #[named]
    pub fn $fn_name(
      &self,
      device: &LogicalDevice,
      shader_manager: &shader_manager::ShaderManager,
      vertex_shader_key: ShaderKey,
      fragment_shader_key: ShaderKey,
      depth_stencil_format: vk::Format,
      color_format: vk::Format,
      allocator: &vk_mem::Allocator,
      discard_pool: &resources::DiscardPool,
      renderpasses: &renderpasses::RenderPasses,
      pipeline_pool: &mut pipelines::PipelinePool,
      timeline: u64,
    ) -> GpuResult<()> {
      let mut archetype_lock = self.$archetype_field.write();
      if archetype_lock.is_some() {
        return Err(crate::gpu_err_device!());
      }

      let (vertex_shader, fragment_shader) = get_validated_shaders(shader_manager, vertex_shader_key, fragment_shader_key)?;

      // -> USING allocator DIRECTLY <-
      let res = unsafe { resources::$resource_struct::new(device, allocator) }?;
      *archetype_lock = Some(res);

      let layout = archetype_lock.as_ref().ok_or(crate::gpu_err_device!())?.pipeline_layout.get();
      let render_pass = renderpasses
        .get_pipeline_render_pass(color_format, depth_stencil_format)?.get();

      let graphics_info = GraphicsInfo::default()
        .with_pre_rasterization(
          PreRasterization::default().with_vertex_module(vertex_shader.module.get())
        )
        .with_fragment_shader(
          FragmentShader::default().with_fragment_module(fragment_shader.module.get())
        )
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL);

      $(
        let graphics_info = {
          let $gi = graphics_info;
          $extra // MAKE SURE THIS DOES NOT END WITH A SEMICOLON
        };
      )?

      let pipeline_graphics_info = graphics_info.apply_presentation_defaults(
        color_format, depth_stencil_format, layout, render_pass
      );

      pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

      let arch_mut = archetype_lock.as_mut().ok_or(crate::gpu_err_device!())?;
      arch_mut.insert_graphics_info(color_format, pipeline_graphics_info.clone(), pipeline_graphics_info.pipeline_key());

      Ok(())
    }
  };

  // 2. Match WITHOUT the keyword (Defaults to `allocator.get_raw()`)
  (
    $fn_name:ident,
    $archetype_field:ident,
    $resource_struct:ident
    $(, |$gi:ident| $extra:block)?
  ) => {
    /// TODO: Document this item
    #[named]
    pub fn $fn_name(
      &self,
      device: &LogicalDevice,
      shader_manager: &shader_manager::ShaderManager,
      vertex_shader_key: ShaderKey,
      fragment_shader_key: ShaderKey,
      depth_stencil_format: vk::Format,
      color_format: vk::Format,
      allocator: &vk_mem::Allocator,
      discard_pool: &resources::DiscardPool,
      renderpasses: &renderpasses::RenderPasses,
      pipeline_pool: &mut pipelines::PipelinePool,
      timeline: u64,
    ) -> GpuResult<()> {
      let mut archetype_lock = self.$archetype_field.write();
      if archetype_lock.is_some() {
        return Err(crate::gpu_err_device!());
      }

      let (vertex_shader, fragment_shader) = get_validated_shaders(shader_manager, vertex_shader_key, fragment_shader_key)?;

      // -> USING allocator.get_raw() <-
      let res = unsafe { resources::$resource_struct::new(device, allocator.get_raw()) }?;
      *archetype_lock = Some(res);

      let layout = archetype_lock.as_ref().ok_or(crate::gpu_err_device!())?.pipeline_layout.get();
      let render_pass = renderpasses
        .get_pipeline_render_pass(color_format, depth_stencil_format)?.get();

      let graphics_info = GraphicsInfo::default()
        .with_pre_rasterization(
          PreRasterization::default().with_vertex_module(vertex_shader.module.get())
        )
        .with_fragment_shader(
          FragmentShader::default().with_fragment_module(fragment_shader.module.get())
        )
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL);

      $(
        let graphics_info = {
          let $gi = graphics_info;
          $extra // MAKE SURE THIS DOES NOT END WITH A SEMICOLON
        };
      )?

      let pipeline_graphics_info = graphics_info.apply_presentation_defaults(
        color_format, depth_stencil_format, layout, render_pass
      );

      pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

      let arch_mut = archetype_lock.as_mut().ok_or(crate::gpu_err_device!())?;
      arch_mut.insert_graphics_info(color_format, pipeline_graphics_info.clone(), pipeline_graphics_info.pipeline_key());

      Ok(())
    }
  };
}
impl Archetypes {
  impl_update_archetype!(
    update_physical_mesh_archetype_for_presentation_engine,
    physical_mesh_render_archetype,
    |archetype, device, write_pipeline, discard_pool, timeline, graphics_info, format| {
      if !archetype.has_format_outline_pipeline_map(format) {
        let mut outline_graphics_info =
          archetype.get_any_graphics_info_outline_pipeline_map().unwrap();
        outline_graphics_info.fragment_out.color_attachment_formats.clear();
        outline_graphics_info.fragment_out.color_attachment_formats.push(format);
        outline_graphics_info.render_pass = graphics_info.render_pass;

        // TODO discard old pipeline (also for others)
        let outline_pipeline_key = outline_graphics_info.pipeline_key();
        write_pipeline.get_or_create_graphics_pipeline(device, &outline_graphics_info)?;
        archetype.insert_graphics_info_outline_pipeline_map(
          format,
          outline_graphics_info,
          outline_pipeline_key,
        );
      }
    }
  );

  impl_update_archetype!(
    update_physical_mesh2_archetype_for_presentation_engine,
    physical_mesh2_render_archetype,
    |archetype, device, write_pipeline, discard_pool, timeline, graphics_info, format| {
      if !archetype.has_format_outline_pipeline_map(format) {
        let mut outline_graphics_info =
          archetype.get_any_graphics_info_outline_pipeline_map().unwrap();
        outline_graphics_info.fragment_out.color_attachment_formats.clear();
        outline_graphics_info.fragment_out.color_attachment_formats.push(format);
        outline_graphics_info.render_pass = graphics_info.render_pass;

        let outline_pipeline_key = outline_graphics_info.pipeline_key();
        write_pipeline.get_or_create_graphics_pipeline(device, &outline_graphics_info)?;
        archetype.insert_graphics_info_outline_pipeline_map(
          format,
          outline_graphics_info,
          outline_pipeline_key,
        );
      }
    }
  );

  impl_update_archetype!(
    update_cursor_archetype_for_presentation_engine,
    cursor_render_archetype
  );
  impl_update_archetype!(
    update_particle_archetype_for_presentation_engine,
    particle_render_archetype
  );
  impl_update_archetype!(
    update_particle2_archetype_for_presentation_engine,
    particle2_render_archetype
  );
  impl_update_archetype!(
    update_sun_archetype_for_presentation_engine,
    sun_render_archetype
  );
  impl_update_archetype!(
    update_sky_archetype_for_presentation_engine,
    sky_render_archetype
  );

  impl_update_archetype!(
    update_grid_archetype_for_presentation_engine,
    grid_render_archetype
  );

  impl_update_archetype!(
    update_minimap_archetype_for_presentation_engine,
    minimap_render_archetype
  );

  impl_update_archetype!(
    update_text_archetype_for_presentation_engine,
    text_render_archetype
  );

  impl_update_archetype!(
    update_text2_archetype_for_presentation_engine,
    text2_render_archetype
  );

  impl_update_archetype!(
    update_bvh_archetype_for_presentation_engine,
    bvh_render_archetype
  );

  impl_update_archetype!(
    update_bvhwire2_archetype_for_presentation_engine,
    bvhwire2_render_archetype
  );

  impl_update_archetype!(
    update_gizmo_archetype_for_presentation_engine,
    gizmo_render_archetype
  );

  impl_update_archetype!(
    update_measurement_archetype_for_presentation_engine,
    measurement_render_archetype
  );

  impl_update_archetype!(
    update_marker_archetype_for_presentation_engine,
    marker_render_archetype
  );

  impl_update_archetype!(
    update_billboard_archetype_for_presentation_engine,
    billboard_render_archetype
  );

  impl_update_archetype!(
    update_trajectory_archetype_for_presentation_engine,
    trajectory_render_archetype
  );

  impl_update_archetype!(
    update_ui_archetype_for_presentation_engine,
    ui_render_archetype
  );

  impl_update_archetype!(
    update_background_archetype_for_presentation_engine,
    background_render_archetype
  );

  // ------------------------------------ Creation ------------------------------------

  impl_create_archetype!(
    create_sun_archetype,
    sun_render_archetype,
    SunRenderResourceArchetype,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::NO_DEPTH_WRITE)
    }
  );

  impl_create_archetype!(
    create_particle_archetype,
    particle_render_archetype,
    ParticleRenderResourceArchetype,
    ref_alloc,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::empty())
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  impl_create_archetype!(
    create_particle2_archetype,
    particle2_render_archetype,
    Particle2RenderResourceArchetype,
    ref_alloc,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::empty())
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  impl_create_archetype!(
    create_trajectory_archetype,
    trajectory_render_archetype,
    TrajectoryRenderResourceArchetype,
    ref_alloc,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::empty())
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  impl_create_archetype!(
    create_ui_archetype,
    ui_render_archetype,
    UiRenderResourceArchetype,
    ref_alloc,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_LIST))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST)
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  #[named]
  pub fn create_background_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut bg_render_archetype = self.background_render_archetype.write();
    if bg_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }
    let (vertex_shader, fragment_shader) =
      get_validated_shaders(shader_manager, vertex_shader_key, fragment_shader_key)?;

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(32)]; // 2x vec4 (2 * 4 * 4 = 32 bytes)
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    *bg_render_archetype = Some(resources::BackgroundRenderResourceArchetype::new(
      pipeline_layout,
    ));

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_LIST).clone(),
      )
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()).clone(),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
          .clone(),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST)
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

    bg_render_archetype.as_mut().unwrap().insert_graphics_info(
      color_format,
      pipeline_graphics_info,
      pipeline_key,
    );

    Ok(())
  }

  impl_create_archetype!(
    create_cursor_archetype,
    cursor_render_archetype,
    CursorRenderResourceArchetype,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_TEST)
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  impl_create_archetype!(
    create_measurement_archetype,
    measurement_render_archetype,
    MeasurementRenderResourceArchetype,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::LINE_LIST))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_TEST | PipelineFlags::NO_DEPTH_WRITE)
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  impl_create_archetype!(
    create_marker_archetype,
    marker_render_archetype,
    MarkerRenderResourceArchetype,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::empty())
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  impl_create_archetype!(
    create_billboard_archetype,
    billboard_render_archetype,
    BillboardRenderResourceArchetype,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_TEST)
        .with_stencil_compare_op(StencilCompareOp::None)
        .with_stencil_logic_op(StencilLogicOp::Replace)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(0)
        .with_stencil_write_mask(u32::MAX)
    }
  );

  /// TODO: Document this item
  #[named]
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
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    if self.physical_mesh_render_archetype.read().is_some() {
      return Err(crate::gpu_err_device!());
    }

    let vertex_shader = shader_manager.get(vertex_shader_key).ok_or(GpuError::InvalidShader)?;
    if vertex_shader.shader_stage != vk::ShaderStageFlags::VERTEX {
      return Err(GpuError::InvalidShader);
    }
    let fragment_shader = shader_manager.get(fragment_shader_key).ok_or(GpuError::InvalidShader)?;
    if fragment_shader.shader_stage != vk::ShaderStageFlags::FRAGMENT {
      return Err(GpuError::InvalidShader);
    }

    let outline_vertex_shader =
      shader_manager.get(outline_vertex_shader_key).ok_or(GpuError::InvalidShader)?;
    if outline_vertex_shader.shader_stage != vk::ShaderStageFlags::VERTEX {
      return Err(GpuError::InvalidShader);
    }
    let outline_fragment_shader =
      shader_manager.get(outline_fragment_shader_key).ok_or(GpuError::InvalidShader)?;
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
        staging_arena,
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
          ), // inTangent
      )
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          // NOTE: Viewport and Scissor setup here is generally ignored during command buffer recording
          // because they are bound as DYNAMIC_STATE. We provide them here as fallback.
          // necessary to call add because count is not a dynamic state
          .add_viewport(ignored_viewport())
          // NOTE: Viewport and Scissor setup here is generally ignored during command buffer recording
          // because they are bound as DYNAMIC_STATE. We provide them here as fallback.
          // necessary to call add because count is not a dynamic state
          .add_scissors(ignored_scissor()),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .with_stencil_attachment_format(depth_stencil_format),
      )
      .with_pipeline_layout(res.pipeline_layout.get())
      .with_pipeline_flags(
        PipelineFlags::CULL_BACK | PipelineFlags::STENCIL_ENABLE | PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_stencil_compare_op(StencilCompareOp::Always)
      .with_stencil_logic_op(StencilLogicOp::Replace)
      .with_stencil_reference(255)
      .with_stencil_write_mask(u32::MAX)
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .with_stencil_compare_op(StencilCompareOp::None)
      .with_stencil_logic_op(StencilLogicOp::Replace)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(0)
      .with_stencil_write_mask(u32::MAX)
      .clone();

    aethervk_oshal_rlib::log!("Creating graphics pipeline for physical_mesh2...");
    pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)
      .inspect_err(|e| aethervk_oshal_rlib::log!("Failed to create graphics pipeline: {:?}", e))?;

    // Note: old code rendered outlines with back faces and stencil buffer. But:
    // - That is the traditional technique because traditional pipelines don't use Stencil Masks.
    // - The flaw with backfaces is that they sit behind the mesh. If your character stands against a wall, the expanded backfaces will clip into the wall, fail the depth test, and the outline will awkwardly disappear.
    // - Because you use a Stencil Mask, you are already stopping the outline from rendering over the mesh itself. Therefore, you can safely change your outline pipeline to render Front Faces. This projects the outline forward, preventing it from clipping into nearby background walls!
    let outline_graphics_info = pipeline_graphics_info
      .clone()
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(outline_vertex_shader.module.get()),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(outline_fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor()),
      )
      .with_pipeline_flags(
        PipelineFlags::STENCIL_ENABLE
          | PipelineFlags::NO_DEPTH_TEST
          | PipelineFlags::NO_DEPTH_WRITE,
      )
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .with_stencil_compare_op(StencilCompareOp::NotEqual)
      .with_stencil_logic_op(StencilLogicOp::None)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(255)
      .with_stencil_write_mask(0)
      .clone();

    let outline_pipeline_key = outline_graphics_info.pipeline_key();

    aethervk_oshal_rlib::log!("Creating outline graphics pipeline for physical_mesh2...");
    pipeline_pool.get_or_create_graphics_pipeline(&device, &outline_graphics_info).inspect_err(
      |e| aethervk_oshal_rlib::log!("Failed to create outline graphics pipeline: {:?}", e),
    )?;

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    let final_res = res
      .with_graphics_info(color_format, pipeline_graphics_info, pipeline_key)
      .with_graphics_info_outline_pipeline_map(
        color_format,
        outline_graphics_info,
        outline_pipeline_key,
      );
    *self.physical_mesh_render_archetype.write() = Some(final_res);

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_physical_mesh2_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    outline_vertex_shader_key: ShaderKey,
    outline_fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    staging_arena: &crate::gpu_backends::vulkan::device::memory::FrameStagingArena,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    if self.physical_mesh2_render_archetype.read().is_some() {
      return Err(crate::gpu_err_device!());
    }

    let vertex_shader = shader_manager.get(vertex_shader_key).ok_or(GpuError::InvalidShader)?;
    if vertex_shader.shader_stage != vk::ShaderStageFlags::VERTEX {
      return Err(GpuError::InvalidShader);
    }
    let fragment_shader = shader_manager.get(fragment_shader_key).ok_or(GpuError::InvalidShader)?;
    if fragment_shader.shader_stage != vk::ShaderStageFlags::FRAGMENT {
      return Err(GpuError::InvalidShader);
    }

    let outline_vertex_shader =
      shader_manager.get(outline_vertex_shader_key).ok_or(GpuError::InvalidShader)?;
    if outline_vertex_shader.shader_stage != vk::ShaderStageFlags::VERTEX {
      return Err(GpuError::InvalidShader);
    }
    let outline_fragment_shader =
      shader_manager.get(outline_fragment_shader_key).ok_or(GpuError::InvalidShader)?;
    if outline_fragment_shader.shader_stage != vk::ShaderStageFlags::FRAGMENT {
      return Err(GpuError::InvalidShader);
    }

    // Create initial struct
    aethervk_oshal_rlib::log!("Creating ForwardMesh2RenderResourceArchetype...");
    let res = unsafe {
      resources::ForwardMesh2RenderResourceArchetype::new(
        device,
        &vertex_shader,
        &fragment_shader,
        allocator,
        discard_pool,
        staging_arena,
        &queue,
      )
    }
    .inspect_err(|e| {
      aethervk_oshal_rlib::log!("ForwardMesh2RenderResourceArchetype::new failed: {:?}", e)
    })?;

    aethervk_oshal_rlib::log!("Populating GraphicsInfo...");
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
          ), // inTangent
      )
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor()),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .with_stencil_attachment_format(depth_stencil_format),
      )
      .with_pipeline_layout(res.pipeline_layout.get())
      .with_pipeline_flags(
        PipelineFlags::CULL_BACK | PipelineFlags::STENCIL_ENABLE | PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_stencil_compare_op(StencilCompareOp::Always)
      .with_stencil_logic_op(StencilLogicOp::Replace)
      .with_stencil_reference(255)
      .with_stencil_write_mask(u32::MAX)
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .with_stencil_compare_op(StencilCompareOp::None)
      .with_stencil_logic_op(StencilLogicOp::Replace)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(0)
      .with_stencil_write_mask(u32::MAX)
      .clone();

    aethervk_oshal_rlib::log!("Creating graphics pipeline for physical_mesh2...");
    pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)
      .inspect_err(|e| aethervk_oshal_rlib::log!("Failed to create graphics pipeline: {:?}", e))?;

    let outline_graphics_info = pipeline_graphics_info
      .clone()
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(outline_vertex_shader.module.get()),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(outline_fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor()),
      )
      .with_pipeline_flags(
        PipelineFlags::STENCIL_ENABLE
          | PipelineFlags::NO_DEPTH_TEST
          | PipelineFlags::NO_DEPTH_WRITE,
      )
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .with_stencil_compare_op(StencilCompareOp::NotEqual)
      .with_stencil_logic_op(StencilLogicOp::None)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(255)
      .with_stencil_write_mask(0)
      .clone();

    let outline_pipeline_key = outline_graphics_info.pipeline_key();

    aethervk_oshal_rlib::log!("Creating outline graphics pipeline for physical_mesh2...");
    pipeline_pool.get_or_create_graphics_pipeline(&device, &outline_graphics_info).inspect_err(
      |e| aethervk_oshal_rlib::log!("Failed to create outline graphics pipeline: {:?}", e),
    )?;

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    let final_res = res
      .with_graphics_info(color_format, pipeline_graphics_info, pipeline_key)
      .with_graphics_info_outline_pipeline_map(
        color_format,
        outline_graphics_info,
        outline_pipeline_key,
      );
    *self.physical_mesh2_render_archetype.write() = Some(final_res);

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_sky_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut sky_render_archetype = self.sky_render_archetype.write();
    if sky_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }
    let (vertex_shader, fragment_shader) =
      get_validated_shaders(shader_manager, vertex_shader_key, fragment_shader_key)?;

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

    let mut arch = resources::SkyRenderResourceArchetype::new(pipeline_layout, set_layout);

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_LIST))
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor()),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST)
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL);

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;
    arch = arch.with_graphics_info(color_format, pipeline_graphics_info, pipeline_key);

    *sky_render_archetype = Some(arch);

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_grid_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut grid_render_archetype = self.grid_render_archetype.write();
    if grid_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }
    let (vertex_shader, fragment_shader) =
      get_validated_shaders(shader_manager, vertex_shader_key, fragment_shader_key)?;

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(128)];
    let pipeline_layout_info =
      vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);
    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None) }?;

    *grid_render_archetype = Some(resources::GridRenderResourceArchetype::new(pipeline_layout));

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP).clone(),
      )
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()).clone(),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
          .clone(),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(PipelineFlags::empty())
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

    grid_render_archetype.as_mut().unwrap().insert_graphics_info(
      color_format,
      pipeline_graphics_info,
      pipeline_key,
    );

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_minimap_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut minimap_render_archetype = self.minimap_render_archetype.write();
    if minimap_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }
    *minimap_render_archetype =
      Some(unsafe { resources::MinimapRenderResourceArchetype::new(device, allocator.get_raw())? });
    let arch_mut = minimap_render_archetype.as_mut().ok_or(crate::gpu_err_device!())?;

    let (vertex_shader, fragment_shader) = get_validated_shaders(shader_manager, vkey, fkey)?;

    let pipeline_graphics_info = pipelines::GraphicsInfo::default()
      .with_vertex_in(
        pipelines::VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP),
      )
      .with_pre_rasterization(
        pipelines::PreRasterization::default().with_vertex_module(vertex_shader.module.get()),
      )
      .with_fragment_shader(
        pipelines::FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor()),
      )
      .with_fragment_out(
        pipelines::FragmentOut::default().add_color_attachment_format(color_format).clone(),
      )
      .with_pipeline_layout(arch_mut.pipeline_layout.get())
      .with_pipeline_flags(
        pipelines::PipelineFlags::NO_DEPTH_TEST | pipelines::PipelineFlags::NO_DEPTH_WRITE,
      )
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    aethervk_oshal_rlib::log!("Creating graphics pipeline for physical_mesh2...");
    pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)
      .inspect_err(|e| aethervk_oshal_rlib::log!("Failed to create graphics pipeline: {:?}", e))?;

    arch_mut.insert_graphics_info(color_format, pipeline_graphics_info, pipeline_key);

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_text_archetype(
    &self,
    device: &vulkan::device::LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut text_render_archetype = self.text_render_archetype.write();
    if text_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

    let (vertex_shader, fragment_shader) =
      get_validated_shaders(shader_manager, vertex_shader_key, fragment_shader_key)?;

    let max_fonts = 256; // Array limit

    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_fonts)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    // flags from descriptor_indexing
    let binding_flags =
      [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
    let mut binding_flags_info =
      vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

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

    let alloc_info =
      vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?[0];

    let mut arch = resources::TextRenderResourceArchetype::new(
      pipeline_layout,
      set_layout,
      pool,
      descriptor_set,
      font_sampler,
      max_fonts,
      allocator,
    );

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor()),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST)
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

    *text_render_archetype =
      Some(arch.with_graphics_info(color_format, pipeline_graphics_info, pipeline_key));

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_text2_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vertex_shader_key: ShaderKey,
    fragment_shader_key: ShaderKey,
    depth_stencil_format: vk::Format,
    queue: &Queue,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut text2_render_archetype = self.text2_render_archetype.write();
    if text2_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

    let (vertex_shader, fragment_shader) =
      get_validated_shaders(shader_manager, vertex_shader_key, fragment_shader_key)?;

    let max_fonts = 256;

    let bindings = [vk::DescriptorSetLayoutBinding::default()
      .binding(0)
      .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
      .descriptor_count(max_fonts)
      .stage_flags(vk::ShaderStageFlags::FRAGMENT)];

    let binding_flags =
      [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND];
    let mut binding_flags_info =
      vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);

    let layout_info = vk::DescriptorSetLayoutCreateInfo::default()
      .bindings(&bindings)
      .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
      .push_next(&mut binding_flags_info);
    let set_layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }?;
    let set_layouts = [set_layout];

    let push_constant_ranges = [vk::PushConstantRange::default()
      .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
      .offset(0)
      .size(72)]; // 72 bytes Push Constant
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

    let alloc_info =
      vk::DescriptorSetAllocateInfo::default().descriptor_pool(pool).set_layouts(&set_layouts);
    let descriptor_set = unsafe { device.allocate_descriptor_sets(&alloc_info) }?[0];

    let mut arch = resources::Text2RenderResourceArchetype::new(
      pipeline_layout,
      set_layout,
      pool,
      descriptor_set,
      font_sampler,
      max_fonts,
      allocator,
      device,
    )?;

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()),
      )
      .with_fragment_shader(
        FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor()),
      )
      .with_fragment_out(
        FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .with_stencil_attachment_format(depth_stencil_format),
      )
      .with_pipeline_layout(pipeline_layout)
      .with_pipeline_flags(
        PipelineFlags::NO_DEPTH_WRITE
          | PipelineFlags::NO_DEPTH_TEST
          | PipelineFlags::STENCIL_ENABLE,
      )
      .with_stencil_compare_op(StencilCompareOp::None)
      .with_stencil_logic_op(StencilLogicOp::Replace)
      .with_stencil_reference(255)
      .with_stencil_compare_mask(0)
      .with_stencil_write_mask(u32::MAX)
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool.get_or_create_graphics_pipeline(device, &pipeline_graphics_info)?;

    *text2_render_archetype =
      Some(arch.with_graphics_info(color_format, pipeline_graphics_info, pipeline_key));

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_bvh_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut bvh_render_archetype = self.bvh_render_archetype.write();
    if bvh_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }
    *bvh_render_archetype =
      Some(unsafe { resources::BvhRenderResourceArchetype::new(device, allocator.get_raw()) }?);
    let archetype = bvh_render_archetype.as_mut().ok_or(crate::gpu_err_device!())?;

    let (vertex_shader, fragment_shader) = get_validated_shaders(shader_manager, vkey, fkey)?;

    let pipeline_graphics_info = pipelines::GraphicsInfo::default()
      .with_vertex_in(
        pipelines::VertexIn::default().with_topology(vk::PrimitiveTopology::LINE_LIST).clone(),
      )
      .with_pre_rasterization(
        pipelines::PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
      )
      .with_fragment_shader(
        pipelines::FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
          .clone(),
      )
      .with_fragment_out(
        pipelines::FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(archetype.pipeline_layout.get())
      .with_pipeline_flags(
        pipelines::PipelineFlags::NO_DEPTH_WRITE | pipelines::PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::LINE)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    aethervk_oshal_rlib::log!("Creating graphics pipeline for bvh...");
    pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)
      .inspect_err(|e| aethervk_oshal_rlib::log!("Failed to create graphics pipeline: {:?}", e))?;

    archetype.insert_graphics_info(color_format, pipeline_graphics_info, pipeline_key);

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_bvhwire2_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut bvhwire2_render_archetype = self.bvhwire2_render_archetype.write();
    if bvhwire2_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }
    *bvhwire2_render_archetype =
      Some(unsafe { resources::Bvhwire2RenderResourceArchetype::new(device, allocator) }?);
    let archetype = bvhwire2_render_archetype.as_mut().ok_or(crate::gpu_err_device!())?;

    let (vertex_shader, fragment_shader) = get_validated_shaders(shader_manager, vkey, fkey)?;

    let pipeline_graphics_info = pipelines::GraphicsInfo::default()
      .with_vertex_in(
        pipelines::VertexIn::default().with_topology(vk::PrimitiveTopology::LINE_LIST).clone(),
      )
      .with_pre_rasterization(
        pipelines::PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
      )
      .with_fragment_shader(
        pipelines::FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
          .clone(),
      )
      .with_fragment_out(
        pipelines::FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(archetype.pipeline_layout.get())
      .with_pipeline_flags(
        pipelines::PipelineFlags::NO_DEPTH_WRITE | pipelines::PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::LINE)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    aethervk_oshal_rlib::log!("Creating graphics pipeline for bvhwire2...");
    pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)
      .inspect_err(|e| aethervk_oshal_rlib::log!("Failed to create graphics pipeline: {:?}", e))?;

    archetype.insert_graphics_info(color_format, pipeline_graphics_info, pipeline_key);

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_gizmo_archetype(
    &self,
    device: &LogicalDevice,
    shader_manager: &shader_manager::ShaderManager,
    vkey: ShaderKey,
    fkey: ShaderKey,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    allocator: &vk_mem::Allocator,
    discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool: &mut pipelines::PipelinePool,
    timeline: u64,
  ) -> GpuResult<()> {
    let mut gizmo_render_archetype = self.gizmo_render_archetype.write();
    if gizmo_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }
    *gizmo_render_archetype =
      Some(unsafe { resources::GizmoRenderResourceArchetype::new(device, allocator.get_raw()) }?);
    let archetype = gizmo_render_archetype.as_mut().ok_or(crate::gpu_err_device!())?;

    let (vertex_shader, fragment_shader) = get_validated_shaders(shader_manager, vkey, fkey)?;

    let pipeline_graphics_info = pipelines::GraphicsInfo::default()
      .with_vertex_in(
        pipelines::VertexIn::default().with_topology(vk::PrimitiveTopology::LINE_LIST).clone(),
      )
      .with_pre_rasterization(
        pipelines::PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
      )
      .with_fragment_shader(
        pipelines::FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
          .clone(),
      )
      .with_fragment_out(
        pipelines::FragmentOut::default()
          .add_color_attachment_format(color_format)
          .with_depth_attachment_format(depth_stencil_format)
          .clone(),
      )
      .with_pipeline_layout(archetype.pipeline_layout.get())
      .with_pipeline_flags(
        pipelines::PipelineFlags::NO_DEPTH_TEST
          | pipelines::PipelineFlags::NO_DEPTH_WRITE
          | pipelines::PipelineFlags::INVERT_FRONT_FACE,
      )
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::LINE)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    aethervk_oshal_rlib::log!("Creating graphics pipeline for physical_mesh2...");
    pipeline_pool
      .get_or_create_graphics_pipeline(&device, &pipeline_graphics_info)
      .inspect_err(|e| aethervk_oshal_rlib::log!("Failed to create graphics pipeline: {:?}", e))?;

    archetype.insert_graphics_info(color_format, pipeline_graphics_info, pipeline_key);

    Ok(())
  }
}

fn ignored_viewport() -> vk::Viewport {
  vk::Viewport::default()
}

fn ignored_scissor() -> vk::Rect2D {
  vk::Rect2D::default()
}