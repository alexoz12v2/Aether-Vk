//! archetypes_struct module.

use crate::gpu::{ArchetypeId, PipelineKey};
use crate::gpu_backends::vulkan::device::{renderpasses::RenderPasses, resources::DiscardPool};
use crate::{
  gpu::PipelineKeyable,
  gpu_backends::vulkan::device::{
    LogicalDevice, Queue,
    locks::DebugTrackedRwLock,
    pipelines::{
      self, FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization,
      StencilCompareOp, StencilLogicOp, VertexIn,
    },
    renderpasses, resources,
    shader_manager::{self, ShaderKey},
    utils::{self, RwLockable},
  },
  types::{GpuError, GpuResult},
};
use alloc::boxed::Box;
use ash::vk;
use function_name::named;

#[named]
pub(super) fn get_validated_shaders<'a>(
  shader_manager: &'a shader_manager::ShaderManager,
  vertex_shader_key: ShaderKey,
  fragment_shader_key: ShaderKey,
) -> GpuResult<(
  alloc::sync::Arc<shader_manager::Shader>,
  alloc::sync::Arc<shader_manager::Shader>,
)> {
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

pub trait RenderArchetype: Send + Sync {
  // Pipeline Getters
  fn pipeline_key(&self) -> PipelineKey;
  fn outline_pipeline_key(&self) -> Option<PipelineKey> {
    None
  }
  fn pipeline_layout(&self) -> vk::PipelineLayout;

  // Lifecycle hooks (replaces prepare_update_sun, prepare_update_sky, ...)
  fn prepare_update(
    &self,
    format: vk::Format,
    passes: &RenderPasses,
  ) -> GpuResult<Option<PreparedArchetypeUpdate>>;
  fn commit_update(&mut self, data: CompiledArchetypeData);
  fn discard_archetype(&mut self, device: &LogicalDevice, pool: &DiscardPool, timeline: u64);

  // Allows downcasting for specific methods (e.g., allocate_sphere_gizmo_instance)
  fn as_any(&self) -> &dyn core::any::Any;
  fn as_any_mut(&mut self) -> &mut dyn core::any::Any;
}

#[derive(Default)]
pub struct Archetypes {
  pub registry: DebugTrackedRwLock<hashbrown::HashMap<ArchetypeId, Box<dyn RenderArchetype>>>,
}

impl Archetypes {
  pub fn has_discardables(&self) -> bool {
    !self.registry.read().is_empty()
  }

  pub fn discard(&self, device: &LogicalDevice, pool: &DiscardPool) {
    let mut reg = self.registry.write();
    for (_, archetype) in reg.iter_mut() {
      archetype.discard_archetype(device, pool, u64::MAX);
    }
    reg.clear();
  }
}

macro_rules! impl_render_archetype {
  ($archetype:ident) => {
    impl RenderArchetype for resources::$archetype {
      fn pipeline_key(&self) -> PipelineKey {
        self.pipeline_key
      }
      fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.arena.upgrade().unwrap().read().pipeline_layout.get()
      }
      fn prepare_update(
        &self,
        format: vk::Format,
        passes: &RenderPasses,
      ) -> GpuResult<Option<PreparedArchetypeUpdate>> {
        let mut graphics_info = self.graphics_info.clone();
        if graphics_info.fragment_out.color_attachment_formats.first().copied() != Some(format) {
          let depth_stencil_format = graphics_info
            .fragment_out
            .depth_attachment_format
            .unwrap_or(vk::Format::UNDEFINED);

          graphics_info.fragment_out.color_attachment_formats.clear();
          graphics_info.fragment_out.color_attachment_formats.push(format);
          graphics_info.render_pass =
            passes.get_pipeline_render_pass(format, depth_stencil_format)?.get();

          Ok(Some(PreparedArchetypeUpdate {
            main_graphics_info: graphics_info,
            outline_graphics_info: None,
          }))
        } else {
          Ok(None)
        }
      }
      fn commit_update(&mut self, data: CompiledArchetypeData) {
        self.pipeline_key = data.pipeline_key;
        self.graphics_info = data.graphics_info;
      }
      fn discard_archetype(&mut self, device: &LogicalDevice, pool: &DiscardPool, timeline: u64) {
        self.discard(device, pool, timeline);
      }
      fn as_any(&self) -> &dyn core::any::Any {
        self
      }
      fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
      }
    }
  };

  ($archetype:ident, with_outline) => {
    impl RenderArchetype for resources::$archetype {
      fn pipeline_key(&self) -> PipelineKey {
        self.pipeline_key
      }
      fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.arena.upgrade().unwrap().read().pipeline_layout.get()
      }
      fn outline_pipeline_key(&self) -> Option<PipelineKey> {
        Some(self.outline_pipeline_key)
      }
      fn prepare_update(
        &self,
        format: vk::Format,
        passes: &RenderPasses,
      ) -> GpuResult<Option<PreparedArchetypeUpdate>> {
        let mut needs_update = false;
        let mut graphics_info = self.graphics_info.clone();
        let mut outline_graphics_info = self.outline_graphics_info.clone();

        if graphics_info.fragment_out.color_attachment_formats.first().copied() != Some(format) {
          let depth_stencil_format = graphics_info
            .fragment_out
            .depth_attachment_format
            .unwrap_or(vk::Format::UNDEFINED);

          graphics_info.fragment_out.color_attachment_formats.clear();
          graphics_info.fragment_out.color_attachment_formats.push(format);
          graphics_info.render_pass =
            passes.get_pipeline_render_pass(format, depth_stencil_format)?.get();
          needs_update = true;
        }

        if outline_graphics_info.fragment_out.color_attachment_formats.first().copied()
          != Some(format)
        {
          let depth_stencil_format = outline_graphics_info
            .fragment_out
            .depth_attachment_format
            .unwrap_or(vk::Format::UNDEFINED);

          outline_graphics_info.fragment_out.color_attachment_formats.clear();
          outline_graphics_info.fragment_out.color_attachment_formats.push(format);
          outline_graphics_info.render_pass =
            passes.get_pipeline_render_pass(format, depth_stencil_format)?.get();
          needs_update = true;
        }

        if needs_update {
          Ok(Some(PreparedArchetypeUpdate {
            main_graphics_info: graphics_info,
            outline_graphics_info: Some(outline_graphics_info),
          }))
        } else {
          Ok(None)
        }
      }
      fn commit_update(&mut self, data: CompiledArchetypeData) {
        self.pipeline_key = data.pipeline_key;
        self.graphics_info = data.graphics_info;
        if let Some((outline_key, outline_info)) = data.outline_data {
          self.outline_pipeline_key = outline_key;
          self.outline_graphics_info = outline_info;
        }
      }
      fn discard_archetype(&mut self, device: &LogicalDevice, pool: &DiscardPool, timeline: u64) {
        self.discard(device, pool, timeline);
      }
      fn as_any(&self) -> &dyn core::any::Any {
        self
      }
      fn as_any_mut(&mut self) -> &mut dyn core::any::Any {
        self
      }
    }
  };
}

// ---------------------- implementations -------------------------------------------

impl_render_archetype!(ForwardMesh2RenderResourceArchetype, with_outline);
impl_render_archetype!(SunRenderResourceArchetype);
impl_render_archetype!(SkyRenderResourceArchetype);
impl_render_archetype!(BackgroundRenderResourceArchetype);
impl_render_archetype!(GridRenderResourceArchetype);
impl_render_archetype!(MeasurementRenderResourceArchetype);
impl_render_archetype!(MarkerRenderResourceArchetype);
impl_render_archetype!(BillboardRenderResourceArchetype);
impl_render_archetype!(TrajectoryRenderResourceArchetype);
impl_render_archetype!(UiRenderResourceArchetype);
impl_render_archetype!(CursorRenderResourceArchetype);
impl_render_archetype!(Text2RenderResourceArchetype);
impl_render_archetype!(SphereGizmoRenderResourceArchetype);
impl_render_archetype!(GizmoRenderResourceArchetype);
impl_render_archetype!(DustRenderArchetype);

pub struct PreparedArchetypeUpdate {
  pub main_graphics_info: crate::gpu_backends::vulkan::device::pipelines::GraphicsInfo,
  pub outline_graphics_info: Option<crate::gpu_backends::vulkan::device::pipelines::GraphicsInfo>,
}

pub struct CompiledArchetypeData {
  pub pipeline_key: crate::gpu::PipelineKey,
  pub graphics_info: crate::gpu_backends::vulkan::device::pipelines::GraphicsInfo,
  pub outline_data: Option<(
    crate::gpu::PipelineKey,
    crate::gpu_backends::vulkan::device::pipelines::GraphicsInfo,
  )>,
}

macro_rules! impl_create_archetype {
  (
    $fn_name:ident,
    $archetype_id:expr,
    $resource_struct:ident,
    $arena_struct:ident
    $(, |$gi:ident| $extra:block)?
  ) => {
    #[named]
    pub fn $fn_name(
      &self,
      device: &LogicalDevice,
      vertex_shader: &shader_manager::Shader,
      fragment_shader: &shader_manager::Shader,
      depth_stencil_format: vk::Format,
      color_format: vk::Format,
      _allocator: vk_mem::AllocatorView,
      _discard_pool: &resources::DiscardPool,
      renderpasses: &renderpasses::RenderPasses,
      pipeline_pool_lock: &pipelines::PipelinePool,
      _timeline: u64,
      arena: alloc::sync::Arc<DebugTrackedRwLock<resources::$arena_struct>>,
      rollback: &mut utils::RollbackContext<'_>,
    ) -> GpuResult<()> {
      let mut registry = self.registry.write();
      if registry.contains_key(&$archetype_id) {
        return Err(crate::gpu_err_device!());
      }

      let layout = arena.read().pipeline_layout.get();
      let render_pass = renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get();

      let mut graphics_info = GraphicsInfo::default()
        .with_pre_rasterization(PreRasterization::default().with_vertex_module(vertex_shader.module.get()))
        .with_fragment_shader(FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
        )
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL);

      $(
        graphics_info = {
          let mut $gi = graphics_info;
          $extra
        };
      )?

      let pipeline_graphics_info = graphics_info.apply_presentation_defaults(color_format, depth_stencil_format, layout, render_pass);
      pipeline_pool_lock.get_or_create_graphics_pipeline(device, &pipeline_graphics_info, rollback)?;
      let pipeline_key = pipeline_graphics_info.pipeline_key();

      let res = resources::$resource_struct { arena: alloc::sync::Arc::downgrade(&arena), pipeline_key, graphics_info: pipeline_graphics_info };
      registry.insert($archetype_id, Box::new(res));

      Ok(())
    }
  };

  (
    $fn_name:ident,
    $archetype_id:expr,
    $resource_struct:ident,
    $arena_struct:ident,
    ref_alloc
    $(, |$gi:ident| $extra:block)?
  ) => {
    impl_create_archetype!($fn_name, $archetype_id, $resource_struct, $arena_struct $(, |$gi| $extra)?);
  };

  (
    $fn_name:ident,
    $archetype_id:expr,
    $resource_struct:ident,
    $arena_struct:ident,
    text
    $(, |$gi:ident| $extra:block)?
  ) => {
    #[named]
    pub fn $fn_name(
      &self,
      device: &LogicalDevice,
      vertex_shader: &shader_manager::Shader,
      fragment_shader: &shader_manager::Shader,
      depth_stencil_format: vk::Format,
      _queue: &Queue,
      color_format: vk::Format,
      _allocator: vk_mem::AllocatorView,
      _discard_pool: &resources::DiscardPool,
      renderpasses: &renderpasses::RenderPasses,
      pipeline_pool_lock: &pipelines::PipelinePool,
      _timeline: u64,
      arena: alloc::sync::Arc<DebugTrackedRwLock<resources::$arena_struct>>,
      rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
    ) -> GpuResult<()> {
      let mut registry = self.registry.write();
      if registry.contains_key(&$archetype_id) {
        return Err(crate::gpu_err_device!());
      }

      let layout = arena.read().pipeline_layout.get();
      let render_pass = renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get();
      let mut graphics_info = GraphicsInfo::default()
        .with_pre_rasterization(PreRasterization::default().with_vertex_module(vertex_shader.module.get()))
        .with_fragment_shader(FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
        )
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL);

      $(
        graphics_info = {
          let mut $gi = graphics_info;
          $extra
        };
      )?

      let pipeline_graphics_info = graphics_info.apply_presentation_defaults(color_format, depth_stencil_format, layout, render_pass);
      pipeline_pool_lock.get_or_create_graphics_pipeline(device, &pipeline_graphics_info, rollback)?;
      let pipeline_key = pipeline_graphics_info.pipeline_key();
      let res = resources::$resource_struct { arena: alloc::sync::Arc::downgrade(&arena), pipeline_key, graphics_info: pipeline_graphics_info };

      registry.insert($archetype_id, Box::new(res));

      Ok(())
    }
  };

  (
    $fn_name:ident,
    $archetype_id:expr,
    $resource_struct:ident,
    $arena_struct:ident,
    mesh
    $(, |$gi:ident| $extra:block)?
  ) => {
    #[named]
    pub fn $fn_name(
      &self,
      device: &LogicalDevice,
      vertex_shader: &shader_manager::Shader,
      fragment_shader: &shader_manager::Shader,
      outline_vertex_shader: &shader_manager::Shader,
      outline_fragment_shader: &shader_manager::Shader,
      depth_stencil_format: vk::Format,
      _queue: &Queue,
      color_format: vk::Format,
      _allocator: vk_mem::AllocatorView,
      _discard_pool: &resources::DiscardPool,
      renderpasses: &renderpasses::RenderPasses,
      pipeline_pool_lock: &pipelines::PipelinePool,
      _timeline: u64,
      arena: alloc::sync::Arc<DebugTrackedRwLock<resources::$arena_struct>>,
      rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
    ) -> GpuResult<()> {
      let mut registry = self.registry.write();
      if registry.contains_key(&$archetype_id) {
        return Err(crate::gpu_err_device!());
      }

      let layout = arena.read().pipeline_layout.get();
      let render_pass = renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get();
      let mut graphics_info = GraphicsInfo::default()
        .with_pre_rasterization(PreRasterization::default().with_vertex_module(vertex_shader.module.get()))
        .with_fragment_shader(FragmentShader::default()
          .with_fragment_module(fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
        )
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL);

      $(
        graphics_info = {
          let mut $gi = graphics_info;
          $extra
        };
      )?

      let pipeline_graphics_info = graphics_info.apply_presentation_defaults(color_format, depth_stencil_format, layout, render_pass);
      pipeline_pool_lock.get_or_create_graphics_pipeline(device, &pipeline_graphics_info, rollback)?;
      let pipeline_key = pipeline_graphics_info.pipeline_key();

      let outline_graphics_info = pipeline_graphics_info.clone()
        .with_pre_rasterization(PreRasterization::default().with_vertex_module(outline_vertex_shader.module.get()))
        .with_fragment_shader(FragmentShader::default()
          .with_fragment_module(outline_fragment_shader.module.get())
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
        )
        .with_pipeline_flags(
          pipelines::PipelineFlags::STENCIL_ENABLE
          | pipelines::PipelineFlags::NO_DEPTH_TEST
          | pipelines::PipelineFlags::NO_DEPTH_WRITE
          | pipelines::PipelineFlags::INVERT_FRONT_FACE
        )
        .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
        .with_stencil_compare_op(pipelines::StencilCompareOp::NotEqual)
        .with_stencil_logic_op(pipelines::StencilLogicOp::None)
        .with_stencil_reference(255)
        .with_stencil_compare_mask(255)
        .with_stencil_write_mask(0);
      pipeline_pool_lock.get_or_create_graphics_pipeline(device, &outline_graphics_info, rollback)?;
      let outline_pipeline_key = outline_graphics_info.pipeline_key();
      let res = resources::$resource_struct {
        arena: alloc::sync::Arc::downgrade(&arena),
        pipeline_key,
        graphics_info: pipeline_graphics_info,
        outline_pipeline_key,
        outline_graphics_info
      };

      registry.insert($archetype_id, Box::new(res));

      Ok(())
    }
  };
}

#[allow(unused_mut)]
impl Archetypes {
  impl_create_archetype!(
    create_sun_archetype,
    ArchetypeId::Sun,
    SunRenderResourceArchetype,
    SunRenderResourceArchetypeArena,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::NO_DEPTH_WRITE)
    }
  );
  impl_create_archetype!(
    create_sky_archetype,
    ArchetypeId::Sky,
    SkyRenderResourceArchetype,
    SkyRenderResourceArchetypeArena,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST)
    }
  );
  impl_create_archetype!(
    create_trajectory_archetype,
    ArchetypeId::Trajectory,
    TrajectoryRenderResourceArchetype,
    TrajectoryRenderResourceArchetypeArena,
    ref_alloc,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE)
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
    ArchetypeId::Ui,
    UiRenderResourceArchetype,
    UiRenderResourceArchetypeArena,
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
  impl_create_archetype!(
    create_cursor_archetype,
    ArchetypeId::Cursor,
    CursorRenderResourceArchetype,
    CursorRenderResourceArchetypeArena,
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
    ArchetypeId::Measurement,
    MeasurementRenderResourceArchetype,
    MeasurementRenderResourceArchetypeArena,
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
    ArchetypeId::Marker,
    MarkerRenderResourceArchetype,
    MarkerRenderResourceArchetypeArena,
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
    ArchetypeId::Billboard,
    BillboardRenderResourceArchetype,
    BillboardRenderResourceArchetypeArena,
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
    create_physical_mesh2_archetype,
    ArchetypeId::Mesh,
    ForwardMesh2RenderResourceArchetype,
    ForwardMesh2RenderResourceArchetypeArena,
    mesh
  );
  impl_create_archetype!(
    create_text2_archetype,
    ArchetypeId::Text,
    Text2RenderResourceArchetype,
    Text2RenderResourceArchetypeArena,
    text
  );

  #[named]
  pub fn create_background_archetype(
    &self,
    device: &LogicalDevice,
    vertex_shader: &shader_manager::Shader,
    fragment_shader: &shader_manager::Shader,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    _allocator: vk_mem::AllocatorView,
    _discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool_lock: &pipelines::PipelinePool,
    _timeline: u64,
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::BackgroundRenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut registry = self.registry.write();
    if registry.contains_key(&ArchetypeId::Background) {
      return Err(crate::gpu_err_device!());
    }

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_LIST).clone(),
      )
      .with_pre_rasterization(
        PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
      .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE | PipelineFlags::NO_DEPTH_TEST)
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool_lock.get_or_create_graphics_pipeline(
      device,
      &pipeline_graphics_info,
      rollback,
    )?;

    let res = resources::BackgroundRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info,
    };
    registry.insert(ArchetypeId::Background, Box::new(res));

    Ok(())
  }

  #[named]
  pub fn create_grid_archetype(
    &self,
    device: &LogicalDevice,
    vertex_shader: &shader_manager::Shader,
    fragment_shader: &shader_manager::Shader,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    _allocator: vk_mem::AllocatorView,
    _discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool_lock: &pipelines::PipelinePool,
    _timeline: u64,
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::GridRenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut registry = self.registry.write();
    if registry.contains_key(&ArchetypeId::Grid) {
      return Err(crate::gpu_err_device!());
    }

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_LIST).clone(),
      )
      .with_pre_rasterization(
        PreRasterization::default()
          .with_vertex_module(vertex_shader.module.get())
          .clone(),
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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
      .with_pipeline_flags(PipelineFlags::empty())
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();
    pipeline_pool_lock.get_or_create_graphics_pipeline(
      device,
      &pipeline_graphics_info,
      rollback,
    )?;

    let res = resources::GridRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info,
    };
    registry.insert(ArchetypeId::Grid, Box::new(res));

    Ok(())
  }

  #[named]
  pub fn create_sphere_gizmo_archetype(
    &self,
    device: &LogicalDevice,
    vertex_shader: &shader_manager::Shader,
    fragment_shader: &shader_manager::Shader,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    _allocator: vk_mem::AllocatorView,
    _discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool_lock: &pipelines::PipelinePool,
    _timeline: u64,
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::SphereGizmoRenderResourceArchetypeArena>>,
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut registry = self.registry.write();
    if registry.contains_key(&ArchetypeId::SphereGizmo) {
      return Err(crate::gpu_err_device!());
    }

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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
      .with_render_pass(
        renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get(),
      )
      .with_subpass(0)
      .with_rasterization_polygon_mode(vk::PolygonMode::LINE)
      .clone();

    let pipeline_key = pipeline_graphics_info.pipeline_key();

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      &device,
      &pipeline_graphics_info,
      rollback,
    )?;

    let res = resources::SphereGizmoRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    };
    registry.insert(ArchetypeId::SphereGizmo, Box::new(res));

    Ok(())
  }

  #[named]
  pub fn create_gizmo_archetype(
    &self,
    device: &LogicalDevice,
    vertex_shader: &shader_manager::Shader,
    fragment_shader: &shader_manager::Shader,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    _allocator: vk_mem::AllocatorView,
    _discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool_lock: &pipelines::PipelinePool,
    _timeline: u64,
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::GizmoRenderResourceArchetypeArena>>,
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut registry = self.registry.write();
    if registry.contains_key(&ArchetypeId::Gizmo) {
      return Err(crate::gpu_err_device!());
    }

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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
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

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      &device,
      &pipeline_graphics_info,
      rollback,
    )?;

    let res = resources::GizmoRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    };
    registry.insert(ArchetypeId::Gizmo, Box::new(res));

    Ok(())
  }

  #[named]
  pub fn create_dust_archetype(
    &self,
    device: &LogicalDevice,
    vertex_shader: &shader_manager::Shader,
    fragment_shader: &shader_manager::Shader,
    depth_stencil_format: vk::Format,
    color_format: vk::Format,
    _allocator: vk_mem::AllocatorView,
    _discard_pool: &resources::DiscardPool,
    renderpasses: &renderpasses::RenderPasses,
    pipeline_pool_lock: &pipelines::PipelinePool,
    _timeline: u64,
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::DustRenderArchetypeArena>>,
    rollback: &mut utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut registry = self.registry.write();
    if registry.contains_key(&ArchetypeId::Particles) {
      return Err(crate::gpu_err_device!());
    }
    let layout = arena.read().pipeline_layout.get();
    let render_pass =
      renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get();

    let graphics_info = pipelines::GraphicsInfo::default()
      .with_pre_rasterization(
        PreRasterization::default().with_vertex_module(vertex_shader.module.get()),
      )
      .with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::POINT_LIST))
      .with_rasterization_polygon_mode(vk::PolygonMode::FILL)
      .with_fragment_shader(
        FragmentShader::default()
          .add_viewport(ignored_viewport())
          .add_scissors(ignored_scissor())
          .with_fragment_module(fragment_shader.module.get()),
      )
      .with_pipeline_flags(pipelines::PipelineFlags::NO_DEPTH_WRITE);
    let pipeline_graphics_info = graphics_info.apply_presentation_defaults(
      color_format,
      depth_stencil_format,
      layout,
      render_pass,
    );

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      device,
      &pipeline_graphics_info,
      rollback,
    )?;
    let pipeline_key = pipeline_graphics_info.pipeline_key();

    let res = resources::DustRenderArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info,
    };
    registry.insert(ArchetypeId::Particles, Box::new(res));
    Ok(())
  }
}

fn ignored_viewport() -> vk::Viewport {
  vk::Viewport::default()
}

fn ignored_scissor() -> vk::Rect2D {
  vk::Rect2D::default()
}
