//! archetypes_struct module.

use crate::{
  gpu::{
    PipelineKeyable,
    vulkan::device::{
        LogicalDevice, Queue,
        locks::DebugTrackedRwLock,
        pipelines::{
          FragmentOut, FragmentShader, GraphicsInfo, PipelineFlags, PreRasterization,
          StencilCompareOp, StencilLogicOp, VertexIn,
        },
        renderpasses,
        resources,
        resources::ForwardMeshRenderResourceArchetype,
        shader_manager,
        shader_manager::ShaderKey,
      },
  },
  gpu_backends::vulkan::device::{pipelines, utils},
  simulation::comet::{NORMAL_COMPONENTS, POSITION_COMPONENTS, UV_COMPONENTS},
  types::{GpuError, GpuResult},
};
use ash::vk;
use function_name::named;

// TODO rewrite error messages

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

#[derive(Default)]
/// TODO: Document this item
pub(super) struct Archetypes {
  pub sun_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::SunRenderResourceArchetype>,
  >,
  pub physical_mesh_render_archetype:
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      Option<ForwardMeshRenderResourceArchetype>,
    >,
  pub physical_mesh2_render_archetype:
    crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
      Option<resources::ForwardMesh2RenderResourceArchetype>,
    >,
  pub billboard_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::BillboardRenderResourceArchetype>,
  >,
  pub particle_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::ParticleRenderResourceArchetype>,
  >,
  pub cursor_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::CursorRenderResourceArchetype>,
  >,
  pub marker_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::MarkerRenderResourceArchetype>,
  >,
  pub measurement_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::MeasurementRenderResourceArchetype>,
  >,
  pub sky_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::SkyRenderResourceArchetype>,
  >,
  pub grid_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::GridRenderResourceArchetype>,
  >,
  pub minimap_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::MinimapRenderResourceArchetype>,
  >,
  pub text_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::TextRenderResourceArchetype>,
  >,
  pub text2_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::Text2RenderResourceArchetype>,
  >,
  pub bvh_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::BvhRenderResourceArchetype>,
  >,
  pub bvhwire2_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::Bvhwire2RenderResourceArchetype>,
  >,
  pub sphere_gizmo_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::SphereGizmoRenderResourceArchetype>,
  >,
  pub gizmo_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::GizmoRenderResourceArchetype>,
  >,
  pub particle2_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::Particle2RenderResourceArchetype>,
  >,
  pub trajectory_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::TrajectoryRenderResourceArchetype>,
  >,
  pub ui_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::UiRenderResourceArchetype>,
  >,
  pub background_render_archetype: crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock<
    Option<resources::BackgroundRenderResourceArchetype>,
  >,
}

impl Archetypes {
  /// TODO: Document this item
  #[named]
  pub fn has_discardables(&self) -> bool {
    use crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock;
    DebugTrackedRwLock::read(&self.sun_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.physical_mesh_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.physical_mesh2_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.billboard_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.particle_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.cursor_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.marker_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.measurement_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.sky_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.grid_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.minimap_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.text_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.text2_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.bvh_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.bvhwire2_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.sphere_gizmo_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.ui_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.gizmo_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.trajectory_render_archetype).is_some()
      || DebugTrackedRwLock::read(&self.background_render_archetype).is_some()
  }

  /// TODO: Document this item
  #[named]
  pub fn discard(&self, _device: &ash::Device, _discard_pool: &resources::DiscardPool) {
    use crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock;
    let _ = DebugTrackedRwLock::write(&self.sun_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.physical_mesh_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.physical_mesh2_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.billboard_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.particle_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.particle2_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.cursor_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.marker_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.measurement_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.sky_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.grid_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.minimap_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.text_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.text2_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.bvh_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.bvhwire2_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.sphere_gizmo_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.ui_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.gizmo_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.trajectory_render_archetype).take();
    let _ = DebugTrackedRwLock::write(&self.background_render_archetype).take();
    // Archetype Arenas are discarded by DeviceResources
  }
}

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

macro_rules! impl_update_archetype {
  (
    $prepare_fn:ident,
    $commit_fn:ident,
    $archetype_field:ident
  ) => {
    #[named]
    pub fn $prepare_fn(
      &self,
      color_format: ash::vk::Format,
      renderpasses: &crate::gpu_backends::vulkan::device::renderpasses::RenderPasses,
    ) -> crate::types::GpuResult<Option<PreparedArchetypeUpdate>> {
      let archetype_lock = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &self.$archetype_field,
      );
      let archetype = match archetype_lock.as_ref() {
        Some(a) => a,
        None => return Ok(None),
      };

      let mut graphics_info = archetype.graphics_info.clone();

      if graphics_info.fragment_out.color_attachment_formats.first() != Some(&color_format) {
        let depth_stencil_format =
          graphics_info.fragment_out.depth_attachment_format.unwrap_or(ash::vk::Format::UNDEFINED);

        graphics_info.fragment_out.color_attachment_formats.clear();
        graphics_info.fragment_out.color_attachment_formats.push(color_format);
        graphics_info.render_pass =
          renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get();

        Ok(Some(PreparedArchetypeUpdate {
          main_graphics_info: graphics_info,
          outline_graphics_info: None,
        }))
      } else {
        Ok(None)
      }
    }

    #[named]
    pub fn $commit_fn(&self, data: CompiledArchetypeData) {
      let mut archetype_lock =
        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &self.$archetype_field,
        );
      if let Some(archetype) = archetype_lock.as_mut() {
        archetype.pipeline_key = data.pipeline_key;
        archetype.graphics_info = data.graphics_info;
      }
    }
  };

  (
    $prepare_fn:ident,
    $commit_fn:ident,
    $archetype_field:ident,
    with_outline
  ) => {
    #[named]
    pub fn $prepare_fn(
      &self,
      color_format: ash::vk::Format,
      renderpasses: &crate::gpu_backends::vulkan::device::renderpasses::RenderPasses,
    ) -> crate::types::GpuResult<Option<PreparedArchetypeUpdate>> {
      let archetype_lock = crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(
        &self.$archetype_field,
      );
      let archetype = match archetype_lock.as_ref() {
        Some(a) => a,
        None => return Ok(None),
      };

      let mut needs_update = false;
      let mut graphics_info = archetype.graphics_info.clone();
      let mut outline_graphics_info = archetype.outline_graphics_info.clone();

      if graphics_info.fragment_out.color_attachment_formats.first() != Some(&color_format) {
        let depth_stencil_format =
          graphics_info.fragment_out.depth_attachment_format.unwrap_or(ash::vk::Format::UNDEFINED);
        graphics_info.fragment_out.color_attachment_formats.clear();
        graphics_info.fragment_out.color_attachment_formats.push(color_format);
        graphics_info.render_pass =
          renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get();
        needs_update = true;
      }

      if outline_graphics_info.fragment_out.color_attachment_formats.first() != Some(&color_format)
      {
        let depth_stencil_format = outline_graphics_info
          .fragment_out
          .depth_attachment_format
          .unwrap_or(ash::vk::Format::UNDEFINED);
        outline_graphics_info.fragment_out.color_attachment_formats.clear();
        outline_graphics_info.fragment_out.color_attachment_formats.push(color_format);
        outline_graphics_info.render_pass =
          renderpasses.get_pipeline_render_pass(color_format, depth_stencil_format)?.get();
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

    #[named]
    pub fn $commit_fn(&self, data: CompiledArchetypeData) {
      let mut archetype_lock =
        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::write(
          &self.$archetype_field,
        );
      if let Some(archetype) = archetype_lock.as_mut() {
        archetype.pipeline_key = data.pipeline_key;
        archetype.graphics_info = data.graphics_info;
        if let Some((outline_key, outline_info)) = data.outline_data {
          archetype.outline_pipeline_key = outline_key;
          archetype.outline_graphics_info = outline_info;
        }
      }
    }
  };
}
macro_rules! impl_create_archetype {
  (
    $fn_name:ident,
    $archetype_field:ident,
    $resource_struct:ident,
    $arena_struct:ident,
    ref_alloc
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
      allocator: vk_mem::AllocatorView,
      discard_pool: &resources::DiscardPool,
      renderpasses: &renderpasses::RenderPasses,
      pipeline_pool_lock: &pipelines::PipelinePool,
      timeline: u64,
      arena: alloc::sync::Arc<DebugTrackedRwLock<resources::$arena_struct>>,
      rollback: &mut utils::RollbackContext<'_>,
    ) -> GpuResult<()> {
      let mut archetype_lock = DebugTrackedRwLock::write(&self.$archetype_field);
      if archetype_lock.is_some() {
        return Err(crate::gpu_err_device!());
      }
      let layout = DebugTrackedRwLock::read(&arena).pipeline_layout.get();
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
          $extra
        };
      )?

      let pipeline_graphics_info = graphics_info.apply_presentation_defaults(
        color_format, depth_stencil_format, layout, render_pass
      );

      pipeline_pool_lock.get_or_create_graphics_pipeline(device, &pipeline_graphics_info, rollback)?;
      let pipeline_key = pipeline_graphics_info.pipeline_key();

      let res = resources::$resource_struct { arena: alloc::sync::Arc::downgrade(&arena), pipeline_key, graphics_info: pipeline_graphics_info };
      *archetype_lock = Some(res);

      aethervk_oshal_rlib::log!("archetype {} created", stringify!($archetype_field));

      Ok(())
    }
  };

  (
    $fn_name:ident,
    $archetype_field:ident,
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
      allocator: vk_mem::AllocatorView,
      discard_pool: &resources::DiscardPool,
      renderpasses: &renderpasses::RenderPasses,
      pipeline_pool_lock: &pipelines::PipelinePool,
      timeline: u64,
      arena: alloc::sync::Arc<DebugTrackedRwLock<resources::$arena_struct>>,
      rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
    ) -> GpuResult<()> {
      let mut archetype_lock = DebugTrackedRwLock::write(&self.$archetype_field);
      if archetype_lock.is_some() {
        return Err(crate::gpu_err_device!());
      }
      let layout = DebugTrackedRwLock::read(&arena).pipeline_layout.get();
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
          $extra
        };
      )?

      let pipeline_graphics_info = graphics_info.apply_presentation_defaults(
        color_format, depth_stencil_format, layout, render_pass
      );

      pipeline_pool_lock.get_or_create_graphics_pipeline(device, &pipeline_graphics_info, rollback)?;
      let pipeline_key = pipeline_graphics_info.pipeline_key();

      let res = resources::$resource_struct { arena: alloc::sync::Arc::downgrade(&arena), pipeline_key, graphics_info: pipeline_graphics_info };
      *archetype_lock = Some(res);

      aethervk_oshal_rlib::log!("archetype {} created", stringify!($archetype_field));

      Ok(())
    }
  };
}

impl Archetypes {
  impl_update_archetype!(
    prepare_update_physical_mesh_archetype,
    commit_update_physical_mesh_archetype,
    physical_mesh_render_archetype,
    with_outline
  );

  impl_update_archetype!(
    prepare_update_physical_mesh2_archetype,
    commit_update_physical_mesh2_archetype,
    physical_mesh2_render_archetype,
    with_outline
  );

  impl_update_archetype!(
    prepare_update_cursor_archetype,
    commit_update_cursor_archetype,
    cursor_render_archetype
  );
  impl_update_archetype!(
    prepare_update_particle_archetype,
    commit_update_particle_archetype,
    particle_render_archetype
  );
  impl_update_archetype!(
    prepare_update_particle2_archetype,
    commit_update_particle2_archetype,
    particle2_render_archetype
  );
  impl_update_archetype!(
    prepare_update_sun_archetype,
    commit_update_sun_archetype,
    sun_render_archetype
  );
  impl_update_archetype!(
    prepare_update_sky_archetype,
    commit_update_sky_archetype,
    sky_render_archetype
  );

  impl_update_archetype!(
    prepare_update_grid_archetype,
    commit_update_grid_archetype,
    grid_render_archetype
  );

  impl_update_archetype!(
    prepare_update_minimap_archetype,
    commit_update_minimap_archetype,
    minimap_render_archetype
  );

  impl_update_archetype!(
    prepare_update_text_archetype,
    commit_update_text_archetype,
    text_render_archetype
  );

  impl_update_archetype!(
    prepare_update_text2_archetype,
    commit_update_text2_archetype,
    text2_render_archetype
  );

  impl_update_archetype!(
    prepare_update_bvh_archetype,
    commit_update_bvh_archetype,
    bvh_render_archetype
  );

  impl_update_archetype!(
    prepare_update_bvhwire2_archetype,
    commit_update_bvhwire2_archetype,
    bvhwire2_render_archetype
  );

  impl_update_archetype!(
    prepare_update_sphere_gizmo_archetype,
    commit_update_sphere_gizmo_archetype,
    sphere_gizmo_render_archetype
  );

  impl_update_archetype!(
    prepare_update_gizmo_archetype,
    commit_update_gizmo_archetype,
    gizmo_render_archetype
  );

  impl_update_archetype!(
    prepare_update_measurement_archetype,
    commit_update_measurement_archetype,
    measurement_render_archetype
  );

  impl_update_archetype!(
    prepare_update_marker_archetype,
    commit_update_marker_archetype,
    marker_render_archetype
  );

  impl_update_archetype!(
    prepare_update_billboard_archetype,
    commit_update_billboard_archetype,
    billboard_render_archetype
  );

  impl_update_archetype!(
    prepare_update_trajectory_archetype,
    commit_update_trajectory_archetype,
    trajectory_render_archetype
  );

  impl_update_archetype!(
    prepare_update_ui_archetype,
    commit_update_ui_archetype,
    ui_render_archetype
  );

  impl_update_archetype!(
    prepare_update_background_archetype,
    commit_update_background_archetype,
    background_render_archetype
  );

  // ------------------------------------ Creation ------------------------------------

  impl_create_archetype!(
    create_sun_archetype,
    sun_render_archetype,
    SunRenderResourceArchetype,
    SunRenderResourceArchetypeArena,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::INVERT_FRONT_FACE | PipelineFlags::NO_DEPTH_WRITE)
    }
  );

  impl_create_archetype!(
    create_sky_archetype,
    sky_render_archetype,
    SkyRenderResourceArchetype,
    SkyRenderResourceArchetypeArena,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE)
    }
  );

  impl_create_archetype!(
    create_particle_archetype,
    particle_render_archetype,
    ParticleRenderResourceArchetype,
    ParticleRenderResourceArchetypeArena,
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
    Particle2RenderResourceArchetypeArena,
    ref_alloc,
    |gi| {
      gi.with_vertex_in(VertexIn::default().with_topology(vk::PrimitiveTopology::TRIANGLE_STRIP))
        .with_pipeline_flags(PipelineFlags::NO_DEPTH_WRITE)
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
    TrajectoryRenderResourceArchetypeArena,
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
    let mut bg_render_archetype = DebugTrackedRwLock::write(&self.background_render_archetype);
    if bg_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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

    *bg_render_archetype = Some(resources::BackgroundRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info,
    });

    Ok(())
  }

  impl_create_archetype!(
    create_cursor_archetype,
    cursor_render_archetype,
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
    measurement_render_archetype,
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
    marker_render_archetype,
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
    billboard_render_archetype,
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

  /// TODO: Document this item
  #[named]
  pub fn create_physical_mesh_archetype(
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
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::ForwardMeshRenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    if DebugTrackedRwLock::read(&self.physical_mesh_render_archetype).is_some() {
      return Err(crate::gpu_err_device!());
    }

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
          .add_binding(
            0,
            POSITION_COMPONENTS * core::mem::size_of::<f32>() as u32,
            vk::VertexInputRate::VERTEX,
          )
          .add_binding(
            1,
            9 * core::mem::size_of::<f32>() as u32,
            vk::VertexInputRate::VERTEX,
          )
          .add_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0) // inPosition
          .add_attribute(1, 1, vk::Format::R32G32B32_SFLOAT, 0) // inNormal
          .add_attribute(
            1,
            2,
            vk::Format::R32G32_SFLOAT,
            NORMAL_COMPONENTS * core::mem::size_of::<f32>() as u32,
          ) // inUV
          .add_attribute(
            1,
            3,
            vk::Format::R32G32B32A32_SFLOAT,
            (NORMAL_COMPONENTS + UV_COMPONENTS) * core::mem::size_of::<f32>() as u32,
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
      .with_pipeline_layout(
        crate::gpu_backends::vulkan::device::locks::DebugTrackedRwLock::read(&arena)
          .pipeline_layout
          .get(),
      )
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

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      device,
      &pipeline_graphics_info,
      rollback,
    )?;
    let pipeline_key = pipeline_graphics_info.pipeline_key();

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

    pipeline_pool_lock.get_or_create_graphics_pipeline(device, &outline_graphics_info, rollback)?;
    let outline_pipeline_key = outline_graphics_info.pipeline_key();

    *DebugTrackedRwLock::write(&self.physical_mesh_render_archetype) =
      Some(resources::ForwardMeshRenderResourceArchetype {
        arena: alloc::sync::Arc::downgrade(&arena),
        pipeline_key,
        graphics_info: pipeline_graphics_info,
        outline_pipeline_key,
        outline_graphics_info,
      });

    Ok(())
  }

  /// TODO: Document this item
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
    let mut grid_render_archetype = DebugTrackedRwLock::write(&self.grid_render_archetype);
    if grid_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        // Single full-screen triangle: TRIANGLE_LIST with 3 verts, no strip seam.
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

    *grid_render_archetype = Some(resources::GridRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info,
    });

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_minimap_archetype(
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
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::MinimapRenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut minimap_render_archetype = DebugTrackedRwLock::write(&self.minimap_render_archetype);
    if minimap_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
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

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      &device,
      &pipeline_graphics_info,
      rollback,
    )?;

    *minimap_render_archetype = Some(resources::MinimapRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    });

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_text_archetype(
    &self,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
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
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::TextRenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut text_render_archetype = DebugTrackedRwLock::write(&self.text_render_archetype);
    if text_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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

    *text_render_archetype = Some(resources::TextRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    });

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_text2_archetype(
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
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::Text2RenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut text2_render_archetype = DebugTrackedRwLock::write(&self.text2_render_archetype);
    if text2_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
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
    pipeline_pool_lock.get_or_create_graphics_pipeline(
      device,
      &pipeline_graphics_info,
      rollback,
    )?;

    *text2_render_archetype = Some(resources::Text2RenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    });

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_bvh_archetype(
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
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::BvhRenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut bvh_render_archetype = DebugTrackedRwLock::write(&self.bvh_render_archetype);
    if bvh_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
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

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      &device,
      &pipeline_graphics_info,
      rollback,
    )?;

    *bvh_render_archetype = Some(resources::BvhRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    });

    Ok(())
  }

  /// TODO: Document this item
  #[named]
  pub fn create_bvhwire2_archetype(
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
    arena: alloc::sync::Arc<DebugTrackedRwLock<resources::Bvhwire2RenderResourceArchetypeArena>>,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut bvhwire2_render_archetype = DebugTrackedRwLock::write(&self.bvhwire2_render_archetype);
    if bvhwire2_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
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

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      &device,
      &pipeline_graphics_info,
      rollback,
    )?;

    *bvhwire2_render_archetype = Some(resources::Bvhwire2RenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    });

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
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut sphere_gizmo_render_archetype =
      DebugTrackedRwLock::write(&self.sphere_gizmo_render_archetype);
    if sphere_gizmo_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
      .with_pipeline_flags(
        pipelines::PipelineFlags::NO_DEPTH_WRITE | pipelines::PipelineFlags::NO_DEPTH_TEST,
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

    *sphere_gizmo_render_archetype = Some(resources::SphereGizmoRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    });

    Ok(())
  }

  /// TODO: Document this item
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
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    let mut gizmo_render_archetype = DebugTrackedRwLock::write(&self.gizmo_render_archetype);
    if gizmo_render_archetype.is_some() {
      return Err(crate::gpu_err_device!());
    }

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

    *gizmo_render_archetype = Some(resources::GizmoRenderResourceArchetype {
      arena: alloc::sync::Arc::downgrade(&arena),
      pipeline_key,
      graphics_info: pipeline_graphics_info.clone(),
    });

    Ok(())
  }

  #[named]
  pub fn create_physical_mesh2_archetype(
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
    arena: alloc::sync::Arc<
      DebugTrackedRwLock<resources::ForwardMesh2RenderResourceArchetypeArena>,
    >,
    rollback: &mut crate::gpu_backends::vulkan::utils::RollbackContext<'_>,
  ) -> GpuResult<()> {
    if DebugTrackedRwLock::read(&self.physical_mesh2_render_archetype).is_some() {
      return Err(crate::gpu_err_device!());
    }

    let pipeline_graphics_info = GraphicsInfo::default()
      .with_vertex_in(
        VertexIn::default()
          .with_topology(vk::PrimitiveTopology::TRIANGLE_LIST)
          .add_binding(
            0,
            POSITION_COMPONENTS * core::mem::size_of::<f32>() as u32,
            vk::VertexInputRate::VERTEX,
          )
          .add_binding(
            1,
            9 * core::mem::size_of::<f32>() as u32,
            vk::VertexInputRate::VERTEX,
          )
          .add_attribute(0, 0, vk::Format::R32G32B32_SFLOAT, 0) // inPosition
          .add_attribute(1, 1, vk::Format::R32G32B32_SFLOAT, 0) // inNormal
          .add_attribute(
            1,
            2,
            vk::Format::R32G32_SFLOAT,
            NORMAL_COMPONENTS * core::mem::size_of::<f32>() as u32,
          ) // inUV
          .add_attribute(
            1,
            3,
            vk::Format::R32G32B32A32_SFLOAT,
            (NORMAL_COMPONENTS + UV_COMPONENTS) * core::mem::size_of::<f32>() as u32,
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
      .with_pipeline_layout(DebugTrackedRwLock::read(&arena).pipeline_layout.get())
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

    pipeline_pool_lock.get_or_create_graphics_pipeline(
      device,
      &pipeline_graphics_info,
      rollback,
    )?;
    let pipeline_key = pipeline_graphics_info.pipeline_key();

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

    pipeline_pool_lock.get_or_create_graphics_pipeline(device, &outline_graphics_info, rollback)?;
    let outline_pipeline_key = outline_graphics_info.pipeline_key();

    *DebugTrackedRwLock::write(&self.physical_mesh2_render_archetype) =
      Some(resources::ForwardMesh2RenderResourceArchetype {
        arena: alloc::sync::Arc::downgrade(&arena),
        pipeline_key,
        graphics_info: pipeline_graphics_info,
        outline_pipeline_key,
        outline_graphics_info,
      });

    Ok(())
  }
}

fn ignored_viewport() -> vk::Viewport {
  vk::Viewport::default()
}

fn ignored_scissor() -> vk::Rect2D {
  vk::Rect2D::default()
}
