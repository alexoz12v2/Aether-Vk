//! pipelines module.

use aethervk_oshal_rlib::hash::FnvHasher;
use alloc::vec::Vec;
use ash::vk;
use bitflags::bitflags;
use core::{
  ffi,
  hash::{Hash, Hasher},
  ptr,
};
use hashbrown::HashMap;

use crate::gpu::vulkan::device::swapchain;
use crate::{
  gpu::{PipelineKey, PipelineKeyable},
  gpu_backends::vulkan::{
    device::{DeviceResource, resources::DiscardPool},
    utils::NonZeroHandle,
  },
  types::GpuResult,
};

// ---------------- COMPUTE PIPELINE HASH ------------------------------------
fn eq_specialization_constants(
  a: &vk::SpecializationMapEntry,
  b: &vk::SpecializationMapEntry,
) -> bool {
  a.size == b.size && a.offset == b.offset && a.constant_id == b.constant_id
}

/// TODO: Document this item
pub struct ComputeInfo {
  pub shader_module: vk::ShaderModule,
  pub pipeline_layout: vk::PipelineLayout,
  pub specialization_constants: Vec<vk::SpecializationMapEntry>,
  pub specialization_constants_values: Vec<u8>,
}

impl Default for ComputeInfo {
  fn default() -> Self {
    Self {
      shader_module: Default::default(),
      pipeline_layout: Default::default(),
      specialization_constants: Vec::with_capacity(8),
      specialization_constants_values: Vec::with_capacity(64),
    }
  }
}

impl ComputeInfo {
  /// TODO: Document this item
  pub fn with_shader_module(&mut self, shader_module: vk::ShaderModule) -> &mut Self {
    self.shader_module = shader_module;
    self
  }

  /// TODO: Document this item
  pub fn with_pipeline_layout(&mut self, pipeline_layout: vk::PipelineLayout) -> &mut Self {
    self.pipeline_layout = pipeline_layout;
    self
  }

  /// TODO: Document this item
  pub fn add_specialization_constant_u32(
    &mut self,
    mut constant: vk::SpecializationMapEntry,
    value: u32,
  ) -> &mut Self {
    constant.offset = self.specialization_constants_values.len() as u32;
    self.specialization_constants.push(constant);
    // little endian right?
    for b in value.to_le_bytes() {
      self.specialization_constants_values.push(b);
    }
    self
  }
}

impl PartialEq for ComputeInfo {
  fn eq(&self, other: &Self) -> bool {
    let result = self.shader_module == other.shader_module
      && self.pipeline_layout == other.pipeline_layout
      && self.specialization_constants.len() == other.specialization_constants.len();
    if !result {
      return false;
    }

    for i in 0..self.specialization_constants.len() {
      if unsafe {
        !eq_specialization_constants(
          self.specialization_constants.get_unchecked(i),
          other.specialization_constants.get_unchecked(i),
        )
      } {
        return false;
      }
    }

    true
  }
}
impl Eq for ComputeInfo {}

impl Hash for ComputeInfo {
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    self.shader_module.hash(state);
    self.pipeline_layout.hash(state);
    for specialization_constant_entry in &self.specialization_constants {
      specialization_constant_entry.constant_id.hash(state);
      specialization_constant_entry.offset.hash(state);
      specialization_constant_entry.size.hash(state);
    }

    self.specialization_constants_values.hash(state);
  }
}

// ---------------- GRAPHICS PIPELINE HASH -----------------------------------
bitflags! {
  #[derive(PartialEq, Eq, Hash, Default, Clone, Copy)]
  /// TODO: Document this item
  pub struct PipelineFlags: u32 {
    const DEPTH_BIAS = 1 << 0;
    const CULL_FRONT = 1 << 1;
    const CULL_BACK  = 1 << 2;
    const CULL_ALL   = Self::CULL_FRONT.bits() | Self::CULL_BACK.bits();
    const INVERT_FRONT_FACE = 1 << 3;
    const NO_DEPTH_WRITE = 1 << 4;
    const STENCIL_ENABLE = 1 << 5;
    const NO_DEPTH_TEST = 1 << 6;
    const NO_LINE_DYNAMIC_STATE = 1 << 7;
  }
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// TODO: Document this item
pub enum StencilCompareOp {
  None = 0,
  Equal,
  NotEqual,
  Always,
}

impl Default for StencilCompareOp {
  fn default() -> Self {
    Self::None
  }
}

#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// TODO: Document this item
pub enum StencilLogicOp {
  // synonym for KEEP in both back and front always
  None = 0,
  // replace on pass (visibility mask and identification of visible objects)
  Replace,
  // depth pass for z-pass shadow volumne counting
  // decrement on front, increment on back
  CountDepthPass,
  // depth pass for Carmack's reverse algorithm
  // count crossings on depth test fail. inc/dec -> front/back
  // https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-11-efficient-and-robust-shadow-volumes-using
  CountDepthFail,
}

impl Default for StencilLogicOp {
  fn default() -> Self {
    Self::None
  }
}

fn eq_vertex_input_attribute_description(
  a: &vk::VertexInputAttributeDescription,
  b: &vk::VertexInputAttributeDescription,
) -> bool {
  a.binding == b.binding && a.format == b.format && a.location == b.location && a.offset == b.offset
}

#[derive(Clone)]
/// TODO: Document this item
pub(super) struct VertexIn {
  // adjacency requires `geometryShader` feature, patch requires `tessellationShader` feature
  pub topology: vk::PrimitiveTopology,
  // defined what's inside each binding, which can be split in multiple
  // locations (globally unique, monotonically increasing) if you need multple
  // pieces of data having a VkFormat
  pub attributes: Vec<vk::VertexInputAttributeDescription>,
  // define how big a single vertex binding or a single instance binding and
  // how it's stepped through memory. *no information of what's inside it*
  pub bindings: Vec<vk::VertexInputBindingDescription>,
}

impl VertexIn {
  /// TODO: Document this item
  pub fn with_topology(mut self, topology: vk::PrimitiveTopology) -> Self {
    self.topology = topology;
    self
  }

  /// TODO: Document this item
  pub fn add_attribute(
    mut self,
    binding: u32,
    location: u32,
    format: vk::Format,
    offset: u32,
  ) -> Self {
    self.attributes.push(
      vk::VertexInputAttributeDescription::default()
        .binding(binding)
        .location(location)
        .format(format)
        .offset(offset),
    );
    self
  }

  /// TODO: Document this item
  pub fn add_binding(mut self, binding: u32, stride: u32, input_rate: vk::VertexInputRate) -> Self {
    self.bindings.push(
      vk::VertexInputBindingDescription::default()
        .binding(binding)
        .input_rate(input_rate)
        .stride(stride),
    );
    self
  }
}

impl Default for VertexIn {
  fn default() -> Self {
    Self {
      topology: Default::default(),
      attributes: Vec::with_capacity(8),
      bindings: Vec::with_capacity(8),
    }
  }
}

fn vertex_input_attribute_description_eq(
  a: &vk::VertexInputAttributeDescription,
  b: &vk::VertexInputAttributeDescription,
) -> bool {
  a.location == b.location && a.binding == b.binding && a.format == b.format && a.offset == b.offset
}

fn vertex_input_binding_description_eq(
  a: &vk::VertexInputBindingDescription,
  b: &vk::VertexInputBindingDescription,
) -> bool {
  a.binding == b.binding && a.input_rate == b.input_rate && a.stride == b.stride
}

impl PartialEq for VertexIn {
  fn eq(&self, other: &Self) -> bool {
    self.topology == other.topology
      && self
        .attributes
        .iter()
        .zip(other.attributes.iter())
        .map(|(a, b)| vertex_input_attribute_description_eq(a, b))
        .reduce(|acc, x| acc && x)
        .unwrap_or(true)
      && self
        .bindings
        .iter()
        .zip(other.bindings.iter())
        .map(|(a, b)| vertex_input_binding_description_eq(a, b))
        .reduce(|acc, x| acc && x)
        .unwrap_or(true)
  }
}

impl Eq for VertexIn {}

impl Hash for VertexIn {
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    self.topology.hash(state);
    for attribute in &self.attributes {
      attribute.binding.hash(state);
      attribute.format.hash(state);
      attribute.location.hash(state);
      attribute.offset.hash(state);
    }
    for binding in &self.bindings {
      binding.binding.hash(state);
      binding.input_rate.hash(state);
      binding.stride.hash(state);
    }
  }
}

/// Note: `geometryShader` has been omitted because of they poor/lack of support from
/// tile-based GPUs (like Apple Silicon ones)
/// `tessellationShader` isn't in our interest for now
#[derive(PartialEq, Eq, Hash, Default, Clone)]
pub(super) struct PreRasterization {
  pub vertex_module: vk::ShaderModule,
}

impl PreRasterization {
  /// TODO: Document this item
  pub fn with_vertex_module(mut self, vertex_module: vk::ShaderModule) -> Self {
    self.vertex_module = vertex_module;
    self
  }
}

#[derive(Clone)]
/// TODO: Document this item
pub(super) struct FragmentShader {
  pub fragment_module: vk::ShaderModule,
  pub viewports: Vec<vk::Viewport>,
  pub scissors: Vec<vk::Rect2D>,
}

impl FragmentShader {
  /// TODO: Document this item
  pub fn with_fragment_module(mut self, fragment_module: vk::ShaderModule) -> Self {
    self.fragment_module = fragment_module;
    self
  }
  /// TODO: Document this item
  pub fn add_viewport(mut self, viewport: vk::Viewport) -> Self {
    self.viewports.push(viewport);
    self
  }
  /// TODO: Document this item
  pub fn add_scissors(mut self, scissors: vk::Rect2D) -> Self {
    self.scissors.push(scissors);
    self
  }
}

impl Default for FragmentShader {
  fn default() -> Self {
    Self {
      fragment_module: Default::default(),
      viewports: Vec::with_capacity(8),
      scissors: Vec::with_capacity(8),
    }
  }
}

impl PartialEq for FragmentShader {
  fn eq(&self, other: &Self) -> bool {
    self.fragment_module == other.fragment_module
      && self.viewports.len() == other.viewports.len()
      && self.scissors.len() == other.scissors.len()
  }
}

impl Eq for FragmentShader {}

impl Hash for FragmentShader {
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    self.fragment_module.hash(state);
    self.viewports.len().hash(state);
    self.scissors.len().hash(state);
  }
}

#[derive(PartialEq, Eq, Hash, Default, Clone)]
/// TODO: Document this item
pub(super) struct FragmentOut {
  pub color_attachment_formats: Vec<vk::Format>,
  pub depth_attachment_format: Option<vk::Format>,
  pub stencil_attachment_format: Option<vk::Format>,
}

impl FragmentOut {
  /// TODO: Document this item
  pub fn add_color_attachment_format(mut self, format: vk::Format) -> Self {
    self.color_attachment_formats.push(format);
    self
  }

  /// TODO: Document this item
  pub fn with_depth_attachment_format(mut self, format: vk::Format) -> Self {
    self.depth_attachment_format = Some(format);
    self
  }

  /// TODO: Document this item
  pub fn with_stencil_attachment_format(mut self, format: vk::Format) -> Self {
    self.stencil_attachment_format = Some(format);
    self
  }
}

#[derive(Default, Clone)]
/// TODO: Document this item
pub struct GraphicsInfo {
  pub specialization_constants: Vec<vk::SpecializationMapEntry>,
  pub specialization_constants_values: Vec<u8>,
  pub vertex_in: VertexIn,
  pub pre_rasterization: PreRasterization,
  pub fragment_shader: FragmentShader,
  pub fragment_out: FragmentOut,
  pub pipeline_layout: vk::PipelineLayout,
  pub pipeline_flags: PipelineFlags,
  pub render_pass: vk::RenderPass,
  pub subpass: u32,
  pub rasterization_polygon_mode: vk::PolygonMode,
  pub stencil_compare_op: StencilCompareOp,
  pub stencil_logic_op: StencilLogicOp,
  pub stencil_reference: u32,
  pub stencil_compare_mask: u32,
  pub stencil_write_mask: u32,
}

impl PartialEq for GraphicsInfo {
  fn eq(&self, other: &Self) -> bool {
    for i in 0..self.specialization_constants.len() {
      if unsafe {
        !eq_specialization_constants(
          self.specialization_constants.get_unchecked(i),
          other.specialization_constants.get_unchecked(i),
        )
      } {
        return false;
      }
    }

    self.vertex_in == other.vertex_in
      && self.pre_rasterization == other.pre_rasterization
      && self.fragment_shader == other.fragment_shader
      && self.fragment_out == other.fragment_out
      && self.pipeline_layout == other.pipeline_layout
      && self.pipeline_flags == other.pipeline_flags
      && self.render_pass == other.render_pass
      && self.subpass == other.subpass
      && self.rasterization_polygon_mode == other.rasterization_polygon_mode
      && self.stencil_compare_op == other.stencil_compare_op
      && self.stencil_logic_op == other.stencil_logic_op
      && self.stencil_reference == other.stencil_reference
      && self.stencil_compare_mask == other.stencil_compare_mask
      && self.stencil_write_mask == other.stencil_write_mask
  }
}

impl Eq for GraphicsInfo {}

impl Hash for GraphicsInfo {
  fn hash<H: Hasher>(&self, state: &mut H) {
    for specialization_constant_entry in &self.specialization_constants {
      specialization_constant_entry.constant_id.hash(state);
      specialization_constant_entry.offset.hash(state);
      specialization_constant_entry.size.hash(state);
    }
    self.specialization_constants_values.hash(state);

    self.vertex_in.hash(state);
    self.pre_rasterization.hash(state);
    self.fragment_shader.hash(state);
    self.fragment_out.hash(state);
    self.pipeline_layout.hash(state);
    self.pipeline_flags.hash(state);
    self.render_pass.hash(state);
    self.subpass.hash(state);
    self.rasterization_polygon_mode.hash(state);
    self.stencil_compare_op.hash(state);
    self.stencil_logic_op.hash(state);
    self.stencil_reference.hash(state);
    self.stencil_compare_mask.hash(state);
    self.stencil_write_mask.hash(state);
  }
}

impl GraphicsInfo {
  // TODO: Add methods to diminish code duplication in [`super::archetypes_struct`]
  /// TODO: Document this item
  pub fn apply_presentation_defaults(
    mut self,
    color_format: vk::Format,
    depth_format: vk::Format,
    layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
  ) -> Self {
    if self.fragment_shader.viewports.is_empty() {
      self.fragment_shader.viewports.push(vk::Viewport::default());
    }
    if self.fragment_shader.scissors.is_empty() {
      self.fragment_shader.scissors.push(vk::Rect2D::default());
    }

    self.fragment_out.color_attachment_formats.push(color_format);
    self.fragment_out.depth_attachment_format = Some(depth_format);
    if self.pipeline_flags.contains(PipelineFlags::STENCIL_ENABLE) {
      self.fragment_out.stencil_attachment_format = Some(depth_format);
    }

    self.pipeline_layout = layout;
    self.render_pass = render_pass;
    self.subpass = 0;
    self
  }

  /// TODO: Document this item
  pub fn add_specialization_constant_u32(
    mut self,
    mut constant: vk::SpecializationMapEntry,
    value: u32,
  ) -> Self {
    constant.offset = self.specialization_constants_values.len() as u32;
    self.specialization_constants.push(constant);
    // little endian right? TODO: query endianness of OS (oshal)
    for b in value.to_le_bytes() {
      self.specialization_constants_values.push(b);
    }
    self
  }

  /// TODO: Document this item
  pub fn with_vertex_in(mut self, vertex_in: VertexIn) -> Self {
    self.vertex_in = vertex_in;
    self
  }

  /// TODO: Document this item
  pub fn with_pre_rasterization(mut self, pre_rasterization: PreRasterization) -> Self {
    self.pre_rasterization = pre_rasterization;
    self
  }

  /// TODO: Document this item
  pub fn with_fragment_shader(mut self, fragment_shader: FragmentShader) -> Self {
    self.fragment_shader = fragment_shader;
    self
  }

  /// TODO: Document this item
  pub fn with_fragment_out(mut self, fragment_out: FragmentOut) -> Self {
    self.fragment_out = fragment_out;
    self
  }

  /// TODO: Document this item
  pub fn with_pipeline_layout(mut self, pipeline_layout: vk::PipelineLayout) -> Self {
    self.pipeline_layout = pipeline_layout;
    self
  }

  /// TODO: Document this item
  pub fn with_pipeline_flags(mut self, pipeline_flags: PipelineFlags) -> Self {
    self.pipeline_flags = pipeline_flags;
    self
  }

  /// TODO: Document this item
  pub fn with_render_pass(mut self, render_pass: vk::RenderPass) -> Self {
    self.render_pass = render_pass;
    self
  }

  /// TODO: Document this item
  pub fn with_subpass(mut self, subpass: u32) -> Self {
    self.subpass = subpass;
    self
  }

  /// TODO: Document this item
  pub fn with_rasterization_polygon_mode(
    mut self,
    rasterization_polygon_mode: vk::PolygonMode,
  ) -> Self {
    self.rasterization_polygon_mode = rasterization_polygon_mode;
    self
  }

  /// TODO: Document this item
  pub fn with_stencil_compare_op(mut self, stencil_compare_op: StencilCompareOp) -> Self {
    self.stencil_compare_op = stencil_compare_op;
    self
  }

  /// TODO: Document this item
  pub fn with_stencil_logic_op(mut self, stencil_logic_op: StencilLogicOp) -> Self {
    self.stencil_logic_op = stencil_logic_op;
    self
  }

  /// TODO: Document this item
  pub fn with_stencil_reference(mut self, stencil_reference: u32) -> Self {
    self.stencil_reference = stencil_reference;
    self
  }

  /// TODO: Document this item
  pub fn with_stencil_compare_mask(mut self, stencil_compare_mask: u32) -> Self {
    self.stencil_compare_mask = stencil_compare_mask;
    self
  }

  /// TODO: Document this item
  pub fn with_stencil_write_mask(mut self, stencil_write_mask: u32) -> Self {
    self.stencil_write_mask = stencil_write_mask;
    self
  }
}

impl PipelineKeyable for ComputeInfo {
  fn pipeline_key(&self) -> PipelineKey {
    let mut hasher = FnvHasher::new();
    self.hash(&mut hasher);
    PipelineKey(hasher.finish())
  }
}

impl PipelineKeyable for GraphicsInfo {
  fn pipeline_key(&self) -> PipelineKey {
    let mut hasher = FnvHasher::new();
    self.hash(&mut hasher);
    PipelineKey(hasher.finish())
  }
}

#[ouroboros::self_referencing]
struct RawComputeInfo<'a> {
  // specs
  compute_info: &'a ComputeInfo,
  // referenced structs
  specialization_info: vk::SpecializationInfo<'a>,
  entry_name: alloc::ffi::CString,
  // main struct
  #[borrows(specialization_info, entry_name)]
  #[covariant]
  compute_pipeline_create_info: vk::ComputePipelineCreateInfo<'this>,
}

impl<'a> From<&'a ComputeInfo> for RawComputeInfo<'a> {
  fn from(compute_info: &'a ComputeInfo) -> Self {
    let mut specialization_info = vk::SpecializationInfo::default();
    if !compute_info.specialization_constants.is_empty() {
      specialization_info = specialization_info
        .map_entries(&compute_info.specialization_constants)
        .data(&compute_info.specialization_constants_values);
    }

    RawComputeInfoBuilder {
      compute_info,
      specialization_info,
      entry_name: alloc::ffi::CString::new("main").unwrap(),
      compute_pipeline_create_info_builder: |specialization_info: &_, entry_name: &_| {
        let stage = vk::PipelineShaderStageCreateInfo::default()
          .module(compute_info.shader_module)
          .stage(vk::ShaderStageFlags::COMPUTE)
          .name(entry_name)
          .specialization_info(specialization_info);

        vk::ComputePipelineCreateInfo::default()
          .base_pipeline_handle(vk::Pipeline::null())
          .base_pipeline_index(-1)
          .layout(compute_info.pipeline_layout)
          .stage(stage)
      },
    }
    .build()
  }
}

#[ouroboros::self_referencing]
struct RawGraphicsInfo<'a> {
  // the rest is in the spec
  graphics_info: &'a GraphicsInfo,
  // referenced resources
  dynamic_states: Vec<vk::DynamicState>,
  color_blend_attachments: Vec<vk::PipelineColorBlendAttachmentState>,
  // components
  #[borrows(graphics_info)]
  #[covariant]
  pipeline_shader_stage_create_infos: Vec<vk::PipelineShaderStageCreateInfo<'this>>,
  #[borrows(graphics_info)]
  #[covariant]
  vertex_input_state: vk::PipelineVertexInputStateCreateInfo<'this>,
  #[borrows(graphics_info)]
  #[covariant]
  input_assembly_state: vk::PipelineInputAssemblyStateCreateInfo<'this>,
  #[borrows(graphics_info)]
  #[covariant]
  tessellation_state: vk::PipelineTessellationStateCreateInfo<'this>,
  #[borrows(graphics_info)]
  #[covariant]
  viewport_state: vk::PipelineViewportStateCreateInfo<'this>,
  #[borrows(graphics_info)]
  #[covariant]
  rasterization_state: vk::PipelineRasterizationStateCreateInfo<'this>,
  #[borrows(graphics_info)]
  #[covariant]
  multisample_state: vk::PipelineMultisampleStateCreateInfo<'this>,
  #[borrows(graphics_info)]
  #[covariant]
  depth_stencil_state: vk::PipelineDepthStencilStateCreateInfo<'this>,
  #[borrows(graphics_info, color_blend_attachments)]
  #[covariant]
  color_blend_state: vk::PipelineColorBlendStateCreateInfo<'this>,
  #[borrows(graphics_info, dynamic_states)]
  #[covariant]
  dynamic_state: vk::PipelineDynamicStateCreateInfo<'this>,
  layout: vk::PipelineLayout,
  render_pass: vk::RenderPass,
  subpass: u32,
  // main struct
  #[borrows(
    pipeline_shader_stage_create_infos,
    vertex_input_state,
    input_assembly_state,
    tessellation_state,
    viewport_state,
    rasterization_state,
    multisample_state,
    depth_stencil_state,
    color_blend_state,
    dynamic_state,
    layout,
    render_pass,
    subpass
  )]
  #[covariant]
  graphics_pipeline_create_info: vk::GraphicsPipelineCreateInfo<'this>,
}

impl<'a> From<&'a GraphicsInfo> for RawGraphicsInfo<'a> {
  fn from(graphics_info: &'a GraphicsInfo) -> Self {
    let mut color_blend_attachments = Vec::with_capacity(
      if graphics_info.fragment_out.color_attachment_formats.is_empty() {
        1
      } else {
        graphics_info.fragment_out.color_attachment_formats.len()
      },
    );
    for _ in 0..color_blend_attachments.capacity() {
      // over operator
      color_blend_attachments.push(
        vk::PipelineColorBlendAttachmentState::default()
          .color_write_mask(vk::ColorComponentFlags::RGBA)
          .blend_enable(true)
          .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
          .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
          .color_blend_op(vk::BlendOp::ADD)
          .src_alpha_blend_factor(vk::BlendFactor::ONE)
          .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
          .alpha_blend_op(vk::BlendOp::ADD),
      );
    }

    let mut dynamic_states = Vec::with_capacity(3);
    dynamic_states.push(vk::DynamicState::VIEWPORT);
    dynamic_states.push(vk::DynamicState::SCISSOR);
    let topology = graphics_info.vertex_in.topology;
    if topology == vk::PrimitiveTopology::LINE_LIST
      || topology == vk::PrimitiveTopology::LINE_STRIP
      || topology == vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY
      || topology == vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY
      || graphics_info.rasterization_polygon_mode == vk::PolygonMode::LINE
      || !graphics_info.pipeline_flags.contains(PipelineFlags::NO_LINE_DYNAMIC_STATE)
    {
      dynamic_states.push(vk::DynamicState::LINE_WIDTH);
    }

    RawGraphicsInfoBuilder {
      graphics_info,
      color_blend_attachments,
      dynamic_states,
      pipeline_shader_stage_create_infos_builder: |graphics_info: &_| {
        let mut stages = Vec::with_capacity(2);
        let name = unsafe { ffi::CStr::from_bytes_with_nul_unchecked(b"main\0") };
        stages.push(
          vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(graphics_info.pre_rasterization.vertex_module)
            .name(name),
        );
        stages.push(
          vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(graphics_info.fragment_shader.fragment_module)
            .name(name),
        );
        stages
      },
      vertex_input_state_builder: |graphics_info: &_| {
        vk::PipelineVertexInputStateCreateInfo::default()
          .vertex_binding_descriptions(&graphics_info.vertex_in.bindings)
          .vertex_attribute_descriptions(&graphics_info.vertex_in.attributes)
      },
      input_assembly_state_builder: |graphics_info: &_| {
        let topology = graphics_info.vertex_in.topology;
        let is_strip = topology == vk::PrimitiveTopology::LINE_STRIP
          || topology == vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY
          || topology == vk::PrimitiveTopology::TRIANGLE_STRIP
          || topology == vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY;

        vk::PipelineInputAssemblyStateCreateInfo::default()
          .topology(topology)
          .primitive_restart_enable(is_strip)
      },
      tessellation_state_builder: |_| vk::PipelineTessellationStateCreateInfo::default(),
      viewport_state_builder: |graphics_info: &_| {
        vk::PipelineViewportStateCreateInfo::default()
          .viewports(&graphics_info.fragment_shader.viewports)
          .scissors(&graphics_info.fragment_shader.scissors)
      },
      rasterization_state_builder: |graphics_info: &_| {
        let mut cull_mode = vk::CullModeFlags::NONE;
        if graphics_info.pipeline_flags.contains(PipelineFlags::CULL_ALL) {
          cull_mode = vk::CullModeFlags::FRONT_AND_BACK;
        } else if graphics_info.pipeline_flags.contains(PipelineFlags::CULL_BACK) {
          cull_mode = vk::CullModeFlags::BACK;
        } else if graphics_info.pipeline_flags.contains(PipelineFlags::CULL_FRONT) {
          cull_mode = vk::CullModeFlags::FRONT;
        }

        let mut front_face = vk::FrontFace::COUNTER_CLOCKWISE;
        if graphics_info.pipeline_flags.contains(PipelineFlags::INVERT_FRONT_FACE) {
          front_face = vk::FrontFace::CLOCKWISE;
        }

        vk::PipelineRasterizationStateCreateInfo::default()
          .depth_clamp_enable(false)
          .rasterizer_discard_enable(false)
          .polygon_mode(graphics_info.rasterization_polygon_mode)
          .cull_mode(cull_mode)
          .front_face(front_face)
          .depth_bias_enable(graphics_info.pipeline_flags.contains(PipelineFlags::DEPTH_BIAS))
          .line_width(1.0)
      },
      multisample_state_builder: |_| {
        vk::PipelineMultisampleStateCreateInfo::default()
          .rasterization_samples(vk::SampleCountFlags::TYPE_1)
          .sample_shading_enable(false)
          .alpha_to_coverage_enable(false)
          .alpha_to_one_enable(false)
      },
      depth_stencil_state_builder: |graphics_info: &_| {
        let mut info = vk::PipelineDepthStencilStateCreateInfo::default()
          .depth_test_enable(!graphics_info.pipeline_flags.contains(PipelineFlags::NO_DEPTH_TEST))
          .depth_write_enable(!graphics_info.pipeline_flags.contains(PipelineFlags::NO_DEPTH_WRITE))
          .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
          .depth_bounds_test_enable(false)
          .min_depth_bounds(0.0)
          .max_depth_bounds(1.0);

        if graphics_info.pipeline_flags.contains(PipelineFlags::STENCIL_ENABLE) {
          info = info.stencil_test_enable(true);
          let mut front = vk::StencilOpState::default();
          front.reference = graphics_info.stencil_reference;
          front.compare_mask = graphics_info.stencil_compare_mask;
          front.write_mask = graphics_info.stencil_write_mask;

          front.compare_op = match graphics_info.stencil_compare_op {
            StencilCompareOp::None | StencilCompareOp::Always => vk::CompareOp::ALWAYS,
            StencilCompareOp::Equal => vk::CompareOp::EQUAL,
            StencilCompareOp::NotEqual => vk::CompareOp::NOT_EQUAL,
          };

          let mut back;
          match graphics_info.stencil_logic_op {
            StencilLogicOp::None => {
              front.fail_op = vk::StencilOp::KEEP;
              front.pass_op = vk::StencilOp::KEEP;
              front.depth_fail_op = vk::StencilOp::KEEP;
              back = front;
            }
            StencilLogicOp::Replace => {
              front.fail_op = vk::StencilOp::KEEP;
              front.pass_op = vk::StencilOp::REPLACE;
              front.depth_fail_op = vk::StencilOp::KEEP;
              back = front;
            }
            StencilLogicOp::CountDepthPass => {
              front.fail_op = vk::StencilOp::KEEP;
              front.pass_op = vk::StencilOp::DECREMENT_AND_WRAP;
              front.depth_fail_op = vk::StencilOp::KEEP;
              back = front;
              back.pass_op = vk::StencilOp::INCREMENT_AND_WRAP;
            }
            StencilLogicOp::CountDepthFail => {
              front.pass_op = vk::StencilOp::KEEP;
              front.fail_op = vk::StencilOp::KEEP;
              front.depth_fail_op = vk::StencilOp::INCREMENT_AND_WRAP;
              back = front;
              back.depth_fail_op = vk::StencilOp::DECREMENT_AND_WRAP;
            }
          }
          info = info.front(front).back(back);
        }
        info
      },
      color_blend_state_builder: |_, color_blend_attachments: &_| {
        vk::PipelineColorBlendStateCreateInfo::default()
          .logic_op_enable(false)
          .attachments(color_blend_attachments)
      },
      dynamic_state_builder: |_, dynamic_states: &_| {
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dynamic_states)
      },
      layout: graphics_info.pipeline_layout,
      render_pass: graphics_info.render_pass,
      subpass: graphics_info.subpass,
      graphics_pipeline_create_info_builder:
        |pipeline_shader_stage_create_infos,
         vertex_input_state,
         input_assembly_state,
         tessellation_state,
         viewport_state,
         rasterization_state,
         multisample_state,
         depth_stencil_state,
         color_blend_state,
         dynamic_state,
         layout,
         render_pass,
         subpass| {
          vk::GraphicsPipelineCreateInfo::default()
            .stages(pipeline_shader_stage_create_infos)
            .vertex_input_state(vertex_input_state)
            .input_assembly_state(input_assembly_state)
            .tessellation_state(tessellation_state)
            .viewport_state(viewport_state)
            .rasterization_state(rasterization_state)
            .multisample_state(multisample_state)
            .depth_stencil_state(depth_stencil_state)
            .color_blend_state(color_blend_state)
            .dynamic_state(dynamic_state)
            .layout(*layout)
            .render_pass(*render_pass)
            .subpass(*subpass)
        },
    }
    .build()
  }
}

/// Vulkan Resource to create and manage Graphics and Compute Pipelines
/// Note: To be wrapped in a RwLock, so won't sweat about implementing Sync and Send
pub(super) struct PipelinePool {
  graphics_pipelines: HashMap<PipelineKey, NonZeroHandle<vk::Pipeline>>,
  compute_pipelines: HashMap<PipelineKey, NonZeroHandle<vk::Pipeline>>,
  /// underlying vulkan cache object to speed up driver-level compilations
  vk_pipeline_cache: NonZeroHandle<vk::PipelineCache>,
}

impl PipelinePool {
  /// Creates a new PipelinePool
  /// `cache_data` can be loaded from disk (from a previous run) to warm up driver's cache
  pub fn new(device: &ash::Device, cache_data: Option<&[u8]>) -> GpuResult<Self> {
    let mut create_info = vk::PipelineCacheCreateInfo::default();
    if let Some(data) = cache_data {
      create_info = create_info.initial_data(data);
    }
    let vk_pipeline_cache =
      unsafe { NonZeroHandle::new_unchecked(device.create_pipeline_cache(&create_info, None)?) };
    Ok(Self {
      graphics_pipelines: HashMap::with_capacity(16),
      compute_pipelines: HashMap::with_capacity(16),
      vk_pipeline_cache,
    })
  }

  /// Extract the binary data from the Vulkan Pipeline Cache to save to disk
  pub fn get_cache_data(&self, device: &ash::Device) -> GpuResult<Vec<u8>> {
    let data = unsafe { device.get_pipeline_cache_data(self.vk_pipeline_cache.get()) }?;
    Ok(data)
  }

  /// TODO: Document this item
  pub fn get_graphics_pipeline(
    &self,
    pipeline_key: PipelineKey,
  ) -> Option<NonZeroHandle<vk::Pipeline>> {
    self.graphics_pipelines.get(&pipeline_key).copied()
  }

  /// TODO: Document this item
  pub fn get_compute_pipeline(
    &self,
    pipeline_key: PipelineKey,
  ) -> Option<NonZeroHandle<vk::Pipeline>> {
    self.compute_pipelines.get(&pipeline_key).copied()
  }

  /// TODO: Document this item
  pub fn discard_graphics_pipeline_if_present(
    &mut self,
    pipeline_key: PipelineKey,
    discard_pool: &DiscardPool,
    timeline: u64,
  ) -> bool {
    if let Some(pipeline) = self.graphics_pipelines.remove(&pipeline_key) {
      discard_pool.discard_pipeline(pipeline.get(), timeline);
      true
    } else {
      false
    }
  }

  /// TODO: Document this item
  pub fn discard_compute_pipeline_if_present(
    &mut self,
    pipeline_key: PipelineKey,
    discard_pool: &DiscardPool,
    timeline: u64,
  ) -> bool {
    if let Some(pipeline) = self.compute_pipelines.remove(&pipeline_key) {
      discard_pool.discard_pipeline(pipeline.get(), timeline);
      true
    } else {
      false
    }
  }

  /// TODO: Document this item
  pub fn get_or_create_compute_pipeline(
    &mut self,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    info: &ComputeInfo,
  ) -> GpuResult<NonZeroHandle<vk::Pipeline>> {
    use crate::gpu_backends::vulkan::device::VulkanDebugNameExt;
    let key = info.pipeline_key();
    if let Some(&pipeline) = self.compute_pipelines.get(&key) {
      return Ok(pipeline);
    }
    let raw_info = RawComputeInfo::from(info);
    let pipeline = unsafe {
      let mut pipeline = vk::Pipeline::null();
      let compute_info = raw_info.borrow_compute_pipeline_create_info();
      let res = (device.fp_v1_0().create_compute_pipelines)(
        device.handle(),
        self.vk_pipeline_cache.get(),
        1u32,
        ptr::from_ref(&compute_info),
        ptr::null(),
        ptr::from_mut(&mut pipeline),
      );
      NonZeroHandle::new_unchecked(
        res.result_with_success(pipeline).with_name(device, "VkPipeline_Compute")?,
      )
    };
    unsafe { self.compute_pipelines.insert_unique_unchecked(key, pipeline) };
    Ok(pipeline)
  }

  /// TODO: Document this item
  pub fn get_or_create_graphics_pipeline(
    &mut self,
    device: &crate::gpu_backends::vulkan::device::LogicalDevice,
    info: &GraphicsInfo,
  ) -> GpuResult<NonZeroHandle<vk::Pipeline>> {
    use crate::gpu_backends::vulkan::device::VulkanDebugNameExt;
    let key = info.pipeline_key();
    if let Some(&pipeline) = self.graphics_pipelines.get(&key) {
      return Ok(pipeline);
    }
    let raw_info = RawGraphicsInfo::from(info);
    let pipeline = unsafe {
      let mut pipeline = vk::Pipeline::null();
      let graphics_info = raw_info.borrow_graphics_pipeline_create_info();
      let res = (device.fp_v1_0().create_graphics_pipelines)(
        device.handle(),
        self.vk_pipeline_cache.get(),
        1,
        ptr::from_ref(&graphics_info),
        ptr::null(),
        ptr::from_mut(&mut pipeline),
      );
      NonZeroHandle::new_unchecked(
        res.result_with_success(pipeline).with_name(device, "VkPipeline_Graphics")?,
      )
    };
    unsafe {
      self.graphics_pipelines.insert_unique_unchecked(key, pipeline);
    }
    Ok(pipeline)
  }
}

impl DeviceResource for PipelinePool {
  fn cleanup(&mut self, device: &ash::Device) {
    unsafe {
      for (_, pipeline) in self.graphics_pipelines.drain() {
        device.destroy_pipeline(pipeline.get(), None);
      }
      for (_, pipeline) in self.compute_pipelines.drain() {
        device.destroy_pipeline(pipeline.get(), None);
      }
      device.destroy_pipeline_cache(self.vk_pipeline_cache.get(), None);
    }
  }
}
