use core::{hash::Hash};
use alloc::{vec::Vec};

use ash::vk;
use bitflags::bitflags;

// ---------------- COMPUTE PIPELINE HASH ------------------------------------
fn eq_specialization_constants(
  a: &vk::SpecializationMapEntry,
  b: &vk::SpecializationMapEntry,
) -> bool {
  a.size == b.size && a.offset == b.offset && a.constant_id == b.constant_id
}

pub struct ComputeInfo {
  pub shader_module: vk::ShaderModule,
  pub pipeline_layout: vk::PipelineLayout,
  pub specialization_constants: Vec<vk::SpecializationMapEntry>,
}

impl ComputeInfo {
  pub fn new(shader_module: vk::ShaderModule, pipeline_layout: vk::PipelineLayout) -> Self {
    Self {
      shader_module,
      pipeline_layout,
      specialization_constants: Vec::with_capacity(8),
    }
  }

  pub fn with_specialization_constants(
    &self,
    specialization_constants: &[vk::SpecializationMapEntry],
  ) -> Self {
    Self {
      shader_module: self.shader_module,
      pipeline_layout: self.pipeline_layout,
      specialization_constants: {
        let mut the_vec = Vec::with_capacity(specialization_constants.len());
        the_vec.extend_from_slice(specialization_constants);
        the_vec
      },
    }
  }
}

impl PartialEq for ComputeInfo {
  fn eq(&self, other: &Self) -> bool {
    let mut result = self.shader_module == other.shader_module
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
  }
}

// ---------------- GRAPHICS PIPELINE HASH -----------------------------------
bitflags! {
  pub struct PipelineFlags: u32 {
    const DEPTH_BIAS = 0x1u32 << 0;
    const CULL = 0x1u32 << 1;
    const CULL_FRONT = 0x3u32 << 1;
    const CULL_BACK = 0x1u32 << 1;
    const CULL_ALL = 0x2u32 << 1;
    const INVERT_FRONT_FACE = 0x1u32 << 3;
    const NO_DEPTH_WRITE = 0x1u32 << 4;
    const STENCIL_ENABLE = 0x1u32 << 5;
  }
}

#[repr(u32)]
pub enum StencilCompareOp {
  None = 0,
  Equal,
  NotEqual,
  Always,
}

#[repr(u32)]
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

fn eq_vertex_input_attribute_description(
  a: &vk::VertexInputAttributeDescription,
  b: &vk::VertexInputAttributeDescription,
) -> bool {
  a.binding == b.binding && a.format == b.format && a.location == b.location && a.offset == b.offset
}

pub struct VertexIn {
  // adjacency requires `geometryShader` feature, patch requires `tessellationShader` feature
  topology: vk::PrimitiveTopology,
  // defined what's inside each binding, which can be split in multiple
  // locations (globally unique, monotonically increasing) if you need multple
  // pieces of data having a VkFormat
  attributes: Vec<vk::VertexInputAttributeDescription>,
  // define how big a single vertex binding or a single instance binding and
  // how it's stepped through memory. *no information of what's inside it*
  bindings: Vec<vk::VertexInputBindingDescription>,
}
