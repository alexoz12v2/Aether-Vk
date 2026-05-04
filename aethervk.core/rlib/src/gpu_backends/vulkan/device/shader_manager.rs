use ash::vk;
use slotmap::{SlotMap, new_key_type};

use alloc::ffi::CString;
use hashbrown::HashMap;

use aethervk_oshal_rlib::os::fs::{self, Path, PathBuf};

use crate::{
  gpu_backends::vulkan::utils::NonZeroHandle,
  types::{GpuError, GpuResult},
};

new_key_type! { pub struct ShaderKey; }

pub fn execution_model_to_shader_flags(
  execution_model: spirv::ExecutionModel,
) -> vk::ShaderStageFlags {
  match execution_model {
    spirv::ExecutionModel::Vertex => vk::ShaderStageFlags::VERTEX,
    spirv::ExecutionModel::TessellationControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
    spirv::ExecutionModel::TessellationEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
    spirv::ExecutionModel::Geometry => vk::ShaderStageFlags::GEOMETRY,
    spirv::ExecutionModel::Fragment => vk::ShaderStageFlags::FRAGMENT,
    spirv::ExecutionModel::GLCompute => vk::ShaderStageFlags::COMPUTE,
    spirv::ExecutionModel::Kernel => vk::ShaderStageFlags::COMPUTE,
    spirv::ExecutionModel::TaskNV => vk::ShaderStageFlags::TASK_NV,
    spirv::ExecutionModel::MeshNV => vk::ShaderStageFlags::MESH_NV,
    spirv::ExecutionModel::RayGenerationKHR => vk::ShaderStageFlags::RAYGEN_KHR,
    spirv::ExecutionModel::IntersectionKHR => vk::ShaderStageFlags::INTERSECTION_KHR,
    spirv::ExecutionModel::AnyHitKHR => vk::ShaderStageFlags::ANY_HIT_KHR,
    spirv::ExecutionModel::ClosestHitKHR => vk::ShaderStageFlags::CLOSEST_HIT_KHR,
    spirv::ExecutionModel::MissKHR => vk::ShaderStageFlags::MISS_KHR,
    spirv::ExecutionModel::CallableKHR => vk::ShaderStageFlags::CALLABLE_KHR,
    spirv::ExecutionModel::TaskEXT => vk::ShaderStageFlags::TASK_EXT,
    spirv::ExecutionModel::MeshEXT => vk::ShaderStageFlags::MESH_EXT,
  }
}

pub fn shader_flags_to_execution_model(
  stage_flags: vk::ShaderStageFlags,
) -> Option<spirv::ExecutionModel> {
  match stage_flags {
    vk::ShaderStageFlags::VERTEX => Some(spirv::ExecutionModel::Vertex),
    vk::ShaderStageFlags::TESSELLATION_CONTROL => Some(spirv::ExecutionModel::TessellationControl),
    vk::ShaderStageFlags::TESSELLATION_EVALUATION => {
      Some(spirv::ExecutionModel::TessellationEvaluation)
    }
    vk::ShaderStageFlags::GEOMETRY => Some(spirv::ExecutionModel::Geometry),
    vk::ShaderStageFlags::FRAGMENT => Some(spirv::ExecutionModel::Fragment),
    vk::ShaderStageFlags::COMPUTE => Some(spirv::ExecutionModel::GLCompute),
    vk::ShaderStageFlags::TASK_NV => Some(spirv::ExecutionModel::TaskNV),
    vk::ShaderStageFlags::MESH_NV => Some(spirv::ExecutionModel::MeshNV),
    vk::ShaderStageFlags::RAYGEN_KHR => Some(spirv::ExecutionModel::RayGenerationKHR),
    vk::ShaderStageFlags::INTERSECTION_KHR => Some(spirv::ExecutionModel::IntersectionKHR),
    vk::ShaderStageFlags::ANY_HIT_KHR => Some(spirv::ExecutionModel::AnyHitKHR),
    vk::ShaderStageFlags::CLOSEST_HIT_KHR => Some(spirv::ExecutionModel::ClosestHitKHR),
    vk::ShaderStageFlags::MISS_KHR => Some(spirv::ExecutionModel::MissKHR),
    vk::ShaderStageFlags::CALLABLE_KHR => Some(spirv::ExecutionModel::CallableKHR),
    // Mixed or non Vulkan
    _ => None,
  }
}

// This struct holds the Vulkan shader module and other relevant data.
pub struct Shader {
  pub module: NonZeroHandle<vk::ShaderModule>,
  pub entry_point: CString,
  pub shader_stage: vk::ShaderStageFlags,
  pub spv_module: spirv_reflect::ShaderModule,
}

impl Shader {
  // Destroys the shader module.
  pub fn destroy(&self, device: &ash::Device) {
    unsafe {
      device.destroy_shader_module(self.module.get(), None);
    }
  }

  pub fn new(
    device: &ash::Device,
    code: &[u32],
    entry_point: &str,
    execution_model: spirv::ExecutionModel,
  ) -> GpuResult<Self> {
    let shader_create_info = vk::ShaderModuleCreateInfo::default().code(code);
    let spv_module = spirv_reflect::create_shader_module(unsafe {
      core::slice::from_raw_parts(
        code.as_ptr() as *const u8,
        code.len() * core::mem::size_of::<u32>(),
      )
    })
    .map_err(|_| GpuError::InvalidShader)?;
    spv_module
      .enumerate_entry_points()
      .map_err(|_| GpuError::InvalidShader)?
      .iter()
      .find(|&entry| entry.spirv_execution_model == execution_model && entry.name == entry_point)
      .ok_or(GpuError::BackendSpecific(alloc::fmt::format(format_args!(
        "Couldn't find entry point {} inside shader",
        entry_point
      ))))?;

    let module = unsafe { device.create_shader_module(&shader_create_info, None) }?;
    let module = unsafe { NonZeroHandle::new_unchecked(module) };

    Ok(Self {
      module,
      entry_point: CString::new(entry_point).map_err(|_| crate::gpu_err!("device error"))?,
      shader_stage: execution_model_to_shader_flags(execution_model),
      spv_module,
    })
  }
}

// Manages loading and storing shaders to avoid duplicates.
pub struct ShaderManager {
  shaders: SlotMap<ShaderKey, Shader>,
  shader_paths: HashMap<PathBuf, ShaderKey>,
}

unsafe impl Sync for ShaderManager {}
unsafe impl Send for ShaderManager {}

impl ShaderManager {
  pub fn new() -> Self {
    Self {
      shaders: SlotMap::with_key(),
      shader_paths: HashMap::new(),
    }
  }

  /// Gets a shader key for a given path, loading the shader if it's not already loaded.
  pub fn get_or_load(
    &mut self,
    device: &ash::Device,
    path: &Path,
    entry_point: &str,
    execution_model: spirv::ExecutionModel,
  ) -> GpuResult<ShaderKey> {
    if let Some(key) = self.shader_paths.get(path) {
      return Ok(*key);
    }

    // Load the SPIR-V shader code from the file.
    let spirv_code = fs::read(&path).map_err(|_| crate::gpu_invalid_arg!("invalid argument"))?;

    // Ensure the code is aligned to 4 bytes (u32).
    let (prefix, code, suffix) = unsafe { spirv_code.align_to::<u32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
      return Err(GpuError::InvalidShader);
    }

    let shader = Shader::new(device, code, entry_point, execution_model)?;
    let key = self.shaders.insert(shader);
    self.shader_paths.insert(path.to_pathbuf(), key);

    Ok(key)
  }

  /// Gets a reference to a shader from its key.
  pub fn get(&self, key: ShaderKey) -> Option<&Shader> {
    self.shaders.get(key)
  }

  /// Destroys all shader modules held by the manager.
  pub fn destroy(&self, device: &ash::Device) {
    for (_, shader) in self.shaders.iter() {
      shader.destroy(device);
    }
  }
}

impl Default for ShaderManager {
  fn default() -> Self {
    Self::new()
  }
}
