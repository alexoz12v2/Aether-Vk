//! shader_manager module.

use alloc::ffi::CString;
use ash::vk;
use function_name::named;
use hashbrown::HashMap;
use slotmap::{SlotMap, new_key_type};

use aethervk_oshal_rlib::os::fs::{self, Path, PathBuf};
use alloc::string::ToString;

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
  /// Path of the SPIR-V file this shader was loaded from, for debug logging.
  pub path: alloc::string::String,
}

impl Shader {
  // Destroys the shader module.
  pub fn destroy(&self, device: &ash::Device) {
    unsafe {
      device.destroy_shader_module(self.module.get(), None);
    }
  }

  #[named]
  pub fn new(
    device: &ash::Device,
    code: &[u32],
    entry_point: &str,
    execution_model: spirv::ExecutionModel,
    path: &str,
  ) -> GpuResult<Self> {
    let mut patched_code = code.to_vec();
    if cfg!(debug_assertions) || cfg!(test) {
      patch_spirv_prevent_inlining(&mut patched_code);
      let spv_u8 = unsafe {
        core::slice::from_raw_parts(
          patched_code.as_ptr() as *const u8,
          patched_code.len() * core::mem::size_of::<u32>(),
        )
      };

      // TODO: Uncomment once we fix stuff .NET Side, logs too polluted
      if false {
        #[cfg(not(windows))]
        {
          let _ = aethervk_oshal_rlib::os::fs::write(PathBuf::from("/tmp/dump.spv"), spv_u8);
          unsafe {
            let cmd = alloc::ffi::CString::new("spirv-dis /tmp/dump.spv").unwrap();
            libc::system(cmd.as_ptr());
          }
        }
        #[cfg(windows)]
        unsafe {
          use windows::Win32::Foundation::{
            CloseHandle, HANDLE, HANDLE_FLAG_INHERIT, HANDLE_FLAGS, SetHandleInformation,
          };
          use windows::Win32::Storage::FileSystem::WriteFile;
          use windows::Win32::System::Console::{
            GetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE,
          };
          use windows::Win32::System::Environment::GetEnvironmentVariableW;
          use windows::Win32::System::Pipes::CreatePipe;
          use windows::Win32::System::Threading::{
            CreateProcessW, INFINITE, PROCESS_CREATION_FLAGS, PROCESS_INFORMATION,
            STARTF_USESTDHANDLES, STARTUPINFOW, WaitForSingleObject,
          };
          use windows::core::{PCWSTR, PWSTR};

          let mut sdk_buf = [0u16; 512];
          let len = GetEnvironmentVariableW(windows::core::w!("VULKAN_SDK"), Some(&mut sdk_buf));
          if len > 0 && len <= 512 {
            let sdk_str = alloc::string::String::from_utf16_lossy(&sdk_buf[..len as usize]);
            let spirv_dis_exe = alloc::format!("{}\\Bin\\spirv-dis.exe", sdk_str);

            let mut h_stdin_read = HANDLE::default();
            let mut h_stdin_write = HANDLE::default();

            let mut sa = windows::Win32::Security::SECURITY_ATTRIBUTES::default();
            sa.nLength =
              core::mem::size_of::<windows::Win32::Security::SECURITY_ATTRIBUTES>() as u32;
            sa.bInheritHandle = true.into();

            if CreatePipe(
              &mut h_stdin_read,
              &mut h_stdin_write,
              Some(&sa),
              1024 * 1024 * 10,
            )
            .is_ok()
            {
              let _ = SetHandleInformation(h_stdin_write, HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0));

              let mut si = STARTUPINFOW::default();
              si.cb = core::mem::size_of::<STARTUPINFOW>() as u32;
              si.dwFlags = STARTF_USESTDHANDLES;
              si.hStdInput = h_stdin_read;
              si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE).unwrap_or(HANDLE::default());
              si.hStdError = GetStdHandle(STD_ERROR_HANDLE).unwrap_or(HANDLE::default());

              let mut pi = PROCESS_INFORMATION::default();
              let cmd_str = alloc::format!("\"{}\" -\0", spirv_dis_exe);
              let mut cmd: alloc::vec::Vec<u16> = cmd_str.encode_utf16().collect();

              if CreateProcessW(
                PCWSTR::null(),
                Some(PWSTR(cmd.as_mut_ptr())),
                None,
                None,
                true,
                PROCESS_CREATION_FLAGS(0),
                None,
                PCWSTR::null(),
                &mut si,
                &mut pi,
              )
              .is_ok()
              {
                let _ = CloseHandle(h_stdin_read);
                let mut written = 0;
                let _ = WriteFile(h_stdin_write, Some(spv_u8), Some(&mut written), None);
                let _ = CloseHandle(h_stdin_write);

                let _ = WaitForSingleObject(pi.hProcess, INFINITE);
                let _ = CloseHandle(pi.hProcess);
                let _ = CloseHandle(pi.hThread);
              } else {
                let _ = CloseHandle(h_stdin_read);
                let _ = CloseHandle(h_stdin_write);
              }
            }
          }
        }
      }
    }

    let shader_create_info = vk::ShaderModuleCreateInfo::default().code(&patched_code);
    let spv_module = spirv_reflect::create_shader_module(unsafe {
      core::slice::from_raw_parts(
        patched_code.as_ptr() as *const u8,
        patched_code.len() * core::mem::size_of::<u32>(),
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
      entry_point: CString::new(entry_point).map_err(|_| crate::gpu_err_device!())?,
      shader_stage: execution_model_to_shader_flags(execution_model),
      spv_module,
      path: alloc::string::String::from(path),
    })
  }
}

// Manages loading and storing shaders to avoid duplicates.
pub struct ShaderManager {
  shaders: SlotMap<ShaderKey, alloc::sync::Arc<Shader>>,
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
  #[named]
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
    let spirv_code = fs::read(&path).map_err(|_| crate::gpu_invalid_arg!())?;

    // Ensure the code is aligned to 4 bytes (u32).
    let (prefix, code, suffix) = unsafe { spirv_code.align_to::<u32>() };
    if !prefix.is_empty() || !suffix.is_empty() {
      return Err(GpuError::InvalidShader);
    }

    let path_str = path
      .to_str_unified()
      .map(|s| s.into_owned())
      .unwrap_or_else(|| alloc::string::String::from("<unknown>"));
    aethervk_oshal_rlib::log!("[ShaderManager] Loading shader: {}", path_str);
    let shader = alloc::sync::Arc::new(Shader::new(
      device,
      code,
      entry_point,
      execution_model,
      &path_str,
    )?);
    let key = self.shaders.insert(shader);
    self.shader_paths.insert(path.to_pathbuf(), key);

    Ok(key)
  }

  /// Gets a reference to a shader from its key.
  pub fn get(&self, key: ShaderKey) -> Option<alloc::sync::Arc<Shader>> {
    self.shaders.get(key).cloned()
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

/// Patches a SPIR-V binary in-place to force all functions to NOT inline.
/// This severely limits monolithic stack frames in CPU Vulkan implementations,
/// avoiding AArch64 link register (x30) clobbering and SIGSEGVs.
pub fn patch_spirv_prevent_inlining(spv: &mut [u32]) {
  // Validate the SPIR-V magic number (native endian)
  if spv.len() < 5 || spv[0] != 0x07230203 {
    return;
  }

  let mut i = 5; // Skip the 5-word SPIR-V header
  while i < spv.len() {
    let inst = spv[i];
    let opcode = inst & 0xFFFF;
    let word_count = (inst >> 16) as usize;

    if word_count == 0 {
      break;
    } // Safety break on malformed SPIR-V

    // 54 == OpFunction
    if opcode == 54 && i + 3 < spv.len() {
      // Function Control bitmask is the 3rd operand (at index i + 3)
      // Bit 0 = Inline (0x1), Bit 1 = DontInline (0x2)

      // Clear the 'Inline' bit (if any) and strictly set the 'DontInline' bit
      spv[i + 3] = (spv[i + 3] & !0x1) | 0x2;
    }

    i += word_count;
  }
}

pub fn disassemble_and_log_spirv(spv_code: &[u8]) {
  #[cfg(target_family = "unix")]
  unsafe {
    let _ = aethervk_oshal_rlib::os::fs::write(&PathBuf::from("/tmp/dump.spv"), spv_code);
    let env_var = alloc::ffi::CString::new("VULKAN_SDK").unwrap();
    let sdk_ptr = libc::getenv(env_var.as_ptr());
    let cmd_str = if sdk_ptr.is_null() {
      "spirv-dis /tmp/dump.spv".to_string()
    } else {
      let sdk = core::ffi::CStr::from_ptr(sdk_ptr).to_string_lossy();
      alloc::format!("{}/bin/spirv-dis /tmp/dump.spv", sdk)
    };
    let cmd = alloc::ffi::CString::new(cmd_str).unwrap();
    let mode = alloc::ffi::CString::new("r").unwrap();
    let fp = libc::popen(cmd.as_ptr(), mode.as_ptr());
    if !fp.is_null() {
      let mut buf = [0i8; 1024];
      let mut out = alloc::string::String::new();
      while !libc::fgets(buf.as_mut_ptr().cast(), buf.len() as i32, fp).is_null() {
        out.push_str(&core::ffi::CStr::from_ptr(buf.as_ptr().cast()).to_string_lossy());
      }
      libc::pclose(fp);
      aethervk_oshal_rlib::log!("SPIR-V Disassembly:\n{}", out);
    }
  }

  #[cfg(windows)]
  unsafe {
    use windows::Win32::Foundation::{
      CloseHandle, HANDLE, HANDLE_FLAG_INHERIT, HANDLE_FLAGS, SetHandleInformation,
    };
    use windows::Win32::Storage::FileSystem::{ReadFile, WriteFile};
    use windows::Win32::System::Environment::GetEnvironmentVariableW;
    use windows::Win32::System::Pipes::CreatePipe;
    use windows::Win32::System::Threading::{
      CreateProcessW, INFINITE, PROCESS_CREATION_FLAGS, PROCESS_INFORMATION, STARTF_USESTDHANDLES,
      STARTUPINFOW, WaitForSingleObject,
    };
    use windows::core::{PCWSTR, PWSTR};

    let mut sdk_buf = [0u16; 512];
    let len = GetEnvironmentVariableW(windows::core::w!("VULKAN_SDK"), Some(&mut sdk_buf));
    if len > 0 && len <= 512 {
      let sdk_str = alloc::string::String::from_utf16_lossy(&sdk_buf[..len as usize]);
      let spirv_dis_exe = alloc::format!("{}\\Bin\\spirv-dis.exe", sdk_str);

      let mut h_stdin_read = HANDLE::default();
      let mut h_stdin_write = HANDLE::default();
      let mut h_stdout_read = HANDLE::default();
      let mut h_stdout_write = HANDLE::default();

      let mut sa = windows::Win32::Security::SECURITY_ATTRIBUTES::default();
      sa.nLength = core::mem::size_of::<windows::Win32::Security::SECURITY_ATTRIBUTES>() as u32;
      sa.bInheritHandle = true.into();

      if CreatePipe(
        &mut h_stdin_read,
        &mut h_stdin_write,
        Some(&sa),
        1024 * 1024 * 10,
      )
      .is_ok()
        && CreatePipe(
          &mut h_stdout_read,
          &mut h_stdout_write,
          Some(&sa),
          1024 * 1024 * 10,
        )
        .is_ok()
      {
        let _ = SetHandleInformation(h_stdin_write, HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0));
        let _ = SetHandleInformation(h_stdout_read, HANDLE_FLAG_INHERIT.0, HANDLE_FLAGS(0));

        let mut si = STARTUPINFOW::default();
        si.cb = core::mem::size_of::<STARTUPINFOW>() as u32;
        si.dwFlags = STARTF_USESTDHANDLES;
        si.hStdInput = h_stdin_read;
        si.hStdOutput = h_stdout_write;
        si.hStdError = h_stdout_write;

        let mut pi = PROCESS_INFORMATION::default();
        let cmd_str = alloc::format!("\"{}\" -\0", spirv_dis_exe);
        let mut cmd: alloc::vec::Vec<u16> = cmd_str.encode_utf16().collect();

        if CreateProcessW(
          PCWSTR::null(),
          Some(PWSTR(cmd.as_mut_ptr())),
          None,
          None,
          true,
          PROCESS_CREATION_FLAGS(0),
          None,
          PCWSTR::null(),
          &mut si,
          &mut pi,
        )
        .is_ok()
        {
          let _ = CloseHandle(h_stdin_read);
          let _ = CloseHandle(h_stdout_write);

          let mut written = 0;
          let _ = WriteFile(h_stdin_write, Some(spv_code), Some(&mut written), None);
          let _ = CloseHandle(h_stdin_write);

          let mut out_bytes = alloc::vec::Vec::new();
          let mut buf = [0u8; 4096];
          loop {
            let mut bytes_read = 0;
            let read_res = ReadFile(h_stdout_read, Some(&mut buf), Some(&mut bytes_read), None);
            if read_res.is_err() || bytes_read == 0 {
              break;
            }
            out_bytes.extend_from_slice(&buf[..bytes_read as usize]);
          }
          let _ = CloseHandle(h_stdout_read);

          let _ = WaitForSingleObject(pi.hProcess, INFINITE);
          let _ = CloseHandle(pi.hProcess);
          let _ = CloseHandle(pi.hThread);

          if let Ok(out_str) = core::str::from_utf8(&out_bytes) {
            aethervk_oshal_rlib::log!("SPIR-V Disassembly:\n{}", out_str);
          }
        } else {
          let _ = CloseHandle(h_stdin_read);
          let _ = CloseHandle(h_stdin_write);
          let _ = CloseHandle(h_stdout_read);
          let _ = CloseHandle(h_stdout_write);
        }
      }
    }
  }
}
