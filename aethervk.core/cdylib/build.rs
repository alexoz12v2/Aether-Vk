use std::{
  env, fs,
  path::{Path, PathBuf},
};

fn main() {
  // rerun build script if it changed (should be default)
  println!("cargo:rerun-if-changed=build.rs");

  // if compiling with `--profile debug` (default)
  if env::var("PROFILE").unwrap() == "debug" {
    println!("Turning on Vulkan Layers");
    println!("cargo:rustc-check-cfg=cfg(vk_debug)");
    println!("cargo:rustc-cfg=vk_debug")
  }

  if let Ok(sdk_env) = env::var("VULKAN_SDK") {
    let sdk_path = PathBuf::from(sdk_env);
    if !sdk_path.exists() {
      panic!("VULKAN_SDK path doesn't exist: '{}'", sdk_path.display());
    }
    process_vulkan_sdk(&sdk_path);
  } else {
    println!("cargo:warning=VULKAN_SDK environment variable not defined.");
    if !cfg!(target_os = "linux") {
      panic!("VULKAN_SDK environment variable is required on Windows and macOS!");
    } else {
      println!("cargo:warning=Proceeding without VULKAN_SDK on Linux (assuming native system packages like libvulkan-dev are used).");
    }
  }

  let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
  if let Ok(bindings) = cbindgen::Builder::new()
    .with_crate(crate_dir)
    .with_language(cbindgen::Language::C)
    .with_no_includes()
    .with_sys_include("stdint.h")
    .with_sys_include("stdbool.h")
    .generate()
  {
    bindings.write_to_file("aethervk_core.h");
  }
}

fn get_binary_pathbuf() -> PathBuf {
  PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
    .join("target")
    .join(env::var("PROFILE").unwrap())
}

fn process_vulkan_sdk(sdk_path: &Path) {
  let binary_root = get_binary_pathbuf();
  let lib_dir = binary_root.join("vulkan").join("lib");
  let icd_dir = binary_root.join("vulkan").join("icd");
  let is_debug = env::var("PROFILE").unwrap() == "debug";

  if cfg!(target_os = "macos") || is_debug {
    fs::create_dir_all(&lib_dir).unwrap();
    fs::create_dir_all(&icd_dir).unwrap();
  }

  // Windows SDK keeps binaries in `Bin`, Unix variants use `lib`
  let src_lib_dir = if cfg!(windows) {
    sdk_path.join("Bin")
  } else {
    sdk_path.join("lib")
  };

  if !src_lib_dir.exists() {
    panic!("{} doesn't exist", src_lib_dir.display());
  }

  // macOS: Copy vulkan loader, MoltenVK and its icd under ${dll_dir}/vulkan/*
  #[cfg(target_os = "macos")]
  {
    // directory structure (just for debug builds, `xtask` for avalonia packaging later)
    let search_path = src_lib_dir.join("libvulkan*.dylib").into_os_string().into_string().unwrap();

    for entry in glob::glob(&search_path).unwrap() {
      let path = entry.unwrap();
      let dest_path = lib_dir.join(path.file_name().unwrap());
      if fs::symlink_metadata(&path).unwrap().file_type().is_symlink() {
        let target = fs::read_link(&path).unwrap();
        println!(
          "Symlink: {:?} -> {:?}",
          &dest_path,
          (&target).file_name().unwrap()
        );
        if (&dest_path).exists() || fs::symlink_metadata(&dest_path).is_ok() {
          fs::remove_file(&dest_path).unwrap();
        }
        std::os::unix::fs::symlink((&target).file_name().unwrap(), &dest_path).unwrap();
      } else {
        println!("Copying {:?}", &path);
        fs::copy(&path, &dest_path).unwrap();
      }
    }

    if !lib_dir.join("libvulkan.dylib").exists() {
      panic!(
        "libvulkan.dylib doesn't exist. Does your VULKAN_SDK path contain it? At {}",
        src_lib_dir.display()
      );
    }

    const MOLTENVK_NAME: &str = "libMoltenVK.dylib";
    fs::copy(src_lib_dir.join(MOLTENVK_NAME), lib_dir.join(MOLTENVK_NAME)).unwrap();

    let icd_file = sdk_path.join("share").join("vulkan").join("icd.d").join("MoltenVK_icd.json");

    if !icd_file.exists() {
      panic!("{} doesn't exist", icd_file.display());
    }
    let dst_icd = icd_dir.join("MoltenVK_icd.json");
    fs::copy(&icd_file, &dst_icd).unwrap();
    fixup_icd(&dst_icd, "ICD", MOLTENVK_NAME);
  }

  // if debug: copy validation layers inside the build under ${dll_dir}/vulkan/*
  if is_debug {
    // Windows often keeps JSONs next to DLLs in Bin. Unix uses share/vulkan/explicit_layer.d
    let src_explicit_layer_dir = if cfg!(windows) {
      sdk_path.join("Bin")
    } else {
      sdk_path.join("share").join("vulkan").join("explicit_layer.d")
    };

    let explicit_layer_dir = binary_root.join("vulkan").join("explicit_layer");
    fs::create_dir_all(&explicit_layer_dir).unwrap();

    // copy dynamic libraries
    let src_layer_libs = validation_layer_lib_names(&src_lib_dir);
    if let Some(path) = src_layer_libs.iter().find(|p| !p.exists()) {
      panic!("{} missing", path.display());
    }
    for src in &src_layer_libs {
      fs::copy(src, lib_dir.join(src.file_name().unwrap())).unwrap();
    }

    // copy layer json files
    let src_layer_jsons = validation_layer_json_names(&src_explicit_layer_dir);
    if let Some(path) = src_layer_jsons.iter().find(|p| !p.exists()) {
      panic!("{} missing", path.display());
    }
    for src in &src_layer_jsons {
      let json_name = src.file_name().unwrap().to_str().unwrap();
      let dylib_name = validation_layer_json_to_dylib(json_name);
      let dst_json_path = explicit_layer_dir.join(json_name);
      fs::copy(src, &dst_json_path).unwrap();
      fixup_icd(&dst_json_path, "layer", &dylib_name);
    }
  }
}

// Removed #[cfg(target_os = "macos")] - this is now cross-platform.
fn fixup_icd(dst_icd: &Path, key_name: &str, dylib_name: &str) {
  let json_bytes =
    fs::read(dst_icd).unwrap_or_else(|e| panic!("Failed to read {}: {e}", dst_icd.display()));
  let mut json: serde_json::Value = serde_json::from_slice(&json_bytes)
    .unwrap_or_else(|e| panic!("Invalid JSON at {}: {}", dst_icd.display(), e));

  let obj = json.as_object_mut().expect("JSON root should be an object");
  let library_path = obj
    .get_mut(key_name)
    .expect("JSON missing key")
    .as_object_mut()
    .expect("JSON at key should be an object")
    .get_mut("library_path")
    .expect("Key `${key}.library_path` doesn't exist");

  let fixed_path = format!("../lib/{}", dylib_name);

  if library_path != fixed_path.as_str() {
    *library_path = serde_json::Value::String(fixed_path);
    let the_content = serde_json::to_string_pretty(&json).unwrap();
    fs::write(dst_icd, &the_content).unwrap();
  }
}

fn validation_layer_lib_names(lib_dir: &Path) -> Vec<PathBuf> {
  let prefix = if cfg!(windows) { "" } else { "lib" };
  vec![
    lib_dir.join(format!(
      "{}VkLayer_khronos_validation.{}",
      prefix, DYNAMIC_LIBRARY_EXTENSION
    )),
    // Note: SDK versions 1.3.231+ merged synchronization2 into the main validation layer.
    // If your SDK is newer, you might want to remove this second file to prevent panics.
    lib_dir.join(format!(
      "{}VkLayer_khronos_synchronization2.{}",
      prefix, DYNAMIC_LIBRARY_EXTENSION
    )),
  ]
}

fn validation_layer_json_names(explicit_layer_dir: &Path) -> Vec<PathBuf> {
  vec![
    explicit_layer_dir.join("VkLayer_khronos_validation.json"),
    explicit_layer_dir.join("VkLayer_khronos_synchronization2.json"),
  ]
}

fn validation_layer_json_to_dylib(json_file_name: &str) -> String {
  let prefix = if cfg!(windows) { "" } else { "lib" };
  let base_name = json_file_name.replace(".json", "");
  format!("{}{}.{}", prefix, base_name, DYNAMIC_LIBRARY_EXTENSION)
}

#[cfg(target_vendor = "apple")]
const DYNAMIC_LIBRARY_EXTENSION: &str = "dylib";
#[cfg(windows)]
const DYNAMIC_LIBRARY_EXTENSION: &str = "dll";
#[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
const DYNAMIC_LIBRARY_EXTENSION: &str = "so";
