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

  let sdk_path =
    PathBuf::from(env::var("VULKAN_SDK").expect("VULKAN_SDK environment variable not defined"));
  if !sdk_path.exists() {
    panic!("VULKAN_SDK path doesn't exist: '{}'", sdk_path.display());
  }

  process_vulkan_sdk(&sdk_path);
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

  // TODO: copy VMA library (actually, vk_mem statically links it, so we'll see)
  if cfg!(target_os = "macos") || is_debug {
    fs::create_dir_all(&lib_dir).unwrap();
    fs::create_dir_all(&icd_dir).unwrap();
  }

  let src_lib_dir = sdk_path.to_path_buf().join("lib");
  if !src_lib_dir.exists() {
    panic!("{} doesn't exist", src_lib_dir.display());
  }

  // macOS: Copy vulkan loader, MoltenVK and its icd under ${dll_dir}/vulkan/*
  #[cfg(target_os = "macos")]
  {
    // directory structure (just for debug builds, `xtask` for avalonia packaging later)
    // copy vulkan loader and molten vk inside lib
    let search_path = src_lib_dir
      .join("libvulkan*.dylib")
      .into_os_string()
      .into_string()
      .unwrap();
    for entry in glob::glob(&search_path).unwrap() {
      let path = entry.unwrap();
      let dest_path = lib_dir.join(path.file_name().unwrap());
      if fs::symlink_metadata(&path)
        .unwrap()
        .file_type()
        .is_symlink()
      {
        // read symlink target and recreate it relative to result
        let target = fs::read_link(&path).unwrap();
        println!(
          "Symlink: {:?} -> {:?}",
          &dest_path,
          (&target).file_name().unwrap()
        );
        if (&dest_path).exists() || fs::symlink_metadata(&dest_path).is_ok() {
          println!("this is to be removed: {}", dest_path.display());
          fs::remove_file(&dest_path).unwrap();
        } else {
          println!("Apparently, this doesn't exist: {}", dest_path.display());
        }
        std::os::unix::fs::symlink((&target).file_name().unwrap(), &dest_path).unwrap();
      } else {
        println!("Copying {:?}", &path);
        fs::copy(&path, &dest_path).unwrap();
      }
    }

    // check whether you have libvulkan. If not, crash
    if !lib_dir.join("libvulkan.dylib").exists() {
      panic!("libvulkan.dylib doens't exist. Does your VULKAN_SDK path contain it? At {}", src_lib_dir.display());
    }

    const MOLTENVK_NAME: &str = "libMoltenVK.dylib";
    fs::copy(src_lib_dir.join(MOLTENVK_NAME), lib_dir.join(MOLTENVK_NAME)).unwrap();

    // copy ICD file inside and rename "library_path" to "../lib/MoltenVK.dylib"
    let icd_file = sdk_path
      .join("share")
      .join("vulkan")
      .join("icd.d")
      .join("MoltenVK_icd.json");
    if !icd_file.exists() {
      panic!("{} doesn't exist", icd_file.display());
    }
    let dst_icd = icd_dir.join("MoltenVK_icd.json");
    fs::copy(&icd_file, &dst_icd).unwrap();
    fixup_icd(&dst_icd, "ICD", MOLTENVK_NAME);
  }
  // if debug: copy validation layers inside the build under ${dll_dir}/vulkan/*
  if is_debug {
    let src_explicit_layer_dir = sdk_path
      .join("share")
      .join("vulkan")
      .join("explicit_layer.d");
    let explicit_layer_dir = binary_root.join("vulkan").join("explicit_layer");
    fs::create_dir_all(&explicit_layer_dir).unwrap();

    // copy dylibs
    let src_layer_libs = validation_layer_lib_names(&src_lib_dir);
    if let Some(path) = src_layer_libs.iter().filter(|p| !p.exists()).next() {
      panic!("{} missing", path.display());
    }
    for src in &src_layer_libs {
      fs::copy(src, lib_dir.join(src.file_name().unwrap())).unwrap();
    }
    // copy layer json
    let src_layer_jsons = validation_layer_json_names(&src_explicit_layer_dir);
    if let Some(path) = src_layer_jsons.iter().filter(|p| !p.exists()).next() {
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

#[cfg(target_os = "macos")]
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
  let fixed_path = {
    let mut s = "../lib/".to_string();
    s.push_str(dylib_name);
    s
  };
  if library_path != fixed_path.as_str() {
    *library_path = serde_json::Value::String(fixed_path);
    let the_content = serde_json::to_string_pretty(&json).unwrap();
    fs::write(dst_icd, &the_content).unwrap();
  }
}

fn validation_layer_lib_names(lib_dir: &Path) -> [PathBuf; 2] {
  #[cfg(target_os = "macos")]
  {
    [
      lib_dir.join("libVkLayer_khronos_validation.dylib"),
      lib_dir.join("libVkLayer_khronos_synchronization2.dylib"),
    ]
  }
  #[cfg(not(target_os = "macos"))]
  {
    todo!();
  }
}

fn validation_layer_json_names(explicit_layer_dir: &Path) -> [PathBuf; 2] {
  [
    explicit_layer_dir.join("VkLayer_khronos_validation.json"),
    explicit_layer_dir.join("VkLayer_khronos_synchronization2.json"),
  ]
}

fn validation_layer_json_to_dylib(json_file_name: &str) -> String {
  let prefix = if cfg!(any(target_os = "macos", target_os = "ios")) {
    "lib"
  } else {
    ""
  };
  let mut the_string = json_file_name.replace("json", DYNAMIC_LIBRARY_EXTENSION);
  if !prefix.is_empty() {
    the_string.insert_str(0, prefix);
  }

  the_string
}

#[cfg(target_vendor = "apple")]
const DYNAMIC_LIBRARY_EXTENSION: &str = "dylib";
#[cfg(windows)]
const DYNAMIC_LIBRARY_EXTENSION: &str = "dll";
#[cfg(all(target_family = "unix", not(target_vendor = "apple")))]
const DYNAMIC_LIBRARY_EXTENSION: &str = "so";
