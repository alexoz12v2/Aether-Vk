use std::{env, fs, path::PathBuf, process::Command};

fn main() {
    let mut cargo_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    
    // Calculate repo root: CARGO_MANIFEST_DIR is likely aethervk.core/rlib
    let mut root_dir = cargo_dir.clone();
    root_dir.pop();
    root_dir.pop();
    
    let assets_dir = root_dir.join("assets");
    println!("cargo:rerun-if-changed={}", assets_dir.display());
    
    // Check if any .spv file is older than its source
    let mut needs_recompile = false;
    
    if let Ok(entries) = fs::read_dir(&assets_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "spv" {
                    let file_name = path.file_name().unwrap().to_string_lossy();
                    let mut src_name = file_name.to_string();
                    let mut found = false;
                    for src_ext in [".comp", ".vert", ".frag"] {
                        if let Some(idx) = src_name.find(src_ext) {
                            src_name.truncate(idx + src_ext.len());
                            found = true;
                            break;
                        }
                    }
                    
                    if found {
                        let src_path = assets_dir.join(&src_name);
                        if src_path.exists() {
                            if let (Ok(spv_meta), Ok(src_meta)) = (fs::metadata(&path), fs::metadata(&src_path)) {
                                if let (Ok(spv_time), Ok(src_time)) = (spv_meta.modified(), src_meta.modified()) {
                                    if spv_time < src_time {
                                        needs_recompile = true;
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    if needs_recompile {
        let is_windows = cfg!(target_os = "windows");
        let script = if is_windows { "compile_shaders.ps1" } else { "compile_shaders.sh" };
        
        let mut cmd = if is_windows {
            let mut c = Command::new("powershell");
            c.arg("-ExecutionPolicy").arg("Bypass").arg("-File").arg(script);
            c
        } else {
            let mut c = Command::new("bash");
            c.arg(script);
            c
        };
        
        cmd.current_dir(&root_dir);
        let status = cmd.status().expect("Failed to execute shader compilation script");
        if !status.success() {
            panic!("Shader compilation failed");
        }
    }

    if env::var("CARGO_FEATURE_TESTBED").is_ok() {
        let prof_dir = env::var("PROFILE").unwrap();
        cargo_dir.pop();
        cargo_dir.push("cdylib/target");
        cargo_dir.push(&prof_dir);
        cargo_dir.push("vulkan");

        if !cargo_dir.is_dir() {
            panic!(
                "'{}' doensn't exist. Have you built cdylib?",
                cargo_dir.display()
            );
        }
    }
}
