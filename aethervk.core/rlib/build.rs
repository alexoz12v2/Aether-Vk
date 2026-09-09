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
  let dirs_to_check = [assets_dir.clone(), assets_dir.join("sim")];

  for dir in &dirs_to_check {
    if let Ok(entries) = fs::read_dir(dir) {
      for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() {
          if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext == "comp" || ext == "vert" || ext == "frag" {
              let mut spv_path = path.clone().into_os_string();
              spv_path.push(".spv");
              let spv_path = PathBuf::from(spv_path);

              if !spv_path.exists() {
                needs_recompile = true;
                break;
              }

              if let (Ok(src_meta), Ok(spv_meta)) = (fs::metadata(&path), fs::metadata(&spv_path)) {
                if let (Ok(src_time), Ok(spv_time)) = (src_meta.modified(), spv_meta.modified()) {
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
    if needs_recompile {
      break;
    }
  }

  if needs_recompile {
    let is_windows = cfg!(target_os = "windows");
    let script = if is_windows {
      "compile_shaders.ps1"
    } else {
      "compile_shaders.sh"
    };

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
    let output = cmd.output().expect("Failed to execute shader compilation script");
    if !output.status.success() {
      let stdout = String::from_utf8_lossy(&output.stdout);
      let stderr = String::from_utf8_lossy(&output.stderr);
      panic!(
        "Shader compilation failed!\n\nSTDOUT:\n{}\n\nSTDERR:\n{}",
        stdout, stderr
      );
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
