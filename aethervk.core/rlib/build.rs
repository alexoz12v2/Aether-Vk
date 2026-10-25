use std::{env, fs, path::PathBuf, process::Command};

/// Git Bash on Windows: `AETHERVK_BASH`, else `bin/bash.exe` next to the `git` on PATH, else the
/// default install. Never `System32\bash.exe` (WSL: a Linux VULKAN_SDK, not the Windows one).
fn find_git_bash() -> Option<PathBuf> {
  if let Some(p) = env::var_os("AETHERVK_BASH").map(PathBuf::from) {
    return p.is_file().then_some(p);
  }
  let from_git = env::var_os("PATH").and_then(|paths| {
    env::split_paths(&paths)
      .filter(|dir| dir.join("git.exe").is_file())
      // <Git>/cmd/git.exe or <Git>/bin/git.exe -> <Git>/bin/bash.exe
      .find_map(|dir| dir.parent().map(|git| git.join("bin").join("bash.exe")).filter(|b| b.is_file()))
  });
  from_git.or_else(|| {
    env::var_os("ProgramFiles")
      .map(|pf| PathBuf::from(pf).join("Git").join("bin").join("bash.exe"))
      .filter(|b| b.is_file())
  })
}

fn main() {
  let mut cargo_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

  // Calculate repo root: CARGO_MANIFEST_DIR is likely aethervk.core/rlib
  let mut root_dir = cargo_dir.clone();
  root_dir.pop();
  root_dir.pop();

  let assets_dir = root_dir.join("assets");
  println!("cargo:rerun-if-changed={}", assets_dir.display());

  // Check if any .spv file is older than its source, or than any included .glsl (a shader that
  // only includes a changed `dust_common.glsl` is stale too: the compute shaders went un-rebuilt
  // for a day of `dust_common.glsl` edits before this check)
  let mut needs_recompile = false;
  let dirs_to_check = [assets_dir.clone(), assets_dir.join("sim")];
  let mut include_time: Option<std::time::SystemTime> = None;
  for dir in &dirs_to_check {
    if let Ok(entries) = fs::read_dir(dir) {
      for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("glsl") {
          if let Ok(t) = fs::metadata(&path).and_then(|m| m.modified()) {
            include_time = Some(include_time.map_or(t, |u| u.max(t)));
          }
        }
      }
    }
  }

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
                  if spv_time < src_time || include_time.is_some_and(|t| spv_time < t) {
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
    // compile_shaders.sh is the maintained script (blob variants, spv1.4, spirv-opt): on Windows
    // it runs under Git Bash; compile_shaders.ps1 is a stale fallback whose inline `-Os` crashes
    // shaderc on dust_emit_mono.comp ("Aligned operand value 0")
    let windows_bash = if cfg!(target_os = "windows") { find_git_bash() } else { None };

    let mut cmd = if cfg!(target_os = "windows") && windows_bash.is_none() {
      println!("cargo:warning=Git Bash not found (set AETHERVK_BASH): falling back to compile_shaders.ps1");
      let mut c = Command::new("powershell");
      c.arg("-ExecutionPolicy").arg("Bypass").arg("-File").arg("compile_shaders.ps1");
      c
    } else {
      let mut c = Command::new(windows_bash.unwrap_or_else(|| PathBuf::from("bash")));
      c.arg("compile_shaders.sh");
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
    } else {
      let stdout = String::from_utf8_lossy(&output.stdout);
      for line in stdout.lines() {
        println!("cargo:warning={}", line);
      }
    }
  }

  println!("cargo:rerun-if-env-changed=AETHERVK_BASH");

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
