use std::{env, path::PathBuf};

fn main() {
  if env::var("CARGO_FEATURE_TESTBED").is_ok() {
    let mut cargo_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
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
