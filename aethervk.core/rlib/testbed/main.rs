use aethervk_core_rlib::{self as lib, gpu::constants };
use heapless::index_map::FnvIndexMap;

extern crate std;

fn main() {
  println!("Hello std");
  let path = {
    let mut p = std::env::current_exe().unwrap();
    for _ in 1..6 {
      let b = p.pop();
      assert!(b);
    }
    p.push("cdylib/target/");
    p.push(if cfg!(debug_assertions) { "debug" } else { "release" });
    p.push("vulkan");

    p
  };
  if !path.is_dir() {
    panic!("{} doesn't exist", path.display());
  }

  let params = lib::types::RuntimeParams {
    render_backend_params: {
      let mut the_map = FnvIndexMap::new();
      let mut the_str = path.display().to_string();
      the_map.insert(constants::RUNTIME_PARAM_VULKAN_ENTRY_BASE_DIR, the_str);

      the_map
    },
  };
  let render_frontend = lib::gpu::new_render_frontend(lib::gpu::VULKAN_RENDER_BACKEND, &params)
    .expect("Couldn't create Vulkan Instance");

  render_frontend.take_and(|render_backend| {
    println!("Created Vulkan Instance");
    Ok(())
  });
}
