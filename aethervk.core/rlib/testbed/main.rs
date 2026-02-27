use aethervk_core_rlib as lib;

fn main() {
  let hello_msg = "Hello\n";
  unsafe {
    libc::write(
      libc::STDOUT_FILENO,
      hello_msg.as_ptr().cast(),
      hello_msg.len(),
    )
  };

  unsafe { lib::gpu_backends::vulkan::instance::Instance::new() }.unwrap();
}
