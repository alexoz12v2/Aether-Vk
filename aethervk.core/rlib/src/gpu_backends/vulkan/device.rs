use crate::gpu::RenderDevice;

pub(super) struct Device {}

impl RenderDevice for Device {
  #[cfg(debug_assertions)]
  fn print_info(&self) -> alloc::string::String {
    todo!()
  }

  fn context_id(&self) -> u64 {
    todo!()
  }
}
