pub struct AppThreads {
  pub logic_thread: Option<std::thread::JoinHandle<()>>,
  pub render_thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for AppThreads {
  fn drop(&mut self) {
    if let Some(logic_thread) = self.logic_thread.take() {
      let _ = logic_thread.join();
    }
    if let Some(render_thread) = self.render_thread.take() {
      let _ = render_thread.join();
    }
  }
}
