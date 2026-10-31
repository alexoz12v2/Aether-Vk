//! Opt-in Tracy instrumentation (`profile-tracy`): zones, thread names and frame marks.
//!
//! The macros expand to nothing without the feature, so the shipped build carries no dependency
//! and no code. With it, `tracy-client` is linked and the process broadcasts to a Tracy profiler
//! (GUI or `tracy-capture`) on port 8086; the GUI version must match the client's protocol
//! (`tracy-client` 0.19 ↔ Tracy 0.14.1, see the crate's version table). The client starts before
//! `main`, so there is no init call: the first zone, thread name or frame mark is enough.
//!
//! The zones are a few dozen hand-placed ones (the logic tick phases, the render frame, the dust
//! GPU passes' recording and submit, the blocking FFI calls), not a tracing stack: no `tracing`
//! spans, no per-function instrumentation. The dust passes' GPU time is a separate Tracy GPU
//! context fed from the Vulkan timestamps the engine already writes
//! (`gpu_backends::vulkan::device::profile` in the core rlib).
//!
//! ```text
//! prof_zone!("logic: fixed update");      // scoped: the zone ends at the end of the block
//! prof_thread_name!("aethervk-logic");    // the calling thread's name in the profiler
//! prof_frame_mark!();                     // one rendered frame
//! ```
//! The zone name must be a string literal: Tracy keeps one static source location per site.

/// `tracy_client`, re-exported for the macros (`$crate::os::profile::tracy_client::…`), so a crate
/// using them needs no dependency of its own.
#[cfg(feature = "profile-tracy")]
pub use tracy_client;

/// Names the calling thread in the profiler. The OS-level name (`SetThreadDescription`,
/// `pthread_setname_np`) is set by `thread::Builder` and the pool as before; this is the one
/// Tracy shows, since it reads thread names from the client, not from the OS.
#[cfg(feature = "profile-tracy")]
pub fn thread_name(name: &str) {
  tracy_client::Client::start().set_thread_name(name);
}

/// [`thread_name`] with formatted arguments (the pool's `worker_N`); formats only with the feature.
#[cfg(feature = "profile-tracy")]
pub fn thread_name_args(args: core::fmt::Arguments<'_>) {
  thread_name(&alloc::format!("{}", args));
}

/// A scoped Tracy zone named by a string literal; a no-op without `profile-tracy`.
#[cfg(feature = "profile-tracy")]
#[macro_export]
macro_rules! prof_zone {
  ($name:literal) => {
    // not `_`: the span must live to the end of the enclosing block
    let _prof_zone = $crate::os::profile::tracy_client::span!($name);
  };
}
/// A scoped Tracy zone named by a string literal; a no-op without `profile-tracy`.
#[cfg(not(feature = "profile-tracy"))]
#[macro_export]
macro_rules! prof_zone {
  ($name:literal) => {};
}

/// Names the calling thread for the profiler (format arguments); a no-op without `profile-tracy`.
#[cfg(feature = "profile-tracy")]
#[macro_export]
macro_rules! prof_thread_name {
  ($($arg:tt)*) => {
    $crate::os::profile::thread_name_args(core::format_args!($($arg)*))
  };
}
/// Names the calling thread for the profiler (format arguments); a no-op without `profile-tracy`.
#[cfg(not(feature = "profile-tracy"))]
#[macro_export]
macro_rules! prof_thread_name {
  ($($arg:tt)*) => {
    ()
  };
}

/// Marks the end of a rendered frame on Tracy's frame timeline; a no-op without `profile-tracy`.
#[cfg(feature = "profile-tracy")]
#[macro_export]
macro_rules! prof_frame_mark {
  () => {
    $crate::os::profile::tracy_client::frame_mark()
  };
}
/// Marks the end of a rendered frame on Tracy's frame timeline; a no-op without `profile-tracy`.
#[cfg(not(feature = "profile-tracy"))]
#[macro_export]
macro_rules! prof_frame_mark {
  () => {
    ()
  };
}
