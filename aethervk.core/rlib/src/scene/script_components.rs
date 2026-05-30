use crate::scene::{Component, EntityId, Scene};

/// Opaque FFI data pointer with ownership semantics.
/// Only the original owner (where `destructor` is `Some`) will free the memory on drop.
/// Clones get a shallow copy with `destructor: None`.
pub struct ForeignUserData {
  pub ptr: *mut core::ffi::c_void,
  pub destructor: Option<unsafe extern "C" fn(*mut core::ffi::c_void)>,
}

// SAFETY: The pointee is exclusively owned by the ForeignUserData (one owner via destructor flag).
// The holder (UpdateComponent) is itself accessed under ECS synchronisation rules.
unsafe impl Send for ForeignUserData {}
unsafe impl Sync for ForeignUserData {}

impl Clone for ForeignUserData {
  fn clone(&self) -> Self {
    // Shallow clone: the clone does NOT own the allocation, so destructor is None.
    Self {
      ptr: self.ptr,
      destructor: None,
    }
  }
}

impl Drop for ForeignUserData {
  fn drop(&mut self) {
    if let Some(dtor) = self.destructor {
      if !self.ptr.is_null() {
        // SAFETY: caller guarantees the destructor matches the allocation behind ptr.
        unsafe { dtor(self.ptr) };
        self.ptr = core::ptr::null_mut();
      }
    }
  }
}

impl core::fmt::Debug for ForeignUserData {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("ForeignUserData")
      .field("ptr", &self.ptr)
      .field("has_destructor", &self.destructor.is_some())
      .finish()
  }
}

#[derive(Clone)]
pub struct UpdateComponent {
  pub entities: [Option<EntityId>; 4],
  pub arbitrary_data: [f64; 4],
  pub user_data: Option<ForeignUserData>,
  pub callback: fn(
    EntityId,
    &Scene,
    &mut [Option<EntityId>; 4],
    &mut [f64; 4],
    Option<&mut ForeignUserData>,
    f32,
  ),
}

impl core::fmt::Debug for UpdateComponent {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("UpdateComponent")
      .field("entities", &self.entities)
      .field("arbitrary_data", &self.arbitrary_data)
      .field("user_data", &self.user_data)
      .field("callback", &"<fn>")
      .finish()
  }
}

impl Component for UpdateComponent {}
