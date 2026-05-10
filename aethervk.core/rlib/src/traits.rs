//! traits module.

use crate::types::{EngineResult, RuntimeParams};

/// TODO: Document this item
pub trait InitWithRuntime<T> {
  fn init_with_runtime(params: &RuntimeParams) -> EngineResult<T>;
}
