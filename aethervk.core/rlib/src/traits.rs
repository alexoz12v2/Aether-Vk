use crate::types::{ EngineResult, RuntimeParams };

pub trait InitWithRuntime<T> {
  fn init_with_runtime(params: &RuntimeParams) -> EngineResult<T>;
}