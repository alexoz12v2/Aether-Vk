use crate::gpu::GpuResourceHandle;
use crate::types::GpuResult;

// ----------------------------------------------------------------------------
//                          COMPUTE ABSTRACTION
// ----------------------------------------------------------------------------

/// A handle to a compute pipeline.
pub type ComputePipelineHandle = GpuResourceHandle;

/// Parameters for a compute dispatch.
pub struct DispatchParams {
  pub pipeline: ComputePipelineHandle,
  pub group_count_x: u32,
  pub group_count_y: u32,
  pub group_count_z: u32,
  // This would also include descriptor sets for the compute shader.
}

/// A backend for executing compute tasks.
pub trait ComputeBackend: Send + Sync {
  /// Dispatches a compute job.
  fn dispatch(&self, params: &DispatchParams) -> GpuResult<()>;
}

/// A frontend for managing and submitting compute work.
pub trait ComputeFrontend {
  /// Submits a compute task to the backend.
  fn submit_task(&mut self, params: DispatchParams) -> TaskHandle;
}

/// A handle to a task in the task graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TaskHandle(u64);
