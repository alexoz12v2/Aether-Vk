import re

with open("physics.rs", "r") as f:
    content = f.read()

# Fix the method signature
content = content.replace("GpuResult<VulkanBuffer<crate::gpu::CollisionPair>>", "crate::types::EngineResult<VulkanBuffer<crate::gpu::CollisionPair>>")

# Add the struct NarrowCcdPushConstants if it's missing
if "struct NarrowCcdPushConstants" not in content:
    struct_def = """
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NarrowCcdPushConstants {
  pub scene_entities: u64,
  pub output_list: u64,
  pub particles: u64,
  pub pair_buffer: u64,
  pub dt: f32,
  pub particle_radius: f32,
}

"""
    content = struct_def + content

with open("physics.rs", "w") as f:
    f.write(content)
