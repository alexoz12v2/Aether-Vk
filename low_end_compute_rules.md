# Low-End Hardware Compute Shader Compatibility Rules

Per the Vulkan 1.1 specification, the absolute minimum guaranteed limit for `maxComputeSharedMemorySize` across all conformant devices is 16,384 bytes (16 KB) per workgroup.

If your goal is to write compute shaders with guaranteed 100% compatibility across low-end mobile devices, older integrated GPUs, and embedded systems running Vulkan 1.1, you must keep your total workgroup shared variables (the Workgroup storage class in SPIR-V or shared memory in GLSL) strictly under this 16 KB threshold.

## Crucial Companion Limits for Low-End Hardware
Shared memory capacity is only one piece of the compatibility puzzle. When targeting the lowest common denominator, your compute shader dispatches and local thread configurations must also respect the minimum guaranteed hardware limits.

Here are the other essential Vulkan 1.1 compute limits you need to build around to prevent crashes on low-end hardware:

- **maxComputeSharedMemorySize**: 16,384 bytes (16 KB) - Total size of all shared variables in a single workgroup.
- **maxComputeWorkGroupInvocations**: 128 - The total number of threads in a workgroup. The product of your local sizes (`local_size_x * local_size_y * local_size_z`) cannot exceed 128.
- **maxComputeWorkGroupSize**: [128, 128, 64] - The maximum allowed value for each individual dimension in your `layout(local_size_x, local_size_y, local_size_z)` declaration.
- **maxComputeWorkGroupCount**: [65535, 65535, 65535] - The maximum number of workgroups you can dispatch in a single `vkCmdDispatch` call per dimension.

**Important Note on Invocations:** While `maxComputeWorkGroupSize` allows an X-dimension of 128, you cannot dispatch a [128, 128, 1] workgroup on low-end hardware. Doing so results in 128 x 128 x 1 = 16,384 total threads, which vastly exceeds the minimum guaranteed `maxComputeWorkGroupInvocations` limit of 128. A safe 2D workgroup size for universal compatibility is 8x8 (since 8 x 8 = 64, which is well under 128).

## Best Practices for Low-End Compute Shaders
- **Avoid Pushing the 16 KB Limit:** Maxing out shared memory limits your GPU's occupancy (the number of workgroups the hardware can run concurrently on a single compute unit), leading to stalled resources and poor performance.
- **Use Subgroup Operations (When Available):** Vulkan 1.1 made subgroup operations a core API feature. Depending on the device's subgroupSize (often 16, 32, or 64 on mobile), you can share data across threads using subgroup intrinsics (like `subgroupAdd` or `subgroupBroadcast`) without touching shared memory. This is fundamentally faster and bypasses the 16 KB bottleneck.
- **Watch the Padding:** The 16 KB limit includes any invisible padding added by the compiler. Tightly pack your shared structs and arrays to avoid wasting precious bytes.