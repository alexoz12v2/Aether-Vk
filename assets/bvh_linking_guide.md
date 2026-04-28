# SPIR-V Shader Linking & BVH Utilities Guide

## 1. SPIR-V Linking and Shared Shaders

Vulkan and SPIR-V historically preferred statically compiling shaders with their dependencies linked together at compile time. However, sharing and linking SPIR-V modules is feasible through a few methods:

### Option A: Compile-Time `#include` (Recommended)
By using `GL_GOOGLE_include_directive` (as enabled in your current shaders), you can inject shared functions at GLSL compilation time:
```glsl
#extension GL_GOOGLE_include_directive : require
#include "bvh_utils.glsl"
```
This is the most robust, highly optimized path because the compiler can aggressively inline and optimize the shared functions for the specific kernel.

### Option B: Offline SPIR-V Linking (`spirv-link`)
SPIR-V tools provide the `spirv-link` binary. You compile your GLSL functions into a library module:
```bash
glslangValidator -V bvh_utils.glsl -o bvh_utils.spv
glslangValidator -V physics_kernel.comp -o physics_kernel.spv
spirv-link bvh_utils.spv physics_kernel.spv -o linked_kernel.spv
```
For this to work, functions in the library must be exported (via linkage linkage attributes), and the consuming shader must declare them as `external`.

### Option C: `VK_EXT_graphics_pipeline_library` / `VK_KHR_pipeline_library` (Vulkan 1.3)
Vulkan 1.3 introduces pipeline libraries, allowing you to compile fragments of pipelines independently and link them fast at runtime. However, this is primarily for vertex/fragment stage combinations, not typically for linking arbitrary compute functions together dynamically.

---

## 2. Architecture: `Kernels` vs. `RenderDevice`

Currently, `Kernels` handles purely compute/physics operations, while `RenderDevice` manages the presentation engine, graphics pipelines, and some rendering-specific compute shaders (`sungen.comp`, `skygen.comp`). 

### Should `sungen.comp` and `skygen.comp` move to `Kernels`?
**Feasibility:** Yes, it is feasible. The `Kernels` trait provides a `Cmd: CommandBuffer` which can execute compute dispatches.
**Suitability:** Moving these to `Kernels` implies separating the generation of rendering assets (like the 3D sun volume) from the rendering pipeline.
- **Pros**: Clean separation of compute logic from drawing logic. `RenderDevice` becomes purely a graphics consumer.
- **Cons**: `sync_compute_to_graphics` (the `KernelRenderBridge`) must manage queue ownership transfers. If `Kernels` runs on an async-compute queue (which your `vulkan::instance` initialization actively searches for), `vkCmdPipelineBarrier` must be used to release the `sunVolume` 3D image from the Compute Queue Family and acquire it on the Graphics Queue Family.
- **Recommendation**: Since `skygen` and `sungen` are highly tied to the visual representation, creating a separate trait (e.g., `RenderCompute`) or keeping them in `RenderDevice` prevents complex queue-ownership transfers. `Kernels` should remain strictly for latency-sensitive physical simulations (like BVH collisions, particles, and rigid body dynamics).

---

## 3. GPU Programming Concepts: From CUDA to Vulkan

To achieve maximum performance in compute shaders (like our BVH traversal), we adapt several low-level GPU programming concepts commonly found in CUDA directly into Vulkan SPIR-V.

### AOS vs SOA vs AOSOA (Memory Coalescing)
- **AOS (Array of Structures)**: The standard object-oriented approach (e.g., `struct Node { vec3 min; vec3 max; }; Node nodes[];`). In CUDA/Vulkan, when 32 threads (a warp/subgroup) read `nodes[i].min.x` sequentially, they hit 32 completely different cache lines scattered across memory. This causes massive memory bandwidth bottlenecks.
- **SOA (Structure of Arrays)**: `struct Nodes { float minX[]; float minY[]; ... };`. Better for coalescing, but terrible for cache locality if a single thread needs `minX`, `minY`, and `minZ` simultaneously, as they are far apart in memory.
- **AOSOA (Array of Structures of Arrays)**: The optimal hybrid. We group data into blocks sized exactly to the hardware's execution width (e.g., `SUBGROUP_SIZE = 32`). 
  ```glsl
  struct BVHNodeBlock { float minX[32]; float minY[32]; ... };
  BVHNodeBlock blocks[];
  ```
  When the subgroup executes `blocks[block_idx].minX[gl_SubgroupInvocationID]`, exactly 32 consecutive floats (128 bytes) are requested. This perfectly aligns with a standard GPU L1/L2 cache line, achieving **100% memory coalescing efficiency**.

### Subgroups (Warps / Wavefronts)
In CUDA, a "Warp" is 32 threads. In AMD/Vulkan, it's a "Wavefront" (typically 64 threads). Vulkan abstracts this as a **Subgroup**.
- `gl_SubgroupInvocationID` maps directly to the thread's lane index within the warp.
- We use Subgroup Intrinsics (`GL_KHR_shader_subgroup_*`) to allow threads within the same subgroup to share data without hitting memory, reducing latency.

### Shared Memory (Workgroup Local Memory)
In CUDA, `__shared__` memory is ultra-fast, user-managed L1 cache. In Vulkan GLSL, this is the `shared` keyword (translating to the `Workgroup` storage class in SPIR-V).
- **Collaborative Loading**: Instead of each thread fetching its own BVH node directly from VRAM (global memory), the entire subgroup collaboratively loads an entire `BVHNodeBlock` from VRAM into `shared` memory in one coalesced read. 
- Once in `shared` memory, individual threads can randomly access any node's data without penalty, bypassing the VRAM bandwidth bottleneck entirely. This is crucial for BVH traversal where divergent branching makes memory access patterns unpredictable.

### Memory Model and Atomics
Vulkan 1.2 introduced the `GL_KHR_memory_scope_semantics` extension, giving us C++11 / CUDA-style atomic memory models.
When writing to the `CollisionPairList`, we use:
```glsl
atomicAdd(outputList.count, 1, gl_ScopeQueueFamily, gl_StorageSemanticsBuffer, gl_SemanticsRelaxed);
```
- `gl_ScopeQueueFamily`: Ensures the atomic operation is visible to all threads across the entire compute queue (not just the local workgroup).
- `gl_SemanticsRelaxed`: Tells the compiler we only care about the atomicity of the counter increment itself, not strictly synchronizing the order of other surrounding memory writes (which avoids expensive memory fences).

---

## 4. SPIR-V Assembly Expectations

When our `bvh_utils.glsl` and compute shaders are compiled to SPIR-V (`.spv`), we expect specific opcodes corresponding to our architectural choices.

### Buffer Layouts & Physical Storage Buffers
Because we use `GL_EXT_buffer_reference` (Physical Storage Buffers), Vulkan uses true 64-bit pointers instead of descriptor bindings.
**Expected SPIR-V:**
```spirv
; PhysicalStorageBuffer class indicates raw pointer access
OpTypeForwardPointer %_ptr_PhysicalStorageBuffer_BVHArray PhysicalStorageBuffer
%BVHArray = OpTypeStruct %_runtimearr_BVHNodeBlockAABB
```
When accessing elements:
```spirv
; OpPtrAccessChain is used instead of OpAccessChain to do pointer arithmetic on 64-bit addresses
%ptr = OpPtrAccessChain %_ptr_PhysicalStorageBuffer_float %base_ptr %offset
%val = OpLoad %float %ptr Aligned 4
```

### Shared Memory (Workgroup Local)
Our `DECLARE_SHARED_BVH_CACHE` macro creates a variable in the `Workgroup` storage class.
**Expected SPIR-V:**
```spirv
; The Workgroup storage class explicitly allocates from the compute unit's shared memory
%_ptr_Workgroup_BVHNodeBlockAABB = OpTypePointer Workgroup %BVHNodeBlockAABB
%my_cache = OpVariable %_ptr_Workgroup_BVHNodeBlockAABB Workgroup
```
And to synchronize the collaborative load:
```spirv
; barrier() translates to OpControlBarrier
OpControlBarrier %uint_2 %uint_2 %uint_264 ; (Workgroup Scope, Workgroup Memory, AcquireRelease)
```

### Atomics
Our highly specified `atomicAdd` translates directly into an atomic instruction with specific memory semantics.
**Expected SPIR-V:**
```spirv
; %int_5 = QueueFamily Scope
; %uint_64 = Relaxed Semantics
%result = OpAtomicIAdd %uint %pointer %int_5 %uint_64 %uint_1
```

### Subgroup Intrinsics
When reading the thread's lane ID (`gl_SubgroupInvocationID`):
```spirv
; Mapped to a BuiltIn
OpDecorate %gl_SubgroupInvocationID BuiltIn SubgroupLocalInvocationId
%id = OpLoad %uint %gl_SubgroupInvocationID
```
