# Vulkan Notes

## Deprecation guidelines

<https://github.com/KhronosGroup/Vulkan-Guide/blob/main/chapters/deprecated.adoc>

## Useful Guides

- Image Copy: <https://docs.vulkan.org/guide/latest/image_copies.html>
- Android Setup: <https://docs.vulkan.org/tutorial/latest/14_Android.html>

## Support Range

Promoted version to the left

**NOTE**: Of the following baseline, use only what you need right **now**

- Baseline: Xiaomi 22126RN91Y (Mali-G52 MC2) <https://vulkan.gpuinfo.org/displayreport.php?id=35260#device>
  - only Vulkan core 1.0 (API version 1.1, but features only from 1.0 and extensions)
  - Instance Extension on MacOS/iOS/iPadOS: `VK_KHR_portability_enumeration`
  - Device Extension on MacOS/iOS/iPadOS: `VK_KHR_portability_subset`
  - Instance Extensions:
    - (1.1) supports `VK_KHR_get_physical_device_properties2` (`vkGetPhysicalDeviceFeatures2KHR`, ...)
    - (n) supports `VK_KHR_get_surface_capabilities2` (`vkGetPhysicalDeviceSurfaceCapabilities2KHR`, `vkGetPhysicalDeviceSurfaceFormats2KHR`)
    - (KHR) supports `VK_EXT_surface_maintenance1` (`VkSurfacePresentModeEXT`, `VkSurfacePresentScalingCapabilitiesEXT`)
    - usual ones (surface, debug report)
    - Android Only: `VK_GOOGLE_surfaceless_query` -> `VK_NULL_HANDLE` on surface queries since the presentation engine is only the device
  - Extensions:
    - supports the usual swapchain
    - `VK_KHR_descriptor_update_template`
    - compute shaders
      - supports `VK_KHR_shader_subgroup_extended_types` (Non Uniform Group Operations in SPIR-V to support 8-bit integer, 16-bit integer, 64-bit integer, 16-bit floating-point)
      - supports `VK_EXT_subgroup_size_control` (1.3) control subgroup size
    - Memory
      - supports `VK_KHR_buffer_device_address` (depends on `VK_KHR_device_group`) (promoted 1.2) Shaders can access buffers through a 64-bit address through the `PhysicalStorage` Buffer class with `SPV_KHR_physical_storage_buffer` SPIR-V Extension
        - Guide: <https://docs.vulkan.org/guide/latest/buffer_device_address.html>
      - supports `VK_KHR_bind_memory2`
      - (1.1) supports `VK_KHR_dedicated_allocation`
    - Multisampled Depth Stencil
      - (1.2) supports `VK_KHR_create_renderpass2`
      - (1.2) supports `VK_KHR_depth_stencil_resolve`
    - Exporting primitives or memory outside vulkan
      - (1.1) supports `VK_KHR_external_fence`
      - supports `VK_KHR_external_fence_fd` (or `_win32`)
      - (1.1) supports `VK_KHR_external_memory`
      - supports `VK_KHR_external_memory_fd` (or `_win32`)
      - (1.1) supports `VK_KHR_external_semaphore`
      - supports `VK_KHR_external_semaphore_fd`(or `_win32`)
      - Linux Kernel only (android/linux): `VK_EXT_external_memory_dma_buf`
      - Android only:
        - `VK_EXT_queue_family_foreign`
        - `VK_ANDROID_external_memory_android_hardware_buffer`
      - Apple MacOS/ios: `VK_EXT_external_memory_metal`
    - Maintenance or Shaders
      - (1.1) supports `VK_KHR_maintenance1`
      - (1.1) supports `VK_KHR_maintenance2`
      - (1.1) supports `VK_KHR_maintenance3`
      - (1.2) supports `VK_KHR_shader_float_controls` (not now)
      - (1.2) supports `VK_KHR_spirv_1_4`
      - (1.2) supports `VK_KHR_uniform_buffer_standard_layout`
      - (1.2) supports `VK_KHR_vulkan_memory_model` (uses `SPV_KHR_vulkan_memory_model` as SPIR-V Dependency)
      - (1.1) supports `VK_KHR_shader_draw_parameters` (uses `SPV_KHR_shader_draw_parameters` as SPIR-V Dependency)
      - (1.3) supports `VK_EXT_inline_uniform_block`
    - (1.2) supports `VK_KHR_timeline_semaphore`
  - Features
    - `depthBiasClamp` (not now)
    - `sampleRateShading` (not now)
    - `drawIndirectFirstInstance` (not now)
    - `timelineSemaphore`
    - `vulkanMemoryModel`
    - `uniformBufferStandardLayout`
    - `shaderSubgroupExtendedTypes`(not now)
    - `shaderDrawParameters`(not now)
    - `fragmentStoresAndAtomics` (atomic operations on fragment shader) (not now)
    - `bufferDeviceAddress`
    - `subgroupSizeControl` (not now) (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` in `pNext` of `VkPipelineShaderStageCreateInfo`)
    - `imageCubeArray` (`VK_IMAGE_VIEW_TYPE_CUBE_ARRAY` image type allowed) (not now)
    - if array of samplers or storage buffers are to indexed with a non constant expression
      - `shaderSampledImageArrayDynamicIndexing`(not now)
      - `shaderStorageBufferArrayDynamicIndexing`(not now)
      - `shaderStorageImageArrayDynamicIndexing`(not now)
      - `shaderUniformBufferArrayDynamicIndexing`(not now)
    - `inlineUniformBlock`
- Baseline, only if needed in future
  - Extensions
    - (KHR->1.4) `VK_EXT_line_rasterization` for [Stippled Line Rasterization](https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-lines-stipple)
      - Might be useful for precise trajectories?

Note that we won't be using

- `geometryShader`
- `tessellationShader`

Because they run worse on Tile-Based GPUs (Mobile, Apple Silicon, Older Desktop GPUs)

- Variable Line (might be absent on both baseline and topline), only if needed in future

  - Texture Comporession

- "Topline" -> Maximum support for our devices

  - Extensions
    - supports `VK_EXT_swapchain_maintenance1` (promoted to KHR) for Present Fences
      - From Docs:
        `The application can destroy the wait semaphores associated with a given presentation operation when at least one of the associated fences is signaled, and can destroy the swapchain when the fences associated with all past presentation requests referring to that swapchain have signaled.`
    - supports `VK_KHR_dynamic_rendering` To create rendering procedures without render passes
    - supports `VK_KHR_copy_commands2`
  - Features
    - `swapchainMaintenance1`
    - `synchronization2`
  - Mobile Only: `VK_EXT_image_compression_control` for AFRC Framebuffer compression

- Desktop only features (and Additional Vulkan extensions/features they require)
  - Video Encoding? Is it necessary?
  - Image Processing?
  - **IN PROGRESS**
    - For Windows < 11: `VK_EXT_full_screen_exclusive` (disable DWM composition for the monitor occupied by your vulkan surface)
      - it's necessary to check whether the extension is available or not
      - when creating the swapchain, use as `pNext` `VkSurfaceFullScreenExclusiveInfoEXT` Always
      - when querying supported present modes, use `vkGetPhysicalDeviceSurfacePresentModes2EXT`
      - to transition to exclusive fullscreen, use `vkAcquireFullScreenExclusiveModeEXT`
      - to exit from exclusive fullscreen, use `vkReleaseFullScreenExclusiveModeEXT`

### Queues and Queue Family Usages

Links to relevant devices

- Mali G52: <https://vulkan.gpuinfo.org/displayreport.php?id=35260#queuefamilies>
- Intel UHD 770: <https://vulkan.gpuinfo.org/displayreport.php?id=43615#queuefamilies>

The only safest, simplest way, is to use a single queue setup from family 0, which
is guaranteed to support GRAPHICS, COMPUTE, TRANSFER

**When To use staging Buffers?**: When it's not a SoC chip (or not integrated?)

- Link: <https://stackoverflow.com/questions/44940684/vulkan-buffer-memory-management-when-do-we-need-staging-buffers>

## Descriptors

- Reference Link: <https://docs.vulkan.org/tutorial/latest/05_Uniform_buffers/00_Descriptor_set_layout_and_buffer.html>

## Why do we need only 1 Depth image (or any other) and not as many as frames in flight

<https://www.reddit.com/r/vulkan/comments/1kt2nei/why_do_you_only_need_a_single_depth_image/>

If I have a maximum of 3 FIF, and a render pass cannot asynchronously write to the same image,
then why is it that we only need a single depth image? It doesn't seem to make much sense,
since the depth buffer is evaluated not at presentation time, but at render time.

Regardless of how many fif you have, you will almost always be rendering a single frame
at a time on the gpu.

Fif is about being able to record more frames while the current one is exexuting on the gpu,
not literally drawing that many frames at once. Thus, gpu only resources like depth and color
buffers do not need to be duplicated per fif, its only resources that the cpu touches
(command buffers, sync objects, descriptor sets) that you duplicate so that the cpu can
touch one copy while the gpu is using a different copy to avoid sync hazards.

The goal will be to only duplicate command dependent resources per command swap
(2x descriptor sets, staging buffers ) and a single one for everything else which would include
almost all attachments and almost all buffers.

There are a few instances in which you WOULD want to double up resources,
like if you are doing some effect that depends on last frame's data.

**Takeaway**: If the CPU Doesn't change it from one frame in flight onto the next, and we
don't need it in the next frame as input, any GPU resource must be allocated **per queue**,
not

## Types of Buffer

| Buffer Type                                                | Main Vulkan Usage Flags                                      | Typical Memory                                            | CPU Access   | Use-cases                                               | Worth its own class?                        |
| ---------------------------------------------------------- | ------------------------------------------------------------ | --------------------------------------------------------- | ------------ | ------------------------------------------------------- | ------------------------------------------- |
| **VertexBuffer**                                           | `VK_BUFFER_USAGE_VERTEX_BUFFER_BIT`                          | `TRANSFER_DST`                                            | Device-local | (via staging) Holds vertex attributes                   | ✅ Yes — `VertexBuffer` wrapper for clarity |
| **IndexBuffer**                                            | `VK_BUFFER_USAGE_INDEX_BUFFER_BIT`                           | `TRANSFER_DST`                                            | Device-local | Holds mesh indices (uint16/uint32)                      | ✅ Yes (thin wrapper around `BufferVk`)     |
| **UniformBuffer (UBO)**                                    | `VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT`                         | Host-visible (dynamic) or device-local (static + staging) | Frequent     | Camera matrices, per-object params                      | ✅ Yes — or via `UniformRingBuffer`         |
| **StorageBuffer (SSBO)**                                   | `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`                         | Device-local or host-visible                              | Sometimes    | Arbitrary structured data, compute I/O, GPU culling     | ✅ Optional — could reuse generic buffer    |
| **IndirectBuffer**                                         | `VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT`                        | Device-local                                              | Rare         | `vkCmdDrawIndirect`, `vkCmdDispatchIndirect` parameters | ⚙️ Usually a generic `BufferVk`             |
| **StagingBuffer**                                          | `TRANSFER_SRC` or `TRANSFER_DST`                             | Host-visible, coherent                                    | Always       | Temporary upload/download buffer                        | ✅ Yes — small utility class or pool        |
| **PixelBuffer / ImageStagingBuffer**                       | `TRANSFER_SRC` or `TRANSFER_DST`, sometimes `STORAGE_BUFFER` | Host-visible                                              | Yes          | Texture uploads, readbacks                              | ⚙️ Can be same as `StagingBuffer`           |
| **StreamingBuffer**                                        | `VERTEX_BUFFER` / `UNIFORM_BUFFER` / `STORAGE_BUFFER`        | Host-visible, coherent                                    | Yes          | Frequently updated data (particles, dynamic meshes)     | ✅ Yes — specialized “dynamic buffer”       |
| **Counter/AtomicBuffer**                                   | `STORAGE_BUFFER_BIT`                                         | Device-local                                              | No           | Counters, prefix sums, append/consume buffers           | ⚙️ Generic SSBO                             |
| **AccelerationStructureBuffer** _(Ray tracing)_            | `ACCELERATION_STRUCTURE_STORAGE_BIT_KHR`                     | Device-local                                              | No           | BLAS/TLAS build data                                    | ⚙️ Special — if you add ray tracing later   |
| **DescriptorBuffer** _(if using VK_EXT_descriptor_buffer)_ | `DESCRIPTOR_BUFFER_BIT_EXT`                                  | Device-local or host-visible                              | No           | Holds serialized descriptors                            | ⚙️ Later feature, can reuse generic buffer  |
| **ShaderDeviceAddressBuffer**                              | `SHADER_DEVICE_ADDRESS_BIT`                                  | Device-local                                              | No           | GPU-side pointer access                                 | ⚙️ Usually just a flag on any buffer        |
| **UniformRingBuffer**                                      | `UNIFORM_BUFFER_BIT`                                         | Host-visible                                              | Yes          | Per-frame suballocations for uniforms                   | ✅ Yes — very useful utility                |

### What is a pixel buffer

A **Pixel buffer** is a portion of memory for raw image/texture data storage (it's not an image)
used for _Pixel Transfer Operations_ (via `vkCmdCopyImageToBuffer`, for example)

- Streaming of mip levels/subimages
- Readback of stream buffers for CPU side processing or post processing
- GPU side processing via compute shaders using storage buffers

Hence it should have `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` and/or `DST_BIT`. If you want to
use them on shaders, use `VK_BUFFER_STORAGE_BUFFER_BIT`.

For host side processing without copying, using Host-Visible, Host-Coherent memory with an
export mechanism is the best option

### What is a streaming buffer

Buffer storing dynamic verted data, instance data, uniform data

- Dynamic Geometry (particle systems, dynamic LODs) (we are interested in the former)
- Per-frame dynamic uniform/storage buffer (camera per-frame data)
- large arrays updated every frame (animations, skinning data)

We have three implementation choices

- Host-Visible memory, such that we can write it every frame host side
  (`vkFlushMappedMemoryRanges` or VMA equivalent if not host coherent)
- _Ring-Allocate_ visible memory to avoid stalls

### What is a Vertex Buffer/Index Buffer

A _Vertex Buffer_ holds per vertex or per instance attributes Used by the graphics pipeline

- Static Meshes: Device-Local Memory + Staging buffer for initial upload
- Dynamic Meshes: Streaming buffer (host visible or via staging)

### What is a Staging Buffer

Temporary host-visible buffer used to upload large data into device-local buffers and images

- Upload mesh/texture at load times or on streaming events
- Readback from devcie-local into staging for CPU side processing

It should be `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` for uploads (`DST_BIT` for downloads) and have
`VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT`

- **Note**: On Integrated GPUs and on mobile devices, where there are less or even one memory heap,
  _The need for a staging buffer is greatly diminished_, and it might even hinder performance
  - Guide Link: <https://www.youtube.com/watch?v=K-2bxdmosH8>
  - Takeaways: Out of all the memory types
    - where most of the ram is (type 0) is usually the fastest
    - if theres a memory heap in which no types are device local, that's actually RAM mapped
      to the device, hence it's really slow
    - host cached memory is slow to write, but faster to read
  - More on memory types:
    - `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT`: This is the fastest memory for the GPU.
      On mobile (Unified Memory Architecture - UMA) systems, this is often shared system RAM,
      but it's still the preferred location for performance-critical resources.
    - `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`: This memory is accessible to the CPU.
      Data needs to be written here and then ideally transferred to device-local memory
      for GPU access if not in a UMA system, though UMA systems can combine both flags.
    - `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`: This is crucial for mobile, tile-based GPUs.
      It can be used for transient attachments (e.g., depth or multi-sampled buffers
      in a render pass) that remain in fast on-chip memory between subpasses and don't
      need to be written back to main memory, saving significant bandwidth

## Mobile and Immediate GPU vs Tiled GPU (and renderapasses)

<https://developer.arm.com/documentation/102662/0100/Tile-based-GPUs>
<https://www.youtube.com/watch?v=BXlo09Kbp2k>
<https://www.youtube.com/watch?v=BD1zXW7Uz8Q>

Focus on vertex and processing stages: in **Immediate Mode GPU** (desktop) vertex processing
happens all at once and stores its results on a GPU FIFO

After all the Geometry is processed, The Fragment Processing can start. It heavily interacts
With framebuffer attachments like _Color_ and _Depth/Stencil_ both reading and writing

- R/W Depth: Depth Testing
- R/W Color: Blending

Since The **On-Screen Framebuffer** (Swapchain image) is in main memory (OS performs compositing
and sends to appropriate device connected to its display), the
**Fragment out R/W operations are expensive and require high bandwidth**

Hence, Mobile devices prefer **Tile Based GPU**, which divide the rasterization pipeline into 2

- Pre-Rasterization Vertex Processing produces Tile result, which are feedback to main memory
- Then, fragment processing happens for each tile separately

This saves bandwidth

Depth Testing on mobile Tiled GPU renderered can use Tile-Local Memory,
since we only need information within the tile. This means that we can request to vulkan
to not create a full memory backing in main memory (_external memory_) for a given
depth/stencil image (`VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT`, `VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT`)

```cpp
VkImageCreateInfo createInfo{};
createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
createInfo.imageType = flags;
createInfo.format = format;
createInfo.extent = extent;
createInfo.samples = sampleCount;
createInfo.usage = VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;

VmaAllocation memory;
VmaAllocationCreateInfo allocInfo{};
allocInfo.usage = memoryUsage;
allocInfo.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;

auto res = vmaCreateImage(vmaAllocator, &createInfo, &allocInfo, &handle, &memory, nullptr);
```

### Renderpasses

The collection of attachments and how they are used define a _RenderPass_.
The processing is divided into multiple _Subpasses_

Each Attachment specifies a **load operation** and a **store operation**

The **Load Operation** specified what should the Vulkan Driver do before rendering to
use the attachment

- `VK_LOAD_OP_LOAD`: Load previous content (**Takeaway: Don't use that**)
- `VK_LOAD_OP_CLEAR`: start anew
- `VK_LOAD_OP_DONT_CARE`: whatever is more optimal

On _Immediate GPUs_, these all perform similiarly. On _Tiled GPUs_, Load is expensive, cause it
needs to stream the whole framebuffer, which defeats the purpose of having tiles at all

- Note: If we clear the attachment, don't do that manually with `vkCmdClearAttachments()`

The **Store Operation** specified what to do after the Renderpass

- `VK_STORE_OP_STORE`: Written back to main memory at the end of the renderpass
  (needed for present images)
- `VK_STORE_OP_DONT_CARE`: It's not going to be used after the renderpass (eg depth/stencil),
  discard that

### Subpasses

In which cases do we need to use more than one subpass? **Deferred Rendering**

- 1st subpass to compute the G-Buffer
  - normals depth and color attachments computed and stored there
- 2nd subpass to compute the lighting
  - use G-Bufer to compute final image. After this, G-Buffer attachemnts are discarded, hence
    - they can be transient and use `LOAD_OP_CLEAR` and `STORE_OP_DONT_CARE`

This can be done only with an optimization called **Subpass Fusion**

- A tile based GPU might process subpasses on a tile-per-tile basis, avoiding unnecessary
  transfers
  - `vkCmdNextSubpass` becomes more optimal, and the G-Buffer becomes effectively transient
  - uses special image type in SPIR-V called `subpassInput`

Between the 2 subpasses there is a dependency, meaning it needs synchronization, which means
**Subpass Dependency** and **Pipeline Barriers**.

**Pipeline Barriers** are synchronization primities which are specified using stage flags, here
are some

- TOP_OF_PIPE: Beginning (helper stage signaling that a command is parsed)
- BOTTOM_OF_PIPE: Beginning (helper stage signaling that a command is being retired)
- VERTEX_INPUT, VERTEX_SHADER: geometry stage
- EARLY_FRAGMENT_TEST, FRAGMENT_SHADER, LATE_FRAGMENT_TEST, COLOR_ATTACHMENT_OUTPUT

Commands are executed out of order across command buffers and across queue submission. Instead
of synchronizing single commands, we split the commands inside the command buffer inserting pipeline
barriers

```cpp
void vkCmdPipelineBarrier(
  VkCommandBuffer              cmdBuf,
  VkPipelineStageFlags         srcStageMask,
  VkPipelineStageFlags         dstStageMask,
  VkDependencyFlags            dependencyFlags,
  uint32_t                     memoryBarrierCount,
  const VkMemoryBarrier*       pMemoryBarriers,
  uint32_t                     bufferMemoryBarrierCount,
  const VkBufferMemoryBarrier* pBufferMemoryBarriers,
  uint32_t                     imageMemoryBarrierCount,
  const VkImageMemoryBarrier*  pImageMemoryBarriers
)
```

- barrier synchronizes everything before it and after it, splitting command buffer in 2
- `srcStageMask` is the stage every command **Before** the barrier should have completed
- `dstStageMask` is the stage every command **After** the barrier will be waiting on until
  stage for previous commands is completed
  Example

```cpp
vkCmdPipelineBarrier(
  commandbuffer,
  VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
  VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
  0, 0, nullptr, 0, nullptr, 1,
  &imageMemoryBarrier
)
```

Indicates that

- All Color attachment output of the previous renderpass should be already completed
- fragment shader of the next renderpass waits for the previous output
- this is the minimal barrier when you don't have any other additional dependencies
- <https://www.youtube.com/watch?v=BXlo09Kbp2k> (11:12, actually 15:56)

### MSAA

GPU rendering can result in jagged edges, and they can be smoothed out using MSAA.

Without MSAA, when a polygon is rasterized, we look at the center of the pixel and rasterize it only if the center of the pixel falls
inside the primitive

With MSAA, we look at multiple samples on the pixel's cell area, and shade each sample which falls inside the primitive

- The fragment shader is still run with the coordinates of the pixel

Then, we combine (**resolve**) all the samples of the results for a given frag coord and get a resulting

The Resolve operation can be done with **Tile-Local Memory**, hence there is no need for writeback to main memory

- Multisample attachment can be _transient_
  - LOAD_OP_CLEAR
  - STORE_OP_DONT_CARE
  - LAZILY_ALLOCATED

**Note**: Resolve images must be specified inside the subpass under `pResolveAttachments` and not manually using `vkCmdResolveImage`

- If you specify the resolve attachment upfront, you writeback to main memory only the resolve result, while if you do it manually
  you writeback to main memory both the multisampled image,

  - which is then passed back to the GPU to perform the resolve operation
  - which is then wrote back to main memory (<https://www.youtube.com/watch?v=BXlo09Kbp2k>, 19:07)

- with `VK_KHR_depth_stencil_resolve` (core in 1.2) you can resolve the Depth/Stencil Attachment too

### CPU Optimization

Draw calls are costly operations. We want a thread-friendly implementation which adds little overhead, batches graphics work, and is steady

The API needs to have

- Command buffer management (create, commit to queue)
- Render Pass Manageemnt (begin, end)
- Pipeline State management (bind program)
- Descriptor Management (bind buffer, buffer data, texture)
- draw/dispatch commands management

#### Command Buffer Management

- Each thread needs its own `VkCommandPool` because command buffer recording is an externally synchronized operation
- `vkALlocateBuffers` is not free, and `vkFreeCommandBuffers` might actually result in a memory free in some implementation, therefore
  - We need to have a Pool of command pools, such that we ensure synchronization
  - `vkQueueSubmit` is a heavy operation, hence we batch command buffer submission, hence when we commit the command buffer we just add the command buffer
    to a wait list for the queue and signal that the `VkCommandPool` is free to use again
    - single submit at the end of the frame with `submitCount = 1`
- After recording a frame we remove all pools with pending command buffers from the command pool
- after a frame has been completed, we can recycle command pools to the global pool after `vkResetCommandPool`

#### Render Passes

The problem here is that upfront, you don't have the "Global view" of the frame, ie you don't know how resources will be used on the future, and you
need to manually insert barriers

- some engines solve that by constructing a render graph

A simpler solution might be to still manually synchronize work, bu specify upfront some typical workload for the renderpass

- set of textures to render (color, depth, resolve color, resolve depth)
- which framebuffer textures need to be loaded from memory
- which framebuffer textures need to be stored to memory
- which framebuffer textures need to be cleared with what initial data
- do we need to do MSAA resolve in end Render Pass? If so, where

`VkRenderPass` and `VKFramebuffer` might be lazily created and cached

To perform **Image Layout Transitions**

- Always perform them **Outside of the graphics scope** (ouside of `vkBeginRenderPass`, `VkEndRenderPass`)
- the API should infer it, hence you should maintain a default state (not current, because doesn't work in multithreaded)
  - default = what state is in between passes
  - Accesses:
    - textures with shader access: `SHADER_READ_ONLY`
    - texture without shader access: `COLOR_ATTACHMENT_OPTIMAL` or `DEPTH_ATTACHMENT_OPTIMAL`
    - read write textures: `GENERAL`
- All image layout transitions should be performed at render pass bounddary: No In-Pass Synchronization
- All image layout transitions are guided by load/store masks
  - image which is not loaded should transition from `UNDEFINED` to `COLOR_ATTACHMENT`
  - image which is not stored is kepth in `COLOR_ATTACHMENT` or `DEPTH_ATTACHMENT`

We also need to infer pipeline barriers between subpasses and renderpasses

- default to fragment to fragment barriers
  - This means that we assume that previously produced textures are used in the fragment shader only
  - if textures are used somewhere else we need to specify it

#### Pipeline State

we lazily create pipeline objects and we have a `VkPipelineCache` object which is serialized to disk and you don't need to recompile its shaders

- Reads and writes inside the cache should be synchronized
- to avoid mutex utilization when searching from the cache, we might have 2 hash tables
  - Read Map is only read-from
  - Write Map is written in a synchronized way
  - the two maps are merged at the end of the frame

#### Descriptor Management

for buffers, we can use a single slot integer specification valid for all stages.
We use at most 2 sets: Buffers and Textures

To Allocate a descriptor set you need to allocate it from a pool. A good compromise is

- A single pool, configured to support the "average" use case in terms of sets, textures, buffers
- Use a pool of pools, such that, similiarly for command pools, each thread can extract its own pool and give it up **at the end of the frame** (not the end of recording)
  - pools are recycled at the end of the frame, never freed (`vkResetDescriptorPool`)

Descriptor sets need to be updated calling on core 1.0 `vkUpdateDescriptorSet`, and this is done **Lazily**

- Use Descriptor Templates if available (saves CPU cost)
- Only update set with changes (eg texture-only changes only need to update 1 set)
- Usually not copying descriptor sets from others

Note that instead of having a buffer for each descriptor set, we want to allocate large buffers and assign **dynamic offsets** to
a descriptor

- works well for most per frame data which is constant, small and dynamic (bindbufferdata)
- instead of allocating a new buffer descriptor every time, we use `pDynamicOffsets`

### Efficient Swapchains

In Mobile Devices, The GPU and DPU (Display Processing Unit) are two different entities on
the System on Chip.

The latter receives the image from `vkQueuePresentKHR` and is able to perform simple composition
before sending it to the display.

The problem is that the DPU can handle

- Rotation, scaling, color conversion, layer composition
- **Or It may not** (depends on hardware capabilities)

Meaning that if the composition is too complex the DPU actually sends back the image to
the GPU, which will handle part of the composition and send back the image to the DPU

To help producing DPU Friendly images (since we cannot query DPU capabilities, assume it
cannot handle rotation)

- Apply pre-rotation as a geometry transform in the application
- Swapchain pre-transform should match surface current transform
- recreate the swapchain when the surface current-transform changes

## Android multithreading

big vs Little threads is important

## WSI

### Windows

Some vulkan functions call the `SendMessage`, hence we need to keep the window procedure active to
not stop it, cause the calling thread blocks until the message has been handled

- `vkCreateSwapchainKHR`
- `vkDestroySwapchainKHR`
- `vkAcquireNextImageKHR and vkAcquireNextImage2KHR`
- `vkQueuePresentKHR`
- `vkReleaseSwapchainImagesKHR`
- `vkAcquireFullScreenExclusiveModeEXT`
- `vkReleaseFullScreenExclusiveModeEXT`

vkSetHdrMetadataEXT

## Windows Notes

- TODO fancy zones

### Links

- [GDI/Graphics](https://learn.microsoft.com/en-us/windows/win32/gdi/nonclient-area)
- [Desktop Window Manager](https://learn.microsoft.com/en-us/windows/win32/dwm/desktop-window-manager-overviews)
- [HWND Handling](https://learn.microsoft.com/en-us/windows/win32/winmsg/window-styles)
- [Modern Window Guide](https://learn.microsoft.com/en-us/windows/apps/desktop/modernize/ui/apply-snap-layout-menu)

## Snippets

How to perform multisampling:

```cpp
VkRenderingAttachmentInfo colorAttachment = {
    .imageView = multisampleColorView,
    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
    .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
    .resolveMode = VK_RESOLVE_MODE_AVERAGE_BIT,
    .resolveImageView = swapchainImageView, // single-sample resolve target
    .resolveImageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
};
```

Refresh window style

```cpp
SetWindowPos(
    hWnd, nullptr/*insert after*/, 0/*x*/, 0/*y*/, 0/*cx*/, 0/*cy*/,
    SWP_NOMOVE | SWP_NOSIZE |  // no position and size params
        SWP_NOZORDER |         // ignore hWndInsertAfter (2nd param)
        SWP_FRAMECHANGED |     // apply styles and send WM_NCCALCSIZE
        SWP_NOOWNERZORDER |    // this is always given
        SWP_SHOWWINDOW);
```

Menu Definition (Non client area content)

```cpp
static int constexpr ID_FILE_OPEN = 1;
HMENU hMenu = CreateMenu();  // DestroyMenu
HMENU hFileSubMenu = CreatePopupMenu();  // DestroyPopupMenu
AppendMenuW(hFileSubMenu, MF_STRING, ID_FILE_OPEN, L"Open"));
AppendMenuW(hMenu, MF_STRING | MF_POPUP, (UINT_PTR)hFileSubMenu, L"File");
```

Blender Win32 window style

```cpp
DWORD style = parent_window ?
WS_POPUPWINDOW | WS_CAPTION | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SIZEBOX
: WS_OVERLAPPEDWINDOW;
/* Forces owned windows onto taskbar and allows minimization. */
DWORD extended_style = parent_window ? WS_EX_APPWINDOW : 0;

if (dialog) {
  /* When we are ready to make windows of this type:
   * style = WS_POPUPWINDOW | WS_CAPTION
   * extended_style = WS_EX_DLGMODALFRAME | WS_EX_TOPMOST
   */
}
```

On `VK_SUBOPTIMAL_KHR` Check true windows size with DWM

```cpp
    if (presentResult == VK_SUBOPTIMAL_KHR) {
#ifdef VK_USE_PLATFORM_WIN32_KHR
      // surface allows variable; use window size converted to pixels
      RECT dwmBounds;
      if (SUCCEEDED(DwmGetWindowAttribute(m_hWindow,
                                          DWMWA_EXTENDED_FRAME_BOUNDS,
                                          &dwmBounds, sizeof(dwmBounds)))) {
        uint32_t visibleW = dwmBounds.right - dwmBounds.left;
        uint32_t visibleH = dwmBounds.bottom - dwmBounds.top;
        if (m_currentExtent.width == visibleW &&
            m_currentExtent.height == visibleH) {
          // std::cout << "Suboptimal is a lie!" << std::endl;
          return ContextResult::Success;
        }
      }
#endif
   }
```

Enable Virtual Console in Windows 10+ (ie. ANSI Escape codes)

```cpp
HANDLE handleOut = GetStdHandle(STD_OUTPUT_HANDLE);
DWORD consoleMode;
GetConsoleMode( handleOut , &consoleMode);
consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
consoleMode |= DISABLE_NEWLINE_AUTO_RETURN;
SetConsoleMode( handleOut , consoleMode );
```

_Windows_ Powershell (not core) command for apps inside

- ```cmd
  %ProgramData%\Microsoft\Windows\Start Menu\Programs
  ```

  whose executable has a **AppUserModelID** registered on the system

  ```powershell
  Get-Startapps | Sort-Object name | Where-Object { $_.Name -like "A*" }
  ```

A Timeline semaphore would exhaust its 64-bit values in years. That said, if you do want
to recreate it, destroy all objects depending on it and wait device idle

```cpp
void Renderer::resetTimelineSemaphore() {
  vkDeviceWaitIdle(device);  // optional if you track GPU completion precisely

  vkDestroySemaphore(device, m_timelineSemaphore, nullptr);

  VkSemaphoreTypeCreateInfo timelineInfo{
    VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO, nullptr,
    VK_SEMAPHORE_TYPE_TIMELINE, 0
  };
  VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
  semInfo.pNext = &timelineInfo;
  vkCreateSemaphore(device, &semInfo, nullptr, &m_timelineSemaphore);

  nextSignalValue = 1;
  lastCompletedValue = 0;
}
```

How to ensure a Dispatch of a compute pipeline covers exacly an image, provided
the shader performs a bound check. Approximate by excess (ceilDiv)

```cpp
uint32_t groupSizeX = 16;
uint32_t groupSizeY = 16;

// Assume you know image width/height
uint32_t imageWidth  = textureWidth;
uint32_t imageHeight = textureHeight;

// Round up: ensure we cover all pixels
uint32_t dispatchX = (imageWidth  + groupSizeX - 1) / groupSizeX;
uint32_t dispatchY = (imageHeight + groupSizeY - 1) / groupSizeY;

// Dispatch
vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, computePipeline);
vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
vkCmdDispatch(cmd, dispatchX, dispatchY, 1);
```
