# Vulkan Notes

## Deprecation guidelines

<https://github.com/KhronosGroup/Vulkan-Guide/blob/main/chapters/deprecated.adoc>

## Useful Guides

- Image Copy: <https://docs.vulkan.org/guide/latest/image_copies.html>
- Android Setup: <https://docs.vulkan.org/tutorial/latest/14_Android.html>

## Support Range

Promoted version to the left

- Baseline: Xiaomi 22126RN91Y (Mali-G52 MC2) <https://vulkan.gpuinfo.org/displayreport.php?id=35260#device>
  - only Vulkan core 1.0 (API version 1.1, but features only from 1.0 and extensions)
  - Instance Extensions:
    - (1.1) supports `VK_KHR_get_physical_device_properties2` (`vkGetPhysicalDeviceFeatures2KHR`, ...)
    - (n) supports `VK_KHR_get_surface_capabilities2` (`vkGetPhysicalDeviceSurfaceCapabilities2KHR`, `vkGetPhysicalDeviceSurfaceFormats2KHR`)
    - (KHR) supports `VK_EXT_surface_maintenance1` (`VkSurfacePresentModeEXT`, `VkSurfacePresentScalingCapabilitiesEXT`)
    - usual ones (surface, debug report)
    - Android Only: `VK_GOOGLE_surfaceless_query` -> `VK_NULL_HANDLE` on surface queries since the presentation engine is only the device
  - Extensions:
    - supports the usual swapchain
    - compute shaders
      - supports `VK_KHR_shader_subgroup_extended_types` (Non Uniform Group Operations in SPIR-V to support 8-bit integer, 16-bit integer, 64-bit integer, 16-bit floating-point)
      - supports `VK_EXT_subgroup_size_control` (1.3) control subgroup size
    - Memory
      - supports `VK_KHR_buffer_device_address` (depends on `VK_KHR_device_group`) (promoted 1.2) Shaders can access buffers through a 64-bit address through the `PhysicalStorage` Buffer class with `SPV_KHR_physical_storage_buffer` SPIR-V Extension
        - Guide: <https://docs.vulkan.org/guide/latest/buffer_device_address.html>
      - supports `VK_KHR_bind_memory2`
      - supports `VK_KHR_dedicated_allocation`
    - Multisampled Depth Stencil
      - supports `VK_KHR_create_renderpass2`
      - supports `VK_KHR_depth_stencil_resolve`
    - Exporting primitives or memory outside vulkan
      - (1.1) supports `VK_KHR_external_fence`
      - supports `VK_KHR_external_fence_fd` (or `_win32`)
      - (1.1) supports `VK_KHR_external_memory`
      - supports `VK_KHR_external_memory_fd` (or `_win32`)
      - (1.1) supports `VK_KHR_external_semaphore`
      - supports `VK_KHR_external_semaphore_fd`(or `_win32`)
      - Linux Kernel only (android/linux): `VK_EXT_external_memory_dma_buf`
      - Android only: `VK_ANDROID_external_memory_android_hardware_buffer`
      - Apple MacOS/ios: `VK_EXT_external_memory_metal`
    - Maintenance or Shaders
      - (1.1) supports `VK_KHR_maintenance1`
      - (1.1) supports `VK_KHR_maintenance2`
      - (1.1) supports `VK_KHR_maintenance3`
      - (1.2) supports `VK_KHR_shader_float_controls`
      - (1.2) supports `VK_KHR_spirv_1_4`
      - (1.2) supports `VK_KHR_uniform_buffer_standard_layout`
      - (1.2) supports `VK_KHR_vulkan_memory_model` (uses `SPV_KHR_vulkan_memory_model` as SPIR-V Dependency)
      - (1.1) supports `VK_KHR_shader_draw_parameters` (uses `SPV_KHR_shader_draw_parameters` as SPIR-V Dependency)
      - (1.3) supports `VK_EXT_inline_uniform_block`
    - (1.2) supports `VK_KHR_timeline_semaphore`
  - Features
    - `depthBiasClamp`
    - `geometryShader`
    - `tessellationShader`
    - `sampleRateShading`
    - `drawIndirectFirstInstance`
    - `timelineSemaphore`
    - `vulkanMemoryModel`
    - `uniformBufferStandardLayout`
    - `shaderSubgroupExtendedTypes`
    - `shaderDrawParameters`
    - `fragmentStoresAndAtomics` (atomic operations on fragment shader)
    - `bufferDeviceAddress`
    - `subgroupSizeControl` (`VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` in `pNext` of `VkPipelineShaderStageCreateInfo`)
    - `imageCubeArray` (`VK_IMAGE_VIEW_TYPE_CUBE_ARRAY` image type allowed)
    - if array of samplers or storage buffers are to indexed with a non constant expression
      - `shaderSampledImageArrayDynamicIndexing`
      - `shaderStorageBufferArrayDynamicIndexing`
      - `shaderStorageImageArrayDynamicIndexing`
      - `shaderUniformBufferArrayDynamicIndexing`
    - `inlineUniformBlock`
  
- Baseline, only if needed in future
  - Extensions
    - (KHR->1.4) `VK_EXT_line_rasterization` for [Stippled Line Rasterization](https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-lines-stipple)
      - Might be useful for precise trajectories?

- Variable Line (might be absent on both baseline and topline), only if needed in future
  - Texture Comporession

- "Topline" -> Maximum support for our devices
  - Extensions
    - supports `VK_EXT_swapchain_maintenance1` (promoted to KHR) for Present Fences
      - From Docs:
        ```The application can destroy the wait semaphores associated with a given presentation operation when at least one of the associated fences is signaled, and can destroy the swapchain when the fences associated with all past presentation requests referring to that swapchain have signaled.```
    - supports `VK_KHR_dynamic_rendering` To create rendering procedures without render passes
    - supports `VK_KHR_copy_commands2`
  - Features
    - `swapchainMaintenance1`
    - `synchronization2`

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

## Descriptors

- Reference Link: <https://docs.vulkan.org/tutorial/latest/05_Uniform_buffers/00_Descriptor_set_layout_and_buffer.html>

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

*Windows* Powershell (not core) command for apps inside

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
