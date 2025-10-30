# Vulkan Notes

## Deprecation guidelines

<https://github.com/KhronosGroup/Vulkan-Guide/blob/main/chapters/deprecated.adoc>

## Instance extensions to use

| Extension                                                                                                                                                                  | Type                 | Purpose                                                                                                                                              |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`VK_EXT_debug_utils`**                                                                                                                                                   | *Instance extension* | Provides debug callbacks, object naming, and markers for validation and debugging.                                                                   |
| **`VK_KHR_surface`**                                                                                                                                                       | *Instance extension* | The cross-platform interface between Vulkan and window-system surfaces (the base “surface” extension).                                               |
| **Platform-specific surface extension** (e.g. `VK_KHR_win32_surface`, `VK_KHR_xcb_surface`, `VK_KHR_xlib_surface`, `VK_KHR_wayland_surface`, `VK_EXT_metal_surface`, etc.) | *Instance extension* | Provides the platform-specific glue between Vulkan and the native windowing system. Required to create `VkSurfaceKHR`.                               |
| **`VK_EXT_surface_maintenance1`**                                                                                                                                          | *Instance extension* | Adds quality-of-life improvements and fixes for surface/swapchain behavior across platforms (introduced 2023–2024).                                  |
| **`VK_KHR_get_surface_capabilities2`**                                                                                                                                     | *Instance extension* | Adds extended surface capability queries, necessary for modern swapchain and surface features.                                                       |
| **`VK_EXT_swapchain_maintenance1`**                                                                                                                                        | *Device extension*   | Extends `VK_KHR_swapchain` with maintenance and bugfix features for better frame management (e.g., rect scaling, suboptimal surface handling, etc.). |

## Random Notes

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

apparently, when using aero snap, i've arrived at this situation: 

```sh
GetClientRect: 1936x1056 
GetWindowRect: 976x1056 
GetMonitorInfoW: 1920x1040 
```

And hence the swapchain is incorrect, as it follows the client info

TODO fancy zones