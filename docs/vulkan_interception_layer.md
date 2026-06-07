# Vulkan In-App Interception Layer

## Overview
The Aether-Vk Vulkan backend implements a custom, in-app validation and interception layer. Because the application utilizes a custom lock-tracking system (`DebugTrackedMutex` and `DebugTrackedRwLock`) in a `#![no_std]` environment, standard Khronos validation layers cannot introspect the application's internal thread-local lock state. 

This interception layer serves to intercept core Vulkan API calls right before they hit the driver to assert that **no global locks are held**, preventing subtle deadlocks.

## Architecture

The interception layer is conditionally compiled using `#[cfg(any(debug_assertions, test))]`. Release builds incur zero overhead as they bypass the hooks entirely.

### 1. Intercepting Device Loading
Instead of relying on the standard `ash::Device::load`, the application uses a custom loader `hooks::load_device_with_hooks`. This function utilizes `ash::Device::load_with` to intercept the `get_device_proc_addr` query. When `ash` attempts to load specific Vulkan function pointers, the loader swaps the real driver pointers with our custom `extern "system"` thunks.

### 2. Handle Tracking (`spin::Mutex` + `BTreeMap`)
To call the original driver functions, we need the correct `vk::Device`-specific function pointer. However, dispatchable handles like `vk::Queue` and `vk::CommandBuffer` do not natively expose their parent `vk::Device`.
- We maintain global `spin::Mutex<BTreeMap<Handle, vk::Device>>` maps for queues and command buffers. 
- *Note:* Raw `spin::Mutex` is used instead of our custom tracked locks to prevent recursive lock-checking deadlocks inside the interception layer itself.
- Dedicated lifecycle hooks for `vkGetDeviceQueue`, `vkAllocateCommandBuffers`, and `vkFreeCommandBuffers` automatically keep these tracking maps up to date.

### 3. The `define_hook!` Macro
The majority of the hooks are generated using the `define_hook!` macro. For each intercepted Vulkan function, this macro creates:
- A storage table for the real function pointers (`..._PTRS`).
- An optional, mutable static function pointer (`..._HOOK`) designed for unit testing.
- The `extern "system"` thunk (`hooked_...`) which:
  1. Calls `assert_no_locks_held()`.
  2. Executes the test hook if one is registered.
  3. Looks up the parent `vk::Device` using the `DispatchableToDevice` trait.
  4. Dispatches the call to the original driver function pointer.

### 4. Testability and Dynamic Hooks
Because the macro generates an `Option<fn(...)>` hook for every intercepted command (e.g., `vkQueueSubmit_HOOK`), unit tests can dynamically inject standard Rust callbacks into the Vulkan pipeline. This allows developers to easily mock behavior, count API calls, or validate arguments at runtime without modifying the core renderer logic.

## Supported Commands
Currently, the layer intercepts the following APIs:
- **Queues & Sync:** `vkQueueSubmit`, `vkQueueWaitIdle`, `vkDeviceWaitIdle`
- **Memory/Resources:** `vkCreateBuffer`, `vkDestroyBuffer`, `vkCreateImage`, `vkDestroyImage`
- **Recording:** `vkCmdDraw`, `vkCmdDrawIndexed`, `vkCmdBindPipeline`, `vkCmdDispatch`
- **Lifecycle:** `vkGetDeviceQueue`, `vkAllocateCommandBuffers`, `vkFreeCommandBuffers`

To add a new intercepted command, simply add a `define_hook!` invocation in `hooks.rs` and register the string name match inside `load_device_with_hooks`.
