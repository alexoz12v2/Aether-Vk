#pragma once

// TODO maybe: move to cmake/bazel
#ifdef AVK_OS_WINDOWS
#  define VK_USE_PLATFORM_WIN32_KHR
#elif defined(AVK_OS_MACOS)  // TODO: also iOS and iPadOS
#  define VK_USE_PLATFORM_METAL_EXT
#elif defined(AVK_OS_ANDROID)
#  define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(AVK_OS_LINUX)
#  if defined(AVK_USE_XCB)
#    define VK_USE_PLATFORM_XCB_KHR
#    error "ADD SUPPORT"
#  else
#    define VK_USE_PLATFORM_WAYLAND_KHR
#    error "ADD SUPPORT"
#  endif
#else
#  error "Not Supported"
#endif

// volk/vulkan before VMA
// clang-format off
#ifndef VK_NO_PROTOTYPES
#  error "IF you are using volk, define VK_NO_PROTOTYPES"
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnullability-extension"
#pragma clang attribute push(__attribute__((no_sanitize("cfi"))), \
                             apply_to = any(function))
#include <volk.h>
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0
#include <vk_mem_alloc.h>
#pragma clang attribute pop
#pragma GCC diagnostic pop
// clang-format on

#include <iostream>
#include <string>
#include <vector>

// our stuff
#include "os/avk-log.h"
#include "os/stackstrace.h"
#include "utils/mixins.h"

/*! This check for any vulkan error (< 0, while > 0 is status for some vk calls,
    eg VK_NOT_READY or VK_SUBOPTIMAL_KHR (note: VK_ERROR_OUT_OF_DATE_KHR is not
    a crashing error, so this should be used only in points in which a failure
    is a non recoverable error).
    TODO insert a logging macro? */
#define VK_CHECK(vkres)                                             \
  do {                                                              \
    VkResult const var = (vkres);                                   \
    if (var < 0) {                                                  \
      std::string err = "[Vulkan Error]: ";                         \
      err.append(::avk::vkResToString(var));                        \
      LOGE << AVK_LOG_RED << err << std::endl;                      \
      LOGE << ::avk::dumpStackTrace() << AVK_LOG_RST << std::flush; \
      ::avk::showErrorScreenAndExit(err.c_str());                   \
    }                                                               \
  } while (0)

/// Macro to check whether a boolean is true, otherwise crash.
/// Intended to be used to check for mandatory feature/extension support
#define AVK_EXT_CHECK(val)                      \
  do {                                          \
    if (!(val)) {                               \
      VK_CHECK(VK_ERROR_EXTENSION_NOT_PRESENT); \
    }                                           \
  } while (0)

namespace avk {

std::string vkResToString(VkResult res);

/// Function used when we want to export our memory outside of vulkan
inline VkExternalMemoryHandleTypeFlags externalMemoryVkFlags() {
#ifdef AVK_OS_WINDOWS
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
#elif AVK_OS_ANDROID
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_ANDROID_HARDWARE_BUFFER_BIT_ANDROID;
#elif AVK_OS_MACOS
  // TODO insert metal handles
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#elif AVK_OS_LINUX
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT |
         VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;
#else
  return 0;
#endif
}

/// Helper class acting as a bag for tracking which extensions are supported and
/// which
// are active
struct Extensions {
  std::vector<VkExtensionProperties> extensions;
  std::vector<char const*> enabled;

  inline bool isSupported(char const* name) const {
    for (VkExtensionProperties const& ext : extensions) {
      if (strcmp(ext.extensionName, name) == 0) {
        return true;
      }
    }
    return false;
  }

  template <typename StrIt>
  inline bool isSupported(StrIt&& beg, StrIt&& end) const {
    for (auto it = beg; it != end; ++it) {
      if (!isSupported(*it)) {
        return false;
      }
    }
    return true;
  }

  inline bool enable(char const* name) {
    bool supported = isSupported(name);
    if (supported) {
      enabled.push_back(name);
      // TODO add logging
      return true;
    }
    // TODO add logging
    return false;
  }

  template <typename StrIt>
  inline bool enable(StrIt&& beg, StrIt&& end) {
    bool failure = false;
    if (beg == end) {
      return true;
    }
    for (auto it = beg; it != end; ++it) {
      const char* name = nullptr;
      const auto& val = *it;

      if constexpr (std::is_same_v<std::decay_t<decltype(val)>, const char*>) {
        name = val;
      } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                          std::string>) {
        name = val.c_str();
      } else if constexpr (std::is_same_v<std::decay_t<decltype(val)>,
                                          std::string_view>) {
        name = val.data();
      } else {
        static_assert(sizeof(val) == 0, "Unsupported string type");
      }

      failure |= !enable(name);
    }
    return !failure;
  }

  inline bool isEnabled(char const* name) const {
    for (char const* enabledName : enabled) {
      if (strcmp(enabledName, name) == 0) {
        return true;
      }
    }
    return false;
  }
};

}  // namespace avk

namespace avk::vk {

class Device;

/// Helper type to define Vulkan Device Operations which
/// Shouldn't be necessarily fatal
template <typename H>
struct Expected {
  H handle;
  VkResult result;

  inline operator bool() const { return result >= 0; }
  inline void crashIfErr() const { VK_CHECK(result); }
  inline H get() const {
    VK_CHECK(result);
    return handle;
  }
};

/// Helper type for handles allocated by VMA
template <typename H>
struct VMAResource {
  VMAResource() = default;
  VMAResource(H handle, VmaAllocation alloc) : handle(handle), alloc(alloc) {}
  H handle;
  VmaAllocation alloc;
};

/// Helper function for Discrete GPU paths to see whether a device local
/// allocation is also host visible, hence can be mapped, or whether it
/// requires a staging buffer and a copy operation
bool isAllocHostVisible(VmaAllocator allocator, VmaAllocation allocation);

/// Used to create the handle for a staging buffer of a given size
/// to be bound multiple times with different memory handles with VMA
/// \warning assumes exclusive queue ownership
Expected<VkBuffer> newUnboundStagingBufferHandle(Device* device, size_t size);
}  // namespace avk::vk
