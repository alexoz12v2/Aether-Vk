#pragma once

#include "os/avk-core-macros.h"
#include "render/vk/common-vk.h"
#include "utils/mixins.h"

namespace avk::vk {

class Instance : public NonMoveable {
 public:
  /// Creates a vulkan instance and, if necessary and supported, a messenger
  Instance();
  ~Instance() noexcept;

  inline VkInstance handle() const { return m_instance; }
  inline operator bool() const { return m_instance != VK_NULL_HANDLE; }
  inline bool portabilityEnumeration() const {
    return m_portabilityEnumeration;
  }
#ifdef AVK_DEBUG
  inline bool debugReport() const { return m_debugReport; }
#else
  inline bool debugReport() const { return false; }
#endif
  inline uint32_t vulkanApiVersion() const { return VK_API_VERSION_1_1; }

 private:
  VkInstance m_instance = VK_NULL_HANDLE;
  bool m_portabilityEnumeration = false;
#ifdef AVK_DEBUG
  bool m_debugReport = false;
  VkDebugReportCallbackEXT m_debugCallback = VK_NULL_HANDLE;
#endif
};

}  // namespace avk::vk
