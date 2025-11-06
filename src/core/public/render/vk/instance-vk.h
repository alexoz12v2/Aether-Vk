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
  inline bool debugUtils() const { return m_debugUtils; }
  inline uint32_t vulkanApiVersion() const { return VK_API_VERSION_1_1; }

 private:
  VkInstance m_instance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT m_messenger = VK_NULL_HANDLE;
  bool m_portabilityEnumeration = false;
  bool m_debugUtils = false;
};

}  // namespace avk::vk
