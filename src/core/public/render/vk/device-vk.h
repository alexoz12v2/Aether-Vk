#pragma once

// device lost error:
// ARM GPUs will shader vertices in groups of 4 contiguous indices
//  - can run inexistent indices. Replicate the last vertex such that multiple
//  of 4
// ARM GPUs will shade indices you skipped inside the index buffer, hence create
// a tighly
//   packed index range to minimze over shaidng
// ARM GPUs for intermediate data geometry needs up to 180MB per render pass
// - Midgard and Bifrost GPUs (old Mali architextures) allocate space for
// min_index to
//    max_index and scans for min and max
// - newer Valhall GPU allocates only for visible vertices

#include "render/vk/common-vk.h"
#include "render/vk/instance-vk.h"
#include "render/vk/surface-vk.h"
#include "utils/mixins.h"

// std
#include <memory>

namespace avk::vk::utils {

struct SampledImageCompressedFormats {
  VkFormat sRGB_RGBA = VK_FORMAT_R8G8B8A8_SRGB;
  VkFormat linear_RGB = VK_FORMAT_R8G8B8_UNORM;
  VkFormat linear_R = VK_FORMAT_R8_UNORM;
};

struct QueueFamilyMap {
  /// queue family supporting graphics, compute, transfer and presentation
  uint32_t universalGraphics;
};

}  // namespace avk::vk::utils

namespace avk::vk {

class Device : public NonMoveable {
 public:
  /// Selects a physical device with baseline features and extension support,
  /// creates a device and takes its first queue supporting
  /// - graphics, compute, transfer, presentation
  /// TODO: Decouple surface for desktop, and don't take it for mobile (android
  /// and metal mandate
  ///  that each graphics queue family is present capable)
  Device(Instance* instance, Surface* surface);
  /// `vkDeviceWaitIdle`, `vmaDestroyAllocator`, `vkDestroyDevice`
  ~Device() noexcept;

  /// Recomputes `vmaGetHeapBudgets` to update memory usage and budget for
  /// each memory heap in the system. When the `VmaAllocator` is used by
  /// multiple threads, such information gets outdated quickly
  void refreshMemoryBudgets(uint32_t frameIndex);
  inline std::vector<VmaBudget> const& heapBudgets() const {
    return m_heapBudgets;
  }

  inline VkPhysicalDevice physicalDevice() const { return m_physicalDevice; }
  inline VkDevice device() const { return m_device; }
  inline VmaAllocator vmaAllocator() const { return m_vmaAllocator; }
  inline VkQueue queue() const { return m_queue; }
  inline uint32_t universalGraphicsQueueFamilyIndex() const {
    return m_queueFamilies.universalGraphics;
  }
  inline VolkDeviceTable const* table() const { return m_table.get(); }

  inline bool swapchainMaintenance1() const { return m_swapchainMaintenance1; }
  inline utils::SampledImageCompressedFormats compressedSampledImageFormat()
      const {
    return m_comprFormats;
  }

  inline operator bool() const {
    return m_device && m_table && m_vmaAllocator && m_queue;
  }

  inline bool isSoC() const { return m_isSoC; }

 private:
  // dependencies which must outlive this object
  struct Deps {
    Instance* instance;
    Surface* surface;
  } m_deps;

  // Handles
  VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
  VkDevice m_device = VK_NULL_HANDLE;
  // TODO Memory pools when work size becomes known after experimentation
  // VmaPool m_pool = VK_NULL_HANDLE;
  VmaAllocator m_vmaAllocator = VK_NULL_HANDLE;

  /// VMA memory budget for each memory heap. Must be refreshed every frame
  std::vector<VmaBudget> m_heapBudgets;

  // TODO: do we need to keep `VmaVulkanFunctions`?
  std::unique_ptr<VolkDeviceTable> m_table = nullptr;
  std::unique_ptr<VmaVulkanFunctions> m_vmaVulkanFunctions = nullptr;

  // queue related data (*submit externally synchronized*)
  VkQueue m_queue = VK_NULL_HANDLE;
  utils::QueueFamilyMap m_queueFamilies;

  // features tracking
  // -- compressed formats for usage SAMPLED_IMAGE, TRANSFER, BLIT_SRC
  // TODO later: store an array of formats for linear, srgb, and hdr (float)
  utils::SampledImageCompressedFormats m_comprFormats;

  // optional extensions/features tracking
  bool m_swapchainMaintenance1 = false;

  // other
  bool m_isSoC = false;
};

}  // namespace avk::vk