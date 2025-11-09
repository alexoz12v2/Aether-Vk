#pragma once

#include "render/vk/common-vk.h"
#include "render/vk/device-vk.h"

// standard
#include <mutex>

namespace avk::vk {

class DescriptorPools : public NonMoveable {
 public:
    DescriptorPools(Device* device);
    ~DescriptorPools();

    VkDescriptorSet allocate(VkDescriptorSetLayout descriptorSetLayout);
    void recycle(VkDescriptorPool pool);

 private:
  // dependencies which must outlive the object
  struct Deps {
    Device* device;
  } m_deps;

  /// when a pool is full it is discarded. after all descriptor sets of the pool
  /// are unused, the pool can be reset and reused. Note: Descriptor *Pools* are
  /// used by a single thread
  std::vector<VkDescriptorPool> m_recycledPools;
  VkDescriptorPool m_activePool = VK_NULL_HANDLE;

  // synchronization
  std::mutex m_mtx;
};

}  // namespace avk::vk