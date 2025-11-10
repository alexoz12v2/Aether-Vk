#pragma once

#include "render/vk/common-vk.h"
#include "render/vk/device-vk.h"

// standard
#include <mutex>

namespace avk::vk {

class DiscardPool;

class DescriptorPools : public NonMoveable {
 public:
  DescriptorPools(Device* device);
  ~DescriptorPools();

  /// allocates descriptor set with desired layout
  /// discard pool and timeline semaphore value used when
  /// we fail allocation due to pool exhaustion, hence discard it
  VkDescriptorSet allocate(VkDescriptorSetLayout descriptorSetLayout,
                           DiscardPool* discardPool, uint64_t value);
  void ensureActivePool();
  /// called by allocate in case of out of memory or fragmented
  void discardActivePool(DiscardPool* discardPool, uint64_t value);

  /// called by DiscardPool
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

  // synchronization: either maintain a pool per thread or mutex
  // we use the latter
  std::mutex m_mtx;
};

}  // namespace avk::vk