#pragma once

#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "render/context-vk.h"

namespace avk {

// -------------------------------------------------------------------
// list of descriptor pools, constructod with a fixed size per resource type.
// when more resources are needed, create another pool
class DiscardPoolVk;
class DescriptorPoolsVk {
 private:
  // sizes copied from blender
  static constexpr uint32_t POOL_SIZE_STORAGE_BUFFER = 1000;
  static constexpr uint32_t POOL_SIZE_DESCRIPTOR_SETS = 250;
  static constexpr uint32_t POOL_SIZE_STORAGE_IMAGE = 250;
  static constexpr uint32_t POOL_SIZE_COMBINED_IMAGE_SAMPLER = 250;
  static constexpr uint32_t POOL_SIZE_UNIFORM_BUFFER = 500;
  static constexpr uint32_t POOL_SIZE_UNIFORM_TEXEL_BUFFER = 100;
  static constexpr uint32_t POOL_SIZE_INPUT_ATTACHMENT = 100;

 public:
  DescriptorPoolsVk() = default;
  DescriptorPoolsVk(DescriptorPoolsVk const&) = delete;
  DescriptorPoolsVk(DescriptorPoolsVk&&) noexcept = delete;
  DescriptorPoolsVk& operator=(DescriptorPoolsVk const&) = delete;
  DescriptorPoolsVk& operator=(DescriptorPoolsVk&&) noexcept = delete;

  void destroy(DeviceVk const& device);

  void init(DeviceVk const& device);
  VkDescriptorSet allocate(DeviceVk const& device, DiscardPoolVk& discardPool,
                           VkDescriptorSetLayout const& descriptorSetLayout);
  void recycle(DeviceVk const& device, VkDescriptorPool descriptorPool);

 private:
  void discardActivePool(DiscardPoolVk& discardPool);
  void ensurePool(DeviceVk const& device);

  std::vector<VkDescriptorPool> m_recycledPools;
  VkDescriptorPool m_activeDescriptorPool = VK_NULL_HANDLE;
  std::mutex m_mutex;
};

// -------------------------------------------------------------------
// vector of pair timestamp, value
// invariant: timeline values should always be sorted (except for wrap around)
template <typename Item>
class TimelineResources : public std::vector<std::pair<uint64_t, Item>> {
 public:
  void appendTimeline(uint64_t timeline, Item item) {
    append(std::make_pair(timeline, item));
  }

  void updateTimeline(uint64_t timeline) {
    for (std::pair<uint64_t, Item>& pair : *this) {
      pair.first = timeline;
    }
  }

  template <typename Deleter,
            typename V = std::enable_if_t<std::is_invocable_v<Deleter, Item>>>
  void removeOld(uint64_t currentTimeline, Deleter&& deleter) {
    int64_t firstIndexToKeep = 0;
    for (std::pair<uint64_t, Item>& item : *this) {
      if (item.first > currentTimeline) {
        break;
      }
      deleter(item.second);
      ++firstIndexToKeep;
    }

    if (firstIndexToKeep > 0) {
      erase(begin(), begin() + firstIndexToKeep);
    }
  }
};

class DiscardPoolVk {
 public:
  void deinit(DeviceVk& device);

  void discardImage(VkImage image, VmaAllocation allocation);
  void discardImageView(VkImageView imageView);
  void discardBuffer(VkBuffer buffer, VmaAllocation allocation);
  void discardBufferView(VkBufferView bufferView);
  void discardShaderModule(VkShaderModule shaderModule);
  void discardPipeline(VkPipeline pipeline);
  void discardPipelineLayout(VkPipelineLayout pipelineLayout);
  void discardDescriptorPoolForReuse(VkDescriptorPool descriptorPool,
                                     DescriptorPoolsVk* descriptorPools);

  // move data from a discard pool into another. the target pool
  void moveData(DiscardPoolVk& srcPool, uint64_t timeline);
  inline std::mutex& getMutex() { return m_mutex; }
  void destroyDiscardedResources(DeviceVk const& device, bool force = false);

 private:
  TimelineResources<std::pair<VkImage, VmaAllocation>> m_images;
  TimelineResources<std::pair<VkBuffer, VmaAllocation>> m_buffers;
  TimelineResources<VkImageView> m_imageViews;
  TimelineResources<VkBufferView> m_bufferViews;
  TimelineResources<VkShaderModule> m_shaderModules;
  TimelineResources<VkPipeline> m_pipelines;
  TimelineResources<VkPipelineLayout> m_pipelineLayouts;
  TimelineResources<std::pair<VkDescriptorPool, DescriptorPoolsVk*>>
      m_descriptorPools;

  std::mutex m_mutex;
  uint64_t m_timeline = UINT64_MAX;
};

}  // namespace avk