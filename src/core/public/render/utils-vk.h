#pragma once

#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include "render/context-vk.h"
#include "utils/mixins.h"

// TODO: split all classes here in different files

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
  using Base = std::vector<std::pair<uint64_t, Item>>;

 public:
  void appendTimeline(uint64_t timeline, Item item) {
    Base::emplace_back(timeline, item);
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
      Base::erase(Base::begin(), Base::begin() + firstIndexToKeep);
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

class BufferVk : public NonCopyable {
 public:
  BufferVk() = default;
#ifdef AVK_DEBUG
  ~BufferVk() noexcept;
#endif

  inline VkBuffer buffer() const { return m_buffer; }

  inline bool isAllocated() const {
    return m_buffer != VK_NULL_HANDLE && m_allocation != VK_NULL_HANDLE;
  }
  inline bool isMapped() const { return m_mappedMemory != nullptr; }

  // TODO see if export memory is worth it
  bool create(ContextVk const& context, size_t size, VkBufferUsageFlags usage,
              VkMemoryPropertyFlags requiredFlags,
              VkMemoryPropertyFlags preferredFlags,
              VmaAllocationCreateFlags vmaAllocFlags, float priority,
              bool exportMemory, uint32_t queueFamilyCount,
              uint32_t* queueFamilies);
  void updateImmediately(void const* data) const;
  void updateSubImmediately(size_t startOffset, size_t size,
                            void const* data) const;
  void flush(ContextVk const& context) const;
  void flushToHostAsync(ContextVk const& context);
  void readSync(ContextVk const& context, void* data) const;
  void readAsync(ContextVk const& context, void* data);

  VkDeviceMemory getExportMemory(ContextVk const& context, size_t& memorySize);

  bool map(ContextVk const& context);
  void unmap(ContextVk const& context);

  // WARNING: necessary to be called before going out of scope
  void free(ContextVk const& context, DiscardPoolVk& discardPool);
  void freeImmediately(ContextVk const& context);

 private:
  size_t m_bytes = 0;
  size_t m_allocBytes = 0;
  VkBuffer m_buffer = VK_NULL_HANDLE;
  VmaAllocation m_allocation = VK_NULL_HANDLE;
  VkMemoryPropertyFlags m_memoryPropertyFlags = 0;
  uint64_t m_asyncTimeline = 0;

  // previous allocation failed: will skip reallocations
  bool m_allocationFailed = false;

  // pointer to device mapped memory
  void* m_mappedMemory = nullptr;

  // TODO requires feature/extension
  VkDeviceAddress m_deviceAddress = 0;
};

// TODO remove in favour of a class supporting image samplers, push constants,
// descriptor sets and specialization constants
// plus, write a shader library and analysis on SPIR-V
VkShaderModule finalizeShaderModule(ContextVk const& context,
                                    uint32_t const* pCode, size_t codeSize);

// TODO organize in a better way (maybe in context?)
VkFormat basicDepthStencilFormat(VkPhysicalDevice physicalDevice);

// TODO organize in a better way
// TODO now it supports GPU local image only
// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html
struct SingleImage2DSpecVk {
  uint32_t width;
  uint32_t height;
  VkFormat format;
  VkImageTiling imageTiling;
  VkImageUsageFlags usage;
  VkSampleCountFlagBits samples;
};
bool createImage(ContextVk const& context, SingleImage2DSpecVk const& spec,
                 VkImage& image, VmaAllocation& allocation);

// TODO better
// TODO see if VkDataGraphProcessingEngineCreateInfoARM is necessary for ARM
// devices
// TODO see if protectedMemory Feature is necessary
VkCommandPool createCommandPool(ContextVk const& context, bool resettable,
                                uint32_t queueFamilyIndex);

// TODO better
bool allocPrimaryCommandBuffers(ContextVk const& context,
                                VkCommandPool commandPool, uint32_t count,
                                VkCommandBuffer* commandBuffers);
}  // namespace avk