#include "render/vk/discard-pool.h"

namespace avk::vk {

DiscardPool::DiscardPool(Device* device) AVK_NO_CFI : m_deps{device} {
  // create timeline semaphore
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();

  VkSemaphoreTypeCreateInfoKHR semType{};
  semType.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
  semType.initialValue = 0;
  semType.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;

  VkSemaphoreCreateInfo semCreateInfo{};
  semCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semCreateInfo.pNext = &semType;

  VK_CHECK(
      vkDevApi->vkCreateSemaphore(dev, &semCreateInfo, nullptr, &m_timeline));

  // reserve some space
  m_images.reserve(64);
  m_buffers.reserve(64);
  m_imageViews.reserve(64);
  m_bufferViews.reserve(64);
  m_shaderModules.reserve(64);
  m_pipelines.reserve(64);
  m_pipelineLayouts.reserve(64);
}

DiscardPool::~DiscardPool() AVK_NO_CFI {
  if (!*this) {
    return;
  }
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  destroyDiscardedResources(true);
  vkDevApi->vkDestroySemaphore(dev, m_timeline, nullptr);
}

uint64_t DiscardPool::queryTime() const {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  uint64_t value = 0;
  VK_CHECK(vkDevApi->vkGetSemaphoreCounterValueKHR(dev, m_timeline, &value));
  return value;
}

void DiscardPool::discardImage(VkImage image, VmaAllocation allocation,
                               uint64_t value) {
  std::lock_guard lk{m_mtx};
  m_images.appendTimeline(value, std::make_pair(image, allocation));
}
void DiscardPool::discardImageView(VkImageView imageView, uint64_t value) {
  std::lock_guard lk{m_mtx};
  m_imageViews.appendTimeline(value, imageView);
}
void DiscardPool::discardBuffer(VkBuffer buffer, VmaAllocation allocation,
                                uint64_t value) {
  std::lock_guard lk{m_mtx};
  m_buffers.appendTimeline(value, std::make_pair(buffer, allocation));
}
void DiscardPool::discardBufferView(VkBufferView bufferView, uint64_t value) {
  std::lock_guard lk{m_mtx};
  m_bufferViews.appendTimeline(value, bufferView);
}
void DiscardPool::discardShaderModule(VkShaderModule shaderModule,
                                      uint64_t value) {
  std::lock_guard lk{m_mtx};
  m_shaderModules.appendTimeline(value, shaderModule);
}
void DiscardPool::discardPipeline(VkPipeline pipeline, uint64_t value) {
  std::lock_guard lk{m_mtx};
  m_pipelines.appendTimeline(value, pipeline);
}
void DiscardPool::discardPipelineLayout(VkPipelineLayout pipelineLayout,
                                        uint64_t value) {
  std::lock_guard lk{m_mtx};
  m_pipelineLayouts.appendTimeline(value, pipelineLayout);
}

void DiscardPool::destroyDiscardedResources(bool force) AVK_NO_CFI {
  std::lock_guard lk{m_mtx};
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  uint64_t timeline = UINT64_MAX;
  if (!force) {
    VK_CHECK(
        vkDevApi->vkGetSemaphoreCounterValueKHR(dev, m_timeline, &timeline));
  }
  // image and image views
  m_imageViews.removeOld(
      timeline, [dev, vkDevApi](VkImageView imageView) AVK_NO_CFI {
        vkDevApi->vkDestroyImageView(dev, imageView, nullptr);
      });
  m_images.removeOld(
      timeline, [device = m_deps.device](
                    std::pair<VkImage, VmaAllocation> const& pair) AVK_NO_CFI {
        vmaDestroyImage(device->vmaAllocator(), pair.first, pair.second);
      });
  // buffer and buffer views
  m_bufferViews.removeOld(
      timeline, [dev, vkDevApi](VkBufferView bufferView) AVK_NO_CFI {
        vkDevApi->vkDestroyBufferView(dev, bufferView, nullptr);
      });
  m_buffers.removeOld(
      timeline, [device = m_deps.device](
                    std::pair<VkBuffer, VmaAllocation> const& pair) AVK_NO_CFI {
        vmaDestroyBuffer(device->vmaAllocator(), pair.first, pair.second);
      });
  // pipeline, pipeline layouts, shader modules
  m_pipelines.removeOld(timeline,
                        [dev, vkDevApi](VkPipeline pipeline) AVK_NO_CFI {
                          vkDevApi->vkDestroyPipeline(dev, pipeline, nullptr);
                        });
  m_pipelineLayouts.removeOld(
      timeline, [dev, vkDevApi](VkPipelineLayout pipelineLayout) AVK_NO_CFI {
        vkDevApi->vkDestroyPipelineLayout(dev, pipelineLayout, nullptr);
      });
  m_shaderModules.removeOld(
      timeline, [dev, vkDevApi](VkShaderModule shaderModule) AVK_NO_CFI {
        vkDevApi->vkDestroyShaderModule(dev, shaderModule, nullptr);
      });
}

}  // namespace avk::vk