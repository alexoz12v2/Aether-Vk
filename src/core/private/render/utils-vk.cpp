#include "render/utils-vk.h"

#include <cassert>
#include <mutex>

namespace avk {

// -------------------------------------------------------------------
void DescriptorPoolsVk::destroy(DeviceVk const& device) {
  for (VkDescriptorPool const descriptorPool : m_recycledPools) {
    vkDestroyDescriptorPool(device.device, descriptorPool, nullptr);
  }
  m_recycledPools.clear();
  if (m_activeDescriptorPool != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(device.device, m_activeDescriptorPool, nullptr);
    m_activeDescriptorPool = VK_NULL_HANDLE;
  }
}

void DescriptorPoolsVk::init(DeviceVk const& device) { ensurePool(device); }

VkDescriptorSet DescriptorPoolsVk::allocate(
    DeviceVk const& device, DiscardPoolVk& discardPool,
    VkDescriptorSetLayout const& descriptorSetLayout) {
  assert(m_activeDescriptorPool != VK_NULL_HANDLE);
  assert(descriptorSetLayout != VK_NULL_HANDLE);

  VkDescriptorSetAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocateInfo.descriptorPool = m_activeDescriptorPool;
  allocateInfo.pSetLayouts = &descriptorSetLayout;

  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  VkResult result =
      vkAllocateDescriptorSets(device.device, &allocateInfo, &descriptorSet);
  if (result == VK_ERROR_OUT_OF_POOL_MEMORY ||
      result == VK_ERROR_FRAGMENTED_POOL) {
    // TODO add syncronization on discard pool (or external)
    discardActivePool(discardPool);
    ensurePool(device);
    return allocate(device, discardPool, descriptorSetLayout);
  }

  return descriptorSet;
}

void DescriptorPoolsVk::recycle(DeviceVk const& device,
                                VkDescriptorPool descriptorPool) {
  vkResetDescriptorPool(device.device, descriptorPool, 0);
  std::scoped_lock<std::mutex> lock(m_mutex);
  m_recycledPools.push_back(descriptorPool);
}

void DescriptorPoolsVk::discardActivePool(DiscardPoolVk& discardPool) {
  discardPool.discardDescriptorPoolForReuse(m_activeDescriptorPool, this);
  m_activeDescriptorPool = VK_NULL_HANDLE;
}

void DescriptorPoolsVk::ensurePool(DeviceVk const& device) {
  if (m_activeDescriptorPool != VK_NULL_HANDLE) {
    return;
  }

  std::scoped_lock<std::mutex> lock{m_mutex};
  if (!m_recycledPools.empty()) {
    m_activeDescriptorPool = m_recycledPools.back();
    m_recycledPools.pop_back();
    return;
  }

  static uint32_t constexpr PoolSizesCount = 6;
  VkDescriptorPoolSize const poolSizes[PoolSizesCount] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, POOL_SIZE_STORAGE_BUFFER},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, POOL_SIZE_STORAGE_IMAGE},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       POOL_SIZE_COMBINED_IMAGE_SAMPLER},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, POOL_SIZE_UNIFORM_BUFFER},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, POOL_SIZE_UNIFORM_TEXEL_BUFFER},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, POOL_SIZE_INPUT_ATTACHMENT},
  };

  VkDescriptorPoolCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  createInfo.maxSets = POOL_SIZE_DESCRIPTOR_SETS;
  createInfo.poolSizeCount = PoolSizesCount;
  createInfo.pPoolSizes = poolSizes;
  vkCreateDescriptorPool(device.device, &createInfo, nullptr,
                         &m_activeDescriptorPool);
}

// -------------------------------------------------------------------

void DiscardPoolVk::deinit(DeviceVk& device) {
  destroyDiscardedResources(device);
}

void DiscardPoolVk::discardImage(VkImage image, VmaAllocation allocation) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_images.appendTimeline(m_timeline, std::make_pair(image, allocation));
}

void DiscardPoolVk::discardImageView(VkImageView imageView) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_imageViews.appendTimeline(m_timeline, imageView);
}

void DiscardPoolVk::discardBuffer(VkBuffer buffer, VmaAllocation allocation) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_buffers.appendTimeline(m_timeline, std::make_pair(buffer, allocation));
}

void DiscardPoolVk::discardBufferView(VkBufferView bufferView) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_bufferViews.appendTimeline(m_timeline, bufferView);
}

void DiscardPoolVk::discardShaderModule(VkShaderModule shaderModule) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_shaderModules.appendTimeline(m_timeline, shaderModule);
}

void DiscardPoolVk::discardPipeline(VkPipeline pipeline) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_pipelines.appendTimeline(m_timeline, pipeline);
}

void DiscardPoolVk::discardPipelineLayout(VkPipelineLayout pipelineLayout) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_pipelineLayouts.appendTimeline(m_timeline, pipelineLayout);
}

void DiscardPoolVk::discardDescriptorPoolForReuse(
    VkDescriptorPool descriptorPool, DescriptorPoolsVk* descriptorPools) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  m_descriptorPools.appendTimeline(
      m_timeline, std::make_pair(descriptorPool, descriptorPools));
}

template <typename T>
static void moveExtend(std::vector<T>& dest, std::vector<T>& source) {
  dest.insert(dest.end(), std::make_move_iterator(source.begin()),
              std::make_move_iterator(source.end()));
  source.clear();  // optional: clear source after move
}

void DiscardPoolVk::moveData(DiscardPoolVk& srcPool, uint64_t timeline) {
  srcPool.m_images.updateTimeline(timeline);
  srcPool.m_buffers.updateTimeline(timeline);
  srcPool.m_imageViews.updateTimeline(timeline);
  srcPool.m_bufferViews.updateTimeline(timeline);
  srcPool.m_shaderModules.updateTimeline(timeline);
  srcPool.m_pipelines.updateTimeline(timeline);
  srcPool.m_pipelineLayouts.updateTimeline(timeline);
  srcPool.m_descriptorPools.updateTimeline(timeline);

  moveExtend(m_images, srcPool.m_images);
  moveExtend(m_buffers, srcPool.m_buffers);
  moveExtend(m_imageViews, srcPool.m_imageViews);
  moveExtend(m_bufferViews, srcPool.m_bufferViews);
  moveExtend(m_shaderModules, srcPool.m_shaderModules);
  moveExtend(m_pipelines, srcPool.m_pipelines);
  moveExtend(m_pipelineLayouts, srcPool.m_pipelineLayouts);
  moveExtend(m_descriptorPools, srcPool.m_descriptorPools);
}

void DiscardPoolVk::destroyDiscardedResources(DeviceVk const& device,
                                              bool force) {
  std::scoped_lock<std::mutex> lk{m_mutex};
  // TODO add timeline semaphores
  uint64_t const currentTimeline =
      force ? UINT64_MAX
            : UINT64_MAX;  // device.getSubmissionFinishedTimeline();

  m_imageViews.removeOld(currentTimeline, [&](VkImageView imageView) {
    vkDestroyImageView(device.device, imageView, nullptr);
  });

  m_images.removeOld(
      currentTimeline, [&](std::pair<VkImage, VmaAllocation> const& pair) {
        // TODO remove from rendergraph resource state tracker
        vmaDestroyImage(device.vmaAllocator, pair.first, pair.second);
      });

  m_bufferViews.removeOld(currentTimeline, [&](VkBufferView bufferView) {
    vkDestroyBufferView(device.device, bufferView, nullptr);
  });

  m_buffers.removeOld(
      currentTimeline, [&](std::pair<VkBuffer, VmaAllocation> const& pair) {
        // TODO remove from rendergraph resource state tracker
        vmaDestroyBuffer(device.vmaAllocator, pair.first, pair.second);
      });

  m_pipelines.removeOld(currentTimeline, [&](VkPipeline pipeline) {
    vkDestroyPipeline(device.device, pipeline, nullptr);
  });

  m_pipelineLayouts.removeOld(
      currentTimeline, [&](VkPipelineLayout pipelineLayout) {
        vkDestroyPipelineLayout(device.device, pipelineLayout, nullptr);
      });

  m_shaderModules.removeOld(currentTimeline, [&](VkShaderModule shaderModule) {
    vkDestroyShaderModule(device.device, shaderModule, nullptr);
  });

  m_descriptorPools.removeOld(
      currentTimeline,
      [&](std::pair<VkDescriptorPool, DescriptorPoolsVk*> const& pair) {
        pair.second->recycle(device, pair.first);
      });
}

}  // namespace avk