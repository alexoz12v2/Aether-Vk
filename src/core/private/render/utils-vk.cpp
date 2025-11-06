#include "render/utils-vk.h"

#include <cassert>
#include <iostream>
#include <mutex>

#include "render/context-vk.h"
#include "utils/bits.h"

namespace avk {

// -------------------------------------------------------------------
void DescriptorPoolsVk::destroy(DeviceVk const& device) AVK_NO_CFI {
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
    VkDescriptorSetLayout const& descriptorSetLayout) AVK_NO_CFI {
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
                                VkDescriptorPool descriptorPool) AVK_NO_CFI {
  vkResetDescriptorPool(device.device, descriptorPool, 0);
  std::scoped_lock<std::mutex> lock(m_mutex);
  m_recycledPools.push_back(descriptorPool);
}

void DescriptorPoolsVk::discardActivePool(DiscardPoolVk& discardPool) {
  discardPool.discardDescriptorPoolForReuse(m_activeDescriptorPool, this);
  m_activeDescriptorPool = VK_NULL_HANDLE;
}

void DescriptorPoolsVk::ensurePool(DeviceVk const& device) AVK_NO_CFI {
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
  destroyDiscardedResources(device, true);
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
                                              bool force) AVK_NO_CFI {
  std::scoped_lock<std::mutex> lk{m_mutex};
  // TODO add timeline semaphores
  uint64_t const currentTimeline =
      force ? m_timeline
            : UINT64_MAX;  // device.getSubmissionFinishedTimeline();

  m_imageViews.removeOld(
      currentTimeline, [&](VkImageView imageView) AVK_NO_CFI {
        vkDestroyImageView(device.device, imageView, nullptr);
      });

  m_images.removeOld(
      currentTimeline,
      [&](std::pair<VkImage, VmaAllocation> const& pair) AVK_NO_CFI {
        // TODO remove from rendergraph resource state tracker
        std::cout << "\033[33m" << "[removeImage] Created VkImage 0x"
                  << std::hex << pair.first << std::dec << "\033[0m"
                  << std::endl;
        vmaDestroyImage(device.vmaAllocator, pair.first, pair.second);
      });

  m_bufferViews.removeOld(
      currentTimeline, [&](VkBufferView bufferView) AVK_NO_CFI {
        vkDestroyBufferView(device.device, bufferView, nullptr);
      });

  m_buffers.removeOld(
      currentTimeline,
      [&](std::pair<VkBuffer, VmaAllocation> const& pair) AVK_NO_CFI {
        // TODO remove from rendergraph resource state tracker
        vmaDestroyBuffer(device.vmaAllocator, pair.first, pair.second);
      });

  m_pipelines.removeOld(currentTimeline, [&](VkPipeline pipeline) AVK_NO_CFI {
    vkDestroyPipeline(device.device, pipeline, nullptr);
  });

  m_pipelineLayouts.removeOld(
      currentTimeline, [&](VkPipelineLayout pipelineLayout) AVK_NO_CFI {
        vkDestroyPipelineLayout(device.device, pipelineLayout, nullptr);
      });

  m_shaderModules.removeOld(
      currentTimeline, [&](VkShaderModule shaderModule) AVK_NO_CFI {
        vkDestroyShaderModule(device.device, shaderModule, nullptr);
      });

  m_descriptorPools.removeOld(
      currentTimeline,
      [&](std::pair<VkDescriptorPool, DescriptorPoolsVk*> const& pair)
          AVK_NO_CFI { pair.second->recycle(device, pair.first); });
}

// ----------------- BufferVk --------------------------------

#ifdef AVK_DEBUG
BufferVk::~BufferVk() noexcept {
  // externally managed (free requires context)
  // actually move should be fine since VK_NULL_HANDLE
  if (!isAllocated()) {
    // TODO add object tag
    std::cerr << "Buffer Not Deallocated " << this
              << ", (if this is caused by a move constructor/assignment, this "
                 "is fine)"
              << std::endl;
  }
}
#endif

bool BufferVk::create(ContextVk const& context, size_t size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags requiredFlags,
                      VkMemoryPropertyFlags preferredFlags,
                      VmaAllocationCreateFlags vmaAllocFlags, float priority,
                      bool exportMemory, uint32_t queueFamilyCount,
                      uint32_t* queueFamilies) AVK_NO_CFI {
  assert(!isAllocated() && !isMapped());
  if (m_allocationFailed) {
    return false;
  }

  bool const share = queueFamilyCount > 1 && queueFamilies;

  m_bytes = size;
  // go to next multiple of 16 bytes
  m_allocBytes = nextMultipleOf<16>(size);
  // TODO: if maintenance4 extension active, check maxBufferSize
  // TODO if uage is VK_BUFFER_USAGE_VIDEO_DECODE or encode, you need a
  // VkVideoProfileList

  // core from 1.1 https://docs.vulkan.org/guide/latest/extensions/external.html
  VkExternalMemoryBufferCreateInfo externalInfo{};
  externalInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;

  VmaAllocator allocator = context.getAllocator();
  assert(allocator && "VmaAllocator not valid");
  VkBufferCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  createInfo.flags = 0;
  createInfo.size = m_allocBytes;
  createInfo.usage = usage;
  // needed if buffer accessed in different queues
  createInfo.sharingMode =
      share ? VK_SHARING_MODE_CONCURRENT : VK_SHARING_MODE_EXCLUSIVE;
  if (share) {
    createInfo.queueFamilyIndexCount = queueFamilyCount;
    createInfo.pQueueFamilyIndices = queueFamilies;
  }

  VmaAllocationCreateInfo allocCreateInfo{};
  allocCreateInfo.flags = vmaAllocFlags;
  allocCreateInfo.priority = priority;
  allocCreateInfo.requiredFlags = requiredFlags;
  allocCreateInfo.preferredFlags = preferredFlags;
  allocCreateInfo.usage = VMA_MEMORY_USAGE_AUTO;
  // example dynamic buffer:
  // allocCreateInfo.flags =
  // VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
  //   VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
  //   VMA_ALLOCATION_CREATE_MAPPED_BIT;

  // TODO parametrize based on usage
  // External Memory -> Either External Image or Pixel Buffer
  // this is a buffer -> pixel buffer pool
  if (exportMemory) {
    externalInfo.handleTypes = externalMemoryVkFlags();
    createInfo.pNext = &externalInfo;  // TODO

    // dedicated allocation to ensure zero offset
    allocCreateInfo.flags |= VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    // TODO create a external memory VmaPool per deivce
    // allocCreateInfo.pool
  }

  bool const useDescriptorBuffer = context.device().extensions.isEnabled(
      VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME);
  if (useDescriptorBuffer) {
    createInfo.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  }

  VkResult result = vmaCreateBuffer(allocator, &createInfo, &allocCreateInfo,
                                    &m_buffer, &m_allocation, nullptr);
  if (result != VK_SUCCESS) {
    m_allocationFailed = true;
    m_bytes = 0;
    m_allocBytes = 0;
    return false;
  }

  // TODO add buffer to tracked resources of the device
  if (useDescriptorBuffer) {
    VkBufferDeviceAddressInfo bufferDeviceAddrInfo{};
    bufferDeviceAddrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    bufferDeviceAddrInfo.buffer = m_buffer;
    m_deviceAddress = vkGetBufferDeviceAddress(context.device().device,
                                               &bufferDeviceAddrInfo);
  }

  vmaGetAllocationMemoryProperties(allocator, m_allocation,
                                   &m_memoryPropertyFlags);
  // If in commadn buffer you use vmaCopyMemoryToAllocation no need to map
  if ((m_memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
      (allocCreateInfo.flags &
       (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT))) {
    return map(context);
  }

  return true;
}

void BufferVk::updateImmediately(void const* data) const {
  updateSubImmediately(0, m_bytes, data);
}

void BufferVk::updateSubImmediately(size_t startOffset, size_t size,
                                    void const* data) const {
  assert(isMapped() && isAllocated());
  memcpy(static_cast<uint8_t*>(m_mappedMemory) + startOffset, data, size);
}

void BufferVk::flush(ContextVk const& context) const AVK_NO_CFI {
  vmaFlushAllocation(context.getAllocator(), m_allocation, 0, m_bytes);
}

void BufferVk::flushToHostAsync([[maybe_unused]] ContextVk const& context) {
  assert(m_asyncTimeline == 0);
  // TODO wait for rendering end
  // TODO set current async timeline
}

void BufferVk::readSync([[maybe_unused]] ContextVk const& context,
                        void* data) const {
  assert(isMapped());
  assert(m_asyncTimeline == 0);
  // TODO: wait rendering end if context is rendering
  memcpy(data, m_mappedMemory, m_bytes);
}

void BufferVk::readAsync(ContextVk const& context, void* data) {
  assert(isMapped());
  if (m_asyncTimeline == 0) {
    flushToHostAsync(context);
  }
  // TODO wait for timeline
  m_asyncTimeline = 0;
  memcpy(data, m_mappedMemory, m_bytes);
}

void BufferVk::free(ContextVk const& context, DiscardPoolVk& discardPool) {
  if (isMapped()) {
    unmap(context);
  }
  discardPool.discardBuffer(m_buffer, m_allocation);
  m_allocation = VK_NULL_HANDLE;
  m_buffer = VK_NULL_HANDLE;
}

void BufferVk::freeImmediately(ContextVk const& context) AVK_NO_CFI {
  assert(isAllocated());
  if (isMapped()) {
    unmap(context);
  }
  // TODO remove from resource tracker if inserted in device context
  vmaDestroyBuffer(context.getAllocator(), m_buffer, m_allocation);
  m_buffer = VK_NULL_HANDLE;
  m_allocation = VK_NULL_HANDLE;
}

VkDeviceMemory BufferVk::getExportMemory(ContextVk const& context,
                                         size_t& memorySize) AVK_NO_CFI {
  assert(isAllocated());
  VmaAllocationInfo info = {};
  vmaGetAllocationInfo(context.getAllocator(), m_allocation, &info);
  // VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT ensures 0 offset
  if (info.offset == 0) {
    assert(false &&
           "Failed to get zero offset export memory for vulkan buffer");
    return nullptr;
  }
  memorySize = info.size;
  return info.deviceMemory;
}

bool BufferVk::map(ContextVk const& context) AVK_NO_CFI {
  assert(!isMapped() && isAllocated());
  VkResult const res =
      vmaMapMemory(context.getAllocator(), m_allocation, &m_mappedMemory);
  return res == VK_SUCCESS;
}

void BufferVk::unmap(ContextVk const& context) AVK_NO_CFI {
  assert(isMapped() && isAllocated());
  vmaUnmapMemory(context.getAllocator(), m_allocation);
  m_mappedMemory = nullptr;
}

VkShaderModule finalizeShaderModule(ContextVk const& context,
                                    uint32_t const* pCode,
                                    size_t codeSize) AVK_NO_CFI {
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pCode = pCode;
  createInfo.codeSize = codeSize;
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  vkCheck(vkCreateShaderModule(context.device().device, &createInfo, nullptr,
                               &shaderModule));
  return shaderModule;
}

VkFormat basicDepthStencilFormat(VkPhysicalDevice physicalDevice) AVK_NO_CFI {
  // assuming we want VK_IMAGE_TILING_OPTIMAL and not linear
  VkFormatProperties formatProperties{};
  // at least one of D24/S8 or D32/S8 are supported per specification
  vkGetPhysicalDeviceFormatProperties(
      physicalDevice, VK_FORMAT_D24_UNORM_S8_UINT, &formatProperties);
  if (formatProperties.optimalTilingFeatures &
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
    return VK_FORMAT_D24_UNORM_S8_UINT;
  }
  vkGetPhysicalDeviceFormatProperties(
      physicalDevice, VK_FORMAT_D32_SFLOAT_S8_UINT, &formatProperties);
  if (formatProperties.optimalTilingFeatures &
      VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
    return VK_FORMAT_D32_SFLOAT_S8_UINT;
  }
  // unreachable
  return VK_FORMAT_UNDEFINED;
}

bool createImage(ContextVk const& context, SingleImage2DSpecVk const& spec,
                 VkImage& image, VmaAllocation& allocation) AVK_NO_CFI {
  VkImageCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  createInfo.imageType = VK_IMAGE_TYPE_2D;
  createInfo.extent.width = spec.width;
  createInfo.extent.height = spec.height;
  createInfo.extent.depth = 1;  // TODO + mips
  createInfo.mipLevels = 1;
  createInfo.arrayLayers = 1;
  createInfo.format = spec.format;
  createInfo.tiling = spec.imageTiling;
  createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  createInfo.usage = spec.usage;
  createInfo.samples = spec.samples;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  // TODO better -> leave this for external memory only
  allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocInfo.priority = 1.f;

  if (VK_SUCCESS != vmaCreateImage(context.getAllocator(), &createInfo,
                                   &allocInfo, &image, &allocation, nullptr)) {
    return false;
  }

  // TODO remove
  std::cout << "\033[33m" << "[createImage] Created VkImage 0x" << std::hex
            << image << std::dec << "\033[0m" << std::endl;
  return true;
}

VkCommandPool createCommandPool(ContextVk const& context, bool resettable,
                                uint32_t queueFamilyIndex) AVK_NO_CFI {
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkCommandPoolCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  if (resettable) {
    createInfo.flags |= VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  }
  createInfo.queueFamilyIndex = queueFamilyIndex;
  vkCheck(vkCreateCommandPool(context.device().device, &createInfo, nullptr,
                              &commandPool));
  return commandPool;
}

bool allocPrimaryCommandBuffers(ContextVk const& context,
                                VkCommandPool commandPool, uint32_t count,
                                VkCommandBuffer* commandBuffers) AVK_NO_CFI {
  VkCommandBufferAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  allocateInfo.commandPool = commandPool;
  allocateInfo.commandBufferCount = count;
  allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

  VkResult const res = vkAllocateCommandBuffers(context.device().device,
                                                &allocateInfo, commandBuffers);
  return VK_SUCCESS == res;
}

VkPipelineLayout createPipelineLayout(
    ContextVk const& context,
    VkDescriptorSetLayout const* pDescriptorSetLayouts,
    uint32_t descriptorSetLayoutCount,
    VkPushConstantRange const* pPushConstantRanges,
    uint32_t pushConstantRangeCount) AVK_NO_CFI {
  // TODO check setLayoutCount must be less than or equal to
  // VkPhysicalDeviceLimits::maxBoundDescriptorSets
  // TODO Any two elements of pPushConstantRanges must not include the same
  // stage in stageFlags
  VkPipelineLayoutCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  createInfo.setLayoutCount = descriptorSetLayoutCount;
  createInfo.pSetLayouts = pDescriptorSetLayouts;
  createInfo.pushConstantRangeCount = pushConstantRangeCount;
  createInfo.pPushConstantRanges = pPushConstantRanges;

  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  vkCheck(vkCreatePipelineLayout(context.device().device, &createInfo, nullptr,
                                 &pipelineLayout));
  return pipelineLayout;
}

}  // namespace avk