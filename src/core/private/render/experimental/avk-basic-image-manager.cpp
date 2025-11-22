#include "render/experimental/avk-basic-image-manager.h"

// library
#include <cassert>

static VkImageCreateInfo startCreateInfo(VkExtent2D extent, VkFormat format,
                                         VkImageUsageFlags usage,
                                         VkSampleCountFlagBits samples) {
  VkImageCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  createInfo.extent = {extent.width, extent.height, 1};
  createInfo.format = format;
  createInfo.imageType = VK_IMAGE_TYPE_2D;
  createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
  createInfo.arrayLayers = 1;
  createInfo.mipLevels = 1;  // TODO generate mipmapped texture
  createInfo.usage = usage;
  createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.samples = samples;
  return createInfo;
}

namespace avk::experimental {

ImageManager::ImageManager(vk::Device* device, size_t cap) : m_deps{device} {
  assert(device);
  m_imageMap.reserve(cap);
}

// Note: add VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT and
// VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT as preferredFlags
int32_t ImageManager::createTransientAttachment(
    uint64_t id, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage,
    VkSampleCountFlagBits samples) AVK_NO_CFI {
  VmaAllocator allocator = m_deps.device->vmaAllocator();

  VkImageCreateInfo const createInfo = startCreateInfo(
      extent, format, usage | VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT, samples);
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  // give it priority on dedicated hardware. on SoC we don't care as we expect
  // it to not be effectively created
  if (!m_deps.device->isSoC()) {
    allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  } else {
    // WARN: This is experimental and it may fail if there's no lazily allocated
    // memory type
    allocInfo.flags = VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
  }
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;  // not needed?
  allocInfo.priority = 1.f;
  // prefer lazily allocated (transient -> tile cacheable) on mobile (+any SoC)
  if (m_deps.device->isSoC()) {
    allocInfo.preferredFlags = VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT;
  }
  VkImage image = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;

  VkResult res = vmaCreateImage(allocator, &createInfo, &allocInfo, &image,
                                &alloc, nullptr);

  if (res == VK_ERROR_OUT_OF_DEVICE_MEMORY && m_deps.device->isSoC()) {
    allocInfo.flags &= ~VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
    LOGW << AVK_LOG_YLW
        "Failed never allocate lazy allocation. retry without it" AVK_LOG_RST
         << std::endl;
    res = vmaCreateImage(allocator, &createInfo, &allocInfo, &image, &alloc,
                         nullptr);
  }

  if (res < 0) {
    return VulkanError;
  }
  bool wasInserted = false;
  {
    std::unique_lock wlock{m_mtx};
    wasInserted = m_imageMap.try_emplace(id, image, alloc).second;
  }
  if (!wasInserted) {
    vmaDestroyImage(allocator, image, alloc);
    return Collision;
  }
  return Success;
}

// sampled texture -> VK_IMAGE_USAGE_SAMPLED_BIT
// storage texture -> VK_IMAGE_USAGE_STORAGE_BIT
int32_t ImageManager::createTexture(uint64_t id, VkExtent2D extent,
                                    VkFormat format, VkImageUsageFlags usage,
                                    VkSampleCountFlagBits samples,
                                    bool streamingNeeded,
                                    bool forceWithinBudget,
                                    bool forceNoAllocation) AVK_NO_CFI {
  VmaAllocator allocator = m_deps.device->vmaAllocator();

  VkImageCreateInfo const createInfo =
      startCreateInfo(extent, format, usage, samples);
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = streamingNeeded ? VMA_MEMORY_USAGE_AUTO
                                    : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  allocInfo.priority = 1.f;
  if (streamingNeeded) {
    allocInfo.flags |=
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
        VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
        VMA_ALLOCATION_CREATE_MAPPED_BIT;
  }
  if (forceWithinBudget) {
    allocInfo.flags |= VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
  }
  if (forceNoAllocation) {
    allocInfo.flags |= VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT;
  }

  VkImage image = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;

  VkResult const res = vmaCreateImage(allocator, &createInfo, &allocInfo,
                                      &image, &alloc, nullptr);
  if (res < 0) {
    return VulkanError;
  }
#ifdef AVK_DEBUG
  if (m_deps.device->isSoC() && streamingNeeded) {
    VkMemoryPropertyFlags memPropFlags = 0;
    vmaGetAllocationMemoryProperties(allocator, alloc, &memPropFlags);
    assert(
        (memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
        "SoC GPUs should have Mappable memory on streaming allocation request");
  }
#endif
  bool wasInserted = false;
  {
    std::unique_lock wlock{m_mtx};
    wasInserted = m_imageMap.try_emplace(id, image, alloc).second;
  }
  if (!wasInserted) {
    vmaDestroyImage(allocator, image, alloc);
    return Collision;
  }
  return Success;
}

bool ImageManager::get(uint64_t id, VkImage& outImage,
                       VmaAllocation& outAlloc) {
  std::shared_lock rlock{m_mtx};
  auto it = m_imageMap.find(id);
  if (it == m_imageMap.end()) {
    return false;
  }
  outImage = it->second.handle;
  outAlloc = it->second.alloc;
  return true;
}

bool ImageManager::discardById(vk::DiscardPool* discardPool, uint64_t id,
                               uint64_t timeline) {
  {
    std::shared_lock rlock{m_mtx};
    auto it = m_imageMap.find(id);
    if (it == m_imageMap.end()) {
      return false;
    }
  }
  std::unique_lock wlock{m_mtx};
  auto it = m_imageMap.find(id);
  if (it == m_imageMap.end()) {
    return false;
  }
  discardPool->discardImage(it->second.handle, it->second.alloc, timeline);
  m_imageMap.erase(it);
  return true;
}

void ImageManager::discardEverything(vk::DiscardPool* discardPool,
                                     uint64_t timeline) {
  std::unique_lock wlock{m_mtx};
  for (auto const& [id, pair] : m_imageMap) {
    discardPool->discardImage(pair.handle, pair.alloc, timeline);
  }
  m_imageMap.clear();
}

}  // namespace avk::experimental