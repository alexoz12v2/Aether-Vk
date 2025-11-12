#include "render/experimental/avk-basic-buffer-manager.h"

#include "render/vk/buffers/buffer-vk.h"

// library
#include <cassert>

// TODO:add support for pNext chain and some flags like descriptor buffer
// https://docs.vulkan.org/refpages/latest/refpages/source/VkBufferCreateInfo.html

// TODO: suppport for periodic memory defragmentation
// https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/defragmentation.html

static VkBufferCreateInfo startCreateInfo(size_t size,
                                          VkBufferCreateFlags usage) {
  VkBufferCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  createInfo.usage = usage;
  createInfo.size = avk::nextMultipleOf<16>(size);
  createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  return createInfo;
}

// TODO logging functions for allocation infos, whet you requested

namespace avk::experimental {

BufferManager::BufferManager(vk::Device* device, size_t cap) : m_deps{device} {
  m_bufferMap.reserve(cap);
}

int32_t BufferManager::createBufferGPUOnly(uint64_t id, size_t bytes,
                                           VkBufferCreateFlags usage) {
  assert(m_deps.device && m_deps.device->device());
  VmaAllocator const allocator = m_deps.device->vmaAllocator();
  bool const isSoC = m_deps.device->isSoC();
  VkBufferCreateInfo createInfo = startCreateInfo(bytes, usage);
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage =
      isSoC ? VMA_MEMORY_USAGE_AUTO : VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
  allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocInfo.priority = 1.f;

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  VkResult const res = vmaCreateBuffer(allocator, &createInfo, &allocInfo,
                                       &buffer, &alloc, nullptr);
  if (res < 0) {
    return VulkanError;
  }
  bool wasInserted = false;
  {
    std::unique_lock wlock{m_mtx};
    wasInserted = m_bufferMap.try_emplace(id, buffer, alloc).second;
  }
  if (!wasInserted) {
    vmaDestroyBuffer(allocator, buffer, alloc);
    return Collision;
  }
  return Success;
}

int32_t BufferManager::createBufferStaging(uint64_t id, size_t bytes) {
  assert(m_deps.device && m_deps.device->device());
  assert(!m_deps.device->isSoC() && "Integrated graphics/SoC -> no staging");
  VmaAllocator const allocator = m_deps.device->vmaAllocator();

  VkBufferCreateInfo createInfo =
      startCreateInfo(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  allocInfo.priority = 1.f;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  allocInfo.preferredFlags =
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;

  VkResult const res = vmaCreateBuffer(allocator, &createInfo, &allocInfo,
                                       &buffer, &alloc, nullptr);
  if (res < 0) {
    return VulkanError;
  }
  bool wasInserted = false;
  {
    std::unique_lock wlock{m_mtx};
    wasInserted = m_bufferMap.try_emplace(id, buffer, alloc).second;
  }
  if (!wasInserted) {
    vmaDestroyBuffer(allocator, buffer, alloc);
    return Collision;
  }
  return Success;
}

int32_t BufferManager::createBufferReadback(uint64_t id, size_t bytes) {
  assert(m_deps.device && m_deps.device->device());
  assert(!m_deps.device->isSoC() && "Integrated graphics/SoC -> no staging");
  VmaAllocator const allocator = m_deps.device->vmaAllocator();

  VkBufferCreateInfo createInfo =
      startCreateInfo(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  allocInfo.priority = 1.f;
  allocInfo.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  allocInfo.preferredFlags =
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
  allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                    VMA_ALLOCATION_CREATE_MAPPED_BIT;

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;

  VkResult const res = vmaCreateBuffer(allocator, &createInfo, &allocInfo,
                                       &buffer, &alloc, nullptr);
  if (res < 0) {
    return VulkanError;
  }
  bool wasInserted = false;
  {
    std::unique_lock wlock{m_mtx};
    wasInserted = m_bufferMap.try_emplace(id, buffer, alloc).second;
  }
  if (!wasInserted) {
    vmaDestroyBuffer(allocator, buffer, alloc);
    return Collision;
  }
  return Success;
}

int32_t BufferManager::createBufferStreaming(uint64_t id, size_t bytes,
                                             VkBufferUsageFlags usage) {
  assert(m_deps.device && m_deps.device->device());
  VmaAllocator const allocator = m_deps.device->vmaAllocator();

  VkBufferCreateInfo createInfo =
      startCreateInfo(bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;
  VkResult const res = vmaCreateBuffer(allocator, &createInfo, &allocInfo,
                                       &buffer, &alloc, nullptr);
  if (res < 0) {
    return VulkanError;
  }
  bool wasInserted = false;
  {
    std::unique_lock wlock{m_mtx};
    wasInserted = m_bufferMap.try_emplace(id, buffer, alloc).second;
  }
  if (!wasInserted) {
    vmaDestroyBuffer(allocator, buffer, alloc);
    return Collision;
  }
  // now check whether we got memory mapped I/O or
  // transfer operation (`vmaCopyMemoryToAllocation`) + `VkBufferMemoryBarrier`
  // is required
  VkMemoryPropertyFlags memPropFlags = 0;
  vmaGetAllocationMemoryProperties(allocator, alloc, &memPropFlags);
  if (memPropFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
    return Success;
  } else {
    return TransferRequired;
  }
}

bool BufferManager::get(uint64_t id, VkBuffer& outBuffer,
                        VmaAllocation& outAlloc) {
  std::shared_lock rlock{m_mtx};
  auto it = m_bufferMap.find(id);
  if (it == m_bufferMap.end()) {
    return false;
  }
  outBuffer = it->second.handle;
  outAlloc = it->second.alloc;
  return true;
}

bool BufferManager::discardById(vk::DiscardPool* discardPool, uint64_t id,
                                uint64_t timeline) {
  {
    std::shared_lock rlock{m_mtx};
    auto it = m_bufferMap.find(id);
    if (it == m_bufferMap.end()) {
      return false;
    }
  }
  std::unique_lock wlock{m_mtx};
  auto it = m_bufferMap.find(id);
  if (it == m_bufferMap.end()) {
    return false;
  }
  discardPool->discardBuffer(it->second.handle, it->second.alloc, timeline);
  m_bufferMap.erase(it);
  return true;
}

void BufferManager::discardEverything(vk::DiscardPool* discard,
                                      uint64_t timeline) {
  std::unique_lock wlock{m_mtx};
  for (auto const& [id, pair] : m_bufferMap) {
    discard->discardBuffer(pair.handle, pair.alloc, timeline);
  }
  m_bufferMap.clear();
}

}  // namespace avk::experimental