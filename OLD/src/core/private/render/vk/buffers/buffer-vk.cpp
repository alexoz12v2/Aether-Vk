#include "render/vk/buffers/buffer-vk.h"

#include "render/vk/device-vk.h"

// library
#include <utils/bits.h>

#include <cassert>

namespace avk::vk {

bool createBuffer(Device* device, size_t size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags required,
                  VkMemoryPropertyFlags preferred,
                  VmaAllocationCreateFlags vmaFlags, VkBuffer* buffer,
                  VmaAllocation* allocation) {
  assert(allocation && buffer && device);
  if ((size & 15) != 0) {
    size = nextMultipleOf<16>(size);
  }
  VmaAllocator const allocator = device->vmaAllocator();

  VkBufferCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  createInfo.size = size;
  createInfo.usage = usage;
  createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.flags = vmaFlags;
  allocInfo.requiredFlags = required;
  allocInfo.preferredFlags = preferred;
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;  // TODO better
  // TODO export memory if requested
  VkResult const res = vmaCreateBuffer(allocator, &createInfo, &allocInfo,
                                       buffer, allocation, nullptr);

  return res >= 0;
}

}  // namespace avk::vk