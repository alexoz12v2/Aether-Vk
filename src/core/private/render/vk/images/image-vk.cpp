#include "render/vk/images/image-vk.h"

// stuff
#include "render/vk/device-vk.h"

// library
#include <cassert>

namespace avk::vk {

bool createImage(Device* device, SingleImage2DSpecVk const& spec,
                 VkImage* image, VmaAllocation* allocation) {
  assert(device && image && allocation);
  VmaAllocator const allocator = device->vmaAllocator();
  VkImageCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  createInfo.extent.width = spec.width;
  createInfo.extent.height = spec.height;
  createInfo.extent.depth = 1;
  createInfo.mipLevels = 1;
  createInfo.arrayLayers = 1;
  createInfo.format = spec.format;
  createInfo.tiling = spec.imageTiling;
  createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  createInfo.usage = spec.usage;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  // TODO better only for extenral memory images
  // allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocInfo.priority = 1.f;

  VkResult const res = vmaCreateImage(allocator, &createInfo, &allocInfo, image,
                                      allocation, nullptr);
  return res >= 0;
}

}  // namespace avk::vk