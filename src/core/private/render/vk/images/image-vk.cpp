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
  createInfo.imageType = VK_IMAGE_TYPE_2D;
  createInfo.extent.width = spec.width;
  createInfo.extent.height = spec.height;
  createInfo.extent.depth = 1;
  createInfo.mipLevels = 1;
  createInfo.arrayLayers = 1;
  // note: format cannot be undefined unless you export this memory
  // which is signaled by a OS-specific pNext
  createInfo.format = spec.format;
  createInfo.tiling = spec.imageTiling;
  createInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  createInfo.usage = spec.usage;
  createInfo.samples = spec.samples;

  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  // TODO better only for extenral memory images
  // allocInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
  allocInfo.priority = 1.f;

  VkResult const res = vmaCreateImage(allocator, &createInfo, &allocInfo, image,
                                      allocation, nullptr);
  return res >= 0;
}

Expected<VkImageView> depthStencilImageView(Device* device, VkImage image,
                                            VkFormat fmt) {
  assert(device && *device);
  auto const* const vkDevApi = device->table();
  VkDevice const dev = device->device();

  VkImageViewCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  createInfo.image = image;
  createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  createInfo.format = fmt;
  createInfo.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  createInfo.subresourceRange.baseArrayLayer = 0;
  createInfo.subresourceRange.layerCount = 1;
  createInfo.subresourceRange.baseMipLevel = 0;
  createInfo.subresourceRange.levelCount = 1;

  VkImageView view = VK_NULL_HANDLE;
  VkResult const res =
      vkDevApi->vkCreateImageView(dev, &createInfo, nullptr, &view);
  return {view, res};
}

}  // namespace avk::vk