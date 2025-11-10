#pragma once

#include "render/vk/common-vk.h"

namespace avk::vk {

class Device;

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
/// Remember to use `VK_IMAGE_USAGE_TRANSIENT_BIT` for attachments
/// you don't need after the renderpass
bool createImage(Device* device, SingleImage2DSpecVk const& spec,
                 VkImage* image, VmaAllocation* allocation);

}  // namespace avk::vk