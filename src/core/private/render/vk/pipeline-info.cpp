#include "render/vk/pipeline-info.h"

#include "render/vk/device-vk.h"

namespace avk::vk {

VkPipelineLayout createPipelineLayout(
    Device* device, VkDescriptorSetLayout const* pDescriptorSetLayouts,
    uint32_t descriptorSetLayoutCount,
    VkPushConstantRange const* pPushConstantRanges,
    uint32_t pushConstantRangeCount) AVK_NO_CFI {
  auto const* const vkDevApi = device->table();
  VkDevice const dev = device->device();
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
  VK_CHECK(vkDevApi->vkCreatePipelineLayout(dev, &createInfo, nullptr,
                                            &pipelineLayout));
  return pipelineLayout;
}

}  // namespace avk::vk