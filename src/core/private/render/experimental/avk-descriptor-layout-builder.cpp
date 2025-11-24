#include "render/experimental/avk-descriptor-layout-builder.h"

#include "render/vk/device-vk.h"

// library
#include <cassert>

namespace avk::experimental {

void DescriptorSetLayoutBuilder::addUniformBuffer(
    uint32_t descriptorCount, VkShaderStageFlags stageFlags) {
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = static_cast<uint32_t>(m_bindings.size());
  binding.descriptorCount = descriptorCount;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
  binding.stageFlags = stageFlags;
  m_bindings.push_back(binding);
}

void DescriptorSetLayoutBuilder::addSampler() {
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = static_cast<uint32_t>(m_bindings.size());
  binding.descriptorCount = 1;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  m_bindings.push_back(binding);
}

void DescriptorSetLayoutBuilder::addSampledImage() {
  VkDescriptorSetLayoutBinding binding{};
  binding.binding = static_cast<uint32_t>(m_bindings.size());
  binding.descriptorCount = 1;
  binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  m_bindings.push_back(binding);
}
VkDescriptorSetLayout DescriptorSetLayoutBuilder::build(
    vk::Device const* const device) const {
  VkDescriptorSetLayoutCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  createInfo.bindingCount = static_cast<uint32_t>(m_bindings.size());
  createInfo.pBindings = m_bindings.data();
  VkDescriptorSetLayout layout = VK_NULL_HANDLE;
  VK_CHECK(device->table()->vkCreateDescriptorSetLayout(
      device->device(), &createInfo, nullptr, &layout));
  return layout;
}

}  // namespace avk::experimental