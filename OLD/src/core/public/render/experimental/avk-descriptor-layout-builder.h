#pragma once

#include "render/vk/common-vk.h"

// library
#include <vector>

namespace avk::vk {
class Device;
}

namespace avk::experimental {

class DescriptorSetLayoutBuilder {
 public:
  DescriptorSetLayoutBuilder() = default;

  inline uint32_t currentBinding() const {
    return static_cast<uint32_t>(m_bindings.size());
  }

  inline void reset() { m_bindings.clear(); }

  void addUniformBuffer(uint32_t descriptorCount,
                        VkShaderStageFlags stageFlags);
  void addSampler();
  void addSampledImage();

  VkDescriptorSetLayout build(vk::Device const* const device) const;

 private:
  // its size is the next binding being used
  std::vector<VkDescriptorSetLayoutBinding> m_bindings;
};

}  // namespace avk::experimental