#include "render/vk/descriptor-pools.h"

// TODO

// TODO possible: query limits and use them
// Limit values from
// https://vulkan.gpuinfo.org/displayreport.php?id=35260#properties
inline uint32_t constexpr POOL_SIZE_STORAGE_BUFFER = 210;
inline uint32_t constexpr POOL_SIZE_STORAGE_IMAGE = 126;
inline uint32_t constexpr POOL_SIZE_COMBINED_IMAGE_SAMPLER = 250;  // much less
inline uint32_t constexpr POOL_SIZE_UNIFORM_BUFFER = 216;
inline uint32_t constexpr POOL_SIZE_UNIFORM_TEXEL_BUFFER = 32;
inline uint32_t constexpr POOL_SIZE_INPUT_ATTACHMENT = 9;

inline uint32_t constexpr MAX_DESCRIPTOR_SETS = 256;

namespace avk::vk {

DescriptorPools::DescriptorPools(Device* device) AVK_NO_CFI : m_deps{device} {
  m_recycledPools.reserve(64);
}

DescriptorPools::~DescriptorPools() AVK_NO_CFI {}

VkDescriptorSet DescriptorPools::allocate(
    VkDescriptorSetLayout descriptorSetLayout) AVK_NO_CFI {
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
  return descriptorSet;
}

void DescriptorPools::recycle(VkDescriptorPool pool) AVK_NO_CFI {}

}  // namespace avk::vk