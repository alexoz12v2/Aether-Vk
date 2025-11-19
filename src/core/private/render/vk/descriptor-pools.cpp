#include "render/vk/descriptor-pools.h"

#include "render/vk/discard-pool.h"

// standard/runtime
#include <cassert>

// TODO

// TODO possible: query limits and use them
// Limit values from
// https://vulkan.gpuinfo.org/displayreport.php?id=35260#properties
inline uint32_t constexpr POOL_SIZE_STORAGE_BUFFER = 210;
inline uint32_t constexpr POOL_SIZE_STORAGE_IMAGE = 126;
inline uint32_t constexpr POOL_SIZE_COMBINED_IMAGE_SAMPLER = 250;  // much less
inline uint32_t constexpr POOL_SIZE_SAMPLER = 32;
inline uint32_t constexpr POOL_SIZE_SAMPLED_IMAGE = 250;
inline uint32_t constexpr POOL_SIZE_UNIFORM_BUFFER = 216;
inline uint32_t constexpr POOL_SIZE_UNIFORM_TEXEL_BUFFER = 32;
inline uint32_t constexpr POOL_SIZE_INPUT_ATTACHMENT = 9;
inline uint32_t constexpr POOL_SIZES = 8;

inline uint32_t constexpr MAX_DESCRIPTOR_SETS = 256;

namespace avk::vk {

DescriptorPools::DescriptorPools(Device* device) AVK_NO_CFI : m_deps{device} {
  m_recycledPools.reserve(64);
  ensureActivePool();
}

DescriptorPools::~DescriptorPools() AVK_NO_CFI {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  for (VkDescriptorPool pool : m_recycledPools) {
    vkDevApi->vkDestroyDescriptorPool(dev, pool, nullptr);
  }
  m_recycledPools.clear();
  if (m_activePool != VK_NULL_HANDLE) {
    vkDevApi->vkDestroyDescriptorPool(dev, m_activePool, nullptr);
    m_activePool = VK_NULL_HANDLE;
  }
}

VkDescriptorSet DescriptorPools::allocate(
    VkDescriptorSetLayout descriptorSetLayout, DiscardPool* discardPool,
    uint64_t value) AVK_NO_CFI {
  // Note: Not acquiring lock here. You are supposed to use 1
  // VkDescriptorPool per thread concurrently
  // also, discardActive pool acquires the lock.
  assert(descriptorSetLayout != VK_NULL_HANDLE &&
         m_activePool != VK_NULL_HANDLE && discardPool);
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  VkDescriptorSet descriptorSet = VK_NULL_HANDLE;

  VkDescriptorSetAllocateInfo allocInfo{};
  allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  allocInfo.descriptorPool = m_activePool;
  allocInfo.descriptorSetCount = 1;
  allocInfo.pSetLayouts = &descriptorSetLayout;
  VkResult const res =
      vkDevApi->vkAllocateDescriptorSets(dev, &allocInfo, &descriptorSet);
  if (res == VK_ERROR_OUT_OF_POOL_MEMORY || res == VK_ERROR_FRAGMENTED_POOL) {
    discardActivePool(discardPool, value);
    ensureActivePool();
  }
  VK_CHECK(res);

  return descriptorSet;
}

void DescriptorPools::discardActivePool(DiscardPool* discardPool,
                                        uint64_t value) {
  assert(discardPool && m_activePool != VK_NULL_HANDLE);
  std::lock_guard lk{m_mtx};  // note: acquiring 2 locks with discardPool too
  discardPool->discardDescriptorPoolForReuse(m_activePool, this, value);
  m_activePool = VK_NULL_HANDLE;
}

void DescriptorPools::recycle(VkDescriptorPool pool) AVK_NO_CFI {
  if (pool == VK_NULL_HANDLE) {
    return;
  }
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  VK_CHECK(vkDevApi->vkResetDescriptorPool(dev, pool, 0));

  std::lock_guard lk{m_mtx};
  m_recycledPools.push_back(pool);
}

void DescriptorPools::ensureActivePool() AVK_NO_CFI {
  if (m_activePool != VK_NULL_HANDLE) {
    return;
  }

  std::lock_guard lk{m_mtx};
  if (!m_recycledPools.empty()) {
    m_activePool = m_recycledPools.back();
    m_recycledPools.pop_back();
    return;
  }

  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  VkDescriptorPoolSize const poolSizes[POOL_SIZES]{
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, POOL_SIZE_STORAGE_BUFFER},
      {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, POOL_SIZE_STORAGE_IMAGE},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
       POOL_SIZE_COMBINED_IMAGE_SAMPLER},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, POOL_SIZE_UNIFORM_BUFFER},
      {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, POOL_SIZE_UNIFORM_TEXEL_BUFFER},
      {VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, POOL_SIZE_INPUT_ATTACHMENT},
      {VK_DESCRIPTOR_TYPE_SAMPLER, POOL_SIZE_SAMPLER},
      {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, POOL_SIZE_SAMPLED_IMAGE},
  };
  // TODO Inline uniform buffer
  VkDescriptorPoolCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  createInfo.maxSets = MAX_DESCRIPTOR_SETS;
  createInfo.poolSizeCount = POOL_SIZES;
  createInfo.pPoolSizes = poolSizes;
  VK_CHECK(vkDevApi->vkCreateDescriptorPool(dev, &createInfo, nullptr,
                                            &m_activePool));
}

}  // namespace avk::vk