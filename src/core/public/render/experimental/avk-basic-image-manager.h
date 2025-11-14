#pragma once

#include "render/vk/common-vk.h"
#include "render/vk/device-vk.h"
#include "render/vk/discard-pool.h"
#include "utils/mixins.h"

// library
#include <shared_mutex>
#include <unordered_map>

// TODO: Same todos of buffer manager

namespace avk::experimental {

/// Vulkan Images and their allocations following VMA Guidelines
/// images can be allocated by multiple threads concurrently, but each
/// thread should free its own images. This class won't destroy its
/// resources automatically
/// Note: There is no "staging image", because you would use a staging buffer
/// for that with `vkCmdCopyBufferToImage`
class ImageManager : public NonMoveable {
 public:
  static constexpr int32_t Success = 0;
  static constexpr int32_t VulkanError = -1;
  static constexpr int32_t Collision = -2;

  explicit ImageManager(vk::Device* device, size_t cap = 256);

  /// create transient attachment (depth/stencil or color)
  /// these are temporary images which do not need a memory backing inside
  /// main memory (SoC systems, where main memory = GPU global memory),
  /// hence **Tile Cache** inside Tile-Based GPUs are sufficient to hold such
  /// information
  /// Note: depth/stencil with more than one sample requires
  /// VK_EXT_depth_stencil_resolve Note: assumes these are critical GPU-only
  /// resources hence `VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT`
  ///
  /// - Note: This is half of the process to ensure attachments fall into Tile
  /// Cache.
  ///  the other half is using `BY_REGION` subpass dependencies
  ///  + `STORE_OP_DONT_CARE`  and `LOAD_OP_DONT_CARE` or `LOAD_OP_CLEAR`
  /// - Note: We now support only 2D images
  /// - Note: If `LAZILY_ALLOCATED_BIT` is chosen, it's probably not
  /// `HOST_VISIBLE`
  int32_t createTransientAttachment(
      uint64_t id, VkExtent2D extent, VkFormat format, VkImageUsageFlags usage,
      VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT);

  /// Create Sampled texture for an object. Assumes it's GPU-local and not
  /// critical. Note: By default, this is GPU-only. if flag `streamingNeeded`
  /// is given, then on SoC systems it should be mappable, while on discrete GPU
  /// we use `VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT`,
  /// meaning we let VMA figure out if the memory chosen can be mapped,
  /// otherwise you need to use a staging buffer and a transfer operation
  ///
  /// Note: Budget is tracked by `Device` class
  ///
  /// \param streamingNeeded if true, signal that CPU frequently writes this
  /// \param forceWithinBudget if true,
  /// `VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT` used, meaning VMA rejects
  /// allocations exceeding the budget for its first VK memory type choice
  /// (default is find next eligible memory type)
  /// \param forceNoAllocation if true,
  /// `VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT` used, meaning VMA only uses
  /// pre-existing memory blocks, otherwise return error
  /// \return 0 on success
  int32_t createTexture(uint64_t id, VkExtent2D extent, VkFormat format,
                        VkImageUsageFlags usage, VkSampleCountFlagBits samples,
                        bool streamingNeeded, bool forceWithinBudget,
                        bool forceNoAllocation);

  /// false if it doesn't find anything
  bool get(uint64_t id, VkImage& outImage, VmaAllocation& outAlloc);

  /// remove the discarded image from hash table and insert it in discard pool
  /// if nothing is found return false
  /// \warning assumes `discardPool` is on the same `vk::Device`
  bool discardById(vk::DiscardPool* discardPool, uint64_t id, uint64_t timeline);

  /// to be called only on program shutdown if needed. Normally each thread discards
  /// its own images
  /// \warning assumes `discardPool` is on the same `vk::Device`
  void discardEverything(vk::DiscardPool* discardPool, uint64_t timeline);

 private:
  // dependencies which must outlive the object
  struct Deps {
    vk::Device* device;
  } m_deps;

  /// structure to hold all handles. Other objects/functions should always
  /// request them from here when needed and discard them manually when not used
  /// anymore
  std::unordered_map<uint64_t, vk::VMAResource<VkImage>> m_imageMap;

  /// synchronization primitive to achieve shared read, single write. Note: When
  /// a thread reads a resource, the element is not locked inside the hash
  /// table, someone else could come and delete it! hence each thread should own
  /// its own resources (preferably, embedding TID inside hash)
  mutable std::shared_mutex m_mtx;
};

}  // namespace avk::experimental