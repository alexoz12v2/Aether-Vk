#pragma once

#include "render/experimental/avk-basic-buffer-manager.h"
#include "render/vk/common-vk.h"
#include "utils/mixins.h"

// std
#include <vector>

namespace avk::experimental {

struct StagingOperation {
  VkBuffer dstBuffer = VK_NULL_HANDLE;
  VmaAllocation dstAlloc = VK_NULL_HANDLE;
  void const* srcData = nullptr;
  size_t srcBytes = 0;
  uint64_t stagingId = fnv1aHash("empty");  // 10398793620191490734
  VkPipelineStageFlags dstStage = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT;
  VkAccessFlags dstAccess = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
};

/// Class which, During VkCommandBuffer recording, it can automate some of
/// the boilerplate necessary to upload some data from a staging buffer obtained
/// from a VMA suballocation, upload to a GPU-Only target buffer, track
/// the necessary pipeline barriers to ensure proper synchronization of these
/// copy operations, and insert them in the command buffer at flush
/// - works with 1 queue only
/// called "transient" because instead of taking vulkan handles as dependencies,
/// they are "refreshed" every timeline
/// - meant to be used by 1 thread only (as it's linked to a command buffer)
class StagingTransientManager : public NonMoveable {
 public:
  StagingTransientManager();

  void enqueue(StagingOperation const& op);
  void flush();
  void refresh(VkCommandBuffer cmd, BufferManager* bufferManager,
               VolkDeviceTable const* vkDevApi, vk::DiscardPool* discardPool,
               VmaAllocator allocator, uint64_t timeline);

 private:
  // transient resources
  struct Transient {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VolkDeviceTable const* vkDevApi = nullptr;
    vk::DiscardPool* discardPool = nullptr;
    BufferManager* bufferManager = nullptr;
    VmaAllocator allocator = VK_NULL_HANDLE;
    uint64_t timeline = -1;
  } m_tmp;

  // do a better job in memory management? note that we are allocating much of
  // the memory at object construction
  std::vector<StagingOperation> m_stagingOps;
  std::unordered_map<VkPipelineStageFlags, std::vector<VkBufferMemoryBarrier>>
      m_barriersPerStage;

  // TODO kept for debugging, remove later
  bool refreshed = false;
};

}  // namespace avk::experimental