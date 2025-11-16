#include "render/experimental/avk-staging-transient-manager.h"

#include <array>
#include <cassert>

namespace avk::experimental {

StagingTransientManager::StagingTransientManager() {
  // some good enough capacity
  m_stagingOps.reserve(64);

  // TODO add more if needed
  static constexpr std::array<VkPipelineStageFlags, 17> StageFlags{
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
      VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
      VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT,
      VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT,
      VK_PIPELINE_STAGE_GEOMETRY_SHADER_BIT,
      VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
      VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
      VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_HOST_BIT,
      VK_PIPELINE_STAGE_ALL_GRAPHICS_BIT,
      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
  };
  // again, some capacity for each stage
  m_barriersPerStage.reserve(64);
  for (auto const stage : StageFlags) {
    m_barriersPerStage.try_emplace(stage);
  }
  m_barriersPerStage.rehash(32);
  for (auto const stage : StageFlags) {
    m_barriersPerStage.at(stage).reserve(16);
  }
}

void StagingTransientManager::enqueue(StagingOperation const& op) {
  assert(refreshed);
  m_stagingOps.push_back(op);
}

// TODO possibly handle out of budget gracefully
void StagingTransientManager::flush() {
  assert(refreshed);
  for (auto const& op : m_stagingOps) {
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    int res = m_tmp.bufferManager->createBufferStaging(
        op.stagingId, op.srcBytes, true, false);
    if (res)
      showErrorScreenAndExit(("Couldn't allocate staging buffer for id " +
                              std::to_string(op.stagingId))
                                 .c_str());
    if (!m_tmp.bufferManager->get(op.stagingId, stagingBuf, stagingAlloc))
      showErrorScreenAndExit(
          ("Couldn't get staging buffer for id " + std::to_string(op.stagingId))
              .c_str());
    // copy host data
    VK_CHECK(vmaCopyMemoryToAllocation(m_tmp.allocator, op.srcData,
                                       stagingAlloc, 0, op.srcBytes));

    // host -> transfer barrier (cannot be grouped as it needs to affect
    // the copy command which comes immediately after)
    VkBufferMemoryBarrier barrier = {};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // host
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // don't care
    barrier.buffer = stagingBuf;
    barrier.offset = 0;
    barrier.size = op.srcBytes;  // VK_WHOLE_SIZE equivalent

    m_tmp.vkDevApi->vkCmdPipelineBarrier(m_tmp.cmd, VK_PIPELINE_STAGE_HOST_BIT,
                                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                         nullptr, 1, &barrier, 0, nullptr);
    // copy staging -> destination buffer
    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size = op.srcBytes;
    m_tmp.vkDevApi->vkCmdCopyBuffer(m_tmp.cmd, stagingBuf, op.dstBuffer, 1,
                                    &copy);
    // transfer -> destination stage pipeline barrier
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = op.dstAccess;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // no ownership tsfr
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // no ownership tsfr
    barrier.buffer = op.dstBuffer;
    barrier.offset = 0;
    barrier.size = op.srcBytes;
    m_barriersPerStage.at(op.dstStage).push_back(barrier);

    // discard staging buffer on next timeline
    m_tmp.bufferManager->discardById(m_tmp.discardPool, op.stagingId,
                                     m_tmp.timeline);
  }
  m_stagingOps.clear();

  for (auto& [stage, barriers] : m_barriersPerStage) {
    if (barriers.empty()) continue;
    m_tmp.vkDevApi->vkCmdPipelineBarrier(
        m_tmp.cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, stage, 0, 0, nullptr,
        static_cast<uint32_t>(barriers.size()), barriers.data(), 0, nullptr);

    barriers.clear();
  }

  // reset everything
  m_tmp.cmd = VK_NULL_HANDLE;
  m_tmp.bufferManager = nullptr;
  m_tmp.vkDevApi = nullptr;
  m_tmp.discardPool = nullptr;
  m_tmp.allocator = VK_NULL_HANDLE;
  m_tmp.timeline = -1;
  refreshed = false;
}

void StagingTransientManager::refresh(VkCommandBuffer cmd,
                                      BufferManager* bufferManager,
                                      VolkDeviceTable const* vkDevApi,
                                      vk::DiscardPool* discardPool,
                                      VmaAllocator allocator,
                                      uint64_t timeline) {
  assert(!refreshed);
  m_tmp.cmd = cmd;
  m_tmp.vkDevApi = vkDevApi;
  m_tmp.discardPool = discardPool;
  m_tmp.bufferManager = bufferManager;
  m_tmp.allocator = allocator;
  m_tmp.timeline = timeline;
  refreshed = true;
}

}  // namespace avk::experimental
