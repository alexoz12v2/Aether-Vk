#pragma once

#include "os/avk-core-macros.h"
#include "render/vk/common-vk.h"
#include "render/vk/device-vk.h"
#include "render/vk/discard-pool.h"
#include "render/vk/pipeline-info.h"
#include "utils/mixins.h"

// libs and stuff
#include <shared_mutex>
#include <unordered_map>

namespace avk::experimental {

// TODO VMA_MEMORY_USAGE_GPU_LAZILY_ALLOCATED when using transient images
// TODO VMA_ALLCOATION_CREATE_WITHIN_BUDGET_BIT
// TODO VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT

/// Builder/Manager class which helps into creating different categories of
/// buffers prefilling the necessary flags for `vk::createBuffer`.
/// all buffers are owned by the buffer manager *but not freed automatically*,
/// hence they should be discarded by going through its own interface
///
/// Factory Methods Follow the following guidelines.
/// <https://gpuopen-librariesandsdks.github.io/VulkanMemoryAllocator/html/usage_patterns.html>
/// Note: Assumes VK_SHARING_MODE_EXCLUSIVE
/// WARN: It does not protect from multithreaded discard. The thread which
/// creates the buffer should be the thread which discards it
class BufferManager : public NonMoveable {
 public:
  static int32_t constexpr Success = 0;
  static int32_t constexpr VulkanError = -1;
  static int32_t constexpr Collision = -2;
  static int32_t constexpr TransferRequired = 1;

  /// start Buffer manager with a given capacity
  BufferManager(vk::Device* device, size_t cap = 256);

  /// Creates a buffer used as a DEVICE_LOCAL only resource.
  /// - On Dedicated GPUs, should be paired with a staging buffer
  /// - On SoC or some Integrated GPUs, used as standalone
  /// They are resources which are either large or recreated frequently,
  /// eg attachments. Why for frequent things? If they are too frequent,
  /// the resulting fragmentation outweighs the memory allocation overhead
  /// If the device is a SoC, then
  /// `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT` will be added
  /// such that the buffer becomes mappable from host
  /// \param forceWithinBudget if true,
  /// `VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT` used, meaning VMA rejects
  /// allocations exceeding the budget for its first VK memory type choice
  /// (default is find next eligible memory type)
  /// \param forceNoAllocation if true,
  /// `VMA_ALLOCATION_CREATE_NEVER_ALLOCATE_BIT` used, meaning VMA only uses
  /// pre-existing memory blocks, otherwise return error
  /// \return 0 on success
  int32_t createBufferGPUOnly(uint64_t id, size_t bytes,
                              VkBufferCreateFlags usage, bool forceWithinBudget,
                              bool forceNoAllocation);

  /// Creates a HOST_VISIBLE buffer (use preferred for host coherent)
  /// uses VMA_ALLOCATION_CREATE_MAPPED_BIT. Should be used to
  /// upload data from host accessible memory to DEVICE_LOCAL memory into
  /// Dedicated GPU setups
  /// The users should use `vmaGetAllocationMemoryProperties` such that it
  /// can decide, whether `VK_MEkORY_PROPERTY_HOST_COHERENT` is present or not,
  /// whether to call `vmaFlushAllocation`
  /// Note: Uses `VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT`,
  /// meaning
  ///   VMA library assumes host will write sequentially to mapped memory
  /// Note: Makes this buffer persistently mapped
  /// `VMA_ALLOCATION_CREATE_MAPPED_BIT`,
  ///  hence no need to call `vmaMapMemory` or `vmaUnmapMemory`
  int32_t createBufferStaging(uint64_t id, size_t bytes, bool forceWithinBudget,
                              bool forceNoAllocation);

  /// Since, if you don't use VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT`,
  /// VMA will try to use allocated blocks, sometimes it's more convenient
  /// to create staging buffers on the spot
  /// see `createBufferStaging` for details on staging buffer in general
  /// TODO: Support for preexisting buffer binding

  /// buffer which should receive data from the result of a computation to
  /// be read back into main memory. As staging buffers, they are useful
  /// for Dedicated GPU setups. Read `createBufferStaging` for more
  /// Differences from staging:
  /// - `VK_BUFFER_USAGE_TRANSFER_DST_BIT` (not `SRC`)
  /// - `VMA_ALLOCATION_CRAETE_HOST_ACCESS_RANDOM_BIT` (not sequential)
  int32_t createBufferReadback(uint64_t id, size_t bytes,
                               bool forceWithinBudget, bool forceNoAllocation);

  /// Buffers frequently written by the CPU and frequently written by the GPU
  /// meaning it is one of
  /// - HOST_VISIBLE memory in RAM on systems with discrete GPU
  /// - integrated graphics/SoC: the only type of memory heap available is
  /// system memory
  ///   fest to access by both host and device
  /// - HOST_VISIBLE | DEVICE_LOCAL on discrete GPU systems, also known as
  ///   `Base Address Bar`, a portion of VRAM mapped to host memory. CPU writes
  ///   go through the PCIe Bus
  ///
  /// Note: By using the
  /// `VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT`,
  ///  VMA Will choose between Device and Host memory (Mappable, hence visible
  ///  to both), whichever it deems will yield more performance. This means that
  ///  on dedicated GPU setups, the user is required to use
  ///  `vmaGetAllocationMemoryProperties` to check whether
  ///  `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT`, as there might be cases in which
  ///   VMA declares that, even for streaming, the optimal solution is a staging
  ///   buffer setup for a particular allocation
  /// \param usage function will add to this the
  /// `VK_BUFFER_USAGE_TRANSFER_DST_BIT to handle transfer results`
  /// \returns `Transfer` value to allow user to know in advance it it needs to
  /// prepare a staging buffer
  int32_t createBufferStreaming(uint64_t id, size_t bytes,
                                VkBufferUsageFlags usage,
                                bool forceWithinBudget, bool forceNoAllocation);

  /// false if it doesn't find anything
  bool get(uint64_t id, VkBuffer& outBuffer, VmaAllocation& outAlloc);

  /// removes the discarded buffer from the hash table. False if not found
  /// \warning assumes `discardPool` is on the same `vk::Device`
  bool discardById(vk::DiscardPool* discardPool, uint64_t id,
                   uint64_t timeline);

  /// To be called only on program shutdown, because each thread should discard
  /// it own buffers
  /// \warning assumes `discardPool` is on the same `vk::Device`
  void discardEverything(vk::DiscardPool* discard, uint64_t timeline);

 private:
  // dependencies which should outlive this object
  struct Deps {
    vk::Device* device;
  } m_deps;
  /// hash table of buffers. Meaning is given by user
  std::unordered_map<uint64_t, vk::VMAResource<VkBuffer>> m_bufferMap;
  /// synchronization for map modifications/reads
  mutable std::shared_mutex m_mtx;
};

}  // namespace avk::experimental
