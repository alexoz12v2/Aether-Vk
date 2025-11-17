#include "render/experimental/avk-ktx2-textures.h"

#include "render/vk/device-vk.h"
#include "render/vk/discard-pool.h"
#include "render/vk/instance-vk.h"

// stuff
#include <ktxvulkan.h>

// std
#include <cassert>
#include <string>
#include <unordered_map>

// TODO see KHR_texture_basisu glTF Extensionw ->
//   KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT

namespace avk::experimental {

static ktxVulkanFunctions ktxTableFromVolkTable(
    VolkDeviceTable const* volkTable) {
  ktxVulkanFunctions ktxTable{};
  ktxTable.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
  ktxTable.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
  ktxTable.vkAllocateCommandBuffers = volkTable->vkAllocateCommandBuffers;
  ktxTable.vkAllocateMemory = volkTable->vkAllocateMemory;
  ktxTable.vkBeginCommandBuffer = volkTable->vkBeginCommandBuffer;
  ktxTable.vkBindBufferMemory = volkTable->vkBindBufferMemory;
  ktxTable.vkBindImageMemory = volkTable->vkBindImageMemory;
  ktxTable.vkCmdBlitImage = volkTable->vkCmdBlitImage;
  ktxTable.vkCmdCopyBufferToImage = volkTable->vkCmdCopyBufferToImage;
  ktxTable.vkCmdPipelineBarrier = volkTable->vkCmdPipelineBarrier;
  ktxTable.vkCreateImage = volkTable->vkCreateImage;
  ktxTable.vkDestroyImage = volkTable->vkDestroyImage;
  ktxTable.vkCreateBuffer = volkTable->vkCreateBuffer;
  ktxTable.vkDestroyBuffer = volkTable->vkDestroyBuffer;
  ktxTable.vkCreateFence = volkTable->vkCreateFence;
  ktxTable.vkDestroyFence = volkTable->vkDestroyFence;
  ktxTable.vkEndCommandBuffer = volkTable->vkEndCommandBuffer;
  ktxTable.vkFreeCommandBuffers = volkTable->vkFreeCommandBuffers;
  ktxTable.vkFreeMemory = volkTable->vkFreeMemory;
  ktxTable.vkGetBufferMemoryRequirements =
      volkTable->vkGetBufferMemoryRequirements;
  ktxTable.vkGetImageMemoryRequirements =
      volkTable->vkGetImageMemoryRequirements;
  ktxTable.vkGetImageSubresourceLayout = volkTable->vkGetImageSubresourceLayout;
  ktxTable.vkGetPhysicalDeviceImageFormatProperties =
      vkGetPhysicalDeviceImageFormatProperties;
  ktxTable.vkGetPhysicalDeviceFormatProperties =
      vkGetPhysicalDeviceFormatProperties;
  ktxTable.vkGetPhysicalDeviceMemoryProperties =
      vkGetPhysicalDeviceMemoryProperties;
  ktxTable.vkMapMemory = volkTable->vkMapMemory;
  ktxTable.vkQueueSubmit = volkTable->vkQueueSubmit;
  ktxTable.vkQueueWaitIdle = volkTable->vkQueueWaitIdle;
  ktxTable.vkUnmapMemory = volkTable->vkUnmapMemory;
  ktxTable.vkWaitForFences = volkTable->vkWaitForFences;
  return ktxTable;
}

struct KtxThreadLocalAlloc {
  std::mutex mtx;
  uint64_t nextId = 1;
  // this is ugly, but it's fine as long as there's one device
  VmaAllocator allocator;
  std::unordered_map<uint64_t, VmaAllocation> allocations;
};

static thread_local KtxThreadLocalAlloc gKtxTls;

static uint64_t ktxAllocMem(VkMemoryAllocateInfo* allocInfo,
                            VkMemoryRequirements* memReq, uint64_t* pageCount) {
  // Register allocation in thread-local ID map
  std::lock_guard<std::mutex> lock(gKtxTls.mtx);
  assert(gKtxTls.allocator);

  // Required by KTX (we do not use paging in VMA)
  if (pageCount) {
    *pageCount = 1;
  }

  // Fill VMA allocation info from Vulkan alloc + memreq
  VmaAllocationCreateInfo ci{};
  ci.usage = VMA_MEMORY_USAGE_AUTO;
  ci.priority = 1.0f;

  // Respect Vulkan's memory type mask
  ci.requiredFlags = 0;  // optional
  ci.memoryTypeBits = allocInfo->memoryTypeIndex < 32
                          ? (1u << allocInfo->memoryTypeIndex)
                          : 0xFFFFFFFFu;

  // Perform allocation with VMA
  VmaAllocation allocation = VK_NULL_HANDLE;
  VmaAllocationInfo outInfo{};

  VkResult const res =
      vmaAllocateMemory(gKtxTls.allocator, memReq, &ci, &allocation, &outInfo);

  if (res != VK_SUCCESS) {
    return 0;  // KTX interprets 0 as "allocation failed"
  }

  uint64_t const id = gKtxTls.nextId++;
  gKtxTls.allocations.try_emplace(id, allocation);

  return id;
}

static VkResult ktxBindBuffer(VkBuffer buffer, uint64_t allocationId) {
  std::lock_guard lock(gKtxTls.mtx);

  auto it = gKtxTls.allocations.find(allocationId);
  if (it == gKtxTls.allocations.end()) {
    return VK_ERROR_MEMORY_MAP_FAILED;
  }

  VmaAllocation alloc = it->second;
  return vmaBindBufferMemory(gKtxTls.allocator, alloc, buffer);
}

static VkResult ktxBindImage(VkImage image, uint64_t allocationId) {
  std::lock_guard lock(gKtxTls.mtx);

  auto it = gKtxTls.allocations.find(allocationId);
  if (it == gKtxTls.allocations.end()) {
    return VK_ERROR_UNKNOWN;  // "unknown allocation"
  }

  VmaAllocation const alloc = it->second;
  return vmaBindImageMemory(gKtxTls.allocator, alloc, image);
}

static VkResult ktxMemoryMap(uint64_t allocId,
                             [[maybe_unused]] uint64_t pageNumber,
                             VkDeviceSize* mapLength, void** dataPtr) {
  if (!dataPtr) return VK_ERROR_UNKNOWN;
  std::lock_guard lock(gKtxTls.mtx);

  auto const it = gKtxTls.allocations.find(allocId);
  if (it == gKtxTls.allocations.end()) return VK_ERROR_MEMORY_MAP_FAILED;

  VmaAllocation const alloc = it->second;
  if (mapLength) {
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(gKtxTls.allocator, alloc, &info);
    *mapLength = info.size;
  }

  return vmaMapMemory(gKtxTls.allocator, alloc, dataPtr);
}

static void ktxMemoryUnmap(uint64_t allocId,
                           [[maybe_unused]] uint64_t pageNumber) {
  std::lock_guard<std::mutex> lock(gKtxTls.mtx);

  auto const it = gKtxTls.allocations.find(allocId);
  if (it == gKtxTls.allocations.end()) return;

  VmaAllocation const alloc = it->second;
  vmaUnmapMemory(gKtxTls.allocator, alloc);
}

static void ktxFreeMem(uint64_t allocId) {
  std::lock_guard lock{gKtxTls.mtx};
  auto const it = gKtxTls.allocations.find(allocId);
  if (it == gKtxTls.allocations.end()) return;
  VmaAllocation const alloc = it->second;
  gKtxTls.allocations.erase(it);
  vmaFreeMemory(gKtxTls.allocator, alloc);
}

static ktxVulkanTexture_subAllocatorCallbacks ktxAllocCallbacksFromVMA(
    vk::Device* device) {
  ktxVulkanTexture_subAllocatorCallbacks callbacks{};
  {
    std::lock_guard lk{gKtxTls.mtx};
    gKtxTls.allocator = device->vmaAllocator();
  }
  callbacks.allocMemFuncPtr = ktxAllocMem;
  callbacks.bindBufferFuncPtr = ktxBindBuffer;
  callbacks.bindImageFuncPtr = ktxBindImage;
  callbacks.memoryMapFuncPtr = ktxMemoryMap;
  callbacks.memoryUnmapFuncPtr = ktxMemoryUnmap;
  callbacks.freeMemFuncPtr = ktxFreeMem;
  return callbacks;
}

static void openTexture(vk::Instance* instance, vk::Device* device,
                        VkCommandPool commandPool) {
  KTX_error_code err = KTX_SUCCESS;
  // construct function called once
  ktxVulkanDeviceInfo ktxDeviceInfo{};
  ktxVulkanFunctions const ktxVulkanTable =
      ktxTableFromVolkTable(device->table());
  err = ktxVulkanDeviceInfo_ConstructEx(
      &ktxDeviceInfo, instance->handle(), device->physicalDevice(),
      device->device(), device->queue(), commandPool, nullptr, &ktxVulkanTable);
  if (err != KTX_SUCCESS) {
    showErrorScreenAndExit("Couldn't create ktxVulkandEviceInfo");
  }

  // VMA bindings
  ktxVulkanTexture_subAllocatorCallbacks const ktrAllocCallbacks =
      ktxAllocCallbacksFromVMA(device);

  // texture creation
  ktxTexture2* texture = nullptr;
  err = ktxTexture2_CreateFromNamedFile("assets/starry-night-uastc.ktx2",
                                        KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
                                        &texture);
  // TODO better
  if (err != KTX_SUCCESS) {
    using namespace std::string_literals;
    std::string const error =
        "[ktxTexture2_CreateFromNamedFile] Error: "s + ktxErrorString(err);
    showErrorScreenAndExit(error.c_str());
  }
  // transcode depending on device features support for compression formats
  // upload to vulkan `ktxTexture2_VkUploadEx_WithSuballocator`
}

}  // namespace avk::experimental