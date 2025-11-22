#include "render/experimental/avk-ktx2-textures.h"

#include "render/vk/device-vk.h"
#include "render/vk/discard-pool.h"
#include "render/vk/instance-vk.h"

// stuff
#pragma clang attribute push(__attribute__((no_sanitize("cfi"))), \
                             apply_to = any(function))
#include <ktxvulkan.h>
#pragma clang attribute pop

// std
#include <cassert>
#include <optional>
#include <string>
#include <unordered_map>

// TODO see KHR_texture_basisu glTF Extensionw ->
//   KTX_TEXTURE_CREATE_CHECK_GLTF_BASISU_BIT

#define KTX_CHECK(res)                       \
  do {                                       \
    KTX_error_code err = (res);              \
    if (err != KTX_SUCCESS) {                \
      showErrorScreenAndExit("KTX failure"); \
    }                                        \
  } while (0)

namespace avk::experimental {

// ---------- Necessary Static Global State for KTX suballoc callbacks -------

struct KtxThreadLocalAlloc : public NonMoveable {
  explicit KtxThreadLocalAlloc(vk::Device* device) : device(device) {
    allocations.reserve(64);
  }
  ~KtxThreadLocalAlloc() noexcept {
    if (device) {
      std::lock_guard lk(mtx);
      // You shouldn't get to the cleanup with some pending allocations
      assert(allocations.empty());
      for (auto const& [id, alloc] : allocations) {
        LOGI << "[Global KTX2 Texture Manager] destroy alloc " << (void*)alloc
             << std::endl;
        vmaFreeMemory(device->vmaAllocator(), alloc);
      }
      LOGW << AVK_LOG_YLW
          "[Global KTX2 Texture Manager] cleaned all VmaAllocation "
          "Objects" AVK_LOG_RST
           << std::endl;
      allocations.clear();
      device = nullptr;
    }
  }

  std::mutex mtx;
  uint64_t nextId = 1;
  // this is ugly, but it's fine as long as there's one device
  vk::Device* device;
  std::unordered_map<uint64_t, VmaAllocation> allocations;
};

static std::optional<KtxThreadLocalAlloc> gKtxAllocManager = std::nullopt;

// Helper: set the allocator once (call in initialize() or before any ktx op)
static void setGlobalKtxDevice(vk::Device* device) {
  if (gKtxAllocManager.has_value()) {
    gKtxAllocManager.reset();
  }
  gKtxAllocManager.emplace(device);
}

// --------------------- Static Stuff ----------------------------------------

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

#define KTX_CALL_PREAMBLE                      \
  assert(gKtxAllocManager.has_value());        \
  std::lock_guard lock(gKtxAllocManager->mtx); \
  vk::Device const* const device = gKtxAllocManager->device

static uint64_t ktxAllocMem(VkMemoryAllocateInfo* allocInfo,
                            VkMemoryRequirements* memReq,
                            uint64_t* pageCount) AVK_NO_CFI {
  // Register allocation in global ID map
  KTX_CALL_PREAMBLE;
  if (!device) return 0;

  // Required by KTX (we do not use paging in VMA)
  if (pageCount) {
    *pageCount = 1;
  }

  // Fill VMA allocation info from Vulkan alloc + memreq
  VmaAllocationCreateInfo ci{};
  ci.usage = VMA_MEMORY_USAGE_UNKNOWN;
  ci.priority = 1.0f;

  // we need to know whether this should be a staging memory area or a GPU only
  VkMemoryPropertyFlags props = 0;
  {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(device->physicalDevice(), &memProps);
    props = memProps.memoryTypes[allocInfo->memoryTypeIndex].propertyFlags;
  }
  bool const isDeviceLocal = props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  bool const isHostVisible = props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
  if (isHostVisible) {
    ci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
  }
  // TODO: VmaPool
  if (isDeviceLocal && isHostVisible && !device->isSoC()) {
    // From VMA's documentation:
    // Systems with a discrete graphics card and separate video memory may or
    // may not expose a memory type that is both HOST_VISIBLE and DEVICE_LOCAL,
    // also known as Base Address Register (BAR). If they do, it represents a
    // piece of VRAM (or entire VRAM, if ReBAR is enabled in the motherboard
    // BIOS) that is available to CPU for mapping. Writes performed by the host
    // to that memory go through PCI Express bus. The performance of these
    // writes may be limited, but it may be fine, especially on PCIe 4.0, as
    // long as rules of using uncached and write-combined memory are followed -
    // only sequential writes and no reads.
    LOGW << AVK_LOG_YLW
        "[ktxAllocMem] detected DEVICE_LOCAL + HOST_VISIBLE. "
        "If this is not a SoC/Integrated GPU Chip, then you are allocating to "
        "the Base Address Register, which is probably not the desired "
        "effect" AVK_LOG_RST
         << std::endl;
  }

  // Respect Vulkan's memory type mask
  ci.requiredFlags = 0;  // optional
  ci.memoryTypeBits =
      allocInfo->memoryTypeIndex < 32 ? (1u << allocInfo->memoryTypeIndex) : 0;

  // Perform allocation with VMA
  VmaAllocation allocation = VK_NULL_HANDLE;
  VmaAllocationInfo outInfo{};

  VkResult const res = vmaAllocateMemory(device->vmaAllocator(), memReq, &ci,
                                         &allocation, &outInfo);

  if (res != VK_SUCCESS) {
    return 0;  // KTX interprets 0 as "allocation failed"
  }
  LOGI << "[KTX Alloc Mem] --------------"
          "-------------- Memory: "
       << (void*)allocation << std::endl;

  uint64_t const id = gKtxAllocManager->nextId++;
  gKtxAllocManager->allocations.try_emplace(id, allocation);

  return id;
}

static VkResult ktxBindBuffer(VkBuffer buffer,
                              uint64_t allocationId) AVK_NO_CFI {
  KTX_CALL_PREAMBLE;
  if (!device) return VK_ERROR_UNKNOWN;

  const auto it = gKtxAllocManager->allocations.find(allocationId);
  if (it == gKtxAllocManager->allocations.end()) {
    return VK_ERROR_MEMORY_MAP_FAILED;
  }

  const VmaAllocation alloc = it->second;
  return vmaBindBufferMemory(device->vmaAllocator(), alloc, buffer);
}

static VkResult ktxBindImage(VkImage image, uint64_t allocationId) AVK_NO_CFI {
  KTX_CALL_PREAMBLE;
  if (!device) return VK_ERROR_UNKNOWN;

  const auto it = gKtxAllocManager->allocations.find(allocationId);
  if (it == gKtxAllocManager->allocations.end()) {
    return VK_ERROR_UNKNOWN;  // "unknown allocation"
  }

  VmaAllocation const alloc = it->second;
  return vmaBindImageMemory(device->vmaAllocator(), alloc, image);
}

static VkResult ktxMemoryMap(uint64_t allocId,
                             [[maybe_unused]] uint64_t pageNumber,
                             VkDeviceSize* mapLength,
                             void** dataPtr) AVK_NO_CFI {
  if (!dataPtr) return VK_ERROR_UNKNOWN;
  KTX_CALL_PREAMBLE;
  if (!device) return VK_ERROR_UNKNOWN;

  auto const it = gKtxAllocManager->allocations.find(allocId);
  if (it == gKtxAllocManager->allocations.end())
    return VK_ERROR_MEMORY_MAP_FAILED;

  VmaAllocation const alloc = it->second;
  if (mapLength) {
    VmaAllocationInfo info{};
    vmaGetAllocationInfo(device->vmaAllocator(), alloc, &info);
    *mapLength = info.size;
  }

  return vmaMapMemory(device->vmaAllocator(), alloc, dataPtr);
}

static void ktxMemoryUnmap(uint64_t allocId,
                           [[maybe_unused]] uint64_t pageNumber) AVK_NO_CFI {
  KTX_CALL_PREAMBLE;

  auto const it = gKtxAllocManager->allocations.find(allocId);
  if (it == gKtxAllocManager->allocations.end()) return;

  VmaAllocation const alloc = it->second;
  vmaUnmapMemory(device->vmaAllocator(), alloc);
}

static void ktxFreeMem(uint64_t allocId) AVK_NO_CFI {
  KTX_CALL_PREAMBLE;

  auto const it = gKtxAllocManager->allocations.find(allocId);
  if (it == gKtxAllocManager->allocations.end()) {
    LOGE << AVK_LOG_RED "[KTX Free Mem] Something not found (id)" << allocId
         << std::endl;
    return;
  }
  VmaAllocation const alloc = it->second;
  gKtxAllocManager->allocations.erase(it);
  LOGI << "[KTX Free Mem] --------------"
          "-------------- free: "
       << (void*)alloc << std::endl;
  vmaFreeMemory(device->vmaAllocator(), alloc);
}

#undef KTX_CALL_PREAMBLE

static ktxVulkanTexture_subAllocatorCallbacks ktxAllocCallbacksFromVMA() {
  ktxVulkanTexture_subAllocatorCallbacks callbacks{};
  callbacks.allocMemFuncPtr = ktxAllocMem;
  callbacks.bindBufferFuncPtr = ktxBindBuffer;
  callbacks.bindImageFuncPtr = ktxBindImage;
  callbacks.memoryMapFuncPtr = ktxMemoryMap;
  callbacks.memoryUnmapFuncPtr = ktxMemoryUnmap;
  callbacks.freeMemFuncPtr = ktxFreeMem;
  return callbacks;
}

// ----------------------- PIMPL Implementation ----------------------------
class TextureLoaderKTX2::Impl : public NonMoveable {
 public:
  Impl(vk::Instance* instance, vk::Device* device);
  ~Impl() noexcept;

  bool loadTexture(uint64_t id, std::string_view filePath,
                   VkImageUsageFlags usage, VkImageLayout finalLayout,
                   TextureInfo& outInfo);

  void discardById(vk::DiscardPool* discardPool, uint64_t id,
                   TextureInfo& inOutInfo, uint64_t timeline);

 private:
  void initialize();
  void cleanup();
  [[nodiscard]] ktx_transcode_fmt_e selectTranscodeFormat() const;

  // dependencies which should outlive this object and are not owned by it
  struct Deps {
    vk::Instance* instance;
    vk::Device* device;
  } m_deps;
  ktxVulkanDeviceInfo m_ktxDevInfo{};
  ktxVulkanTexture_subAllocatorCallbacks m_allocCallbacks{};
  std::unordered_map<uint64_t, ktxVulkanTexture> m_loadedTextures;
  VkCommandPool m_commandPool = VK_NULL_HANDLE;
};

TextureLoaderKTX2::Impl::Impl(vk::Instance* instance, vk::Device* device)
    : m_deps{instance, device} {
  initialize();
}

TextureLoaderKTX2::Impl::~Impl() noexcept { cleanup(); }

// Execution of any methods form texture->vtbl are to be tagged as no cfi
bool TextureLoaderKTX2::Impl::loadTexture(uint64_t id,
                                          std::string_view filePath,
                                          VkImageUsageFlags usage,
                                          VkImageLayout finalLayout,
                                          TextureInfo& outInfo) AVK_NO_CFI {
  assert(gKtxAllocManager.has_value());
  ktxTexture2* texture = nullptr;
  KTX_error_code err = ktxTexture2_CreateFromNamedFile(
      filePath.data(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT, &texture);
  if (err != KTX_SUCCESS) {
    LOGE << AVK_LOG_RED "[TextureLoader] Couldn't Load KTX2 Texture: "
         << ktxErrorString(err) << AVK_LOG_RST << std::endl;
    return false;
  }

  // transcode on compressed format based on device capabilities
  ktx_transcode_fmt_e const transcodeFormat = selectTranscodeFormat();

  // transcode if necessary (should always be since we start from basisu)
  if (ktxTexture2_NeedsTranscoding(texture)) {
    err = ktxTexture2_TranscodeBasis(texture, transcodeFormat, 0);
    if (err != KTX_SUCCESS) {
      LOGE << AVK_LOG_RED "[TextureLoaded] Couldn't Transcode KTX2 Texture: "
           << ktxErrorString(err) << AVK_LOG_RST << std::endl;
      ktxTexture_Destroy(ktxTexture(texture));
      return false;
    }
  }

  // should be created without mipmaps, as we'll generate them (library crashes)
  assert(texture->numLevels == 1);
  texture->generateMipmaps = false;

  // upload with VMA suballocator
  ktxVulkanTexture vkTexture{};
  err = ktxTexture2_VkUploadEx_WithSuballocator(
      texture, &m_ktxDevInfo, &vkTexture, VK_IMAGE_TILING_OPTIMAL, usage,
      finalLayout, &m_allocCallbacks);
  if (err != KTX_SUCCESS) {
    LOGE << AVK_LOG_RED "[TextureLoaded] Couldn't Upload KTX2 Texture: "
         << ktxErrorString(err) << AVK_LOG_RST << std::endl;
    ktxTexture_Destroy(ktxTexture(texture));
    return false;
  }
  // store for cleanup tracking
  static_assert(std::is_trivial_v<decltype(vkTexture)> &&
                std::is_standard_layout_v<decltype(vkTexture)>);
  auto const& [it, wasInserted] = m_loadedTextures.try_emplace(id, vkTexture);
  if (!wasInserted) {
    LOGE << AVK_LOG_RED
        "[TextureLoaded] Couldn't Track KTX2 Texture: (HashTable Collision)"
         << AVK_LOG_RST << std::endl;
    ktxTexture_Destroy(ktxTexture(texture));
    return false;
  }

  // fill output
  outInfo.image = vkTexture.image;
  outInfo.imageView = VK_NULL_HANDLE;  // To be manually created
  outInfo.format = vkTexture.imageFormat;
  outInfo.width = texture->baseWidth;
  outInfo.height = texture->baseHeight;
  outInfo.mipLevels = texture->numLevels;
  outInfo.layerCount = texture->numLayers;

  ktxTexture_Destroy(ktxTexture(texture));
  return true;
}

void TextureLoaderKTX2::Impl::discardById(vk::DiscardPool* discardPool,
                                          uint64_t id, TextureInfo& inOutInfo,
                                          uint64_t timeline) {
  if (!gKtxAllocManager.has_value()) {
    showErrorScreenAndExit(
        "To discard a texture there should be a global manager");
  }
  auto const it = m_loadedTextures.find(id);
  assert(it != m_loadedTextures.end());
  if (it == m_loadedTextures.end()) return;

  VkImage image = it->second.image;
  assert(image == inOutInfo.image);
  VmaAllocation alloc = VK_NULL_HANDLE;
  {
    // get the allocation object from the global map
    std::lock_guard lk{gKtxAllocManager->mtx};
    LOGI << "Fjkdlfjdslfjdlsajfldk _> " << it->second.allocationId << std::endl;
    auto const allocIt =
        gKtxAllocManager->allocations.find(it->second.allocationId);
    assert(allocIt != gKtxAllocManager->allocations.end());
    alloc = allocIt->second;
    gKtxAllocManager->allocations.erase(allocIt);
    assert(alloc != VK_NULL_HANDLE);
  }
  m_loadedTextures.erase(it);
  LOGI << "[KTX Discard] --------------"
          "-------------- discarding: "
       << (void*)alloc << std::endl;

  // now discard
  discardPool->discardImageView(inOutInfo.imageView, timeline);
  discardPool->discardImage(image, VK_NULL_HANDLE, timeline);
  discardPool->discardImage(VK_NULL_HANDLE, alloc, timeline);

  // clean caller's object
  inOutInfo.imageView = VK_NULL_HANDLE;
  inOutInfo.image = VK_NULL_HANDLE;
}

void TextureLoaderKTX2::Impl::initialize() AVK_NO_CFI {
  if (gKtxAllocManager.has_value()) {
    showErrorScreenAndExit("There should only be one Texture Loader");
  }
  setGlobalKtxDevice(m_deps.device);
  m_loadedTextures.reserve(128);
  ktxVulkanFunctions const ktxTable =
      ktxTableFromVolkTable(m_deps.device->table());
  VkCommandPoolCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  createInfo.queueFamilyIndex =
      m_deps.device->universalGraphicsQueueFamilyIndex();

  VK_CHECK(m_deps.device->table()->vkCreateCommandPool(
      m_deps.device->device(), &createInfo, nullptr, &m_commandPool));
  KTX_CHECK(ktxVulkanDeviceInfo_ConstructEx(
      &m_ktxDevInfo, m_deps.instance->handle(), m_deps.device->physicalDevice(),
      m_deps.device->device(), m_deps.device->queue(), m_commandPool, nullptr,
      &ktxTable));
  m_allocCallbacks = ktxAllocCallbacksFromVMA();
}

void TextureLoaderKTX2::Impl::cleanup() AVK_NO_CFI {
  // cleanup all loaded textures. Doesn't discard them because we are at
  // destruction
  if (!m_loadedTextures.empty()) {
    LOGE << AVK_LOG_RED
        "[TextureLoaderKTX2::Impl::cleanup()] there are textures still "
        "alive" AVK_LOG_RST
         << std::endl;
  }
  for (auto& [id, vkTex] : m_loadedTextures) {
    KTX_CHECK(ktxVulkanTexture_Destruct_WithSuballocator(
        &vkTex, m_deps.device->device(), nullptr, &m_allocCallbacks));
  }
  m_loadedTextures.clear();
  // destroy KTX device info
  if (m_ktxDevInfo.device != VK_NULL_HANDLE) {
    ktxVulkanDeviceInfo_Destruct(&m_ktxDevInfo);
  }
  m_deps.device->table()->vkDestroyCommandPool(m_deps.device->device(),
                                               m_commandPool, nullptr);
  m_commandPool = VK_NULL_HANDLE;
  m_ktxDevInfo = {};
  gKtxAllocManager.reset();
}

ktx_transcode_fmt_e TextureLoaderKTX2::Impl::selectTranscodeFormat() const
    AVK_NO_CFI {
  // If querying for texture every texture allocation is slow, cache that
  VkPhysicalDeviceFeatures2 features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  vkGetPhysicalDeviceFeatures2(m_deps.device->physicalDevice(), &features);
  // TODO if necessary distinguish between key VkFormats
  if (features.features.textureCompressionBC) {
    return KTX_TTF_BC7_RGBA;
  } else if (features.features.textureCompressionETC2) {
    return KTX_TTF_ETC2_RGBA;
  } else if (features.features.textureCompressionASTC_LDR) {
    return KTX_TTF_ASTC_4x4_RGBA;
  } else {
    return KTX_TTF_RGBA32;
  }
}

// ------------------- Interface Implementation ----------------------------

TextureLoaderKTX2::TextureLoaderKTX2(vk::Instance* instance, vk::Device* device)
    : m_impl{std::make_unique<Impl>(instance, device)} {}

TextureLoaderKTX2::~TextureLoaderKTX2() noexcept = default;

bool TextureLoaderKTX2::loadTexture(uint64_t id, std::string_view filePath,
                                    VkImageUsageFlags usage,
                                    VkImageLayout finalLayout,
                                    TextureInfo& outInfo) const {
  return m_impl->loadTexture(id, filePath, usage, finalLayout, outInfo);
}

void TextureLoaderKTX2::discardById(vk::DiscardPool* discardPool, uint64_t id,
                                    TextureInfo& inOutInfo,
                                    uint64_t timeline) const {
  m_impl->discardById(discardPool, id, inOutInfo, timeline);
}

}  // namespace avk::experimental