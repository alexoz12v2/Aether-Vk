#pragma once

#include "render/vk/common-vk.h"
#include "utils/mixins.h"

// std
#include <memory>
#include <string_view>

namespace avk::vk {
class Device;
class DiscardPool;
class Instance;
}  // namespace avk::vk

namespace avk::experimental {

// can be discarded only through the texture manager as you lack the
// VmaAllocation object
struct TextureInfo {
  VkImage image = VK_NULL_HANDLE;
  VkImageView imageView = VK_NULL_HANDLE;
  VkFormat format = VK_FORMAT_UNDEFINED;
  uint32_t width = 0;
  uint32_t height = 0;
  // TODO 3D textures? Are they needed
  uint32_t mipLevels = 0;
  uint32_t layerCount = 0;
};

/// \warning There can be only one instance at a time as the KTX library 4.3.2
/// handles suballocation without any user data (hence global state is needed)
class TextureLoaderKTX2 : public NonMoveable {
 public:
  TextureLoaderKTX2(vk::Instance* instance, vk::Device* device);
  ~TextureLoaderKTX2() noexcept;

  // TODO Integrate with Android asset
  // note: Doesn't create image view
  bool loadTexture(uint64_t id, std::string_view filePath,
                   VkImageUsageFlags usage, VkImageLayout finalLayout,
                   TextureInfo& outInfo) const;

  /// \warning `ktxVulkanTexture_Destruct_WithSuballocator` is *not*
  /// used, as in 4.3.2 all it does is calling `vkDestroyImage` and the
  /// free suballocator callback. We want to manage our vulkan handles
  /// with the discard pool
  void discardById(vk::DiscardPool* discardPool, uint64_t id,
                   TextureInfo& inOutInfo, uint64_t timeline) const;

 private:
  class Impl;
  std::unique_ptr<Impl> m_impl = nullptr;
};

}  // namespace avk::experimental