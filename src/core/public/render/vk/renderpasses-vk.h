#pragma once

#include <render/vk/common-vk.h>
#include <render/vk/device-vk.h>

namespace avk::vk {

/// Creates a `VkRenderPass` with Color and Depth Attachment
/// (without multisampling)
Expected<VkRenderPass> basicRenderPass(Device const* device, VkFormat colorFmt,
                                       VkFormat depthFmt);

}  // namespace avk::vk