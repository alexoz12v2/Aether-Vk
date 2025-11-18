#pragma once

#include "os/avk-core-macros.h"
#include "render/vk/discard-pool.h"
#include "render/vk/pipeline-info.h"

// library stuff
#include <glm/glm.hpp>

namespace avk::experimental {

struct Vertex {
  glm::vec3 position;
  glm::vec3 color;
};

enum class StencilEqualityMode { eReplacing = 0, eZeroExpected };

/// Utility Method to crete a `GraphicsInfo` struct (for pipeline creation)
/// Assuming it uses a simple Vertex Structure, a vertex shader and fragment
/// shader where shaders use 1 binding and 2 locations (0 = position, 1 = color)
/// It also assumes the renderPass has 1 subpass
///
/// Note: The renderPass, since it is recreated every time the swapchain is
/// recreated (the surfaceFormat might change in principle), it is not filled in
/// here
///
/// Note: The ownership of the given objects is user defined, as they are simply
/// copied to the struct
vk::GraphicsInfo basicGraphicsInfo(VkPipelineLayout pipelineLayout,
                                   VkShaderModule const* modules,
                                   VkFormat depthStencilFmt,
                                   StencilEqualityMode, bool disableDepthWrite);

/// helper handles discard of all vulkan handles inside the graphics info
/// bool controls whether we should discard variable members, such as the
/// renderPass where variable means that it is recreated after swapchain is
/// recreated
/// TODO: pipelineLayout might become dynamic if descriptors change
void discardGraphicsInfo(vk::DiscardPool* discardPool, uint64_t timeline,
                         vk::GraphicsInfo& inOutGraphicsInfo,
                         bool discardDynamic = false);

}  // namespace avk::experimental