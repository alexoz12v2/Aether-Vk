#include "render/experimental/avk-basic-graphics-info.h"

namespace avk::experimental {

vk::GraphicsInfo basicGraphicsInfo(VkPipelineLayout pipelineLayout,
                                   VkShaderModule modules[2],
                                   VkFormat depthStencilFmt) {
  vk::GraphicsInfo graphicsInfo{};
  graphicsInfo.preRasterization.vertexModule = modules[0];
  graphicsInfo.fragmentShader.fragmentModule = modules[1];
  graphicsInfo.preRasterization.geometryModule = VK_NULL_HANDLE;
#if 0
  // attribute descriptions: 1 binding, 2 attributes
  VkVertexInputBindingDescription binding{};
  binding.binding = 0;
  binding.stride = sizeof(Vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  graphicsInfo.vertexIn.bindings.push_back(binding);
  VkVertexInputAttributeDescription attribute{};
  attribute.binding = 0;
  attribute.location = 0;
  attribute.format = VK_FORMAT_R32G32B32_SFLOAT;  // position float3
  attribute.offset = offsetof(Vertex, position);
  graphicsInfo.vertexIn.attributes.push_back(attribute);
  attribute.binding = 0;
  attribute.location = 1;
  attribute.format = VK_FORMAT_R32G32B32_SFLOAT;  // position float3
  attribute.offset = offsetof(Vertex, color);
  graphicsInfo.vertexIn.attributes.push_back(attribute);
#else
  // attribute description: 1 binding, 1 attribute
  {
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(glm::vec3);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    graphicsInfo.vertexIn.bindings.push_back(binding);
  }
  {
    VkVertexInputAttributeDescription attribute{};
    attribute.binding = 0;
    attribute.offset = 0;
    attribute.location = 0;
    attribute.format = VK_FORMAT_R32G32B32_SFLOAT;
    graphicsInfo.vertexIn.attributes.push_back(attribute);
  }
#endif
  graphicsInfo.vertexIn.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  graphicsInfo.fragmentShader.viewports.resize(1);
  graphicsInfo.fragmentShader.scissors.resize(1);
  // fragment shader and output attachments
  graphicsInfo.fragmentOut.depthAttachmentFormat = depthStencilFmt;
  graphicsInfo.fragmentOut.stencilAttachmentFormat = depthStencilFmt;

  graphicsInfo.opts.rasterizationPolygonMode = VK_POLYGON_MODE_FILL;
  // TODO Cull
  // stencil options
  graphicsInfo.opts.flags |= vk::EPipelineFlags::eStencilEnable;
  graphicsInfo.opts.stencilCompareOp = vk::EStencilCompareOp::eAlways;
  graphicsInfo.opts.stencilLogicalOp = vk::EStencilLogicOp::eReplace;
  graphicsInfo.opts.stencilReference = 1;
  graphicsInfo.opts.stencilCompareMask = 0xFF;
  graphicsInfo.opts.stencilWriteMask = 0xFF;

  // Pipeline Layout
  graphicsInfo.pipelineLayout = pipelineLayout;
  // Renderpass and subpass
  graphicsInfo.subpass = 0;
  // Note: RenderPass filled by user
  return graphicsInfo;
}

void discardGraphicsInfo(vk::DiscardPool *discardPool, uint64_t timeline,
                         vk::GraphicsInfo &inOutGraphicsInfo,
                         bool discardDynamic) {
  if (inOutGraphicsInfo.preRasterization.vertexModule != VK_NULL_HANDLE) {
    discardPool->discardShaderModule(
        inOutGraphicsInfo.preRasterization.vertexModule, timeline);
    inOutGraphicsInfo.preRasterization.vertexModule = VK_NULL_HANDLE;
  }
  if (inOutGraphicsInfo.preRasterization.geometryModule != VK_NULL_HANDLE) {
    discardPool->discardShaderModule(
        inOutGraphicsInfo.preRasterization.geometryModule, timeline);
    inOutGraphicsInfo.preRasterization.geometryModule = VK_NULL_HANDLE;
  }
  if (inOutGraphicsInfo.fragmentShader.fragmentModule != VK_NULL_HANDLE) {
    discardPool->discardShaderModule(
        inOutGraphicsInfo.fragmentShader.fragmentModule, timeline);
    inOutGraphicsInfo.fragmentShader.fragmentModule = VK_NULL_HANDLE;
  }
  if (inOutGraphicsInfo.pipelineLayout != VK_NULL_HANDLE) {
    discardPool->discardPipelineLayout(inOutGraphicsInfo.pipelineLayout,
                                       timeline);
    inOutGraphicsInfo.pipelineLayout = VK_NULL_HANDLE;
  }
  if (discardDynamic) {
    if (inOutGraphicsInfo.renderPass != VK_NULL_HANDLE) {
      discardPool->discardRenderPass(inOutGraphicsInfo.renderPass, timeline);
      inOutGraphicsInfo.renderPass = VK_NULL_HANDLE;
    }
  }
}

}  // namespace avk::experimental