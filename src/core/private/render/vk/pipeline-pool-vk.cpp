#include "render/vk/pipeline-pool-vk.h"

// my stuff
#include "render/vk/device-vk.h"
#include "render/vk/discard-pool.h"

// library and stuff
#include <cassert>

// ---------------------------------------------------------------------------
// Static Utility for Constructor
// ---------------------------------------------------------------------------
static inline void ctorShaderStageCreateInfo(
    VkPipelineShaderStageCreateInfo& stages) {
  stages = {};
  stages.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  stages.pNext = nullptr;
  stages.flags = 0;
  // to be set and reset at construction
  stages.module = VK_NULL_HANDLE;
  // OpEntryPoint in SPIR-V
  stages.pName = "main";
  stages.pSpecializationInfo = nullptr;  // TODO
}

static inline void ctorVertexInputStateCreateInfo(
    VkPipelineVertexInputStateCreateInfo& vertexInputState) {
  // TODO: Add support for "Vertex Pulling" (ie. bindless shaders)
  // draw time with bindings -> vkCmdBindVertexBuffers
  // used only in "Programmable Primitive Shading"
  // alternative: Mesh Shading, which uses workgroups
  vertexInputState = {};
  vertexInputState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  vertexInputState.pNext = nullptr;
  vertexInputState.flags = 0;
  // binding description and attribute description specified during creation
  vertexInputState.vertexAttributeDescriptionCount = 0;
  vertexInputState.pVertexAttributeDescriptions = nullptr;
  vertexInputState.vertexBindingDescriptionCount = 0;
  vertexInputState.pVertexBindingDescriptions = nullptr;
};

static inline void ctorInputAssemblyStateCreateInfo(
    VkPipelineInputAssemblyStateCreateInfo& inputAssemblyState) {
  inputAssemblyState = {};
  inputAssemblyState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  inputAssemblyState.pNext = nullptr;
  inputAssemblyState.flags = 0;
  // topology set at creation
  inputAssemblyState.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  inputAssemblyState.primitiveRestartEnable = VK_FALSE;
};

static inline void ctorTessellationStateCreateInfo(
    VkPipelineTessellationStateCreateInfo& tessellationState) {
  // TODO add tessellation support if necessary
  tessellationState = {};
  tessellationState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
  tessellationState.pNext = nullptr;
  tessellationState.flags = 0;
  // less than VkPhysicalDeviceLimits::maxTessellationSize
  tessellationState.patchControlPoints = 1;
};

static inline void ctorViewportStateCreateInfo(
    VkPipelineViewportStateCreateInfo& viewportState) {
  viewportState = {};
  viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  viewportState.pNext = nullptr;  // TODO later
  viewportState.flags = 0;
  // number of scissors and viewports given at creation.
};

static inline void ctorRasterizationStateCreateInfo(
    VkPipelineRasterizationStateCreateInfo& rasterizationState) {
  // TODO: add parametrization for Depth Bias (useless for UI, good for 3D)
  rasterizationState = {};
  rasterizationState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  rasterizationState.pNext =
      nullptr;  // TODO depth bias or provoking vertex, ...
  rasterizationState.flags = 0;
  // TODO: Enable depth clamp and bias for 3D pipelines
  rasterizationState.depthClampEnable = VK_FALSE;
  // discard should be enabled for 3D pipelines dealing with Opaque objects
  // TODO: make this configurable
  rasterizationState.rasterizerDiscardEnable = VK_FALSE;
  // TODO: make this configurable for wireframes (or find another way?)
  // for lines, you require fillModeNonSolid
  rasterizationState.polygonMode = VK_POLYGON_MODE_FILL;
  // TODO: should configure if back is needed for transparency
  rasterizationState.cullMode = VK_CULL_MODE_BACK_BIT;
  rasterizationState.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
  // TODO configure depth bias for 3D (UI doesn't need that)
  rasterizationState.depthBiasEnable = VK_FALSE;
  rasterizationState.depthBiasConstantFactor = 1.f;
  rasterizationState.depthBiasClamp = 0.f;  // TODO handle better
  rasterizationState.depthBiasSlopeFactor = 1.f;
  // dynamic state -> vkCmdSetLineWidth
  rasterizationState.lineWidth = 1.f;
};

static inline void ctorMultisampleStateCreateInfo(
    VkPipelineMultisampleStateCreateInfo& multisampleState,
    std::vector<VkSampleMask>& sampleMasks) {
  // matters if using render pass (nothing said for dynamic rendering?)
  // or sample shading is used -> more invocations for a given fragment
  // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-sampleshading
  multisampleState = {};
  multisampleState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  multisampleState.pNext = nullptr;
  multisampleState.flags = 0;
  // TODO: scheme configurable per pipeline and reactivate
  multisampleState.rasterizationSamples =
      // VK_SAMPLE_COUNT_4_BIT;      // MSAA
      VK_SAMPLE_COUNT_1_BIT;
  sampleMasks.resize(1);        // max(4 / 32, 1)
  sampleMasks[0] = UINT32_MAX;  // count all (default)
  // TODO: sample shading dynamic
  // https://docs.vulkan.org/samples/latest/samples/extensions/fragment_shading_rate_dynamic/README.html
  // TODO add check for feature sampleRateShading
  multisampleState.sampleShadingEnable = VK_FALSE;
  // pick until you cover 70% of the sample (If enabled)
  multisampleState.minSampleShading = 0.7f;
  multisampleState.pSampleMask = sampleMasks.data();
  // TODO transparency if necessary
  multisampleState.alphaToCoverageEnable = VK_FALSE;
  multisampleState.alphaToOneEnable = VK_FALSE;
};

static inline void ctorDepthStencilStateCreateInfo(
    VkPipelineDepthStencilStateCreateInfo& depthStencilState) {
  depthStencilState = {};
  depthStencilState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  depthStencilState.pNext = nullptr;
  depthStencilState.flags = 0;
  depthStencilState.depthTestEnable = VK_TRUE;
  // TODO: this is disabled when rendering transparent objects
  depthStencilState.depthWriteEnable = VK_TRUE;
  depthStencilState.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
  depthStencilState.depthBoundsTestEnable = VK_FALSE;  // TODO
  depthStencilState.stencilTestEnable = VK_FALSE;      // TODO
  // TODO front and back for stencil test (zeroed out for now)
  depthStencilState.minDepthBounds = 0.f;
  depthStencilState.maxDepthBounds = 1.f;
  // stencil state for fragments of front-facing polygons
  depthStencilState.front = {};
  // stencil state for fragments of back-facing polygons
  depthStencilState.back = {};
};

static inline void ctorColorBlendStateCreateInfo(
    VkPipelineColorBlendStateCreateInfo& colorBlendState,
    VkPipelineColorBlendAttachmentState& blendAttachmentTemplate) {
  // configuration to use the over operator
  colorBlendState = {};
  colorBlendState.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  colorBlendState.pNext = nullptr;
  colorBlendState.flags = 0;
  colorBlendState.logicOpEnable = VK_FALSE;
  colorBlendState.logicOp = VK_LOGIC_OP_CLEAR;  // ignored
  colorBlendState.blendConstants[0] = 1.f;
  colorBlendState.blendConstants[1] = 1.f;
  colorBlendState.blendConstants[2] = 1.f;
  colorBlendState.blendConstants[3] = 1.f;

  // attachmentCount and pAttachments set starting from template
  blendAttachmentTemplate = {};
  blendAttachmentTemplate.colorWriteMask =
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_A_BIT;
  // TODO if needed, make the blend mode configurable
  blendAttachmentTemplate.blendEnable = VK_TRUE;
  blendAttachmentTemplate.colorBlendOp = VK_BLEND_OP_ADD;
  blendAttachmentTemplate.alphaBlendOp = VK_BLEND_OP_ADD;
  blendAttachmentTemplate.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
  blendAttachmentTemplate.dstColorBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  blendAttachmentTemplate.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
  blendAttachmentTemplate.dstAlphaBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
};

// ---------------------------------------------------------------------------
// Static Utility for creation
// ---------------------------------------------------------------------------
static void setDepthStencilStateCreateInfo(
    avk::vk::GraphicsInfo const& graphicsInfo,
    VkPipelineDepthStencilStateCreateInfo& depthStencilState) {
  // TODO handle transparency and more depth modes if necessary
  // TODO if necessary, add disable depth on VK_FORMAT_UNDEFINED
  using namespace avk::vk;
  if (graphicsInfo.opts.flags & EPipelineFlags::eNoDepthWrite) {
    depthStencilState.depthWriteEnable = VK_FALSE;
  }

  if (graphicsInfo.opts.flags & EPipelineFlags::eStencilEnable &&
      graphicsInfo.fragmentOut.stencilAttachmentFormat != VK_FORMAT_UNDEFINED) {
    depthStencilState.stencilTestEnable = VK_TRUE;
    switch (graphicsInfo.opts.stencilCompareOp) {
      case EStencilCompareOp::eEqual:
        depthStencilState.front.compareOp = VK_COMPARE_OP_EQUAL;
        break;
      case EStencilCompareOp::eNotEqual:
        depthStencilState.front.compareOp = VK_COMPARE_OP_NOT_EQUAL;
        break;
      case EStencilCompareOp::eAlways:
      case EStencilCompareOp::eNone:
        depthStencilState.front.compareOp = VK_COMPARE_OP_ALWAYS;
        break;
    }

    // reference, compare mask and write mask for stencil buffer
    depthStencilState.front.reference = graphicsInfo.opts.stencilReference;
    depthStencilState.front.compareMask = graphicsInfo.opts.stencilCompareMask;
    depthStencilState.front.writeMask = graphicsInfo.opts.stencilWriteMask;

    switch (graphicsInfo.opts.stencilLogicalOp) {
      case EStencilLogicOp::eReplace:
        depthStencilState.front.failOp = VK_STENCIL_OP_KEEP;
        depthStencilState.front.passOp = VK_STENCIL_OP_REPLACE;
        depthStencilState.front.depthFailOp = VK_STENCIL_OP_KEEP;
        depthStencilState.back = depthStencilState.front;
        break;
      case EStencilLogicOp::eCountDepthPass:
        depthStencilState.front.failOp = VK_STENCIL_OP_KEEP;
        depthStencilState.front.passOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
        depthStencilState.front.depthFailOp = VK_STENCIL_OP_KEEP;
        depthStencilState.back = depthStencilState.front;
        depthStencilState.back.passOp = VK_STENCIL_OP_INCREMENT_AND_WRAP;
        break;
      case EStencilLogicOp::eCountDepthFail:
        depthStencilState.front.passOp = VK_STENCIL_OP_KEEP;
        depthStencilState.front.failOp = VK_STENCIL_OP_KEEP;
        depthStencilState.front.depthFailOp = VK_STENCIL_OP_INCREMENT_AND_WRAP;
        depthStencilState.back = depthStencilState.front;
        depthStencilState.back.depthFailOp = VK_STENCIL_OP_DECREMENT_AND_WRAP;
        break;
      case EStencilLogicOp::eNone:
        depthStencilState.front.passOp = VK_STENCIL_OP_KEEP;
        depthStencilState.front.failOp = VK_STENCIL_OP_KEEP;
        depthStencilState.front.depthFailOp = VK_STENCIL_OP_KEEP;
        depthStencilState.back = depthStencilState.front;
        break;
    }
  }
}

// ---------------------------------------------------------------------------
// Static Utility for cleanup
// ---------------------------------------------------------------------------

namespace avk::vk {

void PipelinePool::clearGraphicsPipelineStates() {
  // -- Graphics Pipeline: Shader Stages --
  // TODO: specialization contants
  m_graphicsPipelineCreateInfo.stageCount = 0;
  m_pipelineShaderStageCreateInfos[0].module = VK_NULL_HANDLE;
  m_pipelineShaderStageCreateInfos[1].module = VK_NULL_HANDLE;
  m_pipelineShaderStageCreateInfos[2].module = VK_NULL_HANDLE;

  // -- Graphics Pipeline: Vertex Input --
  m_pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = 0;
  m_pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = nullptr;
  m_pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = 0;
  m_pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = nullptr;

  // -- Graphics Pipeline: Input Assembly --
  m_pipelineInputAssemblyStateCreateInfo.topology =
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

  // -- Graphics Pipeline: Tessellation State --
  // nothing

  // -- Graphics Pipeline: Viewport State --
  // TODO if add support of with count dynamic state, then reset everything
  m_pipelineViewportStateCreateInfo.viewportCount = 0;
  m_pipelineViewportStateCreateInfo.scissorCount = 0;

  // -- Graphics Pipeline: Rasterization State --
  m_pipelineRasterizationStateCreateInfo.frontFace =
      VK_FRONT_FACE_COUNTER_CLOCKWISE;
  m_pipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
  m_pipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
  m_pipelineRasterizationStateCreateInfo.depthBiasEnable = VK_FALSE;
  m_pipelineRasterizationStateCreateInfo.depthBiasConstantFactor = 1.f;
  m_pipelineRasterizationStateCreateInfo.depthBiasClamp = 1.f;
  m_pipelineRasterizationStateCreateInfo.depthBiasSlopeFactor = 1.f;
  m_pipelineRasterizationStateCreateInfo.lineWidth = 1.f;

  // -- Graphics Pipeline: Multisample State --
  // nothing

  // -- Graphics Pipeline: Depth Stencil State --
  m_pipelineDepthStencilStateCreateInfo.stencilTestEnable = VK_FALSE;
  m_pipelineDepthStencilStateCreateInfo.depthWriteEnable = VK_TRUE;
  m_pipelineDepthStencilStateCreateInfo.front = {};
  m_pipelineDepthStencilStateCreateInfo.back = {};

  // -- Graphics Pipeline: Color Blend --
  m_pipelineColorBlendAttachmentStates.clear();

  // -- Graphics Pipeline: Dynamic States --
  m_pipelineDynamicStateCreateInfo.dynamicStateCount = 0;
  m_pipelineDynamicStateCreateInfo.pDynamicStates = nullptr;

  // -- Common Values --
  m_graphicsPipelineCreateInfo.flags = 0;
  m_graphicsPipelineCreateInfo.layout = VK_NULL_HANDLE;
  m_graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  m_graphicsPipelineCreateInfo.renderPass = VK_NULL_HANDLE;
  m_graphicsPipelineCreateInfo.subpass = -1;
}

// ---------------------------------------------------------------------------
// API implementation
// ---------------------------------------------------------------------------

PipelinePool::PipelinePool(Device* device) : m_deps{device} {
  // initialize pipeline compute create info
  m_computePipelineCreateInfo = {};
  m_computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  m_computePipelineCreateInfo.layout = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.basePipelineIndex = -1;
  m_computePipelineCreateInfo.stage.sType =
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  m_computePipelineCreateInfo.stage.pName = "main";
  m_computePipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
  // stage.pSpecializationInfo, stage.module populate in create function

  // initialize graphics pipeline create info
  m_graphicsPipelineCreateInfo = {};
  m_graphicsPipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  m_graphicsPipelineCreateInfo.stageCount = ShaderStageCount;
  m_graphicsPipelineCreateInfo.pStages = m_pipelineShaderStageCreateInfos;
  m_graphicsPipelineCreateInfo.pVertexInputState =
      &m_pipelineVertexInputStateCreateInfo;
  m_graphicsPipelineCreateInfo.pInputAssemblyState =
      &m_pipelineInputAssemblyStateCreateInfo;
  m_graphicsPipelineCreateInfo.pTessellationState =
      &m_pipelineTessellationStateCreateInfo;
  m_graphicsPipelineCreateInfo.pViewportState =
      &m_pipelineViewportStateCreateInfo;
  m_graphicsPipelineCreateInfo.pMultisampleState =
      &m_pipelineMultisampleStateCreateInfo;
  m_graphicsPipelineCreateInfo.pDepthStencilState =
      &m_pipelineDepthStencilStateCreateInfo;
  m_graphicsPipelineCreateInfo.pColorBlendState =
      &m_pipelineColorBlendStateCreateInfo;
  m_graphicsPipelineCreateInfo.pRasterizationState =
      &m_pipelineRasterizationStateCreateInfo;
  m_graphicsPipelineCreateInfo.pDynamicState =
      &m_pipelineDynamicStateCreateInfo;
  m_graphicsPipelineCreateInfo.layout =
      VK_NULL_HANDLE;  // given at construction
  m_graphicsPipelineCreateInfo.renderPass = VK_NULL_HANDLE;  // dynamicrendering
  m_graphicsPipelineCreateInfo.subpass = 0;                  // dynamicrendering
  // reuse of another pipeline requires flag VK_PIPELINE_CREATE_DERIVATIVE_BIT
  m_graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  m_graphicsPipelineCreateInfo.basePipelineIndex = -1;

  // -- Graphics Pipeline: Shader Stages --
  for (uint32_t i = 0; i < ShaderStageCount; ++i) {
    ctorShaderStageCreateInfo(m_pipelineShaderStageCreateInfos[i]);
  }

  // 0. VK_SHADER_STAGE_VERTEX_BIT
  m_pipelineShaderStageCreateInfos[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  // 1. VK_SHADER_STAGE_FRAGMENT_BIT
  m_pipelineShaderStageCreateInfos[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  // 2. VK_SHADER_STAGE_GEOMETRY_BIT -> VkDevice with geometryShaderFeature
  m_pipelineShaderStageCreateInfos[2].stage = VK_SHADER_STAGE_GEOMETRY_BIT;

  // -- Graphics Pipeline: Vertex Input --
  // TODO: Add support for "Vertex Pulling" (ie. bindless shaders)
  // draw time with bindings -> vkCmdBindVertexBuffers
  // used only in "Programmable Primitive Shading"
  // alternative: Mesh Shading, which uses workgroups
  ctorVertexInputStateCreateInfo(m_pipelineVertexInputStateCreateInfo);

  // -- Graphics Pipeline: Input Assembly --
  ctorInputAssemblyStateCreateInfo(m_pipelineInputAssemblyStateCreateInfo);

  // -- Graphics Pipeline: Tessellation State --
  // can be null if pStages doesn't have a tessellation stage
  // (so programmable primitive shading only)
  ctorTessellationStateCreateInfo(m_pipelineTessellationStateCreateInfo);

  // -- Graphics Pipeline: Viewport State --
  // dynamic state for viewport and scissor means that each viewport/scissor is
  // dynamic, *but their number is fixed*. This can be null if dynamic state
  // uses viewport with count and scissor with count (from
  // VK_EXT_extended_dynamic_state3)
  ctorViewportStateCreateInfo(m_pipelineViewportStateCreateInfo);

  // -- Graphics Pipeline: Rasterization State --
  ctorRasterizationStateCreateInfo(m_pipelineRasterizationStateCreateInfo);

  // -- Graphics Pipeline: Multisample State --
  ctorMultisampleStateCreateInfo(m_pipelineMultisampleStateCreateInfo,
                                 m_sampleMasks);

  // -- Graphics Pipeline: Depth Stencil State --
  ctorDepthStencilStateCreateInfo(m_pipelineDepthStencilStateCreateInfo);

  // -- Graphics Pipeline: Color Blend --
  ctorColorBlendStateCreateInfo(m_pipelineColorBlendStateCreateInfo,
                                m_pipelineColorBlendAttachmentStateTemplate);

  // -- Graphics Pipeline: Dynamic States --
  // line width is the last as we exclude it in non line topologies
  m_dynamicStates.reserve(64);
  m_dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
  m_dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
  m_dynamicStates.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);
}

PipelinePool::~PipelinePool() { destroyAllPipelines(); }

VkPipeline PipelinePool::getOrCreateComputePipeline(
    ComputeInfo& computeInfo,
    // TODO remove maybe unused
    [[maybe_unused]] bool isStaticShader, VkPipeline pipelineBase) AVK_NO_CFI {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();

  std::lock_guard<std::mutex> lk{m_mutex};
  if (auto it = m_computePipelines.find(computeInfo);
      it != m_computePipelines.end()) {
    VkPipeline result = it->second;
    assert(result != VK_NULL_HANDLE);
    return result;
  }

  // populate parametrized fields of the create info
  // TODO: specialization contants
  m_computePipelineCreateInfo.layout = computeInfo.pipelineLayout;
  m_computePipelineCreateInfo.stage.module = computeInfo.shaderModule;
  // !: if base, assume you desire a pipeline derivative (ie, if delete parent
  // pipeline child is invalid)
  if (pipelineBase != VK_NULL_HANDLE) {
    m_computePipelineCreateInfo.flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    m_computePipelineCreateInfo.basePipelineHandle = pipelineBase;
  }
  // TODO read more about VK_EXT_descriptor_buffer
  // if (context.device().extensions.isEnabled(
  //         VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
  //   m_computePipelineCreateInfo.flags |=
  //       VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
  // }

  // TODO insert pipeline cache static or not
  VkPipeline pipeline = VK_NULL_HANDLE;
  // TODO host memory allocators
  VK_CHECK(vkDevApi->vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1,
                                              &m_computePipelineCreateInfo,
                                              nullptr, &pipeline));
  assert(pipeline != VK_NULL_HANDLE);
  m_computePipelines.try_emplace(computeInfo, pipeline);

  // cleanup
  m_computePipelineCreateInfo.layout = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.flags = 0;
  m_computePipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.stage.module = {};
  // TODO: specialization contants cleanup

  return pipeline;
}

VkPipeline PipelinePool::getOrCreateGraphicsPipeline(
    GraphicsInfo& graphicsInfo, [[maybe_unused]] bool isStaticShader,
    VkPipeline pipelineBase) AVK_NO_CFI {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();

  std::lock_guard<std::mutex> lk{m_mutex};
  if (auto it = m_graphicsPipelines.find(graphicsInfo);
      it != m_graphicsPipelines.end()) {
    VkPipeline pipeline = it->second;
    assert(pipeline != VK_NULL_HANDLE);
    return pipeline;
  }

  // parametrize create info
  // TODO: specialization contants
  // -- Graphics Pipeline: Shader Stages --
  // make geometry module optional
  // TODO: Add support for more stages in programmed primitive shading, and add
  // support for programmed mesh shading
  m_graphicsPipelineCreateInfo.stageCount =
      graphicsInfo.preRasterization.geometryModule == VK_NULL_HANDLE ? 2 : 3;
  m_pipelineShaderStageCreateInfos[0].module =
      graphicsInfo.preRasterization.vertexModule;
  m_pipelineShaderStageCreateInfos[1].module =
      graphicsInfo.fragmentShader.fragmentModule;
  m_pipelineShaderStageCreateInfos[2].module =
      graphicsInfo.preRasterization.geometryModule;

  // -- Graphics Pipeline: Vertex Input --
  m_pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount =
      static_cast<uint32_t>(graphicsInfo.vertexIn.attributes.size());
  m_pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions =
      graphicsInfo.vertexIn.attributes.data();
  m_pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount =
      static_cast<uint32_t>(graphicsInfo.vertexIn.bindings.size());
  m_pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions =
      graphicsInfo.vertexIn.bindings.data();

  // -- Graphics Pipeline: Input Assembly --
  m_pipelineInputAssemblyStateCreateInfo.topology =
      graphicsInfo.vertexIn.topology;
  // TODO if necessary, primitive restart enable

  // -- Graphics Pipeline: Tessellation State --
  // nothing

  // -- Graphics Pipeline: Viewport State --
  m_pipelineViewportStateCreateInfo.viewportCount =
      static_cast<uint32_t>(graphicsInfo.fragmentShader.viewports.size());
  m_pipelineViewportStateCreateInfo.scissorCount =
      static_cast<uint32_t>(graphicsInfo.fragmentShader.scissors.size());

  // -- Graphics Pipeline: Rasterization State --
  if (graphicsInfo.opts.flags & EPipelineFlags::eDepthBias) {
    // VUID-VkGraphicsPipelineCreateInfo-pDynamicStates-00754
    // if depthBias and no depthBiasClamp feature enabled (TODO) then 0
    m_pipelineRasterizationStateCreateInfo.depthBiasEnable = VK_TRUE;
    m_pipelineRasterizationStateCreateInfo.depthBiasConstantFactor = 1.f;
    m_pipelineRasterizationStateCreateInfo.depthBiasClamp = 0;  // -> 0x1p-13f;
    m_pipelineRasterizationStateCreateInfo.depthBiasSlopeFactor = 2.f;
    m_pipelineRasterizationStateCreateInfo.lineWidth = 1.f;
  }
  // TODO: check polygon mode support
  m_pipelineRasterizationStateCreateInfo.polygonMode =
      graphicsInfo.opts.rasterizationPolygonMode;
  if (graphicsInfo.opts.flags & ~EPipelineFlags::eCull) {
    m_pipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_NONE;
  } else if (graphicsInfo.opts.flags & EPipelineFlags::eCullFront) {
    m_pipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
  }  // else default: cull back face

  if (graphicsInfo.opts.flags & EPipelineFlags::eInvertFrontFace) {
    m_pipelineRasterizationStateCreateInfo.frontFace = VK_FRONT_FACE_CLOCKWISE;
  }
  // TODO maybe provoking vertex

  // -- Graphics Pipeline: Multisample State --
  // nothing

  // -- Graphics Pipeline: Depth Stencil State --
  setDepthStencilStateCreateInfo(graphicsInfo,
                                 m_pipelineDepthStencilStateCreateInfo);

  // -- Graphics Pipeline: Color Blend --
  uint32_t const colorAttachmentNum = static_cast<uint32_t>(
      graphicsInfo.fragmentOut.colorAttachmentFormats.size() > 0
          ? graphicsInfo.fragmentOut.colorAttachmentFormats.size()
          : 1);
  m_pipelineColorBlendAttachmentStates.reserve(colorAttachmentNum);
  for (uint32_t i = 0; i < colorAttachmentNum; ++i) {
    m_pipelineColorBlendAttachmentStates.push_back(
        m_pipelineColorBlendAttachmentStateTemplate);
  }
  // VUID-VkGraphicsPipelineCreateInfo-renderPass-06055:
  // pCreateInfos[0].pNext<VkPipelineRenderingCreateInfo>.colorAttachmentCount
  // == pCreateInfos[0].pColorBlendState->attachmentCount
  m_pipelineColorBlendStateCreateInfo.attachmentCount = colorAttachmentNum;
  m_pipelineColorBlendStateCreateInfo.pAttachments =
      m_pipelineColorBlendAttachmentStates.data();

  // TODO if necessary, more operators in color blend

  // -- Graphics Pipeline: Dynamic States --
  m_pipelineDynamicStateCreateInfo = {};
  m_pipelineDynamicStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;

  // line width is the last as we exclude it in non line topologies
  if (graphicsInfo.vertexIn.topology == VK_PRIMITIVE_TOPOLOGY_LINE_LIST ||
      graphicsInfo.vertexIn.topology == VK_PRIMITIVE_TOPOLOGY_LINE_STRIP ||
      graphicsInfo.vertexIn.topology ==
          VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY ||
      graphicsInfo.vertexIn.topology ==
          VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY) {
    m_pipelineDynamicStateCreateInfo.dynamicStateCount =
        static_cast<uint32_t>(m_dynamicStates.size());
  } else {
    m_pipelineDynamicStateCreateInfo.dynamicStateCount =
        static_cast<uint32_t>(m_dynamicStates.size() - 1);
  }

  m_pipelineDynamicStateCreateInfo.pDynamicStates = m_dynamicStates.data();

  // -- Common Values --
  assert(graphicsInfo.pipelineLayout != VK_NULL_HANDLE);
  m_graphicsPipelineCreateInfo.layout = graphicsInfo.pipelineLayout;
  // assume, if base is given, that you want a pipeline derivative
  if (pipelineBase != VK_NULL_HANDLE) {
    m_graphicsPipelineCreateInfo.flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    m_graphicsPipelineCreateInfo.basePipelineHandle = pipelineBase;
  }
  m_graphicsPipelineCreateInfo.renderPass = graphicsInfo.renderPass;
  m_graphicsPipelineCreateInfo.subpass = graphicsInfo.subpass;
  // TODO read more about VK_EXT_descriptor_buffer
  // if (context.device().extensions.isEnabled(
  //         VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
  //   m_computePipelineCreateInfo.flags |=
  //       VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
  // }

  VkPipeline pipeline = VK_NULL_HANDLE;
  // TODO pipeline cache
  VkResult const res = vkDevApi->vkCreateGraphicsPipelines(
      dev, VK_NULL_HANDLE, 1, &m_graphicsPipelineCreateInfo, nullptr,
      &pipeline);
  VK_CHECK(res);
  m_graphicsPipelines.try_emplace(graphicsInfo, pipeline);

  // cleanup create info
  clearGraphicsPipelineStates();

  return pipeline;
}

void PipelinePool::discardAllPipelines(DiscardPool* discardPool,
                                       VkPipelineLayout pipelineLayout,
                                       uint64_t value) {
  std::lock_guard<std::mutex> lk{m_mutex};
  for (auto it = m_computePipelines.begin(); it != m_computePipelines.end();
       /*inside*/) {
    if (it->first.pipelineLayout == pipelineLayout) {
      discardPool->discardPipeline(it->second, value);
      it = m_computePipelines.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = m_graphicsPipelines.begin(); it != m_graphicsPipelines.end();
       /*inside */) {
    if (it->first.pipelineLayout == pipelineLayout) {
      discardPool->discardPipeline(it->second, value);
      it = m_graphicsPipelines.erase(it);
    } else {
      ++it;
    }
  }
}

void PipelinePool::readStaticCacheFromDisk() AVK_NO_CFI {
  // TODO static pipeline cache from disk
}

void PipelinePool::writeStaticCacheToDisk() AVK_NO_CFI {
  // TODO write static pipeline cache to disk
}

void PipelinePool::destroyAllPipelines() AVK_NO_CFI {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();

  std::lock_guard<std::mutex> lk{m_mutex};
  for (auto& [info, pipeline] : m_computePipelines) {
    vkDevApi->vkDestroyPipeline(dev, pipeline, nullptr);
  }
  for (auto& [info, pipeline] : m_graphicsPipelines) {
    vkDevApi->vkDestroyPipeline(dev, pipeline, nullptr);
  }
  // info struct are externally cleaned, so forget them
  m_computePipelines.clear();
  m_graphicsPipelines.clear();
  // TODO destroy pipeline cache
}

}  // namespace avk::vk