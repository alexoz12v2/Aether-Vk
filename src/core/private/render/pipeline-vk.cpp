#include "render/pipeline-vk.h"

#include <slang/slang.h>
#include <vulkan/vulkan_core.h>

#include <cassert>

namespace avk {

// ---------------------------------------------------------------------------

void ShaderModuleVk::destroy(DiscardPoolVk& discardPool) {
  if (shaderModule != VK_NULL_HANDLE) {
    discardPool.discardShaderModule(shaderModule);
    shaderModule = VK_NULL_HANDLE;
  }
}

// ---------------------------------------------------------------------------

VkPipelinePool::VkPipelinePool() {
  // initialize pipeline compute create info
  m_computePipelineCreateInfo = {};
  m_computePipelineCreateInfo.sType =
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  m_computePipelineCreateInfo.pNext = nullptr;
  m_computePipelineCreateInfo.flags = 0;
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
  m_graphicsPipelineCreateInfo.pNext = &m_pipelineRenderingCreateInfo;
  m_graphicsPipelineCreateInfo.flags = 0;
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
    m_pipelineShaderStageCreateInfos[i] = {};
    m_pipelineShaderStageCreateInfos[i].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    m_pipelineShaderStageCreateInfos[i].pNext = nullptr;
    m_pipelineShaderStageCreateInfos[i].flags = 0;
    // to be set and reset at construction
    m_pipelineShaderStageCreateInfos[i].module = VK_NULL_HANDLE;
    // OpEntryPoint in SPIR-V
    m_pipelineShaderStageCreateInfos[i].pName = "main";
    m_pipelineShaderStageCreateInfos[i].pSpecializationInfo = nullptr;  // TODO
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
  m_pipelineVertexInputStateCreateInfo = {};
  m_pipelineVertexInputStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
  m_pipelineVertexInputStateCreateInfo.pNext = nullptr;
  m_pipelineVertexInputStateCreateInfo.flags = 0;
  // binding description and attribute description specified during creation
  m_pipelineVertexInputStateCreateInfo.vertexAttributeDescriptionCount = 0;
  m_pipelineVertexInputStateCreateInfo.pVertexAttributeDescriptions = nullptr;
  m_pipelineVertexInputStateCreateInfo.vertexBindingDescriptionCount = 0;
  m_pipelineVertexInputStateCreateInfo.pVertexBindingDescriptions = nullptr;

  // -- Graphics Pipeline: Input Assembly --
  // TODO: add Support for "Programmable Mesh Shading"
  m_pipelineInputAssemblyStateCreateInfo = {};
  m_pipelineInputAssemblyStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  m_pipelineInputAssemblyStateCreateInfo.pNext = nullptr;
  m_pipelineInputAssemblyStateCreateInfo.flags = 0;
  // topology set at creation
  m_pipelineInputAssemblyStateCreateInfo.topology =
      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  m_pipelineInputAssemblyStateCreateInfo.primitiveRestartEnable = VK_FALSE;

  // -- Graphics Pipeline: Tessellation State --
  // can be null if pStages doesn't have a tessellation stage (so programmable
  // primitive shading only)
  // TODO add tessellation support if necessary
  m_pipelineTessellationStateCreateInfo = {};
  m_pipelineTessellationStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO;
  m_pipelineTessellationStateCreateInfo.pNext = nullptr;
  m_pipelineTessellationStateCreateInfo.flags = 0;
  // less than VkPhysicalDeviceLimits::maxTessellationSize
  m_pipelineTessellationStateCreateInfo.patchControlPoints = 1;

  // -- Graphics Pipeline: Viewport State --
  // dynamic state for viewport and scissor means that each viewport/scissor is
  // dynamic, *but their number is fixed*. This can be null if dynamic state
  // uses viewport with count and scissor with count (from
  // VK_EXT_extended_dynamic_state3)
  m_pipelineViewportStateCreateInfo = {};
  m_pipelineViewportStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  m_pipelineViewportStateCreateInfo.pNext = nullptr;  // TODO later
  m_pipelineViewportStateCreateInfo.flags = 0;
  // number of scissors and viewports given at creation.

  // -- Graphics Pipeline: Rasterization State --
  // TODO: add parametrization for Depth Bias (useless for UI, good for 3D)
  m_pipelineRasterizationStateCreateInfo = {};
  m_pipelineRasterizationStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  m_pipelineRasterizationStateCreateInfo.pNext =
      nullptr;  // TODO depth bias or provoking vertex, ...
  m_pipelineRasterizationStateCreateInfo.flags = 0;
  // TODO: Enable depth clamp and bias for 3D pipelines
  m_pipelineRasterizationStateCreateInfo.depthClampEnable = VK_FALSE;
  // discard should be enabled for 3D pipelines dealing with Opaque objects
  // TODO: make this configurable
  m_pipelineRasterizationStateCreateInfo.rasterizerDiscardEnable = VK_FALSE;
  // TODO: make this configurable for wireframes (or find another way?)
  // for lines, you require fillModeNonSolid
  m_pipelineRasterizationStateCreateInfo.polygonMode = VK_POLYGON_MODE_FILL;
  // TODO: should configure if back is needed for transparency
  m_pipelineRasterizationStateCreateInfo.cullMode = VK_CULL_MODE_BACK_BIT;
  m_pipelineRasterizationStateCreateInfo.frontFace =
      VK_FRONT_FACE_COUNTER_CLOCKWISE;
  // TODO configure depth bias for 3D (UI doesn't need that)
  m_pipelineRasterizationStateCreateInfo.depthBiasEnable = VK_FALSE;
  m_pipelineRasterizationStateCreateInfo.depthBiasConstantFactor = 1.f;
  m_pipelineRasterizationStateCreateInfo.depthBiasClamp = 1.f;
  m_pipelineRasterizationStateCreateInfo.depthBiasSlopeFactor = 1.f;
  // dynamic state -> vkCmdSetLineWidth
  m_pipelineRasterizationStateCreateInfo.lineWidth = 1.f;

  // -- Graphics Pipeline: Multisample State --
  // matters if using render pass (nothing said for dynamic rendering?)
  // or sample shading is used -> more invocations for a given fragment
  // https://docs.vulkan.org/spec/latest/chapters/primsrast.html#primsrast-sampleshading
  m_pipelineMultisampleStateCreateInfo = {};
  m_pipelineMultisampleStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  m_pipelineMultisampleStateCreateInfo.pNext = nullptr;
  m_pipelineMultisampleStateCreateInfo.flags = 0;
  // TODO: scheme configurable per pipeline and reactivate
  m_pipelineMultisampleStateCreateInfo.rasterizationSamples =
      // VK_SAMPLE_COUNT_4_BIT;      // MSAA
      VK_SAMPLE_COUNT_1_BIT;
  m_sampleMasks.resize(1);        // max(4 / 32, 1)
  m_sampleMasks[0] = UINT32_MAX;  // count all (default)
  // TODO: sample shading dynamic
  // https://docs.vulkan.org/samples/latest/samples/extensions/fragment_shading_rate_dynamic/README.html
  // TODO add check for feature sampleRateShading
  m_pipelineMultisampleStateCreateInfo.sampleShadingEnable = VK_TRUE;
  m_pipelineMultisampleStateCreateInfo.minSampleShading =
      0.7f;  // pick until you cover 70% of the sample
  m_pipelineMultisampleStateCreateInfo.pSampleMask = m_sampleMasks.data();
  // TODO transparency if necessary
  m_pipelineMultisampleStateCreateInfo.alphaToCoverageEnable = VK_FALSE;
  m_pipelineMultisampleStateCreateInfo.alphaToOneEnable = VK_FALSE;

  // -- Graphics Pipeline: Depth Stencil State --
  m_pipelineDepthStencilStateCreateInfo = {};
  m_pipelineDepthStencilStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
  m_pipelineDepthStencilStateCreateInfo.pNext = nullptr;
  m_pipelineDepthStencilStateCreateInfo.flags = 0;
  m_pipelineDepthStencilStateCreateInfo.depthTestEnable = VK_TRUE;
  // TODO: this is disabled when rendering transparent objects
  m_pipelineDepthStencilStateCreateInfo.depthWriteEnable = VK_TRUE;
  m_pipelineDepthStencilStateCreateInfo.depthCompareOp =
      VK_COMPARE_OP_LESS_OR_EQUAL;
  m_pipelineDepthStencilStateCreateInfo.depthBoundsTestEnable =
      VK_FALSE;                                                        // TODO
  m_pipelineDepthStencilStateCreateInfo.stencilTestEnable = VK_FALSE;  // TODO
  // TODO front and back for stencil test (zeroed out for now)
  m_pipelineDepthStencilStateCreateInfo.minDepthBounds = 0.f;
  m_pipelineDepthStencilStateCreateInfo.maxDepthBounds = 1.f;
  // stencil state for fragments of front-facing polygons
  m_pipelineDepthStencilStateCreateInfo.front = {};
  // stencil state for fragments of back-facing polygons
  m_pipelineDepthStencilStateCreateInfo.back = {};

  // -- Graphics Pipeline: Color Blend --
  // configuration to use the over operator
  m_pipelineColorBlendStateCreateInfo = {};
  m_pipelineColorBlendStateCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  m_pipelineColorBlendStateCreateInfo.pNext = nullptr;
  m_pipelineColorBlendStateCreateInfo.flags = 0;
  m_pipelineColorBlendStateCreateInfo.logicOpEnable = VK_FALSE;
  m_pipelineColorBlendStateCreateInfo.logicOp = VK_LOGIC_OP_CLEAR;  // ignored
  m_pipelineColorBlendStateCreateInfo.blendConstants[0] = 1.f;
  m_pipelineColorBlendStateCreateInfo.blendConstants[1] = 1.f;
  m_pipelineColorBlendStateCreateInfo.blendConstants[2] = 1.f;
  m_pipelineColorBlendStateCreateInfo.blendConstants[3] = 1.f;

  // attachmentCount and pAttachments set starting from template
  m_pipelineColorBlendAttachmentStateTemplate = {};
  m_pipelineColorBlendAttachmentStateTemplate.colorWriteMask =
      VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_G_BIT |
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_A_BIT;
  // TODO if needed, make the blend mode configurable
  m_pipelineColorBlendAttachmentStateTemplate.blendEnable = VK_TRUE;
  m_pipelineColorBlendAttachmentStateTemplate.colorBlendOp = VK_BLEND_OP_ADD;
  m_pipelineColorBlendAttachmentStateTemplate.alphaBlendOp = VK_BLEND_OP_ADD;
  m_pipelineColorBlendAttachmentStateTemplate.srcColorBlendFactor =
      VK_BLEND_FACTOR_SRC_ALPHA;
  m_pipelineColorBlendAttachmentStateTemplate.dstColorBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
  m_pipelineColorBlendAttachmentStateTemplate.srcAlphaBlendFactor =
      VK_BLEND_FACTOR_ONE;
  m_pipelineColorBlendAttachmentStateTemplate.dstAlphaBlendFactor =
      VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;

  // -- Graphics Pipeline: Dynamic States --
  // line width is the last as we exclude it in non line topologies
  m_dynamicStates.reserve(64);
  m_dynamicStates.push_back(VK_DYNAMIC_STATE_VIEWPORT);
  m_dynamicStates.push_back(VK_DYNAMIC_STATE_SCISSOR);
  m_dynamicStates.push_back(VK_DYNAMIC_STATE_LINE_WIDTH);

  // -- Graphics Pipeline: Extension "VK_KHR_dynamic_rendering" or 1.3 --
  m_pipelineRenderingCreateInfo = {};
  m_pipelineRenderingCreateInfo.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
  m_pipelineRenderingCreateInfo.pNext = nullptr;
  m_pipelineRenderingCreateInfo.viewMask = 0;  // == VkRenderingInfo::viewMask
  // attachment data on creation
}

VkPipeline VkPipelinePool::getOrCreateComputePipeline(
    ContextVk const& context, ComputeInfo& computeInfo,
    // TODO remove maybe unused
    [[maybe_unused]] bool isStaticShader, VkPipeline pipelineBase) {
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
  if (context.device().extensions.isEnabled(
          VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
    m_computePipelineCreateInfo.flags |=
        VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
  }

  // TODO insert pipeline cache static or not
  VkPipeline pipeline = VK_NULL_HANDLE;
  // TODO host memory allocators
  if (vkCheck(vkCreateComputePipelines(context.device().device, VK_NULL_HANDLE,
                                       1, &m_computePipelineCreateInfo, nullptr,
                                       &pipeline))) {
    assert(pipeline != VK_NULL_HANDLE);
    m_computePipelines.try_emplace(computeInfo, pipeline);
  }

  // cleanup
  m_computePipelineCreateInfo.layout = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.flags = 0;
  m_computePipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.stage.module = {};
  // TODO: specialization contants cleanup

  return pipeline;
}

VkPipeline VkPipelinePool::getOrCreateGraphicsPipeline(
    ContextVk const& context, GraphicsInfo& graphicsInfo,
    [[maybe_unused]] bool isStaticShader, VkPipeline pipelineBase) {
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
    // TODO assert device supports depthBias feature
    m_pipelineRasterizationStateCreateInfo.depthBiasEnable = VK_TRUE;
    m_pipelineRasterizationStateCreateInfo.depthBiasConstantFactor = 1.f;
    m_pipelineRasterizationStateCreateInfo.depthBiasClamp = 0x1p-13f;
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
  // TODO handle transparency and more depth modes if necessary
  // TODO if necessary, add disable depth on VK_FORMAT_UNDEFINED
  if (graphicsInfo.opts.flags & EPipelineFlags::eNoDepthWrite) {
    m_pipelineDepthStencilStateCreateInfo.depthWriteEnable = VK_FALSE;
  }

  if (graphicsInfo.opts.flags & EPipelineFlags::eStencilEnable &&
      graphicsInfo.fragmentOut.stencilAttachmentFormat != VK_FORMAT_UNDEFINED) {
    m_pipelineDepthStencilStateCreateInfo.stencilTestEnable = VK_TRUE;
    switch (graphicsInfo.opts.stencilCompareOp) {
      case EStencilCompareOp::eEqual:
        m_pipelineDepthStencilStateCreateInfo.front.compareOp =
            VK_COMPARE_OP_EQUAL;
        break;
      case EStencilCompareOp::eNotEqual:
        m_pipelineDepthStencilStateCreateInfo.front.compareOp =
            VK_COMPARE_OP_NOT_EQUAL;
        break;
      case EStencilCompareOp::eAlways:
      case EStencilCompareOp::eNone:
        m_pipelineDepthStencilStateCreateInfo.front.compareOp =
            VK_COMPARE_OP_ALWAYS;
        break;
    }

    // reference, compare mask and write mask for stencil buffer
    m_pipelineDepthStencilStateCreateInfo.front.reference =
        graphicsInfo.opts.stencilReference;
    m_pipelineDepthStencilStateCreateInfo.front.compareMask =
        graphicsInfo.opts.stencilCompareMask;
    m_pipelineDepthStencilStateCreateInfo.front.writeMask =
        graphicsInfo.opts.stencilWriteMask;

    switch (graphicsInfo.opts.stencilLogicalOp) {
      case EStencilLogicOp::eReplace:
        m_pipelineDepthStencilStateCreateInfo.front.failOp = VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.front.passOp =
            VK_STENCIL_OP_REPLACE;
        m_pipelineDepthStencilStateCreateInfo.front.depthFailOp =
            VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.back =
            m_pipelineDepthStencilStateCreateInfo.front;
        break;
      case EStencilLogicOp::eCountDepthPass:
        m_pipelineDepthStencilStateCreateInfo.front.failOp = VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.front.passOp =
            VK_STENCIL_OP_DECREMENT_AND_WRAP;  // dangerous on wrap!
        m_pipelineDepthStencilStateCreateInfo.front.depthFailOp =
            VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.back =
            m_pipelineDepthStencilStateCreateInfo.front;
        m_pipelineDepthStencilStateCreateInfo.back.passOp =
            VK_STENCIL_OP_INCREMENT_AND_WRAP;
        break;
      case EStencilLogicOp::eCountDepthFail:
        m_pipelineDepthStencilStateCreateInfo.front.passOp = VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.front.failOp = VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.front.depthFailOp =
            VK_STENCIL_OP_INCREMENT_AND_WRAP;
        m_pipelineDepthStencilStateCreateInfo.back =
            m_pipelineDepthStencilStateCreateInfo.front;
        m_pipelineDepthStencilStateCreateInfo.back.depthFailOp =
            VK_STENCIL_OP_DECREMENT_AND_WRAP;
        break;
      case EStencilLogicOp::eNone:
        m_pipelineDepthStencilStateCreateInfo.front.passOp = VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.front.failOp = VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.front.depthFailOp =
            VK_STENCIL_OP_KEEP;
        m_pipelineDepthStencilStateCreateInfo.back =
            m_pipelineDepthStencilStateCreateInfo.front;
        break;
    }
  }

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

  // -- Graphics Pipeline: Extension "VK_KHR_dynamic_rendering" --
  VkFormat colorFallbackFormat = context.surfaceFormat().format;
  assert(colorFallbackFormat != VK_FORMAT_UNDEFINED);
  if (uint32_t num = static_cast<uint32_t>(
          graphicsInfo.fragmentOut.colorAttachmentFormats.size());
      num > 0) {
    m_pipelineRenderingCreateInfo.colorAttachmentCount = num;
    m_pipelineRenderingCreateInfo.pColorAttachmentFormats =
        graphicsInfo.fragmentOut.colorAttachmentFormats.data();
  } else {
    m_pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    m_pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &colorFallbackFormat;
  }
  m_pipelineRenderingCreateInfo.depthAttachmentFormat =
      graphicsInfo.fragmentOut.depthAttachmentFormat;
  m_pipelineRenderingCreateInfo.stencilAttachmentFormat =
      graphicsInfo.fragmentOut.stencilAttachmentFormat;

  // -- Common Values --
  assert(graphicsInfo.pipelineLayout != VK_NULL_HANDLE);
  m_graphicsPipelineCreateInfo.layout = graphicsInfo.pipelineLayout;
  // assume, if base is given, that you want a pipeline derivative
  if (pipelineBase != VK_NULL_HANDLE) {
    m_graphicsPipelineCreateInfo.flags |= VK_PIPELINE_CREATE_DERIVATIVE_BIT;
    m_graphicsPipelineCreateInfo.basePipelineHandle = pipelineBase;
  }
  // TODO read more about VK_EXT_descriptor_buffer
  if (context.device().extensions.isEnabled(
          VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME)) {
    m_computePipelineCreateInfo.flags |=
        VK_PIPELINE_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;
  }

  VkPipeline pipeline = VK_NULL_HANDLE;
  // TODO pipeline cache
  if (vkCheck(vkCreateGraphicsPipelines(context.device().device, VK_NULL_HANDLE,
                                        1, &m_graphicsPipelineCreateInfo,
                                        nullptr, &pipeline))) {
    m_graphicsPipelines.try_emplace(graphicsInfo, pipeline);
  }

  // cleanup create info
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

  // -- Graphics Pipeline: Extension "VK_KHR_dynamic_rendering" or 1.3 --
  m_pipelineRenderingCreateInfo.colorAttachmentCount = 0;
  m_pipelineRenderingCreateInfo.pColorAttachmentFormats = nullptr;
  m_pipelineRenderingCreateInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
  m_pipelineRenderingCreateInfo.stencilAttachmentFormat = VK_FORMAT_UNDEFINED;

  // -- Common Values --
  m_graphicsPipelineCreateInfo.flags = 0;
  m_graphicsPipelineCreateInfo.layout = VK_NULL_HANDLE;
  m_graphicsPipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;

  return pipeline;
}

void VkPipelinePool::discardAllPipelines(DiscardPoolVk& discardPool,
                                         VkPipelineLayout pipelineLayout) {
  std::lock_guard<std::mutex> lk{m_mutex};
  for (auto it = m_computePipelines.begin(); it != m_computePipelines.end();
       /*inside*/) {
    if (it->first.pipelineLayout == pipelineLayout) {
      discardPool.discardPipeline(it->second);
      it = m_computePipelines.erase(it);
    } else {
      ++it;
    }
  }

  for (auto it = m_graphicsPipelines.begin(); it != m_graphicsPipelines.end();
       /*inside */) {
    if (it->first.pipelineLayout == pipelineLayout) {
      discardPool.discardPipeline(it->second);
      it = m_graphicsPipelines.erase(it);
    } else {
      ++it;
    }
  }
}

void VkPipelinePool::readStaticCacheFromDisk() {
  // TODO static pipeline cache from disk
}

void VkPipelinePool::writeStaticCacheToDisk() {
  // TODO write static pipeline cache to disk
}

void VkPipelinePool::destroyAllPipelines(ContextVk const& context) {
  std::lock_guard<std::mutex> lk{m_mutex};
  for (auto& [info, pipeline] : m_computePipelines) {
    // TODO host allocation callbacks in context
    vkDestroyPipeline(context.device().device, pipeline, nullptr);
  }
  for (auto& [info, pipeline] : m_graphicsPipelines) {
    vkDestroyPipeline(context.device().device, pipeline, nullptr);
  }
  // info struct are externally cleaned, so forget them
  m_computePipelines.clear();
  m_graphicsPipelines.clear();
  // TODO destroy pipeline cache
}

}  // namespace avk