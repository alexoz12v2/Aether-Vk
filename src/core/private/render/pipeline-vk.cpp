#include "render/pipeline-vk.h"

#include <slang/slang.h>

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
  m_computePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
  m_computePipelineCreateInfo.pNext = nullptr;
  m_computePipelineCreateInfo.flags = 0;
  m_computePipelineCreateInfo.layout = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.basePipelineHandle = VK_NULL_HANDLE;
  m_computePipelineCreateInfo.basePipelineIndex = 0;

  // initialize graphics pipeline create info
  m_graphicsPipelineCreateInfo = {};
  m_graphicsPipelineCreateInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  m_graphicsPipelineCreateInfo.pNext = &m_pipelineRenderingCreateInfo;
}

}