#pragma once

#include "render/vk/common-vk.h"
#include "render/vk/pipeline-info.h"
#include "utils/mixins.h"

// libraries and stuff
#include <unordered_map>
#include <vector>

namespace avk::vk {

class Device;
class DiscardPool;

class PipelinePool : public NonMoveable {
 public:
  PipelinePool(Device* device);
  // since we don't own the handles in the info structs, we won't
  // destroy them
  ~PipelinePool();

  VkPipeline getOrCreateComputePipeline(ComputeInfo& computeInfo,
                                        bool isStaticShader,
                                        VkPipeline pipelineBase);
  VkPipeline getOrCreateGraphicsPipeline(GraphicsInfo& graphicsInfo,
                                         bool isStaticShader,
                                         VkPipeline pipelineBase);

  // erase pipelines from the maps and inserts them in the discard pool, such
  // that they can live there until they are no longer in use
  // context must be the same used to create the pipelines
  // WARNING: calling this won't discard Vulkan Handles inside
  // the info structs. only VkPipelines
  void discardAllPipelines(DiscardPool* discardPool,
                           VkPipelineLayout pipelineLayout, uint64_t value);

  void readStaticCacheFromDisk();
  void writeStaticCacheToDisk();

  // WARNING: calling this won't destroy Vulkan Handles inside
  // the info structs. only VkPipelines
  void destroyAllPipelines();

 private:
  // dependencies which must outlive the object
  struct Deps {
    Device* device;
  } m_deps;

  std::unordered_map<GraphicsInfo, VkPipeline> m_graphicsPipelines;
  std::unordered_map<ComputeInfo, VkPipeline> m_computePipelines;
  // partially initialized structure to reuse
  VkComputePipelineCreateInfo m_computePipelineCreateInfo;

  static uint32_t constexpr ShaderStageCount = 3;

  VkGraphicsPipelineCreateInfo m_graphicsPipelineCreateInfo;
  VkPipelineShaderStageCreateInfo
      m_pipelineShaderStageCreateInfos[ShaderStageCount];
  VkPipelineInputAssemblyStateCreateInfo m_pipelineInputAssemblyStateCreateInfo;
  VkPipelineVertexInputStateCreateInfo m_pipelineVertexInputStateCreateInfo;

  VkPipelineRasterizationStateCreateInfo m_pipelineRasterizationStateCreateInfo;

  // no provoking vertex info
  std::vector<VkDynamicState> m_dynamicStates;
  VkPipelineDynamicStateCreateInfo m_pipelineDynamicStateCreateInfo;

  VkPipelineViewportStateCreateInfo m_pipelineViewportStateCreateInfo;
  VkPipelineDepthStencilStateCreateInfo m_pipelineDepthStencilStateCreateInfo;

  std::vector<VkSampleMask> m_sampleMasks;  // TODO in GraphicsInfo?
  VkPipelineMultisampleStateCreateInfo m_pipelineMultisampleStateCreateInfo;
  VkPipelineTessellationStateCreateInfo m_pipelineTessellationStateCreateInfo;

  std::vector<VkPipelineColorBlendAttachmentState>
      m_pipelineColorBlendAttachmentStates;
  VkPipelineColorBlendStateCreateInfo m_pipelineColorBlendStateCreateInfo;
  VkPipelineColorBlendAttachmentState
      m_pipelineColorBlendAttachmentStateTemplate;

  // VkSpecializationInfo m_specializationInfo;
  // std::vector<VkSpecializationMapEntry> m_specializationMapEntries;
  // VkPushConstantRange m_pushConstantRange;
  // VkPipelineCache m_pipelineCacheStatic;
  // VkPipelineCache m_pipelineCacheNonStatic;

  // mutex to be acquired whenever getting/creating a pipeline
  std::mutex m_mutex;

  // utility to reset all states after creating a graphics pipeline
  void clearGraphicsPipelineStates();
};

}  // namespace avk::vk