#pragma once

#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "render/utils-vk.h"

namespace avk {

class ShaderModuleVk {
 public:
  std::string combinedSlangSrc;
  std::vector<uint32_t> spirvBinary;
  VkShaderModule shaderModule = VK_NULL_HANDLE;

  void destroy(DiscardPoolVk& discardPool);
};

// struct to identify compute pipeline
struct ComputeInfo {
  VkShaderModule shaderModule;
  VkPipelineLayout pipelineLayout;
  // TODO specialization constants
};

inline bool operator==(ComputeInfo const& a, ComputeInfo const& b) {
  return a.shaderModule == b.shaderModule &&
         a.pipelineLayout == b.pipelineLayout;
}

}  // namespace avk

template <>
struct std::hash<avk::ComputeInfo> {
  size_t operator()(avk::ComputeInfo const& computeInfo) const noexcept {
    size_t hash = reinterpret_cast<size_t>(computeInfo.shaderModule);
    hash = hash * 33 ^ reinterpret_cast<size_t>(computeInfo.pipelineLayout);
    return hash;
  }
};

namespace avk {

// Combines two hash values into one (from Boost's hash_combine)
inline void hashCombine(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

// Generic hash functor for std::vector<T>
template <typename T>
struct VectorHash {
  std::size_t operator()(const std::vector<T>& v) const {
    std::size_t seed = v.size();
    std::hash<T> hasher;
    for (const auto& elem : v) {
      hashCombine(seed, hasher(elem));
    }
    return seed;
  }

  template <typename F>
  std::size_t operator()(const std::vector<T>& v, F&& f) const {
    std::size_t seed = v.size();
    for (const auto& elem : v) {
      hashCombine(seed, f(elem));
    }
    return seed;
  }
};

// struct to identify graphics pipeline dividing it into
// 'VK_EXT_graphics_pipeline_library' vertex_in, pre_rasterization,
// fragment_shader, fragment_out
struct GraphicsInfo {
  struct VertexIn {
    VkPrimitiveTopology topology;
    std::vector<VkVertexInputAttributeDescription> attributes;
    std::vector<VkVertexInputBindingDescription> bindings;

    bool operator==(VertexIn const& other) const {
      auto constexpr attributeEqual =
          [](VkVertexInputAttributeDescription const& a,
             VkVertexInputAttributeDescription const& b) {
            return a.binding == b.binding && a.format == b.format &&
                   a.location == b.location && a.offset == b.offset;
          };
      auto constexpr bindingEqual =
          [](VkVertexInputBindingDescription const& a,
             VkVertexInputBindingDescription const& b) {
            return a.binding == b.binding && a.inputRate == b.inputRate &&
                   a.stride == b.stride;
          };
      bool attributesEqual = attributes.size() == other.attributes.size();
      bool bindingsEqual = bindings.size() == other.bindings.size();
      for (size_t i = 0; i < attributes.size() && attributesEqual; ++i) {
        if (!attributeEqual(attributes[i], other.attributes[i])) {
          attributesEqual = false;
        }
      }
      for (size_t i = 0; i < bindings.size() && bindingsEqual; ++i) {
        if (!bindingEqual(bindings[i], other.bindings[i])) {
          bindingsEqual = false;
        }
      }
      return topology == other.topology && attributesEqual && bindingsEqual;
    }

    uint64_t hash() const {
      uint64_t hash = static_cast<uint64_t>(topology);
      hash = hash * 33 ^
             VectorHash<VkVertexInputAttributeDescription>{}(
                 attributes, [](VkVertexInputAttributeDescription const& x) {
                   return ((x.binding * 33 ^ x.location) * 33 ^ x.offset) * 33 ^
                          x.format;
                 });
      hash = hash * 33 ^
             VectorHash<VkVertexInputBindingDescription>{}(
                 bindings, [](VkVertexInputBindingDescription const& x) {
                   return (x.binding * 33 ^ x.stride) * 33 ^ x.inputRate;
                 });
      return hash;
    }
  };
  struct PreRasterization {
    VkShaderModule vertexModule;
    VkShaderModule geometryModule;

    bool operator==(PreRasterization const& other) const {
      return vertexModule == other.vertexModule &&
             geometryModule == other.geometryModule;
    }

    uint64_t hash() const {
      uint64_t hash = 33 ^ reinterpret_cast<uint64_t>(vertexModule);
      hash = hash * 33 ^ reinterpret_cast<uint64_t>(geometryModule);
      return hash;
    }
  };
  struct FragmentShader {
    VkShaderModule fragmentModule;
    std::vector<VkViewport> viewports;
    std::vector<VkRect2D> scissors;

    bool operator==(FragmentShader const& f) const {
      auto constexpr equalityViewports = [](VkViewport const& a,
                                            VkViewport const& b) {
        return a.height == b.height && a.width == b.width && a.x == b.x &&
               a.y == b.y;
      };
      auto constexpr equalityScissor = [](VkRect2D a, VkRect2D b) {
        return a.extent.height == b.extent.height &&
               a.extent.width == b.extent.width && a.offset.x == b.offset.x &&
               a.offset.y == b.offset.y;
      };
      bool viewportsEqual = viewports.size() == f.viewports.size();
      bool scissorsEqual = scissors.size() == f.scissors.size();

      for (size_t i = 0; i < viewports.size() && viewportsEqual; ++i) {
        if (!equalityViewports(viewports[i], f.viewports[i])) {
          viewportsEqual = false;
        }
      }
      for (size_t i = 0; i < scissors.size() && scissorsEqual; ++i) {
        if (!equalityScissor(scissors[i], f.scissors[i])) {
          scissorsEqual = false;
        }
      }

      return fragmentModule == f.fragmentModule && viewportsEqual &&
             scissorsEqual;
    }

    uint64_t hash() const {
      uint64_t hash = reinterpret_cast<uint64_t>(fragmentModule);
      hash = hash * 33 ^ viewports.size();
      hash = hash * 33 ^ scissors.size();
      return hash;
    }
  };
  struct FragmentOut {
    uint32_t colorAttachmentSize;
    // stuff for dynamic rendering
    VkFormat depthAttachmentFormat;
    VkFormat stencilAttachmentFormat;
    std::vector<VkFormat> colorAttachmentFormats;

    bool operator==(FragmentOut const& other) const {
      if (depthAttachmentFormat != other.depthAttachmentFormat ||
          stencilAttachmentFormat != other.stencilAttachmentFormat ||
          colorAttachmentFormats.size() !=
              other.colorAttachmentFormats.size()) {
        return false;
      }

      if (memcmp(colorAttachmentFormats.data(),
                 other.colorAttachmentFormats.data(),
                 colorAttachmentFormats.size() * sizeof(VkFormat)) == 0) {
        return false;
      }
      return true;
    }

    uint64_t hash() const {
      uint64_t hash = depthAttachmentFormat;
      hash = hash * 33 ^ stencilAttachmentFormat;
      hash = hash * 33 ^
             VectorHash<VkFormat>{}(colorAttachmentFormats,
                                    [](VkFormat format) { return format; });
      return hash;
    }
  };

  VertexIn vertexIn;
  PreRasterization preRasterization;
  FragmentShader fragmentShader;
  FragmentOut fragmentOut;

  // add GPU state?
  VkPipelineLayout pipelineLayout;
  // TODO add specialization constants
};

}  // namespace avk

template <>
struct std::hash<avk::GraphicsInfo> {
  size_t operator()(avk::GraphicsInfo const& graphicsInfo) const {
    uint64_t hash = 33 ^ graphicsInfo.vertexIn.hash();
    hash = hash * 33 ^ graphicsInfo.preRasterization.hash();
    hash = hash * 33 ^ graphicsInfo.fragmentShader.hash();
    hash = hash * 33 ^ graphicsInfo.fragmentOut.hash();
    hash = hash * 33 ^ reinterpret_cast<uint64_t>(graphicsInfo.pipelineLayout);
    return hash;
  }
};

namespace avk {

class VkPipelinePool {
 public:
  VkPipelinePool();
  VkPipelinePool(VkPipelinePool const&) = delete;
  VkPipelinePool(VkPipelinePool&&) noexcept = delete;
  VkPipelinePool& operator=(VkPipelinePool const&) = delete;
  VkPipelinePool& operator=(VkPipelinePool&&) noexcept = delete;

  VkPipeline getOrCreateComputePipeline(ComputeInfo& computeInfo,
                                        bool isStaticShader,
                                        VkPipeline pipelineBase);
  VkPipeline getOrCreateGraphicsPipeline(GraphicsInfo& graphicsInfo,
                                         bool isStaticShader,
                                         VkPipeline pipelineBase);

  void discardAllPipelines(DiscardPoolVk& discardPool,
                           VkPipelineLayout pipelineLayout);

  void readStaticCacheFromDisk();
  void writeStaticCacheToDisk();

  void destroyAllPipelines();

 private:
  std::unordered_map<GraphicsInfo, VkPipeline> m_graphicsPipelines;
  std::unordered_map<ComputeInfo, VkPipeline> m_computePipelines;
  // partially initialized structure to reuse
  VkComputePipelineCreateInfo m_computePipelineCreateInfo;

  VkGraphicsPipelineCreateInfo m_graphicsPipelineCreateInfo;
  VkPipelineRenderingCreateInfo m_pipelineRenderingCreateInfo;
  VkPipelineShaderStageCreateInfo m_pipelineShaderStageCreateInfo[3];
  VkPipelineInputAssemblyStateCreateInfo m_pipelineInputAssemblyStateCreateInfo;
  VkPipelineVertexInputStateCreateInfo m_pipelineVertexInputStateCreateInfo;

  VkPipelineRasterizationStateCreateInfo m_pipelineRasterizationStateCreateInfo;

  // no provoking vertex info
  std::vector<VkDynamicState> m_dynamicState;
  VkPipelineDynamicStateCreateInfo m_pipelineDynamicStateCreateInfo;

  VkPipelineViewportStateCreateInfo m_pipelineViewportStateCreateInfo;
  VkPipelineDepthStencilStateCreateInfo m_pipelineDepthStencilStateCreateInfo;

  VkPipelineMultisampleStateCreateInfo pipelineMultisampleStateCreateInfo;

  std::vector<VkPipelineColorBlendAttachmentState>
      m_pipelineColorBlendAttachmentStates;
  VkPipelineColorBlendStateCreateInfo m_pipelineColorBlendStateCreateInfo;
  VkPipelineColorBlendAttachmentState
      m_pipelineColorBlendAttachmentStateTemplate;

  VkSpecializationInfo m_specializationInfo;
  std::vector<VkSpecializationMapEntry> m_specializationMapEntries;
  VkPushConstantRange m_pushConstantRange;

  VkPipelineCache m_pipelineCacheStatic;
  VkPipelineCache m_pipelineCacheNonStatic;

  std::mutex m_mutex;
};

}  // namespace avk