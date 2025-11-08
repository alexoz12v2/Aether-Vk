#pragma once

#include <string>
#include <type_traits>
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

// -- Compute Pipeline Parameters --

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
    size_t hash = reinterpret_cast<uint64_t>(computeInfo.shaderModule);
    hash = hash * 33 ^ reinterpret_cast<uint64_t>(computeInfo.pipelineLayout);
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

enum class EPipelineFlags : uint32_t {
  eDepthBias = 1u << 0,
  eCull = 1u << 1,
  eCullFront = 1u << 2 | eCull,
  eCullBack = 0u << 2 | eCull,
  eInvertFrontFace = 1u << 3,
  eNoDepthWrite = 1u << 4,
  eStencilEnable = 1u << 5,
  eAll = eDepthBias | eCull | eInvertFrontFace | eCullFront | eNoDepthWrite |
         eStencilEnable,
};

inline bool operator&(EPipelineFlags a, EPipelineFlags b) {
  return static_cast<std::underlying_type_t<EPipelineFlags>>(a) &
         static_cast<std::underlying_type_t<EPipelineFlags>>(b);
}

inline bool operator|(EPipelineFlags a, EPipelineFlags b) {
  return static_cast<std::underlying_type_t<EPipelineFlags>>(a) |
         static_cast<std::underlying_type_t<EPipelineFlags>>(b);
}

inline EPipelineFlags& operator&=(EPipelineFlags& a, EPipelineFlags b) {
  *reinterpret_cast<std::underlying_type_t<EPipelineFlags>*>(&a) &=
      static_cast<std::underlying_type_t<EPipelineFlags>>(b);
  return a;
}

inline EPipelineFlags& operator|=(EPipelineFlags& a, EPipelineFlags b) {
  *reinterpret_cast<std::underlying_type_t<EPipelineFlags>*>(&a) |=
      static_cast<std::underlying_type_t<EPipelineFlags>>(b);
  return a;
}

inline EPipelineFlags operator~(EPipelineFlags a) {
  return static_cast<EPipelineFlags>(
      ~static_cast<std::underlying_type_t<EPipelineFlags>>(a));
}

enum class EStencilCompareOp : uint32_t {
  eNone = 0,
  eEqual,
  eNotEqual,
  eAlways,
};

enum class EStencilLogicOp : uint32_t {
  // synonym for KEEP in both back and front always
  eNone = 0,
  // replace on pass (visibility mask and identification of visible objects)
  eReplace,
  // depth pass for z-pass shadow volume counting (from blender's codebase)
  // decrement on front, increment on back
  // -> used to count volume crossing on depth pass
  eCountDepthPass,
  // depth pass for Carmack's reverse algorithm
  // count crossings on depth test fail. inc/dec -> front/back
  // https://developer.nvidia.com/gpugems/gpugems3/part-ii-light-and-shadows/chapter-11-efficient-and-robust-shadow-volumes-using
  eCountDepthFail
};

// -- Graphics Pipeline Parameters --

// struct to identify graphics pipeline dividing it into
// 'VK_EXT_graphics_pipeline_library' vertex_in, pre_rasterization,
// fragment_shader, fragment_out
struct GraphicsInfo {
  struct VertexIn {
    // adjacency require geometryShader feature,
    // patch requires tessellationShader
    VkPrimitiveTopology topology;
    // defined what's inside each binding, which can be split in multiple
    // locations (globally unique, monotonically increasing) if you need multple
    // pieces of data having a VkFormat
    std::vector<VkVertexInputAttributeDescription> attributes;
    // define how big a single vertex binding or a single instance binding and
    // how it's stepped through memory. *no information of what's inside it*
    std::vector<VkVertexInputBindingDescription> bindings;

    inline bool operator==(VertexIn const& other) const {
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

    inline bool operator!=(VertexIn const& other) const {
      return !((*this) == other);
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

    inline bool operator==(PreRasterization const& other) const {
      return vertexModule == other.vertexModule &&
             geometryModule == other.geometryModule;
    }
    inline bool operator!=(PreRasterization const& other) const {
      return !((*this) == other);
    }

    uint64_t hash() const {
      uint64_t hash = 33 ^ reinterpret_cast<uint64_t>(vertexModule);
      hash = hash * 33 ^ reinterpret_cast<uint64_t>(geometryModule);
      return hash;
    }
  };
  struct FragmentShader {
    VkShaderModule fragmentModule;
    // If dynamic states contains viewport, we care only about size. if dynamic
    // state contains viewport with count, this is unused (same for scissor)
    std::vector<VkViewport> viewports;
    std::vector<VkRect2D> scissors;

    inline bool operator==(FragmentShader const& f) const {
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
    inline bool operator!=(FragmentShader const& f) const {
      return !((*this) == f);
    }

    uint64_t hash() const {
      uint64_t hash = reinterpret_cast<uint64_t>(fragmentModule);
      hash = hash * 33 ^ viewports.size();
      hash = hash * 33 ^ scissors.size();
      return hash;
    }
  };
  struct FragmentOut {
    // for blend state
    uint32_t colorAttachmentCount;
    // stuff for dynamic rendering and depth stencil
    VkFormat depthAttachmentFormat;
    VkFormat stencilAttachmentFormat;
    std::vector<VkFormat> colorAttachmentFormats;

    inline bool operator==(FragmentOut const& other) const {
      if (colorAttachmentCount != other.colorAttachmentCount ||
          depthAttachmentFormat != other.depthAttachmentFormat ||
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
    inline bool operator!=(FragmentOut const& other) const {
      return !((*this) == other);
    }

    uint64_t hash() const {
      uint64_t hash = depthAttachmentFormat;
      hash = hash * 33 ^ stencilAttachmentFormat;
      hash = hash * 33 ^ colorAttachmentCount;
      hash = hash * 33 ^
             VectorHash<VkFormat>{}(colorAttachmentFormats,
                                    [](VkFormat format) { return format; });
      return hash;
    }
  };
  struct PipelineOpts {
    EPipelineFlags flags;
    VkPolygonMode rasterizationPolygonMode;
    EStencilCompareOp stencilCompareOp;
    EStencilLogicOp stencilLogicalOp;
    uint32_t stencilReference;
    uint32_t stencilCompareMask;
    uint32_t stencilWriteMask;

    inline bool operator==(PipelineOpts const& other) const {
      if (flags != other.flags ||
          rasterizationPolygonMode != other.rasterizationPolygonMode ||
          stencilCompareOp != other.stencilCompareOp ||
          stencilLogicalOp != other.stencilLogicalOp ||
          stencilReference != other.stencilReference ||
          stencilCompareMask != other.stencilCompareMask ||
          stencilWriteMask != other.stencilWriteMask) {
        return false;
      }
      return true;
    }
    inline bool operator!=(PipelineOpts const& other) const {
      return !((*this) == other);
    }

    uint64_t hash() const {
      uint64_t hash =
          uint64_t(0) ^
          static_cast<std::underlying_type_t<EPipelineFlags>>(flags);
      hash = hash * 33 ^ rasterizationPolygonMode;
      hash = hash * 33 ^ static_cast<std::underlying_type_t<EStencilCompareOp>>(
                             stencilCompareOp);
      hash = hash * 33 ^ stencilReference;
      hash = hash * 33 ^ static_cast<std::underlying_type_t<EStencilLogicOp>>(
                             stencilLogicalOp);
      hash = hash * 33 ^ stencilCompareMask;
      hash = hash * 33 ^ stencilWriteMask;
      return hash;
    }
  };

  VertexIn vertexIn;
  PreRasterization preRasterization;
  FragmentShader fragmentShader;
  FragmentOut fragmentOut;

  // add GPU state?
  PipelineOpts opts;
  VkPipelineLayout pipelineLayout;
  // TODO add specialization constants
};

inline bool operator==(GraphicsInfo const& a, GraphicsInfo const& b) {
  if (a.vertexIn != b.vertexIn || a.preRasterization != b.preRasterization ||
      a.fragmentShader != b.fragmentShader ||
      a.pipelineLayout != b.pipelineLayout || a.opts != b.opts) {
    return false;
  }
  return true;
}

}  // namespace avk

template <>
struct std::hash<avk::GraphicsInfo> {
  size_t operator()(avk::GraphicsInfo const& graphicsInfo) const {
    uint64_t hash = 33 ^ graphicsInfo.vertexIn.hash();
    hash = hash * 33 ^ graphicsInfo.preRasterization.hash();
    hash = hash * 33 ^ graphicsInfo.fragmentShader.hash();
    hash = hash * 33 ^ graphicsInfo.fragmentOut.hash();
    hash = hash * 33 ^ reinterpret_cast<uint64_t>(graphicsInfo.pipelineLayout);
    hash = hash * 33 ^ graphicsInfo.opts.hash();
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

  VkPipeline getOrCreateComputePipeline(ContextVk const& context,
                                        ComputeInfo& computeInfo,
                                        bool isStaticShader,
                                        VkPipeline pipelineBase);
  VkPipeline getOrCreateGraphicsPipeline(ContextVk const& context,
                                         GraphicsInfo& graphicsInfo,
                                         bool isStaticShader,
                                         VkPipeline pipelineBase);

  // erase pipelines from the maps and inserts them in the discard pool, such
  // that they can live there until they are no longer in use
  // context must be the same used to create the pipelines
  // TODO possible: hash function for context, such that we can filter by
  // context
  // WARNING: calling this won't discard Vulkan Handles inside
  // the info structs. only VkPipelines
  void discardAllPipelines(DiscardPoolVk& discardPool,
                           VkPipelineLayout pipelineLayout);

  void readStaticCacheFromDisk();
  void writeStaticCacheToDisk();

  // WARNING: calling this won't destroy Vulkan Handles inside
  // the info structs. only VkPipelines
  void destroyAllPipelines(ContextVk const& context);

 private:
  std::unordered_map<GraphicsInfo, VkPipeline> m_graphicsPipelines;
  std::unordered_map<ComputeInfo, VkPipeline> m_computePipelines;
  // partially initialized structure to reuse
  VkComputePipelineCreateInfo m_computePipelineCreateInfo;

  static uint32_t constexpr ShaderStageCount = 3;

  VkGraphicsPipelineCreateInfo m_graphicsPipelineCreateInfo;
  VkPipelineRenderingCreateInfo m_pipelineRenderingCreateInfo;
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
};

}  // namespace avk