#pragma once

#include <render/vk/common-vk.h>
#include <render/vk/device-vk.h>

// standard library
#include <thread>
#include <vector>

namespace avk::vk::utils {

// vector of pair timestamp, value
// invariant: timeline values should always be sorted (except for wrap around)
template <typename Item>
class TimelineResources : public std::vector<std::pair<uint64_t, Item>> {
  using Base = std::vector<std::pair<uint64_t, Item>>;

 public:
  void appendTimeline(uint64_t timeline, Item item) {
    Base::emplace_back(timeline, item);
  }

  void updateTimeline(uint64_t timeline) {
    for (std::pair<uint64_t, Item>& pair : *this) {
      pair.first = timeline;
    }
  }

  template <typename Deleter,
            typename V = std::enable_if_t<std::is_invocable_v<Deleter, Item>>>
  void removeOld(uint64_t currentTimeline, Deleter&& deleter) {
    int64_t firstIndexToKeep = 0;
    for (std::pair<uint64_t, Item>& item : *this) {
      if (item.first > currentTimeline) {
        break;
      }
      deleter(item.second);
      ++firstIndexToKeep;
    }

    if (firstIndexToKeep > 0) {
      Base::erase(Base::begin(), Base::begin() + firstIndexToKeep);
    }
  }
};

}  // namespace avk::vk::utils

namespace avk::vk {

// should outlive the discard pool!
class DescriptorPools;
class CommandPools;

class DiscardPool : public NonMoveable {
  friend class DiscardPoolMonitor;

 public:
  DiscardPool(Instance* instance, Device* device);
  /// Assumes its timeline Semaphore is *NOT* being used by a queue
  ~DiscardPool();

  /// will call `vkGetSemaphoreCounterValueKHR` to extract the `value`
  /// parameter for all "discard*" methods
  uint64_t queryTime() const;

  void discardImage(VkImage image, VmaAllocation allocation, uint64_t value);
  void discardImageView(VkImageView imageView, uint64_t value);
  void discardBuffer(VkBuffer buffer, VmaAllocation allocation, uint64_t value);
  void discardBufferView(VkBufferView bufferView, uint64_t value);
  void discardShaderModule(VkShaderModule shaderModule, uint64_t value);
  void discardPipeline(VkPipeline pipeline, uint64_t value);
  void discardPipelineLayout(VkPipelineLayout pipelineLayout, uint64_t value);
  void discardDescriptorPoolForReuse(VkDescriptorPool descriptorPool,
                                     DescriptorPools* pools, uint64_t value);
  void discardCommandPoolForReuse(VkCommandPool commandPool,
                                  CommandPools* pools, std::thread::id tid,
                                  uint64_t value);
  void discardSurface(VkSurfaceKHR surface, uint64_t value);
  // stuff from renderpasses (to see if needed)
  void discardRenderPass(VkRenderPass renderPass, uint64_t value);
  void discardFramebuffer(VkFramebuffer framebuffer, uint64_t value);

  void destroyDiscardedResources(bool force = false);
  inline VkSemaphore timelineSemaphore() const { return m_timeline; }
  inline operator bool() const { return m_timeline != VK_NULL_HANDLE; }

 private:
  // dependencies which must outlive this object
  struct Deps {
    Instance* instance;
    Device* device;
  } m_deps;

  utils::TimelineResources<VMAResource<VkImage>> m_images;
  utils::TimelineResources<VMAResource<VkBuffer>> m_buffers;
  utils::TimelineResources<VkImageView> m_imageViews;
  utils::TimelineResources<VkBufferView> m_bufferViews;
  utils::TimelineResources<VkShaderModule> m_shaderModules;
  utils::TimelineResources<VkPipeline> m_pipelines;
  utils::TimelineResources<VkPipelineLayout> m_pipelineLayouts;

  utils::TimelineResources<std::pair<VkDescriptorPool, DescriptorPools*>>
      m_descriptorPools;

  struct CmdDiscard {
    VkCommandPool pool;
    std::thread::id tid;
    CommandPools* manager;
  };
  utils::TimelineResources<CmdDiscard> m_commandPools;

  // in the rare case of a surface lost on mobile, or when closing
  // secondary windows on desktop
  utils::TimelineResources<VkSurfaceKHR> m_surfaces;

  // stuff from renderPasses (to see if needed)
  utils::TimelineResources<VkRenderPass> m_renderPasses;
  utils::TimelineResources<VkFramebuffer> m_framebuffers;

  // maintain the timeline inside it
  VkSemaphore m_timeline = VK_NULL_HANDLE;

  // synchronization for multithreaded usage
  std::mutex m_mtx;
};

/// Companion class which handles periodic destruction of
/// the discard pool, to be configured to be called every N Frames
/// This checks only the resources which are more frequently siphoned
/// - images, buffers
/// - framebuffers, pipelines
class DiscardPoolMonitor : public NonMoveable {
 public:
  struct Config {
    size_t MaxImages = 32;
    size_t MaxBuffers = 64;
    size_t MaxFramebuffers = 32;
    size_t MaxPipelines = 16;
    uint32_t CheckEveryNFrames = 240;  // 4 seconds on 60fps
  };

  DiscardPoolMonitor(DiscardPool* discardPool) : m_deps{discardPool} {}

  inline void onFrame() {
    if (++m_frameCounter < Conf.CheckEveryNFrames) return;
    m_frameCounter = 0;
    checkAndCleanup();
  }

  Config Conf;

 private:
  void checkAndCleanup() {
    if (!m_deps.discardPool) return;
    size_t const img = m_deps.discardPool->m_images.size();
    size_t const buf = m_deps.discardPool->m_buffers.size();
    size_t const fb = m_deps.discardPool->m_framebuffers.size();
    size_t const pipe = m_deps.discardPool->m_pipelines.size();
    bool const overLimit = img > Conf.MaxImages || buf > Conf.MaxBuffers ||
                           fb > Conf.MaxFramebuffers ||
                           pipe > Conf.MaxPipelines;
    if (overLimit) {
      LOGI << "[DiscardPoolMonitor] Resource Pressure Detected: " << img
           << " images, " << buf << " buffers, " << fb << " framebuffers, "
           << pipe << " pipelines. "
           << "Triggering cleanup..." << std::endl;
      m_deps.discardPool->destroyDiscardedResources();
    }
  }

  // dependencies which must outlive this object
  struct Deps {
    DiscardPool* discardPool;
  } m_deps;

  uint64_t m_frameCounter = 0;
};

}  // namespace avk::vk