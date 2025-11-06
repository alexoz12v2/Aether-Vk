#include <Windows.h>

// Windows related stuff
#include <Windowsx.h>  // GET_X_LPARAM
#include <basetsd.h>
#include <errhandlingapi.h>
#include <vulkan/vulkan_core.h>
#include <wingdi.h>
#include <winspool.h>
#include <winuser.h>

// ExtractIconExW
#include <shellapi.h>
// https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
#include <dwmapi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>

#include "avk-win32-window.h"
#include "fiber/jobs.h"
#include "os/filesystem.h"
#include "render/context-vk.h"
#include "render/pipeline-vk.h"
#include "render/utils-vk.h"
#include "render/vk/common-vk.h"
#include "utils/mixins.h"

// GLM
#include <glm/common.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include "os/stackstrace.h"

// TODO add Ctrl + C handler

struct PerFrameData {
  VkImage resolveColorImage;
  VkImage depthImage;
  VmaAllocation resolveColorAlloc;
  VmaAllocation depthImageAlloc;
  VkImageView resolveColorImageView;
  VkImageView depthImageView;
  VkCommandBuffer commandBuffer;  // TODO better (using timeline semaphores)
  uint64_t timeline;
  VkDescriptorSet uboDescriptorSet;
};

struct ModelViewProj {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

struct PushConstant {
  float scaleFactor;
  float colorAdd;
};

static void printError() {
  DWORD const err = GetLastError();
  wchar_t* messageBuffer = nullptr;

  FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                     FORMAT_MESSAGE_IGNORE_INSERTS,
                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                 reinterpret_cast<wchar_t*>(&messageBuffer), 0, NULL);
  MessageBoxW(nullptr, messageBuffer, L"Error", MB_OK);
  LocalFree(messageBuffer);
}

// TODO better
template <typename F>
static void syncLog(F&& f) {
  static std::mutex coutMtx;
  std::lock_guard<std::mutex> lk{coutMtx};
  f();
}

// -------------------------------------------------------

static std::atomic<bool> g_quit{false};
// static std::atomic<bool> g_presentReady{false};
static std::atomic<HWND> g_window(nullptr);

struct Vertex {
  float position[3];
  float color[3];
};

class BasicSwapchainCallbacks : public avk::ISwapchainCallbacks,
                                public avk::NonCopyable {
 public:
  void onSwapchainRecreationStarted(avk::ContextVk const& context) override;
  void onSwapBufferDrawCallback(
      avk::ContextVk const& context,
      const avk::SwapchainDataVk* swapchainData) override;
  void onSwapBufferAcquiredCallback(avk::ContextVk const& context) override;
  void onSwapchainRecreationCallback(avk::ContextVk const& context,
                                     VkImage const* images, uint32_t numImages,
                                     VkFormat format,
                                     VkExtent2D imageExtent) override;

  void discardEverything(avk::ContextVk const& context);

  // non owning reference to needed data
  VkFormat DepthFormat;
  std::vector<VkSubmitInfo>* SubmitInfos;
  std::vector<PerFrameData>* PerFrameData;
  VkCommandPool CommandPool;
  avk::DiscardPoolVk* DiscardPool;
  VkSemaphore TimelineSemaphore;
  VkDescriptorPool DescriptorPool;
  VkDescriptorSetLayout DescriptorSetLayout;
  avk::BufferVk* ConstantUniformBufferMVP;

 private:
  std::vector<VkDescriptorSetLayout> m_tempSetLayouts;
  std::vector<VkDescriptorSet> m_tempSets;
};

void BasicSwapchainCallbacks::onSwapchainRecreationStarted(
    [[maybe_unused]] avk::ContextVk const& context) {}

void BasicSwapchainCallbacks::onSwapBufferDrawCallback(
    avk::ContextVk const& context, const avk::SwapchainDataVk* swapchainData) AVK_NO_CFI {
  if (SubmitInfos->size() > 0) {
    avk::vkCheck(vkQueueSubmit(context.device().graphicsComputeQueue,
                               static_cast<uint32_t>(SubmitInfos->size()),
                               SubmitInfos->data(),
                               swapchainData->submissionFence));

    DiscardPool->updateTimeline();
  }
}

void BasicSwapchainCallbacks::onSwapBufferAcquiredCallback(
    [[maybe_unused]] avk::ContextVk const& context) {}

void BasicSwapchainCallbacks::onSwapchainRecreationCallback(
    avk::ContextVk const& context, [[maybe_unused]] VkImage const* images,
    uint32_t numImages, VkFormat format, VkExtent2D imageExtent) AVK_NO_CFI {
  uint64_t const timeline = DiscardPool->getTimeline();
  m_tempSetLayouts.clear();
  m_tempSets.clear();

  // TODO remove debug
  uint64_t actualSem = 0;
  vkGetSemaphoreCounterValue(context.device().device, TimelineSemaphore,
                             &actualSem);
  std::cout << "\033[33m" << "Actual Timeline:  " << actualSem
            << "\nDiscard Timeline: " << timeline << "\033[0m" << std::endl;
  // reset frames descriptor sets inside the frame pool
  vkResetDescriptorPool(context.device().device, DescriptorPool, 0);

  // 1. add all resources in the vector into the discard pool
  std::vector<VkCommandBuffer> commandBuffers;
  commandBuffers.reserve(64);
  for (auto const& data : (*PerFrameData)) {
    DiscardPool->discardImageView(data.depthImageView);
    DiscardPool->discardImageView(data.resolveColorImageView);
    DiscardPool->discardImage(data.depthImage, data.depthImageAlloc);
    DiscardPool->discardImage(data.resolveColorImage, data.resolveColorAlloc);
    // TODO handle discarding command buffers better
    commandBuffers.push_back(data.commandBuffer);
  }

  DiscardPool->destroyDiscardedResources(context.device());
  PerFrameData->clear();
  PerFrameData->resize(numImages);

  // 1.1 create new command buffers/reuse old ones
  // TODO better
  if (commandBuffers.size() < PerFrameData->size()) {
    // create missing command buffers
    size_t const offset = commandBuffers.size();
    size_t const count = PerFrameData->size() - commandBuffers.size();
    commandBuffers.resize(count);
    avk::allocPrimaryCommandBuffers(context, CommandPool, count,
                                    commandBuffers.data() + offset);
    // assign the newly created
    for (size_t i = 0; i < PerFrameData->size(); ++i) {
      (*PerFrameData)[i].commandBuffer = commandBuffers[i];
    }
  } else if (commandBuffers.size() > PerFrameData->size()) {
    // handle excess command buffers
    size_t const offset = PerFrameData->size();
    size_t const count = commandBuffers.size() - PerFrameData->size();
    vkFreeCommandBuffers(context.device().device, CommandPool, count,
                         commandBuffers.data() + offset);
    commandBuffers.resize(PerFrameData->size());

    // reassign to perFrameData
    for (size_t i = 0; i < PerFrameData->size(); ++i) {
      (*PerFrameData)[i].commandBuffer = commandBuffers[i];
    }
  } else {  // 3) equal sizes — just reassign existing command buffers
    for (size_t i = 0; i < PerFrameData->size(); ++i) {
      (*PerFrameData)[i].commandBuffer = commandBuffers[i];
    }
  }

  // 2. create new resources
  VkImageViewCreateInfo depthViewsCreateInfo{};
  depthViewsCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  depthViewsCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  depthViewsCreateInfo.format = DepthFormat;
  depthViewsCreateInfo.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
  depthViewsCreateInfo.subresourceRange.baseArrayLayer = 0;
  depthViewsCreateInfo.subresourceRange.layerCount = 1;
  depthViewsCreateInfo.subresourceRange.baseMipLevel = 0;
  depthViewsCreateInfo.subresourceRange.levelCount = 1;

  VkImageViewCreateInfo resolveColorViewCreateInfo{};
  resolveColorViewCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  resolveColorViewCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  resolveColorViewCreateInfo.format = format;
  resolveColorViewCreateInfo.subresourceRange.aspectMask =
      VK_IMAGE_ASPECT_COLOR_BIT;
  resolveColorViewCreateInfo.subresourceRange.baseArrayLayer = 0;
  resolveColorViewCreateInfo.subresourceRange.layerCount = 1;
  resolveColorViewCreateInfo.subresourceRange.baseMipLevel = 0;
  resolveColorViewCreateInfo.subresourceRange.levelCount = 1;

  avk::SingleImage2DSpecVk depthSpec{};
  depthSpec.imageTiling = VK_IMAGE_TILING_OPTIMAL;
  depthSpec.format = DepthFormat;
  // TODO can depth buffering use multisampling?
  depthSpec.samples = VK_SAMPLE_COUNT_1_BIT;
  depthSpec.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
  depthSpec.width = imageExtent.width;
  depthSpec.height = imageExtent.height;

  avk::SingleImage2DSpecVk resolveColorSpec{};
  resolveColorSpec.width = imageExtent.width;
  resolveColorSpec.height = imageExtent.height;
  resolveColorSpec.imageTiling = VK_IMAGE_TILING_OPTIMAL;
  resolveColorSpec.format = format;
  // TODO: resolve not attachment
  resolveColorSpec.usage =
      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
  resolveColorSpec.samples = VK_SAMPLE_COUNT_1_BIT;

  // allocate descriptor sets
  VkDescriptorSetAllocateInfo descAllocInfo{};
  descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descAllocInfo.descriptorPool = DescriptorPool;
  // If you want more, you duplicate the set layout, even if it's the same
  m_tempSetLayouts.resize(PerFrameData->size());
  m_tempSets.resize(PerFrameData->size());
  std::fill_n(m_tempSetLayouts.begin(), PerFrameData->size(),
              DescriptorSetLayout);
  descAllocInfo.descriptorSetCount =
      static_cast<uint32_t>(PerFrameData->size());
  descAllocInfo.pSetLayouts = m_tempSetLayouts.data();
  // TODO better if fails crash
  vkAllocateDescriptorSets(context.device().device, &descAllocInfo,
                           m_tempSets.data());

  uint32_t index = 0;
  for (auto& data : (*PerFrameData)) {
    uint32_t const i = index;
    data.timeline = timeline + index++;

    // 2.1 create depth image
    // TODO: this is a complete failure, crash the application
    if (!avk::createImage(context, depthSpec, data.depthImage,
                          data.depthImageAlloc)) {
      data.depthImageView = VK_NULL_HANDLE;
    } else {
      depthViewsCreateInfo.image = data.depthImage;
      // create depth image views
      avk::vkCheck(vkCreateImageView(context.device().device,
                                     &depthViewsCreateInfo, nullptr,
                                     &data.depthImageView));
    }

    // 3 color attachment images
    // TODO: this is a complete failure, crash the application
    if (!avk::createImage(context, resolveColorSpec, data.resolveColorImage,
                          data.resolveColorAlloc)) {
      data.resolveColorImage = VK_NULL_HANDLE;
    } else {
      resolveColorViewCreateInfo.image = data.resolveColorImage;
      vkCreateImageView(context.device().device, &resolveColorViewCreateInfo,
                        nullptr, &data.resolveColorImageView);
    }

    // assign its descriptor set and write it
    // without writing a descriptor set only gives you an empty container, not a
    // point to the buffer or image memory. to actually create that association
    // you need a call to vkUpdateDescriptorSets or vkWriteDescriptorSet
    data.uboDescriptorSet = m_tempSets[i];

    VkDescriptorBufferInfo descBufferInfo{};
    descBufferInfo.buffer = ConstantUniformBufferMVP->buffer();
    descBufferInfo.offset = 0;
    descBufferInfo.range = sizeof(ModelViewProj);

    VkWriteDescriptorSet descriptorWrite{};
    descriptorWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrite.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptorWrite.dstSet = data.uboDescriptorSet;
    descriptorWrite.dstBinding = 0;  // same as layout
    // only if descriptor set describes an array
    descriptorWrite.dstArrayElement = 0;
    descriptorWrite.descriptorCount = 1;
    descriptorWrite.pBufferInfo = &descBufferInfo;

    vkUpdateDescriptorSets(context.device().device, 1, &descriptorWrite, 0,
                           nullptr);
  }
}

void discardEverything(avk::ContextVk const& contextVk,
                       avk::DiscardPoolVk& discardPool,
                       VkCommandPool* commandPool,
                       std::vector<PerFrameData>& perFrameData) AVK_NO_CFI {
  for (auto const& data : perFrameData) {
    discardPool.discardImageView(data.depthImageView);
    discardPool.discardImageView(data.resolveColorImageView);
    discardPool.discardImage(data.depthImage, data.depthImageAlloc);
    discardPool.discardImage(data.resolveColorImage, data.resolveColorAlloc);
    vkFreeCommandBuffers(contextVk.device().device, *commandPool, 1,
                         &data.commandBuffer);
  }
  vkDestroyCommandPool(contextVk.device().device, *commandPool, nullptr);
  *commandPool = VK_NULL_HANDLE;
}

// TODO manifest
// Thread that owns the window
void messageThreadProc(avk::WindowPayload* windowPayload) {
  syncLog([] { std::cout << "Started UI Thread" << std::endl; });

  HWND hMainWindow = avk::createPrimaryWindow(windowPayload);
  g_window.store(hMainWindow);
  if (!g_window.load()) {
    printError();
    g_quit.store(true);
    return;
  }

  avk::primaryWindowMessageLoop(hMainWindow, windowPayload, g_quit);
}

struct FrameProducerPayload {
  HWND hMainWindow;
  avk::ContextVk* context;
  int jobBufferSize;
};

VkImageMemoryBarrier2 imageMemoryBarrier(
    VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
    VkAccessFlags2 srcAccessMask, VkAccessFlags2 dstAccessMask,
    VkPipelineStageFlags2 srcStage, VkPipelineStageFlags2 dstStage,
    VkImageAspectFlags aspectMask, uint32_t srcIndex = VK_QUEUE_FAMILY_IGNORED,
    uint32_t dstIndex = VK_QUEUE_FAMILY_IGNORED) {
  // initialize struct
  VkImageMemoryBarrier2 imageBarrier{};
  imageBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;

  // stecify the pipelines stages and access masks for the barrier
  imageBarrier.srcStageMask = srcStage;
  imageBarrier.srcAccessMask = srcAccessMask;
  imageBarrier.dstStageMask = dstStage;
  imageBarrier.dstAccessMask = dstAccessMask;

  // specify the old and new layouts of the image
  imageBarrier.oldLayout = oldLayout;
  imageBarrier.newLayout = newLayout;

  // we can change the ownership between queues (case graphics != present)
  imageBarrier.srcQueueFamilyIndex = srcIndex;
  imageBarrier.dstQueueFamilyIndex = dstIndex;

  // specify the image to be affected by this barrier
  imageBarrier.image = image;

  // define subrange of the image
  imageBarrier.subresourceRange.aspectMask = aspectMask;
  imageBarrier.subresourceRange.baseMipLevel = 0;
  imageBarrier.subresourceRange.levelCount = 1;
  imageBarrier.subresourceRange.baseArrayLayer = 0;
  imageBarrier.subresourceRange.layerCount = 1;
  return imageBarrier;
}

void transitionImageLayout(VkCommandBuffer cmd,
                           VkImageMemoryBarrier2 const* imageBarriers,
                           uint32_t imageBarrierCount) AVK_NO_CFI {
  // initialize dependency info
  VkDependencyInfo dependencyInfo{};
  dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
  dependencyInfo.dependencyFlags = 0;
  dependencyInfo.imageMemoryBarrierCount = imageBarrierCount;
  dependencyInfo.pImageMemoryBarriers = imageBarriers;

  // record pipeline barrier into command buffer
  vkCmdPipelineBarrier2(cmd, &dependencyInfo);
}

std::vector<uint32_t> openSpirV(std::filesystem::path const& path) {
  syncLog([&]() { std::wcout << L"Shader Path: " << path << std::endl; });

  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file) {
    throw std::runtime_error("Failed to open SPIR-V file: " + path.string());
  }

  size_t fileSize = static_cast<size_t>(file.tellg());
  if (fileSize % 4 != 0) {
    return {};
  }

  std::vector<uint32_t> shaderCode(fileSize / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(shaderCode.data()), fileSize);

  return shaderCode;
}

void fillTriangleGraphicsInfo(avk::ContextVk const& context,
                              avk::GraphicsInfo& graphicsInfo,
                              VkPipelineLayout pipelineLayout) {
  graphicsInfo.pipelineLayout = pipelineLayout;

  graphicsInfo.vertexIn.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  // 1 binding, 2 attributes
  VkVertexInputBindingDescription bindingDesc{};
  bindingDesc.binding = 0;
  bindingDesc.stride = sizeof(Vertex);
  bindingDesc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  graphicsInfo.vertexIn.bindings.push_back(bindingDesc);

  VkVertexInputAttributeDescription attrDesc{};
  attrDesc.binding = 0;
  attrDesc.format = VK_FORMAT_R32G32_SFLOAT;  // position float2
  attrDesc.offset = offsetof(Vertex, position);
  attrDesc.location = 0;
  graphicsInfo.vertexIn.attributes.push_back(attrDesc);

  attrDesc.location = 1;
  attrDesc.offset = offsetof(Vertex, color);
  attrDesc.format = VK_FORMAT_R32G32B32_SFLOAT;  // color float3
  graphicsInfo.vertexIn.attributes.push_back(attrDesc);
  auto const execPath = avk::getExecutablePath().parent_path();
  auto const vertCode = openSpirV(execPath / "triangle.vert.slang.spv");
  auto const fragCode = openSpirV(execPath / "triangle.frag.slang.spv");
  assert(!vertCode.empty() && !fragCode.empty());
  // TODO add them to the cleanup somehow (eg device resource tracker)
  VkShaderModule vertModule = avk::finalizeShaderModule(
      context, vertCode.data(), vertCode.size() * sizeof(uint32_t));
  VkShaderModule fragModule = avk::finalizeShaderModule(
      context, fragCode.data(), fragCode.size() * sizeof(uint32_t));

  graphicsInfo.preRasterization.vertexModule = vertModule;
  graphicsInfo.preRasterization.geometryModule = VK_NULL_HANDLE;
  graphicsInfo.fragmentShader.fragmentModule = fragModule;
  // viewport and scissor are using dynamic state without count, hence only size
  // of vector matters
  graphicsInfo.fragmentShader.viewports.resize(1);
  graphicsInfo.fragmentShader.scissors.resize(1);

  graphicsInfo.fragmentOut.depthAttachmentFormat =
      avk::basicDepthStencilFormat(context.device().physicalDevice);
  graphicsInfo.fragmentOut.stencilAttachmentFormat =
      avk::basicDepthStencilFormat(context.device().physicalDevice);

  graphicsInfo.opts.rasterizationPolygonMode = VK_POLYGON_MODE_FILL;

  // write stencil
  graphicsInfo.opts.flags |= avk::EPipelineFlags::eStencilEnable;
  graphicsInfo.opts.stencilCompareOp = avk::EStencilCompareOp::eAlways;
  graphicsInfo.opts.stencilLogicalOp = avk::EStencilLogicOp::eReplace;
  graphicsInfo.opts.stencilReference = 1;
  graphicsInfo.opts.stencilCompareMask = 0xFF;
  graphicsInfo.opts.stencilWriteMask = 0xFF;
}

std::vector<uint32_t> uniqueElements(uint32_t const* elems, uint32_t count) {
  std::vector<uint32_t> vElems(count);
  std::copy_n(elems, count, vElems.begin());
  std::sort(vElems.begin(), vElems.end());
  auto it = std::unique(vElems.begin(), vElems.end());
  vElems.resize(std::distance(vElems.begin(), it));
  return vElems;
}

void frameProducer(avk::Scheduler* sched, FrameProducerPayload const* payload) AVK_NO_CFI {
  std::cout << "Started Render Thread" << std::endl;
  avk::ContextVk& contextVk = *payload->context;
  std::cout << avk::dumpStackTrace() << std::endl;

  // per frame data to update every swapchain recreation
  std::vector<PerFrameData> perFrameData;
  perFrameData.reserve(64);

  VkFormat depthFormat =
      avk::basicDepthStencilFormat(contextVk.device().physicalDevice);

  avk::DiscardPoolVk discardPool;

  // TODO create TLS Command Pool
  VkCommandPool commandPool = avk::createCommandPool(
      contextVk, true, contextVk.device().queueIndices.family.graphicsCompute);

  // Timeline Semaphore to keep track of frame index, hence discard when
  // appropriate
  VkSemaphore timelineSemaphore = VK_NULL_HANDLE;
  {
    VkSemaphoreTypeCreateInfoKHR semTimelineCreateInfo{};
    semTimelineCreateInfo.sType =
        VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO_KHR;
    semTimelineCreateInfo.initialValue = 0;
    semTimelineCreateInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE_KHR;

    VkSemaphoreCreateInfo semCreateInfo{};
    semCreateInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semCreateInfo.pNext = &semTimelineCreateInfo;
    if (!avk::vkCheck(vkCreateSemaphore(contextVk.device().device,
                                        &semCreateInfo, nullptr,
                                        &timelineSemaphore))) {
      std::cerr << "\033[31m"
                << "Failed to create frame index global timeline semaphore"
                << "\033[0m" << std::endl;
      g_quit.store(true);
    }
  }

  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  {
    // maximum numbers of descriptor sets that can be created depend on the type
    // TODO add type limit check
    VkDescriptorPoolSize uboSize{};
    uboSize.descriptorCount = 15;
    uboSize.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;

    VkDescriptorPoolCreateInfo poolCreateInfo{};
    poolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolCreateInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolCreateInfo.maxSets = 15;
    poolCreateInfo.poolSizeCount = 1;
    poolCreateInfo.pPoolSizes = &uboSize;

    // TODO if fail crash
    vkCreateDescriptorPool(contextVk.device().device, &poolCreateInfo, nullptr,
                           &descriptorPool);
  }

  VkDescriptorSetLayout uboSetLayout = VK_NULL_HANDLE;
  {
    VkDescriptorSetLayoutBinding uboBinding{};
    uboBinding.binding = 0;  // first number of [[vk::binding(0,0)]]
    uboBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uboBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    // if descriptorCount bigger than 1, then the object is accessed as an array
    // on the shader
    uboBinding.descriptorCount = 1;

    VkDescriptorSetLayoutCreateInfo setLayoutCreateInfo{};
    setLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    setLayoutCreateInfo.bindingCount = 1;
    setLayoutCreateInfo.pBindings = &uboBinding;

    vkCreateDescriptorSetLayout(contextVk.device().device, &setLayoutCreateInfo,
                                nullptr, &uboSetLayout);
  }

  VkPushConstantRange pushRange{};
  pushRange.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT;
  pushRange.offset = 0;
  pushRange.size = sizeof(PushConstant);

  // ----- Pipeline Layout and Descriptors
  ModelViewProj mvp;
  mvp.model = glm::translate(glm::mat4(1.f), {1.f, 0, 0});
  mvp.view = glm::mat4(1.f);
  mvp.proj = glm::mat4(1.f);

  VmaAllocationCreateFlags stagingVmaFlags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;

  avk::BufferVk uniformBufferUBO;
  if (!uniformBufferUBO.create(contextVk, sizeof(mvp),
                               VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                               VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                               0, stagingVmaFlags, 1.f, false, 0, nullptr)) {
    // TODO crash better
    assert(false);
  }
  uniformBufferUBO.updateImmediately(&mvp);

  // TODO better
  std::vector<VkSubmitInfo> submitInfos;
  std::vector<VkTimelineSemaphoreSubmitInfoKHR> timelineSubmitInfos;
  std::vector<std::array<VkSemaphore, 2>> submitSignalSemaphores;
  std::vector<std::array<uint64_t, 2>> submitSignalSemaphoresValues;
  submitInfos.reserve(16);
  timelineSubmitInfos.reserve(16);
  submitSignalSemaphores.reserve(16);
  submitSignalSemaphoresValues.reserve(16);

  BasicSwapchainCallbacks callbacks;
  callbacks.SubmitInfos = &submitInfos;
  callbacks.PerFrameData = &perFrameData;
  callbacks.DiscardPool = &discardPool;
  callbacks.CommandPool = commandPool;
  callbacks.DepthFormat = depthFormat;
  callbacks.TimelineSemaphore = timelineSemaphore;
  callbacks.DescriptorPool = descriptorPool;
  callbacks.DescriptorSetLayout = uboSetLayout;
  callbacks.ConstantUniformBufferMVP = &uniformBufferUBO;

  contextVk.setSwapBufferCallbacks(&callbacks);

  avk::VkPipelinePool pipelinePool;
  avk::GraphicsInfo graphicsInfo;
  VkPipelineLayout pipelineLayout =
      avk::createPipelineLayout(contextVk, &uboSetLayout, 1, &pushRange, 1);

  // must do before pipeline creation as it populates the surface format
  // TODO decouple classes
  contextVk.recreateSwapchain(false);

  fillTriangleGraphicsInfo(contextVk, graphicsInfo, pipelineLayout);

  VkPipeline pipeline = pipelinePool.getOrCreateGraphicsPipeline(
      contextVk, graphicsInfo, true, VK_NULL_HANDLE);

  avk::GraphicsInfo outlineInfo = graphicsInfo;
  outlineInfo.opts.stencilCompareOp = avk::EStencilCompareOp::eNotEqual;
  outlineInfo.opts.stencilLogicalOp =
      avk::EStencilLogicOp::eNone;  // Don't modify stencil
  outlineInfo.opts.stencilReference = 1;
  outlineInfo.opts.stencilCompareMask = 0xFF;
  outlineInfo.opts.stencilWriteMask = 0x00;  // Don't write new stencil values
  outlineInfo.opts.flags |= avk::EPipelineFlags::eStencilEnable;
  // outlineInfo.opts.rasterizationPolygonMode =
  //     VK_POLYGON_MODE_LINE;  // Outline mode

  VkPipeline outlinePipeline = pipelinePool.getOrCreateGraphicsPipeline(
      contextVk, outlineInfo, true, VK_NULL_HANDLE);

  // Create initial Vertex Buffer
  const std::vector<Vertex> vertices = {
      {{0.5f, -0.5f, 0}, {1.0f, 0.0f, 0.0f}},  // RT Vertex 0: Red
      {{-0.5f, 0.5f, 0}, {0.0f, 1.0f, 0.0f}},  // LB Vertex 1: Green
      {{0.5f, 0.5f, 0}, {0.0f, 0.0f, 1.0f}},   // RB Vertex 2: Blue
      {{-0.5f, -0.5f, 0}, {1.0f, 0.0f, 1.0f}}  // LT Vertex 3: Magenta
  };
  const std::vector<std::array<uint32_t, 3>> indices = {{0, 1, 2}, {0, 3, 1}};
  std::vector<uint32_t> uniqueQueueFamilies =
      uniqueElements(contextVk.device().queueIndices.families,
                     avk::DeviceVk::QueueFamilyIndicesCount);
  assert(uniqueQueueFamilies.size() > 0);
  avk::BufferVk vertexBuffer;
  avk::BufferVk indexBuffer;
  VmaAllocationCreateFlags dynamicBufferFlags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
      VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
      VMA_ALLOCATION_CREATE_MAPPED_BIT;
  // TODO create 2 buffers, 1 device visible and not host visible, the other
  // host visible such that you can create a staging buffer
  if (!vertexBuffer.create(contextVk, sizeof(vertices[0]) * vertices.size(),
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           0, dynamicBufferFlags, 1.f, false,
                           static_cast<uint32_t>(uniqueQueueFamilies.size()),
                           uniqueQueueFamilies.data())) {
    std::cerr << "Couldn't allocate Vertex Buffer Host Visible" << std::endl;
  }
  if (!indexBuffer.create(contextVk, sizeof(indices[0]) * indices.size(),
                          VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                          VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                          0, dynamicBufferFlags, 1.f, false,
                          static_cast<uint32_t>(uniqueQueueFamilies.size()),
                          uniqueQueueFamilies.data())) {
    std::cerr << "Couldn't allocate Index Buffer Host Visible" << std::endl;
  }

  vertexBuffer.updateImmediately(vertices.data());
  indexBuffer.updateImmediately(indices.data());

  while (!g_quit.load()) {
    submitInfos.clear();

    contextVk.swapBufferAcquire();
    avk::SwapchainDataVk swapchainData = contextVk.getSwapchainData();
    {  // update timeline
      auto& frame = perFrameData[swapchainData.imageIndex];
      frame.timeline += perFrameData.size();
    }
    auto const& frame = perFrameData[swapchainData.imageIndex];
    // TODO if necessary vkQueueWaitIdle on graphics Queue?
    // if you wait for the queue to be idle, you can delete immediately command
    // buffer, otherwise, discard it and delete it when you pass to another
    // timeline
    // TODO see better timeline semaphores
    // TODO multithreaded command buffer recording and multithreaded data

    VkCommandBuffer commandBuffer = frame.commandBuffer;
    assert(commandBuffer && "Invalid command buffer from frame data");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  // one time

    avk::vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    // before rendering, transition current image to COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier2 imageBarriers[2];
    uint32_t const imageBarrierCount = 2;
    imageBarriers[0] =
        imageMemoryBarrier(frame.resolveColorImage, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           0 /*no need to wait previous operation */,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                           VK_IMAGE_ASPECT_COLOR_BIT);
    imageBarriers[1] = imageMemoryBarrier(
        frame.depthImage, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);
    transitionImageLayout(commandBuffer, imageBarriers, imageBarrierCount);

    VkClearValue clearValue{{{0.017f, 0.017f, 0.17f, 1.0f}}};
    // rendering attachment info and begin rendering
    // TODO attachment info for depth/stencil if necessary
    // TODO populate resolve fields for multisampling.
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = frame.resolveColorImageView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue = clearValue;

    VkRenderingAttachmentInfo depthAttachment{};
    depthAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    depthAttachment.imageView = frame.depthImageView;
    depthAttachment.imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil = {1.0f, 0};

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = {{0, 0}, swapchainData.extent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;  // TODO graphicsInfo?
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
    renderingInfo.pStencilAttachment = &depthAttachment;

    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                            pipelineLayout, 0, 1, &frame.uboDescriptorSet, 0,
                            nullptr);

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    PushConstant thePushConstant;
    thePushConstant.colorAdd = 0;
    thePushConstant.scaleFactor = 1.f;

    // PASS 1: Write stencil + normal quad
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdPushConstants(
        commandBuffer, pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(PushConstant), &thePushConstant);

    // set dynamic states
    VkViewport vp{};
    vp.height = static_cast<float>(swapchainData.extent.height);
    vp.width = static_cast<float>(swapchainData.extent.width);
    vp.minDepth = 0.f;
    vp.maxDepth = 1.f;
    vkCmdSetViewport(commandBuffer, 0, 1, &vp);

    VkRect2D scissor{};
    scissor.extent = swapchainData.extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // bind vertex buffer
    VkBuffer vBuf = vertexBuffer.buffer();
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vBuf, &offset);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer(), 0,
                         VK_INDEX_TYPE_UINT32);

    // issue draw call
    uint32_t const indexCount =
        static_cast<uint32_t>(indices[0].size() * indices.size());
    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);

    // PASS 2: Draw outline
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                      outlinePipeline);
    thePushConstant.colorAdd = 1.f;
    thePushConstant.scaleFactor = 1.25f;
    vkCmdPushConstants(
        commandBuffer, pipelineLayout,
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(PushConstant), &thePushConstant);
    vkCmdDrawIndexed(commandBuffer, indexCount, 1, 0, 0, 0);

    // complete rendering
    vkCmdEndRendering(commandBuffer);

    // after rendering, transition to PRESENT_SRC layout
    imageBarriers[0] = imageMemoryBarrier(
        frame.resolveColorImage, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        0 /*no need to wait previous operation */,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT);
    imageBarriers[1] = imageMemoryBarrier(
        swapchainData.image, VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, 0 /* no access */,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    transitionImageLayout(commandBuffer, imageBarriers, imageBarrierCount);

    // copy from resolve color to swapchain image
    VkImageCopy2 region{};
    region.sType = VK_STRUCTURE_TYPE_IMAGE_COPY_2;
    region.extent.depth = 1;
    region.extent.width = swapchainData.extent.width;
    region.extent.height = swapchainData.extent.height;
    region.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.srcSubresource.layerCount = 1;
    region.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.dstSubresource.layerCount = 1;

    VkCopyImageInfo2 copyInfo{};
    copyInfo.sType = VK_STRUCTURE_TYPE_COPY_IMAGE_INFO_2;
    copyInfo.srcImage = frame.resolveColorImage;
    copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    copyInfo.dstImage = swapchainData.image;
    copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;
    vkCmdCopyImage2(commandBuffer, &copyInfo);

    imageBarriers[0] = imageMemoryBarrier(
        swapchainData.image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        0 /* no access */, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    transitionImageLayout(commandBuffer, imageBarriers, 1);

    // complete recording of the command buffer
    avk::vkCheck(vkEndCommandBuffer(commandBuffer));

    timelineSubmitInfos.clear();
    timelineSubmitInfos.push_back({});
    submitInfos.clear();
    submitInfos.push_back({});
    submitSignalSemaphores.clear();
    submitSignalSemaphores.push_back({});
    submitSignalSemaphoresValues.clear();
    submitSignalSemaphoresValues.push_back({});

    VkPipelineStageFlags waitStage{VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT};
    submitSignalSemaphoresValues[0][1] = frame.timeline;
    submitSignalSemaphores[0][0] = swapchainData.presentSemaphore;
    submitSignalSemaphores[0][1] = timelineSemaphore;

    // TODO better later
    VkTimelineSemaphoreSubmitInfoKHR& timelineSubmitInfo =
        timelineSubmitInfos.back();
    timelineSubmitInfo.sType =
        VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR;
    timelineSubmitInfo.signalSemaphoreValueCount =
        static_cast<uint32_t>(submitSignalSemaphores[0].size());
    timelineSubmitInfo.pSignalSemaphoreValues =
        submitSignalSemaphoresValues[0].data();

    VkSubmitInfo& submitInfo = submitInfos.back();
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.pNext = &timelineSubmitInfo;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &swapchainData.acquireSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount =
        static_cast<uint32_t>(submitSignalSemaphores[0].size());
    submitInfo.pSignalSemaphores = submitSignalSemaphores[0].data();

    // TODO remove debug
    // uint64_t actualSem = 0;
    // vkGetSemaphoreCounterValue(contextVk.device().device, timelineSemaphore,
    //                            &actualSem);
    // syncLog([&]() {
    //   std::cout << "\033[33m" << "Actual Timeline:  " << actualSem <<
    //   "\033[0m"
    //             << std::endl;
    // });

    contextVk.swapBufferRelease();
  }
  sched->waitUntilAllTasksDone();
  contextVk.unsetSwapBufferCallbacks();

  vkDeviceWaitIdle(contextVk.device().device);
  vkDestroySemaphore(contextVk.device().device, timelineSemaphore, nullptr);
  discardEverything(contextVk, discardPool, &commandPool, perFrameData);
  pipelinePool.discardAllPipelines(discardPool, pipelineLayout);
  discardPool.discardShaderModule(graphicsInfo.preRasterization.vertexModule);
  discardPool.discardShaderModule(graphicsInfo.fragmentShader.fragmentModule);
  discardPool.discardPipelineLayout(pipelineLayout);
  std::cout << "\033[93m" << "[Render Thread] Preparing to deinit discard pool"
            << std::endl;
  vkDestroyDescriptorSetLayout(contextVk.device().device, uboSetLayout,
                               nullptr);
  vkDestroyDescriptorPool(contextVk.device().device, descriptorPool, nullptr);
  discardPool.deinit(contextVk.device());

  // cleanup residual
  vertexBuffer.freeImmediately(contextVk);
  indexBuffer.freeImmediately(contextVk);
  uniformBufferUBO.freeImmediately(contextVk);
}

int WINAPI wWinMain([[maybe_unused]] HINSTANCE hInstance,
                    [[maybe_unused]] HINSTANCE hPrevInstance,
                    [[maybe_unused]] LPWSTR lpCmdLine,
                    [[maybe_unused]] int nCmdShow) {
  // Use HeapSetInformation to specify that the process should
  // terminate if the heap manager detects an error in any heap used
  // by the process.
  // The return value is ignored, because we want to continue running in the
  // unlikely event that HeapSetInformation fails.
  HeapSetInformation(NULL, HeapEnableTerminationOnCorruption, NULL, 0);

  // GUI Console window
  // HKCU\Console\%%Startup\DelegationConsole

#ifdef AVK_DEBUG
  AllocConsole();
  FILE* fDummy;
  freopen_s(&fDummy, "CONOUT$", "w", stdout);
  freopen_s(&fDummy, "CONOUT$", "w", stderr);
  std::ios::sync_with_stdio();
  {
    HANDLE handleOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD consoleMode;
    GetConsoleMode(handleOut, &consoleMode);
    consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    consoleMode |= DISABLE_NEWLINE_AUTO_RETURN;
    SetConsoleMode(handleOut, consoleMode);
  }

#endif
  std::cout << "[Main Thread] Thread ID: " << GetCurrentThreadId() << std::endl;
  avk::ContextVkParams params;

  // Vulkan Context Scope. Note how the window outlives the vulkan context
  {
    avk::ContextVk contextVk{};
    std::unique_ptr<avk::WindowPayload> windowPayload =
        std::make_unique<avk::WindowPayload>();
    windowPayload->contextVk = &contextVk;
    // start Win32 message thread (dedicated)
    std::thread msgThread(messageThreadProc, windowPayload.get());
    while (!g_window.load()) {
      std::this_thread::yield();
    }

    // Initialize Vulkan
    std::cout << "Time to Initialize Vulkan" << std::endl;
    params.window = g_window.load();
    if (contextVk.initializeDrawingContext(params) !=
        avk::ContextResult::Success) {
      MessageBoxW(params.window, L"Error, Couldn't initialize Vulkan",
                  L"Fatal Error", MB_OK | MB_ICONEXCLAMATION);
      return 1;
    }

    // frame producer on its own thread (could be main thread in real app)
    constexpr size_t jobPoolSize = 1024;
    constexpr size_t fiberCount = 64;
    avk::MPMCQueue<avk::Job*> highQ(jobPoolSize);
    avk::MPMCQueue<avk::Job*> medQ(jobPoolSize);
    avk::MPMCQueue<avk::Job*> lowQ(jobPoolSize);

    unsigned const hw = std::thread::hardware_concurrency();
    unsigned const workerCount = hw > 3 ? hw - 3 : 1;
    avk::Scheduler sched(fiberCount, &highQ, &medQ, &lowQ, workerCount);
    sched.start();
    // const int chunksPerFrame = 8;
    FrameProducerPayload producerPayload{};
    producerPayload.context = &contextVk;
    producerPayload.hMainWindow = params.window;
    producerPayload.jobBufferSize = 256;
    std::thread producer(frameProducer, &sched, &producerPayload);

    producer.join();
    sched.waitUntilAllTasksDone();

    // TODO sleep with condition variable
    while (!g_quit.load()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(16));
    }

    sched.shutdown();

    if (msgThread.joinable()) msgThread.join();
  }

  // TODO remove
  while (true) {
    std::this_thread::sleep_for(std::chrono::milliseconds(64));
  }

  return 0;
}
