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

struct PerFrameData {
  VkImage resolveColorImage;
  VkImage depthImage;
  VmaAllocation resolveColorAlloc;
  VmaAllocation depthImageAlloc;
  VkImageView resolveColorImageView;
  VkImageView depthImageView;
  VkCommandBuffer commandBuffer;  // TODO better (using timeline semaphores)
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
  float position[2];
  float color[3];
};

void onSwapchainRecreationStarted(avk::ContextVk const& contextVk) {
  while (contextVk.isResizing.load(std::memory_order_relaxed)) {
    // wait
  }
}

void onSwapBuffer(avk::ContextVk const& contextVk,
                  std::vector<VkSubmitInfo>& submitInfos,
                  avk::SwapchainDataVk const& swapchainData) {
  if (submitInfos.size() > 0) {
    avk::vkCheck(vkQueueSubmit(contextVk.device().graphicsComputeQueue,
                               static_cast<uint32_t>(submitInfos.size()),
                               submitInfos.data(),
                               swapchainData.submissionFence));
  }
}

void onAcquire([[maybe_unused]] avk::ContextVk const& contextVk) {}

void discardEverything(avk::ContextVk const& contextVk,
                       avk::DiscardPoolVk& discardPool,
                       VkCommandPool* commandPool,
                       std::vector<PerFrameData>& perFrameData) {
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

void onSwapchainRecreation(avk::ContextVk const& contextVk,
                           avk::DiscardPoolVk& discardPool,
                           VkCommandPool commandPool,
                           [[maybe_unused]] VkImage const* images,
                           uint32_t numImages, VkFormat format,
                           VkExtent2D imageExtent, VkFormat depthFormat,
                           std::vector<PerFrameData>& perFrameData) {
  // TODO add timeline
  // 1. add all resources in the vector into the discard pool
  std::vector<VkCommandBuffer> commandBuffers;
  commandBuffers.reserve(64);
  for (auto const& data : perFrameData) {
    discardPool.discardImageView(data.depthImageView);
    discardPool.discardImageView(data.resolveColorImageView);
    discardPool.discardImage(data.depthImage, data.depthImageAlloc);
    discardPool.discardImage(data.resolveColorImage, data.resolveColorAlloc);
    // TODO handle discarding command buffers better
    commandBuffers.push_back(data.commandBuffer);
  }
  // TODO handle synchronization? how? TODO remove this
  vkDeviceWaitIdle(contextVk.device().device);
  discardPool.destroyDiscardedResources(contextVk.device());

  perFrameData.clear();
  perFrameData.resize(numImages);

  // 1.1 create new command buffers/reuse old ones
  // TODO better
  if (commandBuffers.size() < perFrameData.size()) {
    // create missing command buffers
    size_t const offset = commandBuffers.size();
    size_t const count = perFrameData.size() - commandBuffers.size();
    commandBuffers.resize(count);
    avk::allocPrimaryCommandBuffers(contextVk, commandPool, count,
                                    commandBuffers.data() + offset);
    // assign the newly created
    for (size_t i = 0; i < perFrameData.size(); ++i) {
      perFrameData[i].commandBuffer = commandBuffers[i];
    }
  } else if (commandBuffers.size() > perFrameData.size()) {
    // handle excess command buffers
    size_t const offset = perFrameData.size();
    size_t const count = commandBuffers.size() - perFrameData.size();
    vkFreeCommandBuffers(contextVk.device().device, commandPool, count,
                         commandBuffers.data() + offset);
    commandBuffers.resize(perFrameData.size());

    // reassign to perFrameData
    for (size_t i = 0; i < perFrameData.size(); ++i) {
      perFrameData[i].commandBuffer = commandBuffers[i];
    }
  }
  // 3) equal sizes — just reassign existing command buffers
  else {
    for (size_t i = 0; i < perFrameData.size(); ++i) {
      perFrameData[i].commandBuffer = commandBuffers[i];
    }
  }

  // 2. create new resources
  VkImageViewCreateInfo depthViewsCreateInfo{};
  depthViewsCreateInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  depthViewsCreateInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
  depthViewsCreateInfo.format = depthFormat;
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
  depthSpec.format = depthFormat;
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

  for (auto& data : perFrameData) {
    // 2.1 create depth image
    // TODO: this is a complete failure, crash the application
    if (!avk::createImage(contextVk, depthSpec, data.depthImage,
                          data.depthImageAlloc)) {
      data.depthImageView = VK_NULL_HANDLE;
    } else {
      depthViewsCreateInfo.image = data.depthImage;
      // create depth image views
      avk::vkCheck(vkCreateImageView(contextVk.device().device,
                                     &depthViewsCreateInfo, nullptr,
                                     &data.depthImageView));
    }

    // 3 color attachment images
    // TODO: this is a complete failure, crash the application
    if (!avk::createImage(contextVk, resolveColorSpec, data.resolveColorImage,
                          data.resolveColorAlloc)) {
      data.resolveColorImage = VK_NULL_HANDLE;
    } else {
      resolveColorViewCreateInfo.image = data.resolveColorImage;
      vkCreateImageView(contextVk.device().device, &resolveColorViewCreateInfo,
                        nullptr, &data.resolveColorImageView);
    }
  }
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
                           uint32_t imageBarrierCount) {
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
}

std::vector<uint32_t> uniqueElements(uint32_t const* elems, uint32_t count) {
  std::vector<uint32_t> vElems(count);
  std::copy_n(elems, count, vElems.begin());
  std::sort(vElems.begin(), vElems.end());
  auto it = std::unique(vElems.begin(), vElems.end());
  vElems.resize(std::distance(vElems.begin(), it));
  return vElems;
}

void frameProducer(avk::Scheduler* sched, FrameProducerPayload const* payload) {
  std::cout << "Started Render Thread" << std::endl;
  avk::ContextVk& contextVk = *payload->context;

  // per frame data to update every swapchain recreation
  std::vector<PerFrameData> perFrameData;
  perFrameData.reserve(64);

  VkFormat depthFormat =
      avk::basicDepthStencilFormat(contextVk.device().physicalDevice);

  avk::DiscardPoolVk discardPool;

  // TODO create TLS Command Pool
  VkCommandPool commandPool = avk::createCommandPool(
      contextVk, true, contextVk.device().queueIndices.family.graphicsCompute);

  std::vector<VkSubmitInfo> submitInfos;
  contextVk.setSwapBufferCallbacks(
      [&contextVk, &submitInfos](const avk::SwapchainDataVk* pData) {
        onSwapBuffer(contextVk, submitInfos, *pData);
      },
      [&contextVk]() { onAcquire(contextVk); },
      [&contextVk, &discardPool, &perFrameData, depthFormat, commandPool](
          VkImage const* images, uint32_t numImages, VkFormat format,
          VkExtent2D extent) {
        onSwapchainRecreation(contextVk, discardPool, commandPool, images,
                              numImages, format, extent, depthFormat,
                              perFrameData);
      },
      [&contextVk]() { onSwapchainRecreationStarted(contextVk); });
  // must do before pipeline creation as it populates the surface format
  // TODO decouple classes
  contextVk.recreateSwapchain(false);

  avk::VkPipelinePool pipelinePool;
  avk::GraphicsInfo graphicsInfo;

  VkPipelineLayout pipelineLayout = avk::createPipelineLayout(contextVk);

  fillTriangleGraphicsInfo(contextVk, graphicsInfo, pipelineLayout);

  VkPipeline pipeline = pipelinePool.getOrCreateGraphicsPipeline(
      contextVk, graphicsInfo, true, VK_NULL_HANDLE);

  // Create initial Vertex Buffer
  const std::vector<Vertex> vertices = {
      {{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},  // Vertex 1: Red
      {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},   // Vertex 2: Green
      {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}   // Vertex 3: Blue
  };
  std::vector<uint32_t> uniqueQueueFamilies =
      uniqueElements(contextVk.device().queueIndices.families,
                     avk::DeviceVk::QueueFamilyIndicesCount);
  assert(uniqueQueueFamilies.size() > 0);
  avk::BufferVk vertexBuffer;
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
  vertexBuffer.updateImmediately(vertices.data());

  // uint32_t jobIdx = 0;
  submitInfos.reserve(16);
  while (!g_quit.load()) {
    submitInfos.clear();

    contextVk.swapBufferAcquire();
    avk::SwapchainDataVk swapchainData = contextVk.getSwapchainData();
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
        VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
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

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    // TODO extent actual swapchain extent
    renderingInfo.renderArea = {{0, 0}, swapchainData.extent};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;  // TODO graphicsInfo?
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

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

    // issue draw call
    uint32_t const vertexCount = static_cast<uint32_t>(vertices.size());
    vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);

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

    VkPipelineStageFlags waitStage{VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT};
    submitInfos.clear();
    submitInfos.push_back({});

    VkSubmitInfo& submitInfo = submitInfos.back();
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &swapchainData.acquireSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &swapchainData.presentSemaphore;

    contextVk.swapBufferRelease();
  }
  sched->waitUntilAllTasksDone();

  vkDeviceWaitIdle(contextVk.device().device);
  discardEverything(contextVk, discardPool, &commandPool, perFrameData);
  pipelinePool.discardAllPipelines(discardPool, pipelineLayout);
  discardPool.discardShaderModule(graphicsInfo.preRasterization.vertexModule);
  discardPool.discardShaderModule(graphicsInfo.fragmentShader.fragmentModule);
  discardPool.discardPipelineLayout(pipelineLayout);
  std::cout << "\033[93m" << "[Render Thread] Preparing to deinit discard pool"
            << std::endl;
  discardPool.deinit(contextVk.device());

  // cleanup residual
  vertexBuffer.freeImmediately(contextVk);
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
