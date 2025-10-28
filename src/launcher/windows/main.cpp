#include <Windows.h>

// Windows related stuff
#include <Windowsx.h>  // GET_X_LPARAM
#include <basetsd.h>
#include <errhandlingapi.h>
#include <wingdi.h>
#include <winspool.h>
#include <winuser.h>

// ExtractIconExW
#include <shellapi.h>

#include <algorithm>
#include <atomic>
#include <climits>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>

#include "fiber/jobs.h"
#include "os/filesystem.h"
#include "render/context-vk.h"
#include "render/pipeline-vk.h"
#include "render/utils-vk.h"

struct PerFrameData {
  VkImage depthImage;
  VmaAllocation depthImageAlloc;
  VkImageView swapchainImageView;
  VkImageView depthImageView;
  VkCommandBuffer commandBuffer;  // TODO better (using timeline semaphores)
};

struct WindowPayload {
  avk::ContextVk* contextVk;
};

static LRESULT standardWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam,
                                  LPARAM lParam);

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

struct RenderJobTLS {
  VkCommandPool commandPool;
};

struct Payload {
  int value;
  int id;
};

struct Vertex {
  float position[2];
  float color[3];
};

void onSwapBuffer([[maybe_unused]] avk::ContextVk const& contextVk,
                  [[maybe_unused]] avk::SwapchainDataVk const& pData) {}

void onAcquire([[maybe_unused]] avk::ContextVk const& contextVk) {}

void onSwapchainRecreation(avk::ContextVk const& contextVk,
                           avk::DiscardPoolVk& discardPool,
                           VkCommandPool commandPool, VkImage const* images,
                           uint32_t numImages, VkFormat format,
                           VkExtent2D imageExtent, VkFormat depthFormat,
                           std::vector<PerFrameData>& perFrameData) {
  // TODO add timeline
  // 1. add all resources in the vector into the discard pool
  std::vector<VkCommandBuffer> commandBuffers;
  commandBuffers.reserve(64);
  for (auto const& data : perFrameData) {
    discardPool.discardImageView(data.swapchainImageView);
    discardPool.discardImageView(data.depthImageView);
    discardPool.discardImage(data.depthImage, data.depthImageAlloc);
    // TODO handle discarding command buffers better
    commandBuffers.push_back(data.commandBuffer);
  }

  perFrameData.clear();
  perFrameData.resize(numImages);

  // 1.1 create new command buffers/reuse old ones
  // TODO better
  if (commandBuffers.size() < perFrameData.size()) {
    // create missing command buffers
    size_t const offset = commandBuffers.size();
    size_t const count = perFrameData.size() - commandBuffers.size();
    commandBuffers.resize(perFrameData.size());
    avk::allocPrimaryCommandBuffers(contextVk, commandPool, count,
                                    commandBuffers.data() + offset);
  } else if (commandBuffers.size() > perFrameData.size()) {
    // handle excess command buffers
    size_t const offset = perFrameData.size();
    size_t const count = commandBuffers.size() - perFrameData.size();
    vkFreeCommandBuffers(contextVk.device().device, commandPool, count,
                         commandBuffers.data() + offset);
    commandBuffers.resize(perFrameData.size());
  }

  // 2. create new resources
  uint32_t index = 0;
  for (auto& data : perFrameData) {
    uint32_t const current = index++;

    // 2.1 create depth image
    avk::SingleImage2DSpecVk depthSpec{};
    depthSpec.imageTiling = VK_IMAGE_TILING_OPTIMAL;
    depthSpec.format = depthFormat;
    depthSpec.samples =
        VK_SAMPLE_COUNT_1_BIT;  // TODO can depth buffering use multisampling?
    depthSpec.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthSpec.width = imageExtent.width;
    depthSpec.height = imageExtent.height;
    if (!avk::createImage(contextVk, depthSpec, data.depthImage,
                          data.depthImageAlloc)) {
      data.depthImageView = VK_NULL_HANDLE;
    } else {
      // create depth image views
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = data.depthImage;
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
      createInfo.format = depthFormat;
      createInfo.subresourceRange.aspectMask =
          VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 1;
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      avk::vkCheck(vkCreateImageView(contextVk.device().device, &createInfo,
                                     nullptr, &data.depthImageView));
    }

    // 2.2 swapchain image views
    data.swapchainImageView = VK_NULL_HANDLE;
    VkImageViewCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    createInfo.image = images[current];
    createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    createInfo.format = format;
    createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    createInfo.subresourceRange.baseArrayLayer = 0;
    createInfo.subresourceRange.layerCount = 1;
    createInfo.subresourceRange.baseMipLevel = 0;
    createInfo.subresourceRange.levelCount = 1;
    avk::vkCheck(vkCreateImageView(contextVk.device().device, &createInfo,
                                   nullptr, &data.swapchainImageView));
  }
}

void physicsStep(void* userData, [[maybe_unused]] std::string const& name,
                 [[maybe_unused]] uint32_t threadIndex,
                 [[maybe_unused]] uint32_t fiberIndex) {
  syncLog([]() { std::cout << "Started Physics Job" << std::endl; });
  Payload* p = reinterpret_cast<Payload*>(userData);
  // simulate CPU work
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  p->value = p->value * 2 + 1;
}

void renderStep(void* userData, [[maybe_unused]] std::string const& name,
                [[maybe_unused]] uint32_t threadIndex,
                [[maybe_unused]] uint32_t fiberIndex) {
  [[maybe_unused]] Payload* p = reinterpret_cast<Payload*>(userData);
  syncLog([]() { std::cout << "Started Render Job" << std::endl; });
  // simulate command buffer recording cost
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
}

// Blender's window style:
//  DWORD style = parent_window ?
//                    WS_POPUPWINDOW | WS_CAPTION | WS_MAXIMIZEBOX |
//                    WS_MINIMIZEBOX | WS_SIZEBOX : WS_OVERLAPPEDWINDOW;
//
//  /* Forces owned windows onto taskbar and allows minimization. */
//  DWORD extended_style = parent_window ? WS_EX_APPWINDOW : 0;
//
//  if (dialog) {
//    /* When we are ready to make windows of this type:
//     * style = WS_POPUPWINDOW | WS_CAPTION
//     * extended_style = WS_EX_DLGMODALFRAME | WS_EX_TOPMOST
//     */
//  }

void messageThreadProc(WindowPayload* windowPayload) {
  syncLog([] { std::cout << "Started UI Thread" << std::endl; });

  // MessageBoxW(nullptr, L"Hello, World!", L"Aether-Vk", MB_OK);
  // return 0;

  // TODO image from somewhere, for now steal it from explorer.exe
  HICON hIconLarge = nullptr;
  HICON hIconSmall = nullptr;
  if (ExtractIconExW(L"explorer.exe", 0, &hIconLarge, &hIconSmall, 1) ==
      UINT_MAX) {
    printError();
  }

  // TODO More Cursors?
  HCURSOR standardCursor = LoadCursorW(nullptr, IDC_ARROW);
  if (!standardCursor) {
    printError();
  }

  // background color (DeleteObject(hbrBackground)) when you are finished
  HBRUSH hbrBackground = CreateSolidBrush(RGB(30, 30, 30));
  if (!hbrBackground) {
    printError();
  }

  // Menu definition (probably we will rmove it if we can't style the part
  // outside the client area)
  // static int constexpr ID_FILE_OPEN = 1;
  // HMENU hMenu = CreateMenu();  // DestroyMenu
  // if (!hMenu) {
  //   printError();
  // }
  // HMENU hFileSubMenu = CreatePopupMenu();  // DestroyPopupMenu
  // if (!hFileSubMenu) {
  //   printError();
  // }
  // if (!AppendMenuW(hFileSubMenu, MF_STRING, ID_FILE_OPEN, L"Open")) {
  //   printError();
  // }
  // if (!AppendMenuW(hMenu, MF_STRING | MF_POPUP, (UINT_PTR)hFileSubMenu,
  //                  L"File")) {
  //   printError();
  // }

  WNDCLASSEXW standardWindowSpec{};
  standardWindowSpec.cbSize = sizeof(WNDCLASSEXW);
  standardWindowSpec.style = 0;  // TODO class styles
  standardWindowSpec.lpfnWndProc = standardWindowProc;
  standardWindowSpec.cbClsExtra = 0;  // TODO see how extra bytes can be used
  standardWindowSpec.cbWndExtra = 0;  // TODO see how extra bytes can be used
  standardWindowSpec.hInstance = GetModuleHandleW(nullptr);
  standardWindowSpec.hIcon = hIconLarge;
  standardWindowSpec.hCursor = standardCursor;
  standardWindowSpec.hbrBackground = hbrBackground;
  standardWindowSpec.lpszMenuName = nullptr;  // HMENU passed explicitly
  standardWindowSpec.lpszClassName = L"Standard Window";
  standardWindowSpec.hIconSm = hIconSmall;

  // UnregisterClassEx() when you are finished
  ATOM standardWindowAtom = RegisterClassExW(&standardWindowSpec);
  if (!standardWindowAtom) {
    printError();
  }
  // WS_POPUP vs WS_EX_OVERLAPPED = no top bar, no default content, hence you
  // need to handle WS_PAINT, and size CW_USEDEFAULT doesn't work
  // WS_POPUP needs an explicit size
  // TODO WS_POPUP doesn't support HMENU! (either use WS_EX_OVERLAPPED or window
  // rendered on vulkan, see how it works on mac and linux)
  int winX = 100, winY = 100, winW = 1024, winH = 768;  // TODO better
  g_window.store(CreateWindowExW(0, L"Standard Window", L"Aether VK", WS_POPUP,
                                 winX, winY, winW, winH, nullptr,
                                 nullptr /*hMenu*/, nullptr, windowPayload));
  if (!g_window.load()) {
    printError();
    g_quit.store(true);
    return;
  }
  HWND hMainWindow = g_window.load();

  // TODO: Controller thread (handle messages) and display thread different,
  // hence call ShowWindowAsync with event synchronization on window creation
  // ShowWindow(hMainWindow, SW_SHOWDEFAULT);
  ShowWindowAsync(hMainWindow, SW_SHOW);

  // send the first WM_PAINT message to the window to fill the client area
  UpdateWindow(hMainWindow);

  MSG message{};
  BOOL getMessageRet = false;

  // TODO handle modeless (=non blocking) dialog boxes (Modal -> DialogBox,
  // Modeless -> CreateDialog)
  HWND hCurrentModelessDialog = nullptr;

  // TODO: Mate an accelerator table (IE Mapping between shortcuts and
  // actions, eg CTRL + S -> Save) (LoadAccelerators)
  HACCEL hAccel = nullptr;
  while (!g_quit.load()) {
    // NOTE: Use PeekMessage if you want to interrupt a lengthy sync operation
    // with PM_REMOVE post message in the thread's message queue
    getMessageRet = GetMessageW(&message, nullptr, 0, 0);
    if (getMessageRet == -1) {
      printError();
      break;
    }

    // modeless dialog messages have been already processed
    if (hCurrentModelessDialog != nullptr &&
        IsDialogMessageW(hCurrentModelessDialog, &message)) {
      continue;
    }
    // TODO accelerator message are handled by accelerator
    if (hAccel != nullptr &&
        TranslateAcceleratorW(hMainWindow, hAccel, &message)) {
      continue;
    }

    if (message.message == WM_QUIT) {
      g_quit.store(true);
      break;
    }

    // (TODO: Handle Dialog box and Translate Accelerators for menus)
    TranslateMessage(&message);
    // dispatch to window procedure
    DispatchMessageW(&message);
  }
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
    uint32_t srcIndex = VK_QUEUE_FAMILY_IGNORED,
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
  imageBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
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
                              avk::GraphicsInfo& graphicsInfo) {
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
}

void frameProducer(avk::Scheduler* sched, FrameProducerPayload const* payload) {
  std::cout << "Started Render Thread" << std::endl;
  auto* p = new Payload();
  avk::Job* phys = new avk::Job[payload->jobBufferSize]();
  avk::Job* render = new avk::Job[payload->jobBufferSize]();
  avk::ContextVk& contextVk = *payload->context;

  avk::DiscardPoolVk discardPool;
  avk::VkPipelinePool pipelinePool;
  avk::GraphicsInfo graphicsInfo;
  fillTriangleGraphicsInfo(contextVk, graphicsInfo);

  VkPipeline pipeline = pipelinePool.getOrCreateGraphicsPipeline(
      contextVk, graphicsInfo, true, VK_NULL_HANDLE);
  // TODO create TLS Command Pool
  VkCommandPool commandPool = avk::createCommandPool(
      contextVk, true, contextVk.device().queueIndices.family.graphicsCompute);

  // Create initial Vertex Buffer
  const std::vector<Vertex> vertices = {
      {{0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},  // Vertex 1: Red
      {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},   // Vertex 2: Green
      {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}   // Vertex 3: Blue
  };
  avk::BufferVk vertexBuffer;
  // TODO create 2 buffers, 1 device visible and not host visible, the other
  // host visible such that you can create a staging buffer
  if (!vertexBuffer.create(contextVk, sizeof(vertices[0]) * vertices.size(),
                           VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                           VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                           0, 0, 1.f, false, 4,
                           contextVk.device().queueIndices.families)) {
    std::cerr << "Couldn't allocate Vertex Buffer Host Visible" << std::endl;
  }
  vertexBuffer.updateImmediately(vertices.data());

  // per frame data to update every swapchain recreation
  std::vector<PerFrameData> perFrameData;
  perFrameData.reserve(64);

  VkFormat depthFormat =
      avk::basicDepthStencilFormat(contextVk.device().physicalDevice);

  contextVk.setSwapBufferCallbacks(
      [&contextVk](const avk::SwapchainDataVk* pData) {
        onSwapBuffer(contextVk, *pData);
      },
      [&contextVk]() { onAcquire(contextVk); },
      [&contextVk, &discardPool, &perFrameData, depthFormat, commandPool](
          VkImage const* images, uint32_t numImages, VkFormat format,
          VkExtent2D extent) {
        onSwapchainRecreation(contextVk, discardPool, commandPool, images,
                              numImages, format, extent, depthFormat,
                              perFrameData);
      });
  contextVk.recreateSwapchain(false);

  // uint32_t jobIdx = 0;
  std::vector<VkSubmitInfo> submitInfos;
  submitInfos.reserve(16);
  while (!g_quit.load()) {
    submitInfos.clear();
    // uint32_t const nextJobIdx = (jobIdx + 1) % payload->jobBufferSize;
    // uint32_t const prevJobIdx = jobIdx == 0 ? 0 : jobIdx - 1;
    // create payloads and submit jobs in a burst (simulates a frame)
    p->value = 100;
    p->id = 1000;

    contextVk.swapBufferAcquire();
    avk::SwapchainDataVk swapchainData = contextVk.getSwapchainData();
    auto const& frame = perFrameData[swapchainData.imageIndex];

    // AVK_JOB(&phys[jobIdx], physicsStep, p, avk::JobPriority::Medium,
    //         "PhysicsChunk");

    // TODO if necessary vkQueueWaitIdle on graphics Queue?
    // if you wait for the queue to be idle, you can delete immediately command
    // buffer, otherwise, discard it and delete it when you pass to another
    // timeline
    // TODO see better timeline semaphores

    // TODO multithreaded command buffer recording and multithreaded data
    // transfer AVK_JOB(&render[jobIdx], renderStep, p, avk::JobPriority::Low,
    //         "RenderPrep");

    VkCommandBuffer commandBuffer = frame.commandBuffer;

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;  // one time

    avk::vkCheck(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    // before rendering, transition current image to COLOR_ATTACHMENT_OPTIMAL
    VkImageMemoryBarrier2 imageBarriers[2];
    uint32_t const imageBarrierCount = 2;
    imageBarriers[0] =
        imageMemoryBarrier(swapchainData.image, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                           0 /*no need to wait previous operation */,
                           VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT);
    imageBarriers[1] =
        imageMemoryBarrier(frame.depthImage, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,
                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT);
    transitionImageLayout(commandBuffer, imageBarriers, imageBarrierCount);

    VkClearValue clearValue{{{0.01f, 0.01f, 0.033f, 1.0f}}};
    // rendering attachment info and begin rendering
    // TODO attachment info for depth/stencil if necessary
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = frame.swapchainImageView;
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
    vp.height = swapchainData.extent.width;
    vp.width = swapchainData.extent.height;
    vp.minDepth = 0.f;
    vp.maxDepth = 1.f;
    vkCmdSetViewport(commandBuffer, 0, 1, &vp);

    VkRect2D scissor{};
    scissor.extent.width = swapchainData.extent.width;
    scissor.extent.height = swapchainData.extent.height;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    // bind vertex buffer
    VkBuffer vBuf = vertexBuffer.buffer();
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vBuf, 0);

    // issue draw call
    uint32_t const vertexCount = static_cast<uint32_t>(vertices.size());
    vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);

    // complete rendering
    vkCmdEndRendering(commandBuffer);

    // after rendering, transition to PRESENT_SRC layout
    imageBarriers[0] = imageMemoryBarrier(
        swapchainData.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        0 /* no access */, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);
    transitionImageLayout(commandBuffer, imageBarriers, imageBarrierCount);

    // complete recording of the command buffer
    avk::vkCheck(vkEndCommandBuffer(commandBuffer));

    // TODO submit to queue
    VkPipelineStageFlags waitStage{VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT};
    submitInfos.push_back({});

    VkSubmitInfo submitInfo = submitInfos.back();
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &swapchainData.acquireSemaphore;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &swapchainData.presentSemaphore;

    avk::vkCheck(vkQueueSubmit(contextVk.device().graphicsComputeQueue,
                               static_cast<uint32_t>(submitInfos.size()),
                               submitInfos.data(),
                               swapchainData.submissionFence));

    // // render depends on physics
    // if (jobIdx != 0) {
    //   phys[jobIdx].addDepencency(&phys[prevJobIdx - 1]);
    //   // unless you recycle stuff from the previous frame, this is not
    //   necessary
    //   // render[jobIdx].addDepencency(&render[prevJobIdx]);
    // }
    // render[jobIdx].addDepencency(&phys[jobIdx]);

    // // submit bot (only the top of the DAG)
    // sched->safeSubmitTask(&phys[jobIdx]);
    // sched->safeSubmitTask(&render[jobIdx]);

    // if (nextJobIdx < jobIdx) {
    //   sched->waitFor(&render[nextJobIdx]);
    //   // we'll recycle the job objects in the next loop iteration, therefore
    //   you
    //   // need to be sure that the job mutex is not locked somewhere for any
    //   job!
    //   // WARNING: Assuming continuations are zeroed out, remaining
    //   dependencies
    //   // are at 0
    // }

    // TODO better
    // present ready might be false if user holds resize window and doesn't
    // release it Does it work like this though? TODO verify for now, don't
    // present images
    // bool expected = false;
    // if (g_presentReady.compare_exchange_weak(expected, true,
    //                                          std::memory_order_release)) {
    //   InvalidateRect(payload->hMainWindow, nullptr, false);
    // }
    contextVk.swapBufferRelease();
  }
  sched->waitUntilAllTasksDone();

  // cleanup residual
  vertexBuffer.freeImmediately(contextVk);
  delete[] phys;
  delete[] render;
}
static LRESULT standardWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam,
                                  LPARAM lParam) {
  avk::ContextVk* context = nullptr;

  if (uMsg == WM_CREATE) {
    // Animate window appearing: slide from left to right
    AnimateWindow(hWnd, 300, AW_SLIDE | AW_ACTIVATE | AW_HOR_POSITIVE);

    // get creation parameters and store their address (reference must live
    // beyond the window)
    CREATESTRUCT* createStruct = reinterpret_cast<CREATESTRUCT*>(lParam);
    context = reinterpret_cast<WindowPayload*>(createStruct->lpCreateParams)
                  ->contextVk;
    SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(context));

    return 0;
  }
  // retrieve context for all other messages
  context =
      reinterpret_cast<avk::ContextVk*>(GetWindowLongPtrW(hWnd, GWLP_USERDATA));

  switch (uMsg) {
    case WM_CLOSE:
      // Animate window disappearing: slide out to the right
      AnimateWindow(hWnd, 300, AW_SLIDE | AW_HIDE | AW_HOR_POSITIVE);
      DestroyWindow(hWnd);
      return 0;

    case WM_APP:  // custom message from worker threads
      // InvalidateRect(hWnd, nullptr, TRUE);
      return 0;

    // if you don't use WS_THICKFRAME, then you need to implement your own logic
    // for border detection
    // Petzold, CH7 "Hit-Test Message"
    // Returning HTLEFT, HTTOPRIGHT, etc., enables automatic resizing.
    // Returning HTCAPTION allows window dragging.
    // If you want to set custom cursors manually, handle WM_SETCURSOR after
    // checking WM_NCHITTEST.
    case WM_NCHITTEST: {
      const LONG border = 6;  // TODO better: width of resize area in pixel
      RECT winRect{};
      GetWindowRect(hWnd, &winRect);
      int const x = GET_X_LPARAM(lParam);
      int const y = GET_Y_LPARAM(lParam);
      bool const resizeWidth =
          (x >= winRect.left && x < winRect.left + border) ||
          (x < winRect.right && x >= winRect.right - border);
      bool const resizeHeight =
          (y >= winRect.top && y < winRect.top + border) ||
          (y < winRect.bottom && y >= winRect.bottom - border);
      if (resizeWidth && resizeHeight) {  // corners
        if (x < winRect.left + border && y < winRect.top + border)
          return HTTOPLEFT;
        if (x >= winRect.right - border && y < winRect.top + border)
          return HTTOPRIGHT;
        if (x < winRect.left + border && y >= winRect.bottom - border)
          return HTBOTTOMLEFT;
        else
          return HTBOTTOMRIGHT;
      } else if (resizeWidth) {
        return (x < winRect.left + border) ? HTLEFT : HTRIGHT;
      } else if (resizeHeight) {
        return (y < winRect.top + border) ? HTTOP : HTBOTTOM;
      }
      // dragging?
      return HTCAPTION;
    }

    // Basic paint function
    case WM_PAINT: {
      // unreliable
      // if (g_presentReady.load(std::memory_order_acquire)) {
      //   context->swapBufferRelease();
      //   g_presentReady.store(false, std::memory_order_release);
      // }
      return 0;
    }

    // TODO BETTER: ASSUMES BUTTON POSITION
    case WM_LBUTTONDOWN: {
      return 0;
    }

    case WM_SIZE: {
      // Handled by Vulkan
      // // generate manually a WM_PAINT
      // InvalidateRect(hWnd, nullptr, false);
      return 0;
    }

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      return DefWindowProcW(hWnd, uMsg, wParam, lParam);
  }
}

int WINAPI wWinMain([[maybe_unused]] HINSTANCE hInstance,
                    [[maybe_unused]] HINSTANCE hPrevInstance,
                    [[maybe_unused]] LPWSTR lpCmdLine,
                    [[maybe_unused]] int nCmdShow) {
// GUI Console window
#ifdef AVK_DEBUG
  AllocConsole();
  FILE* fDummy;
  freopen_s(&fDummy, "CONOUT$", "w", stdout);
  freopen_s(&fDummy, "CONOUT$", "w", stderr);
  std::ios::sync_with_stdio();
#endif

  avk::ContextVk contextVk{};
  std::unique_ptr<WindowPayload> windowPayload =
      std::make_unique<WindowPayload>();
  windowPayload->contextVk = &contextVk;
  // start Win32 message thread (dedicated)
  std::thread msgThread(messageThreadProc, windowPayload.get());
  while (!g_window.load()) {
    std::this_thread::yield();
  }

  // Initialize Vulkan
  std::cout << "Time to Initialize Vulkan" << std::endl;
  avk::ContextVkParams params;
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

  return 0;
}
