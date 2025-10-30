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

// inline static uint32_t constexpr style = WS_POPUP |  WS_THICKFRAME |
//                                          WS_MINIMIZEBOX | WS_MAXIMIZEBOX |
//                                          WS_SYSMENU | WS_CAPTION;
inline static uint32_t constexpr style = WS_OVERLAPPEDWINDOW;

struct WindowPayload {
  VkExtent2D lastClientExtent;
  avk::ContextVk* contextVk;
  WINDOWPLACEMENT windowedPlacement;
  bool isFullscreen;
  std::atomic<bool> framebufferResized;
  bool deferResizeHandling;
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

static std::atomic<bool> g_isResizing = false;

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

void onSwapchainRecreation(avk::ContextVk const& contextVk,
                           avk::DiscardPoolVk& discardPool,
                           VkCommandPool commandPool, VkImage const* images,
                           uint32_t numImages, VkFormat format,
                           VkExtent2D imageExtent, VkFormat depthFormat,
                           std::vector<PerFrameData>& perFrameData) {
  while (g_isResizing.load(std::memory_order_relaxed)) {
    // wait
  }
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
//  WS_POPUPWINDOW | WS_CAPTION | WS_MAXIMIZEBOX | WS_MINIMIZEBOX | WS_SIZEBOX
//  : WS_OVERLAPPEDWINDOW;
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

  // Important: Set DPI Awareness (alternative TODO -> In manifest)
  // Windows 10+ recommended:
  SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
  // Fallback (older Windows 8.1+):
  // HRESULT hr = SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);

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

  // use caption and thickframe to get normal window behaviour, but suppress
  // drawing of non client area with VM_NCCALCSIZE
  uint32_t extendedStyle = WS_EX_APPWINDOW;
  int winX = 100, winY = 100, winW = 1024, winH = 768;  // TODO better
  assert(windowPayload);
  g_window.store(CreateWindowExW(
      extendedStyle, L"Standard Window", L"Aether VK", style, winX, winY, winW,
      winH, nullptr, nullptr /*hMenu*/, nullptr, windowPayload));
  SetWindowPos(g_window.load(), NULL, 0, 0, 0, 0,
               SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE);
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
  auto* p = new Payload();
  avk::Job* phys = new avk::Job[payload->jobBufferSize]();
  avk::Job* render = new avk::Job[payload->jobBufferSize]();
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
      });
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
    // uint32_t const nextJobIdx = (jobIdx + 1) % payload->jobBufferSize;
    // uint32_t const prevJobIdx = jobIdx == 0 ? 0 : jobIdx - 1;
    // create payloads and submit jobs in a burst (simulates a frame)
    p->value = 100;
    p->id = 1000;

    contextVk.swapBufferAcquire();
    avk::SwapchainDataVk swapchainData = contextVk.getSwapchainData();
    auto const& frame = perFrameData[swapchainData.imageIndex];
    // vkResetFences(contextVk.device().device, 1,
    // &swapchainData.submissionFence);

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
    assert(commandBuffer && "Invalid command buffer from frame data");

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

    VkClearValue clearValue{{{0.01f, 0.01f, 0.033f, 1.0f}}};
    // rendering attachment info and begin rendering
    // TODO attachment info for depth/stencil if necessary
    // TODO populate resolve fields for multisampling.
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
        swapchainData.image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        0 /* no access */, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, VK_IMAGE_ASPECT_COLOR_BIT);
    transitionImageLayout(commandBuffer, imageBarriers, imageBarrierCount);

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

  // cleanup residual
  vertexBuffer.freeImmediately(contextVk);
  delete[] phys;
  delete[] render;
}

static LRESULT standardWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam,
                                  LPARAM lParam) {
  DWORD windowedStyle = style;

  // handle messages that can arrive before WM_CREATE
  if (uMsg == WM_NCCREATE) {
    CREATESTRUCTW* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
    if (cs && cs->lpCreateParams) {
      WindowPayload* payload =
          reinterpret_cast<WindowPayload*>(cs->lpCreateParams);
      SetWindowLongPtrW(hWnd, GWLP_USERDATA,
                        reinterpret_cast<LONG_PTR>(payload));
      // initialize windowedPlacement etc. here if you want
      payload->isFullscreen = false;
      payload->windowedPlacement = {};
      payload->windowedPlacement.length = sizeof(payload->windowedPlacement);
      payload->framebufferResized = false;
      payload->lastClientExtent = {};
    }
    return DefWindowProcW(hWnd, uMsg, wParam, lParam);
  }
  // retrieve context for all other messages
  // avk::ContextVk& context = *payload->contextVk;

  constexpr UINT WM_APP_RECREATE_SWAPCHAIN = WM_APP + 1;
  constexpr UINT TIMER_DEFER_SIZECHECK = 42;

  switch (uMsg) {
    case WM_CLOSE: {
      // Animate window disappearing: slide out to the right
      AnimateWindow(hWnd, 300, AW_SLIDE | AW_HIDE | AW_HOR_POSITIVE);
      DestroyWindow(hWnd);
      return 0;
    }

    case WM_APP_RECREATE_SWAPCHAIN: {
      // InvalidateRect(hWnd, nullptr, TRUE);
      return 0;
    }

      // --- Disable double-click maximize on the non-client area
    case WM_NCLBUTTONDBLCLK:
      // Do nothing (prevents default double-click maximize).
      // Must return 0 to indicate we handled it.
      return 0;

    // --- Prevent Windows from drawing the non-client area (titlebar) when we
    // hide it
    case WM_NCPAINT:
      // We draw no non-client area (we already made client full-window in
      // WM_NCCALCSIZE). Returning 0 prevents the default caption rendering that
      // sometimes reappears.
      return 0;

    case WM_NCACTIVATE:
      // Prevent Windows from toggling non-client active visuals. Returning 0
      // suppresses default painting. But preserve return semantics: If wParam
      // == TRUE and we want to keep it active, return 1 to indicate handled;
      // returning 0 prevents the default caption draw which we want hidden.
      return 0;

    // DPI awareness code
    case WM_WINDOWPOSCHANGED: {
      WINDOWPOS* wp = reinterpret_cast<WINDOWPOS*>(lParam);
      WindowPayload* payload = reinterpret_cast<WindowPayload*>(
          GetWindowLongPtrW(hWnd, GWLP_USERDATA));
      if (!payload) break;

      // Ignore changes that don't affect size
      if (wp->flags & SWP_NOSIZE) break;

      // When going fullscreen or snapping, let Windows settle
      bool isFrameChange = (wp->flags & SWP_FRAMECHANGED);
      bool isDeferNeeded = payload->isFullscreen || isFrameChange;

      if (isDeferNeeded) {
        // Defer size check to next frame so GetClientRect matches DWM area
        KillTimer(hWnd, TIMER_DEFER_SIZECHECK);
        SetTimer(hWnd, TIMER_DEFER_SIZECHECK, 50, nullptr);
        return 0;
      }

      RECT rc;
      GetClientRect(hWnd, &rc);
      uint32_t w = rc.right - rc.left;
      uint32_t h = rc.bottom - rc.top;

      if (w && h &&
          (w != payload->lastClientExtent.width ||
           h != payload->lastClientExtent.height)) {
        payload->lastClientExtent = {w, h};
        payload->framebufferResized = true;
        PostMessageW(hWnd, WM_APP_RECREATE_SWAPCHAIN, 0, 0);
      }
      return 0;
    }

    case WM_DPICHANGED: {
      // lParam points to suggested RECT in pixels for the new DPI.
      RECT* suggested = reinterpret_cast<RECT*>(lParam);
      WindowPayload* payload = reinterpret_cast<WindowPayload*>(
          GetWindowLongPtrW(hWnd, GWLP_USERDATA));
      if (payload && suggested) {
        uint32_t w = static_cast<uint32_t>(suggested->right - suggested->left);
        uint32_t h = static_cast<uint32_t>(suggested->bottom - suggested->top);
        payload->lastClientExtent.width = w;
        payload->lastClientExtent.height = h;
        payload->framebufferResized = true;

        // apply recommended pos/size so Windows doesn't scale the window
        // visually
        std::cout << "WM_DPICHANGED: " << payload->lastClientExtent.width << "x"
                  << payload->lastClientExtent.height << std::endl;
        SetWindowPos(hWnd, nullptr, suggested->left, suggested->top, w, h,
                     SWP_NOZORDER | SWP_NOACTIVATE);
        PostMessageW(hWnd, WM_APP_RECREATE_SWAPCHAIN, 0, 0);
      }
      return 0;
    }

    // before style changes
    case WM_STYLECHANGING: {
      WindowPayload* payload = reinterpret_cast<WindowPayload*>(
          GetWindowLongPtrW(hWnd, GWLP_USERDATA));
      if (payload) {
        payload->deferResizeHandling = true;
        return 0;
      }
      break;
    }

    // if you don't use WS_THICKFRAME, then you need to implement your own logic
    // for border detection
    // Petzold, CH7 "Hit-Test Message"
    // Returning HTLEFT, HTTOPRIGHT, etc., enables automatic resizing.
    // Returning HTCAPTION allows window dragging.
    // If you want to set custom cursors manually, handle WM_SETCURSOR after
    // checking WM_NCHITTEST.
    case WM_NCHITTEST: {
      WindowPayload* payload = reinterpret_cast<WindowPayload*>(
          GetWindowLongPtrW(hWnd, GWLP_USERDATA));
      if (payload && payload->isFullscreen) {
        return HTCLIENT;
      }

      // Use system metrics so OS and scaling agree
      const int border = GetSystemMetrics(SM_CXSIZEFRAME) +
                         GetSystemMetrics(SM_CXPADDEDBORDER);
      const int caption = GetSystemMetrics(SM_CYCAPTION);

      POINT pt = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
      RECT wr;
      GetWindowRect(hWnd, &wr);

      // Convert to window-local coords
      int x = pt.x - wr.left;
      int y = pt.y - wr.top;
      int w = wr.right - wr.left;
      int h = wr.bottom - wr.top;

      bool left = x < border;
      bool right = x >= (w - border);
      bool top = y < border;
      bool bottom = y >= (h - border);

      if (top && left) return HTTOPLEFT;
      if (top && right) return HTTOPRIGHT;
      if (bottom && left) return HTBOTTOMLEFT;
      if (bottom && right) return HTBOTTOMRIGHT;
      if (left) return HTLEFT;
      if (right) return HTRIGHT;
      if (top) return HTTOP;
      if (bottom) return HTBOTTOM;

      // Small strip at the top acts like a titlebar for dragging &
      // snap/Win+Arrows. Slightly enlarge with the border so snap works
      // reliably with invisible frame.
      if (y < caption + border) return HTCAPTION;

      return HTCLIENT;
    }

    case WM_TIMER: {
      WindowPayload* payload = reinterpret_cast<WindowPayload*>(
          GetWindowLongPtrW(hWnd, GWLP_USERDATA));
      if (wParam == TIMER_DEFER_SIZECHECK) {
        KillTimer(hWnd, TIMER_DEFER_SIZECHECK);

        RECT rc;
        GetClientRect(hWnd, &rc);
        uint32_t w = rc.right - rc.left;
        uint32_t h = rc.bottom - rc.top;

        if (payload && w && h &&
            (w != payload->lastClientExtent.width ||
             h != payload->lastClientExtent.height)) {
          payload->lastClientExtent = {w, h};
          std::cout << "DEFERRED EXTENT: " << w << "x" << h << std::endl;
          PostMessageW(hWnd, WM_APP_RECREATE_SWAPCHAIN, 0, 0);
        }
      }
      return 0;
    }

    case WM_SYSKEYDOWN:
    case WM_KEYDOWN: {
      WindowPayload* payload = reinterpret_cast<WindowPayload*>(
          GetWindowLongPtrW(hWnd, GWLP_USERDATA));
      // TODO remove debug
      ///////////////////////////////////////////////////////////////
      if (wParam == 0x57 /*W*/) {
        RECT rc{};
        GetClientRect(hWnd, &rc);
        std::cout << "GetClientRect: " << rc.right - rc.left << "x"
                  << rc.bottom - rc.top << std::endl;

        GetWindowRect(hWnd, &rc);
        std::cout << "GetWindowRect: " << rc.right - rc.left << "x"
                  << rc.bottom - rc.top << std::endl;

        HMONITOR hMonitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
        MONITORINFO mi = {};
        mi.cbSize = sizeof(mi);
        GetMonitorInfoW(hMonitor, &mi);
        std::cout << "GetMonitorInfoW: " << mi.rcWork.right - mi.rcWork.left
                  << "x" << mi.rcWork.bottom - mi.rcWork.top << std::endl;

        RECT dwmBounds;
        if (SUCCEEDED(DwmGetWindowAttribute(hWnd, DWMWA_EXTENDED_FRAME_BOUNDS,
                                            &dwmBounds, sizeof(dwmBounds)))) {
          uint32_t visibleW = dwmBounds.right - dwmBounds.left;
          uint32_t visibleH = dwmBounds.bottom - dwmBounds.top;
          std::cout << "DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS): "
                    << visibleW << "x" << visibleH << std::endl;
        }

        VkExtent2D ext = payload->contextVk->surfaceExtent();
        std::cout << "VkExtent Swapchain: " << ext.width << "x" << ext.height
                  << std::endl;
      }
      ///////////////////////////////////////////////////////////////

      if (payload && (wParam == VK_RETURN) && (GetKeyState(VK_MENU) & 0x8000)) {
        // ALT+ENTER detected → toggle fullscreen
        payload->isFullscreen = !payload->isFullscreen;

        // DWORD style = GetWindowLongW(hWnd, GWL_STYLE);
        // DWORD exStyle = GetWindowLongW(hWnd, GWL_EXSTYLE);

        if (payload->isFullscreen) {
          // Enter fullscreen
          GetWindowPlacement(hWnd, &payload->windowedPlacement);

          HMONITOR hMon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
          MONITORINFO mi = {};
          mi.cbSize = sizeof(mi);
          GetMonitorInfoW(hMon, &mi);
          RECT mon = mi.rcMonitor;
          uint32_t monW = static_cast<uint32_t>(mon.right - mon.left);
          uint32_t monH = static_cast<uint32_t>(mon.bottom - mon.top);

          SetWindowLongW(hWnd, GWL_STYLE, WS_POPUP | WS_VISIBLE);
          SetWindowPos(hWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                       mi.rcMonitor.right - mi.rcMonitor.left,
                       mi.rcMonitor.bottom - mi.rcMonitor.top,
                       SWP_FRAMECHANGED | SWP_NOOWNERZORDER | SWP_SHOWWINDOW);
          VkExtent2D override = {monW, monH};
          payload->contextVk->recreateSwapchain(
              true /*useHDR?*/,
              &override);  // or false as needed
        } else {
          // Restore normal window
          SetWindowLongW(hWnd, GWL_STYLE, windowedStyle);
          SetWindowLongW(hWnd, GWL_EXSTYLE, WS_EX_APPWINDOW | WS_EX_WINDOWEDGE);
          SetWindowPlacement(hWnd, &payload->windowedPlacement);
          SetWindowPos(hWnd, nullptr, 0, 0, 0, 0,
                       SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER |
                           SWP_NOOWNERZORDER | SWP_FRAMECHANGED |
                           SWP_SHOWWINDOW);
          RedrawWindow(
              hWnd, nullptr, nullptr,
              RDW_INVALIDATE | RDW_UPDATENOW | RDW_FRAME | RDW_ALLCHILDREN);
        }

        // Duplicate Swapchain recreation incorrect?
        // RECT rect;
        // GetClientRect(hWnd, &rect);
        // SendMessage(hWnd, WM_SIZE, SIZE_RESTORED,
        //             MAKELPARAM(rect.right, rect.bottom));
        PostMessageW(hWnd, WM_APP_RECREATE_SWAPCHAIN, 0, 0);

        return 0;
      }

      // You can also support other shortcuts here:
      // e.g. ALT+F4 → close (already handled by WM_CLOSE)
      // CTRL+R → refresh / resize logic
      return 0;
    }

    case WM_SIZE: {
      // Handled by Vulkan
      // // generate manually a WM_PAINT
      // InvalidateRect(hWnd, nullptr, false);
      if (wParam == SIZE_MINIMIZED) {
        // TODO handled in loop with IsIconic()
        std::cout << "WM_SIZE: MINIMIEZE" << std::endl;
      } else {
        // TODO remove: check if aero snap
        // pulls different extents after some time
        // std::cout << "WM_SIZE: posting timer" << std::endl;
        // KillTimer(hWnd, TIMER_DEFER_SIZECHECK);
        // SetTimer(hWnd, TIMER_DEFER_SIZECHECK, 50,
        //          nullptr);  // 50–100 ms works well

        // WindowPayload* payload = reinterpret_cast<WindowPayload*>(
        //     GetWindowLongPtrW(hWnd, GWLP_USERDATA));
        // if (payload) {
        //   payload->framebufferResized = true;
        //   PostMessageW(hWnd, WM_APP_RECREATE_SWAPCHAIN, 0, 0);
        // }
      }
      return 0;
    }

    case WM_ENTERSIZEMOVE: {
      g_isResizing.store(true, std::memory_order_relaxed);
      return 0;
    }

    case WM_EXITSIZEMOVE: {
      g_isResizing.store(false, std::memory_order_relaxed);
      PostMessageW(hWnd, WM_APP_RECREATE_SWAPCHAIN, 0, 0);
      return 0;
    }

    case WM_NCCALCSIZE: {
      if (wParam) {
        // Remove the standard frame — make the client area full window
        NCCALCSIZE_PARAMS* sz = (NCCALCSIZE_PARAMS*)lParam;
        sz->rgrc[0] = sz->rgrc[1];
        return 0;
      }
      break;
    }

    // since we removed manually part the window (outside the client area),
    // the window minemsions query should be properly handled manually
    case WM_GETMINMAXINFO: {
      WindowPayload* payload = reinterpret_cast<WindowPayload*>(
          GetWindowLongPtrW(hWnd, GWLP_USERDATA));
      MINMAXINFO* mmi = (MINMAXINFO*)lParam;
      HMONITOR hMon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
      MONITORINFO mi = {};
      mi.cbSize = sizeof(mi);
      if (GetMonitorInfoW(hMon, &mi)) {
        if (payload && payload->isFullscreen) {
          // Use the *full monitor area*, not work area
          RECT monitor = mi.rcMonitor;
          mmi->ptMaxPosition.x = monitor.left;
          mmi->ptMaxPosition.y = monitor.top;
          mmi->ptMaxSize.x = monitor.right - monitor.left;
          mmi->ptMaxSize.y = monitor.bottom - monitor.top;
        } else {
          // Normal behavior (respect taskbar)
          RECT work = mi.rcWork;
          RECT monitor = mi.rcMonitor;
          mmi->ptMaxPosition.x = work.left - monitor.left;
          mmi->ptMaxPosition.y = work.top - monitor.top;
          mmi->ptMaxSize.x = work.right - work.left;
          mmi->ptMaxSize.y = work.bottom - work.top;
          mmi->ptMinTrackSize.x = 200;
          mmi->ptMinTrackSize.y = 150;
        }
      }

      return 0;
    }

    case WM_SYSCOMMAND: {
      // Allow system move/resize from keyboard (Alt+Space, etc.)
      return DefWindowProcW(hWnd, uMsg, wParam, lParam);
    }

    case WM_DESTROY: {
      PostQuitMessage(0);
      return 0;
    }
  }
  return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

// Guide to activate Virtual Terminal processing: (Windows 10 and above)
//  HANDLE handleOut = GetStdHandle(STD_OUTPUT_HANDLE);
//  DWORD consoleMode;
//  GetConsoleMode( handleOut , &consoleMode);
//  consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
//  consoleMode |= DISABLE_NEWLINE_AUTO_RETURN;
//  SetConsoleMode( handleOut , consoleMode );

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
  {
    HANDLE handleOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD consoleMode;
    GetConsoleMode(handleOut, &consoleMode);
    consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    consoleMode |= DISABLE_NEWLINE_AUTO_RETURN;
    SetConsoleMode(handleOut, consoleMode);
  }

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
