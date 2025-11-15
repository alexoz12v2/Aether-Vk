#include "avk-windows-application.h"

// system
#include <Windows.h>

// libraries
#include <array>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <vector>

// my stuff
#include "os/filesystem.h"
#include "render/experimental/avk-basic-graphics-info.h"
#include "render/vk/images/image-vk.h"
#include "render/vk/renderpasses-vk.h"
#include "render/vk/shader-vk.h"

namespace avk {

static std::vector<uint32_t> openSpirV(std::filesystem::path const& path) {
  std::ifstream file(path, std::ios::ate | std::ios::binary);
  if (!file) {
    return {};
  }

  size_t const fileSize = static_cast<size_t>(file.tellg());
  if (fileSize % 4 != 0) {
    return {};
  }

  std::vector<uint32_t> shaderCode(fileSize / 4);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(shaderCode.data()), fileSize);

  return shaderCode;
}

void WindowsApplication::RTdoOnSurfaceLost() {
  showErrorScreenAndExit(
      "VkSurfaceKHR lost: How did the"
      " primary HWND get destroyed?");
}

void WindowsApplication::RTdoEarlySurfaceRegained() {
  // never called
  showErrorScreenAndExit(
      "VkSurfaceKHR lost: How did the"
      " primary HWND get destroyed?");
}

void WindowsApplication::RTdoLateSurfaceRegained() {
  // never called
  showErrorScreenAndExit(
      "VkSurfaceKHR lost: How did the"
      " primary HWND get destroyed?");
}

void WindowsApplication::RTdoOnDeviceLost() {
  showErrorScreenAndExit("Desktop should never lose device");
}

void WindowsApplication::RTdoOnDeviceRegained() {
  showErrorScreenAndExit("Desktop should never lose device");
}

static VkBuffer s_vBuffer = VK_NULL_HANDLE;
static VmaAllocation s_vAllocation = VK_NULL_HANDLE;

WindowsApplication::~WindowsApplication() noexcept {
  LOGI << "[WindowsApplication] Detructor Running" << std::endl;
  if (windowInitializedOnce()) {
    cleanupVulkanResources();
    destroyConstantVulkanResources();
  }
}

void WindowsApplication::createConstantVulkanResources() AVK_NO_CFI {
  std::array<glm::vec3, 3> triangle{
      glm::vec3{-.5, .5, .5},  // left below
      {.5, .5, .5},            // right below
      {0, -.5, .5},            // top center
  };
  VkBufferCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  createInfo.size = nextMultipleOf<16>(sizeof(triangle));
  createInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
  VmaAllocationCreateInfo allocInfo{};
  allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
  allocInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                    VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT;
  allocInfo.preferredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

  VK_CHECK(vmaCreateBuffer(vmaAllocator(), &createInfo, &allocInfo, &s_vBuffer,
                           &s_vAllocation, nullptr));
  VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(),
                                     glm::value_ptr(triangle[0]), s_vAllocation,
                                     0, sizeof(triangle)));
  // shaders
  std::filesystem::path const exeDir = getExecutablePath().parent_path();
  auto const vertCode = openSpirV(exeDir / "cube-buffers.vert.spv");
  auto const fragCode = openSpirV(exeDir / "cube-buffers.frag.spv");
  VkShaderModule modules[2]{
      vk::createShaderModule(vkDevice(), vertCode.data(), vertCode.size() << 2),
      vk::createShaderModule(vkDevice(), fragCode.data(),
                             fragCode.size() << 2)};
  // this doesn't fill in the render pass
  m_graphicsInfo = experimental::basicGraphicsInfo(
      vk::createPipelineLayout(vkDevice()), modules,
      vk::basicDepthStencilFormat(vkPhysicalDeviceHandle()));
}

void WindowsApplication::destroyConstantVulkanResources() AVK_NO_CFI {
  experimental::discardGraphicsInfo(vkDiscardPool(), timeline(),
                                    m_graphicsInfo);
}

void WindowsApplication::cleanupVulkanResources() AVK_NO_CFI {
  using namespace avk::literals;
  // render pass
  vkDiscardPool()->discardRenderPass(m_graphicsInfo.renderPass, timeline());
  m_graphicsInfo.renderPass = VK_NULL_HANDLE;
  // graphics pipeline
  vkPipelines()->discardAllPipelines(vkDiscardPool(),
                                     m_graphicsInfo.pipelineLayout, timeline());
  m_graphicsPipeline = VK_NULL_HANDLE;
  // depth image
  vkDiscardPool()->discardImageView(m_depthView, timeline());
  m_depthView = VK_NULL_HANDLE;
  imageManager()->discardById(vkDiscardPool(), "depth"_hash, timeline());
  // framebuffer
  for (const VkFramebuffer framebuffer : m_framebuffers) {
    vkDiscardPool()->discardFramebuffer(framebuffer, timeline());
  }
  // command buffer bookkeeping
  m_commandBufferIds.clear();
}

void WindowsApplication::createVulkanResources() AVK_NO_CFI {
  using namespace avk::literals;
  auto const* const vkDevApi = vkDevTable();
  VkDevice const dev = vkDeviceHandle();
  // renderPass
  VkFormat const depthFmt =
      vk::basicDepthStencilFormat(vkPhysicalDeviceHandle());
  m_graphicsInfo.renderPass =
      vk::basicRenderPass(vkDevice(), vkSwapchain()->surfaceFormat().format,
                          depthFmt)
          .get();
  // graphics pipeline
  m_graphicsPipeline = vkPipelines()->getOrCreateGraphicsPipeline(
      m_graphicsInfo, true, VK_NULL_HANDLE);
  // depth image
  int32_t res = imageManager()->createTransientAttachment(
      "depth"_hash, vkSwapchain()->extent(), depthFmt,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_SAMPLE_COUNT_1_BIT);
  if (res) {
    showErrorScreenAndExit("Couldn't create Depth image");
  }
  VkImage depthImage = VK_NULL_HANDLE;
  VmaAllocation depthAlloc = VK_NULL_HANDLE;
  res = imageManager()->get("depth"_hash, depthImage, depthAlloc);
  if (!res) {
    showErrorScreenAndExit("Couldn't get Depth image from manager");
  }
  m_depthView = vk::depthStencilImageView(
                    vkDevice(), depthImage,
                    vk::basicDepthStencilFormat(vkPhysicalDeviceHandle()))
                    .get();
  // framebuffers
  m_framebuffers.resize(vkSwapchain()->imageCount());
  VkFramebufferCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
  createInfo.renderPass = m_graphicsInfo.renderPass;
  createInfo.attachmentCount = 2;
  createInfo.width = vkSwapchain()->extent().width;
  createInfo.height = vkSwapchain()->extent().height;
  createInfo.layers = 1;
  VkImageView attachments[2]{};
  createInfo.pAttachments = attachments;
  attachments[1] = m_depthView;
  uint32_t index = 0;
  for (VkFramebuffer& framebuffer : m_framebuffers) {
    uint32_t const i = index++;
    attachments[0] = vkSwapchain()->imageViewAt(i);
    VK_CHECK(
        vkDevApi->vkCreateFramebuffer(dev, &createInfo, nullptr, &framebuffer));
  }

  // bookkeeping for command buffers: 1 ID per Frame in Flight
  m_commandBufferIds.resize(vkSwapchain()->frameCount());
  // note: SBO allocated Strings
  for (size_t i = 0; i < m_commandBufferIds.size(); ++i) {
    m_commandBufferIds[i] = fnv1aHash("Prim" + std::to_string(i));
  }

  // projection matrix (resolution dependent)
  float const fovy = glm::radians(90.f);  // TODO vary portrait vs landscape
  float const aspect = static_cast<float>(vkSwapchain()->extent().width) /
                       vkSwapchain()->extent().height;
  m_camera.proj = glm::perspective(fovy, aspect, 0.0001f, 100.f);
}

void WindowsApplication::doOnSaveState() {}

void WindowsApplication::doOnRestoreState() {}

void WindowsApplication::RTdoOnWindowInit() AVK_NO_CFI {
#define PREFIX "[WindowsApplication::onWindowInit] "
  createConstantVulkanResources();
  LOGI << PREFIX "Constant Resources Created" << std::endl;
  createVulkanResources();
  LOGI << PREFIX "Resources Created" << std::endl;
#undef PREFIX
}

VkResult WindowsApplication::RTdoOnRender(
    [[maybe_unused]] const vk::utils::SwapchainData& swapchainData) AVK_NO_CFI {
  using namespace avk::literals;
  auto const* const vkDevApi = vkDevTable();

  // acquire a command buffer
  VkCommandBuffer const cmd = vkCommandPools()->allocatePrimary(
      m_commandBufferIds[vkSwapchain()->frameIndex()]);
  // begin command buffer
  VkClearValue clear[2]{};
  clear[1].depthStencil.depth = 1.0f;
  clear[1].depthStencil.stencil = 0.0f;
  VkRect2D rect{};
  rect.extent = vkSwapchain()->extent();
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkDevApi->vkBeginCommandBuffer(cmd, &beginInfo));
  // begin render pass (transition to optimal layout)
  VkRenderPassBeginInfo renderBegin{};
  renderBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderBegin.renderPass = m_graphicsInfo.renderPass;
  renderBegin.clearValueCount = 2;  // always match number of attachments
  renderBegin.pClearValues = clear;
  renderBegin.renderArea = rect;
  renderBegin.framebuffer = m_framebuffers[vkSwapchain()->imageIndex()];

  VkSubpassBeginInfoKHR subpassBegin{};
  subpassBegin.sType = VK_STRUCTURE_TYPE_SUBPASS_BEGIN_INFO_KHR;

  vkDevApi->vkCmdBeginRenderPass2KHR(cmd, &renderBegin, &subpassBegin);
  // bind pipeline and vertex buffer
  vkDevApi->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              m_graphicsPipeline);
  // bind vertex and index buffer
  VkDeviceSize offset = 0;
  vkDevApi->vkCmdBindVertexBuffers(cmd, 0, 1, &s_vBuffer, &offset);

  // set scissor and viewport
  vkDevApi->vkCmdSetScissor(cmd, 0, 1, &rect);
  VkViewport viewport{};
  viewport.width = rect.extent.width;
  viewport.height = rect.extent.height;
  viewport.maxDepth = 1.f;
  vkDevApi->vkCmdSetViewport(cmd, 0, 1, &viewport);

  vkDevApi->vkCmdDraw(cmd, 3, 1, 0, 0);

  // end render pass (transition to present layout)
  VkSubpassEndInfoKHR subEnd{};
  subEnd.sType = VK_STRUCTURE_TYPE_SUBPASS_END_INFO_KHR;
  vkDevApi->vkCmdEndRenderPass2KHR(cmd, &subEnd);
  VK_CHECK(vkDevApi->vkEndCommandBuffer(cmd));
  // queue submit command buffer
  VkSemaphore submitSignalSems[2]{swapchainData.presentSemaphore,
                                  vkDiscardPool()->timelineSemaphore()};
  uint64_t values[2]{0, timeline() + 1};
  VkTimelineSemaphoreSubmitInfoKHR timelineSubmit{};
  timelineSubmit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR;
  timelineSubmit.signalSemaphoreValueCount = 2;
  timelineSubmit.pSignalSemaphoreValues = values;

  VkPipelineStageFlags waitPresentationStage =
      VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.pNext = &timelineSubmit;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  submitInfo.signalSemaphoreCount = 2;
  submitInfo.pSignalSemaphores = submitSignalSems;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &swapchainData.acquireSemaphore;
  // same as wait semaphores
  submitInfo.pWaitDstStageMask = &waitPresentationStage;
  return vkDevApi->vkQueueSubmit(vkDevice()->queue(), 1, &submitInfo,
                                 swapchainData.submissionFence);
}

void WindowsApplication::RTdoOnResize() {
  LOGI << "[WindowsApplication] Resize callback on derived class" << std::endl;
  cleanupVulkanResources();
  createVulkanResources();
}

vk::SurfaceSpec WindowsApplication::doSurfaceSpec() {
  vk::SurfaceSpec spec{};
  spec.instance = GetModuleHandleW(nullptr);
  spec.window = PrimaryWindow;
  return spec;
}

}  // namespace avk