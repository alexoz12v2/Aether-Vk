#include "avk-android-app.h"

// android stuff
#include <android/asset_manager_jni.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>

// library stuff
#include <glm/glm.hpp>

// our stuff
#include "os/avk-log.h"
#include "render/experimental/avk-basic-graphics-info.h"
#include "render/vk/buffers/buffer-vk.h"
#include "render/vk/images/image-vk.h"
#include "render/vk/renderpasses-vk.h"
#include "render/vk/shader-vk.h"

// TODO elsewhere
static std::vector<uint32_t> spirvFromAsset(AAssetManager* assetMgr,
                                            char const* path) {
  AAsset* asset = AAssetManager_open(assetMgr, path, AASSET_MODE_STREAMING);
  if (!asset) return {};
  off_t size = AAsset_getLength(asset);
  std::vector<uint32_t> code(size >> 2);
  AAsset_read(asset, code.data(), size);
  AAsset_close(asset);
  return code;
}

namespace avk {

AndroidApp::AndroidApp(android_app* app) : m_app(app) {
  // attach current thread to JNI to get java objects (`jobject`)
  app->activity->vm->AttachCurrentThread(&m_jniEnv, nullptr);
}

AndroidApp::~AndroidApp() noexcept {
  LOGI << "[AndroidApp] Destructor Running ..." << std::endl;
  if (windowInitializedOnce()) {
    // resources
    cleanupVulkanResources();
    destroyConstantVulkanResources();
  }
  if (m_jniEnv) {
    m_app->activity->vm->DetachCurrentThread();
    m_jniEnv = nullptr;
  }
}

void AndroidApp::cleanupVulkanResources() AVK_NO_CFI {
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
  vkDiscardPool()->discardImage(m_depthImage, m_depthAlloc, timeline());
  m_depthImage = VK_NULL_HANDLE;
  m_depthAlloc = VK_NULL_HANDLE;
  // framebuffer
  for (VkFramebuffer framebuffer : m_framebuffers) {
    vkDiscardPool()->discardFramebuffer(framebuffer, timeline());
  }
  // command buffer bookkeeping
  m_commandBufferIds.clear();
}

void AndroidApp::createVulkanResources() AVK_NO_CFI {
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
  vk::SingleImage2DSpecVk imgSpec{};
  imgSpec.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  imgSpec.width = vkSwapchain()->extent().width;
  imgSpec.height = vkSwapchain()->extent().height;
  imgSpec.imageTiling = VK_IMAGE_TILING_OPTIMAL;
  imgSpec.samples = VK_SAMPLE_COUNT_1_BIT;
  imgSpec.format = depthFmt;
  bool res = vk::createImage(vkDevice(), imgSpec, &m_depthImage, &m_depthAlloc);
  if (!res){
    showErrorScreenAndExit("Couldn't create Depth image");
  }
  m_depthView = vk::depthStencilImageView(
                    vkDevice(), m_depthImage,
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
}

void AndroidApp::createConstantVulkanResources() AVK_NO_CFI {
  // vertex buffer
  using avk::experimental::Vertex;
  std::vector<Vertex> const vertices = {
      {{0.5f, -0.5f, 0}, {1.0f, 0.0f, 0.0f}},  // RT Vertex 0: Red
      {{-0.5f, 0.5f, 0}, {0.0f, 1.0f, 0.0f}},  // LB Vertex 1: Green
      {{0.5f, 0.5f, 0}, {0.0f, 0.0f, 1.0f}},   // RB Vertex 2: Blue
      {{-0.5f, -0.5f, 0}, {1.0f, 0.0f, 1.0f}}  // LT Vertex 3: Magenta
  };
  size_t const vertexBytes = sizeof(Vertex) * vertices.size();
  bool status = vk::createBuffer(
      vkDevice(), sizeof(Vertex) * vertices.size(),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk::HostVisibleCoherent, 0,
      vk::DynamicBufferVmaFlags, &m_vertexBuffer, &m_vertexAlloc);
  assert(status && "Failed to allocate vertex buffer");
  VK_CHECK(vmaCopyMemoryToAllocation(vkDevice()->vmaAllocator(),
                                     vertices.data(), m_vertexAlloc, 0,
                                     vertexBytes));
  // graphicsInfo
  // apparently gameActivity already loads it from java?
  // AAssetManager* assetMgr = AAssetManager_fromJava(m_jniEnv,
  // m_app->activity->assetManager);
  auto const vertCode = spirvFromAsset(m_app->activity->assetManager,
                                       "shaders/basic-triangle.vert.spv");
  auto const fragCode = spirvFromAsset(m_app->activity->assetManager,
                                       "shaders/basic-triangle.frag.spv");
  // shaders
  VkShaderModule modules[2] = {
      vk::createShaderModule(vkDevice(), vertCode.data(), vertCode.size() << 2),
      vk::createShaderModule(vkDevice(), fragCode.data(),
                             fragCode.size() << 2)};
  m_graphicsInfo = experimental::basicGraphicsInfo(
      vk::createPipelineLayout(vkDevice()), modules,
      vk::basicDepthStencilFormat(vkDevice()->physicalDevice()));
}

void AndroidApp::destroyConstantVulkanResources() AVK_NO_CFI {
  // vertex buffer
  vkDiscardPool()->discardBuffer(m_vertexBuffer, m_vertexAlloc, timeline());
  experimental::discardGraphicsInfo(vkDiscardPool(), timeline(),
                                    m_graphicsInfo);
}

void AndroidApp::RTdoOnWindowInit() {
#define PREFIX "[AndroidApp::onWindowInit] "
  createConstantVulkanResources();
  LOGI << PREFIX "Constant Resources Created" << std::endl;
  createVulkanResources();
  LOGI << PREFIX "Resources Created" << std::endl;
#undef PREFIX
}

void AndroidApp::RTdoOnResize() {
  LOGI << "[AndroidApp] Resize callback on derived class" << std::endl;
  cleanupVulkanResources();
  createVulkanResources();
}

VkResult AndroidApp::RTdoOnRender(vk::utils::SwapchainData const& swapchainData)
    AVK_NO_CFI {
  auto const* const vkDevApi = vkDevTable();

  // acquire a command buffer
  VkCommandBuffer const cmd = vkCommandPools()->allocatePrimary(
      m_commandBufferIds[vkSwapchain()->frameIndex()]);
  // begin command buffer
  VkClearValue clear[2]{};
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
  VkDeviceSize offset = 0;
  vkDevApi->vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer, &offset);
  // set scissor and viewport
  vkDevApi->vkCmdSetScissor(cmd, 0, 1, &rect);
  VkViewport viewport{};
  viewport.width = rect.extent.width;
  viewport.height = rect.extent.height;
  viewport.maxDepth = 1.f;
  vkDevApi->vkCmdSetViewport(cmd, 0, 1, &viewport);
  // draw call
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

vk::SurfaceSpec AndroidApp::doSurfaceSpec() {
  vk::SurfaceSpec spec{};
  spec.window = m_app->window;
  return spec;
}

void AndroidApp::RTdoOnSurfaceLost() {
  // nothing for now
}

void AndroidApp::RTdoEarlySurfaceRegained() {
  // discard everything related to surface and swapchain (depth/stencil image
  // could survive, but who cares)
  cleanupVulkanResources();
}

void AndroidApp::RTdoLateSurfaceRegained() {
  // recreate everything related to surface and swapchain (depth/stencil image
  // could survive, but who cares)
  createVulkanResources();
}

void AndroidApp::doOnSaveState() {
  // nothing (TODO pipeline cache and simulation state)
  // pipeline cache -> base class
  // simulation state -> middle class among all platforms
  // OS Specific -> final subclass
}

void AndroidApp::doOnRestoreState() {
  // nothing (TODO pipeline cache and simulation state)
}

void AndroidApp::RTdoOnDeviceLost() {
  cleanupVulkanResources();
  destroyConstantVulkanResources();
}

void AndroidApp::RTdoOnDeviceRegained() {
  createConstantVulkanResources();
  createVulkanResources();
}

}  // namespace avk
