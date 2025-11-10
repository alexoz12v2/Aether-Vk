#include "avk-android-app.h"

// android stuff
#include <android/asset_manager_jni.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>

// library stuff
#include <glm/glm.hpp>

// our stuff
#include "os/avk-log.h"
#include "render/vk/buffers/buffer-vk.h"
#include "render/vk/images/image-vk.h"
#include "render/vk/renderpasses-vk.h"
#include "render/vk/shader-vk.h"

// TODO remove or move
struct Vertex {
  glm::vec3 position;
  glm::vec3 color;
};

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
  m_vkInstance.create();
  LOGI << "[AndroidApp] Vulkan Instance " << std::hex
       << m_vkInstance.get()->handle() << std::dec << " Created" << std::endl;
  AVK_EXT_CHECK(*m_vkInstance.get());
}

AndroidApp::~AndroidApp() noexcept {
  if (m_windowInit) {
    auto const* const vkDevApi = m_vkDevice.get()->table();
    vkDevApi->vkDeviceWaitIdle(m_vkDevice.get()->device());
    m_vkPipelines.destroy();
    m_vkCommandPools.destroy();
    m_vkDescriptorPools.destroy();
    // last before main 3
    m_vkDiscardPool.destroy();

    m_vkSwapchain.destroy();
    m_vkDevice.destroy();
    m_vkSurface.destroy();
  }
  m_vkInstance.destroy();
  if (m_jniEnv) {
    m_app->activity->vm->DetachCurrentThread();
    m_jniEnv = nullptr;
  }
}

void AndroidApp::cleanupVulkanResources() AVK_NO_CFI {
  // render pass
  m_vkDiscardPool.get()->discardRenderPass(m_graphicsInfo.renderPass,
                                           m_timeline);
  m_graphicsInfo.renderPass = VK_NULL_HANDLE;
  // graphics pipeline
  m_vkPipelines.get()->discardAllPipelines(
      m_vkDiscardPool.get(), m_graphicsInfo.pipelineLayout, m_timeline);
  m_graphicsPipeline = VK_NULL_HANDLE;
  // depth image
  m_vkDiscardPool.get()->discardImage(m_depthImage, m_depthAlloc, m_timeline);
  m_depthImage = VK_NULL_HANDLE;
  m_depthAlloc = VK_NULL_HANDLE;
  // framebuffer
  for (VkFramebuffer framebuffer : m_framebuffers) {
    m_vkDiscardPool.get()->discardFramebuffer(framebuffer, m_timeline);
  }
}

void AndroidApp::createVulkanResources() AVK_NO_CFI {
  // renderPass
  m_graphicsInfo.renderPass =
      vk::basicRenderPass(
          m_vkDevice.get(), m_vkSwapchain.get()->surfaceFormat().format,
          vk::basicDepthStencilFormat(m_vkDevice.get()->physicalDevice()))
          .get();
  // graphics pipeline
  m_graphicsPipeline = m_vkPipelines.get()->getOrCreateGraphicsPipeline(
      m_graphicsInfo, true, VK_NULL_HANDLE);
  // depth image
  vk::SingleImage2DSpecVk imgSpec{};
  imgSpec.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                  VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT;
  imgSpec.width = m_vkSwapchain.get()->extent().width;
  imgSpec.height = m_vkSwapchain.get()->extent().height;
  imgSpec.imageTiling = VK_IMAGE_TILING_OPTIMAL;
  imgSpec.samples = VK_SAMPLE_COUNT_1_BIT;
  bool res =
      vk::createImage(m_vkDevice.get(), imgSpec, &m_depthImage, &m_depthAlloc);
  assert(res && "failed allocating depth image");
  // framebuffers
  // TODO
}

void AndroidApp::createConstantVulkanResources() AVK_NO_CFI {
  // vertex buffer
  std::vector<Vertex> const vertices = {
      {{0.5f, -0.5f, 0}, {1.0f, 0.0f, 0.0f}},  // RT Vertex 0: Red
      {{-0.5f, 0.5f, 0}, {0.0f, 1.0f, 0.0f}},  // LB Vertex 1: Green
      {{0.5f, 0.5f, 0}, {0.0f, 0.0f, 1.0f}},   // RB Vertex 2: Blue
      {{-0.5f, -0.5f, 0}, {1.0f, 0.0f, 1.0f}}  // LT Vertex 3: Magenta
  };
  size_t const vertexBytes = sizeof(Vertex) * vertices.size();
  bool status = vk::createBuffer(
      m_vkDevice.get(), sizeof(Vertex) * vertices.size(),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vk::HostVisibleCoherent, 0,
      vk::DynamicBufferVmaFlags, &m_vertexBuffer, &m_vertexAlloc);
  assert(status && "Failed to allocate vertex buffer");
  VK_CHECK(vmaCopyMemoryToAllocation(m_vkDevice.get()->vmaAllocator(),
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
  m_graphicsInfo.preRasterization.vertexModule = vk::createShaderModule(
      m_vkDevice.get(), vertCode.data(), vertCode.size() << 2);
  m_graphicsInfo.fragmentShader.fragmentModule = vk::createShaderModule(
      m_vkDevice.get(), fragCode.data(), fragCode.size() << 2);
  // attribute descriptions: 1 binding, 2 attributes
  m_graphicsInfo.vertexIn.bindings.push_back({});
  VkVertexInputBindingDescription& binding =
      m_graphicsInfo.vertexIn.bindings.back();
  binding.binding = 0;
  binding.stride = sizeof(Vertex);
  binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
  VkVertexInputAttributeDescription attribute{};
  attribute.binding = 0;
  attribute.location = 0;
  attribute.format = VK_FORMAT_R32G32B32_SFLOAT;  // position float3
  attribute.offset = offsetof(Vertex, position);
  m_graphicsInfo.vertexIn.attributes.push_back({});
  attribute.binding = 0;
  attribute.location = 1;
  attribute.format = VK_FORMAT_R32G32B32_SFLOAT;  // position float3
  attribute.offset = offsetof(Vertex, color);
  m_graphicsInfo.vertexIn.attributes.push_back({});

  m_graphicsInfo.fragmentShader.viewports.resize(1);
  m_graphicsInfo.fragmentShader.scissors.resize(1);
  // fragment shader and output attachments
  m_graphicsInfo.fragmentOut.depthAttachmentFormat =
      vk::basicDepthStencilFormat(m_vkDevice.get()->physicalDevice());
  m_graphicsInfo.fragmentOut.stencilAttachmentFormat =
      m_graphicsInfo.fragmentOut.depthAttachmentFormat;
  m_graphicsInfo.opts.rasterizationPolygonMode = VK_POLYGON_MODE_FILL;
  // stencil options
  m_graphicsInfo.opts.flags |= vk::EPipelineFlags::eStencilEnable;
  m_graphicsInfo.opts.stencilCompareOp = vk::EStencilCompareOp::eAlways;
  m_graphicsInfo.opts.stencilLogicalOp = vk::EStencilLogicOp::eReplace;
  m_graphicsInfo.opts.stencilReference = 1;
  m_graphicsInfo.opts.stencilCompareMask = 0xFF;
  m_graphicsInfo.opts.stencilWriteMask = 0xFF;

  // Pipeline Layout
  m_graphicsInfo.pipelineLayout = vk::createPipelineLayout(m_vkDevice.get());
  // Renderpass and subpass
  m_graphicsInfo.subpass = 0;
  // RnderPass on variable resources
}

void AndroidApp::destroyConstantVulkanResources() AVK_NO_CFI {
  // vertex buffer
  m_vkDiscardPool.get()->discardBuffer(m_vertexBuffer, m_vertexAlloc,
                                       m_timeline);
  // TODO refactor on a function to discard graphics info
  m_vkDiscardPool.get()->discardShaderModule(
      m_graphicsInfo.preRasterization.vertexModule, m_timeline);
  m_vkDiscardPool.get()->discardShaderModule(
      m_graphicsInfo.fragmentShader.fragmentModule, m_timeline);
  m_graphicsInfo.preRasterization.vertexModule = VK_NULL_HANDLE;
  m_graphicsInfo.fragmentShader.fragmentModule = VK_NULL_HANDLE;

  // TODO: this is to be recreated if descriptor set layouts change
  m_vkDiscardPool.get()->discardPipelineLayout(m_graphicsInfo.pipelineLayout,
                                               m_timeline);
}

void AndroidApp::onWindowInit() {
  vk::SurfaceSpec surfSpec{};
  surfSpec.window = m_app->window;
  m_vkSurface.create(m_vkInstance.get(), surfSpec);
  AVK_EXT_CHECK(*m_vkSurface.get());
  LOGI << "[AndroidApp] Vulkan Surface " << std::hex
       << m_vkSurface.get()->handle() << std::dec << " Created" << std::endl;
  m_vkDevice.create(m_vkInstance.get(), m_vkSurface.get());
  LOGI << "[AndroidApp] Vulkan Device Created" << std::endl;
  m_vkSwapchain.create(m_vkInstance.get(), m_vkSurface.get(), m_vkDevice.get());
  LOGI << "[AndroidApp] Vulkan Swapchain Created" << std::endl;

  m_vkDiscardPool.create(m_vkDevice.get());
  m_vkCommandPools.create(
      m_vkDevice.get(), m_vkDevice.get()->universalGraphicsQueueFamilyIndex());
  m_vkDescriptorPools.create(m_vkDevice.get());
  m_vkPipelines.create(m_vkDevice.get());

  m_windowInit = true;
}

void AndroidApp::onResize() {
  m_vkSwapchain.get()->recreateSwapchain();
  cleanupVulkanResources();
  createVulkanResources();
}

void AndroidApp::onRender() AVK_NO_CFI {
  using namespace literals;
  auto const* const vkDevApi = m_vkDevice.get()->table();
  VkDevice const dev = m_vkDevice.get()->device();

  if (!m_shouldRender.load(std::memory_order_relaxed)) {
    return;
  }
  // acquire a command buffer
  VkCommandBuffer const cmd =
      m_vkCommandPools.get()->allocatePrimary("Primary"_hash);
  // acquire a swapchain image (handle resize pt.1)
  VkResult res = m_vkSwapchain.get()->acquireNextImage();
  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
    onResize();
    return;
  }
  auto const swapchainData = m_vkSwapchain.get()->swapchainData();
  if (swapchainData.submissionFence != VK_NULL_HANDLE) {
    VK_CHECK(vkDevApi->vkWaitForFences(dev, 1, &swapchainData.submissionFence,
                                       VK_TRUE, UINT64_MAX));
  }
  // begin command buffer
  VkClearValue clear{};
  VkRect2D rect{};
  rect.extent = m_vkSwapchain.get()->extent();
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  VK_CHECK(vkDevApi->vkBeginCommandBuffer(cmd, &beginInfo));
  // begin render pass (transition to optimal layout)
  VkRenderPassBeginInfo renderBegin{};
  renderBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
  renderBegin.renderPass = m_graphicsInfo.renderPass;
  renderBegin.clearValueCount = 1;
  renderBegin.pClearValues = &clear;
  renderBegin.renderArea = rect;

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
                                  m_vkDiscardPool.get()->timelineSemaphore()};
  uint64_t values[2]{0, m_timeline + 1};
  VkTimelineSemaphoreSubmitInfoKHR timelineSubmit{};
  timelineSubmit.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO_KHR;
  timelineSubmit.signalSemaphoreValueCount = 2;
  timelineSubmit.pSignalSemaphoreValues = values;

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.pNext = &timelineSubmit;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &cmd;
  submitInfo.signalSemaphoreCount = 2;
  submitInfo.pSignalSemaphores = submitSignalSems;
  submitInfo.waitSemaphoreCount = 1;
  submitInfo.pWaitSemaphores = &swapchainData.acquireSemaphore;
  VK_CHECK(vkDevApi->vkQueueSubmit(m_vkDevice.get()->queue(), 1, &submitInfo,
                                   swapchainData.submissionFence));

  // queue present
  VkSwapchainKHR const swapchain = m_vkSwapchain.get()->handle();
  uint32_t const imageIndex = m_vkSwapchain.get()->imageIndex();
  VkPresentInfoKHR presentInfo{};
  presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
  presentInfo.swapchainCount = 1;
  presentInfo.pSwapchains = &swapchain;
  presentInfo.pImageIndices = &imageIndex;
  presentInfo.waitSemaphoreCount = 1;
  presentInfo.pWaitSemaphores = &swapchainData.presentSemaphore;

  res = vkDevApi->vkQueuePresentKHR(m_vkDevice.get()->queue(), &presentInfo);
  if (res == VK_SUBOPTIMAL_KHR || res == VK_ERROR_OUT_OF_DATE_KHR) {
    onResize();
  } else {
    VK_CHECK(res);
  }

  m_timeline++;
}

}  // namespace avk
