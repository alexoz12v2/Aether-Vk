#include "avk-android-app.h"

// android stuff
#include <android/asset_manager_jni.h>
#include <game-activity/native_app_glue/android_native_app_glue.h>

// library stuff
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

// our stuff
#include "os/avk-log.h"
#include "render/experimental/avk-basic-graphics-info.h"
#include "render/vk/buffers/buffer-vk.h"
#include "render/vk/images/image-vk.h"
#include "render/vk/renderpasses-vk.h"
#include "render/vk/shader-vk.h"

// testing stuff to remove
#include "render/testing/avk-primitives.h"

// TODO: framebuffer template builder class, which allows to build renderpass
// and framebuffers
// TODO: ease handing of descriptors

// TODO elsewhere
static std::vector<uint32_t> spirvFromAsset(AAssetManager *assetMgr,
                                            char const *path) {
  AAsset *asset = AAssetManager_open(assetMgr, path, AASSET_MODE_STREAMING);
  if (!asset) return {};
  off_t size = AAsset_getLength(asset);
  std::vector<uint32_t> code(size >> 2);
  AAsset_read(asset, code.data(), size);
  AAsset_close(asset);
  return code;
}

namespace avk {

AndroidApp::AndroidApp(android_app *app, JNIEnv *jniEnv)
    : ApplicationBase(), m_app(app), m_jniEnv(jniEnv) {
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
  for (VkFramebuffer framebuffer : m_framebuffers) {
    vkDiscardPool()->discardFramebuffer(framebuffer, timeline());
  }
  // command buffer bookkeeping
  m_commandBufferIds.clear();
}

void AndroidApp::createVulkanResources() AVK_NO_CFI {
  using namespace avk::literals;
  auto const *const vkDevApi = vkDevTable();
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
  for (VkFramebuffer &framebuffer : m_framebuffers) {
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

  // apply prerotation
  vk::utils::SurfacePreRotation const preRot = vkSwapchain()->preRotation();
  m_camera.view = glm::toMat4(preRot.cameraRotation) * m_camera.view;
  m_camera.proj = preRot.projectionAdjust * m_camera.proj;
}

namespace hashes {

using namespace avk::literals;
inline constexpr uint64_t Vertex = "Vertex"_hash;
inline constexpr uint64_t Index = "Index"_hash;
inline constexpr uint64_t Model = "Model"_hash;
inline constexpr uint64_t Camera = "Camera"_hash;
inline constexpr uint64_t Cube = "Cube"_hash;

}  // namespace hashes

void AndroidApp::createConstantVulkanResources() AVK_NO_CFI {
  // buffers
  alignas(16) std::array<glm::vec3, 8> vertexBuffer;
  alignas(16) std::array<glm::uvec3, 12> indexBuffer;
  alignas(16) std::array<std::array<uint32_t, 12>, 8> faceMap;
  // vec4 not 3 because of HLSL like alignment
  alignas(16) std::array<glm::vec4, 6>
      colors{glm::vec4(.5, .5, .5, 0), {.4, .1, .4, 0},
             {.1, .2, .6, 0}, {.6, .1, .0, 0},
             {.1, .5, .2, 0}, {0, 1, .1, 0}};
  size_t const cubeStructBytes = sizeof(faceMap) + sizeof(colors);
  test::cubePrimitive(vertexBuffer, indexBuffer, faceMap);

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;

  // 1. Allocate vertex and index buffers
  int bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Vertex, vertexBuffer.size() * sizeof(glm::vec3),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, false);
  if (bufRes) {
    showErrorScreenAndExit("Couldn't Allocate Vertex Buffer");
  }
  bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Index, indexBuffer.size() * sizeof(glm::uvec3),
      VK_BUFFER_USAGE_INDEX_BUFFER_BIT, true, false);
  if (bufRes) {
    showErrorScreenAndExit("Couldn't Allocate Index Buffer");
  }

  // 1.1 Copy data into vertex and index buffer
  // Note: there is no need to put a pipeline barrier after
  // `1vmaCopyMemoryToAllocation` cause when we get to recording, this copy will
  // be finished
  if (!bufferManager()->get(hashes::Vertex, buffer, alloc))
    showErrorScreenAndExit("Couldn't get Vertex Buffer");
  VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), vertexBuffer.data(), alloc,
                                     0,
                                     vertexBuffer.size() * sizeof(glm::vec3)));
  if (!bufferManager()->get(hashes::Index, buffer, alloc))
    showErrorScreenAndExit("Couldn't get Index Buffer");
  VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), indexBuffer.data(), alloc,
                                     0,
                                     indexBuffer.size() * sizeof(glm::uvec3)));
  // 2. Camera definition
  m_camera.view = glm::lookAt(glm::vec3(0, 0, 0), glm::vec3(0, 1, -0.6f),
                              glm::vec3(0, 0, 1));

  // 3. Cube vert/index -> face mapping table
  bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Cube, cubeStructBytes, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, true,
      false);
  if (bufRes)
    showErrorScreenAndExit("Couldn't allocate buffer for face mapping");
  if (!bufferManager()->get(hashes::Cube, buffer, alloc))
    showErrorScreenAndExit("Couldn't retrieve buffer for face mapping");
  VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), faceMap.data(), alloc, 0,
                                     sizeof(faceMap)));
  VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(),
                                     glm::value_ptr(*colors.data()), alloc,
                                     sizeof(faceMap), sizeof(colors)));

  // 4. model matrix (TODO: Inline Uniform Buffer)
  glm::mat4 const cubeModel =
      glm::translate(glm::mat4(1.f), glm::vec3(0, 1, -1));
  bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Model, sizeof(glm::mat4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      true, false);
  if (bufRes) showErrorScreenAndExit("Couldn't allocate model matrix UBO");
  if (!bufferManager()->get(hashes::Model, buffer, alloc))
    showErrorScreenAndExit("Couldn't retrieve model matrix UBO");
  VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), glm::value_ptr(cubeModel),
                                     alloc, 0, sizeof(glm::mat4)));

  // graphicsInfo
  // apparently gameActivity already loads it from java?
  // AAssetManager* assetMgr = AAssetManager_fromJava(m_jniEnv,
  // m_app->activity->assetManager);
  auto const vertCode = spirvFromAsset(m_app->activity->assetManager,
                                       "shaders/cube-buffers.vert.spv");
  auto const fragCode = spirvFromAsset(m_app->activity->assetManager,
                                       "shaders/cube-buffers.frag.spv");

  // pipeline layout specification step 1: descriptor set layouts
  {
    VkDescriptorSetLayoutBinding binding[2]{};
    binding[0].binding = 0;
    binding[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding[0].descriptorCount = 1;
    binding[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    binding[1].binding = 1;
    binding[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding[1].descriptorCount = 1;
    binding[1].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo desLayoutCreateInfo{};
    desLayoutCreateInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    desLayoutCreateInfo.bindingCount = 2;
    desLayoutCreateInfo.pBindings = binding;
    VK_CHECK(vkDevTable()->vkCreateDescriptorSetLayout(
        vkDeviceHandle(), &desLayoutCreateInfo, nullptr,
        &m_descriptorSetLayout));
  }

  // now create descriptor set for model and mapping
  m_cubeDescriptorSet = vkDescriptorPools()->allocate(
      m_descriptorSetLayout, vkDiscardPool(), timeline());

  // push constant definition (pipeline layout below)
  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = static_cast<uint32_t>(sizeof(Camera));

  // shaders
  VkShaderModule modules[2] = {
      vk::createShaderModule(vkDevice(), vertCode.data(), vertCode.size() << 2),
      vk::createShaderModule(vkDevice(), fragCode.data(),
                             fragCode.size() << 2)};
  m_graphicsInfo = experimental::basicGraphicsInfo(
      vk::createPipelineLayout(vkDevice(), &m_descriptorSetLayout, 1,
                               &pushConstantRange, 1),
      modules, vk::basicDepthStencilFormat(vkDevice()->physicalDevice()));

  // update template creation
  {
    VkDescriptorUpdateTemplateEntryKHR entries[2]{};
    entries[0].dstBinding = 0;
    entries[0].dstArrayElement = 0;  // start byte offset
    entries[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    entries[0].descriptorCount = 1;
    entries[0].offset = 0;
    entries[0].stride = cubeStructBytes;

    entries[1].dstBinding = 1;
    entries[1].dstArrayElement = 0;
    entries[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    entries[1].descriptorCount = 1;
    entries[1].offset = 0;
    entries[1].stride = sizeof(glm::mat4);

    VkDescriptorUpdateTemplateCreateInfoKHR createInfo{};
    createInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
    createInfo.descriptorUpdateEntryCount = 2;
    createInfo.pDescriptorUpdateEntries = entries;
    createInfo.templateType =
        VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR;
    createInfo.descriptorSetLayout = m_descriptorSetLayout;
    createInfo.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    createInfo.pipelineLayout = m_graphicsInfo.pipelineLayout;
    createInfo.set = 0;

    VK_CHECK(vkDevTable()->vkCreateDescriptorUpdateTemplateKHR(
        vkDeviceHandle(), &createInfo, nullptr, &m_descriptorUpdateTemplate));
  }

  // link descriptors to their buffers
  VkDescriptorBufferInfo bufferInfos[2]{};
  bufferManager()->get(hashes::Cube, buffer, alloc);
  bufferInfos[0].buffer = buffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = cubeStructBytes;

  bufferManager()->get(hashes::Model, buffer, alloc);
  bufferInfos[1].buffer = buffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = sizeof(glm::mat4);

  vkDevTable()->vkUpdateDescriptorSetWithTemplateKHR(
      vkDeviceHandle(), m_cubeDescriptorSet, m_descriptorUpdateTemplate,
      bufferInfos);
}

void AndroidApp::destroyConstantVulkanResources() AVK_NO_CFI {
  using namespace avk::literals;
  // vertex buffer
  bufferManager()->discardById(vkDiscardPool(), "VertexTriangle"_hash,
                               timeline());
  experimental::discardGraphicsInfo(vkDiscardPool(), timeline(),
                                    m_graphicsInfo);
  vkDevTable()->vkDestroyDescriptorUpdateTemplateKHR(
      vkDeviceHandle(), m_descriptorUpdateTemplate, nullptr);
  vkDevTable()->vkDestroyDescriptorSetLayout(vkDeviceHandle(),
                                             m_descriptorSetLayout, nullptr);
  m_descriptorSetLayout = VK_NULL_HANDLE;
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

VkResult AndroidApp::RTdoOnRender(vk::utils::SwapchainData const &swapchainData)
AVK_NO_CFI {
  using namespace avk::literals;
  auto const *const vkDevApi = vkDevTable();

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
  // bind vertex and index buffer
  VkDeviceSize offset = 0;
  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;

  bufferManager()->get(hashes::Vertex, buffer, alloc);
  vkDevApi->vkCmdBindVertexBuffers(cmd, 0, 1, &buffer, &offset);

  bufferManager()->get(hashes::Index, buffer, alloc);
  vkDevApi->vkCmdBindIndexBuffer(cmd, buffer, 0, VK_INDEX_TYPE_UINT32);

  // descriptor set and push constant
  vkDevApi->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_graphicsInfo.pipelineLayout, 0, 1,
                                    &m_cubeDescriptorSet, 0, nullptr);
  vkDevApi->vkCmdPushConstants(cmd, m_graphicsInfo.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Camera),
                               &m_camera);

  // set scissor and viewport
  vkDevApi->vkCmdSetScissor(cmd, 0, 1, &rect);
  VkViewport viewport{};
  viewport.width = rect.extent.width;
  viewport.height = rect.extent.height;
  viewport.maxDepth = 1.f;
  vkDevApi->vkCmdSetViewport(cmd, 0, 1, &viewport);
  // draw call
  vkDevApi->vkCmdDrawIndexed(cmd, 36, 1, 0, 0, 0);
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
