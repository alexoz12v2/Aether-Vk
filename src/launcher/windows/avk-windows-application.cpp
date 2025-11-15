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
#include "render/testing/avk-primitives.h"
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

namespace hashes {

using namespace avk::literals;
inline constexpr uint64_t Vertex = "Vertex"_hash;
inline constexpr uint64_t Index = "Index"_hash;
inline constexpr uint64_t Model = "Model"_hash;
inline constexpr uint64_t Cube = "Cube"_hash;
inline constexpr uint64_t Staging = "Staging"_hash;

}  // namespace hashes

WindowsApplication::~WindowsApplication() noexcept {
  LOGI << "[WindowsApplication] Detructor Running" << std::endl;
  if (windowInitializedOnce()) {
    // It's contained in base class, but we want to ensure that the host storage
    // used in here is not locked in some GPU submitted operations
    vkDevTable()->vkDeviceWaitIdle(vkDeviceHandle());
    cleanupVulkanResources();
    destroyConstantVulkanResources();
  }
}

// TODO render/basic-types.h
struct alignas(16) float3 {
  glm::vec3 v;
};

// stride 16 uint (note: this has been observed with disassembler)
struct alignas(16) uintArray12 {
  uint32_t i;
};

// outer array has 192 stride
struct uintArrayArray8 {
  uintArray12 is[12];
};
static_assert(sizeof(uintArrayArray8) == 192);

struct CubeFaceMapping {
  uintArrayArray8 faceMap[8];
  float3 colors[6];

  CubeFaceMapping(std::array<std::array<uint32_t, 12>, 8> const& _faceMap,
                  std::array<glm::vec4, 6> const& _colors)
      : faceMap{} {
    for (uint32_t i = 0; i < 8; i++) {
      for (uint32_t j = 0; j < 12; j++) {
        faceMap[i].is[j].i = _faceMap[i][j];
      }
    }
    for (uint32_t i = 0; i < 6; ++i) {
      colors[i].v = _colors[i];
    }
  }
};

void WindowsApplication::createConstantVulkanResources() AVK_NO_CFI {
  // reserve enough space such that, on swapchain recreation, the host memory
  // holding the source for our push constants is never reallocated.
  m_pushCameras.reserve(64);

  // setup initial camera
  {
    m_camera.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),  // eye position
                                glm::vec3(0.0f, 1.0f, -1.f),  // cube target
                                glm::vec3(0.0f, 0.0f, 1.0f)   // up direction
    );
    float fov = glm::radians(60.0f);
    float aspect = vkSwapchain()->extent().width /
                   static_cast<float>(vkSwapchain()->extent().height);

    m_camera.proj = glm::perspectiveRH_ZO(fov, aspect, 0.1f, 100.0f);
    // Vulkan wants Y flipped vs OpenGL
    m_camera.proj[1][1] *= -1;
  }

  // index/vertex buffers
  [[maybe_unused]] std::array<glm::vec3, 8> vertexBuffer;
  [[maybe_unused]] std::array<glm::uvec3, 12> indexBuffer;
  [[maybe_unused]] std::array<std::array<uint32_t, 12>, 8> faceMap;
  [[maybe_unused]] std::array<glm::vec4, 6> colors;
  test::cubeColors(colors);
  test::cubePrimitive(vertexBuffer, indexBuffer, faceMap);

  VkBuffer buffer = VK_NULL_HANDLE;
  VmaAllocation alloc = VK_NULL_HANDLE;

  // 1. Allocate vertex and index buffers
  int bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Vertex, sizeof(vertexBuffer), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
      true, false);
  if (bufRes) showErrorScreenAndExit("Couldn't Allocate Vertex Buffer");

  bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Index, sizeof(indexBuffer), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      true, false);
  if (bufRes) showErrorScreenAndExit("Couldn't Allocate Index Buffer");

  // Discrete GPUs can have DEVICE_LOCAL memory heaps that are not HOST_VISIBLE
  // hence we'll use some staging buffers
  if (!bufferManager()->get(hashes::Vertex, buffer, alloc))
    showErrorScreenAndExit("Couldn't Retrieve Vertex Buffer");
  // TODO Macro for this. mobile is always true
  if (vkDevice()->isSoC() || vk::isAllocHostVisible(vmaAllocator(), alloc)) {
    VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), vertexBuffer.data(),
                                       alloc, 0, sizeof(vertexBuffer)));
  }  // if not, copy done on first timeline with staging buffer

  if (!bufferManager()->get(hashes::Index, buffer, alloc))
    showErrorScreenAndExit("Couldn't Retrieve Index Buffer");
  if (vkDevice()->isSoC() || vk::isAllocHostVisible(vmaAllocator(), alloc)) {
    VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), indexBuffer.data(),
                                       alloc, 0, sizeof(indexBuffer)));
  }  // if not, copy done on first timeline with staging buffer

  // shaders
  std::filesystem::path const exeDir = getExecutablePath().parent_path();
  auto const vertCode = openSpirV(exeDir / "cube-buffers.vert.spv");
  auto const fragCode = openSpirV(exeDir / "cube-buffers.frag.spv");
  VkShaderModule modules[2]{
      vk::createShaderModule(vkDevice(), vertCode.data(), vertCode.size() << 2),
      vk::createShaderModule(vkDevice(), fragCode.data(),
                             fragCode.size() << 2)};

  // descriptor set layout
  {
    VkDescriptorSetLayoutBinding binding[2]{};
    binding[0].binding = 0;
    binding[0].descriptorCount = 1;
    binding[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    binding[1].binding = 1;
    binding[1].descriptorCount = 1;
    binding[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
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

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = static_cast<uint32_t>(sizeof(Camera));

  // this doesn't fill in the render pass
  m_graphicsInfo = experimental::basicGraphicsInfo(
      vk::createPipelineLayout(vkDevice(), &m_descriptorSetLayout, 1,
                               &pushConstantRange, 1),
      modules, vk::basicDepthStencilFormat(vkPhysicalDeviceHandle()));

  // allocate the descriptor set and its update template
  m_cubeDescriptorSet = vkDescriptorPools()->allocate(
      m_descriptorSetLayout, vkDiscardPool(), timeline());

  {
    VkDescriptorUpdateTemplateEntryKHR entries[2]{};
    entries[0].dstBinding = 0;
    entries[0].dstArrayElement = 0;
    entries[0].descriptorCount = 1;
    entries[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    entries[0].offset = 0;  // start of pData -> bufferInfos[0]
    entries[0].stride = sizeof(VkDescriptorBufferInfo);  // not UBO size

    entries[1].dstBinding = 1;
    entries[1].dstArrayElement = 0;
    entries[1].descriptorCount = 1;
    entries[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    entries[1].offset = sizeof(VkDescriptorBufferInfo);  // pData + bufferInfo
    entries[1].stride = sizeof(VkDescriptorBufferInfo);

    VkDescriptorUpdateTemplateCreateInfoKHR createInfo{};
    createInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
    createInfo.descriptorUpdateEntryCount = 2;
    createInfo.pDescriptorUpdateEntries = entries;
    createInfo.templateType =
        VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR;
    createInfo.descriptorSetLayout = m_descriptorSetLayout;

    VK_CHECK(vkDevTable()->vkCreateDescriptorUpdateTemplateKHR(
        vkDeviceHandle(), &createInfo, nullptr, &m_descriptorUpdateTemplate));
  }

  // allocate GPU side buffers
  // - Cube vert/index -> face mapping table
  bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Cube, sizeof(CubeFaceMapping), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      true, false);
  if (bufRes)
    showErrorScreenAndExit("Couldn't allocate buffer for face mapping");
  if (!bufferManager()->get(hashes::Cube, buffer, alloc))
    showErrorScreenAndExit("Couldn't retrieve buffer for face mapping");
  if (vkDevice()->isSoC() || vk::isAllocHostVisible(vmaAllocator(), alloc)) {
    CubeFaceMapping hostCubeFaceMapping{faceMap, colors};
    VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), &hostCubeFaceMapping,
                                       alloc, 0, sizeof(hostCubeFaceMapping)));
  }

  // - model matrix (TODO: Inline Uniform Buffer)
  bufRes = bufferManager()->createBufferGPUOnly(
      hashes::Model, sizeof(glm::mat4), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      true, false);
  if (bufRes) showErrorScreenAndExit("Couldn't allocate model matrix UBO");
  if (!bufferManager()->get(hashes::Model, buffer, alloc))
    showErrorScreenAndExit("Couldn't retrieve model matrix UBO");
  if (vkDevice()->isSoC() || vk::isAllocHostVisible(vmaAllocator(), alloc)) {
    glm::mat4 const cubeModel =
        glm::translate(glm::mat4(1.f), glm::vec3(0, 1, -1));
    VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(),
                                       glm::value_ptr(cubeModel), alloc, 0,
                                       sizeof(glm::mat4)));
  }
  // TODO: timeline 0 should always insert a barrier for host transfer?
}

void WindowsApplication::destroyConstantVulkanResources() AVK_NO_CFI {
  // template and layout
  if (m_descriptorUpdateTemplate != VK_NULL_HANDLE) {
    vkDevTable()->vkDestroyDescriptorUpdateTemplateKHR(
        vkDeviceHandle(), m_descriptorUpdateTemplate, nullptr);
    m_descriptorUpdateTemplate = VK_NULL_HANDLE;
  }
  // graphics info handles
  experimental::discardGraphicsInfo(vkDiscardPool(), timeline(),
                                    m_graphicsInfo);
  // index/vertex buffers + uniform buffers
  bufferManager()->discardById(vkDiscardPool(), hashes::Cube, timeline());
  bufferManager()->discardById(vkDiscardPool(), hashes::Model, timeline());
  bufferManager()->discardById(vkDiscardPool(), hashes::Vertex, timeline());
  bufferManager()->discardById(vkDiscardPool(), hashes::Index, timeline());

  // set layout only after pipeline layout (hopefully we don't need a discard)
  vkDiscardPool()->destroyDiscardedResources();
  if (m_descriptorSetLayout != VK_NULL_HANDLE) {
    vkDevTable()->vkDestroyDescriptorSetLayout(vkDeviceHandle(),
                                               m_descriptorSetLayout, nullptr);
  }
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
  // how many frames do we have -> maintain that many push const memory
  if (vkSwapchain()->frameCount() > m_pushCameras.size()) {
    m_pushCameras.resize(vkSwapchain()->frameCount());
  }
  assert(vkSwapchain()->frameCount() == m_pushCameras.size());
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
  // TODO refactor, this is the same as initial
  float fov = glm::radians(60.0f);
  float aspect = vkSwapchain()->extent().width /
                 static_cast<float>(vkSwapchain()->extent().height);

  m_camera.proj = glm::perspectiveRH_ZO(fov, aspect, 0.1f, 100.0f);
  // Vulkan wants Y flipped vs OpenGL
  m_camera.proj[1][1] *= -1;
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

  // if first timeline, stage all resources to GPU local memory
  VkBuffer vertBuf = VK_NULL_HANDLE, indexBuf = VK_NULL_HANDLE;
  VmaAllocation vertAlloc = VK_NULL_HANDLE, indexAlloc = VK_NULL_HANDLE;
  bufferManager()->get(hashes::Index, indexBuf, indexAlloc);
  bufferManager()->get(hashes::Vertex, vertBuf, vertAlloc);

  using namespace avk::literals;
  static uint64_t constexpr StagingVert = "StagingVert"_hash;
  static uint64_t constexpr StagingIndex = "StagingIndex"_hash;

  if (timeline() == 0) {
    VmaAllocation stagingAlloc = VK_NULL_HANDLE;
    VkBuffer stagingBuf = VK_NULL_HANDLE;
    // TODO better
    std::vector<VkBufferMemoryBarrier> beforeVertexInput;
    std::vector<VkBufferMemoryBarrier> beforeVertexShader;
    beforeVertexInput.reserve(4);
    beforeVertexShader.reserve(4);

    // TODO not duplicate
    [[maybe_unused]] std::array<glm::vec3, 8> vertexBuffer;
    [[maybe_unused]] std::array<glm::uvec3, 12> indexBuffer;
    [[maybe_unused]] std::array<std::array<uint32_t, 12>, 8> faceMap;
    [[maybe_unused]] std::array<glm::vec4, 6> colors;
    test::cubeColors(colors);
    test::cubePrimitive(vertexBuffer, indexBuffer, faceMap);
    LOGI << "[WindowsApplication::onRender] First Timeline: Upload staging"
         << std::endl;
    assert(vertBuf && indexBuf);
    int bufRes = 0;
    if (!vkDevice()->isSoC() &&
        !vk::isAllocHostVisible(vmaAllocator(), vertAlloc)) {
      bufRes = bufferManager()->createBufferStaging(
          StagingVert, sizeof(vertexBuffer), true, false);
      if (bufRes)
        showErrorScreenAndExit("Couldn't allocate staging buffer for vertex");
      if (!bufferManager()->get(StagingVert, stagingBuf, stagingAlloc))
        showErrorScreenAndExit("Couldn't get staging buffer for vertex");
      VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), vertexBuffer.data(),
                                         stagingAlloc, 0,
                                         sizeof(vertexBuffer)));
      // insert memory barrier from host stage to make sure mmapped write ends
      VkBufferMemoryBarrier hostBarrier{};
      hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      hostBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      hostBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // host
      hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // don't care
      hostBarrier.buffer = stagingBuf;
      hostBarrier.offset = 0;
      hostBarrier.size = VK_WHOLE_SIZE;
      vkDevApi->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 1, &hostBarrier, 0, nullptr);
      // then safely transfer operation
      VkBufferCopy bufCopy{};
      bufCopy.srcOffset = 0;
      bufCopy.dstOffset = 0;
      bufCopy.size = sizeof(vertexBuffer);
      vkDevApi->vkCmdCopyBuffer(cmd, stagingBuf, vertBuf, 1, &bufCopy);

      // push a buffer barrier into the beforeVertex
      VkBufferMemoryBarrier transferBarrier{};
      transferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      transferBarrier.dstAccessMask = VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;
      transferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
      transferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
      transferBarrier.buffer = vertBuf;
      transferBarrier.offset = 0;
      transferBarrier.size = VK_WHOLE_SIZE;
      beforeVertexInput.push_back(transferBarrier);

      // discard staging after this timeline
      bufferManager()->discardById(vkDiscardPool(), StagingVert, timeline());
    }

    if (!vkDevice()->isSoC() &&
        !vk::isAllocHostVisible(vmaAllocator(), indexAlloc)) {
      bufRes = bufferManager()->createBufferStaging(
          StagingIndex, sizeof(indexBuffer), true, false);
      if (bufRes)
        showErrorScreenAndExit("Couldn't allocate staging buffer for index");
      if (!bufferManager()->get(StagingIndex, stagingBuf, stagingAlloc))
        showErrorScreenAndExit("Couldn't get staging buffer for index");
      VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), indexBuffer.data(),
                                         stagingAlloc, 0, sizeof(indexBuffer)));
      // host memory barrier so transfer on queue starts after mmapped copy
      VkBufferMemoryBarrier hostBarrier{};
      hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      hostBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
      hostBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // from host
      hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // don't care
      hostBarrier.buffer = stagingBuf;
      hostBarrier.offset = 0;
      hostBarrier.size = VK_WHOLE_SIZE;
      vkDevApi->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                     nullptr, 1, &hostBarrier, 0, nullptr);
      // then record the transfer operation
      VkBufferCopy bufCopy{};
      bufCopy.srcOffset = 0;
      bufCopy.dstOffset = 0;
      bufCopy.size = sizeof(indexBuffer);
      vkDevApi->vkCmdCopyBuffer(cmd, stagingBuf, indexBuf, 1, &bufCopy);

      // push beforeVertex memory barrier
      VkBufferMemoryBarrier transferBarrier{};
      transferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
      transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
      transferBarrier.dstAccessMask = VK_ACCESS_INDEX_READ_BIT;
      transferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
      transferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
      transferBarrier.buffer = indexBuf;
      transferBarrier.offset = 0;
      transferBarrier.size = VK_WHOLE_SIZE;
      beforeVertexInput.push_back(transferBarrier);
    }

    // setup GPU only buffer for "cube" uniform buffer
    {
      VkBuffer buffer = VK_NULL_HANDLE;
      VmaAllocation alloc = VK_NULL_HANDLE;
      bufferManager()->get(hashes::Cube, buffer, alloc);
      assert(buffer);
      if (!vkDevice()->isSoC() &&
          !vk::isAllocHostVisible(vmaAllocator(), alloc)) {
        // allocate staging buffer and copy stuff there
        bufRes = bufferManager()->createBufferStaging(
            "StagingCube"_hash, sizeof(CubeFaceMapping), true, false);
        if (bufRes)
          showErrorScreenAndExit("Couldn't allcoate staging for cube UBO");
        if (!bufferManager()->get("StagingCube"_hash, stagingBuf, stagingAlloc))
          showErrorScreenAndExit("Couldn't get staging for cube UBO");
        CubeFaceMapping hostCubeFaceMapping{faceMap, colors};

        VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), &hostCubeFaceMapping,
                                           stagingAlloc, 0,
                                           sizeof(hostCubeFaceMapping)));
        // pipeline barrier so host copy finishes before transfer cmd
        VkBufferMemoryBarrier hostBarrier{};
        hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        hostBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // host
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // dont care
        hostBarrier.buffer = stagingBuf;
        hostBarrier.offset = 0;
        hostBarrier.size = VK_WHOLE_SIZE;
        vkDevApi->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                       nullptr, 1, &hostBarrier, 0, nullptr);
        // then do a transfer operation
        VkBufferCopy bufCopy{};
        bufCopy.srcOffset = 0;
        bufCopy.dstOffset = 0;
        bufCopy.size = sizeof(CubeFaceMapping);
        vkDevApi->vkCmdCopyBuffer(cmd, stagingBuf, buffer, 1, &bufCopy);

        // push back another barrier for the beforeVertex pipeline barrier
        // TODO this barrier can come before fragment
        VkBufferMemoryBarrier transferBarrier{};
        transferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transferBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        transferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
        transferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
        transferBarrier.buffer = buffer;
        transferBarrier.offset = 0;
        transferBarrier.size = VK_WHOLE_SIZE;
        beforeVertexShader.push_back(transferBarrier);

        // discard staging, won't be needed after this timeline
        bufferManager()->discardById(vkDiscardPool(), "StagingCube"_hash,
                                     timeline());
      }
    }

    // setup GPU only buffer for "model" uniform buffer
    {
      VkBuffer buffer = VK_NULL_HANDLE;
      VmaAllocation alloc = VK_NULL_HANDLE;
      bufferManager()->get(hashes::Model, buffer, alloc);
      assert(buffer);
      if (!vkDevice()->isSoC() &&
          !vk::isAllocHostVisible(vmaAllocator(), alloc)) {
        // create a staging buffer for the transfer
        bufRes = bufferManager()->createBufferStaging(
            "StagingModel"_hash, sizeof(glm::mat4), true, false);
        if (bufRes)
          showErrorScreenAndExit("Couldn't allocate staging for model UBO");
        if (!bufferManager()->get("StagingModel"_hash, stagingBuf,
                                  stagingAlloc))
          showErrorScreenAndExit("Couldn't get staging for model UBO");
        // transfer to staging and barrier such that host copy can finish
        // before queue transfer begins
        glm::mat4 const cubeModel =
            glm::translate(glm::mat4(1.f), glm::vec3(0, 1, -1));
        VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(),
                                           glm::value_ptr(cubeModel),
                                           stagingAlloc, 0, sizeof(cubeModel)));
        VkBufferMemoryBarrier hostBarrier{};
        hostBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        hostBarrier.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
        hostBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        hostBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // host
        hostBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // dont care
        hostBarrier.buffer = stagingBuf;
        hostBarrier.offset = 0;
        hostBarrier.size = VK_WHOLE_SIZE;
        vkDevApi->vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT,
                                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0,
                                       nullptr, 1, &hostBarrier, 0, nullptr);
        // then we can perform the transfer operation
        VkBufferCopy bufferCopy{};
        bufferCopy.srcOffset = 0;
        bufferCopy.dstOffset = 0;
        bufferCopy.size = sizeof(glm::mat4);
        vkDevApi->vkCmdCopyBuffer(cmd, stagingBuf, buffer, 1, &bufferCopy);
        // finally add a pipeline barrier so that transfer finishes before
        // vertex shader stage
        VkBufferMemoryBarrier transferBarrier{};
        transferBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        transferBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        transferBarrier.dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
        transferBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
        transferBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;  // same
        transferBarrier.buffer = buffer;
        transferBarrier.offset = 0;
        transferBarrier.size = VK_WHOLE_SIZE;
        beforeVertexShader.push_back(transferBarrier);
        // discard staging
        bufferManager()->discardById(vkDiscardPool(), "StagingModel"_hash,
                                     timeline());
      }
    }

    // now update the descriptors with the template
    {
      VkBuffer buffer = VK_NULL_HANDLE;
      VmaAllocation alloc = VK_NULL_HANDLE;

      // link descriptors to their buffers
      VkDescriptorBufferInfo bufferInfos[2]{};
      bufferManager()->get(hashes::Cube, buffer, alloc);
      assert(buffer);
      bufferInfos[0].buffer = buffer;
      bufferInfos[0].offset = 0;
      bufferInfos[0].range = VK_WHOLE_SIZE;

      bufferManager()->get(hashes::Model, buffer, alloc);
      assert(buffer);
      bufferInfos[1].buffer = buffer;
      bufferInfos[1].offset = 0;
      bufferInfos[1].range = VK_WHOLE_SIZE;

      LOGI << "RENDER UPDATE BUFFER INFOS WITH " << std::hex
           << bufferInfos[0].buffer << " AND " << bufferInfos[1].buffer
           << std::dec << std::endl;

      vkDevTable()->vkUpdateDescriptorSetWithTemplateKHR(
          vkDeviceHandle(), m_cubeDescriptorSet, m_descriptorUpdateTemplate,
          bufferInfos);
    }

    // after everything staged, insert necessary pipeline barrier
    if (!beforeVertexInput.empty()) {
      // note: Shader stage as destination is wrong
      vkDevApi->vkCmdPipelineBarrier(
          cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, 0, 0, nullptr,
          static_cast<uint32_t>(beforeVertexInput.size()),
          beforeVertexInput.data(), 0, nullptr);
      beforeVertexInput.clear();
    }
    if (!beforeVertexShader.empty()) {
      // note: Shader stage as destination is wrong
      vkDevApi->vkCmdPipelineBarrier(
          cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
          VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0, 0, nullptr,
          static_cast<uint32_t>(beforeVertexShader.size()),
          beforeVertexShader.data(), 0, nullptr);
      beforeVertexShader.clear();
    }
  }

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

  vkDevApi->vkCmdBindVertexBuffers(cmd, 0, 1, &vertBuf, &offset);

  vkDevApi->vkCmdBindIndexBuffer(cmd, indexBuf, 0, VK_INDEX_TYPE_UINT32);

  // descriptor set and push constant
  vkDevApi->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_graphicsInfo.pipelineLayout, 0, 1,
                                    &m_cubeDescriptorSet, 0, nullptr);
  auto& pushConst = m_pushCameras[vkSwapchain()->frameIndex()];
  pushConst = m_camera;
  vkDevApi->vkCmdPushConstants(cmd, m_graphicsInfo.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Camera),
                               &pushConst);

  // set scissor and viewport
  vkDevApi->vkCmdSetScissor(cmd, 0, 1, &rect);
  VkViewport viewport{};
  viewport.width = rect.extent.width;
  viewport.height = rect.extent.height;
  viewport.maxDepth = 1.f;
  vkDevApi->vkCmdSetViewport(cmd, 0, 1, &viewport);

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