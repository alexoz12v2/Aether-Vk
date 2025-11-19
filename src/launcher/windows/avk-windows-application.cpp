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
#include "render/experimental/avk-ktx2-textures.h"
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

void WindowsApplication::UTdoOnFixedUpdate() {}

void WindowsApplication::UTdoOnUpdate() {
  // rotate camera, then copy to RT
  // deltatime is int64_t in microseconds. convert it to seconcds
  float const dt = static_cast<float>(Time.current().DeltaTime) * 1e-6f;

  // Rotation speed (radians per second) — very slow rotation
  float const rotationSpeed = glm::radians(100.0f);

  // Accumulate angle over time
  m_angle += rotationSpeed * dt;
  // keep angle small (avoid sin/cos precision loss)
  m_angle = glm::mod(m_angle, glm::two_pi<float>());
  glm::vec3 cubeCenter = glm::vec3(0, 1, -1);
  float radius = 3.0f;

  // Orbit in XY plane (since +Y is forward)
  glm::vec3 eye;
  eye.x = cubeCenter.x + radius * cos(m_angle);
  eye.y = cubeCenter.y + radius * sin(m_angle);
  eye.z = cubeCenter.z + 1.f;  // keep same height (Z+ is up)

  // --- Compute stable camera basis (prevents flipping) ---

  // 1. forward direction
  glm::vec3 forward = glm::normalize(cubeCenter - eye);

  // 2. world-up (Z+ in our coordinate system)
  glm::vec3 worldUp(0, 0, 1);

  // 3. right vector (orthogonal to forward & worldUp)
  glm::vec3 right = glm::normalize(glm::cross(worldUp, forward));

  // If forward almost parallel to worldUp, fallback:
  if (glm::length(right) < 0.001f) {
    // Use X+ as alternative up reference
    worldUp = glm::vec3(1, 0, 0);
    right = glm::normalize(glm::cross(worldUp, forward));
  }

  // 4. final camera-up
  glm::vec3 up = glm::cross(forward, right);
  // Recompute camera view matrix
  m_UTcamera = glm::lookAt(eye, cubeCenter, up);
  {
    std::unique_lock wLock{m_swapState};
    m_RTcamera.view = m_UTcamera;
  }
}

void WindowsApplication::UTdoOnInit() {
  // setup initial camera
  m_UTcamera = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),  // eye position
                           glm::vec3(0.0f, 1.0f, -1.f),  // cube target
                           glm::vec3(0.0f, 0.0f, 1.0f)   // up direction
  );
}

void WindowsApplication::RTdoOnDeviceLost() {
  showErrorScreenAndExit("Desktop should never lose device");
}

void WindowsApplication::RTdoOnDeviceRegained() {
  showErrorScreenAndExit("Desktop should never lose device");
}

WindowsApplication::~WindowsApplication() noexcept AVK_NO_CFI {
  LOGI << "[WindowsApplication] Detructor Running" << std::endl;
  if (windowInitializedOnce()) {
    // It's contained in base class, but we want to ensure that the host storage
    // used in here is not locked in some GPU submitted operations
    vkDevTable()->vkDeviceWaitIdle(vkDeviceHandle());
    cleanupVulkanResources();
    destroyConstantVulkanResources();

    vkDiscardPool()->destroyDiscardedResources(true);
  }
}

void WindowsApplication::createConstantVulkanResources() AVK_NO_CFI {
  using namespace avk::literals;

  // texture loader
  m_textureLoader.create(vkInstance(), vkDevice());

  // cubemap texture (TODO cleanup)
  auto const path =
      getExecutablePath().parent_path() / "assets" / "starry-night-uastc.ktx2";
  m_textureLoader.get()->loadTexture(
      "Tex"_hash, path.string(), VK_IMAGE_USAGE_SAMPLED_BIT,
      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, m_cubeTexInfo);

  // reserve enough space such that, on swapchain recreation, the host
  // memory holding the source for our push constants is never reallocated.
  m_pushCameras.reserve(64);
  // ---------------------- setup initial camera ----------------------------
  {
    // useless, overridden by update thread initialization
    m_RTcamera.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.0f),  // eye position
                                  glm::vec3(0.0f, 1.0f, -1.f),  // cube target
                                  glm::vec3(0.0f, 0.0f, 1.0f)   // up direction
    );
    float fov = glm::radians(60.0f);
    float aspect = vkSwapchain()->extent().width /
                   static_cast<float>(vkSwapchain()->extent().height);

    m_RTcamera.proj = glm::perspectiveRH_ZO(fov, aspect, 0.1f, 100.0f);
    // Vulkan wants Y flipped vs OpenGL
    m_RTcamera.proj[1][1] *= -1;
  }

  // TODO: move first copy for SoC to timeline 0

  // --------------------- index/vertex buffers Main --------------------------
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

  // --------------------- index/vertex buffers Skybox ------------------------
  // 1. Allocate vertex and index buffers
  bufRes = bufferManager()->createBufferGPUOnly(
      "SkyboxVertex"_hash, sizeof(vertexBuffer),
      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, true, false);
  if (bufRes) showErrorScreenAndExit("Couldn't Allocate Vertex Buffer");

  bufRes = bufferManager()->createBufferGPUOnly(
      "SkyboxIndex"_hash, sizeof(indexBuffer), VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
      true, false);
  if (bufRes) showErrorScreenAndExit("Couldn't Allocate Index Buffer");

  // 2. copy if not discrete GPU
  if (!bufferManager()->get("SkyboxVertex"_hash, buffer, alloc))
    showErrorScreenAndExit("Couldn't Retrieve Vertex Buffer");
  // TODO Macro for this. mobile is always true
  if (vkDevice()->isSoC() || vk::isAllocHostVisible(vmaAllocator(), alloc)) {
    VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), vertexBuffer.data(),
                                       alloc, 0, sizeof(vertexBuffer)));
  }  // if not, copy done on first timeline with staging buffer

  if (!bufferManager()->get("SkyboxIndex"_hash, buffer, alloc))
    showErrorScreenAndExit("Couldn't Retrieve Index Buffer");
  if (vkDevice()->isSoC() || vk::isAllocHostVisible(vmaAllocator(), alloc)) {
    VK_CHECK(vmaCopyMemoryToAllocation(vmaAllocator(), indexBuffer.data(),
                                       alloc, 0, sizeof(indexBuffer)));
  }  // if not, copy done on first timeline with staging buffer

  // ------------------------- shaders and push constant (shared) --------------
  // shaders
  std::filesystem::path const exeDir = getExecutablePath().parent_path();

  VkPushConstantRange pushConstantRange{};
  pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
  pushConstantRange.offset = 0;
  pushConstantRange.size = static_cast<uint32_t>(sizeof(Camera));

  VkFormat const depthFmt =
      vk::basicDepthStencilFormat(vkPhysicalDeviceHandle());

  // --------------------------- Main Graphics Pipeline --------------------
  {
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
    // shaders
    auto const vertCode = openSpirV(exeDir / "cube-buffers.vert.spv");
    auto const fragCode = openSpirV(exeDir / "cube-buffers.frag.spv");
    VkShaderModule const modules[2]{
        vk::createShaderModule(vkDevice(), vertCode.data(),
                               vertCode.size() << 2),
        vk::createShaderModule(vkDevice(), fragCode.data(),
                               fragCode.size() << 2)};
    // this doesn't fill in the render pass
    m_graphicsInfo = experimental::basicGraphicsInfo(
        vk::createPipelineLayout(vkDevice(), &m_descriptorSetLayout, 1,
                                 &pushConstantRange, 1),
        modules, depthFmt, experimental::StencilEqualityMode::eReplacing,
        false);

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
  }

  // ------------------------- Skybox Graphics Pipeline --------------------
  {
    // Create Image View for the skybox texture info
    {
      VkImageViewCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
      createInfo.image = m_cubeTexInfo.image;
      createInfo.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      createInfo.format = m_cubeTexInfo.format;
      createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
      createInfo.subresourceRange.baseArrayLayer = 0;
      createInfo.subresourceRange.layerCount = 6;  // cube = 6 faces
      createInfo.subresourceRange.baseMipLevel = 0;
      createInfo.subresourceRange.levelCount = 1;
      VK_CHECK(vkDevTable()->vkCreateImageView(
          vkDeviceHandle(), &createInfo, nullptr, &m_cubeTexInfo.imageView));
    }
    // create separate sampler
    {
      VkSamplerCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
      createInfo.magFilter = VK_FILTER_LINEAR;
      createInfo.minFilter = VK_FILTER_LINEAR;
      createInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      createInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      createInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
      createInfo.minLod = 0.f;
      createInfo.maxLod = m_cubeTexInfo.mipLevels;  // TODO now it's 1
      VK_CHECK(vkDevTable()->vkCreateSampler(vkDeviceHandle(), &createInfo,
                                             nullptr, &m_cubeSampler));
    }
    // create descriptor set layout
    {
      VkDescriptorSetLayoutBinding binding[2]{};  // 0 = tex, 1 = sampler
      binding[0].binding = 0;
      binding[0].descriptorCount = 1;
      binding[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      binding[0].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

      binding[1].binding = 1;
      binding[1].descriptorCount = 1;
      binding[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      binding[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

      VkDescriptorSetLayoutCreateInfo createInfo{};
      createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
      createInfo.bindingCount = 2;
      createInfo.pBindings = binding;
      VK_CHECK(vkDevTable()->vkCreateDescriptorSetLayout(
          vkDeviceHandle(), &createInfo, nullptr,
          &m_skyboxDescriptorSetLayout));
    }
    // graphics Info
    {
      auto const skyboxVertCode = openSpirV(exeDir / "skybox.vert.spv");
      auto const skyboxFragCode = openSpirV(exeDir / "skybox.frag.spv");
      VkShaderModule skyboxModules[2]{
          vk::createShaderModule(vkDevice(), skyboxVertCode.data(),
                                 skyboxVertCode.size() << 2),
          vk::createShaderModule(vkDevice(), skyboxFragCode.data(),
                                 skyboxFragCode.size() << 2)};
      // TODO VkPipelineLayout with descriptor
      // TODO: If the sampler's parameters never change, we can bake it into
      // the pipeline layout by using pImmutableSamplers. In that case, you
      // don't need an entry in the descriptor update template
      m_skyboxGraphicsInfo = experimental::basicGraphicsInfo(
          vk::createPipelineLayout(vkDevice(), &m_skyboxDescriptorSetLayout, 1,
                                   &pushConstantRange, 1),
          skyboxModules, depthFmt,
          experimental::StencilEqualityMode::eZeroExpected, true);
    }
    // descriptor update template for skybox (1 image, 1 sampler)
    {
      VkDescriptorUpdateTemplateEntryKHR entries[2];
      entries[0].dstBinding = 0;
      entries[0].dstArrayElement = 0;
      entries[0].descriptorCount = 1;
      entries[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      entries[0].offset = 0;
      entries[0].stride = sizeof(VkDescriptorImageInfo);

      entries[1].dstBinding = 1;
      entries[1].dstArrayElement = 0;
      entries[1].descriptorCount = 1;
      entries[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
      entries[1].offset = sizeof(VkDescriptorImageInfo);
      entries[1].stride = sizeof(VkDescriptorImageInfo);

      VkDescriptorUpdateTemplateCreateInfoKHR createInfo{};
      createInfo.sType =
          VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
      createInfo.descriptorUpdateEntryCount = 2;
      createInfo.pDescriptorUpdateEntries = entries;
      createInfo.templateType =
          VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET_KHR;
      createInfo.descriptorSetLayout = m_skyboxDescriptorSetLayout;

      VK_CHECK(vkDevTable()->vkCreateDescriptorUpdateTemplateKHR(
          vkDeviceHandle(), &createInfo, nullptr,
          &m_skyboxDescriptorUpdateTemplate));
    }
    // create its descriptor set
    {
      m_skyboxDescriptorSet = vkDescriptorPools()->allocate(
          m_skyboxDescriptorSetLayout, vkDiscardPool(), timeline());
    }
  }

  // ------------------ allocate GPU side buffers ----------------------------
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
  if (m_skyboxDescriptorUpdateTemplate != VK_NULL_HANDLE) {
    vkDevTable()->vkDestroyDescriptorUpdateTemplateKHR(
        vkDeviceHandle(), m_skyboxDescriptorUpdateTemplate, nullptr);
    m_skyboxDescriptorUpdateTemplate = VK_NULL_HANDLE;
  }

  // skybox related resources
  if (m_cubeSampler != VK_NULL_HANDLE) {
    vkDevTable()->vkDestroySampler(vkDeviceHandle(), m_cubeSampler, nullptr);
    m_cubeSampler = VK_NULL_HANDLE;
  }
  experimental::discardGraphicsInfo(vkDiscardPool(), timeline(),
                                    m_skyboxGraphicsInfo);

  // graphics info handles
  experimental::discardGraphicsInfo(vkDiscardPool(), timeline(),
                                    m_graphicsInfo);
  // index/vertex buffers + uniform buffers
  using namespace avk::literals;
  bufferManager()->discardById(vkDiscardPool(), "SkyboxVertex"_hash,
                               timeline());
  bufferManager()->discardById(vkDiscardPool(), "SkyboxIndex"_hash,
                               timeline());
  bufferManager()->discardById(vkDiscardPool(), hashes::Cube, timeline());
  bufferManager()->discardById(vkDiscardPool(), hashes::Model, timeline());
  bufferManager()->discardById(vkDiscardPool(), hashes::Vertex, timeline());
  bufferManager()->discardById(vkDiscardPool(), hashes::Index, timeline());

  // set layout only after pipeline layout (hopefully we don't need a discard)
  vkDiscardPool()->destroyDiscardedResources();
  if (m_descriptorSetLayout != VK_NULL_HANDLE) {
    vkDevTable()->vkDestroyDescriptorSetLayout(vkDeviceHandle(),
                                               m_descriptorSetLayout, nullptr);
    m_descriptorSetLayout = VK_NULL_HANDLE;
  }
  if (m_skyboxDescriptorSetLayout != VK_NULL_HANDLE) {
    vkDevTable()->vkDestroyDescriptorSetLayout(
        vkDeviceHandle(), m_skyboxDescriptorSetLayout, nullptr);
    m_skyboxDescriptorSetLayout = VK_NULL_HANDLE;
  }
  // discard KTX texture
  using namespace avk::literals;
  m_textureLoader.get()->discardById(vkDiscardPool(), "Tex"_hash, m_cubeTexInfo,
                                     timeline());

  // KTX2 texture manager (TODO move)
  m_textureLoader.destroy();
}

void WindowsApplication::cleanupVulkanResources() AVK_NO_CFI {
  using namespace avk::literals;
  // render pass (common on both pipelines, same subpass)
  vkDiscardPool()->discardRenderPass(m_graphicsInfo.renderPass, timeline());
  m_graphicsInfo.renderPass = VK_NULL_HANDLE;
  m_skyboxGraphicsInfo.renderPass = VK_NULL_HANDLE;
  // graphics pipelines
  // -- main pipeline
  vkPipelines()->discardAllPipelines(vkDiscardPool(),
                                     m_graphicsInfo.pipelineLayout, timeline());
  m_graphicsPipeline = VK_NULL_HANDLE;
  // -- skybox pipeline
  vkPipelines()->discardAllPipelines(
      vkDiscardPool(), m_skyboxGraphicsInfo.pipelineLayout, timeline());
  m_skyboxPipeline = VK_NULL_HANDLE;
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
  assert(vkSwapchain()->frameCount() <= m_pushCameras.size());
  // renderPass
  VkFormat const depthFmt =
      vk::basicDepthStencilFormat(vkPhysicalDeviceHandle());
  m_graphicsInfo.renderPass =
      vk::basicRenderPass(vkDevice(), vkSwapchain()->surfaceFormat().format,
                          depthFmt)
          .get();
  m_skyboxGraphicsInfo.renderPass = m_graphicsInfo.renderPass;
  // graphics pipelines (main and skybox)
  // TODO study about pipeline derivatives and pipeline cache
  // -- main pipeline
  m_graphicsPipeline = vkPipelines()->getOrCreateGraphicsPipeline(
      m_graphicsInfo, true, VK_NULL_HANDLE);
  // -- skybox pipeline
  m_skyboxPipeline = vkPipelines()->getOrCreateGraphicsPipeline(
      m_skyboxGraphicsInfo, true, VK_NULL_HANDLE);
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

  m_RTcamera.proj = glm::perspectiveRH_ZO(fov, aspect, 0.1f, 100.0f);
  // Vulkan wants Y flipped vs OpenGL
  m_RTcamera.proj[1][1] *= -1;
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
  using namespace avk::literals;
  static uint64_t constexpr StagingVert = "StagingVert"_hash;
  static uint64_t constexpr StagingIndex = "StagingIndex"_hash;

  if (timeline() == 0) {
    // TODO not duplicate
    std::array<glm::vec3, 8> vertexBuffer;
    std::array<glm::uvec3, 12> indexBuffer;
    std::array<std::array<uint32_t, 12>, 8> faceMap;
    std::array<glm::vec4, 6> colors;
    glm::mat4 const cubeModel =
        glm::translate(glm::mat4(1.f), glm::vec3(0, 1, -1));
    test::cubeColors(colors);
    test::cubePrimitive(vertexBuffer, indexBuffer, faceMap);
    CubeFaceMapping hostCubeFaceMapping{faceMap, colors};
    LOGI << "[WindowsApplication::onRender] First Timeline: Upload staging"
         << std::endl;
    // prepare staging manager with our main vulkan handles
    m_staging.refresh(cmd, bufferManager(), vkDevice(), vkDiscardPool(),
                      timeline());

    // ----------------- vertex/index main ----------------------------------
    {
      VkBuffer vertBuf = VK_NULL_HANDLE, indexBuf = VK_NULL_HANDLE;
      VmaAllocation vertAlloc = VK_NULL_HANDLE, indexAlloc = VK_NULL_HANDLE;

      bufferManager()->get(hashes::Index, indexBuf, indexAlloc);
      bufferManager()->get(hashes::Vertex, vertBuf, vertAlloc);
      assert(vertBuf && indexBuf);

      if (!vkDevice()->isSoC() &&
          !vk::isAllocHostVisible(vmaAllocator(), vertAlloc)) {
        m_staging.enqueue({vertBuf, vertAlloc, vertexBuffer.data(),
                           sizeof(vertexBuffer), StagingVert,
                           VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                           VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT});
      }
      if (!vkDevice()->isSoC() &&
          !vk::isAllocHostVisible(vmaAllocator(), indexAlloc)) {
        m_staging.enqueue({indexBuf, indexAlloc, indexBuffer.data(),
                           sizeof(indexBuffer), StagingIndex,
                           VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                           VK_ACCESS_INDEX_READ_BIT});
      }
      // ----------------- vertex/index skybox --------------------------------
      bufferManager()->get("SkyboxIndex"_hash, indexBuf, indexAlloc);
      bufferManager()->get("SkyboxVertex"_hash, vertBuf, vertAlloc);
      assert(vertBuf && indexBuf);
      if (!vkDevice()->isSoC() &&
          !vk::isAllocHostVisible(vmaAllocator(), vertAlloc)) {
        m_staging.enqueue({vertBuf, vertAlloc, vertexBuffer.data(),
                           sizeof(vertexBuffer), "SkyboxStagingVert"_hash,
                           VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                           VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT});
      }
      if (!vkDevice()->isSoC() &&
          !vk::isAllocHostVisible(vmaAllocator(), indexAlloc)) {
        m_staging.enqueue({indexBuf, indexAlloc, indexBuffer.data(),
                           sizeof(indexBuffer), "SkyboxStagingIndex"_hash,
                           VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
                           VK_ACCESS_INDEX_READ_BIT});
      }
    }
    // --------------------- The rest ---------------------------------------
    // setup GPU only buffer for "cube" uniform buffer
    {
      VkBuffer buffer = VK_NULL_HANDLE;
      VmaAllocation alloc = VK_NULL_HANDLE;
      bufferManager()->get(hashes::Cube, buffer, alloc);
      assert(buffer);
      if (!vkDevice()->isSoC() &&
          !vk::isAllocHostVisible(vmaAllocator(), alloc)) {
        // allocate staging buffer and copy stuff there, then discard staging
        m_staging.enqueue({buffer, alloc, &hostCubeFaceMapping,
                           sizeof(hostCubeFaceMapping), "StagingCube"_hash,
                           VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                           VK_ACCESS_UNIFORM_READ_BIT});
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
        // create a staging buffer for the transfer, then discard it
        m_staging.enqueue({buffer, alloc, glm::value_ptr(cubeModel),
                           sizeof(cubeModel), "StagingModel"_hash,
                           VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
                           VK_ACCESS_UNIFORM_READ_BIT});
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

    // now update the image descriptor with the template
    {
      VkDescriptorImageInfo imageInfos[2]{};
      imageInfos[0].imageView = m_cubeTexInfo.imageView;
      imageInfos[0].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

      imageInfos[1].sampler = m_cubeSampler;

      vkDevTable()->vkUpdateDescriptorSetWithTemplateKHR(
          vkDeviceHandle(), m_skyboxDescriptorSet,
          m_skyboxDescriptorUpdateTemplate, imageInfos);
    }

    // after everything staged, insert necessary pipeline barrier
    m_staging.flush();
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

  // -------------------------- Main ---------------------------------------
  // bind pipeline and vertex buffer
  vkDevApi->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              m_graphicsPipeline);
  // bind vertex and index buffer
  VkDeviceSize offset = 0;

  VkBuffer vertBuf = VK_NULL_HANDLE, indexBuf = VK_NULL_HANDLE;
  VmaAllocation vertAlloc = VK_NULL_HANDLE, indexAlloc = VK_NULL_HANDLE;

  bufferManager()->get(hashes::Vertex, vertBuf, vertAlloc);
  bufferManager()->get(hashes::Index, indexBuf, indexAlloc);
  assert(vertBuf && indexBuf);

  vkDevApi->vkCmdBindVertexBuffers(cmd, 0, 1, &vertBuf, &offset);

  vkDevApi->vkCmdBindIndexBuffer(cmd, indexBuf, 0, VK_INDEX_TYPE_UINT32);

  // descriptor set and push constant
  vkDevApi->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_graphicsInfo.pipelineLayout, 0, 1,
                                    &m_cubeDescriptorSet, 0, nullptr);
  auto& pushConst = m_pushCameras[vkSwapchain()->frameIndex()];
  {
    std::shared_lock rLock{m_swapState};
    pushConst = m_RTcamera;
  }
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

  // ------------------- The skybox ---------------------------------------
  // same viewport, scissor
  bufferManager()->get("SkyboxVertex"_hash, vertBuf, vertAlloc);
  bufferManager()->get("SkyboxIndex"_hash, indexBuf, indexAlloc);
  assert(vertBuf && indexBuf);
  vkDevApi->vkCmdBindVertexBuffers(cmd, 0, 1, &vertBuf, &offset);
  vkDevApi->vkCmdBindIndexBuffer(cmd, indexBuf, 0, VK_INDEX_TYPE_UINT32);
  vkDevApi->vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                                    m_skyboxGraphicsInfo.pipelineLayout, 0, 1,
                                    &m_skyboxDescriptorSet, 0, nullptr);
  vkDevApi->vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              m_skyboxPipeline);
  // remove camera position from view matrix (skybox follows you)
  pushConst.view[3] = glm::vec4(0, 0, 0, pushConst.view[3].w);
  vkDevApi->vkCmdPushConstants(cmd, m_skyboxGraphicsInfo.pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(Camera),
                               &pushConst);
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