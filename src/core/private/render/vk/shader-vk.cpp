#include "render/vk/shader-vk.h"

#include "render/vk/device-vk.h"

namespace avk::vk {

VkShaderModule createShaderModule(Device const* device, uint32_t const* code,
                                  size_t codeSize) AVK_NO_CFI {
  auto const* const vkDevApi = device->table();
  VkDevice const dev = device->device();
  VkShaderModule mod = VK_NULL_HANDLE;
  VkShaderModuleCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  createInfo.pCode = code;
  createInfo.codeSize = codeSize;
  // TODO specialization constants
  VK_CHECK(vkDevApi->vkCreateShaderModule(dev, &createInfo, nullptr, &mod));
  return mod;
}

}  // namespace avk::vk