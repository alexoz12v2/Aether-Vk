#pragma once

#include "render/vk/common-vk.h"

namespace avk::vk {
  class Device;

  VkShaderModule createShaderModule(Device const* device, uint32_t const* code, size_t codeSize);

  // TODO Spirv-cross library integration
}