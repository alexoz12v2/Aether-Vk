#pragma once

#include <QuartzCore/CAMetalLayer.h>
#include "render/vk/common-vk.h"

#ifdef __cplusplus
extern "C" {
#endif

VkExtent2D avkGetDrawableExtent(CAMetalLayer const* layer);

#ifdef __cplusplus
}
#endif
