#import "avk-metal-helpers.h"

VkExtent2D avkGetDrawableExtent(CAMetalLayer const* layer) {
  CGSize size = [layer drawableSize];
  VkExtent2D extent;
  extent.width = static_cast<uint32_t>(size.width);
  extent.height = static_cast<uint32_t>(size.height);
  return extent;
}