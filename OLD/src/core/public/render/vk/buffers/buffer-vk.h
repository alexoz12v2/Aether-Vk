#pragma once

#include "render/vk/common-vk.h"

namespace avk::vk {

class Device;

// TODO add extensive comments on each usage
inline constexpr VkMemoryPropertyFlags HostVisibleCoherent =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
inline constexpr VmaAllocationCreateFlags DynamicBufferVmaFlags =
    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
    VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
    VMA_ALLOCATION_CREATE_MAPPED_BIT;


// TODO better and export if needed
/// Creates a `VkBuffer`
/// Assumes Exclusive usage on a single queue
/// Memory Mapping: Use `vmaMapMemory`
///   can be mapped to host mem if
///   (allocCreateInfo.flags &
///   (VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
///   VMA_ALLOCATION_CREATE_HOST_ACCESS_ALLOW_TRANSFER_INSTEAD_BIT |
///   VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT))) {
bool createBuffer(Device* device, size_t size, VkBufferUsageFlags usage,
                  VkMemoryPropertyFlags required,
                  VkMemoryPropertyFlags preferred,
                  VmaAllocationCreateFlags vmaFlags, VkBuffer* buffer,
                  VmaAllocation* allocation);

}  // namespace avk::vk