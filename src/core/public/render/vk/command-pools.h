#pragma once

#include "render/vk/common-vk.h"
#include "render/vk/device-vk.h"

// standard lib
#include <atomic>
#include <cassert>
#include <shared_mutex>
#include <thread>
#include <unordered_map>
#include <vector>

namespace avk::vk::utils {
/// Minimal SPSC ring buffer. Capacity must be a power of two
/// push -> producer thread (discard pool thread)
/// pop -> consumer thread (command pool owner thread)
/// assumes T is trivial (and preferably standard layout)
template <typename T>
class SpscQueue : public NonMoveable {
  static_assert(std::is_trivial_v<T> && std::is_standard_layout_v<T>);

 public:
  explicit SpscQueue(size_t capPow2) {
    assert(capPow2 >= 2 && 0 == (capPow2 & (capPow2 - 1)));
    m_mask = capPow2 - 1;
    m_buffer.resize(capPow2);
    m_head.store(0, std::memory_order_relaxed);
    m_tail.store(0, std::memory_order_relaxed);
  }

  bool tryPush(T const& item) noexcept {
    size_t const tail = m_tail.load(std::memory_order_relaxed);
    size_t const head = m_head.load(
        std::memory_order_acquire);  // acquire to see the latest head
    // if next place is head -> full
    if (((tail + 1) & m_mask) == (m_mask & head)) {
      return false;
    }
    m_buffer[tail & m_mask] = item;
    m_tail.store(tail + 1, std::memory_order_release);  // publish
    return true;
  }

  bool tryPop(T& outItem) noexcept {
    size_t const head = m_head.load(std::memory_order_relaxed);
    size_t const tail = m_tail.load(
        std::memory_order_acquire);  // acquire to see the latest tail
    if ((head & m_mask) == (tail & m_mask)) {
      return false;  // empty
    }
    outItem = m_buffer[head & m_mask];
    m_head.store(head + 1, std::memory_order_release);  // publish
    return true;
  }

  /// Drain elements into a container from consumer thread to count
  /// useful for thread shutdown
  template <typename Container>
  void drainTo(Container& c) noexcept {
    T tmp;
    while (tryPop(tmp)) c.push_back(tmp);
  }

  // debug only, racy
  size_t approxSize() const noexcept {
    size_t const head = m_head.load(std::memory_order_acquire);
    size_t const tail = m_tail.load(std::memory_order_acquire);
    return tail - head;
  }

 private:
  std::vector<T> m_buffer;
  size_t m_mask;
  std::atomic<size_t> m_head;
  std::atomic<size_t> m_tail;
};

}  // namespace avk::vk::utils

namespace avk::vk {

class DiscardPool;

class CommandPools : public NonMoveable {
 public:
  /// TODO Now we support creation from one pool. If required,
  /// desktop machines may optionally have a async compute family
  CommandPools(Device* device, uint32_t queueFamilyIndex);
  ~CommandPools();

  /// allocate a primary command buffer from the current active pool
  /// id should be unique (eg hashed name)
  VkCommandBuffer allocatePrimary(uint64_t id);

  /// allocate a secondary command buffer from current acrive pool
  /// id should be unique (eg hashed name)
  VkCommandBuffer allocateSecondary(uint64_t id);

  /// called by DiscardPool to return a reset pool for reuse
  /// returns pool to appropriate per-thread list (may be called by a thread
  /// different than the owner)
  void recycle(VkCommandPool pool, std::thread::id owner);

  /// Called internally when pool fragmentation or exhaustion occurs
  void discardActivePool(DiscardPool* discardPool, uint64_t value);

  /// called by a caller thread when you want to release resources associated to
  /// the caller but keep the structures around for the others
  /// Note: You can also use this to flush memory occupied by thread storage
  void threadShutdown();

 private:
  // dependencies which must outlive the object
  struct Deps {
    Device* device;
    uint32_t queueFamilyIndex;
  } m_deps;

  struct ThreadPools {
    utils::SpscQueue<VkCommandPool> recycled;
    VkCommandPool active = VK_NULL_HANDLE;
#ifndef AVK_NO_COMMAND_BUFFER_CACHING
    // TODO: Custom using a pool of memory
    std::unordered_map<VkCommandPool,
                       std::unordered_map<uint64_t, VkCommandBuffer>>
        m_cmdCache;
#endif
    explicit ThreadPools(size_t cap) : recycled(cap) {}
  };
  // values need to be unique pointers because a thread might
  // grow the container while others have references to values!
  // TODO: Custom unique pointer using a pool of memory
  std::unordered_map<std::thread::id, std::unique_ptr<ThreadPools>> m_registry;
  std::shared_mutex m_registryMtx;
  /// No thread can request more than 64 VkCommandPools
  size_t m_spscCapacity = 64;

  // WARNING: Unstable references here!
  ThreadPools* ensureThreadPoolsForThisThread();
  ThreadPools* threadPoolsForOwner(std::thread::id tid);

  // called only by owner. access to registry not synchronized
  VkCommandPool createCommandPool(ThreadPools* tp);
  VkCommandBuffer allocateForLevel(uint64_t id, VkCommandBufferLevel level);
};

}  // namespace avk::vk