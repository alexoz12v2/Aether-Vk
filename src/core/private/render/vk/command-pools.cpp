#include "render/vk/command-pools.h"

// stuff
#include "render/vk/discard-pool.h"

// std
#include <cassert>

namespace avk::vk {
// ----------------------- PRIVATE -------------------------------------------

// ensure we have a pseudo-TLS Storage for a given caller thread
CommandPools::ThreadPools* CommandPools::ensureThreadPoolsForThisThread() {
  std::thread::id const tid = std::this_thread::get_id();
  // lookup with read lock first
  if (auto* res = threadPoolsForOwner(tid); res) {
    return res;
  }

  // lookup again cause we gave up the mutex
  std::unique_lock<std::shared_mutex> wlock{m_registryMtx};
  auto it = m_registry.find(tid);
  if (it != m_registry.end()) return it->second.get();

  // if still failed, then construct pseudo-TLS Storage
  auto [it2, wasInserted] = m_registry.try_emplace(
      tid, std::make_unique<ThreadPools>(m_spscCapacity));
  if (!wasInserted || !it2->second) abort();

#ifndef AVK_NO_COMMAND_BUFFER_CACHING
  it2->second->m_cmdCache.reserve(64);
#endif
  return it2->second.get();
}

CommandPools::ThreadPools* CommandPools::threadPoolsForOwner(
    std::thread::id tid) {
  std::shared_lock<std::shared_mutex> rlock{m_registryMtx};
  auto it = m_registry.find(tid);
  return it != m_registry.end() ? it->second.get() : nullptr;
}

VkCommandPool CommandPools::createCommandPool([[maybe_unused]] ThreadPools* tp)
    AVK_NO_CFI {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();
  VkCommandPoolCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  createInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  createInfo.queueFamilyIndex = m_deps.queueFamilyIndex;

  VkCommandPool commandPool = VK_NULL_HANDLE;
  VK_CHECK(
      vkDevApi->vkCreateCommandPool(dev, &createInfo, nullptr, &commandPool));

#ifndef AVK_NO_COMMAND_BUFFER_CACHING
  auto [it, wasInserted] = tp->m_cmdCache.try_emplace(commandPool);
  if (!wasInserted) abort();
  it->second.reserve(64);
#endif

  return commandPool;
}

// ----------------------- PUBLIC API ----------------------------------------

CommandPools::CommandPools(Device* device, uint32_t queueFamilyIndex)
    : m_deps{device, queueFamilyIndex} {}

// TODO Add WARN LOG
CommandPools::~CommandPools() AVK_NO_CFI {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();

  // destroy all pools from registry: drain both active and queued recycled
  // pools
  std::unique_lock<std::shared_mutex> regLk{m_registryMtx};
  for (auto& kv : m_registry) {
    ThreadPools* tp = kv.second.get();
    // drain queued pool (CONSUMER): we are not the owner of the pool, but
    // the destructor relinquishes all resources
    std::vector<VkCommandPool> drained;
    tp->recycled.drainTo(drained);
    for (VkCommandPool p : drained)
      vkDevApi->vkDestroyCommandPool(dev, p, nullptr);
    if (tp->active != VK_NULL_HANDLE) {
      vkDevApi->vkDestroyCommandPool(dev, tp->active, nullptr);
    }
  }
  m_registry.clear();
}

VkCommandBuffer CommandPools::allocateForLevel(
    [[maybe_unused]] uint64_t id, VkCommandBufferLevel level) AVK_NO_CFI {
  ThreadPools* tp = ensureThreadPoolsForThisThread();
  if (tp->active == VK_NULL_HANDLE) {
    // consumer thread (pool owner): recycle a pool first
    if (!tp->recycled.tryPop(tp->active)) {
      tp->active = createCommandPool(tp);
    }
  }

#ifndef AVK_NO_COMMAND_BUFFER_CACHING
  auto& cache = tp->m_cmdCache.at(tp->active);
  // try to get a cached command buffer
  auto it = cache.find(id);
  if (it != cache.end()) return it->second;
#endif

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  VkCommandBufferAllocateInfo alloc{};
  alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc.commandBufferCount = 1;
  alloc.level = level;
  alloc.commandPool = tp->active;

  VK_CHECK(m_deps.device->table()->vkAllocateCommandBuffers(
      m_deps.device->device(), &alloc, &cmd));

#ifndef AVK_NO_COMMAND_BUFFER_CACHING
  // cache the command buffer, abort on collision
  auto [it2, wasInserted] = cache.try_emplace(id, cmd);
  if (!wasInserted) abort();
#endif
  return cmd;
}

VkCommandBuffer CommandPools::allocatePrimary(uint64_t id) AVK_NO_CFI {
  return allocateForLevel(id, VK_COMMAND_BUFFER_LEVEL_PRIMARY);
}

VkCommandBuffer CommandPools::allocateSecondary(uint64_t id) AVK_NO_CFI {
  return allocateForLevel(id, VK_COMMAND_BUFFER_LEVEL_SECONDARY);
}

void CommandPools::recycle(VkCommandPool pool,
                           std::thread::id owner) AVK_NO_CFI {
  // producer thread (Discard pool): Push into owner SPSC
  ThreadPools* tp = threadPoolsForOwner(owner);
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();

  if (!tp) {  // if owner destroyed its storage, just nuke this
    vkDevApi->vkDestroyCommandPool(dev, pool, nullptr);
  } else {
    // try to push. If full, nuke it
    bool ok = tp->recycled.tryPush(pool);
    if (!ok) {
      // TODO warning log
      vkDevApi->vkDestroyCommandPool(dev, pool, nullptr);
    }
  }
}

void CommandPools::discardActivePool(DiscardPool* discardPool, uint64_t value) {
  assert(discardPool);
  ThreadPools* tp = ensureThreadPoolsForThisThread();
  if (tp->active == VK_NULL_HANDLE) return;

  VkCommandPool poolToDiscard = tp->active;
  tp->active = VK_NULL_HANDLE;
  std::thread::id const ownerId = std::this_thread::get_id();
  discardPool->discardCommandPoolForReuse(poolToDiscard, this, ownerId, value);
}

// calling this ensures that each thread destroys its own
// resources, which may be better than having the destructor do it
void CommandPools::threadShutdown() {
  auto const* const vkDevApi = m_deps.device->table();
  VkDevice const dev = m_deps.device->device();

  std::unique_ptr<ThreadPools> tp = nullptr;
  {
    std::unique_lock<std::shared_mutex> lock{m_registryMtx};
    auto it = m_registry.find(std::this_thread::get_id());
    if (it == m_registry.end()) return;
    tp = std::move(it->second);
    // remove registry entry once you stole the storage
    m_registry.erase(it);
  }
  // now drain any recycled pools and destroy them on this thread
  // TODO: Refactor destructor and this
  std::vector<VkCommandPool> drained;
  tp->recycled.drainTo(drained);
  for (auto p : drained) {
    vkDevApi->vkDestroyCommandPool(dev, p, nullptr);
  }
  if (tp->active != VK_NULL_HANDLE) {
    vkDevApi->vkDestroyCommandPool(dev, tp->active, nullptr);
    tp->active = VK_NULL_HANDLE;
  }
}

}  // namespace avk::vk