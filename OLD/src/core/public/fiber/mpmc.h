#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <new>

namespace avk {

// --- Vyukov bounded MPMC queue (template)
// https://www.1024cores.net/home/lock-free-algorithms/queues/bounded-mpmc-queue
// ---
template <typename T>
class MPMCQueue {
 public:
  explicit MPMCQueue(size_t capacity_pow2)
      : m_buffer(nullptr),
        m_capacity(capacity_pow2),
        m_mask(capacity_pow2 - 1) {
    assert((capacity_pow2 & (capacity_pow2 - 1)) == 0);
    m_buffer = reinterpret_cast<Cell*>(
        ::operator new[](sizeof(Cell) * m_capacity, std::nothrow));
    assert(m_buffer);
    for (size_t i = 0; i < m_capacity; ++i) {
      new (&m_buffer[i]) Cell();
      m_buffer[i].seq.store(i, std::memory_order_relaxed);
    }
    m_enqueuePos.store(0);
    m_dequeuePos.store(0);
  }
  MPMCQueue(MPMCQueue const&) = delete;
  MPMCQueue(MPMCQueue&&) noexcept = delete;
  MPMCQueue& operator=(MPMCQueue const&) = delete;
  MPMCQueue& operator=(MPMCQueue&&) noexcept = delete;
  ~MPMCQueue() noexcept {
    for (size_t i = 0; i < m_capacity; ++i) {
      m_buffer[i].~Cell();
    }
    ::operator delete[](m_buffer);
  }

  bool push(const T& v) {
    Cell* cell = nullptr;
    size_t pos = m_enqueuePos.load(std::memory_order_relaxed);
    for (;;) {
      cell = &m_buffer[pos & m_mask];
      size_t const seq = cell->seq.load(std::memory_order_acquire);
      intptr_t const dif =
          static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);
      if (dif == 0) {
        if (m_enqueuePos.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed))
          break;
      } else if (dif < 0) {
        return false;  // full
      } else {
        pos = m_enqueuePos.load(std::memory_order_relaxed);
      }
    }
    cell->value = v;
    cell->seq.store(pos + 1, std::memory_order_release);
    return true;
  }

  bool pop(T& out) {
    Cell* cell = nullptr;
    size_t pos = m_dequeuePos.load(std::memory_order_relaxed);
    for (;;) {
      cell = &m_buffer[pos & m_mask];
      size_t const seq = cell->seq.load(std::memory_order_acquire);
      intptr_t const dif =
          static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);
      if (dif == 0) {
        if (m_dequeuePos.compare_exchange_weak(pos, pos + 1,
                                               std::memory_order_relaxed))
          break;
      } else if (dif < 0) {
        return false;  // empty
      } else {
        pos = m_dequeuePos.load(std::memory_order_relaxed);
      }
    }
    out = cell->value;
    cell->seq.store(pos + m_capacity, std::memory_order_release);
    return true;
  }

 private:
  struct Cell {
    std::atomic<size_t> seq;
    T value;
    Cell() : seq(0), value() {}
  };

  Cell* m_buffer;
  size_t m_capacity;
  size_t m_mask;
  std::atomic<size_t> m_enqueuePos;
  std::atomic<size_t> m_dequeuePos;
};
}  // namespace avk