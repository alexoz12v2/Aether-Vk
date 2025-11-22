#pragma once

#include <memory>
#include <new>
#include <optional>
#include <type_traits>
#include <utility>

namespace avk {

struct NonCopyable {
  NonCopyable(NonCopyable const&) = delete;
  NonCopyable& operator=(NonCopyable&) = delete;

  NonCopyable() = default;
  NonCopyable(NonCopyable&&) noexcept = default;
  NonCopyable& operator=(NonCopyable&&) noexcept = default;
};

// the only one you're "safe" storing references in
struct NonMoveable {
  NonMoveable() = default;
  NonMoveable(NonMoveable const&) = delete;
  NonMoveable& operator=(NonMoveable const&) = delete;
  NonMoveable(NonMoveable&&) noexcept = delete;
  NonMoveable& operator=(NonMoveable&&) noexcept = delete;
};

// getters to non standard layout classes contained as members (like this one)
// give different addresses on Apple Clang MacOS! Magic! Solution: use heap
#if 0
template <typename T>
union DelayedConstruct {
 public:
  DelayedConstruct() {}
  ~DelayedConstruct() {}

  template <typename... Args>
  inline void create(Args&&... args) {
    ::new (reinterpret_cast<void*>(&m_var)) T(std::forward<Args>(args)...);
  }
  inline void destroy() { m_var.~T(); }

  inline T* get() { return std::addressof(m_var); }
  inline T const* get() const { return std::addressof(m_var); }

 private:
  T m_var;
};
#else
template <typename T>
class DelayedConstruct {
 public:
  // Constructors
  DelayedConstruct() noexcept = default;

  DelayedConstruct(const T& value) { create(value); }

  DelayedConstruct(T&& value) { create(std::move(value)); }

  ~DelayedConstruct() { destroy(); }

  // Copy and move
  DelayedConstruct(const DelayedConstruct& other) {
    if (other) create(*other);
  }

  DelayedConstruct(DelayedConstruct&& other) noexcept = default;

  DelayedConstruct& operator=(const DelayedConstruct& other) {
    if (this != &other) {
      if (other)
        create(*other);
      else
        destroy();
    }
    return *this;
  }

  DelayedConstruct& operator=(DelayedConstruct&& other) noexcept = default;

  // Create and destroy
  template <typename... Args>
  void create(Args&&... args) {
    m_ptr = std::make_unique<T>(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void emplace(Args&&... args) {
    m_ptr = std::make_unique<T>(std::forward<Args>(args)...);
  }

  void destroy() { m_ptr.reset(); }
  void reset() { m_ptr.reset(); }

  // Observers
  T* get() noexcept { return m_ptr.get(); }
  const T* get() const noexcept { return m_ptr.get(); }

  T& operator*() & { return *m_ptr; }
  const T& operator*() const& { return *m_ptr; }
  T&& operator*() && { return std::move(*m_ptr); }

  T* operator->() noexcept { return m_ptr.get(); }
  const T* operator->() const noexcept { return m_ptr.get(); }

  explicit operator bool() const noexcept { return static_cast<bool>(m_ptr); }

 private:
  std::unique_ptr<T> m_ptr;
};
#endif

}  // namespace avk