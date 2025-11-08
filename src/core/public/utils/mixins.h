#pragma once

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

}  // namespace avk