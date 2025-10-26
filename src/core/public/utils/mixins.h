#pragma once

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
  NonMoveable(NonMoveable const &) = delete;
  NonMoveable& operator=(NonMoveable const &) = delete;
  NonMoveable(NonMoveable &&) noexcept = delete;
  NonMoveable& operator=(NonMoveable &&) noexcept = delete;
};

}  // namespace avk