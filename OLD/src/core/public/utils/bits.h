#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace avk {

// WARNING: only for POT
template <size_t Base>
inline constexpr size_t nextMultipleOf(size_t x) {
  return (x + Base - 1) & (~size_t(Base - 1));
}

// ------------
// Hash Function: FNV1a 64 bit
// https://www.ietf.org/archive/id/draft-eastlake-fnv-22.html#name-fnv64-code
// ------------

// Constants for FNV-1a 64-bit
inline constexpr uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
inline constexpr uint64_t FNV_PRIME = 1099511628211ULL;

inline constexpr uint64_t fnv1aHashBytes(const unsigned char* data,
                                         size_t len) noexcept {
  std::uint64_t h = FNV_OFFSET_BASIS;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint64_t>(data[i]);
    h *= FNV_PRIME;
  }
  return h;
}

inline constexpr uint64_t fnv1aHashBytes(const char* data,
                                         size_t len) noexcept {
  std::uint64_t h = FNV_OFFSET_BASIS;
  for (std::size_t i = 0; i < len; ++i) {
    h ^= static_cast<std::uint64_t>(data[i]);
    h *= FNV_PRIME;
  }
  return h;
}

// Overload: std::string_view
inline constexpr uint64_t fnv1aHash(std::string_view sv) noexcept {
  return fnv1aHashBytes(sv.data(), sv.size());
}

// Overload: C-string literal (deduces length; excludes NUL)
template <size_t N>
inline constexpr uint64_t fnv1aHash(const char (&s)[N]) noexcept {
  return fnv1aHashBytes(s, N - 1);
}

// compile-time checks
static_assert(fnv1aHash("hello") == fnv1aHash(std::string_view("hello")));

}  // namespace avk

namespace avk::literals {

// Literal operator: "hello"_hash
inline constexpr uint64_t operator""_hash(const char* s,
                                          std::size_t n) noexcept {
  return fnv1aHashBytes(s, n);
}
static_assert("hello"_hash == fnv1aHash(std::string_view("hello")));

}  // namespace avk::literals

namespace avk {

// Combines two hash values into one (from Boost's hash_combine)
template <typename I>
inline I hashCombine(I seed, I value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

// Generic hash functor for std::vector<T>
template <typename T>
uint64_t vectorHash(const std::vector<T>& v) {
  uint64_t seed = v.size();
  std::hash<T> hasher;
  for (const auto& elem : v) {
    hashCombine(seed, static_cast<uint64_t>(hasher(elem)));
  }
  return seed;
}

template <typename T, typename F>
uint64_t vectorHash(const std::vector<T>& v, F&& f) {
  uint64_t seed = v.size();
  for (const auto& elem : v) {
    seed = hashCombine(seed, static_cast<uint64_t>(f(elem)));
  }
  return seed;
}

}  // namespace avk
