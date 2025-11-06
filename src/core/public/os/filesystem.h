#pragma once

#include <filesystem>

namespace avk {
  inline constexpr uint32_t MaxCharsForPath = 32 * 1024;
  // TODO for linux use /proc/self/exe with std::filesystem::canonical
  std::filesystem::path getExecutablePath();
}
