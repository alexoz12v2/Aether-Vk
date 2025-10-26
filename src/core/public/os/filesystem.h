#pragma once

#include <filesystem>

namespace avk {
  // TODO for linux use /proc/self/exe with std::filesystem::canonical
  std::filesystem::path getExecutablePath();
}
