#include "os/filesystem.h"

// our
#include "os/avk-log.h"

// std
#include <vector>

#ifdef AVK_OS_WINDOWS
#  include <Windows.h>
#endif

namespace avk {

#ifdef AVK_OS_ANDROID
std::vector<uint32_t> loadSpvFromAsset(AAssetManager* manager,
                                       char const* filename) {
  // assumes JNI properly attached to the current thread
  AAsset* asset = AAssetManager_open(manager, filename, AASSET_MODE_STREAMING);
  if (!asset) {
    LOGE << "Couldn't load asset " << filename << std::endl;
    return {};
  }

  off_t const size = AAsset_getLength(asset);
#  ifdef AVK_DEBUG
  if ((size & 0x3) != 0 || (size >> 2) <= 0) {
    LOGE << "SPIR-V File " << filename << " doesn't have size multiple of 4"
         << std::endl;
    abort();
  }
#  endif
  std::vector<uint32_t> data(size >> 2);
  AAsset_read(asset, reinterpret_cast<uint8_t*>(data.data()), size);
  AAsset_close(asset);

  return data;
}
#endif

#if defined(AVK_OS_WINDOWS) || defined(AVK_OS_MACOS) || defined(AVK_OS_LINUX)
#  ifdef AVK_OS_WINDOWS

std::filesystem::path getExecutablePath() {
  DWORD bufferSize = 256;  // start small, typical paths are shorter than this
  for (;;) {
    std::vector<wchar_t> buffer(bufferSize);
    // GetModuleFileNameW returns the number of characters copied, *not*
    // including the null terminator.
    DWORD numChars = GetModuleFileNameW(nullptr, buffer.data(), bufferSize);
    if (numChars == 0) {
      // Some error occurred (use GetLastError for details)
      return {};
    }
    if (numChars < bufferSize - 1) {
      // Success — no truncation
      return std::filesystem::path{buffer.begin(), buffer.begin() + numChars};
    }
    // If truncated (i.e., buffer was too small), double the size and retry
    // Note: On truncation, GetModuleFileNameW does *not* set
    // ERROR_INSUFFICIENT_BUFFER reliably.
    bufferSize *= 2;
    if (bufferSize > MaxCharsForPath) {  // absolute sanity limit: 32KB path
      return {};
    }
  }
}
#  endif
#endif
}  // namespace avk