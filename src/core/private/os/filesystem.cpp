#include "os/filesystem.h"

#include <vector>

#ifdef AVK_OS_WINDOWS
#  include <Windows.h>
#else
#  error TODO
#endif

namespace avk {
#ifdef AVK_OS_WINDOWS

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
#endif
}  // namespace avk