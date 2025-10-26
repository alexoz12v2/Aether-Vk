#include "os/filesystem.h"

#include <vector>

#ifdef AVK_OS_WINDOWS
#include <Windows.h>
#else
#error TODO
#endif

namespace avk {
#ifdef AVK_OS_WINDOWS
std::filesystem::path getExecutablePath() {
  // works since this isn't a DLL
  std::vector<wchar_t> characters(1024);
  // TODO if ERROR_INSUFFICIENT_BUFFER (returns nSize) then enlarge vector
  DWORD numChars = GetModuleFileNameW(nullptr, characters.data(),
                                      static_cast<DWORD>(characters.size()));
  return std::filesystem::path{characters.begin(),
                               characters.begin() + numChars};
}
#endif
}  // namespace avk