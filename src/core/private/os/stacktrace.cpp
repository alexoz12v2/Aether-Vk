#include "os/filesystem.h"

// out stuff
#include "os/avk-log.h"
#include "os/stackstrace.h"

#if defined(_WIN32)
// clang-format off
#define DBGHELP_TRANSLATE_TCHAR // ensure UNICODE on dbghelp library
#include <windows.h>
#include <psapi.h>
#include <dbghelp.h>
// clang-format on

#elif defined(__linux__) && !defined(__ANDROID__)
#  include <cxxabi.h>
#  include <execinfo.h>

#elif defined(__APPLE__)
#  include <TargetConditionals.h>
#  if !TARGET_OS_IOS
#    include <cxxabi.h>
#    include <execinfo.h>
#  else
#    include <cxxabi.h>
#    include <dlfcn.h>
#    include <unwind.h>
#  endif
#elif defined(__ANDROID__)
// Note how we are assuming clang
#  include <cxxabi.h>
// backtace is available from API Level 33 (Android 13)
#  if !__BIONIC_AVAILABILITY_GUARD(33)
#    error "TODO: Not Managing API 33 execinfo (backtrace)"
#  endif
#  include <cxxabi.h>
#  include <dlfcn.h>
#  include <execinfo.h>
#  include <unistd.h>
#endif

#include <iostream>
#include <mutex>
#include <sstream>
#include <vector>

namespace avk {
#if defined(_WIN32)

static std::once_flag s_symInitOnceFlag;

static inline void ensureDbgHelpInitialized() {
  std::call_once(s_symInitOnceFlag, []() {
    HANDLE process = GetCurrentProcess();
    SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_LOAD_LINES);
    if (!SymInitializeW(process,
                        L"srv*.;https://msdl.microsoft.com/download/symbols",
                        TRUE)) {
      LOGE << "Couldn't load debug symbols from executable" << std::endl;
    }

    // load symbols for current HMODULE explicitly
    // TODO: (might break if it goes through a DLL)
    std::wstring exepath = getExecutablePath().wstring();
    // Get the actual runtime base address and size
    HMODULE hMod = GetModuleHandleW(nullptr);
    MODULEINFO mi = {};
    GetModuleInformation(process, hMod, &mi, sizeof(mi));
    DWORD64 baseAddr = reinterpret_cast<DWORD64>(hMod);  // BaseOfDll
    const DWORD size = mi.SizeOfImage;                   // SizeOfDll

    // Load symbols with the correct runtime address
    baseAddr = SymLoadModuleExW(process, nullptr, exepath.c_str(), nullptr,
                                baseAddr, size, nullptr, 0);
    if (baseAddr == 0) {
      LOGE << "SymLoadModuleEx failed: " << GetLastError() << std::endl;
    } else {
      LOGI << "Correctly Initialized DbgHelp" << std::endl;
    }

    IMAGEHLP_MODULEW64 moduleInfo;
    ZeroMemory(&moduleInfo, sizeof(moduleInfo));
    moduleInfo.SizeOfStruct = sizeof(moduleInfo);
    if (SymGetModuleInfoW64(process, baseAddr, &moduleInfo)) {
      std::wcout << L"Module: " << moduleInfo.ImageName << L"\n";
      std::wcout << L"  Loaded image: " << moduleInfo.LoadedImageName << L"\n";
      std::wcout << L"  Loaded PDB: " << moduleInfo.LoadedPdbName << L"\n";
      std::wcout << L"  Symbols: "
                 << (moduleInfo.SymType != SymNone ? L"yes" : L"no") << L"\n"
                 << std::flush;
    } else {
      std::cerr << "SymGetModuleInfoW64 failed: " << GetLastError()
                << std::endl;
    }
  });
}

#endif

// TODO doesn't work on android at all

std::string dumpStackTrace([[maybe_unused]] uint32_t maxFrames) {
#if defined(AVK_NO_RDYNAMIC)
  return "<stacktrace disabled>";
#else
  std::ostringstream oss;
  oss << "Stack trace (latest " << maxFrames << " frames):\n";
  if (maxFrames == 0) return oss.str();

#  if defined(_WIN32)
  // --- Windows implementation ---

  // clamp to a sane upper bound to avoid huge allocations
  if (constexpr size_t kMaxAllowedFrames = 1024; maxFrames > kMaxAllowedFrames)
    maxFrames = kMaxAllowedFrames;

  // Move frame buffer to the heap
  std::vector<void *> frameBuffer;
  frameBuffer.resize(maxFrames);

  // Make sure DbgHelp is initialized once, thread-safely
  ensureDbgHelpInitialized();
  const HANDLE process = GetCurrentProcess();

  // CaptureStackBackTrace expects a DWORD count
  const WORD frames = CaptureStackBackTrace(0, static_cast<DWORD>(maxFrames),
                                            frameBuffer.data(), nullptr);

  // Prepare a heap-allocated symbol buffer big enough for long names
  constexpr size_t symBufSize =
      sizeof(SYMBOL_INFOW) + MAX_SYM_NAME * sizeof(TCHAR);

  const std::unique_ptr<char[]> symBuf(new char[symBufSize]);
  const auto symbol = reinterpret_cast<SYMBOL_INFOW *>(symBuf.get());
  ZeroMemory(symbol, symBufSize);
  symbol->SizeOfStruct = sizeof(SYMBOL_INFOW);
  symbol->MaxNameLen = MAX_SYM_NAME;

  for (WORD i = 1; i < frames; ++i) {
    if (const DWORD64 address = reinterpret_cast<DWORD64>(frameBuffer[i]);
        SymFromAddrW(process, address, nullptr, symbol)) {
      std::wstring wname(symbol->Name);
      std::string name(wname.begin(), wname.end());
      oss << "  [" << i << "] " << name << " - 0x" << std::hex
          << symbol->Address << std::dec << "\n";
      // Try to get line info
      IMAGEHLP_LINEW64 line;
      DWORD displacement = 0;
      ZeroMemory(&line, sizeof(line));
      line.SizeOfStruct = sizeof(line);

      if (SymGetLineFromAddrW64(process, address, &displacement, &line)) {
        std::wstring wfile(line.FileName);
        std::string file(wfile.begin(), wfile.end());
        oss << "        at " << file << ":" << line.LineNumber << "\n";
      }
    } else {
      // Fallback: unresolved symbol
      oss << "  [" << i << "] (unknown) - 0x" << std::hex
          << reinterpret_cast<uintptr_t>(frameBuffer[i]) << std::dec << "\n";
    }
  }

#  elif (defined(__linux__) && !defined(__ANDROID__)) || \
      (defined(__APPLE__) && !TARGET_OS_IOS)

  // --- POSIX (Linux/macOS) implementation ---
  std::vector<void*> buffer(maxFrames);
  int frames = backtrace(buffer.data(), (int)maxFrames);
  char** symbols = backtrace_symbols(buffer.data(), frames);

  for (int i = 1; i < frames; ++i) {
    std::string line(symbols[i]);
    // Attempt to demangle C++ symbols
    size_t begin = line.find('(');
    size_t end = line.find('+', begin);
    if (begin != std::string::npos && end != std::string::npos) {
      std::string mangled = line.substr(begin + 1, end - begin - 1);
      int status = 0;
      char* demangled =
          abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
      if (status == 0 && demangled) {
        line.replace(begin + 1, end - begin - 1, demangled);
        free(demangled);
      }
    }
    oss << "  [" << i << "] " << line << "\n";
  }

  free(symbols);

#  elif defined(__ANDROID__)
  // TODO
  int const count = 0;
  void *frames[] = {(void *)&count};
  for (int i = 0; i < count; ++i) {
    Dl_info info;
    if (dladdr(frames[i], &info) && info.dli_sname) {
      int status = 0;
      char *demangled =
          abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
      char const *name =
          (status == 0 && demangled) ? demangled : info.dli_sname;
      oss << "  [" << i << "] ox" << std::hex << info.dli_saddr << std::dec
          << name << '\n';
      if (demangled) free(demangled);
    } else {
      oss << "  [" << i << "] ox" << std::hex << frames[i] << std::dec
          << "???\n";
    }
  }

#  elif (defined(__APPLE__) && TARGET_OS_IOS)
#    error "TODO"
#  else
  oss << "  (stack tracing not supported on this platform)\n";
#  endif
  return oss.str();
#endif
}

void showErrorScreenAndExit(char const *msg) {
#if defined(_WIN32)
  MessageBoxA(NULL, msg, "Fatal Error", MB_OK | MB_ICONERROR);
  abort();
#elif defined(__ANDROID__)
  LOGE << "Fatal Error: " << msg << std::endl;
  abort();
  // TODO
  // show_error_activity(env, activity, msg);
#elif defined(__APPLE__) && !defined(TARGET_OS_IOS)
#  error "TODO
  ShowErrorScreen(msg);
#elif defined(__APPLE__) && defined(TARGET_OS_IOS)
#  error "TODO
#elif defined(__linux__)
  fprintf(stderr, "Fatal Error: %s\n", msg);
  system("zenity --error --title='VK_CHECK Failed' --text='" msg
         "' 2>/dev/null || true");
  raise(SIGINT);
#else
  fprintf(stderr, "Fatal Error: %s\n", msg);
  abort();
#endif
}

}  // namespace avk