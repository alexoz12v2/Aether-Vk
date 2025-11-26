#include <Windows.h>

// Windows related stuff
#include <Windowsx.h>  // GET_X_LPARAM
#include <vulkan/vulkan_core.h>
#include <winuser.h>

// ExtractIconExW
#include <shellapi.h>
// https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
#include <dwmapi.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <thread>

#include "avk-win32-window.h"
#include "avk-windows-application.h"
#include "os/stackstrace.h"

// TODO add Ctrl + C handler
// TODO manifest

int WINAPI wWinMain([[maybe_unused]] HINSTANCE hInstance,
                    [[maybe_unused]] HINSTANCE hPrevInstance,
                    [[maybe_unused]] LPWSTR lpCmdLine,
                    [[maybe_unused]] int nCmdShow) {
  // Use HeapSetInformation to specify that the process should
  // terminate if the heap manager detects an error in any heap used
  // by the process.
  // The return value is ignored, because we want to continue running in the
  // unlikely event that HeapSetInformation fails.
  HeapSetInformation(nullptr, HeapEnableTerminationOnCorruption, NULL, 0);

  // GUI Console window
  // HKCU\Console\%%Startup\DelegationConsole

#ifdef AVK_DEBUG
  AllocConsole();
  FILE *fDummy;
  freopen_s(&fDummy, "CONOUT$", "w", stdout);
  freopen_s(&fDummy, "CONOUT$", "w", stderr);
  std::ios::sync_with_stdio();
  {
    HANDLE handleOut = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD consoleMode;
    GetConsoleMode(handleOut, &consoleMode);
    consoleMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    consoleMode |= DISABLE_NEWLINE_AUTO_RETURN;
    SetConsoleMode(handleOut, consoleMode);
  }
#endif
  LOGI << "[Main Thread] Hello Windows! Thread ID: " << GetCurrentThreadId()
       << std::endl;

  // create application class and primary window
  {
    avk::WindowsApplication app;
    if (!avk::createPrimaryWindow(&app)) {
      avk::showErrorScreenAndExit("Couldn't create application window");
    }

    // run primary HWND message loop (handles render thread termination)
    avk::primaryWindowMessageLoop(&app);
    LOGI << "[Main Thread] Destroy Application" << std::endl;
  }

  LOGI << "[Main Thread] Exiting Application" << std::endl;
#ifdef AVK_DEBUG  // do not close to inspect graceful termination
  while (true) {
    std::this_thread::yield();
  }
#endif
}
