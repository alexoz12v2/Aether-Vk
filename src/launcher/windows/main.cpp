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
// TODO: package main loops of each thread into shared static applicaiton
// functions, such that more functionality (eg time updates) can be made private
static unsigned __stdcall updateThreadFunc(void *arg) {
  auto *app = static_cast<avk::WindowsApplication *>(arg);
  assert(app);
  LOGI << "[UpdateThread] Started Update Thread" << std::endl;
  app->UTonInit();
  while (app->UTshouldRun()) {
    // update timings from last frame
    app->Time.UTupdate();
    while (app->Time.needsFixedUpdate()) {
      app->Time.UTfixedUpdate();
      app->UTonFixedUpdate();
    }
    app->UTonUpdate();
  }
  LOGI << "[UpdateThread] Exiting Update Thread" << std::endl;
  return 0u;
}

static unsigned __stdcall renderThreadFunc(void *arg) {
  auto *app = static_cast<avk::WindowsApplication *>(arg);
  LOGI << "[RenderThread] started with window ready" << std::endl;
  app->RTwindowInit();
  LOGI << "[RenderThread] rendering started" << std::endl;
  // keep running while rendering
  while (app->RTshouldRun()) {
    if (app->RTsurfaceWasLost()) {
      app->RTsurfaceLost();
    }
    // this shouldn't return `true` if the application is
    // currently in the pause state
    if (app->RTshouldUpdate()) {
      // successfully claimed state: render it
      app->RTonRender();
    } else {
      // if CAS failed or nothing to render, fallthrough and wait
      // LOGI << "AVK Render Thread NO UPDATE, GOING TO WAIT" << std::endl;
      app->RTwaitForNextRound();
      // LOGI << "AVK Render Thread WOKE UP" << std::endl;
    }
  }
  LOGI << "[RenderThread] exiting via pthread_exit" << std::endl;
  return 0u;
}

static HANDLE createThreadOrCrash(_beginthreadex_proc_type proc, void *args) {
  uintptr_t const res = _beginthreadex(nullptr, 0, proc, args, 0, nullptr);
  if (!res) {
    unsigned long error = 0;
    _get_doserrno(&error);
    std::string const errorStr =
        "Couldn't create render thread with error " + std::to_string(error);
    avk::showErrorScreenAndExit(errorStr.c_str());
  }
  return reinterpret_cast<HANDLE>(res);
}

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
    // spin render thread
    app.RenderThread = createThreadOrCrash(renderThreadFunc, &app);
    app.UpdateThread = createThreadOrCrash(updateThreadFunc, &app);

    // run primary HWND message loop (handles render thread termination)
    avk::primaryWindowMessageLoop(&app);
    LOGI << "[Main Thread] Destroy Application" << std::endl;
  }

  LOGI << "[Main Thread] Exiting Application" << std::endl;
}
