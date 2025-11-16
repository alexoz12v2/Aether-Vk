// clang-format off
#include <Windows.h>
#include "avk-comutils.h"
#include "avk-win32-window.h"
#include "avk-windows-application.h"
// clang-format on

// WARNING: not supported on lld-link, hence you need to dynamically load user32
// when using versioned symbols
//
// #pragma comment(linker, "/DELAYLOAD:user32.dll")
#pragma comment(lib, "user32.lib")  // TODO move to cmake/bazel

#include <Windowsx.h>
#include <shellapi.h>  // ExtractIconExW

// _beginthreadex
#include <process.h>

// https://learn.microsoft.com/en-us/windows/win32/api/dwmapi/ne-dwmapi-dwmwindowattribute
#include <dwmapi.h>

// https://www.gaijin.at/en/infos/windows-version-numbers
#include <VersionHelpers.h>

// Older DPI APIs (Windows 8 )
#include <shellscalingapi.h>
#pragma comment(lib, "Shcore.lib")  // TODO move to cmake/bazel

// Component Object Model
// https://www.221bluestreet.com/offensive-security/windows-components-object-model/demystifying-windows-component-object-model-com#what-is-a-com-object
#include <Objbase.h>
#pragma comment(lib, "Ole32.lib")  // TODO move to cmake/bazel

#include <atomic>
#include <cassert>
#include <iostream>

// TODO handle WS_EX_ACCEPTFILES to support drag & drop
// TODO multilingual support

namespace avk {

// ------------------------ Static Functions ---------------------------------

static void enqueueMessageCOM(COMPayload& payload, const std::wstring& type,
                              const std::wstring& title,
                              const std::wstring& body, HWND window = nullptr) {
  // Acquire exclusive modification rights
  WaitForSingleObject(payload.hCanWrite, INFINITE);

  payload.messages.push_back({type + L";-;" + title, body, window});

  // Release write access
  SetEvent(payload.hCanWrite);

  // Signal COM thread that there's work
  SetEvent(payload.hHasWork);
}

static HRESULT disableNCRendering(HWND hWnd) {
  HRESULT hr = S_OK;

  DWMNCRENDERINGPOLICY ncrp = DWMNCRP_DISABLED;

  // Disable non-client area rendering on the window.
  hr = ::DwmSetWindowAttribute(hWnd, DWMWA_NCRENDERING_POLICY, &ncrp,
                               sizeof(ncrp));

  if (SUCCEEDED(hr)) {
    // ...
  }

  return hr;
}

static void enableDPIAwareness() {
  // https://learn.microsoft.com/en-us/windows/win32/hidpi/dpi-awareness-context
  // TODO: recover manifest and check if DPI awareness is enabled. If yes skip
  std::cout << "\033[93m"
            << "DPI Awareness should be enabled by the"
               " application manifest if you target Windows 10 or newer"
            << "\033[0m" << std::endl;
  static auto const pSetProcessDpiAwarenessContext =
      reinterpret_cast<decltype(SetProcessDpiAwarenessContext)*>(GetProcAddress(
          GetModuleHandleW(L"user32.dll"), "SetProcessDpiAwarenessContext"));
  // load library, not get module handle, cause it might be that I remove the
  // pragma lib
  static auto const pSetProcessDpiAwareness =
      reinterpret_cast<decltype(SetProcessDpiAwareness)*>(GetProcAddress(
          LoadLibraryW(L"Shcore.dll"), "SetProcessDpiAwareness"));

  // Windows 10+ API: enables full per-monitor V2 awareness.
  // Must be called before any window is created.
  // only in Windows 10, version 1703
  if (IsWindowsVersionOrGreater(10, 0, 15063) &&
      pSetProcessDpiAwarenessContext) {
    if (pSetProcessDpiAwarenessContext(
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2))
      return;
  }
  // Fallback for older Windows 8.1 APIs
  if (IsWindowsVersionOrGreater(8, 1, 0) && pSetProcessDpiAwareness) {
    HRESULT hr = pSetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE);
    if (SUCCEEDED(hr)) return;
  }

  // Fallback for very old systems (Vista/7)
  SetProcessDPIAware();
}

// TODO management of dynamically loaded modules
static uint32_t getDpiForHwnd(HWND hwnd) {
  // Windows 10+ API
  // user32 is linked at load time, no need for runtime check
  // HMODULE user32 = LoadLibraryW(L"user32.dll"); + GetProcAddress
  // Windows 10, version 1607
  static auto const pGetDpiForWindow =
      reinterpret_cast<decltype(GetDpiForWindow)*>(
          GetProcAddress(GetModuleHandleW(L"user32.dll"), "GetDpiForWindow"));
  if (IsWindowsVersionOrGreater(10, 0, 14393) && pGetDpiForWindow) {
    if (uint32_t dpi = pGetDpiForWindow(hwnd); dpi > 0) return dpi;
  }

  UINT dpiX = 96, dpiY = 96;
  // Fallback for Win8.1
  HMODULE shcore = LoadLibraryW(L"Shcore.dll");
  static auto const pGetDpiForMonitor =
      reinterpret_cast<decltype(GetDpiForMonitor)*>(
          GetProcAddress(shcore, "GetDpiForMonitor"));
  if (IsWindowsVersionOrGreater(8, 1, 0) && pGetDpiForMonitor) {
    HMONITOR monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    pGetDpiForMonitor(monitor, MDT_EFFECTIVE_DPI, &dpiX, &dpiY);
  }

  FreeLibrary(shcore);  // TODO
  return dpiX;
}

// WARNING: Assumes primary window doesn't have a menu
static RECT computeWindowRectForClient(HWND hWnd, RECT clientRect, DWORD style,
                                       DWORD exStyle) {
  // 1. compute dpi
  uint32_t const dpi = getDpiForHwnd(hWnd);

  // 2. adjust window rect
  RECT windowRect = clientRect;
  static auto const pAdjustWindowRectExForDpi =
      reinterpret_cast<decltype(AdjustWindowRectExForDpi)*>(GetProcAddress(
          GetModuleHandleW(L"user32.dll"), "AdjustWindowRectExForDpi"));
  if (IsWindowsVersionOrGreater(10, 0, 14393) && pAdjustWindowRectExForDpi) {
    if (pAdjustWindowRectExForDpi(&windowRect, style, false, exStyle, dpi)) {
      return windowRect;
    }
  }

  // 3. fallback: AdjustWindowRectEx with manual scaling
  if (AdjustWindowRectEx(&windowRect, style, false, exStyle) && dpi != 96) {
    // scale by current DPI ratio
    LONG width = windowRect.right - windowRect.left;
    LONG height = windowRect.bottom - windowRect.top;
    width = MulDiv(width, dpi, 96);
    height = MulDiv(height, dpi, 96);
    windowRect.right = windowRect.left + width;
    windowRect.bottom = windowRect.top + height;
  }

  return windowRect;
}

// -------------- Window Procedure Helpers ---------------

static void primaryWindowToggleFullscreen(HWND& hWnd, bool isFullscreen,
                                          WINDOWPLACEMENT* inOutPlacement) {
  if (isFullscreen) {
    // Enter fullscreen
    GetWindowPlacement(hWnd, inOutPlacement);

    HMONITOR hMon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi = {};
    mi.cbSize = sizeof(mi);
    GetMonitorInfoW(hMon, &mi);

    RECT desiredClient = mi.rcMonitor;
    RECT windowRect = computeWindowRectForClient(
        hWnd, desiredClient, primaryWindowFullscreenStyle, 0);

    SetWindowLongW(hWnd, GWL_STYLE, primaryWindowFullscreenStyle);
    SetWindowLongW(hWnd, GWL_EXSTYLE, 0);

    // resize window, refresh style, place on top, and fire WM_NCCALCSIZE
    // @alexoz12v2: Note: Why Do you need to call it twice?
    SetWindowPos(hWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                 windowRect.right - windowRect.left,
                 windowRect.bottom - windowRect.top,
                 SWP_FRAMECHANGED | SWP_NOOWNERZORDER | SWP_SHOWWINDOW);
    SetWindowPos(hWnd, HWND_TOP, mi.rcMonitor.left, mi.rcMonitor.top,
                 windowRect.right - windowRect.left,
                 windowRect.bottom - windowRect.top,
                 SWP_FRAMECHANGED | SWP_NOOWNERZORDER | SWP_SHOWWINDOW);

  } else {
    // Restore normal window
    SetWindowLongW(hWnd, GWL_STYLE, primaryWindowWindowedStyle);
    SetWindowLongW(hWnd, GWL_EXSTYLE, primaryWindowExtendedStyle);
    SetWindowPlacement(hWnd, inOutPlacement);
    RECT rc = inOutPlacement->rcNormalPosition;
    SetWindowPos(
        hWnd, nullptr, rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top,
        SWP_NOZORDER | SWP_NOOWNERZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW);
  }
}

// TODO remove
static void debugPrintPrimaryWindowStats(HWND hWnd) {
  std::cout << "/////////////////////////////////////" << std::endl;
  DWORD hwndStyle = GetWindowLongW(hWnd, GWL_STYLE);
  if (hwndStyle & WS_MAXIMIZE) {
    std::cout << "\033[71m" << "Window Maximized" << "\033[0m" << std::endl;
  }
  RECT rc{};
  GetClientRect(hWnd, &rc);
  std::cout << "GetClientRect: " << rc.right - rc.left << "x"
            << rc.bottom - rc.top << std::endl;

  GetWindowRect(hWnd, &rc);
  std::cout << "GetWindowRect: " << rc.right - rc.left << "x"
            << rc.bottom - rc.top << std::endl;

  HMONITOR hMonitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
  MONITORINFO mi = {};
  mi.cbSize = sizeof(mi);
  GetMonitorInfoW(hMonitor, &mi);
  std::cout << "GetMonitorInfoW: " << mi.rcWork.right - mi.rcWork.left << "x"
            << mi.rcWork.bottom - mi.rcWork.top << std::endl;

  RECT dwmBounds;
  if (SUCCEEDED(DwmGetWindowAttribute(hWnd, DWMWA_EXTENDED_FRAME_BOUNDS,
                                      &dwmBounds, sizeof(dwmBounds)))) {
    uint32_t visibleW = dwmBounds.right - dwmBounds.left;
    uint32_t visibleH = dwmBounds.bottom - dwmBounds.top;
    std::cout << "DwmGetWindowAttribute(DWMWA_EXTENDED_FRAME_BOUNDS): "
              << visibleW << "x" << visibleH << std::endl;
  }
  std::cout << "/////////////////////////////////////" << std::endl;
}

static void primaryWindowKeyDown(HWND hWnd, WPARAM wParam,
                                 WindowPayload* payload) {
  // TODO better with raw input
  const bool winKey =
      GetKeyState(VK_LWIN) & 0x8000 || GetKeyState(VK_RWIN) & 0x8000;
  const bool ctrlKey = GetKeyState(VK_CONTROL) & 0x8000;
  const bool altKey = GetKeyState(VK_MENU) & 0x8000;

  static constexpr uint32_t WKey = 0x57;
  static constexpr uint32_t EKey = 0x45;
  static constexpr uint32_t FKey = 0x46;
  static constexpr uint32_t GKey = 0x47;

  // priority: The more combinations, the better
  if (winKey && ctrlKey && altKey) {
    // TODO nothing
  } else if (winKey && ctrlKey) {
    // TODO nothing
  } else if (winKey && altKey) {
    // TODO nothing
  } else if (ctrlKey && altKey) {
    // TODO nothing
  } else if (winKey) {
    // TODO nothing
  } else if (altKey) {
    if (wParam == VK_RETURN) {
      // ALT+ENTER detected → toggle fullscreen
      payload->isFullscreen = !payload->isFullscreen;
      primaryWindowToggleFullscreen(hWnd, payload->isFullscreen,
                                    &payload->windowedPlacement);
    }
  } else if (ctrlKey) {
    // TODO nothing
  } else {
    // TODO remove debug
    if (wParam == EKey) {
      enqueueMessageCOM(payload->comPayload, L"NOTIFICATION", L"Test Toast",
                        L"This is a notification");
    } else if (wParam == WKey) {
      debugPrintPrimaryWindowStats(hWnd);
    } else if (wParam == FKey) {
      enqueueMessageCOM(payload->comPayload, L"OPENFILE", L"", L"", hWnd);
    } else if (wParam == GKey) {
      enqueueMessageCOM(payload->comPayload, L"OPENFOLDER", L"", L"", hWnd);
    }
  }
}

// ------------------------ Translation Unit Impl ----------------------------

LRESULT primaryWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
  // handle messages that can arrive before WM_CREATE
  if (uMsg == WM_NCCREATE) {
    CREATESTRUCTW* cs = reinterpret_cast<CREATESTRUCTW*>(lParam);
    // If this is the creation of this window (cs->hwndParent == nullptr)
    if (cs && cs->lpCreateParams && cs->hwndParent == nullptr) {
      auto* app = static_cast<WindowsApplication*>(cs->lpCreateParams);
      SetWindowLongPtrW(hWnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(app));
      // initialize windowedPlacement etc. here if you want
      app->WindowPayload.isFullscreen = false;
      app->WindowPayload.windowedPlacement = {};
      app->WindowPayload.windowedPlacement.length =
          sizeof(app->WindowPayload.windowedPlacement);
      app->WindowPayload.framebufferResized = false;
      app->WindowPayload.lastClientExtent = {};
    }
    return DefWindowProcW(hWnd, uMsg, wParam, lParam);
  }

  // might be null for some early messages
  auto* app = reinterpret_cast<WindowsApplication*>(
      GetWindowLongPtrW(hWnd, GWLP_USERDATA));
  auto* payload = app ? &app->WindowPayload : nullptr;

  switch (uMsg) {
    default:
      break;
    case WM_CREATE: {
      LOGI << "[UI::PrimaryWindow] WM_CREATE" << std::endl;
      if (app) {
        app->resumeRendering();
      }
      break;
    }
    case WM_ACTIVATE: {
      // active
      if (wParam & (WA_ACTIVE | WA_CLICKACTIVE)) {
        if (app) {
          app->resumeRendering();
        }
        if (payload && payload->isFullscreen) {
          // revert to top
          SetWindowPos(hWnd, HWND_TOP, 0, 0, 0, 0,
                       SWP_NOACTIVATE | SWP_NOSIZE | SWP_NOMOVE |
                           SWP_NOOWNERZORDER | SWP_FRAMECHANGED);
        }
      }
      // TODO somehow it clears the surface?
      // if (wParam == WA_INACTIVE) {
      //   if (app) {
      //     app->pauseRendering();
      //   }
      // }
      // we still want the default processing
      break;
    }
    case WM_CLOSE: {
      PostQuitMessage(0);
      // Suppress default window closing
      return 0;
    }

    // --- Disable double-click maximize on the non-client area
    case WM_NCLBUTTONDBLCLK:
      // Do nothing (prevents default double-click maximize).
      // Must return 0 to indicate we handled it.
      return 0;

    // --- Prevent Windows from drawing the non-client area (titlebar) when we
    // hide it
    case WM_NCPAINT:
      // We draw no non-client area (we already made client full-window in
      // WM_NCCALCSIZE). Returning 0 prevents the default caption rendering that
      // sometimes reappears.
      return 0;

    case WM_WINDOWPOSCHANGED: {
      break;
    }

    // DPI awareness code
    case WM_DPICHANGED: {
      // UINT newDpi = HIWORD(wParam);
      if (lParam) {
        const auto* const suggestedRect = reinterpret_cast<RECT*>(lParam);

        // Resize your window to the suggested rectangle
        SetWindowPos(hWnd, nullptr, suggestedRect->left, suggestedRect->top,
                     suggestedRect->right - suggestedRect->left,
                     suggestedRect->bottom - suggestedRect->top,
                     SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
      }
    }

    // before style changes
    // TODO remove?
    case WM_STYLECHANGING: {
      if (payload) {
        payload->deferResizeHandling = true;
        return 0;
      }
      break;
    }

    // if you don't use WS_THICKFRAME, then you need to implement your own logic
    // for border detection
    // Petzold, CH7 "Hit-Test Message"
    // Returning HTLEFT, HTTOPRIGHT, etc., enables automatic resizing.
    // Returning HTCAPTION allows window dragging.
    // If you want to set custom cursors manually, handle WM_SETCURSOR after
    // checking WM_NCHITTEST.
    case WM_NCHITTEST: {
      if (payload && payload->isFullscreen) {
        return HTCLIENT;
      }

      // Use system metrics so OS and scaling agree
      const int border = GetSystemMetrics(SM_CXSIZEFRAME) +
                         GetSystemMetrics(SM_CXPADDEDBORDER);
      const int caption = GetSystemMetrics(SM_CYCAPTION);

      POINT const pt = {GET_X_LPARAM(lParam), GET_Y_LPARAM(lParam)};
      RECT wr;
      GetWindowRect(hWnd, &wr);

      // Convert to window-local coords
      int const x = pt.x - wr.left;
      int const y = pt.y - wr.top;
      int const w = wr.right - wr.left;
      int const h = wr.bottom - wr.top;

      bool const left = x < border;
      bool const right = x >= (w - border);
      bool const top = y < border;
      bool const bottom = y >= (h - border);

      if (top && left) return HTTOPLEFT;
      if (top && right) return HTTOPRIGHT;
      if (bottom && left) return HTBOTTOMLEFT;
      if (bottom && right) return HTBOTTOMRIGHT;
      if (left) return HTLEFT;
      if (right) return HTRIGHT;
      if (top) return HTTOP;
      if (bottom) return HTBOTTOM;

      // Small strip at the top acts like a titlebar for dragging &
      // snap/Win+Arrows. Slightly enlarge with the border so snap works
      // reliably with invisible frame.
      if (y < caption + border) return HTCAPTION;

      return HTCLIENT;
    }

    // suppress the beep on Alt+Enter
    case WM_SYSCHAR: {
      if (wParam == VK_RETURN) {
        return 0;  // suppress system beep on Alt+Enter
      }
      break;
    }

    case WM_SYSKEYDOWN: {
      primaryWindowKeyDown(hWnd, wParam, payload);
      break;
    }

    case WM_KEYDOWN: {
      primaryWindowKeyDown(hWnd, wParam, payload);
      return 0;
    }

    case WM_SIZE: {
      if (wParam == SIZE_MINIMIZED) {
        std::cout << "WM_SIZE: MINIMIEZE" << std::endl;
      } else {
      }
      return 0;
    }

    case WM_ENTERSIZEMOVE: {
      if (app) {
        app->pauseRendering();
      }
      return 0;
    }

    case WM_EXITSIZEMOVE: {
      if (app) {
        app->resumeRendering();
      }
      return 0;
    }

    // https://learn.microsoft.com/en-us/windows/win32/dwm/customframe
    case WM_NCCALCSIZE: {
      if (wParam && lParam) {  // TODO why both?
        // Remove the standard frame — make the client area full window
        auto* sz = reinterpret_cast<NCCALCSIZE_PARAMS*>(lParam);
        sz->rgrc[0] = sz->rgrc[1];
        return 0;
      }
      break;
    }

    // since we removed manually part the window (outside the client area),
    // the window minemsions query should be properly handled manually
    case WM_GETMINMAXINFO: {
      auto* mmi = reinterpret_cast<MINMAXINFO*>(lParam);
      if (!mmi) {
        break;
      }
      HMONITOR const hMon = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);
      MONITORINFO mi = {};
      mi.cbSize = sizeof(mi);
      if (GetMonitorInfoW(hMon, &mi)) {
        if (payload && payload->isFullscreen) {
          // FULL monitor area, no constraints
          const auto [left, top, right, bottom] = mi.rcMonitor;
          mmi->ptMaxPosition.x = left;
          mmi->ptMaxPosition.y = top;
          mmi->ptMaxSize.x = right - left;
          mmi->ptMaxSize.y = bottom - top;

          // Prevent Windows from clamping window size
          mmi->ptMaxTrackSize = mmi->ptMaxSize;
          mmi->ptMinTrackSize = mmi->ptMaxSize;
        } else {
          // Normal behavior (respect taskbar)
          auto const [left, top, right, bottom] = mi.rcWork;
          const RECT monitor = mi.rcMonitor;
          mmi->ptMaxPosition.x = left - monitor.left;
          mmi->ptMaxPosition.y = top - monitor.top;
          mmi->ptMaxSize.x = right - left;
          mmi->ptMaxSize.y = bottom - top;
          mmi->ptMinTrackSize.x = 200;
          mmi->ptMinTrackSize.y = 150;
        }

        return 0;
      }

      break;
    }

    case WM_DESTROY: {
      PostQuitMessage(0);
      return 0;
    }
  }
  return DefWindowProcW(hWnd, uMsg, wParam, lParam);
}

static uint32_t __stdcall comThreadEntrypoint(void* args) {
  std::cout << "[COM Thread] Thread ID: " << GetCurrentThreadId() << std::endl;
  std::cout << "[COM Thread] Successfully got in the Entrypoint!" << std::endl;
  avkComThread(reinterpret_cast<COMPayload*>(args));
  _endthreadex(0);
  return 0;
}

HWND createPrimaryWindow(WindowsApplication* app) {
  std::cout << "[UI Thread] Thread ID: " << GetCurrentThreadId() << std::endl;
  assert(app);
  enableDPIAwareness();

  // UI COM Objects from WinRT Require a Single Threaded Apartment inside the
  // Thread which owns the Window (we are assuming that createPrimaryWindow and
  // primaryWindowEntrypoint are called from the same thread)
  avkInitApartmentSingleThreaded();

  BOOL compositionEnabled = FALSE;
  if (FAILED(DwmIsCompositionEnabled(&compositionEnabled) ||
             !compositionEnabled)) {
    LOGW << AVK_LOG_YLW "WARNING: DWM composition not enabled" AVK_LOG_RST
         << std::endl;
  }

  // Enable COM object creation on a dedicated thread
  app->WindowPayload.comPayload.messages.reserve(64);
  app->WindowPayload.comPayload.messages.clear();
  app->WindowPayload.comPayload.shutdown.store(false);
  app->WindowPayload.comPayload.hCanWrite =
      CreateEventW(nullptr, FALSE, TRUE, nullptr);
  app->WindowPayload.comPayload.hHasWork =
      CreateEventW(nullptr, TRUE, FALSE, nullptr);

  app->WindowPayload.hComThread = reinterpret_cast<HANDLE>(
      _beginthreadex(nullptr, 0, comThreadEntrypoint,
                     &app->WindowPayload.comPayload, 0, nullptr));
  if (!app->WindowPayload.hComThread)
    showErrorScreenAndExit("Coudldn't Create COM thread");

  DWORD opRes = 0;

  // TODO image from somewhere, for now steal it from explorer.exe
  HICON hIconLarge = nullptr;
  HICON hIconSmall = nullptr;
  opRes = ExtractIconExW(L"explorer.exe", 0, &hIconLarge, &hIconSmall, 1);
  if (opRes == UINT_MAX) {
    return nullptr;
  }

  // TODO More Cursors?
  HCURSOR standardCursor = LoadCursorW(nullptr, IDC_ARROW);
  if (!standardCursor) {
    return nullptr;
  }

  // background color (DeleteObject(hbrBackground)) when you are finished
  HBRUSH hbrBackground = CreateSolidBrush(RGB(30, 30, 30));
  if (!hbrBackground) {
    return nullptr;
  }

  WNDCLASSEXW standardWindowSpec{};
  standardWindowSpec.cbSize = sizeof(WNDCLASSEXW);
  standardWindowSpec.style = 0;  // TODO class styles
  standardWindowSpec.lpfnWndProc = avk::primaryWindowProc;
  standardWindowSpec.cbClsExtra = 0;  // TODO see how extra bytes can be used
  standardWindowSpec.cbWndExtra = 0;  // TODO see how extra bytes can be used
  standardWindowSpec.hInstance = GetModuleHandleW(nullptr);
  standardWindowSpec.hIcon = hIconLarge;
  standardWindowSpec.hCursor = standardCursor;
  standardWindowSpec.hbrBackground = hbrBackground;
  standardWindowSpec.lpszMenuName = nullptr;  // HMENU passed explicitly
  standardWindowSpec.lpszClassName = primaryWindowClass;
  standardWindowSpec.hIconSm = hIconSmall;

  // TODO UnregisterClassEx() when you are finished
  if (const ATOM standardWindowAtom = RegisterClassExW(&standardWindowSpec);
      !standardWindowAtom) {
    return nullptr;
  }

  // use caption and thickframe to get normal window behaviour, but suppress
  // drawing of non client area with VM_NCCALCSIZE
  constexpr int winW = 1024;  // TODO better
  constexpr int winH = 768;
  app->PrimaryWindow = CreateWindowExW(
      primaryWindowExtendedStyle, primaryWindowClass, L"Aether VK",
      primaryWindowWindowedStyle, CW_USEDEFAULT, CW_USEDEFAULT, winW, winH,
      nullptr, nullptr /*hMenu*/, nullptr, app);
  if (!app->PrimaryWindow) {
    return nullptr;
  }

  disableNCRendering(app->PrimaryWindow);

  // Force window style refresh and fire WM_NCCALCSIZE
  SetWindowPos(app->PrimaryWindow, nullptr, 0, 0, 0, 0,
               SWP_FRAMECHANGED | SWP_NOMOVE | SWP_NOSIZE);
  return app->PrimaryWindow;
}

// TODO remove: trying Windows Events
#if 0
static void CALLBACK winEventProc(HWINEVENTHOOK hook, DWORD event, HWND hwnd,
                                  LONG idObject, LONG idChild,
                                  DWORD eventThread, DWORD msEventTime) {
  std::cout << "EVENT: hook " << hook << " event " << event << " hwnd " << hwnd
            << " idObject " << idObject << " idChild " << idChild
            << " msEventTime " << msEventTime << " eventThread " << eventThread
            << std::endl;
}

static void initDebugHook() {
  HWINEVENTHOOK hook = SetWinEventHook(
      EVENT_OBJECT_LOCATIONCHANGE, EVENT_OBJECT_LOCATIONCHANGE, nullptr,
      winEventProc, 0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS);

  if (!hook)
    MessageBoxW(nullptr, L"SetWinEventHook failed", L"Hook Error",
                MB_ICONERROR);
  else
    std::cout << "Debug hook installed (EVENT_OBJECT_LOCATIONCHANGE)\n";
}
#endif

void primaryWindowMessageLoop(WindowsApplication* app) {
  assert(app && app->PrimaryWindow);
  // TODO Do something useful with windows events
  // initDebugHook();

  // TODO handle modeless (=non blocking) dialog boxes (Modal -> DialogBox,
  // Modeless -> CreateDialog)
  HWND hCurrentModelessDialog = nullptr;

  // TODO: Mate an accelerator table (IE Mapping between shortcuts and
  // actions, eg CTRL + S -> Save) (LoadAccelerators)
  HACCEL hAccel = nullptr;

  // rounded corners (for Windows 11 Build 22000)
  if (IsWindowsVersionOrGreater(10, 0, 22000)) {
    DWM_WINDOW_CORNER_PREFERENCE const cornerPreference = DWMWCP_ROUND;
    constexpr uint32_t bytes = static_cast<uint32_t>(sizeof(cornerPreference));
    if (FAILED(DwmSetWindowAttribute(app->PrimaryWindow,
                                     DWMWA_WINDOW_CORNER_PREFERENCE,
                                     &cornerPreference, bytes))) {
      LOGW << AVK_LOG_YLW
          "[UI::Warning] Coulnd't get rounded corners" AVK_LOG_RST
           << std::endl;
    };
  }

  ShowWindow(app->PrimaryWindow, SW_SHOWDEFAULT);

  LOGI << "[UI] Begin Message Loop" << std::endl;
  MSG message{};
#if 1  // TODO move signaling to render thraed to update thread
  BOOL getMessageRet = false;
  while (true) {
    // WindowPayload* payload = reinterpret_cast<WindowPayload*>(
    //     GetWindowLongPtrW(window, GWLP_USERDATA));
    // NOTE: Use PeekMessage if you want to interrupt a lengthy sync operation
    // with PM_REMOVE post message in the thread's message queue
    getMessageRet = GetMessageW(&message, nullptr, 0, 0);
    if (getMessageRet == -1) {
      break;
    }

    // modeless dialog messages have been already processed
    if (hCurrentModelessDialog != nullptr &&
        IsDialogMessageW(hCurrentModelessDialog, &message)) {
      continue;
    }
    // TODO accelerator message are handled by accelerator
    if (hAccel != nullptr &&
        TranslateAcceleratorW(app->PrimaryWindow, hAccel, &message)) {
      continue;
    }

    if (message.message == WM_QUIT) {
      break;
    }

    // (TODO: Handle Dialog box and Translate Accelerators for menus)
    TranslateMessage(&message);
    // dispatch to window procedure
    DispatchMessageW(&message);
  }
#else
  while (true) {
    // Non-blocking message check
    while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
      // Modeless dialog processing
      if (hCurrentModelessDialog != nullptr &&
          IsDialogMessageW(hCurrentModelessDialog, &message)) {
        continue;
      }

      // Accelerator processing
      if (hAccel != nullptr &&
          TranslateAcceleratorW(app->PrimaryWindow, hAccel, &message)) {
        continue;
      }

      if (message.message == WM_QUIT) {
        goto exit_loop;
      }

      TranslateMessage(&message);
      DispatchMessageW(&message);
    }

    // Always call signalStateUpdated, even if no messages
    app->signalStateUpdated();

    // Optional: sleep a tiny bit to avoid 100% CPU spin
    Sleep(1);
  }
exit_loop:
#endif

  // join with render thread
  // note: if a _beginthreadex exits by calling _endthread, which is the default
  // when it returns, you don't need to call CloseHandle. If wait fails, kill it
  app->signalStopRendering();
  DWORD constexpr waitMilliseconds = 10000;
  assert(app->RenderThread != INVALID_HANDLE_VALUE &&
         app->UpdateThread != INVALID_HANDLE_VALUE);
  if (WaitForSingleObject(app->RenderThread, waitMilliseconds) !=
      WAIT_OBJECT_0) {
    LOGW << AVK_LOG_YLW
        "[Warning] Terminating Render thread manually" AVK_LOG_RST
         << std::endl;
    TerminateThread(app->RenderThread, 1);
    CloseHandle(app->RenderThread);
    app->RenderThread = INVALID_HANDLE_VALUE;
  }

  app->signalStopUpdating();
  if (WaitForSingleObject(app->UpdateThread, waitMilliseconds) !=
      WAIT_OBJECT_0) {
    LOGW << AVK_LOG_YLW
        "[Warning] Terminating Update thread manually" AVK_LOG_RST
         << std::endl;
    TerminateThread(app->UpdateThread, 1);
    CloseHandle(app->UpdateThread);
    app->UpdateThread = INVALID_HANDLE_VALUE;
  }

  // cleanup and join COM thread
  // wake up COM thread so it can exit, then wait for it
  app->WindowPayload.comPayload.shutdown.store(true);
  if (app->WindowPayload.comPayload.hHasWork) {
    SetEvent(app->WindowPayload.comPayload.hHasWork);
  }
  if (WaitForSingleObject(app->WindowPayload.hComThread, waitMilliseconds) !=
      WAIT_OBJECT_0) {
    LOGW << AVK_LOG_YLW
        "[Warning] Terminating Render thread manually" AVK_LOG_RST
         << std::endl;
    TerminateThread(app->WindowPayload.hComThread, 1);
    CloseHandle(app->WindowPayload.hComThread);
    app->WindowPayload.hComThread = INVALID_HANDLE_VALUE;
  }
  CloseHandle(app->WindowPayload.comPayload.hCanWrite);
  CloseHandle(app->WindowPayload.comPayload.hHasWork);

  // Animate window disappearing after all threads have died:
  // slide out to the right
  AnimateWindow(app->PrimaryWindow, 300, AW_SLIDE | AW_HIDE | AW_HOR_POSITIVE);
  DestroyWindow(app->PrimaryWindow);
  UnregisterClassW(primaryWindowClass, nullptr);
  app->PrimaryWindow = nullptr;
}

}  // namespace avk