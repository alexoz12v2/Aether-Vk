#pragma once

// heavy, but this should be used by launcher only, so it's fine
// (included because vscode complains)
// clang-format off
#include "Windows.h"
// clang-format on

#include <WinDef.h>
#include <winuser.h>

#include <atomic>

#include "avk-comutils.h"
#include "render/context-vk.h"

namespace avk {

// -------------------- Constants --------------------------------------------

// TODO drop file support
inline constexpr DWORD primaryWindowExtendedStyle =
    WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
inline constexpr DWORD primaryWindowWindowedStyle = WS_OVERLAPPEDWINDOW;
inline constexpr DWORD primaryWindowFullscreenStyle = WS_POPUP | WS_VISIBLE;
inline constexpr wchar_t const* primaryWindowClass = L"Standard Window";

// -------------------- Types ------------------------------------------------

struct WindowPayload {
  VkExtent2D lastClientExtent;
  avk::ContextVk* contextVk;
  WINDOWPLACEMENT windowedPlacement;
  std::atomic<bool> isFullscreen;
  std::atomic<bool> framebufferResized;
  bool deferResizeHandling;
  COMPayload comPayload;
  HANDLE hComThread;
};

// -------------------- Functions --------------------------------------------

LRESULT primaryWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam, LPARAM lParam);

HWND createPrimaryWindow(WindowPayload* payload);

void primaryWindowMessageLoop(HWND window, WindowPayload* payload,
                              std::atomic<bool>& shouldQuit);

}  // namespace avk