#pragma once

#include "avk-winlib-macros.h"

// TODO: find a way not not enrage vscode's clangd and remove include
#include <Windows.h>

#include <atomic>
#include <string>
#include <vector>

namespace avk {

// TODO better (they should be C types so that we don't have ABI problems)
struct COMMethod {
  std::wstring title;
  std::wstring body;
  HWND parentWindow;
};

// WARNING: Assumes ABI compatibility between C++17 libc and C++20 libc
// TODO: Switch to a C-like type
struct COMPayload {
  std::atomic<bool> shutdown;
  HANDLE hCanWrite;  // mutual exclusion event
  HANDLE hHasWork;   // signals available work
  std::vector<COMMethod> messages;
};

}  // namespace avk

// C linkage functions do NOT have a namespace!
extern "C" {

AVK_COMUTILS_API void avkComThread(avk::COMPayload* payload);
AVK_COMUTILS_API bool avkInitApartmentSingleThreaded();

}
