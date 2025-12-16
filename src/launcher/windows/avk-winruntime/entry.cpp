
#include "Windows.h"

BOOL WINAPI DllMain([[maybe_unused]] HINSTANCE hInstDll, DWORD fdwReason,
                    [[maybe_unused]] LPVOID fImpLoad) {
  // fImpLoad = 0 if explicitly loaded, != 0 if implicitly loaded
  switch (fdwReason) {
    case DLL_PROCESS_ATTACH:
      // the DLL is being mapped into the process address space
      break;
    case DLL_THREAD_ATTACH:
      // a thread is being created
      break;
    case DLL_THREAD_DETACH:
      // a thread is exiting cleanly
      break;
    case DLL_PROCESS_DETACH:
      // the DLL is being unmapped from the process' address space
      break;
    default:;
  }
  return TRUE;  // used only for DLL_PROCESS_DETACH
}