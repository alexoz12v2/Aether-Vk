#include "func.h"

// windows stuff
#include <Windows.h>
#include <fcntl.h>
#include <io.h>

// C++ stuff
#include <cstdio>

extern "C" {

void avkSomeFunc() {
  // allocate new console
  AllocConsole();

  // redirect stdout and stderr to it
  FILE* fp;
  freopen_s(&fp, "CONOUT$", "w", stdout);
  freopen_s(&fp, "CONOUT$", "w", stderr);

  // disable buffering so output is immediate
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);

  // write only to this console
  printf("Hello from a private console!\r\n");
  fprintf(stderr, "Error stream also goes to the private console\r\n");

  // ensure everything is flushed
  fflush(stdout);
  fflush(stderr);

  // small delay so text is visible for a while
  Sleep(2000);

  // Detach and destroy the console
  FreeConsole();
}
}