#include <Windows.h>

int APIENTRY wWinMain([[maybe_unused]] HINSTANCE hInstance,
                      [[maybe_unused]] HINSTANCE hPrevInstance,
                      [[maybe_unused]] LPWSTR lpCmdLine,
                      [[maybe_unused]] int nCmdShow) {
  MessageBoxW(nullptr, L"Hello, World!", L"Aether-Vk", MB_OK);
  return 0;
}
