#include <Windows.h>

// Windows related stuff
#include <Windowsx.h>  // GET_X_LPARAM
#include <basetsd.h>
#include <errhandlingapi.h>
#include <winspool.h>
#include <winuser.h>

// ExtractIconExW
#include <shellapi.h>

#include <climits>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

static void printError() {
  DWORD const err = GetLastError();
  wchar_t* messageBuffer = nullptr;

  FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                     FORMAT_MESSAGE_IGNORE_INSERTS,
                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                 reinterpret_cast<wchar_t*>(&messageBuffer), 0, NULL);
  MessageBoxW(nullptr, messageBuffer, L"Error", MB_OK);
  LocalFree(messageBuffer);
}

static LRESULT standardWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam,
                                  LPARAM lParam) {
  switch (uMsg) {
    case WM_CREATE:
      // Animate window appearing: slide from left to right
      AnimateWindow(hWnd, 300, AW_SLIDE | AW_ACTIVATE | AW_HOR_POSITIVE);
      return 0;

    case WM_CLOSE:
      // Animate window disappearing: slide out to the right
      AnimateWindow(hWnd, 300, AW_SLIDE | AW_HIDE | AW_HOR_POSITIVE);
      DestroyWindow(hWnd);
      return 0;

    // if you don't use WS_THICKFRAME, then you need to implement your own logic
    // for border detection
    // Petzold, CH7 "Hit-Test Message"
    // Returning HTLEFT, HTTOPRIGHT, etc., enables automatic resizing.
    // Returning HTCAPTION allows window dragging.
    // If you want to set custom cursors manually, handle WM_SETCURSOR after
    // checking WM_NCHITTEST.
    case WM_NCHITTEST: {
      const LONG border = 6;  // TODO better: width of resize area in pixel
      RECT winRect{};
      GetWindowRect(hWnd, &winRect);
      int const x = GET_X_LPARAM(lParam);
      int const y = GET_Y_LPARAM(lParam);
      bool const resizeWidth =
          (x >= winRect.left && x < winRect.left + border) ||
          (x < winRect.right && x >= winRect.right - border);
      bool const resizeHeight =
          (y >= winRect.top && y < winRect.top + border) ||
          (y < winRect.bottom && y >= winRect.bottom - border);
      if (resizeWidth && resizeHeight) {  // corners
        if (x < winRect.left + border && y < winRect.top + border)
          return HTTOPLEFT;
        if (x >= winRect.right - border && y < winRect.top + border)
          return HTTOPRIGHT;
        if (x < winRect.left + border && y >= winRect.bottom - border)
          return HTBOTTOMLEFT;
        else
          return HTBOTTOMRIGHT;
      } else if (resizeWidth) {
        return (x < winRect.left + border) ? HTLEFT : HTRIGHT;
      } else if (resizeHeight) {
        return (y < winRect.top + border) ? HTTOP : HTBOTTOM;
      }
      // dragging?
      return HTCAPTION;
    }

    // Basic paint function
    case WM_PAINT: {
      PAINTSTRUCT ps{};
      HDC hdc = BeginPaint(hWnd, &ps);

      RECT rect{};
      GetClientRect(hWnd, &rect);  // note: client area, not the whole thing

      // Draw top bar
      RECT titleBar = {0, 0, rect.right, 30};
      HBRUSH hBrush = CreateSolidBrush(RGB(45, 45, 45));
      FillRect(hdc, &titleBar, hBrush);

      // TODO better: Draw buttons (min, max, close)
      int btnWidth = 45;
      RECT btnClose = {rect.right - btnWidth, 0, rect.right, 30};
      RECT btnMax = {rect.right - 2 * btnWidth, 0, rect.right - btnWidth, 30};
      RECT btnMin = {rect.right - 3 * btnWidth, 0, rect.right - 2 * btnWidth,
                     30};

      FillRect(hdc, &btnClose, (HBRUSH)(COLOR_3DFACE + 1));
      FillRect(hdc, &btnMax, (HBRUSH)(COLOR_3DFACE + 1));
      FillRect(hdc, &btnMin, (HBRUSH)(COLOR_3DFACE + 1));

      DrawTextW(hdc, L"-", -1, &btnMin, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
      DrawTextW(hdc, L"□", -1, &btnMax, DT_CENTER | DT_VCENTER | DT_SINGLELINE);
      DrawTextW(hdc, L"✕", -1, &btnClose,
                DT_CENTER | DT_VCENTER | DT_SINGLELINE);

      EndPaint(hWnd, &ps);
      return 0;
    }

    // TODO BETTER: ASSUMES BUTTON POSITION
    case WM_LBUTTONDOWN: {
      int x = GET_X_LPARAM(lParam);
      int y = GET_Y_LPARAM(lParam);
      RECT rect;
      GetClientRect(hWnd, &rect);
      int btnWidth = 45;

      if (y <= 30) {
        if (x >= rect.right - btnWidth) {
          PostMessage(hWnd, WM_CLOSE, 0, 0);
        } else if (x >= rect.right - 2 * btnWidth) {
          ShowWindow(hWnd, IsZoomed(hWnd) ? SW_RESTORE : SW_MAXIMIZE);
        } else if (x >= rect.right - 3 * btnWidth) {
          ShowWindow(hWnd, SW_MINIMIZE);
        }
      }
      return 0;
    }

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      return DefWindowProcW(hWnd, uMsg, wParam, lParam);
  }
}

int APIENTRY wWinMain([[maybe_unused]] HINSTANCE hInstance,
                      [[maybe_unused]] HINSTANCE hPrevInstance,
                      [[maybe_unused]] LPWSTR lpCmdLine,
                      [[maybe_unused]] int nCmdShow) {
  // MessageBoxW(nullptr, L"Hello, World!", L"Aether-Vk", MB_OK);
  // return 0;

  // TODO image from somewhere, for now steal it from explorer.exe
  HICON hIconLarge = nullptr;
  HICON hIconSmall = nullptr;
  if (ExtractIconExW(L"explorer.exe", 0, &hIconLarge, &hIconSmall, 1) ==
      UINT_MAX) {
    printError();
  }

  // TODO More Cursors?
  HCURSOR standardCursor = LoadCursorW(nullptr, IDC_ARROW);
  if (!standardCursor) {
    printError();
  }

  // background color (DeleteObject(hbrBackground)) when you are finished
  HBRUSH hbrBackground = CreateSolidBrush(RGB(30, 30, 30));
  if (!hbrBackground) {
    printError();
  }

  // Menu definition (probably we will rmove it if we can't style the part
  // outside the client area)
  static int constexpr ID_FILE_OPEN = 1;
  HMENU hMenu = CreateMenu();  // DestroyMenu
  if (!hMenu) {
    printError();
  }
  HMENU hFileSubMenu = CreatePopupMenu();  // DestroyPopupMenu
  if (!hFileSubMenu) {
    printError();
  }
  if (!AppendMenuW(hFileSubMenu, MF_STRING, ID_FILE_OPEN, L"Open")) {
    printError();
  }
  if (!AppendMenuW(hMenu, MF_STRING | MF_POPUP, (UINT_PTR)hFileSubMenu,
                   L"File")) {
    printError();
  }

  WNDCLASSEXW standardWindowSpec{};
  standardWindowSpec.cbSize = sizeof(WNDCLASSEXW);
  standardWindowSpec.style = 0;  // TODO class styles
  standardWindowSpec.lpfnWndProc = standardWindowProc;
  standardWindowSpec.cbClsExtra = 0;  // TODO see how extra bytes can be used
  standardWindowSpec.cbWndExtra = 0;  // TODO see how extra bytes can be used
  standardWindowSpec.hInstance = GetModuleHandleW(nullptr);
  standardWindowSpec.hIcon = hIconLarge;
  standardWindowSpec.hCursor = standardCursor;
  standardWindowSpec.hbrBackground = hbrBackground;
  standardWindowSpec.lpszMenuName = nullptr;  // HMENU passed explicitly
  standardWindowSpec.lpszClassName = L"Standard Window";
  standardWindowSpec.hIconSm = hIconSmall;

  // UnregisterClassEx() when you are finished
  ATOM standardWindowAtom = RegisterClassExW(&standardWindowSpec);
  if (!standardWindowAtom) {
    printError();
  }
  // WS_POPUP vs WS_EX_OVERLAPPED = no top bar, no default content, hence you
  // need to handle WS_PAINT, and size CW_USEDEFAULT doesn't work
  // WS_POPUP needs an explicit size
  // TODO WS_POPUP doesn't support HMENU! (either use WS_EX_OVERLAPPED or window
  // rendered on vulkan, see how it works on mac and linux)
  int winX = 100, winY = 100, winW = 1024, winH = 768;  // TODO better
  HWND hMainWindow =
      CreateWindowExW(0, L"Standard Window", L"Aether VK", WS_POPUP, winX, winY,
                      winW, winH, nullptr, hMenu, nullptr, nullptr);
  if (!hMainWindow) {
    printError();
    return 1;
  }

  // TODO: Controller thread (handle messages) and display thread different,
  // hence call ShowWindowAsync with event synchronization on window creation
  ShowWindow(hMainWindow, SW_SHOWDEFAULT);

  // send the first WM_PAINT message to the window to fill the client area
  if (!UpdateWindow(hMainWindow)) {
    printError();
    return 1;
  }
  MSG message{};
  BOOL getMessageRet = false;

  // TODO handle modeless (=non blocking) dialog boxes (Modal -> DialogBox,
  // Modeless -> CreateDialog)
  HWND hCurrentModelessDialog = nullptr;

  // TODO: Mate an accelerator table (IE Mapping between shortcuts and
  // actions, eg CTRL + S -> Save) (LoadAccelerators)
  HACCEL hAccel = nullptr;

  while ((getMessageRet = GetMessageW(&message, nullptr, 0, 0)) != 0) {
    if (getMessageRet == -1) {
      printError();
      break;
    }
    // modeless dialog messages have been already processed
    if (hCurrentModelessDialog != nullptr &&
        IsDialogMessageW(hCurrentModelessDialog, &message)) {
      continue;
    }
    // TODO accelerator message are handled by accelerator
    if (hAccel != nullptr &&
        TranslateAcceleratorW(hMainWindow, hAccel, &message)) {
      continue;
    }

    // NOTE: Use PeekMessage if you want to interrupt a lengthy sync operation
    // with PM_REMOVE post message in the thread's message queue (TODO: Handle
    // Dialog box and Translate Accelerators for menus)
    TranslateMessage(&message);
    // dispatch to window procedure
    DispatchMessageW(&message);
  }

  return message.wParam;
}
