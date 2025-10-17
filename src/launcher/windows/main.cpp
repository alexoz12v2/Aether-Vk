#include <Windows.h>

// Windows related stuff
#include <Windowsx.h>  // GET_X_LPARAM
#include <basetsd.h>
#include <errhandlingapi.h>
#include <wingdi.h>
#include <winspool.h>
#include <winuser.h>

// ExtractIconExW
#include <shellapi.h>

#include <climits>
#include <memory>
#include <mutex>
#include <ratio>
#include <thread>
#include <type_traits>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include "fiber/jobs.h"

static LRESULT standardWindowProc(HWND hWnd, UINT uMsg, WPARAM wParam,
                                  LPARAM lParam);

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

// TODO better
template <typename F>
static void syncLog(F&& f) {
  static std::mutex coutMtx;
  std::lock_guard<std::mutex> lk{coutMtx};
  f();
}

// -------------------------------------------------------

static std::atomic<bool> g_quit{false};
static std::atomic<HWND> g_window(nullptr);

struct Payload {
  int value;
  int id;
};

void physicsStep(void* userData, [[maybe_unused]] std::string const& name) {
  syncLog([]() { std::cout << "Started Physics Job" << std::endl; });
  Payload* p = reinterpret_cast<Payload*>(userData);
  // simulate CPU work
  std::this_thread::sleep_for(std::chrono::milliseconds(5));
  p->value = p->value * 2 + 1;
}

void renderPrep(void* userData, [[maybe_unused]] std::string const& name) {
  [[maybe_unused]] Payload* p = reinterpret_cast<Payload*>(userData);
  syncLog([]() { std::cout << "Started Render Job" << std::endl; });
  // simulate command buffer recording cost
  std::this_thread::sleep_for(std::chrono::milliseconds(3));
}

// Blender's window style:
//  DWORD style = parent_window ?
//                    WS_POPUPWINDOW | WS_CAPTION | WS_MAXIMIZEBOX |
//                    WS_MINIMIZEBOX | WS_SIZEBOX : WS_OVERLAPPEDWINDOW;
//
//  /* Forces owned windows onto taskbar and allows minimization. */
//  DWORD extended_style = parent_window ? WS_EX_APPWINDOW : 0;
//
//  if (dialog) {
//    /* When we are ready to make windows of this type:
//     * style = WS_POPUPWINDOW | WS_CAPTION
//     * extended_style = WS_EX_DLGMODALFRAME | WS_EX_TOPMOST
//     */
//  }

void messageThreadProc() {
  syncLog([] { std::cout << "Started UI Thread" << std::endl; });

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
  g_window.store(CreateWindowExW(0, L"Standard Window", L"Aether VK", WS_POPUP,
                                 winX, winY, winW, winH, nullptr, hMenu,
                                 nullptr, nullptr));
  if (!g_window.load()) {
    printError();
    g_quit.store(true);
    return;
  }
  HWND hMainWindow = g_window.load();

  // TODO: Controller thread (handle messages) and display thread different,
  // hence call ShowWindowAsync with event synchronization on window creation
  // ShowWindow(hMainWindow, SW_SHOWDEFAULT);
  ShowWindowAsync(hMainWindow, SW_SHOW);

  // send the first WM_PAINT message to the window to fill the client area
  UpdateWindow(hMainWindow);

  MSG message{};
  BOOL getMessageRet = false;

  // TODO handle modeless (=non blocking) dialog boxes (Modal -> DialogBox,
  // Modeless -> CreateDialog)
  HWND hCurrentModelessDialog = nullptr;

  // TODO: Mate an accelerator table (IE Mapping between shortcuts and
  // actions, eg CTRL + S -> Save) (LoadAccelerators)
  HACCEL hAccel = nullptr;
  while (!g_quit.load()) {
    // NOTE: Use PeekMessage if you want to interrupt a lengthy sync operation
    // with PM_REMOVE post message in the thread's message queue
    getMessageRet = GetMessageW(&message, nullptr, 0, 0);
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

    if (message.message == WM_QUIT) {
      g_quit.store(true);
      break;
    }

    // (TODO: Handle Dialog box and Translate Accelerators for menus)
    TranslateMessage(&message);
    // dispatch to window procedure
    DispatchMessageW(&message);
  }
}

void frameProducer(avk::Scheduler* sched, [[maybe_unused]] HWND hMainWindow,
                   int jobBufferSize) {
  std::cout << "Started Render Thread" << std::endl;
  auto* p = new Payload();
  avk::Job* phys = new avk::Job[jobBufferSize]();
  avk::Job* render = new avk::Job[jobBufferSize]();
  uint32_t jobIdx = 0;
  while (!g_quit.load()) {
    uint32_t const nextJobIdx = (jobIdx + 1) % jobBufferSize;
    uint32_t const prevJobIdx = jobIdx == 0 ? 0 : jobIdx - 1;
    // create payloads and submit jobs in a burst (simulates a frame)
    p->value = 100;
    p->id = 1000;

    AVK_JOB(&phys[jobIdx], physicsStep, p, avk::JobPriority::Medium,
            "PhysicsChunk");
    AVK_JOB(&render[jobIdx], physicsStep, p, avk::JobPriority::Low,
            "RenderPrep");

    // render depends on physics
    if (jobIdx != 0) {
      phys[jobIdx].addDepencency(&phys[prevJobIdx - 1]);
      // unless you recycle stuff from the previous frame, this is not necessary
      // render[jobIdx].addDepencency(&render[prevJobIdx]);
    }
    render[jobIdx].addDepencency(&phys[jobIdx]);

    // submit bot (only the top of the DAG)
    sched->safeSubmitTask(&phys[jobIdx]);
    sched->safeSubmitTask(&render[jobIdx]);

    if (nextJobIdx < jobIdx) {
      sched->waitFor(&render[nextJobIdx]);
      // we'll recycle the job objects in the next loop iteration, therefore you
      // need to be sure that the job mutex is not locked somewhere for any job!
      // WARNING: Assuming continuations are zeroed out, remaining dependencies
      // are at 0
    }
  }
  sched->waitUntilAllTasksDone();

  delete[] phys;
  delete[] render;
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

    case WM_APP:  // custom message from worker threads
      // InvalidateRect(hWnd, nullptr, TRUE);
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
      // Double buffering: Create an off-screen HDC
      PAINTSTRUCT ps{};
      HDC hdc = BeginPaint(hWnd, &ps);

      RECT rect{};
      GetClientRect(hWnd, &rect);  // note: client area, not the whole thing
                                   // (WS_POPUP, so it's fine)

      // create offscreen buffer
      HDC memDC = CreateCompatibleDC(hdc);
      HBITMAP memBmp = CreateCompatibleBitmap(hdc, rect.right, rect.bottom);
      HGDIOBJ oldBmp = SelectObject(memDC, memBmp);

      // background
      HBRUSH bg = CreateSolidBrush(RGB(30, 30, 30));
      FillRect(memDC, &rect, bg);

      // titlebar
      RECT titleBar = {0, 0, rect.right, 30};
      HBRUSH bar = CreateSolidBrush(RGB(45, 45, 45));
      FillRect(memDC, &titleBar, bar);

      // Draw buttons (min, max, close)
      int btnWidth = 45;
      RECT btnClose = {rect.right - btnWidth, 0, rect.right, 30};
      RECT btnMax = {rect.right - 2 * btnWidth, 0, rect.right - btnWidth, 30};
      RECT btnMin = {rect.right - 3 * btnWidth, 0, rect.right - 2 * btnWidth,
                     30};
      FillRect(memDC, &btnClose, (HBRUSH)(COLOR_3DFACE + 1));
      FillRect(memDC, &btnMax, (HBRUSH)(COLOR_3DFACE + 1));
      FillRect(memDC, &btnMin, (HBRUSH)(COLOR_3DFACE + 1));

      DrawTextW(memDC, L"-", -1, &btnMin,
                DT_CENTER | DT_VCENTER | DT_SINGLELINE);
      DrawTextW(memDC, L"□", -1, &btnMax,
                DT_CENTER | DT_VCENTER | DT_SINGLELINE);
      DrawTextW(memDC, L"✕", -1, &btnClose,
                DT_CENTER | DT_VCENTER | DT_SINGLELINE);

      // Swap buffers
      BitBlt(hdc, 0, 0, rect.right, rect.bottom, memDC, 0, 0, SRCCOPY);

      // cleanup
      SelectObject(memDC, oldBmp);
      DeleteObject(memBmp);
      DeleteDC(memDC);
      DeleteObject(bar);
      DeleteObject(bg);

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
          PostMessageW(hWnd, WM_CLOSE, 0, 0);
        } else if (x >= rect.right - 2 * btnWidth) {
          ShowWindow(hWnd, IsZoomed(hWnd) ? SW_RESTORE : SW_MAXIMIZE);
        } else if (x >= rect.right - 3 * btnWidth) {
          ShowWindow(hWnd, SW_MINIMIZE);
        }
      }
      return 0;
    }

    case WM_SIZE: {
      // generate manually a WM_PAINT
      InvalidateRect(hWnd, nullptr, TRUE);
      return 0;
    }

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
    default:
      return DefWindowProcW(hWnd, uMsg, wParam, lParam);
  }
}

int WINAPI wWinMain([[maybe_unused]] HINSTANCE hInstance,
                    [[maybe_unused]] HINSTANCE hPrevInstance,
                    [[maybe_unused]] LPWSTR lpCmdLine,
                    [[maybe_unused]] int nCmdShow) {
// GUI Console window
#ifdef AVK_DEBUG
  AllocConsole();
  FILE* fDummy;
  freopen_s(&fDummy, "CONOUT$", "w", stdout);
  freopen_s(&fDummy, "CONOUT$", "w", stderr);
  std::ios::sync_with_stdio();
#endif

  std::cout << "Created window" << std::endl;

  constexpr size_t jobPoolSize = 1024;
  constexpr size_t fiberCount = 64;
  avk::MPMCQueue<avk::Job*> highQ(jobPoolSize);
  avk::MPMCQueue<avk::Job*> medQ(jobPoolSize);
  avk::MPMCQueue<avk::Job*> lowQ(jobPoolSize);

  unsigned const hw = std::thread::hardware_concurrency();
  unsigned const workerCount = hw > 3 ? hw - 3 : 1;
  avk::Scheduler sched(fiberCount, &highQ, &medQ, &lowQ, workerCount);
  sched.start();

  // start Win32 message thread (dedicated)
  std::thread msgThread(messageThreadProc);
  while (!g_window.load()) {
    std::this_thread::yield();
  }

  // frame producer on its own thread (could be main thread in real app)
  const int chunksPerFrame = 8;
  std::thread producer(frameProducer, &sched, g_window.load(), chunksPerFrame);

  producer.join();
  sched.waitUntilAllTasksDone();

  // TODO sleep with condition variable
  while (!g_quit.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(16));
  }

  sched.shutdown();

  if (msgThread.joinable()) msgThread.join();

  return 0;
}
