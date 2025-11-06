#include "avk-comutils.h"

#pragma clang attribute push(__attribute__((no_sanitize("cfi"))), \
                             apply_to = any(function))
#include <Windows.h>

// more windows (shell object)
#include <Shlobj.h>

// Windows Property System
// https://learn.microsoft.com/en-us/windows/win32/api/_properties/
#include <Propvarutil.h>

// https://learn.microsoft.com/en-us/windows/win32/properties/building-property-handlers-property-schemas
// The Windows Software Development Kit (SDK) includes the header file Propkey.h
// that includes a macro definition of each of the System property keys with the
// convention of PKEY_GroupName_PropertyName. For example, PKEY_Photo_DateTaken
// is the property key for the property with canonical name
// System.Photo.DateTaken.
#include <Propkey.h>

#include <stdexcept>

#include "utils/mixins.h"

// TODO cmake/bazel
#pragma comment(lib, "WindowsApp.lib")     // For WinRT core functionality
#pragma comment(lib, "runtimeobject.lib")  // For activation factories
#pragma comment(lib, "Shell32.lib")        // SHGetFolderPathW and other Shlobj

// WinRT Foundation
// https://stackoverflow.com/questions/57546145/a-function-that-returns-auto-cannot-be-used-before-it-is-defined-error-despi
#include <winrt/windows.foundation.h>

// WinRT Foundation Collection
// necessary to use Collection Methods (Clear, Append, ..)
#include <winrt/windows.foundation.collections.h>

// WinRT: ICoreDispatcher interface implementation (necessary for pickers)
// https://stackoverflow.com/questions/57450168/getting-a-function-that-returns-auto-cannot-be-used-before-it-is-defined-whi
#include <winrt/windows.ui.core.h>

// WinRT stuff -> Notifications
#include <winrt/Windows.Data.Xml.Dom.h>
#include <winrt/Windows.UI.Notifications.h>

// WinRT -> Storage.Pickers
#include <winrt/windows.storage.h>
#include <winrt/windows.storage.pickers.h>
#pragma clang attribute pop

// check C++20
#if __cplusplus < 202002L
#  error "WinRT expects C++20"
#endif

#include <filesystem>
#include <iostream>
#include <string>

namespace avk {

// TODO: different name for non development builds
static const std::wstring s_AppId = L"AetherVK.dev";

// ------------------------------------------ Static Methods -----------------

// TODO remove
static void debugXmlDocument(
    const winrt::Windows::Data::Xml::Dom::XmlDocument& xmlDoc) {
  using namespace winrt;
  using namespace Windows::Data::Xml::Dom;

  try {
    // 1. Get the XML content as a winrt::hstring
    hstring xml_hstring = xmlDoc.GetXml();

    // 2. Convert the hstring to a wide string (std::wstring) for printing
    std::wstring xml_wstring{xml_hstring.data()};

    // 3. Print to stdout using wide character stream
    std::wcout << L"--- XML Document Content ---\n";
    std::wcout << xml_wstring << L"\n";
    std::wcout << L"--------------------------\n";
  } catch (const hresult_error& ex) {
    // Handle any WinRT errors during the process
    std::wcout << L"Error getting XML content: " << ex.message().data()
               << L"\n";
  }
}

static void showToast(std::wstring const& appId, std::wstring const& title,
                      std::wstring const& body) {
  using winrt::Windows::Data::Xml::Dom::XmlDocument;
  using winrt::Windows::UI::Notifications::ToastNotification;
  using winrt::Windows::UI::Notifications::ToastNotificationManager;
  using winrt::Windows::UI::Notifications::ToastNotifier;
  try {
    // create notifier
    ToastNotifier notifier =
        ToastNotificationManager::CreateToastNotifier(appId);

    // XML for Toast embedded here
    // clang-format off
    std::wstring xmlString =
        L"<toast>"
        L"  <visual>"
        L"    <binding template='ToastGeneric'>"
        L"      <text>" + title + L"</text>"
        L"      <text>" + body + L"</text>"
        L"    </binding>"
        L"  </visual>"
        L"</toast>";
    // clang-format on
    XmlDocument doc;
    doc.LoadXml(xmlString);

    debugXmlDocument(doc);

    // create and show Toast
    ToastNotification toast{doc};
    notifier.Show(toast);

  } catch (const winrt::hresult_error& e) {
    std::wcerr << L"Toast Failed: " << e.message().c_str() << std::endl;
  }
}

// TODO: each function which uses .get() to block and get an asynchronous result
// immediately is a C++20 coroutine, which can run on a heap allocated stack and
// be waited for with co_await. The COM thread should be able to handle more
// coroutines in flight
static winrt::Windows::Storage::StorageFile pickFileOpen(
    HWND window, const std::wstring& title = L"Select a File",
    std::vector<std::wstring> const& extensions = {L"*"}) {
  using winrt::Windows::Storage::Pickers::FileOpenPicker;
  using winrt::Windows::Storage::Pickers::PickerLocationId;
  using winrt::Windows::Storage::Pickers::PickerViewMode;

  FileOpenPicker picker;
  // TODO maybe configurable defaults?
  picker.ViewMode(PickerViewMode::List);
  picker.SuggestedStartLocation(PickerLocationId::DocumentsLibrary);
  picker.CommitButtonText(title);

  // allowed file types
  picker.FileTypeFilter().Clear();
  for (auto const& ext : extensions) {
    picker.FileTypeFilter().Append(ext);
  }

  // Atttach parent HWND to picker. Query for COM interface
  // IInitializeWithWindow and call Initialize(hwnd)
  // note: Interface is implemented by Windows 10+
  winrt::com_ptr<::IInitializeWithWindow> init;
  init = picker.as<::IInitializeWithWindow>();
  if (!init) {
    // Fallback behavior; depending on platform this might not be available
    std::wcerr << L"IInitializeWithWindow not available for Picker.\n";
  } else {
    init->Initialize(window);
  }

  auto file = picker.PickSingleFileAsync().get();
  return file;
}

static winrt::Windows::Storage::StorageFolder pickFolder(
    HWND window, const std::wstring& title = L"Select Folder") {
  using winrt::Windows::Storage::Pickers::FolderPicker;
  using winrt::Windows::Storage::Pickers::PickerLocationId;
  using winrt::Windows::Storage::Pickers::PickerViewMode;

  FolderPicker picker;
  picker.ViewMode(PickerViewMode::List);
  picker.SuggestedStartLocation(PickerLocationId::DocumentsLibrary);
  picker.CommitButtonText(title);
  picker.FileTypeFilter().Append(L"*");

  // Attach parent HWND to picker. Query for COM interface
  // IInitializeWithWindow and call Initialize(hwnd)
  // note: Interface is implemented by Windows 10+
  winrt::com_ptr<::IInitializeWithWindow> init;
  init = picker.as<::IInitializeWithWindow>();
  if (!init) {
    // Fallback behavior; depending on platform this might not be available
    std::wcerr << L"IInitializeWithWindow not available for Picker.\n";
  } else {
    init->Initialize(window);
  }

  auto folder = picker.PickSingleFolderAsync().get();
  return folder;
}

// -------------------------- Temporary App ID (Not In Final Build) ----------

// TODO: Temporary only in development build
// Creates or deletes a temporary Start Menu shortcut to register an
// AppUserModelID.

struct AppUserModelRegistration : public NonCopyable {
  std::wstring appId;
  std::wstring exePath;
  std::wstring shortcutPath;

  explicit AppUserModelRegistration(std::wstring id) : appId(std::move(id)) {
    // done by thread
    // winrt::init_apartment(winrt::apartment_type::single_threaded);
    namespace fs = std::filesystem;
    std::wcout << L"[COM Thread] : AppUserModelRegistration" << std::endl;

    // Get Start Menu path
    wchar_t startMenuPath[MAX_PATH];
    if (FAILED(SHGetFolderPathW(nullptr, CSIDL_STARTMENU, nullptr, 0,
                                startMenuPath))) {
      throw std::runtime_error("Failed to get Start Menu path");
    }

    // TODO better -> Long path: more
    exePath = []() -> std::wstring {
      wchar_t buffer[MAX_PATH];
      GetModuleFileNameW(nullptr, buffer, MAX_PATH);
      return buffer;
    }();
    std::wcout << L"[COM Thread] Executable Path: '" << exePath << "'"
               << std::endl;

    shortcutPath = fs::path(startMenuPath) / (L"Programs\\" + appId + L".lnk");
    std::wcout << L"[COM Thread] Shortcut Path: '" << shortcutPath << "'"
               << std::endl;

    // Always rebuild if stale or corrupted
    rebuildShortcut();
  }

  ~AppUserModelRegistration() { removeShortcut(); }

 private:
  void rebuildShortcut() {
    namespace fs = std::filesystem;
    if (fs::exists(shortcutPath)) {
      try {
        fs::remove(shortcutPath);
        std::wcout << L"[AppID] Removed stale shortcut: " << shortcutPath
                   << L"\n";
      } catch (const std::exception& e) {
        std::wcerr << L"[AppID] Failed to remove old shortcut: " << e.what()
                   << L"\n";
      }
    }

    createShortcut();
  }

  void createShortcut() {
    // IID_PPV_ARGS uses language extension
    // Visualize COM Class: Open OleView.NET, open CLSIDs by Name, search
    // ShellLink
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlanguage-extension-token"
    winrt::com_ptr<IShellLinkW> shellLink;
    winrt::check_hresult(CoCreateInstance(CLSID_ShellLink, nullptr,
                                          CLSCTX_INPROC_SERVER,
                                          IID_PPV_ARGS(shellLink.put())));
#pragma GCC diagnostic pop
    winrt::check_hresult(shellLink->SetPath(exePath.c_str()));
    winrt::check_hresult(shellLink->SetArguments(L""));

    winrt::com_ptr<IPropertyStore> propStore = shellLink.as<IPropertyStore>();
    PROPVARIANT pv{};
    winrt::check_hresult(InitPropVariantFromString(appId.c_str(), &pv));
    winrt::check_hresult(propStore->SetValue(PKEY_AppUserModel_ID, pv));
    winrt::check_hresult(propStore->Commit());
    PropVariantClear(&pv);

    winrt::com_ptr<IPersistFile> persistFile = shellLink.as<IPersistFile>();
    winrt::check_hresult(persistFile->Save(shortcutPath.c_str(), TRUE));

    std::wcout << L"[AppID] Created shortcut for AppUserModelID: " << appId
               << L"\n"
               << L"        Path: " << shortcutPath << L"\n";
  }

  void removeShortcut() noexcept {
    namespace fs = std::filesystem;
    try {
      if (fs::exists(shortcutPath)) {
        fs::remove(shortcutPath);
        std::wcout << L"[AppID] Deleted temporary shortcut: " << shortcutPath
                   << L"\n"
                   << std::flush;
      }
    } catch (...) {
      std::wcerr << L"[AppID] Failed to delete shortcut " << shortcutPath
                 << L"\n"
                 << std::flush;
    }
  }
};

}  // namespace avk

// ------------------------------------------ DLL Exported Symbols -----------

extern "C" {

using namespace avk;

bool avkInitApartmentSingleThreaded() {
  // C++/WinRT init (calls COInitializeEx for you -> WinRT uses COM Objects)
  // this one is for ones that require STA. OLE, and WinRT UI Elements
  try {
    winrt::init_apartment(winrt::apartment_type::single_threaded);
    return true;
  } catch (...) {
    return false;
  }
}

// TODO: Either create another hidden window, child of the parent or handle
// coroutines properly

void avkComThread(COMPayload* payload) {
  try {
    // C++/WinRT init (calls COInitializeEx for you -> WinRT uses COM Objects)
    // this one is for stuff which doesn't require STA (non UI, eg
    // Notifications)
    winrt::init_apartment(winrt::apartment_type::multi_threaded);

    // TODO Remove if not in dev build
    // - Creates a .lnk under
    // "%AppData%\Microsoft\Windows\Start Menu\Programs\AetherVK.Dev.lnk"
    // - Assigns the AppID AetherVK.Dev to the shortcut.
    // Check With: `Get-StartApps` on powershell when app is running
    AppUserModelRegistration registration{L"AetherVK.Dev"};

    // TODO
    // Check: Notification Center enabled Settings -> Notifications -> On
    // Check: Focus Assist off Settings -> System -> Focus assist

    while (true) {
      // Wait for new work or shutdown
      DWORD waitRes = WaitForSingleObject(payload->hHasWork, INFINITE);
      if (waitRes != WAIT_OBJECT_0) {
        break;  // error or shutdown externally }
      }

      // Check shutdown before consuming
      if (payload->shutdown) {
        break;
      }

      // Acquire modification rights
      WaitForSingleObject(payload->hCanWrite, INFINITE);

      if (payload->messages.empty()) {
        // No messages — reset and release lock
        ResetEvent(payload->hHasWork);
        SetEvent(payload->hCanWrite);
        continue;
      }

      // Pop one message
      COMMethod msg = std::move(payload->messages.back());
      payload->messages.pop_back();

      // If empty now, reset hHasWork
      if (payload->messages.empty()) ResetEvent(payload->hHasWork);

      // Release modification rights
      SetEvent(payload->hCanWrite);

      // === Process the message ===
      std::wcout << L"[COM Thread] Processing: " << msg.title << L" :: "
                 << msg.body << std::endl;

      static std::wstring const separator = L";-;";
      // split title into "Type;-;Title"
      size_t const off = msg.title.find_first_of(separator);
      if (off != std::wstring::npos) {
        try {
          std::wstring const type = msg.title.substr(0, off);
          std::wstring const title = msg.title.substr(off + separator.size());
          // perform COM notification here (toast, etc.)
          if (type == L"NOTIFICATION") {
            showToast(registration.appId, title, msg.body);
          } else if (type == L"OPENFILE") {
            assert(msg.parentWindow);
            // TODO handle return
            pickFileOpen(msg.parentWindow,
                         title.empty() ? L"Select a File" : title);
          } else if (type == L"OPENFOLDER") {
            assert(msg.parentWindow);
            // TODO handle return
            pickFolder(msg.parentWindow,
                       title.empty() ? L"Select a Folder" : title);
          }
        } catch (const winrt::hresult_error& e) {
          std::wcerr << L"(Loop) WinRT Error: " << e.message().c_str()
                     << std::endl;
        } catch (std::exception const& e) {
          std::cerr << "(Loop) Error: " << e.what() << std::endl;
        }
      }

      if (payload->shutdown) {
        break;
      }
    }
  } catch (const winrt::hresult_error& e) {
    std::wcerr << L"WinRT Error: " << e.message().c_str() << std::endl;
  } catch (std::exception const& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  }
  // we are not calling uninit_apartment on purpose, the process exit will
  // sweep it up
}
}