using System;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

namespace AetherVk.UI;

/// <summary>
/// Registers the menu bar's "Window" menu with AppKit (<c>NSApp.windowsMenu</c>). AppKit then
/// fills it like any Mac app's: Move &amp; Resize (the Fn+Ctrl+arrow tiles, Return to Previous
/// Size), Full Screen Tile, Bring All to Front and the window list. Avalonia's NativeMenu builds the
/// NSMenu but does not register one, so without this the app has none of those, shortcuts included.
/// </summary>
[SupportedOSPlatform("macos")]
internal static unsafe class MacWindowsMenu
{
  private const string Objc = "/usr/lib/libobjc.A.dylib";

  [DllImport(Objc, ExactSpelling = true)]
  private static extern nint objc_getClass(byte* name);

  [DllImport(Objc, ExactSpelling = true)]
  private static extern nint sel_registerName(byte* name);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true)]
  private static extern nint Send(nint receiver, nint selector);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true)]
  private static extern nint Send(nint receiver, nint selector, nint arg);

  [DllImport(Objc, EntryPoint = "objc_msgSend", ExactSpelling = true)]
  private static extern nint SendBytes(nint receiver, nint selector, byte* arg);

  private static nint Sel(ReadOnlySpan<byte> nameZ)
  {
    fixed (byte* p = nameZ)
      return sel_registerName(p);
  }

  /// <summary>
  /// Makes the submenu of the top-level item titled <paramref name="title"/> the windows menu,
  /// unless it already is. Returns null while the menu bar does not have it yet (Avalonia builds it
  /// when the window opens), true when it was (re)registered, false when nothing changed.
  /// </summary>
  public static bool? Register(string title)
  {
    nint app;
    fixed (byte* cls = "NSApplication\0"u8)
      app = Send(objc_getClass(cls), Sel("sharedApplication\0"u8));
    nint mainMenu = app == 0 ? 0 : Send(app, Sel("mainMenu\0"u8));
    if (mainMenu == 0)
      return null;
    nint nsTitle;
    var utf8 = System.Text.Encoding.UTF8.GetBytes(title + "\0");
    fixed (byte* cls = "NSString\0"u8)
    fixed (byte* t = utf8)
      nsTitle = SendBytes(objc_getClass(cls), Sel("stringWithUTF8String:\0"u8), t);
    nint item = Send(mainMenu, Sel("itemWithTitle:\0"u8), nsTitle);
    nint submenu = item == 0 ? 0 : Send(item, Sel("submenu\0"u8));
    if (submenu == 0)
      return null;
    if (Send(app, Sel("windowsMenu\0"u8)) == submenu)
      return false;
    Send(app, Sel("setWindowsMenu:\0"u8), submenu);
    return true;
  }
}
