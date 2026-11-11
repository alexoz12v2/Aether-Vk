using System;
using System.Collections.Generic;
using System.Linq;
using Avalonia.Controls;
using Avalonia.LogicalTree;

namespace AetherVk.AppTests;

/// <summary>One menu entry as the tests see it, whichever control <c>MenuMapper</c> built it as.</summary>
/// <param name="GroupName">the radio group (in-window menu only; a native menu item has none: null)</param>
internal sealed record MenuNode(
  string? Header,
  bool IsRadio,
  Func<bool> IsChecked,
  string? GroupName,
  IReadOnlyList<MenuNode> Items
);

/// <summary>
/// The window's menu, read the way <c>UI/MenuMapper.ApplyMenu</c> built it: on macOS a
/// <see cref="NativeMenu"/> set on the window (the menu bar), elsewhere a <see cref="Menu"/> in the
/// window's menu container.
/// </summary>
internal static class MenuProbe
{
  public static IReadOnlyList<MenuNode> TopLevel(Window window)
  {
    if (OperatingSystem.IsMacOS())
    {
      var native = NativeMenu.GetMenu(window) ?? throw new InvalidOperationException("no NativeMenu on the window");
      return Entries(native).Select(FromNative).ToList();
    }
    var menu = window.GetLogicalDescendants().OfType<Menu>().Single();
    return menu.Items.OfType<MenuItem>().Select(FromMenuItem).ToList();
  }

  // a native separator is a NativeMenuItem (header "-"); the in-window Separator is no MenuItem
  private static IEnumerable<NativeMenuItem> Entries(NativeMenu menu) =>
    menu.Items.OfType<NativeMenuItem>().Where(i => i is not NativeMenuItemSeparator);

  private static MenuNode FromNative(NativeMenuItem item) =>
    new(
      item.Header,
      item.ToggleType == NativeMenuItemToggleType.Radio,
      () => item.IsChecked,
      null,
      item.Menu is { } sub ? Entries(sub).Select(FromNative).ToList() : []
    );

  private static MenuNode FromMenuItem(MenuItem item) =>
    new(
      item.Header as string,
      item.ToggleType == MenuItemToggleType.Radio,
      () => item.IsChecked,
      item.GroupName,
      item.Items.OfType<MenuItem>().Select(FromMenuItem).ToList()
    );
}
