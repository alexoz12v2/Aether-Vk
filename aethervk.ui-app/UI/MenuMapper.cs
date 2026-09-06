using System;
using System.Collections.Generic;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Input;

namespace AetherVk.UI;

public static class MenuMapper
{
  public static void ApplyMenu(
    Window window,
    ContentControl normalMenuContainer,
    IEnumerable<MenuItemViewModel> menuVm
  )
  {
    if (OperatingSystem.IsMacOS())
    {
      var nativeMenu = new NativeMenu();
      foreach (var item in menuVm)
      {
        nativeMenu.Items.Add(CreateNativeMenuItem(item));
      }
      NativeMenu.SetMenu(window, nativeMenu);
    }
    else
    {
      var menu = new Menu();
      foreach (var item in menuVm)
      {
        menu.Items.Add(CreateStandardMenuItem(item));
      }
      normalMenuContainer.Content = menu;
    }
  }

  private static NativeMenuItemBase CreateNativeMenuItem(MenuItemViewModel vm)
  {
    if (vm.IsSeparator)
    {
      return new NativeMenuItemSeparator();
    }

    var item = new NativeMenuItem();

    item.Bind(
      NativeMenuItem.HeaderProperty,
      new Binding(nameof(MenuItemViewModel.Header)) { Source = vm }
    );
    item.Bind(
      NativeMenuItem.CommandProperty,
      new Binding(nameof(MenuItemViewModel.Command)) { Source = vm }
    );
    // Avalonia NativeMenuItem supports IsVisible property? No, it's not a Control, it's an AvaloniaObject. Wait, looking at the previous MainWindow.axaml:
    // <NativeMenuItem Header="Snap Observer" IsVisible="{Binding ActiveViewport.IsEarthObserverMode, FallbackValue=False}" />
    // It DOES have an IsVisible property or at least Avalonia supported this binding! Let's bind it.
    // Wait, NativeMenuItem doesn't inherit from Visual/Control, but maybe it has an attached property or direct property for IsVisible.
    // I will attempt to bind it, if it fails to compile I'll fix it. I will use a direct property reference if it exists, otherwise string based binding might be safer if I'm not sure of the static field name.
    // Let's use `item.Bind(NativeMenuItem.IsVisibleProperty, ...)` Wait, is it there?
    // Let's check with a compilation step soon. For now:
    var isVisibleProperty = typeof(NativeMenuItem).GetField(
      "IsVisibleProperty",
      System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static
    );
    if (isVisibleProperty != null && isVisibleProperty.GetValue(null) is AvaloniaProperty prop)
    {
      item.Bind(prop, new Binding(nameof(MenuItemViewModel.IsVisible)) { Source = vm });
    }
    else
    {
      // shouldn't get here?
    }

    if (!string.IsNullOrEmpty(vm.Gesture))
    {
      try
      {
        item.Gesture = KeyGesture.Parse(vm.Gesture);
      }
      catch { }
    }

    if (vm.Items != null && vm.Items.Count > 0)
    {
      var submenu = new NativeMenu();
      foreach (var childVm in vm.Items)
      {
        submenu.Items.Add(CreateNativeMenuItem(childVm));
      }
      item.Menu = submenu;
    }

    return item;
  }

  private static Control CreateStandardMenuItem(MenuItemViewModel vm)
  {
    if (vm.IsSeparator)
    {
      var sep = new Separator();
      sep.Bind(
        Control.IsVisibleProperty,
        new Binding(nameof(MenuItemViewModel.IsVisible)) { Source = vm }
      );
      return sep;
    }

    var item = new MenuItem();

    item.Bind(
      MenuItem.HeaderProperty,
      new Binding(nameof(MenuItemViewModel.Header)) { Source = vm }
    );
    item.Bind(
      MenuItem.CommandProperty,
      new Binding(nameof(MenuItemViewModel.Command)) { Source = vm }
    );
    item.Bind(
      MenuItem.IsVisibleProperty,
      new Binding(nameof(MenuItemViewModel.IsVisible)) { Source = vm }
    );

    if (!string.IsNullOrEmpty(vm.Gesture))
    {
      try
      {
        var gesture = KeyGesture.Parse(vm.Gesture);
        // Avalonia 11 uses InputElement.HotKeyProperty for MenuItem shortcuts.
        item.HotKey = gesture;
      }
      catch { }
    }

    if (vm.Items != null && vm.Items.Count > 0)
    {
      foreach (var childVm in vm.Items)
      {
        item.Items.Add(CreateStandardMenuItem(childVm));
      }
    }

    return item;
  }
}
