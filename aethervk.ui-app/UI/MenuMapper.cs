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
      nativeMenu.Items.Add(CreateMacWindowMenu(window));
      NativeMenu.SetMenu(window, nativeMenu);
      RegisterMacWindowMenuWhenBuilt(window);
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

  /// <summary>The macOS menu bar's "Window" menu (registered with AppKit, which adds the rest).</summary>
  public const string MacWindowMenuTitle = "Window";

  /// <summary>
  /// The standard Mac "Window" menu: Minimize (Cmd+M) and Zoom. Once registered as the windows menu
  /// (<see cref="MacWindowsMenu"/>), AppKit adds Move &amp; Resize (Fn+Ctrl+arrow tiles), Full Screen
  /// Tile and the window list to it.
  /// </summary>
  private static NativeMenuItem CreateMacWindowMenu(Window window)
  {
    var minimize = new NativeMenuItem
    {
      Header = "Minimize",
      Gesture = new KeyGesture(Key.M, KeyModifiers.Meta),
    };
    minimize.Click += (_, _) => window.WindowState = WindowState.Minimized;
    var zoom = new NativeMenuItem { Header = "Zoom" };
    zoom.Click += (_, _) =>
      window.WindowState = window.WindowState == WindowState.Maximized ? WindowState.Normal : WindowState.Maximized;
    var menu = new NativeMenu();
    menu.Items.Add(minimize);
    menu.Items.Add(zoom);
    return new NativeMenuItem { Header = MacWindowMenuTitle, Menu = menu };
  }

  /// <summary>
  /// Registers the Window menu once Avalonia has put the menu bar in place, and again whenever
  /// Avalonia rebuilds it (a changed item can replace the NSMenu): a check every second, for the
  /// window's lifetime.
  /// </summary>
  private static void RegisterMacWindowMenuWhenBuilt(Window window)
  {
    if (!OperatingSystem.IsMacOS())
      return;
    var timer = new Avalonia.Threading.DispatcherTimer { Interval = TimeSpan.FromSeconds(1) };
    int registrations = 0;
    timer.Tick += (_, _) =>
    {
      if (MacWindowsMenu.Register(MacWindowMenuTitle) == true)
        Console.WriteLine($"[Menu] Window menu registered with AppKit (#{++registrations})");
    };
    // ApplyMenu can run after the window opened (its DataContext arrives later): started now,
    // Register waits for the menu bar
    timer.Start();
    window.Closed += (_, _) => timer.Stop();
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

    if (vm.ToggleType != MenuToggle.None)
    {
      item.ToggleType = vm.ToggleType == MenuToggle.Radio
        ? NativeMenuItemToggleType.Radio
        : NativeMenuItemToggleType.CheckBox;
      item.Bind(
        NativeMenuItem.IsCheckedProperty,
        new Binding(nameof(MenuItemViewModel.IsChecked)) { Source = vm, Mode = BindingMode.TwoWay }
      );
      KeepCheckedInSync(vm, item, NativeMenuItem.IsCheckedProperty);
    }
    // a command drives IsEnabled itself (CanExecute); only a command-less entry takes the view model's
    if (vm.Command is null)
    {
      item.Bind(
        NativeMenuItem.IsEnabledProperty,
        new Binding(nameof(MenuItemViewModel.IsEnabled)) { Source = vm }
      );
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

  /// <summary>
  /// The view model's IsChecked always reaches the item, even if a local write by the menu (a click
  /// on a toggle item) replaced the binding: a refused radio choice must show the previous one.
  /// </summary>
  private static void KeepCheckedInSync(MenuItemViewModel vm, AvaloniaObject item, StyledProperty<bool> isChecked)
  {
    vm.PropertyChanged += (_, e) =>
    {
      if (e.PropertyName == nameof(MenuItemViewModel.IsChecked) && item.GetValue(isChecked) != vm.IsChecked)
        item.SetCurrentValue(isChecked, vm.IsChecked);
    };
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
    item.Bind(
      MenuItem.IsEnabledProperty,
      new Binding(nameof(MenuItemViewModel.IsEnabled)) { Source = vm }
    );

    if (vm.ToggleType != MenuToggle.None)
    {
      // the menu checks the item on click (and unchecks the rest of a radio group) before the
      // command runs; the command may then put the view model's state back, which wins
      item.ToggleType = vm.ToggleType == MenuToggle.Radio ? MenuItemToggleType.Radio : MenuItemToggleType.CheckBox;
      if (!string.IsNullOrEmpty(vm.GroupName))
        item.GroupName = vm.GroupName;
      item.Bind(
        MenuItem.IsCheckedProperty,
        new Binding(nameof(MenuItemViewModel.IsChecked)) { Source = vm, Mode = BindingMode.TwoWay }
      );
      KeepCheckedInSync(vm, item, MenuItem.IsCheckedProperty);
    }

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
