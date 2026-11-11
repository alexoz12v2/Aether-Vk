using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Reactive.Subjects;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Layout menu of the main window: the Classic / Compact radio entries follow the
/// <see cref="UiLayoutService"/> whoever changes it, a choice from the menu is remembered in the
/// preferences, Compact is refused (Classic stays checked, the status says why) when the docking
/// area has no 3D viewport, and the headers follow the interface language.
/// </summary>
public sealed class LayoutMenuTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_layoutmenu_" + Guid.NewGuid().ToString("N"));
  private readonly PreferencesService _prefs;
  private readonly UiLayoutService _layout = new();

  public LayoutMenuTests()
  {
    Directory.CreateDirectory(_root);
    _prefs = new PreferencesService(Path.Combine(_root, "preferences.json"), new InputRegistry());
  }

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private MainWindowViewModel CreateVm(DockingManagerViewModel? docking = null, ITranslationService? translation = null)
  {
    var dispatcher = new Mock<IUiThreadDispatcher>();
    dispatcher.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());
    return new MainWindowViewModel(new Mock<INativeRuntimeService>().Object, new BreadcrumbService(),
      new Mock<IFileDialogService>().Object, new Mock<IWindowService>().Object, new Mock<IWindowInputRouter>().Object,
      docking!, dispatcher.Object, null!, new Mock<IPlatformWindowService>().Object,
      preferences: _prefs, translation: translation, uiLayout: _layout);
  }

  /// <summary>A docking area whose only tab is not a 3D viewport.</summary>
  private static DockingManagerViewModel DockingWithoutViewport()
  {
    var factory = new Mock<ITabFactory>();
    var messenger = new Mock<ILayoutMessenger>();
    var root = new TabGroupNodeViewModel(new UITestPanelViewModel(), factory.Object, messenger.Object);
    return new DockingManagerViewModel(factory.Object, messenger.Object, root);
  }

  private static MenuItemViewModel LayoutMenu(MainWindowViewModel vm) => vm.MainMenu.Single(m => m.Header == "Layout");

  private static (MenuItemViewModel Classic, MenuItemViewModel Compact) Radios(MainWindowViewModel vm)
  {
    var items = LayoutMenu(vm).Items;
    return (items.Single(i => i.Header == "Classic"), items.Single(i => i.Header == "Compact"));
  }

  [Fact]
  public void LayoutMenu_FollowsEdit_AndHoldsTwoRadios_ClassicChecked()
  {
    using var vm = CreateVm();
    Assert.Equal(["Edit", "Layout"], vm.MainMenu.Select(m => m.Header)); // no save service: no File menu
    var (classic, compact) = Radios(vm);
    Assert.All(LayoutMenu(vm).Items, i =>
    {
      Assert.Equal(MenuToggle.Radio, i.ToggleType);
      Assert.Equal(MainWindowViewModel.UiLayoutGroupName, i.GroupName);
      Assert.NotNull(i.Command);
    });
    Assert.True(classic.IsChecked);
    Assert.False(compact.IsChecked);
    Assert.False(vm.IsCompactLayout);
  }

  [Fact]
  public void Radios_FollowTheLayoutService_BothWays()
  {
    using var vm = CreateVm();
    var (classic, compact) = Radios(vm);

    _layout.Set(UiLayout.Compact); // e.g. an opened save
    Assert.True(compact.IsChecked);
    Assert.False(classic.IsChecked);
    Assert.True(vm.IsCompactLayout);

    _layout.Set(UiLayout.Classic); // e.g. File › New
    Assert.True(classic.IsChecked);
    Assert.False(compact.IsChecked);
    Assert.False(vm.IsCompactLayout);
  }

  [Fact]
  public void StartsCompact_WhenTheServiceAlreadyIs()
  {
    _layout.Set(UiLayout.Compact);
    using var vm = CreateVm();
    var (classic, compact) = Radios(vm);
    Assert.True(compact.IsChecked);
    Assert.False(classic.IsChecked);
    Assert.True(vm.IsCompactLayout);
  }

  [Fact]
  public void SetLayout_SwitchesAndPersistsThePreference()
  {
    using var vm = CreateVm();
    var (classic, compact) = Radios(vm);

    compact.Command!.Execute(null);
    Assert.Equal(UiLayout.Compact, _layout.Current);
    Assert.Equal(UiLayout.Compact, _prefs.LastUiLayout);
    Assert.True(compact.IsChecked);
    Assert.False(classic.IsChecked);
    Assert.Null(vm.LayoutStatus);
    // written to preferences.json, not only kept in memory
    var reloaded = new PreferencesService(Path.Combine(_root, "preferences.json"), new InputRegistry());
    reloaded.Load();
    Assert.Equal(UiLayout.Compact, reloaded.LastUiLayout);

    vm.SetLayoutCommand.Execute(UiLayout.Classic);
    Assert.Equal(UiLayout.Classic, _layout.Current);
    Assert.Equal(UiLayout.Classic, _prefs.LastUiLayout);
    Assert.True(classic.IsChecked);
    Assert.False(compact.IsChecked);
  }

  [Fact]
  public void Compact_IsRefused_WithoutA3DViewport_ClassicStaysChecked()
  {
    var docking = DockingWithoutViewport();
    Assert.False(docking.CanEnterViewportOnly);
    using var vm = CreateVm(docking);
    var (classic, compact) = Radios(vm);

    // the menu checks the clicked radio before its command runs
    compact.IsChecked = true;
    classic.IsChecked = false;
    compact.Command!.Execute(null);

    Assert.Equal(UiLayout.Classic, _layout.Current);
    Assert.Equal(UiLayout.Classic, _prefs.LastUiLayout);
    Assert.True(classic.IsChecked);
    Assert.False(compact.IsChecked);
    Assert.False(vm.IsCompactLayout);
    Assert.False(docking.IsViewportOnly);
    Assert.Equal("Compact layout needs the 3D viewport tab to be visible.", vm.LayoutStatus);
  }

  [Fact]
  public void Headers_FollowTheInterfaceLanguage()
  {
    var strings = new Dictionary<string, Dictionary<string, string>>
    {
      ["en"] = new() { ["Menu_Layout"] = "Layout", ["Menu_Layout_Classic"] = "Classic", ["Menu_Layout_Compact"] = "Compact" },
      ["it"] = new() { ["Menu_Layout"] = "Layout", ["Menu_Layout_Classic"] = "Classica", ["Menu_Layout_Compact"] = "Compatta" },
    };
    var culture = new BehaviorSubject<CultureInfo>(new CultureInfo("en"));
    var translation = new Mock<ITranslationService>();
    translation.Setup(t => t.CultureChanged).Returns(culture);
    translation.Setup(t => t.GetString(It.IsAny<string>())).Returns<string>(key =>
      strings[culture.Value.TwoLetterISOLanguageName].TryGetValue(key, out var s) ? s : $"[{key}]");
    using var vm = CreateVm(translation: translation.Object);
    var items = LayoutMenu(vm).Items;
    Assert.Equal(["Classic", "Compact"], items.Select(i => i.Header));

    culture.OnNext(new CultureInfo("it"));
    Assert.Equal(["Classica", "Compatta"], items.Select(i => i.Header));
    Assert.Equal("Layout", LayoutMenu(vm).Header);
  }
}
