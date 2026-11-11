using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Reactive.Disposables;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.NativeInput;
using AetherVk.Logic.ViewModels;
using CommunityToolkit.Mvvm.Input;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The compact layout's panel: the observation period's preview, the tab view models it binds to
/// (the docking tree's own, never a second instance), "Import and overlay", and a compact layout
/// that is refused ending in the classic one everywhere.
/// </summary>
public sealed class CompactPanelTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_compactpanel_" + Guid.NewGuid().ToString("N"));

  public CompactPanelTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  // ── Observation period preview ─────────────────────────────────────────────

  private static double RateForPeriod(double hours) => 360.0 * 24.0 / hours;

  [Fact]
  public void ModelRangePreview_StartsNRotationPeriodsBeforeTheImageDate()
  {
    double rate = RateForPeriod(12.4);
    using var range = new RangeEntryViewModel(null, null, null, () => rate);
    Assert.Equal(string.Empty, range.ModelRangePreviewText); // no image date yet

    range.ObservationEpoch = "2021-12-05T21:30:00";
    range.ModelRotations = 3;

    var end = new DateTimeOffset(2021, 12, 5, 21, 30, 0, TimeSpan.Zero);
    var start = end - TimeSpan.FromHours(3 * 12.4);
    Assert.Equal((start, end), RangeEntryViewModel.ModelRangePreview(range.ObservationEpoch, 3, 12.4));
    string text = range.ModelRangePreviewText;
    Assert.Contains(RangeEntryViewModel.FormatUtc(start), text);
    Assert.Contains(RangeEntryViewModel.FormatUtc(end), text);
    Assert.Contains("→", text);
    Assert.True(text.IndexOf(RangeEntryViewModel.FormatUtc(start), StringComparison.Ordinal)
      < text.IndexOf(RangeEntryViewModel.FormatUtc(end), StringComparison.Ordinal), "start comes before end");
    // the range the Apply button proposes
    Assert.Equal(RangeEntryViewModel.ModelRange(end, 3, 12.4), RangeEntryViewModel.ModelRangePreview(range.ObservationEpoch, 3, 12.4));
  }

  [Fact]
  public void ModelRangePreview_FollowsThePeriod_TheRotationsAndTheDate()
  {
    double rate = RateForPeriod(12.4);
    using var range = new RangeEntryViewModel(null, null, null, () => rate);
    range.ObservationEpoch = "2021-12-05T21:30:00";
    range.ModelRotations = 3;
    var end = new DateTimeOffset(2021, 12, 5, 21, 30, 0, TimeSpan.Zero);
    var raised = new List<string?>();
    range.PropertyChanged += (_, e) => raised.Add(e.PropertyName);

    // the rotation period changes (the Comet view model tells the range)
    rate = RateForPeriod(20.0);
    range.RefreshModelSpan();
    Assert.Contains(nameof(RangeEntryViewModel.ModelRangePreviewText), raised);
    Assert.Contains(RangeEntryViewModel.FormatUtc(end - TimeSpan.FromHours(60)), range.ModelRangePreviewText);

    raised.Clear();
    range.ModelRotations = 5;
    Assert.Contains(nameof(RangeEntryViewModel.ModelRangePreviewText), raised);
    Assert.Contains(RangeEntryViewModel.FormatUtc(end - TimeSpan.FromHours(100)), range.ModelRangePreviewText);

    raised.Clear();
    range.ObservationEpoch = "2021-12-06T00:00:00";
    Assert.Contains(nameof(RangeEntryViewModel.ModelRangePreviewText), raised);
    var end2 = new DateTimeOffset(2021, 12, 6, 0, 0, 0, TimeSpan.Zero);
    Assert.Contains(RangeEntryViewModel.FormatUtc(end2 - TimeSpan.FromHours(100)), range.ModelRangePreviewText);
    Assert.Contains(RangeEntryViewModel.FormatUtc(end2), range.ModelRangePreviewText);

    // no rotation period, or a text that is not a date: nothing to preview
    rate = 0;
    range.RefreshModelSpan();
    Assert.Equal(string.Empty, range.ModelRangePreviewText);
    rate = RateForPeriod(20.0);
    range.ObservationEpoch = "not a date";
    Assert.Equal(string.Empty, range.ModelRangePreviewText);
  }

  // ── The docking tree's tab view models ─────────────────────────────────────

  // The tabs' own constructors need the whole application; the panel only has to hold the docking
  // tree's instances, so instances without state stand in for them here.
  private static T Blank<T>() where T : class => (T)RuntimeHelpers.GetUninitializedObject(typeof(T));

  private static NucleusMiniViewModel Mini(Mock<INativeRuntimeService>? runtime = null) =>
    new((runtime ?? new Mock<INativeRuntimeService>()).Object,
      pointer => new VulkanViewportControlViewModel(
        new Mock<IWindowInputRouter>().Object, new Mock<INativeInputHandlerFactory>().Object,
        (runtime ?? new Mock<INativeRuntimeService>()).Object, null!)
      {
        Role = ViewportRole.NucleusMini,
        MiniPointer = pointer,
      });

  private sealed class PinnedViewportTab() : TabItemViewModel("Viewport")
  {
    public override bool IsPinned => true;
    public override bool IsSingleInstance => true;
  }

  [Fact]
  public void Activate_BindsTheDockingTreesTabViewModels_AndNeverMakesASecondOne()
  {
    var factory = new Mock<ITabFactory>();
    var messenger = new Mock<ILayoutMessenger>();
    foreach (var type in new[] { typeof(ModelTabViewModel), typeof(TimelineTabViewModel), typeof(ImportsTabViewModel), typeof(SettingsTabViewModel) })
    {
      var t = type;
      factory.Setup(f => f.CreateScopedTab(t)).Returns(() => (RuntimeHelpers.GetUninitializedObject(t), null));
    }

    // the Comet tab is already in the layout; the others are not
    var comet = Blank<CometTabViewModel>();
    var viewportGroup = new TabGroupNodeViewModel(new PinnedViewportTab(), factory.Object, messenger.Object);
    var sideGroup = new TabGroupNodeViewModel(new UITestPanelViewModel(), factory.Object, messenger.Object);
    sideGroup.Tabs.Add(comet);
    var split = new SplitNodeViewModel(viewportGroup, sideGroup, SplitOrientation.Horizontal);
    viewportGroup.Parent = split;
    sideGroup.Parent = split;
    var docking = new DockingManagerViewModel(factory.Object, messenger.Object, split);

    using var panel = new CompactPanelViewModel(docking.EnsureTab, Mini());
    Assert.Null(panel.Comet);
    panel.Activate();

    Assert.True(panel.IsActive);
    Assert.Same(comet, panel.Comet);
    Assert.Same(docking.LayoutTabs().OfType<ModelTabViewModel>().Single(), panel.Model);
    Assert.Same(docking.LayoutTabs().OfType<TimelineTabViewModel>().Single(), panel.Timeline);
    Assert.Same(docking.LayoutTabs().OfType<ImportsTabViewModel>().Single(), panel.Imports);
    Assert.Same(docking.LayoutTabs().OfType<SettingsTabViewModel>().Single(), panel.Settings);
    factory.Verify(f => f.CreateScopedTab(typeof(CometTabViewModel)), Times.Never);
    // the new ones are parked beside the viewport's group, never in it and never selected
    Assert.Single(viewportGroup.Tabs);
    Assert.IsType<UITestPanelViewModel>(sideGroup.SelectedTab);

    // entering the compact layout again reads the same instances: nothing new is made
    var before = (panel.Comet, panel.Model, panel.Timeline, panel.Imports, panel.Settings);
    panel.Deactivate();
    Assert.False(panel.IsActive);
    panel.Activate();
    Assert.Same(before.Comet, panel.Comet);
    Assert.Same(before.Model, panel.Model);
    Assert.Same(before.Timeline, panel.Timeline);
    Assert.Same(before.Imports, panel.Imports);
    Assert.Same(before.Settings, panel.Settings);
    factory.Verify(f => f.CreateScopedTab(It.IsAny<Type>()), Times.Exactly(4));
    foreach (var type in new[] { typeof(CometTabViewModel), typeof(ModelTabViewModel), typeof(TimelineTabViewModel), typeof(ImportsTabViewModel), typeof(SettingsTabViewModel) })
      Assert.Single(docking.LayoutTabs(), t => t.GetType() == type);
  }

  [Fact]
  public void Activate_ShowsTheNucleusView_Deactivate_HidesIt()
  {
    var mini = Mini();
    using var panel = new CompactPanelViewModel(_ => null, mini);
    Assert.False(mini.IsLayoutCompact);
    panel.Activate();
    Assert.True(mini.IsLayoutCompact);
    panel.Deactivate();
    Assert.False(mini.IsLayoutCompact);
  }

  [Fact]
  public void PickPreset_MatchesTheDesignation_NotALongerNumber()
  {
    var results = new[]
    {
      new CometSearchResult { Name = "167P/CINEOS", PrimaryDesignation = "167P" },
      new CometSearchResult { Name = "67P/Churyumov-Gerasimenko", PrimaryDesignation = "67P" },
      new CometSearchResult { Name = "10P/Tempel 2", PrimaryDesignation = "10P" },
    };
    Assert.Same(results[1], CompactPanelViewModel.PickPreset(results, "67P"));
    Assert.Same(results[2], CompactPanelViewModel.PickPreset(results, "10P"));
    Assert.Null(CompactPanelViewModel.PickPreset(results, "2P"));
  }

  // ── Import and overlay ─────────────────────────────────────────────────────

  private sealed class Billboards : DesktopBillboardsViewModel
  {
    public Billboards() : base() { }
  }

  private static ImportedAsset Texture(ulong id, string key, bool displayable = true) =>
    new(id, AssetKind.Texture, Path.GetFileName(key), key, 100, 80, ImmutableArray<ulong>.Empty, null, null) { IsDisplayable = displayable };

  private static Mock<IImportsTabViewModel> Imports(ObservableCollection<ImportedAsset> textures, Action onImport)
  {
    var imports = new Mock<IImportsTabViewModel>();
    imports.Setup(i => i.TextureAssets).Returns(textures);
    imports.Setup(i => i.ImportCommand).Returns(new AsyncRelayCommand(() =>
    {
      onImport();
      return Task.CompletedTask;
    }));
    return imports;
  }

  [Fact]
  public async Task ImportAndOverlay_PicksTheTextureTheImportAdded_AndCreatesItsBillboard()
  {
    var old = Texture(1, "/data/old.png");
    var textures = new ObservableCollection<ImportedAsset> { old };
    var compressed = Texture(2, "/data/new.ktx2", displayable: false);
    var fits = Texture(3, "/data/comet_2021-12-05.fits");
    var preview = Texture(4, "/data/preview.png");
    var imports = Imports(textures, () =>
    {
      textures.Add(compressed);
      textures.Add(fits);
      textures.Add(preview);
    });
    using var billboards = new Billboards();

    var picked = await CompactPanelViewModel.ImportAndOverlayCoreAsync(imports.Object, billboards);

    // of the textures the import added: a displayable one, the FITS image first
    Assert.Same(fits, picked);
    Assert.Same(fits, billboards.SelectedTexture);
    var row = Assert.Single(billboards.Items);
    Assert.Equal(fits.Key, row.TextureKey);
  }

  [Fact]
  public async Task ImportAndOverlay_WithoutANewTexture_CreatesNothing()
  {
    var old = Texture(1, "/data/old.png");
    var textures = new ObservableCollection<ImportedAsset> { old };
    using var billboards = new Billboards();

    // cancelled, or the file was already imported
    Assert.Null(await CompactPanelViewModel.ImportAndOverlayCoreAsync(Imports(textures, () => { }).Object, billboards));
    Assert.Empty(billboards.Items);

    // only a texture a billboard cannot show
    var compressed = Texture(2, "/data/new.ktx2", displayable: false);
    Assert.Null(await CompactPanelViewModel.ImportAndOverlayCoreAsync(Imports(textures, () => textures.Add(compressed)).Object, billboards));
    Assert.Empty(billboards.Items);
  }

  /// <summary>The application's texture list reaches the Imports tab a dispatcher pass after the
  /// import returns: the texture that arrives late is still the one overlaid. A cancelled dialog
  /// (status unchanged) returns at once.</summary>
  [Fact]
  public async Task ImportAndOverlay_TextureListedAfterTheImportReturns_IsStillOverlaid()
  {
    var textures = new ObservableCollection<ImportedAsset>();
    var fits = Texture(6, "/data/late_2021-12-27.fits");
    string status = "";
    var imports = new Mock<IImportsTabViewModel>();
    imports.Setup(i => i.TextureAssets).Returns(textures);
    imports.Setup(i => i.StatusMessage).Returns(() => status);
    imports.Setup(i => i.ImportCommand).Returns(new AsyncRelayCommand(() =>
    {
      status = "Imported: late_2021-12-27.fits";
      // listed 200 ms later, as by the library's ObserveOn(MainThread)
      _ = Task.Delay(200).ContinueWith(_ => textures.Add(fits), TaskScheduler.Default);
      return Task.CompletedTask;
    }));
    using var billboards = new Billboards();
    Assert.Same(fits, await CompactPanelViewModel.ImportAndOverlayCoreAsync(imports.Object, billboards));
    Assert.Equal(fits.Key, Assert.Single(billboards.Items).TextureKey);

    // cancelled: the status line is left as it was, nothing is waited for
    var cancelled = new Mock<IImportsTabViewModel>();
    cancelled.Setup(i => i.TextureAssets).Returns(textures);
    cancelled.Setup(i => i.StatusMessage).Returns("Imported: late_2021-12-27.fits");
    cancelled.Setup(i => i.ImportCommand).Returns(new AsyncRelayCommand(() => Task.CompletedTask));
    var sw = System.Diagnostics.Stopwatch.StartNew();
    Assert.Null(await CompactPanelViewModel.ImportAndOverlayCoreAsync(cancelled.Object, billboards));
    Assert.True(sw.Elapsed < TimeSpan.FromSeconds(2), $"a cancelled import waited {sw.Elapsed}");
  }

  [Fact]
  public async Task ImportAndOverlay_APlainImage_IsOverlaidToo()
  {
    var textures = new ObservableCollection<ImportedAsset>();
    var png = Texture(5, "/data/photo.png");
    using var billboards = new Billboards();
    Assert.Same(png, await CompactPanelViewModel.ImportAndOverlayCoreAsync(Imports(textures, () => textures.Add(png)).Object, billboards));
    Assert.Equal(png.Key, Assert.Single(billboards.Items).TextureKey);
  }

  // ── A compact layout that is refused ───────────────────────────────────────

  private MainWindowViewModel MainWindow(UiLayoutService layout, PreferencesService prefs, DockingManagerViewModel docking)
  {
    var dispatcher = new Mock<IUiThreadDispatcher>();
    dispatcher.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());
    return new MainWindowViewModel(new Mock<INativeRuntimeService>().Object, new BreadcrumbService(),
      new Mock<IFileDialogService>().Object, new Mock<IWindowService>().Object, new Mock<IWindowInputRouter>().Object,
      docking, dispatcher.Object, null!, new Mock<IPlatformWindowService>().Object,
      preferences: prefs, uiLayout: layout);
  }

  /// <summary>A docking area whose only tab is not a 3D viewport: compact cannot be entered.</summary>
  private static DockingManagerViewModel DockingWithoutViewport()
  {
    var factory = new Mock<ITabFactory>();
    var messenger = new Mock<ILayoutMessenger>();
    var root = new TabGroupNodeViewModel(new UITestPanelViewModel(), factory.Object, messenger.Object);
    return new DockingManagerViewModel(factory.Object, messenger.Object, root);
  }

  private static (MenuItemViewModel Classic, MenuItemViewModel Compact) Radios(MainWindowViewModel vm)
  {
    var items = vm.MainMenu.Single(m => m.Header == "Layout").Items;
    return (items.Single(i => i.Header == "Classic"), items.Single(i => i.Header == "Compact"));
  }

  [Fact]
  public void CompactAskedFromOutside_AndRefused_EndsClassic_ServiceMenuAndPanel()
  {
    var layout = new UiLayoutService();
    var prefs = new PreferencesService(Path.Combine(_root, "preferences.json"), new InputRegistry());
    var docking = DockingWithoutViewport();
    Assert.False(docking.CanEnterViewportOnly);
    using var vm = MainWindow(layout, prefs, docking);
    var (classic, compact) = Radios(vm);
    var changes = new List<UiLayout>();
    layout.Changed += (_, l) => changes.Add(l);

    layout.Set(UiLayout.Compact); // an opened save, or the preferences at startup

    Assert.Equal(UiLayout.Classic, layout.Current);
    Assert.Contains(UiLayout.Classic, changes); // the service said so to whoever listens
    Assert.True(classic.IsChecked);
    Assert.False(compact.IsChecked);
    Assert.False(vm.IsCompactLayout);
    Assert.False(docking.IsViewportOnly);
    Assert.False(vm.CompactPanel.IsActive);
    Assert.False(vm.CompactPanel.NucleusMini.IsLayoutCompact);
    Assert.Equal("Compact layout needs the 3D viewport tab to be visible.", vm.LayoutStatus);
  }

  [Fact]
  public void StartingCompact_AndRefused_EndsClassic()
  {
    var layout = new UiLayoutService();
    layout.Set(UiLayout.Compact);
    var prefs = new PreferencesService(Path.Combine(_root, "preferences.json"), new InputRegistry());
    using var vm = MainWindow(layout, prefs, DockingWithoutViewport());
    var (classic, compact) = Radios(vm);

    Assert.Equal(UiLayout.Classic, layout.Current);
    Assert.True(classic.IsChecked);
    Assert.False(compact.IsChecked);
    Assert.False(vm.IsCompactLayout);
    Assert.False(vm.CompactPanel.IsActive);
    Assert.Equal("Compact layout needs the 3D viewport tab to be visible.", vm.LayoutStatus);
  }

  [Fact]
  public void CompactFromTheMenu_ActivatesThePanel_ClassicDeactivatesIt()
  {
    var layout = new UiLayoutService();
    var prefs = new PreferencesService(Path.Combine(_root, "preferences.json"), new InputRegistry());
    using var vm = MainWindow(layout, prefs, null!);
    Assert.False(vm.CompactPanel.IsActive);

    vm.SetLayoutCommand.Execute(UiLayout.Compact);
    Assert.True(vm.IsCompactLayout);
    Assert.True(vm.CompactPanel.IsActive);
    Assert.True(vm.CompactPanel.NucleusMini.IsLayoutCompact);

    // the panel's "More settings" button
    vm.CompactPanel.ClassicLayoutCommand!.Execute(null);
    Assert.Equal(UiLayout.Classic, layout.Current);
    Assert.False(vm.IsCompactLayout);
    Assert.False(vm.CompactPanel.IsActive);
    Assert.False(vm.CompactPanel.NucleusMini.IsLayoutCompact);
  }
}
