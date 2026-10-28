using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.InteropServices;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Media.Imaging;
using Avalonia.Platform;

namespace AetherVk.UI;

/// <summary>
/// Keeps one <see cref="DesktopBillboardWindow"/> per visible billboard of
/// <see cref="DesktopBillboardService"/>, with the stacking rules of the platform:
/// <list type="bullet">
///   <item>XWayland (or <c>AETHERVK_FORCE_XWAYLAND_PATH=1</c>): unmanaged <c>override_redirect</c>
///   windows (the only way Xwayland composites their alpha), hidden while the application is in the
///   background (they would float over other applications) and shown again when it comes back;</item>
///   <item>plain X11: <c>_NET_WM_WINDOW_TYPE_UTILITY</c> + <c>_NET_WM_STATE_ABOVE</c> while the
///   application is active;</item>
///   <item>Windows / macOS: windows owned by the main window.</item>
/// </list>
/// "The application is active" is <see cref="AppActivationAggregator"/> over the main window and
/// every billboard window, so focus moving between them never blinks the billboards. On Linux the
/// pointer-sensitive part of each window is its image plus the shown gizmo handles (XFixes input
/// shape): clicks in the transparent margin reach the windows behind.
/// </summary>
public sealed class DesktopBillboardWindowManager : IDisposable
{
  private readonly DesktopBillboardService _service;
  private readonly AssetLibraryService _assets;
  private readonly INativeRuntimeService _runtime;
  private readonly IPlatformWindowService _platform;
  private readonly IScreenInfoProvider _screens;
  private readonly ISchedulerProvider _schedulers;
  private readonly Dictionary<Guid, DesktopBillboardWindow> _windows = [];
  private readonly Dictionary<(string Key, ulong Id), Bitmap> _bitmaps = [];
  // windows whose platform stacking setup ran (Avalonia/X11 creates the XID in the constructor,
  // so "has a handle" does not mean "was shown and set up")
  private readonly HashSet<DesktopBillboardWindow> _setUp = [];
  private readonly CompositeDisposable _disposables = [];
  private AppActivationAggregator? _activation;
  private Window? _main;
  private bool _appActive = true;
  private nint _root;

  public enum StackingMode
  {
    OwnedWindow,
    X11Utility,
    OverrideRedirect,
  }

  public StackingMode Stacking { get; }

  /// <summary>The open billboard windows (tests, diagnostics).</summary>
  public IReadOnlyCollection<DesktopBillboardWindow> Windows => _windows.Values;

  public DesktopBillboardWindowManager(
    DesktopBillboardService service,
    AssetLibraryService assets,
    INativeRuntimeService runtime,
    IPlatformWindowService platform,
    IScreenInfoProvider screens,
    ISchedulerProvider schedulers)
  {
    _service = service;
    _assets = assets;
    _runtime = runtime;
    _platform = platform;
    _screens = screens;
    _schedulers = schedulers;
    Stacking = DetectStacking();
  }

  public static StackingMode DetectStacking()
  {
    if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
      return StackingMode.OwnedWindow;
    bool forced = Environment.GetEnvironmentVariable("AETHERVK_FORCE_XWAYLAND_PATH") == "1";
    bool wayland = Environment.GetEnvironmentVariable("WAYLAND_DISPLAY") != null
      || string.Equals(Environment.GetEnvironmentVariable("XDG_SESSION_TYPE"), "wayland", StringComparison.OrdinalIgnoreCase);
    return forced || wayland ? StackingMode.OverrideRedirect : StackingMode.X11Utility;
  }

  /// <summary>Starts managing billboard windows for <paramref name="main"/> (call once it is open).</summary>
  public void Attach(Window main)
  {
    if (_main is not null)
      return;
    _main = main;
    _root = Stacking == StackingMode.X11Utility ? _platform.GetRootWindowHandle() : 0;
    _activation = new AppActivationAggregator(_schedulers.MainThread, main.IsActive);
    main.Activated += OnWindowActivated;
    main.Deactivated += OnWindowDeactivated;
    main.PropertyChanged += OnMainPropertyChanged;
    _activation.IsAppActive
      .ObserveOn(_schedulers.MainThread)
      .Subscribe(SetAppActive)
      .AddDisposableTo(_disposables);
    _service.Billboards
      .ObserveOn(_schedulers.MainThread)
      .Subscribe(Sync)
      .AddDisposableTo(_disposables);
    _assets.Textures
      .ObserveOn(_schedulers.MainThread)
      .Subscribe(_ => RefreshBitmaps())
      .AddDisposableTo(_disposables);
    // a FITS re-stretch rewrites the texels in place (same id): copy them again
    _assets.TexelsChanged
      .ObserveOn(_schedulers.MainThread)
      .Subscribe(ReloadTexels)
      .AddDisposableTo(_disposables);
    if (main.Screens is { } screens)
      screens.Changed += OnScreensChanged;
  }

  private void OnScreensChanged(object? sender, EventArgs e) => _service.KeepOnScreen(_screens.Screens);

  private void OnWindowActivated(object? sender, EventArgs e)
  {
    if (sender is not null)
      _activation?.Activated(sender);
  }

  private void OnWindowDeactivated(object? sender, EventArgs e)
  {
    if (sender is not null)
      _activation?.Deactivated(sender);
  }

  private void OnMainPropertyChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
  {
    if (e.Property == Window.WindowStateProperty && Stacking == StackingMode.OverrideRedirect)
      SetAppActive(_appActive);
  }

  private bool ShouldShow =>
    Stacking != StackingMode.OverrideRedirect
    || (_appActive && _main is { WindowState: not WindowState.Minimized });

  private void SetAppActive(bool active)
  {
    _appActive = active;
    foreach (var w in _windows.Values)
      ApplyStacking(w);
  }

  private void ApplyStacking(DesktopBillboardWindow w)
  {
    switch (Stacking)
    {
      case StackingMode.OverrideRedirect:
        if (ShouldShow && !w.IsVisible)
          ShowWindow(w);
        else if (!ShouldShow && w.IsVisible)
          w.Hide();
        break;
      case StackingMode.X11Utility:
        _platform.SetOverlayAbove(Handle(w), _root, _appActive);
        break;
    }
  }

  private static nint Handle(Window w) => w.TryGetPlatformHandle()?.Handle ?? 0;

  private void Sync(ImmutableArray<DesktopBillboard> billboards)
  {
    foreach (var id in _windows.Keys.ToList())
      if (!billboards.Any(b => b.Id == id && b.Visible))
        Close(id);
    foreach (var b in billboards.Where(b => b.Visible))
    {
      if (_windows.TryGetValue(b.Id, out var w))
      {
        bool textureChanged = w.Billboard.TextureKey != b.TextureKey;
        if (w.Billboard != b)
          w.Apply(b);
        if (textureChanged)
          w.SetBitmap(BitmapFor(b.TextureKey));
      }
      else
      {
        Open(b);
      }
    }
  }

  private void Open(DesktopBillboard b)
  {
    var w = new DesktopBillboardWindow(b, edited => _service.Update(edited));
    w.SetBitmap(BitmapFor(b.TextureKey));
    w.InputRegionChanged += UpdateInputRegion;
    w.Activated += OnWindowActivated;
    w.Deactivated += OnWindowDeactivated;
    _windows[b.Id] = w;
    w.Apply(b);
    if (ShouldShow)
      ShowWindow(w);
  }

  private void ShowWindow(DesktopBillboardWindow w)
  {
    if (_main is null)
      return;
    bool first = _setUp.Add(w);
    w.Show(_main);
    w.Apply(w.Billboard); // RenderScaling is known once shown
    if (!first)
    {
      UpdateInputRegion(w);
      return;
    }
    nint h = Handle(w);
    switch (Stacking)
    {
      case StackingMode.OverrideRedirect:
        _platform.SetOverlayOverrideRedirect(h);
        w.Hide();
        w.Show(_main);
        w.Apply(w.Billboard);
        break;
      case StackingMode.X11Utility:
        _platform.SetWindowAsCompanionPanel(h);
        w.Hide();
        w.Show(_main);
        w.Apply(w.Billboard);
        _platform.SetOverlayAbove(h, _root, _appActive);
        break;
    }
    UpdateInputRegion(w);
  }

  private void UpdateInputRegion(DesktopBillboardWindow w)
  {
    if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
      return;
    nint h = Handle(w);
    if (h == 0)
      return;
    var b = w.Billboard;
    _platform.SetWindowInputRegion(h, DesktopBillboardGeometry.InputRegion(b, b.GizmoMode));
  }

  private void Close(Guid id)
  {
    if (!_windows.Remove(id, out var w))
      return;
    _activation?.Closed(w);
    _setUp.Remove(w);
    w.InputRegionChanged -= UpdateInputRegion;
    w.Activated -= OnWindowActivated;
    w.Deactivated -= OnWindowDeactivated;
    w.Close();
  }

  private void RefreshBitmaps()
  {
    foreach (var w in _windows.Values)
      w.SetBitmap(BitmapFor(w.Billboard.TextureKey));
  }

  private void ReloadTexels(ulong assetId)
  {
    var stale = _bitmaps.Where(kv => kv.Key.Id == assetId).ToList();
    foreach (var kv in stale)
      _bitmaps.Remove(kv.Key);
    RefreshBitmaps();
    // the windows hold the new bitmaps now
    foreach (var kv in stale)
      kv.Value.Dispose();
  }

  /// <summary>Full-resolution bitmap of the texture with asset key <paramref name="key"/>, cached
  /// per (key, asset id); null when it is not loaded or not displayable.</summary>
  private Bitmap? BitmapFor(string key)
  {
    var asset = _assets.CurrentTextures.FirstOrDefault(t => t.Key == key && t.IsDisplayable);
    if (asset is null)
      return null;
    if (_bitmaps.TryGetValue((key, asset.Id), out var cached))
      return cached;
    var texels = _runtime.CopyAssetTexels(asset.Id);
    if (texels is null)
      return null;
    var bmp = new WriteableBitmap(
      new PixelSize(texels.Width, texels.Height),
      new Vector(96, 96),
      PixelFormat.Rgba8888,
      AlphaFormat.Unpremul);
    using (var fb = bmp.Lock())
    {
      int rowBytes = texels.Width * 4;
      for (int y = 0; y < texels.Height; y++)
        Marshal.Copy(texels.Rgba, y * rowBytes, fb.Address + y * fb.RowBytes, rowBytes);
    }
    _bitmaps[(key, asset.Id)] = bmp;
    return bmp;
  }

  public void Dispose()
  {
    _disposables.Dispose();
    foreach (var id in _windows.Keys.ToList())
      Close(id);
    if (_main is not null)
    {
      _main.Activated -= OnWindowActivated;
      _main.Deactivated -= OnWindowDeactivated;
      _main.PropertyChanged -= OnMainPropertyChanged;
      if (_main.Screens is { } screens)
        screens.Changed -= OnScreensChanged;
    }
    _activation?.Dispose();
    foreach (var b in _bitmaps.Values)
      b.Dispose();
    _bitmaps.Clear();
  }
}
