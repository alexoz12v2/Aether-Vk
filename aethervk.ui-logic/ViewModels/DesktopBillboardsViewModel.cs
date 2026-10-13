using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Collections.ObjectModel;
using System.Linq;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>The displays of the desktop (implemented by the app from Avalonia's <c>Screens</c>).</summary>
public interface IScreenInfoProvider
{
  /// <summary>Every display, desktop physical pixels.</summary>
  IReadOnlyList<ScreenInfo> Screens { get; }

  /// <summary>The display showing the main window (new billboards open there).</summary>
  ScreenInfo MainWindowScreen { get; }
}

/// <summary>
/// "Billboards" section of the Settings tab: create a billboard from an imported texture, and one
/// editable row per billboard (desktop position, size, opacity, rotation, visibility). Edits go to
/// <see cref="DesktopBillboardService"/>; changes made by dragging the billboard windows come back
/// from it, so the rows and the windows always agree.
/// </summary>
public partial class DesktopBillboardsViewModel : ObservableObject, IDisposable
{
  private readonly DesktopBillboardService _service;
  private readonly IScreenInfoProvider _screens;
  private readonly CompositeDisposable _disposables = [];
  private ImmutableArray<ImportedAsset> _textures = ImmutableArray<ImportedAsset>.Empty;

  /// <summary>Textures a billboard can show (uncompressed: the window draws their texels).</summary>
  public ObservableCollection<ImportedAsset> TextureOptions { get; } = [];

  public ObservableCollection<DesktopBillboardItemViewModel> Items { get; } = [];

  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(CreateCommand))]
  private ImportedAsset? _selectedTexture;

  [ObservableProperty]
  private bool _hasTextures;

  public DesktopBillboardsViewModel(
    DesktopBillboardService service,
    AssetLibraryService assetLibrary,
    IScreenInfoProvider screens,
    ISchedulerProvider schedulerProvider)
  {
    _service = service;
    _screens = screens;
    assetLibrary.Textures
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(OnTextures)
      .AddDisposableTo(_disposables);
    service.Billboards
      .ObserveOn(schedulerProvider.MainThread)
      .Subscribe(Sync)
      .AddDisposableTo(_disposables);
  }

  /// <summary>Design-time / derived-class constructor: a private service on one 1920×1080 display,
  /// rows kept in sync synchronously (the previewer and the headless view tests create rows).</summary>
  protected DesktopBillboardsViewModel()
  {
    _service = new DesktopBillboardService();
    _screens = new FixedScreens();
    _service.Billboards.Subscribe(Sync).AddDisposableTo(_disposables);
  }

  /// <summary>The billboards this view model edits.</summary>
  public DesktopBillboardService Service => _service;

  private bool CanCreate() => SelectedTexture is not null;

  [RelayCommand(CanExecute = nameof(CanCreate))]
  private void Create()
  {
    if (SelectedTexture is { } t)
      _service.Create(t, _screens.MainWindowScreen);
  }

  private void OnTextures(ImmutableArray<ImportedAsset> textures)
  {
    _textures = textures;
    var displayable = textures.Where(t => t.IsDisplayable).ToList();
    var keep = SelectedTexture?.Key;
    TextureOptions.Clear();
    foreach (var t in displayable)
      TextureOptions.Add(t);
    SelectedTexture = displayable.FirstOrDefault(t => t.Key == keep) ?? displayable.FirstOrDefault();
    HasTextures = displayable.Count > 0;
    foreach (var item in Items)
      item.SetTexture(Lookup(item.TextureKey));
  }

  private ImportedAsset? Lookup(string key) => _textures.FirstOrDefault(t => t.Key == key);

  private void Sync(ImmutableArray<DesktopBillboard> billboards)
  {
    for (int i = Items.Count - 1; i >= 0; i--)
      if (!billboards.Any(b => b.Id == Items[i].Id))
      {
        Items[i].Dispose();
        Items.RemoveAt(i);
      }
    foreach (var b in billboards)
    {
      var item = Items.FirstOrDefault(x => x.Id == b.Id);
      if (item is null)
      {
        item = new DesktopBillboardItemViewModel(b, _service, _screens);
        Items.Add(item);
      }
      item.Apply(b);
      item.SetTexture(Lookup(b.TextureKey));
    }
  }

  public void Dispose()
  {
    _disposables.Dispose();
    foreach (var item in Items)
      item.Dispose();
  }

  private sealed class FixedScreens : IScreenInfoProvider
  {
    private static readonly ScreenInfo Screen = new(new RectI(0, 0, 1920, 1080), 1, true);
    public IReadOnlyList<ScreenInfo> Screens { get; } = [Screen];
    public ScreenInfo MainWindowScreen => Screen;
  }
}

/// <summary>One billboard row of the Settings "Billboards" section.</summary>
public partial class DesktopBillboardItemViewModel : ObservableObject, IDisposable
{
  private readonly DesktopBillboardService _service;
  private readonly IScreenInfoProvider _screens;
  private bool _applying;

  public Guid Id { get; }

  public string TextureKey { get; private set; }

  [ObservableProperty]
  private string _textureLabel = string.Empty;

  [ObservableProperty]
  private AssetThumbnail? _thumbnail;

  /// <summary>The texture was unloaded (the window shows a placeholder until it is imported again).</summary>
  [ObservableProperty]
  private bool _isMissing;

  /// <summary>Centre, desktop px.</summary>
  [ObservableProperty]
  private double _x;

  [ObservableProperty]
  private double _y;

  [ObservableProperty]
  private double _width;

  [ObservableProperty]
  private double _height;

  /// <summary>Width and height change together (the image's aspect is kept).</summary>
  [ObservableProperty]
  private bool _isAspectLocked = true;

  [ObservableProperty]
  private double _alpha;

  [ObservableProperty]
  private double _rotation;

  [ObservableProperty]
  private bool _isVisible;

  /// <summary>"Display 2 · 640, 360" (the centre on its display), or "off screen".</summary>
  [ObservableProperty]
  private string _displayLabel = string.Empty;

  public DesktopBillboardItemViewModel(DesktopBillboard b, DesktopBillboardService service, IScreenInfoProvider screens)
  {
    Id = b.Id;
    TextureKey = b.TextureKey;
    _service = service;
    _screens = screens;
    Apply(b);
  }

  /// <summary>Shows the service's state (no edit is sent back).</summary>
  public void Apply(DesktopBillboard b)
  {
    _applying = true;
    try
    {
      TextureKey = b.TextureKey;
      X = b.CenterX;
      Y = b.CenterY;
      Width = b.WidthPx;
      Height = b.HeightPx;
      Alpha = b.Alpha;
      Rotation = b.RotationDeg;
      IsVisible = b.Visible;
      var (display, local) = DesktopBillboardGeometry.ToDisplayLocal(_screens.Screens, new PointD(b.CenterX, b.CenterY));
      DisplayLabel = display < 0 ? "off screen" : $"Display {display + 1} · {local.X:F0}, {local.Y:F0}";
    }
    finally
    {
      _applying = false;
    }
  }

  public void SetTexture(ImportedAsset? texture)
  {
    IsMissing = texture is null;
    TextureLabel = texture?.Label ?? TextureKey;
    Thumbnail = texture?.Thumbnail;
  }

  partial void OnXChanged(double value) => Send(b => b with { CenterX = value });

  partial void OnYChanged(double value) => Send(b => b with { CenterY = value });

  partial void OnWidthChanged(double oldValue, double newValue)
  {
    if (_applying)
      return;
    if (IsAspectLocked && oldValue > 0)
      Send(b => b with { WidthPx = newValue, HeightPx = b.HeightPx * newValue / oldValue });
    else
      Send(b => b with { WidthPx = newValue });
  }

  partial void OnHeightChanged(double oldValue, double newValue)
  {
    if (_applying)
      return;
    if (IsAspectLocked && oldValue > 0)
      Send(b => b with { HeightPx = newValue, WidthPx = b.WidthPx * newValue / oldValue });
    else
      Send(b => b with { HeightPx = newValue });
  }

  partial void OnAlphaChanged(double value) => Send(b => b with { Alpha = value });

  partial void OnRotationChanged(double value) => Send(b => b with { RotationDeg = value });

  partial void OnIsVisibleChanged(bool value) => Send(b => b with { Visible = value });

  [RelayCommand]
  private void Delete() => _service.Remove(Id);

  private void Send(Func<DesktopBillboard, DesktopBillboard> edit)
  {
    if (_applying)
      return;
    _service.Update(Id, edit);
  }

  public void Dispose()
  {
  }
}
