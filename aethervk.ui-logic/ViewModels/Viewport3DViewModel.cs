using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class Viewport3DViewModel
  : TabItemViewModel,
    IActionHandler,
    IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly IFileDialogService _fileDialogService;

  public ulong PresentationEngineId { get; private set; }
  private ulong _lastRenderTaskId;

  public OperatorStack OperatorStack { get; }

  public System.Collections.ObjectModel.ObservableCollection<BillboardViewModel> Billboards { get; } =
    new();

  [ObservableProperty]
  private uint _width = 800;

  [ObservableProperty]
  private uint _height = 600;

  partial void OnWidthChanged(uint value)
  {
    throw new InvalidOperationException();
  }

  partial void OnHeightChanged(uint value)
  {
    throw new InvalidOperationException();
  }

  [ObservableProperty]
  private bool _isInitialized;

  [ObservableProperty]
  private bool _isLoading;

  [ObservableProperty]
  private bool _isAddingJet;

  [ObservableProperty]
  private bool _isEarthObserverMode;

  partial void OnIsEarthObserverModeChanged(bool value)
  {
    WeakReferenceMessenger.Default.Send(new Messages.EarthObserverModeChangedMessage(value));
  }

  public enum EarthObserverState
  {
    UpZenith,
    EarthPositioning,
    CometOrbiting
  }

  private EarthObserverState _earthObserverState = EarthObserverState.UpZenith;

  [ObservableProperty]
  private bool _hasFirstMeasurementPoint;

  [ObservableProperty]
  private float _firstMeasurementPointX;

  [ObservableProperty]
  private float _firstMeasurementPointY;

  [ObservableProperty]
  private float _firstMeasurementPointZ;

  [ObservableProperty]
  private bool _showNoIntersectionFlyout;

  [ObservableProperty]
  private float _manualMeasurementX;

  [ObservableProperty]
  private float _manualMeasurementY;

  [ObservableProperty]
  private float _manualMeasurementZ;

  [ObservableProperty]
  private ulong _sceneId;

  [ObservableProperty]
  private string _measurementIndicatorText = "";

  [ObservableProperty]
  private double _measurementIndicatorWidth = 0.0;

  [ObservableProperty]
  private bool _showMeasurementIndicator = false;

  [ObservableProperty]
  private bool _isOrbiting;

  [ObservableProperty]
  private bool _isPanning;

  // ── Radial Menu ───────────────────────────────────────────────────────────
  [ObservableProperty]
  [NotifyPropertyChangedFor(
    nameof(RadialHubLeft),
    nameof(RadialHubTop),
    nameof(RadialCometLeft),
    nameof(RadialCometTop),
    nameof(RadialBillboardLeft),
    nameof(RadialBillboardTop),
    nameof(RadialResetCameraLeft),
    nameof(RadialResetCameraTop),
    nameof(RadialSnapLeft),
    nameof(RadialSnapTop),
    nameof(RadialSnapObserverLeft),
    nameof(RadialSnapObserverTop)
  )]
  private bool _isRadialMenuOpen = false;

  [ObservableProperty]
  [NotifyPropertyChangedFor(
    nameof(RadialHubLeft),
    nameof(RadialCometLeft),
    nameof(RadialBillboardLeft),
    nameof(RadialResetCameraLeft),
    nameof(RadialSnapLeft),
    nameof(RadialSnapObserverLeft)
  )]
  private double _radialMenuX = 0.0;

  [ObservableProperty]
  [NotifyPropertyChangedFor(
    nameof(RadialHubTop),
    nameof(RadialCometTop),
    nameof(RadialBillboardTop),
    nameof(RadialResetCameraTop),
    nameof(RadialSnapTop),
    nameof(RadialSnapObserverTop)
  )]
  private double _radialMenuY = 0.0;

  /// <summary>Tracks which radial item the cursor is currently hovering over (null = none).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(
    nameof(IsCometHovered),
    nameof(IsBillboardHovered),
    nameof(IsResetCameraHovered),
    nameof(IsSnapHovered),
    nameof(IsSnapObserverHovered)
  )]
  private string? _hoveredRadialItem;

  public bool IsCometHovered => HoveredRadialItem == "comet";
  public bool IsBillboardHovered => HoveredRadialItem == "billboard";
  public bool IsResetCameraHovered => HoveredRadialItem == "resetcamera";
  public bool IsSnapHovered => HoveredRadialItem == "snap";
  public bool IsSnapObserverHovered => HoveredRadialItem == "snapobserver";

  /// <summary>Dynamic label for the comet radial menu item: "Spawn Comet" or "Destroy Comet".</summary>
  public string CometRadialLabel => !CanSpawnComet() ? "Destroy\nComet" : "Spawn\nComet";

  /// <summary>Context-sensitive tooltip for the comet radial menu item.</summary>
  public string CometRadialTooltip => !CanSpawnComet() ? "Remove comet from scene" : "Spawn a comet in the scene";

  private const double RadialRadius = 100.0;
  private const double ItemSize = 80.0;
  private const double HalfItem = ItemSize / 2.0;
  private const double HubSize = 16.0;

  // Central hub
  public double RadialHubLeft => RadialMenuX - HubSize / 2;
  public double RadialHubTop => RadialMenuY - HubSize / 2;

  // Top ("up")
  public double RadialCometLeft => RadialMenuX - HalfItem;
  public double RadialCometTop => RadialMenuY - RadialRadius - HalfItem;

  // Top-right (45°)
  private static readonly double _cos45 = Math.Cos(Math.PI / 4.0);
  public double RadialBillboardLeft => RadialMenuX + RadialRadius * _cos45 - HalfItem;
  public double RadialBillboardTop => RadialMenuY - RadialRadius * _cos45 - HalfItem;

  // Right (0°)
  public double RadialResetCameraLeft => RadialMenuX + RadialRadius - HalfItem;
  public double RadialResetCameraTop => RadialMenuY - HalfItem;

  // Bottom-right (135°)
  public double RadialSnapLeft => RadialMenuX + RadialRadius * _cos45 - HalfItem;
  public double RadialSnapTop => RadialMenuY + RadialRadius * _cos45 - HalfItem;

  // Bottom (90° down)
  public double RadialSnapObserverLeft => RadialMenuX - HalfItem;
  public double RadialSnapObserverTop => RadialMenuY + RadialRadius - HalfItem;

  partial void OnIsRadialMenuOpenChanged(bool oldValue, bool newValue)
  {
    if (newValue)
    {
      OnPropertyChanged(nameof(HasComet));
      OnPropertyChanged(nameof(CometRadialLabel));
      OnPropertyChanged(nameof(CometRadialTooltip));
      CloseRadialMenuAndSpawnCometCommand.NotifyCanExecuteChanged();
    }
  }

  public bool HasComet => !CanSpawnComet();

  public void OpenRadialMenuAt(double x, double y)
  {
    RadialMenuX = x;
    RadialMenuY = y;
    HoveredRadialItem = null;
    IsRadialMenuOpen = true;
  }

  public void CloseRadialMenu()
  {
    IsRadialMenuOpen = false;
    HoveredRadialItem = null;
  }

  /// <summary>
  /// Called when Alt+S is released. Executes the action for whatever item the cursor is hovering over,
  /// then closes the menu.
  /// </summary>
  public void CommitRadialMenuSelection()
  {
    // TODO if is playing emit breadcrumb and return?
    var item = HoveredRadialItem;
    CloseRadialMenu();

    switch (item)
    {
      // TODO remove or change
      case "comet":
        if (HasComet)
        {
          DestroyCometInternal();
        }
        else
        {
          WeakReferenceMessenger.Default.Send(new Messages.OpenSpawnCometDialogMessage());
        }
        break;
      case "billboard":
        InsertBillboardCommand.Execute(null);
        break;
      case "resetcamera":
        // Call transform to reset
        throw new NotImplementedException();
      case "snap":
        throw new NotImplementedException();
      case "snapobserver":
        throw new NotImplementedException();
    }
    // null or unrecognized: just close, no action
  }

  /// <summary>
  /// Hit-tests the pointer position against radial menu items and updates HoveredRadialItem.
  /// </summary>
  public void UpdateRadialMenuHover(double pointerX, double pointerY)
  {
    if (!IsRadialMenuOpen)
      return;

    if (HitTestItem(pointerX, pointerY, RadialCometLeft, RadialCometTop))
      HoveredRadialItem = "comet";
    else if (HitTestItem(pointerX, pointerY, RadialBillboardLeft, RadialBillboardTop))
      HoveredRadialItem = "billboard";
    else if (HitTestItem(pointerX, pointerY, RadialResetCameraLeft, RadialResetCameraTop))
      HoveredRadialItem = "resetcamera";
    else if (HitTestItem(pointerX, pointerY, RadialSnapLeft, RadialSnapTop))
      HoveredRadialItem = "snap";
    else if (
      IsEarthObserverMode
      && HitTestItem(pointerX, pointerY, RadialSnapObserverLeft, RadialSnapObserverTop)
    )
      HoveredRadialItem = "snapobserver";
    else
      HoveredRadialItem = null;
  }

  private bool HitTestItem(double px, double py, double itemLeft, double itemTop)
  {
    return px >= itemLeft && px <= itemLeft + ItemSize && py >= itemTop && py <= itemTop + ItemSize;
  }

  [RelayCommand]
  private void CloseRadialMenuCmd() => CloseRadialMenu();

  public void SnapCameraToSun()
  {
    // TODO animation toward home position
    throw new NotImplementedException();
  }

  [RelayCommand]
  private void ResetCameraFromRadial()
  {
    CloseRadialMenu();
    SnapCameraToSun();
  }

  [RelayCommand]
  private void SnapToSelectedFromRadial()
  {
    CloseRadialMenu();
    throw new NotImplementedException();
  }

  [RelayCommand(CanExecute = nameof(CanSpawnComet))]
  private void CloseRadialMenuAndSpawnComet()
  {
    CloseRadialMenu();
    // Send a message to open spawn comet dialog from main window.
    WeakReferenceMessenger.Default.Send(new Messages.OpenSpawnCometDialogMessage()
    );
  }

  private bool CanSpawnComet()
  {
    throw new NotImplementedException();
  }

  private void DestroyCometInternal()
  {
    throw new NotImplementedException();
  }

  public ulong CameraId { get; private set; }

  private static int _measurementCounter = 1;

  public IViewportRenderer? Renderer { get; set; }

  private readonly IUiThreadDispatcher _uiThreadDispatcher;
  private readonly BreadcrumbService _breadcrumbService;

  /// <summary>
  /// Opens a native file dialog to permit selecting an image off the disk.
  /// Spawns a Rust ECS entity with ScreenSpaceBillboardComponent, then creates
  /// a <see cref="BillboardViewModel"/> linked to it for Avalonia rendering.
  /// </summary>
  [RelayCommand]
  private async Task InsertBillboard()
  {
    var filters = new[] { "png", "jpg", "jpeg", "bmp" };

    var path = await _fileDialogService.ShowOpenFileDialogAsync("Select Billboard Image", filters);
    if (!string.IsNullOrEmpty(path))
    {
      try
      {
        // Place billboard at center of viewport, NDC [0..1]
        float ndcX = 0.5f;
        float ndcY = 0.5f;

        // Spawn ECS entity with ScreenSpaceBillboardComponent in Rust
        var entityId = _runtimeService.AddScreenSpaceBillboard(
          SceneId,
          path,
          ndcX,
          ndcY,
          1.0f,
          0.0f,
          1.0f,
          1,
          PresentationEngineId
        );

        // TODO remove this probably it will be handled rust side
        if (entityId == 0)
        {
          _breadcrumbService.ShowMessageAsync("Error", "Failed to create billboard entity.");
          return;
        }

        // Create the Avalonia-side ViewModel linked to this entity
        var billboard = new BillboardViewModel
        {
          EntityId = entityId,
          ImageSource = path,
          X = (Width / 2.0) - 50,
          Y = (Height / 2.0) - 50,
          Width = 100,
          Height = 100,
          ZIndex = 1,
          Opacity = 1.0,
          Scale = 1.0,
          Rotation = 0.0,
        };

        Billboards.Add(billboard);

        _breadcrumbService.ShowMessageAsync(
          "Billboard Added",
          $"Loaded image {System.IO.Path.GetFileName(path)}"
        );
      }
      catch (Exception ex)
      {
        _breadcrumbService.ShowMessageAsync("Error", $"Failed to load image: {ex.Message}");
      }
    }
  }

  /// <summary>
  /// Removes a billboard from the viewport and its backing Rust ECS entity.
  /// </summary>
  [RelayCommand]
  private void RemoveBillboard(BillboardViewModel? billboard)
  {
    if (billboard == null)
      return;

    if (billboard.EntityId != 0)
    {
      _runtimeService.RemoveScreenSpaceBillboard(SceneId, billboard.EntityId);
    }
    Billboards.Remove(billboard);
  }

  // HOME_POSITION: (AU) 0.049,0.034,0.039
  private const float HomePosX = 0.049f;
  private const float HomePosY = 0.034f;
  private const float HomePosZ = 0.039f;

  // Rotation: x=0.251,y=-0.131,z=-0.443,w=0.851
  private const float HomeRotW = 0.851f;
  private const float HomeRotX = 0.251f;
  private const float HomeRotY = -0.131f;
  private const float HomeRotZ = -0.443f;

  private void SetupViewport()
  {
    Console.WriteLine($"[SetupViewport] Called. SceneId={SceneId}  PE={PresentationEngineId}  Cam={CameraId}");
    if (PresentationEngineId == 0)
    {
      // TODO viewport numbering service
      if (_runtimeService.AddViewport(SceneId, Width, Height, "Viewport_", out var presentationEngineId, out var camera))
      {
        PresentationEngineId = presentationEngineId;
        CameraId = camera;
      }

      Console.WriteLine($"[SetupViewport] Created PE={PresentationEngineId} CameraId={CameraId}");
    }

    // TODO position camera
    Console.WriteLine($"[SetupViewport] Applying default viewport camera position: pos=({HomePosX}, {HomePosY}, {HomePosZ})");
    throw new NotImplementedException();
  }

  // TODO cleanup.
  public Viewport3DViewModel(
    INativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    IUiThreadDispatcher uiThreadDispatcher,
    IFileDialogService fileDialogService
  )
    : base("Viewport 3D")
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _uiThreadDispatcher = uiThreadDispatcher;
    _fileDialogService = fileDialogService;

    OperatorStack = new OperatorStack(new ViewportBaseOperator(this));
    IsInitialized = true;

    WeakReferenceMessenger.Default.Register<Messages.RenderFrameReadyMessage>(
      this,
      (r, m) => ((Viewport3DViewModel)r).Receive(m)
    );
    WeakReferenceMessenger.Default.Register<Messages.ToggleAddJetModeMessage>(
      this,
      (r, m) => ((Viewport3DViewModel)r).Receive(m)
    );
    // Retry SetupViewport after CreateScene completes (fires SimulationStateUpdatedMessage),
    // which resolves the timing race where IsInitialized=true fires before scene entities exist.
    WeakReferenceMessenger.Default.Register<Messages.SimulationStateUpdatedMessage>(
      this,
      (r, m) =>
      {
        var self = (Viewport3DViewModel)r;
        if (self.CameraId == 0)
        {
          if (self.SceneId == 0)
            self.SceneId = m.SceneId;
          self.SetupViewport();
        }
      }
    );

    // TODO If CAMERA CHANGES UPDATE INDICATOR
    _uiThreadDispatcher.DispatchAsync(() =>
    {
      UpdateMeasurementIndicator();
      return Task.CompletedTask;
    });

    if (IsInitialized)
    {
      SetupViewport();
    }
  }

  public void Dispose()
  {
    Stop();
    if (PresentationEngineId != 0)
    {
      _runtimeService.RemoveViewport(SceneId, PresentationEngineId);
      PresentationEngineId = 0;
      CameraId = 0;
    }
  }

  public void Receive(Messages.ToggleAddJetModeMessage message)
  {
    IsAddingJet = true;
  }

  public bool Process(AppAction action, InputState state)
  {
    return OperatorStack.Process(action, state);
  }

  [RelayCommand]
  private void ToggleCameraMode()
  {
    throw new NotImplementedException();
  }

  // TODO remove and go to nativecontrol
  public void Receive(Messages.RenderFrameReadyMessage message)
  {
    if (message.PresentationEngineId == PresentationEngineId && message.SceneId == SceneId)
    {
      _lastRenderTaskId = message.RenderGeneration;
      if (Renderer != null && Width > 0 && Height > 0)
      {
        _ = ProcessFrameAsync();
      }
    }
  }

  private bool _isProcessingFrame;

  private async Task ProcessFrameAsync()
  {
    if (_isProcessingFrame)
      return;
    _isProcessingFrame = true;

    UpdateMeasurementIndicator();

    try
    {
      nuint bufferSize = Width * Height * 4;
      IntPtr unmanagedBuffer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)bufferSize);

      try
      {
        bool downloaded = await _runtimeService.DownloadImageAsync(
          _lastRenderTaskId,
          unmanagedBuffer,
          bufferSize
        );
        if (downloaded)
        {
          await _uiThreadDispatcher.DispatchAsync(() =>
          {
            Renderer?.UpdateFrame(unmanagedBuffer, bufferSize);
            return Task.CompletedTask;
          });
        }
        else
        {
          Console.WriteLine($"[ProcessFrameAsync] DownloadImageAsync returned false for taskId={_lastRenderTaskId}. Frame skipped.");
        }
      }
      finally
      {
        System.Runtime.InteropServices.Marshal.FreeHGlobal(unmanagedBuffer);
      }
    }
    finally
    {
      _isProcessingFrame = false;
    }
  }

  private void UpdateMeasurementIndicator()
  {
    if (Width <= 0 || Height <= 0)
      return;

    throw new NotImplementedException();

    // var state = _sceneStateManager.GetOrCreateScene(SceneId);
    // if (state.EntityMap.TryGetValue(CameraId, out var entity))
    // {
    //   var camera = entity.Components.OfType<AetherVk.Logic.Models.CameraComponent>().FirstOrDefault();
    //   if (camera != null)
    //   {
    //     double target_px_width = Math.Max(24.0, Width * 0.07);

    //     if (camera.IsOrthographic)
    //     {
    //       double W_au = Width * camera.OrthoScaleFactor;
    //       if (W_au > 0)
    //       {
    //         double min_au = target_px_width * (W_au / Width);
    //         double nice_au = GetNiceNumber(min_au);
    //         MeasurementIndicatorWidth = nice_au * (Width / W_au);
    //         MeasurementIndicatorText = $"{FormatNiceNumber(nice_au)} AU";
    //         ShowMeasurementIndicator = true;
    //       }
    //       else
    //       {
    //         ShowMeasurementIndicator = false;
    //       }
    //     }
    //     else
    //     {
    //       // Perspective
    //       double fovRad = camera.Fov * Math.PI / 180.0;
    //       double hFovRad = 2.0 * Math.Atan(Math.Tan(fovRad / 2.0) * camera.AspectRatio);
    //       double W_arcsec = hFovRad * 180.0 / Math.PI * 3600.0;

    //       if (W_arcsec > 0)
    //       {
    //         double min_arcsec = target_px_width * (W_arcsec / Width);

    //         if (min_arcsec > 3600.0)
    //         {
    //           double min_deg = min_arcsec / 3600.0;
    //           double nice_deg = GetNiceNumber(min_deg);
    //           MeasurementIndicatorWidth = nice_deg * 3600.0 * (Width / W_arcsec);
    //           MeasurementIndicatorText = $"{FormatNiceNumber(nice_deg)} deg";
    //         }
    //         else if (min_arcsec > 60.0)
    //         {
    //           double min_min = min_arcsec / 60.0;
    //           double nice_min = GetNiceNumber(min_min);
    //           MeasurementIndicatorWidth = nice_min * 60.0 * (Width / W_arcsec);
    //           MeasurementIndicatorText = $"{FormatNiceNumber(nice_min)} arcmin";
    //         }
    //         else
    //         {
    //           double nice_arcsec = GetNiceNumber(min_arcsec);
    //           MeasurementIndicatorWidth = nice_arcsec * (Width / W_arcsec);
    //           MeasurementIndicatorText = $"{FormatNiceNumber(nice_arcsec)} arcsec";
    //         }

    //         ShowMeasurementIndicator = true;
    //       }
    //       else
    //       {
    //         ShowMeasurementIndicator = false;
    //       }
    //     }
    //   }
    //   else
    //   {
    //     ShowMeasurementIndicator = false;
    //   }
    // }
    // else
    // {
    //   ShowMeasurementIndicator = false;
    // }
  }

  private double GetNiceNumber(double value)
  {
    if (value <= 0)
      return 1.0;
    double exponent = Math.Floor(Math.Log10(value));
    double fraction = value / Math.Pow(10, exponent);

    double niceFraction;
    if (fraction <= 1.0)
      niceFraction = 1.0;
    else if (fraction <= 2.0)
      niceFraction = 2.0;
    else if (fraction <= 5.0)
      niceFraction = 5.0;
    else
      niceFraction = 10.0;

    return niceFraction * Math.Pow(10, exponent);
  }

  private string FormatNiceNumber(double value)
  {
    if (value >= 1.0)
      return value.ToString("0");
    return value.ToString("0.#####");
  }

  public void Stop() { }
}
