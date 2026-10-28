using System;
using System.Reactive.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public enum EarthObserverState
{
  UpZenith,
  EarthPositioning,
  CometOrbiting,
}

public enum CameraProjectionType
{
  Perspective,
  Orthographic,
}

public partial class Viewport3DViewModel : StatefulTabViewModelBase<ViewportSession>, IActionHandler
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly IFileDialogService _fileDialogService;

  /// <summary>
  /// Authoritative camera-state and movement manager.
  /// Input operators call this — never the native runtime directly.
  /// </summary>
  public Services.CameraService CameraService { get; }

  public ulong PresentationEngineId { get; private set; }

  public OperatorStack OperatorStack { get; }

  [ObservableProperty]
  private uint _width = 800;

  [ObservableProperty]
  private uint _height = 600;

  // set while SetPixelSize writes both sides: one resize push, not an intermediate aspect
  private bool _settingPixelSize;

  partial void OnWidthChanged(uint value)
  {
    if (!_settingPixelSize && CameraId != 0 && CameraService != null)
    {
      CameraService.OnViewportResized(value, _height);
    }
  }

  partial void OnHeightChanged(uint value)
  {
    if (!_settingPixelSize && CameraId != 0 && CameraService != null)
    {
      CameraService.OnViewportResized(_width, value);
    }
  }

  /// <summary>
  /// The render target's physical pixel size (the swapchain's): both sides, then a single resize
  /// push to the camera (two pushes posed a projection with an intermediate aspect).
  /// </summary>
  public void SetPixelSize(uint width, uint height)
  {
    if (width == Width && height == Height)
      return;
    _settingPixelSize = true;
    try
    {
      Width = width;
      Height = height;
    }
    finally
    {
      _settingPixelSize = false;
    }
    if (CameraId != 0 && CameraService != null)
      CameraService.OnViewportResized(width, height);
  }

  /// <summary>Physical pixels of a <paramref name="dip"/> extent at <paramref name="renderScaling"/>
  /// (the native child window is sized the same way); at least 1.</summary>
  public static uint DipToPixels(double dip, double renderScaling) =>
    (uint)Math.Max(1.0, Math.Round(dip * (renderScaling > 0 ? renderScaling : 1.0)));

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

  [ObservableProperty]
  private CameraProjectionType _projectionType = CameraProjectionType.Perspective;

  [ObservableProperty]
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
  private bool _isOrbiting;

  [ObservableProperty]
  private bool _isPanning;

  private readonly IUiThreadDispatcher _uiThreadDispatcher;
  private readonly BreadcrumbService _breadcrumbService;
  private IDisposable? _cameraModeSubscription;

  /// <summary>
  /// The runtime drives one camera and one presentation engine: the viewport tab is unique, and
  /// pinned so closing, replacing or re-docking it never destroys and re-creates the native view.
  /// </summary>
  public override bool IsPinned => true;

  public override bool IsSingleInstance => true;

  private void SetupViewport()
  {
    if (PresentationEngineId != 0)
    {
      Console.WriteLine(
        $"[SetupViewport] PE={PresentationEngineId} CameraId={CameraId} — ready for camera init"
      );
    }
  }

  /// <summary>
  /// Called by <see cref="VulkanViewportControlViewModel"/> after
  /// <see cref="INativeRuntimeService.AddViewport"/> succeeds.
  /// </summary>
  public void OnViewportCreated(ulong presentationEngineId, ulong cameraEntityId)
  {
    PresentationEngineId = presentationEngineId;
    CameraId = cameraEntityId;
    Console.WriteLine(
      $"[Viewport3DViewModel] OnViewportCreated PE={PresentationEngineId} Cam={CameraId}"
    );
    CameraService.OnViewportReady(cameraEntityId, Width, Height);
    SetupViewport();
  }

  public ulong CameraId { get; private set; }
  public VulkanViewportControlViewModel VulkanViewModel { get; }

  /// <summary>
  /// View model for the transparent overlay window that floats above the Vulkan surface.
  /// Created via factory on construction so all overlay state is owned here and
  /// code-behind never needs to resolve services from the DI container.
  /// </summary>
  public ViewportOverlayViewModel OverlayViewModel { get; }

  /// <summary>
  /// Exposes the platform window service so that <c>Viewport3DView</c> code-behind can
  /// pass it to <c>OverlaySynchronizer</c> without performing DI lookups.
  /// </summary>
  public IPlatformWindowService PlatformWindowService { get; }
  public IWindowInputRouter InputRouter { get; }

  public Viewport3DViewModel(
    INativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    IUiThreadDispatcher uiThreadDispatcher,
    IFileDialogService fileDialogService,
    CameraService cameraService,
    Func<Viewport3DViewModel, VulkanViewportControlViewModel> vulkanVmFactory,
    Func<Viewport3DViewModel, ViewportOverlayViewModel> overlayVmFactory,
    IPlatformWindowService platformWindowService,
    IWindowInputRouter inputRouter,
    ITabStateService<ViewportSession> sessionService
  )
    : base("Viewport 3D", sessionService)
  {
    CanClose = false;
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _uiThreadDispatcher = uiThreadDispatcher;
    _fileDialogService = fileDialogService;
    CameraService = cameraService;
    VulkanViewModel = vulkanVmFactory(this);
    OverlayViewModel = overlayVmFactory(this);
    PlatformWindowService = platformWindowService;
    InputRouter = inputRouter;

    OperatorStack = new OperatorStack(new ViewportBaseOperator(this));

    // Keep EarthObserverState and IsEarthObserverMode in sync with CameraService.
    // IsEarthObserverMode is true only for EarthPosition — UpZenith is a zenith-offset
    // sub-mode but does not require SPK earth data to be live.
    // Subscribe to SimulationStateUpdated via Rx instead of WeakReferenceMessenger.
    // Fires from the native callback thread — dispatch to UI thread before mutating properties.
    _cameraModeSubscription = CameraService.CameraModeChanged.Subscribe(mode =>
    {
      IsEarthObserverMode = mode == CameraMode.EarthPosition;
      EarthObserverState = mode switch
      {
        CameraMode.EarthPosition => EarthObserverState.EarthPositioning,
        CameraMode.UpZenith => EarthObserverState.UpZenith,
        CameraMode.CometOrbiting => EarthObserverState.CometOrbiting,
        _ => EarthObserverState.UpZenith,
      };
    });

    // TODO If CAMERA CHANGES UPDATE INDICATOR
    _uiThreadDispatcher.DispatchAsync(() =>
    {
      return Task.CompletedTask;
    });

    if (!IsInitialized)
    {
      SetupViewport();
      IsInitialized = true;
    }
  }

  public override void Dispose()
  {
    _cameraModeSubscription?.Dispose();
    _cameraModeSubscription = null;
    OverlayViewModel.Dispose();
    VulkanViewModel.Dispose();
    if (PresentationEngineId != 0)
    {
      _runtimeService.RemoveViewport(PresentationEngineId);
      PresentationEngineId = 0;
      CameraId = 0;
    }
    base.Dispose(); // IsActive = false + tears down session Rx subscriptions
  }

  public bool Process(AppAction action, InputState state)
  {
    return OperatorStack.Process(action, state);
  }

  [RelayCommand]
  private void ToggleCameraMode()
  {
    CameraService.CycleCameraMode();
  }
}
