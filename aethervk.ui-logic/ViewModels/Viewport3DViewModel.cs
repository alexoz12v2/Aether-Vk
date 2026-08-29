using System;
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
  CometOrbiting
}

public enum CameraProjectionType
{
  Perspective,
  Orthographic
}


public partial class Viewport3DViewModel
  : StatefulTabViewModelBase<ViewportSession>,
    IActionHandler
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly IFileDialogService _fileDialogService;
  private readonly Services.IViewportRegistry _viewportRegistry;

  /// <summary>
  /// Authoritative camera-state and movement manager.
  /// Input operators call this — never the native runtime directly.
  /// </summary>
  public Services.CameraService CameraService { get; }

  public ulong PresentationEngineId { get; private set; }
  // private ulong _lastRenderTaskId;

  public OperatorStack OperatorStack { get; }



  [ObservableProperty]
  private uint _width = 800;

  [ObservableProperty]
  private uint _height = 600;

  partial void OnWidthChanged(uint value)
  {
    if (CameraId != 0 && CameraService != null)
    {
      CameraService.OnViewportResized(value, _height);
    }
  }

  partial void OnHeightChanged(uint value)
  {
    if (CameraId != 0 && CameraService != null)
    {
      CameraService.OnViewportResized(_width, value);
    }
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
    if (PresentationEngineId != 0)
    {
      Console.WriteLine($"[SetupViewport] PE={PresentationEngineId} CameraId={CameraId} — ready for camera init");
    }
  }

  /// <summary>
  /// Called by <see cref="Logic.ViewModels.VulkanViewportControlViewModel"/> after
  /// <see cref="INativeRuntimeService.AddViewport"/> succeeds.
  /// </summary>
  public void OnViewportCreated(ulong presentationEngineId, ulong cameraEntityId)
  {
    PresentationEngineId = presentationEngineId;
    CameraId = cameraEntityId;
    Console.WriteLine($"[Viewport3DViewModel] OnViewportCreated PE={PresentationEngineId} Cam={CameraId}");
    _viewportRegistry.Register(presentationEngineId, cameraEntityId);
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

  // TODO cleanup.
  public Viewport3DViewModel(
    INativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    IUiThreadDispatcher uiThreadDispatcher,
    IFileDialogService fileDialogService,
    Services.CameraService cameraService,
    Func<Viewport3DViewModel, VulkanViewportControlViewModel> vulkanVmFactory,
    Func<Viewport3DViewModel, ViewportOverlayViewModel> overlayVmFactory,
    IPlatformWindowService platformWindowService,
    IWindowInputRouter inputRouter,
    ITabStateService<ViewportSession> sessionService,
    Services.IViewportRegistry viewportRegistry
  )
    : base("Viewport 3D", sessionService)
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _uiThreadDispatcher = uiThreadDispatcher;
    _fileDialogService = fileDialogService;
    CameraService = cameraService;
    VulkanViewModel = vulkanVmFactory(this);
    OverlayViewModel = overlayVmFactory(this);
    PlatformWindowService = platformWindowService;
    InputRouter = inputRouter;
    _viewportRegistry = viewportRegistry;

    // Keep EarthObserverState and IsEarthObserverMode in sync with CameraService.
    // IsEarthObserverMode is true only for EarthPosition — UpZenith is a zenith-offset
    // sub-mode but does not require SPK earth data to be live.
    _cameraModeSubscription = CameraService.CameraModeChanged.Subscribe(mode =>
    {
      IsEarthObserverMode = mode == Services.CameraMode.EarthPosition;
      EarthObserverState  = mode switch
      {
        Services.CameraMode.EarthPosition  => EarthObserverState.EarthPositioning,
        Services.CameraMode.UpZenith       => EarthObserverState.UpZenith,
        Services.CameraMode.CometOrbiting  => EarthObserverState.CometOrbiting,
        _                                  => EarthObserverState.UpZenith,
      };
    });

    OperatorStack = new OperatorStack(new ViewportBaseOperator(this));
    IsInitialized = true;

    // WeakReferenceMessenger.Default.Register<Messages.RenderFrameReadyMessage>(
    //   this,
    //   (r, m) => ((Viewport3DViewModel)r).Receive(m)
    // );
    // WeakReferenceMessenger.Default.Register<Messages.ToggleAddJetModeMessage>(
    //   this,
    //   (r, m) => ((Viewport3DViewModel)r).Receive(m)
    // );
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
      return Task.CompletedTask;
    });

    if (IsInitialized)
    {
      SetupViewport();
    }
  }

  // Hides the base class Dispose() intentionally: we need to call both our native
  // cleanup AND the base class Rx subscription teardown.
  public new void Dispose()
  {
    _cameraModeSubscription?.Dispose();
    _cameraModeSubscription = null;
    OverlayViewModel.Dispose();
    Stop();
    if (PresentationEngineId != 0)
    {
      _viewportRegistry.Unregister(PresentationEngineId);
      _runtimeService.RemoveViewport(PresentationEngineId);
      PresentationEngineId = 0;
      CameraId = 0;
    }
    base.Dispose(); // tears down Rx session subscriptions
  }

  // public void Receive(Messages.ToggleAddJetModeMessage message)
  // {
  //   IsAddingJet = true;
  // }

  public bool Process(AppAction action, InputState state)
  {
    return OperatorStack.Process(action, state);
  }

  [RelayCommand]
  private void ToggleCameraMode()
  {
    CameraService.CycleCameraMode();
  }

  // Image polling logic completely removed for NativeControlHost transition.



  public void Stop() { }
}
