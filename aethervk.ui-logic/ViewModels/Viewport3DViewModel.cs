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
    IRecipient<AetherVk.Logic.Messages.ToggleAddJetModeMessage>,
    IRecipient<AetherVk.Logic.Messages.RenderFrameReadyMessage>,
    IDisposable
{
  private readonly NativeRuntimeService _runtimeService;
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
    UpdateCameraAspectRatio();
  }

  partial void OnHeightChanged(uint value)
  {
    UpdateCameraAspectRatio();
  }

  private void UpdateCameraAspectRatio()
  {
    if (Width <= 0 || Height <= 0 || _sceneStateManager == null || SceneId == 0)
      return;
    var state = _sceneStateManager.GetOrCreateScene(SceneId);
    if (state.EntityMap.TryGetValue(CameraId, out var entity))
    {
      var camera = entity
        .Components.OfType<AetherVk.Logic.Models.CameraComponent>()
        .FirstOrDefault();
      if (camera != null)
      {
        camera.AspectRatio = (float)Width / Height;
      }
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

  public enum EarthObserverState
  {
    None,
    AnimatingToEarth,
    Locked,
    AnimatingBack,
  }

  private EarthObserverState _earthObserverState = EarthObserverState.None;
  private NativeInterop.FfiHighResTransform _originalCameraPos;

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
  public string CometRadialLabel => HasComet ? "Destroy Comet" : "Spawn Comet";

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
      CloseRadialMenuAndSpawnCometCommand.NotifyCanExecuteChanged();
    }
  }

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
    var item = HoveredRadialItem;
    CloseRadialMenu();

    switch (item)
    {
      case "comet":
        if (HasComet)
        {
          // Destroy comet — disabled during playback
          if (_timelineService != null && _timelineService.IsPlaying)
          {
            _breadcrumbService?.ShowMessageAsync(
              "Cannot Destroy",
              "Cannot destroy comet while simulation is playing. Pause first.",
              default,
              5
            );
          }
          else
          {
            DestroyCometInternal();
          }
        }
        else
        {
          CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
            new AetherVk.Logic.Messages.OpenSpawnCometDialogMessage()
          );
        }
        break;
      case "billboard":
        InsertBillboardCommand.Execute(null);
        break;
      case "resetcamera":
        RuntimeService.ResetCamera(SceneId, CameraId);
        break;
      case "snap":
        var selected = SelectedEntity;
        if (selected != null)
          RuntimeService.SnapToEntity(SceneId, CameraId, selected.Id);
        break;
      case "snapobserver":
        SnapObserverCommand.Execute(null);
        break;
      // null or unrecognized: just close, no action
    }
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
    var state = _sceneStateManager.GetOrCreateScene(SceneId);
    var sun = System.Linq.Enumerable.FirstOrDefault(
      state.EntityMap.Values,
      e => System.Linq.Enumerable.Any(e.Components.OfType<AetherVk.Logic.Models.SunComponent>())
    );
    if (sun != null)
      RuntimeService.SnapToEntity(SceneId, CameraId, sun.Id);
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
    var selected = SelectedEntity;
    if (selected != null)
      RuntimeService.SnapToEntity(SceneId, CameraId, selected.Id);
  }

  public bool HasComet => _sceneStateManager.GetOrCreateScene(SceneId).CometEntityId.HasValue;

  [RelayCommand(CanExecute = nameof(CanSpawnComet))]
  private void CloseRadialMenuAndSpawnComet()
  {
    CloseRadialMenu();
    // Send a message to open spawn comet dialog from main window.
    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.OpenSpawnCometDialogMessage()
    );
  }

  private bool CanSpawnComet() => !HasComet;

  /// <summary>
  /// Destroys the currently spawned comet entity and its children.
  /// Clears the CometEntityId from scene state.
  /// </summary>
  private void DestroyCometInternal()
  {
    var state = _sceneStateManager.GetOrCreateScene(SceneId);
    if (!state.CometEntityId.HasValue)
      return;

    var cometId = state.CometEntityId.Value;
    RuntimeService.RemoveEntity(SceneId, cometId);
    state.CometEntityId = null;

    // Notify property changes for radial menu
    OnPropertyChanged(nameof(HasComet));
    OnPropertyChanged(nameof(CometRadialLabel));

    CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.CometDestroyedMessage { SceneId = SceneId }
    );

    _breadcrumbService.ShowMessageAsync(
      "Comet Destroyed",
      "The comet and all its jets have been removed from the scene.",
      default,
      3
    );
  }

  public ulong CameraId { get; private set; }

  /// <summary>Returns the currently selected entity in this viewport's scene, or null.</summary>
  public AetherVk.Logic.Models.Entity? SelectedEntity =>
    _sceneStateManager.GetOrCreateScene(SceneId).SelectedEntity;

  private static int _measurementCounter = 1;

  public IViewportRenderer? Renderer { get; set; }

  private readonly IUiThreadDispatcher _uiThreadDispatcher;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly SceneStateManager _sceneStateManager;
  private readonly AetherVk.Logic.Services.TimelineService _timelineService;

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
        var entityId = _runtimeService.SpawnBillboard(
          SceneId,
          path,
          ndcX,
          ndcY,
          1.0f,
          1.0f,
          PresentationEngineId
        );

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

        // Wire the entity in the scene with the NativeComponent
        var state = _sceneStateManager.GetOrCreateScene(SceneId);
        if (state.EntityMap.TryGetValue(entityId, out var entity))
        {
          var nativeComp = new AetherVk.Logic.Models.ScreenSpaceBillboardComponent
          {
            ImagePath = path,
            NdcX = ndcX,
            NdcY = ndcY,
            Scale = 1.0f,
            RotationDeg = 0.0f,
            Opacity = 1.0f,
            ZIndex = 1,
            ViewportId = PresentationEngineId,
          };
          entity.Components.Add(nativeComp);
          entity.IsDeletable = true;

          // Bind to Rust so property changes push to native
          nativeComp.BindToNative(_runtimeService.SimulationContext, SceneId, entityId);

          // Link to the overlay ViewModel for bidirectional sync
          nativeComp.LinkBillboard(billboard);
        }

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
      RuntimeService.RemoveEntity(SceneId, billboard.EntityId);
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
    System.Console.WriteLine(
      $"[SetupViewport] Called. IsInitialized={_runtimeService.IsInitialized}  AllScenes={_sceneStateManager.AllScenes.Count(s => s.SceneId != 0)}  SceneId={SceneId}  PE={PresentationEngineId}  Cam={CameraId}"
    );

    if (!_runtimeService.IsInitialized)
      return;

    // Never create a scene here — that is InitializeSimulationContext's job.
    // Skip any phantom SceneState(0) entries created before the real scene arrives.
    var existingScene = _sceneStateManager.AllScenes.FirstOrDefault(s => s.SceneId != 0);
    if (existingScene == null)
    {
      System.Console.WriteLine(
        "[SetupViewport] No valid scene yet — will retry on SimulationStateUpdatedMessage."
      );
      return;
    }

    if (SceneId == 0)
      SceneId = existingScene.SceneId;

    System.Console.WriteLine(
      $"[SetupViewport] Using SceneId={SceneId}  EntityMap.Count={existingScene.EntityMap.Count}"
    );

    if (PresentationEngineId == 0)
    {
      PresentationEngineId = _runtimeService.CreatePresentationEngine(Width, Height, SceneId);
      System.Console.WriteLine($"[SetupViewport] Created PE={PresentationEngineId}");
    }

    if (CameraId != 0)
    {
      System.Console.WriteLine(
        $"[SetupViewport] Camera already wired (CameraId={CameraId}), done."
      );
      return;
    }

    // Check the entity tree is populated before trying to add a camera.
    var rootEntity = _runtimeService.GetEntityByName(SceneId, "root");
    System.Console.WriteLine(
      $"[SetupViewport] root entity = {rootEntity?.Id.ToString() ?? "NULL"}"
    );
    if (rootEntity == null)
    {
      System.Console.WriteLine(
        "[SetupViewport] root not found — waiting for SimulationStateUpdatedMessage."
      );
      return;
    }

    CameraId = _runtimeService.AddPerspectiveCamera(
      SceneId,
      PresentationEngineId,
      $"viewport_camera_{PresentationEngineId}",
      45f,
      0.0001f, // near  (~15 000 km at AU scale)
      1000.0f // far   (1 000 AU covers solar system)
    );

    System.Console.WriteLine($"[SetupViewport] AddPerspectiveCamera => CameraId={CameraId}");

    if (CameraId == 0)
    {
      System.Console.WriteLine("[SetupViewport] ERROR: AddPerspectiveCamera returned 0!");
      return;
    }

    System.Console.WriteLine(
      $"[SetupViewport] Applying default viewport camera position: pos=({HomePosX}, {HomePosY}, {HomePosZ})"
    );
    _runtimeService.SetTransformComponent(
      SceneId,
      CameraId,
      HomePosX,
      HomePosY,
      HomePosZ,
      HomeRotW,
      HomeRotX,
      HomeRotY,
      HomeRotZ,
      1f,
      1f,
      1f
    );
  }

  public Viewport3DViewModel(
    NativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    SceneStateManager sceneStateManager,
    IUiThreadDispatcher uiThreadDispatcher,
    IFileDialogService fileDialogService,
    AetherVk.Logic.Services.TimelineService timelineService
  )
    : base("Viewport 3D")
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _sceneStateManager = sceneStateManager;
    _uiThreadDispatcher = uiThreadDispatcher;
    _fileDialogService = fileDialogService;
    _timelineService = timelineService;

    OperatorStack = new OperatorStack(new ViewportBaseOperator(this));

    _runtimeService.PropertyChanged += (s, e) =>
    {
      if (e.PropertyName == nameof(NativeRuntimeService.IsInitialized))
      {
        IsInitialized = _runtimeService.IsInitialized;
        if (IsInitialized)
        {
          SetupViewport();
        }
      }
    };
    IsInitialized = _runtimeService.IsInitialized;
    if (IsInitialized)
    {
      SetupViewport();
    }
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.RenderFrameReadyMessage>(
      this,
      (r, m) => ((Viewport3DViewModel)r).Receive(m)
    );
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.ToggleAddJetModeMessage>(
      this,
      (r, m) => ((Viewport3DViewModel)r).Receive(m)
    );
    // Retry SetupViewport after CreateScene completes (fires SimulationStateUpdatedMessage),
    // which resolves the timing race where IsInitialized=true fires before scene entities exist.
    WeakReferenceMessenger.Default.Register<AetherVk.Logic.Messages.SimulationStateUpdatedMessage>(
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

    _sceneStateManager.PropertyChanged += (s, e) =>
    {
      // If entities changed or camera changed, we should re-eval measurement
      _uiThreadDispatcher.DispatchAsync(() =>
      {
        UpdateMeasurementIndicator();
        return Task.CompletedTask;
      });
    };

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
      _runtimeService.DestroyPresentationEngine(SceneId, PresentationEngineId, CameraId);
      PresentationEngineId = 0;
    }
  }

  public void Receive(AetherVk.Logic.Messages.ToggleAddJetModeMessage message)
  {
    IsAddingJet = true;
  }

  public bool ProcessAction(AppAction action, bool isPressed)
  {
    if (isPressed && action.Id == "viewport.delete")
    {
      // Try billboard-overlay delete first
      var selectedBillboard = Billboards.FirstOrDefault(b => b.IsSelected);
      if (selectedBillboard != null)
      {
        if (selectedBillboard.EntityId != 0)
        {
          RuntimeService.RemoveEntity(SceneId, selectedBillboard.EntityId);
          // Remove from scene state
          var state = _sceneStateManager.GetOrCreateScene(SceneId);
          if (state.EntityMap.TryGetValue(selectedBillboard.EntityId, out var entity))
          {
            // Remove from parent's children collection
            foreach (var parent in state.EntityMap.Values)
            {
              parent.Children.Remove(entity);
            }
            state.EntityMap.Remove(selectedBillboard.EntityId);
          }
        }
        Billboards.Remove(selectedBillboard);
        return true;
      }

      // General deletable entity (no billboard overlay, e.g. future use)
      var selectedEntity = SelectedEntity;
      if (selectedEntity != null && selectedEntity.IsDeletable)
      {
        RuntimeService.RemoveEntity(SceneId, selectedEntity.Id);
        var state = _sceneStateManager.GetOrCreateScene(SceneId);
        foreach (var parent in state.EntityMap.Values)
        {
          parent.Children.Remove(selectedEntity);
        }
        state.EntityMap.Remove(selectedEntity.Id);
        state.SelectedEntity = null;
        return true;
      }
    }
    return OperatorStack.ProcessAction(action, isPressed);
  }

  public bool ProcessPointerDelta(float dx, float dy) => OperatorStack.ProcessPointerDelta(dx, dy);

  public bool ProcessPointerWheel(float deltaY) => OperatorStack.ProcessPointerWheel(deltaY);

  public async void PerformRaycast(double x, double y, double w, double h)
  {
    float ndcX = (float)((x / w) * 2.0 - 1.0);
    float ndcY = (float)((y / h) * 2.0 - 1.0);

    var res = await _runtimeService.RaycastNdcAsync(SceneId, CameraId, ndcX, ndcY);

    var breadcrumb = _breadcrumbService;

    if (IsEarthObserverMode)
      return;

    if (res.hit)
    {
      var state = _sceneStateManager.GetOrCreateScene(SceneId);
      var entity = _runtimeService.GetEntityById(SceneId, res.entityId);

      if (entity != null)
      {
        if (state.SelectedEntity?.Id == entity.Id)
        {
          var emitter = entity
            .Components.OfType<AetherVk.Logic.Models.ParticleEmitterCirclesComponent>()
            .FirstOrDefault();
          var tx = entity
            .Components.OfType<AetherVk.Logic.Models.TransformComponent>()
            .FirstOrDefault();

          if (emitter != null && tx != null && IsAddingJet)
          {
            // Convert world hit point to local space
            var worldPt = new System.Numerics.Vector3(res.px, res.py, res.pz);
            var worldPos = new System.Numerics.Vector3(tx.PosX, tx.PosY, tx.PosZ);
            var worldRot = new System.Numerics.Quaternion(tx.RotX, tx.RotY, tx.RotZ, tx.RotW);

            var localPt = System.Numerics.Vector3.Transform(
              worldPt - worldPos,
              System.Numerics.Quaternion.Inverse(worldRot)
            );

            // Normalize to get spherical coordinates
            var normalizedPt = System.Numerics.Vector3.Normalize(localPt);
            float latitude = (float)(Math.Asin(normalizedPt.Z) * 180.0 / Math.PI);
            float longitude = (float)(Math.Atan2(normalizedPt.Y, normalizedPt.X) * 180.0 / Math.PI);

            var newCircle = new AetherVk.Logic.Models.EmissionCircleItem
            {
              LatitudeDeg = latitude,
              LongitudeDeg = longitude,
              CircleRadiusKm = 0.5f,
              ParticlesPerSecond = 600f,  // 600/s ≈ 10/tick at 60 Hz
              ColorR = 1.0f,
              ColorG = 0.5f,
              ColorB = 0.0f,
              ColorA = 1.0f,
            };

            emitter.Circles.Add(newCircle);

            breadcrumb?.ShowMessageAsync(
              "Raycast Hit",
              $"Added jet at Lat: {latitude:F1}°, Lon: {longitude:F1}°"
            );
            IsAddingJet = false; // Turn off mode after adding one
          }
          else
          {
            breadcrumb?.ShowMessageAsync(
              "Raycast Info",
              "Entity selected but it is not a comet, cannot add jets."
            );
          }
        }
        else
        {
          state.SelectedEntity = entity;
          CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
            new AetherVk.Logic.ViewModels.EntitySelectedMessage(entity)
          );
          breadcrumb?.ShowMessageAsync("Raycast Hit", $"Selected {entity.Name}");
        }
      }
    }
    else
    {
      // Deselect when clicking on empty space
      var state = _sceneStateManager.GetOrCreateScene(SceneId);
      if (state.SelectedEntity != null)
      {
        state.SelectedEntity = null;
        CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
          new AetherVk.Logic.ViewModels.EntitySelectedMessage(null)
        );
      }
    }
  }

  private void HandleMeasurementPoint(float x, float y, float z)
  {
    if (!HasFirstMeasurementPoint)
    {
      HasFirstMeasurementPoint = true;
      FirstMeasurementPointX = x;
      FirstMeasurementPointY = y;
      FirstMeasurementPointZ = z;
    }
    else
    {
      var name = $"Measurement_{_measurementCounter++}";
      _runtimeService.CreateMeasurement(
        SceneId,
        name,
        new[] { FirstMeasurementPointX, FirstMeasurementPointY, FirstMeasurementPointZ },
        new[] { x, y, z }
      );

      // Calculate distance between the two points
      float dx = x - FirstMeasurementPointX;
      float dy = y - FirstMeasurementPointY;
      float dz = z - FirstMeasurementPointZ;
      float distance = (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);

      // Calculate midpoint to place the label approximately (in 2D space this requires projection, but we can just put it at 10,10 for now, or 3D project it later. We will just add the billboard with the text)
      Billboards.Add(
        new BillboardViewModel
        {
          Text = $"{distance:F2} km",
          X = Width / 2.0 - 50,
          Y = Height / 2.0 - 50,
          Width = 100,
          Height = 40,
          ZIndex = 10,
        }
      );

      HasFirstMeasurementPoint = false;
      IsEarthObserverMode = false;
      ShowNoIntersectionFlyout = false;
    }
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void SubmitManualMeasurement()
  {
    HandleMeasurementPoint(ManualMeasurementX, ManualMeasurementY, ManualMeasurementZ);
    ShowNoIntersectionFlyout = false;
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void SubmitCursorMeasurement()
  {
    float cx = 0,
      cy = 0,
      cz = 0;
    var state = _sceneStateManager;
    var rootEntities = state?.GetOrCreateScene(SceneId).RootEntities;
    var cursor = rootEntities?.FirstOrDefault(e =>
      e.Name == "cursor" || e.Components.Any(c => c.Name == "Cursor")
    );
    if (cursor != null)
    {
      var transform = cursor
        .Components.OfType<AetherVk.Logic.Models.TransformComponent>()
        .FirstOrDefault();
      if (transform != null)
      {
        cx = transform.PosX;
        cy = transform.PosY;
        cz = transform.PosZ;
      }
    }
    HandleMeasurementPoint(cx, cy, cz);
    ShowNoIntersectionFlyout = false;
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void UndoMeasurementRaycast()
  {
    ShowNoIntersectionFlyout = false;
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private void ToggleEarthObserverMode()
  {
    if (_timelineService.IsPlaying)
    {
      _ = _breadcrumbService.ShowMessageAsync(
        "Viewport",
        "Cannot switch to Earth Observer Mode while simulation is running.",
        TimeSpan.FromSeconds(3),
        3
      );
      return;
    }

    if (CameraId == 0)
      return;

    if (!IsEarthObserverMode)
    {
      // Entering mode
      IsEarthObserverMode = true;
      _runtimeService.GetHighResTransformComponent(SceneId, CameraId, out _originalCameraPos);

      // Get Earth's position
      var earthPos = _runtimeService.GetEphemerisPosition(
        399,
        _runtimeService.GetSimulationTime(SceneId)
      );
      if (earthPos.HasValue)
      {
        OperatorStack.IsCameraControlEnabled = false;
        _runtimeService.AddCameraAnimation(
          SceneId,
          CameraId,
          earthPos.Value.PosX,
          earthPos.Value.PosY,
          earthPos.Value.PosZ,
          2.0f
        );
        _earthObserverState = EarthObserverState.AnimatingToEarth;
      }
      else
      {
        // Fallback if no SPK loaded
        _runtimeService.AddAlmanacPlanet(SceneId, CameraId, 399);
        _earthObserverState = EarthObserverState.Locked;
      }
    }
    else
    {
      // Exiting mode
      IsEarthObserverMode = false;
      _runtimeService.RemoveAlmanacPlanet(SceneId, CameraId);
      _runtimeService.AddCameraAnimation(
        SceneId,
        CameraId,
        _originalCameraPos.Px,
        _originalCameraPos.Py,
        _originalCameraPos.Pz,
        2.0f
      );
      _earthObserverState = EarthObserverState.AnimatingBack;
    }
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private async Task SnapObserverAsync()
  {
    if (!IsEarthObserverMode)
      return;

    var result = await WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.OpenSnapObserverDialogMessage()
    );
    if (result.HasValue)
    {
      _runtimeService.SetAlmanacPlanetOffset(
        SceneId,
        CameraId,
        (float)result.Value.X,
        (float)result.Value.Y,
        (float)result.Value.Z
      );
    }
  }

  [CommunityToolkit.Mvvm.Input.RelayCommand]
  private async Task InitializeSceneAsync()
  {
    if (!_runtimeService.IsInitialized)
    {
      IsLoading = true;
      await Task.Run(() => _runtimeService.InitializeSimulationContext("Vulkan", null, false));
      IsLoading = false;
    }

    SetupViewport();

    IsInitialized = true;
  }

  public NativeRuntimeService RuntimeService => _runtimeService;

  public void Receive(AetherVk.Logic.Messages.RenderFrameReadyMessage message)
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

    if (_earthObserverState == EarthObserverState.AnimatingToEarth)
    {
      if (_runtimeService.CheckCameraAnimationFinished(SceneId, CameraId))
      {
        _runtimeService.RemoveCameraAnimation(SceneId, CameraId);
        _runtimeService.AddAlmanacPlanet(SceneId, CameraId, 399);
        _earthObserverState = EarthObserverState.Locked;
      }
    }
    else if (_earthObserverState == EarthObserverState.AnimatingBack)
    {
      if (_runtimeService.CheckCameraAnimationFinished(SceneId, CameraId))
      {
        _runtimeService.RemoveCameraAnimation(SceneId, CameraId);
        _earthObserverState = EarthObserverState.None;
        OperatorStack.IsCameraControlEnabled = true;
      }
    }

    UpdateMeasurementIndicator();

    try
    {
      nuint bufferSize = (nuint)(Width * Height * 4);
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
          System.Console.WriteLine(
            $"[ProcessFrameAsync] DownloadImageAsync returned false for taskId={_lastRenderTaskId}. "
              + "Frame skipped."
          );
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

    var state = _sceneStateManager.GetOrCreateScene(SceneId);
    if (state.EntityMap.TryGetValue(CameraId, out var entity))
    {
      var camera = entity
        .Components.OfType<AetherVk.Logic.Models.CameraComponent>()
        .FirstOrDefault();
      if (camera != null)
      {
        double target_px_width = Math.Max(24.0, Width * 0.07);

        if (camera.IsOrthographic)
        {
          double W_au = Width * camera.OrthoScaleFactor;
          if (W_au > 0)
          {
            double min_au = target_px_width * (W_au / Width);
            double nice_au = GetNiceNumber(min_au);
            MeasurementIndicatorWidth = nice_au * (Width / W_au);
            MeasurementIndicatorText = $"{FormatNiceNumber(nice_au)} AU";
            ShowMeasurementIndicator = true;
          }
          else
          {
            ShowMeasurementIndicator = false;
          }
        }
        else
        {
          // Perspective
          double fovRad = camera.Fov * Math.PI / 180.0;
          double hFovRad = 2.0 * Math.Atan(Math.Tan(fovRad / 2.0) * camera.AspectRatio);
          double W_arcsec = hFovRad * 180.0 / Math.PI * 3600.0;

          if (W_arcsec > 0)
          {
            double min_arcsec = target_px_width * (W_arcsec / Width);

            if (min_arcsec > 3600.0)
            {
              double min_deg = min_arcsec / 3600.0;
              double nice_deg = GetNiceNumber(min_deg);
              MeasurementIndicatorWidth = nice_deg * 3600.0 * (Width / W_arcsec);
              MeasurementIndicatorText = $"{FormatNiceNumber(nice_deg)} deg";
            }
            else if (min_arcsec > 60.0)
            {
              double min_min = min_arcsec / 60.0;
              double nice_min = GetNiceNumber(min_min);
              MeasurementIndicatorWidth = nice_min * 60.0 * (Width / W_arcsec);
              MeasurementIndicatorText = $"{FormatNiceNumber(nice_min)} arcmin";
            }
            else
            {
              double nice_arcsec = GetNiceNumber(min_arcsec);
              MeasurementIndicatorWidth = nice_arcsec * (Width / W_arcsec);
              MeasurementIndicatorText = $"{FormatNiceNumber(nice_arcsec)} arcsec";
            }

            ShowMeasurementIndicator = true;
          }
          else
          {
            ShowMeasurementIndicator = false;
          }
        }
      }
      else
      {
        ShowMeasurementIndicator = false;
      }
    }
    else
    {
      ShowMeasurementIndicator = false;
    }
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
