using System;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;
using CommunityToolkit.Mvvm.Input;
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

  public System.Collections.ObjectModel.ObservableCollection<BillboardViewModel> Billboards { get; } = new();

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
      if (Width <= 0 || Height <= 0 || _sceneStateManager == null) return;
      var state = _sceneStateManager.GetOrCreateScene(SceneId);
      if (state.EntityMap.TryGetValue(CameraId, out var entity))
      {
          var camera = entity.Components.OfType<AetherVk.Logic.Models.CameraComponent>().FirstOrDefault();
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
  private bool _isMeasuringMode;

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

  public ulong CameraId { get; private set; }

  private static int _measurementCounter = 1;

  public IViewportRenderer? Renderer { get; set; }

  private readonly IUiThreadDispatcher _uiThreadDispatcher;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly SceneStateManager _sceneStateManager;

  /// <summary>
  /// Opens a native file dialog to permit selecting an image off the disk.
  /// Constructs a new <see cref="BillboardViewModel"/> dynamically and pushes it to the UI Layer.
  /// </summary>
  [RelayCommand]
  private async Task InsertBillboard()
  {
      var filters = new[] { "Images|*.png;*.jpg;*.jpeg;*.bmp" };

      var path = await _fileDialogService.ShowOpenFileDialogAsync("Select Billboard Image", filters);
      if (!string.IsNullOrEmpty(path))
      {
          try
          {
              Billboards.Add(new BillboardViewModel
              {
                  ImageSource = path,
                  X = 10,
                  Y = 10,
                  Width = 100,
                  Height = 100,
                  ZIndex = 1
              });
              _breadcrumbService.ShowMessageAsync("Billboard Added", $"Loaded image {System.IO.Path.GetFileName(path)}");
          }
          catch (Exception ex)
          {
              _breadcrumbService.ShowMessageAsync("Error", $"Failed to load image: {ex.Message}");
          }
      }
  }

  // HOME_POSITION: camera starts 0.02 AU from the Sun, equally distributed across x/y/z (first octant).
  // Coordinate system: +x=right, -y=forward, +z=up. Distance = 0.02 AU → each component = 0.02/√3 ≈ 0.011547.
  private const float HomeDistanceAu = 0.02f;
  // 0.02 / Math.Sqrt(3) precomputed
  private const float HomePosComponent = 0.011547005f;

  // Look-at quaternion: camera at (p,p,p) facing origin with +z=up.
  // look_dir = normalize(-1,-1,-1), right = normalize(look_dir × up), true_up = right × look_dir.
  // Computed offline to match the Rust create_default_scene look_at_axes output.
  private const float HomeRotW =  0.7010574f;
  private const float HomeRotX =  0.4304593f;
  private const float HomeRotY = -0.0922958f;
  private const float HomeRotZ = -0.5609855f;

  private void SetupViewport()
  {
    // Never create a scene here — that is InitializeSimulationContext's job.
    // If no scene exists yet, we'll be retried via SimulationStateUpdatedMessage
    // once CreateScene finishes on the UI thread.
    var existingScene = _sceneStateManager.AllScenes.FirstOrDefault();
    if (existingScene == null)
      return;

    if (SceneId == 0)
      SceneId = existingScene.SceneId;

    if (PresentationEngineId == 0)
      PresentationEngineId = _runtimeService.CreatePresentationEngine(Width, Height, SceneId);

    if (CameraId != 0)
      return; // Already fully set up.

    // Check the entity tree is populated before trying to add a camera.
    var root = _runtimeService.GetEntityByName(SceneId, "root");
    if (root == null)
      return; // SyncEntities hasn't run yet — retry will arrive via SimulationStateUpdatedMessage.

    CameraId = _runtimeService.AddPerspectiveCamera(
      SceneId,
      PresentationEngineId,
      $"viewport_camera_{PresentationEngineId}",
      45f,
      0.0001f,   // near  (~15 000 km at AU scale)
      1000.0f    // far   (1 000 AU covers solar system)
    );

    System.Console.WriteLine($"[Viewport3DViewModel] PE={PresentationEngineId}, Camera={CameraId}");

    // Seed the new camera at the default scene camera's home position + look-at rotation,
    // so the user sees the scene correctly oriented from first render.
    var sceneCamera = _runtimeService.GetEntityByName(SceneId, "camera");
    if (sceneCamera != null && sceneCamera.Id != CameraId)
    {
      var sceneTransform = sceneCamera.Components
        .OfType<AetherVk.Logic.Models.TransformComponent>()
        .FirstOrDefault();
      if (sceneTransform != null
          && (sceneTransform.PosX != 0f || sceneTransform.PosY != 0f || sceneTransform.PosZ != 0f))
      {
        _runtimeService.SetTransformComponent(SceneId, CameraId,
          sceneTransform.PosX, sceneTransform.PosY, sceneTransform.PosZ,
          sceneTransform.RotW, sceneTransform.RotX, sceneTransform.RotY, sceneTransform.RotZ,
          1f, 1f, 1f);
        return;
      }
    }

    // Fallback: canonical home position (0.011547, 0.011547, 0.011547) looking at origin.
    _runtimeService.SetTransformComponent(SceneId, CameraId,
      HomePosComponent, HomePosComponent, HomePosComponent,
      HomeRotW, HomeRotX, HomeRotY, HomeRotZ,
      1f, 1f, 1f);
  }

  public Viewport3DViewModel(
    NativeRuntimeService runtimeService,
    BreadcrumbService breadcrumbService,
    SceneStateManager sceneStateManager,
    IUiThreadDispatcher uiThreadDispatcher,
    IFileDialogService fileDialogService
  )
    : base("Viewport 3D")
  {
    _runtimeService = runtimeService;
    _breadcrumbService = breadcrumbService;
    _sceneStateManager = sceneStateManager;
    _uiThreadDispatcher = uiThreadDispatcher;
    _fileDialogService = fileDialogService;

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
          self.SetupViewport();
      }
    );

    _sceneStateManager.PropertyChanged += (s, e) =>
    {
       // If entities changed or camera changed, we should re-eval measurement
       _uiThreadDispatcher.DispatchAsync(() => { UpdateMeasurementIndicator(); return Task.CompletedTask; });
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
        var selected = Billboards.FirstOrDefault(b => b.IsSelected);
        if (selected != null)
        {
            Billboards.Remove(selected);
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

    if (IsMeasuringMode)
    {
      if (res.hit)
      {
        HandleMeasurementPoint(res.px, res.py, res.pz);
      }
      else
      {
        ShowNoIntersectionFlyout = true;
      }
      return;
    }

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
              CircleRadius = 0.5f,
              ParticlesPerTick = 10,
              ColorR = 1.0f,
              ColorG = 0.5f,
              ColorB = 0.0f,
              ColorA = 1.0f
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
      Billboards.Add(new BillboardViewModel
      {
          Text = $"{distance:F2} km",
          X = Width / 2.0 - 50,
          Y = Height / 2.0 - 50,
          Width = 100,
          Height = 40,
          ZIndex = 10
      });

      HasFirstMeasurementPoint = false;
      IsMeasuringMode = false;
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
  private void ToggleMeasuringMode()
  {
    IsMeasuringMode = !IsMeasuringMode;
    HasFirstMeasurementPoint = false;
    ShowNoIntersectionFlyout = false;
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

    UpdateMeasurementIndicator();

    try
    {
      await _runtimeService.PollTaskAsync(_lastRenderTaskId);

      nuint bufferSize = (nuint)(Width * Height * 4);
      IntPtr unmanagedBuffer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)bufferSize);

      try
      {
        await _runtimeService.DownloadImageAsync(_lastRenderTaskId, unmanagedBuffer, bufferSize);
        await _uiThreadDispatcher.DispatchAsync(() =>
        {
          Renderer?.UpdateFrame(unmanagedBuffer, bufferSize);
          return Task.CompletedTask;
        });
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
    if (Width <= 0 || Height <= 0) return;

    var state = _sceneStateManager.GetOrCreateScene(SceneId);
    if (state.EntityMap.TryGetValue(CameraId, out var entity))
    {
      var camera = entity.Components.OfType<AetherVk.Logic.Models.CameraComponent>().FirstOrDefault();
      if (camera != null)
      {
        double target_px_width = Math.Max(24.0, Width * 0.07);

        if (camera.IsOrthographic)
        {
          double W_au = camera.OrthoRight - camera.OrthoLeft;
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
          double W_arcsec = camera.Fov * camera.AspectRatio * 3600.0;
          if (W_arcsec > 0)
          {
            double min_arcsec = target_px_width * (W_arcsec / Width);
            double nice_arcsec = GetNiceNumber(min_arcsec);
            MeasurementIndicatorWidth = nice_arcsec * (Width / W_arcsec);
            MeasurementIndicatorText = $"{FormatNiceNumber(nice_arcsec)} arcsec";
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
    if (value <= 0) return 1.0;
    double exponent = Math.Floor(Math.Log10(value));
    double fraction = value / Math.Pow(10, exponent);
    
    double niceFraction;
    if (fraction <= 1.0) niceFraction = 1.0;
    else if (fraction <= 2.0) niceFraction = 2.0;
    else if (fraction <= 5.0) niceFraction = 5.0;
    else niceFraction = 10.0;
    
    return niceFraction * Math.Pow(10, exponent);
  }

  private string FormatNiceNumber(double value)
  {
    if (value >= 1.0) return value.ToString("0");
    return value.ToString("0.#####");
  }

  public void Stop() { }
}
