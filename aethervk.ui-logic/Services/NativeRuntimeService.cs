using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Services;

public partial class NativeRuntimeService : ObservableObject, IDisposable
{
  [ObservableProperty]
  private bool _isInitialized;

  [ObservableProperty]
  private bool _isRunning;

  private IntPtr _simulationContext = IntPtr.Zero;

  /// <summary>Raw simulation context pointer — for direct P/Invoke calls that don't have a wrapper yet.</summary>
  public IntPtr SimulationContext => _simulationContext;
  private readonly object _nativeLock = new object();
  private int _activeDownloads = 0;
  private bool _isDisposing = false;
  private readonly HashSet<ulong> _activePresentationEngines = new();

  private readonly SceneStateManager _sceneStateManager;
  public SceneStateManager SceneStateManager => _sceneStateManager;
  private readonly ConsoleService _consoleService;
  private readonly BreadcrumbService _breadcrumbService;
  private readonly INativeBufferPoolService _bufferPool;
  private readonly IUiThreadDispatcher _uiThreadDispatcher;

  // Keep a weak reference to the instance so we don't artificially keep it alive
  private static WeakReference<NativeRuntimeService>? s_currentInstance;

  // Keep static references to the delegates so they act as GC roots and are NEVER Garbage Collected
  private static readonly NativeInterop.LoggerCallback s_loggerCallbackDelegate =
    NativeLogCallbackStatic;

  private static readonly NativeInterop.BreadcrumbCallback s_breadcrumbCallbackDelegate =
    NativeBreadcrumbCallbackStatic;

  private static readonly NativeInterop.SimulationCallback s_simulationCallbackDelegate =
    NativeSimulationCallbackStatic;

  // Rate-limit render error logging: track count and last log timestamp.
  private int _renderErrorCount = 0;
  private long _lastRenderErrorLogTicks = 0;

  private static readonly NativeInterop.RenderCallback s_renderCallbackDelegate =
    NativeRenderCallbackStatic;

  public void Dispose()
  {
    Console.WriteLine(
      $"[NativeRuntimeService] Dispose called on instance {GetHashCode()}! _simulationContext={_simulationContext}"
    );
    lock (_nativeLock)
    {
      Console.WriteLine(
        $"[NativeRuntimeService] Dispose acquired lock on instance {GetHashCode()}"
      );
      _isDisposing = true;
      while (_activeDownloads > 0)
      {
        Console.WriteLine(
          $"[NativeRuntimeService] Dispose waiting for {_activeDownloads} downloads..."
        );
        System.Threading.Monitor.Wait(_nativeLock);
      }

      if (_simulationContext != IntPtr.Zero)
      {
        Console.WriteLine(
          $"[NativeRuntimeService] Calling avkSimulationContext_shutdown on {_simulationContext}"
        );
        NativeInterop.avkSimulationContext_shutdown(_simulationContext);
        Console.WriteLine($"[NativeRuntimeService] avkSimulationContext_shutdown returned");
        _simulationContext = IntPtr.Zero;
      }

      // Detach from the weak reference to stop processing incoming logs
      s_currentInstance = null;

      // Empty scene manager
      foreach (var scene in _sceneStateManager.AllScenes.ToList())
      {
        WeakReferenceMessenger.Default.Send(
          new AetherVk.Logic.Messages.SimulationStateUpdatedMessage(scene.SceneId)
        );
      }

      _sceneStateManager.Clear();
      IsInitialized = false;
    }
  }

  public NativeRuntimeService(
    SceneStateManager sceneStateManager,
    ConsoleService consoleService,
    BreadcrumbService breadcrumbService,
    INativeBufferPoolService bufferPool,
    IUiThreadDispatcher uiThreadDispatcher
  )
  {
    _sceneStateManager = sceneStateManager;
    _consoleService = consoleService;
    _breadcrumbService = breadcrumbService;
    _bufferPool = bufferPool;
    _uiThreadDispatcher = uiThreadDispatcher;

    // Register the active instance
    s_currentInstance = new WeakReference<NativeRuntimeService>(this);

    try
    {
      // Register the STATIC delegates with the native code
      NativeInterop.avkSimulationContext_setLoggerCallback(s_loggerCallbackDelegate);
      NativeInterop.avkSimulationContext_setBreadcrumbCallback(s_breadcrumbCallbackDelegate);
      NativeInterop.avkSimulationContext_setSimulationCallback(s_simulationCallbackDelegate);
      NativeInterop.avkSimulationContext_setRenderCallback(s_renderCallbackDelegate);
    }
    catch (System.DllNotFoundException)
    {
      // Ignore during tests
    }

    WeakReferenceMessenger.Default.Register<EntityVisibilityChangedMessage>(
      this,
      (r, m) =>
      {
        lock (_nativeLock)
        {
          if (_simulationContext != IntPtr.Zero)
          {
            NativeInterop.avkSimulationContext_setEntityVisibility(
              _simulationContext,
              m.Entity.SceneId,
              m.Entity.Id,
              m.Entity.IsVisible
            );
          }
        }
      }
    );

    WeakReferenceMessenger.Default.Register<EntityOutlineChangedMessage>(
      this,
      (r, m) =>
      {
        lock (_nativeLock)
        {
          if (_simulationContext != IntPtr.Zero)
          {
            NativeInterop.avkSimulationContext_setEntityFollowing(
              _simulationContext,
              m.Entity.SceneId,
              m.Entity.Id,
              m.Entity.IsOutlined
            );
          }
        }
      }
    );

    WeakReferenceMessenger.Default.Register<EntityNameChangedMessage>(
      this,
      (r, m) =>
      {
        lock (_nativeLock)
        {
          if (_simulationContext != IntPtr.Zero)
          {
            NativeInterop.avkSimulationContext_setEntityName(
              _simulationContext,
              m.Entity.SceneId,
              m.Entity.Id,
              m.NewName
            );
          }
        }
      }
    );

    WeakReferenceMessenger.Default.Register<BvhNodeVisibilityChangedMessage>(
      this,
      (r, m) =>
      {
        lock (_nativeLock)
        {
          if (_simulationContext != IntPtr.Zero)
          {
            NativeInterop.avkSimulationContext_setBvhNodeVisibility(
              _simulationContext,
              m.SceneId,
              m.EntityId,
              m.NodeIndex,
              m.IsVisible
            );
          }
        }
      }
    );

    WeakReferenceMessenger.Default.Register<AetherVk.Logic.ViewModels.EntitySelectedMessage>(
      this,
      (r, m) =>
      {
        lock (_nativeLock)
        {
          if (_simulationContext != IntPtr.Zero)
          {
            // Deselect all
            if (m.SelectedEntity != null)
            {
              var sceneState = _sceneStateManager.GetOrCreateScene(m.SelectedEntity.SceneId);
              foreach (var entity in sceneState.EntityMap.Values)
              {
                NativeInterop.avkSimulationContext_setEntitySelected(
                  _simulationContext,
                  m.SelectedEntity.SceneId,
                  entity.Id,
                  false
                );
              }
            }

            if (m.SelectedEntity != null)
            {
              NativeInterop.avkSimulationContext_setEntitySelected(
                _simulationContext,
                m.SelectedEntity.SceneId,
                m.SelectedEntity.Id,
                true
              );
            }
          }
        }
      }
    );
  }

  private void NativeBreadcrumbCallback(uint status, IntPtr messagePtr)
  {
    if (messagePtr != IntPtr.Zero)
    {
      string? message = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr);
      if (message != null)
      {
        _breadcrumbService?.ShowMessageAsync("Simulation Context", message, default, (int)status);
      }
    }
  }

  private void NativeLogCallback(IntPtr messagePtr)
  {
    if (messagePtr != IntPtr.Zero)
    {
      string? message = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr);
      if (message != null)
      {
        _consoleService?.Log(message);
      }
    }
  }

  private static void NativeLogCallbackStatic(IntPtr messagePtr)
  {
    // TryGetTarget safely fetches the instance only if it hasn't been Garbage Collected/Disposed.
    if (s_currentInstance != null && s_currentInstance.TryGetTarget(out var service))
    {
      service.NativeLogCallback(messagePtr);
    }
  }

  private static void NativeBreadcrumbCallbackStatic(uint status, IntPtr messagePtr)
  {
    if (s_currentInstance != null && s_currentInstance.TryGetTarget(out var service))
    {
      service.NativeBreadcrumbCallback(status, messagePtr);
    }
  }

  // A small queue to batch updates before dispatching to UI
  private readonly System.Collections.Concurrent.ConcurrentQueue<(
    ulong SceneId,
    ulong EntityId,
    ulong ComponentId,
    IntPtr DataPtr
  )> _simulationUpdateQueue = new();
  private bool _isSimulationUpdatePending = false;

  private void NativeSimulationCallback(
    ulong sceneId,
    ulong entityId,
    ulong componentId,
    IntPtr dataPtr
  )
  {
    // dataPtr may be null for "pull-signal" component IDs (100/101/102 etc.)
    // where Rust tells C# a component changed but doesn't inline the DTO.
    // The per-branch checks below handle null vs non-null appropriately.

    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    if (state.EntityMap.TryGetValue(entityId, out var entity))
    {
      if (componentId == 1 && dataPtr != IntPtr.Zero) // Transform (inline DTO)
      {
        var dto = Marshal.PtrToStructure<NativeInterop.FfiTransform>(dataPtr);
        _uiThreadDispatcher?.Dispatch(() =>
        {
          var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
          if (transform != null)
          {
            transform.SuspendNotifications = true;
            transform.PosX = dto.Px;
            transform.PosY = dto.Py;
            transform.PosZ = dto.Pz;
            transform.RotW = dto.Rw;
            transform.RotX = dto.Rx;
            transform.RotY = dto.Ry;
            transform.RotZ = dto.Rz;
            transform.ScaleX = dto.Sx;
            transform.ScaleY = dto.Sy;
            transform.ScaleZ = dto.Sz;
            transform.SuspendNotifications = false;
          }
        });
      }
      else if (componentId == 2 && dataPtr != IntPtr.Zero) // Camera (inline DTO)
      {
        var dto = Marshal.PtrToStructure<NativeInterop.FfiCamera>(dataPtr);
        _uiThreadDispatcher?.Dispatch(() =>
        {
          var camera = entity.Components.OfType<CameraComponent>().FirstOrDefault();
          if (camera != null)
          {
            camera.SuspendNotifications = true;
            camera.IsOrthographic = dto.IsOrthographic;
            camera.Fov = dto.Fov;
            camera.AspectRatio = dto.Aspect;
            camera.NearPlane = dto.Near;
            camera.FarPlane = dto.Far;
            camera.OrthoLeft = dto.OrthoLeft;
            camera.OrthoRight = dto.OrthoRight;
            camera.OrthoBottom = dto.OrthoBottom;
            camera.OrthoTop = dto.OrthoTop;
            camera.FocusDistance = dto.FocusDistance;
            camera.SuspendNotifications = false;
          }
        });
      }
      else
      {
        // ── Pull-signal: Rust marked this component as changed but sent
        //    no inline data (null pointer).  For derived-data components
        //    (100/101/102), copy position from the entity's Transform.
        //    For unknown IDs, pull all NativeComponents as a fallback.
        _uiThreadDispatcher?.Dispatch(() =>
        {
          switch (componentId)
          {
            case 100: // Comet
            {
              var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
              var comet = entity.Components.OfType<CometComponent>().FirstOrDefault();
              if (transform != null && comet != null)
              {
                comet.PositionX = transform.PosX;
                comet.PositionY = transform.PosY;
                comet.PositionZ = transform.PosZ;
              }
              break;
            }
            case 101: // Planet
            {
              var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
              var planet = entity.Components.OfType<PlanetComponent>().FirstOrDefault();
              if (transform != null && planet != null)
              {
                planet.PositionX = transform.PosX;
                planet.PositionY = transform.PosY;
                planet.PositionZ = transform.PosZ;
              }
              break;
            }
            case 102: // Sun
            {
              var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
              var sun = entity.Components.OfType<SunComponent>().FirstOrDefault();
              if (transform != null && sun != null)
              {
                sun.PositionX = transform.PosX;
                sun.PositionY = transform.PosY;
                sun.PositionZ = transform.PosZ;
              }
              break;
            }
            default:
            {
              // Unknown component ID — try pulling any NativeComponents on this entity
              foreach (var comp in entity.Components.OfType<NativeComponent>())
                comp.PullFromNative();
              break;
            }
          }
        });
      }
    }

    // Schedule a single UI update
    if (!_isSimulationUpdatePending)
    {
      _isSimulationUpdatePending = true;
      if (_uiThreadDispatcher != null)
      {
        _uiThreadDispatcher.Dispatch(() =>
        {
          _isSimulationUpdatePending = false;
          WeakReferenceMessenger.Default.Send(
            new AetherVk.Logic.Messages.SimulationStateUpdatedMessage(sceneId)
          );
        });
      }
      else
      {
        _isSimulationUpdatePending = false;
        WeakReferenceMessenger.Default.Send(
          new AetherVk.Logic.Messages.SimulationStateUpdatedMessage(sceneId)
        );
      }
    }
  }

  private static void NativeSimulationCallbackStatic(
    ulong sceneId,
    ulong entityId,
    ulong componentId,
    IntPtr dataPtr
  )
  {
    if (s_currentInstance != null && s_currentInstance.TryGetTarget(out var service))
    {
      service.NativeSimulationCallback(sceneId, entityId, componentId, dataPtr);
    }
  }

  private void NativeRenderCallback(
    ulong sceneId,
    ulong presentationEngineId,
    ulong renderGeneration
  )
  {
    // ulong.MaxValue is the Rust sentinel for a failed render tasklet.
    // Rate-limit to log at most once per 5 seconds rather than once per frame.
    if (renderGeneration == ulong.MaxValue)
    {
      System.Threading.Interlocked.Increment(ref _renderErrorCount);
      long now = System.Diagnostics.Stopwatch.GetTimestamp();
      long freq = System.Diagnostics.Stopwatch.Frequency;
      long last = System.Threading.Volatile.Read(ref _lastRenderErrorLogTicks);
      if (now - last >= freq * 5)
      {
        System.Threading.Volatile.Write(ref _lastRenderErrorLogTicks, now);
        int count = System.Threading.Interlocked.Exchange(ref _renderErrorCount, 0);
        System.Console.WriteLine(
          $"[NativeRenderCallback] Render frame error x{count} in last 5s "
            + $"(scene={sceneId}, pe={presentationEngineId}). "
            + "Check Rust log for '[render tasklet]' entries."
        );
      }
      return;
    }

    if (_uiThreadDispatcher != null)
    {
      _uiThreadDispatcher.Dispatch(() =>
      {
        WeakReferenceMessenger.Default.Send(
          new AetherVk.Logic.Messages.RenderFrameReadyMessage(
            sceneId,
            presentationEngineId,
            renderGeneration
          )
        );
      });
    }
    else
    {
      WeakReferenceMessenger.Default.Send(
        new AetherVk.Logic.Messages.RenderFrameReadyMessage(
          sceneId,
          presentationEngineId,
          renderGeneration
        )
      );
    }
  }

  private static void NativeRenderCallbackStatic(
    ulong sceneId,
    ulong presentationEngineId,
    ulong renderGeneration
  )
  {
    if (s_currentInstance != null && s_currentInstance.TryGetTarget(out var service))
    {
      service.NativeRenderCallback(sceneId, presentationEngineId, renderGeneration);
    }
  }

  public async Task PollTaskAsync(ulong taskId)
  {
    if (taskId == 0)
      return;
    while (_simulationContext != IntPtr.Zero)
    {
      int status = NativeInterop.avkSimulationContext_getTaskStatus(_simulationContext, taskId);
      if (status == 1)
        return; // Success
      if (status == 2)
        throw new Exception("Native Task Failed");
      if (status == -1)
        throw new Exception("Native Context Destroyed");

      await Task.Delay(1);
    }
  }

  public void SetBvhNodeVisibility(ulong sceneId, ulong entityId, uint nodeIndex, bool isVisible)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setBvhNodeVisibility(
        _simulationContext,
        sceneId,
        entityId,
        nodeIndex,
        isVisible
      );
    }
  }

  public string[] GetAvailableRenderDevices()
  {
    IntPtr ptr = NativeInterop.avkGetAvailableRenderBackends(out uint count);
    if (ptr == IntPtr.Zero || count == 0)
      return Array.Empty<string>();

    var result = new string[count];
    for (int i = 0; i < count; i++)
    {
      IntPtr strPtr = Marshal.ReadIntPtr(ptr, i * IntPtr.Size);
      result[i] = Marshal.PtrToStringAnsi(strPtr);
    }

    NativeInterop.avkFreeStringArray(ptr, count);
    return result;
  }

  public string[] GetAvailableKernels()
  {
    IntPtr ptr = NativeInterop.avkGetAvailableKernels(out uint count);
    if (ptr == IntPtr.Zero || count == 0)
      return Array.Empty<string>();

    var result = new string[count];
    for (int i = 0; i < count; i++)
    {
      IntPtr strPtr = Marshal.ReadIntPtr(ptr, i * IntPtr.Size);
      result[i] = Marshal.PtrToStringAnsi(strPtr);
    }

    NativeInterop.avkFreeStringArray(ptr, count);
    return result;
  }

  public string[] GetEntityComponentNames(ulong sceneId, ulong entityId)
  {
    if (_simulationContext == IntPtr.Zero)
      return Array.Empty<string>();

    uint maxCount = 64; // arbitrary max components per entity
    IntPtr namesPtr = Marshal.AllocHGlobal((int)(maxCount * IntPtr.Size));

    uint count = NativeInterop.avkSimulationContext_getEntityComponentNames(
      _simulationContext,
      sceneId,
      entityId,
      namesPtr,
      maxCount
    );

    var names = new string[count];
    for (int i = 0; i < count; i++)
    {
      IntPtr strPtr = Marshal.ReadIntPtr(namesPtr, i * IntPtr.Size);
      names[i] = Marshal.PtrToStringAnsi(strPtr) ?? "";
    }

    NativeInterop.avkSimulationContext_freeComponentNames(namesPtr, count);
    Marshal.FreeHGlobal(namesPtr);

    return names;
  }

  private static readonly object _staticInitLock = new object();

  public void InitializeSimulationContext(
    string backend = "Vulkan",
    string? assetOverride = null,
    bool populateDefault = true
  )
  {
    lock (_staticInitLock)
    {
      if (IsInitialized)
        return;

      // Resolve absolute path to the published assets folder
      var exePath = System.AppDomain.CurrentDomain.BaseDirectory;

      // Point Vulkan loader to our embedded MoltenVK and layers if they exist
      // TODO: This part up toll set of VK_LAYER_PATH is for MacOS only
      var icdPath = System.IO.Path.Combine(
        exePath,
        "vulkan",
        "share",
        "vulkan",
        "icd.d",
        "MoltenVK_icd.json"
      );
      if (System.IO.File.Exists(icdPath))
      {
        Environment.SetEnvironmentVariable("VK_DRIVER_FILES", icdPath);
        Environment.SetEnvironmentVariable("VK_ICD_FILENAMES", icdPath);
      }

      var layerPath = System.IO.Path.Combine(
        exePath,
        "vulkan",
        "share",
        "vulkan",
        "explicit_layer.d"
      );
      if (System.IO.Directory.Exists(layerPath))
      {
        Environment.SetEnvironmentVariable("VK_LAYER_PATH", layerPath);
      }

      var assetPath = assetOverride ?? System.IO.Path.Combine(exePath, "assets");
      NativeInterop.avkSimulationContext_setAssetPath(assetPath);

      _simulationContext = NativeInterop.avkSimulationContext_startup(backend);

      if (_simulationContext == IntPtr.Zero)
      {
        CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
          new AetherVk.Logic.Messages.CriticalErrorMessage(
            $"CRITICAL ERROR:\nThe '{backend}' simulation backend could not be initialized.\n\nThe application cannot run without the core simulation engine."
          )
        );
        return;
      }

      IsInitialized = true;

      if (_uiThreadDispatcher != null)
      {
        _uiThreadDispatcher.Dispatch(() => _ = CreateScene(populateDefault));
      }
      else
      {
        _ = CreateScene(populateDefault);
      }
      CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
        new ViewModels.SimulationInitializedMessage()
      );
    }
  }

  public void SetSceneDebugName(ulong sceneId, string name)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setSceneDebugName(_simulationContext, sceneId, name);
    }
  }

  public ulong CreatePresentationEngine(uint width, uint height, ulong sceneId)
  {
    if (_simulationContext == IntPtr.Zero)
      return 0;

    ulong id = NativeInterop.avkSimulationContext_createPresentationEngine(
      _simulationContext,
      width,
      height,
      sceneId
    );
    if (id != 0)
    {
      _activePresentationEngines.Add(id);
    }
    return id;
  }

  public ulong AddPerspectiveCamera(
    ulong sceneId,
    ulong presentationEngineId,
    string name,
    float fov,
    float near,
    float far
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return 0;

    // Create camera natively. FFI handles adding it to presentation engine
    ulong id = NativeInterop.avkSimulationContext_addPerspectiveCamera(
      _simulationContext,
      sceneId,
      presentationEngineId,
      name,
      fov,
      near,
      far
    );

    if (id > 0)
    {
      var entity = new Entity(sceneId, id, name);
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.EntityMap[id] = entity;
      WireEntityComponents(sceneId, entity);

      var root = GetEntityByName(sceneId, "root");

      SyncSceneHierarchy(sceneId);

      entity.Components.Add(new TransformComponent());
      entity.Components.Add(new CameraComponent());
    }

    return id;
  }

  /// <summary>
  /// Wires an existing scene entity as the camera for the given presentation engine.
  /// Prefer this over AddPerspectiveCamera when the default scene already has a "camera" entity.
  /// </summary>
  public bool SetCameraForPresentationEngine(
    ulong sceneId,
    ulong presentationEngineId,
    ulong cameraEntityId
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    return NativeInterop.avkSimulationContext_setCameraForPresentationEngine(
      _simulationContext,
      sceneId,
      presentationEngineId,
      cameraEntityId
    );
  }

  public void SetTransformComponent(
    ulong sceneId,
    ulong entityId,
    float px,
    float py,
    float pz,
    float rw,
    float rx,
    float ry,
    float rz,
    float sx,
    float sy,
    float sz
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return;
    var transform = new NativeInterop.FfiTransform
    {
      Px = px,
      Py = py,
      Pz = pz,
      Rw = rw,
      Rx = rx,
      Ry = ry,
      Rz = rz,
      Sx = sx,
      Sy = sy,
      Sz = sz,
    };
    NativeInterop.avkSimulationContext_setTransformComponent(
      _simulationContext,
      sceneId,
      entityId,
      in transform
    );
  }

  public void AddTransformComponent(
    ulong sceneId,
    ulong entityId,
    float px,
    float py,
    float pz,
    float rw,
    float rx,
    float ry,
    float rz,
    float sx,
    float sy,
    float sz
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return;
    NativeInterop.avkSimulationContext_addTransformComponent(
      _simulationContext,
      sceneId,
      entityId,
      px,
      py,
      pz,
      rw,
      rx,
      ry,
      rz,
      sx,
      sy,
      sz
    );
  }

  public ulong AddOrthographicCamera(
    ulong sceneId,
    ulong presentationEngineId,
    string name,
    float left,
    float right,
    float bottom,
    float top
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return 0;

    ulong id = NativeInterop.avkSimulationContext_addOrthographicCamera(
      _simulationContext,
      sceneId,
      presentationEngineId,
      name,
      left,
      right,
      bottom,
      top
    );

    if (id > 0)
    {
      var entity = new Entity(sceneId, id, name);
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.EntityMap[id] = entity;
      WireEntityComponents(sceneId, entity);

      var root = GetEntityByName(sceneId, "root");

      SyncSceneHierarchy(sceneId);

      entity.Components.Add(new TransformComponent());
      entity.Components.Add(new CameraComponent { IsOrthographic = true });
    }

    return id;
  }

  public void DestroyScene(ulong sceneId)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_destroyScene(_simulationContext, sceneId);
      }

      _sceneStateManager.RemoveScene(sceneId);
    }
  }

  public void DestroyPresentationEngine(ulong sceneId, ulong handle, ulong cameraId = 0)
  {
    if (_simulationContext != IntPtr.Zero && handle != 0)
    {
      _activePresentationEngines.Remove(handle);
      NativeInterop.avkSimulationContext_destroyPresentationEngine(
        _simulationContext,
        sceneId,
        handle
      );

      if (cameraId != 0)
      {
        RemoveEntityLocal(sceneId, cameraId);
      }
    }
  }

  private void RemoveEntityLocal(ulong sceneId, ulong id)
  {
    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    if (state.EntityMap.TryGetValue(id, out var entity))
    {
      foreach (var parent in state.EntityMap.Values)
      {
        if (parent.Children.Contains(entity))
        {
          parent.Children.Remove(entity);
          break;
        }
      }

      if (state.RootEntities.Contains(entity))
      {
        state.RootEntities.Remove(entity);
      }

      state.EntityMap.Remove(id);
    }
  }

  public void ResizePresentationEngine(ulong sceneId, ulong engineId, uint width, uint height)
  {
    if (_simulationContext != IntPtr.Zero && engineId != 0)
    {
      NativeInterop.avkSimulationContext_resize(
        _simulationContext,
        sceneId,
        engineId,
        width,
        height
      );
    }
  }

  public NativeInterop.FfiKinematicState? GetEphemerisPosition(int spkId, double epochTaiSec)
  {
    if (_simulationContext == IntPtr.Zero)
      return null;

    if (
      NativeInterop.avkSimulationContext_getEphemerisPosition(
        _simulationContext,
        spkId,
        epochTaiSec,
        out var state
      )
    )
    {
      return state;
    }
    return null;
  }

  public async Task<ulong> UpdateTrajectoryForSpkAsync(
    ulong sceneId,
    ulong entityId,
    int spkId,
    double startEpochTaiSec,
    double endEpochTaiSec,
    double sampleStepDays
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return 0;

    ulong taskId = NativeInterop.avkSimulationContext_updateTrajectoryForSpk(
      _simulationContext,
      sceneId,
      entityId,
      spkId,
      startEpochTaiSec,
      endEpochTaiSec,
      sampleStepDays
    );

    if (taskId != 0)
    {
      await PollTaskAsync(taskId);
      return taskId;
    }
    return 0;
  }

  public void LoadDefaultAlmanacs()
  {
    if (!IsInitialized)
      return;

    var exePath = System.AppDomain.CurrentDomain.BaseDirectory;
    var assetPath = System.IO.Path.Combine(exePath, "assets", "planets");
    if (System.IO.Directory.Exists(assetPath))
    {
      var files = System.IO.Directory.GetFiles(assetPath, "*.bsp");
      foreach (var file in files)
      {
        _ = LoadAlmanacFileAsync(file);
      }
    }
  }

  private void SyncEntities(ulong sceneId)
  {
    if (_simulationContext == IntPtr.Zero)
      return;

    var sceneState = _sceneStateManager.GetOrCreateScene(sceneId);

    // Allocate a single buffer for fetching strings efficiently
    IntPtr namePtr = Marshal.AllocHGlobal(256);

    foreach (var entity in sceneState.EntityMap.Values)
    {
      // Sync Entity Name dynamically
      if (
        NativeInterop.avkSimulationContext_getEntityName(
          _simulationContext,
          sceneId,
          entity.Id,
          namePtr,
          256
        )
      )
      {
        string? nativeName = Marshal.PtrToStringAnsi(namePtr);
        if (nativeName != null && nativeName != entity.Name)
        {
          entity.SuspendNameSync = true;
          entity.Name = nativeName;
          entity.SuspendNameSync = false;
        }
      }

      // Propagate NativeComponent sync manually here only as a fallback,
      // but ideally NativeSimulationCallback delta-syncs it.
      // E.g., at initial load.
      foreach (var comp in entity.Components.OfType<NativeComponent>())
      {
        comp.PullFromNative();
      }

      // Sync non-NativeComponent types that depend on Transform locally for now (e.g. Sun, Planet, Comet)
      var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
      if (transform != null)
      {
        var sun = entity.Components.OfType<SunComponent>().FirstOrDefault();
        if (sun != null)
        {
          sun.PositionX = transform.PosX;
          sun.PositionY = transform.PosY;
          sun.PositionZ = transform.PosZ;
        }

        var planet = entity.Components.OfType<PlanetComponent>().FirstOrDefault();
        if (planet != null)
        {
          planet.PositionX = transform.PosX;
          planet.PositionY = transform.PosY;
          planet.PositionZ = transform.PosZ;
        }

        var comet = entity.Components.OfType<CometComponent>().FirstOrDefault();
        if (comet != null)
        {
          comet.PositionX = transform.PosX;
          comet.PositionY = transform.PosY;
          comet.PositionZ = transform.PosZ;
        }
      }
    }

    Marshal.FreeHGlobal(namePtr);
  }

  public void ShutdownSimulation()
  {
    if (!IsInitialized)
      return;

    lock (_nativeLock)
    {
      NativeInterop.avkSimulationContext_shutdown(_simulationContext);
      _simulationContext = IntPtr.Zero;
    }

    IsInitialized = false;
  }

  public async Task<bool> DownloadImageAsync(
    ulong renderGeneration,
    IntPtr bufferPtr,
    nuint bufferSize
  )
  {
    lock (_nativeLock)
    {
      if (_simulationContext == IntPtr.Zero || _isDisposing)
        return false;
      System.Threading.Interlocked.Increment(ref _activeDownloads);
    }

    try
    {
      // Run the blocking FFI call (which waits for the GPU timeline semaphore and copies memory) on a background thread
      // to prevent blocking the Avalonia UI Thread.
      return await Task.Run(() =>
      {
        if (_simulationContext == IntPtr.Zero)
          return false;

        return NativeInterop.avkSimulationContext_downloadImage(
          _simulationContext,
          renderGeneration,
          bufferPtr,
          bufferSize
        );
      });
    }
    finally
    {
      lock (_nativeLock)
      {
        System.Threading.Interlocked.Decrement(ref _activeDownloads);
        if (_activeDownloads == 0 && _isDisposing)
        {
          System.Threading.Monitor.PulseAll(_nativeLock);
        }
      }
    }
  }

  public void RotateCamera(ulong sceneId, ulong cameraEntityId, float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_rotateCamera(
        _simulationContext,
        sceneId,
        cameraEntityId,
        deltaX,
        deltaY
      );
    }
  }

  public void ZoomCamera(ulong sceneId, ulong cameraEntityId, float amount)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_zoomCamera(
        _simulationContext,
        sceneId,
        cameraEntityId,
        amount
      );
    }
  }

  public void PanCursor(ulong sceneId, float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_panCursor(_simulationContext, sceneId, deltaX, deltaY);
    }
  }

  public void MoveCursor(ulong sceneId, float x, float y, float z)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_moveCursor(_simulationContext, sceneId, x, y, z);
    }
  }

  public void PanCamera(ulong sceneId, ulong cameraEntityId, float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_panCamera(
        _simulationContext,
        sceneId,
        cameraEntityId,
        deltaX,
        deltaY
      );
    }
  }

  public async Task<bool> LoadAlmanacFileAsync(string path)
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    ulong taskId = NativeInterop.avkSimulationContext_loadAlmanacFile(_simulationContext, path);
    await PollTaskAsync(taskId);
    return NativeInterop.avkSimulationContext_getTaskResultBool(_simulationContext, taskId);
  }

  public async Task<bool> UnloadAlmanacFileAsync(string path)
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    ulong taskId = NativeInterop.avkSimulationContext_unloadAlmanacFile(_simulationContext, path);
    await PollTaskAsync(taskId);
    return NativeInterop.avkSimulationContext_getTaskResultBool(_simulationContext, taskId);
  }

  public bool ParseEpochToTaiSec(string epochStr, out double taiSec)
  {
    return NativeInterop.avkSimulationContext_parseEpochToTaiSec(epochStr, out taiSec);
  }

  public async Task<NativeInterop.FfiKinematicState?> LoadCometSpkAsync(int spkid, string epoch_raw)
  {
    if (_simulationContext == IntPtr.Zero)
      return null;
    ulong taskId = NativeInterop.avkSimulationContext_loadCometSpk(
      _simulationContext,
      spkid,
      epoch_raw
    );
    await PollTaskAsync(taskId);
    if (
      NativeInterop.avkSimulationContext_getTaskResultKinematicState(
        _simulationContext,
        taskId,
        out var result
      )
    )
    {
      return result;
    }

    return null;
  }

  public async Task<ulong> ImportModelAsync(string path)
  {
    if (_simulationContext == IntPtr.Zero)
      return 0;
    ulong taskId = NativeInterop.avkSimulationContext_importModel(_simulationContext, path);
    await PollTaskAsync(taskId);
    return NativeInterop.avkSimulationContext_getTaskResultU64(_simulationContext, taskId);
  }

  public System.Collections.Generic.List<(ulong Id, string Path)> GetImportedModels()
  {
    var list = new System.Collections.Generic.List<(ulong Id, string Path)>();
    if (_simulationContext == IntPtr.Zero)
      return list;

    uint count = NativeInterop.avkSimulationContext_getImportedModelsCount(_simulationContext);
    if (count == 0)
      return list;

    IntPtr idsPtr = Marshal.AllocHGlobal((int)(count * sizeof(ulong)));
    IntPtr pathsPtr = Marshal.AllocHGlobal((int)(count * IntPtr.Size));

    uint actualCount = NativeInterop.avkSimulationContext_getImportedModels(
      _simulationContext,
      idsPtr,
      pathsPtr,
      count
    );

    for (int i = 0; i < actualCount; i++)
    {
      ulong id = (ulong)Marshal.ReadInt64(idsPtr, i * sizeof(ulong));
      IntPtr strPtr = Marshal.ReadIntPtr(pathsPtr, i * IntPtr.Size);
      string path = Marshal.PtrToStringAnsi(strPtr) ?? "";
      list.Add((id, path));
    }

    NativeInterop.avkSimulationContext_freeComponentNames(pathsPtr, actualCount);
    Marshal.FreeHGlobal(idsPtr);
    Marshal.FreeHGlobal(pathsPtr);

    return list;
  }

  public void UnloadModel(ulong modelId)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_unloadModel(_simulationContext, modelId);
      }
    }
  }

  public async Task<ulong> SpawnModelInstanceAsync(
    ulong sceneId,
    ulong modelId,
    string name,
    float posX = 0f,
    float posY = 0f,
    float posZ = 0f
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return 0;
    ulong taskId = NativeInterop.avkSimulationContext_spawnModelInstance(
      _simulationContext,
      modelId,
      name
    );
    await PollTaskAsync(taskId);
    ulong instanceId = NativeInterop.avkSimulationContext_getTaskResultU64(
      _simulationContext,
      taskId
    );

    if (instanceId > 0)
    {
      // Add to UI mirroring
      var entity = new Entity(sceneId, instanceId, name);
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.EntityMap[instanceId] = entity;
      WireEntityComponents(sceneId, entity);
      state.RootEntities.Add(entity);

      // Fetch basic components that were created
      entity.Components.Add(
        new TransformComponent
        {
          PosX = posX,
          PosY = posY,
          PosZ = posZ,
        }
      );
      entity.Components.Add(new CometComponent()); // Assume it spawns as a comet for now
    }

    return instanceId;
  }

  public async Task<ulong> SpawnCometAsync(
    ulong sceneId,
    ulong modelId,
    string name,
    float posX,
    float posY,
    float posZ,
    float rotW,
    float rotX,
    float rotY,
    float rotZ,
    float radiusKm,
    float massKg,
    uint physicsType = 0
  )
  {
    var result = SpawnComet(
      sceneId,
      modelId,
      name,
      posX,
      posY,
      posZ,
      rotW,
      rotX,
      rotY,
      rotZ,
      radiusKm,
      massKg,
      physicsType
    );
    return result.CometEntityId;
  }

  public (ulong LcaFrameId, ulong CometEntityId) SpawnComet(
    ulong sceneId,
    ulong modelId,
    string name,
    float posX,
    float posY,
    float posZ,
    float rotW,
    float rotX,
    float rotY,
    float rotZ,
    float radiusKm,
    float massKg,
    uint physicsType
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return (0, 0);

    if (
      NativeInterop.avkSimulationContext_spawnComet(
        _simulationContext,
        sceneId,
        modelId,
        name,
        posX,
        posY,
        posZ,
        rotW,
        rotX,
        rotY,
        rotZ,
        radiusKm,
        massKg,
        physicsType,
        out var result
      )
    )
    {
      var lcaEntity = new Entity(sceneId, result.LcaFrameId, name + "_LCA");
      var cometEntity = new Entity(sceneId, result.CometEntityId, name);

      var state = _sceneStateManager.GetOrCreateScene(sceneId);

      state.EntityMap[result.LcaFrameId] = lcaEntity;
      state.EntityMap[result.CometEntityId] = cometEntity;

      WireEntityComponents(sceneId, lcaEntity);
      WireEntityComponents(sceneId, cometEntity);

      // Query the true Rust-side components.  WireEntityComponents sets up
      // the CollectionChanged handler so any NativeComponent added to Components
      // gets BindToNative + PullFromNative called immediately.
      //
      // Add empty shells — PullFromNative will populate them from the Rust ECS.
      lcaEntity.Components.Add(new TransformComponent());

      cometEntity.Components.Add(new TransformComponent());
      cometEntity.Components.Add(new ParticleEmitterCirclesComponent());
      cometEntity.Components.Add(new SphereGizmoComponent());
      cometEntity.Components.Add(new CometComponent());
      var root = GetEntityByName(sceneId, "root");

      SyncSceneHierarchy(sceneId);

      WeakReferenceMessenger.Default.Send(
        new AetherVk.Logic.Messages.SimulationStateUpdatedMessage(sceneId)
      );

      return (result.LcaFrameId, result.CometEntityId);
    }

    return (0, 0);
  }

  public ulong SpawnTrajectory(
    ulong sceneId,
    ulong parentEntity,
    string name,
    TrajectoryGpu trajectory,
    RationalBezierGpu[] segments
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return 0;

    var entityId = NativeInterop.avkSimulationContext_spawnTrajectory(
      _simulationContext,
      sceneId,
      parentEntity,
      name,
      ref trajectory,
      segments,
      (uint)segments.Length
    );

    if (entityId > 0)
    {
      var trajEntity = new Entity(sceneId, entityId, name);
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.EntityMap[entityId] = trajEntity;

      // If parentEntity is 0, add to root (though usually it will be RootEntity.EntityId)
      // We assume it's attached to root for now
      if (
        parentEntity == 0
        || (state.RootEntities.Count > 0 && state.RootEntities[0].Id == parentEntity)
      )
      {
        state.RootEntities.FirstOrDefault()?.Children.Add(trajEntity);
      }
      else if (state.EntityMap.TryGetValue(parentEntity, out var parent))
      {
        parent.Children.Add(trajEntity);
      }
    }

    return entityId;
  }

  public ulong SpawnTrajectoryFromElements(
    ulong sceneId,
    string name,
    double a,
    double e,
    double iDeg,
    double omegaNodeDeg,
    double argPeriDeg,
    float lineWidth = 2.0f
  )
  {
    // a is in AU (converted at parse time in HorizonJplService)
    // All Bézier construction now happens in Rust via avkSimulationContext_spawnTrajectoryFromElements.
    if (_simulationContext == IntPtr.Zero)
      return 0;

    ulong rootEntity = GetRootEntityId(sceneId);

    var entityId = NativeInterop.avkSimulationContext_spawnTrajectoryFromElements(
      _simulationContext,
      sceneId,
      rootEntity,
      name,
      a,
      e,
      iDeg,
      omegaNodeDeg,
      argPeriDeg,
      1.0f,
      1.0f,
      1.0f,
      0.5f, // white, 50% alpha
      lineWidth
    );

    if (entityId > 0)
    {
      var trajEntity = new Entity(sceneId, entityId, name);
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.EntityMap[entityId] = trajEntity;
      // Trajectory always sits at identity (Rust sets pos=0, rot=I, scale=1).
      trajEntity.Components.Add(new TransformComponent());
      state.RootEntities.FirstOrDefault()?.Children.Add(trajEntity);
    }

    return entityId;
  }

  private ulong GetRootEntityId(ulong sceneId)
  {
    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    var root = state.RootEntities.FirstOrDefault(e => e.Name == "root");
    return root?.Id ?? 0;
  }

  // ─── New post-spawn component setters ────────────────────────────────────────

  /// <summary>
  /// Patches the NAIF/SPK id into a Kinematic comet entity's AlmanacPlanet component.
  /// Must be called after SpawnComet(physicsType=1) so the logic thread queries the
  /// correct body instead of the placeholder id 0 (Solar System Barycenter).
  /// </summary>
  public bool SetAlmanacPlanetNaifId(ulong sceneId, ulong entityId, int naifId)
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    var ok = NativeInterop.avkSimulationContext_setAlmanacPlanetNaifId(
      _simulationContext,
      sceneId,
      entityId,
      naifId
    );
    if (!ok)
      _consoleService.Log(
        $"[Runtime] SetAlmanacPlanetNaifId: entity {entityId} has no AlmanacPlanet component."
      );
    return ok;
  }

  /// <summary>
  /// Injects the initial velocity (km/s, ecliptic J2000) into a Dynamic comet entity.
  /// Rust converts km/s → AU/s before storing on KinematicComponent.velocity.
  /// Must be called after SpawnComet(physicsType=2) and before simulation start.
  /// </summary>
  public bool SetKinematicVelocity(
    ulong sceneId,
    ulong entityId,
    float vxKmS,
    float vyKmS,
    float vzKmS
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    var ok = NativeInterop.avkSimulationContext_setKinematicVelocity(
      _simulationContext,
      sceneId,
      entityId,
      vxKmS,
      vyKmS,
      vzKmS
    );
    if (!ok)
      _consoleService.Log(
        $"[Runtime] SetKinematicVelocity: entity {entityId} has no KinematicComponent."
      );
    return ok;
  }

  /// <summary>
  /// Shows or hides an entity, keeping the C# Entity.IsVisible and Rust HiddenComponent in sync.
  /// Setting visible=false will add HiddenComponent in Rust AND set Entity.IsVisible=false in C#
  /// so the outline eye-icon reflects the true state.
  /// </summary>
  public bool SetEntityHidden(ulong sceneId, ulong entityId, bool visible)
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    if (state.EntityMap.TryGetValue(entityId, out var entity))
    {
      // Setting IsVisible fires OnIsVisibleChanged → EntityVisibilityChangedMessage
      // → NativeRuntimeService message handler → avkSimulationContext_setEntityVisibility.
      // No need to call the FFI directly; the messenger does it.
      entity.IsVisible = visible;
      return true;
    }
    // Fallback: entity not yet in C# map — call FFI directly.
    NativeInterop.avkSimulationContext_setEntityVisibility(
      _simulationContext,
      sceneId,
      entityId,
      visible
    );
    return true;
  }

  /// <summary>
  /// Shows or hides an entity (adds/removes HiddenComponent in Rust only).
  /// Prefer SetEntityHidden when the entity is registered in the C# EntityMap.
  /// </summary>
  public bool SetEntityVisibility(ulong sceneId, ulong entityId, bool visible)
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    // avkSimulationContext_setEntityVisibility returns void; treat call success as true.
    NativeInterop.avkSimulationContext_setEntityVisibility(
      _simulationContext,
      sceneId,
      entityId,
      visible
    );
    return true;
  }

  // ─── Kepler → Cartesian initial conditions for Dynamic spawn ──────────────────

  /// <summary>
  /// Converts Keplerian osculating elements to ecliptic J2000 position (AU) and
  /// velocity (km/s). Uses Newton-Raphson to solve Kepler's equation and the vis-viva
  /// relation for speed. Returns false when eccentricity ≥ 1 (parabolic/hyperbolic).
  /// </summary>
  public static bool KeplerToCartesian(
    double a_au, // semi-major axis in AU
    double e, // eccentricity
    double i_deg, // inclination (degrees)
    double om_deg, // longitude of ascending node Ω (degrees)
    double w_deg, // argument of periapsis ω (degrees)
    double M_deg, // mean anomaly M (degrees)
    out double px,
    out double py,
    out double pz,
    out double vx,
    out double vy,
    out double vz
  )
  {
    px = py = pz = vx = vy = vz = 0.0;
    if (e >= 1.0)
      return false; // not an elliptic orbit

    const double AU = 149_597_870.7; // km per AU
    const double GM_SUN = 1.32712440018e11; // km³/s²

    double a = a_au * AU;
    double i = i_deg * Math.PI / 180.0;
    double Om = om_deg * Math.PI / 180.0;
    double w = w_deg * Math.PI / 180.0;
    double M = M_deg * Math.PI / 180.0;

    // Solve Kepler's equation M = E - e·sin(E) via Newton-Raphson
    double E = M;
    for (int iter = 0; iter < 100; iter++)
    {
      double dE = (M - E + e * Math.Sin(E)) / (1.0 - e * Math.Cos(E));
      E += dE;
      if (Math.Abs(dE) < 1e-12)
        break;
    }

    // True anomaly
    double nu =
      2.0
      * Math.Atan2(Math.Sqrt(1.0 + e) * Math.Sin(E / 2.0), Math.Sqrt(1.0 - e) * Math.Cos(E / 2.0));

    // Perifocal position (km) and velocity (km/s)
    double p = a * (1.0 - e * e);
    double r = p / (1.0 + e * Math.Cos(nu));
    double cosV = Math.Cos(nu),
      sinV = Math.Sin(nu);
    double sqGMp = Math.Sqrt(GM_SUN / p);

    double xp = r * cosV;
    double yp = r * sinV;
    double vxp = -sqGMp * sinV;
    double vyp = sqGMp * (e + cosV);

    // Perifocal → ecliptic J2000 rotation matrix R = Rz(-Ω)·Rx(-i)·Rz(-ω)
    double cosO = Math.Cos(Om),
      sinO = Math.Sin(Om);
    double cosI = Math.Cos(i),
      sinI = Math.Sin(i);
    double cosW = Math.Cos(w),
      sinW = Math.Sin(w);

    double Rxx = cosO * cosW - sinO * sinW * cosI;
    double Rxy = -cosO * sinW - sinO * cosW * cosI;
    double Ryx = sinO * cosW + cosO * sinW * cosI;
    double Ryy = -sinO * sinW + cosO * cosW * cosI;
    double Rzx = sinW * sinI;
    double Rzy = cosW * sinI;

    // Position in AU
    px = (Rxx * xp + Rxy * yp) / AU;
    py = (Ryx * xp + Ryy * yp) / AU;
    pz = (Rzx * xp + Rzy * yp) / AU;

    // Velocity in km/s
    vx = Rxx * vxp + Rxy * vyp;
    vy = Ryx * vxp + Ryy * vyp;
    vz = Rzx * vxp + Rzy * vyp;

    return true;
  }

  // Rotation helpers used by SpawnTrajectoryFromElements
  private static (double x, double y, double z) RotateZ(
    double x,
    double y,
    double z,
    double angle
  ) => (x * Math.Cos(angle) - y * Math.Sin(angle), x * Math.Sin(angle) + y * Math.Cos(angle), z);

  private static (double x, double y, double z) RotateX(
    double x,
    double y,
    double z,
    double angle
  ) => (x, y * Math.Cos(angle) - z * Math.Sin(angle), y * Math.Sin(angle) + z * Math.Cos(angle));

  public Entity? SpawnEntity(ulong sceneId, string name)
  {
    if (_simulationContext == IntPtr.Zero)
      return null;
    ulong id = NativeInterop.avkSimulationContext_spawnEntity(_simulationContext, sceneId, name);
    if (id > 0)
    {
      var entity = new Entity(sceneId, id, name);
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.EntityMap[id] = entity;
      state.RootEntities.Add(entity);
      WireEntityComponents(sceneId, entity);
      return entity;
    }
    return null;
  }

  public bool SetParent(ulong sceneId, ulong entityId, ulong parentId)
  {
    if (_simulationContext == IntPtr.Zero)
      return false;
    bool success = NativeInterop.avkSimulationContext_setParent(
      _simulationContext,
      sceneId,
      entityId,
      parentId
    );
    if (success)
    {
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      if (
        state.EntityMap.TryGetValue(entityId, out var entity)
        && state.EntityMap.TryGetValue(parentId, out var parent)
      )
      {
        if (state.RootEntities.Contains(entity))
        {
          state.RootEntities.Remove(entity);
        }
        parent.Children.Add(entity);
      }
    }
    return success;
  }

  /// <summary>
  /// Retrieves the model local frames (user-defined and simulation frames).
  /// </summary>
  public bool GetModelLocalFrames(
    ulong modelId,
    out NativeInterop.FfiMat3 userFrame,
    out NativeInterop.FfiMat3 simFrame
  )
  {
    userFrame = default;
    simFrame = default;
    if (_simulationContext == IntPtr.Zero)
      return false;

    return NativeInterop.avkSimulationContext_getModelLocalFrames(
      _simulationContext,
      modelId,
      out userFrame,
      out simFrame
    );
  }

  /// <summary>
  /// Overrides the physical properties (mass, radius, inertia) of a model,
  /// assuming a spherical shape, and aligns its simulation frame with the given user frame.
  /// </summary>
  public bool OverrideModelSpherical(
    ulong modelId,
    float radiusKm,
    double massKg,
    ref NativeInterop.FfiMat3 userFrame
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return false;

    return NativeInterop.avkSimulationContext_overrideModelSpherical(
      _simulationContext,
      modelId,
      radiusKm,
      (float)massKg,
      ref userFrame
    );
  }

  /// <summary>
  /// Synchronously spawns a static mesh entity hierarchy: an LCA micro-frame parent entity and
  /// the static mesh child entity. Both are mirrored into the C# scene tree.
  /// </summary>
  /// <returns>
  /// A tuple of (lcaFrameId, meshEntityId). Both are 0 on failure.
  /// </returns>
  public (ulong lcaFrameId, ulong meshEntityId) SpawnStaticMesh(
    ulong sceneId,
    ulong modelId,
    string entityName,
    float posX,
    float posY,
    float posZ,
    float rotW,
    float rotX,
    float rotY,
    float rotZ,
    float radiusKm
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return (0, 0);

    bool ok = NativeInterop.avkSimulationContext_spawnStaticMesh(
      _simulationContext,
      sceneId,
      modelId,
      entityName,
      posX,
      posY,
      posZ,
      rotW,
      rotX,
      rotY,
      rotZ,
      radiusKm,
      out var ffiResult
    );

    if (!ok)
      return (0, 0);

    var state = _sceneStateManager.GetOrCreateScene(sceneId);

    // ── Mirror LCA micro-frame entity ────────────────────────────────────────
    var lcaFrameName = $"{entityName}_microframe";
    var lcaEntity = new Entity(sceneId, ffiResult.LcaFrameId, lcaFrameName);
    state.EntityMap[ffiResult.LcaFrameId] = lcaEntity;
    WireEntityComponents(sceneId, lcaEntity);
    lcaEntity.Components.Add(new TransformComponent()); // PullFromNative fills position/scale

    // Nest under root
    var root = GetEntityByName(sceneId, "root");

    // ── Mirror static mesh entity ─────────────────────────────────────────────
    var meshEntity = new Entity(sceneId, ffiResult.MeshEntityId, entityName);
    state.EntityMap[ffiResult.MeshEntityId] = meshEntity;
    WireEntityComponents(sceneId, meshEntity);
    meshEntity.Components.Add(new TransformComponent()); // PullFromNative fills pos/scale
    meshEntity.Components.Add(new SphereGizmoComponent());
    meshEntity.Components.Add(new ParticleEmitterCirclesComponent());

    // Nest under LCA frame
    SyncSceneHierarchy(sceneId);

    WeakReferenceMessenger.Default.Send(
      new AetherVk.Logic.Messages.SimulationStateUpdatedMessage(sceneId)
    );

    return (ffiResult.LcaFrameId, ffiResult.MeshEntityId);
  }

  public string[] GetLoadedAlmanacFiles()
  {
    if (_simulationContext != IntPtr.Zero)
    {
      IntPtr ptr = NativeInterop.avkSimulationContext_getAlmanacLoadedFiles(
        _simulationContext,
        out uint count
      );
      if (ptr == IntPtr.Zero || count == 0)
        return Array.Empty<string>();

      var result = new string[count];
      for (int i = 0; i < count; i++)
      {
        IntPtr strPtr = Marshal.ReadIntPtr(ptr, i * IntPtr.Size);
        result[i] = Marshal.PtrToStringAnsi(strPtr) ?? "";
      }

      NativeInterop.avkFreeStringArray(ptr, count);
      return result;
    }

    return Array.Empty<string>();
  }

  public void SetTimeScale(ulong sceneId, uint scale)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setTimeScale(_simulationContext, sceneId, scale);
    }
  }

  public void PlayScene(ulong sceneId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_playScene(_simulationContext, sceneId);
    }
  }

  public void PauseScene(ulong sceneId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_pauseScene(_simulationContext, sceneId);
    }
  }

  public ulong SpawnBillboard(
    ulong sceneId,
    string imagePath,
    float ndcX,
    float ndcY,
    float scale,
    float opacity,
    ulong viewportId
  )
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        if (
          NativeInterop.avkSimulationContext_spawnBillboard(
            _simulationContext,
            sceneId,
            imagePath,
            out ulong entityId
          )
        )
        {
          // Also add the screen-space billboard component with full state
          NativeInterop.avkSimulationContext_addScreenSpaceBillboard(
            _simulationContext,
            sceneId,
            entityId,
            imagePath,
            ndcX,
            ndcY,
            scale,
            0.0f,
            opacity,
            1,
            viewportId
          );

          var entity = new Entity(
            sceneId,
            entityId,
            $"Billboard ({System.IO.Path.GetFileName(imagePath)})"
          );
          var state = _sceneStateManager.GetOrCreateScene(sceneId);
          state.EntityMap[entityId] = entity;
          state.RootEntities.Add(entity);
          return entityId;
        }
      }
      return 0;
    }
  }

  public void SetScreenSpaceBillboard(
    ulong sceneId,
    ulong entityId,
    float ndcX,
    float ndcY,
    float scale,
    float rotationDeg,
    float opacity,
    int zIndex
  )
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_setScreenSpaceBillboard(
          _simulationContext,
          sceneId,
          entityId,
          ndcX,
          ndcY,
          scale,
          rotationDeg,
          opacity,
          zIndex
        );
      }
    }
  }

  public bool GetScreenSpaceBillboard(
    ulong sceneId,
    ulong entityId,
    out NativeInterop.FfiScreenSpaceBillboard data
  )
  {
    data = default;
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        return NativeInterop.avkSimulationContext_getScreenSpaceBillboard(
          _simulationContext,
          sceneId,
          entityId,
          out data
        );
      }
      return false;
    }
  }

  public double GetSimulationTime(ulong sceneId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      return NativeInterop.avkSimulationContext_getSimulationTime(_simulationContext, sceneId);
    }

    return 0.0;
  }

  public string GetSimulationTimeUtc(ulong sceneId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      IntPtr ptr = Marshal.AllocHGlobal(256);
      if (
        NativeInterop.avkSimulationContext_getSimulationTimeUtc(
          _simulationContext,
          sceneId,
          ptr,
          256
        )
      )
      {
        var result = Marshal.PtrToStringAnsi(ptr) ?? "";
        Marshal.FreeHGlobal(ptr);
        return result;
      }

      Marshal.FreeHGlobal(ptr);
    }

    return "UNKNOWN";
  }

  public void SetSimulationTime(ulong sceneId, double timeTai)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setSimulationTime(_simulationContext, sceneId, timeTai);
    }
  }

  /// <summary>
  /// Seeks the simulation to a specific epoch, recomputing body positions from ephemeris data
  /// and marking the scene for TLAS rebuild. Use this instead of SetSimulationTime when you
  /// need positions to update visually (e.g., during timeline scrubbing).
  /// </summary>
  public void SeekEpoch(ulong sceneId, double epochTai)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_seekEpoch(_simulationContext, sceneId, epochTai);
    }
  }

  public bool GetEpochLimits(ulong sceneId, out double startTai, out double endTai)
  {
    startTai = 0.0;
    endTai = 0.0;
    if (_simulationContext != IntPtr.Zero)
    {
      return NativeInterop.avkSimulationContext_getEpochLimits(
        _simulationContext,
        sceneId,
        out startTai,
        out endTai
      );
    }

    return false;
  }

  public async Task<(bool hit, ulong entityId, float px, float py, float pz)> RaycastNdcAsync(
    ulong sceneId,
    ulong cameraId,
    float ndcX,
    float ndcY
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return (false, 0UL, 0f, 0f, 0f);

    ulong taskId = NativeInterop.avkSimulationContext_raycastNdc(
      _simulationContext,
      sceneId,
      cameraId,
      ndcX,
      ndcY
    );
    await PollTaskAsync(taskId);

    if (
      NativeInterop.avkSimulationContext_getTaskResultRaycast(
        _simulationContext,
        taskId,
        out var result
      )
    )
    {
      return (result.Hit, result.Entity, result.Px, result.Py, result.Pz);
    }

    return (false, 0UL, 0f, 0f, 0f);
  }

  public void ResetCamera(ulong sceneId, ulong cameraEntityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_resetCamera(
        _simulationContext,
        sceneId,
        cameraEntityId
      );
    }
  }

  public void SnapToEntity(ulong sceneId, ulong cameraEntityId, ulong entityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_snapToEntity(
        _simulationContext,
        sceneId,
        cameraEntityId,
        entityId
      );
    }
  }

  public void FollowEntity(ulong sceneId, ulong cameraEntityId, ulong entityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_followEntity(
        _simulationContext,
        sceneId,
        cameraEntityId,
        entityId,
        true
      );
    }
  }

  public void UnfollowEntity(ulong sceneId, ulong cameraEntityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_unfollowEntity(
        _simulationContext,
        sceneId,
        cameraEntityId
      );
    }
  }

  public void SyncMarkers(ulong sceneId, ulong entityId, CometComponent comet)
  {
    if (_simulationContext == IntPtr.Zero)
      return;

    int count = comet.Jets.Count;
    float[] px = new float[count];
    float[] py = new float[count];
    float[] pz = new float[count];
    float[] cr = new float[count];
    float[] cg = new float[count];
    float[] cb = new float[count];
    float[] sizes = new float[count];

    for (int i = 0; i < count; i++)
    {
      var jet = comet.Jets[i];
      px[i] = jet.PosX;
      py[i] = jet.PosY;
      pz[i] = jet.PosZ;
      cr[i] = jet.ColorR;
      cg[i] = jet.ColorG;
      cb[i] = jet.ColorB;
      sizes[i] = jet.Size;
    }

    NativeInterop.avkSimulationContext_setMarkers(
      _simulationContext,
      sceneId,
      entityId,
      (uint)count,
      px,
      py,
      pz,
      cr,
      cg,
      cb,
      sizes
    );
  }

  public ulong CreateScene(bool populateDefault = true)
  {
    System.Console.WriteLine(
      $"[CreateScene] Called. populateDefault={populateDefault}  ctx={((_simulationContext != IntPtr.Zero) ? "valid" : "NULL")}"
    );

    if (_simulationContext != IntPtr.Zero)
    {
      ulong sceneId;
      if (populateDefault)
      {
        sceneId = NativeInterop.avkSimulationContext_createDefaultScene(_simulationContext);
      }
      else
      {
        sceneId = NativeInterop.avkSimulationContext_createEmptyScene(_simulationContext);
      }

      System.Console.WriteLine($"[CreateScene] createDefaultScene returned sceneId={sceneId}");

      if (sceneId == 0)
      {
        System.Console.WriteLine(
          "[CreateScene] ERROR: native scene creation returned 0. Check Rust logs for details."
        );
        return 0; // Don't create a phantom SceneState(0) in the manager.
      }

      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.Clear();

      uint count = NativeInterop.avkSimulationContext_getEntityCount(_simulationContext, sceneId);
      System.Console.WriteLine($"[CreateScene] getEntityCount({sceneId}) = {count}");

      if (count > 0)
      {
        IntPtr idsPtr = Marshal.AllocHGlobal((int)count * sizeof(long));
        NativeInterop.avkSimulationContext_getEntityIds(_simulationContext, sceneId, idsPtr, count);

        long[] ids = new long[count];
        Marshal.Copy(idsPtr, ids, 0, (int)count);
        Marshal.FreeHGlobal(idsPtr);

        System.Console.WriteLine(
          $"[CreateScene] Found {count} native entities for Scene {sceneId}."
        );

        IntPtr namePtr = Marshal.AllocHGlobal(256);
        foreach (long signedId in ids)
        {
          ulong id = (ulong)signedId;
          string name = "Entity";
          if (
            NativeInterop.avkSimulationContext_getEntityName(
              _simulationContext,
              sceneId,
              id,
              namePtr,
              256
            )
          )
          {
            name = Marshal.PtrToStringAnsi(namePtr) ?? name;
          }

          var entity = new Entity(sceneId, id, name);
          state.EntityMap[id] = entity;
          WireEntityComponents(sceneId, entity);
        }

        Marshal.FreeHGlobal(namePtr);

        // Build hierarchy & default components based on FFI entity type heuristics
        foreach (long signedId in ids)
        {
          ulong id = (ulong)signedId;
          var entity = state.EntityMap[id];
          ulong parentId = NativeInterop.avkSimulationContext_getEntityParent(
            _simulationContext,
            sceneId,
            id
          );
          if (parentId != 0 && state.EntityMap.TryGetValue(parentId, out var parent))
          {
            parent.Children.Add(entity);
            System.Console.WriteLine(
              $"[CreateScene] Parented {entity.Name} ({entity.Id}) to {parent.Name} ({parent.Id})"
            );
          }
          else
          {
            state.RootEntities.Add(entity);
            System.Console.WriteLine(
              $"[CreateScene] Added {entity.Name} ({entity.Id}) to RootEntities."
            );
          }

          // Fetch basic transform logic
          entity.Components.Add(new TransformComponent());

          // Add UI mirrored components by FFI inspection heuristic
          // TODO: No. Do not use heuristic. Add a function which queries the list of components present, and we decide which to spawn
          if (entity.Name == "camera")
            entity.Components.Add(new CameraComponent());
          if (entity.Name == "cursor")
            entity.Components.Add(new CursorComponent());
          if (entity.Name == "sun")
            entity.Components.Add(new SunComponent());
          // TODO: remove sun core. nucleus of sun is included in sun entity itself
          if (entity.Name == "sun_core")
            entity.Components.Add(new CometComponent());
          if (entity.Name == "grid")
            entity.Components.Add(new GridComponent());
          if (entity.Name.Contains("measurement", StringComparison.OrdinalIgnoreCase))
            entity.Components.Add(new MeasurementComponent());
        }

        SyncEntities(sceneId); // Immediately populate real positions
      }

      // Signal all subscribers (e.g. Viewport3DViewModel, OutlineViewModel) that the
      // scene is ready (even if entity count was 0 — they will find no entities and
      // log appropriately rather than leaving CameraId = 0 permanently).
      // This resolves the timing race where IsInitialized fires before CreateScene runs.
      System.Console.WriteLine(
        $"[CreateScene] Sending SimulationStateUpdatedMessage(sceneId={sceneId})"
      );
      WeakReferenceMessenger.Default.Send(
        new AetherVk.Logic.Messages.SimulationStateUpdatedMessage(sceneId)
      );

      return sceneId;
    }

    System.Console.WriteLine("[CreateScene] ERROR: _simulationContext is null/zero!");
    return 0;
  }

  public bool SnapshotScene(ulong sceneId)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        return NativeInterop.avkSimulationContext_snapshotScene(_simulationContext, sceneId);
      }
      return false;
    }
  }

  public bool RestoreSnapshot(ulong sceneId)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        bool result = NativeInterop.avkSimulationContext_restoreSnapshot(
          _simulationContext,
          sceneId
        );
        if (result)
        {
          // Sync entities so UI updates immediately
          SyncEntities(sceneId);
          if (_uiThreadDispatcher != null)
          {
            _uiThreadDispatcher.Dispatch(() =>
            {
              WeakReferenceMessenger.Default.Send(
                new AetherVk.Logic.Messages.SimulationStateUpdatedMessage(sceneId)
              );
            });
          }
          return true;
        }
      }
      return false;
    }
  }

  private void WireEntityComponents(ulong sceneId, Entity entity)
  {
    entity.Components.CollectionChanged += (sender, args) =>
    {
      if (args.NewItems == null)
        return;
      foreach (var item in args.NewItems)
      {
        // Polymorphic Auto-Wiring!
        if (item is NativeComponent nativeComp)
        {
          nativeComp.BindToNative(_simulationContext, sceneId, entity.Id);

          // Fetch initial state immediately upon UI creation
          if (_simulationContext != IntPtr.Zero)
            nativeComp.PullFromNative();
        }

        // Retain custom nested sub-collection bindings like Jets if needed
        if (item is CometComponent comet)
        {
          comet.Jets.CollectionChanged += (s, e) =>
          {
            if (_simulationContext != IntPtr.Zero)
              SyncMarkers(sceneId, entity.Id, comet);
            if (e.NewItems != null)
            {
              foreach (JetMarker jet in e.NewItems)
              {
                if (_simulationContext != IntPtr.Zero)
                {
                  NativeInterop.avkSimulationContext_addJet(
                    _simulationContext,
                    sceneId,
                    entity.Id,
                    jet.RadiusKm,
                    jet.Latitude,
                    jet.Longitude,
                    jet.ColorR,
                    jet.ColorG,
                    jet.ColorB,
                    jet.Mass,
                    jet.ParticlesPerTick,
                    jet.TTL,
                    jet.MeanVelocity
                  );
                }
                jet.PropertyChanged += (js, je) =>
                {
                  SyncMarkers(sceneId, entity.Id, comet);
                };
              }
            }
          };
        }
      }
    };
  }

  public void RefreshBvhNodes(ulong sceneId, ulong entityId, CometComponent comet)
  {
    if (_simulationContext == IntPtr.Zero)
      return;

    comet.BvhTree.Clear();

    IntPtr ptr = NativeInterop.avkSimulationContext_getBvhNodes(
      _simulationContext,
      sceneId,
      entityId,
      out uint count
    );
    if (ptr == IntPtr.Zero || count == 0)
      return;

    var nodes = new NativeInterop.FfiBvhNode[count];
    for (int i = 0; i < count; i++)
    {
      nodes[i] = Marshal.PtrToStructure<NativeInterop.FfiBvhNode>(
        ptr + i * Marshal.SizeOf<NativeInterop.FfiBvhNode>()
      );
    }

    NativeInterop.avkSimulationContext_freeBvhNodes(ptr, count);

    BvhNode? BuildNode(uint index)
    {
      if (index >= count)
        return null;
      var ffiNode = nodes[index];

      var node = new BvhNode
      {
        SceneId = sceneId,
        EntityId = entityId,
        Index = index,
        Name = ffiNode.PrimitiveCount > 0 ? "Leaf Node" : "Inner Node",
      };

      if (ffiNode.NodeType == 0)
      {
        node.Type = BvhNodeType.AABB;
        node.Details =
          $"Min: ({ffiNode.MinX:F1}, {ffiNode.MinY:F1}, {ffiNode.MinZ:F1}), Max: ({ffiNode.MaxX:F1}, {ffiNode.MaxY:F1}, {ffiNode.MaxZ:F1})";
      }
      else
      {
        node.Type = BvhNodeType.OBB;
        node.Details =
          $"Center: ({ffiNode.CenterX:F1}, {ffiNode.CenterY:F1}, {ffiNode.CenterZ:F1}), Extents: ({ffiNode.ExtentsX:F1}, {ffiNode.ExtentsY:F1}, {ffiNode.ExtentsZ:F1})";
      }

      if (ffiNode.PrimitiveCount == 0)
      {
        if (ffiNode.LeftChild != uint.MaxValue)
        {
          var left = BuildNode(ffiNode.LeftChild);
          if (left != null)
            node.Children.Add(left);
        }

        if (ffiNode.RightChild != uint.MaxValue)
        {
          var right = BuildNode(ffiNode.RightChild);
          if (right != null)
            node.Children.Add(right);
        }
      }

      return node;
    }

    if (count > 0)
    {
      var root = BuildNode(0);
      if (root != null)
      {
        comet.BvhTree.Add(root);
      }
    }
  }

  public void RecalculateJetPoints(ulong sceneId, ulong entityId)
  {
    if (_simulationContext == IntPtr.Zero)
      return;

    NativeInterop.avkSimulationContext_recalculateJetPoints(_simulationContext, sceneId, entityId);
  }

  public ulong SpawnProceduralSphere(ulong sceneId, string name, float radius, float mass)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        ulong id = NativeInterop.avkSimulationContext_spawnProceduralSphere(
          _simulationContext,
          sceneId,
          name,
          radius,
          mass
        );
        if (id > 0)
        {
          var entity = new Entity(sceneId, id, name);
          var state = _sceneStateManager.GetOrCreateScene(sceneId);
          state.EntityMap[id] = entity;
          WireEntityComponents(sceneId, entity);
          // For now, don't automatically add to RootEntities if it's a test mesh
          // or maybe we should? Tests might want to check RootEntities.
          // Let's add it to a child of root if possible.
          var root = GetEntityByName(sceneId, "root");

          SyncSceneHierarchy(sceneId);

          return id;
        }
      }
    }

    return 0;
  }

  public Entity SpawnEntity(ulong sceneId, string name, Entity? parent = null)
  {
    // Native spawn
    ulong nativeId = 0;
    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    if (_simulationContext != IntPtr.Zero)
    {
      nativeId = NativeInterop.avkSimulationContext_spawnEntity(_simulationContext, sceneId, name);
    }
    else
    {
      nativeId = (ulong)state.EntityMap.Count + 1; // Fallback mock ID
    }

    var entity = new Entity(sceneId, nativeId, name);
    state.EntityMap[nativeId] = entity;

    WireEntityComponents(sceneId, entity);

    if (parent != null)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_setParent(
          _simulationContext,
          sceneId,
          nativeId,
          parent.Id
        );
      }
    }

    SyncSceneHierarchy(sceneId);

    return entity;
  }

  public Entity CreateGrid(ulong sceneId, Entity parent)
  {
    var grid = SpawnEntity(sceneId, "grid", parent);
    grid.Components.Add(new GridComponent());
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addGridComponent(_simulationContext, sceneId, grid.Id);
    }

    return grid;
  }

  public Entity CreateSky(ulong sceneId, Entity parent)
  {
    var sky = SpawnEntity(sceneId, "sky", parent);
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addSkyComponent(_simulationContext, sceneId, sky.Id);
    }

    return sky;
  }

  public Entity CreateCursor(ulong sceneId, Entity parent)
  {
    var cursor = SpawnEntity(sceneId, "cursor", parent);
    cursor.Components.Add(new TransformComponent());
    cursor.Components.Add(new CursorComponent());
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addTransformComponent(
        _simulationContext,
        sceneId,
        cursor.Id,
        0f,
        0f,
        0f,
        1f,
        0f,
        0f,
        0f,
        1f,
        1f,
        1f
      );
      NativeInterop.avkSimulationContext_addCursorComponent(_simulationContext, sceneId, cursor.Id);
    }

    return cursor;
  }

  public Entity CreateSun(
    ulong sceneId,
    Entity parent,
    uint resX = 128,
    uint resY = 128,
    uint resZ = 128
  )
  {
    float radius = NativeInterop.avkSimulationContext_getBodyRadius(10);
    if (radius <= 0.0001f)
      radius = 0.0696f; // Default 696000 km scaled

    ulong sunId = 0;
    if (_simulationContext != IntPtr.Zero)
    {
      // TODO this is in kilogram. If we decide to use reference frames, scale accordingly
      sunId = NativeInterop.avkSimulationContext_spawnProceduralSphere(
        _simulationContext,
        sceneId,
        "sun",
        radius,
        1.989e30f
      );
      NativeInterop.avkSimulationContext_addSunComponent(
        _simulationContext,
        sceneId,
        sunId,
        resX,
        resY,
        resZ
      );
    }

    if (sunId == 0)
      return SpawnEntity(sceneId, "sun", parent); // Fallback if no context

    var sun = new Entity(sceneId, sunId, "sun");
    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    state.EntityMap[sunId] = sun;
    WireEntityComponents(sceneId, sun);

    sun.Components.Add(new TransformComponent());
    sun.Components.Add(new SunComponent());

    if (parent != null)
    {
      NativeInterop.avkSimulationContext_setParent(_simulationContext, sceneId, sunId, parent.Id);
      parent.Children.Add(sun);
    }
    else
    {
      state.RootEntities.Add(sun);
    }

    return sun;
  }

  public Entity CreateMeasurement(ulong sceneId, string name, float[] p1, float[] p2)
  {
    var entity = SpawnEntity(
      sceneId,
      name,
      _sceneStateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault()
    );
    entity.Components.Add(new MeasurementComponent());

    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addMeasurementComponent(
        _simulationContext,
        sceneId,
        entity.Id,
        p1[0],
        p1[1],
        p1[2],
        p2[0],
        p2[1],
        p2[2]
      );
    }

    return entity;
  }

  public Entity SpawnImageBillboard(
    ulong sceneId,
    string name,
    bool isScreenSpace,
    float width,
    float height
  )
  {
    var entity = SpawnEntity(
      sceneId,
      name,
      _sceneStateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault()
    );
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addImageBillboardComponent(
        _simulationContext,
        sceneId,
        entity.Id,
        isScreenSpace,
        width,
        height
      );
    }

    return entity;
  }

  public Entity CreateCamera(ulong sceneId, Entity parent)
  {
    var camera = SpawnEntity(sceneId, "camera", parent);

    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addTransformComponent(
        _simulationContext,
        sceneId,
        camera.Id,
        0f,
        -400.0f,
        0f,
        1f,
        0f,
        0f,
        0f,
        1f,
        1f,
        1f
      );

      NativeInterop.avkSimulationContext_addCameraComponent(
        _simulationContext,
        sceneId,
        camera.Id,
        false,
        45.0f,
        1.77f,
        0.1f,
        10000.0f,
        -10.0f,
        10.0f,
        -10.0f,
        10.0f
      );
    }

    camera.Components.Add(new TransformComponent { PosY = -400.0f });
    camera.Components.Add(new CameraComponent());

    return camera;
  }

  public Entity? GetEntityByName(ulong sceneId, string name)
  {
    return _sceneStateManager
      .GetOrCreateScene(sceneId)
      .EntityMap.Values.FirstOrDefault(e => e.Name == name);
  }

  public Entity? GetEntityById(ulong sceneId, ulong id)
  {
    return _sceneStateManager.GetOrCreateScene(sceneId).EntityMap.TryGetValue(id, out var entity)
      ? entity
      : null;
  }

  public void SetEntityName(ulong sceneId, ulong id, string name)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setEntityName(_simulationContext, sceneId, id, name);
    }
  }

  public void RemoveEntity(ulong sceneId, ulong id)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_removeEntity(_simulationContext, sceneId, id);
    }

    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    if (state.EntityMap.TryGetValue(id, out var entity))
    {
      foreach (var parent in state.EntityMap.Values)
      {
        if (parent.Children.Contains(entity))
        {
          parent.Children.Remove(entity);
          break;
        }
      }

      if (state.RootEntities.Contains(entity))
      {
        state.RootEntities.Remove(entity);
      }

      state.EntityMap.Remove(id);
    }
  }

  public void SyncSceneHierarchy(ulong sceneId)
  {
    if (_simulationContext == IntPtr.Zero)
      return;

    NativeInterop.avkSimulationContext_getSceneHierarchy(
      _simulationContext,
      sceneId,
      null,
      0,
      out uint count
    );
    if (count == 0)
      return;

    using var buffer = _bufferPool.Rent<SceneHierarchyNodeDTO>((int)count);

    if (
      !NativeInterop.avkSimulationContext_getSceneHierarchy(
        _simulationContext,
        sceneId,
        buffer.Array,
        (uint)buffer.Array.Length,
        out count
      )
    )
      return;

    var state = _sceneStateManager.GetOrCreateScene(sceneId);
    var activeNodes = buffer.Array.AsSpan(0, (int)count);

    var rootNodes = new System.Collections.Generic.HashSet<ulong>();
    var childrenMap = new System.Collections.Generic.Dictionary<
      ulong,
      System.Collections.Generic.List<ulong>
    >();
    var foundEntityIds = new System.Collections.Generic.HashSet<ulong>();

    foreach (var node in activeNodes)
    {
      foundEntityIds.Add(node.EntityId);
      if (node.ParentId == 0)
      {
        rootNodes.Add(node.EntityId);
      }
      else
      {
        if (!childrenMap.TryGetValue(node.ParentId, out var list))
        {
          list = new System.Collections.Generic.List<ulong>();
          childrenMap[node.ParentId] = list;
        }
        list.Add(node.EntityId);
      }
    }

    // Sync RootEntities
    for (int i = state.RootEntities.Count - 1; i >= 0; i--)
    {
      if (!rootNodes.Contains(state.RootEntities[i].Id))
        state.RootEntities.RemoveAt(i);
    }
    foreach (var id in rootNodes)
    {
      if (state.EntityMap.TryGetValue(id, out var entity) && !state.RootEntities.Contains(entity))
        state.RootEntities.Add(entity);
    }

    // Sync Children for all entities
    foreach (var entity in state.EntityMap.Values)
    {
      if (!foundEntityIds.Contains(entity.Id))
        continue;

      var hasChildren = childrenMap.TryGetValue(entity.Id, out var children);
      for (int i = entity.Children.Count - 1; i >= 0; i--)
      {
        if (!hasChildren || !children.Contains(entity.Children[i].Id))
          entity.Children.RemoveAt(i);
      }
      if (hasChildren)
      {
        foreach (var childId in children)
          if (
            state.EntityMap.TryGetValue(childId, out var child) && !entity.Children.Contains(child)
          )
            entity.Children.Add(child);
      }
    }
  }
}
