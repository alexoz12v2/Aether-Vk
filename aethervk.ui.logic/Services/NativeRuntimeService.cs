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
  [ObservableProperty] private bool _isInitialized;

  [ObservableProperty] private bool _isRunning;

  private IntPtr _simulationContext = IntPtr.Zero;
  public ulong ActiveSceneId { get; private set; } = 0;
  public ulong ActivePresentationEngineId { get; private set; } = 0;
  private readonly object _nativeLock = new object();

  // Scene mirroring for UI
  public ObservableCollection<Entity> RootEntities { get; } = new();
  private readonly Dictionary<ulong, Entity> _entityMap = new();

  private static readonly NativeInterop.LoggerCallback _loggerCallbackDelegate = new NativeInterop.LoggerCallback(NativeLogCallback);
  private static readonly NativeInterop.BreadcrumbCallback _breadcrumbCallbackDelegate = new NativeInterop.BreadcrumbCallback(NativeBreadcrumbCallback);

  public void Dispose()
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_shutdown(_simulationContext);
        _simulationContext = IntPtr.Zero;
      }
    }
  }

  public NativeRuntimeService()

  {
    try
    {
      NativeInterop.avkSimulationContext_setLoggerCallback(_loggerCallbackDelegate);
      NativeInterop.avkSimulationContext_setBreadcrumbCallback(_breadcrumbCallbackDelegate);
    }
    catch (System.DllNotFoundException)
    {
      // Ignore during tests
    }

    WeakReferenceMessenger.Default.Register<EntityVisibilityChangedMessage>(this, (r, m) =>
    {
      lock (_nativeLock)
      {
        if (_simulationContext != IntPtr.Zero)
        {
          NativeInterop.avkSimulationContext_setEntityVisibility(_simulationContext, ActiveSceneId, m.Entity.Id,
            m.Entity.IsVisible);
        }
      }
    });

    WeakReferenceMessenger.Default.Register<EntityOutlineChangedMessage>(this, (r, m) =>
    {
      lock (_nativeLock)
      {
        if (_simulationContext != IntPtr.Zero)
        {
          NativeInterop.avkSimulationContext_setEntityFollowing(_simulationContext, ActiveSceneId, m.Entity.Id,
            m.Entity.IsOutlined);
        }
      }
    });

    WeakReferenceMessenger.Default.Register<AetherVk.Logic.ViewModels.EntitySelectedMessage>(this,
      (r, m) =>
      {
        lock (_nativeLock)
        {
          if (_simulationContext != IntPtr.Zero)
          {
            // Deselect all
            foreach (var entity in _entityMap.Values)
            {
              NativeInterop.avkSimulationContext_setEntitySelected(_simulationContext, ActiveSceneId, entity.Id,
                false);
            }

            if (m.SelectedEntity != null)
            {
              NativeInterop.avkSimulationContext_setEntitySelected(_simulationContext,
                ActiveSceneId, m.SelectedEntity.Id, true);
            }
          }
        }
      });
  }

  private static void NativeBreadcrumbCallback(uint status, IntPtr messagePtr)
  {
    if (messagePtr != IntPtr.Zero)
    {
      string? message = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr);
      if (message != null)
      {
        var breadcrumb =
          ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
        breadcrumb?.ShowMessageAsync("Simulation Context", message, default, (int)status);
      }
    }
  }

  private static void NativeLogCallback(IntPtr messagePtr)
  {
    if (messagePtr != IntPtr.Zero)
    {
      string? message = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr);
      if (message != null)
      {
        System.Console.Error.WriteLine($"[Native] {message}");
        var consoleService =
          ServiceLocator.Provider?.GetService(typeof(ConsoleService)) as ConsoleService;
        consoleService?.Log(message);
      }
    }
  }

  private async Task PollTaskAsync(ulong taskId)
  {
    if (taskId == 0) return;
    while (_simulationContext != IntPtr.Zero)
    {
      int status = NativeInterop.avkSimulationContext_getTaskStatus(_simulationContext, taskId);
      if (status == 1) return; // Success
      if (status == 2) throw new Exception("Native Task Failed");
      if (status == -1) throw new Exception("Native Context Destroyed");

      await Task.Delay(1);
    }
  }

  public void SetBvhNodeVisibility(ulong entityId, uint nodeIndex, bool isVisible)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setBvhNodeVisibility(_simulationContext, ActiveSceneId, entityId,
        nodeIndex, isVisible);
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

  private static readonly object _staticInitLock = new object();

  public void InitializeSimulationContext(
    string backend = "Vulkan",
    uint width = 800,
    uint height = 600,
    string assetOverride = null,
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
      var icdPath = System.IO.Path.Combine(exePath, "vulkan", "share", "vulkan", "icd.d",
        "MoltenVK_icd.json");
      if (System.IO.File.Exists(icdPath))
      {
        Environment.SetEnvironmentVariable("VK_DRIVER_FILES", icdPath);
        Environment.SetEnvironmentVariable("VK_ICD_FILENAMES", icdPath);
      }

      var layerPath =
        System.IO.Path.Combine(exePath, "vulkan", "share", "vulkan", "explicit_layer.d");
      if (System.IO.Directory.Exists(layerPath))
      {
        Environment.SetEnvironmentVariable("VK_LAYER_PATH", layerPath);
      }

      var assetPath = assetOverride ?? System.IO.Path.Combine(exePath, "assets");
      NativeInterop.avkSimulationContext_setAssetPath(assetPath);

      _simulationContext = NativeInterop.avkSimulationContext_startup(backend, width, height);

      if (_simulationContext == IntPtr.Zero)
      {
        CommunityToolkit.Mvvm.Messaging.WeakReferenceMessenger.Default.Send(
          new AetherVk.Logic.Messages.CriticalErrorMessage(
            $"CRITICAL ERROR:\nThe '{backend}' simulation backend could not be initialized.\n\nThe application cannot run without the core simulation engine."
          )
        );
        return;
      }

      ActivePresentationEngineId = NativeInterop.avkSimulationContext_createPresentationEngine(_simulationContext, width, height);
      IsInitialized = true;

      if (ServiceLocator.DispatchToUI != null)
      {
        ServiceLocator.DispatchToUI(() => CreateScene(populateDefault));
      }
      else
      {
        CreateScene(populateDefault);
      }
    }
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

  private void SyncEntities()
  {
    if (_simulationContext == IntPtr.Zero)
      return;

    foreach (var entity in _entityMap.Values)
    {
      var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
      if (transform != null)
      {
        bool success = NativeInterop.avkSimulationContext_getTransformComponent(
          _simulationContext,
          ActiveSceneId, entity.Id,
          out float px,
          out float py,
          out float pz,
          out float rw,
          out float rx,
          out float ry,
          out float rz,
          out float sx,
          out float sy,
          out float sz
        );

        if (success)
        {
          // Suspend event handling to avoid circular updates
          transform.SuspendNotifications = true;
          transform.PosX = px;
          transform.PosY = py;
          transform.PosZ = pz;
          transform.RotW = rw;
          transform.RotX = rx;
          transform.RotY = ry;
          transform.RotZ = rz;
          transform.ScaleX = sx;
          transform.ScaleY = sy;
          transform.ScaleZ = sz;
          transform.SuspendNotifications = false;
        }
      }

      var camera = entity.Components.OfType<CameraComponent>().FirstOrDefault();
      if (camera != null)
      {
        IntPtr projPtr = Marshal.AllocHGlobal(16 * sizeof(float));
        bool camSuccess = NativeInterop.avkSimulationContext_getCameraComponent(
          _simulationContext,
          ActiveSceneId, entity.Id,
          projPtr
        );
        if (camSuccess)
        {
          float[] projArray = new float[16];
          Marshal.Copy(projPtr, projArray, 0, 16);
          camera.SuspendNotifications = true;
          camera.ProjectionMatrixPreview =
            $"[{projArray[0]:F2}, {projArray[4]:F2}, {projArray[8]:F2}, {projArray[12]:F2}]\n"
            + $"[{projArray[1]:F2}, {projArray[5]:F2}, {projArray[9]:F2}, {projArray[13]:F2}]\n"
            + $"[{projArray[2]:F2}, {projArray[6]:F2}, {projArray[10]:F2}, {projArray[14]:F2}]\n"
            + $"[{projArray[3]:F2}, {projArray[7]:F2}, {projArray[11]:F2}, {projArray[15]:F2}]";
          camera.SuspendNotifications = false;
        }

        Marshal.FreeHGlobal(projPtr);
      }
    }
  }

  public void StartSimulation()
  {
    if (!IsInitialized || IsRunning)
      return;

    IsRunning = true;
  }

  public void StopSimulation()
  {
    if (!IsRunning)
      return;

    IsRunning = false;
  }

  public void SimulationTick()
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_simulationTick(_simulationContext, ActiveSceneId, 0.0);
        SyncEntities();
      }
    }
  }

  public ulong LastRenderTaskId { get; private set; } = 0;

  public async Task RenderTickAsync()
  {
    if (_simulationContext == IntPtr.Zero || ActivePresentationEngineId == 0) return;

    ulong taskId = NativeInterop.avkSimulationContext_renderTick(_simulationContext, ActivePresentationEngineId, ActiveSceneId, 800, 600);
    if (taskId == 0) return;
    LastRenderTaskId = taskId;

    await PollTaskAsync(taskId);
  }

  public void ShutdownSimulation()
  {
    if (!IsInitialized)
      return;

    StopSimulation();
    lock (_nativeLock)
    {
      NativeInterop.avkSimulationContext_shutdown(_simulationContext);
      _simulationContext = IntPtr.Zero;
    }
    IsInitialized = false;
    RootEntities.Clear();
    _entityMap.Clear();
  }

  public async Task<bool> DownloadImageAsync(IntPtr bufferPtr, nuint bufferSize)
  {
    if (_simulationContext == IntPtr.Zero || LastRenderTaskId == 0) return false;

    bool success = NativeInterop.avkSimulationContext_downloadImage(
      _simulationContext,
      LastRenderTaskId,
      bufferPtr,
      bufferSize
    );
    return success;
  }

  public void SetActiveCamera(ulong cameraEntityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setActiveCamera(_simulationContext, ActiveSceneId, cameraEntityId);
    }
  }

  public void RotateCamera(float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_rotateCamera(
        _simulationContext,
        ActiveSceneId,
        GetActiveCameraId(),
        deltaX,
        deltaY
      );
    }
  }

  public void ZoomCamera(float amount)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_zoomCamera(
        _simulationContext,
        ActiveSceneId,
        GetActiveCameraId(),
        amount
      );
    }
  }

  public void PanCursor(float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_panCursor(
        _simulationContext,
        ActiveSceneId,
        deltaX,
        deltaY
      );
    }
  }

  public void MoveCursor(float x, float y, float z)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_moveCursor(
        _simulationContext,
        ActiveSceneId,
        x,
        y,
        z
      );
    }
  }

  public void PanCamera(float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_panCamera(
        _simulationContext,
        ActiveSceneId,
        GetActiveCameraId(),
        deltaX,
        deltaY
      );
    }
  }

  public async Task<bool> LoadAlmanacFileAsync(string path)
  {
    if (_simulationContext == IntPtr.Zero) return false;
    ulong taskId = NativeInterop.avkSimulationContext_loadAlmanacFile(_simulationContext, path);
    await PollTaskAsync(taskId);
    return NativeInterop.avkSimulationContext_getTaskResultBool(_simulationContext, taskId);
  }

  public async Task<bool> LoadCometSpkAsync(string path, uint spkid)
  {
    if (_simulationContext == IntPtr.Zero) return false;
    ulong taskId = NativeInterop.avkSimulationContext_loadCometSpk(_simulationContext, path, spkid);
    await PollTaskAsync(taskId);
    return NativeInterop.avkSimulationContext_getTaskResultBool(_simulationContext, taskId);
  }

  public async Task<ulong> ImportModelAsync(string path)
  {
    if (_simulationContext == IntPtr.Zero) return 0;
    ulong taskId = NativeInterop.avkSimulationContext_importModel(_simulationContext, path);
    await PollTaskAsync(taskId);
    return NativeInterop.avkSimulationContext_getTaskResultU64(_simulationContext, taskId);
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

    // Cleanup UI mirroring
    var toRemove = _entityMap.Values.Where(e => e.Name.StartsWith($"model_{modelId}") || e.Name == "model").ToList();
    foreach (var entity in toRemove)
    {
       RemoveEntity(entity.Id);
    }
  }

  public async Task<ulong> SpawnModelInstanceAsync(ulong modelId, string name, float posX = 0f, float posY = 0f,
    float posZ = 0f)
  {
    if (_simulationContext == IntPtr.Zero) return 0;
    ulong taskId = NativeInterop.avkSimulationContext_spawnModelInstance(_simulationContext, modelId, name);
    await PollTaskAsync(taskId);
    ulong instanceId = NativeInterop.avkSimulationContext_getTaskResultU64(_simulationContext, taskId);
    
    if (instanceId > 0)
    {
        // Add to UI mirroring
        var entity = new Entity(instanceId, name);
        _entityMap[instanceId] = entity;
        WireEntityComponents(entity);
        RootEntities.Add(entity);

        // Fetch basic components that were created
        entity.Components.Add(new TransformComponent { PosX = posX, PosY = posY, PosZ = posZ });
        entity.Components.Add(new CometComponent()); // Assume it spawns as a comet for now
    }
    return instanceId;
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

  public void SetTimeScale(uint scale)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setTimeScale(_simulationContext, scale);
    }
  }

  public double GetSimulationTime()
  {
    if (_simulationContext != IntPtr.Zero)
    {
      return NativeInterop.avkSimulationContext_getSimulationTime(_simulationContext);
    }

    return 0.0;
  }

  public string GetSimulationTimeUtc()
  {
    if (_simulationContext != IntPtr.Zero)
    {
      IntPtr ptr = Marshal.AllocHGlobal(256);
      if (NativeInterop.avkSimulationContext_getSimulationTimeUtc(_simulationContext, ptr, 256))
      {
        var result = Marshal.PtrToStringAnsi(ptr) ?? "";
        Marshal.FreeHGlobal(ptr);
        return result;
      }

      Marshal.FreeHGlobal(ptr);
    }

    return "UNKNOWN";
  }

  public void SetSimulationTime(double timeTai)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setSimulationTime(_simulationContext, timeTai);
    }
  }

  public bool GetEpochLimits(out double startTai, out double endTai)
  {
    startTai = 0.0;
    endTai = 0.0;
    if (_simulationContext != IntPtr.Zero)
    {
      return NativeInterop.avkSimulationContext_getEpochLimits(
        _simulationContext,
        out startTai,
        out endTai
      );
    }

    return false;
  }

  public async Task<(bool hit, ulong entityId, float px, float py, float pz)> RaycastNdcAsync(
    float ndcX,
    float ndcY
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return (false, 0UL, 0f, 0f, 0f);

    ulong taskId = NativeInterop.avkSimulationContext_raycastNdc(_simulationContext, ActiveSceneId, ndcX, ndcY);
    await PollTaskAsync(taskId);
    
    if (NativeInterop.avkSimulationContext_getTaskResultRaycast(_simulationContext, taskId, out var result))
    {
        return (result.Hit, result.Entity, result.Px, result.Py, result.Pz);
    }
    return (false, 0UL, 0f, 0f, 0f);
  }

  public void ResetCamera()
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_resetCamera(
        _simulationContext,
        ActiveSceneId,
        GetActiveCameraId()
      );
    }
  }

  public void SnapToEntity(ulong entityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_snapToEntity(
        _simulationContext,
        ActiveSceneId,
        GetActiveCameraId(), // Use camera as snap entity for now
        entityId
      );
    }
  }

  public void FollowEntity(ulong entityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_followEntity(
        _simulationContext,
        ActiveSceneId,
        GetActiveCameraId(),
        entityId,
        true
      );
    }
  }

  public void UnfollowEntity()
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_unfollowEntity(
        _simulationContext,
        ActiveSceneId,
        GetActiveCameraId()
      );
    }
  }

  public void SyncMarkers(ulong entityId, CometComponent comet)
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
      ActiveSceneId, entityId,
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

  public ulong GetActiveCameraId()
  {
    ulong fallbackId = 1;
    foreach (var entity in _entityMap.Values)
    {
      var cam = entity.Components.OfType<CameraComponent>().FirstOrDefault();
      if (cam != null)
      {
        fallbackId = entity.Id;
        if (cam.IsActiveCamera)
        {
          return entity.Id;
        }
      }
    }

    return fallbackId;
  }

  public void CreateScene(bool populateDefault = true)
  {
    RootEntities.Clear();
    _entityMap.Clear();

    if (_simulationContext != IntPtr.Zero)
    {
      if (populateDefault)
      {
        ActiveSceneId = NativeInterop.avkSimulationContext_createDefaultScene(_simulationContext);
      }
      else
      {
        ActiveSceneId = NativeInterop.avkSimulationContext_createEmptyScene(_simulationContext);
      }

      uint count = NativeInterop.avkSimulationContext_getEntityCount(_simulationContext, ActiveSceneId);
      if (count > 0)
      {
        IntPtr idsPtr = Marshal.AllocHGlobal((int)count * sizeof(long));
        NativeInterop.avkSimulationContext_getEntityIds(_simulationContext, ActiveSceneId, idsPtr, count);

        long[] ids = new long[count];
        Marshal.Copy(idsPtr, ids, 0, (int)count);
        Marshal.FreeHGlobal(idsPtr);

        IntPtr namePtr = Marshal.AllocHGlobal(256);
        foreach (long signedId in ids)
        {
          ulong id = (ulong)signedId;
          string name = "Entity";
          if (NativeInterop.avkSimulationContext_getEntityName(_simulationContext, ActiveSceneId, id, namePtr, 256))
          {
            name = Marshal.PtrToStringAnsi(namePtr) ?? name;
          }

          var entity = new Entity(id, name);
          _entityMap[id] = entity;
          WireEntityComponents(entity);
        }
        Marshal.FreeHGlobal(namePtr);

        // Build hierarchy & default components based on FFI entity type heuristics
        foreach (long signedId in ids)
        {
          ulong id = (ulong)signedId;
          var entity = _entityMap[id];
          ulong parentId = NativeInterop.avkSimulationContext_getEntityParent(_simulationContext, ActiveSceneId, id);
          if (parentId != 0 && _entityMap.TryGetValue(parentId, out var parent))
          {
            parent.Children.Add(entity);
          }
          else
          {
            RootEntities.Add(entity);
          }

          // Fetch basic transform logic
          entity.Components.Add(new TransformComponent());

          // Add UI mirrored components by FFI inspection heuristic
          // TODO: No. Do not use heuristic. Add a function which queries the list of components present, and we decide which to spawn
          if (entity.Name == "camera") entity.Components.Add(new CameraComponent());
          if (entity.Name == "cursor") entity.Components.Add(new CursorComponent());
          if (entity.Name == "sun") entity.Components.Add(new SunComponent());
          // TODO: remove sun core. nucleus of sun is included in sun entity itself
          if (entity.Name == "sun_core") entity.Components.Add(new CometComponent());
          if (entity.Name == "grid") entity.Components.Add(new GridComponent());
          if (entity.Name.Contains("measurement", StringComparison.OrdinalIgnoreCase)) entity.Components.Add(new MeasurementComponent());
        }

        SyncEntities(); // Immediately populate real positions
      }
    }
  }

  private void WireEntityComponents(Entity entity)
  {
    entity.Components.CollectionChanged += (sender, args) =>
    {
      if (args.NewItems != null)
      {
        foreach (var item in args.NewItems)
        {
          if (item is TransformComponent tc)
          {
            tc.PropertyChanged += (s, e) =>
            {
              if (tc.SuspendNotifications)
                return;

              if (_simulationContext != IntPtr.Zero)
              {
                NativeInterop.avkSimulationContext_setTransformComponent(
                  _simulationContext,
                  ActiveSceneId, entity.Id,
                  tc.PosX,
                  tc.PosY,
                  tc.PosZ,
                  tc.RotW,
                  tc.RotX,
                  tc.RotY,
                  tc.RotZ,
                  tc.ScaleX,
                  tc.ScaleY,
                  tc.ScaleZ
                );
              }
            };
          }
          else if (item is CameraComponent cc)
          {
            cc.PropertyChanged += (s, e) =>
            {
              if (cc.SuspendNotifications)
                return;

              if (_simulationContext != IntPtr.Zero)
              {
                NativeInterop.avkSimulationContext_setCameraComponent(
                  _simulationContext,
                  ActiveSceneId, entity.Id,
                  cc.IsOrthographic,
                  cc.Fov,
                  cc.AspectRatio,
                  cc.NearPlane,
                  cc.FarPlane
                );
              }
            };
          }
          else if (item is CometComponent comet)
          {
            comet.Jets.CollectionChanged += (s, e) =>
            {
              SyncMarkers(entity.Id, comet);
              if (e.NewItems != null)
              {
                foreach (JetMarker jet in e.NewItems)
                {
                  jet.PropertyChanged += (js, je) => { SyncMarkers(entity.Id, comet); };
                }
              }
            };
          }
        }
      }
    };
  }

  public void RefreshBvhNodes(ulong entityId, CometComponent comet)
  {
    if (_simulationContext == IntPtr.Zero) return;

    comet.BvhTree.Clear();

    IntPtr ptr =
      NativeInterop.avkSimulationContext_getBvhNodes(_simulationContext, ActiveSceneId, entityId, out uint count);
    if (ptr == IntPtr.Zero || count == 0) return;

    var nodes = new NativeInterop.FfiBvhNode[count];
    for (int i = 0; i < count; i++)
    {
      nodes[i] =
        Marshal.PtrToStructure<NativeInterop.FfiBvhNode>(ptr +
                                                         i * Marshal
                                                           .SizeOf<NativeInterop.FfiBvhNode>());
    }

    NativeInterop.avkSimulationContext_freeBvhNodes(ptr, count);

    BvhNode? BuildNode(uint index)
    {
      if (index >= count) return null;
      var ffiNode = nodes[index];

      var node = new BvhNode
      {
        EntityId = entityId,
        Index = index,
        Name = ffiNode.PrimitiveCount > 0 ? "Leaf Node" : "Inner Node"
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
          if (left != null) node.Children.Add(left);
        }

        if (ffiNode.RightChild != uint.MaxValue)
        {
          var right = BuildNode(ffiNode.RightChild);
          if (right != null) node.Children.Add(right);
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

  public ulong SpawnProceduralSphere(string name, float radius)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        ulong id =
          NativeInterop.avkSimulationContext_spawnProceduralSphere(_simulationContext, ActiveSceneId, name,
            radius);
        if (id > 0)
        {
          var entity = new Entity(id, name);
          _entityMap[id] = entity;
          WireEntityComponents(entity);
          // For now, don't automatically add to RootEntities if it's a test mesh
          // or maybe we should? Tests might want to check RootEntities.
          // Let's add it to a child of root if possible.
          var root = GetEntityByName("root");
          if (root != null)
          {
            root.Children.Add(entity);
          }
          else
          {
            RootEntities.Add(entity);
          }

          return id;
        }
      }
    }

    return 0;
  }

  public Entity SpawnEntity(string name, Entity? parent = null)
  {
    // Native spawn
    ulong nativeId = 0;
    if (_simulationContext != IntPtr.Zero)
    {
      nativeId = NativeInterop.avkSimulationContext_spawnEntity(_simulationContext, ActiveSceneId, name);
    }
    else
    {
      nativeId = (ulong)_entityMap.Count + 1; // Fallback mock ID
    }

    var entity = new Entity(nativeId, name);
    _entityMap[nativeId] = entity;

    WireEntityComponents(entity);

    if (parent != null)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_setParent(_simulationContext, ActiveSceneId, nativeId, parent.Id);
      }

      parent.Children.Add(entity);
    }
    else if (nativeId != 1) // Avoid adding root again if called manually without parent
    {
      RootEntities.Add(entity);
    }

    return entity;
  }

  public Entity CreateMeasurement(string name, float[] p1, float[] p2)
  {
    var entity = SpawnEntity(name, RootEntities.FirstOrDefault());
    entity.Components.Add(new MeasurementComponent());

    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addMeasurementComponent(
        _simulationContext,
        ActiveSceneId, entity.Id,
        p1[0], p1[1], p1[2],
        p2[0], p2[1], p2[2]
      );
    }

    return entity;
  }

  public Entity SpawnImageBillboard(string name, bool isScreenSpace, float width, float height)
  {
    var entity = SpawnEntity(name, RootEntities.FirstOrDefault());
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addImageBillboardComponent(
        _simulationContext,
        ActiveSceneId, entity.Id,
        isScreenSpace,
        width,
        height
      );
    }

    return entity;
  }

  public Entity CreateCamera(Entity parent)
  {
    var camera = SpawnEntity("camera", parent);
    
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addTransformComponent(
        _simulationContext,
        ActiveSceneId, camera.Id,
        0f, -400.0f, 0f,
        1f, 0f, 0f, 0f,
        1f, 1f, 1f
      );
      
      NativeInterop.avkSimulationContext_addCameraComponent(
        _simulationContext,
        ActiveSceneId, camera.Id,
        45.0f,
        1.77f,
        0.1f,
        10000.0f
      );
    }

    camera.Components.Add(new TransformComponent { PosY = -400.0f });
    camera.Components.Add(new CameraComponent());

    return camera;
  }

  public Entity? GetEntityByName(string name)
  {
    return _entityMap.Values.FirstOrDefault(e => e.Name == name);
  }

  public Entity? GetEntityById(ulong id)
  {
    return _entityMap.TryGetValue(id, out var entity) ? entity : null;
  }

  public void SetEntityName(ulong id, string name)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setEntityName(_simulationContext, ActiveSceneId, id, name);
    }
  }

  public void RemoveEntity(ulong id)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_removeEntity(_simulationContext, ActiveSceneId, id);
    }

    if (_entityMap.TryGetValue(id, out var entity))
    {
      foreach (var parent in _entityMap.Values)
      {
        if (parent.Children.Contains(entity))
        {
          parent.Children.Remove(entity);
          break;
        }
      }

      if (RootEntities.Contains(entity))
      {
        RootEntities.Remove(entity);
      }

      _entityMap.Remove(id);
    }
  }
}
