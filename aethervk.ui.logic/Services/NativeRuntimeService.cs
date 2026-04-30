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
  private readonly object _nativeLock = new object();

  private readonly SceneStateManager _sceneStateManager;

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

  public NativeRuntimeService(SceneStateManager sceneStateManager)
  {
    _sceneStateManager = sceneStateManager;
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
          NativeInterop.avkSimulationContext_setEntityVisibility(_simulationContext, m.Entity.SceneId, m.Entity.Id,
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
          NativeInterop.avkSimulationContext_setEntityFollowing(_simulationContext, m.Entity.SceneId, m.Entity.Id,
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
            if (m.SelectedEntity != null) {
              var sceneState = _sceneStateManager.GetOrCreateScene(m.SelectedEntity.SceneId);
              foreach (var entity in sceneState.EntityMap.Values)
              {
                NativeInterop.avkSimulationContext_setEntitySelected(_simulationContext, m.SelectedEntity.SceneId, entity.Id,
                  false);
              }
            }

            if (m.SelectedEntity != null)
            {
              NativeInterop.avkSimulationContext_setEntitySelected(_simulationContext,
                m.SelectedEntity.SceneId, m.SelectedEntity.Id, true);
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

  public void SetBvhNodeVisibility(ulong sceneId, ulong entityId, uint nodeIndex, bool isVisible)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setBvhNodeVisibility(_simulationContext, sceneId, entityId,
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

      if (ServiceLocator.DispatchToUI != null)
      {
        ServiceLocator.DispatchToUI(() => _ = CreateScene(populateDefault));
      }
      else
      {
        _ = CreateScene(populateDefault);
      }
    }
  }

  public ulong CreatePresentationEngine(uint width, uint height)
  {
    if (_simulationContext == IntPtr.Zero) return 0;
    return NativeInterop.avkSimulationContext_createPresentationEngine(_simulationContext, width, height);
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
    foreach (var entity in sceneState.EntityMap.Values)
    {
      var transform = entity.Components.OfType<TransformComponent>().FirstOrDefault();
      if (transform != null)
      {
        bool success = NativeInterop.avkSimulationContext_getTransformComponent(
          _simulationContext,
          sceneId, entity.Id,
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
          sceneId, entity.Id,
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

  public void SimulationTick(ulong sceneId)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_simulationTick(_simulationContext, sceneId, 0.0);
        SyncEntities(sceneId);
      }
    }
}

  public async Task<ulong> RenderTickAsync(ulong presentationEngineId, ulong sceneId, ulong cameraId, uint width, uint height)
  {
    if (_simulationContext == IntPtr.Zero || presentationEngineId == 0) return 0;

    ulong taskId = NativeInterop.avkSimulationContext_renderTick(_simulationContext, presentationEngineId, sceneId, cameraId, width, height);
    if (taskId == 0) return 0;

    await PollTaskAsync(taskId);
    return taskId;
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
  }

  public async Task<bool> DownloadImageAsync(ulong taskId, IntPtr bufferPtr, nuint bufferSize)
  {
    if (_simulationContext == IntPtr.Zero || taskId == 0) return false;

    bool success = NativeInterop.avkSimulationContext_downloadImage(
      _simulationContext,
      taskId,
      bufferPtr,
      bufferSize
    );
    return success;
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
      _ = NativeInterop.avkSimulationContext_panCursor(
        _simulationContext,
        sceneId,
        deltaX,
        deltaY
      );
    }
  }

  public void MoveCursor(ulong sceneId, float x, float y, float z)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      _ = NativeInterop.avkSimulationContext_moveCursor(
        _simulationContext,
        sceneId,
        x,
        y,
        z
      );
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

    // Cleanup UI mirroring across all scenes
    foreach (var state in _sceneStateManager.AllScenes)
    {
      var toRemove = state.EntityMap.Values.Where(e => e.Name.StartsWith($"model_{modelId}") || e.Name == "model").ToList();
      foreach (var entity in toRemove)
      {
         RemoveEntity(state.SceneId, entity.Id);
      }
    }
  }

  public async Task<ulong> SpawnModelInstanceAsync(ulong sceneId, ulong modelId, string name, float posX = 0f, float posY = 0f,
    float posZ = 0f)
  {
    if (_simulationContext == IntPtr.Zero) return 0;
    ulong taskId = NativeInterop.avkSimulationContext_spawnModelInstance(_simulationContext, modelId, name);
    await PollTaskAsync(taskId);
    ulong instanceId = NativeInterop.avkSimulationContext_getTaskResultU64(_simulationContext, taskId);
    
    if (instanceId > 0)
    {
        // Add to UI mirroring
        var entity = new Entity(sceneId, instanceId, name);
        var state = _sceneStateManager.GetOrCreateScene(sceneId);
        state.EntityMap[instanceId] = entity;
        WireEntityComponents(sceneId, entity);
        state.RootEntities.Add(entity);

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
    ulong sceneId,
    float ndcX,
    float ndcY
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return (false, 0UL, 0f, 0f, 0f);

    ulong taskId = NativeInterop.avkSimulationContext_raycastNdc(_simulationContext, sceneId, ndcX, ndcY);
    await PollTaskAsync(taskId);
    
    if (NativeInterop.avkSimulationContext_getTaskResultRaycast(_simulationContext, taskId, out var result))
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
      sceneId, entityId,
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
      var state = _sceneStateManager.GetOrCreateScene(sceneId);
      state.Clear();

      uint count = NativeInterop.avkSimulationContext_getEntityCount(_simulationContext, sceneId);
      if (count > 0)
      {
        IntPtr idsPtr = Marshal.AllocHGlobal((int)count * sizeof(long));
        NativeInterop.avkSimulationContext_getEntityIds(_simulationContext, sceneId, idsPtr, count);

        long[] ids = new long[count];
        Marshal.Copy(idsPtr, ids, 0, (int)count);
        Marshal.FreeHGlobal(idsPtr);

        IntPtr namePtr = Marshal.AllocHGlobal(256);
        foreach (long signedId in ids)
        {
          ulong id = (ulong)signedId;
          string name = "Entity";
          if (NativeInterop.avkSimulationContext_getEntityName(_simulationContext, sceneId, id, namePtr, 256))
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
          ulong parentId = NativeInterop.avkSimulationContext_getEntityParent(_simulationContext, sceneId, id);
          if (parentId != 0 && state.EntityMap.TryGetValue(parentId, out var parent))
          {
            parent.Children.Add(entity);
          }
          else
          {
            state.RootEntities.Add(entity);
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

        SyncEntities(sceneId); // Immediately populate real positions
      }
      return sceneId;
    }
    return 0;
  }

  private void WireEntityComponents(ulong sceneId, Entity entity)
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
                  sceneId, entity.Id,
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
                  sceneId, entity.Id,
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
              SyncMarkers(sceneId, entity.Id, comet);
              if (e.NewItems != null)
              {
                foreach (JetMarker jet in e.NewItems)
                {
                  jet.PropertyChanged += (js, je) => { SyncMarkers(sceneId, entity.Id, comet); };
                }
              }
            };
          }
        }
      }
    };
  }

  public void RefreshBvhNodes(ulong sceneId, ulong entityId, CometComponent comet)
  {
    if (_simulationContext == IntPtr.Zero) return;

    comet.BvhTree.Clear();

    IntPtr ptr =
      NativeInterop.avkSimulationContext_getBvhNodes(_simulationContext, sceneId, entityId, out uint count);
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
        SceneId = sceneId,
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

  public ulong SpawnProceduralSphere(ulong sceneId, string name, float radius)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        ulong id =
          NativeInterop.avkSimulationContext_spawnProceduralSphere(_simulationContext, sceneId, name,
            radius);
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
          if (root != null)
          {
            root.Children.Add(entity);
          }
          else
          {
            state.RootEntities.Add(entity);
          }

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
        NativeInterop.avkSimulationContext_setParent(_simulationContext, sceneId, nativeId, parent.Id);
      }

      parent.Children.Add(entity);
    }
    else if (nativeId != 1) // Avoid adding root again if called manually without parent
    {
      state.RootEntities.Add(entity);
    }

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

  public Entity CreateSun(ulong sceneId, Entity parent, uint resX = 128, uint resY = 128, uint resZ = 128)
  {
    var sun = SpawnEntity(sceneId, "sun", parent);
    sun.Components.Add(new SunComponent());
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addSunComponent(_simulationContext, sceneId, sun.Id, resX, resY, resZ);
    }
    return sun;
  }

  public Entity CreateMeasurement(ulong sceneId, string name, float[] p1, float[] p2)
  {
    var entity = SpawnEntity(sceneId, name, _sceneStateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault());
    entity.Components.Add(new MeasurementComponent());

    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addMeasurementComponent(
        _simulationContext,
        sceneId, entity.Id,
        p1[0], p1[1], p1[2],
        p2[0], p2[1], p2[2]
      );
    }

    return entity;
  }

  public Entity SpawnImageBillboard(ulong sceneId, string name, bool isScreenSpace, float width, float height)
  {
    var entity = SpawnEntity(sceneId, name, _sceneStateManager.GetOrCreateScene(sceneId).RootEntities.FirstOrDefault());
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addImageBillboardComponent(
        _simulationContext,
        sceneId, entity.Id,
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
        sceneId, camera.Id,
        0f, -400.0f, 0f,
        1f, 0f, 0f, 0f,
        1f, 1f, 1f
      );
      
      NativeInterop.avkSimulationContext_addCameraComponent(
        _simulationContext,
        sceneId, camera.Id,
        45.0f,
        1.77f,
        0.1f,
        10000.0f
      );
    }

    camera.Components.Add(new TransformComponent { PosY = -400.0f });
    camera.Components.Add(new CameraComponent { IsActiveCamera = true });

    return camera;
  }

  public Entity? GetEntityByName(ulong sceneId, string name)
  {
    return _sceneStateManager.GetOrCreateScene(sceneId).EntityMap.Values.FirstOrDefault(e => e.Name == name);
  }

  public Entity? GetEntityById(ulong sceneId, ulong id)
  {
    return _sceneStateManager.GetOrCreateScene(sceneId).EntityMap.TryGetValue(id, out var entity) ? entity : null;
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
}
