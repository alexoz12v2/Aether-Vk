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

  // Scene mirroring for UI
  public ObservableCollection<Entity> RootEntities { get; } = new();
  private readonly Dictionary<ulong, Entity> _entityMap = new();

  private NativeInterop.LoggerCallback _loggerCallbackDelegate;
  private NativeInterop.BreadcrumbCallback _breadcrumbCallbackDelegate;

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
    _loggerCallbackDelegate = new NativeInterop.LoggerCallback(NativeLogCallback);
    _breadcrumbCallbackDelegate = new NativeInterop.BreadcrumbCallback(NativeBreadcrumbCallback);
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
          NativeInterop.avkSimulationContext_setEntityVisibility(_simulationContext, m.Entity.Id,
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
          NativeInterop.avkSimulationContext_setEntityFollowing(_simulationContext, m.Entity.Id,
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
              NativeInterop.avkSimulationContext_setEntitySelected(_simulationContext, entity.Id,
                false);
            }

            if (m.SelectedEntity != null)
            {
              NativeInterop.avkSimulationContext_setEntitySelected(_simulationContext,
                m.SelectedEntity.Id, true);
            }
          }
        }
      });
  }

  private void NativeBreadcrumbCallback(uint status, IntPtr messagePtr)
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

  private void NativeLogCallback(IntPtr messagePtr)
  {
    if (messagePtr != IntPtr.Zero)
    {
      string? message = System.Runtime.InteropServices.Marshal.PtrToStringAnsi(messagePtr);
      if (message != null)
      {
        var consoleService =
          ServiceLocator.Provider?.GetService(typeof(ConsoleService)) as ConsoleService;
        consoleService?.Log(message);
      }
    }
  }

  public void SetBvhNodeVisibility(ulong entityId, uint nodeIndex, bool isVisible)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setBvhNodeVisibility(_simulationContext, entityId,
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

  public void InitializeSimulationContext(
    string backend = "Vulkan",
    uint width = 800,
    uint height = 600,
    string assetOverride = null
  )
  {
    if (IsInitialized)
      return;

    // Resolve absolute path to the published assets folder
    var exePath = System.AppDomain.CurrentDomain.BaseDirectory;

    // Point Vulkan loader to our embedded MoltenVK and layers if they exist
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

    IsInitialized = true;

    if (ServiceLocator.DispatchToUI != null)
    {
      ServiceLocator.DispatchToUI(() => CreateScene());
    }
    else
    {
      CreateScene();
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
        LoadAlmanacFile(file);
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
          entity.Id,
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
          entity.Id,
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

    NativeInterop.avkSimulationContext_startThreads(_simulationContext);
    IsRunning = true;
  }

  public void StopSimulation()
  {
    if (!IsRunning)
      return;

    NativeInterop.avkSimulationContext_stopThreads(_simulationContext);
    IsRunning = false;
  }

  public void SimulationTick()
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        NativeInterop.avkSimulationContext_simulationTick(_simulationContext);
        SyncEntities();
      }
    }
  }

  public async Task RenderTickAsync()
  {
    if (_simulationContext == IntPtr.Zero) return;

    ulong taskId = NativeInterop.avkSimulationContext_renderTick(_simulationContext);
    if (taskId == 0) return;

    await Task.Run(async () =>
    {
      while (true)
      {
        int status = NativeInterop.avkSimulationContext_getTaskStatus(_simulationContext, taskId);
        if (status == 1) break; // Success
        if (status == 2) throw new Exception("GPU Task Failed");
        if (status == -1) throw new Exception("Invalid Simulation Context");

        await Task.Delay(8); // Poll every ~8ms
      }
    });
  }

  public void ShutdownSimulation()
  {
    if (!IsInitialized)
      return;

    StopSimulation();
    NativeInterop.avkSimulationContext_shutdown(_simulationContext);
    _simulationContext = IntPtr.Zero;
    IsInitialized = false;
    RootEntities.Clear();
    _entityMap.Clear();
  }

  public void SetClearColor(float r, float g, float b, float a)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setClearColor(_simulationContext, r, g, b, a);
    }
  }

  public bool DownloadImage(IntPtr bufferPtr, nuint bufferSize)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        return NativeInterop.avkSimulationContext_downloadImage(
          _simulationContext,
          bufferPtr,
          bufferSize
        );
      }
    }

    return false;
  }

  public void SetActiveCamera(ulong cameraEntityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_setActiveCamera(_simulationContext, cameraEntityId);
    }
  }

  public void RotateCamera(float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand
        {
          cmd_type = 0,
          float_val_1 = deltaX,
          float_val_2 = deltaY,
        }
      );
    }
  }

  public void ZoomCamera(float amount)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand { cmd_type = 1, float_val_1 = amount }
      );
    }
  }

  public void PanCursor(float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand
        {
          cmd_type = 3,
          float_val_1 = deltaX,
          float_val_2 = deltaY,
        }
      );
    }
  }

  public void MoveCursor(float x, float y, float z)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand
        {
          cmd_type = 8,
          float_val_1 = x,
          float_val_2 = y,
          ulong_val = (ulong)BitConverter.ToInt32(BitConverter.GetBytes(z), 0),
        }
      );
    }
  }

  public void PanCamera(float deltaX, float deltaY)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand
        {
          cmd_type = 7,
          float_val_1 = deltaX,
          float_val_2 = deltaY,
        }
      );
    }
  }

  public bool LoadAlmanacFile(string path)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      return NativeInterop.avkSimulationContext_loadAlmanacFile(_simulationContext, path);
    }

    return false;
  }

  public ulong ImportModel(string path)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        return NativeInterop.avkSimulationContext_importModel(_simulationContext, path);
      }
    }

    var breadcrumb =
      ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
    breadcrumb?.ShowMessageAsync("Import Error",
      "Please initialize the 3D Viewport before importing models.", TimeSpan.FromSeconds(5), 3);

    return 0;
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

  public void SpawnModelInstance(ulong modelId, string name, float posX = 0f, float posY = 0f,
    float posZ = 0f)
  {
    lock (_nativeLock)
    {
      if (_simulationContext != IntPtr.Zero)
      {
        ulong instanceId =
          NativeInterop.avkSimulationContext_spawnModelInstance(_simulationContext, modelId, name);
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
      }
    }
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

  public Task<(bool hit, ulong entityId, float px, float py, float pz)> RaycastNdcAsync(
    float ndcX,
    float ndcY
  )
  {
    if (_simulationContext == IntPtr.Zero)
      return Task.FromResult((false, 0UL, 0f, 0f, 0f));

    return Task.Run(() =>
    {
      bool success = NativeInterop.avkSimulationContext_raycastNdc(
        _simulationContext,
        ndcX,
        ndcY,
        out ulong outHitEntity,
        out float outPx,
        out float outPy,
        out float outPz
      );
      return (success, outHitEntity, outPx, outPy, outPz);
    });
  }

  public void ResetCamera()
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand { cmd_type = 2 }
      );
    }
  }

  public void SnapToEntity(ulong entityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand { cmd_type = 4, ulong_val = entityId }
      );
    }
  }

  public void FollowEntity(ulong entityId)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand { cmd_type = 5, ulong_val = entityId }
      );
    }
  }

  public void UnfollowEntity()
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_processCommand(
        _simulationContext,
        new NativeInterop.FfiLogicCommand { cmd_type = 6 }
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

  public void CreateScene()
  {
    RootEntities.Clear();
    _entityMap.Clear();

    if (_simulationContext != IntPtr.Zero)
    {
      // If initialized, the native layer already creates these. We must query them.
      // Right now we don't have an API to query the scene graph, so we simulate the mirror tree exactly as it is spawned natively
      var root = new Entity(1, "root");
      RootEntities.Add(root);
      _entityMap[1] = root;
      WireEntityComponents(root);

      var camera = new Entity(2, "camera");
      WireEntityComponents(camera);
      camera.Components.Add(new TransformComponent { PosY = -400.0f });
      camera.Components.Add(new CameraComponent { IsActiveCamera = true });
      root.Children.Add(camera);
      _entityMap[2] = camera;

      var cursor = new Entity(3, "cursor");
      WireEntityComponents(cursor);
      cursor.Components.Add(new TransformComponent());
      cursor.Components.Add(new CursorComponent());
      root.Children.Add(cursor);
      _entityMap[3] = cursor;

      var sun = new Entity(4, "sun");
      WireEntityComponents(sun);
      sun.Components.Add(new TransformComponent());
      sun.Components.Add(new SunComponent());
      root.Children.Add(sun);
      _entityMap[4] = sun;

      var sunCore = new Entity(5, "sun_core");
      WireEntityComponents(sunCore);
      sunCore.Components.Add(new TransformComponent());
      sunCore.Components.Add(new CometComponent());
      sun.Children.Add(sunCore);
      _entityMap[5] = sunCore;

      var grid = new Entity(6, "grid");
      WireEntityComponents(grid);
      grid.Components.Add(new GridComponent());
      root.Children.Add(grid);
      _entityMap[6] = grid;
    }
    else
    {
      // 1. Create Root
      var root = SpawnEntity("root");
      RootEntities.Add(root);

      // 2. Create Sun
      var sun = SpawnEntity("sun", root);
      sun.Components.Add(new TransformComponent());
      sun.Components.Add(new SunComponent());

      // 3. Create Grid
      var grid = SpawnEntity("grid", root);
      grid.Components.Add(new GridComponent());

      // 4. Create Cursor
      var cursor = SpawnEntity("cursor", root);
      cursor.Components.Add(new TransformComponent());
      cursor.Components.Add(new CursorComponent());

      // 5. Create Camera
      CreateCamera(root);
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
                  entity.Id,
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
                  entity.Id,
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
      NativeInterop.avkSimulationContext_getBvhNodes(_simulationContext, entityId, out uint count);
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

  public Entity SpawnEntity(string name, Entity? parent = null)
  {
    // Native spawn
    ulong nativeId = 0;
    if (_simulationContext != IntPtr.Zero)
    {
      nativeId = NativeInterop.avkSimulationContext_spawnEntity(_simulationContext, name);
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
        NativeInterop.avkSimulationContext_setParent(_simulationContext, nativeId, parent.Id);
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
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addMeasurementComponent(
        _simulationContext,
        entity.Id,
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
        entity.Id,
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
    camera.Components.Add(new TransformComponent { PosY = -400.0f });
    camera.Components.Add(new CameraComponent());

    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_addCameraComponent(
        _simulationContext,
        camera.Id,
        45.0f,
        1.77f,
        0.1f,
        10000.0f
      );
    }

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

  public void RemoveEntity(ulong id)
  {
    if (_simulationContext != IntPtr.Zero)
    {
      NativeInterop.avkSimulationContext_removeEntity(_simulationContext, id);
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
