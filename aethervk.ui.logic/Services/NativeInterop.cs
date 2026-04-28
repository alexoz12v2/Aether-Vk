using System;
using System.Runtime.InteropServices;

namespace AetherVk.Logic.Services;

public static class NativeInterop
{
  private const string DllName = "aethervk_core_cdylib";

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern IntPtr avkSimulationContext_startup(string backend, uint width, uint height);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern IntPtr avkGetAvailableRenderBackends(out uint count);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern uint avkSimulationContext_getEntityCount(IntPtr ctx, ulong sceneId);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern uint avkSimulationContext_getEntityIds(IntPtr ctx, ulong sceneId, IntPtr outIds,
    uint maxCount);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getEntityName(IntPtr ctx, ulong sceneId, ulong entity,
    IntPtr outName, uint maxLen);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_getEntityParent(IntPtr ctx, ulong sceneId, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_createPresentationEngine(IntPtr ctx, uint width, uint height);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_createDefaultScene(IntPtr ctx);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_createEmptyScene(IntPtr ctx);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern IntPtr avkGetAvailableKernels(out uint count);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkFreeStringArray(IntPtr arr, uint count);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_shutdown(IntPtr ctx);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_startThreads(IntPtr ctx);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_stopThreads(IntPtr ctx);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_spawnProceduralSphere(IntPtr ctx, ulong sceneId, string name,
    float radius);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_spawnEntity(IntPtr ctx, ulong sceneId, string name);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setEntityName(IntPtr ctx, ulong sceneId, ulong entity,
    [MarshalAs(UnmanagedType.LPStr)] string name);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_removeEntity(IntPtr ctx, ulong sceneId, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_setParent(IntPtr ctx, ulong sceneId, ulong entity, ulong parent);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setEntityVisibility(IntPtr ctx, ulong sceneId,
    ulong entity,
    [MarshalAs(UnmanagedType.I1)] bool visible);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setEntitySelected(IntPtr ctx, ulong sceneId,
    ulong entity, [MarshalAs(UnmanagedType.I1)] bool selected);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setEntityFollowing(IntPtr ctx, ulong sceneId, ulong entity,
    [MarshalAs(UnmanagedType.I1)] bool following);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setBvhNodeVisibility(IntPtr ctx, ulong sceneId, ulong entity,
    uint nodeIndex, [MarshalAs(UnmanagedType.I1)] bool visible);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_addTransformComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    float posX,
    float posY,
    float posZ,
    float rotW,
    float rotX,
    float rotY,
    float rotZ,
    float scaleX,
    float scaleY,
    float scaleZ
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_setTransformComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    float posX,
    float posY,
    float posZ,
    float rotW,
    float rotX,
    float rotY,
    float rotZ,
    float scaleX,
    float scaleY,
    float scaleZ
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getTransformComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    out float posX,
    out float posY,
    out float posZ,
    out float rotW,
    out float rotX,
    out float rotY,
    out float rotZ,
    out float scaleX,
    out float scaleY,
    out float scaleZ
  );

  [StructLayout(LayoutKind.Sequential)]
  public struct FfiBvhNode
  {
    public uint NodeType;
    public float MinX, MinY, MinZ;
    public float MaxX, MaxY, MaxZ;
    public float CenterX, CenterY, CenterZ;
    public float ExtentsX, ExtentsY, ExtentsZ;
    public uint LeftChild;
    public uint RightChild;
    public uint PrimitiveCount;
  }

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern IntPtr avkSimulationContext_getBvhNodes(IntPtr ctx, ulong sceneId, ulong entity,
    out uint count);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_freeBvhNodes(IntPtr ptr, uint count);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_addPhysicalMeshComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    string gltfPath,
    float emissiveIntensity,
    float emissiveR,
    float emissiveG,
    float emissiveB
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addSkyComponent(IntPtr ctx, ulong sceneId, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addCameraComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    float fov,
    float aspect,
    float near,
    float far
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setCameraComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    [MarshalAs(UnmanagedType.I1)] bool isOrthographic,
    float fov,
    float aspect,
    float near,
    float far
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getCameraComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    IntPtr projOut
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addCursorComponent(IntPtr ctx, ulong sceneId, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addSunComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    uint resX,
    uint resY,
    uint resZ
  );

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_loadAlmanacFile(IntPtr ctx, string path);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_loadCometSpk(IntPtr ctx, string path, uint spkid);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_importModel(IntPtr ctx, string path);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_unloadModel(IntPtr ctx, ulong modelId);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_spawnModelInstance(IntPtr ctx, ulong modelId,
    string name);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern IntPtr avkSimulationContext_getAlmanacLoadedFiles(
    IntPtr ctx,
    out uint count
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setTimeScale(IntPtr ctx, uint scale);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern double avkSimulationContext_getSimulationTime(IntPtr ctx);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getSimulationTimeUtc(
    IntPtr ctx,
    IntPtr buffer,
    uint bufferLen
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setSimulationTime(IntPtr ctx, double timeTai);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getEpochLimits(
    IntPtr ctx,
    out double startTai,
    out double endTai
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_raycast(
    IntPtr ctx,
    ulong sceneId,
    float roX,
    float roY,
    float roZ,
    float rdX,
    float rdY,
    float rdZ
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_raycastNdc(
    IntPtr ctx,
    ulong sceneId,
    float ndcX,
    float ndcY
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setMarkers(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    uint count,
    float[] px,
    float[] py,
    float[] pz,
    float[] cr,
    float[] cg,
    float[] cb,
    float[] sizes
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addGridComponent(IntPtr ctx, ulong sceneId, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addMeasurementComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    float p1X, float p1Y, float p1Z,
    float p2X, float p2Y, float p2Z
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addImageBillboardComponent(
    IntPtr ctx,
    ulong sceneId,
    ulong entity,
    [MarshalAs(UnmanagedType.I1)] bool isScreenSpace,
    float width,
    float height
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_simulationTick(IntPtr ctx, ulong sceneId, double deltaTime);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_renderTick(IntPtr ctx, ulong presentationEngineHandle, ulong sceneId, ulong cameraId, uint width, uint height);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern int avkSimulationContext_getTaskStatus(IntPtr ctx, ulong taskId);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_getTaskResultU64(IntPtr ctx, ulong taskId);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getTaskResultBool(IntPtr ctx, ulong taskId);

  [StructLayout(LayoutKind.Sequential)]
  public struct FfiRaycastResult
  {
    [MarshalAs(UnmanagedType.I1)] public bool Hit;
    public ulong Entity;
    public float Px, Py, Pz;
  }

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getTaskResultRaycast(IntPtr ctx, ulong taskId, out FfiRaycastResult result);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_resize(IntPtr ctx, uint width, uint height);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setActiveCamera(IntPtr ctx, ulong sceneId, ulong camera);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_rotateCamera(IntPtr ctx, ulong sceneId, ulong cameraEntity, float deltaX, float deltaY);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_zoomCamera(IntPtr ctx, ulong sceneId, ulong cameraEntity, float amount);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_resetCamera(IntPtr ctx, ulong sceneId, ulong cameraEntity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_panCamera(IntPtr ctx, ulong sceneId, ulong cameraEntity, float deltaX, float deltaY);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_panCursor(IntPtr ctx, ulong sceneId, float deltaX, float deltaY);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_moveCursor(IntPtr ctx, ulong sceneId, float deltaX, float deltaY, float deltaZ);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_snapToEntity(IntPtr ctx, ulong sceneId, ulong snapEntity, ulong targetEntity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_followEntity(IntPtr ctx, ulong sceneId, ulong snapEntity, ulong targetEntity, [MarshalAs(UnmanagedType.I1)] bool unfollowOther);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_unfollowEntity(IntPtr ctx, ulong sceneId, ulong entityId);

  [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
  public delegate void LoggerCallback(IntPtr message);

  [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
  public delegate void BreadcrumbCallback(uint status, IntPtr message);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setLoggerCallback(LoggerCallback cb);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setBreadcrumbCallback(BreadcrumbCallback cb);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setAssetPath(string path);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_downloadImage(
    IntPtr ctx,
    ulong taskId,
    IntPtr bufferPtr,
    nuint bufferSize
  );
}
