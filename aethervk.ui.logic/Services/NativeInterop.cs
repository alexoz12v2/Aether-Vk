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
  public static extern ulong avkSimulationContext_spawnEntity(IntPtr ctx, string name);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setParent(IntPtr ctx, ulong entity, ulong parent);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setEntityVisibility(IntPtr ctx, ulong entity, [MarshalAs(UnmanagedType.I1)] bool visible);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setEntitySelected(IntPtr ctx, ulong entity, [MarshalAs(UnmanagedType.I1)] bool selected);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setEntityFollowing(IntPtr ctx, ulong entity, [MarshalAs(UnmanagedType.I1)] bool following);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setBvhNodeVisibility(IntPtr ctx, ulong entity, uint nodeIndex, [MarshalAs(UnmanagedType.I1)] bool visible);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addTransformComponent(
    IntPtr ctx,
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
  public static extern void avkSimulationContext_setTransformComponent(
    IntPtr ctx,
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

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_addPhysicalMeshComponent(
    IntPtr ctx,
    ulong entity,
    string gltfPath,
    float emissiveIntensity,
    float emissiveR,
    float emissiveG,
    float emissiveB
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addSkyComponent(IntPtr ctx, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addCameraComponent(
    IntPtr ctx,
    ulong entity,
    float fov,
    float aspect,
    float near,
    float far
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setCameraComponent(
    IntPtr ctx,
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
    ulong entity,
    IntPtr projOut
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addCursorComponent(IntPtr ctx, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_addSunComponent(
    IntPtr ctx,
    ulong entity,
    uint resX,
    uint resY,
    uint resZ
  );

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_loadAlmanacFile(IntPtr ctx, string path);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_importModel(IntPtr ctx, string path);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_unloadModel(IntPtr ctx, ulong modelId);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_spawnModelInstance(IntPtr ctx, ulong modelId, string name);

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
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_raycast(
    IntPtr ctx,
    float roX,
    float roY,
    float roZ,
    float rdX,
    float rdY,
    float rdZ,
    out ulong outHitEntity,
    out float outPx,
    out float outPy,
    out float outPz
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_raycastNdc(
    IntPtr ctx,
    float ndcX,
    float ndcY,
    out ulong outHitEntity,
    out float outPx,
    out float outPy,
    out float outPz
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setMarkers(
    IntPtr ctx,
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
  public static extern void avkSimulationContext_addGridComponent(IntPtr ctx, ulong entity);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_renderTick(IntPtr ctx);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_renderTickSync(IntPtr ctx);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_resize(IntPtr ctx, uint width, uint height);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setActiveCamera(IntPtr ctx, ulong camera);

  [StructLayout(LayoutKind.Sequential)]
  public struct FfiLogicCommand
  {
    public uint cmd_type;
    public float float_val_1;
    public float float_val_2;
    public ulong ulong_val;

    [MarshalAs(UnmanagedType.I1)]
    public bool bool_val;
  }

  [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
  public delegate void LoggerCallback(IntPtr message);

  [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
  public delegate void BreadcrumbCallback(uint status, IntPtr message);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setLoggerCallback(LoggerCallback cb);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setBreadcrumbCallback(BreadcrumbCallback cb);

  [DllImport(DllName, CharSet = CharSet.Ansi, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_setAssetPath(IntPtr ctx, string path);

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_processCommand(
    IntPtr ctx,
    FfiLogicCommand command
  );

  [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_downloadImage(
    IntPtr ctx,
    IntPtr bufferPtr,
    nuint bufferSize
  );
}
