using System;
using System.Collections.Immutable;
using System.IO;
using System.Numerics;
using System.Reactive.Disposables;
using System.Reactive.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Utils;

namespace AetherVk.Logic.Services;

#region Interface

// severity: int, utf8Message: byte*
using unsafe BreadcrumbCallbackDelegate = delegate* unmanaged[Cdecl]<uint, byte*, void>;
//stateId: int, structPtr: nint
using unsafe ExternalStateSimulationCallbackDelegate = delegate* unmanaged[Cdecl]<uint, nint, void>;
// out_handle: CNativeWindowHandle*, signal_done: *AtomicBool (byte) — C# fills handle on UI thread then writes 1
using unsafe GetNativeWindowHandleCallbackDelegate = delegate* unmanaged[Cdecl]<
  CNativeWindowHandle*,
  nint,
  void>;
// utf8Buffer: byte*
using unsafe LoggerCallbackDelegate = delegate* unmanaged[Cdecl]<byte*, void>;
// void*, int, void* (don't care about types, we pass them to `executeMainThreadCleanup`)
using unsafe MainThreadDispatchCallbackDelegate = delegate* unmanaged[Cdecl]<nint, int, nint, void>;
using unsafe PanicCallbackDelegate = delegate* unmanaged[Cdecl]<IntPtr, nuint, void>;
// scene id, pe_handle, timeline completion value
using unsafe RenderCallbackDelegate = delegate* unmanaged[Cdecl]<ulong, ulong, ulong, void>;
// scene_id, ext_id, comp_id, data.as_slice().as_ptr().cast()
using unsafe SimulationCallbackDelegate = delegate* unmanaged[Cdecl]<
  ulong,
  ulong,
  ulong,
  nint,
  void>;

/// <summary>
/// Safe, exclusive C# interface for interacting with the Aether-Vk Native Runtime.
/// Implementations of this interface manage the underlying native context pointer.
///
/// Note: `sceneId` has been removed from all interface methods as we are managing a single scene
/// and it's the class' responsability to track that and communicate it to the FFI layer
/// </summary>
public interface INativeRuntimeService : IDisposable
{
  // Lifecycle is managed entirely by the DI container:
  //   - Construction (= Startup)  → NativeRuntimeService constructor calls avkSimulationContext_startup
  //   - Teardown (= ShutdownSync) → IDisposable.Dispose calls avkSimulationContext_shutdownSync
  // Neither method appears on the interface; SplashViewModel uses a factory Func<INativeRuntimeService>
  // to trigger lazy DI construction at the right point in the startup flow.

  // ==========================================
  // Viewport & Rendering
  // ==========================================
  /// <summary>
  /// Creates a windowed or windowless viewport.
  /// <para>
  /// For windowed mode (<paramref name="handleType"/> &gt; 0):
  /// <paramref name="nativeHandleProvider"/> is invoked on the UI thread (wrapped in a
  /// <c>CocoaAutoreleasePool</c> on macOS) to obtain the platform-specific window handle.
  /// Must NOT be called from the UI thread — the callback dispatches work there.
  /// </para>
  /// </summary>
  /// <param name="nativeHandleProvider">
  /// UI-thread provider. Returns the OS window handle struct.
  /// Use <see cref="NativeWindowHandleProvider"/> factories.
  /// Pass <c>null</c> when <paramref name="handleType"/> is 0 (windowless).
  /// </param>
  /// <param name="handleType">
  /// Matches Rust <c>NativeHandleType</c>: 0 = windowless, 1 = Win32, 3 = Xlib, 5 = Metal.
  /// </param>
  bool AddViewport(
    uint width,
    uint height,
    string name,
    Func<CNativeWindowHandle>? nativeHandleProvider,
    uint handleType,
    out ulong presentationEngineId,
    out ulong cameraEntityId
  );

  void RemoveViewport(ulong presentationEngineId);
  void ResizeViewport(ulong presentationEngineId, uint width, uint height);

  // Swapping for swapchain driven rendering, not needed
  // /// <summary>
  // /// Safely polls getTaskStatus without blocking, then copies the frame to the buffer.
  // /// </summary>
  // Task<bool> DownloadImageAsync(ulong taskId, IntPtr bufferPtr, nuint bufferSize);

  // ==========================================
  // Simulation Flow Control
  // ==========================================

  /// <summary>
  /// Fires with the scene ID when a simulation scene becomes active after <see cref="StartSimulation"/>.
  /// Observed on the native callback thread — use <c>ObserveOn(schedulerProvider.MainThread)</c>
  /// before subscribing on the UI thread.
  /// </summary>
  IObservable<ulong> SimulationStateUpdated { get; }

  bool ResetSimulationSync();
  bool PauseSimulationSync();
  bool StartSimulation(int simSpeed);

  // ==========================================
  // ECS Components & Camera
  // ==========================================
  // Note: `inDto` and `outComputedDto` are passed as IntPtr to allow unmanaged struct blasting
  // TODO: Swap this for specific versions, namely, particle system
  // bool ModifyComponent(ulong entityId, uint command, nint inDto, nint outComputedDto);

#if DEBUG
  void DebugECSPrint(uint entityCount, ulong[] entityIds, uint compCount, ulong[] comps);
#endif

  bool AddCameraAnimation(ulong cameraId, AnimationTarget animation);

  /// <summary>
  /// Directly writes the camera's world-space position and orientation.
  /// Returns <c>false</c> (rejected) if a <c>TransformAnimationComponent</c> is still active
  /// on the camera entity — the caller should treat <c>false</c> as a silent no-op.
  /// Result is confirmed asynchronously via <c>SIMULATION_CALLBACK</c>.
  /// </summary>
  bool CameraSetRotoTranslate(
    ulong cameraId,
    System.Numerics.Vector3 position,
    System.Numerics.Quaternion rotation
  );

  /// <summary>Sets the camera projection to perspective. Not blocked by an active animation.</summary>
  bool CameraSetPerspective(ulong cameraId, float fov, float aspectRatio, float near, float far);

  /// <summary>Sets the camera projection to orthographic. Not blocked by an active animation.</summary>
  bool CameraSetOrthographic(
    ulong cameraId,
    float left,
    float right,
    float bottom,
    float top,
    float near,
    float far
  );

  // ==========================================
  // Particle Systems
  // ==========================================
  // TODO: Should we separate model modification, which affects all particle systems, or not?
  //       we are now reflecting Native side, in which model properties are stored separately, and
  //       we are actively synchronizing them
  bool AddParticleSystem(ParticleSystemModel psModel, ParticleSystemJet psJet, out ulong outPsId);
  ParticleSystemComputedProperties? AddFirstParticleSystem(
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ulong outPsId
  );
  bool ModifyParticleSystem(
    ulong psId,
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ParticleSystemComputedProperties outPsComputedProps
  );
  bool RemoveParticleSystem(ulong psId);

  // ==========================================
  // Orbital Mechanics & Almanacs
  // ==========================================

  /// <summary>
  /// Initializes the native comet entity (Phase 1 of two-phase commit).
  /// Sends <c>TryInitComet</c> to the Rust logic thread, which attaches
  /// <c>AlmanacPlanet</c>, force-repositions, and queues trajectory generation
  /// using the supplied Keplerian elements for the orbital track.
  /// </summary>
  /// <param name="spkId">NAIF SPK id of the comet.</param>
  /// <param name="proposedRange">The epoch window selected by the user.</param>
  /// <param name="sbData">SBDB Keplerian elements used to draw the analytical orbit track.</param>
  /// <param name="cometBodyId">Receives the comet body entity id on success.</param>
  bool TryInitComet(int spkId, TimeRange proposedRange, Models.SmallBodyDataComponent sbData, out ulong cometBodyId);

  /// <summary>
  /// Reconfigures the native comet entity (query or DETACH).
  /// </summary>
  /// <param name="commandFlags">
  ///   Bitmask: <c>0</c> = query id only; <c>2</c> = DETACH (CleanupComet).
  /// </param>
  /// <param name="spkId">NAIF SPK id of the comet.</param>
  /// <param name="cometBodyId">Receives the comet body entity id (always populated on success).</param>
  bool ReconfigureComet(int commandFlags, int spkId, out ulong cometBodyId);

  /// <summary>
  /// Synchronously updates the <c>BodyRotationalModel</c> ECS component on the comet body entity.
  /// Takes effect within one logic frame (~16 ms). No-op if the entity does not exist.
  /// </summary>
  bool SetBodyRotationalModel(ulong cometBodyEntityId, BodyRotationalModelDto dto);

  // Async: completion is signalled by a one-shot (transient) handler registered via the
  // `ExternalStateDispatcher` utility inside the implementation. Permanent subscriptions
  // (companion services) use `RegisterExternalStateListener` instead.
  Task<ulong> LoadAlmanacFileAsync(string path);
  bool UnloadAlmanacFile(string path);

  // ==========================================
  // 3D Models & Assets
  // ==========================================
  // handles conversion to UTF-16 or UTF-8 as done in dll preloading
  // void SetAssetPath(string path); // done at startup -> removed from interface
  // external state management with transient
  Task<ulong> ImportModelAsync(string path);
  void UnloadModel(ulong modelId);

  // ==========================================
  // Screen Space Billboards (UI Overlays)
  // ==========================================
  ulong AddScreenSpaceBillboard(string imagePath, ScreenSpaceBillboard billboard);
  bool SetScreenSpaceBillboard(ulong entityId, ScreenSpaceBillboard billboard);
  bool RemoveScreenSpaceBillboard(ulong entityId);
  bool GetScreenSpaceBillboard(ulong entityId, out ScreenSpaceBillboard outData);

  // ==========================================
  // Callbacks & Dispatch
  // ==========================================
  // TODO: panic/logger/breadcrumb/render callbacks will move to constructor params once the
  // Rust constructor is updated. The panic callback is kept here temporarily.
  // -> Made private as callback given at constructor
  // void RegisterPanicCallback(Action<nint, nuint> callback);

  /// <summary>
  /// Register a listener that will be invoked (on the native callback thread) whenever
  /// <c>SIMULATION_CALLBACK</c> fires for the given entity and component.
  /// </summary>
  /// <param name="entityId">ECS external entity ID to filter on.</param>
  /// <param name="componentForeignId">
  ///   Component discriminator (see <see cref="ComponentForeignId"/>).
  ///   Pass <c>0</c> to receive all component updates for the entity.
  /// </param>
  /// <param name="handler">
  ///   Invoked with a raw data pointer valid only for the duration of the call.
  ///   Must not block. Must not throw.
  /// </param>
  /// <returns>Dispose to deregister.</returns>
  IDisposable RegisterSimulationListener(
    ulong entityId,
    ulong componentForeignId,
    Action<nint> handler
  );

  /// <summary>
  /// Register a listener for <c>EXTERNAL_STATE_SIMULATION_CALLBACK</c> filtered by state type.
  /// </summary>
  /// <param name="stateType">External state discriminator to filter on.</param>
  /// <param name="handler">
  ///   Invoked with a raw data pointer valid only for the duration of the call.
  ///   Must not block. Must not throw.
  /// </param>
  /// <returns>Dispose to deregister.</returns>
  IDisposable RegisterExternalStateListener(ExternalStateType stateType, Action<nint> handler);

  // ==========================================
  // Cached State (populated after successful runtime calls)
  // ==========================================

  /// <summary>Camera entity ID — set when <see cref="AddViewport"/> succeeds.</summary>
  ulong? CameraEntityId { get; }

  /// <summary>Windowed presentation engine ID — set when <see cref="AddViewport"/> succeeds.
  /// Pass to <see cref="StartScopedRenderDocCapture"/> to scope frame captures.</summary>
  ulong? PresentationEngineId { get; }

  /// <summary>
  /// Earth body entity ID — populated from <c>CStartupReturn.EarthPlanetEntity</c> at startup.
  /// Used by <see cref="CameraService"/> to register a position-tracking listener.
  /// </summary>
  ulong? EarthEntityId { get; }

  /// <summary>
  /// Comet entity ID — set when <c>avkSimulationContext_reconfigureComet</c> succeeds.
  /// <c>null</c> until the first comet spawn.
  /// TODO (Rust): pending <c>out_comet_entity_id</c> out-param on <c>avkSimulationContext_reconfigureComet</c>.
  /// </summary>
  ulong? CometEntityId { get; }

  /// <summary>
  /// Cancelled when <see cref="IDisposable.Dispose"/> is called on this service.
  /// Companion services (e.g. <see cref="CometConfigService"/>) link their async operations
  /// to this token so they terminate immediately on shutdown rather than waiting for
  /// the internal 30-second wall-clock timeout.
  /// </summary>
  CancellationToken ShutdownToken { get; }

  // ==========================================
  // Timeline (async — fires ExternalState::TimeRange callback on success)
  // ==========================================

  /// <summary>
  /// Submit a new epoch range to the logic thread. Returns <c>true</c> if the command was
  /// enqueued. The observable in <see cref="TimelineService"/> is only updated when the
  /// <c>ExternalState::TimeRange</c> callback fires.
  /// </summary>
  bool SetEpochRange(short startCenturies, ulong startNs, short endCenturies, ulong endNs);

  /// <summary>Synchronous almanac coverage check against loaded SPK data.</summary>
  bool CheckAlmanacCoverage(
    int spkId,
    short startCenturies,
    ulong startNs,
    short endCenturies,
    ulong endNs
  );

#if DEBUG
  // ==========================================
  // Frame Debugging (RenderDoc)
  // ==========================================

  /// <summary>
  /// Returns <c>true</c> if the process was launched under RenderDoc and the
  /// in-app capture API is available.
  /// </summary>
  bool IsRenderDocAvailable();

  /// <summary>
  /// Requests RenderDoc to capture the next rendered frame and write it to a
  /// <c>.rdc</c> file in the current working directory.
  /// No-op (and safe to call) if <see cref="IsRenderDocAvailable"/> is <c>false</c>.
  /// </summary>
  void TriggerRenderDocCapture();

  /// <summary>
  /// Queues a scoped RenderDoc capture bracketing the very next rendered frame
  /// of the specified windowed presentation engine.  Only that swapchain is
  /// captured — Avalonia's own rendering queues are not included.
  /// </summary>
  /// <param name="presentationEngineId">
  ///   The PE id returned by <see cref="AddViewport"/>.
  /// </param>
  /// <returns><c>true</c> if the command was successfully enqueued.</returns>
  bool StartScopedRenderDocCapture(ulong presentationEngineId);

  bool GetDebugTelemetryStats(out DebugTelemetryStats stats);
#endif
}

#endregion

#region pinvoke_interop

/// <summary>
/// PInvoke Interop for the AetherVk Native Runtime.
///
/// - This needs to resolve to the correct file both when running from a development build, where
///   everything is on the same file, *and* when we are running packaged (MacOS Bundle, MSIX
///   Packaged on windows, and flatpak installed on linux)
///
/// Its interface can be glipsed with the following commands
/// - Linux
///   <pre>
///   nm -D --defined-only --no-demangle target/x86_64-unknown-linux-gnu/debug/libaethervk_core_cdylib.so | grep " T " | awk '{print $3}'
///   </pre>
/// </summary>
internal unsafe static class PInvokeAetherVkCore
{
  private const string LibName = "aethervk_core_cdylib";

  /// <summary>The first time this class is loaded, resolve the dynamic library reference</summary>
  static PInvokeAetherVkCore()
  {
#if NET
    NativeLibrary.SetDllImportResolver(typeof(PInvokeAetherVkCore).Assembly, ResolveNativeLibrary);
#else
    PreloadNativeLibrary();
#endif
  }

#if NET
  // Modern .NET (.NET 5+) Implementation
  private static IntPtr ResolveNativeLibrary(
    string libraryName,
    System.Reflection.Assembly assembly,
    DllImportResearchPath? searchPath
  )
  {
    if (libraryName != LibName)
      return IntPtr.Zero;

    string basePath = AppContext.BaseDirectory;

    // -- macOS App Bundle Logic
    if (OperatingSystem.IsMacOS())
    {
      string fileName = "libaethervk_core_cdylib.dylib";
      // 1. Check MacOS Foldder (same folder as the executable)
      string macOsPath = Path.Combine(basePath, fileName);
      if (NativeLibrary.TryLoad(macOsPath, out IntPtr handle1))
        return handle1;

      // 2. Check the Frameworks folder (Apple's standard location for shared libraries)
      // base path is "Contents/MacOS" so we go up one levels and into Frameworks
      string frameworksPath = Path.GetFullPath(
        Path.Combine(basePath, "..", "Frameworks", fileName)
      );
      if (NativeLibrary.TryLoad(frameworksPath, out IntPtr handle2))
        return handle2;

      // 3. Fallback for macOS dynamic linker: @rpath, @executable_path, ...
      return IntPtr.Zero;
    }
    // -- Linux logic
    else if (OperatingSystem.IsLinux())
    {
      string fullPath = Path.Combine(basePath, "libaethervk_core_cdylib.so");
      if (NativeLibrary.TryLoad(fullPath, out IntPtr handle))
        return handle;
    }
    // -- Windows logic
    else if (OperatingSystem.IsWindows())
    {
      string fullPath = Path.Combine(basePath, "aethervk_core_cdylib.dll");
      if (NativeLibrary.TryLoad(fullPath, out IntPtr handle))
        return handle;
    }

    return IntPtr.Zero;
  }
#else
  // .NET Standard 2.0 Fallback (Preload + Zero-Marshalling)
  private static void PreloadNativeLibrary()
  {
    string basePath = AppContext.BaseDirectory;
    if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
    {
      string fileName = "libaethervk_core_cdylib.dylib";
      // 1. Check MacOS folder
      if (TryLoadMac(Path.Combine(basePath, fileName)))
        return;
      if (TryLoadMac(Path.Combine(basePath, basePath, "..", "Frameworks", fileName)))
        return;
    }
    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
    {
      TryLoadLinux(Path.Combine(basePath, "libaethervk_core_cdylib.so"));
    }
    else if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
    {
      TryLoadWindows(Path.Combine(basePath, "aethervk_core_cdylib.dll"));
    }
  }

  // -- OS Specific Utility functions
  private static bool TryLoadWindows(string path)
  {
    if (!File.Exists(path))
      return false;

    // C# strings are UTF-16 and null terminated by the CLR (from microsoft docs)
    // Pinning it gives us a LPCWSTR compatible with Win32 functions
    fixed (char* pPath = path)
      return LoadLibraryW(pPath) != IntPtr.Zero;
  }

  private static bool TryLoadMac(string path)
  {
    if (!File.Exists(path))
      return false;

    // 2 -> RTLD_NOW, 8 -> RTLD_GLOBAL
    return TryLoadUnixDynamic(path, 2 | 8, isMac: true);
  }

  private static bool TryLoadLinux(string path)
  {
    if (!File.Exists(path))
      return false;

    // 2 -> RTLD_NOW, 256 -> RTLD_GLOBAL
    return TryLoadUnixDynamic(path, 2 | 256, isMac: false);
  }

  private static bool TryLoadUnixDynamic(string path, int flags, bool isMac)
  {
    // 1. Calculate UTF-8 length (System.String already has a null terminator)
    int byteCount = Encoding.UTF8.GetByteCount(path);

    // 2. Allocate on the stack (Linux paths max out at 4096 bytes, so should not stack overflow)
    //  add 1 for a null terminator (byte count doesn't account for it?)
    byte* utf8Buffer = stackalloc byte[byteCount + 1];

    // 3. Pin the managed string and transcode directly to stack memory
    fixed (char* pPath = path)
      Encoding.UTF8.GetBytes(pPath, path.Length, utf8Buffer, byteCount);

    utf8Buffer[byteCount] = 0; // null termination

    // 4. invoke appropriate dynamic linker
    if (isMac)
      return dlopen_mac(utf8Buffer, flags) != IntPtr.Zero;
    else
    {
      try
      {
        if (dlopen_linux(utf8Buffer, flags) != IntPtr.Zero)
          return true;
      }
      catch (DllNotFoundException) { }
      try
      {
        if (dlopen_linux_glibc(utf8Buffer, flags) != IntPtr.Zero)
          return true;
      }
      catch (DllNotFoundException) { }
      try
      {
        if (dlopen_linux_musl(utf8Buffer, flags) != IntPtr.Zero)
          return true;
      }
      catch (DllNotFoundException) { }
      return false;
    }
  }

  // -- OS Dynamic Linker P/Invokes (Blittable only, zero-marshalling safe)
  [DllImport("kernel32.dll", ExactSpelling = true, SetLastError = true)]
  private static extern IntPtr LoadLibraryW(char* lpFileName);

  [DllImport("libdl.dylib", EntryPoint = "dlopen", ExactSpelling = true)]
  private static extern IntPtr dlopen_mac(byte* filename, int flags);

  [DllImport("libdl.so.2", EntryPoint = "dlopen", ExactSpelling = true)]
  private static extern IntPtr dlopen_linux(byte* filename, int flags);

  [DllImport("libc.so.6", EntryPoint = "dlopen", ExactSpelling = true)]
  private static extern IntPtr dlopen_linux_glibc(byte* filename, int flags);

  [DllImport("libc.so", EntryPoint = "dlopen", ExactSpelling = true)]
  private static extern IntPtr dlopen_linux_musl(byte* filename, int flags);
#endif

  // =========================================================================
  // Target Library Export
  // =========================================================================

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkProbeSpkFile(
    byte* path,
    int spkId,
    CTimeRange* inTaiParts,
    CTimeRange* outDomainTaiParts,
    int* outDiscoveredNaifId
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkRegisterPanicCallback(PanicCallbackDelegate cb);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetAssetPath(byte* path);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetBreadcrumbCallback(BreadcrumbCallbackDelegate cb);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetExternalStateSimulationCallback(
    ExternalStateSimulationCallbackDelegate cb
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetLoggerCallback(LoggerCallbackDelegate cb);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetMainThreadDispatchCallback(MainThreadDispatchCallbackDelegate cb);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetGetNativeWindowHandleCallback(
    GetNativeWindowHandleCallbackDelegate cb
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetRenderCallback(RenderCallbackDelegate cb);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSetSimulationCallback(SimulationCallbackDelegate cb);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_addCameraAnimation(
    nint ctx,
    ulong sceneId,
    ulong cameraId,
    AnimationTargetDTO* animation
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_addParticleSystem(
    nint ctx,
    ulong sceneId,
    ParticleSystemDTO* particleSystem,
    ulong* outPsId,
    ParticleSystemComputedDTO* outComputedDto
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_addScreenSpaceBillboard(
    nint ctx,
    ulong sceneId,
    ulong entity,
    byte* imagePath,
    float ndcX,
    float ndcY,
    float scale,
    float rotationDeg,
    float opacity,
    int zIndex,
    ulong viewportId
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_addViewport(
    nint ctx,
    ulong sceneId,
    uint width,
    uint height,
    byte* name,
    uint handleType, // 0 = windowless | 1 = Win32 | 3 = Xlib | 4 = Xcb | 5 = Metal
    ulong* outPresentationEngine,
    ulong* outCameraEntity
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_checkAlmanacCoverage(
    nint ctx,
    int spkId,
    CTimeRange* taiRange
  );

  // associated with get task status. We are not using it anymore, so drop it
  // avkSimulationContext_downloadImage

#if DEBUG
  // *mut *const c_char -> nint*. Each nint is a pointer to a UTF-8 null terminated string
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_freeComponentNames(nint* names, uint count);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern uint avkSimulationContext_getEntityComponentNames(
    nint ctx,
    ulong sceneId,
    ulong entityExtId,
    nint* outNames,
    uint maxCount // size of outNames array of pointers
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern uint avkSimulationContext_getEntityCount(nint ctx, ulong sceneId);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern uint avkSimulationContext_getEntityIds(
    nint ctx,
    ulong sceneId,
    ulong* outIds,
    uint maxCount // size of outIds array
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_getEntityName(
    nint ctx,
    ulong sceneId,
    byte* outNameUtf8,
    uint maxLen // length of buffer, including space for null terminator
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_getEntityParent(
    nint ctx,
    ulong sceneId,
    ulong entity
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_getSceneHierarchy(
    nint ctx,
    ulong sceneId,
    SceneHierarchyDTO* outBuffer,
    uint capacity, // number of `SceneHierarchyDTO` in `outBuffer`
    uint* outCount
  );
#endif

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_getScreenSpaceBillboard(
    nint ctx,
    ulong sceneId,
    ulong entity,
    FfiScreenSpaceBillboardDTO* outData
  );

  // associated with download image and rendering callback. We are not using it anymore, so drop it
  // avkSimulationContext_getTaskStatus

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_importModel(nint ctx, byte* utf8Path);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern ulong avkSimulationContext_initEarth();

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_loadAlmanacFile(nint ctx, byte* utf8Path);

  // figure out size of computed dto allocation (implicit on what you request)
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_modifyComponent(
    nint ctx,
    ulong sceneId,
    ulong entityId,
    uint command, // 1,2,3
    nint inDto,
    nint outDto
  );

#if DEBUG
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_debugECSPrint(
    nint ctx,
    ulong sceneId,
    uint entityCount,
    ulong* entityIds,
    uint compCount,
    ulong* comps
  );
#endif

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_modifyParticleSystem(
    nint ctx,
    ulong sceneId,
    ulong psId,
    ParticleSystemDTO* inNewPsData,
    ParticleSystemComputedDTO* outNewComputedData
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_removeParticleSystem(
    nint ctx,
    ulong sceneId,
    ulong psId
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_pauseSimulationSync(nint ctx, ulong sceneId);

  [DllImport(LibName, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_tryInitComet(
    nint ctx,
    ulong sceneId,
    int spkId,
    CTimeRange* proposedRange,
    CKeplerianElementsDTO* elements,
    ulong* outCometId
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_reconfigureComet(
    nint ctx,
    ulong sceneId,
    int commandFlags, // 0=query, 2=DETACH
    int spkId,
    ulong* outCometId
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_removeScreenSpaceBillboard(
    nint ctx,
    ulong sceneId,
    ulong entityId
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_removeViewport(
    nint ctx,
    ulong sceneId,
    ulong peHandle
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_resetSimulationSync(nint ctx, ulong sceneId);

  // associated with windowless. Shouldn't be needed now that we are transitioning towards swapchain
  // avkSimulationContext_resize

  // This is async, callback will handle it. false -> failed to submit to logic thread
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_setEpochRange(
    nint ctx,
    ulong sceneId,
    CTimeRange* inTai
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_setScreenSpaceBillboard(
    nint ctx,
    ulong sceneId,
    ulong entity,
    float ndcX,
    float ndcY,
    float scale,
    float rotationDeg,
    float opacity,
    int zIndex
  );

  // destructor for ctx (To put in dispose of runtime)
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_shutdownSync(nint ctx);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_startSimulation(
    nint ctx,
    ulong sceneId,
    int speed
  );

  // "throwning" constructor for ctx (To put in constructor of runtime)
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_startup(
    CStartupParameters* inParams,
    CStartupReturn* outParams
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_transformStaticCamera(
    nint ctx,
    ulong sceneId,
    ulong cameraId,
    int mode,
    nint buffer
  );

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_unloadAlmanacFile(nint ctx, byte* utf8Path);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkSimulationContext_unloadModel(nint ctx, ulong modelId);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern bool avkSimulationContext_setBodyRotationalModel(
    nint ctx,
    ulong sceneId,
    ulong cometBodyEntityId,
    CBodyRotationalModelDTO* dto
  );

  // never called directly, MainThreadDispatchCallbackDelegate does that
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void executeMainThreadCleanup(
    nint vulkanDevice,
    int command,
    nint signalDonePtr
  );

#if DEBUG
  // ── RenderDoc in-application API ─────────────────────────────────────────

  /// <summary>
  /// Returns 1 if the process was launched under RenderDoc and the in-app API
  /// is available, 0 otherwise.  Triggers the one-time library probe on first call.
  /// Only exported from the native library in Debug builds.
  /// </summary>
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern byte avkDebug_isRenderDocAvailable();

  /// <summary>
  /// Requests RenderDoc to capture the next presented frame.
  /// No-op if RenderDoc is not loaded.
  /// Only exported from the native library in Debug builds.
  /// </summary>
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern void avkDebug_triggerCapture();

  /// <summary>
  /// Queues a scoped RenderDoc capture for the next frame of the given windowed PE.
  /// Returns 1 on success, 0 if unavailable or channel full.
  /// Only exported from the native library in Debug builds.
  /// </summary>
  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  public static extern byte avkDebug_startScopedCapture(nint ctx, ulong peId);

  [DllImport(LibName, ExactSpelling = true, CallingConvention = CallingConvention.Cdecl)]
  [return: MarshalAs(UnmanagedType.I1)]
  public static extern bool avkSimulationContext_getDebugTelemetryStats(
    IntPtr ctx,
    out CDebugTelemetryStatsDTO stats
  );
#endif
}

#endregion

#region comet_dtos

/// <summary>
/// Blittable C-layout DTO matching <c>CBodyRotationalModelDTO</c> in <c>ffi.rs</c>.
/// Passed by pointer to <see cref="PInvokeAetherVkCore.avkSimulationContext_setBodyRotationalModel"/>.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CBodyRotationalModelDTO
{
  public double PoleRaDeg;
  public double PoleDecDeg;
  public double PrimeMeridianDeg;
  public double PoleRaRateDegCen;
  public double PoleDecRateDegCen;
  public double RotRateDegDay;
}

/// <summary>
/// Managed representation of an IAU rotational model for use in <see cref="INativeRuntimeService.SetBodyRotationalModel"/>.
/// Values are the same as <see cref="CBodyRotationalModelDTO"/>.
/// </summary>
public sealed record BodyRotationalModelDto(
  double PoleRaDeg,
  double PoleDecDeg,
  double PrimeMeridianDeg,
  double PoleRaRateDegCen,
  double PoleDecRateDegCen,
  double RotRateDegDay
)
{
  /// <summary>Converts to the blittable FFI struct.</summary>
  public CBodyRotationalModelDTO ToDto() =>
    new()
    {
      PoleRaDeg = PoleRaDeg,
      PoleDecDeg = PoleDecDeg,
      PrimeMeridianDeg = PrimeMeridianDeg,
      PoleRaRateDegCen = PoleRaRateDegCen,
      PoleDecRateDegCen = PoleDecRateDegCen,
      RotRateDegDay = RotRateDegDay,
    };

  /// <summary>Default model: north pole aligned, no rotation.</summary>
  public static BodyRotationalModelDto Default { get; } = new(0, 90, 0, 0, 0, 0);
}

#endregion

#region implementation

// Note: this class is not perfect, as it is aware it is a singleton. Acceptable in our case as it
// maps to a native library services of which we know we'll load only once
public sealed class NativeRuntimeService : INativeRuntimeService
{
  // ── Dispatch table ────────────────────────────────────────────────────────

  // Single-instance reference for static callback entry points (singleton pattern)
  private static NativeRuntimeService? _instance;

  private sealed record SimListenerEntry(ulong EntityId, ulong CompForeignId, Action<nint> Handler);

  // volatile: ImmutableInterlocked ensures writes have full barrier; volatile ensures reads in
  // the static callback see the latest reference without a memory-barrier instruction on each read.
  private volatile ImmutableList<SimListenerEntry> _simulationListeners =
    ImmutableList<SimListenerEntry>.Empty;

  private sealed record ExtStateListenerEntry(ExternalStateType StateType, Action<nint> Handler);

  private volatile ImmutableList<ExtStateListenerEntry> _externalStateListeners =
    ImmutableList<ExtStateListenerEntry>.Empty;

  // ── Cached entity IDs ─────────────────────────────────────────────────────

  /// <inheritdoc/>
  public ulong? CameraEntityId { get; private set; }

  /// <inheritdoc/>
  public ulong? PresentationEngineId { get; private set; }

  /// <inheritdoc/>
  public ulong? EarthEntityId { get; private set; }

  /// <inheritdoc/>
  // Populated at startup from CStartupReturn.CometPlanetEntity.
  // Also updated by ReconfigureComet when the ATTACH flag returns a non-zero entity id.
  public ulong? CometEntityId { get; private set; }

  // ── Private fields ────────────────────────────────────────────────────────

  private readonly IUiThreadDispatcher _uiThreadDispatcher;

  // TODO: populated in constructor once Rust startup is wired
  private nint _ctx = 0;
  private ulong _sceneId = 0;

  // Cancelled in Dispose() so async operations (e.g. CommitCometAsync) can observe shutdown.
  private readonly CancellationTokenSource _shutdownCts = new();

  /// <inheritdoc/>
  public CancellationToken ShutdownToken => _shutdownCts.Token;

  // Fires when StartSimulation completes and the scene becomes active.
  private readonly System.Reactive.Subjects.Subject<ulong> _simulationStateUpdated = new();

  /// <inheritdoc/>
  public IObservable<ulong> SimulationStateUpdated => _simulationStateUpdated.AsObservable();

  // ── Constructor ───────────────────────────────────────────────────────────

  public NativeRuntimeService(
    IUiThreadDispatcher uiThreadDispatcher,
    ConsoleService consoleService,
    BreadcrumbService breadcrumbService,
    Action<nint, nuint> panicCallback
  )
  {
    // panic registration
    _panicCallback = panicCallback;
    unsafe
    {
      PInvokeAetherVkCore.avkRegisterPanicCallback(&PanicCallbackThunk);
    }

    // logger registration
    _loggerCallback = consoleService.Log;
    unsafe
    {
      PInvokeAetherVkCore.avkSetLoggerCallback(&LoggerCallbackThunk);
    }

    // breadcrumb registration
    _breadcrumbCallback = (severity, msg) =>
    {
      _ = breadcrumbService.ShowMessageAsync("Engine", msg, TimeSpan.FromSeconds(5), (int)severity);
    };
    unsafe
    {
      PInvokeAetherVkCore.avkSetBreadcrumbCallback(&BreadcrumbCallbackThunk);
    }

    _uiThreadDispatcher = uiThreadDispatcher;
    _instance = this;

    Startup();
  }

  // ── Static callback installation ──────────────────────────────────────────

  private static unsafe void InstallCallbacks()
  {
    PInvokeAetherVkCore.avkSetSimulationCallback(&SimulationCallbackEntry);
    PInvokeAetherVkCore.avkSetExternalStateSimulationCallback(&ExternalStateCallbackEntry);
    PInvokeAetherVkCore.avkSetGetNativeWindowHandleCallback(&GetNativeWindowHandleThunk);
    PInvokeAetherVkCore.avkSetMainThreadDispatchCallback(&MainThreadDispatchThunk);
  }

  // ── Static unmanaged entry points (pinned by the runtime, called from Rust) ──

  /// <summary>
  /// Temporarily holds the provider lambda set by <see cref="AddViewport"/> before the
  /// FFI call. The Rust spin-wait guarantees the callback fires and completes before
  /// <see cref="AddViewport"/> returns, so this is safe to use as a single-slot store.
  /// </summary>
  private static volatile Func<CNativeWindowHandle>? _pendingWindowHandleProvider;

  /// <summary>
  /// Static unmanaged entry point for <c>GET_NATIVE_WINDOW_HANDLE_CALLBACK</c>.
  /// Called by Rust from a non-UI thread to obtain the OS window handle.
  /// Dispatches to the UI thread, fills <paramref name="outHandle"/>, then writes
  /// <c>1</c> to <paramref name="signalDonePtr"/> (matching Rust's Acquire spin-wait).
  ///
  /// On macOS the body is wrapped in a <see cref="CocoaAutoreleasePool"/> so that
  /// Objective-C message sends (e.g. reading <c>MetalLayerPointer</c>) are safe.
  /// </summary>
  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static unsafe void GetNativeWindowHandleThunk(
    CNativeWindowHandle* outHandle,
    nint signalDonePtr
  )
  {
    var instance = Volatile.Read(ref _instance);

    // Always signal done — even on failure — so Rust never spin-waits forever.
    static void SignalDone(nint ptr)
    {
      if (ptr != 0)
        Volatile.Write(ref *(byte*)ptr, 1); // AtomicBool in Rust is a single byte
    }

    if (instance is null || outHandle == null)
    {
      SignalDone(signalDonePtr);
      return;
    }

    void Execute()
    {
#if TARGET_IS_OSX
      using var pool = new CocoaAutoreleasePool();
#endif
      try
      {
        var provider = Volatile.Read(ref _pendingWindowHandleProvider);
        if (provider is not null)
          *outHandle = provider();
      }
      catch
      { /* must not propagate across unmanaged boundary */
      }
      finally
      {
        SignalDone(signalDonePtr);
      }
    }

    if (instance._uiThreadDispatcher.CheckAccess())
    {
      Execute();
    }
    else
    {
      instance._uiThreadDispatcher.Dispatch(Execute);
    }
  }

  /// <summary>
  /// Static unmanaged entry point for <c>MAIN_THREAD_DISPATCH_CALLBACK</c>.
  /// Called by Rust (from the render thread) when Vulkan cleanup work must run on the main thread.
  /// On macOS, <c>vkDestroySwapchainKHR</c> and <c>vkDestroySurfaceKHR</c> touch
  /// <c>CAMetalLayer</c> (a Core Animation object) and must run on the UI thread.
  ///
  /// <para>Commands (mirror <c>simulation_api.rs</c>):
  /// <list type="bullet">
  ///   <item><c>1</c> — <c>process_main_thread_cleanup_queue()</c>: periodic drain.</item>
  ///   <item><c>2</c> — <c>flush_main_thread_cleanup_queue()</c>: full drain on shutdown.</item>
  /// </list>
  /// </para>
  ///
  /// <para>If <paramref name="signalDonePtr"/> is non-zero, Rust is spin-waiting on it.
  /// <c>executeMainThreadCleanup</c> writes <c>1</c> to signal completion.</para>
  /// </summary>
  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static unsafe void MainThreadDispatchThunk(
    nint vulkanDevicePtr,
    int command,
    nint signalDonePtr
  )
  {
    var instance = Volatile.Read(ref _instance);

    static void SignalDone(nint ptr)
    {
      if (ptr != 0)
        Volatile.Write(ref *(byte*)ptr, 1);
    }

    if (instance is null)
    {
      SignalDone(signalDonePtr);
      return;
    }

    // Run inline if we are already on the UI thread (e.g. during shutdown, where
    // ShutdownSync() is called from the UI thread and the Avalonia event loop is no
    // longer pumping). Posting asynchronously in that case deadlocks: the render thread
    // spin-waits on signal_done, but the posted action can never execute because the
    // UI thread is blocked in pthread_join waiting for the render thread.
    // This mirrors the identical pattern in GetNativeWindowHandleThunk.
    void Execute()
    {
#if TARGET_IS_OSX
      using var pool = new CocoaAutoreleasePool();
#endif
      try
      {
        // executeMainThreadCleanup is the Rust-exported function; it runs the cleanup queue
        // entries (vkDestroySwapchainKHR, vkDestroySurfaceKHR) and writes *signal_done itself.
        PInvokeAetherVkCore.executeMainThreadCleanup(vulkanDevicePtr, command, signalDonePtr);
      }
      catch
      {
        // Fallback: ensure Rust's spin-wait terminates even if the native call threw.
        SignalDone(signalDonePtr);
      }
    }

    if (instance._uiThreadDispatcher.CheckAccess())
      Execute();
    else
      instance._uiThreadDispatcher.Dispatch(Execute);
  }

  /// <summary>
  /// Entry point for <c>SIMULATION_CALLBACK</c>. Fans out to all registered listeners
  /// that match the given entity + component combination.
  /// <para>Called on a native thread. Must not throw or block.</para>
  /// </summary>
  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void SimulationCallbackEntry(
    ulong sceneId,
    ulong entityId,
    ulong compForeignId,
    nint dataPtr
  )
  {
    var instance = Volatile.Read(ref _instance);
    if (instance is null)
      return;

    // Reference read of a volatile ImmutableList — safe, atomic on all supported architectures
    var listeners = instance._simulationListeners;
    foreach (var entry in listeners)
    {
      if (entry.EntityId != entityId)
        continue;
      if (entry.CompForeignId != 0 && entry.CompForeignId != compForeignId)
        continue;
      try
      {
        entry.Handler(dataPtr);
      }
      catch
      { /* Handler errors must never escape to native */
      }
    }
  }

  /// <summary>
  /// Entry point for <c>EXTERNAL_STATE_SIMULATION_CALLBACK</c>. Fans out to listeners
  /// registered for the matching state type.
  /// <para>Called on a native thread. Must not throw or block.</para>
  /// </summary>
  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void ExternalStateCallbackEntry(uint stateId, nint dataPtr)
  {
    var instance = Volatile.Read(ref _instance);
    if (instance is null)
      return;

    var listeners = instance._externalStateListeners;
    foreach (var entry in listeners)
    {
      if ((uint)entry.StateType != stateId)
        continue;
      try
      {
        entry.Handler(dataPtr);
      }
      catch
      { /* Handler errors must never escape to native */
      }
    }
  }

  // ── Registration API ──────────────────────────────────────────────────────

  /// <inheritdoc/>
  public IDisposable RegisterSimulationListener(
    ulong entityId,
    ulong componentForeignId,
    Action<nint> handler
  )
  {
    var entry = new SimListenerEntry(entityId, componentForeignId, handler);
    // Read volatile field into local to avoid CS0420; ImmutableInterlocked provides full memory barrier
    var list = _simulationListeners;
    ImmutableInterlocked.Update(ref list, static (l, e) => l.Add(e), entry);
    _simulationListeners = list;
    return Disposable.Create(() =>
    {
      var current = _simulationListeners;
      ImmutableInterlocked.Update(ref current, static (l, e) => l.Remove(e), entry);
      _simulationListeners = current;
    });
  }

  /// <inheritdoc/>
  public IDisposable RegisterExternalStateListener(
    ExternalStateType stateType,
    Action<nint> handler
  )
  {
    var entry = new ExtStateListenerEntry(stateType, handler);
    var list = _externalStateListeners;
    ImmutableInterlocked.Update(ref list, static (l, e) => l.Add(e), entry);
    _externalStateListeners = list;
    return Disposable.Create(() =>
    {
      var current = _externalStateListeners;
      ImmutableInterlocked.Update(ref current, static (l, e) => l.Remove(e), entry);
      _externalStateListeners = current;
    });
  }

  // ── Callbacks & panic registration ────────────────────────────────────────

  // Stored to prevent GC of the managed Action while the native side holds the function pointer
  // Note: this is ugly, cause it implicitly assumes that the class is a singleton. The class
  // shouldn't be aware of its lifetime
  private static Action<nint, nuint>? _panicCallback;
  private static Action<string>? _loggerCallback;
  private static Action<uint, string>? _breadcrumbCallback;

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static void PanicCallbackThunk(nint messagePtr, nuint length)
  {
    try
    {
      _panicCallback?.Invoke(messagePtr, length);
    }
    catch
    { /* must not propagate to native */
    }
  }

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static unsafe void LoggerCallbackThunk(byte* utf8Message)
  {
    if (_loggerCallback == null || utf8Message == null)
      return;
    try
    {
      var msg = StringUtils.GetStringFromUtf8(utf8Message) ?? "Unknown log message";
      _loggerCallback.Invoke(msg);
    }
    catch
    { /* must not propagate */
    }
  }

  [UnmanagedCallersOnly(CallConvs = [typeof(CallConvCdecl)])]
  private static unsafe void BreadcrumbCallbackThunk(uint severity, byte* utf8Message)
  {
    if (_breadcrumbCallback == null || utf8Message == null)
      return;
    try
    {
      var msg = StringUtils.GetStringFromUtf8(utf8Message) ?? "Unknown breadcrumb message";
      _breadcrumbCallback.Invoke(severity, msg);
    }
    catch
    { /* must not propagate */
    }
  }

  // ── INativeRuntimeService — Lifecycle stubs ───────────────────────────────

  // called by constructor
  private void Startup()
  {
    // 1. Setup Asset path
    var assetPath = Path.Combine(AppContext.BaseDirectory, "assets");
    if (!Directory.Exists(assetPath))
      throw new InvalidOperationException($"Asset Path {assetPath} doesn't exist");

    unsafe
    { // scope stackalloc
      int byteCount = Encoding.UTF8.GetByteCount(assetPath);
      byte* utf8Buffer = stackalloc byte[byteCount + 1];
      fixed (char* pAssetPath = assetPath)
        Encoding.UTF8.GetBytes(pAssetPath, assetPath.Length, utf8Buffer, byteCount);
      utf8Buffer[byteCount] = 0;
      PInvokeAetherVkCore.avkSetAssetPath(utf8Buffer);
    }

    // 2. Setup simulation callback and external state simulation callback
    InstallCallbacks();

    // 3. call the startup native method to create the initial scene
    // TODO: Probably expose this stuff to Preferences menu?
    string start = "2025-10-01T12:00:00";
    string end = "2025-11-11T12:00:00";
    var startupParams = new CStartupParameters { StartRange = CTimeRange.FromStrings(start, end) };
    var returnParams = new CStartupReturn();

    unsafe
    {
      if (!PInvokeAetherVkCore.avkSimulationContext_startup(&startupParams, &returnParams))
        throw new InvalidOperationException("avkSimulationContext_startup failed");
    }

    // 4. Get initial state -> comet id, earth id, scene id
    _sceneId = returnParams.SceneId;
    _ctx = returnParams.Ctx;
    EarthEntityId = returnParams.EarthPlanetEntity;
    CometEntityId = returnParams.CometPlanetEntity;
  }

  // called by dispose
  public void ShutdownSync()
  {
    // assumes checks on ctx already done
    PInvokeAetherVkCore.avkSimulationContext_shutdownSync(_ctx);
  }

  // ── INativeRuntimeService — Viewport & Rendering ──────────────────────────

  public unsafe bool AddViewport(
    uint width,
    uint height,
    string name,
    Func<CNativeWindowHandle>? nativeHandleProvider,
    uint handleType,
    out ulong presentationEngineId,
    out ulong cameraEntityId
  )
  {
    presentationEngineId = 0;
    cameraEntityId = 0;

    // Publish the provider so GetNativeWindowHandleThunk can read it.
    // The Rust spin-wait guarantees the callback executes and completes
    // synchronously (from AddViewport's perspective) before the FFI call returns.
    Volatile.Write(ref _pendingWindowHandleProvider, nativeHandleProvider);
    try
    {
      int byteCount = Encoding.UTF8.GetByteCount(name);
      byte* utf8Name = stackalloc byte[byteCount + 1];
      fixed (char* pName = name)
        Encoding.UTF8.GetBytes(pName, name.Length, utf8Name, byteCount);
      utf8Name[byteCount] = 0;

      ulong pe = 0,
        cam = 0;
      bool ok = PInvokeAetherVkCore.avkSimulationContext_addViewport(
        _ctx,
        _sceneId,
        width,
        height,
        utf8Name,
        handleType,
        &pe,
        &cam
      );

      if (ok)
      {
        presentationEngineId = pe;
        cameraEntityId = cam;
        CameraEntityId = cam;
        PresentationEngineId = pe;
      }

      return ok;
    }
    finally
    {
      Volatile.Write(ref _pendingWindowHandleProvider, null);
    }
  }

  public void RemoveViewport(ulong presentationEngineId)
  {
    PInvokeAetherVkCore.avkSimulationContext_removeViewport(_ctx, _sceneId, presentationEngineId);
  }

  public void ResizeViewport(ulong presentationEngineId, uint width, uint height)
  {
    throw new NotImplementedException();
  }

  // ── INativeRuntimeService — Simulation Flow ───────────────────────────────

  public bool ResetSimulationSync()
  {
    throw new NotImplementedException();
  }

  public bool PauseSimulationSync()
  {
    throw new NotImplementedException();
  }

  public bool StartSimulation(int simSpeed)
  {
    throw new NotImplementedException();
  }

  // ── INativeRuntimeService — ECS & Camera ─────────────────────────────────

  public unsafe bool AddCameraAnimation(ulong cameraId, AnimationTarget animation)
  {
    var dto = animation.ToDTO();
    return PInvokeAetherVkCore.avkSimulationContext_addCameraAnimation(
      _ctx,
      _sceneId,
      cameraId,
      &dto
    );
  }

  // ── Camera transform typed methods (all map to avkSimulationContext_transformStaticCamera) ──

  public unsafe bool CameraSetRotoTranslate(
    ulong cameraId,
    System.Numerics.Vector3 position,
    System.Numerics.Quaternion rotation
  )
  {
    // mode 2: disp_x | disp_y | disp_z | quat_x | quat_y | quat_z | quat_w  [f32; 7]
    float* buf = stackalloc float[7]
    {
      position.X,
      position.Y,
      position.Z,
      rotation.X,
      rotation.Y,
      rotation.Z,
      rotation.W,
    };
    return PInvokeAetherVkCore.avkSimulationContext_transformStaticCamera(
      _ctx,
      _sceneId,
      cameraId,
      mode: 2,
      (nint)buf
    );
  }

  public unsafe bool CameraSetPerspective(
    ulong cameraId,
    float fov,
    float aspectRatio,
    float near,
    float far
  )
  {
    // mode 1: fov | aspect_ratio | near | far  [f32; 4]
    float* buf = stackalloc float[4] { fov, aspectRatio, near, far };
    return PInvokeAetherVkCore.avkSimulationContext_transformStaticCamera(
      _ctx,
      _sceneId,
      cameraId,
      mode: 1,
      (nint)buf
    );
  }

  public unsafe bool CameraSetOrthographic(
    ulong cameraId,
    float left,
    float right,
    float bottom,
    float top,
    float near,
    float far
  )
  {
    // mode 0: left | right | bottom | top | near | far  [f32; 6]
    float* buf = stackalloc float[6] { left, right, bottom, top, near, far };
    return PInvokeAetherVkCore.avkSimulationContext_transformStaticCamera(
      _ctx,
      _sceneId,
      cameraId,
      mode: 0,
      (nint)buf
    );
  }

  // ── INativeRuntimeService — Particle Systems ──────────────────────────────

  public unsafe bool AddParticleSystem(
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ulong outPsId
  )
  {
    var dto = new ParticleSystemDTO(psModel, psJet);
    ulong psId = 0;
    bool ok = PInvokeAetherVkCore.avkSimulationContext_addParticleSystem(
      _ctx,
      _sceneId,
      &dto,
      &psId,
      null
    );
    outPsId = psId;
    return ok;
  }

  public unsafe ParticleSystemComputedProperties? AddFirstParticleSystem(
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ulong outPsId
  )
  {
    var dto = new ParticleSystemDTO(psModel, psJet);
    ulong psId = 0;
    var computed = new ParticleSystemComputedDTO();
    bool ok = PInvokeAetherVkCore.avkSimulationContext_addParticleSystem(
      _ctx,
      _sceneId,
      &dto,
      &psId,
      &computed
    );
    outPsId = psId;
    if (!ok)
      return null;
    return new ParticleSystemComputedProperties(computed.Beta, computed.DustProductionRateAt1AuKgs);
  }

  public unsafe bool ModifyParticleSystem(
    ulong psId,
    ParticleSystemModel psModel,
    ParticleSystemJet psJet,
    out ParticleSystemComputedProperties outPsComputedProps
  )
  {
    var dto = new ParticleSystemDTO(psModel, psJet);
    var computed = new ParticleSystemComputedDTO();
    bool ok = PInvokeAetherVkCore.avkSimulationContext_modifyParticleSystem(
      _ctx,
      _sceneId,
      psId,
      &dto,
      &computed
    );
    outPsComputedProps = new ParticleSystemComputedProperties(
      computed.Beta,
      computed.DustProductionRateAt1AuKgs
    );
    return ok;
  }

  public bool RemoveParticleSystem(ulong psId)
  {
    if (_ctx == 0 || _sceneId == 0 || psId == 0)
      return false;
    return PInvokeAetherVkCore.avkSimulationContext_removeParticleSystem(_ctx, _sceneId, psId);
  }

  // ── INativeRuntimeService — Orbital Mechanics & Almanacs ─────────────────

  public unsafe bool ReconfigureComet(int commandFlags, int spkId, out ulong cometBodyId)
  {
    ulong cometId = 0;
    bool ok = PInvokeAetherVkCore.avkSimulationContext_reconfigureComet(
      _ctx,
      _sceneId,
      commandFlags,
      spkId,
      &cometId
    );
    cometBodyId = cometId;
    // Update cached entity id when ATTACH succeeds and id is non-zero
    if (ok && cometId != 0)
      CometEntityId = cometId;
    return ok;
  }

  public unsafe bool TryInitComet(int spkId, TimeRange proposedRange, Models.SmallBodyDataComponent sbData, out ulong cometBodyId)
  {
    cometBodyId = 0;
    if (_ctx == 0) return false;

    var rangeDto = default(CTimeRange);
    rangeDto.Nanoseconds[0] = proposedRange.StartNs;
    rangeDto.Nanoseconds[1] = proposedRange.EndNs;
    rangeDto.Centuries[0] = proposedRange.StartCenturies;
    rangeDto.Centuries[1] = proposedRange.EndCenturies;

    var keplerianDto = new CKeplerianElementsDTO
    {
      Eccentricity = sbData.E,
      PerihelionDistanceAu = sbData.Q,
      InclinationDeg = sbData.I,
      LongitudeOfAscendingNodeDeg = sbData.Om,
      ArgumentOfPerihelionDeg = sbData.W
    };

    ulong outId = 0;
    bool ok = PInvokeAetherVkCore.avkSimulationContext_tryInitComet(
        _ctx, _sceneId, spkId, &rangeDto, &keplerianDto, &outId);
        
    cometBodyId = outId;
    if (ok && outId != 0)
      CometEntityId = outId;
      
    return ok;
  }

  public unsafe bool SetBodyRotationalModel(ulong cometBodyEntityId, BodyRotationalModelDto dto)
  {
    var cDto = dto.ToDto();
    return PInvokeAetherVkCore.avkSimulationContext_setBodyRotationalModel(
      _ctx,
      _sceneId,
      cometBodyEntityId,
      &cDto
    );
  }

  public unsafe Task<ulong> LoadAlmanacFileAsync(string path)
  {
    int byteCount = Encoding.UTF8.GetByteCount(path);
    byte* utf8Path = stackalloc byte[byteCount + 1];
    fixed (char* pPath = path)
      Encoding.UTF8.GetBytes(pPath, path.Length, utf8Path, byteCount);
    utf8Path[byteCount] = 0;

    // The load is fire-and-forget to the logic thread; completion arrives via
    // ExternalState::AlmanacImported callback. We return a TCS that CometConfigService
    // resolves when that callback fires.
    var tcs = new TaskCompletionSource<ulong>(TaskCreationOptions.RunContinuationsAsynchronously);

    // Cancel the TCS immediately if the service is being disposed so that callers
    // (e.g. CommitCometAsync) unblock without waiting for a callback that will never arrive.
    CancellationTokenRegistration shutdownReg = _shutdownCts.Token.Register(() =>
      tcs.TrySetCanceled()
    );

    // Register a one-shot listener for AlmanacImported that resolves the TCS.
    // The listener stays alive for non-load events (e.g. a concurrent unload, operation=2)
    // and only self-disposes on the load-success event (operation=1).
    IDisposable? token = null;
    token = RegisterExternalStateListener(
      ExternalStateType.AlmanacImported,
      dataPtr =>
      {
        unsafe
        {
          var dto = *(CAlmanacImportedDTO*)dataPtr;
          if (dto.Operation != 1)
            return; // unload or load-failure event — keep listening
        }

        token?.Dispose();
        shutdownReg.Dispose();
        unsafe
        {
          var dto = *(CAlmanacImportedDTO*)dataPtr;
          // Cast via uint first to zero-extend rather than sign-extend the i32 naif_id.
          tcs.TrySetResult((ulong)(uint)dto.NaifId);
        }
      }
    );

    bool enqueued = PInvokeAetherVkCore.avkSimulationContext_loadAlmanacFile(_ctx, utf8Path);
    if (!enqueued)
    {
      token?.Dispose();
      shutdownReg.Dispose();
      tcs.TrySetException(
        new InvalidOperationException(
          $"LoadAlmanacFile: failed to enqueue command for path '{path}'"
        )
      );
    }

    return tcs.Task;
  }

  public unsafe bool UnloadAlmanacFile(string path)
  {
    int byteCount = Encoding.UTF8.GetByteCount(path);
    byte* utf8Path = stackalloc byte[byteCount + 1];
    fixed (char* pPath = path)
      Encoding.UTF8.GetBytes(pPath, path.Length, utf8Path, byteCount);
    utf8Path[byteCount] = 0;
    return PInvokeAetherVkCore.avkSimulationContext_unloadAlmanacFile(_ctx, utf8Path);
  }

  // ── INativeRuntimeService — Timeline ─────────────────────────────────────

  public unsafe bool SetEpochRange(
    short startCenturies,
    ulong startNs,
    short endCenturies,
    ulong endNs
  )
  {
    var range = new CTimeRange();
    range.Nanoseconds[0] = startNs;
    range.Nanoseconds[1] = endNs;
    range.Centuries[0] = startCenturies;
    range.Centuries[1] = endCenturies;
    return PInvokeAetherVkCore.avkSimulationContext_setEpochRange(_ctx, _sceneId, &range);
  }

  public unsafe bool CheckAlmanacCoverage(
    int spkId,
    short startCenturies,
    ulong startNs,
    short endCenturies,
    ulong endNs
  )
  {
    var range = new CTimeRange();
    range.Nanoseconds[0] = startNs;
    range.Nanoseconds[1] = endNs;
    range.Centuries[0] = startCenturies;
    range.Centuries[1] = endCenturies;
    return PInvokeAetherVkCore.avkSimulationContext_checkAlmanacCoverage(_ctx, spkId, &range);
  }

  // ── INativeRuntimeService — 3D Models & Assets ───────────────────────────

  public Task<ulong> ImportModelAsync(string path)
  {
    throw new NotImplementedException();
  }

  public void UnloadModel(ulong modelId)
  {
    throw new NotImplementedException();
  }

  // ── INativeRuntimeService — Screen Space Billboards ──────────────────────

  public ulong AddScreenSpaceBillboard(string imagePath, ScreenSpaceBillboard billboard)
  {
    throw new NotImplementedException();
  }

  public bool SetScreenSpaceBillboard(ulong entityId, ScreenSpaceBillboard billboard)
  {
    throw new NotImplementedException();
  }

  public bool RemoveScreenSpaceBillboard(ulong entityId)
  {
    throw new NotImplementedException();
  }

  public bool GetScreenSpaceBillboard(ulong entityId, out ScreenSpaceBillboard outData)
  {
    throw new NotImplementedException();
  }

#if DEBUG
  // ── INativeRuntimeService — RenderDoc (debug only) ───────────────────────

  /// <inheritdoc/>
  public bool IsRenderDocAvailable()
  {
    try
    {
      return PInvokeAetherVkCore.avkDebug_isRenderDocAvailable() != 0;
    }
    catch (EntryPointNotFoundException)
    {
      // Loaded a release build of the native library in a debug .NET build.
      return false;
    }
  }

  /// <inheritdoc/>
  public void TriggerRenderDocCapture()
  {
    try
    {
      PInvokeAetherVkCore.avkDebug_triggerCapture();
    }
    catch (EntryPointNotFoundException)
    {
      // No-op: loaded a release native library.
    }
  }

  /// <inheritdoc/>
  public bool StartScopedRenderDocCapture(ulong presentationEngineId)
  {
    try
    {
      return PInvokeAetherVkCore.avkDebug_startScopedCapture((nint)_ctx, presentationEngineId) != 0;
    }
    catch (EntryPointNotFoundException)
    {
      // Loaded a release build of the native library in a debug .NET build.
      return false;
    }
  }

  public unsafe void DebugECSPrint(
    uint entityCount,
    ulong[] entityIds,
    uint compCount,
    ulong[] comps
  )
  {
    if (_ctx == 0)
      return;
    fixed (ulong* pEntities = entityIds)
    fixed (ulong* pComps = comps)
    {
      PInvokeAetherVkCore.avkSimulationContext_debugECSPrint(
        _ctx,
        _sceneId,
        entityCount,
        pEntities,
        compCount,
        pComps
      );
    }
  }

  public bool GetDebugTelemetryStats(out DebugTelemetryStats stats)
  {
    if (
      _ctx != 0
      && PInvokeAetherVkCore.avkSimulationContext_getDebugTelemetryStats(_ctx, out var cStats)
    )
    {
      stats = new DebugTelemetryStats(
        cStats.OsPhysicalRamBytes,
        cStats.OsVirtualRamBytes,
        cStats.CpuAllocatedBytes,
        cStats.GpuAllocatedBytes,
        cStats.LogicThreadCpuTimeMs,
        cStats.RenderThreadCpuTimeMs,
        cStats.ReservedGpuExecutionMs
      );
      return true;
    }

    stats = null!;
    return false;
  }
#endif

  // ── IDisposable ───────────────────────────────────────────────────────────

  public void Dispose()
  {
    // Signal shutdown to any in-flight async operations (CommitCometAsync, LoadAlmanacFileAsync)
    // before tearing down the native context, so they can abort rather than hanging.
    _shutdownCts.Cancel();
    _simulationStateUpdated.Dispose();
    _instance = null;
    if (_ctx == 0)
    {
      _shutdownCts.Dispose();
      return;
    }

    ShutdownSync();
    _shutdownCts.Dispose();
  }
}

#endregion


// Probably to move into AetherVk.Logic.Models
#region public_facing_record_classes

public record AnimationTarget(Vector3 Pos, Quaternion Rot, float Seconds)
{
  internal AnimationTargetDTO ToDTO()
  {
    return new AnimationTargetDTO(Pos, Rot, Seconds);
  }
}

/// <summary>
/// Common members among all dust jets
/// </summary>
public record ParticleSystemModel(
  float MassVariabilityPerc,
  float DiametreUm,
  float DensityGCm3,
  float ScatteringEfficiency,
  float Afrho0Cm,
  float AfrhoPower,
  float AfrhoCutoffAu,
  float AfrhoMaxValueCm
);

/// <summary>
/// Visual and dispersion Properties of a dust jet
/// </summary>
public record ParticleSystemJet(
  float LatitudeRad,
  float LongitudeRad,
  float ApertureRad,
  float StartVelocityMean,
  float StartVelocityStd,
  Vector4 StreamColor,
  float NucleusRadiusKm,
  uint Seed
);

public record ParticleSystemComputedProperties(float Beta, float DustProductionRateAt1AuKgs);

public record ScreenSpaceBillboard(
  float NdcX,
  float NdcY,
  float Scale,
  float RotationDeg,
  float Opacity,
  uint ZIndex
);

#if DEBUG
public sealed record DebugTelemetryStats(
  ulong OsPhysicalRamBytes,
  ulong OsVirtualRamBytes,
  ulong CpuAllocatedBytes,
  ulong GpuAllocatedBytes,
  double LogicThreadCpuTimeMs,
  double RenderThreadCpuTimeMs,
  double ReservedGpuExecutionMs
);
#endif

#endregion

#region c_structs

// ==========================================
// Required DTOs and Delegates
// ==========================================

/// <summary>
/// Platform-agnostic 16-byte handle passed to Rust's <c>GET_NATIVE_WINDOW_HANDLE_CALLBACK</c>.
/// Mirrors <c>CNativeWindowHandle</c> in <c>simulation_api.rs</c> exactly
/// (<c>#[repr(C)]</c>, 16 bytes, <c>bytemuck::Pod</c>).
/// <list type="bullet">
///   <item><b>Linux (Xlib)</b>: <see cref="Field0"/> = <c>Display*</c>, <see cref="Field1"/> = <c>Window</c> (XID).
///     Avalonia 11 runs under Xlib/XWayland — native Wayland is not supported.</item>
///   <item><b>Windows</b>: <see cref="Field0"/> = <c>HINSTANCE</c>, <see cref="Field1"/> = <c>HWND</c>.</item>
///   <item><b>macOS</b>: <see cref="Field0"/> = <c>CAMetalLayer*</c>, <see cref="Field1"/> = 0.</item>
/// </list>
/// Use <see cref="NativeWindowHandleProvider"/> to construct platform-specific instances.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct CNativeWindowHandle
{
  /// <summary>Display* (Xlib) | HINSTANCE (Win32) | CAMetalLayer* (macOS) — as u64.</summary>
  public ulong Field0;

  /// <summary>Window/XID (Xlib) | HWND (Win32) | 0 (macOS) — as u64.</summary>
  public ulong Field1;
}

/// <summary>
/// C# representation of the Rust AnimationTargetDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct AnimationTargetDTO(Vector3 pos, Quaternion rot, float durationS)
{
  public readonly float posX = pos.X;
  public readonly float posY = pos.Y;
  public readonly float posZ = pos.Z;
  public readonly float rotX = rot.X;
  public readonly float rotY = rot.Y;
  public readonly float rotZ = rot.Z;
  public readonly float rotW = rot.W;
  public readonly float durationS = durationS;
}

/// <summary>
/// C# representation of the Rust aethervk_core_rlib::scene::ParticleSystemDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct ParticleSystemDTO(ParticleSystemModel model, ParticleSystemJet jet)
{
  // -- start of jet common properties --
  public readonly float MassVariabilityPerc = model.MassVariabilityPerc;
  public readonly float DiametreUm = model.DiametreUm;
  public readonly float DensityGCm3 = model.DensityGCm3;
  public readonly float ScatteringEfficiency = model.ScatteringEfficiency;

  public readonly float Afrho0Cm = model.Afrho0Cm;
  public readonly float AfrhoPower = model.AfrhoPower;
  public readonly float AfrhoCutoffAu = model.AfrhoCutoffAu;
  public readonly float AfrhoMaxValueCm = model.AfrhoMaxValueCm;

  // -- start jet specific properties --
  public readonly float LatitudeRad = jet.LatitudeRad;
  public readonly float LongitudeRad = jet.LongitudeRad;
  public readonly float ApertureRad = jet.ApertureRad;
  public readonly float StartVelocityMean = jet.StartVelocityMean;
  public readonly float StartVelocityStd = jet.StartVelocityStd;

  // unroll cause `fixed float` doesn't let this be `readonly` (can't use `InlineArray` cause we are
  // not in .NET 8 or higher)
  public readonly float StreamColor0 = jet.StreamColor.X;
  public readonly float StreamColor1 = jet.StreamColor.Y;
  public readonly float StreamColor2 = jet.StreamColor.Z;
  public readonly float StreamColor3 = jet.StreamColor.W;

  public readonly float NucleusRadiusKm = jet.NucleusRadiusKm;
  public readonly uint Seed = jet.Seed;
}

/// <summary>
/// C# representation of the Rust aethervk_core_rlib::scene::ParticleSystemComputedDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct ParticleSystemComputedDTO
{
  public readonly float Beta;
  public readonly float DustProductionRateAt1AuKgs;
}

/// <summary>
/// C# representation of the Rust aethervk_core_cdylib::ffi::FfiScreenSpaceBillboardDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct FfiScreenSpaceBillboardDTO
{
  public readonly float NdcX;
  public readonly float NdcY;
  public readonly float Scale;
  public readonly float RotationDeg;
  public readonly float Opacity;
  public readonly uint ZIndex;
  public readonly ulong ViewportId;
}

/// <summary>
/// C# representation of aethervk_core_rlib::simulation_api::external_state::ExternalState
/// identifier.
/// </summary>
public enum ExternalStateType : uint
{
  TimeRange            = 1,
  ModelImported        = 2,
  AlmanacImported      = 3,
  CometInitialized     = 4,
  SunVisibilityChanged = 5,
  /// <summary>
  /// Emitted once by <c>BuildCometTrajectory</c> after <c>force_reposition</c> completes.
  /// Carries the post-commit comet position in AU (heliocentric SUN_ECLIPJ2000, f64).
  /// Payload: <see cref="CCometPositionSnapshotDTO"/>.
  /// </summary>
  CometPositionSnapshot = 6,
}

/// <summary>
/// C# representation of aethervk_core_rlib::simulation_api::external_state::ExternalState
/// <c>CAlamanacImported</c> arm.
///
/// Layout (40 bytes, matches Rust <c>#[repr(C)]</c>):
/// <list type="bullet">
///   <item><c>Operation : u32</c> — 0 = load failed, 1 = loaded, 2 = unloaded.</item>
///   <item><c>NaifId   : i32</c> — discovered NAIF/SPK body ID; 0 if unknown or on unload.</item>
///   <item><c>PathBytes : [u8; 32]</c> — UTF-8 basename, null-terminated.</item>
/// </list>
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CAlmanacImportedDTO
{
  /// <summary>
  /// Operation discriminant mirroring <c>CAlamanacImported.operation</c> in Rust:
  /// <list type="bullet">
  ///   <item>0 = load failed</item>
  ///   <item>1 = loaded successfully</item>
  ///   <item>2 = unloaded successfully</item>
  /// </list>
  /// </summary>
  public uint Operation;

  /// <summary>
  /// Discovered NAIF/SPK body ID. 0 when unknown, multi-body, load failed, or on unload.
  /// </summary>
  public int NaifId;

  public fixed byte PathBytes[32];

  // Helper to extract the string cleanly
  public string GetPath()
  {
    fixed (byte* p = PathBytes)
    {
      // find null terminator (Assuming UTF-8 from Rust)
      int len = 0;
      while (len < 32 && p[len] != 0)
        len++;
      return Encoding.UTF8.GetString(p, len);
    }
  }
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CCometInitializedDTO
{
  public uint Success;
  public int SpkId;
}

/// <summary>
/// C# mirror of <c>CCometPositionSnapshot</c> (Rust, <c>external_state</c> module).
///
/// Emitted via <c>ExternalState::CometPositionSnapshot</c> (state_id = 6) once by
/// <c>BuildCometTrajectory</c> after <c>force_reposition</c> completes. Allows
/// <see cref="CometPositionTrackerService"/> to update its position subject immediately
/// — without requiring the simulation to be running.
///
/// Layout: 32 bytes—matches Rust <c>#[repr(C)]</c>:
/// <list type="bullet">
///   <item><c>SpkId  : i32</c></item>
///   <item><c>_Pad   : i32</c></item>
///   <item><c>PosX   : f64</c> — AU (heliocentric SUN_ECLIPJ2000)</item>
///   <item><c>PosY   : f64</c></item>
///   <item><c>PosZ   : f64</c></item>
/// </list>
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct CCometPositionSnapshotDTO
{
  public readonly int  SpkId;
  public readonly int  _Pad;
  public readonly double PosX;
  public readonly double PosY;
  public readonly double PosZ;
}

/// <summary>
/// C# representation of aethervk_core_rlib::simulation_api::external_state::ExternalState
/// <c>CModelImported</c> arm.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CModelImportedDTO
{
  public uint WasSuccessful; // 0 = no, !=0 = yes
  public fixed byte PathBytes[32];

  /// <summary>Extracts the UTF-8 basename from the fixed buffer.</summary>
  public string GetPath()
  {
    fixed (byte* p = PathBytes)
    {
      int len = 0;
      while (len < 32 && p[len] != 0)
        len++;
      return Encoding.UTF8.GetString(p, len);
    }
  }
}

[StructLayout(LayoutKind.Sequential)]
internal unsafe struct CTimeRange
{
  public fixed ulong Nanoseconds[2];
  public fixed short Centuries[2];

  internal static CTimeRange FromStrings(string isoStart, string isoEnd)
  {
    if (!TimeUtils.TryParseIso8601(isoStart, out DateTimeOffset startDateTime))
      throw new ArgumentException(nameof(isoStart));
    if (!TimeUtils.TryParseIso8601(isoEnd, out DateTimeOffset endDateTime))
      throw new ArgumentException(nameof(isoEnd));

    var (startCenturies, startNanoseconds) = TimeUtils.ToTaiParts(startDateTime);
    var (endCenturies, endNanoseconds) = TimeUtils.ToTaiParts(endDateTime);

    var range = new CTimeRange();
    unsafe
    {
      range.Centuries[0] = startCenturies;
      range.Centuries[1] = endCenturies;
      range.Nanoseconds[0] = startNanoseconds;
      range.Nanoseconds[1] = endNanoseconds;
    }

    return range;
  }
}

/// <summary>
/// DTO emitted by <c>SIMULATION_CALLBACK</c> for <see cref="ComponentForeignId.CometPosition"/>.
/// Three IEEE-754 double-precision floats representing the comet nucleus position in
/// simulation units (AU). Layout must match the Rust Vec3f64 byte order.
/// TODO (Rust): confirm exact field order and size once comp_foreign_id = 3 is stabilized.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct CometPositionDTO
{
  public readonly double X;
  public readonly double Y;
  public readonly double Z;
}

/// <summary>
/// DTO emitted by <c>SIMULATION_CALLBACK</c> for <see cref="ComponentForeignId.HighResTransform"/>.
/// Mirrors <c>aethervk_core_rlib::scene::HighResTransformDTO</c>.
/// Position f64×3 | rotation quat (rw,rx,ry,rz) f32×4 | scale f32×3 | _pad u32 — total 56 bytes.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct HighResTransformDTO
{
  public readonly double PosX;
  public readonly double PosY;
  public readonly double PosZ;
  public readonly float RotW;
  public readonly float RotX;
  public readonly float RotY;
  public readonly float RotZ;
  public readonly float ScaleX;
  public readonly float ScaleY;
  public readonly float ScaleZ;

  /// <summary>Padding to reach 56 bytes. Must not be read.</summary>
  private readonly uint _pad;
}

/// <summary>
/// DTO emitted by <c>SIMULATION_CALLBACK</c> for <see cref="ComponentForeignId.CameraProjection"/>.
/// Mirrors <c>aethervk_core_rlib::scene::CameraDTO</c>.
/// Total: 40 bytes (9×f32 + 1×u8 + 3×u8 pad).
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal readonly struct CameraProjectionDTO
{
  public readonly float Fov;
  public readonly float Aspect;
  public readonly float Near;
  public readonly float Far;
  public readonly float Left;
  public readonly float Right;
  public readonly float Bottom;
  public readonly float Top;
  public readonly float FocusDistance;

  /// <summary>0 = Perspective, 1 = Orthographic.</summary>
  public readonly byte IsOrthographic;
  private readonly byte _pad0;
  private readonly byte _pad1;
  private readonly byte _pad2;
}

/// in param for avkSimulationContext_startup
[StructLayout(LayoutKind.Sequential)]
internal struct CStartupParameters
{
  public CTimeRange StartRange;
}

/// out param for avkSimulationContext_startup
[StructLayout(LayoutKind.Sequential)]
internal readonly struct CStartupReturn
{
  public readonly ulong EarthPlanetEntity;
  public readonly ulong CometPlanetEntity;
  public readonly ulong SceneId;
  public readonly nint Ctx;
}

#if DEBUG
[StructLayout(LayoutKind.Sequential)]
internal struct CDebugTelemetryStatsDTO
{
  public ulong OsPhysicalRamBytes;
  public ulong OsVirtualRamBytes;
  public ulong CpuAllocatedBytes;
  public ulong GpuAllocatedBytes;
  public double LogicThreadCpuTimeMs;
  public double RenderThreadCpuTimeMs;
  public double ReservedGpuExecutionMs;
}

[StructLayout(LayoutKind.Sequential)]
internal struct SceneHierarchyDTO
{
  public ulong entityId;
  public ulong parentId;
}

#endif

#endregion

[StructLayout(LayoutKind.Sequential)]
internal struct CKeplerianElementsDTO
{
  public double Eccentricity;
  public double PerihelionDistanceAu;
  public double InclinationDeg;
  public double LongitudeOfAscendingNodeDeg;
  public double ArgumentOfPerihelionDeg;
}

/// <summary>
/// C# mirror of <c>CSunVisibilityChanged</c> (Rust, <c>external_state</c> module).
///
/// Layout: 12 bytes — matches Rust <c>#[repr(C)]</c>:
/// <list type="bullet">
///   <item><c>IsVisible : u32</c> — 1 = sun entered the camera frustum; 0 = sun exited.</item>
///   <item><c>NdcX : f32</c> — projected NDC X of the sun (may exceed ±1 when off-screen).</item>
///   <item><c>NdcY : f32</c> — projected NDC Y of the sun (may exceed ±1 when off-screen).</item>
/// </list>
///
/// Both <c>NdcX</c> and <c>NdcY</c> are valid regardless of on/off-screen status.
/// Consumers should use <c>IsVisible</c> to gate on/off logic, and the NDC pair to
/// compute the direction angle for the arrowhead indicator.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
internal struct CSunVisibilityChangedDTO
{
  /// <summary>1 = sun entered frustum; 0 = sun exited.</summary>
  public uint  IsVisible;
  /// <summary>Projected NDC X (may exceed ±1 when sun is off-screen).</summary>
  public float NdcX;
  /// <summary>Projected NDC Y (may exceed ±1 when sun is off-screen).</summary>
  public float NdcY;
}
