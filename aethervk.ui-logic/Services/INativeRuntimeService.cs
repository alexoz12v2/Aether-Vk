using System;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

/// <summary>
/// Safe, exclusive C# interface for interacting with the Aether-Vk Native Runtime.
/// Implementations of this interface manage the underlying native context pointer.
///
/// Note: `sceneId` has been removed from all interface methods as we are managing a single scene
/// and it's the class' responsability to track that and communicate it to the FFI layer
/// </summary>
public interface INativeRuntimeService : IDisposable
{
  // ==========================================
  // Lifecycle & Context Management
  // ==========================================
  bool Startup();
  void ShutdownSync();

  // ==========================================
  // Viewport & Rendering
  // ==========================================
  bool AddViewport(uint width, uint height, string name, out ulong presentationEngineId, out ulong cameraEntityId);
  void RemoveViewport(ulong presentationEngineId);
  void ResizeViewport(ulong presentationEngineId, uint width, uint height);

  /// <summary>
  /// Safely polls getTaskStatus without blocking, then copies the frame to the buffer.
  /// </summary>
  Task<bool> DownloadImageAsync(ulong taskId, IntPtr bufferPtr, nuint bufferSize);

  // ==========================================
  // Simulation Flow Control
  // ==========================================
  bool ResetSimulationSync();
  bool PauseSimulationSync();
  bool StartSimulation(int simSpeed);

  // ==========================================
  // ECS Components & Camera
  // ==========================================
  // Note: `inDto` and `outComputedDto` are passed as IntPtr to allow unmanaged struct blasting
  bool ModifyComponent(ulong entityId, uint command, IntPtr inDto, IntPtr outComputedDto);

  bool AddCameraAnimation(ulong cameraId, ref AnimationTargetDTO animation);

  // Mode: 0 = Ortho [f32;6], 1 = Persp [f32;4], 2 = RotoTranslate [f32;7]
  bool TransformStaticCamera(ulong cameraId, int mode, IntPtr buffer);

  // ==========================================
  // Particle Systems
  // ==========================================
  bool AddParticleSystem(ref ParticleSystemDTO particleSystem, out ulong outPsId);
  bool ModifyParticleSystem(ulong psId, ref ParticleSystemDTO particleSystem, out ParticleSystemComputedDTO outPsComputedProps);

  // ==========================================
  // Orbital Mechanics & Almanacs
  // ==========================================
  bool ReconfigureComet();
  // purposefully called differently with respect to its native method as we wait for it by using
  // `ExternalStateDispatcher` utility`
  Task<ulong> LoadAlmanacFileAsync(string path);
  bool UnloadAlmanacFile(string path);

  // ==========================================
  // 3D Models & Assets
  // ==========================================
  void SetAssetPath(string path);
  // external state management with transient
  Task<ulong> ImportModelAsync(string path);
  void UnloadModel(ulong modelId);

  // ==========================================
  // Screen Space Billboards (UI Overlays)
  // ==========================================
  ulong AddScreenSpaceBillboard(string imagePath, float ndcX, float ndcY, float scale, float rotationDeg, float opacity, int zIndex, ulong viewportId);
  bool SetScreenSpaceBillboard(ulong entityId, float ndcX, float ndcY, float scale, float rotationDeg, float opacity, int zIndex);
  bool RemoveScreenSpaceBillboard(ulong entityId);
  // TODO Probably to remove
  bool GetScreenSpaceBillboard(ulong entityId, out FfiScreenSpaceBillboardDTO outData);

  // ==========================================
  // Callbacks & Diagnostics
  // ==========================================
  // Depending on your architecture, you might prefer exposing these as standard C# `event`s
  // inside the implementation rather than interface methods, but they are included here for completeness.
  void RegisterPanicCallback(PanicCallbackDelegate cb);
  void SetLoggerCallback(LoggerCallbackDelegate cb);
  void SetBreadcrumbCallback(BreadcrumbCallbackDelegate cb);
  void SetSimulationCallback(SimulationCallbackDelegate cb);
  void SetExternalStateSimulationCallback(ExternalStateSimulationCallbackDelegate cb);
  void SetRenderCallback(RenderCallbackDelegate cb);
  void SetMainThreadDispatchCallback(MainThreadDispatchCallbackDelegate cb);
}

// ==========================================
// Required DTOs and Delegates
// ==========================================

/// <summary>
/// C# representation of the Rust AnimationTargetDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct AnimationTargetDTO
{
  public readonly float posX, posY, posZ;
  public readonly float rotX, rotY, rotZ, rotW;
  public readonly float durationS;
}

/// <summary>
/// C# representation of the Rust aethervk_core_rlib::scene::ParticleSystemDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct ParticleSystemDTO
{
  // -- start of jet common properties --
  public readonly float MassVariabilityPerc;
  public readonly float DiametreUm;
  public readonly float DensityGCm3;
  public readonly float ScatteringEfficiency;

  public readonly float Afrho0Cm;
  public readonly float AfrhoPower;
  public readonly float AfrhoCutoffAu;
  public readonly float AfrhoMaxValueCm;

  // -- start jet specific properties --
  public readonly float LatitudeRad;
  public readonly float LongitudeRad;
  public readonly float ApertureRad;
  public readonly float StartVelocityMean;
  public readonly float StartVelocityStd;

  // unroll cause `fixed float` doesn't let this be `readonly` (can't use `InlineArray` cause we are
  // not in .NET 8 or higher)
  public readonly float StreamColor0;
  public readonly float StreamColor1;
  public readonly float StreamColor2;
  public readonly float StreamColor3;
}

/// <summary>
/// C# representation of the Rust aethervk_core_rlib::scene::ParticleSystemComputedDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct ParticleSystemComputedDTO
{
  public readonly float Beta;
  public readonly float DustProductionRateAt1AuKgs;
}

/// <summary>
/// C# representation of the Rust aethervk_core_cdylib::ffi::FfiScreenSpaceBillboardDTO
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public readonly struct FfiScreenSpaceBillboardDTO
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
/// identifier
/// </summary>
public enum ExternalStateType : uint
{
  TimeRange = 1,
  ModelImported = 2,
  AlmanacImported = 3
}

/// <summary>
/// C# representation of aethervk_core_rlib::simulation_api::external_state::ExternalState
/// `CAlmanacImported` arm
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public unsafe struct CAlmanacImportedDTO
{
  public uint WasSuccessful; // 0 = no, !=0 = yes
  public fixed byte PathBytes[32];

  // Helper to extract the string cleanly
  public string GetPath()
  {
    fixed (byte* p = PathBytes)
    {
      // find null terminator (Assuming UTF-8 from Rust)
      int len = 0;
      while (len < 32 && p[len] != 0) len++;
      return Encoding.UTF8.GetString(p, len);
    }
  }
}

public delegate void PanicCallbackDelegate(IntPtr message, nuint length);
public delegate void LoggerCallbackDelegate(IntPtr message);
public delegate void BreadcrumbCallbackDelegate(uint level, IntPtr message);
public delegate void SimulationCallbackDelegate(ulong sceneId, ulong entityId, ulong componentId, IntPtr data);
public delegate void ExternalStateSimulationCallbackDelegate(uint stateId, IntPtr stateDto);
public delegate void RenderCallbackDelegate(ulong sceneId, ulong presentationEngineId, ulong taskId);
public delegate void MainThreadDispatchCallbackDelegate(IntPtr context);
