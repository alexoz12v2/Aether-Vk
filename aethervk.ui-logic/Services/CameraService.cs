
using System;
using System.Numerics;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

/// <summary>
/// Current camera mode, governing which transform operations are allowed.
/// </summary>
public enum CameraMode
{
  /// Camera locked to Earth's trajectory. Zoom allowed; pan locked; rotation changes orientation only.
  EarthPosition,
  /// Snap-to-zenith mode (derived from EarthPosition). Pan only; rotation and zoom locked.
  UpZenith,
  /// Camera orbits the comet centre-of-mass. Zoom allowed (with limits); pan locked;
  /// rotation focuses on comet. Camera automatically tracks comet position when simulation is running.
  CometOrbiting,
}

/// <summary>
/// Immutable snapshot of the camera transform state, emitted via <c>SIMULATION_CALLBACK</c>.
/// </summary>
public sealed record CameraTransformState(
  double PosX, double PosY, double PosZ,
  float RotX, float RotY, float RotZ, float RotW);

/// <summary>
/// Immutable snapshot of the camera projection state, emitted via <c>SIMULATION_CALLBACK</c>.
/// </summary>
public sealed record CameraProjectionState(
  bool IsPerspective,
  float Fov, float Aspect, float Near, float Far,
  float Left, float Right, float Bottom, float Top,
  float FocusDistance);

/// <summary>
/// Validates camera movement commands against the current <see cref="CameraMode"/>, submits
/// approved commands to <see cref="INativeRuntimeService"/>, and exposes the runtime's
/// authoritative camera state as <see cref="System.Reactive"/> observables.
///
/// <para>Also manages the <b>comet orbit</b> camera mode: while in
/// <see cref="CameraMode.CometOrbiting"/> and the simulation is running, this service
/// subscribes to <see cref="CometPositionTrackerService.CometPositionRaw"/> and
/// automatically re-issues <c>TransformStaticCamera</c> commands to keep the camera
/// at the configured orbit offset.</para>
///
/// - part of the "Companion Runtime Service" group
/// </summary>
/// <seealso cref="CometPositionTrackerService" />
/// <seealso cref="TimelineService" />
/// <seealso cref="ImportedModelsTrackerService" />
public sealed class CameraService : IDisposable
{
  private readonly INativeRuntimeService _runtimeService;
  private readonly ISchedulerProvider _schedulerProvider;
  private readonly CometPositionTrackerService _cometTracker;

  private readonly BehaviorSubject<CameraTransformState?> _transformSubject = new(null);
  private readonly BehaviorSubject<CameraProjectionState?> _projectionSubject = new(null);

  private CameraMode _mode = CameraMode.EarthPosition;
  private IDisposable? _transformListenerToken;
  private IDisposable? _projectionListenerToken;
  private IDisposable? _cometOrbitSubscription;
  private IDisposable? _earthListenerToken;

  // Orbit offset in simulation units — kept constant while in CometOrbiting mode
  private Vector3 _orbitOffset = new(0f, 0f, 5e-5f); // ~7500 km at 1 AU scale

  // Earth position cache — updated via SIMULATION_CALLBACK for the earth entity.
  // Initialised to 1 AU on +X as a safe fallback before the first callback fires.
  private Vector3 _lastEarthPos = new(1f, 0f, 0f);

  // Animation durations
  private const float ModeSwitchAnimationSeconds = 2.5f;
  // Short duration so the animation is always in-flight when the next comet callback
  // arrives, allowing retarget() to produce smooth continuous orbit tracking.
  private const float OrbitTrackingAnimationSeconds = 0.4f;

  public CameraService(
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider,
    CometPositionTrackerService cometTracker)
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    _cometTracker = cometTracker;

    RegisterSimListeners();
    RegisterEarthListener();
  }

  // ── Observables ────────────────────────────────────────────────────────────

  /// <summary>
  /// Authoritative camera transform state (position + rotation) as confirmed by the runtime.
  /// Scheduler is the caller's choice — default is main-thread.
  /// </summary>
  public IObservable<CameraTransformState?> CameraTransform =>
    _transformSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>
  /// Authoritative camera projection state as confirmed by the runtime.
  /// Scheduler is the caller's choice — default is main-thread.
  /// </summary>
  public IObservable<CameraProjectionState?> CameraProjection =>
    _projectionSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>Current camera mode.</summary>
  public CameraMode CurrentMode => _mode;

  // ── Mode management ────────────────────────────────────────────────────────

  /// <summary>
  /// Switches the active camera mode. Automatically starts/stops the comet orbit
  /// subscription when transitioning to/from <see cref="CameraMode.CometOrbiting"/>.
  /// For every mode change, an animated transition to the canonical starting position
  /// is triggered via <see cref="INativeRuntimeService.AddCameraAnimation"/>.
  /// </summary>
  public void SetCameraMode(CameraMode mode)
  {
    if (_mode == mode) return;
    var previous = _mode;
    _mode = mode;

    if (mode == CameraMode.CometOrbiting)
      StartCometOrbitTracking();
    else if (previous == CameraMode.CometOrbiting)
      StopCometOrbitTracking();

    TriggerModeTransitionAnimation(mode);
  }

  /// <summary>
  /// Sets the orbit offset used when <see cref="CurrentMode"/> is
  /// <see cref="CameraMode.CometOrbiting"/>.
  /// </summary>
  public void SetOrbitOffset(Vector3 offset) => _orbitOffset = offset;

  // ── Mode transition animation ──────────────────────────────────────────────

  private void TriggerModeTransitionAnimation(CameraMode mode)
  {
    ulong? camId = _runtimeService.CameraEntityId;
    if (camId is null) return;

    Vector3 targetPos;
    Quaternion targetRot;

    switch (mode)
    {
      case CameraMode.UpZenith:
        // 1 AU above the Sun (origin), looking inward.
        // 1 AU == 1.0 in engine simulation units.
        targetPos = new Vector3(0f, 0f, 1f);
        targetRot = LookAtOriginFrom(targetPos);
        break;

      case CameraMode.EarthPosition:
        // 2 × Earth radius standoff from the Earth body centre.
        // Earth radius ≈ 6 371 km ≈ 4.26e-5 AU in engine units.
        const float EarthRadiusAu = 4.26e-5f;
        targetPos = _lastEarthPos + new Vector3(0f, 0f, 2f * EarthRadiusAu);
        targetRot = LookAtOriginFrom(targetPos);
        break;

      case CameraMode.CometOrbiting:
        // The comet orbit subscription takes over the moment it fires.
        // SnapCameraToOrbit will issue the first AddCameraAnimation on the next
        // comet position callback — no separate initial animation needed here.
        return;

      default:
        return;
    }

    _runtimeService.AddCameraAnimation(
      camId.Value,
      new AnimationTarget(targetPos, targetRot, ModeSwitchAnimationSeconds));
  }

  /// <summary>
  /// Builds a rotation quaternion that orients the camera to look toward the world
  /// origin from <paramref name="pos"/>. Falls back to +X as world-up when the
  /// position is nearly on the Y axis.
  /// </summary>
  private static Quaternion LookAtOriginFrom(Vector3 pos)
  {
    var forward = Vector3.Normalize(-pos);                                // toward origin
    var up = Math.Abs(forward.Y) < 0.99f ? Vector3.UnitY : Vector3.UnitX;
    var right = Vector3.Normalize(Vector3.Cross(up, forward));
    up = Vector3.Cross(forward, right);
    return Quaternion.CreateFromRotationMatrix(new Matrix4x4(
      right.X, right.Y, right.Z, 0,
      up.X, up.Y, up.Z, 0,
      forward.X, forward.Y, forward.Z, 0,
      0, 0, 0, 1));
  }

  // ── Camera commands ────────────────────────────────────────────────────────

  /// <summary>
  /// Request a camera roto-translation. Validated against current mode constraints.
  /// Visual parameter changes (projection) bypass mode validation.
  /// </summary>
  /// <returns><c>true</c> if the command was accepted by the runtime.</returns>
  public unsafe bool RequestRotoTranslate(Vector3 position, Quaternion rotation)
  {
    if (!IsMoveAllowed()) return false;

    // Build [f32; 7] buffer: dispX, dispY, dispZ, quatX, quatY, quatZ, quatW
    float* buf = stackalloc float[7]
    {
      position.X, position.Y, position.Z,
      rotation.X, rotation.Y, rotation.Z, rotation.W,
    };

    return _runtimeService.TransformStaticCamera(
      _runtimeService.CameraEntityId ?? 0,
      mode: 2,
      buffer: (nint)buf);
  }

  /// <summary>
  /// Request a perspective projection change. Not mode-gated.
  /// </summary>
  public unsafe bool RequestPerspectiveProjection(float fov, float aspectRatio, float near, float far)
  {
    float* buf = stackalloc float[4] { fov, aspectRatio, near, far };
    return _runtimeService.TransformStaticCamera(
      _runtimeService.CameraEntityId ?? 0,
      mode: 1,
      buffer: (nint)buf);
  }

  /// <summary>
  /// Request an orthographic projection change. Not mode-gated.
  /// </summary>
  public unsafe bool RequestOrthographicProjection(
    float left, float right, float bottom, float top, float near, float far)
  {
    float* buf = stackalloc float[6] { left, right, bottom, top, near, far };
    return _runtimeService.TransformStaticCamera(
      _runtimeService.CameraEntityId ?? 0,
      mode: 0,
      buffer: (nint)buf);
  }

  // ── Mode validation ────────────────────────────────────────────────────────

  private bool IsMoveAllowed() => _mode switch
  {
    // EarthPosition: rotation + zoom OK, pan locked
    // CometOrbiting: managed internally by orbit tracking — direct moves are rejected
    // UpZenith: pan only (handled separately), rotation + zoom locked
    CameraMode.EarthPosition => true,  // roto-translate allowed (zoom = change in Z translation)
    CameraMode.CometOrbiting => false, // orbit tracking manages this, reject external roto-translate
    CameraMode.UpZenith => false,
    _ => false,
  };

  // ── Comet orbit tracking ───────────────────────────────────────────────────

  private void StartCometOrbitTracking()
  {
    StopCometOrbitTracking(); // ensure only one subscription at a time
    _cometOrbitSubscription = _cometTracker.CometPositionRaw
      .Where(static pos => pos.HasValue)
      .Subscribe(pos => SnapCameraToOrbit(pos!.Value));
  }

  private void StopCometOrbitTracking()
  {
    _cometOrbitSubscription?.Dispose();
    _cometOrbitSubscription = null;
  }

  private void SnapCameraToOrbit(Vector3 cometPos)
  {
    ulong? camId = _runtimeService.CameraEntityId;
    if (camId is null) return;

    var targetPos = cometPos + _orbitOffset;
    // Look toward the comet nucleus
    var forward = Vector3.Normalize(cometPos - targetPos);
    var up = Vector3.UnitY;
    var right = Vector3.Normalize(Vector3.Cross(up, forward));
    up = Vector3.Cross(forward, right);
    var rot = Quaternion.CreateFromRotationMatrix(new Matrix4x4(
      right.X, right.Y, right.Z, 0,
      up.X, up.Y, up.Z, 0,
      forward.X, forward.Y, forward.Z, 0,
      0, 0, 0, 1));

    // Use AddCameraAnimation instead of TransformStaticCamera.
    // The short duration keeps a TransformAnimationComponent always in-flight on
    // the camera entity.  The next comet callback (~16 ms later) calls this again,
    // which hits retarget() on the Rust side — producing smooth continuous tracking
    // without any extra Rust constructs or FFI changes.
    _runtimeService.AddCameraAnimation(
      camId.Value,
      new AnimationTarget(targetPos, rot, OrbitTrackingAnimationSeconds));
  }

  // ── Simulation listener registration ──────────────────────────────────────

  private void RegisterSimListeners()
  {
    ulong? camId = _runtimeService.CameraEntityId;
    if (camId is null)
    {
      // CameraEntityId is populated after AddViewport — register lazily on first use.
      // TODO: expose an event or observable from runtime service to retry registration.
      return;
    }

    _transformListenerToken = _runtimeService.RegisterSimulationListener(
      camId.Value,
      ComponentForeignId.HighResTransform,
      HandleTransformCallback);

    _projectionListenerToken = _runtimeService.RegisterSimulationListener(
      camId.Value,
      ComponentForeignId.CameraProjection,
      HandleProjectionCallback);
  }

  private void RegisterEarthListener()
  {
    ulong? earthId = _runtimeService.EarthEntityId;
    if (earthId is null) return;

    _earthListenerToken = _runtimeService.RegisterSimulationListener(
      earthId.Value,
      ComponentForeignId.HighResTransform,
      HandleEarthTransformCallback);
  }

  // ── Internal callback handling ─────────────────────────────────────────────

  private unsafe void HandleTransformCallback(nint dataPtr)
  {
    var dto = *(HighResTransformDTO*)dataPtr;
    _transformSubject.OnNext(new CameraTransformState(
      dto.PosX, dto.PosY, dto.PosZ,
      dto.RotW, dto.RotX, dto.RotY, dto.RotZ));
  }

  private unsafe void HandleProjectionCallback(nint dataPtr)
  {
    var dto = *(CameraProjectionDTO*)dataPtr;
    _projectionSubject.OnNext(new CameraProjectionState(
      IsPerspective: dto.IsOrthographic == 0,
      dto.Fov, dto.Aspect, dto.Near, dto.Far,
      dto.Left, dto.Right, dto.Bottom, dto.Top,
      dto.FocusDistance));
  }

  private unsafe void HandleEarthTransformCallback(nint dataPtr)
  {
    var dto = *(HighResTransformDTO*)dataPtr;
    // f32 is sufficient precision for mode-switch animation targets.
    _lastEarthPos = new Vector3((float)dto.PosX, (float)dto.PosY, (float)dto.PosZ);
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    StopCometOrbitTracking();
    _transformListenerToken?.Dispose();
    _projectionListenerToken?.Dispose();
    _earthListenerToken?.Dispose();
    _transformSubject.Dispose();
    _projectionSubject.Dispose();
  }
}
