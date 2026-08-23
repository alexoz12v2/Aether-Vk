using System;
using System.Numerics;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using AetherVk.Logic.Input;

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
  double PosX,
  double PosY,
  double PosZ,
  float RotX,
  float RotY,
  float RotZ,
  float RotW
);

/// <summary>
/// Immutable snapshot of the camera projection state, emitted via <c>SIMULATION_CALLBACK</c>.
/// </summary>
public sealed record CameraProjectionState(
  bool IsPerspective,
  float Fov,
  float Aspect,
  float Near,
  float Far,
  float Left,
  float Right,
  float Bottom,
  float Top,
  float FocusDistance
);

/// <summary>
/// Validates camera movement commands against the current <see cref="CameraMode"/>, submits
/// approved commands to <see cref="INativeRuntimeService"/>, and exposes the runtime's
/// authoritative camera state as <see cref="System.Reactive"/> observables.
///
/// <para>Also manages the <b>comet orbit</b> camera mode: while in
/// <see cref="CameraMode.CometOrbiting"/> and the simulation is running, this service
/// subscribes to <see cref="CometPositionTrackerService.CometPositionRaw"/> and
/// automatically re-issues camera animation commands to keep the camera
/// at the configured orbit offset.</para>
///
/// <para>Also owns all interactive movement math (orbit, pan, zoom) and sensitivity constants.
/// Transient input operators (OrbitCameraOperator etc.) call <c>Request*</c> methods here;
/// they never touch the native runtime directly.</para>
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

  private readonly BehaviorSubject<CameraMode> _modeSubject = new(CameraMode.UpZenith);
  private IDisposable? _transformListenerToken;
  private IDisposable? _projectionListenerToken;
  private IDisposable? _cometOrbitSubscription;
  private IDisposable? _earthListenerToken;

  // Orbit offset in simulation units — kept constant while in CometOrbiting mode
  private Vector3 _orbitOffset = new(0f, 0f, 5e-5f); // ~7500 km at 1 AU scale

  // Earth position cache — updated via SIMULATION_CALLBACK for the earth entity.
  // Initialised to 1 AU on +X as a safe fallback before the first callback fires.
  private Vector3 _lastEarthPos = new(1f, 0f, 0f);

  // Last authoritative transform confirmed by the runtime. Populated by HandleTransformCallback.
  // Read synchronously by Request* methods to compute new absolute target transforms.
  private CameraTransformState? _lastConfirmedTransform;

  // Animation durations
  private const float ModeSwitchAnimationSeconds = 2.5f;

  // Short duration so the animation is always in-flight when the next comet callback
  // arrives, allowing retarget() to produce smooth continuous orbit tracking.
  private const float OrbitTrackingAnimationSeconds = 0.4f;

  // ── Movement sensitivity ─────────────────────────────────────────────────────
  // All units in simulation scale (AU / radian per pixel of drag).
  // Shift modifier applies ShiftFactor for Blender-style fine control.
  private const float OrbitSensitivity = 0.005f; // rad/px
  private const float PanSensitivity = 1e-5f; // AU/px
  private const float ZoomSensitivity = 2e-5f; // AU/px (vertical drag)
  private const float ShiftFactor = 0.2f; // fine-control multiplier

  // ── Comet orbit zoom limits ──────────────────────────────────────────────────
  // Min/max distance (AU) from the comet nucleus when zooming in CometOrbiting mode.
  private const float OrbitMinDistance = 1e-6f; // ~150 km — hard stop before nucleus surface
  private const float OrbitMaxDistance = 1e-2f; // ~1.5 million km — wide view

  public CameraService(
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider,
    CometPositionTrackerService cometTracker
  )
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

  /// <summary>
  /// Emits the new <see cref="CameraMode"/> every time <see cref="SetCameraMode"/> causes
  /// an actual transition. Observed on the main thread.
  /// </summary>
  public IObservable<CameraMode> CameraModeChanged =>
    _modeSubject.ObserveOn(_schedulerProvider.MainThread);

  /// <summary>Current camera mode.</summary>
  public CameraMode CurrentMode => _modeSubject.Value;

  /// <summary>
  /// Last authoritative camera transform confirmed by the runtime callback.
  /// Read synchronously by <c>Request*</c> methods to compute absolute transform targets.
  /// At most one frame stale between callback fires.
  /// </summary>
  public CameraTransformState? LastConfirmedTransform => _lastConfirmedTransform;

  // ── Mode management ────────────────────────────────────────────────────────

  /// <summary>
  /// Switches the active camera mode. Automatically starts/stops the comet orbit
  /// subscription when transitioning to/from <see cref="CameraMode.CometOrbiting"/>.
  /// For every mode change, an animated transition to the canonical starting position
  /// is triggered via <see cref="INativeRuntimeService.AddCameraAnimation"/>.
  /// </summary>
  public void SetCameraMode(CameraMode mode)
  {
    var previous = _modeSubject.Value;
    if (previous == mode)
      return;
    _modeSubject.OnNext(mode);

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

  /// <summary>Advance to the next camera mode (EarthPosition → UpZenith → CometOrbiting → EarthPosition).</summary>
  public void CycleCameraMode() =>
    SetCameraMode(
      _modeSubject.Value switch
      {
        CameraMode.EarthPosition => CameraMode.UpZenith,
        CameraMode.UpZenith => CameraMode.CometOrbiting,
        CameraMode.CometOrbiting => CameraMode.EarthPosition,
        _ => CameraMode.EarthPosition,
      }
    );

  /// <summary>
  /// Re-issues the canonical mode-default animation for the current mode.
  /// Called by <c>ViewportBaseOperator</c> via <c>Viewport3DViewModel.CameraService</c>
  /// when the user presses the Reset camera shortcut.
  /// </summary>
  public void ResetToModeDefault() => TriggerModeTransitionAnimation(_modeSubject.Value);

  /// <summary>Toggle between perspective and orthographic projection.</summary>
  public void ToggleProjection()
  {
    var proj = _projectionSubject.Value;
    if (proj is null)
      return;
    if (proj.IsPerspective)
      RequestOrthographicProjection(-proj.Aspect, proj.Aspect, -1f, 1f, proj.Near, proj.Far);
    else
      RequestPerspectiveProjection(proj.Fov, proj.Aspect, proj.Near, proj.Far);
  }

  // ── Mode-gate predicates (public — also checked by ViewportBaseOperator at push time) ────

  /// <summary>True when the current mode permits orbit (rotation around a pivot).</summary>
  public bool IsOrbitAllowed() =>
    _modeSubject.Value switch
    {
      CameraMode.EarthPosition => true,
      CameraMode.CometOrbiting => false, // comet-tracking animation manages position
      CameraMode.UpZenith => false,
      _ => false,
    };

  /// <summary>Pan (lateral translate) is permitted in all modes.</summary>
  public bool IsPanAllowed() => true;

  /// <summary>True when the current mode permits dolly zoom.</summary>
  public bool IsZoomAllowed() =>
    _modeSubject.Value switch
    {
      CameraMode.EarthPosition => true,
      CameraMode.CometOrbiting => true,
      CameraMode.UpZenith => false,
      _ => false,
    };

  // ── Mode transition animation ──────────────────────────────────────────────

  internal void TriggerModeTransitionAnimation(CameraMode mode)
  {
    ulong? camId = _runtimeService.CameraEntityId;
    if (camId is null)
      return;

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
      new AnimationTarget(targetPos, targetRot, ModeSwitchAnimationSeconds)
    );
  }

  /// <summary>
  /// Builds a rotation quaternion that orients the camera to look toward the world
  /// origin from <paramref name="pos"/>. Falls back to +X as world-up when the
  /// position is nearly on the Y axis.
  /// </summary>
  private static Quaternion LookAtOriginFrom(Vector3 pos)
  {
    var forward = Vector3.Normalize(-pos); // toward origin
    var up = Math.Abs(forward.Y) < 0.99f ? Vector3.UnitY : Vector3.UnitX;
    var right = Vector3.Normalize(Vector3.Cross(up, forward));
    up = Vector3.Cross(forward, right);
#pragma warning disable format
    return Quaternion.CreateFromRotationMatrix(
      // csharpier-ignore-start
      new Matrix4x4(
        right.X,   right.Y,   right.Z,   0,
        up.X,      up.Y,      up.Z,      0,
        forward.X, forward.Y, forward.Z, 0,
        0,         0,         0,         1
      )
    );
#pragma warning restore format
    // csharpier-ignore-end
  }

  // ── Interactive movement (called by transient camera operators) ────────────

  /// <summary>
  /// Orbit the camera around <paramref name="pivotWorld"/> by a screen-space pixel delta.
  /// Applies <see cref="OrbitSensitivity"/> (halved when Shift held).
  /// No-op if <see cref="IsOrbitAllowed"/> returns false or no transform is cached yet.
  /// </summary>
  public bool RequestOrbit(Vector2 pixelDelta, InputModifiers mods, Vector3 pivotWorld)
  {
    if (!IsOrbitAllowed())
      return false;
    var last = _lastConfirmedTransform;
    if (last is null)
      return false;

    float sens = mods.HasFlag(InputModifiers.Shift)
      ? OrbitSensitivity * ShiftFactor
      : OrbitSensitivity;
    float yawRad = -pixelDelta.X * sens;
    float pitchRad = -pixelDelta.Y * sens;

    var camPos = new Vector3((float)last.PosX, (float)last.PosY, (float)last.PosZ);
    var camRot = new Quaternion(last.RotX, last.RotY, last.RotZ, last.RotW);

    var yaw = Quaternion.CreateFromAxisAngle(Vector3.UnitY, yawRad);
    var right = Vector3.Transform(Vector3.UnitX, camRot);
    var pitch = Quaternion.CreateFromAxisAngle(right, pitchRad);
    var rot = pitch * yaw;

    var newPos = pivotWorld + Vector3.Transform(camPos - pivotWorld, rot);
    var newRot = Quaternion.Normalize(camRot * rot);

    bool ok = RotoTranslateDirect(newPos, newRot);
    if (ok)
    {
      _lastConfirmedTransform = new CameraTransformState(
        newPos.X, newPos.Y, newPos.Z,
        newRot.X, newRot.Y, newRot.Z, newRot.W
      );
    }
    return ok;
  }

  /// <summary>
  /// Pan the camera (translate on the view-right / view-up plane).
  /// Rotation is held fixed. Allowed in all modes including <see cref="CameraMode.UpZenith"/>.
  /// </summary>
  public bool RequestPan(Vector2 pixelDelta, InputModifiers mods)
  {
    var last = _lastConfirmedTransform;
    if (last is null)
      return false;

    float sens = mods.HasFlag(InputModifiers.Shift) ? PanSensitivity * ShiftFactor : PanSensitivity;

    var camRot = new Quaternion(last.RotX, last.RotY, last.RotZ, last.RotW);
    var right = Vector3.Transform(Vector3.UnitX, camRot);
    var up = Vector3.Transform(Vector3.UnitY, camRot);
    var worldD = (-right * pixelDelta.X + up * pixelDelta.Y) * sens;

    var newPos = new Vector3(
      (float)last.PosX + worldD.X,
      (float)last.PosY + worldD.Y,
      (float)last.PosZ + worldD.Z
    );

    // Pan bypasses mode gating — structurally always permitted.
    bool ok = RotoTranslateDirect(newPos, camRot);
    if (ok)
    {
      _lastConfirmedTransform = new CameraTransformState(
        newPos.X, newPos.Y, newPos.Z,
        camRot.X, camRot.Y, camRot.Z, camRot.W
      );
    }
    return ok;
  }

  /// <summary>
  /// Dolly zoom along the camera's forward axis.
  /// Positive <paramref name="pixelDy"/> = drag downward = zoom in.
  ///
  /// <para>In <see cref="CameraMode.CometOrbiting"/>: the tracking animation is always
  /// active so <c>CameraSetRotoTranslate</c> would be rejected. Instead mutates
  /// <c>_orbitOffset</c> magnitude (clamped); the change is applied by the next
  /// <see cref="SnapCameraToOrbit"/> tick (≤16 ms).</para>
  ///
  /// No-op if <see cref="IsZoomAllowed"/> returns false.
  /// </summary>
  public bool RequestZoom(float pixelDy, InputModifiers mods)
  {
    if (!IsZoomAllowed())
      return false;

    float sens = mods.HasFlag(InputModifiers.Shift)
      ? ZoomSensitivity * ShiftFactor
      : ZoomSensitivity;

    if (_modeSubject.Value == CameraMode.CometOrbiting)
    {
      // Adjust orbit radius — picked up by the next SnapCameraToOrbit tick.
      float scaleFactor = 1f + pixelDy * sens;
      float currentLen = _orbitOffset.Length();
      var newLen = currentLen * scaleFactor;
      if (newLen < OrbitMinDistance)
        newLen = OrbitMinDistance;
      if (newLen > OrbitMaxDistance)
        newLen = OrbitMaxDistance;
      if (currentLen > 1e-10f)
        _orbitOffset = Vector3.Normalize(_orbitOffset) * newLen;
      return true; // orbit subscription will apply it
    }

    // EarthPosition / UpZenith — direct immediate transform.
    var last = _lastConfirmedTransform;
    if (last is null)
      return false;

    var camRot = new Quaternion(last.RotX, last.RotY, last.RotZ, last.RotW);
    var camForward = Vector3.Transform(-Vector3.UnitZ, camRot); // -Z = forward (Vulkan convention)
    var newPos = new Vector3(
      (float)last.PosX + camForward.X * pixelDy * sens,
      (float)last.PosY + camForward.Y * pixelDy * sens,
      (float)last.PosZ + camForward.Z * pixelDy * sens
    );

    bool ok = RotoTranslateDirect(newPos, camRot);
    if (ok)
    {
      _lastConfirmedTransform = new CameraTransformState(
        newPos.X, newPos.Y, newPos.Z,
        camRot.X, camRot.Y, camRot.Z, camRot.W
      );
    }
    return ok;
  }

  // ── Projection commands ────────────────────────────────────────────────────

  /// <summary>
  /// Request a perspective projection change. Not mode-gated.
  /// </summary>
  public bool RequestPerspectiveProjection(float fov, float aspectRatio, float near, float far) =>
    _runtimeService.CameraSetPerspective(
      _runtimeService.CameraEntityId ?? 0,
      fov,
      aspectRatio,
      near,
      far
    );

  /// <summary>
  /// Request an orthographic projection change. Not mode-gated.
  /// </summary>
  public bool RequestOrthographicProjection(
    float left,
    float right,
    float bottom,
    float top,
    float near,
    float far
  ) =>
    _runtimeService.CameraSetOrthographic(
      _runtimeService.CameraEntityId ?? 0,
      left,
      right,
      bottom,
      top,
      near,
      far
    );

  // ── Legacy roto-translate (kept for external callers not yet migrated) ─────

  /// <summary>
  /// Request a camera roto-translation. Validated against current mode constraints
  /// using the coarse <c>IsMoveAllowed()</c> predicate.
  /// Prefer <see cref="RequestOrbit"/>, <see cref="RequestPan"/>, <see cref="RequestZoom"/>
  /// for interactive use — they apply the correct per-operation gate.
  /// </summary>
  /// <returns><c>true</c> if the command was accepted by the runtime.</returns>
  public bool RequestRotoTranslate(Vector3 position, Quaternion rotation)
  {
    if (!IsMoveAllowed())
      return false;
    return RotoTranslateDirect(position, rotation);
  }

  // ── Internal helpers ───────────────────────────────────────────────────────

  /// <summary>
  /// Apply a validated absolute roto-translate via the typed runtime interface.
  /// All unsafe buffer packing lives in <c>NativeRuntimeService.CameraSetRotoTranslate</c>.
  /// </summary>
  private bool RotoTranslateDirect(Vector3 position, Quaternion rotation) =>
    _runtimeService.CameraSetRotoTranslate(_runtimeService.CameraEntityId ?? 0, position, rotation);

  /// <summary>Coarse move-allowed predicate used by the legacy <see cref="RequestRotoTranslate"/>.</summary>
  private bool IsMoveAllowed() =>
    _modeSubject.Value switch
    {
      // EarthPosition: roto-translate allowed
      // CometOrbiting: managed internally by orbit tracking — reject external roto-translate
      // UpZenith: pan only (via RequestPan which bypasses this gate)
      CameraMode.EarthPosition => true,
      CameraMode.CometOrbiting => false,
      CameraMode.UpZenith => false,
      _ => false,
    };

  // ── Comet orbit tracking ───────────────────────────────────────────────────

  private void StartCometOrbitTracking()
  {
    StopCometOrbitTracking(); // ensure only one subscription at a time
    _cometOrbitSubscription = _cometTracker
      .CometPositionRaw.Where(static pos => pos.HasValue)
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
    if (camId is null)
      return;

    var targetPos = cometPos + _orbitOffset;
    // Look toward the comet nucleus
    var forward = Vector3.Normalize(cometPos - targetPos);
    var up = Vector3.UnitY;
    var right = Vector3.Normalize(Vector3.Cross(up, forward));
    up = Vector3.Cross(forward, right);
    var rot = Quaternion.CreateFromRotationMatrix(
      new Matrix4x4(
        right.X,
        right.Y,
        right.Z,
        0,
        up.X,
        up.Y,
        up.Z,
        0,
        forward.X,
        forward.Y,
        forward.Z,
        0,
        0,
        0,
        0,
        1
      )
    );

    // Use AddCameraAnimation instead of CameraSetRotoTranslate.
    // The short duration keeps a TransformAnimationComponent always in-flight on
    // the camera entity.  The next comet callback (~16 ms later) calls this again,
    // which hits retarget() on the Rust side — producing smooth continuous tracking
    // without any extra Rust constructs or FFI changes.
    _runtimeService.AddCameraAnimation(
      camId.Value,
      new AnimationTarget(targetPos, rot, OrbitTrackingAnimationSeconds)
    );
  }

  // ── Simulation listener registration ──────────────────────────────────────

  /// <summary>
  /// Called by <see cref="Viewport3DViewModel.OnViewportCreated"/> once
  /// <see cref="INativeRuntimeService.AddViewport"/> has succeeded and
  /// <see cref="INativeRuntimeService.CameraEntityId"/> is populated.
  /// Re-attempts simulation listener registration that was skipped in the
  /// constructor because the camera entity did not yet exist.
  /// </summary>
  public void OnViewportReady()
  {
    RegisterSimListeners();
    RegisterEarthListener();
    // Trigger initial mode animation so the runtime assigns a valid transform
    // and fires the SIMULATION_CALLBACK to populate _lastConfirmedTransform.
    TriggerModeTransitionAnimation(_modeSubject.Value);
  }

  private void RegisterSimListeners()
  {
    ulong? camId = _runtimeService.CameraEntityId;
    if (camId is null)
    {
      // CameraEntityId is populated after AddViewport — register lazily on first use.
      // TODO: expose an event or observable from runtime service to retry registration.
      return;
    }

    // Already registered — prevent double-registration if OnViewportReady is called twice.
    if (_transformListenerToken is not null)
      return;

    _transformListenerToken = _runtimeService.RegisterSimulationListener(
      camId.Value,
      ComponentForeignId.HighResTransform,
      HandleTransformCallback
    );

    _projectionListenerToken = _runtimeService.RegisterSimulationListener(
      camId.Value,
      ComponentForeignId.CameraProjection,
      HandleProjectionCallback
    );
  }

  private void RegisterEarthListener()
  {
    ulong? earthId = _runtimeService.EarthEntityId;
    if (earthId is null)
      return;

    // Already registered — prevent double-registration.
    if (_earthListenerToken is not null)
      return;

    _earthListenerToken = _runtimeService.RegisterSimulationListener(
      earthId.Value,
      ComponentForeignId.HighResTransform,
      HandleEarthTransformCallback
    );
  }

  // ── Internal callback handling ─────────────────────────────────────────────

  private unsafe void HandleTransformCallback(nint dataPtr)
  {
    var dto = *(HighResTransformDTO*)dataPtr;
    var state = new CameraTransformState(
      dto.PosX,
      dto.PosY,
      dto.PosZ,
      dto.RotW,
      dto.RotX,
      dto.RotY,
      dto.RotZ
    );
    _lastConfirmedTransform = state;
    _transformSubject.OnNext(state);
  }

  private unsafe void HandleProjectionCallback(nint dataPtr)
  {
    var dto = *(CameraProjectionDTO*)dataPtr;
    _projectionSubject.OnNext(
      new CameraProjectionState(
        IsPerspective: dto.IsOrthographic == 0,
        dto.Fov,
        dto.Aspect,
        dto.Near,
        dto.Far,
        dto.Left,
        dto.Right,
        dto.Bottom,
        dto.Top,
        dto.FocusDistance
      )
    );
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
    _modeSubject.Dispose();
  }
}
