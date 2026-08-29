using System;
using System.Collections.Generic;
using System.Numerics;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
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
  private readonly CometConfigService _cometConfigService;
  private readonly BreadcrumbService _breadcrumbService;

  private readonly BehaviorSubject<CameraTransformState?> _transformSubject = new(null);
  private readonly BehaviorSubject<CameraProjectionState?> _projectionSubject = new(null);

  private readonly BehaviorSubject<CameraMode> _modeSubject = new(CameraMode.UpZenith);
  private IDisposable? _transformListenerToken;
  private IDisposable? _projectionListenerToken;
  private IDisposable? _cometOrbitSubscription;
  private IDisposable? _earthListenerToken;

  // Orbit offset in simulation units — kept constant while in CometOrbiting mode
  private Vector3 _orbitOffset = new(0f, 0f, 5e-5f); // ~7500 km at 1 AU scale
  private readonly object _orbitOffsetLock = new();

  // Earth position cache — updated via SIMULATION_CALLBACK for the earth entity.
  // Initialised to 1 AU on +X as a safe fallback before the first callback fires.
  private Vector3 _lastEarthPos = new(1f, 0f, 0f);

  // Lock protecting _lastEarthPos, _earthOffset, and _earthRotation.
  private readonly object _earthPosLock = new();

  private Vector3 _earthOffset;
  private Quaternion _earthRotation = Quaternion.Identity;

  // Last authoritative transform confirmed by the runtime. Populated by HandleTransformCallback.
  // Read synchronously by Request* methods to compute new absolute target transforms.
  private CameraTransformState? _lastConfirmedTransform;

  // ── Mode state memory ────────────────────────────────────────────────────────
  // Saved (transform, projection) per mode — restored via animation on re-entry.
  private sealed record ModeSnapshot(
    CameraTransformState Transform,
    CameraProjectionState? Projection
  );

  private readonly Dictionary<CameraMode, ModeSnapshot> _modeSnapshots = new();

  // Aspect ratio (W/H) of the Vulkan render target — set by OnViewportReady.
  private float _viewportAspect = 1f;

  // Cancels any pending deferred projection change when a new mode switch fires.
  private CancellationTokenSource? _pendingProjectionCts;

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
    CometPositionTrackerService cometTracker,
    CometConfigService cometConfigService,
    BreadcrumbService breadcrumbService
  )
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    _cometTracker = cometTracker;
    _cometConfigService = cometConfigService;
    _breadcrumbService = breadcrumbService;
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
  public ulong? CameraEntityId { get; private set; }

  /// <summary>
  /// Last authoritative camera transform confirmed by the runtime callback.
  /// Read synchronously by <c>Request*</c> methods to compute absolute transform targets.
  /// At most one frame stale between callback fires.
  /// </summary>
  public CameraTransformState? LastConfirmedTransform => _lastConfirmedTransform;

  /// <summary>Last authoritative projection state confirmed by the runtime callback.</summary>
  public CameraProjectionState? LastConfirmedProjection => _projectionSubject.Value;

  // ── Mode management ────────────────────────────────────────────────────────

  /// <summary>
  /// Switches the active camera mode. Automatically starts/stops the comet orbit
  /// subscription when transitioning to/from <see cref="CameraMode.CometOrbiting"/>.
  /// For every mode change, an animated transition to the canonical starting position
  /// is triggered via <see cref="INativeRuntimeService.AddCameraAnimation"/>.
  /// </summary>
  public void SetCameraMode(CameraMode mode)
  {
    // Guard: CometOrbiting requires a committed comet SPK.
    // Show a warning toast and return without changing mode.
    if (mode == CameraMode.CometOrbiting && !_cometConfigService.IsAlmanacCommittedValue)
    {
      _ = _breadcrumbService.ShowMessageAsync(
        "No Comet Configured",
        "Load and commit a comet SPK to enter Comet Orbiting mode.",
        TimeSpan.FromSeconds(4),
        status: 2 // Warning
      );
      return;
    }

    var previous = _modeSubject.Value;
    if (previous == mode)
      return;

    // Snapshot the outgoing mode's confirmed state for restoration on re-entry.
    SaveModeSnapshot(previous);

    // Cancel any projection change that was still pending from the previous transition.
    _pendingProjectionCts?.Cancel();
    _pendingProjectionCts = null;

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
  public void SetOrbitOffset(Vector3 offset)
  {
    lock (_orbitOffsetLock)
      _orbitOffset = offset;
  }

  /// <summary>Advance to the next camera mode (EarthPosition → UpZenith → CometOrbiting → EarthPosition).</summary>
  public void CycleCameraMode()
  {
    var next = _modeSubject.Value switch
    {
      CameraMode.EarthPosition => CameraMode.UpZenith,
      CameraMode.UpZenith => CameraMode.CometOrbiting,
      CameraMode.CometOrbiting => CameraMode.EarthPosition,
      _ => CameraMode.EarthPosition,
    };
    if (next == CameraMode.CometOrbiting && !_cometConfigService.IsAlmanacCommittedValue)
      next = CameraMode.EarthPosition;
    SetCameraMode(next);
  }

  /// <summary>
  /// Re-issues the canonical mode-default animation for the current mode.
  /// Clears any saved snapshot so the hard-coded first-entry default fires.
  /// Called by <c>ViewportBaseOperator</c> via <c>Viewport3DViewModel.CameraService</c>
  /// when the user presses the Reset camera shortcut.
  /// </summary>
  public void ResetToModeDefault()
  {
    // Discard saved state — next TriggerModeTransitionAnimation will use first-entry defaults.
    _modeSnapshots.Remove(_modeSubject.Value);
    // Cancel any in-flight deferred projection change.
    _pendingProjectionCts?.Cancel();
    _pendingProjectionCts = null;
    TriggerModeTransitionAnimation(_modeSubject.Value);
  }

  /// Capture the confirmed (transform, projection) for <paramref name="mode"/> so it can
  /// be restored next time that mode is entered.
  private void SaveModeSnapshot(CameraMode mode)
  {
    var t = _lastConfirmedTransform;
    if (t is null)
      return; // nothing confirmed yet — skip
    _modeSnapshots[mode] = new ModeSnapshot(t, _projectionSubject.Value);
  }

  public void ToggleProjection()
  {
    var proj = _projectionSubject.Value;
    if (proj is null)
      return;
    if (proj.IsPerspective)
    {
      float height = Math.Abs(proj.Top - proj.Bottom);
      if (height < 1e-5f) height = 2f; // Fallback if 0
      float width = height * _viewportAspect;
      RequestOrthographicProjection(-width / 2f, width / 2f, -height / 2f, height / 2f, proj.Near, proj.Far);
    }
    else
    {
      RequestPerspectiveProjection(proj.Fov, _viewportAspect, proj.Near, proj.Far);
    }
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

  /// <summary>Pan (lateral translate) is permitted only in <see cref="CameraMode.UpZenith"/>.</summary>
  public bool IsPanAllowed() =>
    _modeSubject.Value switch
    {
      CameraMode.UpZenith => true,
      CameraMode.EarthPosition => false, // orbit changes orientation only
      CameraMode.CometOrbiting => false, // tracking animation owns position
      _ => false,
    };

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

  internal void TriggerModeTransitionAnimation(CameraMode mode, bool snapImmediate = false)
  {
    ulong? camId = CameraEntityId;
    if (camId is null)
      return;

    Vector3 targetPos;
    Quaternion targetRot;

    // Projection to apply after the animation completes (null = no change).
    Action? deferredProjection = null;

    if (_modeSnapshots.TryGetValue(mode, out var snap))
    {
      // ── Restore saved state ──────────────────────────────────────────────
      var t = snap.Transform;
      targetPos = new Vector3((float)t.PosX, (float)t.PosY, (float)t.PosZ);
      targetRot = new Quaternion(t.RotX, t.RotY, t.RotZ, t.RotW);
      if (snap.Projection is { } savedProj)
        deferredProjection = () => ApplyProjectionSnapshot(savedProj);
    }
    else
    {
      // ── First-entry defaults ─────────────────────────────────────────────
      switch (mode)
      {
        case CameraMode.UpZenith:
          // 0.05 AU above the Sun (origin), looking inward.
          // Sun radius ≈ 0.00465 AU. To fill ~30% of the half-extent:
          //   halfH = sun_radius / 0.30 ≈ 0.0155 AU
          targetPos = new Vector3(0f, 0f, 0.05f);
          targetRot = LookAtOriginFrom(targetPos);
          deferredProjection = ApplyUpZenithDefaultProjection;
          break;

        case CameraMode.EarthPosition:
          // 2 × Earth radius standoff from the Earth body centre.
          // Earth radius ≈ 6 371 km ≈ 4.26e-5 AU in engine units.
          const float EarthRadiusAu = 4.26e-5f;
          lock (_earthPosLock)
          {
            _earthOffset = new Vector3(0f, 0f, 2f * EarthRadiusAu);
            _earthRotation = LookAtOriginFrom(_lastEarthPos + _earthOffset);
            targetPos = _lastEarthPos + _earthOffset;
            targetRot = _earthRotation;
          }
          break;

        case CameraMode.CometOrbiting:
          // The comet orbit subscription takes over the moment it fires.
          return;

        default:
          return;
      }
    }

    // ── Dispatch the transform animation ─────────────────────────────────
    // snapImmediate=true on the first viewport-ready call so the sun is
    // visible from frame 1, not after a 2.5 s ramp from the Rust default.
    float animDuration = snapImmediate ? 0.01f : ModeSwitchAnimationSeconds;
    _runtimeService.AddCameraAnimation(
      camId.Value,
      new AnimationTarget(targetPos, targetRot, animDuration)
    );

    // ── Defer the projection change until after the animation completes ────
    if (deferredProjection is not null)
    {
      var cts = new CancellationTokenSource();
      _pendingProjectionCts = cts;
      var targetMode = mode; // capture — prevents closure over a variable that may change
      float projDelay = snapImmediate ? 0.05f : ModeSwitchAnimationSeconds;
      _ = Task.Delay(TimeSpan.FromSeconds(projDelay), cts.Token)
        .ContinueWith(
          t =>
          {
            // Guard: if the mode changed again while we were waiting, discard the projection.
            if (t.IsCanceled || _modeSubject.Value != targetMode)
              return;
            // Dispatch to the main-thread scheduler — projection commands touch the runtime
            // and must not be called from a thread-pool continuation.
            _schedulerProvider.MainThread.Schedule(deferredProjection);
          },
          CancellationToken.None,
          TaskContinuationOptions.ExecuteSynchronously,
          TaskScheduler.Default
        );
    }
  }

  /// <summary>
  /// Builds a rotation quaternion compatible with the Rust engine that orients
  /// the camera to look toward the world origin from <paramref name="pos"/>.
  ///
  /// <para>Engine convention (vec4.rs <c>Quat::from_mat4</c>):
  /// column-major — col0 = right (+X), col1 = backward (+Y), col2 = up (+Z).
  /// Forward = local −Y. This function ports <c>from_mat4</c> directly so the
  /// XYZW bytes are interpreted identically on the Rust side.</para>
  /// <para>Falls back to +X as the world-up hint when the camera is nearly on the Z axis.</para>
  /// </summary>
  private static Quaternion LookAtOriginFrom(Vector3 pos)
  {
    var worldFwd = Vector3.Normalize(-pos); // toward origin (engine −Y)

    // World-up hint: prefer +Z; fall back to -Y when nearly on the Z axis to maintain +X right vector.
    var worldUpHint = Math.Abs(worldFwd.Z) < 0.99f ? Vector3.UnitZ : -Vector3.UnitY;

    // right = cross(upHint, fwd) — NOT cross(fwd, upHint).
    // Verified by hand: for fwd=(0,0,-1) + hint=(1,0,0) this gives right=(0,1,0),
    // up=(1,0,0), q=(0.5,0.5,0.5,0.5), q.rotate(0,-1,0)=(0,0,-1) → pitch=−90° ✓.
    var worldRight = Vector3.Normalize(Vector3.Cross(worldUpHint, worldFwd));
    var worldUp = Vector3.Cross(worldFwd, worldRight);

    return EngineQuatFromBasis(worldRight, -worldFwd, worldUp);
  }

  /// <summary>
  /// Ports <c>Quat::from_mat4</c> (vec4.rs) verbatim. Builds the engine-compatible
  /// quaternion from three orthonormal world-space basis vectors:
  /// <paramref name="right"/> → col0, <paramref name="backward"/> → col1,
  /// <paramref name="up"/> → col2.
  /// </summary>
  private static Quaternion EngineQuatFromBasis(Vector3 right, Vector3 backward, Vector3 up)
  {
    // m[row][col] — column-major 3×3: col0=right, col1=backward, col2=up.
    float m00 = right.X,
      m01 = backward.X,
      m02 = up.X;
    float m10 = right.Y,
      m11 = backward.Y,
      m12 = up.Y;
    float m20 = right.Z,
      m21 = backward.Z,
      m22 = up.Z;

    float trace = m00 + m11 + m22;
    float x,
      y,
      z,
      w;

    if (trace > 0f)
    {
      float s = (float)Math.Sqrt(trace + 1f) * 2f; // s = 4w
      float invS = 1f / s;
      x = (m21 - m12) * invS;
      y = (m02 - m20) * invS;
      z = (m10 - m01) * invS;
      w = 0.25f * s;
    }
    else if (m00 > m11 && m00 > m22)
    {
      float s = (float)Math.Sqrt(1f + m00 - m11 - m22) * 2f; // s = 4x
      float invS = 1f / s;
      x = 0.25f * s;
      y = (m01 + m10) * invS;
      z = (m02 + m20) * invS;
      w = (m21 - m12) * invS;
    }
    else if (m11 > m22)
    {
      float s = (float)Math.Sqrt(1f + m11 - m00 - m22) * 2f; // s = 4y
      float invS = 1f / s;
      x = (m01 + m10) * invS;
      y = 0.25f * s;
      z = (m12 + m21) * invS;
      w = (m02 - m20) * invS;
    }
    else
    {
      float s = (float)Math.Sqrt(1f + m22 - m00 - m11) * 2f; // s = 4z
      float invS = 1f / s;
      x = (m02 + m20) * invS;
      y = (m12 + m21) * invS;
      z = 0.25f * s;
      w = (m10 - m01) * invS;
    }

    return new Quaternion(x, y, z, w);
  }

  // ── Interactive movement (called by transient camera operators) ────────────

  /// <summary>
  /// Orbit the camera around <paramref name="pivotWorld"/> by a screen-space pixel delta.
  /// Applies <see cref="OrbitSensitivity"/> (halved when Shift held).
  /// No-op if <see cref="IsOrbitAllowed"/> returns false or no transform is cached yet.
  /// </summary>
  public bool RequestOrbit(Vector2 pixelDelta, InputModifiers mods)
  {
    if (!IsOrbitAllowed())
      return false;

    float sens = mods.HasFlag(InputModifiers.Shift)
      ? OrbitSensitivity * ShiftFactor
      : OrbitSensitivity;

    if (_modeSubject.Value == CameraMode.CometOrbiting)
    {
      float yawRad = -pixelDelta.X * sens;
      float pitchRad = -pixelDelta.Y * sens;

      lock (_orbitOffsetLock)
      {
        var yaw = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, yawRad);
        var currentFwd = Vector3.Normalize(-_orbitOffset);
        var currentUpHint = Math.Abs(currentFwd.Z) < 0.99f ? Vector3.UnitZ : -Vector3.UnitY;
        var currentRight = Vector3.Normalize(Vector3.Cross(currentUpHint, currentFwd));
        var pitch = Quaternion.CreateFromAxisAngle(currentRight, pitchRad);

        var proposedOffset = Vector3.Transform(_orbitOffset, pitch * yaw);
        var proposedFwd = Vector3.Normalize(-proposedOffset);
        if (Math.Abs(proposedFwd.Z) >= 0.98f)
        {
          // Gimbal lock avoidance: skip pitch if too close to poles.
          proposedOffset = Vector3.Transform(_orbitOffset, yaw);
        }
        _orbitOffset = proposedOffset;
      }

      var lastCometPos = _cometTracker.LastKnownCometPosition;
      if (lastCometPos.HasValue)
        SnapCameraToOrbit(lastCometPos.Value);
      return true;
    }

    if (_modeSubject.Value == CameraMode.EarthPosition)
    {
      float earthSens = sens * 0.1f;
      float earthYawRad = -pixelDelta.X * earthSens;
      float earthPitchRad = -pixelDelta.Y * earthSens;

      lock (_earthPosLock)
      {
        var earthYaw = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, earthYawRad);
        var right = Vector3.Transform(Vector3.UnitX, _earthRotation);
        var earthPitch = Quaternion.CreateFromAxisAngle(right, earthPitchRad);
        _earthRotation = Quaternion.Normalize(_earthRotation * earthPitch * earthYaw);

        SnapCameraToEarth(_lastEarthPos);
      }
      return true;
    }

    return false;
  }

  /// <summary>
  /// Pan the camera (translate on the view-right / view-up plane).
  /// Rotation is held fixed. Permitted only in <see cref="CameraMode.UpZenith"/>.
  /// </summary>
  public bool RequestPan(Vector2 pixelDelta, InputModifiers mods)
  {
    if (!IsPanAllowed())
      return false;

    var last = _lastConfirmedTransform;
    if (last is null)
      return false;

    float sens = mods.HasFlag(InputModifiers.Shift) ? PanSensitivity * ShiftFactor : PanSensitivity;

    var camRot = new Quaternion(last.RotX, last.RotY, last.RotZ, last.RotW);
    // Engine convention: +X = Right, +Z = Up
    var right = Vector3.Transform(Vector3.UnitX, camRot);
    var up = Vector3.Transform(Vector3.UnitZ, camRot);

    // Note: pixelDelta.X is positive right, pixelDelta.Y is positive down in Avalonia
    var worldD = (-right * pixelDelta.X + up * pixelDelta.Y) * sens;

    var newPos = new Vector3(
      (float)last.PosX + worldD.X,
      (float)last.PosY + worldD.Y,
      (float)last.PosZ + worldD.Z
    );

    bool ok = RotoTranslateDirect(newPos, camRot);
    if (ok)
    {
      _lastConfirmedTransform = new CameraTransformState(
        newPos.X,
        newPos.Y,
        newPos.Z,
        camRot.X,
        camRot.Y,
        camRot.Z,
        camRot.W
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
      lock (_orbitOffsetLock)
      {
        float scaleFactor = 1f + pixelDy * sens;
        float currentLen = _orbitOffset.Length();
        var newLen = currentLen * scaleFactor;
        if (newLen < OrbitMinDistance)
          newLen = OrbitMinDistance;
        if (newLen > OrbitMaxDistance)
          newLen = OrbitMaxDistance;
        if (currentLen > 1e-10f)
          _orbitOffset = Vector3.Normalize(_orbitOffset) * newLen;
      }
      var lastCometPos = _cometTracker.LastKnownCometPosition;
      if (lastCometPos.HasValue)
        SnapCameraToOrbit(lastCometPos.Value);
      return true;
    }

    if (_modeSubject.Value == CameraMode.EarthPosition)
    {
      lock (_earthPosLock)
      {
        var camForward = Vector3.Transform(-Vector3.UnitY, _earthRotation);
        _earthOffset += camForward * pixelDy * sens;
        SnapCameraToEarth(_lastEarthPos);
      }
      return true;
    }

    return false;
  }

  // ── Projection commands ────────────────────────────────────────────────────

  /// <summary>
  /// Request a perspective projection change. Not mode-gated.
  /// </summary>
  public bool RequestPerspectiveProjection(float fov, float aspectRatio, float near, float far) =>
    _runtimeService.CameraSetPerspective(CameraEntityId ?? 0, fov, aspectRatio, near, far);

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
    _runtimeService.CameraSetOrthographic(CameraEntityId ?? 0, left, right, bottom, top, near, far);

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

  // UpZenith first-entry orthographic half-extent (AU).
  // sun_radius ≈ 0.00465 AU; halfH = sun_radius / 0.30 ≈ 0.0155 AU
  // so the sun fills ~30% of the ±halfH ortho box on first entry.
  private const float UpZenithObservationHalfExtent = 0.0155f;

  /// <summary>
  /// Applies the UpZenith first-entry orthographic projection:
  /// half-height = 0.3 AU, half-width = 0.3 × aspect, mapped to the full VkExtent2D.
  /// near/far are taken from the current confirmed projection state (fallback: 0.001 / 1000).
  /// </summary>
  private void ApplyUpZenithDefaultProjection()
  {
    var cur = _projectionSubject.Value;
    float near = cur?.Near ?? 0.001f;
    float far = cur?.Far ?? 1000f;
    float halfH = UpZenithObservationHalfExtent;
    float halfW = halfH * _viewportAspect;
    RequestOrthographicProjection(-halfW, halfW, -halfH, halfH, near, far);
  }

  /// <summary>Re-applies a previously saved projection snapshot to the runtime.</summary>
  private void ApplyProjectionSnapshot(CameraProjectionState proj)
  {
    if (proj.IsPerspective)
      RequestPerspectiveProjection(proj.Fov, _viewportAspect, proj.Near, proj.Far);
    else
      RequestOrthographicProjection(proj.Left, proj.Right, proj.Bottom, proj.Top, proj.Near, proj.Far);
  }

  /// <summary>
  /// Apply a validated absolute roto-translate via the typed runtime interface.
  /// All unsafe buffer packing lives in <c>NativeRuntimeService.CameraSetRotoTranslate</c>.
  /// </summary>
  private bool RotoTranslateDirect(Vector3 position, Quaternion rotation) =>
    _runtimeService.CameraSetRotoTranslate(CameraEntityId ?? 0, position, rotation);

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
    ulong? camId = CameraEntityId;
    if (camId is null)
      return;

    Vector3 offset;
    lock (_orbitOffsetLock)
      offset = _orbitOffset;

    var targetPos = cometPos + offset;
    var worldFwd = Vector3.Normalize(cometPos - targetPos);
    var worldUpHint = Math.Abs(worldFwd.Z) < 0.99f ? Vector3.UnitZ : -Vector3.UnitY;
    var worldRight = Vector3.Normalize(Vector3.Cross(worldUpHint, worldFwd));
    var worldUp = Vector3.Cross(worldFwd, worldRight);
    var rot = EngineQuatFromBasis(worldRight, -worldFwd, worldUp);

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
  public void OnViewportReady(ulong cameraEntityId, uint viewportWidth, uint viewportHeight)
  {
    CameraEntityId = cameraEntityId;
    _viewportAspect = viewportHeight > 0 ? (float)viewportWidth / viewportHeight : 1f;
    RegisterSimListeners(cameraEntityId);
    RegisterEarthListener();
    // Snap immediately on first viewport-ready so the camera starts at the
    // correct mode position from frame 1 rather than animating over 2.5 s.
    TriggerModeTransitionAnimation(_modeSubject.Value, snapImmediate: true);
  }

  public void OnViewportResized(uint viewportWidth, uint viewportHeight)
  {
    _viewportAspect = viewportHeight > 0 ? (float)viewportWidth / viewportHeight : 1f;

    // We must resend the projection matrix when the viewport aspect ratio changes
    // to prevent the native swapchain from stretching the old projection.
    var last = LastConfirmedProjection;
    if (last is not null)
    {
      if (last.IsPerspective)
      {
        RequestPerspectiveProjection(last.Fov, _viewportAspect, last.Near, last.Far);
      }
      else
      {
        float height = Math.Abs(last.Top - last.Bottom);
        float width = height * _viewportAspect;
        RequestOrthographicProjection(
          -width / 2f,
          width / 2f,
          -height / 2f,
          height / 2f,
          last.Near,
          last.Far
        );
      }
    }
  }

  private void RegisterSimListeners(ulong camId)
  {
    // Already registered — prevent double-registration if OnViewportReady is called twice.
    if (_transformListenerToken is not null)
      return;

    _transformListenerToken = _runtimeService.RegisterSimulationListener(
      camId,
      ComponentForeignId.HighResTransform,
      HandleTransformCallback
    );

    _projectionListenerToken = _runtimeService.RegisterSimulationListener(
      camId,
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
      dto.RotX,
      dto.RotY,
      dto.RotZ,
      dto.RotW
    );
    // Reference write is pointer-width atomic in .NET — no lock needed for _lastConfirmedTransform.
    _lastConfirmedTransform = state;
    // BehaviorSubject.OnNext is not thread-safe for concurrent calls; marshal to the UI thread.
    _schedulerProvider.MainThread.Schedule(() => _transformSubject.OnNext(state));
  }

  private unsafe void HandleProjectionCallback(nint dataPtr)
  {
    var dto = *(CameraProjectionDTO*)dataPtr;
    float fov = 45f;
    float aspect = 1f;
    float near = 0.1f;
    float far = 1000f;
    float left = 0;
    float right = 800;
    float bottom = 0;
    float top = 600;
    float focusDistance = 1f;
    if (dto.IsOrthographic == 0)
    {
      fov = dto.Fov;
      aspect = dto.Aspect;
    }
    else
    {
      left = dto.Left;
      right = dto.Right;
      bottom = dto.Bottom;
      top = dto.Top;
    }
    near = dto.Near;
    far = dto.Far;
    focusDistance = dto.FocusDistance;

    if (_projectionSubject.Value != null)
    {
      if (dto.IsOrthographic == 0)
      {
        left = _projectionSubject.Value.Left;
        right = _projectionSubject.Value.Right;
        bottom = _projectionSubject.Value.Bottom;
        top = _projectionSubject.Value.Top;
      }
      else
      {
        fov = _projectionSubject.Value.Fov;
        aspect = _projectionSubject.Value.Aspect;
      }
    }
    var projState = new CameraProjectionState(
      IsPerspective: dto.IsOrthographic == 0,
      fov,
      aspect,
      near,
      far,
      left,
      right,
      bottom,
      top,
      focusDistance
    );
    // Marshal to the UI thread — same reason as HandleTransformCallback.
    _schedulerProvider.MainThread.Schedule(() => _projectionSubject.OnNext(projState));
  }

  private unsafe void HandleEarthTransformCallback(nint dataPtr)
  {
    var dto = *(HighResTransformDTO*)dataPtr;
    var newPos = new Vector3((float)dto.PosX, (float)dto.PosY, (float)dto.PosZ);
    lock (_earthPosLock)
    {
      _lastEarthPos = newPos;
      if (_modeSubject.Value == CameraMode.EarthPosition)
      {
        SnapCameraToEarth(newPos);
      }
    }
  }

  private void SnapCameraToEarth(Vector3 earthPos)
  {
    ulong? camId = CameraEntityId;
    if (camId is null)
      return;

    _runtimeService.AddCameraAnimation(
      camId.Value,
      new AnimationTarget(earthPos + _earthOffset, _earthRotation, OrbitTrackingAnimationSeconds)
    );
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    StopCometOrbitTracking();
    _pendingProjectionCts?.Cancel();
    _pendingProjectionCts?.Dispose();
    _pendingProjectionCts = null;
    _transformListenerToken?.Dispose();
    _projectionListenerToken?.Dispose();
    _earthListenerToken?.Dispose();
    _transformSubject.Dispose();
    _projectionSubject.Dispose();
    _modeSubject.Dispose();
  }
}
