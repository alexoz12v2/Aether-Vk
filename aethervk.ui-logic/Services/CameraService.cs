using System;
using System.Collections.Generic;
using System.Numerics;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.Messages;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Services;

/// <summary>
/// Current camera mode, governing which transform operations are allowed.
/// </summary>
public enum CameraMode
{
  /// Camera locked to Earth's trajectory. Zoom disabled; pan locked; rotation changes orientation only.
  EarthPosition,

  /// Snap-to-zenith mode (derived from EarthPosition). Pan only; rotation and zoom locked.
  UpZenith,

  /// Camera orbits the comet centre-of-mass. Zoom allowed (with limits); pan locked;
  /// rotation focuses on comet. Camera automatically tracks comet position when simulation is running.
  CometOrbiting,
}

/// <summary>
/// Controls how the Earth Observer camera's look direction behaves while tracking
/// a body-fixed surface point on Earth. Does not affect camera position — that is
/// always derived from the lat/lon anchor rotated by Earth's instantaneous orientation.
/// </summary>
public enum EarthObserverOrientationMode
{
  /// Look direction is fixed in the heliocentric/ecliptic inertial frame.
  /// Dragging the mouse changes (and permanently stores) the inertial look direction.
  Inertial,

  /// Camera always points toward the comet's current position.
  /// Look direction is updated automatically each time the comet position is known.
  CometTracking,

  /// Look direction rotates with Earth's body — like a physical telescope anchored to the ground.
  /// The angle relative to the surface stays constant; Earth's spin carries it through the sky.
  EarthFixed,
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
/// Snapshot of the sun's frustum visibility state, as reported by the logic thread via
/// <c>ExternalState::SunVisibilityChanged</c>.
///
/// <para><c>NdcX</c> and <c>NdcY</c> carry the actual projected NDC coordinates even when
/// the sun is off-screen (values may exceed ±1). Use <see cref="IsVisible"/> to distinguish
/// on/off-screen, and the NDC pair to compute the arrowhead bearing for the overlay
/// indicator.</para>
/// </summary>
public sealed record SunVisibilityState(bool IsVisible, float NdcX, float NdcY);

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

  private readonly ICometMessenger _cometMessenger;

  private readonly BehaviorSubject<CameraTransformState?> _transformSubject = new(null);
  private readonly BehaviorSubject<CameraProjectionState?> _projectionSubject = new(null);

  // Start with IsVisible=true so the overlay indicator is hidden at launch.
  // The Rust engine fires the correct transition on its first logic tick (~16 ms),
  // preventing a brief crosshair flash before the actual state is known.
  private readonly BehaviorSubject<SunVisibilityState> _sunVisibilitySubject =
    new(new SunVisibilityState(IsVisible: true, NdcX: 0f, NdcY: 0f));

  private readonly BehaviorSubject<CameraMode> _modeSubject = new(CameraMode.UpZenith);
  private IDisposable? _transformListenerToken;
  private IDisposable? _projectionListenerToken;
  private IDisposable? _cometOrbitSubscription;
  private IDisposable? _earthListenerToken;
  private IDisposable? _sunVisibilityListenerToken;

  // Orbit offset in simulation units (AU) — kept constant while in CometOrbiting mode.
  // Orbit addition is done in f64 (see SnapCameraToOrbit / TriggerModeTransitionAnimation),
  // so this value does not need to be large to survive f32 cancellation. 5e-5 AU ≈ 7,500 km.
  private Vector3 _orbitOffset = new(0f, 0f, 5e-5f); // ~7,500 km — orbit-offset arithmetic is f64
  private readonly object _orbitOffsetLock = new();

  // Cinemachine OrbitalFollow "Sphere" style: store the camera's position on the sphere as
  // explicit spherical coordinates so interactive drag is drift-free (no quaternion chaining).
  // Engine frame: +X = right, +Y = sideways, +Z = up.
  // Azimuth 0 → offset along +X; elevation +π/2 → offset along +Z (straight above comet).
  // Both fields are guarded by _orbitOffsetLock (written together with _orbitOffset).
  private float _orbitAzimuthRad   = 0f;   // horizontal orbit angle, radians, [0, 2π)
  private float _orbitElevationRad = 0f;   // vertical orbit angle,   radians, (−π/2, +π/2)


  // Earth position cache — updated via SIMULATION_CALLBACK for the earth entity.
  // Initialised to 1 AU on +X as a safe fallback before the first callback fires.
  private Vector3 _lastEarthPos = new(1f, 0f, 0f);

  // Lock protecting all _earth* fields below.
  private readonly object _earthPosLock = new();

  // Earth radius in AU (6 371 km / 149 597 870.7 km per AU ≈ 4.26e-5 AU).
  private const float EarthRadiusAu = 4.26e-5f;

  // Surface anchor in Earth's body-fixed frame (body-fixed Cartesian, AU scale).
  // Default: (0°N, 0°E) → (1, 0, 0) × EarthRadiusAu in the body-fixed frame.
  private Vector3 _earthSurfacePointBf = new(EarthRadiusAu, 0f, 0f);

  // Earth body-fixed → world rotation, updated from HighResTransformDTO every tick.
  private Quaternion _earthBodyRot = Quaternion.Identity;

  // Camera orientation in the inertial frame, updated on every drag or mode switch.
  private Quaternion _earthRotation = Quaternion.Identity;

  // Camera look direction frozen in the inertial frame (Inertial mode anchor).
  private Quaternion _inertialLookDir = Quaternion.Identity;

  // Camera look direction in Earth's body-fixed frame (EarthFixed mode anchor).
  private Quaternion _earthFixedLookDir = Quaternion.Identity;

  // Current Earth Observer orientation sub-mode.
  private EarthObserverOrientationMode _earthOrientationMode = EarthObserverOrientationMode.Inertial;

  // Observable that broadcasts orientation mode changes to the Settings tab.
  private readonly BehaviorSubject<EarthObserverOrientationMode> _earthOrientationModeSubject =
    new(EarthObserverOrientationMode.Inertial);

  /// <summary>
  /// Fires on the main thread whenever <see cref="EarthObserverOrientationMode"/> changes.
  /// Subscribe in <c>ViewportSettingsViewModel</c>.
  /// </summary>
  public IObservable<EarthObserverOrientationMode> EarthObserverOrientationModeChanged =>
    _earthOrientationModeSubject.AsObservable();

  // Last authoritative transform confirmed by the runtime. Populated by HandleTransformCallback.
  // Read synchronously by Request* methods to compute new absolute target transforms.
  private CameraTransformState? _lastConfirmedTransform;

  /// <summary>
  /// If true, orthographic projection bounds are constrained to match the viewport aspect ratio.
  /// This also applies during window resize.
  /// </summary>
  public bool IsOrthoProportionsLocked { get; set; } = true;

  // ── Mode state memory ────────────────────────────────────────────────────────
  // Saved (transform, projection) per mode — restored via animation on re-entry.
  private sealed record ModeSnapshot(
    CameraTransformState Transform,
    CameraProjectionState? Projection
  );

  private readonly Dictionary<CameraMode, ModeSnapshot> _modeSnapshots = new();

  // Aspect ratio (W/H) of the Vulkan render target — set by OnViewportReady.
  private float _viewportAspect = 1f;
  
  public float ViewportAspect => _viewportAspect;
  
  public event Action? ViewportResized;

  // Cancels any pending deferred projection change when a new mode switch fires.
  private CancellationTokenSource? _pendingProjectionCts;

  // Animation durations
  private const float ModeSwitchAnimationSeconds = 2.5f;

  // Short duration so the animation is always in-flight when the next comet callback
  // arrives, allowing retarget() to produce smooth continuous orbit tracking.
  private const float OrbitTrackingAnimationSeconds = 0.4f;

  // 1-frame target (≈60 Hz) used for interactive drag events in CometOrbiting mode.
  // CameraSetRotoTranslate is rejected while a TransformAnimationComponent is active,
  // so we must go through AddCameraAnimation even during drag — but with this duration
  // the Rust retarget() completes within a single render frame, giving instantaneous feel.
  private const float InteractiveDragAnimationSeconds = 0.016f;

  // ── Movement sensitivity ─────────────────────────────────────────────────────
  // All units in simulation scale (AU / radian per pixel of drag).
  // Shift modifier applies ShiftFactor for Blender-style fine control.
  private const float OrbitSensitivity = 0.005f; // rad/px
  private const float PanSensitivity = 1e-5f; // AU/px
  private const float ZoomSensitivity = 2e-5f; // AU/px (vertical drag)
  private const float ShiftFactor = 0.2f; // fine-control multiplier

  // ── Comet orbit zoom limits ──────────────────────────────────────────────────
  // Min/max distance (AU) from the comet nucleus when zooming in CometOrbiting mode.
  // These are instance fields (not consts) so TriggerModeTransitionAnimation can scale
  // them relative to the nucleus radius when Horizon data is available.
  private float _orbitMinDistance = 1e-6f; // ~150 km — hard stop before nucleus surface
  private float _orbitMaxDistance = 1e-2f; // ~1.5 million km — wide view

  // Nucleus radius (km) cached from NucleusRadiusKnownMessage.
  // 0 = unknown (fallback to legacy default orbit offset).
  private float _lastKnownNucleusRadiusKm = 0f;

  public CameraService(
    INativeRuntimeService runtimeService,
    ISchedulerProvider schedulerProvider,
    CometPositionTrackerService cometTracker,
    CometConfigService cometConfigService,
    BreadcrumbService breadcrumbService,
    ICometMessenger cometMessenger
  )
  {
    _runtimeService = runtimeService;
    _schedulerProvider = schedulerProvider;
    _cometTracker = cometTracker;
    _cometConfigService = cometConfigService;
    _breadcrumbService = breadcrumbService;
    _cometMessenger = cometMessenger;

    // Cache the nucleus radius whenever it becomes known (Horizon fetch or manual entry).
    cometMessenger.Register<NucleusRadiusKnownMessage>(this, (_, msg) =>
    {
      _lastKnownNucleusRadiusKm = msg.RadiusKm;
      Console.WriteLine($"[CameraService] NucleusRadius updated: {msg.RadiusKm:F2} km");
    });
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
  /// Emits whenever the sun (world origin) enters or exits the primary camera's frustum.
  /// Only fires on state *transitions*, not every frame.
  /// Observed on the main (UI) thread — intended for overlay display only.
  /// </summary>
  public IObservable<SunVisibilityState> SunVisibilityChanged =>
    _sunVisibilitySubject.ObserveOn(_schedulerProvider.MainThread);

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
    {
      _orbitOffset = offset;
      InitOrbitAnglesFromOffset(offset); // keep spherical angles in sync
    }
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
      if (height < 1e-5f)
        height = 2f; // Fallback if 0
      float width = height * _viewportAspect;
      RequestOrthographicProjection(
        -width / 2f,
        width / 2f,
        -height / 2f,
        height / 2f,
        proj.Near,
        proj.Far
      );
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
      CameraMode.CometOrbiting => true,  // drag rotates orbit offset around comet
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
      CameraMode.EarthPosition  => false, // surface-anchored — zoom not meaningful
      CameraMode.CometOrbiting  => true,
      CameraMode.UpZenith       => false,
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
          // Place the camera at the body-fixed surface anchor (default 0°N, 0°E),
          // oriented toward the Sun (world origin). Snapshots from previous visits
          // are stored above and restored without entering this branch.
          lock (_earthPosLock)
          {
            var surfaceWorld = Vector3.Transform(_earthSurfacePointBf, _earthBodyRot);
            var camPos = _lastEarthPos + surfaceWorld;
            _earthRotation  = LookAtOriginFrom(camPos);
            _inertialLookDir   = _earthRotation;
            _earthFixedLookDir = WorldLookDirToBodyFixed(_earthBodyRot, _earthRotation);
            targetPos = camPos;
            targetRot = _earthRotation;
          }
          deferredProjection = ApplyDefaultPerspectiveProjection;
          break;

        case CameraMode.CometOrbiting:
          var kp = _cometTracker.LastKnownCometPosition;
          Console.WriteLine(
            $"[CameraService] TriggerModeTransitionAnimation(CometOrbiting) — LastKnownCometPosition={kp?.ToString() ?? "null"}"
          );
          if (kp is { } cometPos)
          {
            // ── Auto-set orbit distance from nucleus radius ─────────────────────
            // Goal: nucleus gizmo fills ~30% of screen diameter.
            // At FOV=45°, viewProj[1][1] = 1/tan(22.5°).  For a sphere of radius r_AU
            // at distance d_AU from camera:
            //   projected_diameter_ndc = 2 * r_AU * viewProj[1][1] / d_AU
            // Setting projected_diameter_ndc = 0.30 → d_AU = r_AU / tan(0.15 * FOV/2).
            // All trig done in double (Math.*) — MathF is not available in netstandard2.0.
            const double AuToKm = 149_597_870.7;
            const double TargetFraction = 0.30;           // 30% screen diameter
            const double FovRad = Math.PI / 4.0;          // 45° vertical FOV

            float nucleusRadiusKm = _lastKnownNucleusRadiusKm;
            float autoOrbitDistanceAu;
            if (nucleusRadiusKm > 0f)
            {
              double rAu = nucleusRadiusKm / AuToKm;
              double halfAngTan = Math.Tan(TargetFraction * 0.5 * FovRad);
              autoOrbitDistanceAu = (float)(rAu / halfAngTan);

              // Zoom limits: min = 10% above surface, max = 500× radius (wide survey view).
              // Store as float AU — these small values (≤ 1e-5 AU) fit cleanly in f32.
              _orbitMinDistance = (float)(rAu * 1.1);
              _orbitMaxDistance = (float)(rAu * 500.0);

              Console.WriteLine(
                $"[CameraService] CometOrbiting: nucleusRadius={nucleusRadiusKm:F2} km → "
                + $"orbit={autoOrbitDistanceAu * AuToKm:F1} km, "
                + $"min={_orbitMinDistance * AuToKm:F1} km, "
                + $"max={_orbitMaxDistance * AuToKm:F0} km"
              );
            }
            else
            {
              // No nucleus radius known — preserve the existing offset magnitude.
              float existingMag = _orbitOffset.Length();
              autoOrbitDistanceAu = existingMag > 1e-8f ? existingMag : 5e-5f;
              _orbitMinDistance = 1e-6f;
              _orbitMaxDistance = 1e-2f;
              Console.WriteLine("[CameraService] CometOrbiting: nucleus radius unknown, keeping existing orbit offset.");
            }

            // Re-orient the offset along its current direction (default +Z), scaled to the
            // computed distance.  Must happen BEFORE the f64 addition below so targetPos is
            // based on the correct offset.
            lock (_orbitOffsetLock)
            {
              var dir = _orbitOffset.Length() > 1e-10f
                ? Vector3.Normalize(_orbitOffset)
                : new Vector3(0f, 0f, 1f);
              _orbitOffset = dir * autoOrbitDistanceAu;
              // Sync spherical angles so interactive drag picks up from the right position.
              InitOrbitAnglesFromOffset(_orbitOffset);
            }

            // Add offset to comet position in f64 to prevent catastrophic cancellation at
            // large heliocentric distances (e.g. 5 AU where a 1e-5 AU offset is lost in f32).
            Vector3 offsetSnap;
            lock (_orbitOffsetLock) offsetSnap = _orbitOffset;
            if (_cometTracker.LastKnownCometPositionF64 is { } f64)
            {
              targetPos = new Vector3(
                (float)(f64.X + (double)offsetSnap.X),
                (float)(f64.Y + (double)offsetSnap.Y),
                (float)(f64.Z + (double)offsetSnap.Z));
            }
            else
            {
              targetPos = cometPos + offsetSnap; // f32 fallback (comet near origin)
            }
            targetRot = ComputeLookAtComet(targetPos);
          }
          else
          {
            Console.WriteLine("[CameraService] TriggerModeTransitionAnimation(CometOrbiting): no comet position known — aborting mode transition animation.");
            return;
          }
          // Near plane: 5% of orbit offset magnitude, clamped to [1e-6, 0.001] AU.
          // Default orbit 5e-5 AU → near ≈ 2.5e-6 AU ≈ 374 km.
          // The fixed fallback of 0.001 AU (150 000 km) is 20× the orbital distance and
          // would near-clip both the sphere gizmo and the nearby trajectory arc.
          deferredProjection = () =>
          {
            float orbitMag;
            lock (_orbitOffsetLock) orbitMag = _orbitOffset.Length();
            float cometNear = Math.Max(1e-6f, Math.Min(orbitMag * 0.05f, 0.001f));
            var cur = _projectionSubject.Value;
            float far = cur?.Far ?? 1000f;
            RequestPerspectiveProjection(45f, _viewportAspect, cometNear, far);
          };
          break;


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
  /// Strips any roll component from <paramref name="q"/> by rebuilding the camera basis from
  /// its forward direction alone, constraining world-up to +Z.
  ///
  /// <para>Engine convention: forward = local −Y rotated by <paramref name="q"/>.
  /// The returned quaternion has the same yaw and pitch as <paramref name="q"/> but zero roll.</para>
  ///
  /// <para>Falls back to returning <paramref name="q"/> unchanged when the forward vector is
  /// degenerate (near-zero length).</para>
  /// </summary>
  private static Quaternion StripRoll(Quaternion q)
  {
    // Extract the forward direction: engine forward is local −Y.
    var fwd = Vector3.Transform(-Vector3.UnitY, q);
    if (fwd.LengthSquared() < 1e-10f)
      return q; // degenerate — return unchanged

    // LookAtOriginFrom(pos) builds a rotation toward the origin from pos.
    // Passing -fwd as "pos" gives worldFwd = normalize(-(-fwd)) = normalize(fwd),
    // which is exactly the forward direction we want, constrained to up=+Z.
    return LookAtOriginFrom(-fwd);
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
      // Positive ΔX (drag right) → decrease azimuth (camera moves left around comet → comet appears right).
      // Positive ΔY (drag down in Avalonia) → decrease elevation (tilt down).
      float yawRad   = -pixelDelta.X * sens;
      float pitchRad = -pixelDelta.Y * sens;

      lock (_orbitOffsetLock)
      {
        float radius = _orbitOffset.Length();
        if (radius < 1e-10f)
          radius = 5e-5f; // guard: use default distance if somehow zero

        // ── Cinemachine OrbitalFollow "Sphere" style ──────────────────────────
        // Accumulate horizontal (azimuth) and vertical (elevation) angles separately
        // so there is no quaternion drift — the offset direction is always recomputed
        // from clean trigonometry rather than chained quaternion multiplications.
        // Note: MathF is not available in netstandard2.0; use (float)Math.* instead.
        const float PI = (float)Math.PI;
        _orbitAzimuthRad += yawRad;
        _orbitAzimuthRad %= 2f * PI; // keep in [0, 2π) to avoid float creep

        float newElev = _orbitElevationRad + pitchRad;
        _orbitElevationRad = Math.Max(-PI / 2f + 0.01f,   // south-pole guard
                             Math.Min(+PI / 2f - 0.01f,   // north-pole guard
                                      newElev));

        // Recompute offset from spherical coordinates.
        // Engine frame: +X = right, +Y = sideways, +Z = up.
        // Azimuth 0 → offset in +X; elevation +π/2 → offset in +Z (directly above comet).
        float cosElev = (float)Math.Cos(_orbitElevationRad);
        _orbitOffset = new Vector3(
          cosElev * (float)Math.Cos(_orbitAzimuthRad),
          cosElev * (float)Math.Sin(_orbitAzimuthRad),
          (float)Math.Sin(_orbitElevationRad)
        ) * radius;
      }

      var lastCometPos = _cometTracker.LastKnownCometPosition;
      if (lastCometPos.HasValue)
        // Use a 1-frame animation duration so retarget() in Rust completes within one
        // render tick — CameraSetRotoTranslate is rejected while an animation is active,
        // so we must always go through AddCameraAnimation even during interactive drag.
        SnapCameraToOrbit(lastCometPos.Value, InteractiveDragAnimationSeconds);
      return true;
    }

    if (_modeSubject.Value == CameraMode.EarthPosition)
    {
      // In Earth Observer mode, dragging rotates the camera in place — the surface
      // anchor position does not change.  We apply reduced sensitivity (0.1×) to
      // feel natural at human-scale (surface of a planet vs. solar-system orbit).
      float earthSens = sens * 0.1f;
      float earthYawRad   = -pixelDelta.X * earthSens;
      float earthPitchRad = -pixelDelta.Y * earthSens;

      lock (_earthPosLock)
      {
        // Both yaw (around world +Z) and pitch (around the camera's world-space right)
        // are world-space rotations, so they must be PRE-multiplied onto the base
        // orientation.  In System.Numerics, A * B = "apply B first then A", so:
        //   pitch * yaw * _earthRotation  =  (base → yaw → pitch)
        // The old order (_earthRotation * pitch * yaw) applied the deltas in the
        // camera's already-rotated local frame, making vertical drag bleed into
        // horizontal rotation.
        var yaw   = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, earthYawRad);
        var right = Vector3.Transform(Vector3.UnitX, _earthRotation); // camera's world-right
        var pitch = Quaternion.CreateFromAxisAngle(right, earthPitchRad);
        var newRot = StripRoll(Quaternion.Normalize(pitch * yaw * _earthRotation));

        // Update all cached look directions so a later mode switch has fresh anchors.
        _earthRotation     = newRot;
        _inertialLookDir   = newRot;
        _earthFixedLookDir = WorldLookDirToBodyFixed(_earthBodyRot, newRot);

        // Apply the rotation directly without animation: CameraSetRotoTranslate succeeds
        // in EarthPosition mode because no tracking animation is permanently in-flight.
        // This gives the user instant 1:1 response (no 0.4 s animation lag).
        var surfaceWorld = Vector3.Transform(_earthSurfacePointBf, _earthBodyRot);
        var camPos       = _lastEarthPos + surfaceWorld;
        RotoTranslateDirect(camPos, newRot);
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

    var proj = LastConfirmedProjection;
    if (proj != null && !proj.IsPerspective)
    {
        float halfHeight = Math.Abs(proj.Top - proj.Bottom) / 2f;
        sens *= (halfHeight / 0.0155f);
    }

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
        if (newLen < _orbitMinDistance)
          newLen = _orbitMinDistance;
        if (newLen > _orbitMaxDistance)
          newLen = _orbitMaxDistance;
        if (currentLen > 1e-10f)
          _orbitOffset = Vector3.Normalize(_orbitOffset) * newLen;
      }
      var lastCometPos = _cometTracker.LastKnownCometPosition;
      if (lastCometPos.HasValue)
        SnapCameraToOrbit(lastCometPos.Value, InteractiveDragAnimationSeconds);
      return true;
    }

    return false;
  }

  // ── Earth Observer controls (called by ViewportSettingsViewModel) ──────────

  /// <summary>
  /// Sets the surface anchor to a geographic lat/lon on Earth.
  /// The body-fixed Cartesian offset is recomputed and the camera is immediately
  /// re-snapped to the new position if in <see cref="CameraMode.EarthPosition"/>.
  /// </summary>
  /// <param name="latDeg">Geodetic latitude in degrees (−90 … +90, positive = North).</param>
  /// <param name="lonDeg">Longitude in degrees (−180 … +180, positive = East).</param>
  public void SetEarthObserverLatLon(float latDeg, float lonDeg)
  {
    float lat = latDeg * (float)Math.PI / 180f;
    float lon = lonDeg * (float)Math.PI / 180f;

    // Body-fixed unit vector (IAU/ITRF convention, +Z = North Pole):
    //   X = cos(lat)·cos(lon),  Y = cos(lat)·sin(lon),  Z = sin(lat)
    var bfUnit = new Vector3(
      (float)Math.Cos(lat) * (float)Math.Cos(lon),
      (float)Math.Cos(lat) * (float)Math.Sin(lon),
      (float)Math.Sin(lat)
    );

    lock (_earthPosLock)
    {
      _earthSurfacePointBf = bfUnit * EarthRadiusAu;
      if (_modeSubject.Value == CameraMode.EarthPosition)
        SnapCameraToEarth(_lastEarthPos);
    }
  }

  /// <summary>
  /// Switches the Earth Observer orientation sub-mode.
  /// Snapshots the current look direction into the new mode's anchor before switching.
  /// </summary>
  public void SetEarthObserverOrientationMode(EarthObserverOrientationMode mode)
  {
    lock (_earthPosLock)
    {
      // Snapshot the current look direction into whatever anchor the new mode uses,
      // so the first frame after the switch looks identical to the last frame before it.
      switch (mode)
      {
        case EarthObserverOrientationMode.Inertial:
          _inertialLookDir = _earthRotation;
          break;
        case EarthObserverOrientationMode.EarthFixed:
          _earthFixedLookDir = WorldLookDirToBodyFixed(_earthBodyRot, _earthRotation);
          break;
        // CometTracking: no snapshot needed; look dir is always computed fresh.
      }

      _earthOrientationMode = mode;
      _earthOrientationModeSubject.OnNext(mode);

      if (_modeSubject.Value == CameraMode.EarthPosition)
        SnapCameraToEarth(_lastEarthPos);
    }
  }


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

  /// <summary>Applies the default perspective projection for Earth/Comet modes.</summary>
  private void ApplyDefaultPerspectiveProjection()
  {
    var cur = _projectionSubject.Value;
    float near = cur?.Near ?? 0.001f;
    float far = cur?.Far ?? 1000f;
    RequestPerspectiveProjection(45f, _viewportAspect, near, far);
  }

  /// <summary>Re-applies a previously saved projection snapshot to the runtime.</summary>
  private void ApplyProjectionSnapshot(CameraProjectionState proj)
  {
    if (proj.IsPerspective)
      RequestPerspectiveProjection(proj.Fov, _viewportAspect, proj.Near, proj.Far);
    else
      RequestOrthographicProjection(
        proj.Left,
        proj.Right,
        proj.Bottom,
        proj.Top,
        proj.Near,
        proj.Far
      );
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
    var currentKnown = _cometTracker.LastKnownCometPosition;
    Console.WriteLine(
      $"[CameraService] StartCometOrbitTracking — LastKnownCometPosition={currentKnown?.ToString() ?? "null"}"
    );
    _cometOrbitSubscription = _cometTracker
      .CometPositionRaw.Where(static pos => pos.HasValue)
      .Sample(TimeSpan.FromMilliseconds(50), _schedulerProvider.Background)
      .Subscribe(pos =>
      {
        // Guard: reject any delivery that arrives after the mode has changed.
        // Sample() uses a background scheduler; after Dispose() one extra tick can
        // still fire within the 50 ms window, which would retarget the outgoing
        // UpZenith/EarthPosition animation back toward the comet position.
        if (_modeSubject.Value != CameraMode.CometOrbiting)
          return;
        Console.WriteLine($"[CameraService] CometPositionRaw update → SnapCameraToOrbit({pos!.Value})");
        SnapCameraToOrbit(pos!.Value);
      });
  }

  private void StopCometOrbitTracking()
  {
    _cometOrbitSubscription?.Dispose();
    _cometOrbitSubscription = null;
  }

  private void SnapCameraToOrbit(
    Vector3 cometPos,
    float animSeconds = OrbitTrackingAnimationSeconds)
  {
    ulong? camId = CameraEntityId;
    if (camId is null)
      return;

    Vector3 offset;
    lock (_orbitOffsetLock)
      offset = _orbitOffset;

    Vector3 targetPos;
    if (_cometTracker.LastKnownCometPositionF64 is { } f64)
    {
      // Compute the sum in f64 to avoid catastrophic cancellation in f32 at large
      // heliocentric distances (e.g. 5 AU: a 5e-5 AU offset has <1 ULP in f32).
      // Only the final cast to f32 truncates — the orbit position is distinct from
      // cometPos as long as the offset exceeds a few ULPs of the f64 result, which
      // it easily does (f64 ULP at 10 AU ≈ 9e-13, so 5e-5 is ~5.5×10^7 ULPs).
      targetPos = new Vector3(
        (float)(f64.X + (double)offset.X),
        (float)(f64.Y + (double)offset.Y),
        (float)(f64.Z + (double)offset.Z));
    }
    else
    {
      // f64 not available yet (default position before first snapshot/callback)
      targetPos = cometPos + offset;
    }

    if (targetPos == cometPos)
    {
      // Sentinel: even after f64 arithmetic the final f32 result is indistinguishable
      // from the comet position. The offset is astronomically small — no action needed.
      Console.WriteLine(
        "[WARNING] CameraService: targetPos == cometPos even after f64 arithmetic. Offset may be sub-ULP in f32."
      );
    }

    var worldFwd = Vector3.Normalize(-offset);
    var worldUpHint = Math.Abs(worldFwd.Z) < 0.99f ? Vector3.UnitZ : -Vector3.UnitY;
    var worldRight = Vector3.Normalize(Vector3.Cross(worldUpHint, worldFwd));
    var worldUp = Vector3.Cross(worldFwd, worldRight);
    var rot = EngineQuatFromBasis(worldRight, -worldFwd, worldUp);

    _runtimeService.AddCameraAnimation(
      camId.Value,
      new AnimationTarget(targetPos, rot, animSeconds)
    );
  }
  /// <summary>
  /// Initialises the spherical coordinate angles (<see cref="_orbitAzimuthRad"/> and
  /// <see cref="_orbitElevationRad"/>) from the current <see cref="_orbitOffset"/> direction.
  /// Must be called under <see cref="_orbitOffsetLock"/> whenever <c>_orbitOffset</c> is
  /// reset to a new direction (e.g. on first entry into <see cref="CameraMode.CometOrbiting"/>).
  /// After this call, interactive drag uses clean spherical increments — no prior quaternion
  /// history is inherited.
  /// </summary>
  private void InitOrbitAnglesFromOffset(Vector3 offset)
  {
    float r = offset.Length();
    if (r < 1e-10f)
    {
      _orbitAzimuthRad   = 0f;
      _orbitElevationRad = 0f;
      return;
    }

    var n = offset / r; // unit direction
    // Elevation = arcsin(Nz) — clamp argument to [-1,1] to guard against float rounding.
    // MathF is not available in netstandard2.0; use (float)Math.* instead.
    float nzClamped = Math.Max(-1f, Math.Min(1f, n.Z));
    _orbitElevationRad = (float)Math.Asin(nzClamped);
    // Azimuth = angle of (Nx, Ny) in the XY plane.
    _orbitAzimuthRad   = (float)Math.Atan2(n.Y, n.X);
  }



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
    RegisterSunVisibilityListener();
    // Snap immediately on first viewport-ready so the camera starts at the
    // correct mode position from frame 1 rather than animating over 2.5 s.
    TriggerModeTransitionAnimation(_modeSubject.Value, snapImmediate: true);
  }

  public void OnViewportResized(uint viewportWidth, uint viewportHeight)
  {
    _viewportAspect = viewportHeight > 0 ? (float)viewportWidth / viewportHeight : 1f;
    ViewportResized?.Invoke();

    // We must resend the projection matrix when the viewport aspect ratio changes
    // to prevent the native swapchain from stretching the old projection.
    var last = LastConfirmedProjection;
    if (last is not null)
    {
      if (last.IsPerspective)
      {
        RequestPerspectiveProjection(last.Fov, _viewportAspect, last.Near, last.Far);
      }
      else if (IsOrthoProportionsLocked)
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

  private void RegisterSunVisibilityListener()
  {
    // Already registered — prevent double-registration if OnViewportReady fires twice.
    if (_sunVisibilityListenerToken is not null)
      return;

    _sunVisibilityListenerToken = _runtimeService.RegisterExternalStateListener(
      ExternalStateType.SunVisibilityChanged,
      HandleSunVisibilityCallback
    );
  }

  // ── Internal callback handling ─────────────────────────────────────────────

  private unsafe void HandleTransformCallback(nint dataPtr)
  {
    var dto = *(HighResTransformDTO*)dataPtr;
    // Strip any roll component accumulated via slerp drift in the animation system.
    // This keeps _lastConfirmedTransform (and any mode snapshots derived from it) roll-free
    // on the C# side even before the Rust side's strip_roll has fully propagated.
    var rawRot = new Quaternion(dto.RotX, dto.RotY, dto.RotZ, dto.RotW);
    var cleanRot = StripRoll(rawRot);
    var state = new CameraTransformState(
      dto.PosX,
      dto.PosY,
      dto.PosZ,
      cleanRot.X,
      cleanRot.Y,
      cleanRot.Z,
      cleanRot.W
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
    // Capture the Earth body-fixed → world rotation that AlmanacPlanet::step() computed
    // from the BPC file.  Previously discarded; now used to rotate the surface anchor.
    var newBodyRot = new Quaternion(dto.RotX, dto.RotY, dto.RotZ, dto.RotW);

    lock (_earthPosLock)
    {
      _lastEarthPos = newPos;
      _earthBodyRot = newBodyRot;
      if (_modeSubject.Value == CameraMode.EarthPosition)
        SnapCameraToEarth(newPos);
    }
  }

  private void SnapCameraToEarth(Vector3 earthPos)
  {
    ulong? camId = CameraEntityId;
    if (camId is null)
      return;

    // Transform the body-fixed surface anchor into world space using Earth's
    // current orientation (updated from the BPC callback).
    var surfaceWorld = Vector3.Transform(_earthSurfacePointBf, _earthBodyRot);
    var camPos = earthPos + surfaceWorld;

    // Resolve look direction from the current orientation sub-mode.
    Quaternion camRot = _earthOrientationMode switch
    {
      EarthObserverOrientationMode.Inertial =>
        // Stays fixed in the inertial frame; drag updates _inertialLookDir.
        _inertialLookDir,

      EarthObserverOrientationMode.CometTracking =>
        // Re-compute a look-at toward the comet every tick.
        ComputeLookAtComet(camPos),

      EarthObserverOrientationMode.EarthFixed =>
        // Rotate the body-fixed anchor by the current Earth rotation.
        Quaternion.Normalize(_earthBodyRot * _earthFixedLookDir),

      _ => _earthRotation,
    };

    _earthRotation = camRot;

    _runtimeService.AddCameraAnimation(
      camId.Value,
      new AnimationTarget(camPos, camRot, OrbitTrackingAnimationSeconds)
    );
  }

  /// <summary>
  /// Computes an engine-compatible quaternion that makes the camera look toward
  /// the last-known comet position from <paramref name="camPos"/>.
  /// Falls back to the current <see cref="_earthRotation"/> when the comet position
  /// is not yet known (before simulation starts or no comet loaded).
  /// </summary>
  private Quaternion ComputeLookAtComet(Vector3 camPos)
  {
    var cometPos = _cometTracker.LastKnownCometPosition;
    if (!cometPos.HasValue)
      return _earthRotation; // safe fallback

    var toComet = Vector3.Normalize(cometPos.Value - camPos);

    // World-up hint: prefer +Z; fall back when nearly on the Z axis.
    var upHint = Math.Abs(toComet.Z) < 0.99f ? Vector3.UnitZ : -Vector3.UnitY;
    var right   = Vector3.Normalize(Vector3.Cross(upHint, toComet));
    var up      = Vector3.Cross(toComet, right);

    // Engine forward = −Y; toComet is the desired forward direction.
    return EngineQuatFromBasis(right, -toComet, up);
  }

  /// <summary>
  /// Converts a world-space look-direction quaternion into Earth's body-fixed frame.
  /// Used to snapshot the current look direction when switching to
  /// <see cref="EarthObserverOrientationMode.EarthFixed"/>.
  /// </summary>
  private static Quaternion WorldLookDirToBodyFixed(Quaternion earthBodyRot, Quaternion worldLookDir)
  {
    // q_bf = inv(earthBodyRot) · worldLookDir
    return Quaternion.Normalize(Quaternion.Inverse(earthBodyRot) * worldLookDir);
  }

  private unsafe void HandleSunVisibilityCallback(nint dataPtr)
  {
    var dto = *(CSunVisibilityChangedDTO*)dataPtr;
    var state = new SunVisibilityState(
      IsVisible: dto.IsVisible != 0,
      NdcX: dto.NdcX,
      NdcY: dto.NdcY
    );
    // Marshal to the UI thread — only consumers are overlay ViewModels.
    _schedulerProvider.MainThread.Schedule(() => _sunVisibilitySubject.OnNext(state));
  }

  // ── IDisposable ────────────────────────────────────────────────────────────

  public void Dispose()
  {
    _cometMessenger.Unregister<NucleusRadiusKnownMessage>(this);
    StopCometOrbitTracking();
    _pendingProjectionCts?.Cancel();
    _pendingProjectionCts?.Dispose();
    _pendingProjectionCts = null;
    _transformListenerToken?.Dispose();
    _projectionListenerToken?.Dispose();
    _earthListenerToken?.Dispose();
    _sunVisibilityListenerToken?.Dispose();
    _transformSubject.Dispose();
    _projectionSubject.Dispose();
    _modeSubject.Dispose();
    _sunVisibilitySubject.Dispose();
  }
}
