using System;
using System.Numerics;
using System.Reactive.Concurrency;
using System.Runtime.InteropServices;
using AetherVk.Logic.Services;
using Microsoft.Reactive.Testing;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// Tests for <see cref="CometPositionTrackerService"/> and the
/// <c>ExternalState::CometPositionSnapshot</c> fix that ensures the comet position
/// reaches <see cref="CameraService"/> before entering <see cref="CameraMode.CometOrbiting"/>.
/// </summary>
[Collection("Sequential")]
public class CometPositionSnapshotTests
{
  // ── Helpers ────────────────────────────────────────────────────────────────

  private static ISchedulerProvider MakeSchedulers(IScheduler scheduler)
  {
    var sp = new Mock<ISchedulerProvider>();
    sp.Setup(s => s.MainThread).Returns(ImmediateScheduler.Instance);
    sp.Setup(s => s.Background).Returns(scheduler);
    return sp.Object;
  }

  // Mutable mirror used only for test marshalling (CCometPositionSnapshotDTO is readonly)
  [StructLayout(LayoutKind.Sequential)]
  private struct MutableCometPositionSnapshotDTO
  {
    public int    SpkId;
    public int    Pad;
    public double PosX;
    public double PosY;
    public double PosZ;
  }

  /// <summary>Marshals a mutable snapshot DTO to unmanaged memory and invokes
  /// <paramref name="callback"/> with the pointer, then frees the allocation.</summary>
  private static void InvokeWithSnapshot(
    Action<nint> callback,
    int spkId, double posX, double posY, double posZ)
  {
    var dto = new MutableCometPositionSnapshotDTO
    {
      SpkId = spkId, Pad = 0,
      PosX  = posX, PosY = posY, PosZ = posZ,
    };
    int  size = Marshal.SizeOf<MutableCometPositionSnapshotDTO>();
    nint ptr  = Marshal.AllocHGlobal(size);
    try
    {
      Marshal.StructureToPtr(dto, ptr, false);
      callback(ptr);
    }
    finally
    {
      Marshal.FreeHGlobal(ptr);
    }
  }

  /// <summary>
  /// Creates a wired-up <see cref="CometPositionTrackerService"/> and captures the
  /// <c>CometPositionSnapshot</c> callback registered with the mock runtime.
  /// </summary>
  private static (
    CometPositionTrackerService tracker,
    Action<nint> snapshotCallback
  ) BuildTracker()
  {
    var scheduler = new TestScheduler();
    var schedulers = MakeSchedulers(scheduler);
    var runtime = new Mock<INativeRuntimeService>();

    Action<nint>? capturedSnapshot = null;
    runtime
      .Setup(r => r.RegisterExternalStateListener(
        ExternalStateType.CometPositionSnapshot, It.IsAny<Action<nint>>()))
      .Callback<ExternalStateType, Action<nint>>((_, h) => capturedSnapshot = h)
      .Returns(Mock.Of<IDisposable>());
    runtime
      .Setup(r => r.RegisterExternalStateListener(
        It.IsNotIn(ExternalStateType.CometPositionSnapshot), It.IsAny<Action<nint>>()))
      .Returns(Mock.Of<IDisposable>());

    var breadcrumb  = new BreadcrumbService(Mock.Of<IUiThreadDispatcher>());
    var cometConfig = new CometConfigService(runtime.Object, schedulers);
    var timeline    = new TimelineService(runtime.Object, schedulers, cometConfig, breadcrumb);
    var tracker     = new CometPositionTrackerService(runtime.Object, schedulers, timeline);

    Assert.NotNull(capturedSnapshot);
    return (tracker, capturedSnapshot!);
  }

  // ── Tests ──────────────────────────────────────────────────────────────────

  /// <summary>
  /// Receiving a <c>CometPositionSnapshot</c> external-state event must push the comet
  /// position to <c>LastKnownCometPosition</c> immediately.
  /// </summary>
  [Fact]
  public void CometPositionSnapshot_UpdatesLastKnownCometPosition()
  {
    var (tracker, snapshotCallback) = BuildTracker();

    // Before any snapshot the tracker emits the default position (1, 0, 0).
    var beforeSnapshot = tracker.LastKnownCometPosition;
    Assert.NotNull(beforeSnapshot);
    Assert.Equal(new Vector3(1f, 0f, 0f), beforeSnapshot.Value);

    double px = -1.9201, py = -5.1530, pz = -0.2039;
    InvokeWithSnapshot(snapshotCallback, 1000012, px, py, pz);

    // After snapshot: position must be updated to the real comet coordinates (f64 → f32 cast)
    var known = tracker.LastKnownCometPosition;
    Assert.NotNull(known);
    Assert.True(Math.Abs(known.Value.X - (float)px) < 1e-4f,
      $"Expected X≈{px:F4} got {known.Value.X}");
    Assert.True(Math.Abs(known.Value.Y - (float)py) < 1e-4f,
      $"Expected Y≈{py:F4} got {known.Value.Y}");
    Assert.True(Math.Abs(known.Value.Z - (float)pz) < 1e-4f,
      $"Expected Z≈{pz:F4} got {known.Value.Z}");
    // Sanity-check: value has changed from the default
    Assert.NotEqual(beforeSnapshot.Value, known.Value);
  }

  /// <summary>
  /// When <c>CometPositionSnapshot</c> fires BEFORE the user enters
  /// <c>CometOrbiting</c> mode, <see cref="CameraService.SetCameraMode"/> must animate
  /// toward the snapshot position — not the default (1, 0, 0) AU.
  /// </summary>
  [Fact]
  public void CometOrbiting_AfterSnapshot_AnimatesToSnapshotPosition_NotDefault()
  {
    var testScheduler = new TestScheduler();
    var schedulers = MakeSchedulers(testScheduler);
    var runtime = new Mock<INativeRuntimeService>();

    Action<nint>? snapshotCallback = null;
    runtime
      .Setup(r => r.RegisterExternalStateListener(
        ExternalStateType.CometPositionSnapshot, It.IsAny<Action<nint>>()))
      .Callback<ExternalStateType, Action<nint>>((_, h) => snapshotCallback = h)
      .Returns(Mock.Of<IDisposable>());
    runtime
      .Setup(r => r.RegisterExternalStateListener(
        It.IsNotIn(ExternalStateType.CometPositionSnapshot), It.IsAny<Action<nint>>()))
      .Returns(Mock.Of<IDisposable>());
    runtime
      .Setup(r => r.RegisterSimulationListener(
        It.IsAny<ulong>(), It.IsAny<ulong>(), It.IsAny<Action<nint>>()))
      .Returns(Mock.Of<IDisposable>());
    runtime.Setup(r => r.EarthEntityId).Returns(42UL);

    var breadcrumb  = new BreadcrumbService(Mock.Of<IUiThreadDispatcher>());
    var cometConfig = new CometConfigService(runtime.Object, schedulers);
    var timeline    = new TimelineService(runtime.Object, schedulers, cometConfig, breadcrumb);
    var tracker     = new CometPositionTrackerService(runtime.Object, schedulers, timeline);
    var camera      = new CameraService(runtime.Object, schedulers, tracker, cometConfig, breadcrumb, Mock.Of<ICometMessenger>());
    camera.OnViewportReady(77UL, 800, 600);

    // Pre-commit almanac so CometOrbiting mode is allowed
    var field = typeof(CometConfigService)
      .GetField("_isCommittedSubject",
        System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
    var subject = (System.Reactive.Subjects.BehaviorSubject<bool>?)field?.GetValue(cometConfig);
    subject?.OnNext(true);

    // Fire snapshot: comet is at ~(-1.92, -5.15, -0.20) AU — clearly not (1,0,0)
    Assert.NotNull(snapshotCallback);
    InvokeWithSnapshot(snapshotCallback!, 1000012, -1.920, -5.153, -0.204);

    runtime.Invocations.Clear();

    // Enter CometOrbiting
    camera.SetCameraMode(CameraMode.CometOrbiting);

    // The animation target X must be clearly negative (comet is at ~-1.92 AU, not +1 AU)
    runtime.Verify(r => r.AddCameraAnimation(
        77UL,
        It.Is<AnimationTarget>(t => t.Pos.X < -0.5f)),
      Times.Once,
      "Camera must animate to the snapshot position (X < -0.5 AU), not the default +1 AU.");
  }
}
