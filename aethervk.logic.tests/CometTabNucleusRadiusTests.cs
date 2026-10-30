using System;
using System.Reflection;
using System.Reactive.Subjects;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// Feedback item 10b: the manual nucleus radius is edited in the Comet tab. It stays in
/// <see cref="ModelSession.ManualNucleusRadiusKm"/> (saves unchanged) and reaches the native
/// runtime through the same debounced <see cref="CometConfigService.SetNucleusRadiusKm"/> push.
public class CometTabNucleusRadiusTests
{
  private sealed class Rig
  {
    public TestSchedulerProvider Schedulers { get; } = new();
    public Mock<INativeRuntimeService> Runtime { get; } = new();
    public ModelSession Model { get; } = new();
    public CometSession Comet { get; } = new();
    public CometConfigService CometConfig { get; }
    public CometTabViewModel Vm { get; }

    public Rig(float manualRadiusKm = 2f)
    {
      Model.ManualNucleusRadiusKm = manualRadiusKm;
      var translation = new Mock<ITranslationService>();
      translation.Setup(t => t.CultureChanged)
        .Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
      var cometSessions = new Mock<ITabStateService<CometSession>>();
      var cometIds = new[] { new SessionId(typeof(CometSession), 1) };
      cometSessions.Setup(s => s.ActiveSessionIds).Returns(cometIds);
      cometSessions.Setup(s => s.ObserveSession(It.IsAny<SessionId>()))
        .Returns(System.Reactive.Linq.Observable.Return(Comet));
      cometSessions.Setup(s => s.GetSession(It.IsAny<SessionId>())).Returns(Comet);
      cometSessions.Setup(s => s.ObserveSessionList()).Returns(
        new BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(cometIds));
      var modelSessions = new Mock<ITabStateService<ModelSession>>();
      var modelIds = new[] { new SessionId(typeof(ModelSession), 1) };
      modelSessions.Setup(s => s.ActiveSessionIds).Returns(modelIds);
      modelSessions.Setup(s => s.GetSession(It.IsAny<SessionId>())).Returns(Model);
      CometConfig = new CometConfigService(Runtime.Object, Schedulers);
      var breadcrumbs = new BreadcrumbService();
      var storage = new Mock<ILocalStorageService>();
      var jpl = new HorizonJplService(null, breadcrumbs, storage.Object);
      var timeline = new TimelineService(Runtime.Object, Schedulers, CometConfig, breadcrumbs);
      Vm = new CometTabViewModel(
        translation.Object, Schedulers, cometSessions.Object, jpl, CometConfig, timeline,
        breadcrumbs, storage.Object, Runtime.Object, modelSessions.Object,
        new Mock<ICometMessenger>().Object, new Mock<ITabFactory>().Object);
    }

    /// A committed comet without the SPK round trip: the native radius push needs one.
    public void MarkCommitted()
    {
      var field = typeof(CometConfigService).GetField("_isCommittedSubject", BindingFlags.NonPublic | BindingFlags.Instance)!;
      ((BehaviorSubject<bool>)field.GetValue(CometConfig)!).OnNext(true);
      Schedulers.MainThread.Start();
    }

    public void AdvanceMs(int ms) => Schedulers.MainThread.AdvanceBy(TimeSpan.FromMilliseconds(ms).Ticks);
  }

  [Fact]
  public void RadiusEdit_WritesTheModelSession_AndPushesNativeOnce()
  {
    var rig = new Rig();
    rig.MarkCommitted();
    rig.Runtime.Invocations.Clear();

    // a slider drag: three edits inside the 250 ms debounce
    rig.Vm.ManualNucleusRadiusKm = 3f;
    rig.Vm.ManualNucleusRadiusKm = 4f;
    rig.Vm.ManualNucleusRadiusKm = 4.5f;
    Assert.Equal(4.5f, rig.Model.ManualNucleusRadiusKm);
    rig.Runtime.Verify(r => r.UpdateCometNucleusRadius(It.IsAny<float>()), Times.Never);

    rig.AdvanceMs(300);
    rig.Runtime.Verify(r => r.UpdateCometNucleusRadius(It.IsAny<float>()), Times.Once);
    rig.Runtime.Verify(r => r.UpdateCometNucleusRadius(4.5f), Times.Once);
    Assert.Equal(4.5f, rig.CometConfig.NucleusRadiusKmValue);
  }

  [Fact]
  public void Vm_ReadsTheModelSessionRadius_WithoutAPush()
  {
    var rig = new Rig(manualRadiusKm: 3.25f);
    Assert.Equal(3.25f, rig.Vm.ManualNucleusRadiusKm);
    Assert.Equal(3.25f, rig.Vm.ManualNucleusRadiusKmNullable);
    rig.AdvanceMs(300);
    rig.Runtime.Verify(r => r.UpdateCometNucleusRadius(It.IsAny<float>()), Times.Never);

    // a savefile open replaces the model session's values: the tab re-reads them, no push
    rig.Model.ManualNucleusRadiusKm = 6f;
    rig.Model.NotifyRestored();
    Assert.Equal(6f, rig.Vm.ManualNucleusRadiusKm);
    rig.AdvanceMs(300);
    rig.Runtime.Verify(r => r.UpdateCometNucleusRadius(It.IsAny<float>()), Times.Never);
  }

  [Fact]
  public void ClearedRadius_FallsBackToTheCatalogueRadius()
  {
    var rig = new Rig();
    rig.Vm.ManualNucleusRadiusKmNullable = null;
    Assert.Equal(0f, rig.Model.ManualNucleusRadiusKm);
    Assert.Null(rig.Vm.ManualNucleusRadiusKmNullable);
    Assert.True(rig.Vm.IsNucleusRadiusUnknown);

    rig.Comet.NucleusRadiusKm = 2.2f;
    Assert.False(rig.Vm.IsNucleusRadiusUnknown);
  }
}
