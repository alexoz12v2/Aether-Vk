using System;
using System.Reactive.Subjects;
using System.Reflection;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

internal delegate bool GetJetLitFractionReturns(ulong id, out double litFraction, out TimeRange range);

/// <summary>
/// R1b: a jet whose surface site is never lit over the committed range emits no dust (the
/// reviewer's 67P save had such a site and nothing said so). The jet row carries
/// <see cref="JetViewModel.LitWarning"/> with the range, refreshed after a commit and a jet edit,
/// cleared when the site is lit or the comet is no longer committed.
/// </summary>
public class JetLitWarningTests
{
  /// the reviewer's committed range, TDB parts since J2000 (2021-12-12 → 2021-12-28)
  private static readonly TimeRange Range = new(0, 692_539_200_000_000_000UL, 0, 693_921_600_000_000_000UL);

  private sealed class Rig
  {
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TestSchedulerProvider Schedulers = new();
    public readonly CometConfigService CometConfig;
    public readonly ModelTabViewModel Vm;
    /// the core's answer: lit fraction of every jet (null: no answer, as without a committed comet)
    public double? LitFraction = 1.0;

    public Rig()
    {
      var translation = new Mock<ITranslationService>();
      translation.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
      var modelSessions = new Mock<ITabStateService<ModelSession>>();
      modelSessions.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(ModelSession), 1) });
      modelSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new ModelSession()));
      modelSessions.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
      var cometSessions = new Mock<ITabStateService<CometSession>>();
      cometSessions.Setup(s => s.ActiveSessionIds).Returns(new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
      cometSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(System.Reactive.Linq.Observable.Return(new CometSession()));
      cometSessions.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(new System.Collections.Generic.List<SessionId>()));
      var dispatcher = new Mock<IUiThreadDispatcher>();
      dispatcher.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());

      ulong psId = 7;
      Runtime
        .Setup(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out psId))
        .Returns(new ParticleSystemComputedProperties(0.1f, 1f));
      var computed = new ParticleSystemComputedProperties(0.1f, 1f);
      Runtime
        .Setup(r => r.ModifyParticleSystem(It.IsAny<ulong>(), It.IsAny<ParticleSystemModel>(), It.IsAny<ParticleSystemJet>(), out computed))
        .Returns(true);
      Runtime
        .Setup(r => r.GetJetLitFraction(It.IsAny<ulong>(), out It.Ref<double>.IsAny, out It.Ref<TimeRange>.IsAny))
        .Returns(new GetJetLitFractionReturns((ulong _, out double f, out TimeRange range) =>
        {
          f = LitFraction ?? 0;
          range = Range;
          return LitFraction is not null;
        }));

      CometConfig = new CometConfigService(Runtime.Object, Schedulers);
      var breadcrumbs = new BreadcrumbService();
      var timeline = new TimelineService(Runtime.Object, Schedulers, CometConfig, breadcrumbs);
      Vm = new ModelTabViewModel(
        translation.Object, Schedulers, modelSessions.Object, cometSessions.Object, CometConfig,
        Runtime.Object, dispatcher.Object, new Mock<ICometMessenger>().Object,
        new Mock<IPlatformWindowService>().Object, timeline);
      // deliver the initial "not committed" emission first (it would reset NativePsId later)
      Schedulers.MainThread.Start();
    }

    /// A committed comet without the SPK round trip (the Model tab's commit subscription fires).
    public void SetCommitted(bool committed)
    {
      var field = typeof(CometConfigService).GetField("_isCommittedSubject", BindingFlags.NonPublic | BindingFlags.Instance)!;
      ((BehaviorSubject<bool>)field.GetValue(CometConfig)!).OnNext(committed);
      Schedulers.MainThread.Start();
    }

    /// the model session's default manual radius (2 km) places the jet
    public JetViewModel AddJet()
    {
      Vm.IsCometCommitted = true;
      Vm.AddJetCommand.Execute(null);
      return Assert.Single(Vm.Jets!);
    }
  }

  private static string ExpectedWarning() => string.Format(
    JetNativeSync.NeverLitFormat,
    TimelineTabViewModel.FormatEpoch(Range.StartCenturies, Range.StartNs),
    TimelineTabViewModel.FormatEpoch(Range.EndCenturies, Range.EndNs));

  [Fact]
  public void NeverLitSite_WarnsWithTheCommittedRange()
  {
    var rig = new Rig { LitFraction = 0.0 };
    var jet = rig.AddJet();

    Assert.Equal(7UL, jet.NativePsId);
    Assert.Equal(ExpectedWarning(), jet.LitWarning);
    Assert.Contains("2021-12-12", jet.LitWarning);
    Assert.Contains("2021-12-28", jet.LitWarning);
    Assert.Null(jet.NativeError);
  }

  [Fact]
  public void LitSite_HasNoWarning()
  {
    var rig = new Rig { LitFraction = 0.37 };
    var jet = rig.AddJet();

    Assert.Null(jet.LitWarning);
  }

  [Fact]
  public void Warning_RefreshesAfterAJetEdit()
  {
    var rig = new Rig { LitFraction = 0.0 };
    var jet = rig.AddJet();
    Assert.NotNull(jet.LitWarning);

    // moved to a lit latitude: the debounced modify re-asks the core
    rig.LitFraction = 0.2;
    jet.LatitudeDeg = 12f;
    rig.Schedulers.Background.AdvanceBy(TimeSpan.FromMilliseconds(300).Ticks);
    Assert.Null(jet.LitWarning);

    // and back into the polar night
    rig.LitFraction = 0.0;
    jet.LatitudeDeg = -88f;
    rig.Schedulers.Background.AdvanceBy(TimeSpan.FromMilliseconds(300).Ticks);
    Assert.Equal(ExpectedWarning(), jet.LitWarning);
  }

  [Fact]
  public void Warning_RefreshesAfterACommit_AndClearsWhenDecommitted()
  {
    var rig = new Rig { LitFraction = 0.5 };
    var jet = rig.AddJet();
    Assert.Null(jet.LitWarning);

    // a new committed range leaves the site in the dark: the commit re-asks every jet
    rig.LitFraction = 0.0;
    rig.SetCommitted(true);
    Assert.Equal(7UL, jet.NativePsId);
    Assert.Equal(ExpectedWarning(), jet.LitWarning);

    // no comet: nothing to warn about
    rig.SetCommitted(false);
    Assert.Equal(0UL, jet.NativePsId);
    Assert.Null(jet.LitWarning);
  }

  [Fact]
  public void NoAnswerFromTheCore_NoWarning()
  {
    var rig = new Rig { LitFraction = null };
    var jet = rig.AddJet();

    Assert.Null(jet.LitWarning);
  }
}
