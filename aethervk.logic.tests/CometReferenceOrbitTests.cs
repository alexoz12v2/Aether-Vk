using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// The reference orbit (SBDB solution vs re-osculated at the start epoch) is a required choice
/// before committing a comet, and it is locked while a comet is committed.
public class CometReferenceOrbitTests
{
  private static CometTabViewModel BuildVm() => BuildVm(new CometSession());

  private static CometTabViewModel BuildVm(CometSession session)
  {
    var translation = new Mock<ITranslationService>();
    translation.Setup(t => t.CultureChanged)
      .Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
    var schedulers = new TestSchedulerProvider();
    var cometSessions = new Mock<ITabStateService<CometSession>>();
    cometSessions.Setup(s => s.ActiveSessionIds).Returns(
      new System.Collections.ObjectModel.ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
    cometSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>()))
      .Returns(System.Reactive.Linq.Observable.Return(session));
    cometSessions.Setup(s => s.ObserveSessionList()).Returns(
      new System.Reactive.Subjects.BehaviorSubject<System.Collections.Generic.IReadOnlyList<SessionId>>(
        new System.Collections.Generic.List<SessionId>()));
    var modelSessions = new Mock<ITabStateService<ModelSession>>();
    var runtime = new Mock<INativeRuntimeService>();
    var cometConfig = new CometConfigService(runtime.Object, schedulers);
    var breadcrumbs = new BreadcrumbService();
    var storage = new Mock<ILocalStorageService>();
    var jpl = new HorizonJplService(null, breadcrumbs, storage.Object);
    var timeline = new TimelineService(runtime.Object, schedulers, cometConfig, breadcrumbs);
    return new CometTabViewModel(
      translation.Object, schedulers, cometSessions.Object, jpl, cometConfig, timeline,
      breadcrumbs, storage.Object, runtime.Object, modelSessions.Object,
      new Mock<ICometMessenger>().Object, new Mock<ITabFactory>().Object);
  }

  [Fact]
  public void Commit_IsDisabled_UntilAReferenceOrbitIsChosen()
  {
    var vm = BuildVm();
    vm.SelectedComet = new CometSearchResult { Name = "67P/Churyumov-Gerasimenko", PrimaryDesignation = "67P" };
    vm.SelectedSpkRecord = new SpkRecordItem { RecordId = "90000703", Name = "67P" };
    vm.HasProposedTimeline = true;

    Assert.Equal(-1, vm.ReferenceOrbitIndex);
    Assert.Null(vm.ReferenceOrbitMode);
    Assert.False(vm.DownloadAndCommitCommand.CanExecute(null));

    vm.ReferenceOrbitIndex = 1;
    Assert.Equal(ReferenceOrbitMode.OsculatingAtStart, vm.ReferenceOrbitMode);
    Assert.True(vm.DownloadAndCommitCommand.CanExecute(null));

    vm.ReferenceOrbitIndex = 0;
    Assert.Equal(ReferenceOrbitMode.Sbdb, vm.ReferenceOrbitMode);
  }

  /// Tabs are scoped: closing and reopening the Comet tab builds a new VM over the same session.
  [Fact]
  public void ReferenceOrbit_IsRestored_ByANewVmOverTheSameSession()
  {
    var session = new CometSession();
    var first = BuildVm(session);
    first.ReferenceOrbitIndex = 1;
    Assert.Equal(1, session.ReferenceOrbitIndex);

    var reopened = BuildVm(session);
    Assert.Equal(1, reopened.ReferenceOrbitIndex);
    Assert.Equal(ReferenceOrbitMode.OsculatingAtStart, reopened.ReferenceOrbitMode);
  }

  /// The rest of the form (rotational model, committed comet) is restored as well.
  [Fact]
  public void Form_IsRestored_ByANewVmOverTheSameSession()
  {
    var session = new CometSession
    {
      RotPoleRaDeg = 69.3,
      RotPoleDecDeg = 64.1,
      RotRateDegDay = 696.5,
      RotBodyFixedOrientationIndex = 1,
      IsAlmanacLoaded = true,
      CommittedFullName = "67P/Churyumov-Gerasimenko",
      CommittedSpkRecordId = "90000703",
    };
    var vm = BuildVm(session);
    Assert.Equal(69.3, vm.PoleRaDeg);
    Assert.Equal(64.1, vm.PoleDecDeg);
    Assert.Equal(696.5, vm.RotRateDegDay);
    Assert.Equal(1, vm.BodyFixedOrientationIndex);
    Assert.Equal("67P/Churyumov-Gerasimenko", vm.CommittedCometName);
    Assert.Equal("90000703", vm.CommittedSpkRecordId);
  }

  [Fact]
  public void ReferenceOrbit_IsLocked_WhileCommitted()
  {
    var vm = BuildVm();
    Assert.True(vm.IsReferenceOrbitSelectable);
    var raised = false;
    vm.PropertyChanged += (_, e) => raised |= e.PropertyName == nameof(vm.IsReferenceOrbitSelectable);
    vm.IsAlmanacCommitted = true;
    Assert.False(vm.IsReferenceOrbitSelectable);
    Assert.True(raised);
  }
}
