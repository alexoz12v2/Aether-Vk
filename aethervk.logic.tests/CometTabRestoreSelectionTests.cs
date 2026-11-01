using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Net;
using System.Net.Http;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// A reopened save holds a committed comet: the Comet tab restores the comet / SPK record selection
/// from the session, so Download &amp; Commit, "Recommit with this range" and "Run at this date" work
/// without a new search. Old saves without those fields keep the selection empty.
/// </summary>
public class CometTabRestoreSelectionTests
{
  private sealed class RecordingHandler : HttpMessageHandler
  {
    public readonly List<string> Urls = [];
    protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
    {
      string url = request.RequestUri!.ToString();
      Urls.Add(url);
      if (url.Contains("sbdb.api", StringComparison.Ordinal))
        return Task.FromResult(new HttpResponseMessage(HttpStatusCode.OK)
        {
          Content = new StringContent("""{"object":{"spkid":"1000012","des":"67P","fullname":"67P/Churyumov-Gerasimenko"}}""")
        });
      return Task.FromResult(new HttpResponseMessage(HttpStatusCode.ServiceUnavailable));
    }
  }

  private sealed class Rig
  {
    public readonly TestSchedulerProvider Schedulers = new();
    public readonly Mock<INativeRuntimeService> Runtime = new();
    public readonly TimelineSession Session = new();
    public readonly RecordingHandler Http = new();
    public readonly HorizonJplService Jpl;
    public readonly TimelineService Timeline;
    public readonly CometTabViewModel Vm;

    public Rig(CometSession comet)
    {
      var translation = new Mock<ITranslationService>();
      translation.Setup(t => t.CultureChanged).Returns(Observable.Empty<System.Globalization.CultureInfo>());
      translation.Setup(t => t.GetString(It.IsAny<string>())).Returns<string>(k => k);
      var cometSessions = new Mock<ITabStateService<CometSession>>();
      cometSessions.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(CometSession), 1) });
      cometSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(Observable.Return(comet));
      cometSessions.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
      var stream = new BehaviorSubject<TimelineSession>(Session);
      var timelineSessions = new Mock<ITabStateService<TimelineSession>>();
      timelineSessions.Setup(s => s.ActiveSessionIds).Returns(new ObservableCollection<SessionId> { new SessionId(typeof(TimelineSession), 1) });
      timelineSessions.Setup(s => s.ObserveSessionList()).Returns(new BehaviorSubject<IReadOnlyList<SessionId>>(new List<SessionId>()));
      timelineSessions.Setup(x => x.GetSession(It.IsAny<SessionId>())).Returns(Session);
      timelineSessions.Setup(x => x.ObserveSession(It.IsAny<SessionId>())).Returns(stream);
      timelineSessions.Setup(x => x.UpdateSession(It.IsAny<SessionId>(), It.IsAny<Action<TimelineSession>>()))
        .Callback<SessionId, Action<TimelineSession>>((_, a) => { a(Session); stream.OnNext(Session); });
      Runtime.Setup(r => r.GetSimulationClock(out It.Ref<short>.IsAny, out It.Ref<ulong>.IsAny)).Returns(false);
      var config = new CometConfigService(Runtime.Object, Schedulers);
      var breadcrumbs = new BreadcrumbService();
      var storage = new Mock<ILocalStorageService>();
      storage.Setup(s => s.GetDownloadsPath(It.IsAny<string>())).Returns<string>(f => System.IO.Path.Combine(System.IO.Path.GetTempPath(), f));
      storage.Setup(s => s.SavePersistentAsync(It.IsAny<string>(), It.IsAny<byte[]>())).Returns(Task.CompletedTask);
      Jpl = new HorizonJplService(new ConsoleService(new Mock<IUiThreadDispatcher>().Object), breadcrumbs, storage.Object) { _httpClient = new HttpClient(Http) };
      Timeline = new TimelineService(Runtime.Object, Schedulers, config, breadcrumbs);
      Vm = new CometTabViewModel(translation.Object, Schedulers, cometSessions.Object, Jpl, config, Timeline,
        breadcrumbs, storage.Object, Runtime.Object, new Mock<ITabStateService<ModelSession>>().Object,
        new ForwardingCometMessenger(), new Mock<ITabFactory>().Object, timelineSessions.Object, timeScale: new TimeScaleService(new FakeTimeScaleConverter()));
      Flush();
      Vm.ReferenceOrbitIndex = 0;
    }

    public void Flush()
    {
      for (int i = 0; i < 4; i++) Schedulers.MainThread.AdvanceBy(1);
    }

    public bool SpkRequested() => Http.Urls.Exists(u => u.Contains("EPHEM_TYPE", StringComparison.Ordinal));
  }

  private static CometSession Committed(string? spkFile = null) => new()
  {
    IsAlmanacLoaded = true,
    CommittedDesignation = "67P",
    CommittedFullName = "67P/Churyumov-Gerasimenko",
    CommittedSpkRecordId = "90000703",
    SpkId = 1000012,
    CommittedSpkFilePath = spkFile,
  };

  [Fact]
  public void CommittedSession_RestoresASelection_ThatCanRecommit()
  {
    var rig = new Rig(Committed());

    Assert.Equal("67P/Churyumov-Gerasimenko", rig.Vm.SelectedComet?.Name);
    Assert.Equal("67P", rig.Vm.SelectedComet!.PrimaryDesignation);
    Assert.Equal("90000703", rig.Vm.SelectedSpkRecord?.RecordId);
    // the combo boxes' collections hold the same items
    Assert.Contains(rig.Vm.SelectedComet, rig.Vm.FilteredSearchResults);
    Assert.Contains(rig.Vm.SelectedSpkRecord, rig.Vm.SpkRecords);
    // no network call on restore
    Assert.Empty(rig.Http.Urls);

    rig.Vm.Range.StartEpoch = "2021-12-20T00:00:00";
    rig.Vm.Range.EndEpoch = "2021-12-27T00:14:47";
    rig.Flush();
    Assert.True(rig.Vm.DownloadAndCommitCommand.CanExecute(null));
  }

  [Fact]
  public async Task RecommitRequest_OnARestoredSession_Commits_WithoutADownloadInsideTheCoverage()
  {
    string file = System.IO.Path.GetTempFileName();
    try
    {
      var rig = new Rig(Committed(file));
      rig.Vm.NoteSpkCoverage(CometTabViewModel.SpkCoverageOf("67P", "90000703", file, "2021-12-12", "2021-12-28"));
      rig.Vm.Range.StartEpoch = "2021-12-20T00:00:00";
      rig.Vm.Range.EndEpoch = "2021-12-27T00:14:47";
      rig.Flush();

      var done = rig.Timeline.RequestRecommitAsync();
      rig.Flush();
      await done; // the engine mock refuses the range, so the request is not committed

      Assert.False(rig.SpkRequested(), string.Join("\n", rig.Http.Urls));
      Assert.Contains(rig.Http.Urls, u => u.Contains("sbdb.api", StringComparison.Ordinal));
      rig.Runtime.Verify(r => r.SetEpochRange(It.IsAny<short>(), It.IsAny<ulong>(), It.IsAny<short>(), It.IsAny<ulong>()), Times.Once);
    }
    finally
    {
      if (System.IO.File.Exists(file)) System.IO.File.Delete(file);
    }
  }

  [Fact]
  public void OldSession_WithoutTheFields_RestoresAnEmptySelection()
  {
    var old = new CometSession { IsAlmanacLoaded = true, CommittedFullName = "67P/Churyumov-Gerasimenko", CommittedSpkRecordId = "90000703" };
    var rig = new Rig(old);

    Assert.Null(rig.Vm.SelectedComet);
    Assert.Null(rig.Vm.SelectedSpkRecord);
    Assert.Empty(rig.Jpl.SpkRecordsData);
    Assert.Empty(rig.Http.Urls);
  }
}
