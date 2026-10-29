using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The reviewer's save (2026-10-29, test_assets/saves/review_67p_2021-12: committed 2021-12-12 → 28 with a
/// later proposal "2021-12-17T15:48:00" in the start box), opened in the order SaveFileService
/// uses: the Timeline must show the range the engine runs.
/// </summary>
public class ReviewSaveReproTests
{
  internal static string FixtureDir()
  {
    for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
    {
      var p = Path.Combine(d.FullName, "test_assets", "saves", "review_67p_2021-12");
      if (Directory.Exists(p)) return p;
    }
    throw new DirectoryNotFoundException("test_assets/saves/review_67p_2021-12");
  }

  [Fact]
  public async Task Open_ShowsTheCommittedRange_NotTheLaterProposal()
  {
    var (index, error) = SaveIndexIo.Read(FixtureDir());
    Assert.NotNull(index);
    Assert.Equal("2021-12-17T15:48:00", index!.Timeline.ProposedStart);
    Assert.Equal(4, index.Jets.Count);

    var rig = new AppSaveFileHostTests.Rig();
    var translation = new Mock<ITranslationService>();
    translation.Setup(t => t.CultureChanged).Returns(System.Reactive.Linq.Observable.Empty<System.Globalization.CultureInfo>());
    var vm = new TimelineTabViewModel(translation.Object, rig.Schedulers,
      AppSaveFileHostTests.Rig.Sessions(rig.Timeline).Object, AppSaveFileHostTests.Rig.Sessions(rig.Comet).Object,
      rig.TimelineService, rig.Runtime.Object, rig.Messenger);
    vm.CurrentSession = rig.Timeline;

    // SaveFileService.OpenCoreAsync: model, timeline (uncommitted), commit, timeline (committed)
    rig.Host.ApplyModel(index.Model, index.Jets);
    var spk = index.Spk.Single(s => s.Committed);
    rig.Host.ApplyTimeline(index.Timeline, committed: false);
    Assert.True(await rig.Host.CommitCometAsync(Path.Combine(FixtureDir(), spk.File), spk,
      SaveIndexMapping.ToRange(index.Timeline.CommittedStart!, index.Timeline.CommittedEnd!)));
    rig.Host.ApplyTimeline(index.Timeline, committed: true);

    Assert.True(vm.HasCommittedState);
    Assert.StartsWith("2021-12-12", vm.StartEpoch);
    Assert.StartsWith("2021-12-28", vm.EndEpoch);
    Assert.StartsWith("2021-12-12", rig.Timeline.CommittedStartEpoch);
    Assert.False(vm.IsRangeLocked);
    // all four saved jets were asked of the core (first + three later), ignited one TTL (30 d,
    // the default lead: the save has none) before the committed start, so the tail exists at play
    long ignition = 692_539_200_000_000L - 30L * 86_400L * 1_000_000L;
    ulong ps;
    rig.Runtime.Verify(r => r.AddFirstParticleSystem(It.IsAny<ParticleSystemModel>(),
      It.Is<ParticleSystemJet>(j => j.EmissionStartUs == ignition), out ps), Times.Once);
    rig.Runtime.Verify(r => r.AddParticleSystem(It.IsAny<ParticleSystemModel>(),
      It.Is<ParticleSystemJet>(j => j.EmissionStartUs == ignition), out ps), Times.Exactly(3));
    Assert.Equal(30.0, vm.DustLeadDays);
  }

  [Fact]
  public void TheLeadTime_RoundTripsThroughTheSave()
  {
    var rig = new AppSaveFileHostTests.Rig();
    rig.Host.ApplyTimeline(new TimelineDto { ProposedStart = "2021-12-12", ProposedEnd = "2021-12-28", DustLeadDays = 7.5 }, committed: false);
    Assert.Equal(7.5, rig.TimelineService.DustLeadDays);
    Assert.Equal(7.5, rig.Host.Capture().Timeline.DustLeadDays);

    // an older save without one: the default
    rig.Host.ApplyTimeline(new TimelineDto { ProposedStart = "2021-12-12", ProposedEnd = "2021-12-28" }, committed: false);
    Assert.Equal(TimelineService.MaxDustLeadDays, rig.TimelineService.DustLeadDays);
  }
}
