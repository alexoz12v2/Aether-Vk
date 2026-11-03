using System.Runtime.InteropServices;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The per-jet dynamic model (insolation weighting, the size law along the orbit, the pre-start
/// source, the dust lifetime): its savefile mapping, its native record and the lead-time maximum.
/// An older save has none of the fields, and must come back with the toggles off, the lifetime at
/// 30 days and no q table, which the engine treats exactly as before.
/// </summary>
public class JetDynamicModelTests
{
  [Fact]
  public void OldJetDto_GivesTheTogglesOff_ThirtyDays_NoTable()
  {
    var j = SaveIndexMapping.FromDto(new JetDto { Seed = 1 });
    Assert.False(j.InsolationWeighted);
    Assert.False(j.OrbitSizeLaw);
    Assert.False(j.SpkBeforeStart);
    Assert.Equal(DustLifetime.DefaultDays, j.LifetimeDays);
    Assert.True(j.SizeQTable.IsEmpty);
    Assert.Same(SizeQTable.Empty, j.SizeQTable);
  }

  [Fact]
  public void NewFields_RoundTripThroughTheSaveDto()
  {
    var j = new JetViewModel(new JetGrain(100f, 1f, 1f, 0.04f, 3.5f, 10f, 0.5f))
    {
      InsolationWeighted = true,
      OrbitSizeLaw = true,
      SpkBeforeStart = true,
      LifetimeDays = 120f,
      SizeQTable = SizeQTable.From(new[] { new SizeQPoint(-200, 2.5f), new SizeQPoint(0, 3.8f), new SizeQPoint(40, 4.2f) }),
    };
    var dto = SaveIndexMapping.ToDto(j);
    Assert.True(dto.InsolationWeighted);
    Assert.True(dto.OrbitSizeLaw);
    Assert.True(dto.SpkBeforeStart);
    Assert.Equal(120f, dto.LifetimeDays);
    Assert.Equal(new[] { -200f, 0f, 40f }, dto.SizeQTableDays);
    Assert.Equal(new[] { 2.5f, 3.8f, 4.2f }, dto.SizeQTableQ);

    var json = SaveIndexIo.ToJson(new SaveIndex { Jets = [dto] });
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    var r = SaveIndexMapping.FromDto(back.Jets[0]);
    Assert.True(r.InsolationWeighted);
    Assert.True(r.OrbitSizeLaw);
    Assert.True(r.SpkBeforeStart);
    Assert.Equal(120f, r.LifetimeDays);
    Assert.Equal(j.SizeQTable, r.SizeQTable);
  }

  [Fact]
  public void EmptyTable_IsSavedAsAbsentArrays()
  {
    var dto = SaveIndexMapping.ToDto(new JetViewModel(new JetGrain(100f, 1f, 1f, 0.04f, 3.5f, 10f, 0.5f)));
    Assert.Null(dto.SizeQTableDays);
    Assert.Null(dto.SizeQTableQ);
    Assert.Equal(DustLifetime.DefaultDays, dto.LifetimeDays);
  }

  [Theory]
  [InlineData(0f)]
  [InlineData(0.5f)]
  [InlineData(400f)]
  [InlineData(float.NaN)]
  public void OutOfRangeLifetime_ReadsAsTheDefault(float days)
  {
    Assert.Equal(DustLifetime.DefaultDays, DustLifetime.Effective(days));
    Assert.Equal(DustLifetime.DefaultDays, SaveIndexMapping.FromDto(new JetDto { LifetimeDays = days }).LifetimeDays);
  }

  [Fact]
  public void EmissionSteps_AtThirtyDays_AreTheBandLengthsOver256()
  {
    Assert.Equal("2.8 h / 19.7 h / 6.6 d", DustLifetime.EmissionStepsText(30f));
    var steps = DustLifetime.EmissionStepsDays(30f);
    Assert.Equal(30.0 / 256, steps[0], 12);
    Assert.Equal(7 * 30.0 / 256, steps[1], 12);
    Assert.Equal(56 * 30.0 / 256, steps[2], 12);
    Assert.Equal("0.4 h / 2.6 h / 21.0 h", DustLifetime.EmissionStepsText(4f));
  }

  [Fact]
  public void SizeQTable_SortsClampsAndCuts()
  {
    var rows = new System.Collections.Generic.List<SizeQPoint>();
    for (int i = 0; i < 12; i++)
      rows.Add(new SizeQPoint(100 - 10 * i, 7f));
    rows.Add(new SizeQPoint(float.NaN, 3f));
    var t = SizeQTable.From(rows);
    Assert.Equal(SizeQTable.MaxRows, t.Count);
    Assert.All(t.Rows, r => Assert.Equal(SizeQTable.MaxQ, r.Q));
    for (int i = 1; i < t.Count; i++)
      Assert.True(t.Rows[i - 1].DaysFromPerihelion <= t.Rows[i].DaysFromPerihelion);
  }

  [Fact]
  public void SizeQTable_InterpolatesAndHoldsOutside()
  {
    Assert.Equal(3.5f, SizeQTable.Empty.Evaluate(10, 3.5f));
    var one = SizeQTable.From(new[] { new SizeQPoint(0, 2f) });
    Assert.Equal(2f, one.Evaluate(-1000, 3.5f));
    Assert.Equal(2f, one.Evaluate(1000, 3.5f));
    var t = SizeQTable.From(new[] { new SizeQPoint(-100, 2f), new SizeQPoint(100, 4f) });
    Assert.Equal(2f, t.Evaluate(-500, 3.5f));
    Assert.Equal(3f, t.Evaluate(0, 3.5f), 6);
    Assert.Equal(3.5f, t.Evaluate(50, 3.5f), 6);
    Assert.Equal(4f, t.Evaluate(500, 3.5f));
  }

  [Fact]
  public void NativeRecord_HasTheEngineSize()
  {
    Assert.Equal(ParticleSystemDTO.NativeSize, Marshal.SizeOf<ParticleSystemDTO>());
    Assert.Equal(184, ParticleSystemDTO.NativeSize);
  }

  [Fact]
  public void JetViewModel_TableRows_ReflectTheTable()
  {
    var j = new JetViewModel(new JetGrain(100f, 1f, 1f, 0.04f, 3.5f, 10f, 0.5f));
    Assert.True(j.CanAddSizeQRow);
    for (int i = 0; i < SizeQTable.MaxRows; i++)
      j.AddSizeQRowCommand.Execute(null);
    Assert.Equal(SizeQTable.MaxRows, j.SizeQRows.Count);
    Assert.False(j.CanAddSizeQRow);
    j.AddSizeQRowCommand.Execute(null);
    Assert.Equal(SizeQTable.MaxRows, j.SizeQRows.Count);
    j.RemoveSizeQRowCommand.Execute(j.SizeQRows[0]);
    Assert.Equal(SizeQTable.MaxRows - 1, j.SizeQRows.Count);
    Assert.True(j.CanAddSizeQRow);
    Assert.Equal(SizeQTable.MaxRows - 1, j.SizeQTable.Count);
  }

  [Fact]
  public void MaxDustLead_FollowsTheLongestLifetime()
  {
    var rt = new Moq.Mock<INativeRuntimeService>();
    var sched = new Mocks.TestSchedulerProvider();
    var tl = new TimelineService(rt.Object, sched, new CometConfigService(rt.Object, sched), new BreadcrumbService());
    Assert.Equal(TimelineService.MaxDustLeadDays, tl.MaxDustLeadDaysNow);
    tl.MaxDustLeadDaysNow = 90;
    tl.DustLeadDays = 60;
    Assert.Equal(60, tl.DustLeadDays);
    tl.MaxDustLeadDaysNow = 10;
    Assert.Equal(10, tl.DustLeadDays);
    tl.DustLeadDays = 45;
    Assert.Equal(10, tl.DustLeadDays);
    tl.MaxDustLeadDaysNow = 0;
    Assert.Equal(TimelineService.MaxDustLeadDays, tl.MaxDustLeadDaysNow);
  }

  [Fact]
  public void SpkDownload_StartsALeadBeforeTheRange()
  {
    // 2021-12-12T00:00 TDB
    var range = new TimeRange(0, 692_539_200_000_000_000UL, 0, 693_921_600_000_000_000UL);
    var (s0, e0) = CometTabViewModel.SpkDownloadDays(range);
    var (s30, e30) = CometTabViewModel.SpkDownloadDays(range, 30);
    Assert.Equal(e0, e30);
    Assert.Equal(System.DateTime.Parse(s0, System.Globalization.CultureInfo.InvariantCulture).AddDays(-30),
      System.DateTime.Parse(s30, System.Globalization.CultureInfo.InvariantCulture));
    var widened = CometTabViewModel.RangeWithLead(range, 30);
    Assert.Equal(range.EndNs, widened.EndNs);
    Assert.True(widened.StartNs < range.StartNs);
    Assert.Equal(range, CometTabViewModel.RangeWithLead(range, 0));
  }
}
