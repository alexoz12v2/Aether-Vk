using System.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The grain size law as the smallest / largest grain diameter (µm): the engine keeps (D, F) with
/// D = √(min·max), F = √(max/min); the UI never shows F.
/// </summary>
public class JetGrainDiameterRangeTests
{
  [Fact]
  public void DefaultRange_Is10To1000_TheSamePhysicsAsD100F10()
  {
    var (d, f) = GrainDiameterRange.ToEngine(10, 1000);
    Assert.Equal((100f, 10f), (d, f));
    Assert.Equal((10f, 1000f), (JetGrain.Default.MinDiameterUm, JetGrain.Default.MaxDiameterUm));
    var s = new ModelSession();
    Assert.Equal((10f, 1000f), (s.MinDiameterUm, s.MaxDiameterUm));
    var j = new JetViewModel(JetGrain.Default);
    Assert.Equal((10f, 1000f), (j.MinDiameterUm, j.MaxDiameterUm));
  }

  [Theory]
  [InlineData(30, 100)]
  [InlineData(0.5, 4000)]
  [InlineData(10, 1000)]
  public void EngineConversion_RoundTripsTheRange(double min, double max)
  {
    var (d, f) = GrainDiameterRange.ToEngine(min, max);
    Assert.Equal(System.Math.Sqrt(min * max), d, 3);
    Assert.Equal(System.Math.Sqrt(max / min), f, 4);
    Assert.Equal((float)min, GrainDiameterRange.MinOf(d, f));
    Assert.Equal((float)max, GrainDiameterRange.MaxOf(d, f));
  }

  [Fact]
  public void Validate_NamesTheProblem()
  {
    Assert.Equal(GrainRangeError.None, GrainDiameterRange.Validate(30, 100));
    Assert.Equal(GrainRangeError.None, GrainDiameterRange.Validate(50, 50));
    Assert.Equal(GrainRangeError.MinAboveMax, GrainDiameterRange.Validate(100, 30));
    // the widest pair the bounds allow is exactly the engine's F = 1000 (ratio 1e6)
    Assert.Equal(GrainRangeError.None, GrainDiameterRange.Validate(0.01, 10000));
    Assert.Equal(1000f, GrainDiameterRange.ToEngine(0.01, 10000).SizeRange);
    Assert.Equal(GrainRangeError.OutOfBounds, GrainDiameterRange.Validate(0, 10));
    Assert.Equal(GrainRangeError.OutOfBounds, GrainDiameterRange.Validate(1, 1e6));
    Assert.Equal(GrainRangeError.OutOfBounds, GrainDiameterRange.Validate(double.NaN, 10));
  }

  [Fact]
  public void Session_AppliesAValidPair()
  {
    var s = new ModelSession();
    s.MinDiameterUm = 30;
    s.MaxDiameterUm = 100;
    Assert.Equal(GrainRangeError.None, s.GrainRangeError);
    Assert.Equal((30f, 100f), (s.MinDiameterUm, s.MaxDiameterUm));
    Assert.Equal(System.Math.Sqrt(3000), s.DiametreUm, 3);
    Assert.Equal(System.Math.Sqrt(100.0 / 30), s.SizeRange, 4);
  }

  [Fact]
  public void Session_MinAboveMax_IsAnErrorNotASwap_AndFixingItApplies()
  {
    var s = new ModelSession();
    float d = s.DiametreUm, f = s.SizeRange;
    s.MinDiameterUm = 2000; // above the largest (1000)
    Assert.Equal(GrainRangeError.MinAboveMax, s.GrainRangeError);
    Assert.Equal((d, f), (s.DiametreUm, s.SizeRange)); // nothing applied
    Assert.Equal((2000f, 1000f), (s.MinDiameterUm, s.MaxDiameterUm)); // the typed value is kept, no swap
    s.MaxDiameterUm = 5000;
    Assert.Equal(GrainRangeError.None, s.GrainRangeError);
    Assert.Equal((2000f, 5000f), (s.MinDiameterUm, s.MaxDiameterUm));
  }

  [Fact]
  public void Jet_MinEqualsMax_IsASingleSize()
  {
    var j = new JetViewModel(JetGrain.Default);
    j.MinDiameterUm = 40;
    j.MaxDiameterUm = 40;
    Assert.Equal(GrainRangeError.None, j.GrainRangeError);
    Assert.Equal((40f, 1f), (j.DiametreUm, j.SizeRange));
    Assert.Equal((40f, 40f), (j.MinDiameterUm, j.MaxDiameterUm));
  }

  [Fact]
  public void ExternalChange_DropsAPendingEdit()
  {
    var j = new JetViewModel(JetGrain.Default);
    j.MinDiameterUm = 5000; // invalid, pending
    Assert.Equal(GrainRangeError.MinAboveMax, j.GrainRangeError);
    j.Grain = JetGrain.Default with { DiametreUm = 20, SizeRange = 2 };
    Assert.Equal(GrainRangeError.None, j.GrainRangeError);
    Assert.Equal((10f, 40f), (j.MinDiameterUm, j.MaxDiameterUm));
  }

  [Fact]
  public void SaveRoundTrip_KeepsTheRange_ModelAndJet()
  {
    var model = new ModelSession();
    model.MinDiameterUm = 30;
    model.MaxDiameterUm = 100;
    var jet = new JetViewModel(model.DefaultGrain);
    jet.MinDiameterUm = 3;
    jet.MaxDiameterUm = 300;

    var md = SaveIndexMapping.ToDto(model);
    var jd = SaveIndexMapping.ToDto(jet);
    Assert.Equal((30f, 100f), (md.MinDiameterUm!.Value, md.MaxDiameterUm!.Value));
    Assert.Equal((3f, 300f), (jd.MinDiameterUm!.Value, jd.MaxDiameterUm!.Value));

    var json = SaveIndexIo.ToJson(new SaveIndex { Model = md, Jets = [jd] });
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    var model2 = new ModelSession();
    SaveIndexMapping.Apply(back.Model, model2);
    Assert.Equal((30f, 100f), (model2.MinDiameterUm, model2.MaxDiameterUm));
    var jet2 = SaveIndexMapping.FromDto(back.Jets[0], model2.DefaultGrain);
    Assert.Equal((3f, 300f), (jet2.MinDiameterUm, jet2.MaxDiameterUm));
  }

  [Fact]
  public void OldSave_WithoutMinMax_DerivesFromDiameterAndF()
  {
    var m = System.Text.Json.JsonSerializer.Deserialize(
      "{\"diametre_um\": 100, \"size_range\": 10}", SaveIndexJsonContext.Default.ModelDto)!;
    Assert.Null(m.MinDiameterUm);
    var session = new ModelSession();
    SaveIndexMapping.Apply(m, session);
    Assert.Equal((10f, 1000f), (session.MinDiameterUm, session.MaxDiameterUm));

    // a jet of an old save inherits the model's D and F, then states them as a range
    var j = System.Text.Json.JsonSerializer.Deserialize("{\"seed\": 3, \"diametre_um\": 50}", SaveIndexJsonContext.Default.JetDto)!;
    var jet = SaveIndexMapping.FromDto(j, session.DefaultGrain);
    Assert.Equal((5f, 500f), (jet.MinDiameterUm, jet.MaxDiameterUm));
  }

  [Fact]
  public void SizePdfAndPlot_SpanExactlyTheRange()
  {
    var (d, f) = GrainDiameterRange.ToEngine(30, 100);
    var (a, b) = DustCurves.DiameterBoundsUm(d, f);
    Assert.Equal(30.0, a, 3);
    Assert.Equal(100.0, b, 3);
    Assert.Equal(0.0, DustCurves.SizePdf(29, d, f, 3.5));
    Assert.Equal(0.0, DustCurves.SizePdf(101, d, f, 3.5));
    Assert.True(DustCurves.SizePdf(50, d, f, 3.5) > 0);
    var plot = DustCurves.SizeDistributionPlot(d, f, 3.5);
    Assert.Equal(30.0, plot.XMin, 3);
    Assert.Equal(100.0, plot.XMax, 3);
    Assert.Contains("diameter", plot.XLabel);
    Assert.Equal(plot.XMin, plot.Series[0].Points.First().X, 6);
    Assert.Equal(plot.XMax, plot.Series[0].Points.Last().X, 6);
  }

  /// <summary>
  /// β ∝ Q_pr/(ρ·s): the engine's β is the reference grain's (diameter D = √(D_min·D_max)), so the
  /// range shown next to it, β/F .. β·F, is β of the largest and of the smallest simulated grain.
  /// One grain: 10-1000 µm (D 100, F 10), 0.8 g/cm³, Q_pr 1.
  /// </summary>
  [Fact]
  public void BetaRange_SpansTheLargestToTheSmallestGrain()
  {
    // β = 3L/(16π c GM) · Q_pr / (ρ s), the engine's constant, ρ in kg/m³ and s in m
    const double k = 3 * 3.828e26 / (16 * System.Math.PI * 299_792_458.0 * 1.3271244e20);
    static double BetaOf(double diameterUm) => k * 1.0 / (800.0 * diameterUm * 0.5e-6);

    var (d, f) = GrainDiameterRange.ToEngine(10, 1000);
    double beta = BetaOf(d);
    Assert.InRange(beta, 0.01435, 0.01436);
    var (lo, hi) = GrainDiameterRange.BetaRange(beta, f);
    Assert.Equal(BetaOf(1000), lo, 9);
    Assert.Equal(BetaOf(10), hi, 9);
    Assert.Equal("0.00144 – 0.144", GrainDiameterRange.BetaRangeText(beta, f));

    // the jet's text follows its β (from the engine) and its range factor
    var jet = new JetViewModel(new JetGrain(100f, 0.8f, 1f, 0.04f, 3.5f, 10f, 2f));
    Assert.Equal("", jet.BetaRangeText); // β not computed yet
    var raised = new System.Collections.Generic.List<string?>();
    jet.PropertyChanged += (_, e) => raised.Add(e.PropertyName);
    jet.Beta = 0.0143f;
    Assert.Equal("0.00143 – 0.143", jet.BetaRangeText);
    jet.MaxDiameterUm = 100f; // 10-100 µm: F = √10
    Assert.Equal("0.00452 – 0.0452", jet.BetaRangeText);
    Assert.Contains(nameof(JetViewModel.BetaRangeText), raised);
    // one size (F = 1): a single β
    Assert.Equal("0.02 – 0.02", GrainDiameterRange.BetaRangeText(0.02, 1));
  }
}
