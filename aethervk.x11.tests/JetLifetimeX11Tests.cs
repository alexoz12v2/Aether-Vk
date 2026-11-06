using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// The jet's dust lifetime T sets the emission steps, and nothing else does: the engine emits each
/// of its three age tiers ([0, T), [T, 8T), [8T, 64T)) over 256 windows, so a tier's step is its
/// band / 256. The Model tab shows those steps read-only next to the lifetime box. This test types
/// into the real box with real X key events and checks three things agree for every value: the
/// text the panel shows, <see cref="DustLifetime.EmissionStepsDays"/>, and the engine's own tier
/// bands read back through <see cref="INativeRuntimeService.DustStats"/>. The view model and the
/// engine are only read, never written. No network: the committed 67P review fixture (a save
/// written before the lifetime existed, so it opens at 30 days).
/// </summary>
[Collection("X11")]
public sealed class JetLifetimeX11Tests : IDisposable
{
  private readonly ITestOutputHelper _output;

  public JetLifetimeX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  public void Dispose()
  {
  }

  private const double DaySeconds = 86_400.0;

  private static string FixtureIndex() =>
    Path.Combine(AppRig.RepoRoot(), "test_assets", "saves", "review_67p_2021-12", SaveIndex.FileName);

  private void Step(string s)
  {
    _output.WriteLine($"── {s}");
    Console.Error.WriteLine($"[JetLifetimeX11Tests] {s}"); // progress in the run log if the run hangs
  }

  private static bool Close(double a, double b, double rel = 1e-9) =>
    Math.Abs(a - b) <= rel * Math.Max(Math.Abs(a), Math.Abs(b));

  // ── The Model tab, driven by real clicks ────────────────────────────────────────────────

  /// <summary>Scrolls <paramref name="c"/> into view and clicks its centre.</summary>
  private static void ClickInView(AppRig rig, Control c)
  {
    Ui(() => c.BringIntoView());
    Settle(400);
    rig.Click(c);
  }

  private static void ExpandByClick(AppRig rig, Expander e, string what)
  {
    if (Ui(() => e.IsExpanded)) return;
    var toggle = Ui(() => e.GetVisualDescendants().OfType<ToggleButton>().First());
    ClickInView(rig, toggle);
    Assert.True(WaitFor(() => Ui(() => e.IsExpanded), 5_000), $"{what} expanded");
  }

  private static ListBox JetList(AppRig rig) =>
    Ui(() => rig.Main.GetVisualDescendants().OfType<ListBox>().First(l => l.Name == "JetListBox"));

  /// <summary>The row showing <paramref name="jet"/> (the list keeps recycled containers in the tree).</summary>
  private static ListBoxItem JetRow(AppRig rig, JetViewModel jet)
  {
    var list = JetList(rig);
    ListBoxItem? row = null;
    Assert.True(WaitFor(() => (row = Ui(() => list.GetVisualDescendants().OfType<ListBoxItem>()
      .FirstOrDefault(r => ReferenceEquals(r.DataContext, jet) && r.IsVisible))) is not null, 5_000), $"the row of jet #{jet.DisplayIndex}");
    return row!;
  }

  /// <summary>The Model tab shown, its Dust section open.</summary>
  private static void ShowModelTab(AppRig rig)
  {
    rig.Click(rig.Require<TextBlock>("Tab.ModelTab"));
    ExpandByClick(rig, rig.Require<Expander>("DustSection"), "Dust section");
  }

  /// <summary>
  /// Removes jets with each row's "−" button until one is left: <c>dust_stats</c> sums the
  /// clusters over the scene's systems but reports one system's age band, so the bands are only
  /// that jet's when it is the only one.
  /// </summary>
  private static void KeepOneJet(AppRig rig, ModelTabViewModel model)
  {
    while (Ui(() => model.Jets!.Count) > 1)
    {
      int n = Ui(() => model.Jets!.Count);
      var row = JetRow(rig, Ui(() => model.Jets![^1]));
      var remove = Ui(() => row.GetVisualDescendants().OfType<Button>().First(b => (b.Content as string) == "−"));
      ClickInView(rig, remove);
      Assert.True(WaitFor(() => Ui(() => model.Jets!.Count) == n - 1, 10_000), $"jet {n} removed");
    }
  }

  /// <summary>Jet #1 selected (a click on its row) and its "Advanced: dynamic model" open.</summary>
  private static void OpenJetDynamics(AppRig rig, ModelTabViewModel model)
  {
    if (Ui(() => model.SelectedJet) is null)
    {
      ClickInView(rig, JetRow(rig, Ui(() => model.Jets![0])));
      Assert.True(WaitFor(() => Ui(() => model.SelectedJet) is not null, 5_000), "jet #1 selected");
    }
    ExpandByClick(rig, rig.Require<Expander>("JetDynamicsExpander"), "Advanced: dynamic model");
    var steps = rig.Require<TextBlock>("JetEmissionStepsText");
    Ui(() => steps.BringIntoView());
    Settle(400);
  }

  /// <summary>Real keystrokes into the lifetime box: select all, type, Enter (the box's commit).</summary>
  private static void TypeInto(AppRig rig, Control box, string text)
  {
    ClickInView(rig, box);
    XDo("key --clearmodifiers ctrl+a");
    XDo($"type --delay 40 {text}");
    XDo("key Return");
    Settle(300);
  }

  // ── What the engine and the panel say ───────────────────────────────────────────────────

  private static IReadOnlyList<DustTierStats> EngineTiers(INativeRuntimeService runtime) => runtime.DustStats();

  private static bool BandsAre(IReadOnlyList<DustTierStats> tiers, double ttlDays)
  {
    if (tiers.Count != DustLifetime.TierBands.Length) return false;
    double ttl = ttlDays * DaySeconds;
    for (int i = 0; i < tiers.Count; i++)
      if (!Close(tiers[i].BandMinS, DustLifetime.TierBands[i].MinTtl * ttl)
          || !Close(tiers[i].BandMaxS, DustLifetime.TierBands[i].MaxTtl * ttl))
        return false;
    return true;
  }

  private void Log(string label, AppRig rig, INativeRuntimeService runtime)
  {
    var tiers = EngineTiers(runtime);
    var box = rig.Find<TextBox>("JetLifetimeDays")!;
    var steps = rig.Find<TextBlock>("JetEmissionStepsText")!;
    _output.WriteLine($"{label}: box '{Ui(() => box.Text)}', steps '{Ui(() => steps.Text)}'");
    for (int i = 0; i < tiers.Count; i++)
    {
      var t = tiers[i];
      _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"  tier {i}: band {t.BandMinS / DaySeconds:0.###}–{t.BandMaxS / DaySeconds:0.###} d, "
        + $"band/256 {(t.BandMaxS - t.BandMinS) / DustLifetime.WindowsPerTtl / 3600:0.####} h, live {t.LiveClusters}/{t.Capacity}, "
        + $"youngest {t.YoungestAgeS / 3600:0.##} h, oldest {t.OldestAgeS / 3600:0.##} h, caught up {t.CaughtUp}, "
        + $"unlit windows {t.UnlitWindows}, jet unavailable {t.JetUnavailable}"));
    }
  }

  /// <summary>"2.8 h" / "6.6 d" back to days.</summary>
  private static double ParseStep(string s)
  {
    var parts = s.Trim().Split(' ');
    Assert.Equal(2, parts.Length);
    double v = double.Parse(parts[0], CultureInfo.InvariantCulture);
    return parts[1] switch
    {
      "h" => v / 24.0,
      "d" => v,
      _ => throw new Xunit.Sdk.XunitException($"not a step: '{s}'"),
    };
  }

  /// <summary>
  /// Box, view model, panel text and engine bands all say <paramref name="ttlDays"/>: the shown
  /// steps are the engine's band / 256 within the display rounding, and the formula's exactly.
  /// </summary>
  private void AssertAgree(AppRig rig, ModelTabViewModel model, INativeRuntimeService runtime, double ttlDays,
    string? expectedText, string label)
  {
    var box = rig.Find<TextBox>("JetLifetimeDays")!;
    var stepsText = rig.Find<TextBlock>("JetEmissionStepsText")!;
    bool bands = WaitFor(() => BandsAre(EngineTiers(runtime), ttlDays), 30_000);
    Log(label, rig, runtime);
    _output.WriteLine($"  view model: lifetime {Ui(() => model.SelectedJet!.LifetimeDays)} d, simulation running {Ui(() => model.IsSimulationRunning)}");
    Assert.True(bands, $"{label}: the engine's tier bands are not {{0,1,8}}–{{1,8,64}} × {ttlDays} d");
    Assert.Equal(ttlDays.ToString("0.#", CultureInfo.CurrentCulture), Ui(() => box.Text));
    Assert.Equal((float)ttlDays, Ui(() => model.SelectedJet!.LifetimeDays));
    string text = Ui(() => stepsText.Text) ?? string.Empty;
    if (expectedText is not null)
      Assert.Equal(expectedText, text);

    var shown = text.Split(" / ");
    var formula = DustLifetime.EmissionStepsDays((float)ttlDays);
    var tiers = EngineTiers(runtime);
    Assert.Equal(tiers.Count, shown.Length);
    for (int i = 0; i < tiers.Count; i++)
    {
      double engineDays = (tiers[i].BandMaxS - tiers[i].BandMinS) / DustLifetime.WindowsPerTtl / DaySeconds;
      Assert.True(Close(engineDays, formula[i]), $"{label}: tier {i} engine step {engineDays} d vs formula {formula[i]} d");
      double shownDays = ParseStep(shown[i]);
      double tol = shown[i].EndsWith(" h", StringComparison.Ordinal) ? 0.05 / 24.0 : 0.05;
      Assert.True(Math.Abs(shownDays - engineDays) <= tol + 1e-12,
        $"{label}: tier {i} shows '{shown[i]}', the engine emits every {engineDays * 24:0.####} h");
    }
  }

  /// One sequential test: the core is a process singleton and each step builds on the last.
  [SkippableFact]
  public void LifetimeBox_SetsTheEmissionStepsTheEngineUses()
  {
    Skip.IfNot(File.Exists(FixtureIndex()), $"fixture missing: {FixtureIndex()}");
    using var rig = new AppRig();
    var model = rig.Tab<ModelTabViewModel>();
    var runtime = rig.Host.Services.GetRequiredService<INativeRuntimeService>();

    // ── 1. the fixture, one jet ─────────────────────────────────────────────────────────────
    Step("1: open the fixture folder");
    rig.Dialogs.Answer(FixtureIndex());
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.Contains("review_67p_2021-12", StringComparison.Ordinal), 90_000),
      $"fixture open, title: {rig.XTitle}");
    Assert.True(WaitFor(() => Ui(() => (model.Jets?.Count ?? 0) > 0 && model.Jets!.All(j => j.NativePsId != 0)), 60_000),
      "the fixture's jets in the engine");
    Step($"1: keep one of {Ui(() => model.Jets!.Count)} jets");
    ShowModelTab(rig);
    KeepOneJet(rig, model);
    OpenJetDynamics(rig, model);

    // ── 2. a save from before the lifetime: 30 days ─────────────────────────────────────────
    Step("2: the fixture's lifetime");
    AssertAgree(rig, model, runtime, 30, "2.8 h / 19.7 h / 6.6 d", "fixture");

    // ── 3. typed values, the clamped ones and a refused one ─────────────────────────────────
    var box = rig.Require<TextBox>("JetLifetimeDays");
    foreach (var (typed, days, text) in new (string, double, string?)[]
    {
      ("1", 1, null), // 5.25 h is a rounding midpoint: checked against the engine only
      ("365", 365, "1.4 d / 10.0 d / 79.8 d"),
      ("90", 90, "8.4 h / 2.5 d / 19.7 d"),
      ("0", 1, null),
      ("400", 365, "1.4 d / 10.0 d / 79.8 d"),
    })
    {
      Step($"3: type '{typed}'");
      TypeInto(rig, box, typed);
      AssertAgree(rig, model, runtime, days, text, $"typed '{typed}'");
    }

    Step("3: type 'abc' (not a number: the box reverts)");
    TypeInto(rig, box, "abc");
    Settle(1500); // past the 250 ms forwarding throttle
    AssertAgree(rig, model, runtime, 365, "1.4 d / 10.0 d / 79.8 d", "typed 'abc'");

    // ── 4. the steps themselves cannot be typed into ────────────────────────────────────────
    Step("4: the emission steps are read-only");
    var steps = rig.Require<TextBlock>("JetEmissionStepsText");
    // the lifetime grid: the box on row 0, the steps on row 2, and nothing editable on row 2
    var grid = Ui(() => steps.Parent as Grid);
    Assert.NotNull(grid);
    int stepsRow = Ui(() => Grid.GetRow(steps));
    Assert.Empty(Ui(() => grid!.Children.Where(c => Grid.GetRow(c) == stepsRow)
      .SelectMany(c => c.GetVisualDescendants().Prepend(c)).OfType<TextBox>().ToList()));
    ClickInView(rig, steps);
    XDo("type --delay 40 5");
    XDo("key Return");
    Settle(1500);
    AssertAgree(rig, model, runtime, 365, "1.4 d / 10.0 d / 79.8 d", "typed on the steps");

    // ── 5. the engine emits on that grid ────────────────────────────────────────────────────
    Step("5: the dust history at 90 days");
    TypeInto(rig, box, "90");
    AssertAgree(rig, model, runtime, 90, "8.4 h / 2.5 d / 19.7 d", "typed '90' again");
    Assert.True(WaitFor(() => EngineTiers(runtime).All(t => t.CaughtUp), 120_000), "every tier caught up");
    Log("caught up", rig, runtime);
    var tiers = EngineTiers(runtime);
    double step0 = (tiers[0].BandMaxS - tiers[0].BandMinS) / DustLifetime.WindowsPerTtl;
    Assert.True(tiers[0].LiveClusters > 0, "no young dust");
    Assert.True(tiers[0].YoungestAgeS <= step0 * (1 + 1e-9),
      $"the youngest dust is {tiers[0].YoungestAgeS / 3600:0.##} h old, more than one {step0 / 3600:0.##} h window");
    Assert.True(tiers[^1].OldestAgeS <= tiers[^1].BandMaxS * (1 + 1e-9),
      $"the oldest dust ({tiers[^1].OldestAgeS / DaySeconds:0.#} d) is older than its band ({tiers[^1].BandMaxS / DaySeconds:0.#} d)");

    // ── 6. the lifetime survives Save As and a reopen ───────────────────────────────────────
    Step("6: 365 days, Save As");
    TypeInto(rig, box, "365");
    AssertAgree(rig, model, runtime, 365, "1.4 d / 10.0 d / 79.8 d", "before saving");
    var avk = Path.Combine(rig.SavesDir, "lifetime.avk");
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Save As...");
    Assert.True(WaitFor(() => File.Exists(avk) && rig.XTitle.EndsWith("— lifetime", StringComparison.Ordinal), 90_000),
      $"the .avk save, title: {rig.XTitle}");
    using (var zip = ZipFile.OpenRead(avk))
    using (var s = zip.GetEntry(SaveIndex.FileName)!.Open())
    using (var doc = JsonDocument.Parse(s))
    {
      var jets = doc.RootElement.GetProperty("jets");
      Assert.Equal(1, jets.GetArrayLength());
      Assert.Equal(365f, jets[0].GetProperty("lifetime_days").GetSingle());
    }

    Step("6: File › New, then reopen");
    rig.Menu("File", "New");
    Assert.True(WaitFor(() => Ui(() => (model.Jets?.Count ?? 0) == 0), 60_000), "a fresh session");
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.Contains("lifetime", StringComparison.Ordinal)
      && Ui(() => model.Jets?.Count == 1 && model.Jets[0].NativePsId != 0), 90_000), $"reopened, title: {rig.XTitle}");
    ShowModelTab(rig);
    OpenJetDynamics(rig, model);
    AssertAgree(rig, model, runtime, 365, "1.4 d / 10.0 d / 79.8 d", "reopened");
    Assert.Equal(0, rig.Prompt.Asked);
  }
}
