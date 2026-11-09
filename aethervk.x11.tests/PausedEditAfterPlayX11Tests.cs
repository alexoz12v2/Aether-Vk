using System.Globalization;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.UI;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Controls.Primitives;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// The model that would not show after a Play: a jet edited while paused after some playback
/// used to restore the scene snapshot taken at Play, which rewound the clock and left the engine
/// without the comet's state, so the dust stayed unbuilt until a seek or a reopen. Now a paused
/// edit stays at the current date and the engine rebuilds the dust there. This test drives the
/// real window: the committed 67P review fixture opened, Play for a few seconds, Pause, then a
/// new ejection speed typed into jet #1's velocity box with real X key events. The engine's
/// <see cref="INativeRuntimeService.DustStats"/> must report the system complete again within
/// 30 s, with its jet state available, and the clock (the engine's, the viewport epoch label and
/// the Timeline tab's epoch box) must read the same as right after the Pause. No network: the
/// fixture is a committed save.
/// </summary>
[Collection("X11")]
public sealed class PausedEditAfterPlayX11Tests : IDisposable
{
  private readonly ITestOutputHelper _output;

  public PausedEditAfterPlayX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  public void Dispose()
  {
  }

  private void Step(string s)
  {
    _output.WriteLine($"── {s}");
    Console.Error.WriteLine($"[PausedEditAfterPlayX11Tests] {s}"); // progress in the run log if the run hangs
  }

  // ── The Model tab, driven by real clicks ────────────────────────────────────────────────

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

  private static ListBoxItem JetRow(AppRig rig, JetViewModel jet)
  {
    var list = JetList(rig);
    ListBoxItem? row = null;
    Assert.True(WaitFor(() => (row = Ui(() => list.GetVisualDescendants().OfType<ListBoxItem>()
      .FirstOrDefault(r => ReferenceEquals(r.DataContext, jet) && r.IsVisible))) is not null, 5_000), $"the row of jet #{jet.DisplayIndex}");
    return row!;
  }

  /// <summary>The Model tab shown, its Dust section open, jet #1 selected (a click on its row).</summary>
  private static void ShowJetOne(AppRig rig, ModelTabViewModel model)
  {
    rig.Click(rig.Require<TextBlock>("Tab.ModelTab"));
    ExpandByClick(rig, rig.Require<Expander>("DustSection"), "Dust section");
    if (Ui(() => model.SelectedJet) is null)
    {
      ClickInView(rig, JetRow(rig, Ui(() => model.Jets![0])));
      Assert.True(WaitFor(() => Ui(() => model.SelectedJet) is not null, 5_000), "jet #1 selected");
    }
    var box = rig.Require<TextBox>("JetVelocityBox");
    Ui(() => box.BringIntoView());
    Settle(400);
  }

  /// <summary>Real keystrokes into a box: select all, type, Enter (the box's commit).</summary>
  private static void TypeInto(AppRig rig, Control box, string text)
  {
    ClickInView(rig, box);
    XDo("key --clearmodifiers ctrl+a");
    XDo($"type --delay 40 {text}");
    XDo("key Return");
    Settle(300);
  }

  // ── What the engine and the window say ──────────────────────────────────────────────────

  private static bool Complete(IReadOnlyList<DustTierStats> tiers) =>
    tiers.Count > 0 && tiers.All(t => t.CaughtUp && !t.JetUnavailable);

  /// <summary>The engine clock (TDB parts) as one number of microseconds since J2000.</summary>
  private static long EngineClockUs(INativeRuntimeService runtime)
  {
    Assert.True(runtime.GetSimulationClock(out short centuries, out ulong ns), "the engine clock");
    const long UsPerDay = 86_400L * 1_000_000L;
    return centuries * 36525L * UsPerDay + (long)(ns / 1000UL);
  }

  /// <summary>Every viewport overlay's epoch label (its view model's string), in window order.</summary>
  private static string[] OverlayEpochLabels() =>
    Ui(() => (Application.Current!.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?.Windows
      .OfType<OverlayWindow>()
      .Select(w => (w.DataContext as ViewportOverlayViewModel)?.CurrentEpochString ?? string.Empty)
      .ToArray() ?? []);

  private void Log(string label, INativeRuntimeService runtime, TimelineTabViewModel timeline)
  {
    var tiers = runtime.DustStats();
    _output.WriteLine($"{label}: engine clock {EngineClockUs(runtime)} µs, epoch box '{Ui(() => timeline.CurrentEpochText)}', "
      + $"overlay [{string.Join(" | ", OverlayEpochLabels())}], playing {Ui(() => timeline.IsPlaying)}");
    for (int i = 0; i < tiers.Count; i++)
    {
      var t = tiers[i];
      _output.WriteLine(string.Create(CultureInfo.InvariantCulture,
        $"  tier {i}: band {t.BandMinS / 86_400.0:0.###}–{t.BandMaxS / 86_400.0:0.###} d, live {t.LiveClusters}/{t.Capacity}, "
        + $"youngest {t.YoungestAgeS / 3600:0.##} h, oldest {t.OldestAgeS / 3600:0.##} h, caught up {t.CaughtUp}, "
        + $"unlit windows {t.UnlitWindows}, jet unavailable {t.JetUnavailable}"));
    }
  }

  /// One sequential test: the core is a process singleton and each step builds on the last.
  [SkippableFact]
  public void VelocityTypedAfterPlayWhilePaused_RebuildsTheDustAtTheSameDate()
  {
    var fixture = AetherVk.TestAssets.TestAssets.Require("saves/review_67p_2021-12");
    using var rig = new AppRig();
    var model = rig.Tab<ModelTabViewModel>();
    var timeline = rig.Tab<TimelineTabViewModel>();
    var runtime = rig.Host.Services.GetRequiredService<INativeRuntimeService>();

    // ── 1. the fixture ──────────────────────────────────────────────────────────────────────
    Step("1: open the fixture folder");
    rig.Dialogs.Answer(rig.CopyOfSave(fixture));
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.Contains("review_67p_2021-12", StringComparison.Ordinal), 90_000),
      $"fixture open, title: {rig.XTitle}");
    Assert.True(WaitFor(() => Ui(() => (model.Jets?.Count ?? 0) > 0 && model.Jets!.All(j => j.NativePsId != 0)), 60_000),
      "the fixture's jets in the engine");
    Assert.True(WaitFor(() => Complete(runtime.DustStats()), 120_000), "the opened save's dust built");
    ShowJetOne(rig, model);
    Log("opened", runtime, timeline);

    // the save opens ~70 s before its range's end, where Play has nowhere to go: seek back first
    Step("1b: seek to 2021-12-26 00:00:00");
    long clockAtOpen = EngineClockUs(runtime);
    Ui(() =>
    {
      timeline.CurrentEpochText = "2021-12-26 00:00:00";
      timeline.SeekToEpochCommand.Execute(null);
    });
    Assert.True(WaitFor(() => EngineClockUs(runtime) < clockAtOpen, 30_000), "the seek moved the clock back");
    Assert.True(WaitFor(() => Complete(runtime.DustStats()), 120_000), "the dust built after the seek");
    Log("seeked", runtime, timeline);

    // ── 2. Play a few seconds, then Pause ──────────────────────────────────────────────────
    Step("2: Play");
    Ui(() =>
    {
      timeline.SelectedSpeed ??= TimelineTabViewModel.SimulationSpeed.OneHourPerSec;
      timeline.IsPlaying = true;
    });
    Assert.True(WaitFor(() => Ui(() => model.IsSimulationRunning), 10_000), "the simulation running");
    long clockAtPlay = EngineClockUs(runtime);
    Settle(3_000);
    Step("2: Pause");
    Ui(() => timeline.IsPlaying = false);
    Assert.True(WaitFor(() => !Ui(() => model.IsSimulationRunning), 10_000), "the simulation paused");
    Assert.True(WaitFor(() => Complete(runtime.DustStats()), 60_000), "the dust caught up after the pause");
    long clockAtPause = EngineClockUs(runtime);
    Assert.True(clockAtPause > clockAtPlay, $"the clock did not move while playing ({clockAtPlay} → {clockAtPause} µs)");
    // the labels settle on the paused clock
    Settle(1_500);
    var overlayAtPause = OverlayEpochLabels();
    var boxAtPause = Ui(() => timeline.CurrentEpochText);
    Log("paused", runtime, timeline);

    // ── 3. a new ejection speed typed into jet #1 ───────────────────────────────────────────
    Step("3: type a velocity");
    var jet = Ui(() => model.SelectedJet!);
    float before = Ui(() => jet.StartVelocityMeanMs);
    string typed = Math.Abs(before - 7.5f) < 0.01f ? "12.5" : "7.5";
    float want = float.Parse(typed, CultureInfo.InvariantCulture);
    var box = rig.Require<TextBox>("JetVelocityBox");
    TypeInto(rig, box, typed);
    Assert.True(WaitFor(() => Math.Abs(Ui(() => jet.StartVelocityMeanMs) - want) < 0.01f, 5_000),
      $"the velocity box took '{typed}' (view model {Ui(() => jet.StartVelocityMeanMs)} m/s)");

    // ── 4. the engine rebuilds the dust at the paused date ─────────────────────────────────
    Step("4: the dust complete again, the clock unchanged");
    Settle(1_500); // past the 250 ms forwarding throttle: the re-emit requested
    bool complete = WaitFor(() => Complete(runtime.DustStats()) && runtime.DustStats().Any(t => t.LiveClusters > 0), 30_000);
    Log("edited", runtime, timeline);
    Assert.True(complete, "the dust did not complete within 30 s of the paused edit");
    Assert.False(Ui(() => model.IsSimulationRunning), "the edit started the simulation");
    Assert.Equal(clockAtPause, EngineClockUs(runtime));
    Settle(1_500);
    Assert.Equal(boxAtPause, Ui(() => timeline.CurrentEpochText));
    Assert.Equal(overlayAtPause, OverlayEpochLabels());
    Assert.All(Ui(() => model.Jets!.Select(j => j.NativeError).ToArray()), e => Assert.Null(e));
    Assert.Equal(0, rig.Prompt.Asked);
  }
}
