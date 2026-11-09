using System.Globalization;
using OpenQA.Selenium;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// A jet edited while paused after some playback: the review save opened, a seek back into its
/// range (it opens about 70 s before the range's end, where Play has nowhere to go), Play for a
/// few seconds, Pause, then a new ejection speed typed into jet #1's velocity box and committed
/// with Enter. The edit used to restore the scene taken at Play, which rewound the clock and left
/// the dust unbuilt until a Play, a seek or a reopen. Now, with no Play and no reopen, the
/// viewport's date label stays where the Pause left it and the viewport redraws visibly (not
/// black) within 10 s. The Windows twin of the X11 test of the same name. Screenshots go to
/// TestResults/e2e/paused-edit-after-play/.
/// </summary>
public sealed class PausedEditAfterPlayTests(ITestOutputHelper output)
{
  /// <summary>The seek target, inside the save's committed range (2021-12-12 → 2021-12-28).</summary>
  private const string SeekText = "2021-12-26 00:00:00";

  private static readonly DateTime SeekTarget = new(2021, 12, 26);

  /// <summary>A frame counts as changed past this share of moved cells (PlaybackTests' seek threshold).</summary>
  private const double ChangedFrame = 0.02;

  /// <summary>A rendered viewport's mean luminance (0 on a black one; the sky gradient alone is above this).</summary>
  private const double NotBlack = 0.005;

  /// <summary>How long the paused edit may take to show.</summary>
  private static readonly TimeSpan RedrawWithin = TimeSpan.FromSeconds(10);

  /// <summary>After a pause the dust holds its preview for 2 s, then the full render draws and settles.</summary>
  private static readonly TimeSpan PauseSettle = TimeSpan.FromSeconds(6);

  private void Step(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");

  [SkippableFact]
  public void PausedEditAfterPlay_KeepsTheDateAndRedrawsTheDust()
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var (artifacts, saveDir) = ReviewSave.Prepare("paused-edit-after-play");

    var s = AppSession.Launch(Path.Combine(artifacts, "app.log"), diagnosticPanels: false);
    int? exitCode = null;
    try
    {
      // ── 1. the review save, the comet on screen ──────────────────────────────────────────
      Step("open the review save");
      ReviewSave.Open(s, Path.Combine(saveDir, "aethervk_index.json"));
      var vp = Ui.WaitFor(() => ViewportProbe.Discover(s) is { Count: > 0 } v ? v[0] : null,
        TimeSpan.FromSeconds(30), "a 3D viewport");
      output.WriteLine(vp.ToString());
      // the save's FITS billboard window sits over the comet: the screen would show it, not the dust
      ReviewSave.HideTheImage(s);
      Ui.WaitUntil(() => ViewportProbe.Mean(ViewportProbe.Sample(vp)) > NotBlack, TimeSpan.FromSeconds(60), "the viewport to draw");
      Shot(vp, artifacts, "00-opened.png");

      // ── 2. seek back into the range, as a user does: the epoch box, then Go ─────────────
      Step($"seek to {SeekText}");
      s.SelectTab("TimelineTab");
      ReviewSave.Pause(s);
      s.TypeInto(s.Find("CurrentEpochBox"), SeekText, enter: false);
      s.Click("SeekToEpochButton");
      Ui.WaitUntil(() => ReviewSave.Near(ReviewSave.Epoch(s), SeekTarget), TimeSpan.FromSeconds(30), $"the clock at {SeekText}");
      Ui.WaitUntil(() => ViewportProbe.SameMinute(ViewportProbe.LabelEpoch(EpochLabel(s, vp)), SeekTarget),
        TimeSpan.FromSeconds(60), $"the viewport's date label at {SeekText}");
      Thread.Sleep(PauseSettle);
      output.WriteLine($"after the seek: label \"{EpochLabel(s, vp)}\", epoch box \"{s.Find("CurrentEpochBox").Text}\"");
      Shot(vp, artifacts, "01-seeked.png");

      // ── 3. Play 3 s, Pause; the paused label and picture ─────────────────────────────────
      Step("play 3 s, pause");
      ReviewSave.Play(s);
      Thread.Sleep(3000);
      ReviewSave.Pause(s);
      var boxAtPause = StableText(() => s.Find("CurrentEpochBox").Text, "the epoch box after the pause");
      Assert.True(ReviewSave.ParseEpoch(boxAtPause) > SeekTarget + TimeSpan.FromMinutes(1),
        $"the clock did not move while playing (epoch box \"{boxAtPause}\")");
      Thread.Sleep(PauseSettle);
      var labelAtPause = StableText(() => EpochLabel(s, vp), "the viewport's date label after the pause");
      var paused = ViewportProbe.Sample(vp);
      var (ringPaused, skyPaused) = ViewportProbe.CentreRingAndSky(vp);
      output.WriteLine($"paused: label \"{labelAtPause}\", epoch box \"{boxAtPause}\", mean {ViewportProbe.Mean(paused):F4}, "
        + $"centre ring {ringPaused:F4} vs sky {skyPaused:F4}");
      Shot(vp, artifacts, "02-paused.png");

      // ── 4. a new velocity typed into jet #1's box, committed with Enter ─────────────────
      Step("type a new velocity into jet #1");
      s.SelectTab("ModelTab");
      s.Expand("DustSection", "AddJetButton");
      Ui.WaitUntil(() => s.Items("JetListBox").Count > 0, TimeSpan.FromSeconds(60), "the jet rows");
      s.ScrollIntoView(s.Items("JetListBox")[0]).RealClick();
      var box = s.ScrollIntoView(Ui.WaitFor(() => s.TryFind("JetVelocityBox"), TimeSpan.FromSeconds(10), "jet #1's velocity box"));
      double before = Ui.ParseNum(box.Text);
      // a change the picture shows: the save's jets eject at tens of m/s
      double wanted = Math.Round(before * 0.7, 1);
      if (Math.Abs(wanted - before) < 2.5)
        wanted = before + 2.5;
      output.WriteLine($"jet #1 velocity {Ui.Num(before)} → {Ui.Num(wanted)} m/s");
      s.TypeInto(box, Ui.Num(wanted));
      Ui.WaitUntil(() => Math.Abs(Ui.ParseNum(s.Find("JetVelocityBox").Text) - wanted) < 0.01,
        TimeSpan.FromSeconds(5), $"the velocity box to hold {Ui.Num(wanted)}");
      var committed = DateTime.UtcNow;

      // ── 5. same date, a new picture, not black: no Play, no reopen ───────────────────────
      Step("the date holds and the viewport redraws");
      double changed = 0, mean = 0;
      int polls = 0;
      while (true)
      {
        Thread.Sleep(500);
        polls++;
        var label = EpochLabel(s, vp);
        Assert.True(label == labelAtPause, $"the paused edit moved the viewport's date: \"{labelAtPause}\" → \"{label}\"");
        var frame = ViewportProbe.Sample(vp);
        (changed, mean) = (FrameMetrics.Changed(paused, frame), ViewportProbe.Mean(frame));
        if (polls % 4 == 0)
          output.WriteLine($"  +{(DateTime.UtcNow - committed).TotalSeconds:0.0} s: changed {ViewportProbe.Pct(changed)}, mean {mean:F4}");
        if (changed > ChangedFrame && mean > NotBlack)
          break;
        if (DateTime.UtcNow - committed > RedrawWithin)
        {
          Shot(vp, artifacts, "03-not-redrawn.png");
          Assert.Fail($"no visible redraw within {RedrawWithin.TotalSeconds:0} s of the paused edit "
            + $"(changed {ViewportProbe.Pct(changed)}, mean {mean:F4})");
        }
      }
      output.WriteLine($"redrawn after {(DateTime.UtcNow - committed).TotalSeconds:0.0} s: changed {ViewportProbe.Pct(changed)}, mean {mean:F4}");
      Shot(vp, artifacts, "03-edited.png");

      // the rebuilt picture, once its full render settles: still the paused date, still drawn
      Thread.Sleep(PauseSettle);
      var settled = ViewportProbe.Sample(vp);
      var (ring, sky) = ViewportProbe.CentreRingAndSky(vp);
      double changedSettled = FrameMetrics.Changed(paused, settled), meanSettled = ViewportProbe.Mean(settled);
      var labelAfter = EpochLabel(s, vp);
      Shot(vp, artifacts, "04-edited-settled.png");
      s.App.SaveScreenshot(Path.Combine(artifacts, "04-window.png"));
      output.WriteLine($"settled: label \"{labelAfter}\", changed {ViewportProbe.Pct(changedSettled)}, mean {meanSettled:F4}, "
        + $"centre ring {ring:F4} vs sky {sky:F4}");
      Assert.Equal(labelAtPause, labelAfter);
      Assert.True(meanSettled > NotBlack, $"the viewport is black after the paused edit (mean luminance {meanSettled:F4})");
      Assert.True(changedSettled > ChangedFrame,
        $"the settled picture after the edit barely differs from the paused one ({ViewportProbe.Pct(changedSettled)})");
      if (ringPaused > skyPaused)
        Assert.True(ring > sky, $"the comet's surroundings went dark after the edit (ring {ring:F4}, sky {sky:F4}; paused {ringPaused:F4} vs {skyPaused:F4})");

      s.SelectTab("TimelineTab");
      Assert.False(ReviewSave.IsPlaying(s), "the paused edit started the playback");
      Assert.Equal(boxAtPause, s.Find("CurrentEpochBox").Text);

      // ── 6. close ─────────────────────────────────────────────────────────────────────────
      Step("close");
      s.Process.CloseMainWindow();
      if (!s.Process.WaitForExit(5_000))
        try
        {
          // the edit made the save dirty: discard it
          Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("Discard").FirstOrDefault(e => e.Displayed),
            TimeSpan.FromSeconds(10), "the save prompt").Click();
        }
        catch (Exception e) when (e is TimeoutException or WebDriverException) { }
      if (s.Process.WaitForExit(30_000))
        exitCode = s.Process.ExitCode;
    }
    catch
    {
      try { s.Desktop.SaveScreenshot(Path.Combine(artifacts, "failure.png")); } catch { }
      throw;
    }
    finally
    {
      s.Dispose();
    }
    output.WriteLine($"exit code {exitCode?.ToString(CultureInfo.InvariantCulture) ?? "none (killed after 30 s)"}");
    Assert.Equal(0, exitCode);
  }

  /// <summary>The viewport's date label ("yyyy-MM-dd HH:mm (UTC)"), empty when not shown.</summary>
  private static string EpochLabel(AppSession s, Viewport vp) =>
    ViewportProbe.OverlayText(s, vp, "ViewportEpochLabel") ?? string.Empty;

  /// <summary>A text once two reads 1 s apart agree (a label settling on the paused clock).</summary>
  private static string StableText(Func<string> read, string what)
  {
    string last = read();
    string? stable = null;
    Ui.WaitUntil(() =>
    {
      Thread.Sleep(1000);
      var now = read();
      bool same = now == last && now.Length > 0;
      last = now;
      if (same)
        stable = now;
      return same;
    }, TimeSpan.FromSeconds(20), what);
    return stable!;
  }

  private static void Shot(Viewport vp, string artifacts, string name) =>
    Win32.SavePng(Path.Combine(artifacts, name), vp.Rect.Left, vp.Rect.Top, vp.Rect.Width, vp.Rect.Height);
}
