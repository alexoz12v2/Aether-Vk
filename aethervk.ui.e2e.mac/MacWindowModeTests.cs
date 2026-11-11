using System.Globalization;
using System.Text.RegularExpressions;
using AetherVk.UI.E2E.Shared;
using Xunit.Abstractions;
using static AetherVk.UI.E2E.Shared.E2ESteps;
using Ids = AetherVk.Logic.Automation.CompactAutomationIds;

namespace AetherVk.UI.E2E.Mac;

/// <summary>
/// The window modes macOS users switch between: native full screen (Ctrl+Cmd+F, its own Space) and
/// the macOS 15+ half-screen tiles (Window › Move &amp; Resize › Left / Right, also Fn+Ctrl+← / →;
/// back with Return to Previous Size), in both layouts.
/// For each mode and back: the window takes the expected share of the display, every shown native
/// viewport (the main one, and the mini-view in Compact) gets a swapchain of its new pixel size,
/// and the main viewport still shows a picture. Run alone, like the harness:
/// <c>dotnet test aethervk.ui.e2e.mac -p:BuildE2E=true --filter "FullyQualifiedName~MacWindowModeTests"</c>.
/// Artifacts (screenshots per mode, app log) go to TestResults/e2e-mac/window-*/.
/// </summary>
public sealed partial class MacWindowModeTests(ITestOutputHelper output)
{
  private static readonly MacE2EHost Host = new();

  private void Log(string text) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {text}");

  public enum Mode
  {
    FullScreen,
    LeftHalf,
    RightHalf,
  }

  [SkippableTheory]
  [InlineData(Mode.FullScreen, false)]
  [InlineData(Mode.FullScreen, true)]
  [InlineData(Mode.LeftHalf, false)]
  [InlineData(Mode.LeftHalf, true)]
  [InlineData(Mode.RightHalf, false)]
  [InlineData(Mode.RightHalf, true)]
  public void WindowMode_ResizesViewports(Mode mode, bool compact)
  {
    var unavailable = Host.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    using var prefs = new E2EPreferencesBackup();
    string name = $"window-{mode.ToString().ToLowerInvariant()}-{(compact ? "compact" : "classic")}";
    var artifacts = Artifacts(Host, name);
    var index = CopyFixture(artifacts);
    var app = (MacE2EApp)Host.Launch(Path.Combine(artifacts, "app.log"));
    try
    {
      OpenSave(app, index); // a comet: the mini-view shows its host, and the frame is not empty
      SetLayout(app, compact);
      if (compact)
        SetMiniExpanded(app, true);
      var display = MacScreen.MainBoundsPoints;
      var original = app.WindowFrame();
      Log($"display {display.Width}×{display.Height} pt, window {Fmt(original)}");
      CheckViewports(app, artifacts, "windowed", compact, sinceLine: 0);

      int mark = app.Lines.Count;
      Enter(app, mode);
      E2EWait.Until(() => InMode(app.WindowFrame(), mode, display), TimeSpan.FromSeconds(20),
        $"the window in {mode} (now {Fmt(app.WindowFrame())})");
      Thread.Sleep(2000); // the Space animation, the layout pass and the swapchain recreation
      Log($"{mode}: window {Fmt(app.WindowFrame())}");
      CheckViewports(app, artifacts, mode.ToString(), compact, mark);

      mark = app.Lines.Count;
      Leave(app, mode);
      E2EWait.Until(() => Near(app.WindowFrame(), original, 6), TimeSpan.FromSeconds(20),
        $"the window back to {Fmt(original)} (now {Fmt(app.WindowFrame())})");
      Thread.Sleep(2000);
      Log($"restored: window {Fmt(app.WindowFrame())}");
      CheckViewports(app, artifacts, "restored", compact, mark);
      app.EnsureAlive("switching window modes");
    }
    catch
    {
      try { app.SaveScreenshot(Path.Combine(artifacts, "failure.png")); } catch (Exception) { }
      try { File.WriteAllText(Path.Combine(artifacts, "failure.xml"), app.Session.Driver.PageSource); } catch (Exception) { }
      // never leave the machine in full screen or tiled for the next test
      try { Leave(app, mode); Thread.Sleep(1500); } catch (Exception) { }
      throw;
    }
    finally
    {
      app.Dispose();
    }
  }

  private const string WindowMenu = "Window", MoveAndResize = "Move & Resize";

  private static void Enter(MacE2EApp app, Mode mode)
  {
    app.Foreground();
    switch (mode)
    {
      case Mode.FullScreen:
        app.Chord("f", MacE2EApp.Control | MacE2EApp.Command);
        break;
      // the Window menu's tiles (AppKit's, in the menu MenuMapper registers): the action behind
      // Fn+Ctrl+arrow, which Mac2 cannot press (it drops Fn, and Ctrl+arrow switches Spaces)
      case Mode.LeftHalf:
        app.Menu(WindowMenu, MoveAndResize, "Left");
        break;
      case Mode.RightHalf:
        app.Menu(WindowMenu, MoveAndResize, "Right");
        break;
    }
  }

  private static void Leave(MacE2EApp app, Mode mode)
  {
    app.Foreground();
    if (mode == Mode.FullScreen)
      app.Chord("f", MacE2EApp.Control | MacE2EApp.Command);
    else
      app.Menu(WindowMenu, MoveAndResize, "Return to Previous Size");
  }

  /// <summary>
  /// Full screen: the window covers the display. A half tile: at that edge, 40–52 % of the width
  /// (the tiles may keep margins), most of the height below the menu bar.
  /// </summary>
  private static bool InMode(PointRect w, Mode mode, (double X, double Y, double Width, double Height) d)
  {
    switch (mode)
    {
      case Mode.FullScreen:
        return Near(w, new PointRect(d.X, d.Y, d.Width, d.Height), 2);
      default:
        double share = w.Width / d.Width;
        bool edge = mode == Mode.LeftHalf
          ? Math.Abs(w.X - d.X) < 24
          : Math.Abs(w.X + w.Width - (d.X + d.Width)) < 24;
        return edge && share is > 0.40 and < 0.52 && w.Height > 0.8 * (d.Height - 40);
    }
  }

  private static bool Near(PointRect a, PointRect b, double tolerance) =>
    Math.Abs(a.X - b.X) <= tolerance && Math.Abs(a.Y - b.Y) <= tolerance
    && Math.Abs(a.Width - b.Width) <= tolerance && Math.Abs(a.Height - b.Height) <= tolerance;

  private static string Fmt(PointRect r) =>
    string.Create(CultureInfo.InvariantCulture, $"{r.X:0},{r.Y:0} {r.Width:0}×{r.Height:0}");

  [GeneratedRegex(@"\[Swapchain\] create (\d+)x(\d+)")]
  private static partial Regex SwapchainCreate();

  /// <summary>The swapchain sizes created since the log line <paramref name="sinceLine"/>.</summary>
  private static List<(int W, int H)> SwapchainsSince(MacE2EApp app, int sinceLine) =>
    app.Lines.Skip(sinceLine)
      .Select(l => SwapchainCreate().Match(l))
      .Where(m => m.Success)
      .Select(m => (int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture), int.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture)))
      .ToList();

  /// <summary>The pixel size of each viewport at the previous step.</summary>
  private readonly Dictionary<string, (int W, int H)> _previous = new();

  /// <summary>
  /// Each shown viewport whose size changed since the previous step has a swapchain of its new
  /// pixel size (±2 px of rounding), created since the mark (a viewport of a fixed size, the
  /// mini-view, keeps its swapchain). The main viewport's picture is not black (Mac2's element
  /// screenshot: no Screen Recording needed).
  /// </summary>
  private void CheckViewports(MacE2EApp app, string artifacts, string step, bool compact, int sinceLine)
  {
    var ids = new List<string> { MainViewportId };
    if (compact)
      ids.Add(Ids.NucleusMiniViewHost);
    foreach (var id in ids)
    {
      var rect = id == MainViewportId ? MainViewport(app) : app.ViewportRect(id);
      bool resized = !_previous.TryGetValue(id, out var before)
                     || Math.Abs(before.W - rect.Width) > 2 || Math.Abs(before.H - rect.Height) > 2;
      _previous[id] = (rect.Width, rect.Height);
      if (!resized)
      {
        Log($"{step}: {id} {rect.Width}×{rect.Height} px, unchanged");
        continue;
      }
      bool Matches() => SwapchainsSince(app, sinceLine).Any(s => Math.Abs(s.W - rect.Width) <= 2 && Math.Abs(s.H - rect.Height) <= 2);
      try
      {
        E2EWait.Until(Matches, TimeSpan.FromSeconds(15), $"a {rect.Width}×{rect.Height} swapchain for {id}");
      }
      catch (TimeoutException)
      {
        var seen = string.Join(", ", SwapchainsSince(app, sinceLine).Select(s => $"{s.W}×{s.H}"));
        throw new Xunit.Sdk.XunitException(
          $"{step}: {id} is {rect.Width}×{rect.Height} px but no swapchain of that size was created (seen since the mode change: {seen})");
      }
      Log($"{step}: {id} {rect.Width}×{rect.Height} px, swapchain matches");
    }
    var shot = app.ElementScreenshot(MainViewportId);
    File.WriteAllBytes(Path.Combine(artifacts, $"{step}-viewport.png"), PngCodec.EncodeRgb(shot.Rgba, shot.Width, shot.Height));
    app.SaveScreenshot(Path.Combine(artifacts, $"{step}.png"));
    double mean = shot.Luminance().Average();
    Log($"{step}: main viewport mean luminance {mean:0.0000}");
    Assert.True(mean > 0.005, $"{step}: the main viewport is black");
  }
}
