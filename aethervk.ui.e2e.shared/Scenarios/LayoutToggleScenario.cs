// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac.
using System;
using System.IO;
using System.Linq;
using System.Threading;
using Xunit;
using static AetherVk.UI.E2E.Shared.E2ESteps;

namespace AetherVk.UI.E2E.Shared.Scenarios;

/// <summary>
/// Layout › Classic ↔ Compact ten times on the reviewer's save while it plays: every time exactly
/// one main viewport is shown, at the same rectangle as the first time in that layout (the native
/// host is resized, never re-created or moved to another parent), and its frames keep changing.
/// </summary>
public static class LayoutToggleScenario
{
  public const int Toggles = 10;

  public static void Run(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "layout-toggle");
    var index = CopyFixture(artifacts);
    WithApp(host, artifacts, "app", app =>
    {
      Step(log, "open the reviewer's save");
      OpenSave(app, index);
      if (IsCompact(app))
        SetLayout(app, compact: false);
      var classic = MainViewport(app);
      Step(log, $"classic viewport {classic}");

      SetLayout(app, compact: true);
      var compact = MainViewport(app);
      Step(log, $"compact viewport {compact}");
      Assert.True(compact.Width < classic.Width, $"the compact viewport {compact} is not narrower than the classic one {classic}");
      app.SavePng(compact, Path.Combine(artifacts, "compact-viewport.png"));

      Step(log, "play at 1 h/s");
      Play(app);

      for (int i = 1; i <= Toggles; i++)
      {
        foreach (var (toCompact, expected) in new[] { (false, classic), (true, compact) })
        {
          SetLayout(app, toCompact);
          app.EnsureAlive($"toggle {i}");
          var now = MainViewport(app);
          var changed = ChangedPairs(app, MainViewportId, pairs: 2, gap: TimeSpan.FromSeconds(1));
          log($"  toggle {i} {(toCompact ? "compact" : "classic")}: viewport {now}, changed {string.Join(", ", changed.Select(Pct))}");
          Assert.True(now.Area > 0, $"toggle {i}: the main viewport has no area");
          Assert.True(now.Near(expected, 2), $"toggle {i}: the viewport moved to {now} (first time in this layout: {expected})");
          Assert.All(changed, c => Assert.True(c > PlayingChanged, $"toggle {i}: a frame pair barely changed ({Pct(c)})"));
        }
      }
      Pause(app);
    });
  }
}
