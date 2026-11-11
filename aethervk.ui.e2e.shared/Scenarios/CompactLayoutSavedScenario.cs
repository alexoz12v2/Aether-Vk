// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac.
using System;
using System.IO;
using Xunit;
using static AetherVk.UI.E2E.Shared.E2ESteps;

namespace AetherVk.UI.E2E.Shared.Scenarios;

/// <summary>
/// The layout travels with the save: saved in Compact, the index has <c>"ui_layout": "compact"</c>
/// and the save reopens compact in a new session started in Classic; saved in Classic, the index
/// has no <c>ui_layout</c> key at all (older builds and saves stay byte-identical) and reopens classic.
/// </summary>
public static class CompactLayoutSavedScenario
{
  public static void Run(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "compact-layout-saved");
    var index = CopyFixture(artifacts);
    var compactAvk = Path.Combine(artifacts, "layout_compact.avk");
    var classicAvk = Path.Combine(artifacts, "layout_classic.avk");

    WithApp(host, artifacts, "app-save", app =>
    {
      Step(log, "open, compact, Save As");
      OpenSave(app, index);
      SetLayout(app, compact: true);
      SaveAs(app, compactAvk);
      var compact = UiLayout(compactAvk);
      log($"  {Path.GetFileName(compactAvk)}: ui_layout present {compact.Present}, \"{compact.Value}\"");
      Assert.True(compact.Present, "a compact save has no ui_layout");
      Assert.Equal("compact", compact.Value);

      Step(log, "classic, Save As");
      SetLayout(app, compact: false);
      SaveAs(app, classicAvk);
      var classic = UiLayout(classicAvk);
      log($"  {Path.GetFileName(classicAvk)}: ui_layout present {classic.Present}");
      Assert.False(classic.Present, $"a classic save has ui_layout \"{classic.Value}\"");
      log($"  quit: {app.Quit(TimeSpan.FromSeconds(30))}");
    });

    WithApp(host, artifacts, "app-reopen", app =>
    {
      Step(log, "a new session in Classic opens the compact save compact");
      if (IsCompact(app))
        SetLayout(app, compact: false);
      OpenSave(app, compactAvk, "layout_compact");
      E2EWait.Until(() => IsCompact(app), TimeSpan.FromSeconds(20), "the compact save to open compact");
      MainViewport(app);

      Step(log, "the classic save opens classic");
      OpenSave(app, classicAvk, "layout_classic");
      E2EWait.Until(() => !IsCompact(app), TimeSpan.FromSeconds(20), "the classic save to open classic");
      MainViewport(app);
    });
  }
}
