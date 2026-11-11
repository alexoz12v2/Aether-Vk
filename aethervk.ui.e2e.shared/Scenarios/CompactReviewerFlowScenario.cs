// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac.
using System;
using System.IO;
using System.Linq;
using System.Threading;
using Xunit;
using static AetherVk.UI.E2E.Shared.E2ESteps;
using Ids = AetherVk.Logic.Automation.CompactAutomationIds;

namespace AetherVk.UI.E2E.Shared.Scenarios;

/// <summary>
/// The reviewer's loop, all in the compact panel: the 67P save → the observation period (date
/// from the image, rotation period, number of rotations, Apply; the preview spans N × period and
/// ends at the image date) → jet 1's velocity → a FITS imported and overlaid → Export: two PNGs,
/// the model one the main viewport's size and not black.
/// </summary>
public static class CompactReviewerFlowScenario
{
  public const double RotationPeriodHours = 12.4;
  public const int Rotations = 3;

  public static void Run(IE2EHost host, Action<string> log)
  {
    using var prefs = new E2EPreferencesBackup();
    var artifacts = Artifacts(host, "compact-reviewer-flow");
    var index = CopyFixture(artifacts);
    var fits = CopyFits(artifacts);
    WithApp(host, artifacts, "app", app =>
    {
      Step(log, "open the reviewer's save, compact layout");
      OpenSave(app, index);
      SetLayout(app, compact: true);

      Step(log, "observation period");
      if (app.TryFind(Ids.PeriodFromImage) is { Displayed: true, Enabled: true })
        app.Click(Ids.PeriodFromImage);
      var imageDate = E2EWait.For(() => Epochs(app.Text(Ids.PeriodImageDate)) is { Count: > 0 } d ? (object)d[0] : null,
        TimeSpan.FromSeconds(20), "an image date") is DateTime t ? t : throw new InvalidOperationException();
      log($"  image date {imageDate:u}");
      app.TypeInto(Ids.PeriodRotationPeriodHours, Num(RotationPeriodHours));
      app.TypeInto(Ids.PeriodRotations, Num(Rotations));
      app.Click(Ids.PeriodApply);
      var span = TimeSpan.FromHours(RotationPeriodHours * Rotations);
      string preview = "";
      E2EWait.Until(() =>
      {
        preview = app.Text(Ids.PeriodPreview);
        var e = Epochs(preview);
        return e.Count >= 2 && ((e[1] - e[0]) - span).Duration() <= TimeSpan.FromMinutes(2);
      }, TimeSpan.FromSeconds(60), $"the preview to span {Rotations} × {RotationPeriodHours} h");
      log($"  preview \"{preview}\"");
      var end = Epochs(preview)[1];
      Assert.True((end - imageDate).Duration() <= TimeSpan.FromSeconds(70),
        $"the preview ends at {end:u}, not at the image date {imageDate:u}");

      Step(log, "jet 1 velocity");
      E2EWait.Until(() => app.ListCount(Ids.JetsList) > 0, TimeSpan.FromSeconds(20), "the jet rows");
      app.SelectListItem(Ids.JetsList, 0);
      double v0 = ParseNum(app.Text(Ids.JetVelocity));
      double v1 = Math.Round(v0 * 1.1 + 1, 1);
      app.TypeInto(Ids.JetVelocity, Num(v1));
      E2EWait.Until(() => Math.Abs(ParseNum(app.Text(Ids.JetVelocity)) - v1) <= Math.Max(1e-3, 1e-3 * Math.Abs(v1)),
        TimeSpan.FromSeconds(10), $"jet 1 velocity {v1}");
      log($"  velocity {v0} → {v1}");

      Step(log, "import and overlay a FITS");
      int before = app.ListCount(Ids.FitsList);
      app.Click(Ids.FitsImportAndOverlay);
      Thread.Sleep(1000);
      app.AnswerOpenDialog(fits);
      E2EWait.Until(() => app.ListCount(Ids.FitsList) > before, TimeSpan.FromSeconds(90), "a new FITS row");
      log($"  FITS rows {before} → {app.ListCount(Ids.FitsList)}");
      Thread.Sleep(3000);

      Step(log, "export");
      var main = MainViewport(app);
      var (model, withFits) = Export(app, Path.Combine(artifacts, "shot.png"));
      AssertModelPng(model, main, log);
      var composite = PngCodec.Read(withFits);
      var png = PngCodec.Read(model);
      Assert.Equal((png.Width, png.Height), (composite.Width, composite.Height));
      if (app.TryFind(Ids.ExportStatus) is { } status)
        log($"  export status \"{status.Text}\"");
      app.EnsureAlive("after the export");
    });
  }
}
