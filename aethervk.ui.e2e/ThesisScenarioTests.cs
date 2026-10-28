using System.Globalization;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E;

/// <summary>
/// The thesis "Experiments" flow, once per comet, in the real app window: timeline, Horizons
/// download and commit, jets, Earth observer at the telescope site in comet tracking, the
/// observation image as a translucent desktop billboard over the viewport, a desktop screenshot
/// (what the user grabs by hand for the overlay figure), then Save As, a fresh process, Open and
/// the restored values (the observer site is the stale-panel regression).
///
/// Artifacts: TestResults/e2e/&lt;scenario&gt;/ (screenshots and the save folder). Real
/// observation images are taken from AETHERVK_E2E_IMAGES/&lt;scenario&gt;.png|.jpg when present.
/// </summary>
public sealed class ThesisScenarioTests(ITestOutputHelper output)
{
  private static readonly TimeSpan DownloadTimeout = TimeSpan.FromMinutes(5);

  private static TimeSpan RunTimeout =>
    TimeSpan.FromMinutes(double.TryParse(Environment.GetEnvironmentVariable("AETHERVK_E2E_RUN_MINUTES"),
      NumberStyles.Float, CultureInfo.InvariantCulture, out var m) ? m : 30);

  [SkippableTheory]
  [MemberData(nameof(Scenarios.All), MemberType = typeof(Scenarios))]
  public void Scenario_ReplaysAndSurvivesSaveAndOpen(string scenarioId)
  {
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var sc = Scenarios.ById(scenarioId);
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", sc.Id);
    var saveDir = Path.Combine(artifacts, sc.Id);
    if (Directory.Exists(saveDir))
      Directory.Delete(saveDir, recursive: true);
    Directory.CreateDirectory(saveDir);
    var image = ObservationImage(sc, artifacts);

    InSession(artifacts, "build", s =>
    {
      Step("timeline");
      SetTimeline(s, sc);
      Step("comet");
      CommitComet(s, sc);
      Step("rotation");
      SetRotation(s, sc.Spin);
      Step("jets");
      AddJets(s, sc);
      AssertCameraMode(s, "CometOrbiting", "jet editing orbits the nucleus");
      s.App.SaveScreenshot(Path.Combine(artifacts, "jets.png"));
      var buildLog = Path.Combine(artifacts, "app-build.log");
      if (sc.NucleusMesh is { } mesh)
      {
        Step("nucleus mesh");
        ShowNucleusMesh(s, Path.Combine(AppSession.RepoRoot(), "assets", mesh), buildLog,
          Path.Combine(artifacts, "mesh.png"));
      }
      // tracking keeps the comet centred from the telescope site; it is entered at the
      // observation epoch, as at the telescope
      Step("run to the observation epoch");
      RunToEnd(s);
      Step("close range");
      CloseRange(s, buildLog, artifacts);
      Step("observer");
      EnterObserver(s, sc);
      AssertCometInView(s);
      s.App.SaveScreenshot(Path.Combine(artifacts, "observer-early.png"));
      WaitForDustSettled(buildLog);
      s.App.SaveScreenshot(Path.Combine(artifacts, "observer.png"));
      AssertNotWashedOut(s.Find("ViewportHost"), "the settled observer view (observer.png)");
      AssertNotFlat(s.Find("ViewportHost"), "the settled observer view (observer.png)");
      Step("tracers");
      ShowTracers(s, Path.Combine(artifacts, "tracers.png"));
      Step("billboard");
      ImportAndShowBillboard(s, image);

      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "desktop.png"));
      s.App.SaveScreenshot(Path.Combine(artifacts, "main-window.png"));

      Step("save as");
      s.Menu("File", "Save As...");
      s.AnswerFileDialog(saveDir);
      Ui.WaitUntil(() => s.WindowTitle.Contains(sc.Id) && !s.WindowTitle.EndsWith('*'),
        TimeSpan.FromSeconds(60), "the save to finish");
      Assert.True(File.Exists(Path.Combine(saveDir, "aethervk_index.json")), "no aethervk_index.json written");
    });

    InSession(artifacts, "reopen", s =>
    {
      Step("open");
      // an untouched application has nothing to save: Open goes straight to the file dialog
      Thread.Sleep(2000);
      Assert.False(s.WindowTitle.EndsWith('*'), $"a fresh application reports unsaved changes: '{s.WindowTitle}'");
      s.Menu("File", "Open...");
      Thread.Sleep(1000);
      Assert.True(s.Desktop.FindElementsByAccessibilityId("Discard").All(e => !e.Displayed),
        "Open asked to save the changes of a fresh application");
      s.AnswerFileDialog(Path.Combine(saveDir, "aethervk_index.json"));
      Ui.WaitUntil(() => s.WindowTitle.Contains(sc.Id), TimeSpan.FromSeconds(60), "the save to open");
      Thread.Sleep(3000);
      AssertRestored(s, sc);
      // the dust feedback restarts 3 s after Open and re-climbs against the final view
      Thread.Sleep(5000);
      s.App.SaveScreenshot(Path.Combine(artifacts, "reopened.png"));
      s.Desktop.SaveScreenshot(Path.Combine(artifacts, "desktop-reopened.png"));
      AssertDustDetailNotCoarser(Path.Combine(artifacts, "app-build.log"), Path.Combine(artifacts, "app-reopen.log"));
    });
  }

  /// <summary>One app process; on a failure the desktop is saved as failure-&lt;phase&gt;.png.</summary>
  private void InSession(string artifacts, string phase, Action<AppSession> body)
  {
    using var s = AppSession.Launch(Path.Combine(artifacts, $"app-{phase}.log"));
    try
    {
      body(s);
    }
    catch
    {
      var shot = Path.Combine(artifacts, $"failure-{phase}.png");
      try
      {
        s.Desktop.SaveScreenshot(shot);
        output.WriteLine($"screenshot: {shot}");
      }
      catch (Exception e)
      {
        output.WriteLine($"no failure screenshot: {e.Message}");
      }
      throw;
    }
  }

  private void Step(string name) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {name}");

  private static string Epoch(DateTime utc) => utc.ToString("yyyy-MM-dd'T'HH:mm:ss'Z'", CultureInfo.InvariantCulture);

  private static void SetTimeline(AppSession s, Scenario sc)
  {
    s.SelectTab("TimelineTab");
    s.TypeInto(s.Find("StartEpochBox"), Epoch(sc.ObservationUtc.AddDays(-sc.LeadDays)));
    s.TypeInto(s.Find("EndEpochBox"), Epoch(sc.ObservationUtc));
    // an out-of-range value is reverted without a message: the proposed range tells
    var expected = $"{Epoch(sc.ObservationUtc.AddDays(-sc.LeadDays))} to {Epoch(sc.ObservationUtc)}";
    Ui.WaitUntil(() => s.Find("ProposedRangeText").Text == expected, what: $"proposed range {expected}");
  }

  private static void CommitComet(AppSession s, Scenario sc)
  {
    s.SelectTab("CometTab");
    s.TypeInto(s.Find("CometSearchBox"), sc.Search);
    var result = Ui.WaitFor(() => s.Items("CometResultsList")
        .FirstOrDefault(i => Ui.ItemText(i).Contains(sc.Designation, StringComparison.OrdinalIgnoreCase)),
      TimeSpan.FromSeconds(60), $"a search result for {sc.Designation}");
    s.ScrollIntoView(result).RealClick();

    SelectRecord(s, sc.ObservationUtc.Year);

    s.Choose("ReferenceOrbitPicker", t => t.StartsWith("Osculating", StringComparison.Ordinal));
    s.Click("DownloadCommitButton");
    Ui.WaitUntil(() =>
    {
      if (s.TryFind("DownloadStatusText")?.Text is { } status
          && (status.Contains("fail", StringComparison.OrdinalIgnoreCase) || status.Contains("error", StringComparison.OrdinalIgnoreCase)))
        throw new InvalidOperationException($"download failed: {status}");
      return s.TryFind("CommittedCometName") is not null;
    }, DownloadTimeout, $"{sc.Designation} to be committed");
  }

  /// <summary>
  /// The nucleus IAU model, in the Comet tab once committed: with its default rate 0 the jet
  /// sites never turn and the run cannot show the spawn points sweeping.
  /// </summary>
  private static void SetRotation(AppSession s, Rotation spin)
  {
    s.SelectTab("CometTab");
    s.Expand("RotationalModelSection", "PoleRaSlider.Input");
    // the °/day rate lives in the collapsed "Advanced: rates" expander (the period is the main control)
    s.Expand("PolePrecessionExpander", "RotRateSlider.Input");
    s.SetSlider("PoleRaSlider", spin.PoleRaDeg);
    s.SetSlider("PoleDecSlider", spin.PoleDecDeg);
    s.SetSlider("PrimeMeridianSlider", spin.PrimeMeridianDeg);
    s.SetSlider("RotRateSlider", spin.RateDegPerDay);
  }

  private static void AssertRotation(AppSession s, Rotation spin)
  {
    s.SelectTab("CometTab");
    s.Expand("RotationalModelSection", "PoleRaSlider.Input");
    // the °/day rate lives in the collapsed "Advanced: rates" expander (the period is the main control)
    s.Expand("PolePrecessionExpander", "RotRateSlider.Input");
    Assert.Equal(spin.PoleRaDeg, s.ReadSlider("PoleRaSlider"), 2);
    Assert.Equal(spin.PoleDecDeg, s.ReadSlider("PoleDecSlider"), 2);
    Assert.Equal(spin.PrimeMeridianDeg, s.ReadSlider("PrimeMeridianSlider"), 2);
    Assert.Equal(spin.RateDegPerDay, s.ReadSlider("RotRateSlider"), 2);
  }

  /// <summary>
  /// The SPK record for the observation: the latest solution of that year or before (the first one
  /// when all are later). Picked with the keyboard: the list scrolls inside its own small viewport,
  /// where a row below its fold still lies inside the window, so a click would land on whatever is
  /// drawn there. The records are listed oldest first.
  /// </summary>
  private static void SelectRecord(AppSession s, int year)
  {
    var records = Ui.WaitFor(() => s.Items("SpkRecordsList") is { Count: > 0 } r ? r : null,
      TimeSpan.FromSeconds(60), "the SPK records");
    s.ScrollIntoView(records[0]).RealClick();
    string Selected() => Ui.WaitFor(() => s.Items("SpkRecordsList").FirstOrDefault(i => i.Selected) is { } i
      ? Ui.ItemText(i) : null, what: "the selected SPK record");
    for (int k = 0; k < 100; k++)
    {
      var current = Selected();
      s.App.Keyboard.SendKeys(OpenQA.Selenium.Keys.Down);
      Thread.Sleep(150);
      var next = Selected();
      if (next == current)
        break; // the last record
      if (RecordYear(next) > year)
      {
        s.App.Keyboard.SendKeys(OpenQA.Selenium.Keys.Up);
        Thread.Sleep(150);
        break;
      }
    }
  }

  private static int? RecordYear(string row) =>
    System.Text.RegularExpressions.Regex.Match(row, @"\b(19|20)\d{2}\b") is { Success: true } m
      ? int.Parse(m.Value, CultureInfo.InvariantCulture) : null;

  private static void AddJets(AppSession s, Scenario sc)
  {
    s.SelectTab("ModelTab");
    s.Expand("DustSection", "AddJetButton");
    var add = s.Find("AddJetButton");
    if (!add.Enabled)
      s.SetSlider("NucleusRadiusSlider", sc.FallbackNucleusRadiusKm);
    for (int k = 0; k < sc.Jets.Count; k++)
    {
      s.Click("AddJetButton");
      var items = Ui.WaitFor(() => s.Items("JetListBox") is { } l && l.Count == k + 1 ? l : null,
        what: $"jet #{k + 1} in the list");
      // the list toggles selection: click only an unselected row
      if (!items[^1].Selected)
        s.ScrollIntoView(items[^1]).RealClick();
      s.SetSlider("JetLatitudeSlider", sc.Jets[k].LatitudeDeg);
      s.SetSlider("JetLongitudeSlider", sc.Jets[k].LongitudeDeg);
      if (sc.Jets[k].ApertureDeg is { } aperture)
        s.SetSlider("JetApertureSlider", aperture);
      if (sc.Jets[k].SpeedMs is { } speed)
      {
        s.TypeInto(s.Find("JetVelocityBox"), Ui.Num(speed));
        s.TypeInto(s.Find("JetVelocityStdBox"), Ui.Num(sc.Jets[k].SpeedStdFraction * speed));
      }
    }
  }

  /// <summary>
  /// For a moment, the dust tracers (one real particle in 256 as a dot at its exact position, not
  /// the flow marks) from the observation view: the projected spiral of the rotating jets, then off
  /// again (the observer view and the saved state stay as the thesis figure).
  /// </summary>
  private static void ShowTracers(AppSession s, string shot)
  {
    s.SelectTab("ModelTab");
    s.Expand("DustSection", "DustTracersToggle");
    SetToggle(s, "DustTracersToggle", true);
    Thread.Sleep(2000);
    s.App.SaveScreenshot(shot);
    SetToggle(s, "DustTracersToggle", false);
  }

  private static void SetToggle(AppSession s, string id, bool on)
  {
    if (s.Find(id).Selected != on)
      s.ScrollIntoView(s.Find(id)).RealClick();
    Ui.WaitUntil(() => s.Find(id).Selected == on, what: $"{id} {(on ? "on" : "off")}");
  }

  private static void AssertCameraMode(AppSession s, string mode, string why)
  {
    s.SelectTab("SettingsTab");
    s.Expand("ViewportSettingsSection", "CameraModeText");
    Ui.WaitUntil(() => s.Find("CameraModeText").Text.Contains(mode), TimeSpan.FromSeconds(10),
      $"camera mode {mode} ({why})");
  }

  private static void EnterObserver(AppSession s, Scenario sc)
  {
    // the viewport is a native child window: a click focuses it, then its row-1 binding
    // (viewport.switch_to_earth_position) as a key to that focused window
    s.Find("ViewportHost").RealClick();
    Win32.PressKey(0x31); // VK '1'

    s.SelectTab("SettingsTab");
    s.Expand("ViewportSettingsSection", "CameraModeText");
    Ui.WaitUntil(() => s.Find("CameraModeText").Text.Contains("EarthPosition"), TimeSpan.FromSeconds(30),
      "camera mode EarthPosition (key 1 on the viewport)");
    // the thesis site: the observer stands on the surface (the sliders show only then)
    if (!s.Find("ObserverTopocentricCheckBox", TimeSpan.FromSeconds(30)).Selected)
      s.Click("ObserverTopocentricCheckBox");
    s.Find("ObserverLatSlider.Input", TimeSpan.FromSeconds(30));
    s.SetSlider("ObserverLatSlider", sc.Site.LatitudeDeg);
    s.SetSlider("ObserverLonSlider", sc.Site.LongitudeDeg);
    s.Click("OrientCometTracking");
    Ui.WaitUntil(() => s.Find("OrientCometTracking").Selected, what: "comet tracking");
    SetOrthographic(s, sc);
  }

  /// <summary>
  /// Orthographic at the thesis figure's framing: kilometres, proportions locked to the viewport
  /// (the half-height follows), the half-width from the figure's scale bar. The sliders live in the
  /// collapsed "Advanced: projection" expander.
  /// </summary>
  private static void SetOrthographic(AppSession s, Scenario sc)
  {
    s.Click("ProjectionOrthographic");
    Ui.WaitUntil(() => s.Find("ProjectionOrthographic").Selected, what: "orthographic projection");
    s.Expand("ProjectionAdvancedExpander", "OrthoUnitPicker");
    s.ChooseAt("OrthoUnitPicker", 1); // Kilometers (km)
    if (!s.Find("OrthoProportionsLock").Selected)
      s.Click("OrthoProportionsLock");
    s.SetSlider("OrthoHalfWidthSlider", sc.OrthoHalfWidthKm, tolerance: 1);
  }

  /// <summary>
  /// The camera HUD's observer line ("CometTracking · aim 2.1e-10 rad · target +34.2°", in the
  /// overlay window: the desktop session): the view axis is on the comet and it is above the horizon.
  /// </summary>
  private static void AssertCometInView(AppSession s)
  {
    var line = Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("EarthObserverStatus")
        .Select(e => e.Text).FirstOrDefault(t => t.Contains("aim", StringComparison.Ordinal)),
      TimeSpan.FromSeconds(30), "the HUD observer status");
    var m = System.Text.RegularExpressions.Regex.Match(line, @"aim (\S+) rad · target ([+-]\S+)°");
    Assert.True(m.Success, $"unexpected HUD line \"{line}\"");
    double aim = Ui.ParseNum(m.Groups[1].Value);
    double elevation = Ui.ParseNum(m.Groups[2].Value);
    Assert.True(aim < 1e-3, $"the telescope is {aim} rad off the comet ({line})");
    Assert.True(elevation > 0, $"the comet is below the horizon ({line})");
  }

  /// <summary>Tracking and the run must not have replaced the projection the user set.</summary>
  private static void AssertOrthographic(AppSession s, Scenario sc)
  {
    s.SelectTab("SettingsTab");
    s.Expand("ViewportSettingsSection", "CameraModeText");
    Assert.True(s.Find("ProjectionOrthographic").Selected, "projection is no longer orthographic");
    s.Expand("ProjectionAdvancedExpander", "OrthoHalfWidthSlider.Input");
    Assert.Equal(sc.OrthoHalfWidthKm, s.ReadSlider("OrthoHalfWidthSlider"), 0);
  }

  /// <summary>
  /// The imported mesh as the nucleus, in the jet-editing orbit view: the import and the switch
  /// log no error, and the nucleus region changes from the default wireframe to a drawn surface.
  /// </summary>
  private static void ShowNucleusMesh(AppSession s, string mesh, string log, string shot)
  {
    var nucleus = Centre(s.Find("ViewportHost"), 0.4);
    var wireframe = Sample(nucleus);
    var errors = ErrorLines(log);

    s.SelectTab("ImportsTab");
    s.Click("ImportButton");
    s.AnswerFileDialog(mesh);
    s.SelectTab("SettingsTab");
    s.Expand("CometAppearanceSection", "CometDisplayModeCustom");
    Ui.WaitUntil(() => s.Find("CometDisplayModeCustom").Enabled, TimeSpan.FromSeconds(30),
      "the custom display mode (enabled once a mesh is imported)");
    if (!s.Find("CometDisplayModeCustom").Selected)
      s.Click("CometDisplayModeCustom");
    var stem = Path.GetFileNameWithoutExtension(mesh);
    s.Choose("CometMeshPicker", t => t.Contains(stem, StringComparison.OrdinalIgnoreCase));
    Thread.Sleep(3000);
    s.App.SaveScreenshot(shot);

    var drawn = Sample(nucleus);
    var newErrors = ErrorLines(log).Skip(errors.Count).ToList();
    Assert.True(newErrors.Count == 0, $"the mesh logged errors:\n{string.Join('\n', newErrors)}");
    // the nucleus is dark (a few % albedo) under the white wireframe gizmo, which stays: count the
    // samples the mesh visibly changed rather than the mean change
    double changed = drawn.Zip(wireframe, (a, b) => Math.Abs(a - b) > 0.01).Count(c => c) / (double)drawn.Length;
    Assert.True(changed > 0.05, $"the nucleus region did not change with the mesh ({changed:P0} of it changed)");
    Assert.True(drawn.Average() < 0.95, $"the nucleus region is saturated (mean {drawn.Average():F3})");
  }

  /// <summary>
  /// At the observation epoch, in the jet-editing orbit view (a few km from the nucleus), once
  /// the dust detail feedback has settled: the near-nucleus stems (tracers on and off) leave the
  /// jet sites and turned with the nucleus since jets.png; then a few wheel notches out.
  /// </summary>
  private static void CloseRange(AppSession s, string log, string artifacts)
  {
    WaitForDustSettled(log);
    var vp = s.Find("ViewportHost");
    s.App.SaveScreenshot(Path.Combine(artifacts, "spin.png"));
    AssertNotWashedOut(vp, "the orbit view at the observation epoch (spin.png)");
    ShowTracers(s, Path.Combine(artifacts, "spin-tracers.png"));
    PhysicalLighting(s, Path.Combine(artifacts, "spin-physical.png"));

    Win32.WheelAt(vp.Location.X + vp.Size.Width / 2, vp.Location.Y + vp.Size.Height / 2, 6);
    WaitForDustSettled(log);
    s.App.SaveScreenshot(Path.Combine(artifacts, "spin-far.png"));
    AssertNotWashedOut(vp, "the orbit view zoomed out (spin-far.png)");
  }

  /// <summary>
  /// Comet debug: the physical lighting switch (offered in CometOrbiting only) shades the nucleus
  /// differently (sunlit side at (1 AU / r)², unlit side tinted); captured, then switched back.
  /// </summary>
  private static void PhysicalLighting(AppSession s, string png)
  {
    var vp = s.Find("ViewportHost");
    var before = Sample(Centre(vp, 0.4));
    s.SelectTab("SettingsTab");
    s.Expand("ViewportSettingsSection", "PhysicalLightingToggle");
    s.Click("PhysicalLightingToggle");
    Thread.Sleep(2000);
    s.App.SaveScreenshot(png);
    var after = Sample(Centre(vp, 0.4));
    s.Click("PhysicalLightingToggle");
    Thread.Sleep(1000);
    double changed = before.Zip(after, (a, b) => Math.Abs(a - b) > 0.01 ? 1.0 : 0.0).Average();
    Assert.True(changed > 0.02, $"physical lighting did not change the nucleus ({changed:P0} of samples)");
  }

  /// <summary>
  /// Until the dust detail feedback and the exposure are quiet (no new EWA / volume / exposure line for 8 s): the
  /// level it settled on is what the user looks at, not the first frames after a jump.
  /// </summary>
  private static void WaitForDustSettled(string log)
  {
    var re = new System.Text.RegularExpressions.Regex(@"\[Dust (EWA|volume)\].*for the next frames|\[Dust exposure\]");
    var end = DateTime.UtcNow + TimeSpan.FromSeconds(90);
    int last = -1;
    var since = DateTime.UtcNow;
    while (DateTime.UtcNow < end)
    {
      int count = LogLines(log).Count(l => re.IsMatch(l));
      if (count != last)
        (last, since) = (count, DateTime.UtcNow);
      else if (DateTime.UtcNow - since >= TimeSpan.FromSeconds(8))
        return;
      Thread.Sleep(1000);
    }
  }

  /// <summary>Most of the viewport saturated is a coma drawn as a wall, not as stems.</summary>
  private static void AssertNotWashedOut(OpenQA.Selenium.Appium.AppiumWebElement vp, string what)
  {
    var lum = Sample(Centre(vp, 0.9));
    double bright = lum.Count(l => l > 0.8) / (double)lum.Length;
    Assert.True(bright < 0.5, $"{what}: {bright:P0} of the viewport is saturated");
  }

  /// <summary>
  /// A lit viewport of one uniform grey is the dust drawn at a too coarse level and exposed to its
  /// median, not a coma (which has a bright core fading out).
  /// </summary>
  private static void AssertNotFlat(OpenQA.Selenium.Appium.AppiumWebElement vp, string what)
  {
    var lum = Sample(Centre(vp, 0.9));
    var lit = lum.Where(l => l > 0.05).OrderBy(l => l).ToArray();
    if (lit.Length < lum.Length / 4)
      return;
    double median = lit[lit.Length / 2];
    double flat = lit.Count(l => Math.Abs(l - median) < 0.05) / (double)lum.Length;
    Assert.True(flat < 0.6, $"{what}: {flat:P0} of the viewport is one grey ({median:F2})");
  }

  private static (int X, int Y, int W, int H) Centre(OpenQA.Selenium.Appium.AppiumWebElement e, double fraction)
  {
    int side = (int)(Math.Min(e.Size.Width, e.Size.Height) * fraction);
    return (e.Location.X + (e.Size.Width - side) / 2, e.Location.Y + (e.Size.Height - side) / 2, side, side);
  }

  private static double[] Sample((int X, int Y, int W, int H) r) => Win32.SampleLuminance(r.X, r.Y, r.W, r.H, 32);

  private static List<string> ErrorLines(string log)
  {
    var re = new System.Text.RegularExpressions.Regex(@"\b(error|panic|VUID-)", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
    return LogLines(log).Where(l => re.IsMatch(l)).ToList();
  }

  private static List<string> LogLines(string log)
  {
    using var f = new FileStream(log, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
    using var r = new StreamReader(f);
    var lines = new List<string>();
    for (string? line; (line = r.ReadLine()) != null;)
      lines.Add(line);
    return lines;
  }

  private static void RunToEnd(AppSession s)
  {
    s.SelectTab("TimelineTab");
    s.ChooseAt("SpeedPicker", ^1); // the fastest preset (1 day / s)
    s.Click("RunToEndButton");
    Ui.WaitUntil(() => s.TryFind("TimelineProgress") is { } p && Ui.ReadRange(p) >= 99.9,
      RunTimeout, "the simulation to reach the observation epoch");
  }

  private static void ImportAndShowBillboard(AppSession s, string image)
  {
    s.SelectTab("ImportsTab");
    s.Click("ImportButton");
    s.AnswerFileDialog(image);

    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    var stem = Path.GetFileNameWithoutExtension(image);
    s.Choose("BillboardTexturePicker", t => t.Contains(stem, StringComparison.OrdinalIgnoreCase));
    s.Click("CreateBillboardButton");
    s.Find("BillboardAlpha");

    // over the viewport, translucent: centred on it (X/Y are the centre) and as wide as it, the
    // scale the half-width was chosen for; the aspect lock (on, from the image) sets the height
    var vp = s.Find("ViewportHost");
    s.SetNumeric("BillboardWidth", vp.Size.Width);
    s.SetNumeric("BillboardX", vp.Location.X + vp.Size.Width / 2);
    s.SetNumeric("BillboardY", vp.Location.Y + vp.Size.Height / 2);
    s.SetRange("BillboardAlpha", 0.5, step: 0.1, tolerance: 0.05);
  }

  private static void AssertRestored(AppSession s, Scenario sc)
  {
    s.SelectTab("TimelineTab");
    Assert.Equal(sc.ObservationUtc.AddDays(-sc.LeadDays), ParseEpoch(s.Find("StartEpochBox").Text));
    Assert.Equal(sc.ObservationUtc, ParseEpoch(s.Find("EndEpochBox").Text));
    // the save sits at the observation epoch, paused, with its speed: Play is available
    Assert.True(Ui.ReadRange(s.Find("TimelineProgress")) >= 99.9, "the clock is not back at the observation epoch");
    Assert.True(s.Find("PlayPauseButton").Enabled, "no playback speed restored");
    var date = Ui.WaitFor(() => s.Desktop.FindElementsByAccessibilityId("ViewportEpochLabel")
      .FirstOrDefault(e => e.Displayed && e.Text.Length > 0), TimeSpan.FromSeconds(10), "the viewport date");
    Assert.StartsWith(sc.ObservationUtc.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture), date.Text);

    s.SelectTab("CometTab");
    Assert.Contains(sc.Designation, s.Find("CommittedCometName", TimeSpan.FromSeconds(60)).Text,
      StringComparison.OrdinalIgnoreCase);
    AssertRotation(s, sc.Spin);

    s.SelectTab("ModelTab");
    s.Expand("DustSection", "AddJetButton");
    var jets = s.Items("JetListBox");
    Assert.Equal(sc.Jets.Count, jets.Count);
    for (int k = 0; k < jets.Count; k++)
    {
      // row summary "Jet #n  λ 40,0°  φ 0,0°  ∠ 30°"
      var numbers = System.Text.RegularExpressions.Regex.Matches(Ui.ItemText(jets[k]).Replace('−', '-'), @"-?\d+(?:[.,]\d+)?")
        .Select(m => Ui.ParseNum(m.Value)).ToList();
      Assert.Contains(sc.Jets[k].LatitudeDeg, numbers);
      Assert.Contains(sc.Jets[k].LongitudeDeg, numbers);
    }

    s.SelectTab("SettingsTab");
    s.Expand("ViewportSettingsSection", "CameraModeText");
    Assert.Equal(sc.Site.LatitudeDeg, s.ReadSlider("ObserverLatSlider"), 2);
    Assert.Equal(sc.Site.LongitudeDeg, s.ReadSlider("ObserverLonSlider"), 2);
    Assert.True(s.Find("ObserverTopocentricCheckBox").Selected, "topocentric observer not restored");
    Assert.True(s.Find("OrientCometTracking").Selected, "comet tracking not restored");
    AssertOrthographic(s, sc);

    s.Expand("BillboardsSection", "CreateBillboardButton");
    Assert.InRange(Ui.ReadRange(s.Find("BillboardAlpha")), 0.45, 0.55); // as set: 0.5 ± the SetRange tolerance
  }

  /// <summary>
  /// The reopened save draws its dust at no coarser level than the session that built it: the last
  /// EWA and volume biases the adaptive feedback logged are compared.
  /// </summary>
  private static void AssertDustDetailNotCoarser(string buildLog, string reopenLog)
  {
    foreach (var kind in new[] { "EWA", "volume" })
    {
      int? built = LastBias(buildLog, kind), reopened = LastBias(reopenLog, kind);
      if (built is { } b && reopened is { } r)
        Assert.True(r <= b, $"dust {kind} bias {r} after reopen, {b} when built: coarser dust");
    }
  }

  private static int? LastBias(string log, string kind)
  {
    var re = new System.Text.RegularExpressions.Regex($@"\[Dust {kind}\].*bias (\d+) for the next frames");
    int? last = null;
    using var f = new FileStream(log, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
    using var r = new StreamReader(f);
    for (string? line; (line = r.ReadLine()) != null;)
      if (re.Match(line) is { Success: true } m)
        last = int.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture);
    return last;
  }

  private static DateTime ParseEpoch(string text) =>
    DateTime.Parse(text, CultureInfo.InvariantCulture, DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal);

  /// <summary>
  /// The telescope image of the scenario: panel (a) of its thesis chapter 5 figure, extracted from
  /// the PDF into ThesisImages/&lt;id&gt;.png (AETHERVK_E2E_IMAGES overrides the folder); the
  /// generated pattern only when neither has it.
  /// </summary>
  private static string ObservationImage(Scenario sc, string artifacts)
  {
    var dirs = new[]
    {
      Environment.GetEnvironmentVariable("AETHERVK_E2E_IMAGES"),
      Path.Combine(AppSession.RepoRoot(), "aethervk.ui.e2e", "ThesisImages"),
    };
    foreach (var dir in dirs.Where(d => !string.IsNullOrEmpty(d)))
    {
      foreach (var ext in new[] { ".png", ".jpg", ".jpeg" })
      {
        var real = Path.Combine(dir, sc.Id + ext);
        if (File.Exists(real))
          return real;
      }
    }
    Directory.CreateDirectory(artifacts);
    var path = Path.Combine(artifacts, $"{sc.Id}-pattern.png");
    TestPng.Write(path);
    return path;
  }
}
