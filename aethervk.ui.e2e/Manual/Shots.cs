using System.Drawing;
using OpenQA.Selenium.Appium;

namespace AetherVk.UI.E2E.Manual;

/// <summary>What a shot keeps: a desktop rectangle, and optionally the callouts (else the capture reads them from the UIA tree).</summary>
/// <param name="Rect">Physical desktop px.</param>
/// <param name="Explicit">The callouts, when the crop is a popup the app's tree does not hold; <c>[]</c> for none.</param>
/// <param name="AnchorId">An element of the crop, used to line up the tree's coordinates with WinAppDriver's.</param>
internal sealed record Cropped(Rectangle Rect, IReadOnlyList<RawCallout>? Explicit = null, string? AnchorId = null, bool Clipped = false);

/// <summary>One screenshot of the manual, as data: <paramref name="Session"/> is "empty" or "committed".</summary>
internal sealed record Shot(
  string Key, string State, string Session, Action<ShotCtx> Setup, Func<ShotCtx, Cropped> Crop,
  Action<ShotCtx>? After = null);

/// <summary>A shot that is not captured, and why (the manifest lists it).</summary>
internal sealed record SkippedShot(string Key, string Reason);

/// <summary>
/// The session state a shot's setup and crop work on. Rectangles are physical desktop px. The window
/// is maximised once on the primary monitor and never moved, so WinAppDriver's element places are
/// the desktop's; the small correction in <see cref="Delta"/> covers any difference the window
/// position reports.
/// </summary>
internal sealed class ShotCtx(AppSession s, Win32.Monitor monitor, Action<string> log)
{
  public AppSession S => s;
  public Win32.Monitor Monitor => monitor;
  public Action<string> Log => log;

  /// <summary>Top-level windows present before a click that raises a popup (<see cref="NewPopup"/>).</summary>
  public HashSet<IntPtr> Before { get; set; } = [];

  /// <summary>A dialog window a shot opened, for its <c>After</c> to close.</summary>
  public IntPtr Dialog { get; set; }

  /// <summary>The "Model rotations" text before the lock shots edited it; null when untouched.</summary>
  public string? RotationsBefore { get; set; }

  /// <summary>The app is gone: the rest of the session is not attempted.</summary>
  public bool Dead { get; set; }

  private (int Dx, int Dy)? _delta;

  /// <summary>Real window corner minus the corner WinAppDriver reports, added to every element place.</summary>
  public (int Dx, int Dy) Delta
  {
    get
    {
      if (_delta is null)
      {
        var real = Win32.WindowCorner(s.MainWindow);
        var reported = s.App.Manage().Window.Position;
        _delta = (real.X - reported.X, real.Y - reported.Y);
      }
      return _delta.Value;
    }
  }

  public void ResetDelta() => _delta = null;

  public Rectangle Rect(AppiumWebElement e)
  {
    var (p, size) = (e.Location, e.Size);
    return new Rectangle(p.X + Delta.Dx, p.Y + Delta.Dy, size.Width, size.Height);
  }

  public Rectangle? RectOf(string id) => s.TryFind(id) is { } e ? Rect(e) : null;

  public Rectangle Need(string id) => RectOf(id) ?? throw new InvalidOperationException($"'{id}' is not on screen");

  /// <summary>The main window's visible bounds, within the monitor.</summary>
  public Rectangle Window => Rectangle.Intersect(Win32.FrameBounds(s.MainWindow), new Rectangle(monitor.Left, monitor.Top, monitor.Width, monitor.Height));

  /// <summary>The union of the elements of <paramref name="ids"/> that are on screen, grown by <paramref name="pad"/>.</summary>
  public Rectangle UnionFound(int pad, params string[] ids)
  {
    Rectangle? union = null;
    foreach (var id in ids)
      if (RectOf(id) is { } r)
        union = union is null ? r : Rectangle.Union(union.Value, r);
    if (union is null)
      throw new InvalidOperationException($"none of {string.Join(", ", ids)} is on screen");
    return Rectangle.Inflate(union.Value, pad, pad);
  }

  /// <summary>Keeps a rectangle inside the main window; <c>Clipped</c> tells that something was cut off.</summary>
  public Cropped Fit(Rectangle r, string? anchor = null, IReadOnlyList<RawCallout>? explicitCallouts = null)
  {
    var win = Window;
    var fitted = Rectangle.Intersect(r, win);
    return new Cropped(fitted, explicitCallouts, anchor, Clipped: fitted != r);
  }

  /// <summary>A popup window the last click raised (menu, combo drop-down, flyout): the largest new top-level window.</summary>
  public Rectangle NewPopup(TimeSpan? timeout = null)
  {
    var deadline = DateTime.UtcNow + (timeout ?? TimeSpan.FromSeconds(5));
    do
    {
      var found = Win32.TopLevelWindows(s.Process.Id)
        .Where(w => !Before.Contains(w.Handle))
        .Select(w => Win32.FrameBounds(w.Handle))
        .Where(r => r.Width >= 30 && r.Height >= 20)
        .OrderByDescending(r => (long)r.Width * r.Height)
        .ToList();
      if (found.Count > 0)
        return found[0];
      Thread.Sleep(150);
    } while (DateTime.UtcNow < deadline);
    throw new TimeoutException("no popup window appeared");
  }

  public void Snapshot() => Before = Win32.TopLevelWindows(s.Process.Id).Select(w => w.Handle).ToHashSet();

  /// <summary>The pointer far from the app so hover styles and tooltips fade.</summary>
  public void Park()
  {
    var other = Win32.Monitors().Skip(1).FirstOrDefault();
    if (other is not null)
      Win32.MoveCursor(other.Left + other.Width / 2, other.Top + other.Height / 2);
    else
      Win32.MoveCursor(monitor.Left + monitor.Width / 2, monitor.Top + 1);
    Thread.Sleep(300);
  }

  public void Escape() => Win32.PressKey(Win32.VK_ESCAPE);

  /// <summary>The first element of the app's tree of a class, displayed (a list item, a text part).</summary>
  public AppiumWebElement? FirstOfClass(string className) =>
    s.App.FindElementsByClassName(className).FirstOrDefault(e => e.Displayed);
}

/// <summary>The steps shared by the shot setups.</summary>
internal static class Steps
{
  public static readonly string[] SettingsSections =
  [
    "TimeScaleSection", "GeneralSection", "OrbitalElementsSection", "CometAppearanceSection", "BillboardsSection",
    "ParticleRenderingSection", "InputBindingsSection", "ViewportSettingsSection",
  ];

  public static readonly string[] CometSections = ["SimulatedRangeSection", "NucleusSection", "RotationalModelSection"];

  public static readonly string[] ModelSections = ["ModelSunSection", "NucleusSpinSection", "DustSection"];

  public static bool IsExpanded(AppiumWebElement e) => e.GetAttribute("ExpandCollapse.ExpandCollapseState") == "Expanded";

  public static void Mode(ShotCtx c, string buttonId)
  {
    c.S.Click(buttonId);
    Thread.Sleep(800);
  }

  // sections opened the way the scenario e2e opens them: by a child that shows once open
  private static readonly Dictionary<string, string> OpenProbes = new()
  {
    ["ViewportSettingsSection"] = "CameraModeText",
  };

  public static void OpenSection(ShotCtx c, string id)
  {
    if (OpenProbes.TryGetValue(id, out var probe))
    {
      c.S.Expand(id, probe);
      return;
    }
    var e = c.S.ScrollIntoView(c.S.Find(id));
    // an expander made from a DataTemplate may not report its ExpandCollapse state: then it counts as
    // open once it grew taller than its header (a second toggle would close it again)
    bool reports = e.GetAttribute("ExpandCollapse.ExpandCollapseState") is "Expanded" or "Collapsed";
    int h0 = e.Size.Height;
    for (int attempt = 0; attempt < 2; attempt++)
    {
      if (reports ? IsExpanded(e) : attempt > 0 && e.Size.Height > h0 + 20)
        return;
      if (!reports && attempt == 0 && h0 > 80)
        return; // already open
      Ui.ToggleExpander(e);
      Thread.Sleep(600);
    }
  }

  public static void CloseSection(ShotCtx c, string id)
  {
    if (c.S.TryFind(id) is not { } e)
      return;
    c.S.ScrollIntoView(e);
    var state = e.GetAttribute("ExpandCollapse.ExpandCollapseState");
    if (state == "Expanded" || (state is not "Collapsed" && e.Size.Height > 80))
    {
      // a tall open section counts as in view while its header is scrolled off: bring the header up
      AlignTop(c, id);
      e = c.S.Find(id);
      Ui.ToggleExpander(e);
      Thread.Sleep(400);
    }
  }

  /// <summary>Selects the tab, closes the other sections of it, opens <paramref name="keep"/> and puts its header at the top of the pane.</summary>
  public static void OnlyOpen(ShotCtx c, string tab, string[] sections, string keep)
  {
    c.S.SelectTab(tab);
    foreach (var other in sections.Where(x => x != keep))
      CloseSection(c, other);
    OpenSection(c, keep);
    AlignTop(c, keep);
  }

  /// <summary>Wheels the pane until the element's top is just under the pane header (or the pane ends).</summary>
  public static void AlignTop(ShotCtx c, string id)
  {
    var e = c.S.ScrollIntoView(c.S.Find(id));
    var win = c.Window;
    int top = win.Y + 150, bottom = win.Bottom - 40, last = int.MinValue, prev = 0;
    for (int i = 0; i < 40; i++)
    {
      var r = c.Rect(e);
      int dy = r.Y > top + 6 ? 1 : r.Y < top - 60 ? -1 : 0;
      // a wheel notch can be taller than the band: once the direction flips we have bracketed it,
      // and after a flip to "up" (-1) the header is below the top, i.e. visible: stop there
      if (dy == 0 || r.Y == last || (prev != 0 && dy != prev))
      {
        if (prev == 1 && dy == -1)
          Win32.WheelAt(r.X + Math.Min(r.Width / 2, 40), Math.Clamp(r.Y + r.Height / 2, top + 20, bottom - 20), -1);
        break;
      }
      prev = dy;
      last = r.Y;
      Win32.WheelAt(r.X + Math.Min(r.Width / 2, 40), Math.Clamp(r.Y + r.Height / 2, top + 20, bottom - 20), dy);
    }
    Thread.Sleep(300);
  }

  public static Cropped Section(ShotCtx c, string id, int pad = 6) =>
    c.Fit(Rectangle.Inflate(c.Need(id), pad, pad), id);

  /// <summary>
  /// The dock pane of a tab (WinAppDriver has no parent axis, so it is measured from the tab headers):
  /// from just above the tab's header to the window's right edge, down to the next pane's header below
  /// it (the Imports pane stops above the Timeline's) or the window bottom.
  /// </summary>
  public static Rectangle PaneOf(ShotCtx c, string tab)
  {
    var win = c.Window;
    var t = c.Rect(c.S.Find("Tab." + tab));
    int x0 = Math.Max(win.X, t.X - 14), y0 = Math.Max(win.Y, t.Y - 10), bottom = win.Bottom;
    foreach (var other in new[] { "TimelineTab", "ImportsTab", "SettingsTab", "CometTab", "ModelTab" })
    {
      if (other == tab || c.S.TryFind("Tab." + other) is not { } e)
        continue;
      var r = c.Rect(e);
      if (r.Y > t.Bottom + 40 && r.Right > x0)
        bottom = Math.Min(bottom, r.Y - 14);
    }
    return Rectangle.Intersect(Rectangle.FromLTRB(x0, y0, win.Right, bottom), win);
  }

  /// <summary>Until no new dust detail / exposure line has come from the app for <paramref name="quiet"/>.</summary>
  public static void WaitDustSettled(ShotCtx c, int quietSeconds = 6, int maxSeconds = 60)
  {
    var re = new System.Text.RegularExpressions.Regex(@"\[Dust (EWA|volume)\].*for the next frames|\[Dust exposure\]");
    var end = DateTime.UtcNow.AddSeconds(maxSeconds);
    int last = -1;
    var since = DateTime.UtcNow;
    while (DateTime.UtcNow < end)
    {
      int count = c.S.Lines.Count(l => re.IsMatch(l.Line));
      if (count != last)
        (last, since) = (count, DateTime.UtcNow);
      else if (DateTime.UtcNow - since >= TimeSpan.FromSeconds(quietSeconds))
        return;
      Thread.Sleep(1000);
    }
  }

  /// <summary>The viewport on the desktop: from the app's own report when it gives one, else the host element.</summary>
  public static Rectangle ViewportRect(ShotCtx c)
  {
    if (FitsComparisonSteps.ReadViewportDesktopRect(c.S) is { } r)
      return new Rectangle((int)Math.Round(r.X), (int)Math.Round(r.Y), (int)Math.Round(r.W), (int)Math.Round(r.H));
    return c.Need("ViewportHost");
  }

  /// <summary>
  /// A pending Start edit locks the range, as in ReviewerMailTests R2 (a pending rotations edit
  /// does not); <see cref="UnlockTheRange"/> presses Restore.
  /// </summary>
  public static void LockTheRange(ShotCtx c)
  {
    c.S.OpenRangeEntry();
    var box = c.S.Find("StartEpochBox");
    c.RotationsBefore ??= box.Text;
    var start = DateTime.Parse(box.Text.Trim().TrimEnd('Z'), System.Globalization.CultureInfo.InvariantCulture);
    c.S.TypeInto(box, start.AddDays(1).ToString("yyyy-MM-ddTHH:mm:ss", System.Globalization.CultureInfo.InvariantCulture), enter: false);
    Ui.WaitUntil(() => c.S.TryFind("ModeBarRangeLockText") is not null, TimeSpan.FromSeconds(10), "the range lock text");
  }

  public static void UnlockTheRange(ShotCtx c)
  {
    if (c.RotationsBefore is null)
      return;
    c.S.OpenRangeEntry("RestoreButton");
    c.S.Click("RestoreButton");
    c.RotationsBefore = null;
    Thread.Sleep(800);
  }

  /// <summary>The menu entries of the main menu, with their rectangles, as callouts "menu:File/Open...".</summary>
  public static (Rectangle Union, List<RawCallout> Callouts) MenuEntries(ShotCtx c, string menu, string[] entries)
  {
    var header = c.Rect(c.S.App.FindElementByName(menu));
    var union = header;
    var callouts = new List<RawCallout>();
    foreach (var entry in entries)
    {
      var e = c.S.Desktop.FindElementsByName(entry).FirstOrDefault(x => x.Displayed);
      if (e is null)
        continue;
      var r = new Rectangle(e.Location, e.Size);
      union = Rectangle.Union(union, r);
      callouts.Add(new RawCallout($"menu:{menu}/{entry}", entry, r, "MenuItem"));
    }
    return (Rectangle.Inflate(union, 6, 6), callouts);
  }

  /// <summary>The first jet row of the Dust section's list (a list item), selected unless a jet is already.</summary>
  public static void SelectFirstJet(ShotCtx c)
  {
    if (c.S.TryFind("JetLatitudeSlider") is not null)
      return;
    var row = c.FirstOfClass("ListBoxItem") ?? throw new InvalidOperationException("no jet row in the Dust section");
    c.S.ScrollIntoView(row).RealClick();
    Ui.WaitFor(() => c.S.TryFind("JetLatitudeSlider"), what: "the jet detail");
  }

  public static void ShowBillboardImage(ShotCtx c, bool visible)
  {
    FitsComparisonSteps.SetVisible(c.S, 0, visible);
    Thread.Sleep(1500);
  }

  public static IntPtr FindWindow(ShotCtx c, Func<Win32.Window, bool> match, int seconds = 10)
  {
    var end = DateTime.UtcNow.AddSeconds(seconds);
    while (DateTime.UtcNow < end)
    {
      if (Win32.TopLevelWindows(c.S.Process.Id).FirstOrDefault(match) is { } w)
        return w.Handle;
      Thread.Sleep(250);
    }
    return IntPtr.Zero;
  }
}

/// <summary>The shot list of PLAN §3.</summary>
internal static class Shots
{
  /// <summary>Rows of the plan that this fixture does not capture.</summary>
  public static IReadOnlyList<SkippedShot> Skipped { get; } =
  [
    new("dlg-manage-imports", "no menu, button or hot key opens the Manage Imports window (the command exists in MainWindowViewModel only)"),
    new("dlg-spawn-mesh", "no UI path opens the Spawn Mesh Instance window"),
    new("dlg-spawn-image", "no UI path opens the Spawn Image Billboard window"),
  ];

  private static readonly string[] ModeBarIds =
    ["ModeBarOrbitButton", "ModeBarNucleusButton", "ModeBarModelButton", "ModeBarHintText", "ModeBarRangeLockText", "ModeBarRecommitButton"];

  private static readonly string[] TimelineIds =
  [
    "TimelineStartEpochLabel", "TimelineStartEpochBox", "TimelineEndEpochLabel", "TimelineEndEpochBox", "CommittedRangeText",
    "PendingRangeEditText", "SpeedPicker", "ResetButton", "PlayPauseButton", "RunToEndButton", "TimelineProgress",
    "CurrentEpochLabel", "CurrentEpochBox", "SeekToEpochButton",
  ];

  private static readonly string[] OverlayIds =
    ["GizmoLegendPanel", "ViewportRulerText", "DustPreviewBadge", "DustRenderFullButton", "DustRenderingBadge", "ViewportEpochLabel", "ViewportFrameLabel"];

  public static IEnumerable<Shot> All()
  {
    // ---- 1. main window
    foreach (var session in new[] { "empty", "committed" })
      yield return new("main", session, session, _ => { }, c => c.Fit(c.Window));

    // ---- 2. mode bar
    foreach (var (state, button) in new[] { ("orbit", "ModeBarOrbitButton"), ("nucleus", "ModeBarNucleusButton"), ("model", "ModeBarModelButton") })
      yield return new("main-modebar", state, "committed", c => Steps.Mode(c, button),
        c => c.Fit(c.UnionFound(6, ModeBarIds), ModeBarIds[0]));

    // ---- 3. menus (the popups are not in the app's tree: callouts are the entries)
    yield return new("menu-file", "committed", "committed", c => c.S.Menu("File"),
      c =>
      {
        var (union, callouts) = Steps.MenuEntries(c, "File", ["New", "Open...", "Open Recent", "Save", "Save As...", "Save As Folder..."]);
        return new Cropped(union, callouts);
      }, c => c.Escape());
    yield return new("menu-edit", "committed", "committed", c => c.S.Menu("Edit"),
      c =>
      {
        var (union, callouts) = Steps.MenuEntries(c, "Edit", ["Settings...", "Toggle Theme"]);
        return new Cropped(union, callouts);
      }, c => c.Escape());

    // ---- 4. docking: the tab strip of one pane with its +, swap and close buttons
    yield return new("docking", "committed", "committed", _ => { }, c =>
    {
      var tabs = new[] { "SettingsTab", "CometTab", "ModelTab", "ImportsTab", "TimelineTab" }
        .Select(t => c.RectOf("Tab." + t)).Where(r => r is not null).Select(r => r!.Value).ToList();
      if (tabs.Count == 0)
        throw new InvalidOperationException("no dock tab on screen");
      var strip = tabs.Aggregate(Rectangle.Union);
      var row = Rectangle.Inflate(strip, 0, 6);
      var union = strip;
      foreach (var name in new[] { "+", "⇆", "✕" })
        foreach (var e in c.S.App.FindElementsByName(name).Where(x => x.Displayed))
        {
          var r = c.Rect(e);
          if (r.Y + r.Height / 2 >= row.Y && r.Y + r.Height / 2 <= row.Bottom)
            union = Rectangle.Union(union, r);
        }
      return c.Fit(Rectangle.Inflate(union, 8, 8));
    });

    // ---- 5. comet tab
    foreach (var session in new[] { "empty", "committed" })
    {
      yield return new("comet-guided", session, session, c => c.S.SelectTab("CometTab"),
        c => c.Fit(c.UnionFound(8, "GuidedSteps", "GuidedStepHelpText"), "GuidedSteps"));
      yield return new("comet-selection", session, session,
        c =>
        {
          c.S.SelectTab("CometTab");
          if (session == "empty")
          {
            c.S.TypeInto(c.S.Find("CometSearchBox"), "67P", enter: false);
            c.S.Click("CometSearchButton");
            try
            {
              Ui.WaitUntil(() => c.S.Items("CometResultsList").Count > 0, TimeSpan.FromSeconds(30), "the search results");
            }
            catch (TimeoutException)
            {
              c.Log("comet-selection: no search results in 30 s (offline?); captured without them");
            }
          }
        },
        c =>
        {
          var r = c.UnionFound(8, "CommittedCometName", "CometSearchBox", "CometSearchButton", "CometResultsList", "SpkRecordsList");
          return c.Fit(new Rectangle(r.X, r.Y - 34, r.Width, r.Height + 34), "CometSearchBox"); // up to the header
        });
      yield return new("comet-commit", session, session, c => c.S.SelectTab("CometTab"),
        c => c.Fit(c.UnionFound(8, "ReferenceOrbitPicker", "DownloadCommitButton", "DownloadStatusText"), "DownloadCommitButton"));
    }

    // ---- 6. timeline pane (empty); the committed ones are below, playing last
    yield return new("timeline", "empty", "empty", c => { c.S.Click("Tab.TimelineTab"); Thread.Sleep(800); }, // SelectTab waits for the speed picker
      c => c.Fit(Steps.PaneOf(c, "TimelineTab"), "TimelineStartEpochBox")); // no speed picker before a commit
    yield return new("model-dust", "empty", "empty", c => Steps.OnlyOpen(c, "ModelTab", Steps.ModelSections, "DustSection"),
      c => Steps.Section(c, "DustSection"));
    yield return new("imports", "empty", "empty", c => c.S.SelectTab("ImportsTab"),
      c => c.Fit(Steps.PaneOf(c, "ImportsTab"), "ImportButton"));

    // ---- committed session
    yield return new("comet-simulated-range", "committed", "committed",
      c => Steps.OnlyOpen(c, "CometTab", Steps.CometSections, "SimulatedRangeSection"),
      c => Steps.Section(c, "SimulatedRangeSection"));
    yield return new("comet-range-advanced", "committed", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "CometTab", Steps.CometSections, "SimulatedRangeSection");
        Steps.OpenSection(c, "RangeAdvancedExpander");
        Steps.AlignTop(c, "RangeAdvancedExpander");
      },
      c => Steps.Section(c, "RangeAdvancedExpander"));
    yield return new("comet-nucleus", "committed", "committed",
      c => Steps.OnlyOpen(c, "CometTab", Steps.CometSections, "NucleusSection"),
      c => Steps.Section(c, "NucleusSection"));
    yield return new("comet-rotation", "committed", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "CometTab", Steps.CometSections, "RotationalModelSection");
        Steps.OpenSection(c, "PolePrecessionExpander");
      },
      c => Steps.Section(c, "RotationalModelSection"));

    yield return new("timeline", "committed", "committed", c => { Steps.Mode(c, "ModeBarOrbitButton"); c.S.SelectTab("TimelineTab"); },
      c => c.Fit(Steps.PaneOf(c, "TimelineTab"), "SpeedPicker"));
    yield return new("imports", "committed", "committed", c => c.S.SelectTab("ImportsTab"),
      c => c.Fit(Steps.PaneOf(c, "ImportsTab"), "ImportButton"));

    yield return new("model-dust", "committed", "committed", c => Steps.OnlyOpen(c, "ModelTab", Steps.ModelSections, "DustSection"),
      c => Steps.Section(c, "DustSection"));
    yield return new("model-dust-jet", "selected", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "ModelTab", Steps.ModelSections, "DustSection");
        Steps.SelectFirstJet(c);
        Steps.OpenSection(c, "JetGrainExpander");
        Steps.OpenSection(c, "JetActivityExpander");
        Steps.AlignTop(c, "JetLatitudeSlider");
      },
      c => c.Fit(c.UnionFound(8, "JetLatitudeSlider", "JetLongitudeSlider", "JetApertureSlider", "JetVelocityBox", "JetStreamColorPicker",
        "JetBetaBox", "JetBetaRangeText", "JetQDustBox", "JetGrainExpander", "JetActivityExpander", "JetSizeDistributionPlot", "JetDustRateCurvePlot"), "JetLatitudeSlider"));
    yield return new("model-dust-defaults", "committed", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "ModelTab", Steps.ModelSections, "DustSection");
        Steps.OpenSection(c, "ActivityAdvancedExpander");
        Steps.AlignTop(c, "JetDefaultsHelp");
      },
      c => c.Fit(c.UnionFound(8, "JetDefaultsHelp", "DustDensityBox", "DustSizeQBox", "SizeRangeNote", "ActivityAdvancedExpander", "DustRateCurvePlot"),
        "JetDefaultsHelp"));

    // ---- settings
    foreach (var (key, id) in new[]
             {
               // GeneralSection and OrbitalElementsSection are hidden in the app (IsVisible="False");
               // the Model tab's Sun and Nucleus spin sections are legacy (EnableLegacyExpanders): no shots
               ("settings-time-scale", "TimeScaleSection"),
               ("settings-particle-rendering", "ParticleRenderingSection"), ("settings-input-bindings", "InputBindingsSection"),
             })
      yield return new(key, "committed", "committed", c => Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, id),
        c => Steps.Section(c, id));
    yield return new("settings-comet-appearance", "committed", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "CometAppearanceSection");
        c.S.Click("CometDisplayModeDefault");
        Thread.Sleep(500);
      },
      c => Steps.Section(c, "CometAppearanceSection"));
    yield return new("settings-comet-appearance", "custom", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "CometAppearanceSection");
        c.S.Click("CometDisplayModeCustom");
        Thread.Sleep(600);
      },
      c => Steps.Section(c, "CometAppearanceSection"),
      c => c.S.Click("CometDisplayModeDefault"));
    yield return new("settings-billboards", "committed", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "BillboardsSection");
        EnsureBillboardRowOpen(c);
      },
      c => Steps.Section(c, "BillboardsSection"));
    yield return new("settings-billboards-fits", "committed", "committed",
      c =>
      {
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "BillboardsSection");
        EnsureBillboardRowOpen(c);
        Steps.OpenSection(c, "BillboardFitsSection");
        Steps.AlignTop(c, "BillboardFitsSection");
      },
      c => Steps.Section(c, "BillboardFitsSection"));

    // ---- viewport settings (the shots of the sample)
    foreach (var (state, button) in new[] { ("orbit", "ModeBarOrbitButton"), ("nucleus", "ModeBarNucleusButton"), ("model", "ModeBarModelButton") })
      yield return new("settings-viewport", state, "committed",
        c =>
        {
          Steps.Mode(c, button);
          Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "ViewportSettingsSection");
          Steps.CloseSection(c, "ProjectionAdvancedExpander");
        },
        c => Steps.Section(c, "ViewportSettingsSection"));
    yield return new("settings-viewport-persp", "orbit", "committed",
      c =>
      {
        Steps.Mode(c, "ModeBarOrbitButton");
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "ViewportSettingsSection");
        c.S.Click("ProjectionPerspective");
        Steps.OpenSection(c, "ProjectionAdvancedExpander");
        Steps.AlignTop(c, "ProjectionAdvancedExpander");
      },
      c => Steps.Section(c, "ProjectionAdvancedExpander"));
    yield return new("settings-viewport-ortho", "orbit", "committed",
      c =>
      {
        Steps.Mode(c, "ModeBarOrbitButton");
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "ViewportSettingsSection");
        c.S.Click("ProjectionOrthographic");
        Thread.Sleep(500);
        Steps.OpenSection(c, "ProjectionAdvancedExpander");
        Steps.AlignTop(c, "ProjectionAdvancedExpander");
      },
      c => Steps.Section(c, "ProjectionAdvancedExpander"),
      c => c.S.Click("ProjectionPerspective"));
    yield return new("settings-viewport-nucleus", "nucleus", "committed",
      c =>
      {
        Steps.Mode(c, "ModeBarNucleusButton");
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "ViewportSettingsSection");
        Steps.CloseSection(c, "ProjectionAdvancedExpander");
      },
      c => Steps.Section(c, "ViewportSettingsSection"));

    // ---- help flyout, combo drop-down (popups: the crop is the new window)
    yield return new("help-flyout", "flyout", "committed",
      c =>
      {
        // the Timeline pane never scrolls: no section juggling before the click
        c.S.Find("Tab.TimelineTab").Click();
        c.S.Find("SpeedPickerHelp");
        c.Snapshot();
        c.S.ClickWhenSettled("SpeedPickerHelp");
        // a click-opened flyout is pinned: it must still be up 1 s later (it used to vanish on
        // Win32 after a fraction of a second), and the pointer parking away does not close it
        Thread.Sleep(1000);
        try { c.NewPopup(TimeSpan.FromMilliseconds(300)); }
        catch (TimeoutException)
        {
          var h = c.S.Find("SpeedPickerHelp");
          var all = Win32.TopLevelWindows(c.S.Process.Id).Select(w => $"{w.ClassName}:{Win32.FrameBounds(w.Handle)}");
          throw new TimeoutException($"no popup; help at {h.Location} {h.Size}, windows {string.Join(" | ", all)}");
        }
      },
      c => new Cropped(Rectangle.Inflate(c.NewPopup(), 4, 4), []),
      c => c.Escape());
    yield return new("combo-dropdown", "dropdown", "committed",
      c =>
      {
        c.S.SelectTab("TimelineTab");
        c.Snapshot();
        c.S.Click("SpeedPicker");
        Thread.Sleep(600);
      },
      c => new Cropped(Rectangle.Inflate(c.NewPopup(), 4, 4), []),
      c => c.Escape());

    // ---- dialogs
    yield return new("dlg-settings", "dialog", "committed",
      c =>
      {
        c.S.Menu("Edit", "Settings...");
        c.Dialog = Steps.FindWindow(c, w => w.Title == "Settings");
        if (c.Dialog == IntPtr.Zero)
          throw new TimeoutException("the Settings window did not appear");
        Thread.Sleep(800);
      },
      c => new Cropped(Win32.FrameBounds(c.Dialog), []),
      c =>
      {
        if (c.Dialog != IntPtr.Zero)
          Win32.Close(c.Dialog);
        c.Dialog = IntPtr.Zero;
        Thread.Sleep(600);
      });

    // ---- the desktop billboard window (the session hides the image: show it, then hide it again)
    yield return new("billboard-window", "committed", "committed",
      c =>
      {
        Steps.ShowBillboardImage(c, visible: true);
        c.Dialog = Steps.FindWindow(c, w => w.Title == "AetherVk billboard", seconds: 15);
        if (c.Dialog == IntPtr.Zero)
          throw new TimeoutException("the billboard window did not appear");
      },
      c => new Cropped(Win32.FrameBounds(c.Dialog), []),
      c => { c.Dialog = IntPtr.Zero; Steps.ShowBillboardImage(c, visible: false); });

    // ---- viewport and overlay (the image is hidden, the dust settled)
    foreach (var (state, button) in new[] { ("orbit", "ModeBarOrbitButton"), ("nucleus", "ModeBarNucleusButton"), ("model", "ModeBarModelButton") })
      yield return new("viewport", state, "committed",
        c => { Steps.Mode(c, button); Steps.WaitDustSettled(c); },
        c => new Cropped(Steps.ViewportRect(c), []));
    foreach (var (state, button) in new[] { ("orbit", "ModeBarOrbitButton"), ("model", "ModeBarModelButton") })
      yield return new("overlay", state, "committed",
        c => { Steps.Mode(c, button); Steps.WaitDustSettled(c); },
        c =>
        {
          var callouts = new List<RawCallout>();
          foreach (var id in OverlayIds)
            if (c.S.Desktop.FindElementsByAccessibilityId(id).FirstOrDefault(e => e.Displayed) is { } e)
              callouts.Add(new RawCallout(id, e.Text, new Rectangle(e.Location, e.Size)));
          return new Cropped(Steps.ViewportRect(c), callouts);
        });

    // ---- the range lock (edits the pending range; the second shot puts it back)
    yield return new("main-modebar", "locked", "committed",
      c => { Steps.Mode(c, "ModeBarOrbitButton"); Steps.LockTheRange(c); },
      c => c.Fit(c.UnionFound(6, ModeBarIds), ModeBarIds[0]));
    yield return new("comet-simulated-range", "locked", "committed",
      c => Steps.OnlyOpen(c, "CometTab", Steps.CometSections, "SimulatedRangeSection"),
      c => Steps.Section(c, "SimulatedRangeSection"),
      Steps.UnlockTheRange);

    // ---- the more observer-heavy settings, then the clock (changes the epoch: last)
    yield return new("settings-viewport-observer", "nucleus", "committed",
      c =>
      {
        Steps.Mode(c, "ModeBarNucleusButton");
        Steps.OnlyOpen(c, "SettingsTab", Steps.SettingsSections, "ViewportSettingsSection");
        if (c.S.TryFind("ObserverTopocentricCheckBox") is null)
        {
          c.S.Click("ViewNucleusFromEarthButton");
          Thread.Sleep(1500);
        }
        Steps.AlignTop(c, "ViewNucleusFromEarthButton");
      },
      c => c.Fit(c.UnionFound(8, "ViewNucleusFromEarthButton", "NucleusExposureSlider", "ObserverTopocentricCheckBox", "ObserverNorthUpCheckBox",
        "ObserverLatSlider", "ObserverLonSlider", "EarthObserverScaleText"), "ViewNucleusFromEarthButton"),
      c => Steps.Mode(c, "ModeBarOrbitButton"));
    yield return new("timeline", "playing", "committed",
      c =>
      {
        Steps.Mode(c, "ModeBarOrbitButton");
        ReviewSave.ResetToStart(c.S);
        ReviewSave.Play(c.S);
        Thread.Sleep(1500);
      },
      c => c.Fit(Steps.PaneOf(c, "TimelineTab"), "SpeedPicker"),
      c => ReviewSave.Pause(c.S));
  }

  /// <summary>The Billboards section's first row (an expander without an id) open, so its fields are on screen.</summary>
  private static void EnsureBillboardRowOpen(ShotCtx c)
  {
    if (c.S.TryFind("BillboardX") is not null)
      return;
    var section = c.S.Find("BillboardsSection");
    var rows = section.FindElementsByClassName("Expander").Where(e => e.Displayed).ToList();
    var row = rows.FirstOrDefault(e => e.GetAttribute("AutomationId") is not ("BillboardFitsSection" or "BillboardsSection"))
              ?? throw new InvalidOperationException("no billboard row in the Billboards section");
    c.S.ScrollIntoView(row);
    if (!Steps.IsExpanded(row))
      Ui.ToggleExpander(row);
    Thread.Sleep(600);
    Ui.WaitFor(() => c.S.TryFind("BillboardX"), what: "the billboard row fields");
  }
}
