using System.Drawing;
using System.Globalization;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace AetherVk.UI.E2E;

/// <summary>
/// A point of observation the app offers: one camera mode of the viewport, picked by the top
/// mode bar for the active viewport, named in the overlay's frame label and persisted as the
/// save's <c>viewports[i].mode</c>. The Model view is the Earth observer (geocentric, or the
/// topocentric site), the Orbit view looks at the orbit plane from above the Sun, the Nucleus
/// view sits at the comet. The app offers no spacecraft or other observer.
/// </summary>
public sealed record Observer(string Name, string ModeBarButton, string CameraMode, string SaveMode, string[] FrameLabelStarts)
{
  public static readonly Observer Earth =
    new("Earth", "ModeBarModelButton", "EarthPosition", "earth_position", ["From Earth"]);

  public static readonly Observer Sun =
    new("Sun (orbit plane)", "ModeBarOrbitButton", "UpZenith", "up_zenith", ["Orbit-plane frame"]);

  public static readonly Observer Comet =
    new("Comet nucleus", "ModeBarNucleusButton", "CometOrbiting", "comet_orbiting", ["Body-fixed frame", "Nucleus from Earth"]);

  /// <summary>In the order the viewports get them (viewport i → All[i % 3]).</summary>
  public static readonly Observer[] All = [Earth, Sun, Comet];

  public bool Labels(string? frameLabel) =>
    frameLabel is not null && FrameLabelStarts.Any(p => frameLabel.StartsWith(p, StringComparison.Ordinal));

  /// <summary>The observer an overlay frame label names; null for an empty or unknown label.</summary>
  public static Observer? FromFrameLabel(string? frameLabel) => All.FirstOrDefault(o => o.Labels(frameLabel));

  public override string ToString() => Name;
}

/// <summary>One 3D viewport of the layout: its native host element and its rectangle (WinAppDriver's frame).</summary>
public sealed class Viewport(int index, WindowsElement host)
{
  public int Index { get; } = index;

  public WindowsElement Host { get; } = host;

  public Rectangle Rect { get; } = new(host.Location.X, host.Location.Y, host.Size.Width, host.Size.Height);

  public Point Centre => new(Rect.X + Rect.Width / 2, Rect.Y + Rect.Height / 2);

  public override string ToString() => $"viewport {Index + 1} at {Rect.X},{Rect.Y} {Rect.Width}×{Rect.Height}";
}

/// <summary>
/// What one viewport's overlay shows (its own top-level window over the viewport): the clock
/// label, the frame label, the scale bar, the ephemeris table, the Earth observer HUD and the
/// viewport rectangle the app reports (the last two with the diagnostic panels on).
/// </summary>
public sealed record ViewportReadout(
  string? EpochLabel, string? FrameLabel, string? Ruler,
  string? EphemerisEpoch, string? R, string? Delta, string? Phase, string? Elongation, string? PaOfSun, string? PsAng,
  string? EarthObserver, string? DesktopRect)
{
  public Observer? Observer => Observer.FromFrameLabel(FrameLabel);

  /// <summary>The ephemeris rows a committed comet fills ("—" without one).</summary>
  public bool HasEphemeris => R is { Length: > 0 } r && r != ViewportProbe.NoValue && Delta is { Length: > 0 } d && d != ViewportProbe.NoValue;

  /// <summary>The geocentric ephemeris rows, which do not depend on the viewport's camera.</summary>
  public string EphemerisKey => string.Join(" | ", EphemerisEpoch, R, Delta, Phase, Elongation, PaOfSun, PsAng);

  public override string ToString() =>
    $"epoch \"{EpochLabel}\", frame \"{FrameLabel}\", ruler \"{Ruler}\", r {R}, Δ {Delta}, phase {Phase}, PA of Sun {PaOfSun}, PsAng {PsAng}" +
    (string.IsNullOrEmpty(EarthObserver) ? string.Empty : $", Earth observer \"{EarthObserver}\"");
}

/// <summary>
/// The viewports of the layout and what each one shows, for a layout holding several of them at
/// once: every overlay element is read per viewport by the rectangle it lies in, the active
/// viewport is chosen with a click in it before a mode bar button.
/// </summary>
public static class ViewportProbe
{
  public const string NoValue = "—";

  /// <summary>The overlay's ids read per viewport.</summary>
  private static readonly string[] OverlayIds =
  [
    "ViewportEpochLabel", "ViewportFrameLabel", "ViewportRulerText",
    "EphemerisEpoch", "EphemerisR", "EphemerisDelta", "EphemerisPhase", "EphemerisElongation", "EphemerisPaOfSun", "EphemerisPsAng",
    "EarthObserverStatus", "DiagnosticsViewportDesktopRect",
  ];

  /// <summary>A camera mode switch flies the camera for 2.5 s; the dust hold and full render follow.</summary>
  public static readonly TimeSpan ModeSettle = TimeSpan.FromSeconds(6);

  /// <summary>The shown 3D viewports, left to right then top to bottom.</summary>
  public static IReadOnlyList<Viewport> Discover(AppSession s)
  {
    var hosts = s.App.FindElementsByAccessibilityId("ViewportHost")
      .Where(e => e.Displayed && e.Size.Width > 50 && e.Size.Height > 50)
      .OrderBy(e => e.Location.X).ThenBy(e => e.Location.Y)
      .ToList();
    return hosts.Select((h, i) => new Viewport(i, h)).ToList();
  }

  /// <summary>
  /// The displayed overlay element with <paramref name="automationId"/> whose centre lies in the
  /// viewport (each overlay window sits over its own viewport); null when there is none.
  /// </summary>
  public static string? OverlayText(AppSession s, Viewport vp, string automationId)
  {
    var rect = vp.Rect;
    rect.Inflate(6, 6);
    foreach (var e in s.Desktop.FindElementsByAccessibilityId(automationId))
    {
      try
      {
        if (!e.Displayed)
          continue;
        var (p, size) = (e.Location, e.Size);
        if (rect.Contains(p.X + size.Width / 2, p.Y + size.Height / 2))
          return e.Text;
      }
      catch (WebDriverException)
      {
        // an overlay rebuilt under the read: the next poll sees the new one
      }
    }
    return null;
  }

  public static ViewportReadout Read(AppSession s, Viewport vp)
  {
    var t = OverlayIds.ToDictionary(id => id, id => OverlayText(s, vp, id));
    return new ViewportReadout(
      t["ViewportEpochLabel"], t["ViewportFrameLabel"], t["ViewportRulerText"],
      t["EphemerisEpoch"], t["EphemerisR"], t["EphemerisDelta"], t["EphemerisPhase"], t["EphemerisElongation"],
      t["EphemerisPaOfSun"], t["EphemerisPsAng"],
      t["EarthObserverStatus"], t["DiagnosticsViewportDesktopRect"]);
  }

  public static IReadOnlyList<ViewportReadout> ReadAll(AppSession s, IReadOnlyList<Viewport> viewports) =>
    viewports.Select(vp => Read(s, vp)).ToList();

  /// <summary>
  /// Makes <paramref name="vp"/> the active viewport (the one the mode bar drives): a click in its
  /// upper middle, clear of the HUD (top right), the legend (top left) and the bottom labels.
  /// </summary>
  public static void Activate(AppSession s, Viewport vp)
  {
    s.Foreground();
    Win32.ClickAt(vp.Rect.X + vp.Rect.Width / 2, vp.Rect.Y + vp.Rect.Height / 4);
    Thread.Sleep(400);
  }

  /// <summary>
  /// Puts <paramref name="vp"/> on <paramref name="observer"/> through the mode bar and waits for
  /// its frame label to name it, then for the camera flight and the dust render to settle.
  /// </summary>
  public static void SetObserver(AppSession s, Viewport vp, Observer observer, TimeSpan? settle = null)
  {
    Activate(s, vp);
    Ui.WaitUntil(() => s.Find(observer.ModeBarButton).Enabled, TimeSpan.FromSeconds(60), $"{observer.ModeBarButton} enabled");
    // the mode bar is above the scroll band ScrollIntoView keeps clear: a direct click
    s.Find(observer.ModeBarButton).RealClick();
    Ui.WaitUntil(() => Observer.FromFrameLabel(OverlayText(s, vp, "ViewportFrameLabel")) == observer, TimeSpan.FromSeconds(45),
      $"{vp} on the {observer} observer (its frame label)");
    Thread.Sleep(settle ?? ModeSettle);
  }

  /// <summary>
  /// Opens one more 3D viewport beside <paramref name="beside"/>: the tab strip's "+" flyout
  /// ("Viewport 3D"), then the new tab's drag handle dropped on the right edge zone of its group
  /// (a horizontal split). False when the flyout offers no viewport (the layout refuses a second
  /// one) or the split did not take.
  /// </summary>
  public static bool TryAddViewportBeside(AppSession s, Viewport beside, Action<string>? log = null)
  {
    int before = Discover(s).Count;
    var header = TabHeaderOf(s, beside);
    if (header is null)
    {
      log?.Invoke($"no Tab.Viewport3D header above {beside}");
      return false;
    }
    var add = AddTabButtonFor(s, header);
    if (add is null)
    {
      log?.Invoke("no \"+\" (add tab) button in the viewport's tab strip");
      return false;
    }
    s.Foreground();
    add.RealClick();
    WindowsElement? item;
    try
    {
      item = Ui.WaitFor(() => s.Desktop.FindElementsByName("Viewport 3D").FirstOrDefault(e => e.Displayed && e.TagName != "Text"),
        TimeSpan.FromSeconds(4), "the \"Viewport 3D\" flyout item");
    }
    catch (TimeoutException)
    {
      Win32.PressKey(Win32.VK_ESCAPE);
      Thread.Sleep(300);
      log?.Invoke("the add-tab flyout offers no \"Viewport 3D\": the layout holds a single 3D viewport");
      return false;
    }
    item.RealClick();
    Thread.Sleep(1500);

    // the new tab opened in the same group: its header is the last Tab.Viewport3D of the strip
    var headers = TabHeaders(s).Where(h => Math.Abs(h.Location.Y - header.Location.Y) < 20).OrderBy(h => h.Location.X).ToList();
    if (headers.Count < 2)
    {
      log?.Invoke($"after the flyout the strip has {headers.Count} viewport tab(s): the viewport was refused");
      return false;
    }
    var newTab = headers[^1];
    // the drag handle sits left of the title (an 8 px margin, 12 px icon, the icon glyph): two
    // starts are tried; the drop lands in the group's right edge zone (the last 30 %)
    foreach (int handleDx in new[] { 16, 36 })
    {
      int x0 = newTab.Location.X - handleDx, y0 = newTab.Location.Y + newTab.Size.Height / 2;
      int x1 = beside.Rect.Right - beside.Rect.Width / 10, y1 = beside.Rect.Y + beside.Rect.Height / 2;
      log?.Invoke($"drag the new viewport tab from ({x0}, {y0}) to ({x1}, {y1})");
      s.Foreground();
      Win32.DragFromTo(x0, y0, x1, y1, steps: 30);
      Thread.Sleep(2000);
      if (Discover(s).Count > before)
        return true;
    }
    log?.Invoke("the tab drag did not split the group");
    return false;
  }

  private static IReadOnlyList<WindowsElement> TabHeaders(AppSession s) =>
    s.App.FindElementsByAccessibilityId("Tab.Viewport3D").Where(e => e.Displayed).ToList();

  /// <summary>The viewport tab header of the strip above <paramref name="vp"/> (its group's strip).</summary>
  private static WindowsElement? TabHeaderOf(AppSession s, Viewport vp) =>
    TabHeaders(s)
      .Where(h => h.Location.Y < vp.Rect.Y && h.Location.X >= vp.Rect.X - 10 && h.Location.X < vp.Rect.Right)
      .OrderByDescending(h => h.Location.Y)
      .FirstOrDefault();

  /// <summary>
  /// The strip's "+" (add tab) button: <c>AddTabButton</c> when the view names it, else the "+"
  /// button on the header's row, right of it.
  /// </summary>
  private static WindowsElement? AddTabButtonFor(AppSession s, WindowsElement header)
  {
    bool OnRow(WindowsElement e) => Math.Abs(e.Location.Y + e.Size.Height / 2 - (header.Location.Y + header.Size.Height / 2)) < 20
                                   && e.Location.X > header.Location.X;
    var named = s.App.FindElementsByAccessibilityId("AddTabButton").Where(e => e.Displayed && OnRow(e)).OrderBy(e => e.Location.X).FirstOrDefault();
    if (named is not null)
      return named;
    return s.App.FindElementsByName("+")
      .Where(e => e.Displayed && e.Size.Width <= 40 && OnRow(e))
      .OrderBy(e => e.Location.X)
      .FirstOrDefault();
  }

  /// <summary>
  /// The <paramref name="index"/>-th viewport's section of the Settings tab ("Viewport N", one per
  /// camera in creation order), expanded.
  /// </summary>
  public static WindowsElement SettingsSection(AppSession s, int index)
  {
    s.SelectTab("SettingsTab");
    var section = Ui.WaitFor(() =>
    {
      var sections = s.App.FindElementsByAccessibilityId("ViewportSettingsSection").Where(e => e.Displayed).OrderBy(e => e.Location.Y).ToList();
      return sections.Count > index ? sections[index] : null;
    }, what: $"the settings section of viewport {index + 1}");
    s.ScrollIntoView(section);
    for (int attempt = 0; attempt < 2 && section.GetAttribute("ExpandCollapse.ExpandCollapseState") != "Expanded"; attempt++)
    {
      Ui.ToggleExpander(section);
      Thread.Sleep(300);
    }
    return section;
  }

  /// <summary>A control of the viewport's settings section, scrolled into view.</summary>
  public static AppiumWebElement SectionControl(AppSession s, WindowsElement section, string automationId)
  {
    var e = Ui.WaitFor(() => section.FindElementsByAccessibilityId(automationId).FirstOrDefault(),
      what: $"'{automationId}' in the viewport's settings section");
    return s.ScrollIntoView(e);
  }

  /// <summary>The viewport's luminance grid (<see cref="FrameMetrics.Sample"/>).</summary>
  public static double[] Sample(Viewport vp) => FrameMetrics.Sample(vp.Host);

  /// <summary>The mean luminance of a grid: 0 on a black viewport.</summary>
  public static double Mean(double[] grid) => grid.Average();

  /// <summary>The ring around the viewport's centre against the sky in its corners (the nucleus is centred in the comet view).</summary>
  public static (double Ring, double Sky) CentreRingAndSky(Viewport vp) =>
    FrameMetrics.RingAndSky(vp.Host, (vp.Centre.X, vp.Centre.Y));

  /// <summary>"yyyy-MM-dd HH:mm (UTC)" / "yyyy-MM-dd HH:mm UTC" labels as a date; null for an empty label.</summary>
  public static DateTime? LabelEpoch(string? label)
  {
    if (string.IsNullOrWhiteSpace(label))
      return null;
    try
    {
      return ReviewSave.ParseEpoch(label);
    }
    catch (FormatException)
    {
      return null;
    }
  }

  /// <summary>A label to the minute against the clock box to the second, in either time scale (UTC / TDB differ by 69 s).</summary>
  public static bool SameMinute(DateTime? label, DateTime clock) =>
    label is { } l && (l - clock).Duration() <= TimeSpan.FromSeconds(60 + 70);

  /// <summary>The viewport modes of a save's index (<c>viewports[i].mode</c>, in camera order).</summary>
  public static List<string> SavedViewportModes(string indexJsonPath)
  {
    using var doc = System.Text.Json.JsonDocument.Parse(File.ReadAllText(indexJsonPath));
    var modes = new List<string>();
    if (doc.RootElement.TryGetProperty("viewports", out var viewports))
      foreach (var v in viewports.EnumerateArray())
        modes.Add(v.TryGetProperty("mode", out var m) ? m.GetString() ?? string.Empty : string.Empty);
    return modes;
  }

  public static string Describe(IReadOnlyList<Viewport> viewports, IReadOnlyList<ViewportReadout> readouts) =>
    string.Join("\n", viewports.Zip(readouts, (vp, r) => $"  {vp}: {r}"));

  public static string Pct(double v) => v.ToString("P1", CultureInfo.InvariantCulture);
}
