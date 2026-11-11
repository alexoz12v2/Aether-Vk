using System.Globalization;
using System.Text.RegularExpressions;
using AetherVk.UI.E2E.Shared;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;

namespace AetherVk.UI.E2E.Mac;

/// <summary>
/// The shared scenarios' seam (<see cref="IE2EApp"/>) over a Mac2 session:
/// <list type="bullet">
/// <item>lookup by <c>accessibility id</c> (Avalonia's AutomationId as the AX identifier: TreeDump proves it);</item>
/// <item>element frames in points (the <c>frame</c> attribute) → screen pixels by <see cref="MacScreen.Scale"/>;</item>
/// <item>menus in the menu bar: <c>XCUIElementTypeMenuBarItem</c> → <c>XCUIElementTypeMenuItem</c> by title;</item>
/// <item>keys through <c>macos: keys</c> (chords with XCUIKeyModifierFlags), clicks at points through <c>macos: click</c>;</item>
/// <item>NSOpenPanel / NSSavePanel: Cmd+Shift+G, the path, Return, Return (Save: the name field first);</item>
/// <item>pixels through CoreGraphics (<see cref="MacScreen"/>).</item>
/// </list>
/// Checked state: Mac2 exposes <c>value</c> and <c>selected</c>; <see cref="IsChecked"/> reads
/// <c>value</c> first ("1"/"0", "true"/"false", a check mark), then <c>selected</c>. Which one a
/// NativeMenu radio item and an Avalonia toggle actually fill is what
/// <c>MenuBar_LayoutRadio_ReflectsState</c> records (it logs both raw values).
/// </summary>
public sealed partial class MacE2EApp(MacAppSession session) : IE2EApp
{
  // XCUIKeyModifierFlags
  public const int Shift = 1 << 1;
  public const int Control = 1 << 2;
  public const int Option = 1 << 3;
  public const int Command = 1 << 4;

  // XCUIKeyboardKey values
  public const string Return = "\r";
  public const string Escape = "\u001b";
  public const string UpArrow = "";
  public const string DownArrow = "";

  public MacAppSession Session { get; } = session;

  private ISearchContext Root => Session.Driver;

  public string Platform => "macos";

  public int ProcessId => Session.Process.Id;

  public IReadOnlyList<string> Lines => Session.Lines.Select(l => l.Line).ToList();

  public string WindowTitle => MacAppSession.MainWindowOf(Session.Driver)?.GetAttribute("title") ?? string.Empty;

  // ── driver commands ─────────────────────────────────────────────────────────

  public object? Exec(string command, Dictionary<string, object> args) => Session.Driver.ExecuteScript(command, args);

  /// <summary><c>macos: keys</c>: each item a string typed as is, or a key with modifier flags.</summary>
  public void Keys(params object[] items)
  {
    Exec("macos: keys", new Dictionary<string, object> { ["keys"] = items });
    Thread.Sleep(150);
  }

  public static Dictionary<string, object> Key(string key, int modifiers = 0) =>
    new() { ["key"] = key, ["modifierFlags"] = modifiers };

  public void Chord(string key, int modifiers) => Keys(Key(key, modifiers));

  /// <summary>A click at a screen point (points, not pixels).</summary>
  public void ClickAtPoint(double x, double y)
  {
    Exec("macos: click", new Dictionary<string, object> { ["x"] = x, ["y"] = y });
    Thread.Sleep(150);
  }

  // ── lookup ──────────────────────────────────────────────────────────────────

  private static string? Attr(IWebElement e, string name)
  {
    try
    {
      return e.GetAttribute(name);
    }
    catch (WebDriverException)
    {
      return null;
    }
  }

  private static bool Shown(IWebElement e)
  {
    try
    {
      return e.Displayed;
    }
    catch (WebDriverException)
    {
      return false;
    }
  }

  public IWebElement? TryElement(string automationId) =>
    Root.FindElements(MobileBy.AccessibilityId(automationId)).FirstOrDefault(Shown);

  public IWebElement Element(string automationId, TimeSpan? timeout = null) =>
    E2EWait.For(() => TryElement(automationId), timeout, $"element '{automationId}'");

  [GeneratedRegex(@"-?\d+(?:\.\d+)?")]
  private static partial Regex Numbers();

  /// <summary>The element's frame in points: the <c>frame</c> attribute ("{{x, y}, {w, h}}"), else its rect.</summary>
  public static PointRect FramePoints(IWebElement e)
  {
    if (Attr(e, "frame") is { } frame)
    {
      var n = Numbers().Matches(frame).Select(m => double.Parse(m.Value, CultureInfo.InvariantCulture)).ToList();
      if (n.Count >= 4)
        return new PointRect(n[0], n[1], n[2], n[3]);
    }
    return new PointRect(e.Location.X, e.Location.Y, e.Size.Width, e.Size.Height);
  }

  public static E2ERect PixelRect(IWebElement e) => ScreenScale.ToPixels(FramePoints(e), MacScreen.Scale);

  private static E2EElement Snapshot(string id, IWebElement e) =>
    new(id, PixelRect(e), ReadText(e), e.Enabled, Shown(e));

  public E2EElement? TryFind(string automationId)
  {
    try
    {
      return TryElement(automationId) is { } e ? Snapshot(automationId, e) : null;
    }
    catch (WebDriverException)
    {
      return null;
    }
  }

  public E2EElement Find(string automationId, TimeSpan? timeout = null) =>
    Snapshot(automationId, Element(automationId, timeout));

  private PointRect WindowFrame() =>
    MacAppSession.MainWindowOf(Session.Driver) is { } w ? FramePoints(w) : new PointRect(0, 0, 100_000, 100_000);

  /// <summary>
  /// Scrolls the panel under the element until its centre is inside the window's content band
  /// (a native scroll at a point of the window; the direction is learnt from the first step).
  /// </summary>
  public IWebElement ScrollIntoView(IWebElement e)
  {
    var win = WindowFrame();
    double top = win.Y + 60, bottom = win.Y + win.Height - 30;
    double sign = 1;
    double? lastY = null;
    for (int i = 0; i < 40; i++)
    {
      var f = FramePoints(e);
      double cy = f.Y + f.Height / 2;
      int dir = cy > bottom ? 1 : cy < top ? -1 : 0;
      if (dir == 0)
        return e;
      // a step that moved the element the wrong way (or not at all) flips the direction
      if (lastY is { } ly && (dir > 0 ? cy >= ly : cy <= ly))
        sign = -sign;
      lastY = cy;
      double x = Math.Clamp(f.X + Math.Min(f.Width / 2, 40), win.X + 5, win.X + win.Width - 5);
      double y = Math.Clamp(cy, top + 10, bottom - 10);
      Exec("macos: scroll", new Dictionary<string, object> { ["x"] = x, ["y"] = y, ["deltaX"] = 0, ["deltaY"] = -dir * sign * 5 });
      Thread.Sleep(200);
    }
    return e;
  }

  // ── input ───────────────────────────────────────────────────────────────────

  public void Click(string automationId)
  {
    var e = ScrollIntoView(Element(automationId));
    e.Click();
    Thread.Sleep(200);
  }

  public void ClickAt(int x, int y)
  {
    var (px, py) = ScreenScale.PixelToPoint(x, y, MacScreen.Scale);
    ClickAtPoint(px, py);
  }

  private static bool IsTextField(IWebElement e) =>
    e.TagName is "XCUIElementTypeTextField" or "XCUIElementTypeTextView" or "XCUIElementTypeSearchField";

  /// <summary>The slider's "&lt;id&gt;.Input", else the element if it is a text field, else its first inner text field.</summary>
  private IWebElement TextTarget(string automationId)
  {
    if (TryElement(automationId + ".Input") is { } input)
      return input;
    var e = Element(automationId);
    if (IsTextField(e))
      return e;
    return e.FindElements(By.XPath(".//XCUIElementTypeTextField")).FirstOrDefault()
           ?? e.FindElements(By.XPath(".//XCUIElementTypeTextView")).FirstOrDefault()
           ?? e;
  }

  public void TypeInto(string automationId, string text, bool enter = true)
  {
    Foreground();
    var box = ScrollIntoView(TextTarget(automationId));
    box.Click();
    Thread.Sleep(150);
    Chord("a", Command);
    Keys(text);
    if (enter)
      Keys(Return);
    Thread.Sleep(250);
  }

  /// <summary>A click near the element's top-left (an expander's header chevron, a check box's box).</summary>
  public void Toggle(string automationId)
  {
    var e = ScrollIntoView(Element(automationId));
    var f = FramePoints(e);
    ClickAtPoint(f.X + Math.Min(f.Width / 2, 14), f.Y + Math.Min(f.Height / 2, 14));
    Thread.Sleep(300);
  }

  /// <summary>value ("1"/"0", "true"/"false", a check mark) first, then selected.</summary>
  public static bool? CheckedFrom(string? value, string? selected)
  {
    switch (value?.Trim())
    {
      case "1" or "true" or "True" or "✓" or "✔" or "on":
        return true;
      case "0" or "false" or "False" or "off":
        return false;
    }
    return selected?.Trim() switch
    {
      "true" or "True" or "1" => true,
      "false" or "False" or "0" => false,
      _ => null,
    };
  }

  public bool? IsChecked(string automationId)
  {
    var e = Element(automationId);
    return CheckedFrom(Attr(e, "value"), Attr(e, "selected"));
  }

  private static string ReadText(IWebElement e)
  {
    var value = Attr(e, "value");
    if (!string.IsNullOrEmpty(value))
      return value;
    string? text = null;
    try { text = e.Text; } catch (WebDriverException) { }
    if (!string.IsNullOrEmpty(text))
      return text;
    return Attr(e, "title") is { Length: > 0 } title ? title : Attr(e, "label") ?? string.Empty;
  }

  public string Text(string automationId) => ReadText(TextTarget(automationId));

  /// <summary>
  /// A list's rows: <c>XCUIElementTypeCell</c>, else <c>XCUIElementTypeRow</c>, else its direct
  /// children (how Avalonia's ListBoxItem appears is recorded by TreeDump).
  /// </summary>
  public IReadOnlyList<IWebElement> ListItems(string listId)
  {
    var list = Element(listId);
    foreach (var xpath in new[] { ".//XCUIElementTypeCell", ".//XCUIElementTypeRow", "./*" })
    {
      var items = list.FindElements(By.XPath(xpath)).Where(Shown).ToList();
      if (items.Count > 0)
        return items;
    }
    return [];
  }

  public int ListCount(string listId) => ListItems(listId).Count;

  public void SelectListItem(string listId, int index)
  {
    var item = E2EWait.For(() => ListItems(listId).ElementAtOrDefault(index), what: $"item {index} of {listId}");
    ScrollIntoView(item).Click();
    Thread.Sleep(300);
  }

  /// <summary>Opens the combo box with a click, Up to the first item, Down <paramref name="index"/> times, Return.</summary>
  public void ChooseAt(string comboId, int index)
  {
    Click(comboId);
    Thread.Sleep(400);
    for (int i = 0; i < 12; i++)
      Keys(UpArrow);
    for (int i = 0; i < index; i++)
      Keys(DownArrow);
    Keys(Return);
    Thread.Sleep(300);
  }

  public void PressKey(char key) => Keys(key.ToString());

  // ── menus ───────────────────────────────────────────────────────────────────

  /// <summary>NativeMenu titles as typed, or with "..." as the ellipsis character.</summary>
  private static IEnumerable<string> TitleVariants(string title) =>
    title.Contains("...", StringComparison.Ordinal) ? new[] { title, title.Replace("...", "…", StringComparison.Ordinal) } : new[] { title };

  private IWebElement MenuBarItem(string title) =>
    E2EWait.For(() => TitleVariants(title)
      .SelectMany(t => Root.FindElements(By.XPath($"//XCUIElementTypeMenuBarItem[@title=\"{t}\"]")))
      .FirstOrDefault(), what: $"menu bar item '{title}'");

  private IWebElement MenuItemUnder(IWebElement parent, string title) =>
    E2EWait.For(() => TitleVariants(title)
      .SelectMany(t => parent.FindElements(By.XPath($".//XCUIElementTypeMenuItem[@title=\"{t}\"]")))
      .FirstOrDefault(), what: $"menu item '{title}'");

  public void Menu(params string[] path)
  {
    Foreground();
    IWebElement current = MenuBarItem(path[0]);
    current.Click();
    foreach (var entry in path.Skip(1))
    {
      Thread.Sleep(300);
      current = MenuItemUnder(current, entry);
      current.Click();
    }
    Thread.Sleep(300);
  }

  /// <summary>
  /// The checked state of a menu item and the raw attributes it was read from: the menu is opened,
  /// read and closed with Escape.
  /// </summary>
  public (bool? Checked, string Raw) MenuItemState(string menu, string item)
  {
    Foreground();
    var bar = MenuBarItem(menu);
    bar.Click();
    Thread.Sleep(400);
    try
    {
      var e = MenuItemUnder(bar, item);
      string? value = Attr(e, "value"), selected = Attr(e, "selected");
      return (CheckedFrom(value, selected), $"value=\"{value}\" selected=\"{selected}\"");
    }
    finally
    {
      Keys(Escape);
      Thread.Sleep(300);
    }
  }

  // ── file dialogs ────────────────────────────────────────────────────────────

  private IWebElement? OpenPanel() =>
    Root.FindElements(By.XPath("//XCUIElementTypeSheet")).FirstOrDefault(Shown)
    ?? Root.FindElements(By.XPath("//XCUIElementTypeDialog")).FirstOrDefault(Shown);

  private IWebElement? WaitForPanel()
  {
    try
    {
      return E2EWait.For(OpenPanel, TimeSpan.FromSeconds(20), "a file panel");
    }
    catch (TimeoutException)
    {
      return null; // a panel shown as a plain window: answered by keys all the same
    }
  }

  /// <summary>Cmd+Shift+G, the path, Return: the panel goes to the folder (and selects a named file).</summary>
  private void GoTo(string path)
  {
    Chord("g", Command | Shift);
    Thread.Sleep(800);
    Chord("a", Command);
    Keys(path);
    Thread.Sleep(300);
    Keys(Return);
    Thread.Sleep(1000);
  }

  private void WaitPanelGone() =>
    E2EWait.Until(() => OpenPanel() is null, TimeSpan.FromSeconds(15), "the file panel to close");

  public void AnswerOpenDialog(string path)
  {
    WaitForPanel();
    Thread.Sleep(500);
    GoTo(Path.GetFullPath(path));
    Keys(Return); // Open
    WaitPanelGone();
  }

  public void AnswerSaveDialog(string path)
  {
    var full = Path.GetFullPath(path);
    var panel = WaitForPanel();
    Thread.Sleep(500);
    var name = panel?.FindElements(By.XPath(".//XCUIElementTypeTextField[@identifier=\"saveAsNameTextField\"]")).FirstOrDefault()
               ?? panel?.FindElements(By.XPath(".//XCUIElementTypeTextField")).FirstOrDefault();
    name?.Click();
    Chord("a", Command);
    Keys(Path.GetFileName(full));
    GoTo(Path.GetDirectoryName(full)!);
    Keys(Return); // Save
    Thread.Sleep(800);
    // an existing file: confirm the replace sheet
    if (Root.FindElements(By.XPath("//XCUIElementTypeButton[@title=\"Replace\"]")).FirstOrDefault(Shown) is { } replace)
      replace.Click();
    WaitPanelGone();
  }

  // ── pixels ──────────────────────────────────────────────────────────────────

  public E2ERect ViewportRect(string automationId) => PixelRect(Element(automationId));

  public double[] CaptureLuminance(E2ERect rect) => MacScreen.CaptureLuminance(rect);

  public double[] SampleLuminance(E2ERect rect, int n) =>
    FrameMetricsCore.Grid(CaptureLuminance(rect), rect.Width, rect.Height, n);

  public void SavePng(E2ERect rect, string path)
  {
    var image = MacScreen.Capture(rect);
    PngCodec.WriteRgb(path, image.Rgba, image.Width, image.Height);
  }

  /// <summary>Mac2's screenshot of one element, at the element's pixel size (the probe cross-check).</summary>
  public PngImage ElementScreenshot(string automationId)
  {
    var e = Element(automationId);
    var shot = ((ITakesScreenshot)e).GetScreenshot();
    return PngCodec.Decode(shot.AsByteArray);
  }

  public void SaveScreenshot(string path)
  {
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
    File.WriteAllBytes(path, Session.Driver.GetScreenshot().AsByteArray);
  }

  // ── lifetime ────────────────────────────────────────────────────────────────

  public void EnsureAlive(string doing)
  {
    if (Session.IsAlive)
      return;
    string code;
    try { code = Session.Process.ExitCode.ToString(CultureInfo.InvariantCulture); }
    catch (InvalidOperationException) { code = "unknown"; }
    var tail = Session.Lines.TakeLast(15).Select(l => l.Line);
    throw new MacAppGoneException($"AetherVk closed while {doing} (exit {code}); log tail:\n{string.Join('\n', tail)}");
  }

  public void Foreground()
  {
    try
    {
      Exec("macos: activateApp", new Dictionary<string, object> { ["bundleId"] = MacAppSession.BundleId() });
      Thread.Sleep(200);
    }
    catch (WebDriverException)
    {
    }
  }

  /// <summary>Cmd+Q; a save prompt is answered with Discard; killed after the timeout.</summary>
  public E2EExit Quit(TimeSpan? timeout = null)
  {
    var p = Session.Process;
    int? Code()
    {
      try { return p.ExitCode; }
      catch (InvalidOperationException) { return null; } // not our child (driver mode)
    }
    if (!Session.IsAlive)
      return new E2EExit(true, Code(), false);
    Foreground();
    try
    {
      Chord("q", Command);
    }
    catch (WebDriverException)
    {
    }
    if (!p.WaitForExit(5_000))
      try
      {
        E2EWait.For(() => TryElement("Discard"), TimeSpan.FromSeconds(10), "the save prompt").Click();
      }
      catch (Exception e) when (e is TimeoutException or WebDriverException) { }
    if (p.WaitForExit((int)(timeout ?? TimeSpan.FromSeconds(30)).TotalMilliseconds))
      return new E2EExit(true, Code(), false);
    p.Kill(entireProcessTree: true);
    p.WaitForExit(5_000);
    return new E2EExit(true, null, true);
  }

  public void Dispose() => Session.Dispose();
}

/// <summary>The app closed under a running test.</summary>
public sealed class MacAppGoneException(string message) : Exception(message);

/// <summary>Launches the packaged app for the shared scenarios; artifacts under TestResults/e2e-mac/.</summary>
public sealed class MacE2EHost : IE2EHost
{
  public string Platform => "macos";

  public string ArtifactsRoot => Path.Combine(E2ESteps.RepoRoot(), "TestResults", "e2e-mac");

  public string? UnavailableReason() => MacAppSession.UnavailableReason();

  public IE2EApp Launch(string logPath, string culture = E2ESteps.Culture) =>
    new MacE2EApp(MacAppSession.Launch(logPath, culture));

  /// <summary>macOS crash reports of the app (~/Library/Logs/DiagnosticReports/AetherVk*) and the app's own.</summary>
  public IReadOnlyList<string> CrashReportsSince(DateTime sinceUtc)
  {
    var home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
    var found = new List<string>();
    var diagnostic = Path.Combine(home, "Library", "Logs", "DiagnosticReports");
    if (Directory.Exists(diagnostic))
      found.AddRange(Directory.GetFiles(diagnostic, "AetherVk*").Where(f => File.GetLastWriteTimeUtc(f) >= sinceUtc));
    var own = MacAppSession.LogDirectory();
    if (Directory.Exists(own))
      found.AddRange(Directory.GetFiles(own, "aethervk-crash-*.txt").Where(f => File.GetLastWriteTimeUtc(f) >= sinceUtc));
    return found;
  }
}
