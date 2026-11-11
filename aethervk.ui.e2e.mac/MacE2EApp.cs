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
  public const string LeftArrow = "\uF702";
  public const string RightArrow = "\uF703";
  public const string UpArrow = "";
  public const string DownArrow = "";

  public MacAppSession Session { get; } = session;

  /// <summary>
  /// Every scroll-into-view and click, one line each, in TestResults/e2e-mac/clicks.log: where a
  /// scenario's click went is otherwise only in a screenshot.
  /// </summary>
  /// <summary>A screenshot next to clicks.log (TestResults/e2e-mac/trace-&lt;name&gt;.png).</summary>
  private void Shot(string name)
  {
    try
    {
      SaveScreenshot(Path.Combine(E2ESteps.RepoRoot(), "TestResults", "e2e-mac", $"trace-{name}.png"));
    }
    catch (Exception e) when (e is IOException or WebDriverException or InvalidOperationException)
    {
    }
  }

  private static void Trace(string line)
  {
    try
    {
      File.AppendAllText(Path.Combine(E2ESteps.RepoRoot(), "TestResults", "e2e-mac", "clicks.log"),
        $"{DateTime.Now:HH:mm:ss.fff} {line}{Environment.NewLine}");
    }
    catch (IOException) { }
  }

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
    // Mac2 takes one character per string item ("Input key '…' is too long … Use typeText"):
    // a longer string goes as its characters
    var keys = items
      .SelectMany(i => i is string { Length: > 1 } s ? s.Select(c => (object)c.ToString()) : [i])
      .ToArray();
    Exec("macos: keys", new Dictionary<string, object> { ["keys"] = keys });
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
    // one round trip (each costs ~250 ms): Mac2's "frame" attribute is a JSON object, which
    // GetAttribute does not turn into text, and Location and Size are a request each
    if (e is AppiumElement ae)
    {
      var r = ae.Rect;
      return new PointRect(r.X, r.Y, r.Width, r.Height);
    }
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

  /// <summary>The main window's frame in points (screen coordinates, top-left origin).</summary>
  /// <remarks>Retried while the window element goes stale (full-screen and tiling transitions).</remarks>
  public PointRect WindowFrame()
  {
    for (int attempt = 0; ; attempt++)
    {
      try
      {
        return MacAppSession.MainWindowOf(Session.Driver) is { } w ? FramePoints(w) : new PointRect(0, 0, 100_000, 100_000);
      }
      catch (WebDriverException) when (attempt < 5)
      {
        Thread.Sleep(500);
      }
    }
  }

  /// <summary>Title of a desktop billboard's window (UI/DesktopBillboardWindow).</summary>
  public const string BillboardWindowTitle = "AetherVk billboard";

  /// <summary>
  /// The desktop billboards' windows, in screen pixels. On macOS the whole window takes the pointer,
  /// its transparent margin included (Linux shapes the input to the image), so a click meant for
  /// what lies behind must avoid them.
  /// </summary>
  public IReadOnlyList<E2ERect> BillboardFrames() =>
    ((ISearchContext)Session.Driver)
      .FindElements(By.XPath($"//XCUIElementTypeWindow[@title=\"{BillboardWindowTitle}\"]"))
      .Select(PixelRect).ToList();

  /// <summary>
  /// Scrolls the panel under the element until its centre is inside the window's content band
  /// (a native scroll at a point of the window, in the element's column).
  /// </summary>
  /// <remarks>
  /// The wheel goes over the panel's own vertical scroll bar when one is shown beside the element
  /// (an Avalonia ScrollBar ignores the wheel, which bubbles to its ScrollViewer). Elsewhere in the
  /// column the wheel can hit what scrolls past the cursor: a numeric field or a list takes it
  /// (and changes its value), a native viewport (the mini-view's NSView) eats it, a fixed footer
  /// is outside the viewer. Those points are only fallbacks, tried in turn, the first one that
  /// moves the element kept; per point the sign is learnt from the first step that moves it.
  /// </remarks>
  /// <summary>
  /// Scrolls <paramref name="e"/> into the visible band of its innermost scroll viewer; when that
  /// one cannot bring it there (an inner list partly under the panel's footer), through the next
  /// one out, and so on. Throws rather than leave it where a click would land outside the window.
  /// </summary>
  public IWebElement ScrollIntoView(IWebElement e)
  {
    var viewers = ScrollViewportsOf(e);
    for (int i = 0; ; i++)
    {
      try
      {
        return ScrollIntoView(e, i < viewers.Count ? viewers[i] : null);
      }
      catch (InvalidOperationException) when (i < viewers.Count)
      {
      }
    }
  }

  private IWebElement ScrollIntoView(IWebElement e, (PointRect Viewer, PointRect Bar)? scroller)
  {
    var win = WindowFrame();
    double top = win.Y + 60, bottom = win.Y + win.Height - 30;
    var start = FramePoints(e);
    double columnX = Math.Clamp(start.X + Math.Min(start.Width / 2, 40), win.X + 5, win.X + win.Width - 5);
    var points = new List<(double X, double Y)>();
    if (scroller is var (viewer, bar))
    {
      points.Add((bar.X + bar.Width / 2, bar.Y + bar.Height / 2));
      // inside its scroll viewer's viewport: above or below it, the element is clipped under what is
      // pinned there (the compact panel's mini-view, its footer), not shown
      top = Math.Max(top, viewer.Y + start.Height / 2);
      bottom = Math.Min(bottom, viewer.Y + viewer.Height - start.Height / 2);
      if (bottom < top) // taller than the viewport: its centre at the viewport's centre
      {
        top = viewer.Y + viewer.Height / 2 - 4;
        bottom = top + 8;
      }
    }
    double startCy = start.Y + start.Height / 2;
    Trace($"scroll {e.GetAttribute("identifier")} at {start} into {top:0}–{bottom:0} (viewer {(scroller is var (v, _) ? v.ToString() : "none")})");
    if (startCy >= top && startCy <= bottom)
      return e;
    foreach (var h in new[] { bottom - 60, top + (bottom - top) * 0.75, (top + bottom) / 2, top + (bottom - top) * 0.25, top + 40 })
      points.Add((columnX, h));
    int point = 0;
    double sign = 1;
    bool signKnown = false;
    double perUnit = 0; // points the element moves per unit of deltaY, learnt from the first move
    double target = (top + bottom) / 2;
    // the band can be out of reach (the last field of a panel ends under its fixed footer): at the
    // panel's end the rubber band throws each step back and forth, so stop once the element has not
    // come closer for a few steps, where the panel lets it be
    double closest = double.MaxValue;
    int stale = 0;
    for (int i = 0; i < 150 && point < points.Count; i++)
    {
      var before = FramePoints(e);
      double cy = before.Y + before.Height / 2;
      int dir = cy > bottom ? 1 : cy < top ? -1 : 0;
      if (dir == 0)
        return e;
      double distance = dir > 0 ? cy - bottom : top - cy;
      if (distance < closest - 1)
      {
        closest = distance;
        stale = 0;
      }
      else if (signKnown && ++stale >= 6)
        break;
      // a probing step of 10 until the rate is known, then most of the way to the band's centre
      double units = perUnit > 0 ? Math.Clamp(Math.Abs(cy - target) / perUnit * 0.9, 1, 400) : 10;
      double delta = -dir * sign * Math.Round(units);
      Exec("macos: scroll", new Dictionary<string, object>
      {
        ["x"] = points[point].X, ["y"] = points[point].Y, ["deltaX"] = 0, ["deltaY"] = delta,
      });
      Thread.Sleep(perUnit > 0 ? 350 : 200);
      double moved = FramePoints(e).Y + before.Height / 2 - cy;
      if (Math.Abs(moved) < 0.5)
      {
        if (signKnown)
        {
          point++; // this point scrolled before and stopped: the panel's end, or it lost the wheel
          signKnown = false;
          sign = 1;
          stale = 0;
        }
        else if (sign > 0)
          sign = -1; // maybe the other way
        else
        {
          sign = 1;
          point++; // neither way: not over the scroll viewer
        }
        continue;
      }
      if (!signKnown && Math.Sign(moved) != -dir)
        sign = -sign; // moved, but away from the band
      else if (Math.Sign(moved) == -dir)
        perUnit = Math.Abs(moved) / Math.Abs(delta); // the latest rate (wheel acceleration varies it)
      signKnown = true;
    }
    // never click outside the window: the point would land on another application or the Dock
    var end = FramePoints(e);
    Trace($"  ended at {end}");
    double endX = end.X + end.Width / 2, endY = end.Y + end.Height / 2;
    double slack = end.Height / 2; // the band is for the centre: half the element may stick out
    if (endX < win.X || endX > win.X + win.Width || endY < Math.Max(win.Y + 28, top - slack) || endY > Math.Min(win.Y + win.Height, bottom + slack))
      throw new InvalidOperationException(
        $"the element at {end} could not be scrolled into the visible band {top:0}–{bottom:0} pt of the window {win}");
    return e;
  }

  /// <summary>
  /// The scroll viewer the element scrolls in: the innermost ancestor with a shown vertical scroll
  /// bar spanning its whole height (an Avalonia ScrollViewer is a group whose frame is its viewport,
  /// its bar inside). Null for an element that does not scroll (pinned above or below a viewer).
  /// </summary>
  private List<(PointRect Viewer, PointRect Bar)> ScrollViewportsOf(IWebElement e)
  {
    var found = new List<(PointRect Viewer, PointRect Bar)>();
    IReadOnlyList<IWebElement> ancestors;
    try
    {
      // from the root: WDA evaluates an element's XPath over that element's subtree only, so
      // ancestor:: finds nothing; the scroll viewers are the groups with a scroll bar child
      var id = e.GetAttribute("identifier");
      if (string.IsNullOrEmpty(id))
        return found;
      ancestors = Root.FindElements(By.XPath($"//*[XCUIElementTypeScrollBar][.//*[@identifier=\"{id}\"]]"));
    }
    catch (WebDriverException)
    {
      return found;
    }
    var win = WindowFrame();
    var el = FramePoints(e);
    double elX = el.X + el.Width / 2;
    foreach (var anc in ancestors.Reverse()) // document order: the innermost last
    {
      try
      {
        var a = FramePoints(anc);
        // inside the main window and over the element's column (a combo box's closed popup
        // has been reported holding a scroll viewer far above the window)
        if (a.Y < win.Y - 1 || a.Y + a.Height > win.Y + win.Height + 1 || elX < a.X || elX > a.X + a.Width)
          continue;
        foreach (var bar in anc.FindElements(By.XPath(".//XCUIElementTypeScrollBar")))
        {
          // not bar.Displayed: an overlay scroll bar collapsed to its thin idle state reads as hidden
          var f = FramePoints(bar);
          if (f.Width <= 0 || f.Height <= 0)
            continue;
          bool vertical = f.Height > 2 * f.Width;
          bool spans = Math.Abs(f.Y - a.Y) <= 4 && Math.Abs(f.Height - a.Height) <= 8
                       && Math.Abs(f.X + f.Width - (a.X + a.Width)) <= 24;
          if (vertical && spans)
          {
            found.Add((a, f));
            break;
          }
        }
      }
      catch (WebDriverException)
      {
      }
    }
    return found;
  }

  // ── input ───────────────────────────────────────────────────────────────────

  public void Click(string automationId)
  {
    // a click goes to whatever is on top at that point: another app's window (a terminal's tip
    // panel took one) must not be
    Foreground();
    var e = ScrollIntoView(Element(automationId));
    Trace($"click {automationId} at {FramePoints(e)}");
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
    Foreground();
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
    // an Avalonia ListBox item is a TableRow (inside border › scroll viewer › presenters)
    foreach (var xpath in new[] { ".//XCUIElementTypeTableRow", ".//XCUIElementTypeCell", ".//XCUIElementTypeRow", "./*" })
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
    // one request: each one waits for the app to idle
    Keys([.. Enumerable.Repeat<object>(UpArrow, 12), .. Enumerable.Repeat<object>(DownArrow, index), Return]);
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
      .SelectMany(t => parent.FindElements(By.XPath($".//XCUIElementTypeMenuItem[{TitlePredicate(t)}]"))
        .Where(e => !t.Contains('&') || e.GetAttribute("title") == t))
      .FirstOrDefault(), what: $"menu item '{title}'");

  /// <summary>
  /// An XPath predicate on the title. WebDriverAgent matches no title holding '&amp;' (AppKit's
  /// "Move &amp; Resize") by equality, so such a title is matched by its parts (then compared exactly).
  /// </summary>
  private static string TitlePredicate(string title) =>
    title.Contains('&')
      ? string.Join(" and ", title.Split('&', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries)
          .Select(part => $"contains(@title, \"{part}\")"))
      : $"@title=\"{title}\"";

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
  // measured on macOS 26 (dark menu, 1×): an unchecked item 0 px, a radio dot ~50 px
  private const int MarkInkThreshold = 8;

  /// <summary>
  /// Pixels of the menu item's left gutter (a square of the item's height, where AppKit draws the
  /// state mark) that stand out from the gutter's median luminance.
  /// </summary>
  private static int MarkGutterInk(IWebElement menuItem)
  {
    var png = PngCodec.Decode(((ITakesScreenshot)menuItem).GetScreenshot().AsByteArray);
    int gutter = Math.Min(png.Height, png.Width / 4);
    var lum = new List<int>(gutter * png.Height);
    for (int y = 0; y < png.Height; y++)
      for (int x = 0; x < gutter; x++)
      {
        int i = (y * png.Width + x) * 4;
        lum.Add((png.Rgba[i] * 299 + png.Rgba[i + 1] * 587 + png.Rgba[i + 2] * 114) / 1000);
      }
    var sorted = lum.Order().ToList();
    int median = sorted[sorted.Count / 2];
    return lum.Count(l => Math.Abs(l - median) > 60);
  }

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
      if (CheckedFrom(value, null) is { } fromValue)
        return (fromValue, $"value=\"{value}\" selected=\"{selected}\"");
      // Avalonia's NativeMenu draws its mark (radio dot, check) as the item's on-state image: AX then
      // exposes no mark char, value is "" and selected false. Read the mark gutter's pixels instead.
      int ink = MarkGutterInk(e);
      return (ink >= MarkInkThreshold, $"value=\"{value}\" selected=\"{selected}\" mark-gutter ink={ink}");
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

  /// <summary>
  /// Cmd+Shift+G, the path, Return: the panel goes to the folder (and selects a named file).
  /// </summary>
  /// <remarks>
  /// The path is typed, all its characters in one request: a pasted one (pbcopy, Cmd+V) did not
  /// replace the sheet's preselected text, and the panel stayed in the folder it used last. Typed,
  /// each key waits for the panel to idle, and the Open panel refreshes its suggestions after every
  /// key (a full path hung WebDriverAgent for 3 minutes), so the folder is reached through a short
  /// symbolic link, /tmp/avk/&lt;hash&gt;.
  /// </remarks>
  private void GoTo(string path)
  {
    Chord("g", Command | Shift);
    Thread.Sleep(800);
    Chord("a", Command);
    Keys(ShortPath(path));
    Thread.Sleep(300);
    Keys(Return);
    Thread.Sleep(1000);
  }

  /// <summary>
  /// <paramref name="path"/> through /tmp/avk/&lt;8 hex digits&gt;, a symbolic link to its folder
  /// (a folder itself: the link is to it).
  /// </summary>
  internal static string ShortPath(string path)
  {
    bool isDir = Directory.Exists(path);
    string dir = isDir ? path : Path.GetDirectoryName(path)!;
    string hash = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(System.Text.Encoding.UTF8.GetBytes(dir)))[..8].ToLowerInvariant();
    string root = "/tmp/avk";
    Directory.CreateDirectory(root);
    string link = Path.Combine(root, hash);
    var info = new FileInfo(link);
    if (info.LinkTarget != dir)
    {
      if (info.Exists || info.LinkTarget is not null)
        info.Delete();
      else if (Directory.Exists(link))
        Directory.Delete(link);
      File.CreateSymbolicLink(link, dir);
    }
    return isDir ? link + "/" : Path.Combine(link, Path.GetFileName(path));
  }

  /// <summary>Writes <paramref name="text"/> to the clipboard (pbcopy), or reads it (pbpaste) when null.</summary>
  private static string? Pasteboard(string? text)
  {
    var psi = new System.Diagnostics.ProcessStartInfo(text is null ? "/usr/bin/pbpaste" : "/usr/bin/pbcopy")
    {
      RedirectStandardInput = text is not null,
      RedirectStandardOutput = text is null,
      UseShellExecute = false,
    };
    using var p = System.Diagnostics.Process.Start(psi)!;
    string? read = null;
    if (text is not null)
    {
      p.StandardInput.Write(text);
      p.StandardInput.Close();
    }
    else
      read = p.StandardOutput.ReadToEnd();
    p.WaitForExit(5000);
    return read;
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
    Trace($"save panel {(panel is null ? "not found" : $"{panel.TagName} at {FramePoints(panel)}")}, name field {(name is null ? "not found" : FramePoints(name).ToString())} → {full}");
    // the folder first: going to it puts the app's suggested name back in the name field
    GoTo(Path.GetDirectoryName(full)!);
    // found again: the navigation may have rebuilt the panel's accessibility elements
    panel = OpenPanel() ?? panel;
    name = panel?.FindElements(By.XPath(".//XCUIElementTypeTextField[@identifier=\"saveAsNameTextField\"]")).FirstOrDefault() ?? name;
    string fileName = Path.GetFileName(full);
    for (int attempt = 0; attempt < 3; attempt++)
    {
      if (name is null)
      {
        Chord("a", Command);
        Keys(fileName);
        break;
      }
      // cleared through the element: after the panel went to the folder, Cmd+A in the field
      // no longer selects its text, and the name was typed into the middle of the old one
      name.Click();
      try
      {
        name.Clear();
      }
      catch (WebDriverException)
      {
        Keys("\uF72B"); // End
        Keys([.. Enumerable.Repeat<object>("\b", (name.GetAttribute("value") ?? "").Length)]);
      }
      name.SendKeys(fileName);
      Thread.Sleep(200);
      string? typed = name?.GetAttribute("value");
      Trace($"  name now \"{typed}\"");
      // a hidden extension shows the name without it
      if (typed == fileName || typed == Path.GetFileNameWithoutExtension(fileName))
        break;
    }
    Shot("save-1-named");
    Keys(Return); // Save
    Thread.Sleep(800);
    Shot("save-2-after-return");
    Trace($"  after Return: panel {(OpenPanel() is { } after ? $"{after.TagName} at {FramePoints(after)}" : "gone")}");
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
