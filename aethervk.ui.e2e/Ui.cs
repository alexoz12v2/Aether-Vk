using System.Globalization;
using System.Text.RegularExpressions;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace AetherVk.UI.E2E;

/// <summary>
/// Element lookup and input on top of WinAppDriver, in the vocabulary of the app: AutomationIds
/// set in the .axaml views, sliders typed into, native dialogs answered by path.
/// </summary>
public static partial class Ui
{
  public static readonly TimeSpan DefaultTimeout = TimeSpan.FromSeconds(20);

  /// <summary>The app parses and formats numbers with the current culture (it-IT: "45,4").</summary>
  public static string Num(double value) => value.ToString("0.###", CultureInfo.CurrentCulture);

  public static double ParseNum(string text)
  {
    var m = NumberRegex().Match(text.Replace('−', '-'));
    if (!m.Success)
      throw new FormatException($"no number in \"{text}\"");
    return double.Parse(m.Value.Replace(',', '.'), CultureInfo.InvariantCulture);
  }

  // G5 displays switch to exponents past five digits ("2,174E+05")
  [GeneratedRegex(@"-?\d+(?:[.,]\d+)?(?:[eE][+-]?\d+)?")]
  private static partial Regex NumberRegex();

  public static T WaitFor<T>(Func<T?> probe, TimeSpan? timeout = null, string? what = null) where T : class
  {
    var deadline = DateTime.UtcNow + (timeout ?? DefaultTimeout);
    Exception? last = null;
    while (DateTime.UtcNow < deadline)
    {
      try
      {
        if (probe() is { } value)
          return value;
      }
      catch (WebDriverException e)
      {
        last = e;
      }
      Thread.Sleep(250);
    }
    throw new TimeoutException($"timed out waiting for {what ?? "condition"}", last);
  }

  public static void WaitUntil(Func<bool> condition, TimeSpan? timeout = null, string? what = null) =>
    WaitFor(() => condition() ? "ok" : null, timeout, what);

  public static WindowsElement? TryFind(this AppSession s, string automationId) =>
    s.App.FindElementsByAccessibilityId(automationId).FirstOrDefault(e => e.Displayed);

  public static WindowsElement Find(this AppSession s, string automationId, TimeSpan? timeout = null) =>
    WaitFor(() => s.TryFind(automationId), timeout, $"element '{automationId}'");

  /// <summary>Wheel-scrolls the panel under the element until it is inside the main window.</summary>
  public static T ScrollIntoView<T>(this AppSession s, T element) where T : AppiumWebElement
  {
    var win = s.App.Manage().Window;
    int top = win.Position.Y + 80, bottom = win.Position.Y + win.Size.Height - 40;
    for (int i = 0; i < 60; i++)
    {
      var (p, size) = (element.Location, element.Size);
      int dy = p.Y + size.Height > bottom ? 1 : p.Y < top ? -1 : 0;
      if (dy == 0)
        return element;
      Win32.WheelAt(p.X + Math.Min(size.Width / 2, 40), Math.Clamp(p.Y + size.Height / 2, top + 20, bottom - 20), dy * 2);
    }
    return element;
  }

  public static void Click(this AppSession s, string automationId) =>
    s.ScrollIntoView(s.Find(automationId)).RealClick();

  /// <summary>A mouse click at the element's centre (see <see cref="Win32.ClickAt"/>).</summary>
  public static void RealClick(this AppiumWebElement e)
  {
    var (p, size) = (e.Location, e.Size);
    Win32.ClickAt(p.X + size.Width / 2, p.Y + size.Height / 2);
  }

  /// <summary>An element of each tab's content: present only while the tab is the selected one.</summary>
  private static readonly Dictionary<string, string> TabMarkers = new()
  {
    ["ModelTab"] = "DustSection",
    ["SettingsTab"] = "ViewportSettingsSection",
    ["CometTab"] = "CometSearchBox",
    ["ImportsTab"] = "ImportButton",
    ["TimelineTab"] = "StartEpochBox",
  };

  /// <summary>Selects the tab, clicking again while its content has not appeared (a click can be lost).</summary>
  public static void SelectTab(this AppSession s, string tab)
  {
    for (int attempt = 0; ; attempt++)
    {
      s.Click("Tab." + tab);
      Thread.Sleep(400);
      if (!TabMarkers.TryGetValue(tab, out var marker))
        return;
      try
      {
        WaitUntil(() => s.App.FindElementsByAccessibilityId(marker).Count > 0, TimeSpan.FromSeconds(3), $"the {tab} content");
        return;
      }
      catch (TimeoutException) when (attempt < 3)
      {
      }
    }
  }

  /// <summary>
  /// Opens a collapsed expander and brings <paramref name="childId"/> into view. An open one is not
  /// clicked (that would close it): its child may only lie below the fold of the panel.
  /// </summary>
  public static void Expand(this AppSession s, string expanderId, string childId)
  {
    if (s.TryFind(childId) is not null)
      return;
    var expander = s.ScrollIntoView(s.Find(expanderId));
    if (expander.GetAttribute("ExpandCollapse.ExpandCollapseState") != "Expanded")
      expander.RealClick();
    s.ScrollIntoView(WaitFor(() => s.App.FindElementsByAccessibilityId(childId).FirstOrDefault(),
      what: $"element '{childId}' in {expanderId}"));
    s.Find(childId);
  }

  public static void TypeInto(this AppSession s, AppiumWebElement box, string text, bool enter = true)
  {
    s.ScrollIntoView(box).RealClick();
    Thread.Sleep(150);
    s.App.Keyboard.SendKeys(Keys.Control + "a" + Keys.Control);
    s.App.Keyboard.SendKeys(text);
    if (enter)
      s.App.Keyboard.SendKeys(Keys.Enter);
    Thread.Sleep(250);
  }

  /// <summary>
  /// An UnboundedSlider: a tap (no drag) opens its box for typing, Enter commits. The inner box is
  /// "&lt;id&gt;.Input" and reads back the committed value.
  /// </summary>
  public static void SetSlider(this AppSession s, string sliderId, double value, double tolerance = 1e-3)
  {
    var input = s.Find(sliderId + ".Input");
    s.TypeInto(input, Num(value));
    WaitUntil(() => Math.Abs(ParseNum(s.Find(sliderId + ".Input").Text) - value) <= tolerance,
      what: $"{sliderId} = {value}");
  }

  public static double ReadSlider(this AppSession s, string sliderId) =>
    ParseNum(s.Find(sliderId + ".Input").Text);

  /// <summary>A NumericUpDown: its text part takes the value, Enter commits.</summary>
  public static void SetNumeric(this AppSession s, string id, double value)
  {
    var box = s.Find(id).FindElementsByClassName("TextBox").FirstOrDefault() ?? s.Find(id);
    s.TypeInto(box, Num(value));
  }

  /// <summary>A standard Slider (RangeValue pattern): Home, then arrow steps up to the value.</summary>
  public static void SetRange(this AppSession s, string id, double value, double step, double tolerance)
  {
    var slider = s.ScrollIntoView(s.Find(id));
    slider.RealClick();
    s.App.Keyboard.SendKeys(Keys.Home);
    for (int i = 0; i < 200 && ReadRange(slider) < value - tolerance; i++)
      s.App.Keyboard.SendKeys(Keys.Right);
    WaitUntil(() => Math.Abs(ReadRange(slider) - value) <= tolerance + step, what: $"{id} = {value}");
  }

  public static double ReadRange(AppiumWebElement slider) =>
    double.Parse(slider.GetAttribute("RangeValue.Value"), CultureInfo.InvariantCulture);

  /// <summary>Opens a combo box and picks the drop-down item (a popup window: the desktop session).</summary>
  public static void Choose(this AppSession s, string comboId, Func<string, bool> itemText)
  {
    s.Click(comboId);
    var item = WaitFor(() => s.Desktop.FindElementsByClassName("ComboBoxItem")
      .FirstOrDefault(i => i.Displayed && itemText(ItemText(i))), what: $"an item of {comboId}");
    item.RealClick();
    Thread.Sleep(300);
  }

  /// <summary>Picks the drop-down item at <paramref name="index"/> (^1 = last): label-independent.</summary>
  public static void ChooseAt(this AppSession s, string comboId, Index index)
  {
    s.Click(comboId);
    var items = WaitFor(() => s.Desktop.FindElementsByClassName("ComboBoxItem").Where(i => i.Displayed).ToList()
      is { Count: > 0 } l ? l : null, what: $"the items of {comboId}");
    items[index].RealClick();
    Thread.Sleep(300);
  }

  /// <summary>
  /// Text of a list / combo item: its text blocks joined, else its name (for a templated item the
  /// name is the data object's ToString(), e.g. "AetherVk.Logic.Models.CometSearchResult").
  /// </summary>
  public static string ItemText(AppiumWebElement item)
  {
    var text = string.Join(" ", item.FindElementsByClassName("TextBlock").Select(t => t.Text)
      .Where(t => !string.IsNullOrWhiteSpace(t)));
    return text.Length > 0 ? text : item.GetAttribute("Name") ?? string.Empty;
  }

  public static IReadOnlyList<AppiumWebElement> Items(this AppSession s, string listId) =>
    s.Find(listId).FindElementsByClassName("ListBoxItem").ToList();

  /// <summary>
  /// Answers the native Open / Save / folder dialog the last click raised, the way its platform
  /// expects (WinAppDriver drives Windows only; other platforms need their own driver).
  /// </summary>
  public static void AnswerFileDialog(this AppSession s, string path)
  {
    if (OperatingSystem.IsWindows())
      s.AnswerWindowsFileDialog(path);
    else
      throw new PlatformNotSupportedException("native file dialogs are automated on Windows only");
  }

  /// <summary>
  /// Windows common item dialog: the containing folder in the address bar at the top (Alt+D),
  /// Enter navigates there; then only the file (or folder) name in the name box at the bottom,
  /// Enter accepts. A folder picker that only navigated into the named folder is accepted with
  /// its Select Folder button (control id 1).
  /// </summary>
  private static void AnswerWindowsFileDialog(this AppSession s, string path)
  {
    var dialog = WaitFor(() => s.Desktop.FindElementsByClassName("#32770").FirstOrDefault(d => d.Displayed),
      TimeSpan.FromSeconds(20), "a file dialog");
    var handle = new IntPtr(Convert.ToInt64(dialog.GetAttribute("NativeWindowHandle")));
    Win32.SetForegroundWindow(handle);
    Thread.Sleep(400);

    var full = Path.GetFullPath(path).TrimEnd(Path.DirectorySeparatorChar);
    // a folder picker (Save As): the address bar opens the folder itself, "Select Folder" takes
    // it; a name typed in its Folder box would only navigate into a subfolder
    bool folder = Directory.Exists(full);
    Win32.Chord(Win32.VK_MENU, (byte)'D');
    Thread.Sleep(300);
    Win32.Chord(Win32.VK_CONTROL, (byte)'A');
    Win32.Paste(folder ? full : Path.GetDirectoryName(full)!);
    Win32.PressKey(Win32.VK_RETURN);
    Thread.Sleep(1000);
    if (folder)
    {
      dialog.FindElementsByAccessibilityId("1").First(e => e.Displayed).RealClick();
      WaitUntil(() => !s.Desktop.FindElementsByClassName("#32770").Any(d => d.Displayed),
        TimeSpan.FromSeconds(10), "the folder dialog to close");
      return;
    }

    var nameBox = WaitFor(() => dialog.FindElementsByAccessibilityId("1148").FirstOrDefault(e => e.Displayed)
                                ?? dialog.FindElementsByAccessibilityId("1152").FirstOrDefault(e => e.Displayed),
      what: "the file name box");
    nameBox.RealClick();
    Win32.Chord(Win32.VK_CONTROL, (byte)'A');
    Win32.Paste(Path.GetFileName(full));
    Win32.PressKey(Win32.VK_RETURN);
    Thread.Sleep(1500);
    if (s.Desktop.FindElementsByClassName("#32770").FirstOrDefault(d => d.Displayed) is { } still)
    {
      still.FindElementsByAccessibilityId("1").FirstOrDefault()?.RealClick();
      WaitUntil(() => !s.Desktop.FindElementsByClassName("#32770").Any(d => d.Displayed),
        TimeSpan.FromSeconds(10), "the file dialog to close");
    }
  }

  /// <summary>Clicks a main-menu path, e.g. ("File", "Save As...").</summary>
  public static void Menu(this AppSession s, params string[] path)
  {
    s.App.FindElementByName(path[0]).RealClick();
    foreach (var entry in path.Skip(1))
    {
      Thread.Sleep(300);
      WaitFor(() => s.Desktop.FindElementsByName(entry).FirstOrDefault(e => e.Displayed), what: $"menu '{entry}'")
        .RealClick();
    }
  }

  public static void SaveScreenshot(this WindowsDriver<WindowsElement> driver, string path)
  {
    Directory.CreateDirectory(Path.GetDirectoryName(path)!);
    driver.GetScreenshot().SaveAsFile(path, ScreenshotImageFormat.Png);
  }
}
