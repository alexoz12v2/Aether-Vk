using AetherVk.UI.E2E.Shared;
using OpenQA.Selenium;
using OpenQA.Selenium.Appium;
using OpenQA.Selenium.Appium.Windows;

namespace AetherVk.UI.E2E;

/// <summary>
/// The shared scenarios' seam (<see cref="IE2EApp"/>) over the existing WinAppDriver session,
/// <see cref="Ui"/> and <see cref="Win32"/>: the same locators, clicks, dialogs and BitBlt probes
/// the Windows suite has always used.
/// </summary>
public sealed class WindowsE2EApp(AppSession session) : IE2EApp
{
  public AppSession Session { get; } = session;

  public string Platform => "windows";

  public int ProcessId => Session.Process.Id;

  public IReadOnlyList<string> Lines => Session.Lines.Select(l => l.Line).ToList();

  public string WindowTitle => Session.WindowTitle;

  private static E2EElement Snapshot(string id, AppiumWebElement e) =>
    new(id, FrameMetrics.RectOf(e), e.Text ?? string.Empty, e.Enabled, e.Displayed);

  public E2EElement? TryFind(string automationId)
  {
    try
    {
      return Session.TryFind(automationId) is { } e ? Snapshot(automationId, e) : null;
    }
    catch (WebDriverException)
    {
      return null; // rebuilt under the read
    }
  }

  public E2EElement Find(string automationId, TimeSpan? timeout = null) =>
    Snapshot(automationId, Session.Find(automationId, timeout));

  public void Click(string automationId) => Session.Click(automationId);

  public void ClickAt(int x, int y) => Win32.ClickAt(x, y);

  /// <summary>The slider's "&lt;id&gt;.Input", else the element's inner TextBox (a NumericUpDown), else the element.</summary>
  private AppiumWebElement TextTarget(string automationId)
  {
    if (Session.TryFind(automationId + ".Input") is { } input)
      return input;
    var e = Session.Find(automationId);
    return e.FindElementsByClassName("TextBox").FirstOrDefault() ?? e;
  }

  public void TypeInto(string automationId, string text, bool enter = true) =>
    Session.TypeInto(TextTarget(automationId), text, enter);

  private static string? Attribute(AppiumWebElement e, string name)
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

  public void Toggle(string automationId)
  {
    var e = Session.ScrollIntoView(Session.Find(automationId));
    if (Attribute(e, "ExpandCollapse.ExpandCollapseState") is "Expanded" or "Collapsed")
      Ui.ToggleExpander(e);
    else
      e.RealClick();
    Thread.Sleep(300);
  }

  public bool? IsChecked(string automationId)
  {
    var e = Session.Find(automationId);
    return Attribute(e, "Toggle.ToggleState") switch
    {
      "1" or "On" => true,
      "0" or "Off" => false,
      _ => Attribute(e, "SelectionItem.IsSelected") switch
      {
        "True" or "true" => true,
        "False" or "false" => false,
        _ => Attribute(e, "ExpandCollapse.ExpandCollapseState") switch
        {
          "Expanded" => true,
          "Collapsed" => false,
          _ => null,
        },
      },
    };
  }

  public string Text(string automationId)
  {
    var target = TextTarget(automationId);
    var text = target.Text;
    if (string.IsNullOrEmpty(text))
      text = Attribute(target, "Value.Value") ?? Attribute(target, "Name") ?? string.Empty;
    return text;
  }

  public int ListCount(string listId) => Session.Items(listId).Count;

  public void SelectListItem(string listId, int index)
  {
    var item = Ui.WaitFor(() => Session.Items(listId).ElementAtOrDefault(index), what: $"item {index} of {listId}");
    Session.ScrollIntoView(item).RealClick();
    Thread.Sleep(300);
  }

  public void ChooseAt(string comboId, int index) => Session.ChooseAt(comboId, index);

  public void PressKey(char key) => Win32.PressKey((byte)char.ToUpperInvariant(key));

  /// <summary>
  /// The in-window menu: the top item on the main window, each further entry in its popup (a
  /// top-level window: the desktop session), menu items preferred over other elements of that name.
  /// </summary>
  public void Menu(params string[] path)
  {
    static bool IsMenuItem(WindowsElement e) => e.TagName == "ControlType.MenuItem";
    var top = Session.App.FindElementsByName(path[0]).Where(e => e.Displayed).OrderBy(e => IsMenuItem(e) ? 0 : 1).FirstOrDefault()
              ?? throw new NoSuchElementException($"menu '{path[0]}'");
    top.RealClick();
    foreach (var entry in path.Skip(1))
    {
      Thread.Sleep(300);
      Ui.WaitFor(() => Session.Desktop.FindElementsByName(entry).Where(e => e.Displayed)
          .OrderBy(e => IsMenuItem(e) ? 0 : 1).FirstOrDefault(), what: $"menu '{entry}'")
        .RealClick();
    }
  }

  public void AnswerOpenDialog(string path) => Session.AnswerFileDialog(path);

  public void AnswerSaveDialog(string path) => Session.AnswerFileDialog(path);

  public E2ERect ViewportRect(string automationId) => FrameMetrics.RectOf(Session.Find(automationId));

  public double[] SampleLuminance(E2ERect rect, int n) => Win32.SampleLuminance(rect.X, rect.Y, rect.Width, rect.Height, n);

  public double[] CaptureLuminance(E2ERect rect) => Win32.CaptureLuminance(rect.X, rect.Y, rect.Width, rect.Height);

  public void SavePng(E2ERect rect, string path)
  {
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
    Win32.SavePng(path, rect.X, rect.Y, rect.Width, rect.Height);
  }

  public void SaveScreenshot(string path) => Session.Desktop.SaveScreenshot(path);

  public void EnsureAlive(string doing) => Session.EnsureAlive(doing);

  public void Foreground() => Session.Foreground();

  /// <summary>Closes the main window; a save prompt is answered with Discard (as the export test does).</summary>
  public E2EExit Quit(TimeSpan? timeout = null)
  {
    var p = Session.Process;
    if (p.HasExited)
      return new E2EExit(true, p.ExitCode, false);
    p.CloseMainWindow();
    if (!p.WaitForExit(5_000))
      try
      {
        Ui.WaitFor(() => Session.Desktop.FindElementsByAccessibilityId("Discard").FirstOrDefault(e => e.Displayed),
          TimeSpan.FromSeconds(10), "the save prompt").Click();
      }
      catch (Exception e) when (e is TimeoutException or WebDriverException) { }
    if (p.WaitForExit((int)(timeout ?? TimeSpan.FromSeconds(30)).TotalMilliseconds))
      return new E2EExit(true, p.ExitCode, false);
    p.Kill(entireProcessTree: true);
    p.WaitForExit(5_000);
    return new E2EExit(true, null, true);
  }

  public void Dispose() => Session.Dispose();
}

/// <summary>Launches the app for the shared scenarios; artifacts under TestResults/e2e/.</summary>
public sealed class WindowsE2EHost : IE2EHost
{
  public string Platform => "windows";

  public string ArtifactsRoot => Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e");

  public string? UnavailableReason() => AppSession.UnavailableReason();

  public IE2EApp Launch(string logPath, string culture = E2ESteps.Culture) =>
    new WindowsE2EApp(AppSession.Launch(logPath, diagnosticPanels: false, culture: culture));

  /// <summary>The app's own crash reports (<c>%LOCALAPPDATA%\Aether-Vk\logs\aethervk-crash-*.txt</c>).</summary>
  public IReadOnlyList<string> CrashReportsSince(DateTime sinceUtc)
  {
    var dir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "Aether-Vk", "logs");
    return Directory.Exists(dir)
      ? Directory.GetFiles(dir, "aethervk-crash-*.txt").Where(f => File.GetLastWriteTimeUtc(f) >= sinceUtc).ToList()
      : [];
  }
}
