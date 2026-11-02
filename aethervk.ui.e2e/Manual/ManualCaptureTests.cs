using System.Diagnostics;
using System.Drawing;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E.Manual;

/// <summary>
/// The screenshots of the user manual (docs/manual/PLAN.md), taken from the real app.
/// <c>AETHERVK_MANUAL_CAPTURE=1</c> runs it; <c>AETHERVK_MANUAL_ONLY=&lt;regex&gt;</c> keeps the shots whose
/// <c>key--state</c> matches (e.g. <c>^settings-viewport</c>); <c>AETHERVK_MANUAL_THEME</c> is light by default.
/// It owns the desktop while it runs: do not touch the mouse or keyboard.
/// Output: <c>TestResults/e2e/manual/&lt;yyyyMMdd-HHmm&gt;/</c>, per shot a clean crop, a tagged crop with numbered
/// markers and the callouts json, plus <c>manifest.json</c>; <c>docs/manual/tools/make_adoc.py</c> turns that into AsciiDoc.
/// </summary>
[Trait("Category", "Manual")]
public sealed class ManualCaptureTests(ITestOutputHelper output)
{
  private const string Culture = "en-US";

  private string _theme = "light";
  private string _dir = string.Empty;
  private Manifest _manifest = null!;
  private Win32.Monitor _monitor = null!;
  private Regex? _only;
  private Inventory _inventory = Inventory.Empty;

  [SkippableFact]
  public void CaptureEveryPanel()
  {
    Skip.If(Environment.GetEnvironmentVariable("AETHERVK_MANUAL_CAPTURE") != "1", "set AETHERVK_MANUAL_CAPTURE=1");
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);

    _theme = Environment.GetEnvironmentVariable("AETHERVK_MANUAL_THEME") is { Length: > 0 } t ? t : "light";
    if (Environment.GetEnvironmentVariable("AETHERVK_MANUAL_ONLY") is { Length: > 0 } only)
      _only = new Regex(only, RegexOptions.CultureInvariant);
    _monitor = Win32.Monitors()[0]; // the primary: at the origin, so WinAppDriver's places are the desktop's
    _inventory = Inventory.Load();
    _dir = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "manual", DateTime.Now.ToString("yyyyMMdd-HHmm"));
    Directory.CreateDirectory(_dir);
    _manifest = new Manifest(AppVersion(), GitCommit(), Culture, _theme,
      $"{_monitor.Left},{_monitor.Top} {_monitor.Width}x{_monitor.Height}", Win32.ScalingOf(_monitor));
    if (Math.Abs(Win32.ScalingOf(_monitor) - 1.0) > 0.01)
      _manifest.Warnings.Add($"the monitor scales {Win32.ScalingOf(_monitor):0.##}x: sizes differ from the 100 % reference");
    Log($"output: {_dir}");

    var preferencesExisted = File.Exists(PreferencesPath);
    using var backup = new PreferencesBackup();
    try
    {
      PatchPreferences(_theme);
      var shots = Shots.All().Where(Wanted).ToList();
      foreach (var skipped in Shots.Skipped.Where(x => _only is null || _only.IsMatch(x.Key)))
      {
        Log($"SKIP {skipped.Key}: {skipped.Reason}");
        _manifest.Add(new ShotRecord(skipped.Key, "-", "-", "skipped", skipped.Reason, 0, 0, 0, 0, false, 0, null));
      }

      if (_only is null || _only.IsMatch("splash--launch"))
        Guard("splash", "launch", CaptureSplash);
      foreach (var session in new[] { "empty", "committed" })
      {
        var mine = shots.Where(x => x.Session == session).ToList();
        if (mine.Count > 0)
          RunSession(session, mine);
      }
      if (_only is null || _only.IsMatch("dlg-fatal--dialog"))
        Guard("dlg-fatal", "dialog", CaptureFatal);
    }
    finally
    {
      _manifest.Write(Path.Combine(_dir, "manifest.json"));
      backup.Dispose();
      if (!preferencesExisted && File.Exists(PreferencesPath))
        File.Delete(PreferencesPath);
      int failed = _manifest.Shots.Count(x => x.Status == "failed");
      Log($"done: {_manifest.Shots.Count(x => x.Status == "ok")} captured, {failed} failed, " +
          $"{_manifest.Shots.Count(x => x.Status == "skipped")} skipped; {_dir}");
    }
  }

  private bool Wanted(Shot shot) => _only is null || _only.IsMatch($"{shot.Key}--{shot.State}");

  private void Log(string message) => output.WriteLine($"{DateTime.Now:HH:mm:ss} {message}");

  // ── sessions ──

  private void RunSession(string name, List<Shot> shots)
  {
    Log($"session '{name}': {shots.Count} shots");
    AppSession? s = null;
    ShotCtx? c = null;
    try
    {
      s = AppSession.Launch(Path.Combine(_dir, $"app-{name}.log"), diagnosticPanels: false, culture: Culture);
      // once, and the window never moves again: WinAppDriver's places go stale after a move
      Win32.MaximizeOn(s.MainWindow, _monitor);
      Thread.Sleep(1500);
      c = new ShotCtx(s, _monitor, Log);
      var win = Win32.FrameBounds(s.MainWindow);
      Log($"window {win}");
      if (Math.Abs(win.Width - _monitor.Width) > 20 || Math.Abs(win.Height - _monitor.Height) > 120)
        _manifest.Warnings.Add($"session {name}: the window is {win.Width}x{win.Height} on a {_monitor.Width}x{_monitor.Height} monitor");
      if (name == "committed")
        OpenTheSave(c);
    }
    catch (Exception e)
    {
      Log($"session '{name}' did not start: {e.Message}");
      foreach (var shot in shots)
        Fail(shot.Key, shot.State, $"session did not start: {e.Message}", null);
      s?.Dispose();
      return;
    }

    try
    {
      foreach (var shot in shots)
        Take(c!, shot);
      Steps.UnlockTheRange(c!); // safety net: never leave the pending range edited
    }
    catch (Exception e)
    {
      Log($"session '{name}' ended early: {e.Message}");
    }
    finally
    {
      s?.Dispose();
    }
  }

  private void OpenTheSave(ShotCtx c)
  {
    var s = c.S;
    var (artifacts, saveDir) = ReviewSave.Prepare("manual-save"); // not under TestResults/e2e/manual: that holds the captures
    Log("open the review save");
    Thread.Sleep(2000);
    s.Foreground();
    Win32.CtrlChord((byte)'O'); // File > Open...
    Thread.Sleep(1000);
    s.AnswerFileDialog(Path.Combine(saveDir, "aethervk_index.json"));
    Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
      TimeSpan.FromSeconds(90), "the save to open");
    Thread.Sleep(5000);
    ReviewSave.SetQuality(s, 3, "Medium");
    ReviewSave.HideTheImage(s); // its window sits over the comet; billboard-window shows it again
    ReviewSave.Pause(s);
    Steps.WaitDustSettled(c);
    Log($"save open, artifacts {artifacts}");
  }

  // ── one shot ──

  private void Take(ShotCtx c, Shot shot)
  {
    var stem = $"{shot.Key}--{shot.State}--{_theme}";
    if (c.Dead)
    {
      Fail(shot.Key, shot.State, "the app is gone", null);
      return;
    }
    Log($"shot {stem}");
    try
    {
      c.S.EnsureAlive($"before {stem}");
      c.ResetDelta();
      shot.Setup(c);
      if (shot.Key != "help-flyout") // a help flyout hides once the pointer leaves it
        c.Park();
      Thread.Sleep(700); // layout, hover fade
      var crop = shot.Crop(c);
      var rect = Rectangle.Intersect(crop.Rect, new Rectangle(_monitor.Left, _monitor.Top, _monitor.Width, _monitor.Height));
      if (rect.Width < 8 || rect.Height < 8)
        throw new InvalidOperationException($"the crop {crop.Rect} is outside the monitor");
      var bgra = Win32.Capture(rect.X, rect.Y, rect.Width, rect.Height);

      int[]? calibration = null;
      IReadOnlyList<RawCallout> raw = crop.Explicit ?? TreeCallouts(c, crop, out calibration);
      var callouts = ShotTools.Renumber(ShotTools.ToRelative(raw, rect), rect.Width, rect.Height);
      ShotTools.WritePng(Path.Combine(_dir, stem + ".png"), bgra, rect.Width, rect.Height);
      var tagged = ShotTools.Tagged(bgra, rect.Width, rect.Height, callouts, out int tw, out int th);
      ShotTools.WritePng(Path.Combine(_dir, stem + "--tagged.png"), tagged, tw, th);
      File.WriteAllText(Path.Combine(_dir, stem + ".callouts.json"), ShotTools.ToJson(callouts));
      _manifest.Add(new ShotRecord(shot.Key, shot.State, stem + ".png", "ok", crop.Clipped ? "cropped at the window edge" : null,
        rect.X, rect.Y, rect.Width, rect.Height, crop.Clipped, callouts.Count, calibration));
      Log($"  {rect} {callouts.Count} callouts{(crop.Clipped ? " (clipped)" : "")}");
    }
    catch (Exception e)
    {
      Log($"  FAILED {stem}: {e.GetType().Name}: {e.Message}");
      Screenshot(Path.Combine(_dir, $"{shot.Key}--{shot.State}--FAILED.png"));
      Fail(shot.Key, shot.State, $"{e.GetType().Name}: {e.Message}", null);
      if (!c.S.IsAlive)
        c.Dead = true;
      else
      {
        // a failed setup may leave a menu or popup open
        try { c.Escape(); c.Escape(); } catch (Exception) { }
      }
    }
    finally
    {
      if (!c.Dead)
      {
        try { shot.After?.Invoke(c); }
        catch (Exception e) { Log($"  after {stem}: {e.GetType().Name}: {e.Message}"); }
      }
    }
  }

  private void Fail(string key, string state, string reason, int[]? calibration) =>
    _manifest.Add(new ShotRecord(key, state, "-", "failed", reason, 0, 0, 0, 0, false, 0, calibration));

  private void Screenshot(string path)
  {
    try
    {
      var bgra = Win32.Capture(_monitor.Left, _monitor.Top, _monitor.Width, _monitor.Height);
      ShotTools.WritePng(path, bgra, _monitor.Width, _monitor.Height);
    }
    catch (Exception e)
    {
      Log($"  no failure screenshot: {e.Message}");
    }
  }

  private void Guard(string key, string state, Action action)
  {
    try
    {
      action();
    }
    catch (Exception e)
    {
      Log($"FAILED {key}--{state}: {e.GetType().Name}: {e.Message}");
      Screenshot(Path.Combine(_dir, $"{key}--{state}--FAILED.png"));
      Fail(key, state, $"{e.GetType().Name}: {e.Message}", null);
    }
  }

  // ── callouts from the UIA tree ──

  private static readonly HashSet<string> ExcludedTypes =
    ["StackPanel", "Border", "Panel", "Grid", "ProgressBar", "ItemsControl", "ScrollViewer"];

  private static readonly HashSet<string> ExcludedIds =
    ["ViewportHost", "ViewportDesktopRect", "DiagnosticsViewportDesktopRect"];

  /// <summary>
  /// Every element with an AutomationId under the crop (WinAppDriver's page source), without the
  /// containers and the <c>*Help</c> buttons (those are attached to their property as <c>help_id</c>).
  /// </summary>
  private List<RawCallout> TreeCallouts(ShotCtx c, Cropped crop, out int[]? calibration)
  {
    calibration = null;
    XDocument doc;
    try
    {
      doc = XDocument.Parse(c.S.App.PageSource);
    }
    catch (Exception e)
    {
      Log($"  page source unreadable: {e.Message}");
      return [];
    }
    var all = new List<(string Id, string Name, Rectangle Rect, bool Offscreen)>();
    foreach (var e in doc.Descendants())
    {
      var id = Attr(e, "AutomationId");
      if (string.IsNullOrWhiteSpace(id))
        continue;
      if (!int.TryParse(Attr(e, "x"), out int x) || !int.TryParse(Attr(e, "y"), out int y)
          || !int.TryParse(Attr(e, "width"), out int w) || !int.TryParse(Attr(e, "height"), out int h))
        continue;
      all.Add((id, Attr(e, "Name") ?? string.Empty, new Rectangle(x, y, w, h), string.Equals(Attr(e, "IsOffscreen"), "true", StringComparison.OrdinalIgnoreCase)));
    }

    // line the tree's places up with WinAppDriver's (stale after a window move)
    int dx = 0, dy = 0;
    var anchor = crop.AnchorId ?? all.Select(a => a.Id).FirstOrDefault(i => i.StartsWith("Tab.", StringComparison.Ordinal));
    var hit = all.FirstOrDefault(x => x.Id == anchor);
    if (anchor is not null && hit.Id is not null && c.RectOf(anchor) is { } known)
    {
      dx = known.X - hit.Rect.X;
      dy = known.Y - hit.Rect.Y;
    }
    calibration = [dx, dy];
    if (dx != 0 || dy != 0)
      Log($"  tree offset ({dx}, {dy}) from {anchor}");

    var present = all.Select(a => a.Id).ToHashSet();
    var result = new List<RawCallout>();
    var seen = new HashSet<string>();
    foreach (var (id, name, rect0, offscreen) in all)
    {
      if (offscreen || id.EndsWith("Help", StringComparison.Ordinal) || id.Contains('{') || id.Contains('.') && !id.StartsWith("Tab.", StringComparison.Ordinal)
          || ExcludedIds.Contains(id) || !seen.Add(id))
        continue;
      var info = _inventory.Get(id);
      // only the app's own controls: template parts (PART_*, the scroll bars' "Line up", "Page down")
      // are not in the AXAML inventory
      if (id.StartsWith("PART_", StringComparison.Ordinal) || info.Type is null && !id.StartsWith("Tab.", StringComparison.Ordinal))
        continue;
      if (info.Type is not null && (ExcludedTypes.Contains(info.Type) || info.Type == "TextBlock" && id.EndsWith("Label", StringComparison.Ordinal)))
        continue;
      var rect = new Rectangle(rect0.X + dx, rect0.Y + dy, rect0.Width, rect0.Height);
      if (!crop.Rect.Contains(new Point(rect.X + rect.Width / 2, rect.Y + rect.Height / 2)))
        continue;
      var label = name.Length > 0 && !name.StartsWith("AetherVk.", StringComparison.Ordinal) ? name : info.Label ?? id;
      result.Add(new RawCallout(id, Shorten(label), rect, info.Type, present.Contains(id + "Help") ? id + "Help" : null));
    }
    return result;
  }

  private static string? Attr(XElement e, string name) =>
    e.Attributes().FirstOrDefault(a => string.Equals(a.Name.LocalName, name, StringComparison.OrdinalIgnoreCase))?.Value;

  private static string Shorten(string s)
  {
    s = s.Replace("\r", " ").Replace("\n", " ").Trim();
    return s.Length <= 70 ? s : s[..67] + "...";
  }

  // ── splash and fatal dialog: processes of their own ──

  private Process StartApp(bool crash)
  {
    var exe = AppSession.AppPath();
    var p = new Process
    {
      StartInfo = new ProcessStartInfo(exe) { WorkingDirectory = Path.GetDirectoryName(exe)!, UseShellExecute = false, CreateNoWindow = true },
    };
    p.StartInfo.Environment["AETHERVK_E2E"] = "1";
    p.StartInfo.Environment["AETHERVK_UI_CULTURE"] = Culture;
    if (crash)
      p.StartInfo.Environment["AETHERVK_TEST_CRASH"] = "1";
    if (!p.Start())
      throw new InvalidOperationException($"could not start {exe}");
    return p;
  }

  private void CaptureSplash()
  {
    using var p = StartApp(crash: false);
    try
    {
      var end = DateTime.UtcNow.AddSeconds(30);
      while (DateTime.UtcNow < end && !p.HasExited)
      {
        var splash = Win32.TopLevelWindows(p.Id).FirstOrDefault(w => w.Title.Contains("Starting", StringComparison.Ordinal));
        if (splash is not null)
        {
          Thread.Sleep(500); // painted
          Emit("splash", "launch", Win32.FrameBounds(splash.Handle));
          return;
        }
        Thread.Sleep(40);
      }
      throw new TimeoutException("the splash window was not seen (too fast, or too slow to appear)");
    }
    finally
    {
      if (!p.HasExited)
        p.Kill(entireProcessTree: true);
    }
  }

  /// <summary>Best effort: the test crash switch may raise the unhandled-exception path instead of the Fatal Engine Error window.</summary>
  private void CaptureFatal()
  {
    using var p = StartApp(crash: true);
    try
    {
      var end = DateTime.UtcNow.AddSeconds(60);
      while (DateTime.UtcNow < end && !p.HasExited)
      {
        var fatal = Win32.TopLevelWindows(p.Id).FirstOrDefault(w => w.Title.Contains("Fatal Engine Error", StringComparison.Ordinal));
        if (fatal is not null)
        {
          Thread.Sleep(800);
          Emit("dlg-fatal", "dialog", Win32.FrameBounds(fatal.Handle));
          return;
        }
        Thread.Sleep(250);
      }
      throw new TimeoutException("AETHERVK_TEST_CRASH raised no 'Fatal Engine Error' window (it writes a crash report instead)");
    }
    finally
    {
      if (!p.HasExited)
        p.Kill(entireProcessTree: true);
    }
  }

  /// <summary>A shot of a window of a process outside any session: no callouts.</summary>
  private void Emit(string key, string state, Rectangle rect)
  {
    rect = Rectangle.Intersect(rect, new Rectangle(_monitor.Left, _monitor.Top, _monitor.Width, _monitor.Height));
    var stem = $"{key}--{state}--{_theme}";
    var bgra = Win32.Capture(rect.X, rect.Y, rect.Width, rect.Height);
    ShotTools.WritePng(Path.Combine(_dir, stem + ".png"), bgra, rect.Width, rect.Height);
    ShotTools.WritePng(Path.Combine(_dir, stem + "--tagged.png"), bgra, rect.Width, rect.Height);
    File.WriteAllText(Path.Combine(_dir, stem + ".callouts.json"), "[]");
    _manifest.Add(new ShotRecord(key, state, stem + ".png", "ok", null, rect.X, rect.Y, rect.Width, rect.Height, false, 0, null));
    Log($"shot {stem} {rect}");
  }

  // ── preferences, version, commit ──

  private static string PreferencesPath =>
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aethervk", "preferences.json");

  /// <summary>Theme, no diagnostic panels, no recent saves (the File menu shows none); the backup puts the file back.</summary>
  private static void PatchPreferences(string theme)
  {
    JsonObject root = File.Exists(PreferencesPath)
      ? JsonNode.Parse(File.ReadAllText(PreferencesPath)) as JsonObject ?? new JsonObject()
      : new JsonObject { ["format_version"] = 1 };
    root["theme"] = theme;
    root["show_diagnostic_panels"] = false;
    root["recent_saves"] = new JsonArray();
    Directory.CreateDirectory(Path.GetDirectoryName(PreferencesPath)!);
    File.WriteAllText(PreferencesPath, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
  }

  private static string AppVersion()
  {
    try
    {
      var info = FileVersionInfo.GetVersionInfo(AppSession.AppPath());
      return info.ProductVersion ?? info.FileVersion ?? "unknown";
    }
    catch (Exception)
    {
      return "unknown";
    }
  }

  /// <summary>Read-only: <c>git rev-parse HEAD</c>.</summary>
  private static string GitCommit()
  {
    try
    {
      using var p = Process.Start(new ProcessStartInfo("git", "rev-parse HEAD")
      {
        WorkingDirectory = AppSession.RepoRoot(), RedirectStandardOutput = true, UseShellExecute = false, CreateNoWindow = true,
      })!;
      var text = p.StandardOutput.ReadToEnd().Trim();
      p.WaitForExit(5000);
      return text.Length > 0 ? text : "unknown";
    }
    catch (Exception)
    {
      return "unknown";
    }
  }
}

/// <summary>The type and English label of each AutomationId, from <c>docs/manual/tools/inventory.json</c> (empty when it is missing).</summary>
internal sealed class Inventory(Dictionary<string, (string? Type, string? Label)> items)
{
  public static readonly Inventory Empty = new(new Dictionary<string, (string? Type, string? Label)>());

  public (string? Type, string? Label) Get(string id) => items.TryGetValue(id, out var v) ? v : (null, null);

  public static Inventory Load()
  {
    var path = Path.Combine(AppSession.RepoRoot(), "docs", "manual", "tools", "inventory.json");
    if (!File.Exists(path))
      return Empty;
    var items = new Dictionary<string, (string? Type, string? Label)>();
    using var doc = JsonDocument.Parse(File.ReadAllText(path));
    foreach (var e in doc.RootElement.EnumerateArray())
    {
      var id = e.TryGetProperty("id", out var i) && i.ValueKind == JsonValueKind.String ? i.GetString() : null;
      if (string.IsNullOrEmpty(id) || items.ContainsKey(id))
        continue;
      var type = e.TryGetProperty("type", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString() : null;
      var label = e.TryGetProperty("label", out var l) && l.ValueKind == JsonValueKind.String ? l.GetString() : null;
      items[id] = (type, label is "?" or "" ? null : label);
    }
    return new Inventory(items);
  }
}
