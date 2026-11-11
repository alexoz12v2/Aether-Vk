// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac: the steps the shared
// scenarios are made of, written only against IE2EApp.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using Xunit;
using Ids = AetherVk.Logic.Automation.CompactAutomationIds;

namespace AetherVk.UI.E2E.Shared;

public static partial class E2ESteps
{
  /// <summary>The scenarios run in English with the en-US number format on both platforms.</summary>
  public const string Culture = "en-US";

  /// <summary>The main 3D viewport's host (not a compact id: it predates the compact panel).</summary>
  public const string MainViewportId = "ViewportHost";

  /// <summary>The menu titles in the en-US culture (the drivers map them to their own locators).</summary>
  public const string FileMenu = "File";
  public const string OpenItem = "Open...";
  public const string SaveAsItem = "Save As...";
  public const string LayoutMenu = "Layout";
  public const string ClassicItem = "Classic";
  public const string CompactItem = "Compact";

  /// <summary>The playback speed items, as in the Timeline tab: 10 s/s, 1 min/s, 1 h/s, 3 h/s, 1 d/s.</summary>
  public const int OneHourPerSecondIndex = 2;

  /// <summary>A not-black image: this share of pixels brighter than <see cref="LitLuminance"/>.</summary>
  public const double MinLitShare = 1.0 / 200;

  public const double LitLuminance = 0.05;

  /// <summary>A frame pair while playing changes at least this share of grid cells.</summary>
  public const double PlayingChanged = 0.02;

  public const string FixtureName = "review_67p_2021-12";

  /// <summary>The FITS fixture the import steps use (a 4 × 3 ramp with the 67P header).</summary>
  public const string FitsAsset = "fits_lt_67p_header_4x3.fits";

  public static string RepoRoot()
  {
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "AetherVk.sln")))
      dir = dir.Parent;
    return dir?.FullName ?? throw new InvalidOperationException("AetherVk.sln not found above the test output");
  }

  /// <summary>A fresh artifacts folder <c>&lt;host root&gt;/&lt;name&gt;</c>.</summary>
  public static string Artifacts(IE2EHost host, string name)
  {
    var dir = Path.Combine(host.ArtifactsRoot, name);
    if (Directory.Exists(dir))
      Directory.Delete(dir, recursive: true);
    Directory.CreateDirectory(dir);
    return dir;
  }

  /// <summary>A copy of the reviewer's save under <paramref name="artifacts"/>; returns its index path.</summary>
  public static string CopyFixture(string artifacts)
  {
    var from = AetherVk.TestAssets.TestAssets.Require("saves/" + FixtureName);
    var to = Path.Combine(artifacts, FixtureName);
    Directory.CreateDirectory(to);
    foreach (var f in Directory.GetFiles(from, "*", SearchOption.AllDirectories))
    {
      var dest = Path.Combine(to, Path.GetRelativePath(from, f));
      Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
      File.Copy(f, dest, overwrite: true);
    }
    return Path.Combine(to, "aethervk_index.json");
  }

  /// <summary>
  /// A copy of the FITS fixture under <paramref name="artifacts"/>, made new to the library: the
  /// reviewer's save already holds this fixture, and the asset library skips an image whose
  /// decoded pixels it has ("Already imported"), compared after the 8-bit stretch. The copy has its
  /// first two pixels swapped (the fixture's differ: 100 and 200).
  /// </summary>
  public static string CopyFits(string artifacts, string name = "import_67p.fits")
  {
    var dest = Path.Combine(artifacts, name);
    var bytes = File.ReadAllBytes(AetherVk.TestAssets.TestAssets.Require(FitsAsset));
    File.WriteAllBytes(dest, SwapFirstTwoPixels(bytes));
    return dest;
  }

  /// <summary>
  /// The primary HDU's first two data values swapped. The data start at the 2880-byte block after
  /// the END card; a value is |BITPIX|/8 bytes.
  /// </summary>
  internal static byte[] SwapFirstTwoPixels(byte[] fits)
  {
    const int Card = 80, Block = 2880;
    var copy = (byte[])fits.Clone();
    int bitpix = 0;
    for (int i = 0; (i + 1) * Card <= copy.Length; i++)
    {
      string card = System.Text.Encoding.ASCII.GetString(copy, i * Card, Card);
      if (card.StartsWith("BITPIX  =", StringComparison.Ordinal))
        bitpix = int.Parse(card[10..30].Trim(), System.Globalization.CultureInfo.InvariantCulture);
      if (card.TrimEnd() != "END")
        continue;
      int data = ((i + 1) * Card + Block - 1) / Block * Block;
      int size = Math.Abs(bitpix) / 8;
      if (size == 0 || data + 2 * size > copy.Length)
        break;
      var first = copy[data..(data + size)];
      Array.Copy(copy, data + size, copy, data, size);
      first.CopyTo(copy, data + size);
      return copy;
    }
    throw new InvalidOperationException("no primary image data in the FITS fixture");
  }

  public static void Step(Action<string> log, string text) => log($"{DateTime.Now:HH:mm:ss} {text}");

  // ── save files ─────────────────────────────────────────────────────────────

  /// <summary>File › Open... on <paramref name="path"/>, until the title names the save.</summary>
  public static void OpenSave(IE2EApp app, string path, string titleContains = "review_67p")
  {
    Thread.Sleep(2000);
    app.Foreground();
    app.Menu(FileMenu, OpenItem);
    Thread.Sleep(1000);
    app.AnswerOpenDialog(path);
    E2EWait.Until(() => app.WindowTitle.Contains(titleContains, StringComparison.OrdinalIgnoreCase),
      TimeSpan.FromSeconds(90), $"the save \"{titleContains}\" to open");
    Thread.Sleep(5000);
    app.EnsureAlive("opening the save");
  }

  /// <summary>File › Save As... to <paramref name="avk"/>, until the file exists and the title is clean.</summary>
  public static void SaveAs(IE2EApp app, string avk)
  {
    app.Foreground();
    app.Menu(FileMenu, SaveAsItem);
    Thread.Sleep(1000);
    app.AnswerSaveDialog(avk);
    E2EWait.Until(() => File.Exists(avk) && !app.WindowTitle.EndsWith('*'), TimeSpan.FromSeconds(90), $"the save {Path.GetFileName(avk)}");
    Thread.Sleep(1000);
  }

  /// <summary>The save index's text: an .avk's first entry, or a folder's / an index file's own.</summary>
  public static string ReadIndexJson(string path)
  {
    if (Directory.Exists(path))
      return File.ReadAllText(Path.Combine(path, "aethervk_index.json"));
    if (path.EndsWith(".avk", StringComparison.OrdinalIgnoreCase))
    {
      using var zip = ZipFile.OpenRead(path);
      var entry = zip.GetEntry("aethervk_index.json") ?? throw new InvalidDataException($"{path} has no aethervk_index.json");
      using var reader = new StreamReader(entry.Open());
      return reader.ReadToEnd();
    }
    return File.ReadAllText(path);
  }

  /// <summary>The index's <c>ui_layout</c>: (present, value).</summary>
  public static (bool Present, string? Value) UiLayout(string savePath)
  {
    using var doc = JsonDocument.Parse(ReadIndexJson(savePath));
    return doc.RootElement.TryGetProperty("ui_layout", out var v)
      ? (true, v.ValueKind == JsonValueKind.String ? v.GetString() : v.GetRawText())
      : (false, null);
  }

  // ── layout ─────────────────────────────────────────────────────────────────

  public static bool IsCompact(IE2EApp app) =>
    app.TryFind(Ids.CompactPanel) is { Displayed: true } p && p.Rect.Width > 20 && p.Rect.Height > 20;

  /// <summary>Layout › Classic / Compact, until the compact panel is shown or gone.</summary>
  public static void SetLayout(IE2EApp app, bool compact)
  {
    app.Foreground();
    app.Menu(LayoutMenu, compact ? CompactItem : ClassicItem);
    E2EWait.Until(() => IsCompact(app) == compact, TimeSpan.FromSeconds(20), compact ? "the compact panel" : "the classic layout");
    Thread.Sleep(1500);
  }

  /// <summary>The main viewport's rectangle: the one displayed <c>ViewportHost</c>, never the mini-view.</summary>
  public static E2ERect MainViewport(IE2EApp app)
  {
    E2EWait.Until(() => app.TryFind(MainViewportId) is { Displayed: true } e && e.Rect.Width > 50 && e.Rect.Height > 50,
      TimeSpan.FromSeconds(30), "the main viewport");
    return app.ViewportRect(MainViewportId);
  }

  // ── the nucleus mini-view ───────────────────────────────────────────────────

  public static bool MiniShown(IE2EApp app) =>
    app.TryFind(Ids.NucleusMiniViewHost) is { Displayed: true } h && h.Rect.Width > 50 && h.Rect.Height > 50;

  /// <summary>Opens or closes the mini-view's expander until its host is shown or hidden.</summary>
  public static void SetMiniExpanded(IE2EApp app, bool expanded)
  {
    for (int attempt = 0; attempt < 3 && MiniShown(app) != expanded; attempt++)
    {
      app.Toggle(Ids.NucleusMiniViewExpander);
      try
      {
        E2EWait.Until(() => MiniShown(app) == expanded, TimeSpan.FromSeconds(5));
      }
      catch (TimeoutException) when (attempt < 2)
      {
      }
    }
    Assert.True(MiniShown(app) == expanded, $"the nucleus mini-view did not {(expanded ? "expand" : "collapse")}");
    Thread.Sleep(expanded ? 3000 : 1000);
  }

  // ── numbers and epochs ──────────────────────────────────────────────────────

  public static string Num(double value) => value.ToString("0.###", CultureInfo.GetCultureInfo(Culture));

  public static double ParseNum(string text)
  {
    var m = NumberRegex().Match(text.Replace('−', '-'));
    if (!m.Success)
      throw new FormatException($"no number in \"{text}\"");
    return double.Parse(m.Value.Replace(",", ""), CultureInfo.InvariantCulture);
  }

  [GeneratedRegex(@"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")]
  private static partial Regex NumberRegex();

  /// <summary>Every epoch in a text ("2021-12-12 00:00:00", "2021-12-12T00:00", "2021-12-12").</summary>
  public static IReadOnlyList<DateTime> Epochs(string text) =>
    EpochRegex().Matches(text).Select(m =>
    {
      var time = m.Groups["time"].Success ? m.Groups["time"].Value : "00:00";
      var seconds = m.Groups["sec"].Success ? m.Groups["sec"].Value : "00";
      return DateTime.ParseExact($"{m.Groups["date"].Value} {time}:{seconds}", "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
    }).ToList();

  [GeneratedRegex(@"(?<date>\d{4}-\d{2}-\d{2})(?:[ T](?<time>\d{2}:\d{2})(?::(?<sec>\d{2}))?)?")]
  private static partial Regex EpochRegex();

  // ── playback ────────────────────────────────────────────────────────────────

  /// <summary>The engine clock in the compact Playback card; null while it shows none.</summary>
  public static DateTime? Epoch(IE2EApp app) =>
    app.TryFind(Ids.PlaybackEpoch) is null ? null : Epochs(app.Text(Ids.PlaybackEpoch)).Cast<DateTime?>().FirstOrDefault();

  /// <summary>The clock moves forward over <paramref name="gap"/> (default 1.5 s).</summary>
  public static bool Advancing(IE2EApp app, TimeSpan? gap = null)
  {
    var a = Epoch(app);
    Thread.Sleep(gap ?? TimeSpan.FromSeconds(1.5));
    var b = Epoch(app);
    return a is { } x && b is { } y && y > x;
  }

  /// <summary>Paused, back at the start, then playing at 1 h/s (compact Playback card).</summary>
  public static void Play(IE2EApp app, int speedIndex = OneHourPerSecondIndex)
  {
    app.ChooseAt(Ids.PlaybackSpeed, speedIndex);
    if (Advancing(app))
      return;
    if (app.TryFind(Ids.PlaybackReset) is { Displayed: true, Enabled: true })
    {
      app.Click(Ids.PlaybackReset);
      Thread.Sleep(2000);
    }
    app.Click(Ids.PlaybackPlayPause);
    E2EWait.Until(() => Advancing(app), TimeSpan.FromSeconds(20), "the clock to advance (playing)");
  }

  public static void Pause(IE2EApp app)
  {
    if (Advancing(app))
      app.Click(Ids.PlaybackPlayPause);
    E2EWait.Until(() => !Advancing(app), TimeSpan.FromSeconds(10), "the clock to hold (paused)");
  }

  // ── frames ──────────────────────────────────────────────────────────────────

  /// <summary>The element's central 90 % on the 32² grid.</summary>
  public static double[] Grid(IE2EApp app, string automationId) =>
    FrameMetricsCore.Sample(app.SampleLuminance, app.ViewportRect(automationId));

  /// <summary>The changed share of <paramref name="pairs"/> consecutive frame pairs <paramref name="gap"/> apart.</summary>
  public static List<double> ChangedPairs(IE2EApp app, string automationId, int pairs, TimeSpan gap)
  {
    var changed = new List<double>();
    var previous = Grid(app, automationId);
    for (int i = 0; i < pairs; i++)
    {
      Thread.Sleep(gap);
      var frame = Grid(app, automationId);
      changed.Add(FrameMetricsCore.Changed(previous, frame));
      previous = frame;
    }
    return changed;
  }

  /// <summary>
  /// Where to click the viewport to give it the keyboard: its upper middle, unless a desktop
  /// billboard's window covers that point (the reviewer's save shows one over the viewport), then
  /// the middle band beside the billboards, from the centre outwards.
  /// </summary>
  public static (int X, int Y) ViewportClickPoint(IE2EApp app, E2ERect viewport)
  {
    var billboards = app.BillboardFrames();
    int x0 = (int)viewport.CentreX, y0 = viewport.Y + viewport.Height / 4;
    if (!billboards.Any(b => b.Contains(x0, y0)))
      return (x0, y0);
    int y = viewport.Y + viewport.Height / 2;
    for (int step = 0; step <= 8; step++)
      foreach (int sign in step == 0 ? [1] : new[] { -1, 1 })
      {
        int x = x0 + sign * step * viewport.Width / 20;
        if (x > viewport.X + 4 && x < viewport.Right - 4 && !billboards.Any(b => b.Contains(x, y)))
          return (x, y);
      }
    throw new InvalidOperationException($"no point of the viewport {viewport} is free of the billboards");
  }

  public static string Pct(double v) => v.ToString("P1", CultureInfo.InvariantCulture);

  // ── export ──────────────────────────────────────────────────────────────────

  /// <summary>
  /// The compact Export button, answered with <paramref name="chosenPng"/>: waits for the model
  /// image and the image with the FITS overlay next to it.
  /// </summary>
  public static (string Model, string WithFits) Export(IE2EApp app, string chosenPng)
  {
    var dir = Path.GetDirectoryName(chosenPng)!;
    var stem = Path.GetFileNameWithoutExtension(chosenPng);
    var model = Path.Combine(dir, stem + "_model.png");
    var fits = Path.Combine(dir, stem + "_with_fits.png");
    app.Click(Ids.ExportButton);
    Thread.Sleep(1000);
    app.AnswerSaveDialog(chosenPng);
    E2EWait.Until(() => File.Exists(model) && File.Exists(fits), TimeSpan.FromSeconds(60), "both exported PNGs");
    Thread.Sleep(500);
    return (model, fits);
  }

  /// <summary>The exported model image is the main viewport's size (±2 px) and not black.</summary>
  public static PngImage AssertModelPng(string modelPath, E2ERect mainViewport, Action<string> log)
  {
    var png = PngCodec.Read(modelPath);
    double lit = FrameMetricsCore.LitShare(png.Luminance(), LitLuminance);
    log($"model PNG {png.Width}×{png.Height}, main viewport {mainViewport}, lit share {Pct(lit)}");
    Assert.InRange(png.Width, mainViewport.Width - 2, mainViewport.Width + 2);
    Assert.InRange(png.Height, mainViewport.Height - 2, mainViewport.Height + 2);
    Assert.True(lit > MinLitShare, $"the model image is black ({Pct(lit)} lit)");
    return png;
  }

  // ── shutdown ────────────────────────────────────────────────────────────────

  /// <summary>Quits; asserts a clean exit, no crash report since <paramref name="sinceUtc"/> and no panic in the log.</summary>
  public static void QuitClean(IE2EHost host, IE2EApp app, DateTime sinceUtc, Action<string> log)
  {
    var exit = app.Quit(TimeSpan.FromSeconds(30));
    log($"quit: {exit}");
    var panics = app.Lines.Where(l => l.Contains("panicked at", StringComparison.Ordinal)
                                      || l.Contains("Unhandled exception", StringComparison.OrdinalIgnoreCase)).ToList();
    Assert.True(exit.Clean, $"the app did not exit cleanly: {exit}");
    Assert.True(panics.Count == 0, "the log has a panic / unhandled exception:\n" + string.Join('\n', panics.Take(5)));
    var reports = host.CrashReportsSince(sinceUtc);
    Assert.True(reports.Count == 0, "crash reports written: " + string.Join(", ", reports));
  }

  /// <summary>Runs <paramref name="body"/> on a launched app; a failure leaves a screenshot, the app is always closed.</summary>
  public static void WithApp(IE2EHost host, string artifacts, string logName, Action<IE2EApp> body)
  {
    var app = host.Launch(Path.Combine(artifacts, logName + ".log"));
    try
    {
      body(app);
    }
    catch
    {
      try { app.SaveScreenshot(Path.Combine(artifacts, $"failure-{logName}.png")); } catch (Exception) { }
      throw;
    }
    finally
    {
      app.Dispose();
    }
  }
}

/// <summary>Puts the user's <c>~/.aethervk/preferences.json</c> back (layout and mini-view state are kept there).</summary>
public sealed class E2EPreferencesBackup : IDisposable
{
  private static readonly string PreferencesPath =
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aethervk", "preferences.json");

  private readonly byte[]? _saved = File.Exists(PreferencesPath) ? File.ReadAllBytes(PreferencesPath) : null;

  public void Dispose()
  {
    try
    {
      if (_saved is not null)
        File.WriteAllBytes(PreferencesPath, _saved);
      else if (File.Exists(PreferencesPath))
        File.Delete(PreferencesPath); // there was none before the test
    }
    catch (IOException)
    {
    }
  }
}
