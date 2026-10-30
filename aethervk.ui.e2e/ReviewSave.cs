using System.Diagnostics;
using System.Globalization;
using System.Text.RegularExpressions;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace AetherVk.UI.E2E;

/// <summary>
/// The reviewer's 67P save (<c>test_assets/saves/review_67p_2021-12</c>, four jets, one FITS
/// billboard, committed 2021-12-12 → 2021-12-28, saved at the end epoch) and the timeline steps
/// the playback and mail tests share.
/// </summary>
public static partial class ReviewSave
{
  public const string FixtureName = "review_67p_2021-12";

  /// <summary>SpeedPicker items, in <c>TimelineTabViewModel.AvailableSpeeds</c> order: 10 s/s, 1 min/s, 1 h/s, 3 h/s, 1 d/s.</summary>
  public const int OneHourPerSecondIndex = 2;

  /// <summary>The item after <see cref="OneHourPerSecondIndex"/>: 3 h/s.</summary>
  public const int ThreeHoursPerSecondIndex = 3;

  /// <summary>
  /// Two epoch boxes agree within this: the committed range is shown as committed (in the save's
  /// time scale), the clock in the chosen one, and TDB and UTC differ by 69.2 s.
  /// </summary>
  public static readonly TimeSpan EpochTolerance = TimeSpan.FromSeconds(70);

  public static string FixtureDir => Path.Combine(AppSession.RepoRoot(), "test_assets", "saves", FixtureName);

  /// <summary>A fresh <c>TestResults/e2e/&lt;name&gt;</c> holding a copy of the save.</summary>
  public static (string Artifacts, string SaveDir) Prepare(string artifactsName)
  {
    var artifacts = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", artifactsName);
    if (Directory.Exists(artifacts))
      Directory.Delete(artifacts, recursive: true);
    var saveDir = Path.Combine(artifacts, FixtureName);
    ReviewerMailTests.CopyDir(FixtureDir, saveDir);
    return (artifacts, saveDir);
  }

  /// <summary>File → Open... on <paramref name="path"/> (an index or an .avk), until the title names the save.</summary>
  public static void Open(AppSession s, string path)
  {
    Thread.Sleep(2000);
    s.Menu("File", "Open...");
    Thread.Sleep(1000);
    s.AnswerFileDialog(path);
    Ui.WaitUntil(() => s.WindowTitle.Contains("review_67p", StringComparison.OrdinalIgnoreCase),
      TimeSpan.FromSeconds(90), "the save to open");
    Thread.Sleep(5000);
  }

  /// <summary>Low particle quality (a GTX 1070 class GPU). The slider writes the preferences.</summary>
  public static void SetLowQuality(AppSession s)
  {
    s.SelectTab("SettingsTab");
    s.Expand("ParticleRenderingSection", "ParticleQualitySlider");
    s.SetRange("ParticleQualitySlider", 2, step: 1, tolerance: 0);
    Ui.WaitUntil(() => s.Find("ParticleQualityLabel").Text.Contains("Low", StringComparison.Ordinal),
      TimeSpan.FromSeconds(10), "the Low preset");
  }

  /// <summary>Hides the save's FITS billboard: its window sits over the comet and would be measured instead of the dust.</summary>
  public static void HideTheImage(AppSession s) => FitsComparisonSteps.SetVisible(s, 0, visible: false);

  /// <summary>The text of a displayed element of the overlay window (a top-level window of its own).</summary>
  public static string? OverlayText(AppSession s, string automationId) =>
    s.Desktop.FindElementsByAccessibilityId(automationId).FirstOrDefault(e => e.Displayed)?.Text;

  /// <summary>An epoch box's text ("2021-12-12 00:00:00", "2021-12-12T00:00", "2021-12-12") as a date.</summary>
  public static DateTime ParseEpoch(string text)
  {
    var m = EpochRegex().Match(text);
    if (!m.Success)
      throw new FormatException($"no epoch in \"{text}\"");
    var time = m.Groups["time"].Success ? m.Groups["time"].Value : "00:00";
    var seconds = m.Groups["sec"].Success ? m.Groups["sec"].Value : "00";
    return DateTime.ParseExact($"{m.Groups["date"].Value} {time}:{seconds}", "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
  }

  [GeneratedRegex(@"(?<date>\d{4}-\d{2}-\d{2})(?:[ T](?<time>\d{2}:\d{2})(?::(?<sec>\d{2}))?)?")]
  private static partial Regex EpochRegex();

  /// <summary>The engine clock (the Timeline tab's epoch box).</summary>
  public static DateTime Epoch(AppSession s) => ParseEpoch(s.Find("CurrentEpochBox").Text);

  public static DateTime StartEpoch(AppSession s) => ParseEpoch(s.Find("TimelineStartEpochBox").Text);

  public static DateTime EndEpoch(AppSession s) => ParseEpoch(s.Find("TimelineEndEpochBox").Text);

  public static bool Near(DateTime a, DateTime b) => (a - b).Duration() <= EpochTolerance;

  /// <summary>Playing: Reset is shown only while paused (the play button's icon has no text).</summary>
  public static bool IsPlaying(AppSession s) => s.TryFind("ResetButton") is null;

  public static void Play(AppSession s)
  {
    s.SelectTab("TimelineTab");
    if (!IsPlaying(s))
      s.Click("PlayPauseButton");
    Ui.WaitUntil(() => IsPlaying(s), TimeSpan.FromSeconds(10), "playing");
  }

  public static void Pause(AppSession s)
  {
    s.SelectTab("TimelineTab");
    if (IsPlaying(s))
      s.Click("PlayPauseButton");
    Ui.WaitUntil(() => !IsPlaying(s), TimeSpan.FromSeconds(10), "paused");
  }

  /// <summary>Paused, then Reset: the clock back at the committed start (the save opens at its end).</summary>
  public static void ResetToStart(AppSession s)
  {
    Pause(s);
    s.Click("ResetButton");
    var start = StartEpoch(s);
    Ui.WaitUntil(() => Near(Epoch(s), start), TimeSpan.FromSeconds(60), $"the clock back at the start {start:u}");
  }

  /// <summary>
  /// Simulated seconds per wall-clock second over <paramref name="samples"/> reads of the epoch
  /// box <paramref name="gap"/> apart; asserts the epoch never goes back.
  /// </summary>
  public static double MeasureRate(AppSession s, int samples, TimeSpan gap, Action<string>? log = null)
  {
    var box = s.Find("CurrentEpochBox");
    var clock = Stopwatch.StartNew();
    var reads = new List<(double Wall, DateTime Epoch)>();
    for (int i = 0; i < samples; i++)
    {
      if (i > 0)
        Thread.Sleep(gap);
      var epoch = ParseEpoch(box.Text);
      reads.Add((clock.Elapsed.TotalSeconds, epoch));
    }
    log?.Invoke("epochs: " + string.Join(", ", reads.Select(r => $"{r.Wall:0.00} s → {r.Epoch:u}")));
    for (int i = 1; i < reads.Count; i++)
      Assert.True(reads[i].Epoch >= reads[i - 1].Epoch, $"the epoch went back ({reads[i - 1].Epoch:u} -> {reads[i].Epoch:u})");
    var (first, last) = (reads[0], reads[^1]);
    return (last.Epoch - first.Epoch).TotalSeconds / (last.Wall - first.Wall);
  }
}

/// <summary>Puts the user's preferences.json back (the quality slider writes it).</summary>
public sealed class PreferencesBackup : IDisposable
{
  private static readonly string Path_ =
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aethervk", "preferences.json");

  private readonly byte[]? _saved = File.Exists(Path_) ? File.ReadAllBytes(Path_) : null;

  public void Dispose()
  {
    if (_saved is not null)
      File.WriteAllBytes(Path_, _saved);
  }
}

/// <summary>
/// One app session on the reviewer's save, shared by the tests of a class (an xunit class
/// fixture). It is launched on the first <see cref="Session"/> call, after the test's skip check,
/// so off Windows nothing starts; a failed open is not retried by every following test.
/// <see cref="Close"/> ends it for a test that needs a session of its own.
/// </summary>
public abstract class ReviewSaveFixture : IDisposable
{
  private AppSession? _session;
  private Exception? _failure;
  private PreferencesBackup? _preferences;
  private int _launches;

  protected abstract string ArtifactsName { get; }

  protected virtual bool LowQuality => false;

  public string Artifacts { get; private set; } = string.Empty;

  public string SaveDir { get; private set; } = string.Empty;

  public AppSession Session(ITestOutputHelper output)
  {
    if (_failure is not null)
      throw new InvalidOperationException("the shared session on the reviewer's save did not open", _failure);
    if (_session is not null)
      return _session;
    if (_launches == 0)
      (Artifacts, SaveDir) = ReviewSave.Prepare(ArtifactsName);
    _launches++;
    var s = AppSession.Launch(Path.Combine(Artifacts, $"app-{_launches}.log"), diagnosticPanels: false);
    try
    {
      output.WriteLine($"{DateTime.Now:HH:mm:ss} open the reviewer's save");
      ReviewSave.Open(s, Path.Combine(SaveDir, "aethervk_index.json"));
      Opened(s);
      if (LowQuality)
      {
        _preferences ??= new PreferencesBackup();
        ReviewSave.SetLowQuality(s);
      }
      ReviewSave.HideTheImage(s);
      Thread.Sleep(3000);
    }
    catch (Exception e)
    {
      _failure = e;
      try { s.Desktop.SaveScreenshot(Path.Combine(Artifacts, "failure-open.png")); } catch (Exception) { }
      s.Dispose();
      throw;
    }
    return _session = s;
  }

  /// <summary>Checks right after the open, before the image is hidden.</summary>
  protected virtual void Opened(AppSession s)
  {
  }

  public void Screenshot(string name)
  {
    try { _session?.Desktop.SaveScreenshot(Path.Combine(Artifacts, name)); } catch (Exception) { }
  }

  public void Close()
  {
    _session?.Dispose();
    _session = null;
  }

  public void Dispose()
  {
    Close();
    _preferences?.Dispose();
  }
}

/// <summary>
/// Runs a class's tests in the natural order of their names ("R2" before "R10"), so the ones
/// sharing a session see it in a known state and the ones that close it run last.
/// </summary>
public sealed partial class NaturalNameOrderer : ITestCaseOrderer
{
  public const string TypeName = "AetherVk.UI.E2E.NaturalNameOrderer";
  public const string AssemblyName = "AetherVk.UI.E2E";

  public IEnumerable<TTestCase> OrderTestCases<TTestCase>(IEnumerable<TTestCase> testCases) where TTestCase : ITestCase =>
    testCases.OrderBy(t => t.TestMethod.Method.Name, Comparer<string>.Create(Compare));

  private static int Compare(string a, string b)
  {
    var (x, y) = (Parts().Matches(a), Parts().Matches(b));
    for (int i = 0; i < Math.Min(x.Count, y.Count); i++)
    {
      var (p, q) = (x[i].Value, y[i].Value);
      int c = char.IsDigit(p[0]) && char.IsDigit(q[0])
        ? long.Parse(p, CultureInfo.InvariantCulture).CompareTo(long.Parse(q, CultureInfo.InvariantCulture))
        : string.CompareOrdinal(p, q);
      if (c != 0)
        return c;
    }
    return x.Count.CompareTo(y.Count);
  }

  [GeneratedRegex(@"\d+|\D+")]
  private static partial Regex Parts();
}
