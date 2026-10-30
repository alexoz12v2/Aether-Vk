using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Utils;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Controls.Primitives;
using Avalonia.Markup.Xaml.Styling;
using Avalonia.Themes.Fluent;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// Single-file <c>.avk</c> saves through the real main window on the X11 backend: the application
/// composition root (<see cref="ServiceCollectionExtensions.AddCommonServices"/>, the real core,
/// <see cref="SaveFileService"/>, <see cref="MainWindowViewModel"/>, <see cref="MainWindow"/>) is
/// built in this process under an isolated <c>HOME</c>, so <c>~/.aethervk/open</c> (the archive
/// work folders) and <c>preferences.json</c> (Open Recent) land in a temp folder. Menus, tabs and
/// buttons are clicked with <c>xdotool</c> at their screen positions; the file pickers are the one
/// thing not driven: <see cref="IFileDialogService"/> is replaced at the composition root by a
/// scripted one (the app has no path hook, and the X11 picker is a portal or managed dialog), so
/// each File › Open… / Save As… gets its path from the test.
/// </summary>
[Collection("X11")]
public sealed class SaveArchiveX11Tests : IDisposable
{
  private readonly ITestOutputHelper _output;

  public SaveArchiveX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  public void Dispose()
  {
  }

  /// <summary>Answers the next file dialog with a queued path (null = cancelled).</summary>
  private sealed class ScriptedFileDialogs : IFileDialogService
  {
    private readonly Queue<string?> _answers = new();
    public List<string> Requests { get; } = [];

    public void Answer(string? path) => _answers.Enqueue(path);

    private Task<string?> Next(string kind, string title)
    {
      Requests.Add($"{kind}: {title}");
      return Task.FromResult(_answers.Count > 0 ? _answers.Dequeue() : null);
    }

    public Task<string?> ShowOpenFileDialogAsync(string title, string[]? filters = null) => Next("open", title);

    public Task<string?> ShowSaveFileDialogAsync(string title, string defaultExtension, string[]? filters = null) =>
      Next("save", title);

    public Task<string?> PickFolderAsync(string title) => Next("folder", title);
  }

  /// <summary>The unsaved-changes prompt must not appear in these steps: it cancels and records.</summary>
  private sealed class RecordingPrompt : IUnsavedChangesPrompt
  {
    public int Asked;

    public Task<UnsavedChoice> AskAsync(string saveName)
    {
      Interlocked.Increment(ref Asked);
      return Task.FromResult(UnsavedChoice.Cancel);
    }
  }

  /// <summary>The application composition under an isolated HOME, with the main window shown.</summary>
  private sealed class Rig : IDisposable
  {
    private static bool _stylesAdded;
    private readonly string? _homeBefore;

    public string Root { get; }
    public string Home { get; }
    public string SavesDir { get; }
    public IHost Host { get; }
    public ScriptedFileDialogs Dialogs { get; } = new();
    public RecordingPrompt Prompt { get; } = new();
    public MainWindowViewModel Vm { get; private set; }
    public MainWindow Main { get; private set; }
    public SaveFileService Saves => Host.Services.GetRequiredService<SaveFileService>();
    public PreferencesService Preferences => Host.Services.GetRequiredService<PreferencesService>();

    public Rig()
    {
      Root = Path.Combine(Path.GetTempPath(), "aethervk-x11-save-" + Guid.NewGuid().ToString("N")[..8]);
      Home = Path.Combine(Root, "home");
      SavesDir = Path.Combine(Root, "saves");
      Directory.CreateDirectory(Home);
      Directory.CreateDirectory(SavesDir);
      // LocalStorageService, PreferencesService and SaveArchive.DefaultWorkRoot() read the user
      // profile (HOME) when they are built: set it before the container resolves anything
      _homeBefore = Environment.GetEnvironmentVariable("HOME");
      Environment.SetEnvironmentVariable("HOME", Home);

      Host = Microsoft.Extensions.Hosting.Host.CreateDefaultBuilder([])
        .ConfigureServices(services =>
        {
          services.AddCommonServices();
          services.AddViewModels();
          // the test's file pickers (registered last: wins the single resolution)
          services.AddSingleton<IFileDialogService>(Dialogs);
          services.AddSingleton<IUnsavedChangesPrompt>(Prompt);
        })
        .Build();
      App.Host = Host;
      Host.Start();

      Ui(() =>
      {
        var app = Application.Current!;
        if (!_stylesAdded)
        {
          // what App.axaml gives the real application (the test process runs X11TestApp)
          _stylesAdded = true;
          app.Styles.Add(new FluentTheme());
          foreach (var src in new[]
          {
            "avares://Avalonia.Controls.DataGrid/Themes/Fluent.xaml",
            "avares://Avalonia.Controls.ColorPicker/Themes/Fluent/Fluent.xaml",
            "avares://AetherVk/Styles/Controls.axaml",
            "avares://AetherVk/Styles/Specs.axaml",
          })
            app.Styles.Add(new StyleInclude(new Uri("avares://AetherVk/")) { Source = new Uri(src) });
          app.Resources.MergedDictionaries.Add(
            new ResourceInclude(new Uri("avares://AetherVk/")) { Source = new Uri("avares://AetherVk/Styles/Colors.axaml") });
          app.DataTemplates.Add(new ViewLocator());
        }
      });

      // the core, off the UI thread as the splash screen does it
      _ = Host.Services.GetRequiredService<INativeRuntimeService>();

      Vm = null!;
      Main = null!;
      Ui(() =>
      {
        Preferences.Load();
        Vm = Host.Services.GetRequiredService<MainWindowViewModel>();
        Main = new MainWindow
        {
          DataContext = Vm,
          Width = 1600,
          Height = 1000,
          Position = new PixelPoint(0, 0),
        };
        if (Application.Current!.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
          desktop.MainWindow = Main;
        Main.Show();
      });
      Assert.True(WaitFor(() => XWinInfo(MainXid).Viewable, 20_000), "main window mapped");
      XDo($"windowfocus --sync {Hex(MainXid)}");
      Settle(1000);
    }

    public nint MainXid => Ui(() => Main.TryGetPlatformHandle()!.Handle);

    /// <summary>The window title as the X server has it.</summary>
    public string XTitle => Run("xdotool", $"getwindowname {Hex(MainXid)}").Trim();

    public T? Find<T>(string automationId) where T : Control =>
      Ui(() => Main.GetVisualDescendants().OfType<T>()
        .FirstOrDefault(c => AutomationProperties.GetAutomationId(c) == automationId));

    public T Require<T>(string automationId) where T : Control
    {
      Assert.True(WaitFor(() => Ui(() => Main.GetVisualDescendants().OfType<T>()
        .Any(c => AutomationProperties.GetAutomationId(c) == automationId && c.IsVisible)), 10_000), $"{automationId} visible");
      return Find<T>(automationId)!;
    }

    /// <summary>A real left click at the control's centre (it must be on screen).</summary>
    public void Click(Control c)
    {
      var p = Ui(() =>
      {
        var b = c.Bounds;
        return c.PointToScreen(new Point(b.Width / 2, b.Height / 2));
      });
      XDo($"mousemove {p.X} {p.Y} sleep 0.1 click 1");
      Settle(300);
    }

    /// <summary>The top-level File menu item (built by MenuMapper from the view model).</summary>
    public MenuItem TopMenu(string header) =>
      Ui(() => Main.GetVisualDescendants().OfType<Menu>().First().Items.OfType<MenuItem>()
        .First(m => string.Equals(m.Header as string, header, StringComparison.Ordinal)));

    public static MenuItem Child(MenuItem parent, string header) =>
      Ui(() => parent.Items.OfType<MenuItem>()
        .First(m => string.Equals(m.Header as string, header, StringComparison.Ordinal)));

    /// <summary>Opens the top-level menu by clicking it and clicks the entry (both real clicks).</summary>
    public MenuItem Menu(string top, string entry)
    {
      var menu = TopMenu(top);
      Click(menu);
      Assert.True(WaitFor(() => Ui(() => menu.IsSubMenuOpen)), $"{top} menu open");
      var item = Child(menu, entry);
      Assert.True(WaitFor(() => Ui(() => item.IsVisible && item.GetVisualRoot() is not null)), $"{top} › {entry} shown");
      Click(item);
      return item;
    }

    public void CloseMenus()
    {
      XDo("key Escape sleep 0.1 key Escape");
      Settle(300);
    }

    public void Dispose()
    {
      try
      {
        var done = Task.Run(() =>
        {
          Ui(() =>
          {
            // as App does at shutdown: the viewport goes before its window
            var runtime = Host.Services.GetService<INativeRuntimeService>();
            if (runtime?.PresentationEngineId is ulong pe)
              runtime.RemoveViewport(pe);
            if (Application.Current!.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
              desktop.MainWindow = null;
            Main.Close();
          });
          Settle(300);
          // stop, not dispose: the Avalonia loop outlives this rig (the app's ends first), and
          // its timers (title, camera) would read disposed services; the process exits after the run
          Host.StopAsync().Wait();
        });
        done.Wait(TimeSpan.FromSeconds(30));
      }
      catch (Exception)
      {
        // a shutdown hang must not take the test result with it
      }
      finally
      {
        App.Host = null;
        Environment.SetEnvironmentVariable("HOME", _homeBefore);
      }
    }
  }

  private static string RepoRoot()
  {
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "AetherVk.sln")))
      dir = dir.Parent;
    return dir?.FullName ?? throw new InvalidOperationException("AetherVk.sln not found above the test output");
  }

  private static string FixtureIndex() =>
    Path.Combine(RepoRoot(), "test_assets", "saves", "review_67p_2021-12", SaveIndex.FileName);

  private static int JetCount(string avk)
  {
    using var zip = ZipFile.OpenRead(avk);
    using var s = zip.GetEntry(SaveIndex.FileName)!.Open();
    using var doc = JsonDocument.Parse(s);
    return doc.RootElement.GetProperty("jets").GetArrayLength();
  }

  private static bool Clean(string title) => !title.TrimEnd().EndsWith('*');

  private void Step(string s) => _output.WriteLine($"── {s}");

  /// Save As a single .avk (title clean, Open Recent, zip layout); edit a jet through the UI and
  /// Save (mtime advances, dirty marker clears); open a truncated copy (refused, the open save
  /// stays). One sequential test: the core is a process singleton and each step builds on the last.
  [SkippableFact]
  public void SaveAsThenSaveThenRefuseTruncatedArchive()
  {
    Skip.IfNot(File.Exists(FixtureIndex()), $"fixture missing: {FixtureIndex()}");
    using var rig = new Rig();
    var avk = Path.Combine(rig.SavesDir, "x.avk");

    // ── 1. open the fixture folder, Save As x.avk ───────────────────────────────────────────
    Step("1: open the fixture folder");
    rig.Dialogs.Answer(FixtureIndex());
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.Contains("review_67p_2021-12", StringComparison.Ordinal), 90_000),
      $"fixture open, title: {rig.XTitle}");
    Assert.Equal(0, rig.Prompt.Asked);

    Step("1: Save As x.avk");
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Save As...");
    Assert.True(WaitFor(() => File.Exists(avk) && rig.XTitle.EndsWith("— x", StringComparison.Ordinal), 90_000),
      $"the .avk save, title: {rig.XTitle}");
    Assert.Equal(Ui(() => rig.Vm.WindowTitle), rig.XTitle);
    Assert.True(Clean(rig.XTitle), rig.XTitle);
    Assert.Equal(avk, rig.Saves.CurrentArchivePath);
    _output.WriteLine($".avk: {new FileInfo(avk).Length} bytes; dialogs: {string.Join(" | ", rig.Dialogs.Requests)}");

    // a zip whose first entry is the index
    using (var f = File.OpenRead(avk))
    {
      var magic = new byte[4];
      Assert.Equal(4, f.Read(magic, 0, 4));
      Assert.Equal(new byte[] { 0x50, 0x4B, 0x03, 0x04 }, magic);
    }
    using (var zip = ZipFile.OpenRead(avk))
      Assert.Equal(SaveIndex.FileName, zip.Entries[0].FullName);

    // Open Recent lists it (the menu entries the window shows, and the preferences behind them)
    var file = rig.TopMenu("File");
    rig.Click(file);
    Assert.True(WaitFor(() => Ui(() => file.IsSubMenuOpen)), "File menu open");
    var recent = Rig.Child(file, "Open Recent");
    rig.Click(recent);
    Assert.True(WaitFor(() => Ui(() => recent.IsSubMenuOpen)), "Open Recent open");
    var recentHeaders = Ui(() => recent.Items.OfType<MenuItem>().Where(m => m.IsVisible)
      .Select(m => m.Header as string ?? string.Empty).ToList());
    rig.CloseMenus();
    _output.WriteLine("Open Recent: " + string.Join(" | ", recentHeaders));
    Assert.Contains(recentHeaders, h => h.EndsWith(avk, StringComparison.Ordinal));
    Assert.DoesNotContain(recentHeaders, h => h == "(none)");
    Assert.Equal(avk, rig.Preferences.RecentSaves.First());
    Assert.True(Directory.Exists(Path.Combine(rig.Home, ".aethervk", "open")), "work folders under the isolated HOME");

    // ── 2. add a jet through the Model tab, File › Save ─────────────────────────────────────
    Step("2: add a jet in the Model tab");
    int jetsBefore = JetCount(avk);
    rig.Click(rig.Require<TextBlock>("Tab.ModelTab"));
    var dust = rig.Require<Expander>("DustSection");
    var toggle = Ui(() => dust.GetVisualDescendants().OfType<ToggleButton>().FirstOrDefault());
    if (toggle is not null)
      rig.Click(toggle);
    Assert.True(WaitFor(() => Ui(() => dust.IsExpanded)), "Dust section expanded");
    var addJet = rig.Require<Button>("AddJetButton");
    Assert.True(WaitFor(() => Ui(() => addJet.IsEnabled), 60_000), "Add jet enabled (comet committed)");
    Ui(() => addJet.BringIntoView());
    Settle(500);
    rig.Click(addJet);
    // the title timer (1 s) shows the dirty marker
    Assert.True(WaitFor(() => rig.XTitle.EndsWith("*"), 10_000), $"dirty marker, title: {rig.XTitle}");

    Step("2: File › Save");
    var mtimeBefore = File.GetLastWriteTimeUtc(avk);
    var sizeBefore = new FileInfo(avk).Length;
    Thread.Sleep(1100); // mtime resolution on coarse filesystems
    rig.Menu("File", "Save");
    Assert.True(WaitFor(() => File.GetLastWriteTimeUtc(avk) > mtimeBefore && Clean(rig.XTitle), 90_000),
      $"saved in place: mtime {File.GetLastWriteTimeUtc(avk):O} vs {mtimeBefore:O}, title: {rig.XTitle}");
    Assert.True(rig.XTitle.EndsWith("— x", StringComparison.Ordinal), rig.XTitle);
    Assert.Equal(jetsBefore + 1, JetCount(avk));
    _output.WriteLine($"saved: {sizeBefore} → {new FileInfo(avk).Length} bytes, jets {jetsBefore} → {JetCount(avk)}");
    Assert.Equal(0, rig.Prompt.Asked);

    // ── 3. a truncated copy is refused, x.avk stays open ────────────────────────────────────
    Step("3: open a truncated copy");
    var half = Path.Combine(rig.SavesDir, "half.avk");
    var bytes = File.ReadAllBytes(avk);
    File.WriteAllBytes(half, bytes[..(bytes.Length / 2)]);
    var breadcrumbs = rig.Vm.BreadcrumbViewModel;
    rig.Dialogs.Answer(half);
    rig.Menu("File", "Open...");
    BreadcrumbMessage? refusal = null;
    Assert.True(WaitFor(() =>
    {
      refusal = Ui(() => breadcrumbs.Messages.FirstOrDefault(m =>
        m.Title == "Open" && m.Content.Contains("could not be opened", StringComparison.Ordinal)));
      return refusal is not null;
    }, 30_000), "the refusal message");
    _output.WriteLine($"refusal: {refusal!.Content}");
    Assert.StartsWith("half.avk could not be opened", refusal.Content, StringComparison.Ordinal);
    Assert.Equal(3, refusal.Status);
    // the overlay that shows it is a mapped window while the message is up
    var overlay = Ui(() => (Application.Current!.ApplicationLifetime as IClassicDesktopStyleApplicationLifetime)?
      .Windows.OfType<UI.BreadcrumbWindow>().FirstOrDefault()?.TryGetPlatformHandle()?.Handle ?? 0);
    Assert.NotEqual(0, overlay);
    Assert.True(WaitFor(() => XWinInfo(overlay).Viewable, 5_000), "breadcrumb overlay mapped");

    Settle(500);
    Assert.True(rig.XTitle.EndsWith("— x", StringComparison.Ordinal), $"x.avk still open, title: {rig.XTitle}");
    Assert.Equal(avk, rig.Saves.CurrentArchivePath);
    Assert.DoesNotContain(half, rig.Preferences.RecentSaves);
    Assert.Equal(0, rig.Prompt.Asked);
  }
}
