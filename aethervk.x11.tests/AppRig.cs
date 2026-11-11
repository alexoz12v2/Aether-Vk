using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;
using Avalonia.Markup.Xaml.Styling;
using Avalonia.Themes.Fluent;
using Avalonia.VisualTree;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using AetherVk.Utils;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Xunit;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>Answers the next file dialog with a queued path (null = cancelled).</summary>
internal sealed class ScriptedFileDialogs : IFileDialogService
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
internal sealed class RecordingPrompt : IUnsavedChangesPrompt
{
  public int Asked;

  public Task<UnsavedChoice> AskAsync(string saveName)
  {
    Interlocked.Increment(ref Asked);
    return Task.FromResult(UnsavedChoice.Cancel);
  }
}

/// <summary>The application composition under an isolated HOME, with the main window shown.</summary>
internal sealed class AppRig : IDisposable
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
  /// <summary>Set by <see cref="Dispose"/>: what the shutdown threw (null = nothing).</summary>
  public Exception? ShutdownError { get; private set; }

  /// <summary>Set by <see cref="Dispose"/>: the shutdown finished within its time limit.</summary>
  public bool ShutdownCompleted { get; private set; }

  /// <summary>Called by <see cref="Dispose"/> on its worker thread after the window closed and the
  /// host stopped, right before the engine stops.</summary>
  public Action? BeforeEngineShutdown { get; set; }

  public SaveFileService Saves => Host.Services.GetRequiredService<SaveFileService>();
  public PreferencesService Preferences => Host.Services.GetRequiredService<PreferencesService>();

  public AppRig()
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

  /// <summary>
  /// The live view model of a docked tab (the layout's scoped tabs: Comet, Model, Timeline…),
  /// found through the tab groups the window shows; the tab need not be the selected one.
  /// </summary>
  public T Tab<T>() where T : TabItemViewModel =>
    Ui(() => Main.GetVisualDescendants().Select(v => v.DataContext).OfType<TabGroupNodeViewModel>().First()
      .LayoutTabs().OfType<T>().First());

  /// <summary>Runs <paramref name="start"/> on the UI thread and returns its task (awaited by the caller).</summary>
  public static Task<T> UiTask<T>(Func<Task<T>> start) => Ui(start);

  public static Task UiTask(Func<Task> start) => Ui(start);

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
          // as App does at shutdown: every viewport (the nucleus mini-view first, then the main
          // one) goes before its window, none may present to a closed window
          Host.Services.GetService<INativeRuntimeService>()?.RemoveAllViewports();
          if (Application.Current!.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop)
            desktop.MainWindow = null;
          Main.Close();
        });
        Settle(300);
        // stop, not dispose: the Avalonia loop outlives this rig (the app's ends first), and
        // its timers (title, camera) would read disposed services; the process exits after the run
        Host.StopAsync().Wait();
        BeforeEngineShutdown?.Invoke();
        // the engine stops as the app's runtime Dispose stops it: threads joined, device and
        // instance released while the X Display is still open (left running, the process exit
        // tore them down mid-call: SIGSEGV in the driver, or SIGABRT under the validation layer).
        // Not Dispose: the timers above may still query the service, which answers defaults now.
        Ui(() => (Host.Services.GetService<INativeRuntimeService>() as NativeRuntimeService)?.ShutdownSync());
      });
      ShutdownCompleted = done.Wait(TimeSpan.FromSeconds(30));
    }
    catch (Exception e)
    {
      // a shutdown failure must not take the test result with it (a test that checks the
      // shutdown reads ShutdownError)
      ShutdownError = e;
    }
    finally
    {
      App.Host = null;
      Environment.SetEnvironmentVariable("HOME", _homeBefore);
    }
  }

  /// <summary>
  /// A copy of the save folder <paramref name="fixtureDir"/> (from test_assets/ASSETS.tsv, a shared
  /// cache that opening a save would rewrite) in <see cref="SavesDir"/> under the same name; its index.
  /// </summary>
  public string CopyOfSave(string fixtureDir)
  {
    var to = Path.Combine(SavesDir, Path.GetFileName(fixtureDir.TrimEnd(Path.DirectorySeparatorChar)));
    foreach (var f in Directory.GetFiles(fixtureDir, "*", SearchOption.AllDirectories))
    {
      var target = Path.Combine(to, Path.GetRelativePath(fixtureDir, f));
      Directory.CreateDirectory(Path.GetDirectoryName(target)!);
      File.Copy(f, target);
    }
    return Path.Combine(to, SaveIndex.FileName);
  }

  /// <summary>The checkout root (the folder holding AetherVk.sln).</summary>
  public static string RepoRoot()
  {
    var dir = new DirectoryInfo(AppContext.BaseDirectory);
    while (dir is not null && !File.Exists(Path.Combine(dir.FullName, "AetherVk.sln")))
      dir = dir.Parent;
    return dir?.FullName ?? throw new InvalidOperationException("AetherVk.sln not found above the test output");
  }
}
