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
    var fixture = AetherVk.TestAssets.TestAssets.Require("saves/review_67p_2021-12");
    using var rig = new AppRig();
    var avk = Path.Combine(rig.SavesDir, "x.avk");

    // ── 1. open the fixture folder, Save As x.avk ───────────────────────────────────────────
    Step("1: open the fixture folder");
    rig.Dialogs.Answer(rig.CopyOfSave(fixture));
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
    var recent = AppRig.Child(file, "Open Recent");
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
