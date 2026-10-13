using System;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// File menu of the main window over a <see cref="SaveFileService"/> with a fake application:
/// the entries, Save → Save As when unsaved, the dirty title, the unsaved-changes prompt (Save /
/// Discard / Cancel) before New and Open, and Open Recent.
/// </summary>
public sealed class FileMenuTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_filemenu_" + Guid.NewGuid().ToString("N"));
  private readonly FakeSaveFileHost _host;
  private readonly SaveFileService _saves;
  private readonly PreferencesService _prefs;
  private readonly Mock<IFileDialogService> _dialogs = new();
  private readonly Mock<IUnsavedChangesPrompt> _prompt = new();
  private readonly MainWindowViewModel _vm;

  public FileMenuTests()
  {
    Directory.CreateDirectory(_root);
    _host = new FakeSaveFileHost(_root);
    _prefs = new PreferencesService(Path.Combine(_root, "preferences.json"), new InputRegistry());
    _saves = new SaveFileService(_host, new SystemClock(), _prefs);
    var dispatcher = new Mock<IUiThreadDispatcher>();
    dispatcher.Setup(d => d.Dispatch(It.IsAny<Action>())).Callback<Action>(a => a());
    _vm = new MainWindowViewModel(new Mock<INativeRuntimeService>().Object, new BreadcrumbService(), _dialogs.Object,
      new Mock<IWindowService>().Object, new Mock<IWindowInputRouter>().Object, null!, dispatcher.Object, null!,
      new Mock<IPlatformWindowService>().Object, _saves, _prefs, _prompt.Object);
  }

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private MenuItemViewModel File => _vm.MainMenu.Single(m => m.Header == "File");

  private MenuItemViewModel Item(string header) => File.Items.Single(i => i.Header == header);

  [Fact]
  public void FileMenu_HasTheEntries_AndShortcuts()
  {
    Assert.Equal(["New", "Open...", "Open Recent", null, "Save", "Save As..."], File.Items.Select(i => i.Header));
    Assert.Equal("Ctrl+S", Item("Save").Gesture);
    Assert.Equal("Ctrl+Shift+S", Item("Save As...").Gesture);
    Assert.Equal("AetherVk — Untitled", _vm.WindowTitle);
  }

  [Fact]
  public async Task Save_AsksForAFolder_TheFirstTime_ThenSavesInPlace()
  {
    var dir = Path.Combine(_root, "mysave");
    _dialogs.Setup(d => d.PickFolderAsync(It.IsAny<string>())).ReturnsAsync(dir);
    await ((CommunityToolkit.Mvvm.Input.IAsyncRelayCommand)Item("Save").Command!).ExecuteAsync(null);
    Assert.True(System.IO.File.Exists(SaveIndexIo.IndexPath(dir)));
    Assert.Equal("AetherVk — mysave", _vm.WindowTitle);

    _host.Model.DiametreUm = 999;
    _vm.RefreshTitle();
    Assert.Equal("AetherVk — mysave *", _vm.WindowTitle);
    Assert.True(_vm.HasUnsavedChanges);

    await ((CommunityToolkit.Mvvm.Input.IAsyncRelayCommand)Item("Save").Command!).ExecuteAsync(null);
    _dialogs.Verify(d => d.PickFolderAsync(It.IsAny<string>()), Times.Once);
    Assert.Equal(999, SaveIndexIo.Read(dir).Index!.Model.DiametreUm);
    Assert.Equal("AetherVk — mysave", _vm.WindowTitle);
    Assert.Equal(Path.GetFullPath(dir), _prefs.RecentSaves[0]);
  }

  [Theory]
  [InlineData(UnsavedChoice.Cancel, false, false)]
  [InlineData(UnsavedChoice.Discard, true, false)]
  [InlineData(UnsavedChoice.Save, true, true)]
  public async Task New_AsksAboutUnsavedChanges(UnsavedChoice choice, bool resets, bool saves)
  {
    var dir = Path.Combine(_root, "s");
    _dialogs.Setup(d => d.PickFolderAsync(It.IsAny<string>())).ReturnsAsync(dir);
    _host.Model.DiametreUm = 500;
    _prompt.Setup(p => p.AskAsync("Untitled")).ReturnsAsync(choice);

    await ((CommunityToolkit.Mvvm.Input.IAsyncRelayCommand)Item("New").Command!).ExecuteAsync(null);
    _prompt.Verify(p => p.AskAsync("Untitled"), Times.Once);
    Assert.Equal(resets, _host.Calls.Contains("reset"));
    Assert.Equal(saves, System.IO.File.Exists(SaveIndexIo.IndexPath(dir)));
  }

  [Fact]
  public async Task NoPrompt_WhenNothingChanged_AndOpenRecentOpensTheSave()
  {
    var dir = Path.Combine(_root, "recent");
    await _saves.SaveAsync(dir);
    _vm.RefreshFileState();
    var recent = Item("Open Recent");
    var visible = recent.Items.Where(i => i.IsVisible).ToList();
    var entry = Assert.Single(visible);
    Assert.Contains(Path.GetFullPath(dir), entry.Header);

    _host.Calls.Clear();
    await ((CommunityToolkit.Mvvm.Input.IAsyncRelayCommand)entry.Command!).ExecuteAsync(null);
    _prompt.Verify(p => p.AskAsync(It.IsAny<string>()), Times.Never);
    Assert.Contains("reset", _host.Calls);
    Assert.Equal(Path.GetFullPath(dir), _saves.CurrentSaveDir);
  }

  [Fact]
  public async Task Open_RefusesAFileThatIsNotAnIndex()
  {
    _dialogs.Setup(d => d.ShowOpenFileDialogAsync(It.IsAny<string>(), It.IsAny<string[]>())).ReturnsAsync("/tmp/other.json");
    await ((CommunityToolkit.Mvvm.Input.IAsyncRelayCommand)Item("Open...").Command!).ExecuteAsync(null);
    Assert.DoesNotContain("reset", _host.Calls);
  }
}
