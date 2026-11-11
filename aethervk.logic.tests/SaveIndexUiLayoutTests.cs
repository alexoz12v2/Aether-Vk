using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;
using AetherVk.Logic.Input;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The UI layout in the save index (<c>ui_layout</c>, docs/savefile.md): classic is not written, so
/// older saves stay byte-identical; compact round-trips; the saved layout wins over the preference
/// on Open, the preference applies on New; switching marks the session dirty.
/// </summary>
public sealed class SaveIndexUiLayoutTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_uilayout_" + Guid.NewGuid().ToString("N"));

  public SaveIndexUiLayoutTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private sealed class Clock : ISystemClock
  {
    public DateTimeOffset UtcNow { get; set; } = new(2026, 11, 10, 12, 0, 0, TimeSpan.Zero);
  }

  private string SaveDir => Path.Combine(_root, "save");
  private string IndexText => File.ReadAllText(SaveIndexIo.IndexPath(SaveDir));

  private PreferencesService Preferences(UiLayout last)
  {
    var p = new PreferencesService(Path.Combine(_root, "prefs", PreferencesService.FileName), new InputRegistry());
    p.SetLastUiLayout(last);
    return p;
  }

  [Fact]
  public async Task Classic_IsNotWritten_OldSavesStayByteIdentical()
  {
    // the DTO: null is not written, an older index reads back and writes out byte for byte
    string old = SaveIndexIo.ToJson(new SaveIndex { Billboards = [new BillboardDto { Id = Guid.NewGuid(), TextureKey = "/a.png" }] });
    Assert.DoesNotContain("ui_layout", old);
    var read = JsonSerializer.Deserialize(old, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Null(read.UiLayout);
    Assert.Equal(old, SaveIndexIo.ToJson(read));

    // a classic session saves no key
    var host = new FakeSaveFileHost(_root);
    await new SaveFileService(host, new Clock()).SaveAsync(SaveDir);
    Assert.DoesNotContain("ui_layout", IndexText);
  }

  [Fact]
  public async Task Compact_RoundTrips()
  {
    var host = new FakeSaveFileHost(_root) { UiLayout = UiLayoutService.ToDto(UiLayout.Compact) };
    await new SaveFileService(host, new Clock()).SaveAsync(SaveDir);
    Assert.Contains("\"ui_layout\": \"compact\"", IndexText);

    var reopened = new FakeSaveFileHost(_root);
    var r = await new SaveFileService(reopened, new Clock()).OpenAsync(SaveDir);
    Assert.True(r.Success);
    Assert.Equal("compact", reopened.UiLayout);
  }

  [Fact]
  public async Task Absent_OpensClassic()
  {
    await new SaveFileService(new FakeSaveFileHost(_root), new Clock()).SaveAsync(SaveDir);
    Assert.DoesNotContain("ui_layout", IndexText);

    var host = new FakeSaveFileHost(_root) { UiLayout = "compact" };
    var r = await new SaveFileService(host, new Clock()).OpenAsync(SaveDir);
    Assert.True(r.Success);
    Assert.Null(host.UiLayout);
    Assert.Contains("layout null", host.Calls);
  }

  [Fact]
  public async Task UnknownValue_OpensClassic()
  {
    await new SaveFileService(new FakeSaveFileHost(_root) { UiLayout = "compact" }, new Clock()).SaveAsync(SaveDir);
    // a hand-edited (or future) value
    File.WriteAllText(SaveIndexIo.IndexPath(SaveDir), IndexText.Replace("\"ui_layout\": \"compact\"", "\"ui_layout\": \"sidebar\""));
    Assert.Contains("sidebar", IndexText);

    var host = new FakeSaveFileHost(_root) { UiLayout = "compact" };
    var r = await new SaveFileService(host, new Clock()).OpenAsync(SaveDir);
    Assert.True(r.Success);
    Assert.Null(host.UiLayout);

    Assert.Equal(UiLayout.Compact, UiLayoutService.FromDto("COMPACT"));
    Assert.Equal(UiLayout.Classic, UiLayoutService.FromDto("sidebar"));
    Assert.Equal(UiLayout.Classic, UiLayoutService.FromDto(null));
    Assert.Equal(UiLayout.Classic, UiLayoutService.FromDto(""));
    Assert.Null(UiLayoutService.ToDto(UiLayout.Classic));
    Assert.Equal("compact", UiLayoutService.ToDto(UiLayout.Compact));
  }

  [Fact]
  public async Task SaveInPlace_AfterSwitchToClassic_DropsTheKey()
  {
    var host = new FakeSaveFileHost(_root) { UiLayout = "compact" };
    var svc = new SaveFileService(host, new Clock());
    await svc.SaveAsync(SaveDir);
    Assert.Contains("ui_layout", IndexText);

    host.UiLayout = null;
    await svc.SaveAsync(SaveDir); // in place: the open index is reused
    Assert.DoesNotContain("ui_layout", IndexText);
    Assert.Null(svc.CurrentIndex!.UiLayout);

    // a fresh service saving over the folder starts from the index already there
    host.UiLayout = "compact";
    await new SaveFileService(host, new Clock()).SaveAsync(SaveDir);
    host.UiLayout = null;
    await new SaveFileService(host, new Clock()).SaveAsync(SaveDir);
    Assert.DoesNotContain("ui_layout", IndexText);
  }

  [Fact]
  public async Task SwitchingLayout_MarksDirty()
  {
    var host = new FakeSaveFileHost(_root);
    var svc = new SaveFileService(host, new Clock());
    await svc.SaveAsync(SaveDir);
    Assert.False(svc.IsDirty);
    host.UiLayout = "compact";
    Assert.True(svc.IsDirty);
    await svc.SaveAsync(SaveDir);
    Assert.False(svc.IsDirty);
    host.UiLayout = null;
    Assert.True(svc.IsDirty);

    // the classic fingerprint is the one older builds took (no key)
    var snap = host.Capture();
    Assert.DoesNotContain("ui_layout", SaveFileService.Fingerprint(snap));
    Assert.Contains("ui_layout", SaveFileService.Fingerprint(snap with { UiLayout = "compact" }));
  }

  [Fact]
  public async Task New_AppliesThePreferredLayout_WithoutMarkingDirty()
  {
    var host = new FakeSaveFileHost(_root);
    var svc = new SaveFileService(host, new Clock(), Preferences(UiLayout.Compact));
    Assert.Null(host.UiLayout);
    await svc.NewAsync();
    Assert.Equal("compact", host.UiLayout);
    Assert.False(svc.IsDirty);

    // and back: a classic preference switches a compact session to classic on New
    var classic = new FakeSaveFileHost(_root) { UiLayout = "compact" };
    var svc2 = new SaveFileService(classic, new Clock(), Preferences(UiLayout.Classic));
    await svc2.NewAsync();
    Assert.Null(classic.UiLayout);
    Assert.False(svc2.IsDirty);
  }

  [Fact]
  public async Task Open_AppliesTheSavedLayout_OverThePreference()
  {
    await new SaveFileService(new FakeSaveFileHost(_root) { UiLayout = "compact" }, new Clock()).SaveAsync(SaveDir);
    var host = new FakeSaveFileHost(_root);
    var svc = new SaveFileService(host, new Clock(), Preferences(UiLayout.Classic));
    var r = await svc.OpenAsync(SaveDir);
    Assert.True(r.Success);
    Assert.Equal("compact", host.UiLayout);
    Assert.False(svc.IsDirty);

    // the reverse: a classic save opens classic under a compact preference
    var classicDir = Path.Combine(_root, "classic");
    await new SaveFileService(new FakeSaveFileHost(_root), new Clock()).SaveAsync(classicDir);
    var host2 = new FakeSaveFileHost(_root);
    var svc2 = new SaveFileService(host2, new Clock(), Preferences(UiLayout.Compact));
    await svc2.NewAsync();
    Assert.Equal("compact", host2.UiLayout);
    Assert.True((await svc2.OpenAsync(classicDir)).Success);
    Assert.Null(host2.UiLayout);
    Assert.False(svc2.IsDirty);
  }

  [Fact]
  public void Service_RaisesChanged_OnlyOnAChange()
  {
    var svc = new UiLayoutService();
    var seen = new List<UiLayout>();
    svc.Changed += (_, layout) => seen.Add(layout);
    Assert.Equal(UiLayout.Classic, svc.Current);
    svc.Set(UiLayout.Classic);
    svc.Set(UiLayout.Compact);
    svc.Set(UiLayout.Compact);
    svc.Set(UiLayout.Classic);
    Assert.Equal(new[] { UiLayout.Compact, UiLayout.Classic }, seen);
    Assert.Equal(UiLayout.Classic, svc.Current);
  }
}

/// <summary>The last chosen UI layout in preferences.json (<c>ui_layout</c>).</summary>
public sealed class PreferencesUiLayoutTests : IDisposable
{
  private readonly string _dir = Path.Combine(Path.GetTempPath(), "avk_prefs_layout_" + Guid.NewGuid().ToString("N"));
  private string PrefsPath => Path.Combine(_dir, PreferencesService.FileName);

  public void Dispose()
  {
    if (Directory.Exists(_dir))
      Directory.Delete(_dir, true);
  }

  [Fact]
  public void LastUiLayout_PersistsAcrossInstances()
  {
    var p = new PreferencesService(PrefsPath, new InputRegistry());
    p.Load();
    Assert.Equal(UiLayout.Classic, p.LastUiLayout);
    p.SetLastUiLayout(UiLayout.Compact);
    Assert.Equal(UiLayout.Compact, p.LastUiLayout);
    Assert.Contains("\"ui_layout\": \"compact\"", File.ReadAllText(PrefsPath));

    var p2 = new PreferencesService(PrefsPath, new InputRegistry());
    p2.Load();
    Assert.Equal(UiLayout.Compact, p2.LastUiLayout);
    p2.SetLastUiLayout(UiLayout.Classic);

    var p3 = new PreferencesService(PrefsPath, new InputRegistry());
    p3.Load();
    Assert.Equal(UiLayout.Classic, p3.LastUiLayout);
  }
}
