using System;
using System.IO;
using System.Linq;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>preferences.json and the input registry editing API (docs/savefile.md).</summary>
public sealed class PreferencesServiceTests : IDisposable
{
  private readonly string _dir = Path.Combine(Path.GetTempPath(), "avk_prefs_" + Guid.NewGuid().ToString("N"));
  private string PrefsPath => Path.Combine(_dir, "preferences.json");

  public void Dispose()
  {
    if (Directory.Exists(_dir))
      Directory.Delete(_dir, true);
  }

  private static InputRegistry Defaults()
  {
    var r = new InputRegistry();
    r.Register("Viewport", new InputChord(Key: "V"), new AppAction("viewport.switch_camera_mode"));
    r.Register("Viewport", new InputChord(Key: "Tab"), new AppAction("viewport.switch_camera_mode"));
    r.Register("Viewport", new InputChord(Key: "NumPad5"), new AppAction("viewport.toggle_projection"));
    r.CaptureDefaults();
    return r;
  }

  [Fact]
  public void MissingFile_KeepsTheDefaults()
  {
    var r = Defaults();
    var p = new PreferencesService(PrefsPath, r);
    p.Load();
    Assert.Equal("defaults", p.LoadOutcome);
    Assert.Equal(AppTheme.System, p.Theme);
    Assert.Equal(r.Defaults, r.Snapshot());
  }

  [Fact]
  public void EditedBindings_ThemeAndRecentSaves_SurviveARestart()
  {
    var r = Defaults();
    var p = new PreferencesService(PrefsPath, r);
    p.Load();
    Assert.True(r.Unregister("Viewport", new InputChord(Key: "Tab")));
    r.Register("Viewport", new InputChord(Key: "P", Ctrl: true), new AppAction("viewport.toggle_projection"));
    p.SaveBindings();
    p.SetTheme(AppTheme.Dark);
    var save = Directory.CreateDirectory(Path.Combine(_dir, "save1")).FullName;
    p.AddRecentSave(save);
    p.AddRecentSave(Directory.CreateDirectory(Path.Combine(_dir, "save2")).FullName);
    p.AddRecentSave(save); // back to the front, not duplicated

    var r2 = Defaults();
    var p2 = new PreferencesService(PrefsPath, r2);
    p2.Load();
    Assert.Equal("loaded", p2.LoadOutcome);
    Assert.Equal(AppTheme.Dark, p2.Theme);
    Assert.Equal(r.Snapshot(), r2.Snapshot());
    Assert.Null(r2.Resolve("Viewport", new InputChord(Key: "Tab")));
    Assert.Equal("viewport.toggle_projection", r2.Resolve("Viewport", new InputChord(Key: "P", Ctrl: true))?.Id);
    Assert.Equal(save, p2.RecentSaves[0]);
    Assert.Equal(2, p2.RecentSaves.Count);
    // a deleted save disappears from the list
    Directory.Delete(save);
    var p3 = new PreferencesService(PrefsPath, Defaults());
    p3.Load();
    Assert.Single(p3.RecentSaves);
  }

  [Fact]
  public void CorruptOrNewerFiles_FallBackToDefaults_AndTheCorruptOneIsKept()
  {
    Directory.CreateDirectory(_dir);
    File.WriteAllText(PrefsPath, "{ broken");
    var r = Defaults();
    var p = new PreferencesService(PrefsPath, r);
    p.Load();
    Assert.Equal("corrupt", p.LoadOutcome);
    Assert.True(File.Exists(PrefsPath + ".bad"));
    Assert.Equal(r.Defaults, r.Snapshot());

    File.WriteAllText(PrefsPath, "{\"format_version\": 9, \"theme\": \"dark\"}");
    var p2 = new PreferencesService(PrefsPath, r);
    p2.Load();
    Assert.Equal("newer", p2.LoadOutcome);
    Assert.Equal(AppTheme.System, p2.Theme);
  }

  [Fact]
  public void Registry_FindsConflicts_AndResetsAContextToItsDefaults()
  {
    var r = Defaults();
    Assert.Equal("viewport.switch_camera_mode", r.FindConflict("Viewport", new InputChord(Key: "V")));
    Assert.Null(r.FindConflict("Viewport", new InputChord(Key: "X")));
    r.Register("Other", new InputChord(Key: "X"), new AppAction("other.x"));
    r.Clear("Viewport");
    Assert.Empty(r.Snapshot().Where(b => b.Context == "Viewport"));
    r.ResetToDefaults("Viewport");
    Assert.Equal(3, r.Snapshot().Count(b => b.Context == "Viewport"));
    Assert.Equal("other.x", r.Resolve("Other", new InputChord(Key: "X"))?.Id); // untouched
  }
}
