using System;
using System.IO;
using System.Linq;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>The Settings "Input bindings" editor (docs/savefile.md, preferences).</summary>
public sealed class InputBindingsViewModelTests : IDisposable
{
  private readonly string _dir = Path.Combine(Path.GetTempPath(), "avk_bindings_" + Guid.NewGuid().ToString("N"));

  public void Dispose()
  {
    if (Directory.Exists(_dir))
      Directory.Delete(_dir, true);
  }

  private (InputBindingsViewModel Vm, InputRegistry Registry, PreferencesService Prefs) Build()
  {
    var r = new InputRegistry();
    r.Register("Viewport", new InputChord(Key: "V"), new AppAction("viewport.switch_camera_mode"));
    r.Register("Viewport", new InputChord(Key: "Tab"), new AppAction("viewport.switch_camera_mode"));
    r.Register("Viewport", new InputChord(Key: "NumPad5"), new AppAction("viewport.toggle_projection"));
    r.CaptureDefaults();
    var p = new PreferencesService(Path.Combine(_dir, "preferences.json"), r);
    p.Load();
    return (new InputBindingsViewModel(r, p), r, p);
  }

  [Fact]
  public void ListsEveryActionWithItsChords()
  {
    var (vm, _, _) = Build();
    var ctx = Assert.Single(vm.Contexts);
    Assert.Equal("Viewport", ctx.Context);
    var cycle = ctx.Actions.Single(a => a.ActionId == "viewport.switch_camera_mode");
    Assert.Equal("Switch camera mode", cycle.Name);
    Assert.Equal(["Tab", "V"], cycle.Chords.Select(c => c.Text).OrderBy(t => t));
  }

  [Fact]
  public void Rebind_ReplacesTheChord_IsLiveAndPersisted()
  {
    var (vm, r, _) = Build();
    var v = vm.Contexts[0].Actions.Single(a => a.ActionId == "viewport.switch_camera_mode").Chords.Single(c => c.Text == "V");
    v.RebindCommand.Execute(null);
    Assert.True(vm.IsCapturing);
    Assert.True(vm.SubmitCapturedChord(new InputChord(Key: "C", Shift: true)));
    Assert.False(vm.IsCapturing);
    Assert.Null(r.Resolve("Viewport", new InputChord(Key: "V")));
    Assert.Equal("viewport.switch_camera_mode", r.Resolve("Viewport", new InputChord(Key: "C", Shift: true))?.Id);
    // persisted: a fresh registry loading the file gets the same table
    var r2 = new InputRegistry();
    var p2 = new PreferencesService(Path.Combine(_dir, "preferences.json"), r2);
    p2.Load();
    Assert.Equal(r.Snapshot(), r2.Snapshot());
  }

  [Fact]
  public void ConflictingChord_IsRefused_AndEscCancels()
  {
    var (vm, r, _) = Build();
    var before = r.Snapshot();
    vm.Contexts[0].Actions.Single(a => a.ActionId == "viewport.toggle_projection").AddChordCommand.Execute(null);
    Assert.False(vm.SubmitCapturedChord(new InputChord(Key: "V")));
    Assert.Contains("Switch camera mode", vm.ConflictMessage);
    Assert.Equal(before, r.Snapshot());
    vm.Contexts[0].Actions[0].AddChordCommand.Execute(null);
    vm.CancelCapture();
    Assert.False(vm.SubmitCapturedChord(new InputChord(Key: "Q")));
    Assert.Equal(before, r.Snapshot());
  }

  [Fact]
  public void RemoveAndReset()
  {
    var (vm, r, _) = Build();
    vm.Contexts[0].Actions.Single(a => a.ActionId == "viewport.toggle_projection").Chords[0].RemoveCommand.Execute(null);
    Assert.Null(r.Resolve("Viewport", new InputChord(Key: "NumPad5")));
    // the action stays listed (from the defaults) with no chord, ready for Add chord
    Assert.Empty(vm.Contexts[0].Actions.Single(a => a.ActionId == "viewport.toggle_projection").Chords);
    vm.Contexts[0].ResetCommand.Execute(null);
    Assert.Equal(r.Defaults, r.Snapshot());
  }
}
