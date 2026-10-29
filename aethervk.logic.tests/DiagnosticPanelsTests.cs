using System;
using System.IO;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels.Debug;
using AetherVk.Logic.Tests.Mocks;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The diagnostic HUD panels (camera coordinates, dust history, observer line, orbit debug) are
/// hidden by default (reviewer report 2026-10-29: the coordinate windows in the viewport would not
/// close and meant nothing to a user). Settings turns them on for every viewport and remembers it;
/// <see cref="PreferencesService.DiagnosticPanelsEnvVar"/> = 1 forces them on for the e2e.
/// </summary>
public sealed class DiagnosticPanelsTests : IDisposable
{
  private readonly string _dir = Path.Combine(Path.GetTempPath(), "avk_diag_" + Guid.NewGuid().ToString("N"));
  private string PrefsPath => Path.Combine(_dir, "preferences.json");

  public void Dispose()
  {
    if (Directory.Exists(_dir))
      Directory.Delete(_dir, true);
  }

  private PreferencesService Prefs()
  {
    var p = new PreferencesService(PrefsPath, new InputRegistry());
    p.Load();
    return p;
  }

  [Fact]
  public void Hidden_ByDefault_AndTheChoiceSurvivesARestart()
  {
    var p = Prefs();
    Assert.False(p.ShowDiagnosticPanels);
    bool? raised = null;
    p.ShowDiagnosticPanelsChanged += v => raised = v;

    p.SetShowDiagnosticPanels(true);

    Assert.True(raised);
    Assert.True(Prefs().ShowDiagnosticPanels);
    Assert.Contains("\"show_diagnostic_panels\": true", File.ReadAllText(PrefsPath));
  }

  [Fact]
  public void TheEnvironmentVariable_ForcesThemOn()
  {
    var before = Environment.GetEnvironmentVariable(PreferencesService.DiagnosticPanelsEnvVar);
    try
    {
      Environment.SetEnvironmentVariable(PreferencesService.DiagnosticPanelsEnvVar, "1");
      var p = Prefs();
      p.SetShowDiagnosticPanels(false);
      Assert.True(p.ShowDiagnosticPanels);
    }
    finally
    {
      Environment.SetEnvironmentVariable(PreferencesService.DiagnosticPanelsEnvVar, before);
    }
  }

  [Fact]
  public void TheOrbitPanel_NeedsBothTheNucleusModeAndThePreference()
  {
    var p = Prefs();
    var vm = new CometOrbitDebugViewModel(p) { IsVisible = true };
    Assert.False(vm.IsShown);

    p.SetShowDiagnosticPanels(true);
    Assert.True(vm.IsShown);
    vm.IsVisible = false;
    Assert.False(vm.IsShown);

    // a disposed panel no longer follows the preference
    vm.IsVisible = true;
    vm.Dispose();
    p.SetShowDiagnosticPanels(false);
    Assert.True(vm.IsShown);
  }

  [Fact]
  public void TheCameraPanel_FollowsThePreference_OnTheUiThread()
  {
    var p = Prefs();
    var schedulers = new TestSchedulerProvider();
    var vm = new CameraMatrixDebugViewModel(new Mock<INativeRuntimeService>().Object, schedulers, p);
    Assert.False(vm.IsShown);

    p.SetShowDiagnosticPanels(true);
    Assert.False(vm.IsShown); // not before the UI thread runs
    schedulers.MainThread.AdvanceBy(1);
    Assert.True(vm.IsShown);

    vm.Dispose();
    p.SetShowDiagnosticPanels(false);
    schedulers.MainThread.AdvanceBy(1);
    Assert.True(vm.IsShown);
  }

  [Fact]
  public void WithoutPreferences_ThePanelsShow()
  {
    using var cam = new CameraMatrixDebugViewModel(new Mock<INativeRuntimeService>().Object, new TestSchedulerProvider());
    using var orbit = new CometOrbitDebugViewModel { IsVisible = true };
    Assert.True(cam.IsShown);
    Assert.True(orbit.IsShown);
  }
}
