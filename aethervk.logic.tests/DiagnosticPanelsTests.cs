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

  /// <summary>The ephemeris is not a diagnostic: with the panels off the camera panel still shows
  /// once a comet is committed, with its diagnostic sections hidden.</summary>
  [Fact]
  public void TheEphemeris_ShowsWithTheDiagnosticsOff_OnceACometIsCommitted()
  {
    var p = Prefs();
    var schedulers = new TestSchedulerProvider();
    var runtime = new Mock<INativeRuntimeService>();
    using var vm = new CameraMatrixDebugViewModel(runtime.Object, schedulers, p);
    void Poll()
    {
      schedulers.Background.AdvanceBy(TimeSpan.FromMilliseconds(500).Ticks);
      schedulers.MainThread.AdvanceBy(1);
    }
    Poll();
    Assert.False(vm.IsShown);

    runtime.Setup(r => r.CometEphemeris()).Returns(new CometEphemeris(
      new DateTimeOffset(2021, 12, 27, 0, 0, 0, TimeSpan.Zero), 2.36, 0.49, 1.388, 0.466, 0.43, 2.51, 4.74, 232.7));
    Poll();
    Assert.True(vm.HasEphemeris);
    Assert.True(vm.IsShown);
    Assert.False(vm.ShowDiagnostics);

    runtime.Setup(r => r.CometEphemeris()).Returns((CometEphemeris?)null);
    Poll();
    Assert.False(vm.IsShown);
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
