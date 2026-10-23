using System;
using System.IO;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>The Settings "Particle rendering" section: the dust volume detail and temporal reuse
/// tolerance (preferences.json).</summary>
public sealed class ParticleRenderingViewModelTests : IDisposable
{
  private readonly string _dir = Path.Combine(Path.GetTempPath(), "avk_particles_" + Guid.NewGuid().ToString("N"));
  private string PrefsPath => Path.Combine(_dir, "preferences.json");

  public void Dispose()
  {
    if (Directory.Exists(_dir))
      Directory.Delete(_dir, true);
  }

  private PreferencesService LoadPrefs()
  {
    var p = new PreferencesService(PrefsPath, new InputRegistry());
    p.Load();
    return p;
  }

  private (ParticleRenderingViewModel Vm, Mock<INativeRuntimeService> Runtime, PreferencesService Prefs) Build()
  {
    var runtime = new Mock<INativeRuntimeService>();
    var prefs = LoadPrefs();
    return (new ParticleRenderingViewModel(runtime.Object, prefs), runtime, prefs);
  }

  [Fact]
  public void ClampVolumeDetail_TakesTheLargestPowerOfTwoBelow()
  {
    Assert.Equal(1, ParticleRenderingViewModel.ClampVolumeDetail(-5));
    Assert.Equal(1, ParticleRenderingViewModel.ClampVolumeDetail(0));
    Assert.Equal(2, ParticleRenderingViewModel.ClampVolumeDetail(3));
    Assert.Equal(16, ParticleRenderingViewModel.ClampVolumeDetail(17));
    Assert.Equal(16, ParticleRenderingViewModel.ClampVolumeDetail(1000));
    foreach (int d in ParticleRenderingViewModel.Steps)
      Assert.Equal(d, ParticleRenderingViewModel.ClampVolumeDetail(d));
  }

  /// <summary>No stored value: 16 (today's image), and building the section sends nothing (the
  /// app pushes the stored value at startup).</summary>
  [Fact]
  public void VolumeDetail_DefaultsTo16_AndConstructionSendsNothing()
  {
    var (vm, runtime, prefs) = Build();
    Assert.Equal(ParticleRenderingViewModel.VolumeDetailDefault, vm.VolumeDetail);
    Assert.Equal(16, vm.VolumeDetail);
    Assert.Equal(0, vm.VolumeDetailIndex);
    Assert.Equal(16, prefs.DustVolumeDetail);
    runtime.Verify(r => r.SetDustVolumeDetail(It.IsAny<int>()), Times.Never);
  }

  [Fact]
  public void VolumeDetail_IsSentClampedAndStored()
  {
    var (vm, runtime, prefs) = Build();
    vm.VolumeDetail = 4;
    runtime.Verify(r => r.SetDustVolumeDetail(4), Times.Once);
    Assert.Equal(4, prefs.DustVolumeDetail);
    Assert.Equal(2, vm.VolumeDetailIndex);

    vm.VolumeDetail = 3;
    Assert.Equal(2, vm.VolumeDetail);
    runtime.Verify(r => r.SetDustVolumeDetail(2), Times.Once);
    runtime.Verify(r => r.SetDustVolumeDetail(3), Times.Never);
    vm.VolumeDetail = 0;
    Assert.Equal(1, vm.VolumeDetail);
    runtime.Verify(r => r.SetDustVolumeDetail(1), Times.Once);
    vm.VolumeDetail = 1000;
    Assert.Equal(16, vm.VolumeDetail);
    runtime.Verify(r => r.SetDustVolumeDetail(16), Times.Once);
    Assert.Equal(16, prefs.DustVolumeDetail);

    // the picker's index
    vm.VolumeDetailIndex = 1;
    Assert.Equal(8, vm.VolumeDetail);
    runtime.Verify(r => r.SetDustVolumeDetail(8), Times.Once);
    vm.VolumeDetailIndex = -1; // a cleared picker
    Assert.Equal(8, vm.VolumeDetail);
  }

  [Fact]
  public void VolumeDetail_SurvivesARestart()
  {
    var (vm, _, _) = Build();
    vm.VolumeDetail = 2;

    var prefs2 = LoadPrefs();
    Assert.Equal("loaded", prefs2.LoadOutcome);
    Assert.Equal(2, prefs2.DustVolumeDetail);
    var runtime2 = new Mock<INativeRuntimeService>();
    var vm2 = new ParticleRenderingViewModel(runtime2.Object, prefs2);
    Assert.Equal(2, vm2.VolumeDetail);
    Assert.Equal(3, vm2.VolumeDetailIndex);
    runtime2.Verify(r => r.SetDustVolumeDetail(It.IsAny<int>()), Times.Never);
  }

  /// <summary>A file without the key (written before the setting) means 16; a hand-edited value
  /// off the steps clamps.</summary>
  [Fact]
  public void VolumeDetail_MissingKeyMeans16_AndAnOffStepValueClamps()
  {
    Directory.CreateDirectory(_dir);
    File.WriteAllText(PrefsPath, "{\"format_version\": 1, \"theme\": \"dark\"}");
    var p = LoadPrefs();
    Assert.Equal("loaded", p.LoadOutcome);
    Assert.Equal(16, p.DustVolumeDetail);

    File.WriteAllText(PrefsPath, "{\"format_version\": 1, \"dust_volume_detail\": 5}");
    var p2 = LoadPrefs();
    Assert.Equal(4, p2.DustVolumeDetail);
    Assert.Equal(4, new ParticleRenderingViewModel(null, p2).VolumeDetail);
  }

  [Fact]
  public void ClampHazeDetail_TakesTheLargestPowerOfTwoBelow()
  {
    Assert.Equal(1, ParticleRenderingViewModel.ClampHazeDetail(-5));
    Assert.Equal(1, ParticleRenderingViewModel.ClampHazeDetail(0));
    Assert.Equal(2, ParticleRenderingViewModel.ClampHazeDetail(3));
    Assert.Equal(16, ParticleRenderingViewModel.ClampHazeDetail(17));
    Assert.Equal(16, ParticleRenderingViewModel.ClampHazeDetail(1000));
    foreach (int d in ParticleRenderingViewModel.HazeDetailSteps)
      Assert.Equal(d, ParticleRenderingViewModel.ClampHazeDetail(d));
  }

  /// <summary>No stored value: 16 (today's image), and building the section sends nothing.</summary>
  [Fact]
  public void HazeDetail_DefaultsTo16_AndConstructionSendsNothing()
  {
    var (vm, runtime, prefs) = Build();
    Assert.Equal(ParticleRenderingViewModel.HazeDetailDefault, vm.HazeDetail);
    Assert.Equal(16, vm.HazeDetail);
    Assert.Equal(0, vm.HazeDetailIndex);
    Assert.Equal(16, prefs.DustHazeDetail);
    runtime.Verify(r => r.SetDustHazeDetail(It.IsAny<int>()), Times.Never);
  }

  [Fact]
  public void HazeDetail_IsSentClampedAndStored()
  {
    var (vm, runtime, prefs) = Build();
    vm.HazeDetail = 4;
    runtime.Verify(r => r.SetDustHazeDetail(4), Times.Once);
    Assert.Equal(4, prefs.DustHazeDetail);
    Assert.Equal(2, vm.HazeDetailIndex);

    vm.HazeDetail = 3;
    Assert.Equal(2, vm.HazeDetail);
    runtime.Verify(r => r.SetDustHazeDetail(2), Times.Once);
    runtime.Verify(r => r.SetDustHazeDetail(3), Times.Never);
    vm.HazeDetail = 0;
    Assert.Equal(1, vm.HazeDetail);
    runtime.Verify(r => r.SetDustHazeDetail(1), Times.Once);
    vm.HazeDetail = 1000;
    Assert.Equal(16, vm.HazeDetail);
    runtime.Verify(r => r.SetDustHazeDetail(16), Times.Once);
    Assert.Equal(16, prefs.DustHazeDetail);
    runtime.Verify(r => r.SetDustVolumeDetail(It.IsAny<int>()), Times.Never);
    runtime.Verify(r => r.SetDustPropagateDetail(It.IsAny<int>()), Times.Never);
    runtime.Verify(r => r.SetDustReuseTolerance(It.IsAny<int>()), Times.Never);
    Assert.Equal(16, prefs.DustVolumeDetail);

    // the picker's index
    vm.HazeDetailIndex = 1;
    Assert.Equal(8, vm.HazeDetail);
    runtime.Verify(r => r.SetDustHazeDetail(8), Times.Once);
    vm.HazeDetailIndex = -1; // a cleared picker
    Assert.Equal(8, vm.HazeDetail);
  }

  [Fact]
  public void HazeDetail_SurvivesARestart()
  {
    var (vm, _, _) = Build();
    vm.HazeDetail = 2;

    var prefs2 = LoadPrefs();
    Assert.Equal("loaded", prefs2.LoadOutcome);
    Assert.Equal(2, prefs2.DustHazeDetail);
    Assert.Contains("\"dust_haze_detail\": 2", File.ReadAllText(PrefsPath));
    var runtime2 = new Mock<INativeRuntimeService>();
    var vm2 = new ParticleRenderingViewModel(runtime2.Object, prefs2);
    Assert.Equal(2, vm2.HazeDetail);
    Assert.Equal(3, vm2.HazeDetailIndex);
    runtime2.Verify(r => r.SetDustHazeDetail(It.IsAny<int>()), Times.Never);
  }

  [Fact]
  public void ClampPropagateDetail_TakesTheLargestStepBelow()
  {
    Assert.Equal(5, ParticleRenderingViewModel.ClampPropagateDetail(-5));
    Assert.Equal(5, ParticleRenderingViewModel.ClampPropagateDetail(0));
    Assert.Equal(5, ParticleRenderingViewModel.ClampPropagateDetail(8));
    Assert.Equal(9, ParticleRenderingViewModel.ClampPropagateDetail(12));
    Assert.Equal(9, ParticleRenderingViewModel.ClampPropagateDetail(16));
    Assert.Equal(17, ParticleRenderingViewModel.ClampPropagateDetail(1000));
    foreach (int d in ParticleRenderingViewModel.PropagateDetailSteps)
      Assert.Equal(d, ParticleRenderingViewModel.ClampPropagateDetail(d));
  }

  /// <summary>No stored value: 17 (today's image), and building the section sends nothing.</summary>
  [Fact]
  public void PropagateDetail_DefaultsTo17_AndConstructionSendsNothing()
  {
    var (vm, runtime, prefs) = Build();
    Assert.Equal(ParticleRenderingViewModel.PropagateDetailDefault, vm.PropagateDetail);
    Assert.Equal(17, vm.PropagateDetail);
    Assert.Equal(0, vm.PropagateDetailIndex);
    Assert.Equal(17, prefs.DustPropagateDetail);
    runtime.Verify(r => r.SetDustPropagateDetail(It.IsAny<int>()), Times.Never);
  }

  [Fact]
  public void PropagateDetail_IsSentClampedAndStored()
  {
    var (vm, runtime, prefs) = Build();
    vm.PropagateDetail = 9;
    runtime.Verify(r => r.SetDustPropagateDetail(9), Times.Once);
    Assert.Equal(9, prefs.DustPropagateDetail);
    Assert.Equal(1, vm.PropagateDetailIndex);

    vm.PropagateDetail = 7;
    Assert.Equal(5, vm.PropagateDetail);
    runtime.Verify(r => r.SetDustPropagateDetail(5), Times.Once);
    runtime.Verify(r => r.SetDustPropagateDetail(7), Times.Never);
    vm.PropagateDetail = 1000;
    Assert.Equal(17, vm.PropagateDetail);
    runtime.Verify(r => r.SetDustPropagateDetail(17), Times.Once);
    Assert.Equal(17, prefs.DustPropagateDetail);
    runtime.Verify(r => r.SetDustVolumeDetail(It.IsAny<int>()), Times.Never);
    runtime.Verify(r => r.SetDustReuseTolerance(It.IsAny<int>()), Times.Never);

    // the picker's index
    vm.PropagateDetailIndex = 2;
    Assert.Equal(5, vm.PropagateDetail);
    runtime.Verify(r => r.SetDustPropagateDetail(5), Times.Exactly(2));
    vm.PropagateDetailIndex = -1; // a cleared picker
    Assert.Equal(5, vm.PropagateDetail);
  }

  [Fact]
  public void PropagateDetail_SurvivesARestart()
  {
    var (vm, _, _) = Build();
    vm.PropagateDetail = 9;

    var prefs2 = LoadPrefs();
    Assert.Equal("loaded", prefs2.LoadOutcome);
    Assert.Equal(9, prefs2.DustPropagateDetail);
    Assert.Contains("\"dust_propagate_detail\": 9", File.ReadAllText(PrefsPath));
    var runtime2 = new Mock<INativeRuntimeService>();
    var vm2 = new ParticleRenderingViewModel(runtime2.Object, prefs2);
    Assert.Equal(9, vm2.PropagateDetail);
    Assert.Equal(1, vm2.PropagateDetailIndex);
    runtime2.Verify(r => r.SetDustPropagateDetail(It.IsAny<int>()), Times.Never);
  }

  /// <summary>A file without the key (written before the setting) means 17; a hand-edited value
  /// off the steps clamps.</summary>
  [Fact]
  public void PropagateDetail_MissingKeyMeans17_AndAnOffStepValueClamps()
  {
    Directory.CreateDirectory(_dir);
    File.WriteAllText(PrefsPath, "{\"format_version\": 1, \"theme\": \"dark\"}");
    var p = LoadPrefs();
    Assert.Equal("loaded", p.LoadOutcome);
    Assert.Equal(17, p.DustPropagateDetail);

    File.WriteAllText(PrefsPath, "{\"format_version\": 1, \"dust_propagate_detail\": 12}");
    var p2 = LoadPrefs();
    Assert.Equal(9, p2.DustPropagateDetail);
    Assert.Equal(9, new ParticleRenderingViewModel(null, p2).PropagateDetail);
  }

  [Fact]
  public void ClampReuseTolerance_KeepsOff_AndTakesTheLargestPowerOfTwoBelow()
  {
    Assert.Equal(0, ParticleRenderingViewModel.ClampReuseTolerance(-5));
    Assert.Equal(0, ParticleRenderingViewModel.ClampReuseTolerance(0));
    Assert.Equal(1, ParticleRenderingViewModel.ClampReuseTolerance(1));
    Assert.Equal(2, ParticleRenderingViewModel.ClampReuseTolerance(3));
    Assert.Equal(4, ParticleRenderingViewModel.ClampReuseTolerance(7));
    Assert.Equal(16, ParticleRenderingViewModel.ClampReuseTolerance(17));
    Assert.Equal(16, ParticleRenderingViewModel.ClampReuseTolerance(1000));
    foreach (int px in ParticleRenderingViewModel.ReuseToleranceSteps)
      Assert.Equal(px, ParticleRenderingViewModel.ClampReuseTolerance(px));
  }

  /// <summary>No stored value: off (every frame drawn fresh), and building the section sends nothing
  /// (the app pushes the stored value at startup).</summary>
  [Fact]
  public void ReuseTolerance_DefaultsToOff_AndConstructionSendsNothing()
  {
    var (vm, runtime, prefs) = Build();
    Assert.Equal(ParticleRenderingViewModel.ReuseToleranceDefault, vm.ReuseTolerance);
    Assert.Equal(0, vm.ReuseTolerance);
    Assert.Equal(0, vm.ReuseToleranceIndex);
    Assert.Equal(0, prefs.DustReuseTolerance);
    Assert.Equal(new[] { "Off", "1 px", "2 px", "4 px", "8 px", "16 px" }, vm.ReuseToleranceOptions);
    runtime.Verify(r => r.SetDustReuseTolerance(It.IsAny<int>()), Times.Never);
  }

  [Fact]
  public void ReuseTolerance_IsSentClampedAndStored()
  {
    var (vm, runtime, prefs) = Build();
    vm.ReuseTolerance = 4;
    runtime.Verify(r => r.SetDustReuseTolerance(4), Times.Once);
    Assert.Equal(4, prefs.DustReuseTolerance);
    Assert.Equal(3, vm.ReuseToleranceIndex);

    vm.ReuseTolerance = 3;
    Assert.Equal(2, vm.ReuseTolerance);
    runtime.Verify(r => r.SetDustReuseTolerance(2), Times.Once);
    runtime.Verify(r => r.SetDustReuseTolerance(3), Times.Never);
    vm.ReuseTolerance = -7;
    Assert.Equal(0, vm.ReuseTolerance);
    runtime.Verify(r => r.SetDustReuseTolerance(0), Times.Once);
    runtime.Verify(r => r.SetDustReuseTolerance(-7), Times.Never);
    vm.ReuseTolerance = 1000;
    Assert.Equal(16, vm.ReuseTolerance);
    runtime.Verify(r => r.SetDustReuseTolerance(16), Times.Once);
    Assert.Equal(16, prefs.DustReuseTolerance);

    // the picker's index
    vm.ReuseToleranceIndex = 2;
    Assert.Equal(2, vm.ReuseTolerance);
    runtime.Verify(r => r.SetDustReuseTolerance(2), Times.Exactly(2));
    vm.ReuseToleranceIndex = -1; // a cleared picker
    Assert.Equal(2, vm.ReuseTolerance);
  }

  /// <summary>The two settings are independent: changing one sends nothing for the other.</summary>
  [Fact]
  public void ReuseTolerance_AndVolumeDetail_DoNotCrossDispatch()
  {
    var (vm, runtime, prefs) = Build();
    vm.ReuseTolerance = 8;
    runtime.Verify(r => r.SetDustVolumeDetail(It.IsAny<int>()), Times.Never);
    vm.VolumeDetail = 2;
    runtime.Verify(r => r.SetDustReuseTolerance(It.IsAny<int>()), Times.Once);
    Assert.Equal(8, prefs.DustReuseTolerance);
    Assert.Equal(2, prefs.DustVolumeDetail);
  }

  [Fact]
  public void ReuseTolerance_SurvivesARestart()
  {
    var (vm, _, _) = Build();
    vm.ReuseTolerance = 8;

    var prefs2 = LoadPrefs();
    Assert.Equal("loaded", prefs2.LoadOutcome);
    Assert.Equal(8, prefs2.DustReuseTolerance);
    Assert.Contains("\"dust_reuse_px\": 8", File.ReadAllText(PrefsPath));
    var runtime2 = new Mock<INativeRuntimeService>();
    var vm2 = new ParticleRenderingViewModel(runtime2.Object, prefs2);
    Assert.Equal(8, vm2.ReuseTolerance);
    Assert.Equal(4, vm2.ReuseToleranceIndex);
    runtime2.Verify(r => r.SetDustReuseTolerance(It.IsAny<int>()), Times.Never);
  }

  /// <summary>A file without the key (written before the setting) means off; a hand-edited value
  /// off the steps clamps, a negative one is off.</summary>
  [Fact]
  public void ReuseTolerance_MissingKeyMeansOff_AndAnOffStepValueClamps()
  {
    Directory.CreateDirectory(_dir);
    File.WriteAllText(PrefsPath, "{\"format_version\": 1, \"theme\": \"dark\"}");
    var p = LoadPrefs();
    Assert.Equal("loaded", p.LoadOutcome);
    Assert.Equal(0, p.DustReuseTolerance);

    File.WriteAllText(PrefsPath, "{\"format_version\": 1, \"dust_reuse_px\": 5}");
    var p2 = LoadPrefs();
    Assert.Equal(4, p2.DustReuseTolerance);
    Assert.Equal(4, new ParticleRenderingViewModel(null, p2).ReuseTolerance);

    File.WriteAllText(PrefsPath, "{\"format_version\": 1, \"dust_reuse_px\": -3}");
    Assert.Equal(0, LoadPrefs().DustReuseTolerance);
  }

  /// <summary>The off item follows the localized label; the selection stays.</summary>
  [Fact]
  public void ReuseToleranceOffLabel_RelabelsTheFirstItem()
  {
    var (vm, runtime, _) = Build();
    vm.ReuseTolerance = 4;
    vm.ReuseToleranceOffLabel = "Disattivato";
    Assert.Equal("Disattivato", vm.ReuseToleranceOptions[0]);
    Assert.Equal("4 px", vm.ReuseToleranceOptions[3]);
    Assert.Equal(3, vm.ReuseToleranceIndex);
    runtime.Verify(r => r.SetDustReuseTolerance(It.IsAny<int>()), Times.Once);
  }
}
