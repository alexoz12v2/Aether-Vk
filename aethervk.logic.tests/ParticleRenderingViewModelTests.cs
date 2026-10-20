using System;
using System.IO;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>The Settings "Particle rendering" section: the dust volume detail (preferences.json).</summary>
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
}
