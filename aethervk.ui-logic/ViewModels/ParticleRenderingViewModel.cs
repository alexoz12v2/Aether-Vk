using System.Collections.Generic;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// "Particle rendering" section of the Settings tab.
///
/// <b>Volume detail</b>: the pieces a volumetric dust packet is drawn with, 16 (every piece, the
/// default: the reference image) down to 1; lower is faster and slightly coarser. Visual only.
/// An application preference (<c>preferences.json</c>, <see cref="PreferencesService.DustVolumeDetail"/>),
/// not part of a save: the app pushes the stored value to the runtime at startup, this section
/// every change at once.
/// </summary>
public partial class ParticleRenderingViewModel : ObservableObject
{
  /// <summary>Default volume detail (every piece: today's image).</summary>
  public const int VolumeDetailDefault = 16;
  public const int VolumeDetailMin = 1;
  public const int VolumeDetailMax = 16;

  /// <summary>The picker's steps, finest first.</summary>
  public static readonly IReadOnlyList<int> Steps = new[] { 16, 8, 4, 2, 1 };

  /// <summary>The largest power of two ≤ <paramref name="v"/> in [<see cref="VolumeDetailMin"/>,
  /// <see cref="VolumeDetailMax"/>] (0 and below → 1); mirror of <c>dust::clamp_volume_detail</c>.</summary>
  public static int ClampVolumeDetail(int v)
  {
    int p = VolumeDetailMin;
    while (p < VolumeDetailMax && p * 2 <= v)
      p *= 2;
    return p;
  }

  private readonly INativeRuntimeService? _runtimeService;
  private readonly PreferencesService? _preferences;

  /// <summary>Suppresses the push while the VM itself loads the stored value.</summary>
  private bool _suppressDispatch;

  /// <summary>Volume detail: 16, 8, 4, 2 or 1 (other values clamp to a step).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(VolumeDetailIndex))]
  private int _volumeDetail = VolumeDetailDefault;

  /// <summary>The picker's items (<see cref="Steps"/>).</summary>
  public IReadOnlyList<int> VolumeDetailOptions => Steps;

  /// <summary>Picker index over <see cref="Steps"/> (a negative index, a cleared picker, is ignored).</summary>
  public int VolumeDetailIndex
  {
    get
    {
      for (int i = 0; i < Steps.Count; i++)
        if (Steps[i] == VolumeDetail)
          return i;
      return 0;
    }
    set
    {
      if (value >= 0 && value < Steps.Count)
        VolumeDetail = Steps[value];
    }
  }

  public ParticleRenderingViewModel(INativeRuntimeService? runtimeService, PreferencesService? preferences)
  {
    _runtimeService = runtimeService;
    _preferences = preferences;
    // the runtime already has the stored value (App startup): no push
    _suppressDispatch = true;
    VolumeDetail = preferences?.DustVolumeDetail ?? VolumeDetailDefault;
    _suppressDispatch = false;
  }

  partial void OnVolumeDetailChanged(int value)
  {
    int d = ClampVolumeDetail(value);
    if (d != value)
    {
      VolumeDetail = d;
      return;
    }
    if (_suppressDispatch)
      return;
    _preferences?.SetDustVolumeDetail(d);
    _runtimeService?.SetDustVolumeDetail(d);
  }
}
