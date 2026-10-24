using System.Collections.Generic;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// "Particle rendering" section of the Settings tab.
///
/// <b>Volume detail</b>: the pieces a volumetric dust packet is drawn with, 16 (every piece, the
/// default: the reference image) down to 1; lower is faster and slightly coarser. Visual only.
///
/// <b>Trajectory detail</b>: the size-polyline edges the dust propagate solves exactly, 17 (every
/// edge, the default: the reference image), 9 or 5 (the rest interpolated between the solved
/// ones; faster, the dust tails a little straighter). Visual only.
///
/// <b>Temporal reuse tolerance</b>: dust that moved less than this many pixels since it was drawn
/// is reused instead of drawn again; Off (0, the default) draws every frame fresh, 1 up to 16 px
/// are faster but can smear moving dust. Visual only.
///
/// Both are application preferences (<c>preferences.json</c>, <see cref="PreferencesService.DustVolumeDetail"/>,
/// <see cref="PreferencesService.DustReuseTolerance"/>), not part of a save: the app pushes the
/// stored values to the runtime at startup, this section every change at once.
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

  /// <summary>Default trajectory detail (every size edge solved: today's image).</summary>
  public const int PropagateDetailDefault = 17;
  public const int PropagateDetailMin = 5;
  public const int PropagateDetailMax = 17;

  /// <summary>The trajectory picker's steps, finest first (17, 9, 5 solved edges).</summary>
  public static readonly IReadOnlyList<int> PropagateDetailSteps = new[] { 17, 9, 5 };

  /// <summary>The largest step of <see cref="PropagateDetailSteps"/> ≤ <paramref name="v"/>
  /// (below the smallest → 5); mirror of <c>dust::clamp_propagate_detail</c>.</summary>
  public static int ClampPropagateDetail(int v)
  {
    foreach (int s in PropagateDetailSteps)
      if (v >= s)
        return s;
    return PropagateDetailMin;
  }

  /// <summary>Default haze detail (every sub-packet: today's image).</summary>
  public const int HazeDetailDefault = 16;
  public const int HazeDetailMin = 1;
  public const int HazeDetailMax = 16;

  /// <summary>The haze picker's steps, finest first (sub-packets per cluster at the coarsest level).</summary>
  public static readonly IReadOnlyList<int> HazeDetailSteps = new[] { 16, 8, 4, 2, 1 };

  /// <summary>The largest power of two ≤ <paramref name="v"/> in [<see cref="HazeDetailMin"/>,
  /// <see cref="HazeDetailMax"/>] (0 and below → 1); mirror of <c>dust::clamp_haze_detail</c>.</summary>
  public static int ClampHazeDetail(int v)
  {
    int p = HazeDetailMin;
    while (p < HazeDetailMax && p * 2 <= v)
      p *= 2;
    return p;
  }

  /// <summary>Default reuse tolerance: 16 px (round 9 step R; Off draws every frame fresh).</summary>
  public const int ReuseToleranceDefault = 16;
  public const int ReuseToleranceMax = 16;

  /// <summary>The reuse picker's steps in pixels, off first (more reuse: faster, coarser).</summary>
  public static readonly IReadOnlyList<int> ReuseToleranceSteps = new[] { 0, 1, 2, 4, 8, 16 };

  /// <summary>0 and below → 0 (off), otherwise the largest power of two ≤ min(<paramref name="px"/>,
  /// <see cref="ReuseToleranceMax"/>); mirror of the native clamp of
  /// <c>avkSimulationContext_setDustReuseTolerance</c>.</summary>
  public static int ClampReuseTolerance(int px)
  {
    if (px <= 0)
      return 0;
    int p = 1;
    while (p < ReuseToleranceMax && p * 2 <= px)
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

  /// <summary>Trajectory detail: 17, 9 or 5 (other values clamp to a step).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(PropagateDetailIndex))]
  private int _propagateDetail = PropagateDetailDefault;

  /// <summary>The trajectory picker's items (<see cref="PropagateDetailSteps"/>).</summary>
  public IReadOnlyList<int> PropagateDetailOptions => PropagateDetailSteps;

  /// <summary>Picker index over <see cref="PropagateDetailSteps"/> (a negative index, a cleared
  /// picker, is ignored).</summary>
  public int PropagateDetailIndex
  {
    get
    {
      for (int i = 0; i < PropagateDetailSteps.Count; i++)
        if (PropagateDetailSteps[i] == PropagateDetail)
          return i;
      return 0;
    }
    set
    {
      if (value >= 0 && value < PropagateDetailSteps.Count)
        PropagateDetail = PropagateDetailSteps[value];
    }
  }

  /// <summary>Haze detail: 16, 8, 4, 2 or 1 (other values clamp to a step).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(HazeDetailIndex))]
  private int _hazeDetail = HazeDetailDefault;

  /// <summary>The haze picker's items (<see cref="HazeDetailSteps"/>).</summary>
  public IReadOnlyList<int> HazeDetailOptions => HazeDetailSteps;

  /// <summary>Picker index over <see cref="HazeDetailSteps"/> (a negative index, a cleared picker,
  /// is ignored).</summary>
  public int HazeDetailIndex
  {
    get
    {
      for (int i = 0; i < HazeDetailSteps.Count; i++)
        if (HazeDetailSteps[i] == HazeDetail)
          return i;
      return 0;
    }
    set
    {
      if (value >= 0 && value < HazeDetailSteps.Count)
        HazeDetail = HazeDetailSteps[value];
    }
  }

  /// <summary>Temporal reuse tolerance in pixels: 0 (off), 1, 2, 4, 8 or 16 (other values clamp
  /// to a step).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ReuseToleranceIndex))]
  private int _reuseTolerance = ReuseToleranceDefault;

  /// <summary>The localized label of the off step (the Settings tab sets it on a culture change).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(ReuseToleranceOptions))]
  [NotifyPropertyChangedFor(nameof(ReuseToleranceIndex))]
  private string _reuseToleranceOffLabel = "Off";

  /// <summary>The reuse picker's items (<see cref="ReuseToleranceSteps"/>): the off label, then "N px".</summary>
  public IReadOnlyList<string> ReuseToleranceOptions
  {
    get
    {
      var labels = new string[ReuseToleranceSteps.Count];
      for (int i = 0; i < labels.Length; i++)
        labels[i] = ReuseToleranceSteps[i] == 0 ? ReuseToleranceOffLabel : $"{ReuseToleranceSteps[i]} px";
      return labels;
    }
  }

  /// <summary>Picker index over <see cref="ReuseToleranceSteps"/> (a negative index, a cleared
  /// picker, is ignored).</summary>
  public int ReuseToleranceIndex
  {
    get
    {
      for (int i = 0; i < ReuseToleranceSteps.Count; i++)
        if (ReuseToleranceSteps[i] == ReuseTolerance)
          return i;
      return 0;
    }
    set
    {
      if (value >= 0 && value < ReuseToleranceSteps.Count)
        ReuseTolerance = ReuseToleranceSteps[value];
    }
  }

  public ParticleRenderingViewModel(INativeRuntimeService? runtimeService, PreferencesService? preferences)
  {
    _runtimeService = runtimeService;
    _preferences = preferences;
    // the runtime already has the stored value (App startup): no push
    _suppressDispatch = true;
    VolumeDetail = preferences?.DustVolumeDetail ?? VolumeDetailDefault;
    PropagateDetail = preferences?.DustPropagateDetail ?? PropagateDetailDefault;
    HazeDetail = preferences?.DustHazeDetail ?? HazeDetailDefault;
    ReuseTolerance = preferences?.DustReuseTolerance ?? ReuseToleranceDefault;
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

  partial void OnPropagateDetailChanged(int value)
  {
    int d = ClampPropagateDetail(value);
    if (d != value)
    {
      PropagateDetail = d;
      return;
    }
    if (_suppressDispatch)
      return;
    _preferences?.SetDustPropagateDetail(d);
    _runtimeService?.SetDustPropagateDetail(d);
  }

  partial void OnHazeDetailChanged(int value)
  {
    int d = ClampHazeDetail(value);
    if (d != value)
    {
      HazeDetail = d;
      return;
    }
    if (_suppressDispatch)
      return;
    _preferences?.SetDustHazeDetail(d);
    _runtimeService?.SetDustHazeDetail(d);
  }

  partial void OnReuseToleranceChanged(int value)
  {
    int px = ClampReuseTolerance(value);
    if (px != value)
    {
      ReuseTolerance = px;
      return;
    }
    if (_suppressDispatch)
      return;
    _preferences?.SetDustReuseTolerance(px);
    _runtimeService?.SetDustReuseTolerance(px);
  }
}
