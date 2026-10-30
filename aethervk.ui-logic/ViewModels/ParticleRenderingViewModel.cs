using System;
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
/// is reused instead of drawn again; Off (0) draws every frame fresh, 1 up to 16 px (the default)
/// are faster but can smear moving dust. Visual only.
///
/// <b>Playback preview quality</b>: how the dust is drawn while playing and for 2 s after a pause
/// or a seek (Off, Smooth, Balanced — the default —, Quality): coarser, without temporal reuse, with
/// the flow marks moving out of the nucleus; the full render follows. Not part of the presets.
///
/// <b>Quality</b>: five presets (<see cref="Presets"/>) over those four values, 1 (lowest, where the
/// app starts) to 5 (ultra); editing a value by hand turns it to custom (0), unless the values match
/// a preset again.
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

  /// <summary>Default reuse tolerance: 16 px (Off draws every frame fresh).</summary>
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

  /// <summary>One quality preset: the four dust values it sets.</summary>
  public sealed record QualityPreset(string Name, int VolumeDetail, int PropagateDetail, int HazeDetail, int ReuseTolerance);

  public const int QualityCustom = 0;
  public const int QualityMin = 1;
  public const int QualityMax = 5;

  /// <summary>Playback preview levels: 0 Off, 1 Smooth, 2 Balanced (the default), 3 Quality;
  /// mirror of <c>dust::play_quality::PlayPreview</c>.</summary>
  public const int PlayPreviewDefault = 2;
  public const int PlayPreviewMax = 3;

  /// <summary>A level in [0, <see cref="PlayPreviewMax"/>], else the default (the native clamp).</summary>
  public static int ClampPlayPreview(int level) => level is >= 0 and <= PlayPreviewMax ? level : PlayPreviewDefault;

  /// <summary>Playback preview level, the picker's index (<see cref="PlayPreviewOptions"/>).</summary>
  [ObservableProperty]
  private int _playPreview = PlayPreviewDefault;

  /// <summary>The localized picker items, Off … Quality (the Settings tab sets them on a culture change).</summary>
  [ObservableProperty]
  private IReadOnlyList<string> _playPreviewOptions = new[] { "Off", "Smooth", "Balanced", "Quality" };
  /// <summary>The app starts on the lowest preset (missing preference).</summary>
  public const int QualityDefault = QualityMin;

  /// <summary>Presets 1 … 5, lowest first.</summary>
  public static readonly IReadOnlyList<QualityPreset> Presets = new QualityPreset[]
  {
    new("Lowest", 1, 5, 1, 16),
    new("Low", 2, 5, 2, 16),
    new("Medium", 4, 9, 4, 8),
    new("High", 8, 17, 8, 4),
    new("Ultra", 16, 17, 16, 0),
  };

  /// <summary>The preset of level <paramref name="quality"/> (1 … 5), <c>null</c> for custom.</summary>
  public static QualityPreset? PresetFor(int quality) =>
    quality >= QualityMin && quality <= QualityMax ? Presets[quality - 1] : null;

  /// <summary>The level whose values are these, else <see cref="QualityCustom"/>.</summary>
  public static int MatchQuality(int volume, int propagate, int haze, int reuse)
  {
    for (int i = 0; i < Presets.Count; i++)
    {
      var p = Presets[i];
      if (p.VolumeDetail == volume && p.PropagateDetail == propagate && p.HazeDetail == haze && p.ReuseTolerance == reuse)
        return i + 1;
    }
    return QualityCustom;
  }

  private readonly INativeRuntimeService? _runtimeService;
  private readonly PreferencesService? _preferences;

  /// <summary>Suppresses the push while the VM itself loads the stored value.</summary>
  private bool _suppressDispatch;

  /// <summary>Set while a preset writes the four values (they must not turn it to custom).</summary>
  private bool _applyingPreset;

  /// <summary>The quality level: 1 … 5, or 0 (custom).</summary>
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(QualityLabel))]
  [NotifyPropertyChangedFor(nameof(QualitySliderValue))]
  private int _quality = QualityDefault;

  private int _lastPresetLevel = QualityDefault;

  /// <summary>The slider's position, 1 … 5: the level, or the last one while custom.</summary>
  public double QualitySliderValue
  {
    get => Quality == QualityCustom ? _lastPresetLevel : Quality;
    set
    {
      int level = (int)Math.Round(value);
      if (level >= QualityMin && level <= QualityMax && level != Quality)
        Quality = level;
    }
  }

  /// <summary>"Lowest" … "Ultra", or "Custom".</summary>
  public string QualityLabel => PresetFor(Quality)?.Name ?? "Custom";

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
    // without preferences (designer) the start level's values
    var start = PresetFor(QualityDefault)!;
    VolumeDetail = preferences?.DustVolumeDetail ?? start.VolumeDetail;
    PropagateDetail = preferences?.DustPropagateDetail ?? start.PropagateDetail;
    HazeDetail = preferences?.DustHazeDetail ?? start.HazeDetail;
    ReuseTolerance = preferences?.DustReuseTolerance ?? start.ReuseTolerance;
    Quality = preferences?.ParticleQuality ?? QualityDefault;
    PlayPreview = preferences?.DustPlayPreview ?? PlayPreviewDefault;
    if (Quality != QualityCustom)
      _lastPresetLevel = Quality;
    _suppressDispatch = false;
  }

  partial void OnQualityChanged(int value)
  {
    if (_suppressDispatch)
      return;
    _preferences?.SetParticleQuality(value);
    if (PresetFor(value) is not { } p)
      return;
    _lastPresetLevel = value;
    _applyingPreset = true;
    try
    {
      VolumeDetail = p.VolumeDetail;
      PropagateDetail = p.PropagateDetail;
      HazeDetail = p.HazeDetail;
      ReuseTolerance = p.ReuseTolerance;
    }
    finally
    {
      _applyingPreset = false;
    }
  }

  /// <summary>A value edited by hand: the matching preset, else custom.</summary>
  private void OnValueEdited()
  {
    if (_applyingPreset)
      return;
    Quality = MatchQuality(VolumeDetail, PropagateDetail, HazeDetail, ReuseTolerance);
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
    OnValueEdited();
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
    OnValueEdited();
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
    OnValueEdited();
  }

  partial void OnPlayPreviewChanged(int value)
  {
    // a cleared picker (-1) or an unknown level: the native clamp's value
    int level = ClampPlayPreview(value);
    if (level != value)
    {
      PlayPreview = level;
      return;
    }
    if (_suppressDispatch)
      return;
    _preferences?.SetDustPlayPreview(level);
    _runtimeService?.SetDustPlayPreview(level);
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
    OnValueEdited();
  }
}
