using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Hand-written partial extension of <see cref="ISettingsTabViewModel"/>
/// for the live camera-projection parameter bindings.
/// Not localized — these labels are fixed English strings in AXAML.
/// Kept separate from the source-generator output so it survives a full
/// code-gen pass.
/// </summary>
public partial interface ISettingsTabViewModel
{
  string CameraModeName  { get; }
  bool HasActiveViewport { get; }

  System.Collections.ObjectModel.ObservableCollection<ViewportSettingsViewModel> ActiveViewports { get; }

  CometAppearanceViewModel CometAppearance { get; }

  DesktopBillboardsViewModel Billboards { get; }

  InputBindingsViewModel InputBindings { get; }

  ParticleRenderingViewModel ParticleRendering { get; }

  /// <summary>Time scale for dates: 0 = UTC, 1 = TDB (<c>TimeScaleService</c>, preferences.json).</summary>
  int TimeScaleIndex { get; set; }

  /// <summary>The time-scale help (the two scales, the offset, the leap-second table).</summary>
  string TimeScaleHelpText { get; }
}
