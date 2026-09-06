using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Non-localized members of the Model tab view-model interface.
/// The localized string members are emitted by the <c>[GenerateLocalizedStrings]</c>
/// source generator into <c>IModelTabViewModel.LocalizedStrings.g.cs</c>.
/// </summary>
public partial interface IModelTabViewModel : ICursorWarpingViewModel
{
  /// <summary>Live collection of dust emission jets managed for this scene.</summary>
  ObservableCollection<JetViewModel>? Jets { get; }

  /// <summary>The jet currently selected in the list, or <c>null</c> when none is selected.</summary>
  JetViewModel? SelectedJet { get; set; }

  /// <summary>Adds a new jet with random physical defaults and registers it with the native runtime.</summary>
  IRelayCommand AddJetCommand { get; }

  /// <summary>Removes the given jet from the list.</summary>
  IRelayCommand<JetViewModel?> RemoveJetCommand { get; }

  /// <summary>
  /// Manual override for nucleus bounding-sphere radius in km.
  /// When &gt; 0, takes precedence over the Horizon-fetched value from CometSession.
  /// <c>AddJetCommand</c> is disabled until a comet is committed AND this or the Horizon radius is &gt; 0.
  /// </summary>
  float ManualNucleusRadiusKm { get; set; }

  /// <summary>
  /// Nullable proxy for <see cref="ManualNucleusRadiusKm"/>.
  /// <c>null</c> = no manual override set (shows watermark in NumericUpDown).
  /// </summary>
  float? ManualNucleusRadiusKmNullable { get; set; }

  /// <summary><c>true</c> when no nucleus radius is available — shows the warning hint.</summary>
  bool IsNucleusRadiusUnknown { get; }

  /// <summary>
  /// <c>true</c> when a comet has been committed to the native runtime.
  /// <c>AddJetCommand</c> is disabled until this is <c>true</c>.
  /// Bind to this to show a "Commit a comet first" hint in the view.
  /// </summary>
  bool IsCometCommitted { get; }

  /// <summary>
  /// When true, enables the legacy Sun and Nucleus expanders for debugging purposes.
  /// </summary>
  bool EnableLegacyExpanders { get; set; }

  /// <summary>
  /// The current model session.
  /// </summary>
  AetherVk.Logic.Services.ModelSession? CurrentSession { get; }
}
