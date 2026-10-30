using System.Collections.ObjectModel;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Non-localized members of the Comet tab view-model interface.
/// The localized string members are emitted by the <c>[GenerateLocalizedStrings]</c>
/// source generator into <c>ICometTabViewModel.LocalizedStrings.g.cs</c>.
/// </summary>
public partial interface ICometTabViewModel
{
  // ── Proposed Timeline display (read-only) ─────────────────────────────────

  /// <summary>Formatted start epoch of the proposed timeline (ISO 8601 UTC).</summary>
  string ProposedStartEpoch { get; }

  /// <summary>Formatted end epoch of the proposed timeline (ISO 8601 UTC).</summary>
  string ProposedEndEpoch { get; }

  /// <summary><c>true</c> when a proposed timeline is available in the Timeline tab.</summary>
  bool HasProposedTimeline { get; }

  /// <summary>
  /// The simulated range, entered in this tab (reviewer feedback item 7): Start/End, the Model
  /// range, the dust lead time, and the lock + Recommit after a commit.
  /// </summary>
  RangeEntryViewModel Range { get; }

  // ── Guided flow (reviewer feedback item 3) ────────────────────────────────

  /// <summary>1 Search comet, 2 Pick the SPK record, 3 Set the range, 4 Commit; 5 = all done.</summary>
  int CurrentGuidedStep { get; }
  bool IsStep1Done { get; }
  bool IsStep2Done { get; }
  bool IsStep3Done { get; }
  bool IsStep4Done { get; }
  bool IsStep1Current { get; }
  bool IsStep2Current { get; }
  bool IsStep3Current { get; }
  bool IsStep4Current { get; }

  /// <summary>One-line help of the current step.</summary>
  string CurrentStepHelp { get; }

  /// <summary>Reference orbit ComboBox index (-1 = none chosen, 0 = SBDB, 1 = re-osculated).</summary>
  int ReferenceOrbitIndex { get; set; }

  /// <summary>Whether the reference orbit can be changed (no comet committed).</summary>
  bool IsReferenceOrbitSelectable { get; }

  // ── Comet search ──────────────────────────────────────────────────────────

  /// <summary>User-entered search query for the comet list.</summary>
  string SearchQuery { get; set; }

  /// <summary>Results from the JPL SBDB comet list, filtered by <see cref="SearchQuery"/>.</summary>
  ObservableCollection<CometSearchResult> FilteredSearchResults { get; }

  /// <summary>Currently selected comet in the search results list.</summary>
  CometSearchResult? SelectedComet { get; set; }

  /// <summary><c>true</c> while a SBDB comet search is in progress.</summary>
  bool IsSearching { get; }

  // ── SPK record selection ──────────────────────────────────────────────────

  /// <summary>Available SPK records for the selected comet.</summary>
  ObservableCollection<SpkRecordItem> SpkRecords { get; }

  /// <summary>Currently selected SPK record.</summary>
  SpkRecordItem? SelectedSpkRecord { get; set; }

  /// <summary><c>true</c> while SPK records are being fetched.</summary>
  bool IsLoadingSpkRecords { get; }

  // ── Commit state ──────────────────────────────────────────────────────────

  /// <summary><c>true</c> when an SPK is loaded and AlmanacPlanet is attached in the native runtime.</summary>
  bool IsAlmanacCommitted { get; }

  /// <summary>Display name of the committed comet, e.g. "67P/Churyumov-Gerasimenko".</summary>
  string CommittedCometName { get; }

  /// <summary>Status text for download / commit progress.</summary>
  string DownloadStatus { get; }

  /// <summary><c>true</c> while the SPK download or commit is in progress.</summary>
  bool IsDownloading { get; }

  /// <summary>
  /// <c>true</c> when the proposed timeline has changed since the last commit,
  /// indicating that a re-commit is needed to update the comet trajectory.
  /// </summary>
  bool HasTimelineChangedAfterCommit { get; }

  bool IsSimulationRunning { get; }

  // ── Rotational model ──────────────────────────────────────────────────────

  double PoleRaDeg { get; set; }
  double PoleDecDeg { get; set; }
  double PrimeMeridianDeg { get; set; }
  double PoleRaRateDegCen { get; set; }
  double PoleDecRateDegCen { get; set; }
  double RotRateDegDay { get; set; }
  /// <summary>Sidereal rotation period in hours (the rate seen another way).</summary>
  double RotationPeriodHours { get; set; }
  int BodyFixedOrientationIndex { get; set; }

  // ── Nucleus (feedback item 10b: the radius belongs here, not in the Model tab) ──

  /// <summary>
  /// Manual nucleus bounding-sphere radius in km, stored in <c>ModelSession.ManualNucleusRadiusKm</c>
  /// (saves unchanged). When &gt; 0 it takes precedence over the catalogue radius of the comet;
  /// the Model tab's jet button needs this or the catalogue radius &gt; 0.
  /// </summary>
  float ManualNucleusRadiusKm { get; set; }

  /// <summary>Nullable proxy for <see cref="ManualNucleusRadiusKm"/> (<c>null</c> = 0, no override).</summary>
  float? ManualNucleusRadiusKmNullable { get; set; }

  /// <summary><c>true</c> when neither a manual nor a catalogue radius is known.</summary>
  bool IsNucleusRadiusUnknown { get; }

  // ── Commands ──────────────────────────────────────────────────────────────

  /// <summary>Updates <see cref="FilteredSearchResults"/> from JPL API (async).</summary>
  IAsyncRelayCommand SearchCometsCommand { get; }

  /// <summary>Fetches SPK records for <see cref="SelectedComet"/>.</summary>
  IAsyncRelayCommand LoadSpkRecordsCommand { get; }

  /// <summary>Downloads SPK, commits it to native, and tracks comet state.</summary>
  IAsyncRelayCommand DownloadAndCommitCommand { get; }

  /// <summary>Undoes commit, detaching the SPK from the simulation.</summary>
  IRelayCommand DecommitCometCommand { get; }
  
  // ── Debug Commands ────────────────────────────────────────────────────────
  
  System.Collections.ObjectModel.ObservableCollection<JetViewModel>? DebugJets { get; }
  IRelayCommand DebugQueryCometCommand { get; }
  IRelayCommand<ulong> DebugQueryJetCommand { get; }
}
