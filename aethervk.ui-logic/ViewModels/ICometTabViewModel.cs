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

  // ── Rotational model ──────────────────────────────────────────────────────

  double PoleRaDeg { get; set; }
  double PoleDecDeg { get; set; }
  double PrimeMeridianDeg { get; set; }
  double PoleRaRateDegCen { get; set; }
  double PoleDecRateDegCen { get; set; }
  double RotRateDegDay { get; set; }

  // ── Commands ──────────────────────────────────────────────────────────────

  /// <summary>Executes a comet search against the JPL SBDB API.</summary>
  IAsyncRelayCommand SearchCometsCommand { get; }

  /// <summary>Fetches SPK records for the currently selected comet.</summary>
  IAsyncRelayCommand LoadSpkRecordsCommand { get; }


  /// <summary>Downloads the selected SPK file and commits it to the native runtime.</summary>
  IAsyncRelayCommand DownloadAndCommitCommand { get; }

  /// <summary>Detaches the current comet configuration and resets to defaults.</summary>
  IRelayCommand DecommitCometCommand { get; }
}
