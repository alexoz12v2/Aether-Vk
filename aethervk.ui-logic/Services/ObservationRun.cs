using System;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

/// <summary>
/// The loaded telescope image as the observation chain sees it (reviewer feedback 2026-10-31,
/// slide 1): its observation date and its "Set up comparison". Implemented by the FITS billboard
/// row's view-model; <see cref="TimelineService.ObservationImageSource"/> finds the current one.
/// </summary>
public interface IObservationImage
{
  /// <summary>The observation epoch, UTC (the middle of the exposure, else DATE-OBS); null without one.</summary>
  DateTimeOffset? ObservationUtc { get; }

  /// <summary>A viewport is attached: the comparison can be set up.</summary>
  bool CanSetUpComparison { get; }

  /// <summary>Earth observer, seek, site, match, rotate, centre, lock; the outcome in <see cref="StatusMessage"/>.</summary>
  Task SetUpComparisonAsync();

  /// <summary>The last step's outcome.</summary>
  string StatusMessage { get; }
}

/// <summary>
/// What "Run at this date" did: the range proposed, the comet committed with it, the clock at the
/// date, the comparison set up. <see cref="Status"/> is the one line the Comet tab and the mode
/// bar show.
/// </summary>
public sealed record ObservationRunResult(bool Proposed, bool Committed, bool Seeked, bool Compared, string Status)
{
  /// <summary>No Comet tab is listening (the range entry is not built).</summary>
  public static ObservationRunResult NoHandler { get; } =
    new(false, false, false, false, "The Comet tab is not open: nothing ran.");
}

/// <summary>
/// A "Run at this date" asked through <see cref="TimelineService.RunAtObservationAsync"/> (the mode
/// bar's Model): the Comet tab's range entry runs the chain and completes it.
/// </summary>
public sealed class ObservationRunRequest
{
  private readonly TaskCompletionSource<ObservationRunResult> _tcs =
    new(TaskCreationOptions.RunContinuationsAsynchronously);

  public ObservationRunRequest(DateTimeOffset utc) => Utc = utc;

  /// <summary>The observation date, UTC.</summary>
  public DateTimeOffset Utc { get; }

  public Task<ObservationRunResult> Completion => _tcs.Task;

  public void Complete(ObservationRunResult result) => _tcs.TrySetResult(result);
}

/// <summary>
/// A "Recommit with this range" asked through <see cref="TimelineService.RequestRecommitAsync"/>:
/// the Comet tab runs its Download &amp; Commit and completes it with whether the comet is now
/// committed with the proposed range.
/// </summary>
public sealed class RecommitRequest
{
  private readonly TaskCompletionSource<bool> _tcs = new(TaskCreationOptions.RunContinuationsAsynchronously);

  public Task<bool> Completion => _tcs.Task;

  public void Complete(bool committed) => _tcs.TrySetResult(committed);
}
