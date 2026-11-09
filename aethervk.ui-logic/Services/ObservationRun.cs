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

  /// <summary>The header's date card as written (<c>DATE-OBS = '…'</c>), shown when
  /// <see cref="ObservationUtc"/> is null so the user sees why; null when no date card was found.</summary>
  string? DateCardText { get; }
}

/// <summary>The "no image" / "an image without a readable date" texts the observation chain shows.</summary>
public static class ObservationImageText
{
  /// <summary>Resource key: no telescope image is loaded.</summary>
  public const string NoImageKey = "Observation_NoImage";

  /// <summary>Resource key: a FITS image is loaded but its date cannot be read (<c>{Date}</c>: <see cref="DateDetail"/>).</summary>
  public const string FitsWithoutDateKey = "Observation_FitsWithoutDate";

  /// <summary>Resource key: the date keywords tried, none found.</summary>
  public const string DateKeywordsNotFoundKey = "Observation_DateKeywordsNotFound";

  /// <summary>What the image's header says about its date: the raw card, else the keywords tried
  /// (<paramref name="getString"/> resolves the resource keys).</summary>
  public static string DateDetail(IObservationImage image, Func<string, string> getString) =>
    image.DateCardText is { Length: > 0 } card ? card : getString(DateKeywordsNotFoundKey);

  /// <summary>Fills <paramref name="template"/>'s <c>{Date}</c> with <see cref="DateDetail"/>.</summary>
  public static string FormatWithoutDate(string template, IObservationImage image, Func<string, string> getString) =>
    AetherVk.Logic.Utils.NamedFormatter.Format(template,
      new System.Collections.Generic.Dictionary<string, string> { ["Date"] = DateDetail(image, getString) });

  /// <summary>The date card from header facts: the engine's raw card when it reports one, else the
  /// card the date came from with its value (a value the app could not parse), else null.</summary>
  public static string? CardText(AetherVk.Logic.Models.FitsFacts facts)
  {
    if (facts.DateCardRaw is { Length: > 0 } raw)
      return raw.IndexOf('=') >= 0 ? raw : string.Concat(facts.DateSource ?? "DATE-OBS", " = '", raw, "'");
    if (facts.DateObs is { Length: > 0 } value)
      return string.Concat(facts.DateSource ?? "DATE-OBS", " = '", value, "'");
    return null;
  }
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
