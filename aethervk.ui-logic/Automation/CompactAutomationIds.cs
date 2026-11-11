using System;

namespace AetherVk.Logic.Automation
{
  /// <summary>
  /// The AutomationIds of the compact panel and of the nucleus mini-view, one constant per id. The
  /// views set them with <c>x:Static</c> and the UI tests read them from this same file (it is linked
  /// as source there, so it depends on nothing but System): an id is renamed in one place only.
  /// </summary>
  public static class CompactAutomationIds
  {
    public const string CompactPanel = "CompactPanel";

    public const string CometCard = "Compact.Comet.Card";
    public const string CometPreset67P = "Compact.Comet.Preset67P";
    public const string CometPreset10P = "Compact.Comet.Preset10P";
    public const string CometSearch = "Compact.Comet.Search";
    public const string CometCommit = "Compact.Comet.Commit";

    public const string PeriodImageDate = "Compact.Period.ImageDate";
    public const string PeriodRotationPeriodHours = "Compact.Period.RotationPeriodHours";
    public const string PeriodRotations = "Compact.Period.Rotations";
    public const string PeriodFromImage = "Compact.Period.FromImage";
    public const string PeriodApply = "Compact.Period.Apply";
    public const string PeriodPreview = "Compact.Period.Preview";

    public const string NucleusRadiusKm = "Compact.Nucleus.RadiusKm";
    public const string NucleusPoleRa = "Compact.Nucleus.PoleRa";
    public const string NucleusPoleDec = "Compact.Nucleus.PoleDec";

    public const string JetsList = "Compact.Jets.List";
    public const string JetsAdd = "Compact.Jets.Add";
    public const string JetsRemove = "Compact.Jets.Remove";
    public const string JetLatitude = "Compact.Jet.Latitude";
    public const string JetLongitude = "Compact.Jet.Longitude";
    public const string JetAperture = "Compact.Jet.Aperture";
    public const string JetVelocity = "Compact.Jet.Velocity";
    public const string JetLifetime = "Compact.Jet.Lifetime";
    public const string JetEmissionSteps = "Compact.Jet.EmissionSteps";

    public const string DustMinDiameter = "Compact.Dust.MinDiameter";
    public const string DustMaxDiameter = "Compact.Dust.MaxDiameter";
    public const string DustSizeQ = "Compact.Dust.SizeQ";
    public const string DustDensity = "Compact.Dust.Density";
    public const string DustAlbedo = "Compact.Dust.Albedo";
    public const string DustBeta = "Compact.Dust.Beta";
    public const string DustCopyToAllJets = "Compact.Dust.CopyToAllJets";

    public const string FitsImportAndOverlay = "Compact.Fits.ImportAndOverlay";
    public const string FitsList = "Compact.Fits.List";
    public const string FitsVisible = "Compact.Fits.Visible";
    public const string FitsOpacity = "Compact.Fits.Opacity";
    public const string FitsPlane = "Compact.Fits.Plane";

    public const string ViewOrbit = "Compact.View.Orbit";
    public const string ViewNucleus = "Compact.View.Nucleus";
    public const string ViewModel = "Compact.View.Model";
    public const string ViewDustFlow = "Compact.View.DustFlow";

    public const string PlaybackPlayPause = "Compact.Playback.PlayPause";
    public const string PlaybackSpeed = "Compact.Playback.Speed";
    public const string PlaybackReset = "Compact.Playback.Reset";
    public const string PlaybackRunToEnd = "Compact.Playback.RunToEnd";
    public const string PlaybackEpoch = "Compact.Playback.Epoch";
    public const string PlaybackSeek = "Compact.Playback.Seek";

    public const string ExportButton = "Compact.Export.Button";
    public const string ExportStatus = "Compact.Export.Status";

    public const string MoreSettings = "Compact.MoreSettings";

    public const string NucleusMiniViewExpander = "NucleusMiniViewExpander";
    public const string NucleusMiniViewHost = "NucleusMiniViewHost";
    public const string NucleusMiniViewResetToEarth = "NucleusMiniView.ResetToEarth";
    public const string NucleusMiniViewCaption = "NucleusMiniView.Caption";

    /// <summary>Every id above.</summary>
    public static readonly string[] All =
    {
      CompactPanel,
      CometCard,
      CometPreset67P,
      CometPreset10P,
      CometSearch,
      CometCommit,
      PeriodImageDate,
      PeriodRotationPeriodHours,
      PeriodRotations,
      PeriodFromImage,
      PeriodApply,
      PeriodPreview,
      NucleusRadiusKm,
      NucleusPoleRa,
      NucleusPoleDec,
      JetsList,
      JetsAdd,
      JetsRemove,
      JetLatitude,
      JetLongitude,
      JetAperture,
      JetVelocity,
      JetLifetime,
      JetEmissionSteps,
      DustMinDiameter,
      DustMaxDiameter,
      DustSizeQ,
      DustDensity,
      DustAlbedo,
      DustBeta,
      DustCopyToAllJets,
      FitsImportAndOverlay,
      FitsList,
      FitsVisible,
      FitsOpacity,
      FitsPlane,
      ViewOrbit,
      ViewNucleus,
      ViewModel,
      ViewDustFlow,
      PlaybackPlayPause,
      PlaybackSpeed,
      PlaybackReset,
      PlaybackRunToEnd,
      PlaybackEpoch,
      PlaybackSeek,
      ExportButton,
      ExportStatus,
      MoreSettings,
      NucleusMiniViewExpander,
      NucleusMiniViewHost,
      NucleusMiniViewResetToEarth,
      NucleusMiniViewCaption,
    };
  }
}
