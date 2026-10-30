using System.Globalization;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Services;

/// <summary>
/// Creates the model session's jets in the native runtime. Idempotent: jets that already have a
/// native particle system are skipped, so the Model tab (on commit) and a savefile open can both
/// call it on the UI thread without creating a jet twice.
/// </summary>
public static class JetNativeSync
{
  /// <summary>
  /// The native grain and activity parameters of <paramref name="j"/>: its own grain and Q_d₀, and
  /// the model-wide shape of the Q_d law (exponent, cutoff, max) from <paramref name="s"/>.
  /// </summary>
  public static ParticleSystemModel BuildModel(ModelSession s, JetViewModel j) => new(
    DiametreUm: j.DiametreUm,
    DensityGCm3: j.DensityGCm3,
    ScatteringEfficiency: j.ScatteringEfficiency,
    GeometricAlbedo: j.GeometricAlbedo,
    SizeQ: j.SizeQ,
    SizeRange: j.SizeRange,
    DustRate1AuKgs: j.DustRate1AuKgs,
    DustRatePower: s.DustRatePower,
    DustRateCutoffAu: s.DustRateCutoffAu,
    DustRateMaxKgs: s.DustRateMaxKgs);

  public static ParticleSystemJet BuildJet(JetViewModel j, float nucleusRadiusKm, long emissionStartUs = long.MaxValue) => new(
    LatitudeRad: j.LatitudeRad,
    LongitudeRad: j.LongitudeRad,
    ApertureRad: j.ApertureRad,
    StartVelocityMean: j.StartVelocityMeanMs,
    StartVelocityStd: j.StartVelocityStdMs,
    StreamColor: j.StreamColor,
    NucleusRadiusKm: nucleusRadiusKm,
    Seed: j.Seed,
    EmissionStartUs: emissionStartUs);

  /// <summary>The radius jets are emitted from: the Model tab's manual one, else the comet's.</summary>
  public static float EffectiveNucleusRadiusKm(ModelSession model, CometSession? comet) =>
    model.ManualNucleusRadiusKm > 0f ? model.ManualNucleusRadiusKm : comet?.NucleusRadiusKm ?? 0f;

  /// <summary>Shown on a jet the core refused (<see cref="JetViewModel.NativeError"/>).</summary>
  public const string RefusedText =
    "Not in the simulation: the core refused this jet (the log has the reason). Retry, or commit again.";

  /// <summary>
  /// English text of <see cref="JetViewModel.LitWarning"/> ({0} / {1}: the committed range ends);
  /// the Model tab passes the localized <c>Tabs_Model_JetNeverLit</c> instead.
  /// </summary>
  public const string NeverLitFormat =
    "This site is never lit between {0} and {1}: the jet emits no dust. Move it to another latitude, or change the range.";

  /// <summary>
  /// Refreshes <see cref="JetViewModel.LitWarning"/> from the core's lit fraction of the site
  /// over the committed range (<see cref="INativeRuntimeService.GetJetLitFraction"/>): set when
  /// the site is never lit, cleared when it is lit, not in the simulation, or the core has no
  /// answer (no committed comet). <paramref name="format"/>: the localized text, else
  /// <see cref="NeverLitFormat"/>; the dates in <paramref name="timeScale"/>'s scale (null: TDB).
  /// Not per frame: the core steps the ephemeris for it.
  /// </summary>
  public static void RefreshLitWarning(INativeRuntimeService runtime, JetViewModel jet, string? format = null, TimeScaleService? timeScale = null)
  {
    if (jet.NativePsId == 0
        || !runtime.GetJetLitFraction(jet.NativePsId, out double litFraction, out var range)
        || litFraction > 0)
    {
      jet.LitWarning = null;
      return;
    }
    string f = string.IsNullOrEmpty(format) ? NeverLitFormat : format!;
    jet.LitWarning = string.Format(CultureInfo.InvariantCulture, f,
      LitWarningEpoch(range.StartCenturies, range.StartNs, timeScale),
      LitWarningEpoch(range.EndCenturies, range.EndNs, timeScale));
  }

  /// <summary>A range end of the lit warning: "yyyy-MM-dd HH:mm:ss (UTC)" in the chosen scale.</summary>
  private static string LitWarningEpoch(short c, ulong ns, TimeScaleService? timeScale) =>
    timeScale is null
      ? TimelineTabViewModel.FormatEpoch(c, ns)
      : timeScale.Format(c, ns, TimelineTabViewModel.EpochBoxFormat) + " (" + timeScale.ScaleName + ")";

  /// <summary><see cref="RefreshLitWarning"/> on every jet of <paramref name="session"/>.</summary>
  public static void RefreshLitWarnings(INativeRuntimeService runtime, ModelSession session, string? format = null, TimeScaleService? timeScale = null)
  {
    foreach (var jet in session.Jets)
      RefreshLitWarning(runtime, jet, format, timeScale);
  }

  /// <summary>
  /// Adds every jet of <paramref name="session"/> without a native system (the comet must be
  /// committed). The first system of the scene goes through <c>AddFirstParticleSystem</c>.
  /// Returns how many jets the core refused; each keeps <see cref="JetViewModel.NativeError"/>.
  /// <paramref name="emissionStartUs"/>: the ignition (<see cref="TimelineService.IgnitionUs()"/>).
  /// Every jet's <see cref="JetViewModel.LitWarning"/> is refreshed (the committed range may be new).
  /// </summary>
  public static int EnsureNative(INativeRuntimeService runtime, ModelSession session, float nucleusRadiusKm,
    long emissionStartUs = long.MaxValue, string? litWarningFormat = null, TimeScaleService? timeScale = null)
  {
    bool any = false;
    foreach (var jet in session.Jets)
      any |= jet.NativePsId != 0;
    int refused = 0;
    foreach (var jet in session.Jets)
    {
      if (jet.NativePsId != 0) continue;
      AddNative(runtime, BuildModel(session, jet), BuildJet(jet, nucleusRadiusKm, emissionStartUs), jet, isFirst: !any);
      any |= jet.NativePsId != 0;
      if (jet.NativePsId == 0) refused++;
    }
    RefreshLitWarnings(runtime, session, litWarningFormat, timeScale);
    return refused;
  }

  /// <summary>
  /// Creates <paramref name="jet"/>'s native particle system and stores its id, β and dust
  /// production rate. <c>AddParticleSystem</c> returns no computed values, and every jet has its
  /// own (its own grain and Q_d₀): a later jet reads them back with an unchanged modify.
  /// A refusal (id 0: no comet, render frontend or clock, or the device could not create the
  /// system) is not silent: it sets <see cref="JetViewModel.NativeError"/>, which a success clears.
  /// </summary>
  public static void AddNative(INativeRuntimeService runtime, ParticleSystemModel model,
    ParticleSystemJet psJet, JetViewModel jet, bool isFirst, string? litWarningFormat = null, TimeScaleService? timeScale = null)
  {
    if (isFirst)
    {
      var computed = runtime.AddFirstParticleSystem(model, psJet, out ulong psId);
      jet.NativePsId = psId;
      jet.NativeError = psId == 0 ? RefusedText : null;
      if (computed is not null)
      {
        jet.Beta = computed.Beta;
        jet.DustProductionRateAt1AuKgs = computed.DustProductionRateAt1AuKgs;
      }
      RefreshLitWarning(runtime, jet, litWarningFormat, timeScale);
      return;
    }
    bool ok = runtime.AddParticleSystem(model, psJet, out ulong id);
    if (!ok) id = 0; // never trust an id the call did not stand behind
    jet.NativePsId = id;
    jet.NativeError = id == 0 ? RefusedText : null;
    if (id != 0 && runtime.ModifyParticleSystem(id, model, psJet, out var props) && props is not null)
    {
      jet.Beta = props.Beta;
      jet.DustProductionRateAt1AuKgs = props.DustProductionRateAt1AuKgs;
    }
    RefreshLitWarning(runtime, jet, litWarningFormat);
  }
}
