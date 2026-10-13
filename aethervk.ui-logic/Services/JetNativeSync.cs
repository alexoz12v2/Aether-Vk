using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Services;

/// <summary>
/// Creates the model session's jets in the native runtime. Idempotent: jets that already have a
/// native particle system are skipped, so the Model tab (on commit) and a savefile open can both
/// call it on the UI thread without creating a jet twice.
/// </summary>
public static class JetNativeSync
{
  public static ParticleSystemModel BuildModel(ModelSession s) => new(
    DiametreUm: s.DiametreUm,
    DensityGCm3: s.DensityGCm3,
    ScatteringEfficiency: s.ScatteringEfficiency,
    GeometricAlbedo: s.GeometricAlbedo,
    SizeQ: s.SizeQ,
    SizeRange: s.SizeRange,
    Afrho0Cm: s.Afrho0Cm,
    AfrhoPower: s.AfrhoPower,
    AfrhoCutoffAu: s.AfrhoCutoffAu,
    AfrhoMaxValueCm: s.AfrhoMaxValueCm);

  public static ParticleSystemJet BuildJet(JetViewModel j, float nucleusRadiusKm) => new(
    LatitudeRad: j.LatitudeRad,
    LongitudeRad: j.LongitudeRad,
    ApertureRad: j.ApertureRad,
    StartVelocityMean: j.StartVelocityMeanMs,
    StartVelocityStd: j.StartVelocityStdMs,
    StreamColor: j.StreamColor,
    NucleusRadiusKm: nucleusRadiusKm,
    Seed: j.Seed);

  /// <summary>The radius jets are emitted from: the Model tab's manual one, else the comet's.</summary>
  public static float EffectiveNucleusRadiusKm(ModelSession model, CometSession? comet) =>
    model.ManualNucleusRadiusKm > 0f ? model.ManualNucleusRadiusKm : comet?.NucleusRadiusKm ?? 0f;

  /// <summary>
  /// Adds every jet of <paramref name="session"/> without a native system (the comet must be
  /// committed). The first system of the scene goes through <c>AddFirstParticleSystem</c>.
  /// </summary>
  public static void EnsureNative(INativeRuntimeService runtime, ModelSession session, float nucleusRadiusKm)
  {
    var model = BuildModel(session);
    bool any = false;
    foreach (var jet in session.Jets)
      any |= jet.NativePsId != 0;
    foreach (var jet in session.Jets)
    {
      if (jet.NativePsId != 0) continue;
      var psJet = BuildJet(jet, nucleusRadiusKm);
      if (!any)
      {
        var computed = runtime.AddFirstParticleSystem(model, psJet, out ulong psId);
        jet.NativePsId = psId;
        if (computed is not null)
        {
          jet.Beta = computed.Beta;
          jet.DustProductionRateAt1AuKgs = computed.DustProductionRateAt1AuKgs;
        }
      }
      else
      {
        runtime.AddParticleSystem(model, psJet, out ulong psId);
        jet.NativePsId = psId;
      }
      any |= jet.NativePsId != 0;
    }
  }
}
