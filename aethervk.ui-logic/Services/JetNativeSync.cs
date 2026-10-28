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
  /// The native grain and activity parameters of <paramref name="j"/>: its own grain and Afρ₀, and
  /// the model-wide shape of the Afρ law (exponent, cutoff, max) from <paramref name="s"/>.
  /// </summary>
  public static ParticleSystemModel BuildModel(ModelSession s, JetViewModel j) => new(
    DiametreUm: j.DiametreUm,
    DensityGCm3: j.DensityGCm3,
    ScatteringEfficiency: j.ScatteringEfficiency,
    GeometricAlbedo: j.GeometricAlbedo,
    SizeQ: j.SizeQ,
    SizeRange: j.SizeRange,
    Afrho0Cm: j.Afrho0Cm,
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
    bool any = false;
    foreach (var jet in session.Jets)
      any |= jet.NativePsId != 0;
    foreach (var jet in session.Jets)
    {
      if (jet.NativePsId != 0) continue;
      AddNative(runtime, BuildModel(session, jet), BuildJet(jet, nucleusRadiusKm), jet, isFirst: !any);
      any |= jet.NativePsId != 0;
    }
  }

  /// <summary>
  /// Creates <paramref name="jet"/>'s native particle system and stores its id, β and dust
  /// production rate. <c>AddParticleSystem</c> returns no computed values, and every jet has its
  /// own (its own grain and Afρ₀): a later jet reads them back with an unchanged modify.
  /// </summary>
  public static void AddNative(INativeRuntimeService runtime, ParticleSystemModel model,
    ParticleSystemJet psJet, JetViewModel jet, bool isFirst)
  {
    if (isFirst)
    {
      var computed = runtime.AddFirstParticleSystem(model, psJet, out ulong psId);
      jet.NativePsId = psId;
      if (computed is not null)
      {
        jet.Beta = computed.Beta;
        jet.DustProductionRateAt1AuKgs = computed.DustProductionRateAt1AuKgs;
      }
      return;
    }
    runtime.AddParticleSystem(model, psJet, out ulong id);
    jet.NativePsId = id;
    if (id != 0 && runtime.ModifyParticleSystem(id, model, psJet, out var props) && props is not null)
    {
      jet.Beta = props.Beta;
      jet.DustProductionRateAt1AuKgs = props.DustProductionRateAt1AuKgs;
    }
  }
}
