using System;
using System.Linq;
using System.Numerics;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>The DTO ↔ live conversions round-trip exactly, through JSON too.</summary>
public class SaveIndexMappingTests
{
  [Fact]
  public void EnumNames_AreSnakeCase_AndParseBack()
  {
    Assert.Equal("comet_orbiting", SaveIndexMapping.Snake(CameraMode.CometOrbiting));
    Assert.Equal("sun_tracking", SaveIndexMapping.Snake(EarthObserverOrientationMode.SunTracking));
    foreach (EarthObserverOrientationMode m in Enum.GetValues(typeof(EarthObserverOrientationMode)))
      Assert.Equal(m, SaveIndexMapping.ParseSnake(SaveIndexMapping.Snake(m), EarthObserverOrientationMode.Free));
    Assert.Equal(CameraMode.UpZenith, SaveIndexMapping.ParseSnake("bogus", CameraMode.UpZenith));
  }

  /// Saves from before the simplification: a retired lock-in opens as a free look along the saved
  /// inertial look, and the observer at the Earth's centre (the old site is kept for the checkbox).
  [Theory]
  [InlineData("comet_lock_in")]
  [InlineData("sun_lock_in")]
  public void RetiredLockIn_OpensAsFree_Geocentric(string orientation)
  {
    var restored = SaveIndexMapping.FromDto(new ViewportDto
    {
      Mode = "earth_position",
      Observer = new ObserverDto
      {
        LatDeg = 40f, LonDeg = 10f, Orientation = orientation,
        InertialLook = [0, 0, 0.6, 0.8], BodyFixedLook = [0.5, 0.5, 0.5, 0.5],
      },
    });
    Assert.Equal(EarthObserverOrientationMode.Free, restored.ObserverOrientation);
    Assert.Equal(new double[] { 0, 0, 0.6, 0.8 }, restored.ObserverInertialLook);
    Assert.False(restored.ObserverTopocentric);
    Assert.Equal((40f, 10f), (restored.ObserverLatDeg, restored.ObserverLonDeg));
  }

  [Fact]
  public void CameraState_RoundTrips_ThroughTheIndex()
  {
    var state = new CameraPersistedState(
      CameraMode.EarthPosition,
      new CameraTransformState(1.25, -0.5, 0.125, 0.1, 0.2, 0.3, 0.927),
      new CameraProjectionState(true, 0.7f, 1.6f, 1e-6f, 1e3f, -1, 1, -0.5f, 0.5f, 2.5f),
      new Vector3(0.001f, -0.002f, 0.003f),
      45.5f, 9.25f, 120.0,
      EarthObserverOrientationMode.CometTracking,
      [0, 0.38268343236508984, 0, 0.9238795325112867],
      [0.5, 0.5, 0.5, 0.5],
      ObserverTopocentric: true);
    var index = new SaveIndex { Viewports = [SaveIndexMapping.ToDto(state)] };
    var json = SaveIndexIo.ToJson(index);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    var restored = SaveIndexMapping.FromDto(back.Viewports[0]);
    Assert.Equal(state.Mode, restored.Mode);
    Assert.Equal(state.Transform, restored.Transform);
    Assert.Equal(state.Projection, restored.Projection);
    Assert.Equal(state.OrbitOffsetAu, restored.OrbitOffsetAu);
    Assert.Equal((state.ObserverLatDeg, state.ObserverLonDeg, state.ObserverElevationM, state.ObserverOrientation),
      (restored.ObserverLatDeg, restored.ObserverLonDeg, restored.ObserverElevationM, restored.ObserverOrientation));
    Assert.Equal(state.ObserverInertialLook, restored.ObserverInertialLook);
    Assert.Equal(state.ObserverBodyFixedLook, restored.ObserverBodyFixedLook);
    Assert.True(restored.ObserverTopocentric);
    Assert.Contains("\"topocentric\": true", json);
    Assert.Contains("\"mode\": \"earth_position\"", json);
    Assert.Contains("\"orientation\": \"comet_tracking\"", json);
  }

  [Fact]
  public void ViewportDisplay_RoundTrips_AndOlderSavesHaveNone()
  {
    var state = new CameraPersistedState(CameraMode.EarthPosition, null, null, Vector3.Zero, 0, 0, 0,
      EarthObserverOrientationMode.SunTracking, [0, 0, 0, 1], [0, 0, 0, 1], new ViewportDisplayState(1, 2, false));
    var json = SaveIndexIo.ToJson(new SaveIndex { Viewports = [SaveIndexMapping.ToDto(state)] });
    Assert.Contains("\"ortho_unit\": \"km\"", json);
    Assert.Contains("\"fov_unit\": \"arcsec\"", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(new ViewportDisplayState(1, 2, false), SaveIndexMapping.FromDto(back.Viewports[0]).Display);
    Assert.Contains("\"show_behind_camera\": true", json);
    // the behind-camera choice round-trips; a display block written before it reads as on
    var hidden = state with { Display = new ViewportDisplayState(1, 2, false, ShowBehindCamera: false) };
    var hiddenBack = System.Text.Json.JsonSerializer.Deserialize(
      SaveIndexIo.ToJson(new SaveIndex { Viewports = [SaveIndexMapping.ToDto(hidden)] }),
      SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.False(SaveIndexMapping.FromDto(hiddenBack.Viewports[0]).Display!.ShowBehindCamera);
    Assert.True(System.Text.Json.JsonSerializer.Deserialize(
      "{\"ortho_unit\": \"km\"}", SaveIndexJsonContext.Default.ViewportDisplayDto)!.ShowBehindCamera);
    // a save written before the field: no display state (the camera keeps its own)
    Assert.Null(SaveIndexMapping.FromDto(new ViewportDto()).Display);
    // an unknown unit name falls back to the first unit
    Assert.Equal(new ViewportDisplayState(0, 0, true), SaveIndexMapping.FromDto(new ViewportDto
    {
      Display = new ViewportDisplayDto { OrthoUnit = "parsec", FovUnit = "deg" },
    }).Display);
  }

  /// <summary>
  /// A jet's own grain and activity survive the save (mapping and JSON); a jet without them (an
  /// older save) takes the model's values given to <c>FromDto</c>.
  /// </summary>
  [Fact]
  public void JetGrainFields_RoundTrip()
  {
    var grain = new AetherVk.Logic.Models.JetGrain(250f, 0.8f, 1.2f, 0.05f, 3.2f, 20f, 900f);
    var jet = new AetherVk.Logic.ViewModels.JetViewModel(10, -20, 33, 2.5f, 0.25f, new Vector4(0.1f, 0.2f, 0.3f, 1), true, 7u, grain);
    var dto = SaveIndexMapping.ToDto(jet);
    Assert.Equal((250f, 0.8f, 1.2f, 0.05f, 3.2f, 20f, 900f),
      (dto.DiametreUm, dto.DensityGCm3, dto.ScatteringEfficiency, dto.GeometricAlbedo, dto.SizeQ, dto.SizeRange, dto.DustRate1AuKgs));
    var back = System.Text.Json.JsonSerializer.Deserialize(
      SaveIndexIo.ToJson(new SaveIndex { Jets = [dto] }), SaveIndexJsonContext.Default.SaveIndex)!;
    // the model's values must not override a jet's own
    var other = new AetherVk.Logic.Models.JetGrain(1f, 2f, 0.6f, 0.3f, 4.5f, 2f, 3f);
    Assert.Equal(grain, SaveIndexMapping.FromDto(back.Jets[0], other).Grain);
    Assert.Equal(grain, SaveIndexMapping.FromDto(back.Jets[0]).Grain);

    // a JSON jet without the fields: the model's values, else the built-in defaults
    var old = System.Text.Json.JsonSerializer.Deserialize(
      "{\"latitude_deg\": 5, \"seed\": 3}", SaveIndexJsonContext.Default.JetDto)!;
    Assert.Null(old.DiametreUm);
    Assert.Equal(other, SaveIndexMapping.FromDto(old, other).Grain);
    Assert.Equal(AetherVk.Logic.Models.JetGrain.Default, SaveIndexMapping.FromDto(old).Grain);
  }

  [Fact]
  public void ModelJetsSbdbRotationBillboards_RoundTrip()
  {
    var model = new ModelSession { DiametreUm = 321, DustRatePower = 1.5f, DustFlow = true, ManualNucleusRadiusKm = 4.5f };
    var m2 = new ModelSession();
    SaveIndexMapping.Apply(SaveIndexMapping.ToDto(model), m2);
    Assert.Equal(SaveIndexIo.ToJson(new SaveIndex { Model = SaveIndexMapping.ToDto(model) }),
      SaveIndexIo.ToJson(new SaveIndex { Model = SaveIndexMapping.ToDto(m2) }));

    var jet = new AetherVk.Logic.ViewModels.JetViewModel(10, -20, 33, 2.5f, 0.25f, new Vector4(0.1f, 0.2f, 0.3f, 1), false, 4242u);
    var j2 = SaveIndexMapping.FromDto(SaveIndexMapping.ToDto(jet));
    Assert.Equal((jet.LatitudeDeg, jet.LongitudeDeg, jet.ApertureDeg, jet.StartVelocityMeanMs, jet.StartVelocityStdMs, jet.StreamColor, jet.IsPreviewVisible, jet.Seed),
      (j2.LatitudeDeg, j2.LongitudeDeg, j2.ApertureDeg, j2.StartVelocityMeanMs, j2.StartVelocityStdMs, j2.StreamColor, j2.IsPreviewVisible, j2.Seed));

    var sb = new SmallBodyDataComponent { SpkId = 1000012, Designation = "67P", E = 0.64, Q = 1.24, I = 7, Om = 50, W = 12, Tp = 2460000.5, IsNeo = true };
    var sb2 = SaveIndexMapping.FromDto(SaveIndexMapping.ToDto(sb));
    Assert.Equal((sb.SpkId, sb.Designation, sb.E, sb.Q, sb.I, sb.Om, sb.W, sb.Tp, sb.IsNeo), (sb2.SpkId, sb2.Designation, sb2.E, sb2.Q, sb2.I, sb2.Om, sb2.W, sb2.Tp, sb2.IsNeo));
    Assert.True(double.IsNaN(sb2.ElementsEpochJd));

    var comet = new CometSession { RotPoleRaDeg = 69.3, RotPoleDecDeg = 64.1, RotRateDegDay = 649.1, RotBodyFixedOrientationIndex = 1 };
    var c2 = new CometSession();
    SaveIndexMapping.Apply(SaveIndexMapping.RotationOf(comet), c2);
    Assert.Equal((comet.RotPoleRaDeg, comet.RotPoleDecDeg, comet.RotRateDegDay, comet.RotBodyFixedOrientationIndex),
      (c2.RotPoleRaDeg, c2.RotPoleDecDeg, c2.RotRateDegDay, c2.RotBodyFixedOrientationIndex));

    var b = new DesktopBillboard(Guid.NewGuid(), "/x.png", 1, 2, 3, 4, 0.5, 30, false) { GizmoMode = BillboardGizmoMode.Rotate };
    Assert.Equal(b with { GizmoMode = BillboardGizmoMode.None }, SaveIndexMapping.FromDto(SaveIndexMapping.ToDto(b)));
  }

  /// <summary>The trajectory lines, coma 1/ρ normalisation and dust exposure survive a save.</summary>
  [Fact]
  public void ModelDustDisplayOptions_RoundTrip()
  {
    var model = new ModelSession { DustTrajectories = true, DustRhoNorm = true, DustExposureEv = -2.75 };
    var json = SaveIndexIo.ToJson(new SaveIndex { Model = SaveIndexMapping.ToDto(model) });
    var dto = System.Text.Json.JsonSerializer.Deserialize(
      System.Text.Json.JsonSerializer.Serialize(SaveIndexMapping.ToDto(model), SaveIndexJsonContext.Default.ModelDto),
      SaveIndexJsonContext.Default.ModelDto)!;
    var m2 = new ModelSession();
    SaveIndexMapping.Apply(dto, m2);
    Assert.Equal((true, true, -2.75), (m2.DustTrajectories, m2.DustRhoNorm, m2.DustExposureEv));
    Assert.Equal(json, SaveIndexIo.ToJson(new SaveIndex { Model = SaveIndexMapping.ToDto(m2) }));
  }

  /// <summary>A save written before these options existed opens with them at their defaults.</summary>
  [Fact]
  public void ModelDustDisplayOptions_OldSave_MapsToDefaults()
  {
    var old = System.Text.Json.JsonSerializer.Deserialize(
      "{\"dust_tracers\": true, \"dust_flow\": true, \"dust_flow_speed\": 2}", SaveIndexJsonContext.Default.ModelDto)!;
    var m = new ModelSession { DustTrajectories = true, DustRhoNorm = true, DustExposureEv = 3 };
    SaveIndexMapping.Apply(old, m);
    Assert.True(m.DustTracers);
    Assert.True(m.DustFlow);
    Assert.Equal((false, false, 0.0), (m.DustTrajectories, m.DustRhoNorm, m.DustExposureEv));
  }

  [Fact]
  public void Appearance_ByKey_FallsBackWithoutItsMesh()
  {
    var a = new CometAppearanceDto { Mesh = 5, Albedo = 6, Yaw = 10, Pitch = 20, Roll = 30, TranslationZ = 0.5f, Mode = CometDisplayMode.Custom };
    var dto = SaveIndexMapping.ToDto(a, id => id switch { 5 => "m", 6 => "t", _ => null });
    Assert.Equal(("custom", "m", "t", null as string), (dto.Mode, dto.MeshKey, dto.AlbedoKey, dto.NormalKey));
    var back = SaveIndexMapping.FromDto(dto, k => k switch { "m" => 50ul, "t" => 60ul, _ => 0ul }, out bool fell);
    Assert.False(fell);
    Assert.Equal((50ul, 60ul, 10f, 20f, 30f, 0.5f, CometDisplayMode.Custom), (back.Mesh, back.Albedo, back.Yaw, back.Pitch, back.Roll, back.TranslationZ, back.Mode));
    back = SaveIndexMapping.FromDto(dto, k => k == "t" ? 60ul : 0ul, out fell);
    Assert.True(fell);
    Assert.Equal(CometDisplayMode.Default, back.Mode);
  }

  /// A billboard's physical size (plate scale, native size, binding) is saved; an older save has
  /// none and opens unbound.
  [Fact]
  public void BillboardPhysicalSize_RoundTrips_AndOlderSavesHaveNone()
  {
    var b = new AetherVk.Logic.Models.DesktopBillboard(Guid.NewGuid(), "/a.fits", 1, 2, 300, 300)
    { ImageArcsecPerPx = 0.3037, ImageWidthPx = 1200, ImageHeightPx = 1200, PhysicalSizeBound = true };
    var json = SaveIndexIo.ToJson(new SaveIndex { Billboards = [SaveIndexMapping.ToDto(b)] });
    Assert.Contains("\"image_arcsec_per_px\": 0.3037", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(b, SaveIndexMapping.FromDto(back.Billboards[0]));
    var old = SaveIndexMapping.FromDto(new BillboardDto { Id = b.Id, TextureKey = "/a.png", WidthPx = 10, HeightPx = 10 });
    Assert.Null(old.ImageArcsecPerPx);
    Assert.False(old.PhysicalSizeBound);
  }

  /// The Earth observer's projection remembered in another mode round-trips; older saves have none.
  [Fact]
  public void ObserverProjection_RoundTrips()
  {
    var proj = new CameraProjectionState(true, 0.002f, 1.5f, 1e-9f, 10f, 0, 0, 0, 0, 1f);
    var state = new CameraPersistedState(CameraMode.UpZenith, null, null, Vector3.Zero, 0, 0, 0,
      EarthObserverOrientationMode.CometTracking, [0, 0, 0, 1], [0, 0, 0, 1], ObserverProjection: proj);
    var json = SaveIndexIo.ToJson(new SaveIndex { Viewports = [SaveIndexMapping.ToDto(state)] });
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(proj, SaveIndexMapping.FromDto(back.Viewports[0]).ObserverProjection);
    Assert.Null(SaveIndexMapping.FromDto(new ViewportDto()).ObserverProjection);
  }
  /// The Earth observer's view up round-trips; an older save without it opens ecliptic-up (the
  /// view it was saved with), while a new observer defaults to celestial north.
  [Fact]
  public void ObserverUp_RoundTrips_AndOlderSavesAreEcliptic()
  {
    var state = new CameraPersistedState(CameraMode.EarthPosition, null, null, Vector3.Zero, 0, 0, 0,
      EarthObserverOrientationMode.Free, [0, 0, 0, 1], [0, 0, 0, 1], ObserverUp: ObserverUp.CelestialNorth);
    var json = SaveIndexIo.ToJson(new SaveIndex { Viewports = [SaveIndexMapping.ToDto(state)] });
    Assert.Contains("\"up\": \"celestial_north\"", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(ObserverUp.CelestialNorth, SaveIndexMapping.FromDto(back.Viewports[0]).ObserverUp);

    var node = System.Text.Json.Nodes.JsonNode.Parse(json)!;
    Assert.True(node["viewports"]![0]!["observer"]!.AsObject().Remove("up")); // a save before the field
    var older = node.ToJsonString();
    Assert.DoesNotContain("celestial_north", older);
    var old = System.Text.Json.JsonSerializer.Deserialize(older, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(ObserverUp.Ecliptic, SaveIndexMapping.FromDto(old.Viewports[0]).ObserverUp);
    Assert.Equal(ObserverUp.Ecliptic, SaveIndexMapping.FromDto(new ViewportDto()).ObserverUp);
  }

  /// A billboard's image height at the comet (km) round-trips; older saves have none.
  [Fact]
  public void BillboardImageHeightKm_RoundTrips_AndOlderSavesHaveNone()
  {
    var b = new AetherVk.Logic.Models.DesktopBillboard(Guid.NewGuid(), "/a.jpg", 1, 2, 300, 300)
    { ImageHeightKm = 120000, ImageWidthPx = 1300, ImageHeightPx = 1200, PhysicalSizeBound = true };
    var json = SaveIndexIo.ToJson(new SaveIndex { Billboards = [SaveIndexMapping.ToDto(b)] });
    Assert.Contains("\"image_height_km\": 120000", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(b, SaveIndexMapping.FromDto(back.Billboards[0]));
    Assert.Null(SaveIndexMapping.FromDto(new BillboardDto { Id = b.Id, TextureKey = "/a.png" }).ImageHeightKm);
  }

  /// A billboard's nucleus pixel round-trips; older saves (and half a pair) have none: the image
  /// centre.
  [Fact]
  public void BillboardNucleusPixel_RoundTrips_AndOlderSavesHaveNone()
  {
    var b = new AetherVk.Logic.Models.DesktopBillboard(Guid.NewGuid(), "/a.fits", 1, 2, 300, 300)
    { ImageWidthPx = 1200, ImageHeightPx = 1200, ImageNucleusPx = new AetherVk.Logic.Utils.PointD(593.1, 604.3) };
    var json = SaveIndexIo.ToJson(new SaveIndex { Billboards = [SaveIndexMapping.ToDto(b)] });
    Assert.Contains("\"image_nucleus_x\": 593.1", json);
    Assert.Contains("\"image_nucleus_y\": 604.3", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(b, SaveIndexMapping.FromDto(back.Billboards[0]));
    Assert.Null(SaveIndexMapping.FromDto(new BillboardDto { Id = b.Id, TextureKey = "/a.png" }).ImageNucleusPx);
    Assert.Null(SaveIndexMapping.FromDto(new BillboardDto { Id = b.Id, TextureKey = "/a.png", ImageNucleusX = 3 }).ImageNucleusPx);
  }

  /// <summary>
  /// A pre-Q_d index (afrho_* keys, no dust_rate_*) maps to converted rates; a new index round-trips
  /// its Q_d law and its JSON carries no Afρ key.
  /// </summary>
  [Fact]
  public void PreQdIndex_IsConverted_AndNewIndexWritesNoAfrho()
  {
    const string oldJson = """
      {"model": {"diametre_um": 100, "density_g_cm3": 0.8, "geometric_albedo": 0.04, "size_range": 10,
                 "afrho0_cm": 100, "afrho_power": 2, "afrho_cutoff_au": 15, "afrho_max_value_cm": 100000},
       "jets": [{"velocity_mean_ms": 2, "velocity_std_ms": 0.5, "afrho0_cm": 100},
                {"velocity_mean_ms": 1.5, "velocity_std_ms": 0.3, "diametre_um": 20, "density_g_cm3": 0.5,
                 "geometric_albedo": 0.03, "size_range": 3, "afrho0_cm": 8.128}]}
      """;
    var old = System.Text.Json.JsonSerializer.Deserialize(oldJson, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Null(old.Model.DustRate1AuKgs);
    Assert.Equal(100f, old.Model.Afrho0Cm);
    var m = new ModelSession();
    SaveIndexMapping.Apply(old.Model, m);
    float q0 = (float)LegacyAfrho.DustRateKgs(100, 100, 0.8, 2.5, 0.625, 0.04, 10);
    Assert.Equal(q0, m.DustRate1AuKgs, 1e-6f * q0);
    Assert.Equal(q0 * 1000f, m.DustRateMaxKgs, q0);
    Assert.Equal((2f, 15f), (m.DustRatePower, m.DustRateCutoffAu));
    Assert.Equal(1.5510886453350254f, SaveIndexMapping.FromDto(old.Jets[0], m.DefaultGrain).DustRate1AuKgs, 1e-5f);
    Assert.Equal(0.013690648521032838f, SaveIndexMapping.FromDto(old.Jets[1], m.DefaultGrain).DustRate1AuKgs, 1e-7f);

    // a new-format index: the law round-trips and no Afρ key is written
    var fresh = new ModelSession { DustRate1AuKgs = 0.25f, DustRatePower = 2.5f, DustRateCutoffAu = 9, DustRateMaxKgs = 50 };
    var jet = new AetherVk.Logic.ViewModels.JetViewModel(1, 2, 3, 2.5f, 0.25f, new Vector4(1, 1, 1, 1), true, 1u,
      AetherVk.Logic.Models.JetGrain.Default with { DustRate1AuKgs = 0.125f });
    var json = SaveIndexIo.ToJson(new SaveIndex { Model = SaveIndexMapping.ToDto(fresh), Jets = [SaveIndexMapping.ToDto(jet)] });
    Assert.DoesNotContain("afrho", json, StringComparison.OrdinalIgnoreCase);
    Assert.Contains("\"dust_rate_1au_kgs\": 0.25", json);
    Assert.Contains("\"dust_rate_max_kgs\": 50", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    var m2 = new ModelSession();
    SaveIndexMapping.Apply(back.Model, m2);
    Assert.Equal((0.25f, 2.5f, 9f, 50f), (m2.DustRate1AuKgs, m2.DustRatePower, m2.DustRateCutoffAu, m2.DustRateMaxKgs));
    Assert.Equal(0.125f, SaveIndexMapping.FromDto(back.Jets[0], m2.DefaultGrain).DustRate1AuKgs);
    // a new-format jet keeps its Q_d₀ even next to a stale Afρ₀
    back.Jets[0].Afrho0Cm = 100;
    Assert.Equal(0.125f, SaveIndexMapping.FromDto(back.Jets[0]).DustRate1AuKgs);
  }

  /// <summary>The reviewer's save (pre-Q_d, Afρ₀ 100 cm per jet) opens with every jet's Q_d₀ converted.</summary>
  [Fact]
  public void ReviewFixture_JetsGetConvertedDustRates()
  {
    var (index, error) = SaveIndexIo.Read(ReviewSaveReproTests.FixtureDir());
    Assert.NotNull(index);
    Assert.All(index!.Jets, j => Assert.Null(j.DustRate1AuKgs));
    Assert.All(index.Jets, j => Assert.NotNull(j.Afrho0Cm));
    var m = new ModelSession();
    SaveIndexMapping.Apply(index.Model, m);
    var jets = index.Jets.Select(j => SaveIndexMapping.FromDto(j, m.DefaultGrain)).ToList();
    Assert.All(jets, j => Assert.True(j.DustRate1AuKgs > 0));
    var j0 = index.Jets[0];
    float expected = (float)LegacyAfrho.DustRateKgs(j0.Afrho0Cm!.Value, j0.DiametreUm ?? m.DiametreUm,
      j0.DensityGCm3 ?? m.DensityGCm3, j0.VelocityMeanMs, j0.VelocityStdMs, j0.GeometricAlbedo ?? m.GeometricAlbedo,
      j0.SizeRange ?? m.SizeRange);
    Assert.Equal(expected, jets[0].DustRate1AuKgs);
  }
}
