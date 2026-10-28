using System;
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

  [Fact]
  public void ModelJetsSbdbRotationBillboards_RoundTrip()
  {
    var model = new ModelSession { DiametreUm = 321, AfrhoPower = 1.5f, DustFlow = true, ManualNucleusRadiusKm = 4.5f };
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
}
