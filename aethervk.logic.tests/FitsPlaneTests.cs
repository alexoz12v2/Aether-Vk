using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;
using static AetherVk.Logic.Tests.FitsBillboardTests;

namespace AetherVk.Logic.Tests;

/// <summary>
/// FITS cubes: the plane selector of the billboard's FITS panel (shown only for a cube, 1-based for
/// the user, 0-based to the engine, off the UI thread), its persistence in the save index (absent
/// for the first plane, so older saves stay byte-identical), its re-apply after a load, and the
/// facts rows for the image, planes, date source, WCS projection / distortion / frame and HiPS.
/// </summary>
public class FitsPlaneTests
{
  /// <see cref="LtJson"/> as a 3-plane cube (plane 0 shown).
  internal static readonly string CubeJson = LtJson.Replace(
    "\"naxis1\": 1200, \"naxis2\": 1200, \"bitpix\": 32,",
    "\"naxis1\": 1200, \"naxis2\": 1200, \"bitpix\": 32, \"hdu\": 1, \"compressed\": false, \"planes\": 3, \"plane\": 0,");

  /// A 2 × 3 cube (NAXIS3 = 2, NAXIS4 = 3).
  internal static readonly string Cube4dJson = LtJson.Replace(
    "\"naxis1\": 1200, \"naxis2\": 1200, \"bitpix\": 32,",
    "\"naxis1\": 1200, \"naxis2\": 1200, \"bitpix\": 32, \"hdu\": 2, \"compressed\": true, \"planes\": 6, \"plane\": 0, \"planeAxes\": [2, 3],");

  private static Rig CubeRig(string? json = null)
  {
    var rig = new Rig(fitsJson: json ?? CubeJson);
    rig.Fake.Runtime.Setup(r => r.SetFitsPlane(It.IsAny<ulong>(), It.IsAny<ulong>())).Returns(true);
    return rig;
  }

  private static string? Fact(FitsImageViewModel fits, string automationId) =>
    fits.Facts.SingleOrDefault(f => f.AutomationId == automationId)?.Value;

  [Fact]
  public void PlaneSelector_IsHiddenForAPlainImage()
  {
    var rig = new Rig();
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    Assert.False(fits.HasPlanes);
    Assert.Equal(1, fits.PlaneCount);
    Assert.Null(Fact(fits, "FitsFact_Planes"));
    Assert.Equal(string.Empty, fits.PlaneAxesText);
  }

  /// Plane 3 of 3 asks the engine for the 0-based plane 2 on the background scheduler, every
  /// billboard of the texture records it, the texels are re-read and the facts follow.
  [Fact]
  public async Task PlaneSelector_ShownForACube_CallsTheServiceWithTheZeroBasedIndex()
  {
    var rig = CubeRig();
    var row = rig.Create("/data/lt_67p.fits");
    var twin = rig.Create("/data/lt_67p.fits");
    var fits = row.Fits!;
    Assert.True(fits.HasPlanes);
    Assert.Equal(3, fits.PlaneCount);
    Assert.Equal(3m, fits.PlaneMaximum);
    Assert.Equal(1m, fits.PlaneValue);
    Assert.Equal("of 3", fits.PlaneOfText);
    Assert.Equal("Plane", fits.PlaneLabel);
    Assert.Equal("1 of 3", Fact(fits, "FitsFact_Planes"));
    Assert.Equal(string.Empty, fits.PlaneAxesText); // a 3-D cube: no per-axis text

    fits.PlaneValue = 3;
    rig.Fake.Runtime.Verify(r => r.SetFitsPlane(It.IsAny<ulong>(), It.IsAny<ulong>()), Times.Never); // not on the UI thread
    Assert.True(fits.IsApplyingPlane);
    rig.Fake.Drain();
    await Task.Yield();
    rig.Fake.Drain();
    rig.Fake.Runtime.Verify(r => r.SetFitsPlane(FitsId, 2UL), Times.Once);
    Assert.Contains(FitsId, rig.TexelsChanged);
    Assert.Equal(2, rig.Svc.Find(row.Id)!.FitsPlane);
    Assert.Equal(2, rig.Svc.Find(twin.Id)!.FitsPlane);
    Assert.Equal("3 of 3", Fact(fits, "FitsFact_Planes"));
    // the twin's row follows without a second engine call
    Assert.Equal(3m, twin.Fits!.PlaneValue);
    rig.Fake.Drain();
    rig.Fake.Runtime.Verify(r => r.SetFitsPlane(It.IsAny<ulong>(), It.IsAny<ulong>()), Times.Once);

    // out of range is clamped to the last plane; back to plane 1 asks for plane 0
    fits.PlaneValue = 9;
    Assert.Equal(3m, fits.PlaneValue);
    fits.PlaneValue = 1;
    rig.Fake.Drain();
    rig.Fake.Runtime.Verify(r => r.SetFitsPlane(FitsId, 0UL), Times.Once);
    Assert.Equal(0, rig.Svc.Find(row.Id)!.FitsPlane);
  }

  /// A refused plane (unreadable source) says so and is retried by the next reconcile.
  [Fact]
  public void PlaneSelector_RefusedPlane_ShowsAStatus()
  {
    var rig = new Rig(fitsJson: CubeJson);
    rig.Fake.Runtime.Setup(r => r.SetFitsPlane(It.IsAny<ulong>(), It.IsAny<ulong>())).Returns(false);
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    fits.PlaneValue = 2;
    rig.Fake.Drain();
    rig.Fake.Drain();
    Assert.StartsWith("Plane refused", fits.StatusMessage);
    Assert.DoesNotContain(FitsId, rig.TexelsChanged);
  }

  /// With two or more extra axes the per-axis indices (1-based, NAXIS3 fastest) are shown.
  [Fact]
  public void PlaneAxes_ShowThePerAxisIndices()
  {
    var rig = CubeRig(Cube4dJson);
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    Assert.Equal(6, fits.PlaneCount);
    Assert.Equal("NAXIS3 = 1, NAXIS4 = 1", fits.PlaneAxesText);
    fits.PlaneValue = 4; // flat index 3 = i3 1 + 2 · i4 1
    Assert.Equal("NAXIS3 = 2, NAXIS4 = 2", fits.PlaneAxesText);
    Assert.Equal("4 of 6 (NAXIS3 = 2, NAXIS4 = 2)", Fact(fits, "FitsFact_Planes"));
    Assert.Equal("NAXIS3 = 2, NAXIS4 = 3, NAXIS5 = 1", FitsImageViewModel.FormatPlaneAxes([2, 3, 4], 5));
    Assert.Equal(string.Empty, FitsImageViewModel.FormatPlaneAxes([3], 2));
  }

  // ── persistence ──

  [Fact]
  public void FitsPlane_RoundTripsThroughTheSaveIndex()
  {
    var b = new DesktopBillboard(Guid.NewGuid(), "/a.fits", 1, 2, 300, 300) { FitsPlane = 2 };
    var json = SaveIndexIo.ToJson(new SaveIndex { Billboards = [SaveIndexMapping.ToDto(b)] });
    Assert.Contains("\"fits_plane\": 2", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(2, back.Billboards[0].FitsPlane);
    Assert.Equal(b, SaveIndexMapping.FromDto(back.Billboards[0]));
    // a negative value read from a hand-edited file is the first plane
    Assert.Equal(0, SaveIndexMapping.FromDto(new BillboardDto { Id = b.Id, TextureKey = "/a.fits", FitsPlane = -4 }).FitsPlane);
  }

  /// The first plane is not written: an older save (no field) reads back and writes out byte for
  /// byte as it was.
  [Fact]
  public void FitsPlane_AbsentForTheFirstPlane_OldSavesStayByteIdentical()
  {
    var b = new DesktopBillboard(Guid.NewGuid(), "/a.fits", 1, 2, 300, 300)
    { FitsStretch = new FitsStretch(FitsStretchMode.Manual, FitsTransfer.Log, -10.5, 2000) };
    Assert.Equal(0, b.FitsPlane);
    Assert.Null(SaveIndexMapping.ToDto(b).FitsPlane);
    string old = SaveIndexIo.ToJson(new SaveIndex { Billboards = [SaveIndexMapping.ToDto(b)] });
    Assert.DoesNotContain("fits_plane", old);
    var read = System.Text.Json.JsonSerializer.Deserialize(old, SaveIndexJsonContext.Default.SaveIndex)!;
    var restored = SaveIndexMapping.FromDto(read.Billboards[0]);
    Assert.Equal(0, restored.FitsPlane);
    string again = SaveIndexIo.ToJson(new SaveIndex { Billboards = [SaveIndexMapping.ToDto(restored)] });
    Assert.Equal(old, again);
  }

  /// After a load the engine shows plane 0: the saved plane is applied again, once, after the
  /// asset is there, and the row shows it; a save without a plane makes no engine call.
  [Fact]
  public void SavedPlane_IsReappliedAfterALoad()
  {
    var rig = CubeRig();
    var saved = new DesktopBillboard(Guid.NewGuid(), "/data/lt_67p.fits", 1, 2, 300, 300) { FitsPlane = 2 };
    rig.Svc.Restore([SaveIndexMapping.FromDto(SaveIndexMapping.ToDto(saved))]);
    rig.Fake.Drain();
    rig.Fake.Drain();
    rig.Fake.Runtime.Verify(r => r.SetFitsPlane(FitsId, 2UL), Times.Once);
    Assert.Contains(FitsId, rig.TexelsChanged);
    var fits = rig.Vm.Items.Single().Fits!;
    Assert.Equal(3m, fits.PlaneValue);
    Assert.Equal("3 of 3", Fact(fits, "FitsFact_Planes"));
    // unrelated edits do not switch again
    rig.Svc.Update(saved.Id, x => x with { CenterX = 50 });
    rig.Fake.Drain();
    rig.Fake.Runtime.Verify(r => r.SetFitsPlane(It.IsAny<ulong>(), It.IsAny<ulong>()), Times.Once);

    var plain = CubeRig();
    plain.Svc.Restore([SaveIndexMapping.FromDto(SaveIndexMapping.ToDto(saved with { FitsPlane = 0 }))]);
    plain.Fake.Drain();
    plain.Fake.Drain();
    plain.Fake.Runtime.Verify(r => r.SetFitsPlane(It.IsAny<ulong>(), It.IsAny<ulong>()), Times.Never);
  }

  // ── facts rows ──

  /// Every new row with the engine's JSON keys, and hidden when the engine has no value.
  [Fact]
  public void FactsRows_ImagePlanesDateSourceProjectionDistortionFrameHips()
  {
    // the Liverpool frame: no hdu (an older engine), no WCS keys, no HiPS
    var lt = new Rig().Create("/data/lt_67p.fits").Fits!;
    Assert.Equal("1200 × 1200, BITPIX 32", Fact(lt, "FitsFact_Image"));
    Assert.Equal("DATE-OBS", Fact(lt, "FitsFact_DateSource"));
    foreach (var id in new[] { "FitsFact_Planes", "FitsFact_Projection", "FitsFact_Distortion", "FitsFact_Frame", "FitsFact_Hips" })
      Assert.Null(Fact(lt, id));

    string json = Cube4dJson
      .Replace("\"timesysAssumed\": true", "\"timesysAssumed\": true, \"dateOnly\": true, \"dateIsFileDate\": false, "
        + "\"wcsProjection\": \"TAN\", \"wcsDistortion\": \"SIP\", \"wcsFrame\": \"FK5\", \"wcsEquinox\": 2000.0, \"wcsFrameAssumed\": false");
    var fits = CubeRig(json).Create("/data/lt_67p.fits").Fits!;
    Assert.Equal("1200 × 1200, BITPIX 32, HDU 2, compressed", Fact(fits, "FitsFact_Image"));
    Assert.Equal("1 of 6 (NAXIS3 = 1, NAXIS4 = 1)", Fact(fits, "FitsFact_Planes"));
    Assert.Equal("DATE-OBS · date only (midnight)", Fact(fits, "FitsFact_DateSource"));
    Assert.Equal("TAN (gnomonic)", Fact(fits, "FitsFact_Projection"));
    Assert.Equal("SIP", Fact(fits, "FitsFact_Distortion"));
    Assert.Equal("FK5 J2000", Fact(fits, "FitsFact_Frame"));

    // the formatters one by one
    Assert.Equal("ZEA (zenithal equal-area)", fits.ProjectionText("ZEA"));
    Assert.Equal("XYZ", fits.ProjectionText("XYZ")); // unknown code: as written
    Assert.Null(fits.ProjectionText(null));
    Assert.Equal("none", fits.DistortionText("none"));
    Assert.Equal("TPV", fits.DistortionText("TPV"));
    Assert.Equal("ICRS (assumed)", fits.FrameText(new FitsFacts { WcsFrame = "ICRS", WcsFrameAssumed = true }));
    Assert.Equal("ICRS", fits.FrameText(new FitsFacts { WcsFrame = "ICRS", WcsFrameAssumed = false }));
    Assert.Equal("Galactic", fits.FrameText(new FitsFacts { WcsFrame = "galactic", WcsFrameAssumed = false }));
    Assert.Equal("Ecliptic J2000", fits.FrameText(new FitsFacts { WcsFrame = "ecliptic" }));
    Assert.Equal("FK4 B1950", fits.FrameText(new FitsFacts { WcsFrame = "FK4", WcsEquinox = 1950.0 }));
    Assert.Null(fits.FrameText(new FitsFacts()));
    Assert.Equal("DATE · file date (DATE), not the observation",
      fits.DateSourceText(new FitsFacts { DateSource = "DATE", DateIsFileDate = true }));
    Assert.Equal("MJD-OBS", fits.DateSourceText(new FitsFacts { DateSource = "MJD-OBS", DateOnly = false }));
    Assert.Null(fits.DateSourceText(new FitsFacts()));
    Assert.Equal("order 3, tile 5 · Equatorial · DSS colored",
      fits.HipsText(new FitsHipsFacts { Order = 3, Npix = 5, Frame = "equatorial", FrameSource = "properties", SurveyTitle = "DSS colored" }));
    Assert.Equal("order 2, Allsky · Galactic (assumed)",
      fits.HipsText(new FitsHipsFacts { Allsky = true, Order = 2, Frame = "galactic", FrameSource = "assumed" }));
    Assert.Null(fits.HipsText(null));
  }

  /// The engine's JSON with the new keys: planes, plane axes, HDU, compression, WCS facts and the
  /// HiPS object reach the model; the HiPS row shows.
  [Fact]
  public void Metadata_ParsesTheNewKeys_AndTheHipsRowShows()
  {
    string json = """
      {"assetId":7,"textureWidth":4,"textureHeight":4,"rowOrder":"topDown","flipped":true,
       "stretch":{"mode":0,"transfer":0,"black":0.0,"white":1.0},
       "naxis1":4,"naxis2":4,"bitpix":16,"hdu":1,"compressed":false,"planes":1,"plane":0,"cards":[],
       "facts":{"object":null,"wcsError":null,"wcsProjection":null,"wcsDistortion":null,"wcsFrame":null,"wcsEquinox":null,"wcsFrameAssumed":null},
       "hips":{"hips":true,"allsky":false,"order":3,"npix":5,"tile_width":4,"frame":"equatorial","frame_source":"assumed",
               "ident_source":"path","survey_order":null,"survey_title":null,"root":null,
               "center_ra_deg":12.5,"center_dec_deg":-3.25,"pixel_scale_arcsec":1611.2}}
      """;
    var m = FitsMetadata.Parse(json)!;
    Assert.Equal((4, 4, 16, 1, false, 1L, 0L), (m.Naxis1, m.Naxis2, m.Bitpix, m.Hdu, m.Compressed, m.Planes, m.Plane));
    Assert.Empty(m.PlaneAxes);
    Assert.Null(m.Facts.WcsProjection);
    Assert.NotNull(m.Hips);
    Assert.Equal((false, 3, 5UL, 4, "equatorial", "assumed"), (m.Hips!.Allsky, m.Hips.Order, m.Hips.Npix, m.Hips.TileWidth, m.Hips.Frame, m.Hips.FrameSource));
    Assert.Equal(12.5, m.Hips.CenterRaDeg);

    var cube = FitsMetadata.Parse(Cube4dJson)!;
    Assert.Equal((6L, 2), (cube.Planes, cube.Hdu));
    Assert.Equal([2L, 3L], cube.PlaneAxes);
    var tan = FitsMetadata.Parse(LtJson.Replace("\"wcsError\": null", "\"wcsError\": null, \"wcsProjection\": \"TAN\", \"wcsDistortion\": \"none\", \"wcsFrame\": \"ICRS\", \"wcsEquinox\": null, \"wcsFrameAssumed\": true"))!;
    Assert.Equal(("TAN", "none", "ICRS", (double?)null, (bool?)true),
      (tan.Facts.WcsProjection, tan.Facts.WcsDistortion, tan.Facts.WcsFrame, tan.Facts.WcsEquinox, tan.Facts.WcsFrameAssumed));

    var rig = new Rig(fitsJson: json);
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    Assert.Equal("order 3, tile 5 · Equatorial (assumed)", Fact(fits, "FitsFact_Hips"));
    Assert.Equal("4 × 4, BITPIX 16, HDU 1", Fact(fits, "FitsFact_Image"));
    Assert.False(fits.HasPlanes);
  }
}
