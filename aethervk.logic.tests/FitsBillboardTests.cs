using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Moq;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// FITS billboards: the metadata JSON (<c>docs/fits_ffi_contract.md</c>), the "FITS image" part of a
/// billboard row (facts, cards, stretch, purpose buttons), the sky maths behind "Orient to image"
/// and the stretch persistence / re-apply.
/// </summary>
public class FitsBillboardTests
{
  private const ulong FitsId = 7;

  /// The Liverpool Telescope 67P frame (test_assets/fits_lt_67p_header_4x3.fits) as the engine
  /// describes it, abridged; the nanoseconds exceed 2^53 on purpose.
  private const string LtJson = """
    {
      "assetId": 7, "textureWidth": 1200, "textureHeight": 1200, "rowOrder": "topDown", "flipped": true,
      "stretch": { "mode": 0, "transfer": 0, "black": -1004.76, "white": 1020.78 },
      "naxis1": 1200, "naxis2": 1200, "bitpix": 32,
      "cards": [
        { "key": "SIMPLE", "value": "T", "comment": "A valid FITS file" },
        { "key": "OBJECT", "value": "67P", "comment": "Actual name of the current observation source" },
        { "key": "LONGITUD", "value": "-17.8792", "comment": "[degrees West] Telescope longitude" },
        { "key": "COMMENT", "value": "free text", "comment": "" }
      ],
      "facts": {
        "object": "67P", "telescope": "Liverpool Telescope", "instrument": "IO:O", "filter": "SDSS-G",
        "exposureS": 20.0, "dateObs": "2021-12-27T00:14:37.345", "dateSource": "DATE-OBS", "dateIsMid": false,
        "dateMidUtc": "2021-12-27T00:14:47.345Z", "dateMidTdbCenturies": 0, "dateMidTdbNanoseconds": 693749756529000001,
        "dateMidTdbUs": 693749756529000, "timesysAssumed": true,
        "pixelScaleArcsec": 0.30368, "pixelScaleSource": "CD", "fovWidthArcmin": 6.07, "fovHeightArcmin": 6.07,
        "crvalRaDeg": 135.5634, "crvalDecDeg": 28.1569, "centerRaDeg": 135.6, "centerDecDeg": 28.1,
        "radesys": "FK5", "wcsError": null, "rotationDeg": 10.0, "mirrored": false,
        "siteLatDeg": 28.7624, "siteLonDeg": -17.8792, "siteElevM": null, "siteSource": "LATITUDE/LONGITUD",
        "siteLonWestConvention": false, "siteLonRawDeg": -17.8792, "siteLonFromLst": true
      }
    }
    """;

  private static void Near(double want, double got, double tol = 1e-6) =>
    Assert.True(Math.Abs(want - got) <= tol, $"expected {want}, got {got}");

  /// Records every viewport call in order.
  internal sealed class FakeProbe : IBillboardViewportProbe
  {
    public readonly List<string> Log = [];
    public double? ArcsecPerPx { get; set; } = 2.0;
    public PointD? CometDesktopPx { get; set; } = new PointD(700, 300);
    public bool IsEarthObserver { get; set; } = true;
    public PointD? ViewportCenterDesktopPx { get; set; } = new PointD(960, 540);
    public ViewportFrame? ViewportFrame { get; set; }
    public event Action? ViewportMoved { add { } remove { } }
    public event Action? ViewChanged { add { } remove { } }
    public System.Threading.Tasks.Task RestoredViewConfirmed => System.Threading.Tasks.Task.CompletedTask;
    public bool ViewUpIsEclipticSouth { get; set; }
    public bool ViewUpIsCelestialNorth { get; set; } = true;
    public double? CometDistanceKm { get; set; } = 0.47 * 149_597_870.7;
    public double ViewportHeightPx { get; set; } = 1000;
    /// Perspective vertical field (rad); null: the scale is <see cref="ArcsecPerPx"/> alone.
    public double? VerticalFovRad { get; set; }
    public bool IsOrthographic { get; set; }
    public Vector3d? Locked;
    public (double Lat, double Lon, double? Elev)? Site;
    private bool _locked;

    /// Logs the field as the image's angular height in arcsec ("match 364.416″" for 1200 px at
    /// 0.30368″/px); perspective: the viewport's height px then show that angle.
    public bool MatchField(double heightKm, double angleRad)
    {
      Log.Add(FormattableString.Invariant($"match {angleRad / DesktopBillboardGeometry.ArcsecToRad:G6}″"));
      MatchedKm = heightKm;
      ArcsecPerPx = angleRad / ViewportHeightPx / DesktopBillboardGeometry.ArcsecToRad;
      return true;
    }

    public double MatchedKm;

    public bool EnterEarthObserver()
    {
      Log.Add("enter");
      IsEarthObserver = true;
      return true;
    }

    public void SetObserverSite(double latDeg, double lonDeg, double? elevM)
    {
      Log.Add("site");
      Site = (latDeg, lonDeg, elevM);
    }

    public bool LockToDirection(Vector3d worldDirection)
    {
      Log.Add("orient");
      if (!IsEarthObserver)
        return false;
      Locked = worldDirection;
      return true;
    }

    public bool IsComparisonLocked
    {
      get => _locked;
      set
      {
        Log.Add($"lock {value}");
        _locked = value;
      }
    }
  }

  internal sealed class Rig
  {
    public readonly FakeAssetRuntime Fake = new();
    public readonly AssetLibraryService Lib;
    public readonly DesktopBillboardService Svc = new();
    public readonly FitsImageService Fits;
    public readonly DesktopBillboardsViewModel Vm;
    public readonly FakeProbe Probe = new();
    public readonly List<(short C, ulong Ns)> Seeks = [];
    public readonly List<ulong> TexelsChanged = [];

    /// <summary>Every "Model range" request (the observation epoch, UTC), in order.</summary>
    public readonly List<DateTimeOffset> ModelRanges = [];

    /// <param name="withModelRange">Wire the Timeline's "Model range" request (logged as "range").</param>
    /// <param name="onModelRange">Also forwards each request (e.g. to <c>TimelineService.RequestModelRange</c>, as the app does); implies <paramref name="withModelRange"/>.</param>
    public Rig(bool withModelRange = false, Action<DateTimeOffset>? onModelRange = null)
    {
      withModelRange |= onModelRange is not null;
      Fake.NativeAssets.Add(Tex(FitsId, "/data/lt_67p.fits"));
      Fake.NativeAssets.Add(Tex(8, "/data/tail.png"));
      Fake.Runtime.Setup(r => r.GetFitsMetadataJson(FitsId)).Returns(LtJson);
      Fake.Runtime
        .Setup(r => r.SetFitsStretch(It.IsAny<ulong>(), It.IsAny<uint>(), It.IsAny<uint>(), It.IsAny<double>(), It.IsAny<double>()))
        .Returns(new FitsStretchResult(-12.5, 340.25));
      Lib = Fake.CreateService();
      Lib.Refresh();
      Lib.TexelsChanged.Subscribe(TexelsChanged.Add);
      Fits = new FitsImageService(Fake.Runtime.Object, Lib, Svc, Fake.Schedulers);
      Vm = new DesktopBillboardsViewModel(
        Svc, Lib, new FixedScreen(), Fake.Schedulers, Fits,
        (c, ns) =>
        {
          Probe.Log.Add("seek");
          Seeks.Add((c, ns));
          return Task.FromResult(true);
        },
        withModelRange
          ? end =>
          {
            Probe.Log.Add("range");
            ModelRanges.Add(end);
            onModelRange?.Invoke(end);
          }
          : null);
      Fake.Drain();
      Vm.AttachViewport(Probe, poll: false); // a polling interval never drains
    }

    public DesktopBillboardItemViewModel Create(string key)
    {
      Vm.SelectedTexture = Vm.TextureOptions.Single(t => t.Key == key);
      Vm.CreateCommand.Execute(null);
      Fake.Drain();
      return Vm.Items.Last();
    }
  }

  private sealed class FixedScreen : IScreenInfoProvider
  {
    private static readonly ScreenInfo S = new(new RectI(0, 0, 1920, 1080), 1, true);
    public IReadOnlyList<ScreenInfo> Screens => [S];
    public ScreenInfo MainWindowScreen => S;
  }

  private static ImportedAsset Tex(ulong id, string key) =>
    new(id, AssetKind.Texture, System.IO.Path.GetFileNameWithoutExtension(key), key, 1200, 1200, [], null, null)
    { IsDisplayable = true };

  // ── metadata ──

  /// ROWORDER = 'TOP-DOWN' is not flipped by the engine: texture up is pixel −j, so the facts'
  /// rotation turns half a turn and the handedness swaps.
  [Fact]
  public void Metadata_UnflippedTexture_TurnsTheRotationAndSwapsTheHandedness()
  {
    var m = FitsMetadata.Parse(LtJson.Replace("\"flipped\": true", "\"flipped\": false"))!;
    Assert.False(m.Flipped);
    Near(-170.0, m.Facts.RotationDeg!.Value);
    Assert.True(m.Facts.Mirrored);
    var standard = FitsMetadata.Parse(LtJson)!;
    Near(10.0, standard.Facts.RotationDeg!.Value);
    Assert.False(standard.Facts.Mirrored);
  }

  [Fact]
  public void Metadata_ParsesCardsFactsAndTheExactTdbParts()
  {
    var m = FitsMetadata.Parse(LtJson)!;
    Assert.Equal(FitsId, m.AssetId);
    Assert.Equal((1200, 1200), (m.TextureWidth, m.TextureHeight));
    Assert.True(m.Flipped);
    Assert.Equal(FitsStretch.Default, m.Stretch);
    Assert.Equal(new FitsStretchResult(-1004.76, 1020.78), m.StretchPoints);
    Assert.Equal(4, m.Cards.Count);
    Assert.Equal(new FitsCard("OBJECT", "67P", "Actual name of the current observation source"), m.Cards[1]);
    Assert.Equal(new FitsCard("COMMENT", "free text", ""), m.Cards[3]);
    var f = m.Facts;
    Assert.Equal("Liverpool Telescope", f.Telescope);
    Assert.Equal((short)0, f.DateMidTdbCenturies);
    Assert.Equal(693749756529000001UL, f.DateMidTdbNanoseconds); // > 2^53: not through a double
    Assert.Null(f.SiteElevM);
    Assert.Null(f.WcsError);
    Assert.Equal((135.6, 28.1), f.SkyCenter);
    Assert.Null(FitsMetadata.Parse("not json"));
    Assert.Null(FitsMetadata.Parse(null));
  }

  // ── the row ──

  /// Only a FITS texture gets the FITS part; it summarises the facts, lists every card and, once,
  /// gives a billboard without a plate scale the header's.
  [Fact]
  public void FitsRow_DetectsFits_SummarisesFacts_ListsCards_AndAutoFillsThePlateScale()
  {
    var rig = new Rig();
    var png = rig.Create("/data/tail.png");
    Assert.Null(png.Fits);
    Assert.False(png.IsFits);
    var row = rig.Create("/data/lt_67p.fits");
    Assert.True(row.IsFits);
    var fits = row.Fits!;
    Assert.Equal(4, fits.Cards.Count);

    var facts = fits.Facts.ToDictionary(x => x.Label, x => x.Value);
    Assert.Equal("67P", facts["Object"]);
    Assert.Equal("2021-12-27T00:14:37.345", facts["DATE-OBS"]);
    Assert.Equal("2021-12-27T00:14:47.345Z · no TIMESYS: read as UTC", facts["Mid-exposure UTC"]);
    Assert.Equal("Liverpool Telescope / IO:O", facts["Telescope"]);
    Assert.Equal("SDSS-G", facts["Filter"]);
    Assert.Equal("20 s", facts["Exposure"]);
    Assert.Equal("0.30368″/px (CD)", facts["Plate scale"]);
    Assert.Equal("6.07′ × 6.07′", facts["Field"]);
    Assert.Equal("lat 28.7624° · lon -17.8792° E · raw -17.8792 (sign from the header's LST) (LATITUDE/LONGITUD)",
      facts["Site"]);
    Assert.Equal("RA 09:02:24.00 (135.60000°) · Dec +28:06:00.0 (28.10000°) FK5", facts["WCS centre"]);
    Assert.Equal("10° E of N (image up)", facts["Rotation"]);
    Assert.False(facts.ContainsKey("WCS")); // no WCS error
    Assert.Equal("no: east left with north up (det CD < 0)", facts["Mirrored"]);
    // the values the end-to-end test reads
    var ids = fits.Facts.Where(x => x.AutomationId is not null).ToDictionary(x => x.AutomationId!, x => x.Label);
    Assert.Equal(new Dictionary<string, string>
    {
      ["FitsFact_DateMid"] = "Mid-exposure UTC", ["FitsFact_DateMidTdb"] = "Mid-exposure TDB", ["FitsFact_Scale"] = "Plate scale",
      ["FitsFact_Rotation"] = "Rotation", ["FitsFact_Mirrored"] = "Mirrored",
    }, ids);

    // the plate scale came from the header; the PNG has none
    Assert.Equal(0.30368, rig.Svc.Find(row.Id)!.ImageArcsecPerPx);
    Assert.Equal("0.30368", row.ImageArcsecPerPxText);
    Assert.Null(rig.Svc.Find(png.Id)!.ImageArcsecPerPx);
    // once: cleared by the user, it stays cleared
    row.ImageArcsecPerPxText = "";
    rig.Lib.Refresh();
    rig.Fake.Drain();
    Assert.Null(rig.Svc.Find(row.Id)!.ImageArcsecPerPx);
  }

  // ── scale match: image px vs viewport px (review item R5) ──

  private const double LtScale = 0.30368; // the header's ″/px, auto-filled into the row

  /// N = image ″/px ÷ viewport ″/px, 2 significant figures; below 1 the inverse is shown too.
  [Theory]
  [InlineData(1.0, "1 image px = 1.0 viewport px")]
  [InlineData(2.5, "1 image px = 2.5 viewport px")]
  [InlineData(0.31, "1 image px = 0.31 viewport px · 1 viewport px = 3.2 image px")]
  public void PixelRatio_FormatsTwoSignificantFigures_AndTheInverseBelowOne(double n, string want)
  {
    var rig = new Rig();
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    rig.Probe.ArcsecPerPx = LtScale / n; // no field reported: the probe's scale
    rig.Vm.Probe();
    Assert.Equal(want, fits.PixelRatioText);
    Near(n, fits.ViewportPxPerImagePx!.Value, 1e-9);
    Assert.Equal(want, FitsImageViewModel.FormatPixelRatio(rig.Vm.Strings, n));
  }

  [Theory]
  [InlineData(1.0, "1.0")]
  [InlineData(0.31, "0.31")]
  [InlineData(1 / 0.31, "3.2")]
  [InlineData(9.96, "10")]
  [InlineData(1234.0, "1200")]
  public void PixelRatio_Sig2_RoundsToTwoSignificantFigures(double v, string want) =>
    Assert.Equal(want, FitsImageViewModel.Sig2(v));

  /// The viewport's centre pixel from its perspective field and height: a zoom (field) or a resize
  /// (height) changes the line on the next probe tick.
  [Fact]
  public void PixelRatio_FollowsTheFieldOfView_AndTheViewportHeight()
  {
    var rig = new Rig();
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    rig.Probe.ArcsecPerPx = 99; // ignored while a perspective field is known
    double pixelRad = LtScale * DesktopBillboardGeometry.ArcsecToRad;
    // the field whose centre pixel on 1000 px subtends exactly one image px
    double fov = 2 * Math.Atan(1000 * Math.Tan(pixelRad / 2));
    rig.Probe.VerticalFovRad = fov;
    rig.Probe.ViewportHeightPx = 1000;
    rig.Vm.Probe();
    Assert.Equal("1 image px = 1.0 viewport px", fits.PixelRatioText);
    Near(1.0, fits.ViewportPxPerImagePx!.Value, 1e-9);
    Near(LtScale, FitsImageViewModel.CentrePixelArcsec(fov, 1000), 1e-12);

    // the window shrinks to 400 px: each viewport px covers 2.5 image px
    rig.Probe.ViewportHeightPx = 400;
    rig.Vm.Probe();
    Assert.Equal("1 image px = 0.40 viewport px · 1 viewport px = 2.5 image px", fits.PixelRatioText);

    // back to 1000 px, zoomed in to half the field
    rig.Probe.ViewportHeightPx = 1000;
    rig.Probe.VerticalFovRad = fov / 2;
    rig.Vm.Probe();
    Assert.Equal("1 image px = 2.0 viewport px", fits.PixelRatioText);
  }

  /// Orthographic: no field of view; the angle exists only at the comet's distance (the mapping the
  /// comparison matches with), else n/a.
  [Fact]
  public void PixelRatio_Orthographic_IsNotApplicable_WithoutTheCometMapping()
  {
    var rig = new Rig();
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    rig.Probe.IsOrthographic = true;
    rig.Probe.VerticalFovRad = 0.01; // stale: an orthographic view has none
    rig.Probe.ArcsecPerPx = null;
    rig.Vm.Probe();
    Assert.Equal("1 image px = n/a (orthographic)", fits.PixelRatioText);
    Assert.Null(fits.ViewportPxPerImagePx);

    rig.Probe.ArcsecPerPx = LtScale / 2.5; // the comet's distance gives the angle per px
    rig.Vm.Probe();
    Assert.Equal("1 image px = 2.5 viewport px (orthographic: the angle at the comet's distance)", fits.PixelRatioText);
  }

  /// A Gaussian nucleus at (x0, y0) image px on a flat background, row 0 = the top.
  private static IntensityImage Nucleus(int w, int h, double x0, double y0)
  {
    var v = new float[w * h];
    for (int j = 0; j < h; j++)
    for (int i = 0; i < w; i++)
      v[j * w + i] = (float)(10 + 500 * Math.Exp(-((i - x0) * (i - x0) + (j - y0) * (j - y0)) / 8.0));
    return new IntensityImage(w, h, v);
  }

  /// A FITS billboard finds its nucleus on its physical raster when it is first shown (not on the
  /// stretched texels); "Center on comet" then puts that pixel on the comet. Any image can detect
  /// it on demand from its texels; clearing a box returns to the image centre.
  [Fact]
  public void Nucleus_IsDetectedOnTheFitsRaster_AndOnDemandFromTexels()
  {
    var rig = new Rig();
    rig.Fake.Runtime.Setup(r => r.CopyFitsRaster(FitsId)).Returns(Nucleus(1200, 1200, 593.1, 604.3));
    var row = rig.Create("/data/lt_67p.fits");
    var n = rig.Svc.Find(row.Id)!.ImageNucleusPx!.Value;
    Near(593.1, n.X, 0.01);
    Near(604.3, n.Y, 0.01);
    Near(593.1, row.NucleusX!.Value, 0.01);
    Assert.Contains("nucleus 593.1", row.GeometryLabel);
    rig.Vm.Probe();
    row.CenterOnCometCommand.Execute(null);
    var b = rig.Svc.Find(row.Id)!;
    Near(700, DesktopBillboardGeometry.NucleusDesktop(b).X, 1e-9);
    Near(300, DesktopBillboardGeometry.NucleusDesktop(b).Y, 1e-9);
    // 1 image px = 1 desktop px (native size): the centre is (599.5 − 593.1, 599.5 − 604.3) off
    Near(700 + 6.4, b.CenterX, 0.02);
    Near(300 - 4.8, b.CenterY, 0.02);

    // a PNG: nothing until asked, then from its texels (luminance)
    var png = rig.Create("/data/tail.png");
    Assert.Null(rig.Svc.Find(png.Id)!.ImageNucleusPx);
    var grey = Nucleus(1200, 1200, 610.25, 590.75);
    var rgba = new byte[grey.Values.Length * 4];
    for (int k = 0; k < grey.Values.Length; k++)
    {
      rgba[4 * k] = rgba[4 * k + 1] = rgba[4 * k + 2] = (byte)Math.Round(grey.Values[k] / 2);
      rgba[4 * k + 3] = 255;
    }
    rig.Fake.Runtime.Setup(r => r.CopyAssetTexels(8)).Returns(new AssetThumbnail(1200, 1200, rgba));
    Assert.True(png.DetectNucleusCommand.CanExecute(null));
    png.DetectNucleusCommand.Execute(null);
    rig.Fake.Drain();
    n = rig.Svc.Find(png.Id)!.ImageNucleusPx!.Value;
    Near(610.25, n.X, 0.1);
    Near(590.75, n.Y, 0.1);

    // typed by hand (the JPEG of the same field); one box clears it
    png.NucleusX = 593.1;
    png.NucleusY = 604.3;
    Assert.Equal(new PointD(593.1, 604.3), rig.Svc.Find(png.Id)!.ImageNucleusPx);
    png.NucleusY = null;
    Assert.Null(rig.Svc.Find(png.Id)!.ImageNucleusPx);
    rig.Fake.Drain();
    Assert.Null(png.NucleusX);
  }

  [Fact]
  public void FlipSiteLon_FlipsTheAppliedLongitude_AndTheSummary()
  {
    var rig = new Rig();
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    Assert.True(fits.HasSiteLon);
    fits.FlipSiteLon = true;
    Assert.Equal(17.8792, fits.SiteLonDeg);
    Assert.Contains("lon 17.8792° E · raw -17.8792 (sign from the header's LST) · flipped",
      fits.Facts.Single(x => x.Label == "Site").Value);
    fits.UseObservatorySiteCommand.Execute(null);
    Assert.Equal((28.7624, 17.8792, (double?)null), rig.Probe.Site);
  }

  /// Apply stretch: the FFI runs on the background scheduler with the mode / transfer codes, the
  /// texture's texels and thumbnail are re-read, the result shows, and every billboard of the
  /// texture records the stretch (no second, automatic re-stretch).
  [Fact]
  public async Task ApplyStretch_RunsTheFfiOffThread_ThenRefreshesTheTexels()
  {
    var rig = new Rig();
    var row = rig.Create("/data/lt_67p.fits");
    var twin = rig.Create("/data/lt_67p.fits");
    var fits = row.Fits!;
    Assert.Equal("black -1004.76 · white 1020.78", fits.StretchResultText);
    Assert.False(fits.HasStretchParams);
    fits.StretchMode = FitsStretchMode.Percentile;
    Assert.True(fits.HasStretchParams);
    Assert.Equal(("Low %", "High %"), (fits.Param0Label, fits.Param1Label));
    Assert.Equal(("0.5", "99.5"), (fits.Param0Text, fits.Param1Text));
    fits.Transfer = FitsTransfer.Asinh;
    fits.Param0Text = "1";
    fits.Param1Text = "abc";
    Assert.False(fits.ApplyStretchCommand.CanExecute(null)); // does not parse
    fits.Param1Text = "99";
    Assert.True(fits.ApplyStretchCommand.CanExecute(null));
    int thumbs = rig.Fake.ThumbnailCalls[FitsId];

    var run = fits.ApplyStretchCommand.ExecuteAsync(null);
    rig.Fake.Runtime.Verify(
      r => r.SetFitsStretch(It.IsAny<ulong>(), It.IsAny<uint>(), It.IsAny<uint>(), It.IsAny<double>(), It.IsAny<double>()),
      Times.Never); // not on the calling (UI) thread
    Assert.True(fits.IsApplyingStretch);
    rig.Fake.Drain();
    await run;

    rig.Fake.Runtime.Verify(r => r.SetFitsStretch(FitsId, 1u, 2u, 1.0, 99.0), Times.Once);
    Assert.Equal([FitsId], rig.TexelsChanged);
    Assert.Equal(thumbs + 1, rig.Fake.ThumbnailCalls[FitsId]);
    Assert.Equal("black -12.5 · white 340.25", fits.StretchResultText);
    Assert.False(fits.IsApplyingStretch);
    var want = new FitsStretch(FitsStretchMode.Percentile, FitsTransfer.Asinh, 1, 99);
    Assert.Equal(want, rig.Svc.Find(row.Id)!.FitsStretch);
    Assert.Equal(want, rig.Svc.Find(twin.Id)!.FitsStretch);

    // manual starts from the last black / white points; back to zscale clears the saved stretch
    fits.StretchMode = FitsStretchMode.Manual;
    Assert.Equal(("Black", "White"), (fits.Param0Label, fits.Param1Label));
    Assert.Equal(("-12.5", "340.25"), (fits.Param0Text, fits.Param1Text));
    fits.StretchMode = FitsStretchMode.ZScale;
    fits.Transfer = FitsTransfer.Linear;
    run = fits.ApplyStretchCommand.ExecuteAsync(null);
    rig.Fake.Drain();
    await run;
    rig.Fake.Runtime.Verify(r => r.SetFitsStretch(FitsId, 0u, 0u, 0.0, 0.0), Times.Once);
    Assert.Null(rig.Svc.Find(row.Id)!.FitsStretch);
  }

  [Fact]
  public void SeekToObservation_SeeksTheMidExposureTdb()
  {
    var rig = new Rig();
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    Assert.True(fits.SeekToObservationCommand.CanExecute(null));
    fits.SeekToObservationCommand.Execute(null);
    Assert.Equal([((short)0, 693749756529000001UL)], rig.Seeks);
  }

  [Fact]
  public void UseObservatorySite_PutsTheObserverOnTheHeaderSite()
  {
    var rig = new Rig();
    var fits = rig.Create("/data/lt_67p.fits").Fits!;
    rig.Probe.IsEarthObserver = false; // works in any mode (the site is stored)
    rig.Vm.Probe();
    Assert.True(fits.UseObservatorySiteCommand.CanExecute(null));
    fits.UseObservatorySiteCommand.Execute(null);
    Assert.Equal((28.7624, -17.8792, (double?)null), rig.Probe.Site);
  }

  /// Orient: Earth observer only; looks at the WCS centre, turns the billboard like the sky in the
  /// view and moves it to the viewport centre.
  [Fact]
  public void OrientToImage_LooksAtTheWcsCentre_AndTurnsTheBillboard()
  {
    var rig = new Rig();
    var row = rig.Create("/data/lt_67p.fits");
    var fits = row.Fits!;
    rig.Probe.IsEarthObserver = false;
    rig.Vm.Probe();
    Assert.False(fits.OrientToImageCommand.CanExecute(null));
    rig.Probe.IsEarthObserver = true;
    rig.Vm.Probe();
    Assert.True(fits.OrientToImageCommand.CanExecute(null));
    fits.OrientToImageCommand.Execute(null);

    var want = SkyMath.EclipticDirection(135.6, 28.1);
    Assert.Equal(want, rig.Probe.Locked);
    var b = rig.Svc.Find(row.Id)!;
    Near(-10.0, b.RotationDeg); // north up: the WCS sky position angle alone
    Near(960, b.CenterX);
    Near(540, b.CenterY);
    Assert.Equal(string.Empty, fits.StatusMessage);
  }

  [Fact]
  public void MatchCameraToImage_IsTheRowsCommand()
  {
    var rig = new Rig();
    var row = rig.Create("/data/lt_67p.fits");
    Assert.Same(row.MatchCameraToImageCommand, row.Fits!.MatchCameraToImageCommand);
    row.Fits.MatchCameraToImageCommand.Execute(null);
    Assert.Equal(["match 364.416″"], rig.Probe.Log); // 1200 px × 0.30368″/px
    // the field's height at the comet: 2·D·tan(α / 2)
    double angle = 1200 * 0.30368 * DesktopBillboardGeometry.ArcsecToRad;
    Near(2 * rig.Probe.CometDistanceKm!.Value * Math.Tan(angle / 2), rig.Probe.MatchedKm, 1e-6);
  }

  /// Set up comparison: Earth observer → seek → site → match → rotate → centre → lock, in order.
  /// Match aims at the comet (tracking) and the camera is never re-aimed at the WCS centre: the
  /// Liverpool 67P stack is comet-aligned and its WCS is night 1's, 633″ (1.7 fields) off the comet
  /// at the averaged epoch, so a look at the WCS centre would lose the comet. The billboard is sized
  /// by the image → viewport conversion (here the viewport's height), turned by the sky position
  /// angle alone (north up) and centred on the comet.
  [Fact]
  public async Task SetUpComparison_KeepsTheCometCentred_RotatesFromTheWcs_AndLocksTheViewport()
  {
    var rig = new Rig();
    var row = rig.Create("/data/lt_67p.fits");
    rig.Probe.IsEarthObserver = false;
    rig.Probe.ArcsecPerPx = null; // outside the Earth observer
    rig.Vm.Probe();
    Assert.True(row.Fits!.SetUpComparisonCommand.CanExecute(null));
    await row.Fits.SetUpComparisonCommand.ExecuteAsync(null);
    rig.Fake.Drain();

    Assert.Equal(["enter", "seek", "site", "match 364.416″", "lock True"], rig.Probe.Log);
    Assert.Null(rig.Probe.Locked); // no free look at the WCS centre
    Assert.True(rig.Probe.IsComparisonLocked);
    var b = rig.Svc.Find(row.Id)!;
    Near(1000, b.HeightPx); // 364.416″ over the matched viewport's 0.364416″/px
    Near(1000, b.WidthPx);
    Assert.True(b.PhysicalSizeBound);
    Near(700, b.CenterX); // on the comet, not on the viewport centre (960, 540)
    Near(300, b.CenterY);
    Near(-10.0, b.RotationDeg);
    Assert.Equal("Comparison set up; the viewport is locked.", row.Fits.StatusMessage);
  }

  // ── sky maths ──

  [Fact]
  public void SkyMath_TdbParts_AndEclipticDirection()
  {
    Assert.Equal(((short)0, 0UL), SkyMath.TdbMicrosToParts(0));
    Assert.Equal(((short)-1, (ulong)(SkyMath.NanosecondsPerCentury - 1000)), SkyMath.TdbMicrosToParts(-1));
    long oneCenturyUs = SkyMath.NanosecondsPerCentury / 1000;
    Assert.Equal(((short)1, 5000UL), SkyMath.TdbMicrosToParts(oneCenturyUs + 5));
    var x = SkyMath.EclipticDirection(0.0, 0.0);
    Near(1, x.X, 1e-12);
    // the ecliptic north pole (RA 270°, Dec 90° − ε) is the world +Z
    double eps = SkyMath.ObliquityRad * 180.0 / Math.PI;
    var pole = SkyMath.EclipticDirection(270.0, 90.0 - eps);
    Near(1, pole.Z, 1e-12);
  }

  /// At the vernal equinox the ecliptic north pole lies ε west of celestial north (PA −ε, i.e. ε
  /// clockwise of north on a sky shown east-left). The view's up is that pole, so celestial north
  /// sits ε counter-clockwise of screen up and an image whose +Y points north turns ε
  /// counter-clockwise: rotation (clockwise) = ψ − θ = −ε.
  [Fact]
  public void BillboardRotation_IsTheViewUpPositionAngleMinusTheImageRotation()
  {
    double eps = SkyMath.ObliquityRad * 180.0 / Math.PI;
    Near(0, SkyMath.PositionAngleDeg(10, 20, SkyMath.EclipticDirection(0, 90)), 1e-9); // celestial north
    Near(-eps, SkyMath.PositionAngleDeg(0, 0, Vector3d.UnitZ), 1e-9);
    Near(-eps, SkyMath.BillboardRotationDeg(0, 0, 0, false), 1e-9);
    Near(-eps - 10, SkyMath.BillboardRotationDeg(0, 0, 10, false), 1e-9);
    Near(180 - eps, SkyMath.BillboardRotationDeg(0, 0, 0, true), 1e-9);
    // north up: the view up is celestial north (ψ = 0) whatever the site, the rotation is −θ
    Near(-10, SkyMath.BillboardRotationDeg(0, 0, 10, false, viewUpIsCelestialNorth: true), 1e-12);
    Near(-10, SkyMath.BillboardRotationDeg(135.6, 28.1, 10, true, viewUpIsCelestialNorth: true), 1e-12);
    // at the solstice (RA 90°) the ecliptic pole is due north
    Near(0, SkyMath.PositionAngleDeg(90, 23.44, Vector3d.UnitZ), 1e-9);
  }

  // ── persistence ──

  [Fact]
  public void FitsStretch_RoundTripsThroughTheSaveIndex_AndTheDefaultIsNotWritten()
  {
    var s = new FitsStretch(FitsStretchMode.Manual, FitsTransfer.Log, -10.5, 2000);
    var b = new DesktopBillboard(Guid.NewGuid(), "/a.fits", 1, 2, 300, 300) { FitsStretch = s };
    var json = SaveIndexIo.ToJson(new SaveIndex { Billboards = [SaveIndexMapping.ToDto(b)] });
    Assert.Contains("\"mode\": \"manual\"", json);
    Assert.Contains("\"transfer\": \"log\"", json);
    var back = System.Text.Json.JsonSerializer.Deserialize(json, SaveIndexJsonContext.Default.SaveIndex)!;
    Assert.Equal(b, SaveIndexMapping.FromDto(back.Billboards[0]));

    Assert.Null(SaveIndexMapping.ToDto(b with { FitsStretch = FitsStretch.Default }).FitsStretch);
    Assert.Null(SaveIndexMapping.FromDto(new BillboardDto { Id = b.Id, TextureKey = "/a.fits" }).FitsStretch);
  }

  /// After a savefile load the engine renders the import's zscale: the billboards' stretch is
  /// applied again, once, and again after a re-import under a new asset id.
  [Fact]
  public void SavedStretch_IsReappliedAfterALoad_AndAfterAReimport()
  {
    var rig = new Rig();
    var s = new FitsStretch(FitsStretchMode.Percentile, FitsTransfer.Log, 2, 98);
    var saved = new DesktopBillboard(Guid.NewGuid(), "/data/lt_67p.fits", 1, 2, 300, 300) { FitsStretch = s };
    var dto = SaveIndexMapping.ToDto(saved);
    rig.Svc.Restore([SaveIndexMapping.FromDto(dto)]);
    rig.Fake.Drain();
    rig.Fake.Runtime.Verify(r => r.SetFitsStretch(FitsId, 1u, 1u, 2.0, 98.0), Times.Once);
    Assert.Equal([FitsId], rig.TexelsChanged);
    // the row shows the saved stretch
    var fits = rig.Vm.Items.Single().Fits!;
    Assert.Equal((FitsStretchMode.Percentile, FitsTransfer.Log, "2", "98"),
      (fits.StretchMode, fits.Transfer, fits.Param0Text, fits.Param1Text));
    // unrelated edits do not re-stretch
    rig.Svc.Update(saved.Id, x => x with { CenterX = 50 });
    rig.Fake.Drain();
    rig.Fake.Runtime.Verify(r => r.SetFitsStretch(FitsId, 1u, 1u, 2.0, 98.0), Times.Once);

    // re-imported (new id): stretched again
    rig.Fake.NativeAssets.RemoveAll(a => a.Id == FitsId);
    rig.Fake.NativeAssets.Add(Tex(17, "/data/lt_67p.fits"));
    rig.Fake.Runtime.Setup(r => r.GetFitsMetadataJson(17)).Returns(LtJson.Replace("\"assetId\": 7", "\"assetId\": 17"));
    rig.Lib.Refresh();
    rig.Fake.Drain();
    rig.Fake.Drain(); // the textures reach the service on the main thread, then its background work
    rig.Fake.Runtime.Verify(r => r.SetFitsStretch(17, 1u, 1u, 2.0, 98.0), Times.Once);
  }
}
