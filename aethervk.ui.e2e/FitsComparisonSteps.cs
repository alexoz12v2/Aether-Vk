using System.Globalization;
using System.Text.RegularExpressions;
using OpenQA.Selenium.Appium;

namespace AetherVk.UI.E2E;

/// <summary>
/// The FITS comparison path, as steps a scenario calls once its comet is committed and the
/// timeline covers the observation: import the Liverpool Telescope 67P stack (FITS) and a JPEG of
/// the same field, make a billboard of each, "Set up comparison" on the FITS one, give the JPEG the
/// same plate scale and nucleus pixel, then check the header facts, that the two billboards
/// coincide and that the nucleus sits on the comet.
/// <para>Billboard rows are addressed by index (creation order): every row repeats the same
/// AutomationIds, so a row's control is the <c>row</c>-th element with that id. Geometry is read
/// from the row's full-precision <c>BillboardGeometry</c> text (the X / Y / size boxes round to
/// whole px): desktop physical px, as the app's billboards use.</para>
/// <para>The image files are data for the app only: this code passes their paths to the app's file
/// dialog and never opens them itself.</para>
/// </summary>
public static partial class FitsComparisonSteps
{
  /// <summary>The FITS stack (AETHERVK_E2E_FITS).</summary>
  public static string FitsPath =>
    Environment.GetEnvironmentVariable("AETHERVK_E2E_FITS") is { Length: > 0 } p
      ? p
      : @"Y:\Downloads\67p 20211225+26+27+28 Liverpool2m0 g+r+i+z 1x-02.fit";

  /// <summary>A JPEG of the same field (AETHERVK_E2E_JPEG).</summary>
  public static string JpegPath =>
    Environment.GetEnvironmentVariable("AETHERVK_E2E_JPEG") is { Length: > 0 } p
      ? p
      : @"Y:\Downloads\Image20261027165736.jpg";

  /// <summary>The stack's plate scale, ″ per image px (SECPIX / √|det CD|).</summary>
  public const double ArcsecPerPx = 0.303677;

  /// <summary>Position angle of image up, degrees east of north.</summary>
  public const double RotationDegEofN = 0.136;

  /// <summary>Mid-exposure, UTC (DATE-OBS, "averaged").</summary>
  public const string DateMidUtc = "2021-12-27T00:14:47";

  /// <summary>The nucleus in the stack, image px (x right, y down, pixel centres at whole numbers):
  /// FITS (594.1, 595.7), rows bottom-up, 1200 rows.</summary>
  public const double NucleusX = 593.1, NucleusY = 604.3;

  private static readonly TimeSpan SetUpTimeout = TimeSpan.FromMinutes(2);

  /// <summary>
  /// The whole path: FITS and JPEG billboards (rows <paramref name="fitsRow"/> and the next),
  /// comparison set up on the FITS, the JPEG made like it, then every assertion.
  /// </summary>
  public static void Run(AppSession s, int fitsRow = 0)
  {
    ImportImage(s, FitsPath);
    ImportImage(s, JpegPath);
    // a new billboard opens at the desktop's centre, a window over the app's own controls: each
    // stays hidden until it is placed over the viewport, so no click lands on it
    int fits = CreateBillboard(s, FitsPath);
    SetVisible(s, fits, false);
    int jpeg = CreateBillboard(s, JpegPath);
    SetVisible(s, jpeg, false);
    Assert.Equal(fitsRow, fits);
    SetUpComparison(s, fits);
    SetVisible(s, fits, true);
    AssertFitsFacts(s);
    AssertNucleusDetected(s, fits);
    SetUpJpegLike(s, ArcsecPerPx, NucleusX, NucleusY, jpeg, fits);
    // each on the comet first (the physical claim), then the two on each other
    AssertNucleusOnComet(s, fits);
    AssertNucleusOnComet(s, jpeg);
    AssertBillboardsCoincide(s, fits, jpeg);
    // interchangeable: the JPEG shown, then the FITS again (the saved state shows the FITS)
    SetVisible(s, fits, false);
    SetVisible(s, jpeg, true);
    Thread.Sleep(1000);
    s.Desktop.SaveScreenshot(Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "comparison-jpeg.png"));
    SetVisible(s, jpeg, false);
    SetVisible(s, fits, true);
  }

  // ── steps ──

  /// <summary>Imports a texture through the Imports tab's file dialog.</summary>
  public static void ImportImage(AppSession s, string path)
  {
    s.SelectTab("ImportsTab");
    s.Click("ImportButton");
    s.AnswerFileDialog(path);
  }

  /// <summary>Creates a billboard of the imported <paramref name="path"/> (Settings ›
  /// Billboards); returns its row index.</summary>
  public static int CreateBillboard(AppSession s, string path)
  {
    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    int before = Rows(s);
    var stem = Path.GetFileNameWithoutExtension(path);
    Ui.WaitFor(() => s.TryFind("BillboardTexturePicker"), what: "the texture picker");
    s.Choose("BillboardTexturePicker", t => t.Contains(stem, StringComparison.OrdinalIgnoreCase));
    s.Click("CreateBillboardButton");
    Ui.WaitUntil(() => Rows(s) == before + 1, what: $"the billboard row of {stem}");
    return before;
  }

  /// <summary>
  /// "Set up comparison" on the FITS row (Earth observer → seek → site → match the field → rotate
  /// → nucleus on the comet → lock), waiting for its outcome.
  /// </summary>
  public static void SetUpComparison(AppSession s, int row = 0)
  {
    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    s.Expand("BillboardFitsSection", "BillboardSetUpComparison");
    s.Click("BillboardSetUpComparison");
    var status = Ui.WaitFor(() => s.TryFind("BillboardFitsStatus") is { } t
      && t.Text.Contains("Comparison set up", StringComparison.Ordinal) ? t : null, SetUpTimeout, "the comparison set-up");
    Assert.DoesNotContain("without", status.Text, StringComparison.Ordinal); // every step ran
    WaitForSettledGeometry(s, row);
  }

  /// <summary>
  /// Makes the JPEG row like the FITS one by hand: the plate scale and nucleus pixel typed, the
  /// FITS billboard's rotation, the physical size applied at the viewport's scale (bound), and
  /// the nucleus centred on the comet.
  /// </summary>
  public static void SetUpJpegLike(AppSession s, double arcsecPerPx, double nucleusX, double nucleusY, int row = 1, int fitsRow = 0)
  {
    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    // the plate-scale box takes dot decimals (invariant); the numeric boxes the current culture
    s.TypeInto(At(s, "BillboardArcsecPerPx", row), arcsecPerPx.ToString("R", CultureInfo.InvariantCulture));
    SetNumericAt(s, "BillboardNucleusX", row, nucleusX);
    SetNumericAt(s, "BillboardNucleusY", row, nucleusY);
    Ui.WaitUntil(() => ReadGeometry(s, row).Nucleus is var n
      && Math.Abs(n.X - nucleusX) < 1e-3 && Math.Abs(n.Y - nucleusY) < 1e-3, what: "the JPEG nucleus pixel");

    double rotation = ReadGeometry(s, fitsRow).RotationDeg;
    s.TypeInto(At(s, "BillboardRotation.Input", row), Ui.Num(rotation));
    Ui.WaitUntil(() => Math.Abs(ReadGeometry(s, row).RotationDeg - rotation) <= 0.001, what: "the JPEG rotation");

    ClickAt(s, "BillboardApplyPhysicalSize", row);
    ClickAt(s, "BillboardCenterOnComet", row);
    WaitForSettledGeometry(s, row);
  }

  // ── assertions ──

  /// <summary>The FITS header facts: plate scale, rotation, handedness, mid-exposure, and the
  /// image → sky → viewport conversion string.</summary>
  public static void AssertFitsFacts(AppSession s, int row = 0)
  {
    s.Expand("BillboardFitsSection", "FitsFact_Scale");
    var scale = s.Find("FitsFact_Scale").Text;
    Assert.Contains("″/px", scale);
    Assert.Equal(ArcsecPerPx, Invariant(scale), 6);
    Assert.InRange(Invariant(s.Find("FitsFact_Rotation").Text), RotationDegEofN - 0.01, RotationDegEofN + 0.01);
    Assert.StartsWith("no", s.Find("FitsFact_Mirrored").Text, StringComparison.Ordinal);
    Assert.StartsWith(DateMidUtc, s.Find("FitsFact_DateMid").Text, StringComparison.Ordinal);

    // "image 0.3037″/px × 1200 px = 364.4″ (123300 km) → viewport 0.4012″/px → 908 px · 1 image px = 0.757 desktop px"
    var c = ConversionRegex().Match(At(s, "BillboardConversion", row).Text);
    Assert.True(c.Success, "no image → sky → viewport conversion shown");
    Assert.Equal(ArcsecPerPx, Num(c.Groups["scale"].Value), 4);
    Assert.Equal(1200, Num(c.Groups["px"].Value));
    Assert.Equal(1200 * ArcsecPerPx, Num(c.Groups["arcsec"].Value), 1);
    Assert.True(Num(c.Groups["km"].Value) > 0, "no km at the comet");
    // the billboard's height is the angle over the viewport's scale, and the px ratio its quotient
    var g = ReadGeometry(s, row);
    Assert.InRange(Num(c.Groups["vpx"].Value), g.Height - 1, g.Height + 1);
    Assert.InRange(Num(c.Groups["ratio"].Value), g.Height / g.ImageHeight - 6e-4, g.Height / g.ImageHeight + 6e-4);
  }

  /// <summary>The FITS import found the nucleus near the known pixel (not the image centre).</summary>
  public static void AssertNucleusDetected(AppSession s, int row = 0, double tolerancePx = 1.0)
  {
    var n = ReadGeometry(s, row).Nucleus;
    Assert.InRange(n.X, NucleusX - tolerancePx, NucleusX + tolerancePx);
    Assert.InRange(n.Y, NucleusY - tolerancePx, NucleusY + tolerancePx);
  }

  /// <summary>Two billboards drawn on top of each other: centres within 1 px, sizes within 1 px,
  /// rotations within 0.01°.</summary>
  public static void AssertBillboardsCoincide(AppSession s, int a, int b)
  {
    var ga = ReadGeometry(s, a);
    var gb = ReadGeometry(s, b);
    double dc = Math.Sqrt((ga.CenterX - gb.CenterX) * (ga.CenterX - gb.CenterX) + (ga.CenterY - gb.CenterY) * (ga.CenterY - gb.CenterY));
    Assert.True(dc <= 1.0, $"centres {dc:F2} px apart: {ga} / {gb}");
    Assert.InRange(gb.Width, ga.Width - 1, ga.Width + 1);
    Assert.InRange(gb.Height, ga.Height - 1, ga.Height + 1);
    Assert.InRange(gb.RotationDeg, ga.RotationDeg - 0.01, ga.RotationDeg + 0.01);
  }

  /// <summary>
  /// The row's nucleus pixel, mapped through its centre, size and rotation (recomputed here, not
  /// the app's own "→ desktop" figure), lies within <paramref name="tolerancePx"/> of the comet on
  /// screen.
  /// </summary>
  public static void AssertNucleusOnComet(AppSession s, int row, double tolerancePx = 2.0)
  {
    var g = ReadGeometry(s, row);
    var nucleus = g.NucleusOnDesktop();
    var app = g.NucleusDesktop;
    Assert.True(Math.Abs(nucleus.X - app.X) < 0.01 && Math.Abs(nucleus.Y - app.Y) < 0.01,
      $"the app maps the nucleus to {app}, the test to {nucleus}");
    var comet = CometScreenPx(s);
    double d = Math.Sqrt((nucleus.X - comet.X) * (nucleus.X - comet.X) + (nucleus.Y - comet.Y) * (nucleus.Y - comet.Y));
    Assert.True(d <= tolerancePx, $"nucleus at {nucleus}, comet at {comet}: {d:F2} px");
  }

  // ── reading ──

  /// <summary>A billboard row's geometry (desktop physical px, degrees clockwise on screen).</summary>
  public sealed record BillboardGeometry(
    double CenterX, double CenterY, double Width, double Height, double RotationDeg,
    int ImageWidth, int ImageHeight, (double X, double Y) Nucleus, (double X, double Y) NucleusDesktop)
  {
    /// <summary>
    /// The nucleus pixel on the desktop: image px (centres at whole numbers) → offset from the
    /// image centre ((W − 1) / 2, (H − 1) / 2) × desktop px per image px → turned clockwise on
    /// screen (y down) by the rotation θ: (dx cos θ − dy sin θ, dx sin θ + dy cos θ) → + centre.
    /// </summary>
    public (double X, double Y) NucleusOnDesktop()
    {
      double dx = (Nucleus.X - (ImageWidth - 1) / 2.0) * Width / ImageWidth;
      double dy = (Nucleus.Y - (ImageHeight - 1) / 2.0) * Height / ImageHeight;
      double t = RotationDeg * Math.PI / 180.0, c = Math.Cos(t), sn = Math.Sin(t);
      return (CenterX + dx * c - dy * sn, CenterY + dx * sn + dy * c);
    }
  }

  /// <summary>Parses the row's <c>BillboardGeometry</c> text (invariant dot decimals).</summary>
  public static BillboardGeometry ReadGeometry(AppSession s, int row)
  {
    var text = At(s, "BillboardGeometry", row).Text;
    var m = GeometryRegex().Match(text);
    if (!m.Success)
      throw new FormatException($"unexpected billboard geometry \"{text}\"");
    double G(string name) => Num(m.Groups[name].Value);
    return new BillboardGeometry(G("cx"), G("cy"), G("w"), G("h"), G("rot"), (int)G("iw"), (int)G("ih"),
      (G("nx"), G("ny")), (G("dx"), G("dy")));
  }

  /// <summary>
  /// The comet on the desktop, physical px: the "· screen (x, y)" the Earth observer HUD appends
  /// (viewport px from its top-left corner, offset by the viewport's place on the screen) when it
  /// is there; else the viewport's centre, where comet tracking keeps the comet.
  /// </summary>
  public static (double X, double Y) CometScreenPx(AppSession s)
  {
    var vp = s.Find("ViewportHost");
    // the HUD is in the overlay window, a separate top-level window: search the desktop
    var hud = s.Desktop.FindElementsByAccessibilityId("EarthObserverStatus").FirstOrDefault()?.Text ?? string.Empty;
    var m = ScreenRegex().Match(hud.Replace('−', '-'));
    if (m.Success)
      return (vp.Location.X + Ui.ParseNum(m.Groups["x"].Value), vp.Location.Y + Ui.ParseNum(m.Groups["y"].Value));
    return (vp.Location.X + vp.Size.Width / 2.0, vp.Location.Y + vp.Size.Height / 2.0);
  }

  // ── helpers ──

  /// <summary>Shows or hides the <paramref name="row"/>-th billboard (its Visible check box).</summary>
  public static void SetVisible(AppSession s, int row, bool visible)
  {
    s.SelectTab("SettingsTab");
    s.Expand("BillboardsSection", "CreateBillboardButton");
    var box = At(s, "BillboardVisible", row);
    if (box.Selected != visible)
      box.RealClick();
    Ui.WaitUntil(() => At(s, "BillboardVisible", row).Selected == visible,
      what: $"billboard row {row} {(visible ? "shown" : "hidden")}");
  }

  private static int Rows(AppSession s) => s.App.FindElementsByAccessibilityId("BillboardGeometry").Count;

  /// <summary>The <paramref name="row"/>-th element with <paramref name="automationId"/>.</summary>
  private static AppiumWebElement At(AppSession s, string automationId, int row) =>
    s.ScrollIntoView(Ui.WaitFor(() => s.App.FindElementsByAccessibilityId(automationId).ElementAtOrDefault(row),
      what: $"'{automationId}' of billboard row {row}"));

  private static void ClickAt(AppSession s, string automationId, int row)
  {
    var button = At(s, automationId, row);
    Ui.WaitUntil(() => button.Enabled, what: $"'{automationId}' of row {row} to be enabled");
    button.RealClick();
    Thread.Sleep(300);
  }

  private static void SetNumericAt(AppSession s, string automationId, int row, double value)
  {
    var box = At(s, automationId, row);
    s.TypeInto(box.FindElementsByClassName("TextBox").FirstOrDefault() ?? box, Ui.Num(value));
  }

  /// <summary>Waits until two reads of the row's geometry 0.75 s apart agree (the viewport probe
  /// runs every 250 ms and a bound billboard follows it).</summary>
  public static void WaitForSettledGeometry(AppSession s, int row)
  {
    var last = ReadGeometry(s, row);
    Ui.WaitUntil(() =>
    {
      Thread.Sleep(750);
      var now = ReadGeometry(s, row);
      bool same = Math.Abs(now.CenterX - last.CenterX) < 0.01 && Math.Abs(now.CenterY - last.CenterY) < 0.01
        && Math.Abs(now.Height - last.Height) < 0.01 && Math.Abs(now.RotationDeg - last.RotationDeg) < 1e-4;
      last = now;
      return same;
    }, TimeSpan.FromSeconds(30), $"billboard row {row} to settle");
  }

  private static double Num(string text) => double.Parse(text, NumberStyles.Float, CultureInfo.InvariantCulture);

  /// <summary>The first number of an invariant-formatted fact ("0.303677″/px (CD)").</summary>
  private static double Invariant(string text) => Num(FirstNumberRegex().Match(text.Replace('−', '-')).Value);

  private const string N = @"-?\d+(?:\.\d+)?";

  [GeneratedRegex(@"centre (?<cx>" + N + "), (?<cy>" + N + @") px · size (?<w>" + N + ") × (?<h>" + N + @") px · rotation (?<rot>"
    + N + @")° · image (?<iw>\d+) × (?<ih>\d+) px · nucleus (?<nx>" + N + "), (?<ny>" + N + @") px(?: \(image centre\))? → desktop (?<dx>"
    + N + "), (?<dy>" + N + ") px")]
  private static partial Regex GeometryRegex();

  [GeneratedRegex(@"image (?<scale>" + N + @")″/px × (?<px>\d+) px = (?<arcsec>" + N + @")″ \((?<km>" + N + @") km\) → viewport "
    + N + @"″/px → (?<vpx>" + N + @") px · 1 image px = (?<ratio>" + N + ") desktop px")]
  private static partial Regex ConversionRegex();

  [GeneratedRegex(@"screen \((?<x>-?\d+(?:[.,]\d+)?)[,;] (?<y>-?\d+(?:[.,]\d+)?)\)")]
  private static partial Regex ScreenRegex();

  [GeneratedRegex(N)]
  private static partial Regex FirstNumberRegex();
}
