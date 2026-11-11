using System.Diagnostics;
using System.Globalization;
using System.IO.Compression;
using System.Text.Json;
using AetherVk.Logic.Automation;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Services.ImageExport;
using AetherVk.Logic.Utils;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// The Classic / Compact layouts on the real application window: the Layout menu's radio entries
/// (real clicks), the main viewport's native host kept through every switch, the compact panel's
/// cards driving the same view models as the classic tabs, the layout stored in a save, and the
/// viewport's native window and overlay following it into the right-hand column.
/// </summary>
[Collection("X11")]
public sealed class CompactLayoutX11Tests
{
  private readonly ITestOutputHelper _output;

  public CompactLayoutX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  private void Report(string s)
  {
    _output.WriteLine(s);
    Console.Error.WriteLine($"[CompactLayoutX11Tests] {s}");
  }

  private void Step(string s) => Report($"── {s}");

  /// <summary>
  /// Proves that switching the layout never rebuilds the main 3D view: ten Classic → Compact →
  /// Classic round trips through the real Layout menu keep the same <c>VulkanViewportControl</c>
  /// instance, its native X window (same XID, still known to the X server) and its engine
  /// (same presentation engine id, never initialised again); there is exactly one ViewportHost in
  /// the window at every step; after each switch the main engine still answers a capture and its
  /// presented-frame count keeps increasing; and the radio entries' checks match the layout shown.
  /// </summary>
  [SkippableFact]
  public void LayoutToggle10x_KeepsEngineViewportAndRenders()
  {
    using var rig = new AppRig();
    var runtime = CompactX11.Runtime(rig);
    ulong pe = CompactX11.RequireMainPe(rig);
    var host = CompactX11.MainHost(rig);
    var hostVm = CompactX11.HostVm(host);
    nint xid = Ui(() => hostVm.NativeHandle);
    int inits = Ui(() => hostVm.InitializeCount);
    Report($"start: main PE {pe}, XID {Hex(xid)}, initializations {inits}, frames {runtime.ViewportFrameCount(pe)}");
    Assert.NotEqual((nint)0, xid);
    Assert.Equal((true, false), CompactX11.LayoutChecks(rig));
    Assert.Equal(1, CompactX11.ViewportHostCount(rig));

    for (int i = 1; i <= 10; i++)
    {
      foreach (bool compact in new[] { true, false })
      {
        string when = $"switch {i} to {(compact ? "Compact" : "Classic")}";
        ulong framesBefore = runtime.ViewportFrameCount(pe);
        CompactX11.SetLayoutByMenu(rig, compact);

        var checks = CompactX11.LayoutChecks(rig);
        Assert.True(checks == (!compact, compact), $"{when}: the radio entries read Classic {checks.Classic}, Compact {checks.Compact}");
        var now = CompactX11.MainHost(rig);
        Assert.True(ReferenceEquals(now, host), $"{when}: the main viewport's native control was replaced");
        int hosts = CompactX11.ViewportHostCount(rig);
        Assert.True(hosts == 1, $"{when}: {hosts} ViewportHost controls in the window");
        nint xidNow = Ui(() => hostVm.NativeHandle);
        Assert.True(xidNow == xid, $"{when}: the native window changed, {Hex(xid)} → {Hex(xidNow)}");
        Assert.True(CompactX11.XidAlive(xid), $"{when}: the X server lost window {Hex(xid)}");
        Assert.True(Ui(() => hostVm.InitializeCount) == inits, $"{when}: the native window was handed over again ({Ui(() => hostVm.InitializeCount)} times)");
        Assert.True(runtime.PresentationEngineId == pe, $"{when}: the main engine changed, {pe} → {runtime.PresentationEngineId}");
        Assert.True(Ui(() => hostVm.PresentationEngineId) == pe, $"{when}: the control's engine is {Ui(() => hostVm.PresentationEngineId)}, not {pe}");

        var sw = Stopwatch.StartNew();
        var image = CompactX11.RequireCapture(runtime, pe, when);
        long ms = sw.ElapsedMilliseconds;
        Assert.True(WaitFor(() => runtime.ViewportFrameCount(pe) > framesBefore, 10_000),
          $"{when}: the main frame count stayed at {runtime.ViewportFrameCount(pe)} (was {framesBefore})");
        Report($"{when}: capture {image.Width}×{image.Height} in {ms} ms, frames {framesBefore} → {runtime.ViewportFrameCount(pe)}, "
          + $"XID mapped {CompactX11.XidViewable(xid)}");
      }
    }
  }

  /// <summary>
  /// A copy of a test-asset FITS with one data byte changed: the asset library merges an import
  /// by content, and the review save already holds this header fixture's raster, so the unchanged
  /// file would add no texture (and no overlay).
  /// </summary>
  private static string UniqueFitsCopy(string source, string target)
  {
    var bytes = File.ReadAllBytes(source);
    int end = -1;
    for (int card = 0; (card + 1) * 80 <= bytes.Length; card++)
    {
      if (bytes[card * 80] == (byte)'E' && bytes[card * 80 + 1] == (byte)'N' && bytes[card * 80 + 2] == (byte)'D'
          && bytes[card * 80 + 3] == (byte)' ')
      {
        end = card;
        break;
      }
    }
    Assert.True(end >= 0, $"{source}: no END card");
    int dataStart = ((end + 1) * 80 + 2879) / 2880 * 2880;
    Assert.True(dataStart + 2 < bytes.Length, $"{source}: no data after the header ({bytes.Length} bytes, data at {dataStart})");
    bytes[dataStart + 1] ^= 0x01;
    Directory.CreateDirectory(Path.GetDirectoryName(target)!);
    File.WriteAllBytes(target, bytes);
    return target;
  }

  private static void WaitExport(AppRig rig, string modelPath, string what) =>
    Assert.True(WaitFor(() => File.Exists(modelPath) && CompactX11.Readable(modelPath), 90_000),
      $"{what}: {modelPath} not written; status '{Ui(() => rig.Vm.CompactPanel.ExportStatus)}', dialogs: {string.Join(" | ", rig.Dialogs.Requests)}");

  /// <summary>
  /// Proves the compact panel is a complete front end over the classic tabs' state, driven only by
  /// its own controls (looked up by <see cref="CompactAutomationIds"/>): on the review save, the
  /// observation period (image date, rotations) is typed and applied, a jet's latitude is typed,
  /// a FITS image is imported with "Import and overlay" (scripted open dialog) and becomes a new
  /// billboard, and two PNGs are exported with the panel's export button (scripted save dialog).
  /// Both exports are valid PNGs of the main viewport's pixel size. Between them the panel's View
  /// card switches the main view to Orbit, so the scenario changes on purpose and the two model
  /// PNGs must differ in a large part of the frame (more than 5 % of the pixels). Finally the
  /// classic Comet, Model and Settings tabs are the very instances the panel edited and show the
  /// typed values.
  /// </summary>
  [SkippableFact]
  public void CompactPanel_FullReviewerFlow_67P()
  {
    var fitsSource = AetherVk.TestAssets.TestAssets.Require("fits_lt_67p_header_4x3.fits");
    using var rig = new AppRig();
    var runtime = CompactX11.Runtime(rig);
    ulong pe = CompactX11.RequireMainPe(rig);
    CompactX11.OpenReviewSave(rig, Step);
    var cometTab = rig.Tab<CometTabViewModel>();
    var modelTab = rig.Tab<ModelTabViewModel>();
    Assert.True(WaitFor(() => Ui(() => (modelTab.Jets?.Count ?? 0) > 0 && modelTab.Jets!.All(j => j.NativePsId != 0)), 60_000),
      "the review save's jets in the engine");

    Step("switch to Compact");
    CompactX11.SetLayoutByMenu(rig, compact: true);
    var panel = Ui(() => rig.Vm.CompactPanel);
    Assert.True(Ui(() => panel.IsActive), "the compact panel is active");
    rig.Require<Control>(CompactAutomationIds.CompactPanel);
    Assert.True(ReferenceEquals(Ui(() => panel.Comet), cometTab), "the panel's Comet card is not the Comet tab's view model");
    Assert.True(ReferenceEquals(Ui(() => panel.Model), modelTab), "the panel's jet cards are not the Model tab's view model");
    var range = Ui(() => cometTab.Range);

    // ── the observation period ──────────────────────────────────────────────────────────────
    Step("observation period: image date and rotations, Apply");
    string dateBefore = Ui(() => range.ObservationEpoch);
    int rotationsBefore = Ui(() => range.ModelRotations);
    Report($"before: image date '{dateBefore}', rotations {rotationsBefore}, start '{Ui(() => range.StartEpoch)}', end '{Ui(() => range.EndEpoch)}'");
    var baseDate = DateTime.TryParseExact(dateBefore, "yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture,
      DateTimeStyles.AdjustToUniversal | DateTimeStyles.AssumeUniversal, out var parsed)
      ? parsed
      : new DateTime(2021, 12, 28, 0, 0, 0, DateTimeKind.Utc);
    // six hours before the save's date: inside its ephemeris, and a different proposal
    string date = baseDate.AddHours(-6).ToString("yyyy-MM-ddTHH:mm:ss", CultureInfo.InvariantCulture);
    int rotations = Math.Clamp(rotationsBefore + 1, 1, 1000);
    CompactX11.TypeInto(rig, rig.Require<TextBox>(CompactAutomationIds.PeriodImageDate), date);
    CompactX11.TypeInto(rig, CompactX11.NumericText(rig, CompactAutomationIds.PeriodRotations), rotations.ToString(CultureInfo.InvariantCulture));
    Assert.True(WaitFor(() => Ui(() => range.ObservationEpoch) == date, 5_000), $"the image date reads '{Ui(() => range.ObservationEpoch)}', typed '{date}'");
    Assert.True(WaitFor(() => Ui(() => range.ModelRotations) == rotations, 5_000), $"the rotations read {Ui(() => range.ModelRotations)}, typed {rotations}");
    string preview = Ui(() => range.ModelRangePreviewText);
    Report($"preview: {preview}");
    CompactX11.ClickInView(rig, rig.Require<Button>(CompactAutomationIds.PeriodApply));
    Assert.True(WaitFor(() => Ui(() => range.EndEpoch).StartsWith(date[..10], StringComparison.Ordinal), 10_000),
      $"Apply: the proposed end is '{Ui(() => range.EndEpoch)}', expected the image date {date}; error '{Ui(() => range.ErrorMessage)}'");
    Assert.False(Ui(() => range.HasError), $"Apply reported an error: {Ui(() => range.ErrorMessage)}");
    Report($"applied: start '{Ui(() => range.StartEpoch)}', end '{Ui(() => range.EndEpoch)}', range locked {Ui(() => range.IsRangeLocked)}");

    // ── a jet value ─────────────────────────────────────────────────────────────────────────
    Step("jet: select #1, type its latitude");
    var jetList = rig.Require<ListBox>(CompactAutomationIds.JetsList);
    if (Ui(() => modelTab.SelectedJet) is null)
    {
      var firstJet = Ui(() => modelTab.Jets![0]);
      ListBoxItem? row = null;
      Ui(() => jetList.BringIntoView());
      Settle(400);
      Assert.True(WaitFor(() => (row = Ui(() => jetList.GetVisualDescendants().OfType<ListBoxItem>()
        .FirstOrDefault(r => ReferenceEquals(r.DataContext, firstJet) && r.IsVisible))) is not null, 5_000), "jet #1's row in the panel");
      CompactX11.ClickInView(rig, row!);
      Assert.True(WaitFor(() => Ui(() => modelTab.SelectedJet) is not null, 5_000), "jet #1 selected from the panel");
    }
    var jet = Ui(() => modelTab.SelectedJet!);
    float latBefore = Ui(() => jet.LatitudeDeg);
    float lat = latBefore <= 80f ? latBefore + 5f : latBefore - 5f;
    string latText = lat.ToString("0.##", CultureInfo.InvariantCulture);
    CompactX11.TypeInto(rig, CompactX11.NumericText(rig, CompactAutomationIds.JetLatitude), latText);
    Assert.True(WaitFor(() => Math.Abs(Ui(() => jet.LatitudeDeg) - lat) < 0.01f, 10_000),
      $"jet #{Ui(() => jet.DisplayIndex)} latitude reads {Ui(() => jet.LatitudeDeg)}, typed {latText} (was {latBefore})");
    Report($"jet #{Ui(() => jet.DisplayIndex)}: latitude {latBefore} → {Ui(() => jet.LatitudeDeg)}");

    // ── a FITS image, imported and overlaid ─────────────────────────────────────────────────
    Step("FITS: Import and overlay");
    var settingsTab = Ui(() => panel.Settings!);
    var billboardItems = Ui(() => settingsTab.Billboards!.Items);
    int billboardsBefore = Ui(() => billboardItems.Count);
    var fits = UniqueFitsCopy(fitsSource, Path.Combine(rig.Root, "fits", "review_67p_overlay_4x3.fits"));
    rig.Dialogs.Answer(fits);
    CompactX11.ClickInView(rig, rig.Require<Button>(CompactAutomationIds.FitsImportAndOverlay));
    Assert.True(WaitFor(() => Ui(() => billboardItems.Count) > billboardsBefore, 60_000),
      $"Import and overlay added no billboard ({Ui(() => billboardItems.Count)}); status '{Ui(() => panel.FitsStatus)}', "
      + $"dialogs: {string.Join(" | ", rig.Dialogs.Requests)}");
    Assert.True(WaitFor(() => Ui(() => panel.SelectedBillboard) is not null && Ui(() => panel.FitsStatus).Length > 0, 10_000),
      $"the FITS card selected the new billboard ({Ui(() => panel.SelectedBillboard?.TextureLabel)}), status '{Ui(() => panel.FitsStatus)}'");
    var overlaid = Ui(() => panel.SelectedBillboard!);
    Assert.True(Ui(() => billboardItems.Contains(overlaid)), "the selected billboard is not in the Settings tab's list");
    Report($"overlaid: '{Ui(() => overlaid.TextureLabel)}', status '{Ui(() => panel.FitsStatus)}', billboards {billboardsBefore} → {Ui(() => billboardItems.Count)}");
    Settle(3000); // the overlay's first frames

    // ── two exports from the panel, the view changed between them ───────────────────────────
    var mainSize = CompactX11.RequireCapture(runtime, pe, "the main viewport before the exports");
    Report($"main viewport capture: {mainSize.Width}×{mainSize.Height}");

    Step("export 1 (the save's view)");
    var first = Path.Combine(rig.Root, "export", "review_67p_compact_1.png");
    var (firstModel, firstFits) = ImageExportService.FileNames(first);
    rig.Dialogs.Answer(first);
    CompactX11.ClickInView(rig, rig.Require<Button>(CompactAutomationIds.ExportButton));
    WaitExport(rig, firstModel, "export 1");

    Step("View › Orbit, export 2");
    CompactX11.ClickInView(rig, rig.Require<Button>(CompactAutomationIds.ViewOrbit));
    Assert.True(WaitFor(() => Ui(() => rig.Vm.ModeBar!.IsOrbitActive), 10_000), "the panel's Orbit button selected the Orbit view");
    Settle(3000);
    var second = Path.Combine(rig.Root, "export", "review_67p_compact_2.png");
    var (secondModel, secondFits) = ImageExportService.FileNames(second);
    rig.Dialogs.Answer(second);
    CompactX11.ClickInView(rig, rig.Require<Button>(CompactAutomationIds.ExportButton));
    WaitExport(rig, secondModel, "export 2");
    Report($"status: '{Ui(() => panel.ExportStatus)}'; composites written: {File.Exists(firstFits)}, {File.Exists(secondFits)}");

    var images = new List<(byte[] Rgba, int W, int H)>();
    foreach (var path in new[] { firstModel, secondModel })
    {
      Assert.True(CompactX11.HasPngSignature(path), $"{path} does not start with the PNG signature");
      var rgba = PngCodec.ReadRgba(path, out int w, out int h);
      Report($"{Path.GetFileName(path)}: {w}×{h}, {new FileInfo(path).Length} bytes, lit {CompactX11.LitCount(rgba, w, h)} px");
      Assert.True(Math.Abs(w - mainSize.Width) <= 2 && Math.Abs(h - mainSize.Height) <= 2,
        $"{Path.GetFileName(path)} is {w}×{h}, the main viewport {mainSize.Width}×{mainSize.Height}");
      images.Add((rgba, w, h));
    }
    Assert.Equal((images[0].W, images[0].H), (images[1].W, images[1].H));
    double changed = CompactX11.DiffFraction(images[0].Rgba, images[1].Rgba, images[0].W, images[0].H);
    Report($"the two exports differ at {changed:P1} of the pixels (the view went from the save's to Orbit)");
    Assert.True(changed > 0.05, $"the Orbit export differs from the first at only {changed:P2} of the pixels: the view change did not reach the export");

    // ── the classic tabs show the same values (same instances) ──────────────────────────────
    Step("back to Classic: the tabs show what the panel typed");
    CompactX11.SetLayoutByMenu(rig, compact: false);
    Assert.True(ReferenceEquals(rig.Tab<CometTabViewModel>(), cometTab), "the Comet tab was replaced");
    Assert.True(ReferenceEquals(rig.Tab<ModelTabViewModel>(), modelTab), "the Model tab was replaced");
    Assert.True(ReferenceEquals(rig.Tab<SettingsTabViewModel>(), settingsTab), "the Settings tab is not the one the panel used");
    Assert.Equal(date, Ui(() => cometTab.Range.ObservationEpoch));
    Assert.Equal(rotations, Ui(() => cometTab.Range.ModelRotations));
    Assert.True(Math.Abs(Ui(() => modelTab.SelectedJet!.LatitudeDeg) - lat) < 0.01f,
      $"the Model tab's jet latitude reads {Ui(() => modelTab.SelectedJet!.LatitudeDeg)}, the panel typed {latText}");
    Assert.True(Ui(() => settingsTab.Billboards!.Items.Contains(overlaid)), "the Settings tab lost the overlaid billboard");
    Assert.Equal(0, rig.Prompt.Asked);
  }

  private static JsonDocument Index(string avk)
  {
    using var zip = ZipFile.OpenRead(avk);
    using var s = zip.GetEntry(SaveIndex.FileName)!.Open();
    using var ms = new MemoryStream();
    s.CopyTo(ms);
    return JsonDocument.Parse(ms.ToArray());
  }

  private static string? UiLayoutOf(string avk)
  {
    using var doc = Index(avk);
    return doc.RootElement.TryGetProperty("ui_layout", out var v) ? v.ToString() : null;
  }

  private static bool Clean(string title) => !title.TrimEnd().EndsWith('*');

  /// <summary>
  /// Proves the layout travels with a save and with the preferences: in Compact, Save As writes
  /// <c>"ui_layout": "compact"</c> into the archive's index; File › New starts in the preferred
  /// layout (Compact, the last one chosen; then Classic once the preference says Classic);
  /// reopening the save shows Compact again; switching to Classic and saving removes the key from
  /// the index (a classic save stays byte-compatible with older ones). No unsaved-changes prompt
  /// appears at any step.
  /// </summary>
  [SkippableFact]
  public void CompactLayout_SavedAndReopened()
  {
    using var rig = new AppRig();
    CompactX11.RequireMainPe(rig);
    var avk = Path.Combine(rig.SavesDir, "layout.avk");

    Step("Compact, Save As layout.avk");
    CompactX11.SetLayoutByMenu(rig, compact: true);
    Assert.Equal(UiLayout.Compact, Ui(() => rig.Preferences.LastUiLayout));
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Save As...");
    Assert.True(WaitFor(() => File.Exists(avk) && rig.XTitle.EndsWith("— layout", StringComparison.Ordinal) && Clean(rig.XTitle), 90_000),
      $"saved, title: {rig.XTitle}");
    string? saved = UiLayoutOf(avk);
    Report($"index ui_layout after Save As in Compact: {saved ?? "(absent)"}");
    Assert.Equal("compact", saved);

    Step("File › New: the preferred layout (Compact)");
    rig.Menu("File", "New");
    Assert.True(WaitFor(() => !rig.XTitle.Contains("layout", StringComparison.Ordinal), 30_000), $"New, title: {rig.XTitle}");
    Settle(500);
    Assert.True(Ui(() => rig.Vm.IsCompactLayout), "New did not start in the preferred Compact layout");
    Assert.Equal((false, true), CompactX11.LayoutChecks(rig));

    Step("preference Classic, File › New: Classic");
    Ui(() => rig.Preferences.SetLastUiLayout(UiLayout.Classic));
    rig.Menu("File", "New");
    Assert.True(WaitFor(() => !Ui(() => rig.Vm.IsCompactLayout), 30_000), "New did not follow the preference to Classic");
    Assert.Equal((true, false), CompactX11.LayoutChecks(rig));

    Step("reopen layout.avk: Compact");
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.EndsWith("— layout", StringComparison.Ordinal), 90_000), $"reopened, title: {rig.XTitle}");
    Assert.True(WaitFor(() => Ui(() => rig.Vm.IsCompactLayout), 10_000), "the reopened save did not apply its Compact layout");
    Assert.Equal((false, true), CompactX11.LayoutChecks(rig));
    Settle(1000);
    Assert.True(Clean(rig.XTitle), $"the reopened save is dirty: {rig.XTitle}");

    Step("Classic, File › Save: the key is gone");
    CompactX11.SetLayoutByMenu(rig, compact: false);
    var mtime = File.GetLastWriteTimeUtc(avk);
    Thread.Sleep(1100); // mtime resolution on coarse filesystems
    rig.Menu("File", "Save");
    Assert.True(WaitFor(() => File.GetLastWriteTimeUtc(avk) > mtime && Clean(rig.XTitle), 90_000),
      $"saved in place: mtime {File.GetLastWriteTimeUtc(avk):O} vs {mtime:O}, title: {rig.XTitle}");
    string? after = UiLayoutOf(avk);
    Report($"index ui_layout after Save in Classic: {after ?? "(absent)"}");
    Assert.Null(after);
    Assert.Equal(UiLayout.Classic, Ui(() => rig.Preferences.LastUiLayout));
    Assert.Equal(0, rig.Prompt.Asked);
  }

  /// <summary>The control's rectangle on the screen, physical pixels.</summary>
  private static (int X, int Y, int W, int H) ScreenRect(Control c) => Ui(() =>
  {
    var tl = c.PointToScreen(new Point(0, 0));
    var br = c.PointToScreen(new Point(c.Bounds.Width, c.Bounds.Height));
    return (tl.X, tl.Y, br.X - tl.X, br.Y - tl.Y);
  });

  private static bool Near(WinInfo a, (int X, int Y, int W, int H) b, int tol = 2) =>
    Math.Abs(a.X - b.X) <= tol && Math.Abs(a.Y - b.Y) <= tol && Math.Abs(a.Width - b.W) <= tol && Math.Abs(a.Height - b.H) <= tol;

  private static string Show(WinInfo w) => $"{w.Width}×{w.Height}+{w.X}+{w.Y}";

  /// <summary>
  /// Proves the main viewport's native window really moves into the compact layout's right-hand
  /// column and back: its X geometry (xwininfo of its XID) lies inside the docking column (the
  /// work area's third column) and fills most of it in Compact, starts right of the panel's
  /// minimum width, and returns to the classic rectangle in Classic. Its overlay window (the
  /// viewport's <c>OverlayWindow</c>: labels and buttons over the 3D view) and the viewport frame
  /// the desktop billboards follow (<c>DesktopBillboardService.ViewportFrame</c>) match that
  /// geometry in both layouts. Billboard windows themselves need a save with billboards and are
  /// covered through the frame they follow; when no overlay window exists (a configuration
  /// without one) the test says so and checks the native window only.
  /// </summary>
  [SkippableFact]
  public void CompactLayout_OverlayFollowsViewport()
  {
    using var rig = new AppRig();
    CompactX11.RequireMainPe(rig);
    var hostVm = CompactX11.HostVm(CompactX11.MainHost(rig));
    nint xid = Ui(() => hostVm.NativeHandle);
    var billboards = rig.Host.Services.GetRequiredService<DesktopBillboardService>();
    var work = Ui(() => rig.Main.GetVisualDescendants().OfType<Grid>().First(g => g.Name == "WorkArea"));
    var dockColumn = Ui(() => work.Children.OfType<Control>().First(c => Grid.GetColumn(c) == 2));
    var window = XWinInfo(rig.MainXid);

    var overlays = CompactX11.OverlayXids(rig);
    Report(overlays.Count == 0
      ? "no viewport OverlayWindow in this configuration: the native window and the billboards' viewport frame are checked"
      : $"viewport overlay windows: {string.Join(", ", overlays.Select(Hex))}");

    WinInfo Stable(string phase)
    {
      WinInfo last = XWinInfo(xid);
      Assert.True(WaitFor(() =>
      {
        Thread.Sleep(300);
        var now = XWinInfo(xid);
        bool same = now == last;
        last = now;
        return same;
      }, 10_000), $"{phase}: the viewport's window kept moving (last {Show(last)})");
      return last;
    }

    void Followers(string phase, WinInfo vp)
    {
      foreach (var o in overlays)
      {
        WinInfo seen = XWinInfo(o);
        bool ok = WaitFor(() => Near(seen = XWinInfo(o), (vp.X, vp.Y, vp.Width, vp.Height)), 10_000);
        Report($"{phase}: overlay {Hex(o)} at {Show(seen)}, mapped {seen.Viewable}");
        Assert.True(ok, $"{phase}: overlay {Hex(o)} at {Show(seen)}, the viewport at {Show(vp)}");
      }
      ViewportFrame? frame = null;
      bool framed = WaitFor(() =>
      {
        frame = Ui(() => billboards.ViewportFrame);
        return frame is { } f && Math.Abs(f.Origin.X - vp.X) <= 2 && Math.Abs(f.Origin.Y - vp.Y) <= 2
          && Math.Abs(f.WidthPx - vp.Width) <= 2 && Math.Abs(f.HeightPx - vp.Height) <= 2;
      }, 10_000);
      Report($"{phase}: billboards' viewport frame {(frame is { } fr ? $"{fr.WidthPx}×{fr.HeightPx}+{fr.Origin.X}+{fr.Origin.Y}" : "unknown")}");
      Assert.True(framed, $"{phase}: the billboards' viewport frame {frame} does not match the viewport at {Show(vp)}");
    }

    Step("Classic");
    var classic = Stable("Classic");
    Report($"Classic: viewport {Show(classic)}, window {Show(window)}");
    Followers("Classic", classic);

    Step("Compact");
    CompactX11.SetLayoutByMenu(rig, compact: true);
    var panelRect = ScreenRect(rig.Require<Control>(CompactAutomationIds.CompactPanel));
    WinInfo compact = classic;
    Assert.True(WaitFor(() => (compact = Stable("Compact")) != classic, 10_000), $"Compact: the viewport stayed at {Show(classic)}");
    var column = ScreenRect(dockColumn);
    Report($"Compact: viewport {Show(compact)}, docking column {column.W}×{column.H}+{column.X}+{column.Y}, "
      + $"panel {panelRect.W}×{panelRect.H}+{panelRect.X}+{panelRect.Y}");
    Assert.True(compact.X >= window.X + MainWindow.CompactPanelMinWidth - 2,
      $"Compact: the viewport starts at x {compact.X}, inside the panel's {MainWindow.CompactPanelMinWidth} px (window x {window.X})");
    Assert.True(compact.X >= panelRect.X + panelRect.W - 2, $"Compact: the viewport (x {compact.X}) overlaps the panel (right edge {panelRect.X + panelRect.W})");
    Assert.True(compact.X >= column.X - 2 && compact.Y >= column.Y - 2
      && compact.X + compact.Width <= column.X + column.W + 2 && compact.Y + compact.Height <= column.Y + column.H + 2,
      $"Compact: the viewport {Show(compact)} is not inside the docking column {column.W}×{column.H}+{column.X}+{column.Y}");
    double fill = (double)compact.Width * compact.Height / Math.Max(1, column.W * column.H);
    Report($"Compact: the viewport fills {fill:P0} of the docking column");
    Assert.True(fill > 0.7, $"Compact: the viewport fills only {fill:P0} of the docking column (viewport-only mode)");
    Assert.True(CompactX11.XidViewable(xid), "Compact: the viewport's window is not mapped");
    Followers("Compact", compact);

    Step("Classic again");
    CompactX11.SetLayoutByMenu(rig, compact: false);
    WinInfo back = compact;
    bool returned = WaitFor(() => Near(back = Stable("Classic again"), (classic.X, classic.Y, classic.Width, classic.Height)), 10_000);
    Report($"Classic again: viewport {Show(back)}");
    Assert.True(returned, $"Classic again: the viewport is at {Show(back)}, it was at {Show(classic)}");
    Followers("Classic again", back);
  }
}
