using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.VisualTree;
using Microsoft.Extensions.DependencyInjection;
using Xunit;
using Xunit.Abstractions;
using static AetherVk.X11Tests.X11Host;

namespace AetherVk.X11Tests;

/// <summary>
/// The plane selector of a FITS cube on the real application: the committed 4 × 3 × 3 cube
/// (<c>test_assets/fits_cube_4x3x3.fits</c>, plane p = the ramp 100..1200 + 1000 p) is imported
/// into the start-up scene and becomes a billboard; its FITS panel shows "1 of 3", plane 3 is
/// typed into the selector with real X key events, and the engine's raster (copyFitsRaster through
/// the runtime service) then holds plane 2's values. Saved as an .avk, File › New, reopened: the
/// billboard comes back on plane 3 and the engine shows it again.
/// </summary>
[Collection("X11")]
public sealed class FitsPlaneX11Tests : IDisposable
{
  private readonly ITestOutputHelper _output;

  public FitsPlaneX11Tests(ITestOutputHelper output)
  {
    _output = output;
    Skip.If(SkipReason is not null, SkipReason);
    EnsureStarted();
  }

  public void Dispose()
  {
  }

  private static string CubePath() => AetherVk.TestAssets.TestAssets.Require("fits_cube_4x3x3.fits");

  private void Step(string s)
  {
    _output.WriteLine($"── {s}");
    Console.Error.WriteLine($"[FitsPlaneX11Tests] {s}");
  }

  /// <summary>The raster of plane <paramref name="p"/> in texture row order (FITS row 0 last).</summary>
  private static float[] PlaneRaster(int p)
  {
    var r = new float[12];
    for (int row = 0; row < 3; row++)
      for (int i = 0; i < 4; i++)
        r[(2 - row) * 4 + i] = 100f * (row * 4 + i + 1) + 1000f * p;
    return r;
  }

  private static bool RasterIs(INativeRuntimeService runtime, ulong assetId, int plane, out string seen)
  {
    var img = runtime.CopyFitsRaster(assetId);
    seen = img is null ? "null" : $"{img.Width}×{img.Height} [{string.Join(", ", img.Values.Take(12))}]";
    return img is { Width: 4, Height: 3 } && img.Values.Take(12).SequenceEqual(PlaneRaster(plane));
  }

  private static void ClickInView(AppRig rig, Control c)
  {
    Ui(() => c.BringIntoView());
    Settle(400);
    rig.Click(c);
  }

  private static void ExpandByClick(AppRig rig, Expander e, string what)
  {
    if (Ui(() => e.IsExpanded)) return;
    var toggle = Ui(() => e.GetVisualDescendants().OfType<ToggleButton>().First());
    ClickInView(rig, toggle);
    Assert.True(WaitFor(() => Ui(() => e.IsExpanded), 5_000), $"{what} expanded");
  }

  /// <summary>The Settings tab shown with its billboards section and the cube row's FITS panel open.</summary>
  private static void ShowFitsPanel(AppRig rig)
  {
    rig.Click(rig.Require<TextBlock>("Tab.SettingsTab"));
    ExpandByClick(rig, rig.Require<Expander>("BillboardsSection"), "Billboards section");
    ExpandByClick(rig, rig.Require<Expander>("BillboardFitsSection"), "FITS image section");
  }

  private static string? FactText(AppRig rig, string automationId) =>
    rig.Find<SelectableTextBlock>(automationId) is { } t ? Ui(() => t.Text) : null;

  /// One sequential test: the core is a process singleton and each step builds on the last.
  [SkippableFact]
  public void CubePlane_TypedInThePanel_ReachesTheEngine_AndSurvivesSaveNewReopen()
  {
    var cubePath = CubePath();
    using var rig = new AppRig();
    var runtime = rig.Host.Services.GetRequiredService<INativeRuntimeService>();
    var billboards = rig.Host.Services.GetRequiredService<DesktopBillboardService>();
    var assets = rig.Host.Services.GetRequiredService<AssetLibraryService>();
    var settings = rig.Tab<SettingsTabViewModel>();
    var avk = Path.Combine(rig.SavesDir, "cube.avk");

    // ── 1. the start-up scene: no billboards. (Not the review save: its 4 × 3 header fixture holds
    // the cube's plane-0 ramp, and the library would merge the cube into that asset by content.)
    Step("1: the start-up scene");
    Assert.True(WaitFor(() => Ui(() => billboards.Current.Length) == 0, 30_000), "no billboards at start-up");

    // ── 2. the cube imported, a billboard of it ─────────────────────────────────────────────
    Step("2: import the cube and make it a billboard");
    var before = Ui(() => assets.CurrentTextures.Select(t => t.Id).ToHashSet());
    var import = AppRig.UiTask(() => assets.ImportAsync(cubePath));
    Assert.True(import.Wait(TimeSpan.FromSeconds(60)) && import.Result.Success, "the cube imported");
    ImportedAsset? cube = null;
    Assert.True(WaitFor(() => (cube = Ui(() => assets.CurrentTextures.FirstOrDefault(t => !before.Contains(t.Id)))) is not null, 30_000),
      "the cube in the texture list: " + string.Join(" | ", Ui(() => assets.CurrentTextures.Select(t => $"{t.Id} {t.Key}"))));
    _output.WriteLine($"cube: asset {cube!.Id}, key {cube.Key}");
    Assert.True(RasterIs(runtime, cube!.Id, 0, out var seen0), $"plane 0 at import: {seen0}");
    Ui(() =>
    {
      var vm = settings.Billboards;
      vm.SelectedTexture = vm.TextureOptions.First(t => t.Id == cube.Id);
      vm.CreateCommand.Execute(null);
    });
    Assert.True(WaitFor(() => Ui(() => billboards.Current.Length) == 1, 10_000), "one billboard");
    var id = Ui(() => billboards.Current[0].Id);
    // its window opens over the app's own controls: hide it
    Ui(() => billboards.Update(id, b => b with { Visible = false }));

    // ── 3. the panel: 1 of 3, then plane 3 typed ────────────────────────────────────────────
    Step("3: the FITS panel shows 1 of 3");
    ShowFitsPanel(rig);
    Assert.True(WaitFor(() => FactText(rig, "FitsFact_Planes") == "1 of 3", 20_000),
      $"the planes fact: {FactText(rig, "FitsFact_Planes")}");
    var of = rig.Require<TextBlock>("FitsPlaneOf");
    Assert.Equal("of 3", Ui(() => of.Text));
    var box = rig.Require<TextBox>("FitsPlane"); // the NumericUpDown's text box carries its id
    Assert.Equal("1", Ui(() => box.Text));

    Step("3: type plane 3");
    ClickInView(rig, box);
    XDo("key --clearmodifiers ctrl+a");
    XDo("type --delay 40 3");
    XDo("key Return");
    Settle(300);
    Assert.True(WaitFor(() => Ui(() => billboards.Find(id)?.FitsPlane) == 2, 10_000),
      $"the billboard records plane index 2 (has {Ui(() => billboards.Find(id)?.FitsPlane)})");
    string seen = "";
    Assert.True(WaitFor(() => RasterIs(runtime, cube.Id, 2, out seen), 30_000), $"the engine shows plane 2: {seen}");
    Assert.True(WaitFor(() => FactText(rig, "FitsFact_Planes") == "3 of 3", 10_000), $"the planes fact: {FactText(rig, "FitsFact_Planes")}");
    var json = runtime.GetFitsMetadataJson(cube.Id);
    Assert.Contains("\"plane\":2,", json ?? "");

    // ── 4. Save As, New, reopen ─────────────────────────────────────────────────────────────
    Step("4: Save As cube.avk");
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Save As...");
    Assert.True(WaitFor(() => File.Exists(avk) && rig.XTitle.EndsWith("— cube", StringComparison.Ordinal)
      && !rig.XTitle.TrimEnd().EndsWith('*'), 90_000), $"saved, title: {rig.XTitle}");

    Step("4: File › New");
    rig.Menu("File", "New");
    Assert.True(WaitFor(() => Ui(() => billboards.Current.Length) == 0, 30_000), "New cleared the billboards");
    Assert.True(WaitFor(() => !Ui(() => assets.CurrentTextures.Any(t => t.Id == cube.Id)), 30_000), "New unloaded the cube");

    Step("4: reopen cube.avk");
    rig.Dialogs.Answer(avk);
    rig.Menu("File", "Open...");
    Assert.True(WaitFor(() => rig.XTitle.EndsWith("— cube", StringComparison.Ordinal), 90_000), $"reopened, title: {rig.XTitle}");
    Assert.True(WaitFor(() => Ui(() => billboards.Current.Length) == 1, 30_000), "the billboard restored");
    Assert.Equal(2, Ui(() => billboards.Current[0].FitsPlane));
    ImportedAsset? again = null;
    Assert.True(WaitFor(() => (again = Ui(() => assets.CurrentTextures.FirstOrDefault(t =>
      t.Key == billboards.Current[0].TextureKey))) is not null, 60_000), "the cube re-imported");
    Assert.True(WaitFor(() => RasterIs(runtime, again!.Id, 2, out seen), 30_000), $"plane 2 re-applied after the load: {seen}");

    Step("4: the panel shows plane 3 again");
    ShowFitsPanel(rig);
    Assert.True(WaitFor(() => FactText(rig, "FitsFact_Planes") == "3 of 3", 20_000), $"the planes fact: {FactText(rig, "FitsFact_Planes")}");
    var box2 = rig.Require<TextBox>("FitsPlane");
    Assert.True(WaitFor(() => Ui(() => box2.Text) == "3", 10_000), $"the selector: {Ui(() => box2.Text)}");
    Assert.Equal(0, rig.Prompt.Asked);
  }
}
