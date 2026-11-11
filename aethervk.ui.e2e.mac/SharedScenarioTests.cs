using AetherVk.UI.E2E.Shared;
using AetherVk.UI.E2E.Shared.Scenarios;
using Xunit.Abstractions;

namespace AetherVk.UI.E2E.Mac;

/// <summary>
/// The scenarios shared with the Windows suite (aethervk.ui.e2e.shared/Scenarios), driven through
/// <see cref="MacE2EApp"/>. The test names are the same in aethervk.ui.e2e (Windows), so the two
/// handoff reports compare like for like. Artifacts go to TestResults/e2e-mac/&lt;scenario&gt;/.
/// </summary>
public sealed class SharedScenarioTests(ITestOutputHelper output)
{
  private static readonly MacE2EHost Host = new();

  private void Run(Action<IE2EHost, Action<string>> scenario)
  {
    var unavailable = Host.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    scenario(Host, output.WriteLine);
  }

  [SkippableFact]
  public void LayoutToggle10x_KeepsViewportAndRenders() => Run(LayoutToggleScenario.Run);

  [SkippableFact]
  public void CompactPanel_FullReviewerFlow_67P() => Run(CompactReviewerFlowScenario.Run);

  [SkippableFact]
  public void CompactLayout_SavedAndReopened() => Run(CompactLayoutSavedScenario.Run);

  [SkippableFact]
  public void NucleusMiniView_ShowsTheLitNucleusCentred() => Run(NucleusMiniViewScenarios.ShowsTheLitNucleusCentred);

  [SkippableFact]
  public void NucleusMiniView_CollapseStopsItsFrames_MainKeepsRendering() =>
    Run(NucleusMiniViewScenarios.CollapseStopsItsFrames_MainKeepsRendering);

  [SkippableFact]
  public void NucleusMiniView_LayoutToggle10x_SameHost() => Run(NucleusMiniViewScenarios.LayoutToggle10x_SameHost);

  [SkippableFact]
  public void NucleusMiniView_ExportUsesMainOnly() => Run(NucleusMiniViewScenarios.ExportUsesMainOnly);

  [SkippableFact]
  public void NucleusMiniView_KeyboardStillDrivesMainCamera() => Run(NucleusMiniViewScenarios.KeyboardStillDrivesMainCamera);

  [SkippableFact]
  public void NucleusMiniView_ShutdownClean_5x() => Run(NucleusMiniViewScenarios.ShutdownClean_5x);

  [SkippableFact]
  public void Smoke_OpenPlayExport_67P() => Run(SmokeScenario.Run);
}
