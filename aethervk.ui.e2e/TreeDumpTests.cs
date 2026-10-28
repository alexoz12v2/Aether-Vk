namespace AetherVk.UI.E2E;

/// <summary>
/// Locator work: AETHERVK_E2E_DUMP=1 writes the UIA tree WinAppDriver sees with each tab
/// selected to TestResults/e2e/tree/&lt;tab&gt;.xml.
/// </summary>
public sealed class TreeDumpTests
{
  private static readonly string[] Tabs = ["SettingsTab", "CometTab", "ModelTab", "ImportsTab", "TimelineTab"];

  [SkippableFact]
  public void DumpTree()
  {
    Skip.If(Environment.GetEnvironmentVariable("AETHERVK_E2E_DUMP") != "1", "set AETHERVK_E2E_DUMP=1");
    var unavailable = AppSession.UnavailableReason();
    Skip.If(unavailable is not null, unavailable);
    var dir = Path.Combine(AppSession.RepoRoot(), "TestResults", "e2e", "tree");
    Directory.CreateDirectory(dir);

    using var s = AppSession.Launch();
    File.WriteAllText(Path.Combine(dir, "startup.xml"), s.App.PageSource);
    foreach (var tab in Tabs)
    {
      if (s.TryFind("Tab." + tab) is null)
        continue;
      s.SelectTab(tab);
      Thread.Sleep(1000);
      File.WriteAllText(Path.Combine(dir, tab + ".xml"), s.App.PageSource);
      s.Desktop.SaveScreenshot(Path.Combine(dir, tab + ".png"));
    }
  }
}
