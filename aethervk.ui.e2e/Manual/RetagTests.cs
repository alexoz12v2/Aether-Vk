using System.Text.Json;

namespace AetherVk.UI.E2E.Manual;

/// <summary>
/// Redraws the tagged PNGs of a capture from its clean PNGs and callouts files, without the app:
/// <c>AETHERVK_MANUAL_RETAG=TestResults/e2e/manual/&lt;stamp&gt;</c>. For changes to the marker layout.
/// </summary>
public sealed class RetagTests
{
  [SkippableFact]
  public void RetagCapture()
  {
    var dir = Environment.GetEnvironmentVariable("AETHERVK_MANUAL_RETAG");
    Skip.If(string.IsNullOrEmpty(dir), "set AETHERVK_MANUAL_RETAG to a capture folder");
    if (!Path.IsPathRooted(dir))
      dir = Path.Combine(AppSession.RepoRoot(), dir);
    var options = new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower };
    int done = 0;
    foreach (var json in Directory.GetFiles(dir, "*.callouts.json"))
    {
      var stem = Path.GetFileName(json)[..^".callouts.json".Length];
      var clean = Path.Combine(dir, stem + ".png");
      if (!File.Exists(clean))
        continue;
      var callouts = JsonSerializer.Deserialize<List<CalloutOut>>(File.ReadAllText(json), options) ?? [];
      var bgra = ShotTools.ReadPng(clean, out int w, out int h);
      callouts = ShotTools.Renumber(callouts, w, h);
      File.WriteAllText(json, ShotTools.ToJson(callouts));
      var tagged = ShotTools.Tagged(bgra, w, h, callouts, out int tw, out int th);
      ShotTools.WritePng(Path.Combine(dir, stem + "--tagged.png"), tagged, tw, th);
      done++;
    }
    Assert.True(done > 0, $"no shots in {dir}");
  }
}
