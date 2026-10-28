using System.IO;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

/// The engine loads a per-range copy of the cached SPK, never the persistent cache file: the loaded
/// file stays mapped while the comet is committed (an expiry delete or a recommit with a new end
/// date used to fail on it).
public class HorizonJplSpkDeliveryTests
{
  private static (string Dir, string Cache, string Save) Files()
  {
    var dir = Directory.CreateTempSubdirectory("avk_spk_").FullName;
    var cache = Path.Combine(dir, "spk_m30_90000703_2021-12-15_2021-12-28.bsp");
    File.WriteAllBytes(cache, [1, 2, 3, 4]);
    return (dir, cache, Path.Combine(dir, "spk", "spk_67P_90000703.bsp"));
  }

  [Fact]
  public void DeliverSpk_CopiesToAPerRangeSessionFile_NotTheCache()
  {
    var (dir, cache, save) = Files();
    Directory.CreateDirectory(Path.GetDirectoryName(save)!);
    try
    {
      var a = HorizonJplService.DeliverSpk(cache, save, "2021-12-15", "2021-12-28");
      var b = HorizonJplService.DeliverSpk(cache, save, "2021-12-15", "2022-01-27");

      Assert.NotEqual(cache, a);
      Assert.NotEqual(a, b); // a new end date never overwrites the file of the old range
      Assert.EndsWith("spk_67P_90000703_2021-12-15_2021-12-28.bsp", a);
      Assert.Equal(File.ReadAllBytes(cache), File.ReadAllBytes(a));
    }
    finally { Directory.Delete(dir, true); }
  }

  [Fact]
  public void DeliverSpk_ReusesTheCopyStillInUse()
  {
    var (dir, cache, save) = Files();
    Directory.CreateDirectory(Path.GetDirectoryName(save)!);
    try
    {
      var first = HorizonJplService.DeliverSpk(cache, save, "2021-12-15", "2021-12-28");
      // the committed comet holds it (the native almanac maps it)
      using (new FileStream(first, FileMode.Open, FileAccess.Read, FileShare.Read))
        Assert.Equal(first, HorizonJplService.DeliverSpk(cache, save, "2021-12-15", "2021-12-28"));
    }
    finally { Directory.Delete(dir, true); }
  }
}
