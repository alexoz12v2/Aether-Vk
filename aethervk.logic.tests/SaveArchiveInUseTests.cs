using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Tests.Mocks;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The in-use marker of an <c>.avk</c> work folder (<see cref="SaveArchive.MarkInUse(string)"/>):
/// another instance's <see cref="SaveArchive.CleanWorkRoot"/> must not remove a save that is open
/// but idle past <see cref="SaveArchive.WorkFolderExpiry"/>, while a marker left by an exited
/// process (or a reused pid) does not keep a folder forever.
/// </summary>
public sealed class SaveArchiveInUseTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_inuse_" + Guid.NewGuid().ToString("N"));
  private string WorkRoot => Path.Combine(_root, "open");

  public SaveArchiveInUseTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private sealed class WallClock : ISystemClock
  {
    public DateTimeOffset UtcNow => DateTimeOffset.UtcNow;
  }

  private FakeSaveFileHost SavedHost(float diametreUm)
  {
    var host = new FakeSaveFileHost(_root);
    Directory.CreateDirectory(Path.GetDirectoryName(host.SpkSource)!);
    File.WriteAllText(host.SpkSource, "DAF/SPK 67P");
    host.Comet = new SpkDto { NaifId = 1000012, Designation = "67P", FullName = "67P/Churyumov-Gerasimenko", RecordId = "90000703" };
    host.Timeline.CommittedStart = new EpochParts(0, 1000);
    host.Timeline.CommittedEnd = new EpochParts(0, 9000);
    host.Timeline.CurrentEpoch = new EpochParts(0, 4000);
    host.Model = new ModelDto { DiametreUm = diametreUm };
    host.Jets = [new JetDto { LatitudeDeg = -67, Seed = 633654374 }];
    return host;
  }

  private static void Age(string dir, DateTime when)
  {
    foreach (var f in Directory.GetFiles(dir, "*", SearchOption.AllDirectories))
      File.SetLastWriteTime(f, when);
    Directory.SetLastWriteTime(dir, when);
  }

  // the saving services' own work folders and markers: only the sessions under test remain
  private void ClearWorkRoot()
  {
    foreach (var d in Directory.GetDirectories(WorkRoot)) Directory.Delete(d, recursive: true);
    foreach (var f in Directory.GetFiles(WorkRoot, "*.lock")) File.Delete(f);
  }

  private static DateTime Expired => DateTime.Now - SaveArchive.WorkFolderExpiry - TimeSpan.FromHours(1);

  private string StaleFolder(string name)
  {
    var dir = Directory.CreateDirectory(Path.Combine(WorkRoot, name)).FullName;
    File.WriteAllText(Path.Combine(dir, SaveIndex.FileName), "{}");
    Age(dir, Expired);
    return dir;
  }

  /// <summary>The open save of a session left idle for days survives another session's open,
  /// and saving it in place still works; the marker sits beside the folder and is not packed.</summary>
  [Fact]
  public async Task AnIdleOpenSave_SurvivesAnotherSessionsOpen()
  {
    var clock = new WallClock();
    var a = Path.Combine(_root, "a.avk");
    var b = Path.Combine(_root, "b.avk");
    Assert.True(await new SaveFileService(SavedHost(100), clock, archiveWorkRoot: WorkRoot).SaveAsync(a));
    Assert.True(await new SaveFileService(SavedHost(200), clock, archiveWorkRoot: WorkRoot).SaveAsync(b));
    ClearWorkRoot();

    var hostA = new FakeSaveFileHost(_root);
    var first = new SaveFileService(hostA, clock, archiveWorkRoot: WorkRoot);
    Assert.True((await first.OpenAsync(a)).Success);
    var openA = first.CurrentSaveDir!;
    Assert.True(File.Exists(SaveArchive.InUseMarker(openA)));
    Age(openA, Expired);

    var second = new SaveFileService(new FakeSaveFileHost(_root), clock, archiveWorkRoot: WorkRoot);
    Assert.True((await second.OpenAsync(b)).Success);

    Assert.True(Directory.Exists(openA));
    Assert.Equal(SaveIndexError.None, SaveIndexIo.Read(openA).Error);
    hostA.Model.DiametreUm = 111;
    Assert.True(await first.SaveAsync(first.CurrentSavePath!));
    using (var zip = ZipFile.OpenRead(a))
      Assert.DoesNotContain(zip.Entries, e => e.FullName.EndsWith(".lock", StringComparison.Ordinal));
    var check = new FakeSaveFileHost(_root);
    Assert.True((await new SaveFileService(check, clock, archiveWorkRoot: WorkRoot).OpenAsync(a)).Success);
    Assert.Equal(111, check.Capture().Model.DiametreUm);
  }

  /// <summary>A marker naming a process that is gone does not protect the folder; it is removed
  /// with it, and a marker whose folder is gone is removed too.</summary>
  [Fact]
  public void AMarkerOfAnExitedProcess_DoesNotKeepTheFolder()
  {
    using var gone = Process.Start(new ProcessStartInfo(OperatingSystem.IsWindows() ? "cmd.exe" : "true")
    {
      Arguments = OperatingSystem.IsWindows() ? "/c exit 0" : "",
      UseShellExecute = false,
      CreateNoWindow = true,
    })!;
    gone.WaitForExit();
    var dir = StaleFolder("exited");
    SaveArchive.MarkInUse(dir, gone.Id, 0);
    var orphan = Path.Combine(WorkRoot, "no-folder.lock");
    File.WriteAllText(orphan, $"{Process.GetCurrentProcess().Id} 0");

    Assert.Equal(1, SaveArchive.CleanWorkRoot(WorkRoot, DateTime.Now));
    Assert.False(Directory.Exists(dir));
    Assert.False(File.Exists(SaveArchive.InUseMarker(dir)));
    Assert.False(File.Exists(orphan));
  }

  /// <summary>A live pid with another start time is a reused pid, not the owner.</summary>
  [Fact]
  public void AMarkerOfAReusedPid_DoesNotKeepTheFolder()
  {
    var dir = StaleFolder("reused");
    using var self = Process.GetCurrentProcess();
    SaveArchive.MarkInUse(dir, self.Id, self.StartTime.ToUniversalTime().AddHours(-5).Ticks);

    Assert.Equal(1, SaveArchive.CleanWorkRoot(WorkRoot, DateTime.Now));
    Assert.False(Directory.Exists(dir));
  }

  /// <summary>A live owner keeps an expired folder; a malformed marker does not.</summary>
  [Fact]
  public void ALiveOwner_KeepsAnExpiredFolder_AndAMalformedMarkerDoesNot()
  {
    var kept = StaleFolder("kept");
    SaveArchive.MarkInUse(kept);
    var bad = StaleFolder("bad");
    File.WriteAllText(SaveArchive.InUseMarker(bad), "not a pid");

    Assert.Equal(1, SaveArchive.CleanWorkRoot(WorkRoot, DateTime.Now));
    Assert.True(Directory.Exists(kept));
    Assert.False(Directory.Exists(bad));
  }

  /// <summary>Opening another save, or a new session, releases the previous work folder's marker.</summary>
  [Fact]
  public async Task OpeningAnotherSave_ReleasesThePreviousMarker()
  {
    var clock = new WallClock();
    var a = Path.Combine(_root, "a.avk");
    var b = Path.Combine(_root, "b.avk");
    Assert.True(await new SaveFileService(SavedHost(100), clock, archiveWorkRoot: WorkRoot).SaveAsync(a));
    Assert.True(await new SaveFileService(SavedHost(200), clock, archiveWorkRoot: WorkRoot).SaveAsync(b));
    ClearWorkRoot();

    var service = new SaveFileService(new FakeSaveFileHost(_root), clock, archiveWorkRoot: WorkRoot);
    Assert.True((await service.OpenAsync(a)).Success);
    var first = service.CurrentSaveDir!;
    Assert.True((await service.OpenAsync(b)).Success);

    Assert.False(File.Exists(SaveArchive.InUseMarker(first)));
    Assert.True(File.Exists(SaveArchive.InUseMarker(service.CurrentSaveDir!)));
    Assert.Single(Directory.GetFiles(WorkRoot, "*.lock"));
  }
}
