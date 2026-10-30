using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Services;
using AetherVk.Logic.Tests.Mocks;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The single-file save (<see cref="SaveArchive"/>, reviewer report 2026-10-29: the save folder was
/// unmanageable): a round trip gives the same index and files, a hostile zip is refused, a failed
/// pack leaves the previous file, and the reviewer's save packs smaller than its folder.
/// </summary>
public sealed class SaveArchiveTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_archive_" + Guid.NewGuid().ToString("N"));
  private string WorkRoot => Path.Combine(_root, "open");

  public SaveArchiveTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  internal sealed class Clock : ISystemClock
  {
    public DateTimeOffset UtcNow { get; set; } = new(2026, 10, 29, 12, 0, 0, TimeSpan.Zero);
  }

  private static string FixtureDir()
  {
    var dir = AppContext.BaseDirectory;
    while (dir is not null && !Directory.Exists(Path.Combine(dir, "test_assets", "saves", "review_67p_2021-12")))
      dir = Path.GetDirectoryName(dir);
    return Path.Combine(dir ?? throw new DirectoryNotFoundException("test_assets"), "test_assets", "saves", "review_67p_2021-12");
  }

  internal static string[] Tree(string dir) =>
    Directory.GetFiles(dir, "*", SearchOption.AllDirectories)
      .Select(f => f.Substring(dir.Length + 1).Replace('\\', '/') + " " + Convert.ToBase64String(
        System.Security.Cryptography.SHA256.Create().ComputeHash(File.ReadAllBytes(f))))
      .OrderBy(s => s, StringComparer.Ordinal).ToArray();

  private static long FolderBytes(string dir) =>
    Directory.GetFiles(dir, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length);

  [Fact]
  public void PackThenExtract_GivesTheSameFiles_IndexFirst()
  {
    var avk = Path.Combine(_root, "review.avk");
    SaveArchive.Pack(FixtureDir(), avk);

    using (var zip = ZipFile.OpenRead(avk))
      Assert.Equal(SaveIndex.FileName, zip.Entries[0].FullName);
    var dir = SaveArchive.Extract(avk, WorkRoot);
    Assert.StartsWith(WorkRoot, dir);
    Assert.Equal(Tree(FixtureDir()), Tree(dir));
    Assert.Equal(SaveIndexError.None, SaveIndexIo.Read(dir).Error);
  }

  [Fact]
  public void TheReviewSave_IsOneFile_SmallerThanItsFolder()
  {
    var avk = Path.Combine(_root, "review.avk");
    SaveArchive.Pack(FixtureDir(), avk);
    Assert.True(new FileInfo(avk).Length < FolderBytes(FixtureDir()),
      $"{new FileInfo(avk).Length} B packed vs {FolderBytes(FixtureDir())} B");
  }

  [Theory]
  [InlineData("../evil.txt")]
  [InlineData("spk/../../evil.txt")]
  [InlineData("/etc/evil.txt")]
  [InlineData("..\\evil.txt")]
  public void AnEntryLeavingTheFolder_IsRefused_AndLeavesNothing(string entry)
  {
    var avk = Path.Combine(_root, "evil.avk");
    using (var zip = ZipFile.Open(avk, ZipArchiveMode.Create))
    {
      using (var w = new StreamWriter(zip.CreateEntry(SaveIndex.FileName).Open()))
        w.Write("{}");
      using (var w = new StreamWriter(zip.CreateEntry(entry).Open()))
        w.Write("pwned");
    }

    var e = Assert.Throws<InvalidDataException>(() => SaveArchive.Extract(avk, WorkRoot));
    Assert.Contains("leaves the save folder", e.Message);
    Assert.False(File.Exists(Path.Combine(_root, "evil.txt")));
    Assert.False(File.Exists(Path.Combine(WorkRoot, "evil.txt")));
    Assert.Empty(Directory.GetDirectories(WorkRoot));
  }

  [Fact]
  public void AZipWithoutAnIndex_IsRefused()
  {
    var avk = Path.Combine(_root, "plain.avk");
    using (var zip = ZipFile.Open(avk, ZipArchiveMode.Create))
    using (var w = new StreamWriter(zip.CreateEntry("readme.txt").Open()))
      w.Write("hi");
    Assert.Throws<InvalidDataException>(() => SaveArchive.Extract(avk, WorkRoot));
    Assert.Empty(Directory.GetDirectories(WorkRoot));
  }

  [Fact]
  public void AFailedPack_LeavesThePreviousArchive_AndNoTemporaryFile()
  {
    var avk = Path.Combine(_root, "keep.avk");
    SaveArchive.Pack(FixtureDir(), avk);
    var before = File.ReadAllBytes(avk);

    Assert.Throws<IOException>(() => SaveArchive.Pack(FixtureDir(), avk,
      afterEntry: rel => { if (rel != SaveIndex.FileName) throw new IOException("disk full"); }));

    Assert.Equal(before, File.ReadAllBytes(avk));
    Assert.Equal(["keep.avk"], Directory.GetFiles(_root).Select(Path.GetFileName));
  }

  [Fact]
  public void OldWorkFolders_AreCleaned_ButNotTheOpenOne()
  {
    var now = new DateTime(2026, 10, 29, 12, 0, 0);
    var old = Directory.CreateDirectory(Path.Combine(WorkRoot, "old")).FullName;
    var open = Directory.CreateDirectory(Path.Combine(WorkRoot, "open")).FullName;
    var fresh = Directory.CreateDirectory(Path.Combine(WorkRoot, "fresh")).FullName;
    foreach (var d in new[] { old, open, fresh })
      File.WriteAllText(Path.Combine(d, "f"), "x");
    File.SetLastWriteTime(Path.Combine(old, "f"), now.AddDays(-3));
    File.SetLastWriteTime(Path.Combine(open, "f"), now.AddDays(-3));
    File.SetLastWriteTime(Path.Combine(fresh, "f"), now.AddHours(-1));

    Assert.Equal(1, SaveArchive.CleanWorkRoot(WorkRoot, now, keep: open));
    Assert.False(Directory.Exists(old));
    Assert.True(Directory.Exists(open));
    Assert.True(Directory.Exists(fresh));
  }

  [Fact]
  public async Task SaveAvk_ThenOpen_RebuildsTheSession_AndSaveRepacksTheSameFile()
  {
    var saved = new FakeSaveFileHost(_root);
    Directory.CreateDirectory(Path.GetDirectoryName(saved.SpkSource)!);
    File.WriteAllText(saved.SpkSource, "DAF/SPK 67P");
    saved.Comet = new SpkDto { NaifId = 1000012, Designation = "67P", FullName = "67P/Churyumov-Gerasimenko", RecordId = "90000703" };
    saved.Timeline.CommittedStart = new EpochParts(0, 1000);
    saved.Timeline.CommittedEnd = new EpochParts(0, 9000);
    saved.Timeline.CurrentEpoch = new EpochParts(0, 4000);
    saved.Model = new ModelDto { DiametreUm = 250 };
    saved.Jets = [new JetDto { LatitudeDeg = -67, Seed = 633654374 }];
    var avk = Path.Combine(_root, "67P.avk");

    Assert.True(await new SaveFileService(saved, new Clock(), archiveWorkRoot: WorkRoot).SaveAsync(avk));
    Assert.True(File.Exists(avk));

    var host = new FakeSaveFileHost(_root);
    var svc = new SaveFileService(host, new Clock(), archiveWorkRoot: WorkRoot);
    var r = await svc.OpenAsync(avk);
    Assert.True(r.Success);
    Assert.True(r.StateRestored);
    Assert.Equal(Path.GetFullPath(avk), svc.CurrentSavePath);
    Assert.StartsWith(WorkRoot, svc.CurrentSaveDir);
    Assert.Equal(633654374u, Assert.Single(host.Capture().Jets).Seed);
    Assert.False(svc.IsDirty);

    // Save in place writes the work folder and packs the same .avk again
    host.Model.DiametreUm = 999;
    Assert.True(await svc.SaveAsync(svc.CurrentSavePath!));
    var again = new FakeSaveFileHost(_root);
    Assert.True((await new SaveFileService(again, new Clock(), archiveWorkRoot: WorkRoot).OpenAsync(avk)).Success);
    Assert.Equal(999, again.Capture().Model.DiametreUm);
  }

  [Fact]
  public async Task ARefusedArchive_OpensNothing_AndSaysWhy()
  {
    var avk = Path.Combine(_root, "evil.avk");
    using (var zip = ZipFile.Open(avk, ZipArchiveMode.Create))
    using (var w = new StreamWriter(zip.CreateEntry("../x").Open()))
      w.Write("x");
    var host = new FakeSaveFileHost(_root);

    var r = await new SaveFileService(host, new Clock(), archiveWorkRoot: WorkRoot).OpenAsync(avk);

    Assert.False(r.Success);
    Assert.DoesNotContain("reset", host.Calls);
    Assert.Contains(host.Notes, n => n.Contains("leaves the save folder"));
  }
}
