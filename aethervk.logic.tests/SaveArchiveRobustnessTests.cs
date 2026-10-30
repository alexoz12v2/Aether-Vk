using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Threading.Tasks;
using AetherVk.Logic.Models;
using AetherVk.Logic.SaveFile;
using AetherVk.Logic.Tests.Mocks;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The single-file save under stress: a state of half a gigabyte, damaged files, a target held
/// open while saving, and two sessions sharing one work root.
/// </summary>
public sealed class SaveArchiveRobustnessTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_robust_" + Guid.NewGuid().ToString("N"));
  private string WorkRoot => Path.Combine(_root, "open");

  public SaveArchiveRobustnessTests() => Directory.CreateDirectory(_root);

  public void Dispose()
  {
    try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
  }

  private static void CopyDir(string from, string to)
  {
    foreach (var f in Directory.GetFiles(from, "*", SearchOption.AllDirectories))
    {
      var dest = Path.Combine(to, Path.GetRelativePath(from, f));
      Directory.CreateDirectory(Path.GetDirectoryName(dest)!);
      File.Copy(f, dest);
    }
  }

  private static byte[] Sha256(string file)
  {
    using var s = File.OpenRead(file);
    using var sha = SHA256.Create();
    return sha.ComputeHash(s);
  }

  /// <summary>A saved session of the fake app (comet, range, one jet).</summary>
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

  // ── a large state ──────────────────────────────────────────────────────────

  private const long LargeStateBytes = 512L << 20;
  private static readonly TimeSpan LargeStateBound = TimeSpan.FromSeconds(120);

  /// <summary>
  /// A 512 MiB scene state (pseudo-random, so it does not compress) in a copy of the reviewer's
  /// save packs and extracts within <see cref="LargeStateBound"/>, and comes back bit for bit.
  /// </summary>
  [Fact]
  public void AHalfGigabyteState_PacksAndExtracts_InTime_AndIntact()
  {
    var folder = Path.Combine(_root, "large");
    CopyDir(ReviewSaveReproTests.FixtureDir(), folder);
    var rel = Path.Combine("states", "0123456789abcdef", "scene.bin");
    var state = Path.Combine(folder, rel);
    Directory.CreateDirectory(Path.GetDirectoryName(state)!);
    byte[] written;
    try
    {
      // xorshift64*, fixed seed: deterministic, incompressible
      using (var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256))
      using (var f = new FileStream(state, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 20))
      {
        var buffer = new byte[1 << 20];
        ulong x = 0x9E3779B97F4A7C15UL;
        for (long done = 0; done < LargeStateBytes; done += buffer.Length)
        {
          for (int i = 0; i < buffer.Length; i += 8)
          {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            BitConverter.TryWriteBytes(buffer.AsSpan(i, 8), x * 0x2545F4914F6CDD1DUL);
          }
          f.Write(buffer, 0, buffer.Length);
          hash.AppendData(buffer);
        }
        written = hash.GetHashAndReset();
      }
      Assert.Equal(LargeStateBytes, new FileInfo(state).Length);

      var avk = Path.Combine(_root, "large.avk");
      var clock = Stopwatch.StartNew();
      SaveArchive.Pack(folder, avk);
      var packed = clock.Elapsed;
      var dir = SaveArchive.Extract(avk, WorkRoot);
      clock.Stop();

      Assert.True(clock.Elapsed < LargeStateBound,
        $"pack {packed.TotalSeconds:F1} s + extract {(clock.Elapsed - packed).TotalSeconds:F1} s > {LargeStateBound.TotalSeconds} s");
      Assert.Equal(written, Sha256(Path.Combine(dir, rel)));
      Assert.Equal(SaveIndexError.None, SaveIndexIo.Read(dir).Error);
    }
    finally
    {
      // a gigabyte and a half of temporary data: gone now, not at Dispose
      try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }
  }

  // ── damaged files ──────────────────────────────────────────────────────────

  private string Damaged(string kind)
  {
    var avk = Path.Combine(_root, kind + ".avk");
    switch (kind)
    {
      case "empty":
        File.WriteAllBytes(avk, []);
        break;
      case "truncated":
        var whole = Path.Combine(_root, "whole.avk");
        SaveArchive.Pack(ReviewSaveReproTests.FixtureDir(), whole);
        var bytes = File.ReadAllBytes(whole);
        File.WriteAllBytes(avk, bytes.AsSpan(0, bytes.Length / 2).ToArray());
        File.Delete(whole);
        break;
      case "text":
        File.WriteAllText(avk, "This is not a save, only a text file renamed to .avk.\n");
        break;
      default:
        throw new ArgumentException(kind);
    }
    return avk;
  }

  [Theory]
  [InlineData("empty")]
  [InlineData("truncated")]
  [InlineData("text")]
  public void ADamagedFile_IsRefused_AndLeavesNothing(string kind)
  {
    var avk = Damaged(kind);
    Directory.CreateDirectory(WorkRoot);

    Assert.Throws<InvalidDataException>(() => SaveArchive.Extract(avk, WorkRoot));
    Assert.Empty(Directory.GetFileSystemEntries(WorkRoot));
  }

  [Theory]
  [InlineData("empty")]
  [InlineData("truncated")]
  [InlineData("text")]
  public async Task ADamagedFile_OpensNothing_AndSaysWhy(string kind)
  {
    var avk = Damaged(kind);
    var host = new FakeSaveFileHost(_root);
    var svc = new SaveFileService(host, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot);

    var r = await svc.OpenAsync(avk);

    Assert.False(r.Success);
    Assert.Equal(SaveIndexError.Unreadable, r.Error);
    Assert.Contains(host.Notes, n => n.Contains("could not be opened") && n.Contains(Path.GetFileName(avk)));
    Assert.DoesNotContain("reset", host.Calls);
    Assert.Null(svc.CurrentSavePath);
    Assert.True(!Directory.Exists(WorkRoot) || Directory.GetFileSystemEntries(WorkRoot).Length == 0);
  }

  // ── a target held open ─────────────────────────────────────────────────────

  /// <summary>
  /// Windows: another program holding the .avk without sharing makes the replace fail. The save
  /// reports it (no exception escapes), the old archive is untouched and no temporary file is left.
  /// </summary>
  [SkippableFact]
  public async Task Windows_ATargetHeldOpen_FailsTheSave_AndKeepsTheOldArchive()
  {
    Skip.IfNot(RuntimeInformation.IsOSPlatform(OSPlatform.Windows), "share modes are enforced on Windows only");
    var avk = Path.Combine(_root, "held.avk");
    Assert.True(await new SaveFileService(SavedHost(250), new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot).SaveAsync(avk));
    var before = File.ReadAllBytes(avk);
    var host = SavedHost(999);
    var svc = new SaveFileService(host, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot);

    bool ok;
    using (new FileStream(avk, FileMode.Open, FileAccess.Read, FileShare.None))
      ok = await svc.SaveAsync(avk);

    Assert.False(ok);
    Assert.Contains(host.Notes, n => n.Contains("Could not write") && n.Contains("held.avk"));
    Assert.Equal(before, File.ReadAllBytes(avk));
    Assert.Equal(["held.avk"], Directory.GetFiles(_root).Select(Path.GetFileName));
    // the work folder holds the save: once the file is free, Save packs it
    Assert.True(await svc.SaveAsync(avk));
    var again = new FakeSaveFileHost(_root);
    Assert.True((await new SaveFileService(again, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot).OpenAsync(avk)).Success);
    Assert.Equal(999, again.Capture().Model.DiametreUm);
  }

  /// <summary>
  /// Linux / macOS: share modes do not stop a rename, so the replace succeeds. The holder keeps
  /// reading the old archive (its handle is on the old file), the path names the new one, and no
  /// temporary file is left.
  /// </summary>
  [SkippableFact]
  public async Task Unix_ATargetHeldOpen_IsReplaced_TheHolderKeepsTheOldBytes()
  {
    Skip.If(RuntimeInformation.IsOSPlatform(OSPlatform.Windows), "Unix rename semantics");
    var avk = Path.Combine(_root, "held.avk");
    Assert.True(await new SaveFileService(SavedHost(250), new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot).SaveAsync(avk));
    var before = File.ReadAllBytes(avk);
    var host = SavedHost(999);
    var svc = new SaveFileService(host, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot);

    using (var held = new FileStream(avk, FileMode.Open, FileAccess.Read, FileShare.None))
    {
      Assert.True(await svc.SaveAsync(avk));
      var seen = new byte[before.Length + 1];
      int n = 0, k;
      while ((k = held.Read(seen, n, seen.Length - n)) > 0) n += k;
      Assert.Equal(before, seen.AsSpan(0, n).ToArray());
    }

    Assert.Empty(host.Notes);
    Assert.NotEqual(before, File.ReadAllBytes(avk));
    Assert.Equal(["held.avk"], Directory.GetFiles(_root).Select(Path.GetFileName));
    var again = new FakeSaveFileHost(_root);
    Assert.True((await new SaveFileService(again, new SaveArchiveTests.Clock(), archiveWorkRoot: WorkRoot).OpenAsync(avk)).Success);
    Assert.Equal(999, again.Capture().Model.DiametreUm);
  }

  // ── two sessions ───────────────────────────────────────────────────────────

  private sealed class WallClock : ISystemClock
  {
    public DateTimeOffset UtcNow => DateTimeOffset.UtcNow;
  }

  /// <summary>
  /// Two applications share <c>~/.aethervk/open</c>: the second one's open cleans the root, which
  /// removes a folder untouched for <see cref="SaveArchive.WorkFolderExpiry"/> but not the first
  /// application's open save (fresh), which can still be saved in place.
  /// </summary>
  [Fact]
  public async Task AnotherSessionsOpen_KeepsThisSessionsWorkFolder_AndRemovesStaleOnes()
  {
    // the work folders' age is their files' time: the services run on the wall clock
    var clock = new WallClock();
    var a = Path.Combine(_root, "a.avk");
    var b = Path.Combine(_root, "b.avk");
    Assert.True(await new SaveFileService(SavedHost(100), clock, archiveWorkRoot: WorkRoot).SaveAsync(a));
    Assert.True(await new SaveFileService(SavedHost(200), clock, archiveWorkRoot: WorkRoot).SaveAsync(b));
    foreach (var d in Directory.GetDirectories(WorkRoot))
      Directory.Delete(d, recursive: true);

    var hostA = new FakeSaveFileHost(_root);
    var first = new SaveFileService(hostA, clock, archiveWorkRoot: WorkRoot);
    Assert.True((await first.OpenAsync(a)).Success);
    var openA = first.CurrentSaveDir!;

    var stale = Directory.CreateDirectory(Path.Combine(WorkRoot, "0000000000000000-20261001-000000-000")).FullName;
    var staleFile = Path.Combine(stale, SaveIndex.FileName);
    File.WriteAllText(staleFile, "{}");
    var old = DateTime.Now - SaveArchive.WorkFolderExpiry - TimeSpan.FromHours(1);
    File.SetLastWriteTime(staleFile, old);
    Directory.SetLastWriteTime(stale, old);

    var hostB = new FakeSaveFileHost(_root);
    var second = new SaveFileService(hostB, clock, archiveWorkRoot: WorkRoot);
    Assert.True((await second.OpenAsync(b)).Success);

    Assert.False(Directory.Exists(stale));
    Assert.True(Directory.Exists(openA));
    Assert.Equal(SaveIndexError.None, SaveIndexIo.Read(openA).Error);
    Assert.NotEqual(openA, second.CurrentSaveDir);
    Assert.Equal(2, Directory.GetDirectories(WorkRoot).Length);

    // the first session still saves in place
    hostA.Model.DiametreUm = 111;
    Assert.True(await first.SaveAsync(first.CurrentSavePath!));
    var check = new FakeSaveFileHost(_root);
    Assert.True((await new SaveFileService(check, clock, archiveWorkRoot: WorkRoot).OpenAsync(a)).Success);
    Assert.Equal(111, check.Capture().Model.DiametreUm);
    Assert.Equal(200, hostB.Capture().Model.DiametreUm);
  }
}
