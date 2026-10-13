using System;
using System.Diagnostics;
using System.IO;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>Session directories: pid file, sweep of abandoned sessions, downloads inside the session.</summary>
public sealed class LocalStorageServiceTests : IDisposable
{
  private readonly string _root = Path.Combine(Path.GetTempPath(), "avk_storage_" + Guid.NewGuid().ToString("N"));

  public void Dispose()
  {
    if (Directory.Exists(_root))
      Directory.Delete(_root, true);
  }

  private string FakeSession(string? pidLine, DateTime? createdUtc = null)
  {
    var dir = Directory.CreateDirectory(Path.Combine(_root, Guid.NewGuid().ToString("N"))).FullName;
    if (pidLine is not null)
      File.WriteAllText(Path.Combine(dir, LocalStorageService.PidFileName), pidLine);
    if (createdUtc is { } c)
      Directory.SetCreationTimeUtc(dir, c);
    return dir;
  }

  [Fact]
  public void IsAbandoned_FollowsTheOwnerProcess()
  {
    var now = DateTime.UtcNow;
    var start = new DateTime(2026, 10, 12, 20, 0, 0, DateTimeKind.Utc);
    var dir = FakeSession($"4242 {start.Ticks}");
    Assert.False(LocalStorageService.IsAbandoned(dir, now, pid => pid == 4242 ? start : null)); // running
    Assert.True(LocalStorageService.IsAbandoned(dir, now, _ => null));                          // gone (crash, kill -9)
    Assert.True(LocalStorageService.IsAbandoned(dir, now, _ => start.AddMinutes(5)));          // pid recycled
    Assert.True(LocalStorageService.IsAbandoned(FakeSession("garbage"), now, _ => start));
    // legacy folders without a pid file: one day
    Assert.False(LocalStorageService.IsAbandoned(FakeSession(null, now.AddHours(-2)), now, _ => null));
    Assert.True(LocalStorageService.IsAbandoned(FakeSession(null, now.AddDays(-2)), now, _ => null));
  }

  [Fact]
  public void Startup_SweepsDeadSessions_KeepsLiveOnes_AndWritesItsOwnPid()
  {
    using var me = Process.GetCurrentProcess();
    var live = FakeSession($"{me.Id} {me.StartTime.ToUniversalTime().Ticks}");
    var dead = FakeSession($"2147483 {DateTime.UtcNow.Ticks}");
    var storage = new LocalStorageService(_root);
    Assert.True(Directory.Exists(live));
    Assert.False(Directory.Exists(dead));
    var pid = File.ReadAllText(Path.Combine(storage.SessionDirectory, LocalStorageService.PidFileName)).Split(' ')[0];
    Assert.Equal(me.Id.ToString(), pid);
    // downloads live in the session and go with it
    var spk = storage.GetDownloadsPath("spk_67P.bsp");
    Assert.Equal(Path.Combine(storage.SessionDirectory, "spk", "spk_67P.bsp"), spk);
    Assert.True(Directory.Exists(Path.GetDirectoryName(spk)));
    storage.Dispose();
    Assert.False(Directory.Exists(storage.SessionDirectory));
  }
}
