using System;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;

namespace AetherVk.Logic.SaveFile;

/// <summary>
/// The single-file save (<c>.avk</c>): a zip of a save folder's layout (docs/savefile.md), so a save
/// travels as one file (reviewer report 2026-10-29: the save folder was unmanageable). Saving writes
/// the folder first, then packs it next to the target and replaces the target atomically; opening
/// extracts into its own work folder under <c>~/.aethervk/open/</c> and opens that folder.
/// </summary>
public static class SaveArchive
{
  public const string Extension = ".avk";

  /// <summary>Most entries an archive may hold (a save has a few dozen).</summary>
  public const int MaxEntries = 10_000;

  /// <summary>Most bytes an archive may expand to (the dumped states dominate).</summary>
  public const long MaxExtractedBytes = 8L << 30;

  /// <summary>Work folders not used for this long are removed by <see cref="CleanWorkRoot"/>.</summary>
  public static readonly TimeSpan WorkFolderExpiry = TimeSpan.FromDays(2);

  public static bool IsArchive(string path) =>
    string.Equals(Path.GetExtension(path), Extension, StringComparison.OrdinalIgnoreCase);

  /// <summary>The default work root, <c>~/.aethervk/open</c>.</summary>
  public static string DefaultWorkRoot() =>
    Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".aethervk", "open");

  /// <summary>
  /// Packs every file under <paramref name="folder"/> into <paramref name="archivePath"/>. The zip
  /// is written to a temporary file in the target's folder and moved over the target only once
  /// complete, so a failure leaves the previous archive as it was. <paramref name="afterEntry"/> is a
  /// test hook called after each entry.
  /// </summary>
  public static void Pack(string folder, string archivePath, Action<string>? afterEntry = null)
  {
    folder = Path.GetFullPath(folder);
    archivePath = Path.GetFullPath(archivePath);
    var targetDir = Path.GetDirectoryName(archivePath)!;
    Directory.CreateDirectory(targetDir);
    var tmp = Path.Combine(targetDir, "." + Path.GetFileName(archivePath) + "." + Guid.NewGuid().ToString("N") + ".tmp");
    try
    {
      using (var stream = new FileStream(tmp, FileMode.CreateNew, FileAccess.Write, FileShare.None))
      using (var zip = new ZipArchive(stream, ZipArchiveMode.Create))
      {
        // the index first: a reader can tell a save from any zip by its first entry
        var files = Directory.GetFiles(folder, "*", SearchOption.AllDirectories)
          .Select(f => (full: f, rel: Relative(folder, f)))
          .OrderBy(f => f.rel == SaveIndex.FileName ? 0 : 1)
          .ThenBy(f => f.rel, StringComparer.Ordinal);
        foreach (var (full, rel) in files)
        {
          var entry = zip.CreateEntry(rel, CompressionLevel.Optimal);
          entry.LastWriteTime = File.GetLastWriteTime(full);
          using (var src = new FileStream(full, FileMode.Open, FileAccess.Read, FileShare.Read))
          using (var dst = entry.Open())
            src.CopyTo(dst);
          afterEntry?.Invoke(rel);
        }
      }
      if (File.Exists(archivePath))
        File.Replace(tmp, archivePath, null);
      else
        File.Move(tmp, archivePath);
    }
    catch
    {
      try
      {
        if (File.Exists(tmp))
          File.Delete(tmp);
      }
      catch (IOException) { }
      throw;
    }
  }

  /// <summary>
  /// Extracts <paramref name="archivePath"/> into a new work folder under
  /// <paramref name="workRoot"/> (named by the archive's content hash and the time, so the folder of
  /// a save still open is never overwritten) and returns it. Refuses (<see cref="InvalidDataException"/>)
  /// an entry with a rooted path or one that leaves the folder, more than <see cref="MaxEntries"/>
  /// entries, more than <see cref="MaxExtractedBytes"/> in total, and an archive without
  /// <see cref="SaveIndex.FileName"/>; a refused archive leaves no folder behind.
  /// </summary>
  public static string Extract(string archivePath, string workRoot, DateTime? now = null)
  {
    string hash;
    using (var s = File.OpenRead(archivePath))
    using (var sha = SHA256.Create())
      hash = string.Concat(sha.ComputeHash(s).Take(8).Select(b => b.ToString("x2")));
    var stamp = (now ?? DateTime.Now).ToString("yyyyMMdd-HHmmss-fff");
    var dest = Path.GetFullPath(Path.Combine(workRoot, $"{hash}-{stamp}"));
    var root = dest.EndsWith(Path.DirectorySeparatorChar.ToString(), StringComparison.Ordinal)
      ? dest : dest + Path.DirectorySeparatorChar;
    Directory.CreateDirectory(dest);
    try
    {
      using var zip = ZipFile.OpenRead(archivePath);
      if (zip.Entries.Count > MaxEntries)
        throw new InvalidDataException($"the archive holds {zip.Entries.Count} entries (at most {MaxEntries})");
      long total = 0;
      bool hasIndex = false;
      foreach (var e in zip.Entries)
      {
        var name = e.FullName.Replace('\\', '/');
        if (name.Length == 0 || name.StartsWith("/", StringComparison.Ordinal) || Path.IsPathRooted(name)
            || name.Split('/').Any(p => p == ".."))
          throw new InvalidDataException($"the archive entry \"{e.FullName}\" leaves the save folder");
        var target = Path.GetFullPath(Path.Combine(dest, name));
        if (!target.StartsWith(root, StringComparison.Ordinal))
          throw new InvalidDataException($"the archive entry \"{e.FullName}\" leaves the save folder");
        if (name.EndsWith("/", StringComparison.Ordinal))
        {
          Directory.CreateDirectory(target);
          continue;
        }
        hasIndex |= name == SaveIndex.FileName;
        Directory.CreateDirectory(Path.GetDirectoryName(target)!);
        // count what is written, not what the header claims
        using var src = e.Open();
        using var dst = new FileStream(target, FileMode.CreateNew, FileAccess.Write, FileShare.None);
        var buffer = new byte[81920];
        int n;
        while ((n = src.Read(buffer, 0, buffer.Length)) > 0)
        {
          total += n;
          if (total > MaxExtractedBytes)
            throw new InvalidDataException($"the archive expands past {MaxExtractedBytes >> 30} GB");
          dst.Write(buffer, 0, n);
        }
      }
      if (!hasIndex)
        throw new InvalidDataException($"the archive has no {SaveIndex.FileName}");
      return dest;
    }
    catch
    {
      TryDeleteDir(dest);
      throw;
    }
  }

  /// <summary>Removes work folders untouched for <see cref="WorkFolderExpiry"/>, except
  /// <paramref name="keep"/> (the open save's). A folder in use (files held open) stays.</summary>
  public static int CleanWorkRoot(string workRoot, DateTime now, string? keep = null)
  {
    if (!Directory.Exists(workRoot)) return 0;
    int removed = 0;
    foreach (var dir in Directory.GetDirectories(workRoot))
    {
      if (keep is not null && string.Equals(Path.GetFullPath(dir), Path.GetFullPath(keep), StringComparison.Ordinal))
        continue;
      DateTime last;
      try
      {
        last = Directory.GetFiles(dir, "*", SearchOption.AllDirectories).Select(File.GetLastWriteTime)
          .DefaultIfEmpty(Directory.GetLastWriteTime(dir)).Max();
      }
      catch (Exception e) when (e is IOException or UnauthorizedAccessException)
      {
        continue;
      }
      if (now - last < WorkFolderExpiry) continue;
      if (TryDeleteDir(dir)) removed++;
    }
    return removed;
  }

  private static string Relative(string folder, string file) =>
    file.Substring(folder.Length).TrimStart(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar)
      .Replace(Path.DirectorySeparatorChar, '/');

  private static bool TryDeleteDir(string dir)
  {
    try
    {
      Directory.Delete(dir, recursive: true);
      return true;
    }
    catch (Exception e) when (e is IOException or UnauthorizedAccessException)
    {
      return false;
    }
  }
}
