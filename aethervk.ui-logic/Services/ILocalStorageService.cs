using System;
using System.IO;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

public interface ILocalStorageService : IDisposable
{
  string PersistentDirectory { get; }
  string SessionDirectory { get; }
  string SessionId { get; }

  string GetPersistentPath(string relativePath);
  string GetSessionPath(string relativePath);

  /// <summary>
  /// Returns the absolute path for a downloaded file (SPK kernels): <c>&lt;session&gt;/spk/{fileName}</c>,
  /// removed with the session (saves copy what they need).
  /// </summary>
  string GetDownloadsPath(string fileName);

  Task SavePersistentAsync(string relativePath, byte[] data);
  Task SavePersistentAsync(string relativePath, Stream data);

  Task SaveSessionAsync(string relativePath, byte[] data);
  Task SaveSessionAsync(string relativePath, Stream data);
}

