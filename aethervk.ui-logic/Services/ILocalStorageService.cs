using System.IO;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

public interface ILocalStorageService
{
  string PersistentDirectory { get; }
  string SessionDirectory { get; }
  string SessionId { get; }

  string GetPersistentPath(string relativePath);
  string GetSessionPath(string relativePath);

  Task SavePersistentAsync(string relativePath, byte[] data);
  Task SavePersistentAsync(string relativePath, Stream data);

  Task SaveSessionAsync(string relativePath, byte[] data);
  Task SaveSessionAsync(string relativePath, Stream data);
}
