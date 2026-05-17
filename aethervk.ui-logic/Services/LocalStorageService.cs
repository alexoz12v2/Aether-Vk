using System;
using System.IO;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

public class LocalStorageService : ILocalStorageService
{
    public string PersistentDirectory { get; }
    public string SessionDirectory { get; }
    public string SessionId { get; }

    public LocalStorageService()
    {
        var homeDir = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        PersistentDirectory = Path.Combine(homeDir, ".aethervk");
        
        SessionId = Guid.NewGuid().ToString("N");
        SessionDirectory = Path.Combine(PersistentDirectory, SessionId);

        EnsureDirectories();
        CleanupOldSessions();
    }

    private void EnsureDirectories()
    {
        if (!Directory.Exists(PersistentDirectory))
        {
            Directory.CreateDirectory(PersistentDirectory);
        }
        if (!Directory.Exists(SessionDirectory))
        {
            Directory.CreateDirectory(SessionDirectory);
        }
    }

    private void CleanupOldSessions()
    {
        try
        {
            if (!Directory.Exists(PersistentDirectory)) return;

            var dirs = Directory.GetDirectories(PersistentDirectory);
            var cutoffTime = DateTime.UtcNow.AddDays(-1);

            foreach (var dir in dirs)
            {
                var dirInfo = new DirectoryInfo(dir);
                // Skip the current session directory
                if (dirInfo.Name == SessionId) continue;

                // Check if it matches GUID format (length 32 for "N" format)
                if (dirInfo.Name.Length == 32 && Guid.TryParseExact(dirInfo.Name, "N", out _))
                {
                    if (dirInfo.CreationTimeUtc < cutoffTime)
                    {
                        try
                        {
                            dirInfo.Delete(true);
                        }
                        catch
                        {
                            // Ignore if we can't delete right now (e.g. file in use)
                        }
                    }
                }
            }
        }
        catch
        {
            // Logging can be added here if needed, but startup cleanup should not crash the app
        }
    }

    public string GetPersistentPath(string relativePath)
    {
        var path = Path.Combine(PersistentDirectory, relativePath);
        EnsureParentDirectoryExists(path);
        return path;
    }

    public string GetSessionPath(string relativePath)
    {
        var path = Path.Combine(SessionDirectory, relativePath);
        EnsureParentDirectoryExists(path);
        return path;
    }

    private void EnsureParentDirectoryExists(string path)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir) && !Directory.Exists(dir))
        {
            Directory.CreateDirectory(dir);
        }
    }

    public async Task SavePersistentAsync(string relativePath, byte[] data)
    {
        var path = GetPersistentPath(relativePath);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true);
        await fs.WriteAsync(data, 0, data.Length);
    }

    public async Task SavePersistentAsync(string relativePath, Stream data)
    {
        var path = GetPersistentPath(relativePath);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true);
        data.Position = 0;
        await data.CopyToAsync(fs);
    }

    public async Task SaveSessionAsync(string relativePath, byte[] data)
    {
        var path = GetSessionPath(relativePath);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true);
        await fs.WriteAsync(data, 0, data.Length);
    }

    public async Task SaveSessionAsync(string relativePath, Stream data)
    {
        var path = GetSessionPath(relativePath);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true);
        data.Position = 0;
        await data.CopyToAsync(fs);
    }
}
