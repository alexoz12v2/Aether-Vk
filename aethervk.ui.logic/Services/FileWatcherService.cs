using System;
using System.Collections.Concurrent;
using System.IO;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Services;

public class FileWatcherService : IDisposable
{
  private readonly ConcurrentDictionary<string, (FileSystemWatcher watcher, Entity entity)> _watchers = new();
  private readonly BreadcrumbService _breadcrumbService;

  public FileWatcherService(BreadcrumbService breadcrumbService)
  {
      _breadcrumbService = breadcrumbService;
  }

  public void WatchImageFile(string filePath, Entity entity)
  {
    if (_watchers.ContainsKey(filePath))
      return;

    var dir = Path.GetDirectoryName(filePath);
    var file = Path.GetFileName(filePath);
    
    if (dir == null || file == null) return;

    var watcher = new FileSystemWatcher(dir, file)
    {
      NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite | NotifyFilters.CreationTime,
      EnableRaisingEvents = true
    };

    watcher.Deleted += (s, e) => HandleFileMissing(filePath, entity);
    watcher.Renamed += (s, e) => HandleFileMissing(filePath, entity); // Treats renamed as missing
    watcher.Created += (s, e) => HandleFileRestored(filePath, entity);
    // Note: To handle rename where it comes back to the watched name, we could check e.Name == file

    _watchers[filePath] = (watcher, entity);
  }

  private void HandleFileMissing(string filePath, Entity entity)
  {
    entity.IsVisible = false;
    _breadcrumbService.ShowMessageAsync("FileWatcher", $"Image file {Path.GetFileName(filePath)} is missing!", default, 2);
  }

  private void HandleFileRestored(string filePath, Entity entity)
  {
    entity.IsVisible = true;
    _breadcrumbService.ShowMessageAsync("FileWatcher", $"Image file {Path.GetFileName(filePath)} restored.", default, 0);
  }

  public void Dispose()
  {
    foreach (var kvp in _watchers)
    {
      kvp.Value.watcher.Dispose();
    }
    _watchers.Clear();
  }
}

