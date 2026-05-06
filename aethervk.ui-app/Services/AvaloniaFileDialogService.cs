using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;

namespace AetherVk.Services
{
  public class AvaloniaFileDialogService : IFileDialogService
  {
    private Window? GetMainWindow()
    {
      if (
        Application.Current?.ApplicationLifetime is IClassicDesktopStyleApplicationLifetime desktop
      )
      {
        return desktop.MainWindow;
      }
      return null;
    }

    public async Task<string?> ShowOpenFileDialogAsync(string title, string[]? filters = null)
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return null;

      var dialog = new OpenFileDialog { Title = title, AllowMultiple = false };

      if (filters != null && filters.Length > 0)
      {
        dialog.Filters = new List<FileDialogFilter>
        {
          new FileDialogFilter { Name = "Files", Extensions = filters.ToList() },
        };
      }

      var result = await dialog.ShowAsync(mainWindow);
      return result?.FirstOrDefault();
    }

    public async Task<string?> ShowSaveFileDialogAsync(
      string title,
      string defaultExtension,
      string[]? filters = null
    )
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return null;

      var dialog = new SaveFileDialog { Title = title, DefaultExtension = defaultExtension };

      if (filters != null && filters.Length > 0)
      {
        dialog.Filters = new List<FileDialogFilter>
        {
          new FileDialogFilter { Name = "Files", Extensions = filters.ToList() },
        };
      }

      var result = await dialog.ShowAsync(mainWindow);
      return result;
    }
  }
}
