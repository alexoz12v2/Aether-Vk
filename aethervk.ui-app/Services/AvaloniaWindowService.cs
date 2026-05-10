using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;

namespace AetherVk.Services
{
  public class AvaloniaWindowService : IWindowService
  {
    private readonly NativeRuntimeService _runtimeService;
    private readonly BreadcrumbService _breadcrumbService;
    private readonly FileWatcherService _fileWatcherService;
    private readonly ConsoleService _consoleService;

    public AvaloniaWindowService(
      NativeRuntimeService runtimeService,
      BreadcrumbService breadcrumbService,
      FileWatcherService fileWatcherService,
      ConsoleService consoleService
    )
    {
      _runtimeService = runtimeService;
      _breadcrumbService = breadcrumbService;
      _fileWatcherService = fileWatcherService;
      _consoleService = consoleService;
    }

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

    public async Task ShowSpawnImageDialogAsync(string imagePath)
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return;

      try
      {
        // Parse dimensions
        using var stream = System.IO.File.OpenRead(imagePath);
        var bitmap = new Avalonia.Media.Imaging.Bitmap(stream);
        float width = (float)bitmap.Size.Width;
        float height = (float)bitmap.Size.Height;

        var fileName = System.IO.Path.GetFileNameWithoutExtension(imagePath);

        var spawnDialog = new Views.SpawnImageDialogWindow
        {
          DataContext = new SpawnImageViewModel(fileName + " Billboard", width, height),
        };

        var dlgResult = await spawnDialog.ShowDialog<bool>(mainWindow);
        if (dlgResult && spawnDialog.DataContext is SpawnImageViewModel vm)
        {
          var entity = _runtimeService.SpawnImageBillboard(
            1,
            vm.EntityName,
            vm.IsScreenSpace,
            vm.Width,
            vm.Height
          );

          _fileWatcherService.WatchImageFile(imagePath, entity);
        }
      }
      catch (Exception ex)
      {
        _breadcrumbService.ShowMessageAsync(
          "Import Error",
          $"Failed to load image: {ex.Message}",
          default,
          3
        );
      }
    }

    public async Task ShowManageImportsDialogAsync()
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return;

      var window = new Views.ManageImportsWindow { DataContext = mainWindow.DataContext };
      await window.ShowDialog(mainWindow);
    }

    public async Task ShowSettingsDialogAsync()
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return;

      var inputRegistry =
        Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<AetherVk.Logic.Input.InputRegistry>(
          App.Host!.Services
        );

      var settingsWindow = new Views.SettingsWindow
      {
        DataContext = new SettingsViewModel(inputRegistry),
      };

      await settingsWindow.ShowDialog(mainWindow);
    }

    public Task OpenMeshViewerAsync(string meshId)
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return Task.CompletedTask;

      var isLightTheme =
        Application.Current?.ActualThemeVariant == Avalonia.Styling.ThemeVariant.Light;
      var vm = mainWindow.DataContext as MainWindowViewModel;
      var model =
        vm != null
          ? System.Linq.Enumerable.FirstOrDefault(vm.ImportedModels, m => m.Id.ToString() == meshId)
          : null;

      if (model != null)
      {
        var meshViewer = new Views.MeshViewerWindow
        {
          DataContext = new MeshViewerViewModel(
            model.Id,
            model.FullPath,
            model.Name,
            isLightTheme,
            _runtimeService,
            _consoleService
          ),
        };
        var inputRegistry =
          Microsoft.Extensions.DependencyInjection.ServiceProviderServiceExtensions.GetRequiredService<AetherVk.Logic.Input.InputRegistry>(
            App.Host!.Services
          );
        _ = new AetherVk.Input.GlobalInputRouter(meshViewer, inputRegistry);
        meshViewer.Show(mainWindow);
      }
      return Task.CompletedTask;
    }

    public async Task<ulong> ShowSpawnMeshDialogAsync(string modelId, string modelName)
    {
      var mainWindow = GetMainWindow();
      if (mainWindow == null)
        return 0;

      var dialog = new Views.SpawnMeshDialogWindow
      {
        DataContext = new SpawnMeshViewModel(modelName + " Instance"),
      };

      var result = await dialog.ShowDialog<bool>(mainWindow);
      if (result && dialog.DataContext is SpawnMeshViewModel vm)
      {
        return await _runtimeService.SpawnModelInstanceAsync(
          1,
          ulong.Parse(modelId),
          vm.EntityName,
          vm.PosX,
          vm.PosY,
          vm.PosZ
        );
      }

      return 0;
    }
  }
}
