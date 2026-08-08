using System;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.ApplicationLifetimes;

namespace AetherVk.Services
{
  public class AvaloniaWindowService(
      INativeRuntimeService runtimeService,
      BreadcrumbService breadcrumbService,
      ConsoleService consoleService,
      IUiThreadDispatcher uiThreadDispatcher,
      HorizonJplService horizonService
    ) : IWindowService
  {
    private readonly INativeRuntimeService _runtimeService = runtimeService;
    private readonly BreadcrumbService _breadcrumbService = breadcrumbService;
    private readonly ConsoleService _consoleService = consoleService;
    private readonly IUiThreadDispatcher _uiThreadDispatcher = uiThreadDispatcher;
    private readonly HorizonJplService _horizonService = horizonService;

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

    public Task ShowSpawnImageDialogAsync(string imagePath)
    {
      throw new NotImplementedException();
    }

    public Task ShowManageImportsDialogAsync()
    {
      throw new NotImplementedException();
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
      throw new NotImplementedException();
    }

    public Task<ulong> ShowSpawnMeshDialogAsync(string modelId, string modelName)
    {
      throw new NotImplementedException();
    }

    public Task<ulong> ShowSpawnCometDialogAsync(
      System.Collections.Generic.IEnumerable<object> models, // TODO: restore ImportedModelItem when SpawnCometWindow rework is complete
      ulong? preselectedModelId = null
    )
    {
      throw new NotImplementedException();
    }

    public Task ShowSpawnBillboardDialogAsync()
    {
      throw new NotImplementedException();
    }

    public Task<(double X, double Y, double Z)?> ShowSnapObserverDialogAsync()
    {
      throw new NotImplementedException();
    }
  }
}
