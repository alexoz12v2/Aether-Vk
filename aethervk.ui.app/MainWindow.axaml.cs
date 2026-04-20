using System;
using System.Diagnostics;
using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk;

public partial class MainWindow : Window
{
  public MainWindow()
  {
    InitializeComponent();

    WeakReferenceMessenger.Default.Register<ImportModelRequestMessage>(this, async (r, m) =>
    {
      await ShowImportModelDialogAsync();
    });

    WeakReferenceMessenger.Default.Register<OpenImportedModelsDialogMessage>(this, (r, m) =>
    {
      ShowImportedModelsFlyout();
    });

    KeyDown += OnKeyDown;
  }

  private void ShowImportedModelsFlyout()
  {
    if (DataContext is MainWindowViewModel vm)
    {
      var flyout = new MenuFlyout();
      foreach (var model in vm.ImportedModels)
      {
        var item = new MenuItem { Header = model.Name };
        var spawnItem = new MenuItem { Header = "Spawn Instance", Command = model.SpawnCommand };
        var unloadItem = new MenuItem { Header = "Unload", Command = model.UnloadCommand };
        item.Items.Add(spawnItem);
        item.Items.Add(unloadItem);
        flyout.Items.Add(item);
      }
      
      if (vm.ImportedModels.Count == 0)
      {
        flyout.Items.Add(new MenuItem { Header = "No models imported", IsEnabled = false });
      }

      flyout.ShowAt(this);
    }
  }

  private void OnKeyDown(object? sender, KeyEventArgs e)
  {
    bool isMacOs = System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX);
    
    if (isMacOs)
    {
      // Cmd + Ctrl + F
      if (e.KeyModifiers.HasFlag(KeyModifiers.Meta | KeyModifiers.Control) && e.Key == Key.F)
      {
        ToggleFullscreen();
        e.Handled = true;
      }
    }
    else
    {
      // Alt + Enter
      if (e.KeyModifiers.HasFlag(KeyModifiers.Alt) && e.Key == Key.Enter)
      {
        ToggleFullscreen();
        e.Handled = true;
      }
    }
  }

  private void ToggleFullscreen()
  {
    if (WindowState == WindowState.FullScreen)
    {
      WindowState = WindowState.Normal;
    }
    else
    {
      WindowState = WindowState.FullScreen;
    }
  }

  private async Task ShowImportModelDialogAsync()
  {
    var dialog = new OpenFileDialog
    {
      Title = "Import 3D Model",
      AllowMultiple = false,
      Filters = new System.Collections.Generic.List<FileDialogFilter>
      {
        new FileDialogFilter { Name = "GLTF/GLB Models", Extensions = { "gltf", "glb" } }
      }
    };

    var result = await dialog.ShowAsync(this);
    if (result != null && result.Length > 0)
    {
      var filePath = result[0];
      var runtime = ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
      
      if (runtime != null)
      {
        var modelId = runtime.ImportModel(filePath);
        if (modelId > 0)
        {
          if (DataContext is MainWindowViewModel vm)
          {
            var fileName = System.IO.Path.GetFileName(filePath);
            vm.ImportedModels.Add(new ImportedModelItem(modelId, fileName));
          }
        }
      }
    }
  }
}
