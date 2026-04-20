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

    WeakReferenceMessenger.Default.Register<OpenMeshViewerMessage>(this, (r, m) =>
    {
       var isLightTheme = Avalonia.Application.Current?.ActualThemeVariant == Avalonia.Styling.ThemeVariant.Light;
       var meshViewer = new Views.MeshViewerWindow
       {
           DataContext = new MeshViewerViewModel(m.Model.FullPath, m.Model.Name, isLightTheme)
       };
       meshViewer.Show(this);
    });

    KeyDown += OnKeyDown;
  }

  private void ShowImportedModelsFlyout()
  {
    if (DataContext is MainWindowViewModel vm)
    {
      var window = new Views.ManageImportsWindow
      {
        DataContext = vm
      };
      window.ShowDialog(this);
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

  private async Task ShowImportImageDialogAsync()
  {
    var dialog = new OpenFileDialog
    {
      Title = "Import Image",
      AllowMultiple = false,
      Filters = new System.Collections.Generic.List<FileDialogFilter>
      {
        new FileDialogFilter { Name = "Images", Extensions = { "png", "jpg", "jpeg", "bmp", "tga" } }
      }
    };

    var result = await dialog.ShowAsync(this);
    if (result != null && result.Length > 0)
    {
      var filePath = result[0];
      try
      {
        // Parse dimensions
        using var stream = System.IO.File.OpenRead(filePath);
        var bitmap = new Avalonia.Media.Imaging.Bitmap(stream);
        float width = (float)bitmap.Size.Width;
        float height = (float)bitmap.Size.Height;
        
        var fileName = System.IO.Path.GetFileNameWithoutExtension(filePath);

        var spawnDialog = new Views.SpawnImageDialogWindow
        {
          DataContext = new SpawnImageViewModel(fileName + " Billboard", width, height)
        };

        var dlgResult = await spawnDialog.ShowDialog<bool>(this);
        if (dlgResult && spawnDialog.DataContext is SpawnImageViewModel vm)
        {
          var runtime = ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
          if (runtime != null)
          {
            var entity = runtime.SpawnImageBillboard(vm.EntityName, vm.IsScreenSpace, vm.Width, vm.Height);
            
            var watcherService = ServiceLocator.Provider?.GetService(typeof(FileWatcherService)) as FileWatcherService;
            watcherService?.WatchImageFile(filePath, entity);
          }
        }
      }
      catch (System.Exception ex)
      {
        var breadcrumb = ServiceLocator.Provider?.GetService(typeof(BreadcrumbService)) as BreadcrumbService;
        breadcrumb?.ShowMessageAsync("Import Error", $"Failed to load image: {ex.Message}", default, 3);
      }
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
            vm.ImportedModels.Add(new ImportedModelItem(modelId, fileName, filePath));
          }
        }
      }
    }
  }
}
