using System.Collections.ObjectModel;
using System.IO;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public partial class AlmanacExplorerViewModel : TabItemViewModel
{
  private readonly NativeRuntimeService _runtimeService;

  [ObservableProperty]
  private ObservableCollection<string> _loadedFiles = new();

  [ObservableProperty]
  private bool _isInitialized;

  [ObservableProperty]
  private bool _isLoading;

  public AlmanacExplorerViewModel(NativeRuntimeService runtimeService, IUiThreadDispatcher uiThreadDispatcher)
    : base("Almanac Explorer")
  {
    _runtimeService = runtimeService;
    _runtimeService.PropertyChanged += (s, e) =>
    {
      if (e.PropertyName == nameof(NativeRuntimeService.IsInitialized))
      {
        uiThreadDispatcher.Dispatch(() =>
        {
          IsInitialized = _runtimeService.IsInitialized;
          RefreshLoadedFiles();
        });
      }
    };
    IsInitialized = _runtimeService.IsInitialized;
    RefreshLoadedFiles();
  }

  [RelayCommand]
  private void RefreshLoadedFiles()
  {
    LoadedFiles.Clear();
    if (_runtimeService.IsInitialized)
    {
      var files = _runtimeService.GetLoadedAlmanacFiles();
      foreach (var f in files)
      {
        LoadedFiles.Add(Path.GetFileName(f));
      }
    }
  }

  [RelayCommand]
  private async Task LoadDefaultAlmanacs()
  {
    if (!_runtimeService.IsInitialized)
      return;

    IsLoading = true;
    await Task.Run(() => _runtimeService.LoadDefaultAlmanacs());
    IsLoading = false;

    RefreshLoadedFiles();
  }
}
