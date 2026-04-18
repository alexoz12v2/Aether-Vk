using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public enum AppTheme
{
    System,
    Light,
    Dark
}

public partial class MainWindowViewModel : ViewModelBase
{
    [ObservableProperty]
    private DockingManagerViewModel _dockingManager = new();

    [ObservableProperty]
    private AppTheme _currentTheme;

    public MainWindowViewModel()
    {
        // Set initial theme to system default
        CurrentTheme = AppTheme.System;
    }

    [RelayCommand]
    private void ToggleTheme()
    {
        // Simple toggle: if Dark -> Light, otherwise -> Dark (covers Light and System)
        CurrentTheme = CurrentTheme == AppTheme.Dark ? AppTheme.Light : AppTheme.Dark;
    }
}
