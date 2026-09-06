using System.Collections.ObjectModel;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class MenuItemViewModel : ObservableObject
{
    [ObservableProperty]
    private string? _header;

    [ObservableProperty]
    private ICommand? _command;

    [ObservableProperty]
    private bool _isVisible = true;

    [ObservableProperty]
    private bool _isSeparator;

    [ObservableProperty]
    private string? _gesture;

    public ObservableCollection<MenuItemViewModel> Items { get; } = new ObservableCollection<MenuItemViewModel>();
}
