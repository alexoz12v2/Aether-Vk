using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class SettingsViewModel : ViewModelBase
{
    public ObservableCollection<ISettingsCategory> Categories { get; } = new();

    [ObservableProperty]
    private ISettingsCategory? _selectedCategory;

    public SettingsViewModel(AetherVk.Logic.Input.InputRegistry inputRegistry)
    {
        Categories.Add(new KeyBindingsViewModel(inputRegistry));
        SelectedCategory = Categories[0];
    }
}
