using Avalonia.Controls;
using Avalonia.Interactivity;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Views;

public partial class SpawnMeshDialogWindow : Window
{
    public SpawnMeshDialogWindow()
    {
        InitializeComponent();
    }

    private void OnCancelClick(object? sender, RoutedEventArgs e)
    {
        Close(false);
    }

    private void OnSpawnClick(object? sender, RoutedEventArgs e)
    {
        Close(true);
    }
}
