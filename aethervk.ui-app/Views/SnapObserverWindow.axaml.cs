using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Markup.Xaml;

namespace AetherVk.Views;

public partial class SnapObserverWindow : Window
{
  public SnapObserverWindow()
  {
    InitializeComponent();
#if DEBUG
    this.AttachDevTools();
#endif
  }

  private void InitializeComponent()
  {
    AvaloniaXamlLoader.Load(this);
  }

  private void OnCancelClick(object? sender, RoutedEventArgs e)
  {
    Close(null);
  }

  private void OnSnapClick(object? sender, RoutedEventArgs e)
  {
    if (DataContext is SnapObserverViewModel vm)
    {
      Close(vm.CalculateSimulationOffset());
    }
    else
    {
      Close(null);
    }
  }
}
