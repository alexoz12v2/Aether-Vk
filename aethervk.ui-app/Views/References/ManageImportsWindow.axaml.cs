using Avalonia.Controls;
using Avalonia.Interactivity;

namespace AetherVk.Views;

public partial class ManageImportsWindow : Window
{
  public ManageImportsWindow()
  {
    InitializeComponent();
  }

  private void CloseButton_Click(object? sender, RoutedEventArgs e)
  {
    Close();
  }
}
