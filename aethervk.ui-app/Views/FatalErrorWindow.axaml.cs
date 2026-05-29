using System;
using Avalonia.Controls;
using Avalonia.Interactivity;

namespace AetherVk.Views;

public partial class FatalErrorWindow : Window
{
  public FatalErrorWindow()
  {
    InitializeComponent();
  }

  public FatalErrorWindow(string message)
    : this()
  {
    ErrorMessageText.Text = message;
  }

  private void OnExitClicked(object? sender, RoutedEventArgs e)
  {
    Environment.Exit(1);
  }
}
