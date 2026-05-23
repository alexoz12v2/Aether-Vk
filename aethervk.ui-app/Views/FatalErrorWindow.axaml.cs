using Avalonia.Controls;
using Avalonia.Interactivity;
using System;

namespace AetherVk.Views;

public partial class FatalErrorWindow : Window
{
    public FatalErrorWindow()
    {
        InitializeComponent();
    }

    public FatalErrorWindow(string message) : this()
    {
        ErrorMessageText.Text = message;
    }

    private void OnExitClicked(object? sender, RoutedEventArgs e)
    {
        Environment.Exit(1);
    }
}
