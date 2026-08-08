using System;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;
using AetherVk.Logic.ViewModels;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Views;

public partial class Viewport3DView : UserControl
{
  private Viewport3DViewModel? _viewModel;

  public Viewport3DView()
  {
    AvaloniaXamlLoader.Load(this);
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);
    _viewModel = DataContext as Viewport3DViewModel;
  }
}
