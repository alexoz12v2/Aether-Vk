using System;
using System.Collections.Specialized;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Data;

namespace AetherVk.Views;

public partial class HorizonJplView : UserControl
{
  private HorizonJplViewModel? _viewModel;

  public HorizonJplView()
  {
    InitializeComponent();
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    if (_viewModel != null)
    {
      _viewModel.Headers.CollectionChanged -= OnHeadersChanged;
      _viewModel.CometsHeaders.CollectionChanged -= OnCometsHeadersChanged;
    }

    _viewModel = DataContext as HorizonJplViewModel;

    if (_viewModel != null)
    {
      _viewModel.Headers.CollectionChanged += OnHeadersChanged;
      _viewModel.CometsHeaders.CollectionChanged += OnCometsHeadersChanged;
      RebuildColumns();
      RebuildCometsColumns();
    }
  }

  private void OnHeadersChanged(object? sender, NotifyCollectionChangedEventArgs e)
  {
    RebuildColumns();
  }

  private void OnCometsHeadersChanged(object? sender, NotifyCollectionChangedEventArgs e)
  {
    RebuildCometsColumns();
  }

  private void RebuildColumns()
  {
    if (_viewModel == null || ResultDataGrid == null)
      return;

    ResultDataGrid.Columns.Clear();
    for (int i = 0; i < _viewModel.Headers.Count; i++)
    {
      var header = _viewModel.Headers[i];
      ResultDataGrid.Columns.Add(
        new DataGridTextColumn { Header = header, Binding = new Binding($"[{i}]") }
      );
    }
  }

  private void RebuildCometsColumns()
  {
    if (_viewModel == null || CometsResultDataGrid == null)
      return;

    CometsResultDataGrid.Columns.Clear();
    for (int i = 0; i < _viewModel.CometsHeaders.Count; i++)
    {
      var header = _viewModel.CometsHeaders[i];
      CometsResultDataGrid.Columns.Add(
        new DataGridTextColumn { Header = header, Binding = new Binding($"[{i}]") }
      );
    }
  }

  private void OnCometsGridDoubleTapped(object? sender, Avalonia.Input.TappedEventArgs e)
  {
    if (_viewModel != null && _viewModel.SelectedComet != null)
    {
      _viewModel.FetchDataCommand.Execute(null);
    }
  }
}
