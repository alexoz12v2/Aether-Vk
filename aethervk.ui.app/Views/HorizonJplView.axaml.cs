using System;
using System.Collections.Specialized;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Data;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Views;

public partial class HorizonJplView : UserControl, IRecipient<RequestSaveFileMessage>
{
  private HorizonJplViewModel? _viewModel;

  public HorizonJplView()
  {
    InitializeComponent();
  }

  protected override void OnAttachedToVisualTree(Avalonia.VisualTreeAttachmentEventArgs e)
  {
      base.OnAttachedToVisualTree(e);
      WeakReferenceMessenger.Default.Register<RequestSaveFileMessage>(this);
  }

  protected override void OnDetachedFromVisualTree(Avalonia.VisualTreeAttachmentEventArgs e)
  {
      base.OnDetachedFromVisualTree(e);
      WeakReferenceMessenger.Default.Unregister<RequestSaveFileMessage>(this);
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    if (_viewModel != null)
    {
      _viewModel.CometsHeaders.CollectionChanged -= OnCometsHeadersChanged;
      _viewModel.SpkRecordsHeaders.CollectionChanged -= OnSpkRecordsHeadersChanged;
    }

    _viewModel = DataContext as HorizonJplViewModel;

    if (_viewModel != null)
    {
      _viewModel.CometsHeaders.CollectionChanged += OnCometsHeadersChanged;
      _viewModel.SpkRecordsHeaders.CollectionChanged += OnSpkRecordsHeadersChanged;
      RebuildCometsColumns();
      RebuildSpkRecordsColumns();
    }
  }

  private void OnCometsHeadersChanged(object? sender, NotifyCollectionChangedEventArgs e)
  {
    RebuildCometsColumns();
  }

  private void OnSpkRecordsHeadersChanged(object? sender, NotifyCollectionChangedEventArgs e)
  {
    RebuildSpkRecordsColumns();
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

  private void RebuildSpkRecordsColumns()
  {
    if (_viewModel == null || SpkRecordsDataGrid == null)
      return;

    SpkRecordsDataGrid.Columns.Clear();
    for (int i = 0; i < _viewModel.SpkRecordsHeaders.Count; i++)
    {
      var header = _viewModel.SpkRecordsHeaders[i];
      SpkRecordsDataGrid.Columns.Add(
        new DataGridTextColumn { Header = header, Binding = new Binding($"[{i}]") }
      );
    }
  }

  public async void Receive(RequestSaveFileMessage message)
  {
    var topLevel = TopLevel.GetTopLevel(this);
    if (topLevel == null)
    {
      message.Result.SetResult(null);
      return;
    }

    var file = await topLevel.StorageProvider.SaveFilePickerAsync(
      new Avalonia.Platform.Storage.FilePickerSaveOptions
      {
        Title = "Save SPK File",
        DefaultExtension = "bsp",
        SuggestedFileName = message.DefaultFileName,
        FileTypeChoices = new[]
        {
          new Avalonia.Platform.Storage.FilePickerFileType("BSP Files")
          {
            Patterns = new[] { "*.bsp" }
          },
          new Avalonia.Platform.Storage.FilePickerFileType("All Files")
          {
            Patterns = new[] { "*.*" }
          }
        }
      }
    );

    message.Result.SetResult(file?.Path.LocalPath);
  }
}
