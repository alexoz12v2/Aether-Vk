using System;
using System.ComponentModel;
using System.Globalization;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Data;
using Avalonia.Media;

namespace AetherVk.Views;

public class EnumToBoolConverter : Avalonia.Data.Converters.IValueConverter
{
  public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    if (value is SplitOrientation orientation && parameter is SplitOrientation expectedOrientation)
    {
      return orientation == expectedOrientation;
    }
    return false;
  }

  public object? ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
  {
    return null;
  }
}

/// <summary>
/// Converts a [0..1] double SplitRatio to a Star <see cref="GridLength"/> for the first pane.
/// e.g. 0.7 → GridLength("0.7*")
/// </summary>
public class RatioToStarConverter : Avalonia.Data.Converters.IValueConverter
{
  public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
    value is double d ? new GridLength(d, GridUnitType.Star) : GridLength.Auto;

  public object ConvertBack(
    object? value,
    Type targetType,
    object? parameter,
    CultureInfo culture
  ) => value is GridLength gl && gl.IsStar ? gl.Value : 0.5;
}

/// <summary>
/// Converts a [0..1] double SplitRatio to a Star <see cref="GridLength"/> for the second pane (1 - ratio).
/// e.g. 0.7 → GridLength("0.3*")
/// </summary>
public class InverseRatioToStarConverter : Avalonia.Data.Converters.IValueConverter
{
  public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture) =>
    value is double d ? new GridLength(1.0 - d, GridUnitType.Star) : GridLength.Auto;

  public object ConvertBack(
    object? value,
    Type targetType,
    object? parameter,
    CultureInfo culture
  ) => value is GridLength gl && gl.IsStar ? 1.0 - gl.Value : 0.5;
}

public partial class SplitNodeView : UserControl
{
  private SplitNodeViewModel? _vm;

  // Viewport-only collapse: the track lengths of the active grid before collapsing, and the
  // Animation-priority overrides that zero one side. Disposing an override lets the bound (or
  // splitter-dragged) length show again; a track whose override could not be undone gets its
  // saved length back.
  private GridLength[]? _savedTracks;
  private Grid? _collapsedGrid;
  private readonly IDisposable?[] _trackOverrides = new IDisposable?[3];

  public SplitNodeView()
  {
    InitializeComponent();
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    if (_vm is not null)
      _vm.PropertyChanged -= OnViewModelPropertyChanged;
    RestoreTracks();

    _vm = DataContext as SplitNodeViewModel;
    if (_vm is not null)
    {
      _vm.PropertyChanged += OnViewModelPropertyChanged;
      ApplyCollapsed(_vm.Collapsed);
    }
  }

  private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs args)
  {
    if (args.PropertyName == nameof(SplitNodeViewModel.Collapsed) && sender is SplitNodeViewModel vm)
      ApplyCollapsed(vm.Collapsed);
  }

  /// <summary>
  /// Zeroes the collapsed side's track and the splitter's track of the active orientation grid, so
  /// the shown side takes the whole area; <see cref="CollapsedSide.None"/> restores the saved tracks.
  /// The children stay where they are: only lengths and visibilities change.
  /// </summary>
  private void ApplyCollapsed(CollapsedSide side)
  {
    RestoreTracks();
    if (side == CollapsedSide.None || _vm is null)
      return;

    var horizontal = _vm.Orientation == SplitOrientation.Horizontal;
    var grid = horizontal ? HorizontalGrid : VerticalGrid;
    var star = new GridLength(1, GridUnitType.Star);
    var zero = new GridLength(0, GridUnitType.Pixel);
    GridLength[] lengths = side == CollapsedSide.Second ? [star, zero, zero] : [zero, zero, star];

    _collapsedGrid = grid;
    _savedTracks = new GridLength[3];
    for (var i = 0; i < 3; ++i)
    {
      if (horizontal)
      {
        var column = grid.ColumnDefinitions[i];
        _savedTracks[i] = column.Width;
        _trackOverrides[i] = column.SetValue(ColumnDefinition.WidthProperty, lengths[i], BindingPriority.Animation);
      }
      else
      {
        var row = grid.RowDefinitions[i];
        _savedTracks[i] = row.Height;
        _trackOverrides[i] = row.SetValue(RowDefinition.HeightProperty, lengths[i], BindingPriority.Animation);
      }
    }
  }

  private void RestoreTracks()
  {
    if (_collapsedGrid is { } grid && _savedTracks is { } saved)
    {
      for (var i = 0; i < 3; ++i)
      {
        if (_trackOverrides[i] is { } undo)
          undo.Dispose();
        else if (grid == HorizontalGrid)
          grid.ColumnDefinitions[i].Width = saved[i];
        else
          grid.RowDefinitions[i].Height = saved[i];
        _trackOverrides[i] = null;
      }
    }
    _collapsedGrid = null;
    _savedTracks = null;
  }

  private class SplitRatioToWidth(SplitNodeView self, SplitOrientation orientation)
    : Avalonia.Data.Converters.IValueConverter
  {
    public object? Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    {
      if (value is not null)
      {
        var size =
          orientation == SplitOrientation.Horizontal ? self.Bounds.Width : self.Bounds.Height;
        var dval = (double)value;
        return dval * size;
      }

      return double.NaN;
    }

    public object? ConvertBack(
      object? value,
      Type targetType,
      object? parameter,
      CultureInfo culture
    )
    {
      var size =
        orientation == SplitOrientation.Horizontal ? self.Bounds.Width : self.Bounds.Height;
      if (value is not null && size != 0)
      {
        var dval = (double)value;
        return dval / size;
      }

      return double.NaN;
    }
  }

  // private void BuildGrid(SplitNodeViewModel vm)
  // {
  //   // 1. Reset Layout
  //   ContainerGrid.Children.Clear();
  //   ContainerGrid.RowDefinitions.Clear();
  //   ContainerGrid.ColumnDefinitions.Clear();

  //   // 2. Create content control that will render the child view models using
  //   // the data templates
  //   var firstContent = new ContentControl
  //   {
  //     [!ContentControl.ContentProperty] = new Binding("FirstChild"),
  //   };
  //   var secondContent = new ContentControl
  //   {
  //     [!ContentControl.ContentProperty] = new Binding("SecondChild"),
  //   };
  //   var splitter = new GridSplitter
  //   {
  //     Background = (ISolidColorBrush)Application.Current!.Resources!["Bg.Outline"]!,
  //   };

  //   // TODO: Instead of listening to ratio, create a new binding
  //   var converter = new SplitRatioToWidth(this, vm.Orientation);
  //   if (vm.Orientation == SplitOrientation.Horizontal)
  //   {
  //     ContainerGrid.ColumnDefinitions.Add(
  //       new ColumnDefinition(vm.SplitRatio, GridUnitType.Star)
  //       {
  //         [!WidthProperty] = new Binding("SplitRatio") { Converter = converter },
  //       }
  //     );
  //     ContainerGrid.ColumnDefinitions.Add(new ColumnDefinition(GridLength.Auto));
  //     ContainerGrid.ColumnDefinitions.Add(
  //       new ColumnDefinition(1 - vm.SplitRatio, GridUnitType.Star)
  //       {
  //         [!WidthProperty] = new Binding("SplitRatio") { Converter = converter },
  //       }
  //     );

  //     Grid.SetColumn(firstContent, 0);
  //     Grid.SetColumn(splitter, 1);
  //     Grid.SetColumn(secondContent, 2);
  //   }
  //   else
  //   {
  //     ContainerGrid.RowDefinitions.Add(
  //       new RowDefinition(vm.SplitRatio, GridUnitType.Star)
  //       {
  //         [!HeightProperty] = new Binding("SplitRatio") { Converter = converter },
  //       }
  //     );
  //     ContainerGrid.RowDefinitions.Add(new RowDefinition(GridLength.Auto));
  //     ContainerGrid.RowDefinitions.Add(
  //       new RowDefinition(1 - vm.SplitRatio, GridUnitType.Star)
  //       {
  //         [!HeightProperty] = new Binding("SplitRatio") { Converter = converter },
  //       }
  //     );

  //     Grid.SetRow(firstContent, 0);
  //     Grid.SetRow(splitter, 1);
  //     Grid.SetRow(secondContent, 2);
  //   }

  //   ContainerGrid.Children.Add(firstContent);
  //   ContainerGrid.Children.Add(splitter);
  //   ContainerGrid.Children.Add(secondContent);
  // }
}
