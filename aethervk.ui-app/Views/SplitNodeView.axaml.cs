using System;
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
  public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    => value is double d ? new GridLength(d, GridUnitType.Star) : GridLength.Auto;

  public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    => value is GridLength gl && gl.IsStar ? gl.Value : 0.5;
}

/// <summary>
/// Converts a [0..1] double SplitRatio to a Star <see cref="GridLength"/> for the second pane (1 - ratio).
/// e.g. 0.7 → GridLength("0.3*")
/// </summary>
public class InverseRatioToStarConverter : Avalonia.Data.Converters.IValueConverter
{
  public object Convert(object? value, Type targetType, object? parameter, CultureInfo culture)
    => value is double d ? new GridLength(1.0 - d, GridUnitType.Star) : GridLength.Auto;

  public object ConvertBack(object? value, Type targetType, object? parameter, CultureInfo culture)
    => value is GridLength gl && gl.IsStar ? 1.0 - gl.Value : 0.5;
}

public partial class SplitNodeView : UserControl
{
  public SplitNodeView()
  {
    InitializeComponent();
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    if (DataContext is SplitNodeViewModel vm)
    {
      // Run first build
      // BuildGrid(vm);

      // Listen changes at runtime
      vm.PropertyChanged += (s, args) =>
      {
        if (
          args.PropertyName == nameof(vm.Orientation)
          || args.PropertyName == nameof(vm.FirstChild)
          || args.PropertyName == nameof(vm.SecondChild)
        )
        {
          //BuildGrid(vm);
          Console.WriteLine("FJDKSLFJDKSLFJ");
        }
      };
    }
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
