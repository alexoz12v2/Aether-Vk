using Avalonia;
using Avalonia.Controls;

namespace AetherVk.Utils;

public class GridExtensions
{
  public static readonly AttachedProperty<string> ColumnDefinitionsProperty =
    AvaloniaProperty.RegisterAttached<GridExtensions, Grid, string>("ColumnDefinitions");

  public static readonly AttachedProperty<string> RowDefinitionsProperty =
    AvaloniaProperty.RegisterAttached<GridExtensions, Grid, string>("RowDefinitions");

  static GridExtensions()
  {
    ColumnDefinitionsProperty.Changed.AddClassHandler<Grid>(
      (grid, e) =>
      {
        if (e.NewValue is string cols)
        {
          grid.ColumnDefinitions = ColumnDefinitions.Parse(cols);
        }
      }
    );

    RowDefinitionsProperty.Changed.AddClassHandler<Grid>(
      (grid, e) =>
      {
        if (e.NewValue is string rows)
        {
          grid.RowDefinitions = RowDefinitions.Parse(rows);
        }
      }
    );
  }

  public static string GetColumnDefinitions(Grid element) =>
    element.GetValue(ColumnDefinitionsProperty);

  public static void SetColumnDefinitions(Grid element, string value) =>
    element.SetValue(ColumnDefinitionsProperty, value);

  public static string GetRowDefinitions(Grid element) => element.GetValue(RowDefinitionsProperty);

  public static void SetRowDefinitions(Grid element, string value) =>
    element.SetValue(RowDefinitionsProperty, value);
}
