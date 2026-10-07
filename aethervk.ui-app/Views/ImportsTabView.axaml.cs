using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;

namespace AetherVk.Views;

public partial class ImportsTabView : UserControl
{
  /// <summary>Below this width the mesh and texture sections are stacked instead of side by side.</summary>
  public const double StackedLayoutThreshold = 640;

  private bool? _sideBySide;

  public ImportsTabView()
  {
    InitializeComponent();
    SizeChanged += (_, e) => ApplyLayout(e.NewSize.Width);
  }

  /// <summary>Whether the sections are currently laid out as two columns (tests).</summary>
  public bool IsSideBySide => _sideBySide ?? true;

  private void ApplyLayout(double width)
  {
    bool sideBySide = width >= StackedLayoutThreshold;
    if (_sideBySide == sideBySide) return;
    _sideBySide = sideBySide;

    if (sideBySide)
    {
      SectionsGrid.ColumnDefinitions = new ColumnDefinitions("*,12,*");
      SectionsGrid.RowDefinitions = new RowDefinitions("Auto,0,Auto");
      Grid.SetColumn(MeshSection, 0);
      Grid.SetRow(MeshSection, 0);
      Grid.SetColumn(TextureSection, 2);
      Grid.SetRow(TextureSection, 0);
    }
    else
    {
      SectionsGrid.ColumnDefinitions = new ColumnDefinitions("*,0,0");
      SectionsGrid.RowDefinitions = new RowDefinitions("Auto,12,Auto");
      Grid.SetColumn(MeshSection, 0);
      Grid.SetRow(MeshSection, 0);
      Grid.SetColumn(TextureSection, 0);
      Grid.SetRow(TextureSection, 2);
    }
  }
}
