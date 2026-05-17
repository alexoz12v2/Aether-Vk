using System.Linq;
using AetherVk.Logic.ViewModels;
using AetherVk.Views;
using Avalonia.Controls;
using Avalonia.LogicalTree;
using Avalonia.Headless.XUnit;
using Xunit;

namespace AetherVk.AppTests;

public class UITestPanelTests
{
  [AvaloniaFact]
  public void UITestPanelView_Should_Render_And_Initialize()
  {
    // Arrange
    var viewModel = new UITestPanelViewModel();
    var view = new UITestPanelView { DataContext = viewModel };

    // Act
    var window = new Window { Content = view };

    window.Show();

    // Assert
    Assert.NotNull(view);

    // Find a TextBlock to ensure view is loaded
    var textBlocks = view.GetLogicalDescendants().OfType<TextBlock>().ToList();
    var titleBlock = textBlocks.FirstOrDefault(tb => tb.Text == "UI Components Test Panel");

    Assert.NotNull(titleBlock);
  }
}
