using AetherVk.Views;
using Avalonia.Controls;
using Avalonia.LogicalTree;
using Avalonia.Headless.XUnit;
using System.Linq;
using Xunit;

namespace AetherVk.AppTests;

public class FatalErrorWindowTests
{
  [AvaloniaFact]
  public void FatalErrorWindow_Should_Render_And_Initialize()
  {
    var window = new FatalErrorWindow("Test error message");
    
    window.Show();

    Assert.NotNull(window);
    Assert.True(window.IsVisible);

    var textBlocks = window.GetLogicalDescendants().OfType<TextBlock>().ToList();
    var messageBlock = textBlocks.FirstOrDefault(tb => tb.Text == "Test error message");
    Assert.NotNull(messageBlock);
  }
}
