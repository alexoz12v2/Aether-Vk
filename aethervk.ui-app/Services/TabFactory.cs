using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Services
{
  public class TabFactory(
      Func<UITestPanelViewModel> createUITestPanel,
      Func<ConsoleViewModel> createConsole,
      Func<DebugUiViewModel> createDebugUi,
      Func<Viewport3DViewModel> createViewport
    ) : ITabFactory
  {
    private readonly Func<UITestPanelViewModel> _createUITestPanel = createUITestPanel;
    private readonly Func<ConsoleViewModel> _createConsole = createConsole;
    private readonly Func<DebugUiViewModel> _createDebugUi = createDebugUi;
    private readonly Func<Viewport3DViewModel> _createViewport = createViewport;

    public object CreateTab(string tabType)
    {
      // TODO enum string
      return tabType switch
      {
        "UITestPanel" => _createUITestPanel(),
        "Console" => _createConsole(),
        "DebugUi" => _createDebugUi(),
        "Viewport3D" => _createViewport(),
        _ => throw new ArgumentException($"Unknown tab type: {tabType}"),
      };
    }
  }
}
