using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Services
{
  public class TabFactory : ITabFactory
  {
    private readonly Func<UITestPanelViewModel> _createUITestPanel;
    private readonly Func<ConsoleViewModel> _createConsole;
    private readonly Func<DebugUiViewModel> _createDebugUi;
    private readonly Func<HorizonJplViewModel> _createHorizonJpl;
    private readonly Func<ulong, OutlineViewModel> _createOutline;
    private readonly Func<ulong, PropertiesViewModel> _createProperties;
    private readonly Func<ulong, TimelineViewModel> _createTimeline;
    private readonly Func<AlmanacExplorerViewModel> _createAlmanac;
    private readonly Func<Viewport3DViewModel> _createViewport;
    private readonly Func<AssetBrowserViewModel> _createAssetBrowser;
    private readonly SceneStateManager _stateManager;

    public TabFactory(
      Func<UITestPanelViewModel> createUITestPanel,
      Func<ConsoleViewModel> createConsole,
      Func<DebugUiViewModel> createDebugUi,
      Func<HorizonJplViewModel> createHorizonJpl,
      Func<ulong, OutlineViewModel> createOutline,
      Func<ulong, PropertiesViewModel> createProperties,
      Func<ulong, TimelineViewModel> createTimeline,
      Func<AlmanacExplorerViewModel> createAlmanac,
      Func<Viewport3DViewModel> createViewport,
      Func<AssetBrowserViewModel> createAssetBrowser,
      SceneStateManager stateManager
    )
    {
      _createUITestPanel = createUITestPanel;
      _createConsole = createConsole;
      _createDebugUi = createDebugUi;
      _createHorizonJpl = createHorizonJpl;
      _createOutline = createOutline;
      _createProperties = createProperties;
      _createTimeline = createTimeline;
      _createAlmanac = createAlmanac;
      _createViewport = createViewport;
      _createAssetBrowser = createAssetBrowser;
      _stateManager = stateManager;
    }

    public object CreateTab(string tabType)
    {
      var activeScene = System.Linq.Enumerable.FirstOrDefault(_stateManager.AllScenes);
      ulong targetSceneId = activeScene != null ? activeScene.SceneId : 1UL;

      return tabType switch
      {
        "UITestPanel" => _createUITestPanel(),
        "Console" => _createConsole(),
        "DebugUi" => _createDebugUi(),
        "HorizonJPL" => _createHorizonJpl(),
        "Almanac" => _createAlmanac(),
        "Outline" => _createOutline(targetSceneId),
        "Properties" => _createProperties(targetSceneId),
        "Timeline" => _createTimeline(targetSceneId),
        "Viewport3D" => _createViewport(),
        "AssetBrowser" => _createAssetBrowser(),
        _ => throw new ArgumentException($"Unknown tab type: {tabType}"),
      };
    }
  }
}
