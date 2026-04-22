#if DEBUG
namespace AetherVk.Logic.ViewModels.Design;

public class TabGroupNodeViewModelDesign : TabGroupNodeViewModel
{
  public TabGroupNodeViewModelDesign()
    : base(new TabItemViewModel(title: "Home"))
  {
    Tabs.Add(new TabItemViewModel(title: "Settings"));
    Tabs.Add(new TabItemViewModel(title: "Logs"));
  }
}

public class SplitNodeViewModelDesign : SplitNodeViewModel
{
  public SplitNodeViewModelDesign()
    : base(
      new TabGroupNodeViewModelDesign(),
      new TabGroupNodeViewModelDesign(),
      SplitOrientation.Horizontal
    ) { }
}
#endif
