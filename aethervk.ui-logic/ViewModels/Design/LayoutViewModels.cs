#if DEBUG
namespace AetherVk.Logic.ViewModels.Design;

public class TabGroupNodeViewModelDesign : TabGroupNodeViewModel
{
  public TabGroupNodeViewModelDesign()
    : base(new TabItemViewModel(title: "Home"), null!, null!)
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
public class DockingManagerViewModelDesign : DockingManagerViewModel
{
  public DockingManagerViewModelDesign()
    : base(new SplitNodeViewModelDesign()) { }
}

public class MainWindowViewModelDesign : MainWindowViewModel
{
  public MainWindowViewModelDesign()
    : base(null!, null!, null!, null!, null!, new DockingManagerViewModelDesign(), null!, null!, null!) { }
}
#endif
