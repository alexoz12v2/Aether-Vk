using Avalonia.Controls;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Controls;

/// <summary>
/// Code-behind for CommonTabHeader.
/// Watches <see cref="Control.DataContext"/> changes and builds/refreshes a
/// <see cref="CommonTabHeaderViewModel"/> whenever the selected tab changes.
/// All session logic stays in <see cref="StatefulTabViewModelBase{TSession}"/>.
/// </summary>
public partial class CommonTabHeader : UserControl
{
  public CommonTabHeader()
  {
    InitializeComponent();
    DataContextChanged += OnDataContextChanged;
  }

  private void OnDataContextChanged(object? sender, System.EventArgs e)
  {
    if (DataContext is IStatefulTabHeader header)
    {
      if (base.DataContext is CommonTabHeaderViewModel existing)
      {
        existing.Refresh(header);
      }
      else
      {
        // Replace with a concrete typed VM so the AXAML compiled bindings work.
        // We assign it back to DataContext, which will trigger this handler again
        // but the guard above will take the Refresh path instead.
        base.DataContext = new CommonTabHeaderViewModel(header);
      }
    }
  }
}
