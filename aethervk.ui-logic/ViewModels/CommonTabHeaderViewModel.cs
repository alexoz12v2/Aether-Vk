using System;
using System.Collections.Generic;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Lightweight per-session row ViewModel used by <c>CommonTabHeader</c>'s session list.
/// </summary>
public partial class CommonTabHeaderSessionItemViewModel : ObservableObject
{
  public string Label { get; }
  public IRelayCommand SwitchCommand { get; }
  public IRelayCommand DeleteCommand { get; }

  public CommonTabHeaderSessionItemViewModel(
    string         label,
    Action         switchAction,
    Action         deleteAction)
  {
    Label         = label;
    SwitchCommand = new RelayCommand(switchAction);
    DeleteCommand = new RelayCommand(deleteAction);
  }
}

/// <summary>
/// DataContext for <c>CommonTabHeader</c>.
/// Built from the selected tab's <see cref="IStatefulTabHeader"/> implementation in the
/// <c>CommonTabHeader</c> code-behind whenever <c>SelectedTab</c> changes.
/// Provides a fully concrete, compiled-binding-safe type for the AXAML.
/// </summary>
public partial class CommonTabHeaderViewModel : ObservableObject
{
  [ObservableProperty] private string _sessionLabel = string.Empty;
  [ObservableProperty] private bool   _canAddSession;
  [ObservableProperty] private bool   _isExclusiveSession;
  [ObservableProperty] private IReadOnlyList<CommonTabHeaderSessionItemViewModel> _sessionItems
    = Array.Empty<CommonTabHeaderSessionItemViewModel>();

  public IRelayCommand NewSessionCommand { get; }

  public CommonTabHeaderViewModel(IStatefulTabHeader source)
  {
    NewSessionCommand = source.NewSessionCommand;
    Refresh(source);
  }

  /// <summary>Rebuilds the ViewModel from the current state of the source header.</summary>
  public void Refresh(IStatefulTabHeader source)
  {
    SessionLabel       = source.SessionId.ToString();
    IsExclusiveSession = source.IsExclusiveSession;
    CanAddSession      = !source.IsExclusiveSession;
    SessionItems       = source.AvailableSessions
      .Select(id => new CommonTabHeaderSessionItemViewModel(
        id.ToString(),
        () => source.SwitchSessionCommand.Execute(id.Number),
        () => source.DeleteSessionCommand.Execute(id.Number)))
      .ToArray();
  }
}
