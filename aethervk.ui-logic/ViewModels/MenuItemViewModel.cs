using System.Collections.ObjectModel;
using System.Windows.Input;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>How a menu entry shows its checked state (mapped to the toolkit's toggle types by the view).</summary>
public enum MenuToggle
{
  None,
  CheckBox,
  /// <summary>One of the entries sharing <see cref="MenuItemViewModel.GroupName"/> is checked.</summary>
  Radio,
}

public partial class MenuItemViewModel : ObservableObject
{
  [ObservableProperty]
  private string? _header;

  [ObservableProperty]
  private ICommand? _command;

  [ObservableProperty]
  private bool _isVisible = true;

  [ObservableProperty]
  private bool _isEnabled = true;

  [ObservableProperty]
  private bool _isSeparator;

  [ObservableProperty]
  private string? _gesture;

  /// <summary>None for a plain entry; CheckBox or Radio for an entry with a checked state.</summary>
  [ObservableProperty]
  private MenuToggle _toggleType;

  /// <summary>The checked state of a CheckBox or Radio entry (the view writes it back on click).</summary>
  [ObservableProperty]
  private bool _isChecked;

  /// <summary>The radio group of a Radio entry.</summary>
  [ObservableProperty]
  private string? _groupName;

  public ObservableCollection<MenuItemViewModel> Items { get; } = new ObservableCollection<MenuItemViewModel>();
}
