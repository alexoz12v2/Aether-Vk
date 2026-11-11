using System;
using System.ComponentModel;
using AetherVk.Controls;
using AetherVk.Logic.Automation;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;

namespace AetherVk.Views;

/// <summary>
/// The "Nucleus seen from Earth" view of the compact panel (see NucleusMiniView.axaml). It creates
/// the native control once, when <see cref="NucleusMiniViewModel.Viewport"/> first exists, keeps it
/// square at <see cref="NucleusMiniViewModel.HostSide"/> of the width it is given, and tells the
/// view model when the window is minimized.
/// </summary>
public partial class NucleusMiniView : UserControl
{
  // the Expander's content padding and border, both sides: not available to the native area
  private const double ExpanderChrome = 18.0;

  private NucleusMiniViewModel? _vm;
  private VulkanViewportControl? _host;
  private IDisposable? _windowState;

  public NucleusMiniView()
  {
    InitializeComponent();
    DataContextChanged += (_, _) => OnViewModelChanged();
    SizeChanged += (_, _) => UpdateSide();
  }

  /// <summary>The native control, null until the view was first shown.</summary>
  public VulkanViewportControl? Host => _host;

  private void OnViewModelChanged()
  {
    if (_vm is not null)
      _vm.PropertyChanged -= OnViewModelPropertyChanged;
    _vm = DataContext as NucleusMiniViewModel;
    if (_vm is not null)
      _vm.PropertyChanged += OnViewModelPropertyChanged;
    EnsureHost();
    PushWindowState();
  }

  private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs e)
  {
    if (e.PropertyName is nameof(NucleusMiniViewModel.Viewport))
      EnsureHost();
    else if (e.PropertyName is nameof(NucleusMiniViewModel.IsHostVisible) && _host is not null && _vm is not null)
      _host.IsVisible = _vm.IsHostVisible;
  }

  // once: the control is never removed nor moved afterwards (that would destroy its native window)
  private void EnsureHost()
  {
    if (_host is not null || _vm?.Viewport is not { } viewport)
      return;
    double side = Side();
    _host = new VulkanViewportControl
    {
      // read by the control when it creates its native window: set before it enters the tree
      DataContext = viewport,
      Width = side,
      Height = side,
      IsVisible = _vm.IsHostVisible,
    };
    AutomationProperties.SetAutomationId(_host, CompactAutomationIds.NucleusMiniViewHost);
    HostSlot.Children.Insert(0, _host);
    UpdateSide();
  }

  private double Side() => Math.Max(1.0, NucleusMiniViewModel.HostSide(Bounds.Width - ExpanderChrome));

  private void UpdateSide()
  {
    double side = Side();
    HostSlot.Width = side;
    HostSlot.Height = side;
    if (_host is not null)
    {
      _host.Width = side;
      _host.Height = side;
    }
  }

  protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    if (TopLevel.GetTopLevel(this) is Window window)
      _windowState = window.GetObservable(Window.WindowStateProperty).Subscribe(new StateObserver(this));
  }

  protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
  {
    _windowState?.Dispose();
    _windowState = null;
    base.OnDetachedFromVisualTree(e);
  }

  private void PushWindowState()
  {
    if (_vm is not null && TopLevel.GetTopLevel(this) is Window window)
      _vm.IsWindowMinimized = window.WindowState == WindowState.Minimized;
  }

  private sealed class StateObserver(NucleusMiniView view) : IObserver<WindowState>
  {
    public void OnCompleted() { }
    public void OnError(Exception error) { }
    public void OnNext(WindowState value) => view.PushWindowState();
  }
}
