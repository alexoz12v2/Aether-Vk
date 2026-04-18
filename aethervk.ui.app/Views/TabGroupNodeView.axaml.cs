using System;
using System.Threading.Tasks;
using AetherVk.Logic.Messages;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Views;

public partial class TabGroupNodeView : UserControl, IDragSourceView
{
  private static readonly DataFormat<string> ChildViewModelFormat =
    DataFormat.CreateStringApplicationFormat("AetherVk.TabItemViewModel");

  private static TabItemViewModel? _draggedTabReference;

  private bool _isInitiatingDrag;
  private Point _dragStartPoint;
  private TabItemViewModel? _tabToDrag;

  public TabGroupNodeView()
  {
    InitializeComponent();

    // 1. CRITICAL FIX: Forces Avalonia to accept drops even if you forget it in XAML
    DragDrop.SetAllowDrop(this, true);

    // Register to listen for the cleanup message from the logic layer
    WeakReferenceMessenger.Default.Register<DragCompletedMessage>(this, (_, m) =>
    {
      if (ReferenceEquals(m.View, this))
      {
        Avalonia.Threading.Dispatcher.UIThread.InvokeAsync(ClearDragState);
      }
    });
  }

  // NOTE: Avalonia Drag and Drop events are often registered via XAML in the new versions to work properly with compiled bindings.
  // Instead of code-behind AddHandler in the constructor, we ensure we use OnDragOver, OnDragLeave, OnDrop methods
  // bound using the attached events on the UserControl level.
  // Since they are attached events, we MUST use AddHandler after the visual tree is built, or in XAML.
  // Adding them in OnAttachedToVisualTree ensures they are registered correctly for the routed event system.
  protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    AddHandler(DragDrop.DragOverEvent, OnDragOver, Avalonia.Interactivity.RoutingStrategies.Bubble,
      true);
    AddHandler(DragDrop.DragLeaveEvent, OnDragLeave,
      Avalonia.Interactivity.RoutingStrategies.Bubble, true);
    AddHandler(DragDrop.DropEvent, OnDrop, Avalonia.Interactivity.RoutingStrategies.Bubble, true);

    // Pointer Handlers for tracking the drag.
    AddHandler(InputElement.PointerMovedEvent, OnTabPointerMoved,
      Avalonia.Interactivity.RoutingStrategies.Bubble, handledEventsToo: true);
    AddHandler(InputElement.PointerReleasedEvent, OnTabPointerReleased,
      Avalonia.Interactivity.RoutingStrategies.Bubble, handledEventsToo: true);
  }

  protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnDetachedFromVisualTree(e);
    RemoveHandler(DragDrop.DragOverEvent, OnDragOver);
    RemoveHandler(DragDrop.DragLeaveEvent, OnDragLeave);
    RemoveHandler(DragDrop.DropEvent, OnDrop);

    RemoveHandler(InputElement.PointerMovedEvent, OnTabPointerMoved);
    RemoveHandler(InputElement.PointerReleasedEvent, OnTabPointerReleased);
  }

  #region Drag Handlers

  private void OnDragOver(object? sender, DragEventArgs e)
  {
    if (!e.DataTransfer.Contains(ChildViewModelFormat) || _draggedTabReference is null)
    {
      e.DragEffects = DragDropEffects.None;
      e.Handled = true;
      return;
    }

    e.DragEffects = e.KeyModifiers.HasFlag(KeyModifiers.Control)
      ? DragDropEffects.Copy
      : DragDropEffects.Move;
    var pos = e.GetPosition(this);
    var zone = CalculateDockZone(pos, Bounds.Size);
    ShowPreview(zone);
    e.Handled = true;
  }

  private void OnDragLeave(object? sender, DragEventArgs e)
  {
    PreviewBox.IsVisible = false;
    e.Handled = true;
  }

  private void OnDrop(object? sender, DragEventArgs e)
  {
    PreviewBox.IsVisible = false;

    if (e.DataTransfer.Contains(ChildViewModelFormat) &&
        _draggedTabReference is { } draggedTab && DataContext is TabGroupNodeViewModel targetNode)
    {
      bool isCopy = e.KeyModifiers.HasFlag(KeyModifiers.Control) ||
                    e.DragEffects == DragDropEffects.Copy;
      var pos = e.GetPosition(this);
      var zone = CalculateDockZone(pos, Bounds.Size);

      WeakReferenceMessenger.Default.Send(new TabDroppedMessage(draggedTab, targetNode, zone,
        isCopy));

      e.DragEffects = isCopy ? DragDropEffects.Copy : DragDropEffects.Move;
    }

    _draggedTabReference = null;
    e.Handled = true;
  }

  #endregion

  private void OnTabPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    if (e.GetCurrentPoint(this).Properties.IsLeftButtonPressed &&
        sender is Border { DataContext: TabItemViewModel clickedTab } &&
        DataContext is TabGroupNodeViewModel vm)
    {
      vm.SelectedTab = clickedTab;
    }
  }

  private void OnTabIconPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed) return;

    if (sender is not Border { DataContext: TabItemViewModel clickedTab }) return;

    // UX Fix: Select the tab when grabbing the drag handle!
    if (DataContext is TabGroupNodeViewModel vm) vm.SelectedTab = clickedTab;

    e.Handled = true;
    _isInitiatingDrag = true;
    _dragStartPoint = e.GetPosition(this);
    _tabToDrag = clickedTab;
  }

  private void OnTabPointerMoved(object? sender, PointerEventArgs e)
  {
    if (!_isInitiatingDrag || _tabToDrag == null) return;

    if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
    {
      _isInitiatingDrag = false;
      _tabToDrag = null;
      return;
    }

    var currentPoint = e.GetPosition(this);
    var dx = currentPoint.X - _dragStartPoint.X;
    var dy = currentPoint.Y - _dragStartPoint.Y;

    if (Math.Sqrt(dx * dx + dy * dy) < 10)
    {
      return;
    }

    _isInitiatingDrag = false;
    StartDrag(e, _tabToDrag);
    _tabToDrag = null;
  }

  private void OnTabPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    _isInitiatingDrag = false;
    _tabToDrag = null;
  }

  private void StartDrag(PointerEventArgs e, TabItemViewModel draggedTab)
  {
    if (DataContext is TabGroupNodeViewModel vm && vm.IsRoot() && vm.Tabs.Count <= 1) return;

    var data = new DataTransfer();
    var item = new DataTransferItem();
    item.Set(ChildViewModelFormat, "DragActive");
    data.Add(item);

    _draggedTabReference = draggedTab;

    // Store task, we do not await it here per requirement to not use async/await
    var dragTask = DragDrop.DoDragDropAsync(e, data, DragDropEffects.Move | DragDropEffects.Copy);

    // Provide the task to the logic layer
    var logicalTask = dragTask.ContinueWith(t => t.Result.ToString(), TaskScheduler.Default);
    WeakReferenceMessenger.Default.Send(new TabDragTaskMessage(draggedTab, logicalTask, this));
  }

  public void ClearDragState()
  {
    _draggedTabReference = null;
    PreviewBox.IsVisible = false;
  }

  private static DockZone CalculateDockZone(Point pos, Size bounds)
  {
    const double edgeThreshold = 0.3;
    var isLeft = pos.X < bounds.Width * edgeThreshold;
    var isRight = pos.X > bounds.Width * (1 - edgeThreshold);
    var isTop = pos.Y < bounds.Height * edgeThreshold;
    var isBottom = pos.Y > bounds.Height * (1 - edgeThreshold);

    if (!isLeft && !isRight && !isTop && !isBottom) return DockZone.Center;

    var distLeft = pos.X;
    var distRight = bounds.Width - pos.X;
    var distTop = pos.Y;
    var distBottom = bounds.Height - pos.Y;
    var minDist = Math.Min(Math.Min(distLeft, distRight), Math.Min(distTop, distBottom));

    if (Math.Abs(minDist - distLeft) < 1e-6) return DockZone.Left;
    if (Math.Abs(minDist - distRight) < 1e-6) return DockZone.Right;
    if (Math.Abs(minDist - distTop) < 1e-6) return DockZone.Top;
    return Math.Abs(minDist - distBottom) < 1e-6 ? DockZone.Bottom : DockZone.Center;
  }

  private void ShowPreview(DockZone zone)
  {
    if (zone == DockZone.Center)
    {
      PreviewBox.IsVisible = false;
      return;
    }

    PreviewBox.IsVisible = true;
    double splitRatio = 0.5;
    double w = Bounds.Width;
    double h = Bounds.Height;

    switch (zone)
    {
      case DockZone.Left:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Left;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Stretch;
        PreviewBox.Width = w * splitRatio;
        PreviewBox.Height = h;
        break;
      case DockZone.Right:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Right;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Stretch;
        PreviewBox.Width = w * splitRatio;
        PreviewBox.Height = h;
        break;
      case DockZone.Top:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Top;
        PreviewBox.Width = w;
        PreviewBox.Height = h * splitRatio;
        break;
      case DockZone.Bottom:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Bottom;
        PreviewBox.Width = w;
        PreviewBox.Height = h * splitRatio;
        break;
    }
  }
}
