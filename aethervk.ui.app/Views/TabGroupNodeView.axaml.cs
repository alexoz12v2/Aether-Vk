using System;
using System.Linq;
using System.Threading.Tasks;
using AetherVk.Logic.Messages;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Threading;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Views;

public partial class TabGroupNodeView : UserControl
{
  // TODO: move this logic in view model as much as possible
  // OS-Level drag flag, we can't format typed bytes due to security risks ...
  public static readonly DataFormat<string> ChildViewModelFormat =
    DataFormat.CreateStringApplicationFormat("AetherVk.TabItemViewModel");

  // ... therefore we need to hold a complex object in memory while drag active
  // Note: Why Static? In a Desktop environment, there's only 1 cursor.
  private static TabItemViewModel? _draggedTabReference;

  public TabGroupNodeView()
  {
    InitializeComponent();
    // Drag is triggered by PointerPressed and DoDragDrop(), passing TabItemViewModel as payload
    AddHandler(DragDrop.DragOverEvent, OnDragOver);
    AddHandler(DragDrop.DragLeaveEvent, OnDragLeave);
    AddHandler(DragDrop.DropEvent, OnDrop);
  }

  #region Drag Handlers
  private void OnDragOver(object? sender, DragEventArgs e)
  {
    if (!e.DataTransfer.Contains(ChildViewModelFormat) || _draggedTabReference is null)
    {
      e.DragEffects = DragDropEffects.None;
      return;
    }

    // 1. Calculate mouse position relative to this UserControl
    var pos = e.GetPosition(this);
    var zone = CalculateDockZone(pos, Bounds.Size);

    // 2. Update PreviewBox Margin/Alignment based on the zone
    ShowPreview(zone);
    e.DragEffects = DragDropEffects.Move;
  }

  private void OnDragLeave(object? sender, DragEventArgs e)
  {
    PreviewBox.IsVisible = false;
  }

  private void OnDrop(object? sender, DragEventArgs e)
  {
    PreviewBox.IsVisible = false;
    if (
      e.DataTransfer.Contains(ChildViewModelFormat)
      && _draggedTabReference is { } draggedTab
      && DataContext is TabGroupNodeViewModel targetNode
    )
    {
      var pos = e.GetPosition(this);
      var zone = CalculateDockZone(pos, Bounds.Size);

      // TODO: is movement enough?

      // Send message to Root Manager to handle the tree mutation
      // Dispatcher.UIThread.Post(() =>
      // {
      //   WeakReferenceMessenger.Default.Send(new TabDroppedMessage(draggedTab, targetNode, zone));
      // });
      WeakReferenceMessenger.Default.Send(new TabDroppedMessage(draggedTab, targetNode, zone));
    }
  }
  #endregion

  /// <summary>
  /// Initializes the drag
  /// </summary>
  private async void OnTabPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    // ensure we drag only on left click
    if (!e.GetCurrentPoint(this).Properties.IsLeftButtonPressed)
      return;

    // check if this is the last tab in the manager
    if (DataContext is TabGroupNodeViewModel vm && vm.IsRoot() && vm.Tabs.Count <= 1)
      return; // optional: shake or show different cursor?

    if (sender is Control control && control.DataContext is TabItemViewModel draggedTab)
    {
      var data = new DataTransfer();
      var item = new DataTransferItem();
      item.Set(ChildViewModelFormat, "DragActive");
      data.Add(item);

      _draggedTabReference = draggedTab;

      // initiate the drag-and-drop operation
      await DragDrop.DoDragDropAsync(e, data, DragDropEffects.Move);

      // Note: In Avalonia, execution pauses here until the drop completes
      _draggedTabReference = null;
    }
  }

  private static DockZone CalculateDockZone(Point pos, Size bounds)
  {
    // 30% from the edge triggers a split. The middle 40% triggers a coalesce (center).
    double edgeThreshold = 0.3;

    bool isLeft = pos.X < bounds.Width * edgeThreshold;
    bool isRight = pos.X > bounds.Width * (1 - edgeThreshold);
    bool isTop = pos.Y < bounds.Height * edgeThreshold;
    bool isBottom = pos.Y > bounds.Height * (1 - edgeThreshold);

    // If we are in the edges, resolve corners by finding the closest edge
    if (isLeft || isRight || isTop || isBottom)
    {
      double distLeft = pos.X;
      double distRight = bounds.Width - pos.X;
      double distTop = pos.Y;
      double distBottom = bounds.Height - pos.Y;

      double minDist = Math.Min(Math.Min(distLeft, distRight), Math.Min(distTop, distBottom));

      if (minDist == distLeft)
        return DockZone.Left;
      if (minDist == distRight)
        return DockZone.Right;
      if (minDist == distTop)
        return DockZone.Top;
      if (minDist == distBottom)
        return DockZone.Bottom;
    }
    return DockZone.Center;
  }

  private void ShowPreview(DockZone zone)
  {
    PreviewBox.IsVisible = true;

    // This represents the visual preview of the 50% split ratio
    double splitRatio = 0.5;
    switch (zone)
    {
      case DockZone.Left:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Left;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Stretch;
        PreviewBox.Width = Bounds.Width * splitRatio;
        PreviewBox.Height = double.NaN; // resets explicitly to Auto
        break;

      case DockZone.Right:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Right;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Stretch;
        PreviewBox.Width = Bounds.Width * splitRatio;
        PreviewBox.Height = double.NaN;
        break;

      case DockZone.Top:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Top;
        PreviewBox.Width = double.NaN;
        PreviewBox.Height = Bounds.Height * splitRatio;
        break;

      case DockZone.Bottom:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Bottom;
        PreviewBox.Width = double.NaN;
        PreviewBox.Height = Bounds.Height * splitRatio;
        break;

      case DockZone.Center:
        PreviewBox.HorizontalAlignment = Avalonia.Layout.HorizontalAlignment.Stretch;
        PreviewBox.VerticalAlignment = Avalonia.Layout.VerticalAlignment.Stretch;
        PreviewBox.Width = double.NaN;
        PreviewBox.Height = double.NaN;
        break;
    }
  }
}
