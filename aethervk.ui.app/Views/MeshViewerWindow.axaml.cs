using System;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;

namespace AetherVk.Views;

public partial class MeshViewerWindow : Window
{
  private MeshViewerViewModel? _viewModel;
  private WriteableBitmap? _bitmap;
  private DateTime _lastFrameTime = DateTime.Now;
  private bool _isLeftDragging = false;
  private bool _isRightDragging = false;
  private bool _isMiddleDragging = false;
  private Avalonia.Point _lastPointerPos;

  public MeshViewerWindow()
  {
    InitializeComponent();
    Closing += (s, e) => _viewModel?.Stop();
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    if (_viewModel != null)
    {
      _viewModel.OnFrameReady -= HandleFrameReady;
    }

    _viewModel = DataContext as MeshViewerViewModel;

    if (_viewModel != null)
    {
      _bitmap = new WriteableBitmap(
        new Avalonia.PixelSize((int)_viewModel.Width, (int)_viewModel.Height),
        new Avalonia.Vector(96, 96),
        PixelFormat.Bgra8888,
        AlphaFormat.Opaque
      );

      RenderTargetImage.Source = _bitmap;
      _viewModel.OnFrameReady += HandleFrameReady;
    }
  }

  private void HandleFrameReady()
  {
    if (_bitmap == null || _viewModel == null)
      return;

    Dispatcher.UIThread.Post(async () =>
    {
      if (_bitmap == null || _viewModel == null)
        return;

      _lastFrameTime = DateTime.Now;

      using (var frame = _bitmap.Lock())
      {
        await _viewModel.CopyFrameToBuffer(
          frame.Address,
          (nuint)(_viewModel.Width * _viewModel.Height * 4)
        );
      }
      RenderTargetImage.InvalidateVisual();
    });
  }

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    var point = e.GetCurrentPoint(RenderTargetImage);
    _lastPointerPos = point.Position;

    if (point.Properties.IsLeftButtonPressed)
      _isLeftDragging = true;
    if (point.Properties.IsRightButtonPressed)
      _isRightDragging = true;
    if (point.Properties.IsMiddleButtonPressed)
      _isMiddleDragging = true;

    RenderTargetImage.Focus();
    e.Handled = true;
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (e.InitialPressMouseButton == MouseButton.Left)
      _isLeftDragging = false;
    if (e.InitialPressMouseButton == MouseButton.Right)
      _isRightDragging = false;
    if (e.InitialPressMouseButton == MouseButton.Middle)
      _isMiddleDragging = false;

    e.Handled = true;
  }

  private void OnPointerMoved(object? sender, PointerEventArgs e)
  {
    var point = e.GetCurrentPoint(RenderTargetImage);
    var currentPos = point.Position;
    var deltaX = (float)(currentPos.X - _lastPointerPos.X);
    var deltaY = (float)(currentPos.Y - _lastPointerPos.Y);

    if (_isRightDragging)
    {
      _viewModel?.RuntimeService.RotateCamera(
        _viewModel.SceneId,
        _viewModel.CameraId,
        deltaX,
        deltaY
      );
    }
    else if (_isMiddleDragging || _isLeftDragging)
    {
      _viewModel?.RuntimeService.PanCursor(_viewModel.SceneId, deltaX, deltaY);
    }

    _lastPointerPos = currentPos;
  }

  private void OnPointerWheelChanged(object? sender, PointerWheelEventArgs e)
  {
    var scroll_amount = (float)e.Delta.Y;
    _viewModel?.RuntimeService.ZoomCamera(_viewModel.SceneId, _viewModel.CameraId, scroll_amount);
  }
}
