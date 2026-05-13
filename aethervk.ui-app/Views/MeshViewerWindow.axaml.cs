using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;

namespace AetherVk.Views;

public partial class MeshViewerWindow : Window, IViewportRenderer
{
  private MeshViewerViewModel? _viewModel;
  private WriteableBitmap? _bitmap;
  private DateTime _lastFrameTime = DateTime.Now;
  private bool _isLeftDragging = false;
  private bool _isRightDragging = false;
  private bool _isMiddleDragging = false;
  private Avalonia.Point _lastPointerPos;

  private DispatcherTimer? _resizeTimer;
  private Avalonia.Size _pendingSize;

  public MeshViewerWindow()
  {
    InitializeComponent();
    Closing += (s, e) => _viewModel?.Stop();

    _resizeTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(150) };
    _resizeTimer.Tick += OnResizeTimerTick;

    SizeChanged += OnSizeChanged;
  }

  private void OnSizeChanged(object? sender, SizeChangedEventArgs e)
  {
    _pendingSize = e.NewSize;
    _resizeTimer?.Stop();
    _resizeTimer?.Start();
  }

  private void OnResizeTimerTick(object? sender, EventArgs e)
  {
    _resizeTimer?.Stop();

    if (_viewModel == null || _pendingSize.Width <= 0 || _pendingSize.Height <= 0)
      return;

    uint newWidth = (uint)_pendingSize.Width;
    uint newHeight = (uint)_pendingSize.Height;

    if (newWidth == _viewModel.Width && newHeight == _viewModel.Height)
      return;

    _viewModel.Width = newWidth;
    _viewModel.Height = newHeight;

    _bitmap = new WriteableBitmap(
      new Avalonia.PixelSize((int)newWidth, (int)newHeight),
      new Avalonia.Vector(96, 96),
      PixelFormat.Bgra8888,
      AlphaFormat.Opaque
    );

    RenderTargetImage.Source = _bitmap;

    _viewModel.RuntimeService.ResizePresentationEngine(
      _viewModel.SceneId,
      _viewModel.PresentationEngineId,
      newWidth,
      newHeight
    );
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    if (_viewModel != null)
    {
      _viewModel.Renderer = null;
    }

    _viewModel = DataContext as MeshViewerViewModel;

    if (_viewModel != null)
    {
      _viewModel.Renderer = this;
      _bitmap = new WriteableBitmap(
        new Avalonia.PixelSize((int)_viewModel.Width, (int)_viewModel.Height),
        new Avalonia.Vector(96, 96),
        PixelFormat.Bgra8888,
        AlphaFormat.Opaque
      );

      RenderTargetImage.Source = _bitmap;
    }
  }

  protected override void OnOpened(EventArgs e)
  {
    base.OnOpened(e);
    if (_viewModel != null)
    {
      _viewModel.Renderer = this;
    }
  }

  protected override void OnClosed(EventArgs e)
  {
    base.OnClosed(e);
    if (_viewModel != null)
    {
      _viewModel.Renderer = null;
      _viewModel.Dispose();
    }
  }

  public void UpdateFrame(IntPtr buffer, nuint bufferSize)
  {
    if (_bitmap == null || _viewModel == null)
      return;

    _lastFrameTime = DateTime.Now;

    using (var frame = _bitmap.Lock())
    {
      unsafe
      {
        System.Buffer.MemoryCopy(
          buffer.ToPointer(),
          frame.Address.ToPointer(),
          bufferSize,
          bufferSize
        );
      }
    }
    RenderTargetImage.InvalidateVisual();
  }

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    var point = e.GetCurrentPoint(RenderTargetImage);
    _lastPointerPos = point.Position;

    RenderTargetImage.Focus();
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e) { }

  private void OnPointerMoved(object? sender, PointerEventArgs e)
  {
    var point = e.GetCurrentPoint(RenderTargetImage);
    var currentPos = point.Position;
    var deltaX = (float)(currentPos.X - _lastPointerPos.X);
    var deltaY = (float)(currentPos.Y - _lastPointerPos.Y);

    _viewModel?.OperatorStack.ProcessPointerDelta(deltaX, deltaY);

    _lastPointerPos = currentPos;
  }

  private void OnPointerWheelChanged(object? sender, PointerWheelEventArgs e)
  {
    var scroll_amount = (float)e.Delta.Y;
    _viewModel?.OperatorStack.ProcessPointerWheel(scroll_amount);
  }
}