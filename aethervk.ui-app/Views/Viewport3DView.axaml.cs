using System;
using AetherVk.Logic.ViewModels;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Media;
using Avalonia.Media.Imaging;
using Avalonia.Platform;
using Avalonia.Threading;

namespace AetherVk.Views;

public partial class Viewport3DView : UserControl
{
  private Viewport3DViewModel? _viewModel;
  private WriteableBitmap? _bitmap;
  private DispatcherTimer? _livelinessTimer;
  private DateTime _lastFrameTime;

  private bool _isMiddleDragging = false;
  private bool _isRightDragging = false;
  private bool _isZoomDragging = false;
  private Avalonia.Point _lastPointerPos;

  private DispatcherTimer? _resizeTimer;
  private Avalonia.Size _pendingSize;

  public Viewport3DView()
  {
    InitializeComponent();

    _livelinessTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
    _livelinessTimer.Tick += OnLivelinessTick;
    _livelinessTimer.Start();
    
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

    if (_viewModel == null || _pendingSize.Width <= 0 || _pendingSize.Height <= 0) return;

    uint newWidth = (uint)_pendingSize.Width;
    uint newHeight = (uint)_pendingSize.Height;

    if (newWidth == _viewModel.Width && newHeight == _viewModel.Height) return;

    _viewModel.Width = newWidth;
    _viewModel.Height = newHeight;

    _bitmap = new WriteableBitmap(
      new Avalonia.PixelSize((int)newWidth, (int)newHeight),
      new Avalonia.Vector(96, 96),
      PixelFormat.Bgra8888,
      AlphaFormat.Opaque
    );

    RenderTargetImage.Source = _bitmap;

    _viewModel.RuntimeService.ResizePresentationEngine(_viewModel.SceneId, _viewModel.PresentationEngineId, newWidth, newHeight);
  }

  private void OnLivelinessTick(object? sender, EventArgs e)
  {
    if (_viewModel == null || !_viewModel.IsInitialized)
      return;

    var age = (DateTime.Now - _lastFrameTime).TotalMilliseconds;
    // Interpolate from Green (0) to Red (1000ms+)
    var t = Math.Clamp(age / 1000.0, 0.0, 1.0);
    byte r = (byte)(255 * t);
    byte g = (byte)(255 * (1.0 - t));
    LivelinessIndicator.Fill = new SolidColorBrush(Color.FromRgb(r, g, 0));
  }

  protected override void OnDataContextChanged(EventArgs e)
  {
    base.OnDataContextChanged(e);

    if (_viewModel != null)
    {
      _viewModel.OnFrameReady -= HandleFrameReady;
    }

    _viewModel = DataContext as Viewport3DViewModel;

    if (_viewModel != null)
    {
      _bitmap = new WriteableBitmap(
        new Avalonia.PixelSize((int)_viewModel.Width, (int)_viewModel.Height),
        new Avalonia.Vector(96, 96),
        PixelFormat.Bgra8888,
        AlphaFormat.Opaque
      );

      RenderTargetImage.Source = _bitmap;
    }
  }

  protected override void OnAttachedToVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnAttachedToVisualTree(e);
    if (_viewModel != null)
    {
      _viewModel.OnFrameReady -= HandleFrameReady;
      _viewModel.OnFrameReady += HandleFrameReady;
    }
  }

  protected override void OnDetachedFromVisualTree(VisualTreeAttachmentEventArgs e)
  {
    base.OnDetachedFromVisualTree(e);
    if (_viewModel != null)
    {
      _viewModel.OnFrameReady -= HandleFrameReady;
    }
  }

  private async void HandleFrameReady()
  {
    if (_bitmap == null || _viewModel == null)
      return;

    _lastFrameTime = DateTime.Now;

    nuint bufferSize = (nuint)(_viewModel.Width * _viewModel.Height * 4);
    IntPtr unmanagedBuffer = System.Runtime.InteropServices.Marshal.AllocHGlobal((int)bufferSize);

    try
    {
      await _viewModel.CopyFrameToBuffer(unmanagedBuffer, bufferSize);

      await Dispatcher.UIThread.InvokeAsync(() =>
      {
        if (_bitmap != null)
        {
          using (var frame = _bitmap.Lock())
          {
            unsafe
            {
              System.Buffer.MemoryCopy(
                unmanagedBuffer.ToPointer(),
                frame.Address.ToPointer(),
                bufferSize,
                bufferSize
              );
            }
          }
          RenderTargetImage.InvalidateVisual();
        }
      });
    }
    finally
    {
      System.Runtime.InteropServices.Marshal.FreeHGlobal(unmanagedBuffer);
    }
  }

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    var point = e.GetCurrentPoint(RenderTargetImage);
    _lastPointerPos = point.Position;

    if (point.Properties.IsRightButtonPressed && e.KeyModifiers.HasFlag(KeyModifiers.Shift))
    {
      _viewModel?.PerformRaycast(
        point.Position.X,
        point.Position.Y,
        RenderTargetImage.Bounds.Width,
        RenderTargetImage.Bounds.Height
      );
      e.Handled = true;
      return;
    }

    RenderTargetImage.Focus();
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
  }

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
