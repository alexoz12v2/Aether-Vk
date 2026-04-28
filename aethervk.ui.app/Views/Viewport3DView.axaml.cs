using System;
using AetherVk.Logic.ViewModels;
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
  private Avalonia.Point _lastPointerPos;

  public Viewport3DView()
  {
    InitializeComponent();

    _livelinessTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(100) };
    _livelinessTimer.Tick += OnLivelinessTick;
    _livelinessTimer.Start();
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
      _viewModel.OnFrameReady += HandleFrameReady;
      _viewModel.PropertyChanged += (s, args) =>
      {
        if (args.PropertyName == nameof(Viewport3DViewModel.IsAddingJet))
        {
          Cursor = _viewModel.IsAddingJet
            ? new Cursor(StandardCursorType.Hand)
            : new Cursor(StandardCursorType.Arrow);
        }
      };
    }
  }

  private async void HandleFrameReady()
  {
    if (_bitmap == null || _viewModel == null)
      return;

    _lastFrameTime = DateTime.Now;

    // Lock the bitmap memory to allow native C++ code to write pixels
    using (var frame = _bitmap.Lock())
    {
      await _viewModel.CopyFrameToBuffer(
        frame.Address,
        (nuint)(_viewModel.Width * _viewModel.Height * 4)
      );
    }
    // Notify the Avalonia rendering system that the bitmap has been updated
    Dispatcher.UIThread.Post(() => RenderTargetImage.InvalidateVisual());
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

    if (point.Properties.IsMiddleButtonPressed)
      _isMiddleDragging = true;
    if (point.Properties.IsRightButtonPressed)
      _isRightDragging = true;

    RenderTargetImage.Focus();
    e.Handled = true;
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (e.InitialPressMouseButton == MouseButton.Middle)
      _isMiddleDragging = false;
    if (e.InitialPressMouseButton == MouseButton.Right)
      _isRightDragging = false;

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
      _viewModel?.RuntimeService.RotateCamera(deltaX, deltaY);
    }
    else if (_isMiddleDragging)
    {
      _viewModel?.RuntimeService.PanCamera(deltaX, deltaY);
    }

    _lastPointerPos = currentPos;
  }

  private void OnPointerWheelChanged(object? sender, PointerWheelEventArgs e)
  {
    _viewModel?.RuntimeService.ZoomCamera((float)e.Delta.Y * 10f);
  }

  private void OnKeyDown(object? sender, KeyEventArgs e)
  {
    if (_viewModel?.IsAddingJet == true && e.Key == Key.Escape)
    {
      _viewModel.IsAddingJet = false;
      Cursor = new Cursor(StandardCursorType.Arrow);
      e.Handled = true;
      return;
    }

    if (e.Key == Key.Tab)
    {
      _viewModel?.ToggleMeasuringModeCommand.Execute(null);
      e.Handled = true;
      return;
    }

    if (e.Key == Key.R)
      _viewModel?.RuntimeService.ResetCamera();
    else if (e.Key == Key.Up)
      _viewModel?.RuntimeService.MoveCursor(0.0f, -0.5f, 0.0f);
    else if (e.Key == Key.Down)
      _viewModel?.RuntimeService.MoveCursor(0.0f, 0.5f, 0.0f);
    else if (e.Key == Key.Left)
      _viewModel?.RuntimeService.MoveCursor(-0.5f, 0.0f, 0.0f);
    else if (e.Key == Key.Right)
      _viewModel?.RuntimeService.MoveCursor(0.5f, 0.0f, 0.0f);
    else if (e.Key == Key.E)
      _viewModel?.RuntimeService.MoveCursor(0.0f, 0.0f, 0.5f);
    else if (e.Key == Key.Q)
      _viewModel?.RuntimeService.MoveCursor(0.0f, 0.0f, -0.5f);
  }
}
