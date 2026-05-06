using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;

namespace AetherVk.Controls;

public partial class UnboundedSlider : UserControl
{
  public static readonly StyledProperty<float> ValueProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    float
  >(nameof(Value), defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

  public float Value
  {
    get => GetValue(ValueProperty);
    set => SetValue(ValueProperty, value);
  }

  public static readonly StyledProperty<float> StepProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    float
  >(nameof(Step), 1.0f);

  public float Step
  {
    get => GetValue(StepProperty);
    set => SetValue(StepProperty, value);
  }

  private Point _lastPos;
  private bool _isDragging;
  private bool _hasMoved;

  public UnboundedSlider()
  {
    InitializeComponent();
  }

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    if (e.GetCurrentPoint(this).Properties.IsLeftButtonPressed && !InputBox.IsFocused)
    {
      _isDragging = true;
      _hasMoved = false;
      _lastPos = e.GetPosition(this);
      e.Pointer.Capture(sender as IInputElement);
      e.Handled = true;
    }
  }

  private void OnPointerMoved(object? sender, PointerEventArgs e)
  {
    if (_isDragging)
    {
      var pos = e.GetPosition(this);
      var delta = (float)(pos.X - _lastPos.X);
      if (System.Math.Abs(delta) > 1.0f)
        _hasMoved = true;

      if (_hasMoved)
      {
        var mult = e.KeyModifiers.HasFlag(KeyModifiers.Shift) ? 10.0f : 1.0f;
        Value += delta * Step * mult * 0.1f;
        _lastPos = pos;
      }
      e.Handled = true;
    }
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (_isDragging)
    {
      _isDragging = false;
      e.Pointer.Capture(null);
      e.Handled = true;
    }
  }

  private void OnTapped(object? sender, TappedEventArgs e)
  {
    if (!_hasMoved && !_isDragging)
    {
      InputBox.IsHitTestVisible = true;
      InputBox.Focus();
    }
  }

  private void OnInputLostFocus(object? sender, RoutedEventArgs e)
  {
    InputBox.IsHitTestVisible = false;
  }

  private void OnInputKeyDown(object? sender, KeyEventArgs e)
  {
    if (e.Key == Key.Enter || e.Key == Key.Escape)
    {
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
  }
}
