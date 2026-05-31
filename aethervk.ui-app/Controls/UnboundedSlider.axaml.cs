using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;

namespace AetherVk.Controls;

public partial class UnboundedSlider : UserControl
{
  public static readonly StyledProperty<double> ValueProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    double
  >(nameof(Value), defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

  public double Value
  {
    get => GetValue(ValueProperty);
    set => SetValue(ValueProperty, value);
  }

  public static readonly StyledProperty<double> StepProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    double
  >(nameof(Step), 1.0);

  public double Step
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
      var delta = pos.X - _lastPos.X;
      if (System.Math.Abs(delta) > 1.0)
        _hasMoved = true;

      if (_hasMoved)
      {
        var mult = e.KeyModifiers.HasFlag(KeyModifiers.Shift) ? 10.0 : 1.0;
        Value += delta * Step * mult * 0.1;
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
    if (double.TryParse(InputBox.Text, out double parsed))
    {
      Value = parsed;
    }
  }

  private void OnInputKeyDown(object? sender, KeyEventArgs e)
  {
    if (e.Key == Key.Enter)
    {
      if (double.TryParse(InputBox.Text, out double parsed))
      {
        Value = parsed;
      }
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
    else if (e.Key == Key.Escape)
    {
      InputBox.Text = Value.ToString("0.###");
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
  }
}
