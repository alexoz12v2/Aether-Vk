using System.Linq;
using AetherVk.Behaviors;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Xaml.Interactivity;

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

  public static readonly StyledProperty<bool> IsWrappedProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    bool
  >(nameof(IsWrapped), false);

  public bool IsWrapped
  {
    get => GetValue(IsWrappedProperty);
    set => SetValue(IsWrappedProperty, value);
  }

  public static readonly StyledProperty<double> WrapMinProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    double
  >(nameof(WrapMin), 0.0);

  public double WrapMin
  {
    get => GetValue(WrapMinProperty);
    set => SetValue(WrapMinProperty, value);
  }

  public static readonly StyledProperty<double> WrapMaxProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    double
  >(nameof(WrapMax), 360.0);

  public double WrapMax
  {
    get => GetValue(WrapMaxProperty);
    set => SetValue(WrapMaxProperty, value);
  }

  public static readonly StyledProperty<bool> IsLogarithmicProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    bool
  >(nameof(IsLogarithmic), false);

  public bool IsLogarithmic
  {
    get => GetValue(IsLogarithmicProperty);
    set => SetValue(IsLogarithmicProperty, value);
  }

  public static readonly StyledProperty<bool> HasBoundsProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    bool
  >(nameof(HasBounds), false);

  public bool HasBounds
  {
    get => GetValue(HasBoundsProperty);
    set => SetValue(HasBoundsProperty, value);
  }

  public static readonly StyledProperty<double> MinBoundProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    double
  >(nameof(MinBound), double.MinValue);

  public double MinBound
  {
    get => GetValue(MinBoundProperty);
    set => SetValue(MinBoundProperty, value);
  }

  public static readonly StyledProperty<double> MaxBoundProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    double
  >(nameof(MaxBound), double.MaxValue);

  public double MaxBound
  {
    get => GetValue(MaxBoundProperty);
    set => SetValue(MaxBoundProperty, value);
  }

  public static readonly StyledProperty<double> DragSensitivityProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    double
  >(nameof(DragSensitivity), 1.0);

  public double DragSensitivity
  {
    get => GetValue(DragSensitivityProperty);
    set => SetValue(DragSensitivityProperty, value);
  }

  public static readonly StyledProperty<bool> IsDraggingProperty = AvaloniaProperty.Register<
    UnboundedSlider,
    bool
  >(nameof(IsDragging), false);

  public bool IsDragging
  {
    get => GetValue(IsDraggingProperty);
    private set => SetValue(IsDraggingProperty, value);
  }

  /// <summary>
  /// Gets or sets the raw text currently displayed in the input box.
  /// Intended for behavior interop — prefer binding <see cref="Value"/> directly.
  /// </summary>
  internal string? InputText
  {
    get => InputBox.Text;
    set => InputBox.Text = value;
  }

  public static readonly StyledProperty<object?> InnerRightContentProperty =
    AvaloniaProperty.Register<UnboundedSlider, object?>(nameof(InnerRightContent), null);

  public object? InnerRightContent
  {
    get => GetValue(InnerRightContentProperty);
    set => SetValue(InnerRightContentProperty, value);
  }

  private Point _lastPos;
  private bool _hasMoved;

  public UnboundedSlider()
  {
    InitializeComponent();
  }

  private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
  {
    if (e.GetCurrentPoint(this).Properties.IsLeftButtonPressed && !InputBox.IsFocused)
    {
      IsDragging = true;
      _hasMoved = false;
      _lastPos = e.GetPosition(this);
      e.Pointer.Capture(sender as IInputElement);
      e.Handled = true;
    }
  }

  private void OnPointerMoved(object? sender, PointerEventArgs e)
  {
    if (IsDragging)
    {
      var pos = e.GetPosition(this);
      var delta = pos.X - _lastPos.X;
      if (System.Math.Abs(delta) > 1.0)
        _hasMoved = true;

      if (_hasMoved)
      {
        var mult = e.KeyModifiers.HasFlag(KeyModifiers.Shift) ? 10.0 : 1.0;
        double newValue;

        if (IsLogarithmic)
        {
          double minLog = HasBounds && MinBound > 0 ? System.Math.Log10(MinBound) : -10.0;
          double currentLog = Value > 0 ? System.Math.Log10(Value) : minLog;

          double deltaLog = delta * Step * mult * 0.005 * DragSensitivity; // Base sensitivity for log
          double newLog = currentLog + deltaLog;

          newValue = System.Math.Pow(10, newLog);
        }
        else
        {
          newValue = Value + delta * Step * mult * 0.1 * DragSensitivity;
        }

        Value = Constrain(newValue);
        _lastPos = pos;
        
        // --- Cursor Wrapping Logic ---
        var topLevel = TopLevel.GetTopLevel(this);
        if (topLevel != null)
        {
          var screen = topLevel.Screens?.ScreenFromVisual(this);
          if (screen != null)
          {
            var currentScreenPoint = this.PointToScreen(pos);
            var bounds = screen.Bounds;
            int newX;
            bool warped = AetherVk.Logic.Utils.CursorWrapHelper.TryWrapCursor(
                currentScreenPoint.X, bounds.X, bounds.Right, 2, 10, out newX);
            int newY = currentScreenPoint.Y;

            if (warped)
            {
              var newScreenPt = new PixelPoint(newX, newY);
              
              if (this.DataContext is AetherVk.Logic.ViewModels.ICursorWarpingViewModel cursorVm)
              {
                cursorVm.SetCursorPosition(newScreenPt.X, newScreenPt.Y);
              }
              
              // Update our local pos so the next event doesn't register a huge jump
              var newLocalPt = this.PointToClient(newScreenPt);
              _lastPos = newLocalPt;
            }
          }
        }
      }
      e.Handled = true;
    }
  }

  private void OnPointerReleased(object? sender, PointerReleasedEventArgs e)
  {
    if (IsDragging)
    {
      IsDragging = false;
      e.Pointer.Capture(null);
      e.Handled = true;
    }
  }

  private void OnTapped(object? sender, TappedEventArgs e)
  {
    if (!_hasMoved && !IsDragging)
    {
      InputBox.IsHitTestVisible = true;
      InputBox.Focus();
    }
  }

  protected override void OnGotFocus(GotFocusEventArgs e)
  {
    base.OnGotFocus(e);
    // When focus arrives via Tab (keyboard navigation), activate edit mode
    // so the user can type immediately without needing to click.
    if (e.NavigationMethod == NavigationMethod.Tab)
    {
      InputBox.IsHitTestVisible = true;
      InputBox.Focus();
    }
  }

  private void OnInputLostFocus(object? sender, RoutedEventArgs e)
  {
    InputBox.IsHitTestVisible = false;
    if (HasCommitBehavior())
      return; // behavior owns the commit

    if (double.TryParse(InputBox.Text, out double parsed))
    {
      Value = Constrain(parsed);
    }
  }

  private void OnInputKeyDown(object? sender, KeyEventArgs e)
  {
    if (e.Key == Key.Enter)
    {
      if (!HasCommitBehavior() && double.TryParse(InputBox.Text, out double parsed))
      {
        Value = Constrain(parsed);
      }
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
    else if (e.Key == Key.Escape)
    {
      InputBox.Text = Value.ToString("0.###");
      TopLevel.GetTopLevel(this)?.FocusManager?.ClearFocus();
    }
  }

  /// <summary>
  /// Returns true when at least one attached behavior implements <see cref="IHandlesCommit"/>,
  /// meaning the slider should not do its own parse-and-set on focus loss / Enter.
  /// </summary>
  private bool HasCommitBehavior() =>
    Interaction.GetBehaviors(this).Cast<IBehavior>().OfType<IHandlesCommit>().Any();

  /// <summary>
  /// Applies the slider's wrap and/or clamp rules to <paramref name="value"/>.
  /// Wrap is applied first; bounds clamping is applied second.
  /// Both can be active simultaneously.
  /// </summary>
  private double Constrain(double value)
  {
    if (IsWrapped)
      value = NumericConstraint.Wrap(value, WrapMin, WrapMax);
    if (HasBounds)
      value = NumericConstraint.Clamp(value, MinBound, MaxBound);
    return value;
  }
}
