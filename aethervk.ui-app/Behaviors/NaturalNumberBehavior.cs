using System;
using System.Globalization;
using System.Linq;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Xaml.Interactivity;
using AetherVk.Controls;

namespace AetherVk.Behaviors;

/// <summary>
/// Behavior that restricts input to non-negative integer digits.
/// Works with both <see cref="TextBox"/> and <see cref="UnboundedSlider"/>.
/// <para>
/// Commit rules (on focus loss or Enter):
/// <list type="bullet">
///   <item>Float input (e.g. "3.7") → truncated to integer (3).</item>
///   <item>Negative number → rolled back to previous valid value.</item>
///   <item>Non-numeric input → rolled back to previous valid value.</item>
///   <item>Empty input → resets to <see cref="Minimum"/>.</item>
/// </list>
/// </para>
/// Note: live keystroke filtering (<c>TextChanging</c>) only applies when attached to a
/// <see cref="TextBox"/>; on an <see cref="UnboundedSlider"/> the commit rules still apply.
/// </summary>
public class NaturalNumberBehavior : Behavior<Control>, IHandlesCommit
{
  // ── Optional range ─────────────────────────────────────────────────────────

  public static readonly StyledProperty<int> MinimumProperty =
    AvaloniaProperty.Register<NaturalNumberBehavior, int>(
      nameof(Minimum),
      defaultValue: 0);

  /// <summary>Inclusive lower bound applied on commit. Default: 0.</summary>
  public int Minimum
  {
    get => GetValue(MinimumProperty);
    set => SetValue(MinimumProperty, value);
  }

  public static readonly StyledProperty<int> MaximumProperty =
    AvaloniaProperty.Register<NaturalNumberBehavior, int>(
      nameof(Maximum),
      defaultValue: int.MaxValue);

  /// <summary>Inclusive upper bound applied on commit. Default: <see cref="int.MaxValue"/> (unclamped).</summary>
  public int Maximum
  {
    get => GetValue(MaximumProperty);
    set => SetValue(MaximumProperty, value);
  }

  // ── State ──────────────────────────────────────────────────────────────────

  /// <summary>Holds the last successfully committed value so invalid input can roll back.</summary>
  private int _lastCommittedValue;

  // ── Avalonia lifecycle ─────────────────────────────────────────────────────

  protected override void OnAttached()
  {
    base.OnAttached();
    if (AssociatedObject is null) return;

    // Live keystroke filtering is only possible on TextBox.
    if (AssociatedObject is TextBox tb)
      tb.TextChanging += OnTextChanging;

    AssociatedObject.LostFocus += OnCommit;
    AssociatedObject.KeyDown   += OnKeyDown;

    // Seed rollback value from whatever is currently displayed.
    if (int.TryParse(GetText(), out int initial))
      _lastCommittedValue = Math.Clamp(initial, Minimum, Maximum);
    else
      _lastCommittedValue = Minimum;
  }

  protected override void OnDetaching()
  {
    base.OnDetaching();
    if (AssociatedObject is null) return;

    if (AssociatedObject is TextBox tb)
      tb.TextChanging -= OnTextChanging;

    AssociatedObject.LostFocus -= OnCommit;
    AssociatedObject.KeyDown   -= OnKeyDown;
  }

  // ── Text access ────────────────────────────────────────────────────────────

  private string? GetText() => AssociatedObject switch
  {
    TextBox tb        => tb.Text,
    UnboundedSlider s => s.InputText,
    _                 => null,
  };

  private void SetText(string? text)
  {
    switch (AssociatedObject)
    {
      case TextBox tb:        tb.Text     = text; break;
      case UnboundedSlider s: s.InputText = text; break;
    }
  }

  // ── Handlers ───────────────────────────────────────────────────────────────

  private void OnTextChanging(object? sender, TextChangingEventArgs e)
  {
    // Only reachable when AssociatedObject is a TextBox (see OnAttached).
    // Allow digits and a decimal point (floats are truncated on commit).
    if (sender is not TextBox tb || string.IsNullOrEmpty(tb.Text))
      return;

    if (!tb.Text.All(c => char.IsDigit(c) || c == '.'))
      e.Handled = true;
  }

  private void OnKeyDown(object? sender, Avalonia.Input.KeyEventArgs e)
  {
    if (e.Key == Avalonia.Input.Key.Enter)
      OnCommit(sender, e);
  }

  private void OnCommit(object? sender, Avalonia.Interactivity.RoutedEventArgs e)
  {
    var text = GetText();

    // Empty / whitespace → reset to Minimum.
    if (string.IsNullOrWhiteSpace(text))
    {
      _lastCommittedValue = Minimum;
      SetText(Minimum.ToString());
      return;
    }

    // Try parsing as a floating-point number first so "3.7" → 3.
    if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double parsed)
        || parsed < 0) // negative numbers → rollback
    {
      SetText(_lastCommittedValue.ToString());
      return;
    }

    // Truncate float to int, then clamp to [Minimum, Maximum].
    int result = Math.Clamp((int)Math.Truncate(parsed), Minimum, Maximum);
    _lastCommittedValue = result;
    SetText(result.ToString());
  }
}
