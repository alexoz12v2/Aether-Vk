using System;
using Avalonia;
using Avalonia.Animation;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Controls.Presenters;
using Avalonia.Controls.Templates;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Media.Transformation;
using Avalonia.Styling;
using Avalonia.Threading;

namespace AetherVk.UI.Controls;

/// <summary>
/// A HUD panel with a toggle icon at its top-right. Clicking the icon collapses the panel (fade +
/// vertical scale, 0.18 s) down to just the icon; clicking it again re-expands it. Collapsed is not
/// hidden: the host keeps deciding visibility. Once collapsed the body leaves the layout and the hit
/// test, so only the icon-sized area takes pointer input.
/// </summary>
public sealed class CollapsibleHudPanel : ContentControl
{
  private static readonly TimeSpan Duration = TimeSpan.FromMilliseconds(180);

  public static readonly StyledProperty<bool> IsExpandedProperty =
    AvaloniaProperty.Register<CollapsibleHudPanel, bool>(nameof(IsExpanded), true, defaultBindingMode: Avalonia.Data.BindingMode.TwoWay);

  /// <summary>AutomationId given to the toggle icon.</summary>
  public static readonly StyledProperty<string> ToggleAutomationIdProperty =
    AvaloniaProperty.Register<CollapsibleHudPanel, string>(nameof(ToggleAutomationId), string.Empty);

  public bool IsExpanded
  {
    get => GetValue(IsExpandedProperty);
    set => SetValue(IsExpandedProperty, value);
  }

  public string ToggleAutomationId
  {
    get => GetValue(ToggleAutomationIdProperty);
    set => SetValue(ToggleAutomationIdProperty, value);
  }

  private Border? _body;
  private Button? _toggle;
  private readonly DispatcherTimer _hideTimer;

  public CollapsibleHudPanel()
  {
    _hideTimer = new DispatcherTimer { Interval = Duration };
    _hideTimer.Tick += (_, _) =>
    {
      _hideTimer.Stop();
      if (!IsExpanded && _body is not null)
        _body.IsVisible = false;
    };
    Template = new FuncControlTemplate<CollapsibleHudPanel>(Build);
  }

  private static IBrush Res(string key, IBrush fallback) =>
    Application.Current is { } app && app.TryGetResource(key, app.ActualThemeVariant, out var v) && v is IBrush b ? b : fallback;

  private Control Build(CollapsibleHudPanel owner, INameScope scope)
  {
    var presenter = new ContentPresenter { [!ContentPresenter.ContentProperty] = owner[!ContentProperty] };
    _body = new Border
    {
      Child = presenter,
      CornerRadius = new CornerRadius(8),
      BorderThickness = new Thickness(1),
      Padding = new Thickness(10, 8, 32, 8),
      Opacity = 0.9,
      Background = Res("Bg.Surface", Brushes.Black),
      BorderBrush = Res("Bg.Outline", Brushes.Gray),
      RenderTransformOrigin = new RelativePoint(1, 0, RelativeUnit.Relative),
      Transitions =
      [
        new DoubleTransition { Property = OpacityProperty, Duration = Duration },
        new TransformOperationsTransition { Property = RenderTransformProperty, Duration = Duration },
      ],
    };
    _body.RenderTransform = TransformOperations.Parse("scaleY(1)");
    _toggle = new Button
    {
      HorizontalAlignment = HorizontalAlignment.Right,
      VerticalAlignment = VerticalAlignment.Top,
      Margin = new Thickness(0, 4, 4, 0),
      Padding = new Thickness(6, 0),
      FontSize = 11,
    };
    AutomationProperties.SetAutomationId(_toggle, ToggleAutomationId);
    _toggle.Click += (_, _) => IsExpanded = !IsExpanded;
    Apply(animate: false);
    return new Grid { Children = { _body, _toggle } };
  }

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);
    if (change.Property == IsExpandedProperty)
      Apply(animate: true);
    else if (change.Property == ToggleAutomationIdProperty && _toggle is not null)
      AutomationProperties.SetAutomationId(_toggle, ToggleAutomationId);
  }

  private void Apply(bool animate)
  {
    if (_body is null || _toggle is null)
      return;
    bool open = IsExpanded;
    _toggle.Content = open ? "✕" : "☰";
    ToolTip.SetTip(_toggle, open ? "Hide panel" : "Show panel");
    _hideTimer.Stop();
    if (open)
    {
      _body.IsVisible = true;
      _body.IsHitTestVisible = true;
    }
    else
    {
      _body.IsHitTestVisible = false;
    }
    SetBodyState(open);
    if (!open)
    {
      if (animate)
        _hideTimer.Start();
      else
        _body.IsVisible = false;
    }
  }

  private void SetBodyState(bool open)
  {
    _body!.Opacity = open ? 0.9 : 0;
    _body.RenderTransform = TransformOperations.Parse(open ? "scaleY(1)" : "scaleY(0)");
  }
}
