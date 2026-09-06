using System;
using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Primitives;
using Avalonia.Input;
using Avalonia.Threading;
using Avalonia.Xaml.Interactivity;
using Avalonia.Interactivity;

namespace AetherVk.Behaviors;

/// <summary>
/// A reusable behavior that opens the control's AttachedFlyout when hovered for a specified duration,
/// and keeps it open if the pointer remains within the control's bounds or moves into the flyout.
/// </summary>
public class ShowFlyoutOnHoverBehavior : Behavior<Control>
{
    private DispatcherTimer? _showTimer;
    private DispatcherTimer? _hideTimer;
    private TopLevel? _flyoutTopLevel;
    
    private bool _isPointerInControl;
    private bool _isPointerInFlyout;
    private bool _isTrackingTopLevel;

    public static readonly StyledProperty<int> DelayMsProperty =
        AvaloniaProperty.Register<ShowFlyoutOnHoverBehavior, int>(nameof(DelayMs), 500);

    public int DelayMs
    {
        get => GetValue(DelayMsProperty);
        set => SetValue(DelayMsProperty, value);
    }

    protected override void OnAttached()
    {
        base.OnAttached();
        if (AssociatedObject != null)
        {
            AssociatedObject.PointerEntered += OnPointerEntered;
            AssociatedObject.PointerExited += OnPointerExited;
            AssociatedObject.AddHandler(InputElement.PointerPressedEvent, OnPointerPressed, RoutingStrategies.Tunnel | RoutingStrategies.Bubble);
        }
    }

    protected override void OnDetaching()
    {
        base.OnDetaching();
        if (AssociatedObject != null)
        {
            AssociatedObject.PointerEntered -= OnPointerEntered;
            AssociatedObject.PointerExited -= OnPointerExited;
            AssociatedObject.RemoveHandler(InputElement.PointerPressedEvent, OnPointerPressed);
        }
        StopShowTimer();
        StopHideTimer();
        DetachFromFlyoutContent();
        StopTrackingTopLevel();
    }

    private void OnPointerPressed(object? sender, PointerPressedEventArgs e)
    {
        var flyout = FlyoutBase.GetAttachedFlyout(AssociatedObject!);
        if (flyout != null && flyout.IsOpen)
        {
            // The user clicked the control while the flyout is open.
            // Mark it as handled so the global LightDismiss layer ignores it
            // and the flyout doesn't randomly close (which would also reset the hover logic).
            e.Handled = true;
        }
    }

    private void OnPointerEntered(object? sender, PointerEventArgs e)
    {
        _isPointerInControl = true;
        StopHideTimer();
        StopShowTimer();
        
        var flyout = FlyoutBase.GetAttachedFlyout(AssociatedObject!);
        if (flyout != null && flyout.IsOpen)
        {
            return; // Already open
        }

        _showTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(DelayMs) };
        _showTimer.Tick += OnShowTimerTick;
        _showTimer.Start();
    }

    private void OnPointerExited(object? sender, PointerEventArgs e)
    {
        StopShowTimer();
        
        var flyout = FlyoutBase.GetAttachedFlyout(AssociatedObject!);
        if (flyout != null && flyout.IsOpen)
        {
            // Avalonia emits PointerExited when a Flyout opens because the control loses pointer-over state.
            // Check if the pointer is actually physically outside our bounds.
            var point = e.GetCurrentPoint(AssociatedObject).Position;
            var bounds = new Rect(0, 0, AssociatedObject!.Bounds.Width, AssociatedObject.Bounds.Height);
            _isPointerInControl = bounds.Contains(point);
            
            UpdateHideTimer();
            return;
        }

        _isPointerInControl = false;
        UpdateHideTimer();
    }

    private void OnShowTimerTick(object? sender, EventArgs e)
    {
        StopShowTimer();
        if (AssociatedObject != null)
        {
            var flyout = FlyoutBase.GetAttachedFlyout(AssociatedObject);
            if (flyout != null)
            {
                flyout.ShowAt(AssociatedObject);
                AttachToFlyoutContent(flyout);
                StartTrackingTopLevel();
                _isPointerInControl = true;
            }
        }
    }

    private void OnHideTimerTick(object? sender, EventArgs e)
    {
        StopHideTimer();
        if (AssociatedObject != null)
        {
            var flyout = FlyoutBase.GetAttachedFlyout(AssociatedObject);
            flyout?.Hide();
            DetachFromFlyoutContent();
            StopTrackingTopLevel();
            _isPointerInControl = false;
            _isPointerInFlyout = false;
        }
    }

    private void UpdateHideTimer()
    {
        if (!_isPointerInControl && !_isPointerInFlyout)
        {
            if (_hideTimer == null)
            {
                _hideTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(200) };
                _hideTimer.Tick += OnHideTimerTick;
                _hideTimer.Start();
            }
        }
        else
        {
            StopHideTimer();
        }
    }

    private void StartTrackingTopLevel()
    {
        if (_isTrackingTopLevel) return;
        var topLevel = TopLevel.GetTopLevel(AssociatedObject);
        if (topLevel != null)
        {
            topLevel.AddHandler(InputElement.PointerMovedEvent, TopLevel_PointerMoved, RoutingStrategies.Tunnel | RoutingStrategies.Bubble);
            topLevel.AddHandler(InputElement.PointerPressedEvent, TopLevel_PointerPressed, RoutingStrategies.Tunnel);
            _isTrackingTopLevel = true;
        }
    }

    private void StopTrackingTopLevel()
    {
        if (!_isTrackingTopLevel) return;
        var topLevel = TopLevel.GetTopLevel(AssociatedObject);
        if (topLevel != null)
        {
            topLevel.RemoveHandler(InputElement.PointerMovedEvent, TopLevel_PointerMoved);
            topLevel.RemoveHandler(InputElement.PointerPressedEvent, TopLevel_PointerPressed);
            _isTrackingTopLevel = false;
        }
    }

    private void TopLevel_PointerPressed(object? sender, PointerPressedEventArgs e)
    {
        if (AssociatedObject == null) return;
        
        var point = e.GetCurrentPoint(AssociatedObject).Position;
        var bounds = new Rect(0, 0, AssociatedObject.Bounds.Width, AssociatedObject.Bounds.Height);
        
        if (bounds.Contains(point))
        {
            // Intercept the click at the absolute root of the window before it can reach
            // the LightDismissOverlayLayer. This completely prevents the flyout from closing.
            e.Handled = true;
        }
    }

    private void TopLevel_PointerMoved(object? sender, PointerEventArgs e)
    {
        if (AssociatedObject == null) return;
        
        var point = e.GetCurrentPoint(AssociatedObject).Position;
        var bounds = new Rect(0, 0, AssociatedObject.Bounds.Width, AssociatedObject.Bounds.Height);
        
        _isPointerInControl = bounds.Contains(point);
        UpdateHideTimer();
    }

    private void AttachToFlyoutContent(FlyoutBase flyout)
    {
        DetachFromFlyoutContent();
        
        Dispatcher.UIThread.Post(() =>
        {
            if (flyout is Flyout f && f.Content is Control content)
            {
                var topLevel = TopLevel.GetTopLevel(content);
                if (topLevel != null)
                {
                    _flyoutTopLevel = topLevel;
                    _flyoutTopLevel.PointerEntered += FlyoutContent_PointerEntered;
                    _flyoutTopLevel.PointerExited += FlyoutContent_PointerExited;
                }
            }
        }, DispatcherPriority.Background);
    }

    private void DetachFromFlyoutContent()
    {
        if (_flyoutTopLevel != null)
        {
            _flyoutTopLevel.PointerEntered -= FlyoutContent_PointerEntered;
            _flyoutTopLevel.PointerExited -= FlyoutContent_PointerExited;
            _flyoutTopLevel = null;
        }
    }

    private void FlyoutContent_PointerEntered(object? sender, PointerEventArgs e)
    {
        _isPointerInFlyout = true;
        UpdateHideTimer();
    }

    private void FlyoutContent_PointerExited(object? sender, PointerEventArgs e)
    {
        _isPointerInFlyout = false;
        UpdateHideTimer();
    }

    private void StopShowTimer()
    {
        if (_showTimer != null)
        {
            _showTimer.Stop();
            _showTimer.Tick -= OnShowTimerTick;
            _showTimer = null;
        }
    }

    private void StopHideTimer()
    {
        if (_hideTimer != null)
        {
            _hideTimer.Stop();
            _hideTimer.Tick -= OnHideTimerTick;
            _hideTimer = null;
        }
    }
}
