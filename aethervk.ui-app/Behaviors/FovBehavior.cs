using Avalonia;
using Avalonia.Controls;
using Avalonia.Interactivity;
using Avalonia.Xaml.Interactivity;
using AetherVk.Controls;
using System;

namespace AetherVk.Behaviors;

/// <summary>
/// Behavior for an UnboundedSlider representing Field of View.
/// Handles switching between Degrees, Arc Minutes, and Arc Seconds,
/// updates formatting/stepping automatically, and prevents precision collapse.
/// </summary>
public class FovBehavior : Behavior<UnboundedSlider>, IHandlesCommit
{
    private ComboBox? _unitSelector;
    private bool _isUpdating;

    public enum FovUnit
    {
        Deg,
        ArcMin,
        ArcSec
    }

    public static readonly StyledProperty<FovUnit> CurrentUnitProperty =
        AvaloniaProperty.Register<FovBehavior, FovUnit>(
            nameof(CurrentUnit),
            defaultValue: FovUnit.Deg);

    public FovUnit CurrentUnit
    {
        get => GetValue(CurrentUnitProperty);
        set => SetValue(CurrentUnitProperty, value);
    }

    protected override void OnAttached()
    {
        base.OnAttached();
        if (AssociatedObject is null) return;

        _unitSelector = new ComboBox
        {
            ItemsSource = new[] { "deg", "arcmin", "arcsec" },
            SelectedIndex = (int)CurrentUnit,
            Margin = new Thickness(4, 0, 0, 0),
            VerticalAlignment = Avalonia.Layout.VerticalAlignment.Center
        };
        _unitSelector.SelectionChanged += OnUnitSelectionChanged;

        AssociatedObject.InnerRightContent = _unitSelector;

        AssociatedObject.LostFocus += OnCommit;
        AssociatedObject.KeyDown += OnKeyDown;
        AssociatedObject.PropertyChanged += OnSliderPropertyChanged;

        UpdateStepAndBounds();
        UpdateTextFromValue();
    }

    protected override void OnDetaching()
    {
        base.OnDetaching();
        if (AssociatedObject is null) return;

        if (_unitSelector != null)
            _unitSelector.SelectionChanged -= OnUnitSelectionChanged;

        AssociatedObject.LostFocus -= OnCommit;
        AssociatedObject.KeyDown -= OnKeyDown;
        AssociatedObject.PropertyChanged -= OnSliderPropertyChanged;
    }

    private void OnUnitSelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (_unitSelector == null) return;
        
        var newUnit = (FovUnit)_unitSelector.SelectedIndex;
        if (newUnit != CurrentUnit)
        {
            CurrentUnit = newUnit;
            UpdateStepAndBounds();
            UpdateTextFromValue();
        }
    }

    private void OnSliderPropertyChanged(object? sender, AvaloniaPropertyChangedEventArgs e)
    {
        if (e.Property == UnboundedSlider.ValueProperty && !_isUpdating)
        {
            CheckAutoSwitchUnit();
            UpdateTextFromValue();
        }
    }

    private void CheckAutoSwitchUnit()
    {
        if (AssociatedObject == null) return;
        double deg = AssociatedObject.Value;

        FovUnit newUnit = CurrentUnit;

        if (deg >= 2.0) 
        {
            newUnit = FovUnit.Deg;
        }
        else if (deg >= (2.0 / 60.0))
        {
            // Between 2 arcmin and 2 deg. 
            // If we are currently in ArcSec, bump up to ArcMin. 
            if (CurrentUnit == FovUnit.ArcSec) 
                newUnit = FovUnit.ArcMin;
            else if (CurrentUnit == FovUnit.Deg && deg < 1.0)
                newUnit = FovUnit.ArcMin;
        }
        else
        {
            // Below 2 arcmin
            if (CurrentUnit == FovUnit.Deg)
                newUnit = FovUnit.ArcMin;
            if (deg < (1.0 / 60.0))
                newUnit = FovUnit.ArcSec;
        }

        if (newUnit != CurrentUnit)
        {
            CurrentUnit = newUnit;
            if (_unitSelector != null)
                _unitSelector.SelectedIndex = (int)newUnit;
            UpdateStepAndBounds();
        }
    }

    private void UpdateStepAndBounds()
    {
        if (AssociatedObject == null) return;
        
        // Always clamp to [1 arcsec, 179 deg] to prevent f32 cancellation downstream.
        AssociatedObject.HasBounds = true;
        AssociatedObject.MinBound = 1.0 / 3600.0;
        AssociatedObject.MaxBound = 179.0;
        
        switch (CurrentUnit)
        {
            case FovUnit.Deg:
                AssociatedObject.Step = 1.0;
                AssociatedObject.IsLogarithmic = false;
                break;
            case FovUnit.ArcMin:
                AssociatedObject.Step = 1.0 / 60.0;
                AssociatedObject.IsLogarithmic = false;
                break;
            case FovUnit.ArcSec:
                // For arc seconds, step is small and it's nice to have a larger drag feel
                AssociatedObject.Step = 10.0 / 3600.0;
                AssociatedObject.IsLogarithmic = true;
                break;
        }
    }

    private void UpdateTextFromValue()
    {
        if (AssociatedObject == null || AssociatedObject.IsFocused) return;
        
        _isUpdating = true;
        double displayVal = GetDisplayValue(AssociatedObject.Value);
        // We use reflection/dynamic here because InputText is marked internal in UnboundedSlider
        SetInputText(AssociatedObject, displayVal.ToString("0.###"));
        _isUpdating = false;
    }

    private void SetInputText(UnboundedSlider slider, string text)
    {
        var prop = typeof(UnboundedSlider).GetProperty("InputText", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (prop != null)
        {
            prop.SetValue(slider, text);
        }
    }

    private string? GetInputText(UnboundedSlider slider)
    {
        var prop = typeof(UnboundedSlider).GetProperty("InputText", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        if (prop != null)
        {
            return prop.GetValue(slider) as string;
        }
        return null;
    }

    private double GetDisplayValue(double deg)
    {
        return CurrentUnit switch
        {
            FovUnit.Deg => deg,
            FovUnit.ArcMin => deg * 60.0,
            FovUnit.ArcSec => deg * 3600.0,
            _ => deg
        };
    }

    private double GetDegFromDisplay(double display)
    {
        return CurrentUnit switch
        {
            FovUnit.Deg => display,
            FovUnit.ArcMin => display / 60.0,
            FovUnit.ArcSec => display / 3600.0,
            _ => display
        };
    }

    private void OnKeyDown(object? sender, Avalonia.Input.KeyEventArgs e)
    {
        if (e.Key == Avalonia.Input.Key.Enter)
            CommitText();
    }

    private void OnCommit(object? sender, RoutedEventArgs e)
    {
        CommitText();
    }

    private void CommitText()
    {
        if (AssociatedObject == null) return;
        
        var inputText = GetInputText(AssociatedObject);
        if (double.TryParse(inputText, out double parsed))
        {
            double deg = GetDegFromDisplay(parsed);
            deg = Math.Clamp(deg, 1.0 / 3600.0, 179.0);
            
            _isUpdating = true;
            AssociatedObject.Value = deg;
            _isUpdating = false;

            CheckAutoSwitchUnit();
            UpdateTextFromValue();
        }
        else
        {
            UpdateTextFromValue(); // revert to previous valid text
        }
    }
}
