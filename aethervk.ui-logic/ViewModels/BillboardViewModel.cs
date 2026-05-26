using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Represents an individual Screen Space Billboard rendered dynamically atop the 3D viewport.
/// Supports absolute canvas positioning, arbitrary dimensioning, and Z-Index stacking.
/// </summary>
public partial class BillboardViewModel : ObservableObject
{
    [ObservableProperty]
    private object? _imageSource;

    [ObservableProperty]
    private string? _text;

    [ObservableProperty]
    private bool _isSelected;

    [ObservableProperty]
    private double _x;

    [ObservableProperty]
    private double _y;

    [ObservableProperty]
    private double _width = 100;

    [ObservableProperty]
    private double _height = 100;

    [ObservableProperty]
    private int _zIndex = 1;

    [ObservableProperty]
    private double _opacity = 1.0;
}
