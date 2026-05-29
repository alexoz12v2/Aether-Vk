using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// Represents an individual Screen Space Billboard rendered dynamically atop the 3D viewport.
/// Supports absolute canvas positioning, arbitrary dimensioning, uniform scaling,
/// rotation, opacity, and Z-Index stacking. Linked to a Rust ECS entity via EntityId.
/// </summary>
public partial class BillboardViewModel : ObservableObject
{
  /// <summary>The Rust ECS entity ID backing this billboard. 0 = UI-only (e.g., measurement labels).</summary>
  [ObservableProperty]
  private ulong _entityId;

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

  /// <summary>Uniform scale factor (1.0 = original size).</summary>
  [ObservableProperty]
  private double _scale = 1.0;

  /// <summary>Rotation in degrees around the center.</summary>
  [ObservableProperty]
  private double _rotation;
}
