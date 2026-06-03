using Avalonia;
using Avalonia.Controls;
using Avalonia.Markup.Xaml;

namespace AetherVk.Controls;

/// <summary>
/// Reusable editor for the IAU rotational model (pole RA/Dec, prime meridian, rates)
/// with a live DualRotationGizmo preview. Computes Euler angles from the IAU pole
/// direction for the gizmo.
/// </summary>
public partial class RotationalModelEditor : UserControl
{
  // ── IAU model properties (two-way bindable) ────────────────────────────────

  public static readonly StyledProperty<double> PoleRaDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleRaDeg), defaultValue: 90.0);

  public static readonly StyledProperty<double> PoleDecDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleDecDeg), defaultValue: 90.0 - AetherVk.Logic.ViewModels.IauRotationMath.ObliquityDeg);

  public static readonly StyledProperty<double> PrimeMeridianDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PrimeMeridianDeg), defaultValue: 180.0);

  public static readonly StyledProperty<double> PoleRaRateDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleRaRateDeg));

  public static readonly StyledProperty<double> PoleDecRateDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleDecRateDeg));

  public static readonly StyledProperty<double> RotationRateDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(RotationRateDeg));

  // ── Editability (disabled during simulation) ───────────────────────────────

  public static readonly StyledProperty<bool> IsEditableProperty =
    AvaloniaProperty.Register<RotationalModelEditor, bool>(nameof(IsEditable), defaultValue: true);

  // ── Computed Quaternion for the DualRotationGizmo ────────────────────────

  public static readonly DirectProperty<RotationalModelEditor, float> QuatWProperty =
    AvaloniaProperty.RegisterDirect<RotationalModelEditor, float>(
      nameof(QuatW), o => o.QuatW);

  public static readonly DirectProperty<RotationalModelEditor, float> QuatXProperty =
    AvaloniaProperty.RegisterDirect<RotationalModelEditor, float>(
      nameof(QuatX), o => o.QuatX);

  public static readonly DirectProperty<RotationalModelEditor, float> QuatYProperty =
    AvaloniaProperty.RegisterDirect<RotationalModelEditor, float>(
      nameof(QuatY), o => o.QuatY);

  public static readonly DirectProperty<RotationalModelEditor, float> QuatZProperty =
    AvaloniaProperty.RegisterDirect<RotationalModelEditor, float>(
      nameof(QuatZ), o => o.QuatZ);

  // ── CLR accessors ─────────────────────────────────────────────────────────

  public double PoleRaDeg
  {
    get => GetValue(PoleRaDegProperty);
    set => SetValue(PoleRaDegProperty, value);
  }

  public double PoleDecDeg
  {
    get => GetValue(PoleDecDegProperty);
    set => SetValue(PoleDecDegProperty, value);
  }

  public double PrimeMeridianDeg
  {
    get => GetValue(PrimeMeridianDegProperty);
    set => SetValue(PrimeMeridianDegProperty, value);
  }

  public double PoleRaRateDeg
  {
    get => GetValue(PoleRaRateDegProperty);
    set => SetValue(PoleRaRateDegProperty, value);
  }

  public double PoleDecRateDeg
  {
    get => GetValue(PoleDecRateDegProperty);
    set => SetValue(PoleDecRateDegProperty, value);
  }

  public double RotationRateDeg
  {
    get => GetValue(RotationRateDegProperty);
    set => SetValue(RotationRateDegProperty, value);
  }

  public bool IsEditable
  {
    get => GetValue(IsEditableProperty);
    set => SetValue(IsEditableProperty, value);
  }

  private float _quatW;
  public float QuatW
  {
    get => _quatW;
    private set => SetAndRaise(QuatWProperty, ref _quatW, value);
  }

  private float _quatX;
  public float QuatX
  {
    get => _quatX;
    private set => SetAndRaise(QuatXProperty, ref _quatX, value);
  }

  private float _quatY;
  public float QuatY
  {
    get => _quatY;
    private set => SetAndRaise(QuatYProperty, ref _quatY, value);
  }

  private float _quatZ;
  public float QuatZ
  {
    get => _quatZ;
    private set => SetAndRaise(QuatZProperty, ref _quatZ, value);
  }

  public RotationalModelEditor()
  {
    AvaloniaXamlLoader.Load(this);
    RecomputeGizmoAngles();
  }

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);
    if (change.Property == PoleRaDegProperty
        || change.Property == PoleDecDegProperty
        || change.Property == PrimeMeridianDegProperty)
    {
      RecomputeGizmoAngles();
    }
  }

  /// <summary>
  /// Converts IAU pole (RA, Dec, W) → Quaternion for the DualRotationGizmo
  /// </summary>
  private void RecomputeGizmoAngles()
  {
    var (w, x, y, z) = AetherVk.Logic.ViewModels.IauRotationMath.IauToQuaternion(
      PoleRaDeg, PoleDecDeg, PrimeMeridianDeg);

    QuatW = (float)w;
    QuatX = (float)x;
    QuatY = (float)y;
    QuatZ = (float)z;
  }
}
