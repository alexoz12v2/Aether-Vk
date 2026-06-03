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
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleRaDeg), defaultValue: 270.0);

  public static readonly StyledProperty<double> PoleDecDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleDecDeg), defaultValue: 90.0);

  public static readonly StyledProperty<double> PrimeMeridianDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PrimeMeridianDeg));

  public static readonly StyledProperty<double> PoleRaRateDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleRaRateDeg));

  public static readonly StyledProperty<double> PoleDecRateDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(PoleDecRateDeg));

  public static readonly StyledProperty<double> RotationRateDegProperty =
    AvaloniaProperty.Register<RotationalModelEditor, double>(nameof(RotationRateDeg));

  // ── Editability (disabled during simulation) ───────────────────────────────

  public static readonly StyledProperty<bool> IsEditableProperty =
    AvaloniaProperty.Register<RotationalModelEditor, bool>(nameof(IsEditable), defaultValue: true);

  // ── Computed Euler angles for the DualRotationGizmo ────────────────────────

  public static readonly DirectProperty<RotationalModelEditor, float> GizmoPitchProperty =
    AvaloniaProperty.RegisterDirect<RotationalModelEditor, float>(
      nameof(GizmoPitch), o => o.GizmoPitch);

  public static readonly DirectProperty<RotationalModelEditor, float> GizmoYawProperty =
    AvaloniaProperty.RegisterDirect<RotationalModelEditor, float>(
      nameof(GizmoYaw), o => o.GizmoYaw);

  public static readonly DirectProperty<RotationalModelEditor, float> GizmoRollProperty =
    AvaloniaProperty.RegisterDirect<RotationalModelEditor, float>(
      nameof(GizmoRoll), o => o.GizmoRoll);

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

  private float _gizmoPitch;
  public float GizmoPitch
  {
    get => _gizmoPitch;
    private set => SetAndRaise(GizmoPitchProperty, ref _gizmoPitch, value);
  }

  private float _gizmoYaw;
  public float GizmoYaw
  {
    get => _gizmoYaw;
    private set => SetAndRaise(GizmoYawProperty, ref _gizmoYaw, value);
  }

  private float _gizmoRoll;
  public float GizmoRoll
  {
    get => _gizmoRoll;
    private set => SetAndRaise(GizmoRollProperty, ref _gizmoRoll, value);
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
  /// Converts IAU pole (RA, Dec, W) → Euler (Pitch, Yaw, Roll) for the
  /// DualRotationGizmo, using the shared <see cref="AetherVk.Logic.ViewModels.IauRotationMath"/>.
  /// </summary>
  private void RecomputeGizmoAngles()
  {
    var (w, x, y, z) = AetherVk.Logic.ViewModels.IauRotationMath.IauToQuaternion(
      PoleRaDeg, PoleDecDeg, PrimeMeridianDeg);

    var (pitch, yaw, roll) = AetherVk.Logic.ViewModels.IauRotationMath.QuaternionToGizmoEuler(
      w, x, y, z);

    GizmoPitch = (float)pitch;
    GizmoYaw = (float)yaw;
    GizmoRoll = (float)roll;
  }
}
