using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels.Debug;

/// <summary>
/// Debug telemetry for CometOrbiting mode, shown in the overlay panel.
/// Visible only when <see cref="IsVisible"/> is true (i.e. camera is in CometOrbiting mode).
/// </summary>
public sealed partial class CometOrbitDebugViewModel : ObservableObject
{
    /// <summary>In CometOrbiting mode (set by the overlay).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsShown))]
    private bool _isVisible;

    /// <summary>Closed by the user: stays closed for the session.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsShown))]
    private bool _isDismissed;

    /// <summary>What the panel binds: in CometOrbiting and not closed.</summary>
    public bool IsShown => IsVisible && !IsDismissed;

    [RelayCommand]
    private void Dismiss() => IsDismissed = true;

    // ── Distance telemetry ────────────────────────────────────────────────────

    [ObservableProperty] private string _actualDistanceKm   = "–";
    [ObservableProperty] private string _expectedDistanceKm = "–";

    /// <summary>Absolute distance drift between actual and expected orbit radius.</summary>
    [ObservableProperty] private string _driftKm = "–";

    /// <summary>"ok" | "warn" | "error" — drives the drift label colour in AXAML.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsDriftOk), nameof(IsDriftWarn), nameof(IsDriftError))]
    private string _driftLevel = "ok";

    public bool IsDriftOk    => DriftLevel == "ok";
    public bool IsDriftWarn  => DriftLevel == "warn";
    public bool IsDriftError => DriftLevel == "error";

    // ── Orientation telemetry ─────────────────────────────────────────────────

    [ObservableProperty] private string _orbitAzimuthDeg    = "–";
    [ObservableProperty] private string _orbitElevationDeg  = "–";
    [ObservableProperty] private string _cameraOrientationQ = "–";

    // ── Ortho projection invariant ────────────────────────────────────────────

    /// <summary>Actual ortho half-height string (formatted km), or "perspective" if not ortho.</summary>
    [ObservableProperty] private string _orthoHalfHeightKm = "–";

    /// <summary>Expected ortho half-height string (3 × nucleus radius).</summary>
    [ObservableProperty] private string _orthoExpectedKm = "–";

    /// <summary>Drives label colour: true = within 1%, false = out of tolerance.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsOrthoOk), nameof(IsOrthoErr))]
    private bool _orthoInvariantPassed = true;

    public bool IsOrthoOk  =>  OrthoInvariantPassed;
    public bool IsOrthoErr => !OrthoInvariantPassed;
}
