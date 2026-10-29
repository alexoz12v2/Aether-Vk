using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels.Debug;

/// <summary>
/// Debug telemetry for CometOrbiting mode, shown in the overlay panel.
/// Shown only when <see cref="IsVisible"/> is true (i.e. camera is in CometOrbiting mode).
/// </summary>
public sealed partial class CometOrbitDebugViewModel : ObservableObject, System.IDisposable
{
    private readonly PreferencesService? _preferences;

    public CometOrbitDebugViewModel(PreferencesService? preferences = null)
    {
        _preferences = preferences;
        _isExpanded = preferences?.HudOrbitExpanded ?? true;
        _diagnosticsShown = preferences?.ShowDiagnosticPanels ?? true;
        if (preferences is not null)
            preferences.ShowDiagnosticPanelsChanged += OnDiagnosticsChanged;
    }

    private void OnDiagnosticsChanged(bool shown) => DiagnosticsShown = shown;

    public void Dispose()
    {
        if (_preferences is not null)
            _preferences.ShowDiagnosticPanelsChanged -= OnDiagnosticsChanged;
    }

    /// <summary><see cref="PreferencesService.ShowDiagnosticPanels"/> (shown without preferences).
    /// Set on the thread that changed the preference: the Settings tab, the UI thread.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsShown))]
    private bool _diagnosticsShown = true;

    /// <summary>In CometOrbiting mode (set by the overlay).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsShown))]
    private bool _isVisible;

    /// <summary>Panel body open (true) or collapsed to its icon; collapsed is still shown.</summary>
    [ObservableProperty]
    private bool _isExpanded = true;

    partial void OnIsExpandedChanged(bool value) => _preferences?.SetHudOrbitExpanded(value);

    /// <summary>What the panel binds: in CometOrbiting (expanded or collapsed) with the diagnostic
    /// panels on.</summary>
    public bool IsShown => IsVisible && DiagnosticsShown;

    [RelayCommand]
    private void ToggleExpanded() => IsExpanded = !IsExpanded;

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
