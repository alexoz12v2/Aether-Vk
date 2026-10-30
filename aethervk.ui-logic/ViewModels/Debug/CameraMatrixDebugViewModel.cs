using System;
using System.Reactive.Concurrency;
using System.Reactive.Linq;
using System.Reactive.Disposables;
using AetherVk.Logic.Attributes;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels.Debug;

/// <summary>
/// The viewport's ephemeris panel (reviewer docx item 15): the committed comet's ephemeris at the
/// shown epoch (<see cref="INativeRuntimeService.CometEphemeris"/>, polled at 2 Hz) formatted as a
/// JPL Horizons observer table, with the camera position and rotation emitted by the Rust render
/// loop under a collapsed "Debug" expander (<see cref="IsDebugExpanded"/>). Shown in all builds:
/// the ephemeris whenever a comet is committed, the diagnostic sections with <see cref="ShowDiagnostics"/>.
/// </summary>
[GenerateLocalizedStrings(keyPrefix: "Ephemeris_", designTitle: "Ephemeris")]
public sealed partial class CameraMatrixDebugViewModel : ObservableObject, IDisposable
{
    /// <summary>Shown in every ephemeris row without a committed comet.</summary>
    public const string NoValue = "—";

    private readonly CompositeDisposable _disposables = new();
    private readonly PreferencesService? _preferences;
    private readonly ITranslationService _translationService = null!;

    /// <summary>Panel body open (true) or collapsed to its icon.</summary>
    [ObservableProperty] private bool _isExpanded = true;

    partial void OnIsExpandedChanged(bool value) => _preferences?.SetHudCameraExpanded(value);

    [RelayCommand] private void ToggleExpanded() => IsExpanded = !IsExpanded;

    /// <summary>The diagnostic sections (Debug expander, Earth observer, dust history) are shown
    /// (<see cref="PreferencesService.ShowDiagnosticPanels"/>; shown without preferences).</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsShown))]
    private bool _showDiagnostics = true;

    /// <summary>A comet is committed and the almanac covers the shown epoch: the ephemeris rows
    /// have values.</summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(IsShown))]
    private bool _hasEphemeris;

    /// <summary>The panel is shown at all: the ephemeris is not a diagnostic (the reviewer asked
    /// for it in place of the position table), so it shows whenever it has values.</summary>
    public bool IsShown => HasEphemeris || ShowDiagnostics;

    /// <summary>Dust history per age tier (<see cref="FormatDustHistory"/>), empty without dust.</summary>
    [ObservableProperty] private string _dustHistory = string.Empty;

    /// <summary>Native Earth observer status (<see cref="FormatEarthObserver"/>), empty when the UI
    /// poses the camera.</summary>
    [ObservableProperty] private string _earthObserver = string.Empty;

    /// <summary>The "Debug" expander (the cartesian camera position and rotation) open; collapsed
    /// by default.</summary>
    [ObservableProperty] private bool _isDebugExpanded;

    // ── Ephemeris rows (formatted, NoValue without a committed comet) ────────────

    /// <summary>The shown epoch, "yyyy-MM-dd HH:mm UTC" (the app's epoch display format).</summary>
    [ObservableProperty] private string _ephemerisEpoch = NoValue;
    /// <summary>Astrometric J2000 right ascension, "07h 51m 23.4s".</summary>
    [ObservableProperty] private string _ephemerisRa = NoValue;
    /// <summary>Astrometric J2000 declination, "+21° 05′ 12″".</summary>
    [ObservableProperty] private string _ephemerisDec = NoValue;
    /// <summary>Heliocentric distance, "1.2731 AU".</summary>
    [ObservableProperty] private string _ephemerisR = NoValue;
    /// <summary>Geocentric distance, "0.4402 AU".</summary>
    [ObservableProperty] private string _ephemerisDelta = NoValue;
    /// <summary>Phase angle Sun–comet–Earth, "48.2°".</summary>
    [ObservableProperty] private string _ephemerisPhase = NoValue;
    /// <summary>Solar elongation Sun–Earth–comet, "112.3°".</summary>
    [ObservableProperty] private string _ephemerisElongation = NoValue;
    /// <summary>Position angle of the extended Sun→comet vector, north through east (Horizons'
    /// PsAng), "287.6°".</summary>
    [ObservableProperty] private string _ephemerisSunPa = NoValue;

    [ObservableProperty] private double _posX;
    [ObservableProperty] private double _posY;
    [ObservableProperty] private double _posZ;

    [ObservableProperty] private double _rotX;
    [ObservableProperty] private double _rotY;
    [ObservableProperty] private double _rotZ;
    [ObservableProperty] private double _rotW;

    private const double AuToKm = 149_597_870.7;

    /// <param name="cometScreen">The comet's screen position in the viewport and the viewport's
    /// physical size (<see cref="CameraService.GetCometScreenPosition"/>), appended to the observer
    /// line; null leaves it out.</param>
    /// <param name="translationService">The row labels' strings (<c>Ephemeris_*</c>); null leaves
    /// them empty.</param>
    public CameraMatrixDebugViewModel(INativeRuntimeService runtimeService, ISchedulerProvider schedulerProvider, PreferencesService? preferences = null,
        Func<(CometScreenPosition? Position, float WidthPx, float HeightPx)>? cometScreen = null,
        ITranslationService? translationService = null)
    {
        _preferences = preferences;
        _isExpanded = preferences?.HudCameraExpanded ?? true;
        if (translationService is not null)
        {
            _translationService = translationService;
            RefreshStrings();
            _translationService.CultureChanged.Skip(1)
                .ObserveOn(schedulerProvider.MainThread)
                .Subscribe(_ => RefreshStrings())
                .AddDisposableTo(_disposables);
        }
        if (preferences is not null)
        {
            _showDiagnostics = preferences.ShowDiagnosticPanels;
            Action<bool> shown = v => schedulerProvider.MainThread.Schedule(() => ShowDiagnostics = v);
            preferences.ShowDiagnosticPanelsChanged += shown;
            _disposables.Add(Disposable.Create(() => preferences.ShowDiagnosticPanelsChanged -= shown));
        }
        _disposables.Add(
            Observable.Timer(TimeSpan.Zero, TimeSpan.FromMilliseconds(500), schedulerProvider.Background)
                .Select(_ =>
                {
                    string dust = FormatDustHistory(runtimeService.DustStats());
                    var status = runtimeService.EarthObserverStatus();
                    var screen = status is not null && cometScreen is not null
                        ? cometScreen()
                        : default((CometScreenPosition? Position, float WidthPx, float HeightPx)?);
                    string earth = FormatEarthObserver(status, screen?.Position, screen?.WidthPx ?? 0f, screen?.HeightPx ?? 0f);
                    var ephemeris = FormatEphemeris(runtimeService.CometEphemeris());
                    bool hasCam = false;
                    double px = 0, py = 0, pz = 0, rx = 0, ry = 0, rz = 0, rw = 0;
                    if (runtimeService.CameraEntityId.HasValue &&
                        runtimeService.DebugCameraState(runtimeService.CameraEntityId.Value, out px, out py, out pz, out rx, out ry, out rz, out rw))
                    {
                        hasCam = true;
                        px *= AuToKm;
                        py *= AuToKm;
                        pz *= AuToKm;
                    }
                    return (dust, earth, ephemeris, hasCam, px, py, pz, rx, ry, rz, rw);
                })
                .DistinctUntilChanged()
                .ObserveOn(schedulerProvider.MainThread)
                .Subscribe(s =>
                {
                    DustHistory = s.dust;
                    EarthObserver = s.earth;
                    HasEphemeris = !ReferenceEquals(s.ephemeris, CometEphemerisText.Empty);
                    EphemerisEpoch = s.ephemeris.Epoch;
                    EphemerisRa = s.ephemeris.Ra;
                    EphemerisDec = s.ephemeris.Dec;
                    EphemerisR = s.ephemeris.R;
                    EphemerisDelta = s.ephemeris.Delta;
                    EphemerisPhase = s.ephemeris.Phase;
                    EphemerisElongation = s.ephemeris.Elongation;
                    EphemerisSunPa = s.ephemeris.SunPa;
                    if (s.hasCam)
                    {
                        PosX = s.px;
                        PosY = s.py;
                        PosZ = s.pz;
                        RotX = s.rx;
                        RotY = s.ry;
                        RotZ = s.rz;
                        RotW = s.rw;
                    }
                })
        );
    }

    /// <summary>
    /// One line per age tier, e.g. <c>T0 1,391/16,384 0.0–0.2 d [0–30 d] building</c>: live clusters
    /// over the tier capacity, the age span present, the tier's age band, and whether every
    /// emission window due so far exists yet.
    /// </summary>
    /// <summary>
    /// e.g. <c>CometTracking · aim 2.1e-10 rad · target +34.2° · screen (561.5, 347.0) px of 1123×694</c>,
    /// or <c>SunTracking · target −12.0° below horizon: facing its azimuth</c>. The aim is the
    /// camera's view axis against the comet as drawn (light-time retarded), measured on the camera's
    /// current pose; the screen part is where the comet is drawn (physical px from the top-left;
    /// centred = half the viewport), present when <paramref name="screen"/> is.
    /// </summary>
    public static string FormatEarthObserver(EarthObserverStatus? s, CometScreenPosition? screen = null,
        float viewportWidthPx = 0f, float viewportHeightPx = 0f)
    {
        if (s is null)
            return string.Empty;
        string mode = Enum.GetName(typeof(EarthObserverOrientationMode), s.Mode) ?? s.Mode.ToString();
        double elevDeg = s.TargetElevationRad * 180.0 / Math.PI;
        bool tracking = s.Mode == (int)EarthObserverOrientationMode.CometTracking
                     || s.Mode == (int)EarthObserverOrientationMode.SunTracking;
        string where = screen is null
            ? string.Empty
            : FormattableString.Invariant(
                $" · screen ({screen.XPx:0.0}, {screen.YPx:0.0}) px of {viewportWidthPx:0}×{viewportHeightPx:0}{(screen.OnScreen ? "" : " off-screen")}");
        if (tracking && elevDeg < 0.0)
            return FormattableString.Invariant($"{mode} · target {elevDeg:+0.0;-0.0}° below horizon: facing its azimuth{where}");
        return FormattableString.Invariant($"{mode} · aim {s.AimErrorRad:0.0e+0} rad · target {elevDeg:+0.0;-0.0}°{where}");
    }

    /// <summary>The ephemeris rows as shown: every value <see cref="NoValue"/> for null (no
    /// committed comet).</summary>
    public sealed record CometEphemerisText(string Epoch, string Ra, string Dec, string R, string Delta,
        string Phase, string Elongation, string SunPa)
    {
        public static readonly CometEphemerisText Empty =
            new(NoValue, NoValue, NoValue, NoValue, NoValue, NoValue, NoValue, NoValue);
    }

    /// <summary>
    /// Formats an ephemeris as Horizons prints one: the epoch as "yyyy-MM-dd HH:mm UTC", RA in
    /// hours/minutes/seconds to 0.1 s, Dec in signed degrees/arcminutes/arcseconds, r and Δ in AU
    /// to 1e-4, the angles in degrees to 0.1°. Invariant culture.
    /// </summary>
    public static CometEphemerisText FormatEphemeris(CometEphemeris? e)
    {
        if (e is null)
            return CometEphemerisText.Empty;
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        return new CometEphemerisText(
            e.Epoch.UtcDateTime.ToString("yyyy-MM-dd HH:mm", inv) + " UTC",
            FormatRightAscension(e.RaRad),
            FormatDeclination(e.DecRad),
            string.Format(inv, "{0:0.0000} AU", e.RAu),
            string.Format(inv, "{0:0.0000} AU", e.DeltaAu),
            string.Format(inv, "{0:0.0}°", e.PhaseRad * 180.0 / Math.PI),
            string.Format(inv, "{0:0.0}°", e.ElongationRad * 180.0 / Math.PI),
            string.Format(inv, "{0:0.0}°", e.SunPaRad * 180.0 / Math.PI));
    }

    /// <summary>"07h 51m 23.4s": sexagesimal hours of a right ascension (rad, any value: wrapped
    /// to [0, 24h)), seconds to 0.1 s (1.5″), a rounding carry wrapping 24h to 0h.</summary>
    public static string FormatRightAscension(double raRad)
    {
        double hours = raRad * 12.0 / Math.PI;
        hours -= Math.Floor(hours / 24.0) * 24.0;
        // round in tenths of a second so the carry is exact
        long tenths = (long)Math.Round(hours * 36000.0);
        tenths %= 24L * 36000L;
        long h = tenths / 36000L, m = tenths / 600L % 60L;
        double sec = tenths % 600L / 10.0;
        return string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0:00}h {1:00}m {2:00.0}s", h, m, sec);
    }

    /// <summary>"+21° 05′ 12″": a signed sexagesimal declination (rad), to 1″.</summary>
    public static string FormatDeclination(double decRad)
    {
        double deg = decRad * 180.0 / Math.PI;
        char sign = deg < 0.0 ? '-' : '+';
        long arcsec = (long)Math.Round(Math.Abs(deg) * 3600.0);
        long d = arcsec / 3600L, m = arcsec / 60L % 60L, s = arcsec % 60L;
        return string.Format(System.Globalization.CultureInfo.InvariantCulture, "{0}{1:00}° {2:00}′ {3:00}″", sign, d, m, s);
    }

    public static string FormatDustHistory(System.Collections.Generic.IReadOnlyList<DustTierStats>? tiers)
    {
        if (tiers is null)
            return string.Empty;
        var sb = new System.Text.StringBuilder();
        var inv = System.Globalization.CultureInfo.InvariantCulture;
        for (int i = 0; i < tiers.Count; i++)
        {
            var t = tiers[i];
            if (i > 0) sb.Append('\n');
            sb.Append(string.Format(inv, "T{0} {1:N0}/{2:N0} {3:0.0}–{4:0.0} d [{5:0}–{6:0} d]{7}",
                i, t.LiveClusters, t.Capacity, t.YoungestAgeS / 86400.0, t.OldestAgeS / 86400.0,
                t.BandMinS / 86400.0, t.BandMaxS / 86400.0, t.CaughtUp ? "" : " building"));
        }
        return sb.ToString();
    }

    /// <inheritdoc/>
    public void Dispose() => _disposables.Dispose();
}
