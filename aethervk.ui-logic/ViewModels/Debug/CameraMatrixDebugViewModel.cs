using System;
using System.Reactive.Linq;
using System.Reactive.Disposables;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels.Debug;

/// <summary>
/// Debug panel showing the camera position and rotation emitted by the Rust render loop.
/// Shown in all builds (overlay position tracker).
/// </summary>
public sealed partial class CameraMatrixDebugViewModel : ObservableObject, IDisposable
{
    private readonly CompositeDisposable _disposables = new();

    /// <summary>Dust history per age tier (<see cref="FormatDustHistory"/>), empty without dust.</summary>
    [ObservableProperty] private string _dustHistory = string.Empty;

    /// <summary>Native Earth observer status (<see cref="FormatEarthObserver"/>), empty when the UI
    /// poses the camera.</summary>
    [ObservableProperty] private string _earthObserver = string.Empty;

    [ObservableProperty] private double _posX;
    [ObservableProperty] private double _posY;
    [ObservableProperty] private double _posZ;

    [ObservableProperty] private double _rotX;
    [ObservableProperty] private double _rotY;
    [ObservableProperty] private double _rotZ;
    [ObservableProperty] private double _rotW;

    private const double AuToKm = 149_597_870.7;

    public CameraMatrixDebugViewModel(INativeRuntimeService runtimeService, ISchedulerProvider schedulerProvider)
    {
        _disposables.Add(
            Observable.Timer(TimeSpan.Zero, TimeSpan.FromMilliseconds(500), schedulerProvider.Background)
                .Select(_ =>
                {
                    string dust = FormatDustHistory(runtimeService.DustStats());
                    string earth = FormatEarthObserver(runtimeService.EarthObserverStatus());
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
                    return (dust, earth, hasCam, px, py, pz, rx, ry, rz, rw);
                })
                .DistinctUntilChanged()
                .ObserveOn(schedulerProvider.MainThread)
                .Subscribe(s =>
                {
                    DustHistory = s.dust;
                    EarthObserver = s.earth;
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
    /// e.g. <c>CometTracking · aim 2.1e-10 rad · target +34.2°</c>, or
    /// <c>SunTracking · target −12.0° below horizon: facing its azimuth</c>.
    /// </summary>
    public static string FormatEarthObserver(EarthObserverStatus? s)
    {
        if (s is null)
            return string.Empty;
        string mode = Enum.GetName(typeof(EarthObserverOrientationMode), s.Mode) ?? s.Mode.ToString();
        double elevDeg = s.TargetElevationRad * 180.0 / Math.PI;
        bool tracking = s.Mode == (int)EarthObserverOrientationMode.CometTracking
                     || s.Mode == (int)EarthObserverOrientationMode.SunTracking;
        if (tracking && elevDeg < 0.0)
            return FormattableString.Invariant($"{mode} · target {elevDeg:+0.0;-0.0}° below horizon: facing its azimuth");
        return FormattableString.Invariant($"{mode} · aim {s.AimErrorRad:0.0e+0} rad · target {elevDeg:+0.0;-0.0}°");
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
