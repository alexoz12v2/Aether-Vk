using System;
using System.Collections.ObjectModel;
using System.Threading.Tasks;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels.Debug;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class ViewportOverlayViewModel : ObservableObject, IDisposable
{
    private readonly CameraService _cameraService;
    private readonly INativeRuntimeService _runtimeService;
    private readonly BreadcrumbService _breadcrumbService;
    private readonly IUiThreadDispatcher _dispatcher;
    private readonly IFileDialogService _fileDialogService;
    private readonly Viewport3DViewModel _viewportVm;

    private readonly IDisposable _modeSubscription;
    private readonly IDisposable _sunSubscription;
    private readonly IDisposable _projectionSubscription;
    private CameraProjectionState? _currentProjection;

    // ── Camera Mode Badge ──────────────────────────────────────────────────────
    [ObservableProperty]
    private bool _isModeIndicatorExpanded;

    [ObservableProperty]
    [NotifyPropertyChangedFor(
        nameof(CameraModeName),
        nameof(IsModeEarthPosition),
        nameof(IsModeUpZenith),
        nameof(IsModeCometOrbiting)
    )]
    private EarthObserverState _currentMode = EarthObserverState.UpZenith;

    public string CameraModeName => CurrentMode switch
    {
        EarthObserverState.EarthPositioning => "Earth Position",
        EarthObserverState.UpZenith => "Up Zenith",
        EarthObserverState.CometOrbiting => "Comet Orbiting",
        _ => string.Empty,
    };

    public bool IsModeEarthPosition => CurrentMode == EarthObserverState.EarthPositioning;
    public bool IsModeUpZenith => CurrentMode == EarthObserverState.UpZenith;
    public bool IsModeCometOrbiting => CurrentMode == EarthObserverState.CometOrbiting;

    // ── Radial Menu ───────────────────────────────────────────────────────────
    [ObservableProperty]
    [NotifyPropertyChangedFor(
        nameof(RadialHubLeft),
        nameof(RadialHubTop),
        nameof(RadialCometLeft),
        nameof(RadialCometTop),
        nameof(RadialBillboardLeft),
        nameof(RadialBillboardTop),
        nameof(RadialResetCameraLeft),
        nameof(RadialResetCameraTop),
        nameof(RadialSnapLeft),
        nameof(RadialSnapTop),
        nameof(RadialSnapObserverLeft),
        nameof(RadialSnapObserverTop)
    )]
    private bool _isRadialMenuOpen = false;

    [ObservableProperty]
    [NotifyPropertyChangedFor(
        nameof(RadialHubLeft),
        nameof(RadialCometLeft),
        nameof(RadialBillboardLeft),
        nameof(RadialResetCameraLeft),
        nameof(RadialSnapLeft),
        nameof(RadialSnapObserverLeft)
    )]
    private double _radialMenuX = 0.0;

    [ObservableProperty]
    [NotifyPropertyChangedFor(
        nameof(RadialHubTop),
        nameof(RadialCometTop),
        nameof(RadialBillboardTop),
        nameof(RadialResetCameraTop),
        nameof(RadialSnapTop),
        nameof(RadialSnapObserverTop)
    )]
    private double _radialMenuY = 0.0;

    [ObservableProperty]
    [NotifyPropertyChangedFor(
        nameof(IsCometHovered),
        nameof(IsBillboardHovered),
        nameof(IsResetCameraHovered),
        nameof(IsSnapHovered),
        nameof(IsSnapObserverHovered)
    )]
    private string? _hoveredRadialItem;

    public bool IsCometHovered => HoveredRadialItem == "comet";
    public bool IsBillboardHovered => HoveredRadialItem == "billboard";
    public bool IsResetCameraHovered => HoveredRadialItem == "resetcamera";
    public bool IsSnapHovered => HoveredRadialItem == "snap";
    public bool IsSnapObserverHovered => HoveredRadialItem == "snapobserver";

    public string CometRadialLabel => !CanSpawnComet() ? "Destroy\nComet" : "Spawn\nComet";
    public string CometRadialTooltip => !CanSpawnComet() ? "Remove comet from scene" : "Spawn a comet in the scene";

    public bool HasComet => !CanSpawnComet();

    private const double RadialRadius = 100.0;
    private const double ItemSize = 80.0;
    private const double HalfItem = ItemSize / 2.0;
    private const double HubSize = 16.0;

    public double RadialHubLeft => RadialMenuX - HubSize / 2;
    public double RadialHubTop => RadialMenuY - HubSize / 2;

    public double RadialCometLeft => RadialMenuX - HalfItem;
    public double RadialCometTop => RadialMenuY - RadialRadius - HalfItem;

    private static readonly double _cos45 = Math.Cos(Math.PI / 4.0);
    public double RadialBillboardLeft => RadialMenuX + RadialRadius * _cos45 - HalfItem;
    public double RadialBillboardTop => RadialMenuY - RadialRadius * _cos45 - HalfItem;

    public double RadialResetCameraLeft => RadialMenuX + RadialRadius - HalfItem;
    public double RadialResetCameraTop => RadialMenuY - HalfItem;

    public double RadialSnapLeft => RadialMenuX + RadialRadius * _cos45 - HalfItem;
    public double RadialSnapTop => RadialMenuY + RadialRadius * _cos45 - HalfItem;

    public double RadialSnapObserverLeft => RadialMenuX - HalfItem;
    public double RadialSnapObserverTop => RadialMenuY + RadialRadius - HalfItem;

    // Measurement indicator
    [ObservableProperty]
    private string _measurementIndicatorText = "";

    [ObservableProperty]
    private double _measurementIndicatorWidth = 0.0;

    [ObservableProperty]
    private bool _showMeasurementIndicator = false;

    // ── Sun direction indicator ──────────────────────────────────────────────
    /// <summary>
    /// True when the sun is outside the camera frustum and the arrowhead should be shown.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SunIndicatorLeft), nameof(SunIndicatorTop))]
    private bool _isSunOffScreen = false;

    /// <summary>
    /// Rotation angle of the arrowhead in degrees, measured clockwise from screen-up.
    /// 0° = sun is above centre, 90° = right, 180° = below, 270° = left.
    /// </summary>
    [ObservableProperty]
    [NotifyPropertyChangedFor(nameof(SunIndicatorLeft), nameof(SunIndicatorTop))]
    private double _sunIndicatorAngleDeg = 0.0;

    /// <summary>Radius of the indicator circle: min(32 epx, min(W,H)/2).</summary>
    private const double SunIndicatorMaxRadiusEpx = 32.0;

    private double ComputeIndicatorRadius() =>
        Math.Min(SunIndicatorMaxRadiusEpx,
                 Math.Min(_viewportVm.Width, _viewportVm.Height) / 2.0);

    /// <summary>Canvas.Left of the arrowhead border (centred on the indicator circle point).</summary>
    public double SunIndicatorLeft
    {
        get
        {
            double r   = ComputeIndicatorRadius();
            double rad = SunIndicatorAngleDeg * Math.PI / 180.0;
            return (_viewportVm.Width  / 2.0) + r * Math.Sin(rad) - 12.0;
        }
    }

    /// <summary>Canvas.Top of the arrowhead border (centred on the indicator circle point).</summary>
    public double SunIndicatorTop
    {
        get
        {
            double r   = ComputeIndicatorRadius();
            double rad = SunIndicatorAngleDeg * Math.PI / 180.0;
            return (_viewportVm.Height / 2.0) - r * Math.Cos(rad) - 12.0;
        }
    }

    // Billboards
    public ObservableCollection<BillboardViewModel> Billboards { get; } = new();

#if DEBUG
    /// <summary>
    /// RenderDoc frame-capture sub-ViewModel. Non-null only in DEBUG builds.
    /// The overlay binds the button's <c>IsVisible</c> and <c>Command</c> here.
    /// </summary>
    public RenderDocCaptureViewModel? RenderDoc { get; }

    public DebugTelemetryPanelViewModel? DebugTelemetry { get; }
#else
    /// <summary>Always null in Release — ContentControl DataTemplate never fires.</summary>
    public object? RenderDoc => null;
    public object? DebugTelemetry => null;
#endif

    public ViewportOverlayViewModel(
        CameraService cameraService,
        INativeRuntimeService runtimeService,
        BreadcrumbService breadcrumbService,
        IUiThreadDispatcher dispatcher,
        IFileDialogService fileDialogService,
        Viewport3DViewModel viewportVm)
    {
        _cameraService = cameraService;
        _runtimeService = runtimeService;
        _breadcrumbService = breadcrumbService;
        _dispatcher = dispatcher;
        _fileDialogService = fileDialogService;
        _viewportVm = viewportVm;
#if DEBUG
        RenderDoc = new RenderDocCaptureViewModel(runtimeService);
        DebugTelemetry = new DebugTelemetryPanelViewModel(runtimeService);
#endif

        _modeSubscription = _cameraService.CameraModeChanged.Subscribe(mode =>
        {
            CurrentMode = mode switch
            {
                CameraMode.EarthPosition => EarthObserverState.EarthPositioning,
                CameraMode.UpZenith => EarthObserverState.UpZenith,
                CameraMode.CometOrbiting => EarthObserverState.CometOrbiting,
                _ => EarthObserverState.UpZenith,
            };
            IsModeIndicatorExpanded = true;
            _ = Task.Delay(1800).ContinueWith(
                _ => _dispatcher.Dispatch(() => IsModeIndicatorExpanded = false)
            );
        });

        _dispatcher.DispatchAsync(() =>
        {
            UpdateMeasurementIndicator();
            return Task.CompletedTask;
        });

        // ── Sun direction indicator subscription ─────────────────────────────
        _sunSubscription = _cameraService.SunVisibilityChanged.Subscribe(state =>
        {
            if (state.IsVisible)
            {
                // Sun re-entered the frustum — hide the arrowhead.
                IsSunOffScreen = false;
                return;
            }

            // NDC convention from Rust: ndc_x > 0 = right, ndc_y > 0 = above screen centre.
            // atan2(ndc_x, ndc_y) → 0° = up, 90° = right, 180° = down, 270° = left (clockwise).
            double angleDeg = Math.Atan2(state.NdcX, state.NdcY) * 180.0 / Math.PI;
            SunIndicatorAngleDeg = angleDeg;
            IsSunOffScreen = true;
        });

        _projectionSubscription = _cameraService.CameraProjection.Subscribe(proj =>
        {
            _currentProjection = proj;
            _dispatcher.DispatchAsync(() =>
            {
                UpdateMeasurementIndicator();
                return Task.CompletedTask;
            });
        });
    }

    public void Dispose()
    {
        _modeSubscription.Dispose();
        _sunSubscription.Dispose();
        _projectionSubscription.Dispose();
    }

    partial void OnIsRadialMenuOpenChanged(bool oldValue, bool newValue)
    {
        if (newValue)
        {
            OnPropertyChanged(nameof(HasComet));
            OnPropertyChanged(nameof(CometRadialLabel));
            OnPropertyChanged(nameof(CometRadialTooltip));
            CloseRadialMenuAndSpawnCometCommand.NotifyCanExecuteChanged();
        }
    }

    public void OpenRadialMenuAt(double x, double y)
    {
        RadialMenuX = x;
        RadialMenuY = y;
        HoveredRadialItem = null;
        IsRadialMenuOpen = true;
    }

    public void CloseRadialMenu()
    {
        IsRadialMenuOpen = false;
        HoveredRadialItem = null;
    }

    public void UpdateRadialMenuHover(double pointerX, double pointerY)
    {
        if (!IsRadialMenuOpen) return;

        if (HitTestItem(pointerX, pointerY, RadialCometLeft, RadialCometTop))
            HoveredRadialItem = "comet";
        else if (HitTestItem(pointerX, pointerY, RadialBillboardLeft, RadialBillboardTop))
            HoveredRadialItem = "billboard";
        else if (HitTestItem(pointerX, pointerY, RadialResetCameraLeft, RadialResetCameraTop))
            HoveredRadialItem = "resetcamera";
        else if (HitTestItem(pointerX, pointerY, RadialSnapLeft, RadialSnapTop))
            HoveredRadialItem = "snap";
        else if (_viewportVm.IsEarthObserverMode && HitTestItem(pointerX, pointerY, RadialSnapObserverLeft, RadialSnapObserverTop))
            HoveredRadialItem = "snapobserver";
        else
            HoveredRadialItem = null;
    }

    private bool HitTestItem(double px, double py, double itemLeft, double itemTop)
    {
        return px >= itemLeft && px <= itemLeft + ItemSize && py >= itemTop && py <= itemTop + ItemSize;
    }

    [RelayCommand]
    private void CloseRadialMenuCmd() => CloseRadialMenu();

    [RelayCommand]
    private void ResetCameraFromRadial()
    {
        CloseRadialMenu();
        _cameraService.ResetToModeDefault();
    }

    [RelayCommand]
    private void SnapToSelectedFromRadial()
    {
        CloseRadialMenu();
        throw new NotImplementedException();
    }

    [RelayCommand(CanExecute = nameof(CanSpawnComet))]
    private void CloseRadialMenuAndSpawnComet()
    {
        CloseRadialMenu();
        WeakReferenceMessenger.Default.Send(new Messages.OpenSpawnCometDialogMessage());
    }

    private bool CanSpawnComet()
    {
        // TODO: return true when no comet exists, false when one already does.
        return false;
    }

    private void DestroyCometInternal()
    {
        // TODO: call _runtimeService.ReconfigureComet with destroy flags.
    }

    [RelayCommand]
    private async Task InsertBillboard()
    {
        var filters = new[] { "png", "jpg", "jpeg", "bmp" };
        var path = await _fileDialogService.ShowOpenFileDialogAsync("Select Billboard Image", filters);
        if (!string.IsNullOrEmpty(path))
        {
            try
            {
                float ndcX = 0.5f;
                float ndcY = 0.5f;

                var entityId = _runtimeService.AddScreenSpaceBillboard(
                    path!,
                    new ScreenSpaceBillboard(
                        NdcX: ndcX,
                        NdcY: ndcY,
                        Scale: 1.0f,
                        RotationDeg: 0.0f,
                        Opacity: 1.0f,
                        ZIndex: 1
                    )
                );

                if (entityId == 0)
                {
                    _ = _breadcrumbService.ShowMessageAsync("Error", "Failed to create billboard entity.");
                    return;
                }

                var billboard = new BillboardViewModel
                {
                    EntityId = entityId,
                    ImageSource = path,
                    X = (_viewportVm.Width / 2.0) - 50,
                    Y = (_viewportVm.Height / 2.0) - 50,
                    Width = 100,
                    Height = 100,
                    ZIndex = 1,
                    Opacity = 1.0,
                    Scale = 1.0,
                    Rotation = 0.0,
                };

                Billboards.Add(billboard);
                _ = _breadcrumbService.ShowMessageAsync("Billboard Added", $"Loaded image {System.IO.Path.GetFileName(path)}");
            }
            catch (Exception ex)
            {
                _ = _breadcrumbService.ShowMessageAsync("Error", $"Failed to load image: {ex.Message}");
            }
        }
    }

    [RelayCommand]
    private void RemoveBillboard(BillboardViewModel? billboard)
    {
        if (billboard == null) return;

        if (billboard.EntityId != 0)
        {
            _runtimeService.RemoveScreenSpaceBillboard(billboard.EntityId);
        }
        Billboards.Remove(billboard);
    }

    public void UpdateMeasurementIndicator()
    {
        if (_viewportVm.Width <= 0 || _viewportVm.Height <= 0 || _currentProjection == null)
        {
            ShowMeasurementIndicator = false;
            return;
        }

        double target_px_width = Math.Max(24.0, _viewportVm.Width * 0.07);

        if (!_currentProjection.IsPerspective)
        {
            double W_au = _currentProjection.Right - _currentProjection.Left;
            if (W_au > 0)
            {
                double min_au = target_px_width * (W_au / _viewportVm.Width);
                
                if (min_au < 1e-6)
                {
                    double min_km = min_au * 1.495978707e8;
                    if (min_km < 1.0)
                    {
                        double min_m = min_km * 1000.0;
                        double nice_m = GetNiceNumber(min_m);
                        MeasurementIndicatorWidth = nice_m * (_viewportVm.Width / (W_au * 1.495978707e11));
                        MeasurementIndicatorText = $"{FormatNiceNumber(nice_m)} m";
                    }
                    else
                    {
                        double nice_km = GetNiceNumber(min_km);
                        MeasurementIndicatorWidth = nice_km * (_viewportVm.Width / (W_au * 1.495978707e8));
                        MeasurementIndicatorText = $"{FormatNiceNumber(nice_km)} km";
                    }
                }
                else
                {
                    double nice_au = GetNiceNumber(min_au);
                    MeasurementIndicatorWidth = nice_au * (_viewportVm.Width / W_au);
                    MeasurementIndicatorText = $"{FormatNiceNumber(nice_au)} AU";
                }
                ShowMeasurementIndicator = true;
            }
            else
            {
                ShowMeasurementIndicator = false;
            }
        }
        else
        {
            double W_rad = 2.0 * Math.Atan(Math.Tan(_currentProjection.Fov / 2.0) * _currentProjection.Aspect);
            double W_arcsec = (W_rad * 180.0 / Math.PI) * 3600.0;

            if (W_arcsec > 0)
            {
                double min_arcsec = target_px_width * (W_arcsec / _viewportVm.Width);

                if (min_arcsec >= 3600.0)
                {
                    double min_deg = min_arcsec / 3600.0;
                    double nice_deg = GetNiceNumber(min_deg);
                    MeasurementIndicatorWidth = nice_deg * 3600.0 * (_viewportVm.Width / W_arcsec);
                    MeasurementIndicatorText = $"{FormatNiceNumber(nice_deg)} deg";
                }
                else if (min_arcsec >= 60.0)
                {
                    double min_min = min_arcsec / 60.0;
                    double nice_min = GetNiceNumber(min_min);
                    MeasurementIndicatorWidth = nice_min * 60.0 * (_viewportVm.Width / W_arcsec);
                    MeasurementIndicatorText = $"{FormatNiceNumber(nice_min)} arcmin";
                }
                else
                {
                    double nice_arcsec = GetNiceNumber(min_arcsec);
                    MeasurementIndicatorWidth = nice_arcsec * (_viewportVm.Width / W_arcsec);
                    MeasurementIndicatorText = $"{FormatNiceNumber(nice_arcsec)} arcsec";
                }
                ShowMeasurementIndicator = true;
            }
            else
            {
                ShowMeasurementIndicator = false;
            }
        }
    }

    private double GetNiceNumber(double value)
    {
        if (value <= 0) return 1.0;
        double exponent = Math.Floor(Math.Log10(value));
        double fraction = value / Math.Pow(10, exponent);

        double niceFraction;
        if (fraction <= 1.0) niceFraction = 1.0;
        else if (fraction <= 2.0) niceFraction = 2.0;
        else if (fraction <= 5.0) niceFraction = 5.0;
        else niceFraction = 10.0;

        return niceFraction * Math.Pow(10, exponent);
    }

    private string FormatNiceNumber(double value)
    {
        if (value >= 1.0) return value.ToString("0");
        return value.ToString("0.#####");
    }
}
