using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Numerics;
using System.Reactive.Linq;
using AetherVk.Logic.Models;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

/// <summary>One legend line: a colour swatch and what that colour draws in the nucleus view.</summary>
public sealed record GizmoLegendRow(string Key, string Label, string ColorHex);

/// <summary>What the legend describes: the view, the gizmo state (Nucleus view) and whether the
/// orbit lines are drawn (Model and Orbit views).</summary>
public sealed record LegendContext(
  EarthObserverState Mode,
  bool GizmoVisible,
  uint Parts,
  IReadOnlyList<(int Index, Vector4 Color)> Jets,
  bool OrbitLinesShown = true);

/// <summary>
/// The legend of the viewport: the Nucleus view's gizmo (reviewer feedback: the coloured lines
/// were unexplained) or, in the Model and Orbit views, the orbit lines. (reviewer feedback: the coloured lines were unexplained).
/// Shown while the view is the Nucleus view and the sphere gizmo is on; its rows follow the gizmo
/// parts mask (<see cref="SphereGizmoPart"/>), a note on the dashed, dimmed axes (an axis pointing
/// away from the viewer, behind the nucleus) whenever an axis is drawn, plus one row per jet, in
/// that jet's colour. The colours mirror <c>assets/sphere_gizmo.vert</c> (grid white, body X red,
/// body Y green, spin axis light blue, Sun yellow, velocity magenta, an away axis at 0.45 of its
/// colour) and the jets' <c>stream_color</c> (their direction lines).
/// </summary>
public sealed partial class SphereGizmoLegendViewModel : ObservableObject, IDisposable
{
  // sphere_gizmo.vert: default white; X (1,0,0); Y (0,1,0); SPIN_COLOR (0.45,0.68,1); Sun (1,1,0);
  // VELOCITY_COLOR (1,0.35,1); AXIS_AWAY_DIM 0.45 (shown on white)
  internal const string GridColor = "#FFFFFF";
  internal const string BodyXColor = "#FF0000";
  internal const string BodyYColor = "#00FF00";
  internal const string SpinColor = "#73ADFF";
  internal const string AxesAwayColor = "#737373";
  internal const string SunColor = "#FFFF00";
  internal const string VelocityColor = "#FF59FF";

  // Orbit lines. Comet reference orbit: logic_thread.rs ~2394-2400 [1, 0.2, 0.2]; Earth orbit:
  // logic_thread.rs ~2586-2592 [0.3, 0.6, 1.0]; travelled path: scene/trajectory.rs
  // EffectiveTrajectoryComponent::COLOR [1, 0.85, 0.1].
  internal const string CometOrbitColor = "#FF3333";
  internal const string EarthOrbitColor = "#4D99FF";
  internal const string TravelledPathColor = "#FFD91A";

  private static readonly IReadOnlyDictionary<string, string> English = new Dictionary<string, string>
  {
    ["Overlay_LegendTitle"] = "Legend",
    ["Overlay_LegendGrid"] = "Latitude / longitude grid",
    ["Overlay_LegendBodyX"] = "Body X axis",
    ["Overlay_LegendBodyY"] = "Body Y axis",
    ["Overlay_LegendSpin"] = "Spin axis (body Z)",
    ["Overlay_LegendSun"] = "Direction of the Sun",
    ["Overlay_LegendVelocity"] = "Velocity direction (heliocentric)",
    ["Overlay_LegendAxesAway"] = "Dashed, dimmed: the axis points away from you (behind the nucleus)",
    ["Overlay_LegendJet"] = "Jet {0} direction",
    ["Overlay_LegendCometOrbit"] = "Comet orbit (reference, from the orbital elements)",
    ["Overlay_LegendEarthOrbit"] = "Earth orbit",
    ["Overlay_LegendTravelledPath"] = "Path travelled in this run",
  };

  private readonly Dictionary<string, string> _strings = new(English.ToDictionary(kv => kv.Key, kv => kv.Value));
  private readonly System.Reactive.Disposables.CompositeDisposable _disposables = [];
  private readonly Action<Action>? _dispatch;

  private bool _gizmoVisible = true;
  private uint _parts = SphereGizmoPart.Default;
  private EarthObserverState _mode = EarthObserverState.UpZenith;
  private bool _orbitLinesShown = true;
  private bool _built;
  private IReadOnlyList<(int Index, Vector4 Color)> _jets = [];

  /// <summary>The legend lines, in the gizmo's drawing order.</summary>
  public ObservableCollection<GizmoLegendRow> Rows { get; } = [];

  /// <summary>The panel is shown: the Nucleus view with the gizmo on, or the Model / Orbit view
  /// with the orbit lines on.</summary>
  [ObservableProperty]
  private bool _isShown;

  /// <summary>The panel is open (collapses to its toggle icon otherwise).</summary>
  [ObservableProperty]
  private bool _isExpanded = true;

  [ObservableProperty]
  private string _title = English["Overlay_LegendTitle"];

  public SphereGizmoLegendViewModel(ITranslationService? translation = null, Action<Action>? dispatch = null)
  {
    _dispatch = dispatch;
    if (translation is null)
      return;
    foreach (var key in English.Keys.ToArray())
    {
      var k = key;
      translation.GetStringObservable(k)
        .Subscribe(s => Post(() =>
        {
          _strings[k] = s;
          Rebuild();
        }))
        .AddDisposableTo(_disposables);
    }
  }

  private void Post(Action a)
  {
    if (_dispatch is null)
      a();
    else
      _dispatch(a);
  }

  /// <summary>Takes the legend context; rebuilds the rows only when something changed.</summary>
  public void Refresh(LegendContext ctx)
  {
    uint parts = ctx.Parts & SphereGizmoPart.All;
    bool same = ctx.Mode == _mode && ctx.GizmoVisible == _gizmoVisible && parts == _parts
      && ctx.OrbitLinesShown == _orbitLinesShown && ctx.Jets.SequenceEqual(_jets);
    if (same && _built)
      return;
    _built = true;
    _mode = ctx.Mode;
    _gizmoVisible = ctx.GizmoVisible;
    _parts = parts;
    _orbitLinesShown = ctx.OrbitLinesShown;
    _jets = ctx.Jets.ToArray();
    Rebuild();
  }

  private string S(string key) => _strings.TryGetValue(key, out var v) ? v : English[key];

  private void Rebuild()
  {
    Title = S("Overlay_LegendTitle");
    bool nucleus = _mode == EarthObserverState.CometOrbiting;
    var rows = nucleus
      ? (_gizmoVisible ? RowsFor(_parts, _jets, S) : [])
      : (_orbitLinesShown ? OrbitRows(S) : []);
    Rows.Clear();
    foreach (var r in rows)
      Rows.Add(r);
    IsShown = Rows.Count > 0;
  }

  /// <summary>The orbit lines' rows (Model and Orbit views).</summary>
  internal static IReadOnlyList<GizmoLegendRow> OrbitRows(Func<string, string>? text = null)
  {
    text ??= k => English[k];
    return
    [
      new("orbit_comet", text("Overlay_LegendCometOrbit"), CometOrbitColor),
      new("orbit_earth", text("Overlay_LegendEarthOrbit"), EarthOrbitColor),
      new("orbit_path", text("Overlay_LegendTravelledPath"), TravelledPathColor),
    ];
  }

  /// <summary>The rows for a parts mask and the jets (their 1-based index and RGBA colour).</summary>
  internal static IReadOnlyList<GizmoLegendRow> RowsFor(
    uint parts,
    IReadOnlyList<(int Index, Vector4 Color)> jets,
    Func<string, string>? text = null)
  {
    text ??= k => English[k];
    var rows = new List<GizmoLegendRow>();
    if ((parts & SphereGizmoPart.Grid) != 0)
      rows.Add(new("grid", text("Overlay_LegendGrid"), GridColor));
    if ((parts & SphereGizmoPart.BodyAxes) != 0)
    {
      rows.Add(new("body_x", text("Overlay_LegendBodyX"), BodyXColor));
      rows.Add(new("body_y", text("Overlay_LegendBodyY"), BodyYColor));
    }
    if ((parts & SphereGizmoPart.SpinAxis) != 0)
      rows.Add(new("spin", text("Overlay_LegendSpin"), SpinColor));
    if ((parts & SphereGizmoPart.Sun) != 0)
      rows.Add(new("sun", text("Overlay_LegendSun"), SunColor));
    if ((parts & SphereGizmoPart.Velocity) != 0)
      rows.Add(new("velocity", text("Overlay_LegendVelocity"), VelocityColor));
    const uint axes = SphereGizmoPart.BodyAxes | SphereGizmoPart.SpinAxis | SphereGizmoPart.Sun | SphereGizmoPart.Velocity;
    if ((parts & axes) != 0)
      rows.Add(new("axes_away", text("Overlay_LegendAxesAway"), AxesAwayColor));
    foreach (var (index, color) in jets)
      rows.Add(new($"jet_{index}", string.Format(System.Globalization.CultureInfo.CurrentCulture, text("Overlay_LegendJet"), index), Hex(color)));
    return rows;
  }

  internal static string Hex(Vector4 c)
  {
    static int B(float v) => (int)Math.Round(Math.Max(0f, Math.Min(1f, v)) * 255f);
    return $"#{B(c.X):X2}{B(c.Y):X2}{B(c.Z):X2}";
  }

  public void Dispose() => _disposables.Dispose();
}
