using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Models;

/// <summary>
/// Data from the JPL Small-Body Database (SBDB) API.
/// Stores the canonical NAIF SPKID and object metadata.
/// </summary>
public partial class SmallBodyDataComponent : ObservableObject
{
  /// <summary>NAIF SPK ID as stored in SPK files (e.g. 1000012 for 67P).</summary>
  [ObservableProperty]
  private int _spkId;

  /// <summary>Short designation (e.g. "67P").</summary>
  [ObservableProperty]
  private string _designation = string.Empty;

  /// <summary>Full name (e.g. "67P/Churyumov-Gerasimenko").</summary>
  [ObservableProperty]
  private string _fullName = string.Empty;

  /// <summary>Object kind code (e.g. "cn" for comet numbered).</summary>
  [ObservableProperty]
  private string _kind = string.Empty;

  /// <summary>Object prefix (e.g. "P" for periodic comet).</summary>
  [ObservableProperty]
  private string _prefix = string.Empty;

  /// <summary>Orbit class name (e.g. "Jupiter-family Comet").</summary>
  [ObservableProperty]
  private string _orbitClassName = string.Empty;

  /// <summary>Orbit class code (e.g. "JFc").</summary>
  [ObservableProperty]
  private string _orbitClassCode = string.Empty;

  /// <summary>Whether this body is a Near-Earth Object.</summary>
  [ObservableProperty]
  private bool _isNeo;

  /// <summary>Whether this body is a Potentially Hazardous Asteroid.</summary>
  [ObservableProperty]
  private bool _isPha;

  /// <summary>Orbit solution ID (e.g. "K213/6").</summary>
  [ObservableProperty]
  private string _orbitId = string.Empty;

  /// <summary>Eccentricity (e).</summary>
  [ObservableProperty]
  private double _e;

  /// <summary>Perihelion distance in AU (q).</summary>
  [ObservableProperty]
  private double _q;

  /// <summary>Inclination in degrees (i).</summary>
  [ObservableProperty]
  private double _i;

  /// <summary>Longitude of the Ascending Node in degrees (om).</summary>
  [ObservableProperty]
  private double _om;

  /// <summary>Argument of Perihelion in degrees (w).</summary>
  [ObservableProperty]
  private double _w;
}
