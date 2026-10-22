using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using AetherVk.Logic.Input;
using AetherVk.Logic.ViewModels;

namespace AetherVk.Logic.Services;

/// <summary><c>~/.aethervk/preferences.json</c>, format 1 (docs/savefile.md).</summary>
public sealed class PreferencesFile
{
  public const int CurrentFormatVersion = 1;
  public const int MaxRecentSaves = 10;

  public int FormatVersion { get; set; } = CurrentFormatVersion;
  public string Theme { get; set; } = "system";
  public List<string> RecentSaves { get; set; } = [];
  public List<BindingDto>? InputBindings { get; set; }
  /// <summary>Dust volume detail (Settings › Particle rendering): 16, 8, 4, 2 or 1; missing = 16.</summary>
  public int? DustVolumeDetail { get; set; }
  /// <summary>Dust trajectory detail (Settings › Particle rendering): 17, 9 or 5 solved size
  /// edges; missing = 17.</summary>
  [JsonPropertyName("dust_propagate_detail")]
  public int? DustPropagateDetail { get; set; }
  /// <summary>Dust temporal reuse tolerance in pixels (Settings › Particle rendering): 0 (off), 1, 2,
  /// 4, 8 or 16; missing = 0.</summary>
  [JsonPropertyName("dust_reuse_px")]
  public int? DustReuseTolerance { get; set; }
}

public sealed class BindingDto
{
  public string Context { get; set; } = "";
  public string Action { get; set; } = "";
  public ChordDto Chord { get; set; } = new();
}

public sealed class ChordDto
{
  public string? Key { get; set; }
  public bool Shift { get; set; }
  public bool Ctrl { get; set; }
  public bool Alt { get; set; }
  public string? Pointer { get; set; }
}

[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.SnakeCaseLower, WriteIndented = true)]
[JsonSerializable(typeof(PreferencesFile))]
public partial class PreferencesJsonContext : JsonSerializerContext;

/// <summary>
/// Application preferences: theme, recent saves, the dust volume detail and temporal reuse
/// tolerance, and the input bindings of
/// <see cref="InputRegistry"/>. <see cref="Load"/> at startup (after the default bindings are
/// registered and captured); every change is written at once. A missing, unreadable or newer file
/// leaves the defaults; an unreadable one is kept as <c>preferences.json.bad</c>.
/// </summary>
public sealed class PreferencesService
{
  public const string FileName = "preferences.json";
  private readonly string _path;
  private readonly InputRegistry _registry;
  private PreferencesFile _file = new();

  public PreferencesService(ILocalStorageService storage, InputRegistry registry)
    : this(Path.Combine(storage.PersistentDirectory, FileName), registry) { }

  /// <summary>Explicit file (tests).</summary>
  public PreferencesService(string path, InputRegistry registry)
  {
    _path = path;
    _registry = registry;
  }

  public string FilePath => _path;

  public AppTheme Theme => _file.Theme switch
  {
    "light" => AppTheme.Light,
    "dark" => AppTheme.Dark,
    _ => AppTheme.System,
  };

  public IReadOnlyList<string> RecentSaves => _file.RecentSaves;

  /// <summary>The dust volume detail, clamped (<see cref="ParticleRenderingViewModel.ClampVolumeDetail"/>);
  /// 16 when the file has none.</summary>
  public int DustVolumeDetail =>
    ParticleRenderingViewModel.ClampVolumeDetail(_file.DustVolumeDetail ?? ParticleRenderingViewModel.VolumeDetailDefault);

  /// <summary>The dust trajectory detail, clamped (<see cref="ParticleRenderingViewModel.ClampPropagateDetail"/>);
  /// 17 when the file has none.</summary>
  public int DustPropagateDetail =>
    ParticleRenderingViewModel.ClampPropagateDetail(_file.DustPropagateDetail ?? ParticleRenderingViewModel.PropagateDetailDefault);

  /// <summary>The dust temporal reuse tolerance in pixels, clamped
  /// (<see cref="ParticleRenderingViewModel.ClampReuseTolerance"/>); 0 (off) when the file has none.</summary>
  public int DustReuseTolerance =>
    ParticleRenderingViewModel.ClampReuseTolerance(_file.DustReuseTolerance ?? ParticleRenderingViewModel.ReuseToleranceDefault);

  /// <summary>What <see cref="Load"/> did: "defaults", "loaded", "corrupt", "newer".</summary>
  public string LoadOutcome { get; private set; } = "defaults";

  public void Load()
  {
    _file = new PreferencesFile();
    LoadOutcome = "defaults";
    if (!File.Exists(_path))
      return;
    try
    {
      var json = File.ReadAllText(_path, Encoding.UTF8);
      var file = JsonSerializer.Deserialize(json, PreferencesJsonContext.Default.PreferencesFile);
      if (file is null)
        throw new JsonException("empty");
      if (file.FormatVersion > PreferencesFile.CurrentFormatVersion)
      {
        LoadOutcome = "newer";
        return;
      }
      _file = file;
      _file.RecentSaves = _file.RecentSaves.Where(Directory.Exists).Take(PreferencesFile.MaxRecentSaves).ToList();
      if (_file.InputBindings is { } bindings)
        _registry.Replace(bindings.Select(FromDto));
      LoadOutcome = "loaded";
    }
    catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException or InvalidOperationException)
    {
      LoadOutcome = "corrupt";
      try
      {
        File.Copy(_path, _path + ".bad", overwrite: true);
      }
      catch (IOException) { }
      _file = new PreferencesFile();
    }
  }

  public void SetTheme(AppTheme theme)
  {
    _file.Theme = theme switch
    {
      AppTheme.Light => "light",
      AppTheme.Dark => "dark",
      _ => "system",
    };
    Save();
  }

  public void SetDustVolumeDetail(int detail)
  {
    _file.DustVolumeDetail = ParticleRenderingViewModel.ClampVolumeDetail(detail);
    Save();
  }

  public void SetDustPropagateDetail(int detail)
  {
    _file.DustPropagateDetail = ParticleRenderingViewModel.ClampPropagateDetail(detail);
    Save();
  }

  public void SetDustReuseTolerance(int px)
  {
    _file.DustReuseTolerance = ParticleRenderingViewModel.ClampReuseTolerance(px);
    Save();
  }

  /// <summary>Moves <paramref name="saveDir"/> to the front of the recent saves.</summary>
  public void AddRecentSave(string saveDir)
  {
    var full = Path.GetFullPath(saveDir);
    _file.RecentSaves.RemoveAll(p => string.Equals(p, full, StringComparison.Ordinal));
    _file.RecentSaves.Insert(0, full);
    if (_file.RecentSaves.Count > PreferencesFile.MaxRecentSaves)
      _file.RecentSaves.RemoveRange(PreferencesFile.MaxRecentSaves, _file.RecentSaves.Count - PreferencesFile.MaxRecentSaves);
    Save();
  }

  /// <summary>Stores the registry's current table (call after every binding edit).</summary>
  public void SaveBindings()
  {
    _file.InputBindings = _registry.Snapshot().Select(ToDto).ToList();
    Save();
  }

  private void Save()
  {
    Directory.CreateDirectory(Path.GetDirectoryName(_path)!);
    var tmp = _path + ".tmp";
    File.WriteAllText(tmp, JsonSerializer.Serialize(_file, PreferencesJsonContext.Default.PreferencesFile), Encoding.UTF8);
    if (File.Exists(_path))
      File.Replace(tmp, _path, null);
    else
      File.Move(tmp, _path);
  }

  private static BindingDto ToDto(InputBinding b) => new()
  {
    Context = b.Context,
    Action = b.ActionId,
    Chord = new ChordDto { Key = b.Chord.Key, Shift = b.Chord.Shift, Ctrl = b.Chord.Ctrl, Alt = b.Chord.Alt, Pointer = b.Chord.Pointer },
  };

  private static InputBinding FromDto(BindingDto d) =>
    new(d.Context, new InputChord(d.Chord.Key, d.Chord.Shift, d.Chord.Ctrl, d.Chord.Alt, d.Chord.Pointer), d.Action);
}
