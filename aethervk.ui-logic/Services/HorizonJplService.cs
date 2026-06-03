using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Services;

// ─────────────────────────────────────────────────────────────── Data models

public class PlanetOrbitData
{
  /// <summary>Semi-major axis in AU (converted from km at parse time).</summary>
  public double SemiMajorAxisAu { get; set; }
  public double Eccentricity { get; set; }
  public double Inclination { get; set; }
  public double MeanAnomaly { get; set; }
  public double AscendingNodeLongitude { get; set; }
  public double ArgumentOfPerifocus { get; set; }
  public string RawConstants { get; set; } = string.Empty;
  public double CometRadiusKm { get; set; } = 1.0;

  /// <summary>
  /// GM in km³/s² as returned by JPL (null / 0 when reported as 'n.a.').
  /// </summary>
  public double GmKm3s2 { get; set; }

  /// <summary>
  /// Mass in kg derived from GM, or null when GM is 'n.a.' and no fallback is used.
  /// For Static / Kinematic this value is irrelevant (no gravitational emitter created).
  /// For Dynamic mode the user must supply it via the mass slider when null.
  /// </summary>
  public double? MassKg { get; set; }

  /// <summary>
  /// Density-based mass estimate (kg) using a 600 kg/m³ average comet density.
  /// Always populated when CometRadiusKm is known; used as the slider default for Dynamic.
  /// </summary>
  public double EstimatedMassKg =>
    (4.0 / 3.0) * Math.PI * Math.Pow(CometRadiusKm * 1000.0, 3) * 600.0;
}

// ─────────────────────────────────────────────────────────────── Service

public class HorizonJplService
{
  // ── Infrastructure
  public HttpClient _httpClient;
  private readonly ConsoleService _console;
  private readonly BreadcrumbService _breadcrumb;
  private readonly ILocalStorageService _storage;

  private const string HorizonsBase = "https://ssd.jpl.nasa.gov/api/horizons.api";
  private const string SbdbBase = "https://ssd-api.jpl.nasa.gov/sbdb_query.api";

  // ── Observable collections consumed by ViewModels
  public ObservableCollection<CometSearchResult> CometsData { get; } = new();
  public ObservableCollection<SpkRecordItem> SpkRecordsData { get; } = new();
  public ObservableCollection<ObjectDataProperty> ObjectData { get; } = new();

  // ─────────────────────────────────────────────────────── Constructor

  public HorizonJplService(
    ConsoleService console,
    BreadcrumbService breadcrumb,
    ILocalStorageService storage
  )
  {
    _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(3) };
    _httpClient.DefaultRequestHeaders.Add("User-Agent", "AetherVk/1.0");
    _console = console;
    _breadcrumb = breadcrumb;
    _storage = storage;
  }

  /// <summary>Returns the session-scoped file path where a downloaded SPK .bsp should be saved.</summary>
  public string GetSpkSavePath(int naifId) => _storage.GetSessionPath($"spk_{naifId}.bsp");

  // ═══════════════════════════════════════════════════════════════════════════
  //  1. COMET LIST  (SBDB JSON — only place we use JSON, because it's clean)
  // ═══════════════════════════════════════════════════════════════════════════

  public virtual async Task FetchCometsAsync()
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Downloading comet database…");
    try
    {
      // Persistent cache: comet list rarely changes
      const string cacheKey = "comets_list.json";
      var cachePath = _storage.GetPersistentPath(cacheKey);

      string json;
      if (File.Exists(cachePath))
      {
        _console.Log("[HorizonJpl] Loading comet list from persistent cache.");
        json = File.ReadAllText(cachePath);
      }
      else
      {
        var url = $"{SbdbBase}?sb-kind=c&fields=full_name,pdes";
        _console.Log($"[HorizonJpl] GET {url}");
        using var resp = await _httpClient.GetAsync(url);
        if (!resp.IsSuccessStatusCode)
        {
          await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status {(int)resp.StatusCode}");
          return;
        }
        json = await resp.Content.ReadAsStringAsync();
        _console.Log($"[HorizonJpl] Comet list fetched ({json.Length} bytes).");
        await _storage.SavePersistentAsync(cacheKey, System.Text.Encoding.UTF8.GetBytes(json));
      }

      ParseCometsJson(json);
      await _breadcrumb.ShowMessageAsync(
        "Horizon API",
        $"{CometsData.Count} comets loaded.",
        status: 1
      );
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] FetchComets error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  private void ParseCometsJson(string json)
  {
    CometsData.Clear();
    try
    {
      using var doc = System.Text.Json.JsonDocument.Parse(json);
      if (doc.RootElement.TryGetProperty("data", out var data))
      {
        foreach (var row in data.EnumerateArray())
        {
          var cols = row.EnumerateArray().ToArray();
          if (cols.Length >= 2)
          {
            CometsData.Add(
              new CometSearchResult
              {
                Name =
                  cols[0].ValueKind == System.Text.Json.JsonValueKind.Null
                    ? ""
                    : cols[0].ToString(),
                PrimaryDesignation =
                  cols[1].ValueKind == System.Text.Json.JsonValueKind.Null
                    ? ""
                    : cols[1].ToString(),
              }
            );
          }
        }
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] ParseComets: {ex.Message}");
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  1b. SMALL BODY DATA  (SBDB single-object API → SmallBodyDataComponent)
  // ═══════════════════════════════════════════════════════════════════════════

  private const string SbdbSingleBase = "https://ssd-api.jpl.nasa.gov/sbdb.api";

  /// <summary>
  /// Fetches the canonical SBDB record for a comet/asteroid designation.
  /// Returns a <see cref="SmallBodyDataComponent"/> with the NAIF SPKID, or null on failure.
  /// The result is persistently cached per designation.
  /// </summary>
  public async Task<SmallBodyDataComponent?> FetchSmallBodyDataAsync(string designation)
  {
    var cacheKey = $"sbdb_{Sanitize(designation)}.json";
    var cachePath = _storage.GetPersistentPath(cacheKey);

    string json;
    if (File.Exists(cachePath))
    {
      _console.Log($"[HorizonJpl] SBDB cache hit: {cacheKey}");
      json = File.ReadAllText(cachePath);
    }
    else
    {
      var sstr = Uri.EscapeDataString(designation);
      var url = $"{SbdbSingleBase}?sstr={sstr}";
      _console.Log($"[HorizonJpl] SBDB GET: {url}");

      using var resp = await _httpClient.GetAsync(url);
      if (!resp.IsSuccessStatusCode)
      {
        _console.Log($"[HorizonJpl] SBDB HTTP {(int)resp.StatusCode}");
        return null;
      }
      json = await resp.Content.ReadAsStringAsync();
      _console.Log($"[HorizonJpl] SBDB response ({json.Length} bytes)");
      await _storage.SavePersistentAsync(cacheKey, System.Text.Encoding.UTF8.GetBytes(json));
    }

    return ParseSmallBodyJson(json);
  }

  private SmallBodyDataComponent? ParseSmallBodyJson(string json)
  {
    try
    {
      using var doc = System.Text.Json.JsonDocument.Parse(json);
      if (!doc.RootElement.TryGetProperty("object", out var obj))
        return null;

      var result = new SmallBodyDataComponent();

      if (obj.TryGetProperty("spkid", out var spkidProp) &&
          int.TryParse(spkidProp.GetString(), out int spkid))
        result.SpkId = spkid;

      if (obj.TryGetProperty("des", out var des))
        result.Designation = des.GetString() ?? string.Empty;

      if (obj.TryGetProperty("fullname", out var fn))
        result.FullName = fn.GetString() ?? string.Empty;

      if (obj.TryGetProperty("kind", out var kind))
        result.Kind = kind.GetString() ?? string.Empty;

      if (obj.TryGetProperty("prefix", out var prefix))
        result.Prefix = prefix.GetString() ?? string.Empty;

      if (obj.TryGetProperty("neo", out var neo))
        result.IsNeo = neo.ValueKind == System.Text.Json.JsonValueKind.True;

      if (obj.TryGetProperty("pha", out var pha))
        result.IsPha = pha.ValueKind == System.Text.Json.JsonValueKind.True;

      if (obj.TryGetProperty("orbit_id", out var oid))
        result.OrbitId = oid.GetString() ?? string.Empty;

      if (obj.TryGetProperty("orbit_class", out var oc))
      {
        if (oc.TryGetProperty("name", out var ocName))
          result.OrbitClassName = ocName.GetString() ?? string.Empty;
        if (oc.TryGetProperty("code", out var ocCode))
          result.OrbitClassCode = ocCode.GetString() ?? string.Empty;
      }

      _console.Log($"[HorizonJpl] SBDB parsed: spkid={result.SpkId} des={result.Designation} name={result.FullName}");
      return result;
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] SBDB parse error: {ex.Message}");
      return null;
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  2. SPK RECORD ENUMERATION  (Horizons text, EPHEM_TYPE=SPK)
  // ═══════════════════════════════════════════════════════════════════════════

  public async Task FetchSpkRecordsAsync(string pdes, string startTime, string stopTime)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Enumerating SPK records…");
    try
    {
      // DES= prefix causes Horizons to list all SPK records for the designation.
      // All parameter values must be URL-encoded (including the enclosing single-quotes),
      // matching the same pattern used by FetchPlanetOrbitDataAsync and GetPlanetDataAsync.
      var start = Uri.EscapeDataString($"'{startTime}'");
      var stop = Uri.EscapeDataString($"'{stopTime}'");
      var cmd = Uri.EscapeDataString($"'DES={pdes};'");
      var cen = Uri.EscapeDataString("'@10'");
      var yes = Uri.EscapeDataString("'YES'");
      var no = Uri.EscapeDataString("'NO'");
      var spk = Uri.EscapeDataString("'SPK'");
      var step = Uri.EscapeDataString("'1 d'");

      var url =
        $"{HorizonsBase}?format=text"
        + $"&COMMAND={cmd}"
        + $"&MAKE_EPHEM={yes}&EPHEM_TYPE={spk}&OBJ_DATA={no}"
        + $"&CENTER={cen}"
        + $"&START_TIME={start}&STOP_TIME={stop}&STEP_SIZE={step}";

      _console.Log($"[HorizonJpl] SPK Records GET: {url}");
      using var resp = await _httpClient.GetAsync(url);
      if (!resp.IsSuccessStatusCode)
      {
        await _breadcrumb.ShowMessageAsync(
          "Horizon API Error",
          $"Status {(int)resp.StatusCode}",
          status: 3
        );
        return;
      }
      var text = await resp.Content.ReadAsStringAsync();
      _console.Log($"[HorizonJpl] SPK Records Response:\n{text}");
      ParseSpkRecordsText(text, startTime, stopTime);
      await _breadcrumb.ShowMessageAsync(
        "Horizon API",
        $"{SpkRecordsData.Count} records found.",
        status: 1
      );
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] FetchSpkRecords error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  private void ParseSpkRecordsText(string text, string startTime, string stopTime)
  {
    SpkRecordsData.Clear();

    var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    bool inData = false;

    foreach (var line in lines)
    {
      var trimmed = line.Trim();
      if (trimmed.StartsWith("---"))
      {
        if (inData)
          break;
        inData = true;
        continue;
      }
      if (!inData)
        continue;
      if (trimmed.StartsWith("*") || string.IsNullOrWhiteSpace(trimmed))
        break;

      var parts = trimmed.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
      if (parts.Length >= 5)
      {
        var id = parts[0];
        var epoch = parts[1];
        var match = parts[2];
        var prim = parts[3];
        var name = string.Join(" ", parts.Skip(4));

        // Skip placeholder / informational rows whose id isn't a valid positive integer
        // e.g. "(9 match — enter record # (integer), followed by semi-colon.)"
        if (!int.TryParse(id, out int numId) || numId <= 0)
          continue;

        SpkRecordsData.Add(
          new SpkRecordItem
          {
            RecordId = id,
            EpochYear = epoch,
            MatchDesig = match,
            PrimaryDesig = prim,
            Name = name,
          }
        );
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  3. OBJECT CONSTANTS  (Horizons text, OBJ_DATA=YES, MAKE_EPHEM=NO)
  //     Returns nucleus radius & mass in addition to populating ObjectData.
  // ═══════════════════════════════════════════════════════════════════════════

  /// <summary>
  /// Fetches physical object constants for the given SPK/designation.
  /// Populates ObjectData for display and returns (radiusKm, massKg).
  /// Uses persistent cache so repeated calls are free.
  /// </summary>
  public async Task<(double radiusKm, double massKg)> FetchObjectConstantsAsync(string targetId)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Fetching object constants…");
    try
    {
      var cacheKey = $"obj_{Sanitize(targetId)}.txt";
      var cachePath = _storage.GetPersistentPath(cacheKey);

      string text;
      if (File.Exists(cachePath))
      {
        _console.Log($"[HorizonJpl] Obj constants from cache: {cacheKey}");
        text = File.ReadAllText(cachePath);
      }
      else
      {
        var cmd = Uri.EscapeDataString($"'{targetId};'");
        var cen = Uri.EscapeDataString("'@10'");
        var no = Uri.EscapeDataString("'NO'");
        var yes = Uri.EscapeDataString("'YES'");

        var url =
          $"{HorizonsBase}?format=text"
          + $"&COMMAND={cmd}"
          + $"&MAKE_EPHEM={no}&OBJ_DATA={yes}&CENTER={cen}";
        _console.Log($"[HorizonJpl] Object Constants: {url}");

        using var resp = await _httpClient.GetAsync(url);
        if (!resp.IsSuccessStatusCode)
        {
          await _breadcrumb.ShowMessageAsync(
            "Horizon API Error",
            $"Status {(int)resp.StatusCode}",
            status: 3
          );
          return (1.0, 1e13);
        }
        text = await resp.Content.ReadAsStringAsync();
        _console.Log($"[HorizonJpl] Object Constants Response:\n{text}");
        await _storage.SavePersistentAsync(cacheKey, System.Text.Encoding.UTF8.GetBytes(text));
      }

      ParseObjectDataText(text);

      double radiusKm = ParseValue(text, @"R_vol\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0)
        radiusKm = ParseValue(text, @"RAD\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0)
        radiusKm = 1.0;

      double gm = ParseValue(text, @"GM\s*=\s*([^\s,+]+)");
      double massKg;
      if (gm > 0)
        massKg = gm / 6.6743e-20; // G in km³/(kg·s²)
      else
      {
        double rm = radiusKm * 1000.0;
        massKg = (4.0 / 3.0) * Math.PI * rm * rm * rm * 600.0; // density 600 kg/m³
      }

      return (radiusKm, massKg);
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] FetchObjectConstants error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
      return (1.0, 1e13);
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  // kept for HorizonJplViewModel download flow
  public async Task FetchObjectDataAsync(string targetId) =>
    await FetchObjectConstantsAsync(targetId);

  private void ParseObjectDataText(string text)
  {
    ObjectData.Clear();
    var block = ExtractConstantsBlock(text);
    var lines = block.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    foreach (var line in lines)
    {
      if (line.StartsWith("*") || string.IsNullOrWhiteSpace(line))
        continue;
      var norm = Regex.Replace(line, @"([=:])\s+", "$1");
      var parts = norm.Split(new[] { "  ", "\t" }, StringSplitOptions.RemoveEmptyEntries);
      foreach (var part in parts)
      {
        var p = part.Trim();
        if (string.IsNullOrEmpty(p))
          continue;
        if (p.Contains('=') || p.Contains(':'))
        {
          var sep = p.Contains('=') ? '=' : ':';
          var kv = p.Split(new[] { sep }, 2, StringSplitOptions.RemoveEmptyEntries);
          if (kv.Length == 2)
            ObjectData.Add(
              new ObjectDataProperty { Property = kv[0].Trim(), Value = kv[1].Trim() }
            );
          else
            ObjectData.Add(new ObjectDataProperty { Property = "Info", Value = p });
        }
        else
        {
          ObjectData.Add(new ObjectDataProperty { Property = "Info", Value = p });
        }
      }
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  4. EPA — Orbital Elements  (Horizons text, EPHEM_TYPE=ELEMENTS)
  // ═══════════════════════════════════════════════════════════════════════════

  /// <summary>
  /// Fetches osculating orbital elements (EPA) for the given target.
  /// </summary>
  public virtual async Task<PlanetOrbitData?> FetchPlanetOrbitDataAsync(
    string targetId,
    string center,
    DateTime startDate,
    DateTime stopDate,
    string stepSize
  )
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Fetching orbital elements…");
    try
    {
      var start = Uri.EscapeDataString($"'{startDate.ToString("yyyy-MM-dd")}'");
      var stop = Uri.EscapeDataString($"'{stopDate.ToString("yyyy-MM-dd")}'");
      var step = Uri.EscapeDataString($"'{stepSize}'");
      var cmd = Uri.EscapeDataString($"'{targetId};'");
      var cen = Uri.EscapeDataString($"'{center}'");
      var yes = Uri.EscapeDataString("'YES'");
      var no = Uri.EscapeDataString("'NO'");
      var elems = Uri.EscapeDataString("'ELEMENTS'");

      var url =
        $"{HorizonsBase}?format=text"
        + $"&COMMAND={cmd}"
        + $"&MAKE_EPHEM={yes}&EPHEM_TYPE={elems}&OBJ_DATA={no}"
        + $"&CENTER={cen}"
        + $"&START_TIME={start}&STOP_TIME={stop}&STEP_SIZE={step}";

      _console.Log($"[HorizonJpl] EPA: {url}");
      using var resp = await _httpClient.GetAsync(url);
      if (!resp.IsSuccessStatusCode)
      {
        await _breadcrumb.ShowMessageAsync(
          "Horizon API Error",
          $"Status {(int)resp.StatusCode}",
          status: 3
        );
        return null;
      }

      var text = await resp.Content.ReadAsStringAsync();
      _console.Log($"[HorizonJpl] EPA Response:\n{text}");
      int soeIdx = text.IndexOf("$$SOE");
      int eoeIdx = text.IndexOf("$$EOE");

      if (soeIdx == -1 || eoeIdx <= soeIdx)
      {
        // Log portion of the response for debugging
        _console.Log(
          $"[HorizonJpl] EPA: $$SOE not found. Response snippet:\n{text.Substring(0, Math.Min(600, text.Length))}"
        );
        await _breadcrumb.ShowMessageAsync(
          "Horizon API",
          "No orbital elements returned.",
          status: 2
        );
        return null;
      }

      var epaBlock = text.Substring(soeIdx + 5, eoeIdx - (soeIdx + 5)).Trim();

      const double KmPerAu2 = 149_597_870.7;
      return new PlanetOrbitData
      {
        SemiMajorAxisAu = ParseValue(epaBlock, @"(?<![A-Za-z])A\s*=\s*([^\s,]+)") / KmPerAu2,
        Eccentricity = ParseValue(epaBlock, @"EC\s*=\s*([^\s,]+)"),
        Inclination = ParseValue(epaBlock, @"IN\s*=\s*([^\s,]+)"),
        MeanAnomaly = ParseValue(epaBlock, @"MA\s*=\s*([^\s,]+)"),
        AscendingNodeLongitude = ParseValue(epaBlock, @"OM\s*=\s*([^\s,]+)"),
        ArgumentOfPerifocus = ParseValue(epaBlock, @"W\s*=\s*([^\s,]+)"),
      };
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] FetchEpa error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
      return null;
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  5. FULL "Get Planet Data" (constants + EPA in sequence)
  //     Used by SpawnCometViewModel Step 3 "Fetch Orbit Data" button.
  // ═══════════════════════════════════════════════════════════════════════════

  public virtual async Task<PlanetOrbitData?> GetPlanetDataAsync(
    string targetId,
    string center,
    DateTime startDate,
    DateTime stopDate,
    string stepSize
  )
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Fetching orbital data...");
    try
    {
      string startDateStr = startDate.ToString("yyyy-MM-dd");
      string stopDateStr = stopDate.ToString("yyyy-MM-dd");

      var cacheKey = $"planet_data_{targetId.Replace("/", "_")}_{startDateStr}_{stopDateStr}.txt";
      var cacheFile = _storage.GetSessionPath(cacheKey);

      string epaText = "";
      string objDataText = "";

      if (System.IO.File.Exists(cacheFile) && System.IO.File.Exists(cacheFile + ".obj"))
      {
        _console.Log($"[HorizonJpl] Loading orbital data from cache: {cacheKey}");
        epaText = System.IO.File.ReadAllText(cacheFile);
        objDataText = System.IO.File.ReadAllText(cacheFile + ".obj");
      }
      else
      {
        var start = Uri.EscapeDataString($"'{startDateStr}'");
        var stop = Uri.EscapeDataString($"'{stopDateStr}'");
        var step = Uri.EscapeDataString($"'{stepSize}'");
        var cmd = Uri.EscapeDataString($"'{targetId};'");
        var cen = Uri.EscapeDataString($"'{center}'");
        var yes = Uri.EscapeDataString("'YES'");
        var no = Uri.EscapeDataString("'NO'");
        var elems = Uri.EscapeDataString("'ELEMENTS'");

        // 1. Fetch Object Constants
        var objUrl =
          $"{HorizonsBase}?format=text"
          + $"&COMMAND={cmd}"
          + $"&MAKE_EPHEM={no}&OBJ_DATA={yes}&CENTER={cen}";

        _console.Log($"[HorizonJpl] GET Object Data: {objUrl}");
        using (var objResponse = await _httpClient.GetAsync(objUrl))
        {
          if (!objResponse.IsSuccessStatusCode)
          {
            throw new Exception(
              $"Horizon API Error: GET Object Data failed with Status: {(int)objResponse.StatusCode}. URL: {objUrl}"
            );
          }
          objDataText = await objResponse.Content.ReadAsStringAsync();
          _console.Log($"[HorizonJpl] Object Data Response:\n{objDataText}");
        }

        // 2. Fetch EPA
        var epaUrl =
          $"{HorizonsBase}?format=text"
          + $"&COMMAND={cmd}"
          + $"&MAKE_EPHEM={yes}&EPHEM_TYPE={elems}&OBJ_DATA={no}"
          + $"&CENTER={cen}"
          + $"&START_TIME={start}&STOP_TIME={stop}&STEP_SIZE={step}";

        _console.Log($"[HorizonJpl] GET EPA Data: {epaUrl}");
        using (var epaResponse = await _httpClient.GetAsync(epaUrl))
        {
          if (!epaResponse.IsSuccessStatusCode)
          {
            throw new Exception(
              $"Horizon API Error: GET EPA Data failed with Status: {(int)epaResponse.StatusCode}. URL: {epaUrl}"
            );
          }
          epaText = await epaResponse.Content.ReadAsStringAsync();
          _console.Log($"[HorizonJpl] EPA Response:\n{epaText}");
        }

        await _storage.SaveSessionAsync(cacheKey, System.Text.Encoding.UTF8.GetBytes(epaText));
        await _storage.SaveSessionAsync(
          cacheKey + ".obj",
          System.Text.Encoding.UTF8.GetBytes(objDataText)
        );
      }

      string constantsBlock = ExtractConstantsBlock(objDataText);

      // Extract only the ephemeris data block to avoid matching header values
      int soeIdx = epaText.IndexOf("$$SOE");
      int eoeIdx = epaText.IndexOf("$$EOE");
      if (soeIdx == -1 || eoeIdx <= soeIdx)
      {
        _console.Log(
          $"[HorizonJpl] GetPlanetData: $$SOE not found. EPA Text length: {epaText.Length}. Snippet:\n{epaText.Substring(0, Math.Min(400, epaText.Length))}"
        );
        await _breadcrumb.ShowMessageAsync(
          "Horizon API",
          "No ephemeris data returned for this record. Try a different SPK record or date range.",
          status: 2
        );
        return null;
      }
      var epaBlock = epaText.Substring(soeIdx + 5, eoeIdx - (soeIdx + 5)).Trim();

      // ── EPA element parsing
      // NOTE: The 'A' field shares the letter with MA/TA/AD — use a negative lookbehind
      // so we never accidentally capture the A-suffix of another two-letter field name.
      const double KmPerAu = 149_597_870.7;
      double a_km = ParseValue(epaBlock, @"(?<![A-Za-z])A\s*=\s*([^\s,]+)");
      double ec = ParseValue(epaBlock, @"EC\s*=\s*([^\s,]+)");
      double in_ = ParseValue(epaBlock, @"IN\s*=\s*([^\s,]+)");
      double ma = ParseValue(epaBlock, @"MA\s*=\s*([^\s,]+)");
      double om = ParseValue(epaBlock, @"OM\s*=\s*([^\s,]+)");
      double w = ParseValue(epaBlock, @"W\s*=\s*([^\s,]+)");

      // ── Physical constants
      double radiusKm = ParseValue(objDataText, @"R_vol\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0.0)
        radiusKm = ParseValue(objDataText, @"RAD\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0.0)
        radiusKm = 1.0;

      double gm = ParseValue(objDataText, @"GM\s*=\s*([^\s,+]+)"); // km³/s²
      double? massKg = null;
      if (gm > 0.0)
      {
        const double G_km3_per_kg_s2 = 6.6743e-20;
        massKg = gm / G_km3_per_kg_s2;
      }
      // When GM is 'n.a.' MassKg stays null → Dynamic mode slider will request it.

      return new PlanetOrbitData
      {
        SemiMajorAxisAu = a_km / KmPerAu, // convert km → AU for trajectory renderer
        Eccentricity = ec,
        Inclination = in_,
        MeanAnomaly = ma,
        AscendingNodeLongitude = om,
        ArgumentOfPerifocus = w,
        RawConstants = constantsBlock,
        CometRadiusKm = radiusKm,
        GmKm3s2 = gm,
        MassKg = massKg,
      };
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] GetPlanetDataAsync Exception: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Exception", ex.Message);
      throw; // Rethrow to fail tests explicitly
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  6. SPK BINARY DOWNLOAD
  // ═══════════════════════════════════════════════════════════════════════════

  public async Task<string?> DownloadSpkByIdAsync(
    string pdes,
    string spkId,
    string savePath,
    string startTime,
    string stopTime
  )
  {
    // ── Persistent cache: SPK data is immutable for a given ID + epoch range ──
    var cacheKey = $"spk_{Sanitize(spkId)}_{Sanitize(startTime)}_{Sanitize(stopTime)}.bsp";
    var cachePath = _storage.GetPersistentPath(cacheKey);

    if (File.Exists(cachePath) && new FileInfo(cachePath).Length > 0)
    {
      _console.Log($"[HorizonJpl] SPK persistent cache hit: {cacheKey}");
      // Copy to caller-requested path if different
      if (!string.Equals(cachePath, savePath, StringComparison.OrdinalIgnoreCase))
        File.Copy(cachePath, savePath, overwrite: true);
      return cachePath;
    }

    try
    {
      var start = Uri.EscapeDataString($"'{startTime}'");
      var stop = Uri.EscapeDataString($"'{stopTime}'");
      var cmd = Uri.EscapeDataString($"'{spkId};'");
      var yes = Uri.EscapeDataString("'YES'");
      var no = Uri.EscapeDataString("'NO'");
      var spk = Uri.EscapeDataString("'SPK'");

      var url =
        $"{HorizonsBase}?format=text"
        + $"&COMMAND={cmd}"
        + $"&MAKE_EPHEM={yes}&EPHEM_TYPE={spk}&OBJ_DATA={no}"
        + $"&START_TIME={start}&STOP_TIME={stop}";

      _console.Log($"[HorizonJpl] SPK download (streaming): {url}");

      // Stream response line-by-line to avoid Large Object Heap allocations.
      // Base64 content is decoded in 4 096-char chunks (staying well under the 85 kB LOH threshold).
      using var resp = await _httpClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
      if (!resp.IsSuccessStatusCode)
      {
        _console.Log($"[HorizonJpl] SPK download HTTP {(int)resp.StatusCode}");
        return null;
      }

      using var netStream = await resp.Content.ReadAsStreamAsync();
      using var reader = new StreamReader(
        netStream,
        System.Text.Encoding.ASCII,
        detectEncodingFromByteOrderMarks: false,
        bufferSize: 4096,
        leaveOpen: true
      );
      using var outFs = new FileStream(
        cachePath,
        FileMode.Create,
        FileAccess.Write,
        FileShare.None,
        65536,
        true
      );

      const int ChunkChars = 4096; // divisible by 4 → clean Base64 boundary
      var sb = new System.Text.StringBuilder(ChunkChars + 80);
      bool markerSeen = false;
      string? line;
      long totalDecoded = 0;

      while ((line = await reader.ReadLineAsync()) != null)
      {
        if (!markerSeen)
        {
          if (line.Contains("REFGL1NQ"))
          {
            markerSeen = true;
            sb.Append(line.Trim());
          }
          continue;
        }
        if (string.IsNullOrWhiteSpace(line))
          break; // end of base64 block
        sb.Append(line.Trim());

        // Decode and flush full 4 096-char (3 072-byte) chunks
        while (sb.Length >= ChunkChars)
        {
          var chunk = sb.ToString(0, ChunkChars);
          sb.Remove(0, ChunkChars);
          var decoded = Convert.FromBase64String(chunk);
          await outFs.WriteAsync(decoded, 0, decoded.Length);
          totalDecoded += decoded.Length;
        }
      }

      // Flush any remainder (pad to 4-char boundary if needed)
      if (sb.Length > 0)
      {
        while (sb.Length % 4 != 0)
          sb.Append('=');
        var decoded = Convert.FromBase64String(sb.ToString());
        await outFs.WriteAsync(decoded, 0, decoded.Length);
        totalDecoded += decoded.Length;
      }

      if (!markerSeen || totalDecoded == 0)
      {
        _console.Log("[HorizonJpl] SPK binary marker not found or empty payload.");
        outFs.Close();
        File.Delete(cachePath);
        return null;
      }

      _console.Log(
        $"[HorizonJpl] SPK saved to persistent cache ({totalDecoded:N0} bytes): {cacheKey}"
      );

      outFs.Close(); // MUST close the FileShare.None handle before we can copy the file!

      // Copy to caller-requested path if different
      if (!string.Equals(cachePath, savePath, StringComparison.OrdinalIgnoreCase))
        File.Copy(cachePath, savePath, overwrite: true);

      return cachePath;
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] SPK download error: {ex.Message}");
      // Clean up partial file
      if (File.Exists(cachePath))
        File.Delete(cachePath);
      return null;
    }
  }

  public async Task<bool> DownloadObservationAsync(
    string pdes,
    string spkId,
    DateTimeOffset start,
    DateTimeOffset stop,
    string savePath
  )
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Downloading observation package…");
    try
    {
      string startStr = start.ToString("yyyy-MM-dd");
      string stopStr = stop.ToString("yyyy-MM-dd");

      var spkResult = await DownloadSpkByIdAsync(pdes, spkId, savePath, startStr, stopStr);
      if (spkResult == null)
        throw new Exception("SPK download failed.");

      await FetchObjectDataAsync(spkId);
      // EPA stored in ObjectData display; call FetchEpaAsync if needed separately

      await _breadcrumb.ShowMessageAsync(
        "Horizon API",
        $"Observation saved to {savePath}",
        status: 1
      );
      return true;
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] DownloadObservation error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
      return false;
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  Helpers
  // ═══════════════════════════════════════════════════════════════════════════

  private static string ExtractConstantsBlock(string text)
  {
    int startIdx = text.IndexOf("PHYSICAL PROPERTIES");

    if (startIdx != -1)
    {
      int endIdx = text.IndexOf("*****************", startIdx);
      if (endIdx != -1 && endIdx > startIdx)
      {
        return text.Substring(startIdx, endIdx - startIdx).Trim();
      }
      return text.Substring(startIdx).Trim();
    }
    return "Constants block not found.";
  }

  private static double ParseValue(string text, string pattern)
  {
    var m = Regex.Match(text, pattern, RegexOptions.IgnoreCase);
    if (
      m.Success
      && double.TryParse(
        m.Groups[1].Value,
        System.Globalization.NumberStyles.Any,
        System.Globalization.CultureInfo.InvariantCulture,
        out double v
      )
    )
      return v;
    return 0.0;
  }

  private static string Sanitize(string s) => Regex.Replace(s, @"[^a-zA-Z0-9_\-]", "_");
}
