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
  public double SemiMajorAxis  { get; set; }
  public double Eccentricity   { get; set; }
  public double Inclination    { get; set; }
  public double MeanAnomaly    { get; set; }
  public string RawConstants   { get; set; } = string.Empty;
  public double CometRadiusKm  { get; set; } = 1.0;
  public double MassKg         { get; set; } = 1e13;
}

// ─────────────────────────────────────────────────────────────── Service

public class HorizonJplService
{
  // ── Infrastructure
  private readonly HttpClient          _httpClient;
  private readonly ConsoleService      _console;
  private readonly BreadcrumbService   _breadcrumb;
  private readonly ILocalStorageService _storage;

  private const string HorizonsBase = "https://ssd.jpl.nasa.gov/api/horizons.api";
  private const string SbdbBase     = "https://ssd-api.jpl.nasa.gov/sbdb_query.api";

  // ── Observable collections consumed by ViewModels
  public ObservableCollection<CometSearchResult>  CometsData    { get; } = new();
  public ObservableCollection<SpkRecordItem>      SpkRecordsData { get; } = new();
  public ObservableCollection<ObjectDataProperty> ObjectData    { get; } = new();

  // ─────────────────────────────────────────────────────── Constructor

  public HorizonJplService(ConsoleService console, BreadcrumbService breadcrumb, ILocalStorageService storage)
  {
    _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(3) };
    _httpClient.DefaultRequestHeaders.Add("User-Agent", "AetherVk/1.0");
    _console   = console;
    _breadcrumb = breadcrumb;
    _storage   = storage;
  }

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
        var resp = await _httpClient.GetAsync(url);
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
      await _breadcrumb.ShowMessageAsync("Horizon API", $"{CometsData.Count} comets loaded.", status: 1);
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] FetchComets error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
    }
    finally { _breadcrumb.RemoveMessage(loadMsg); }
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
            CometsData.Add(new CometSearchResult
            {
              Name                = cols[0].ValueKind == System.Text.Json.JsonValueKind.Null ? "" : cols[0].ToString(),
              PrimaryDesignation  = cols[1].ValueKind == System.Text.Json.JsonValueKind.Null ? "" : cols[1].ToString(),
            });
          }
        }
      }
    }
    catch (Exception ex) { _console.Log($"[HorizonJpl] ParseComets: {ex.Message}"); }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  2. SPK RECORD ENUMERATION  (Horizons text, EPHEM_TYPE=SPK)
  // ═══════════════════════════════════════════════════════════════════════════

  public async Task FetchSpkRecordsAsync(string pdes, string startTime, string stopTime)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Enumerating SPK records…");
    try
    {
      // For small-body designation queries we use DES= prefix and NO semicolon
      var desEncoded = Uri.EscapeDataString(pdes);
      var start      = Uri.EscapeDataString(startTime);
      var stop       = Uri.EscapeDataString(stopTime);

      var url = $"{HorizonsBase}?format=text&COMMAND='DES={desEncoded}'&EPHEM_TYPE=SPK" +
                $"&START_TIME='{start}'&STOP_TIME='{stop}'&MAKE_EPHEM=YES&OBJ_DATA=NO";

      _console.Log($"[HorizonJpl] SPK Records: {url}");
      var resp = await _httpClient.GetAsync(url);
      if (!resp.IsSuccessStatusCode)
      {
        await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status {(int)resp.StatusCode}", status: 3);
        return;
      }
      var text = await resp.Content.ReadAsStringAsync();
      _console.Log($"[HorizonJpl] SPK Records Response:\n{text}");
      ParseSpkRecordsText(text, startTime, stopTime);
      await _breadcrumb.ShowMessageAsync("Horizon API", $"{SpkRecordsData.Count} records found.", status: 1);
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] FetchSpkRecords error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
    }
    finally { _breadcrumb.RemoveMessage(loadMsg); }
  }

  private void ParseSpkRecordsText(string text, string startTime, string stopTime)
  {
    SpkRecordsData.Clear();

    int startYear = int.MinValue, stopYear = int.MaxValue;
    if (DateTime.TryParse(startTime, out var dtStart)) startYear = dtStart.Year;
    if (DateTime.TryParse(stopTime,  out var dtStop))  stopYear  = dtStop.Year;

    var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    bool inData = false;

    foreach (var line in lines)
    {
      var trimmed = line.Trim();
      if (trimmed.StartsWith("---"))
      {
        if (inData) break;
        inData = true;
        continue;
      }
      if (!inData) continue;
      if (trimmed.StartsWith("*") || string.IsNullOrWhiteSpace(trimmed)) break;

      var parts = trimmed.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
      if (parts.Length >= 5)
      {
        var id    = parts[0];
        var epoch = parts[1];
        var match = parts[2];
        var prim  = parts[3];
        var name  = string.Join(" ", parts.Skip(4));

        bool include = true;
        if (int.TryParse(epoch, out int ey))
          include = ey >= startYear && ey <= stopYear;

        if (include)
          SpkRecordsData.Add(new SpkRecordItem { RecordId = id, EpochYear = epoch, MatchDesig = match, PrimaryDesig = prim, Name = name });
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
      var cacheKey  = $"obj_{Sanitize(targetId)}.txt";
      var cachePath = _storage.GetPersistentPath(cacheKey);

      string text;
      if (File.Exists(cachePath))
      {
        _console.Log($"[HorizonJpl] Obj constants from cache: {cacheKey}");
        text = File.ReadAllText(cachePath);
      }
      else
      {
        var cmd = Uri.EscapeDataString($"{targetId};");
        var url = $"{HorizonsBase}?format=text&COMMAND='{cmd}'&OBJ_DATA=YES&MAKE_EPHEM=NO";
        _console.Log($"[HorizonJpl] Object Constants: {url}");

        var resp = await _httpClient.GetAsync(url);
        if (!resp.IsSuccessStatusCode)
        {
          await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status {(int)resp.StatusCode}", status: 3);
          return (1.0, 1e13);
        }
        text = await resp.Content.ReadAsStringAsync();
        _console.Log($"[HorizonJpl] Object Constants Response:\n{text}");
        await _storage.SavePersistentAsync(cacheKey, System.Text.Encoding.UTF8.GetBytes(text));
      }

      ParseObjectDataText(text);

      double radiusKm = ParseValue(text, @"R_vol\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0) radiusKm = ParseValue(text, @"RAD\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0) radiusKm = 1.0;

      double gm = ParseValue(text, @"GM\s*=\s*([^\s,+]+)");
      double massKg;
      if (gm > 0)
        massKg = gm / 6.6743e-20;           // G in km³/(kg·s²)
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
    finally { _breadcrumb.RemoveMessage(loadMsg); }
  }

  // kept for HorizonJplViewModel download flow
  public async Task FetchObjectDataAsync(string targetId)
    => await FetchObjectConstantsAsync(targetId);

  private void ParseObjectDataText(string text)
  {
    ObjectData.Clear();
    var block = ExtractConstantsBlock(text);
    var lines = block.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    foreach (var line in lines)
    {
      if (line.StartsWith("*") || string.IsNullOrWhiteSpace(line)) continue;
      var norm  = Regex.Replace(line, @"([=:])\s+", "$1");
      var parts = norm.Split(new[] { "  ", "\t" }, StringSplitOptions.RemoveEmptyEntries);
      foreach (var part in parts)
      {
        var p = part.Trim();
        if (string.IsNullOrEmpty(p)) continue;
        if (p.Contains('=') || p.Contains(':'))
        {
          var sep = p.Contains('=') ? '=' : ':';
          var kv  = p.Split(new[] { sep }, 2, StringSplitOptions.RemoveEmptyEntries);
          if (kv.Length == 2)
            ObjectData.Add(new ObjectDataProperty { Property = kv[0].Trim(), Value = kv[1].Trim() });
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
    string  stepSize)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Fetching orbital elements…");
    try
    {
      var cmd      = Uri.EscapeDataString($"{targetId};");
      var ctr      = Uri.EscapeDataString(center);
      var start    = Uri.EscapeDataString(startDate.ToString("yyyy-MM-dd"));
      var stop     = Uri.EscapeDataString(stopDate.ToString("yyyy-MM-dd"));
      var step     = Uri.EscapeDataString(stepSize);

      var url = $"{HorizonsBase}?format=text&COMMAND='{cmd}'&MAKE_EPHEM=YES" +
                $"&EPHEM_TYPE=ELEMENTS&CENTER='{ctr}'" +
                $"&START_TIME='{start}'&STOP_TIME='{stop}'&STEP_SIZE='{step}'&OBJ_DATA=NO";

      _console.Log($"[HorizonJpl] EPA: {url}");
      var resp = await _httpClient.GetAsync(url);
      if (!resp.IsSuccessStatusCode)
      {
        await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status {(int)resp.StatusCode}", status: 3);
        return null;
      }

      var text   = await resp.Content.ReadAsStringAsync();
      _console.Log($"[HorizonJpl] EPA Response:\n{text}");
      int soeIdx = text.IndexOf("$$SOE");
      int eoeIdx = text.IndexOf("$$EOE");

      if (soeIdx == -1 || eoeIdx <= soeIdx)
      {
        // Log portion of the response for debugging
        _console.Log($"[HorizonJpl] EPA: $$SOE not found. Response snippet:\n{text.Substring(0, Math.Min(600, text.Length))}");
        await _breadcrumb.ShowMessageAsync("Horizon API", "No orbital elements returned.", status: 2);
        return null;
      }

      var epaBlock = text.Substring(soeIdx + 5, eoeIdx - (soeIdx + 5)).Trim();

      return new PlanetOrbitData
      {
        SemiMajorAxis = ParseValue(epaBlock, @"A\s*=\s*([^\s,]+)"),
        Eccentricity  = ParseValue(epaBlock, @"EC\s*=\s*([^\s,]+)"),
        Inclination   = ParseValue(epaBlock, @"IN\s*=\s*([^\s,]+)"),
        MeanAnomaly   = ParseValue(epaBlock, @"MA\s*=\s*([^\s,]+)"),
      };
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] FetchEpa error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
      return null;
    }
    finally { _breadcrumb.RemoveMessage(loadMsg); }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  5. FULL "Get Planet Data" (constants + EPA in sequence)
  //     Used by SpawnCometViewModel Step 3 "Fetch Orbit Data" button.
  // ═══════════════════════════════════════════════════════════════════════════

  public virtual async Task<PlanetOrbitData?> GetPlanetDataAsync(string targetId, string center, DateTime startDate, DateTime stopDate, string stepSize)
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
          var targetIdEncoded = Uri.EscapeDataString(targetId.EndsWith(";") ? targetId : targetId + ";");
          
          // 1. Fetch Object Constants
          var objUrl = $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text"
            + $"&COMMAND='{targetIdEncoded}'"
            + $"&OBJ_DATA='YES'"
            + $"&MAKE_EPHEM='NO'";

          _console.Log($"[HorizonJpl] GET Object Data: {objUrl}");
          var objResponse = await _httpClient.GetAsync(objUrl);
          if (!objResponse.IsSuccessStatusCode)
          {
            await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status: {(int)objResponse.StatusCode}");
            return null;
          }
          objDataText = await objResponse.Content.ReadAsStringAsync();
          _console.Log($"[HorizonJpl] Object Data Response:\n{objDataText}");

          // 2. Fetch EPA
          var epaUrl = $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text"
            + $"&COMMAND='{targetIdEncoded}'"
            + $"&MAKE_EPHEM='YES'"
            + $"&EPHEM_TYPE='ELEMENTS'"
            + $"&CENTER='{Uri.EscapeDataString(center)}'"
            + $"&START_TIME='{Uri.EscapeDataString(startDateStr)}'"
            + $"&STOP_TIME='{Uri.EscapeDataString(stopDateStr)}'"
            + $"&STEP_SIZE='{Uri.EscapeDataString(stepSize)}'"
            + $"&OBJ_DATA='NO'";

          _console.Log($"[HorizonJpl] GET EPA Data: {epaUrl}");
          var epaResponse = await _httpClient.GetAsync(epaUrl);
          if (!epaResponse.IsSuccessStatusCode)
          {
            await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status: {(int)epaResponse.StatusCode}");
            return null;
          }
          epaText = await epaResponse.Content.ReadAsStringAsync();
          _console.Log($"[HorizonJpl] EPA Response:\n{epaText}");

          await _storage.SaveSessionAsync(cacheKey, System.Text.Encoding.UTF8.GetBytes(epaText));
          await _storage.SaveSessionAsync(cacheKey + ".obj", System.Text.Encoding.UTF8.GetBytes(objDataText));
      }

      string constantsBlock = ExtractConstantsBlock(objDataText);

      double a = ParseValue(epaText, @"A\s*=\s*([^\s,]+)");
      double ec = ParseValue(epaText, @"EC\s*=\s*([^\s,]+)");
      double in_ = ParseValue(epaText, @"IN\s*=\s*([^\s,]+)");
      double ma = ParseValue(epaText, @"MA\s*=\s*([^\s,]+)");

      // Nucleus radius: try volumetric-equivalent radius first (R_vol, km),
      // then the generic RAD field. Default to 1 km if absent.
      double radiusKm = ParseValue(objDataText, @"R_vol\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0.0)
        radiusKm = ParseValue(objDataText, @"RAD\s*=\s*([^\s,+]+)");
      if (radiusKm <= 0.0)
        radiusKm = 1.0;

      double gm = ParseValue(objDataText, @"GM\s*=\s*([^\s,+]+)"); // in km^3/s^2 usually
      double massKg = 1e13; // default
      if (gm > 0.0)
      {
        // G in km^3 / (kg s^2) = 6.6743e-20
        massKg = gm / 6.6743e-20;
      }
      else
      {
        // Volume = 4/3 * pi * r^3 (in meters^3)
        // Average comet density = 600 kg/m^3
        double r_m = radiusKm * 1000.0;
        massKg = (4.0 / 3.0) * Math.PI * r_m * r_m * r_m * 600.0;
      }

      return new PlanetOrbitData
      {
        SemiMajorAxis = a,
        Eccentricity = ec,
        Inclination = in_,
        MeanAnomaly = ma,
        RawConstants = constantsBlock,
        CometRadiusKm = radiusKm,
        MassKg = massKg,
      };
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] GetPlanetDataAsync Exception: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Exception", ex.Message);
      return null;
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  // ═══════════════════════════════════════════════════════════════════════════
  //  6. SPK BINARY DOWNLOAD
  // ═══════════════════════════════════════════════════════════════════════════

  public async Task<string?> DownloadSpkByIdAsync(string pdes, string spkId, string savePath, string startTime, string stopTime)
  {
    try
    {
      var cmd   = Uri.EscapeDataString($"{spkId};");
      var start = Uri.EscapeDataString(startTime);
      var stop  = Uri.EscapeDataString(stopTime);
      var url   = $"{HorizonsBase}?format=text&COMMAND='{cmd}'&OBJ_DATA=NO&MAKE_EPHEM=YES&EPHEM_TYPE=SPK&START_TIME='{start}'&STOP_TIME='{stop}'";

      _console.Log($"[HorizonJpl] SPK download: {url}");
      var resp = await _httpClient.GetAsync(url);
      if (!resp.IsSuccessStatusCode) return null;

      var text     = await resp.Content.ReadAsStringAsync();
      _console.Log($"[HorizonJpl] SPK fetched ({text.Length} bytes).");
      int startIdx = text.IndexOf("REFGL1NQ");
      if (startIdx == -1) { _console.Log("[HorizonJpl] SPK binary marker not found."); return null; }

      var lines      = text.Substring(startIdx).Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
      var base64Data = string.Join("", lines.TakeWhile(l => !string.IsNullOrWhiteSpace(l)));
      var bytes      = Convert.FromBase64String(base64Data);

      using var fs = new FileStream(savePath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true);
      await fs.WriteAsync(bytes, 0, bytes.Length);
      return savePath;
    }
    catch (Exception ex) { _console.Log($"[HorizonJpl] SPK download error: {ex.Message}"); return null; }
  }

  public async Task<bool> DownloadObservationAsync(string pdes, string spkId, DateTimeOffset start, DateTimeOffset stop, string savePath)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Downloading observation package…");
    try
    {
      string startStr = start.ToString("yyyy-MM-dd");
      string stopStr  = stop.ToString("yyyy-MM-dd");

      var spkResult = await DownloadSpkByIdAsync(pdes, spkId, savePath, startStr, stopStr);
      if (spkResult == null) throw new Exception("SPK download failed.");

      await FetchObjectDataAsync(spkId);
      // EPA stored in ObjectData display; call FetchEpaAsync if needed separately

      await _breadcrumb.ShowMessageAsync("Horizon API", $"Observation saved to {savePath}", status: 1);
      return true;
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] DownloadObservation error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message, status: 3);
      return false;
    }
    finally { _breadcrumb.RemoveMessage(loadMsg); }
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
    if (m.Success && double.TryParse(m.Groups[1].Value,
          System.Globalization.NumberStyles.Any,
          System.Globalization.CultureInfo.InvariantCulture, out double v))
      return v;
    return 0.0;
  }

  private static string Sanitize(string s)
    => Regex.Replace(s, @"[^a-zA-Z0-9_\-]", "_");
}
