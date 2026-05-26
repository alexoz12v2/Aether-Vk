using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using AetherVk.Logic.Models;

namespace AetherVk.Logic.Services;

public class PlanetOrbitData
{
  public double SemiMajorAxis { get; set; }
  public double Eccentricity { get; set; }
  public double Inclination { get; set; }
  public double MeanAnomaly { get; set; }
  public string RawConstants { get; set; } = string.Empty;
  public double CometRadiusKm { get; set; } = 1.0;
}

public class HorizonJplService
{
  private readonly HttpClient _httpClient;
  private readonly ConsoleService _console;
  private readonly BreadcrumbService _breadcrumb;

  public ObservableCollection<CometSearchResult> CometsData { get; } = new();
  public ObservableCollection<SpkRecordItem> SpkRecordsData { get; } = new();
  public ObservableCollection<ObjectDataProperty> ObjectData { get; } = new();
  public ObservableCollection<string[]> SessionData { get; } = new(); // Used for CSV EPA, if needed

  public HorizonJplService(ConsoleService console, BreadcrumbService breadcrumb)
  {
    _httpClient = new HttpClient { Timeout = TimeSpan.FromMinutes(2) }; // Downloading SPK might take time
    _httpClient.DefaultRequestHeaders.Add("User-Agent", "AetherVk/1.0");
    _console = console;
    _breadcrumb = breadcrumb;
  }

  // Comets Search (Using JSON SBDB API since it works best for comet lookup)
  public async Task FetchCometsAsync()
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Downloading list of comets...");
    try
    {
      var url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api?sb-kind=c&fields=full_name,pdes";
      _console.Log($"[HorizonJpl] GET {url}");

      var response = await _httpClient.GetAsync(url);
      if (response.IsSuccessStatusCode)
      {
        var text = await response.Content.ReadAsStringAsync();
        ParseCometsJson(text);
        await _breadcrumb.ShowMessageAsync(
          "Horizon API (Comets)",
          $"Success: {CometsData.Count} comets fetched."
        );
      }
      else
      {
        await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status: {(int)response.StatusCode}");
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] Exception: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Exception", ex.Message);
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
      var root = doc.RootElement;

      if (root.TryGetProperty("data", out var data))
      {
        foreach (var row in data.EnumerateArray())
        {
          var rowData = row.EnumerateArray().ToArray();
          if (rowData.Length >= 2)
          {
            CometsData.Add(new CometSearchResult
            {
              Name = rowData[0].ValueKind == System.Text.Json.JsonValueKind.Null ? "" : rowData[0].ToString(),
              PrimaryDesignation = rowData[1].ValueKind == System.Text.Json.JsonValueKind.Null ? "" : rowData[1].ToString()
            });
          }
        }
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] JSON Parse Exception: {ex.Message}");
    }
  }

  // Fetch SPK Records in text format
  public async Task FetchSpkRecordsAsync(string pdes, string startTime, string stopTime)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Downloading list of observation records...");
    try
    {
      // Semicolon url-encoded as %3B for Horizons API when looking up small bodies
      var pdesEncoded = Uri.EscapeDataString($"{pdes};");
      var start = Uri.EscapeDataString(startTime);
      var stop = Uri.EscapeDataString(stopTime);

      var url = $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='DES%3D{pdesEncoded}'&EPHEM_TYPE=SPK&START_TIME='{start}'&STOP_TIME='{stop}'&MAKE_EPHEM=YES";

      _console.Log($"[HorizonJpl] GET SPK Records: {url}");
      var response = await _httpClient.GetAsync(url);

      if (response.IsSuccessStatusCode)
      {
        var text = await response.Content.ReadAsStringAsync();
        ParseSpkRecordsText(text, startTime, stopTime);
        await _breadcrumb.ShowMessageAsync(
          "Horizon API (SPK Records)",
          $"Success: {SpkRecordsData.Count} records found."
        );
      }
      else
      {
        await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status: {(int)response.StatusCode}");
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] SPK Records Exception: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Exception", ex.Message);
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  private void ParseSpkRecordsText(string text, string startTime, string stopTime)
  {
    SpkRecordsData.Clear();
    
    int startYear = int.MinValue;
    int stopYear = int.MaxValue;
    if (DateTime.TryParse(startTime, out var dtStart)) startYear = dtStart.Year;
    if (DateTime.TryParse(stopTime, out var dtStop)) stopYear = dtStop.Year;

    var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    bool inData = false;

    foreach (var line in lines)
    {
      if (line.Trim().StartsWith("--------"))
      {
        if (inData) break; // Reached the bottom of the table
        inData = true;
        continue;
      }
      
      if (inData)
      {
        if (line.Trim().StartsWith("*") || string.IsNullOrWhiteSpace(line)) break;
        
        var parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length >= 5)
        {
          var id = parts[0];
          var epochStr = parts[1];
          var match = parts[2];
          var primary = parts[3];
          var name = string.Join(" ", parts.Skip(4));

          if (int.TryParse(epochStr, out int epochYear))
          {
            if (epochYear >= startYear && epochYear <= stopYear)
            {
              SpkRecordsData.Add(new SpkRecordItem { RecordId = id, EpochYear = epochStr, MatchDesig = match, PrimaryDesig = primary, Name = name });
            }
          }
          else
          {
            SpkRecordsData.Add(new SpkRecordItem { RecordId = id, EpochYear = epochStr, MatchDesig = match, PrimaryDesig = primary, Name = name });
          }
        }
      }
    }
  }

  // Fetch Object Constants in text format
  public async Task FetchObjectDataAsync(string spkId)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Downloading object data...");
    try
    {
      var spkIdEncoded = Uri.EscapeDataString($"{spkId};");
      var url = $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='{spkIdEncoded}'&OBJ_DATA=YES&MAKE_EPHEM=NO";
      
      _console.Log($"[HorizonJpl] GET Object Data: {url}");
      var response = await _httpClient.GetAsync(url);
      
      if (response.IsSuccessStatusCode)
      {
        var text = await response.Content.ReadAsStringAsync();
        ParseObjectDataText(text);
      }
      else
      {
        await _breadcrumb.ShowMessageAsync("Horizon API Error", $"Status: {(int)response.StatusCode}");
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] Object Data Exception: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Exception", ex.Message);
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  private void ParseObjectDataText(string text)
  {
    ObjectData.Clear();
    var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    foreach (var line in lines)
    {
      if (line.StartsWith("*") || string.IsNullOrWhiteSpace(line)) continue;
      
      var normalizedLine = Regex.Replace(line, @"([=:])\s+", "$1");
      var parts = normalizedLine.Split(new[] { "  ", "\t" }, StringSplitOptions.RemoveEmptyEntries);
      bool hasKeyValue = false;
      
      foreach (var part in parts)
      {
        var p = part.Trim();
        if (string.IsNullOrEmpty(p)) continue;

        if (p.Contains("=") || p.Contains(":"))
        {
          var sep = p.Contains("=") ? '=' : ':';
          var kv = p.Split(new[] { sep }, 2, StringSplitOptions.RemoveEmptyEntries);
          if (kv.Length == 2)
          {
            ObjectData.Add(new ObjectDataProperty { Property = kv[0].Trim(), Value = kv[1].Trim() });
            hasKeyValue = true;
          }
          else
          {
            ObjectData.Add(new ObjectDataProperty { Property = "Info", Value = p });
          }
        }
        else if (!hasKeyValue)
        {
          ObjectData.Add(new ObjectDataProperty { Property = "Info", Value = p });
        }
      }
    }
  }

  // Download SPK file directly from text format by extracting base64 string
  public async Task<string?> DownloadSpkByIdAsync(string pdes, string spkId, string savePath, string startTime, string stopTime)
  {
    try
    {
      var spkIdEncoded = Uri.EscapeDataString($"{spkId};");
      var start = Uri.EscapeDataString(startTime);
      var stop = Uri.EscapeDataString(stopTime);

      var url = $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='{spkIdEncoded}'&OBJ_DATA=NO&MAKE_EPHEM=YES&EPHEM_TYPE=SPK&START_TIME='{start}'&STOP_TIME='{stop}'";

      _console.Log($"[HorizonJpl] Requesting SPK: {url}");
      var response = await _httpClient.GetAsync(url);

      if (!response.IsSuccessStatusCode)
        return null;

      var text = await response.Content.ReadAsStringAsync();
      
      // Look for the REFGL1NQ sequence
      int startIdx = text.IndexOf("REFGL1NQ");
      if (startIdx == -1)
      {
        _console.Log("[HorizonJpl] SPK binary data not found in response.");
        return null;
      }

      // The base64 text is from startIdx to the end of the text (or an empty line)
      var lines = text.Substring(startIdx).Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
      
      var base64Spk = string.Join("", lines.TakeWhile(l => !string.IsNullOrWhiteSpace(l)));
      
      var binarySpk = Convert.FromBase64String(base64Spk);
      using (var fs = new FileStream(savePath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true))
      {
        await fs.WriteAsync(binarySpk, 0, binarySpk.Length);
      }
      return savePath;
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] SPK Download Exception: {ex.Message}");
      return null;
    }
  }

  // Added: Single Observation Download Method
  public async Task<bool> DownloadObservationAsync(string pdes, string spkId, DateTimeOffset start, DateTimeOffset stop, string savePath)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Downloading complete observation package...");
    try
    {
      string startStr = start.ToString("yyyy-MM-dd");
      string stopStr = stop.ToString("yyyy-MM-dd");

      // 1. Download SPK
      _console.Log("[HorizonJpl] Downloading SPK...");
      var spkResult = await DownloadSpkByIdAsync(pdes, spkId, savePath, startStr, stopStr);
      if (spkResult == null) throw new Exception("Failed to download SPK.");

      // 2. Download Object Data
      _console.Log("[HorizonJpl] Fetching Object Data...");
      await FetchObjectDataAsync(spkId);

      // 3. Fetch EPA (Orbital Elements)
      _console.Log("[HorizonJpl] Fetching Orbital Elements (EPA)...");
      await FetchEpaAsync(spkId, startStr, stopStr);

      await _breadcrumb.ShowMessageAsync("Horizon API", $"Successfully downloaded observation to {savePath}");
      return true;
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] Observation Download Error: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Error", ex.Message);
      return false;
    }
    finally
    {
      _breadcrumb.RemoveMessage(loadMsg);
    }
  }

  // Fetch EPA using ELEMENTS text format
  public async Task FetchEpaAsync(string spkId, string startTime, string stopTime)
  {
    try
    {
      var spkIdEncoded = Uri.EscapeDataString($"{spkId};");
      var start = Uri.EscapeDataString(startTime);
      var stop = Uri.EscapeDataString(stopTime);

      var url = $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='{spkIdEncoded}'&MAKE_EPHEM=YES&EPHEM_TYPE=ELEMENTS&CENTER='500@0'&START_TIME='{start}'&STOP_TIME='{stop}'&STEP_SIZE='1%20d'&OBJ_DATA=NO";

      var response = await _httpClient.GetAsync(url);
      if (response.IsSuccessStatusCode)
      {
        var text = await response.Content.ReadAsStringAsync();
        
        int soeIdx = text.IndexOf("$$SOE");
        int eoeIdx = text.IndexOf("$$EOE");
        if (soeIdx != -1 && eoeIdx != -1 && eoeIdx > soeIdx)
        {
          var epaBlock = text.Substring(soeIdx + 5, eoeIdx - (soeIdx + 5)).Trim();
          _console.Log($"[HorizonJpl] EPA Block fetched successfully ({epaBlock.Length} chars)");
          // The EPA block can now be used for initialization of the object, e.g. saving it somewhere.
          // Since the objective is just to verify it works, we just log it for now.
        }
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] EPA Fetch Error: {ex.Message}");
    }
  }

  public async Task<PlanetOrbitData?> GetPlanetDataAsync(string targetId, DateTime targetDate)
  {
    var loadMsg = _breadcrumb.ShowLoadingMessage("Horizon API", "Fetching orbital data...");
    try
    {
      string dateStr = targetDate.ToString("yyyy-MM-dd");
      string nextDateStr = targetDate.AddDays(1).ToString("yyyy-MM-dd");

      var url =
        $"https://ssd.jpl.nasa.gov/api/horizons.api?format=json"
        + $"&COMMAND='{Uri.EscapeDataString(targetId)}'"
        + $"&OBJ_DATA='YES'"
        + $"&MAKE_EPHEM='YES'"
        + $"&EPHEM_TYPE='ELEMENTS'"
        + $"&CENTER='@10'"
        + $"&START_TIME='{Uri.EscapeDataString(dateStr)}'"
        + $"&STOP_TIME='{Uri.EscapeDataString(nextDateStr)}'"
        + $"&STEP_SIZE='1 d'";

      _console.Log($"[HorizonJpl] GET Planet Data: {url}");

      // We can't use GetFromJsonAsync cleanly in some older netstandard setups without Microsoft.Net.Http.Json
      // Let's use standard GetAsync + deserialization to be safe in .NET Standard 2.0.
      var response = await _httpClient.GetAsync(url);
      if (!response.IsSuccessStatusCode)
      {
        await _breadcrumb.ShowMessageAsync(
          "Horizon API Error",
          $"Status: {(int)response.StatusCode}"
        );
        return null;
      }

      var json = await response.Content.ReadAsStringAsync();
      using var doc = System.Text.Json.JsonDocument.Parse(json);

      if (doc.RootElement.TryGetProperty("result", out var resultElement))
      {
        string rawText = resultElement.GetString() ?? "";

        string constantsBlock = ExtractConstantsBlock(rawText);

        double a = ParseValue(rawText, @"A\s*=\s*([^\s]+)");
        double ec = ParseValue(rawText, @"EC\s*=\s*([^\s]+)");
        double in_ = ParseValue(rawText, @"IN\s*=\s*([^\s]+)");
        double ma = ParseValue(rawText, @"MA\s*=\s*([^\s]+)");

        // Nucleus radius: try volumetric-equivalent radius first (R_vol, km),
        // then the generic RAD field. Default to 1 km if absent.
        double radiusKm = ParseValue(rawText, @"R_vol\s*=\s*([^\s,+]+)");
        if (radiusKm <= 0.0)
          radiusKm = ParseValue(rawText, @"RAD\s*=\s*([^\s,+]+)");
        if (radiusKm <= 0.0)
          radiusKm = 1.0;

        return new PlanetOrbitData
        {
          SemiMajorAxis = a,
          Eccentricity = ec,
          Inclination = in_,
          MeanAnomaly = ma,
          RawConstants = constantsBlock,
          CometRadiusKm = radiusKm,
        };
      }
      return null;
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

  private static string ExtractConstantsBlock(string text)
  {
    int startIdx = text.IndexOf("PHYSICAL PROPERTIES");
    int endIdx = text.IndexOf("$$SOE");

    if (startIdx != -1 && endIdx != -1 && endIdx > startIdx)
    {
      return text.Substring(startIdx, endIdx - startIdx).Trim();
    }
    return "Constants block not found.";
  }

  private static double ParseValue(string text, string pattern)
  {
      var match = Regex.Match(text, pattern, RegexOptions.IgnoreCase);
      if (match.Success && double.TryParse(match.Groups[1].Value, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out double value))
      {
          return value;
      }
      return 0.0;
  }
}
