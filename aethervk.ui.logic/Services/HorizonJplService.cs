using System;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;

namespace AetherVk.Logic.Services;

public class HorizonJplService
{
  private readonly HttpClient _httpClient;
  private readonly ConsoleService _console;
  private readonly BreadcrumbService _breadcrumb;

  public ObservableCollection<string[]> SessionData { get; } = new();
  public ObservableCollection<string> Headers { get; } = new();

  public ObservableCollection<string[]> CometsData { get; } = new();
  public ObservableCollection<string> CometsHeaders { get; } = new();

  public HorizonJplService(ConsoleService console, BreadcrumbService breadcrumb)
  {
    _httpClient = new HttpClient { Timeout = TimeSpan.FromSeconds(10) };
    _console = console;
    _breadcrumb = breadcrumb;
  }

  public async Task FetchCometsAsync(string startTime, string stopTime)
  {
    try
    {
      var url =
        $"https://ssd-api.jpl.nasa.gov/sbdb_query.api?sb-kind=c&fields=full_name,pdes&sb-cdata=%7B%22AND%22%3A%5B%22first_obs%7CGE%7C{Uri.EscapeDataString(startTime)}%22%2C%22first_obs%7CLE%7C{Uri.EscapeDataString(stopTime)}%22%5D%7D";

      _console.Log($"[HorizonJpl] GET {url}");

      var response = await _httpClient.GetAsync(url);

      _console.Log(
        $"[HorizonJpl] Response Status: {(int)response.StatusCode} {response.ReasonPhrase}"
      );

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
        await _breadcrumb.ShowMessageAsync(
          "Horizon API Error",
          $"Status: {(int)response.StatusCode}"
        );
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] Exception: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Exception", ex.Message);
    }
  }

  public void ParseCometsJson(string json)
  {
    CometsData.Clear();
    CometsHeaders.Clear();

    try
    {
      using var doc = System.Text.Json.JsonDocument.Parse(json);
      var root = doc.RootElement;

      if (root.TryGetProperty("fields", out var fields))
      {
        foreach (var field in fields.EnumerateArray())
        {
          CometsHeaders.Add(field.GetString() ?? "");
        }
      }

      if (root.TryGetProperty("data", out var data))
      {
        foreach (var row in data.EnumerateArray())
        {
          var rowData = new string[CometsHeaders.Count];
          int i = 0;
          foreach (var item in row.EnumerateArray())
          {
            if (i < rowData.Length)
            {
              rowData[i] =
                item.ValueKind == System.Text.Json.JsonValueKind.Null ? "" : item.ToString();
              i++;
            }
          }
          CometsData.Add(rowData);
        }
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] JSON Parse Exception: {ex.Message}");
    }
  }

  public async Task FetchDataAsync(
    string command,
    string startTime,
    string stopTime,
    string stepSize,
    string center,
    System.Threading.CancellationToken cancellationToken = default
  )
  {
    try
    {
      using var request = new HttpRequestMessage(HttpMethod.Get, 
        $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='{Uri.EscapeDataString(command)}'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='OBSERVER'&CENTER='{Uri.EscapeDataString(center)}'&START_TIME='{Uri.EscapeDataString(startTime)}'&STOP_TIME='{Uri.EscapeDataString(stopTime)}'&STEP_SIZE='{Uri.EscapeDataString(stepSize)}'&CSV_FORMAT='YES'"
      );

      _console.Log($"[HorizonJpl] GET {request.RequestUri}");

      // Send the request. Using ResponseContentRead to avoid hanging on stream parsing of chunked encoding
      using var cts = System.Threading.CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
      cts.CancelAfter(TimeSpan.FromSeconds(5)); // Force 5s timeout on stream
      var response = await _httpClient.SendAsync(request, HttpCompletionOption.ResponseContentRead, cts.Token);

      _console.Log(
        $"[HorizonJpl] Response Status: {(int)response.StatusCode} {response.ReasonPhrase}"
      );
      foreach (var header in response.Headers)
      {
        _console.Log($"[HorizonJpl] Header: {header.Key} = {string.Join(", ", header.Value)}");
      }

      if (response.IsSuccessStatusCode)
      {
        var text = await response.Content.ReadAsStringAsync();
        ParseText(text);
        await _breadcrumb.ShowMessageAsync(
          "Horizon API",
          $"Success: {SessionData.Count} rows fetched."
        );
      }
      else
      {
        await _breadcrumb.ShowMessageAsync(
          "Horizon API Error",
          $"Status: {(int)response.StatusCode}"
        );
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] Exception: {ex.Message}");
      await _breadcrumb.ShowMessageAsync("Horizon API Exception", ex.Message);
    }
  }

  public ObservableCollection<string[]> SpkRecordsData { get; } = new();
  public ObservableCollection<string> SpkRecordsHeaders { get; } = new();

  public async Task FetchSpkRecordsAsync(string pdes, string startTime, string stopTime)
  {
      try
      {
          var url = $"https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='DES={Uri.EscapeDataString(pdes)};'&EPHEM_TYPE='SPK'&START_TIME='{Uri.EscapeDataString(startTime)}'&STOP_TIME='{Uri.EscapeDataString(stopTime)}'&MAKE_EPHEM='YES'";
          
          _console.Log($"[HorizonJpl] GET SPK Records: {url}");
          var response = await _httpClient.GetAsync(url);
          
          if (response.IsSuccessStatusCode)
          {
              var json = await response.Content.ReadAsStringAsync();
              ParseSpkRecordsJson(json);
              await _breadcrumb.ShowMessageAsync("Horizon API (SPK Records)", $"Success: {SpkRecordsData.Count} records found.");
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
  }

  private void ParseSpkRecordsJson(string json)
  {
      SpkRecordsData.Clear();
      SpkRecordsHeaders.Clear();
      
      // Simple headers for now
      SpkRecordsHeaders.Add("Record ID");
      SpkRecordsHeaders.Add("Start Time");
      SpkRecordsHeaders.Add("Stop Time");
      
      try
      {
          using var doc = System.Text.Json.JsonDocument.Parse(json);
          var root = doc.RootElement;
          
          if (root.TryGetProperty("spk_records", out var records) && records.ValueKind == System.Text.Json.JsonValueKind.Array)
          {
              foreach (var record in records.EnumerateArray())
              {
                  var id = record.GetProperty("id").GetString() ?? "";
                  var start = record.GetProperty("start").GetString() ?? "";
                  var stop = record.GetProperty("stop").GetString() ?? "";
                  SpkRecordsData.Add(new[] { id, start, stop });
              }
          }
      }
      catch (Exception ex)
      {
          _console.Log($"[HorizonJpl] SPK JSON Parse Exception: {ex.Message}");
      }
  }

  public async Task<string?> DownloadSpkByIdAsync(string pdes, string spkId, string savePath)
  {
      try
      {
          var url = "https://ssd.jpl.nasa.gov/api/horizons.api";
          var query = $"?format=json&COMMAND='DES={Uri.EscapeDataString(pdes)};'&OBJ_DATA=NO&MAKE_EPHEM=YES&EPHEM_TYPE=SPK&SPK_ID='{Uri.EscapeDataString(spkId)}'";
          
          _console.Log($"[HorizonJpl] Requesting SPK: {url}{query}");
          var response = await _httpClient.GetAsync(url + query);
          
          if (!response.IsSuccessStatusCode) return null;
          
          var json = await response.Content.ReadAsStringAsync();
          using var doc = System.Text.Json.JsonDocument.Parse(json);
          var root = doc.RootElement;
          
          if (root.TryGetProperty("spk", out var spkElement))
          {
              var base64Spk = spkElement.GetString();
              if (string.IsNullOrEmpty(base64Spk)) return null;
              
              var binarySpk = Convert.FromBase64String(base64Spk);
              using (var fs = new FileStream(savePath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true))
              {
                  await fs.WriteAsync(binarySpk, 0, binarySpk.Length);
              }
              return savePath;
          }
          return null;
      }
      catch (Exception ex)
      {
          _console.Log($"[HorizonJpl] SPK Download Exception: {ex.Message}");
          return null;
      }
  }

  public void ParseText(string text)
  {
    SessionData.Clear();
    Headers.Clear();
    var lines = text.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
    bool inEphem = false;

    for (int i = 0; i < lines.Length; i++)
    {
      var line = lines[i];
      if (line.StartsWith("$$SOE"))
      {
        // The header line is typically 2 lines above $$SOE
        if (i >= 2)
        {
          var headerLine = lines[i - 2];
          var headers = headerLine.Split(',').Select(h => h.Trim()).ToList();

          // The last column is usually empty in JPL CSV, remove it if so
          if (headers.Count > 0 && string.IsNullOrWhiteSpace(headers.Last()))
          {
            headers.RemoveAt(headers.Count - 1);
          }

          foreach (var h in headers)
          {
            Headers.Add(string.IsNullOrWhiteSpace(h) ? "Col" : h);
          }
        }

        inEphem = true;
        continue;
      }
      if (line.StartsWith("$$EOE"))
      {
        inEphem = false;
        break;
      }

      if (inEphem)
      {
        var parts = line.Split(',').Select(p => p.Trim()).ToArray();
        if (parts.Length > 0)
        {
          // Filter parts to match header count just in case
          var rowData = parts.Take(Headers.Count).ToArray();
          if (rowData.Length < Headers.Count)
          {
            var padded = new string[Headers.Count];
            Array.Copy(rowData, padded, rowData.Length);
            for (int p = rowData.Length; p < padded.Length; p++)
              padded[p] = "";
            rowData = padded;
          }
          SessionData.Add(rowData);
        }
      }
    }
  }

  public async Task<string?> DownloadSpkAsync(string spkid, string startTime, string stopTime)
  {
    try
    {
      var cacheDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "cache", "spk");
      if (!Directory.Exists(cacheDir))
      {
        Directory.CreateDirectory(cacheDir);
      }

      var fileName = $"{spkid}_{startTime}_{stopTime}.bsp".Replace(" ", "_").Replace(":", "-");
      var filePath = Path.Combine(cacheDir, fileName);

      if (File.Exists(filePath))
      {
        _console.Log($"[HorizonJpl] SPK for {spkid} already cached at {filePath}");
        return filePath;
      }

      var url = "https://ssd.jpl.nasa.gov/api/horizons.api";
      var query = $"?format=json&COMMAND='{Uri.EscapeDataString(spkid)}'&OBJ_DATA=NO&MAKE_EPHEM=YES&EPHEM_TYPE=SPK&START_TIME='{Uri.EscapeDataString(startTime)}'&STOP_TIME='{Uri.EscapeDataString(stopTime)}'";

      _console.Log($"[HorizonJpl] Requesting SPK: {url}{query}");

      var response = await _httpClient.GetAsync(url + query);
      if (!response.IsSuccessStatusCode)
      {
        _console.Log($"[HorizonJpl] SPK Download failed: {response.StatusCode}");
        return null;
      }

      var json = await response.Content.ReadAsStringAsync();
      using var doc = System.Text.Json.JsonDocument.Parse(json);
      var root = doc.RootElement;

      if (root.TryGetProperty("spk", out var spkElement))
      {
        var base64Spk = spkElement.GetString();
        if (string.IsNullOrEmpty(base64Spk))
        {
          _console.Log("[HorizonJpl] SPK field is empty in response.");
          return null;
        }

        var binarySpk = Convert.FromBase64String(base64Spk);
        using (var fs = new FileStream(filePath, FileMode.Create, FileAccess.Write, FileShare.None, 4096, true))
        {
            await fs.WriteAsync(binarySpk, 0, binarySpk.Length);
        }
        _console.Log($"[HorizonJpl] Successfully saved SPK to {filePath} ({binarySpk.Length / 1024.0:F2} KB)");
        return filePath;
      }
      else
      {
        var error = root.TryGetProperty("error", out var err) ? err.GetString() : "Unknown error";
        _console.Log($"[HorizonJpl] Failed to generate SPK. API Error: {error}");
        return null;
      }
    }
    catch (Exception ex)
    {
      _console.Log($"[HorizonJpl] SPK Download Exception: {ex.Message}");
      return null;
    }
  }
}
