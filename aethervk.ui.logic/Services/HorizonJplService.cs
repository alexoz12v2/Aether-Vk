using System;
using System.Collections.ObjectModel;
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
    _httpClient = new HttpClient();
    _console = console;
    _breadcrumb = breadcrumb;
  }

  public async Task FetchCometsAsync(string startTime, string stopTime)
  {
    try
    {
      var url =
        $"https://ssd-api.jpl.nasa.gov/sbdb_query.api?sb-kind=c&fields=full_name,first_obs,soln_date,spkid&sb-cdata=%7B%22AND%22%3A%5B%22first_obs%7CGE%7C{Uri.EscapeDataString(startTime)}%22%2C%22first_obs%7CLE%7C{Uri.EscapeDataString(stopTime)}%22%5D%7D";

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
    string center
  )
  {
    try
    {
      var url =
        $"https://ssd.jpl.nasa.gov/api/horizons.api?format=text&COMMAND='{Uri.EscapeDataString(command)}'&OBJ_DATA='YES'&MAKE_EPHEM='YES'&EPHEM_TYPE='OBSERVER'&CENTER='{Uri.EscapeDataString(center)}'&START_TIME='{Uri.EscapeDataString(startTime)}'&STOP_TIME='{Uri.EscapeDataString(stopTime)}'&STEP_SIZE='{Uri.EscapeDataString(stepSize)}'&CSV_FORMAT='YES'";

      _console.Log($"[HorizonJpl] GET {url}");

      var response = await _httpClient.GetAsync(url);

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
}
