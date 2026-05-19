using System;
using System.IO;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Models;

public partial class SpkFileModel : ObservableObject
{
  [ObservableProperty]
  private string _filePath = string.Empty;

  [ObservableProperty]
  private string _fileName = string.Empty;

  [ObservableProperty]
  private string _displayName = string.Empty;

  [ObservableProperty]
  private bool _isLoaded;

  [ObservableProperty]
  private bool _isSelected;

  [ObservableProperty]
  private bool _isSelectable = true;

  public SpkFileModel(string filePath)
  {
    FilePath = filePath;
    FileName = Path.GetFileName(filePath);
    DisplayName = ParseDisplayName(FileName);
  }

  private string ParseDisplayName(string fileName)
  {
    // Expected format: {pdes}-{spkId}-{startTime}-{stopTime}.spk
    // Example: 1P_Halley-90000033-2024-01-01-2024-01-31.spk
    try
    {
      var nameWithoutExt = Path.GetFileNameWithoutExtension(fileName);
      var parts = nameWithoutExt.Split('-');
      
      // We expect at least pdes, spkId, and dates
      if (parts.Length >= 4)
      {
        var pdes = parts[0].Replace("_", " ");
        var spkId = parts[1];
        
        // The dates might be split further if they contain hyphens, but they are already formatted as yyyy-MM-dd
        // Actually the format is {pdes}-{spkId}-{yyyy}-{MM}-{dd}-{yyyy}-{MM}-{dd}
        // Let's just find the first two parts and join the rest as the date range
        if (parts.Length >= 8) // pdes, spkId, yyyy, mm, dd, yyyy, mm, dd
        {
           var start = $"{parts[2]}-{parts[3]}-{parts[4]}";
           var end = $"{parts[5]}-{parts[6]}-{parts[7]}";
           return $"{pdes} (ID: {spkId}) | {start} to {end}";
        }
        else
        {
          // Fallback if dates weren't formatted with hyphens or something else
          var dates = string.Join("-", parts, 2, parts.Length - 2);
          return $"{pdes} (ID: {spkId}) | {dates}";
        }
      }
    }
    catch
    {
      // Ignore parsing errors and fallback
    }

    return fileName;
  }
}
