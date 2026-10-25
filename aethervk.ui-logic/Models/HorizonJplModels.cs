using System;

namespace AetherVk.Logic.Models;

public class CometSearchResult
{
  public string Name { get; set; } = string.Empty;
  public string PrimaryDesignation { get; set; } = string.Empty;
}

public class SpkRecordItem
{
  public string RecordId { get; set; } = string.Empty;
  public string EpochYear { get; set; } = string.Empty;
  public string MatchDesig { get; set; } = string.Empty;
  public string PrimaryDesig { get; set; } = string.Empty;
  public string Name { get; set; } = string.Empty;

  /// <summary>
  /// True when RecordId is a parseable positive integer — i.e. a real SPK record,
  /// not the SBDB placeholder row "(9 match — enter record # …)" — or the designation
  /// command of a single-record comet ("DES=2025 A6").
  /// </summary>
  public bool IsValid =>
    (int.TryParse(RecordId, out int id) && id > 0)
    || (RecordId.StartsWith("DES=", StringComparison.Ordinal) && RecordId.Length > 4);
}

public class ObjectDataProperty
{
  public string Property { get; set; } = string.Empty;
  public string Value { get; set; } = string.Empty;
}
