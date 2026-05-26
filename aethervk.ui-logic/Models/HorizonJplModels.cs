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
}

public class ObjectDataProperty
{
    public string Property { get; set; } = string.Empty;
    public string Value { get; set; } = string.Empty;
}
