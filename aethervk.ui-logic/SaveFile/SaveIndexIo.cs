using System;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace AetherVk.Logic.SaveFile;

/// <summary><c>[centuries, nanoseconds]</c> ⇄ <see cref="EpochParts"/> (nanoseconds stay exact u64).</summary>
public sealed class EpochPartsConverter : JsonConverter<EpochParts>
{
  public override EpochParts? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
  {
    if (reader.TokenType == JsonTokenType.Null)
      return null;
    if (reader.TokenType != JsonTokenType.StartArray)
      throw new JsonException("epoch: expected [centuries, nanoseconds]");
    reader.Read();
    short c = reader.GetInt16();
    reader.Read();
    ulong ns = reader.GetUInt64();
    reader.Read();
    if (reader.TokenType != JsonTokenType.EndArray)
      throw new JsonException("epoch: expected [centuries, nanoseconds]");
    return new EpochParts(c, ns);
  }

  public override void Write(Utf8JsonWriter writer, EpochParts value, JsonSerializerOptions options)
  {
    writer.WriteStartArray();
    writer.WriteNumberValue(value.Centuries);
    writer.WriteNumberValue(value.Nanoseconds);
    writer.WriteEndArray();
  }
}

/// <summary>Why <see cref="SaveIndexIo.Read"/> refused an index.</summary>
public enum SaveIndexError
{
  None,
  Missing,
  Unreadable,
  NewerFormat,
}

/// <summary>Reading (with the version gate and migrations) and atomic writing of the index.</summary>
public static class SaveIndexIo
{
  public static string IndexPath(string saveDir) => Path.Combine(saveDir, SaveIndex.FileName);

  public static string ToJson(SaveIndex index) =>
    JsonSerializer.Serialize(index, SaveIndexJsonContext.Default.SaveIndex);

  /// <summary>
  /// Reads <c>&lt;saveDir&gt;/aethervk_index.json</c>. A newer format is refused; older ones go
  /// through <see cref="Migrate"/>.
  /// </summary>
  public static (SaveIndex? Index, SaveIndexError Error) Read(string saveDir)
  {
    var path = IndexPath(saveDir);
    if (!File.Exists(path))
      return (null, SaveIndexError.Missing);
    try
    {
      var json = File.ReadAllText(path, Encoding.UTF8);
      using var doc = JsonDocument.Parse(json);
      int version = doc.RootElement.TryGetProperty("format_version", out var v) && v.TryGetInt32(out var n) ? n : 0;
      if (version > SaveIndex.CurrentFormatVersion)
        return (null, SaveIndexError.NewerFormat);
      if (version < 1)
        return (null, SaveIndexError.Unreadable);
      var index = JsonSerializer.Deserialize(Migrate(json, version), SaveIndexJsonContext.Default.SaveIndex);
      return index is null ? (null, SaveIndexError.Unreadable) : (index, SaveIndexError.None);
    }
    catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException or InvalidOperationException)
    {
      return (null, SaveIndexError.Unreadable);
    }
  }

  /// <summary>The migration chain vN → vN+1 → … → current, on the raw JSON. Format 1 is the first.</summary>
  public static string Migrate(string json, int fromVersion)
  {
    // no older formats yet: when format 2 appears, add `if (fromVersion == 1) { json = V1ToV2(json); fromVersion = 2; }`
    return json;
  }

  /// <summary>
  /// Writes the index atomically: to <c>aethervk_index.json.tmp</c>, flushed to disk, then renamed
  /// over the index. A crash leaves the previous index (and at most a stale .tmp, overwritten
  /// next time).
  /// </summary>
  public static void Write(string saveDir, SaveIndex index)
  {
    Directory.CreateDirectory(saveDir);
    var path = IndexPath(saveDir);
    var tmp = path + ".tmp";
    using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None))
    {
      var bytes = Encoding.UTF8.GetBytes(ToJson(index));
      fs.Write(bytes, 0, bytes.Length);
      fs.Flush(true);
    }
    if (File.Exists(path))
      File.Replace(tmp, path, null);
    else
      File.Move(tmp, path);
  }

  /// <summary>64-bit FNV-1a of the UTF-8 bytes, 16 lowercase hex digits (the state's <c>compat_hash</c>).</summary>
  public static string CompatHash(string compatibilityJson)
  {
    const ulong offset = 0xcbf29ce484222325, prime = 0x100000001b3;
    ulong h = offset;
    foreach (byte b in Encoding.UTF8.GetBytes(compatibilityJson))
    {
      h ^= b;
      h = unchecked(h * prime);
    }
    return h.ToString("x16");
  }
}
