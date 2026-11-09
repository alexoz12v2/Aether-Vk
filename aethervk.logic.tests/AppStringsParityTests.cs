using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Resources;
using System.Xml.Linq;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>
/// The English and Italian string tables stay in step: every key of AppStrings.resx has a
/// non-empty Italian value in AppStrings.it.resx (and the other way round), no key is declared
/// twice, and the embedded resources resolve every key in both cultures at run time.
/// </summary>
public class AppStringsParityTests
{
  private static string ResourcePath(string file)
  {
    for (var d = new DirectoryInfo(AppContext.BaseDirectory); d is not null; d = d.Parent)
    {
      var p = Path.Combine(d.FullName, "aethervk.ui-logic", "Resources", file);
      if (File.Exists(p)) return p;
    }
    throw new FileNotFoundException($"aethervk.ui-logic/Resources/{file}");
  }

  private static List<(string Name, string Value)> Entries(string file) =>
    XDocument.Load(ResourcePath(file)).Root!.Elements("data")
      .Select(d => ((string?)d.Attribute("name") ?? "", d.Element("value")?.Value ?? ""))
      .ToList();

  [Theory]
  [InlineData("AppStrings.resx")]
  [InlineData("AppStrings.it.resx")]
  public void NoKeyIsDeclaredTwice(string file)
  {
    var dupes = Entries(file).GroupBy(e => e.Name).Where(g => g.Count() > 1).Select(g => g.Key).ToList();
    Assert.True(dupes.Count == 0, $"{file}: duplicate keys {string.Join(", ", dupes)}");
  }

  [Fact]
  public void EveryEnglishKeyHasANonEmptyItalianValue()
  {
    var en = Entries("AppStrings.resx");
    var it = Entries("AppStrings.it.resx").GroupBy(e => e.Name).ToDictionary(g => g.Key, g => g.First().Value);
    Assert.NotEmpty(en);
    var missing = en.Where(e => !it.ContainsKey(e.Name)).Select(e => e.Name).ToList();
    var empty = en.Where(e => it.TryGetValue(e.Name, out var v) && string.IsNullOrWhiteSpace(v)).Select(e => e.Name).ToList();
    Assert.True(missing.Count == 0, $"missing in AppStrings.it.resx: {string.Join(", ", missing)}");
    Assert.True(empty.Count == 0, $"empty in AppStrings.it.resx: {string.Join(", ", empty)}");
  }

  [Fact]
  public void EveryItalianKeyExistsInEnglishWithANonEmptyValue()
  {
    var en = Entries("AppStrings.resx").GroupBy(e => e.Name).ToDictionary(g => g.Key, g => g.First().Value);
    var it = Entries("AppStrings.it.resx");
    var extra = it.Where(e => !en.ContainsKey(e.Name)).Select(e => e.Name).ToList();
    var empty = en.Where(e => string.IsNullOrWhiteSpace(e.Value)).Select(e => e.Key).ToList();
    Assert.True(extra.Count == 0, $"only in AppStrings.it.resx: {string.Join(", ", extra)}");
    Assert.True(empty.Count == 0, $"empty in AppStrings.resx: {string.Join(", ", empty)}");
  }

  /// <summary>The embedded resources (the Italian satellite assembly included) resolve every key; an
  /// inline value resolves to itself, a file reference (<c>ResXFileRef</c>, e.g. the orbital
  /// element help texts) to its non-empty file text.</summary>
  [Fact]
  public void EmbeddedResourcesResolveEveryKeyInBothCultures()
  {
    var rm = new ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly);
    var it = new CultureInfo("it");
    var en = new CultureInfo("en");
    var bad = new List<string>();
    var fileRefs = XDocument.Load(ResourcePath("AppStrings.resx")).Root!.Elements("data")
      .Where(d => ((string?)d.Attribute("type") ?? "").StartsWith("System.Resources.ResXFileRef", StringComparison.Ordinal))
      .Select(d => (string?)d.Attribute("name") ?? "")
      .ToHashSet();
    foreach (var (name, value) in Entries("AppStrings.resx"))
    {
      var enValue = rm.GetString(name, en);
      // a Windows checkout keeps CRLF in the compiled value, the XML reader gives LF
      if (fileRefs.Contains(name) ? string.IsNullOrWhiteSpace(enValue) : enValue?.ReplaceLineEndings("\n") != value.ReplaceLineEndings("\n")) bad.Add($"{name} (en)");
      var itValue = rm.GetResourceSet(it, createIfNotExists: true, tryParents: false)?.GetString(name);
      if (string.IsNullOrWhiteSpace(itValue)) bad.Add($"{name} (it)");
    }
    Assert.True(bad.Count == 0, $"unresolved: {string.Join(", ", bad)}");
  }

  /// <summary>The observation hints split "no FITS image" from "a FITS image without a readable
  /// date"; both texts exist in English and Italian, the date ones with their {Date} slot.</summary>
  [Theory]
  [InlineData("ModeBar_FitsWithoutDateHint", true)]
  [InlineData("Observation_NoImage", false)]
  [InlineData("Observation_FitsWithoutDate", true)]
  [InlineData("Observation_DateKeywordsNotFound", false)]
  public void ObservationDateHints_ExistInEnglishAndItalian(string key, bool hasDateSlot)
  {
    var rm = new ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly);
    var en = rm.GetString(key, new CultureInfo("en"));
    var it = rm.GetResourceSet(new CultureInfo("it"), createIfNotExists: true, tryParents: false)?.GetString(key);
    Assert.False(string.IsNullOrWhiteSpace(en), $"{key} (en)");
    Assert.False(string.IsNullOrWhiteSpace(it), $"{key} (it)");
    Assert.NotEqual(en, it);
    Assert.Equal(hasDateSlot, en!.Contains("{Date}"));
    Assert.Equal(hasDateSlot, it!.Contains("{Date}"));
  }

  /// <summary>The two-digit-year century selector of the FITS billboard panel: its label, help and
  /// both options exist in English and Italian (the texts translated, the century digits kept).</summary>
  [Theory]
  [InlineData("Billboard_FitsCentury_Label", null)]
  [InlineData("Billboard_FitsCentury_Help", "19")]
  [InlineData("Billboard_FitsCentury_TwoThousands", "20xx")]
  [InlineData("Billboard_FitsCentury_NineteenHundreds", "19xx")]
  public void FitsCenturyStrings_ExistInEnglishAndItalian(string key, string? digits)
  {
    var rm = new ResourceManager("AetherVk.Logic.Resources.AppStrings", typeof(ITranslationService).Assembly);
    var en = rm.GetString(key, new CultureInfo("en"));
    var it = rm.GetResourceSet(new CultureInfo("it"), createIfNotExists: true, tryParents: false)?.GetString(key);
    Assert.False(string.IsNullOrWhiteSpace(en), $"{key} (en)");
    Assert.False(string.IsNullOrWhiteSpace(it), $"{key} (it)");
    Assert.NotEqual(en, it);
    if (digits is not null)
    {
      Assert.Contains(digits, en);
      Assert.Contains(digits, it);
    }
  }
}
