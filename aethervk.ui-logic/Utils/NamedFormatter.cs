using System;
using System.Collections.Generic;
using System.Text;

namespace AetherVk.Logic.Utils;

/// <summary>
/// AOT-safe, zero-reflection named-placeholder formatter.
/// Parses <c>{Key}</c> tokens in a template string and resolves them from a
/// dictionary of pre-rendered string values.
/// </summary>
/// <remarks>
/// <para>
/// Callers are responsible for converting typed values to their final string
/// representation (e.g. <c>value.ToString("N0", culture)</c>) before passing
/// them in the dictionary. This keeps the formatter itself reflection-free and
/// fully compatible with .NET Native AOT.
/// </para>
/// <para>
/// Escape sequences: <c>{{</c> → literal <c>{</c>, <c>}}</c> → literal <c>}</c>.
/// Unknown keys are left intact (e.g. <c>{UnknownKey}</c> is emitted as-is).
/// </para>
/// </remarks>
public static class NamedFormatter
{
  /// <summary>
  /// Formats <paramref name="template"/> by replacing every <c>{Key}</c> token
  /// with the matching value from <paramref name="args"/>.
  /// </summary>
  /// <param name="template">
  /// Template string, e.g. <c>"Loaded {Count} particles from {Name}"</c>.
  /// </param>
  /// <param name="args">
  /// Pre-rendered named values keyed by placeholder name.
  /// </param>
  /// <returns>The formatted string, or <see cref="string.Empty"/> if
  /// <paramref name="template"/> is <see langword="null"/> or empty.</returns>
  public static string Format(
    string template,
    IReadOnlyDictionary<string, string> args)
  {
    if (string.IsNullOrEmpty(template))
      return template ?? string.Empty;

    if (args == null || args.Count == 0)
      return template;

    var sb = new StringBuilder(template.Length + 16);
    var i = 0;

    while (i < template.Length)
    {
      var c = template[i];

      // Escape: {{ → literal {
      if (c == '{' && i + 1 < template.Length && template[i + 1] == '{')
      {
        sb.Append('{');
        i += 2;
        continue;
      }

      // Escape: }} → literal }
      if (c == '}' && i + 1 < template.Length && template[i + 1] == '}')
      {
        sb.Append('}');
        i += 2;
        continue;
      }

      // Placeholder start
      if (c == '{')
      {
        var end = template.IndexOf('}', i + 1);
        if (end < 0)
        {
          // No closing brace — emit remaining literal and stop
          sb.Append(template, i, template.Length - i);
          break;
        }

        // Inner content: "Key" or "Key:format"
        var inner = template.Substring(i + 1, end - i - 1);

        // Split on ':' to separate key from optional format specifier (not applied here)
        var colon = inner.IndexOf(':');
        var key = colon < 0 ? inner : inner.Substring(0, colon);

        if (args.TryGetValue(key, out var value))
          sb.Append(value);
        else
          // Unknown key: preserve the original placeholder
          sb.Append(template, i, end - i + 1);

        i = end + 1;
        continue;
      }

      sb.Append(c);
      i++;
    }

    return sb.ToString();
  }
}
