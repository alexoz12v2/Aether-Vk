using System.Globalization;

namespace AetherVk.Logic.Utils;

/// <summary>
/// The app's process culture (reviewer docx item 8): numbers read and print with a dot on every
/// OS locale (an Italian Windows otherwise shows and parses "0,5"); dates and the rest of the
/// locale are kept. Called once at startup by <c>Program.Main</c>.
/// </summary>
public static class NumberCulture
{
  /// <summary><paramref name="locale"/> with the invariant number format.</summary>
  public static CultureInfo WithDotNumbers(CultureInfo locale)
  {
    var culture = (CultureInfo)locale.Clone();
    culture.NumberFormat = CultureInfo.InvariantCulture.NumberFormat;
    return culture;
  }

  /// <summary>
  /// Makes <paramref name="name"/> (a culture name such as "en-US") the current and UI culture of
  /// the process and of new threads; returns false and changes nothing when it is empty or not a
  /// known culture. Called before <see cref="Apply"/>, which keeps the dot numbers.
  /// </summary>
  public static bool ApplyOverride(string? name)
  {
    if (string.IsNullOrWhiteSpace(name)) return false;
    string wanted = name.Trim();
    // only a predefined culture (ICU would otherwise make up one for any well-formed name)
    CultureInfo? culture = null;
    foreach (var c in CultureInfo.GetCultures(CultureTypes.AllCultures))
      if (c.Name.Length > 0 && string.Equals(c.Name, wanted, System.StringComparison.OrdinalIgnoreCase))
      {
        culture = c;
        break;
      }
    if (culture is null) return false;
    CultureInfo.CurrentCulture = culture;
    CultureInfo.CurrentUICulture = culture;
    CultureInfo.DefaultThreadCurrentCulture = culture;
    CultureInfo.DefaultThreadCurrentUICulture = culture;
    return true;
  }

  /// <summary>Makes <see cref="WithDotNumbers"/> of the current culture the current and the
  /// default thread culture; returns it.</summary>
  public static CultureInfo Apply()
  {
    var culture = WithDotNumbers(CultureInfo.CurrentCulture);
    CultureInfo.CurrentCulture = culture;
    CultureInfo.DefaultThreadCurrentCulture = culture;
    return culture;
  }
}
