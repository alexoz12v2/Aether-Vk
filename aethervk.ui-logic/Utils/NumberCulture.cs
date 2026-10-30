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
