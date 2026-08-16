using System;
using System.Text;

namespace AetherVk.Logic.Utils;

public static class StringUtils
{
  /// <summary>
  /// Converts a pinned pointer to a UTF-8 character array, null terminated, using
  /// `System.Memory`'s ReadOnlySpan to search for null terminator, and `System.Text.Encoding`
  /// to parse into a standard UTF-16 `System.String`'
  /// </summary>
  public static unsafe string? GetStringFromUtf8(byte* utf8Ptr)
  {
    if (utf8Ptr == null)
      return null;

    // 1. Create a span up to max length to leverage vectorized searching from System.Memory
    // This does not allocate memory; it just creates a view
    ReadOnlySpan<byte> searchSpan = new(utf8Ptr, int.MaxValue);

    // 2. Use SIMD-accelerated IndexOf to find the null terminator (0x00)
    int length = searchSpan.IndexOf((byte)0);
    if (length == 0)
      return string.Empty;

    if (length < 0)
    {
      // Failsafe in case memory is corrupted and lacks a terminator
      throw new ArgumentException("Null terminator not found.");
    }

    // 3. Decode directly from the pointer (Encoding in .NET Standard 2.0
    // doesn't have a GetString overload that takes a span)
    return Encoding.UTF8.GetString(utf8Ptr, length);
  }
}
