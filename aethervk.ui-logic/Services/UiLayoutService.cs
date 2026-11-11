using System;

namespace AetherVk.Logic.Services;

/// <summary>The arrangement of the main window's panels.</summary>
public enum UiLayout
{
  Classic,
  Compact,
}

/// <summary>
/// The UI layout in use (Classic or Compact). A save stores it as <c>ui_layout</c>
/// (docs/savefile.md); the preferences remember the user's last choice for new sessions. Runs on
/// the UI thread.
/// </summary>
public sealed class UiLayoutService
{
  public UiLayout Current { get; private set; } = UiLayout.Classic;

  /// <summary>Raised with the new layout, only when it changed.</summary>
  public event EventHandler<UiLayout>? Changed;

  public void Set(UiLayout layout)
  {
    if (layout == Current) return;
    Current = layout;
    Changed?.Invoke(this, layout);
  }

  /// <summary>The index / preferences value: "compact", or null for Classic (not written, so
  /// saves without the field stay byte-identical).</summary>
  public static string? ToDto(UiLayout layout) => layout == UiLayout.Compact ? "compact" : null;

  /// <summary>"compact" (any case) is Compact; anything else, or null, is Classic.</summary>
  public static UiLayout FromDto(string? value) =>
    string.Equals(value?.Trim(), "compact", StringComparison.OrdinalIgnoreCase) ? UiLayout.Compact : UiLayout.Classic;
}
