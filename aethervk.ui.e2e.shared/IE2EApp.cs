// Source-linked into aethervk.ui.e2e (Windows) and aethervk.ui.e2e.mac: the seam the shared
// scenarios are written against. Every lookup is by AutomationId (on macOS the accessibility
// identifier); every rectangle is in screen pixels.
using System;
using System.Collections.Generic;

namespace AetherVk.UI.E2E.Shared;

/// <summary>What a lookup saw of one element, read at once (the element may be rebuilt afterwards).</summary>
public sealed record E2EElement(string AutomationId, E2ERect Rect, string Text, bool Enabled, bool Displayed);

/// <summary>How the app ended: <see cref="Code"/> is null when the driver cannot read it (not our child, or killed).</summary>
public sealed record E2EExit(bool Exited, int? Code, bool Killed)
{
  public bool Clean => Exited && !Killed && Code is null or 0;

  public override string ToString() => !Exited ? "still running" : Killed ? "killed" : $"exit {Code?.ToString() ?? "(unknown code)"}";
}

/// <summary>One running AetherVk driven by a platform's automation driver.</summary>
public interface IE2EApp : IDisposable
{
  /// <summary>"windows" or "macos" (for artifact names and messages only: scenarios never branch on it).</summary>
  string Platform { get; }

  int ProcessId { get; }

  /// <summary>The app's log lines so far (stdout/stderr, or the tailed log file).</summary>
  IReadOnlyList<string> Lines { get; }

  string WindowTitle { get; }

  E2EElement? TryFind(string automationId);

  E2EElement Find(string automationId, TimeSpan? timeout = null);

  /// <summary>Brings the element into view and clicks its centre.</summary>
  void Click(string automationId);

  /// <summary>A click at a screen pixel.</summary>
  void ClickAt(int x, int y);

  /// <summary>
  /// The desktop billboards' windows in screen pixels, where they take the pointer over the whole
  /// window (macOS: the transparent margin too). Empty where clicks reach what is behind them.
  /// </summary>
  System.Collections.Generic.IReadOnlyList<E2ERect> BillboardFrames() => [];

  /// <summary>
  /// Selects all of the element's text box (the element itself, its inner text box, or the slider's
  /// "&lt;id&gt;.Input"), types <paramref name="text"/> and, with <paramref name="enter"/>, commits with Enter / Return.
  /// </summary>
  void TypeInto(string automationId, string text, bool enter = true);

  /// <summary>Flips a check box, toggle button or expander (an expander at its header, not its content).</summary>
  void Toggle(string automationId);

  /// <summary>The checked / selected / expanded state the driver exposes; null when it exposes none.</summary>
  bool? IsChecked(string automationId);

  /// <summary>The element's text or value (for a slider, its "&lt;id&gt;.Input" box).</summary>
  string Text(string automationId);

  int ListCount(string listId);

  void SelectListItem(string listId, int index);

  /// <summary>Picks a combo box's item by position (labels are localised).</summary>
  void ChooseAt(string comboId, int index);

  /// <summary>A plain key (a digit or a letter) to whatever has keyboard focus.</summary>
  void PressKey(char key);

  /// <summary>Clicks a menu path by its visible titles, e.g. ("Layout", "Compact").</summary>
  void Menu(params string[] path);

  /// <summary>Answers the native Open dialog the last action raised with a file's full path.</summary>
  void AnswerOpenDialog(string path);

  /// <summary>Answers the native Save dialog the last action raised with a file's full path.</summary>
  void AnswerSaveDialog(string path);

  /// <summary>The displayed element's rectangle in screen pixels.</summary>
  E2ERect ViewportRect(string automationId);

  /// <summary>An n × n grid of luminance cell centres over a pixel rectangle (<see cref="FrameMetricsCore.Grid"/>).</summary>
  double[] SampleLuminance(E2ERect rect, int n);

  /// <summary>Every pixel's luminance in a pixel rectangle, rows from the top.</summary>
  double[] CaptureLuminance(E2ERect rect);

  void SavePng(E2ERect rect, string path);

  /// <summary>A screenshot of the whole screen or window for a failure (best effort).</summary>
  void SaveScreenshot(string path);

  /// <summary>Throws with the log tail when the app has exited or lost its window.</summary>
  void EnsureAlive(string doing);

  void Foreground();

  /// <summary>Closes the app as a user would, answers the save prompt with Discard, waits, kills on timeout.</summary>
  E2EExit Quit(TimeSpan? timeout = null);
}

/// <summary>A platform's way to start the app and where its artifacts go.</summary>
public interface IE2EHost
{
  string Platform { get; }

  /// <summary>TestResults/e2e (Windows) or TestResults/e2e-mac.</summary>
  string ArtifactsRoot { get; }

  /// <summary>Why the suite cannot run here; null when it can.</summary>
  string? UnavailableReason();

  /// <summary>Starts the app (AETHERVK_E2E=1, the culture forced) and attaches the driver, log to <paramref name="logPath"/>.</summary>
  IE2EApp Launch(string logPath, string culture = E2ESteps.Culture);

  /// <summary>Crash reports of the app written since <paramref name="sinceUtc"/> (the app's own and the OS's).</summary>
  IReadOnlyList<string> CrashReportsSince(DateTime sinceUtc);
}
