using System;
using System.Collections.ObjectModel;
using System.Linq;
using AetherVk.Logic.Input;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// "Input bindings" section of the Settings tab: per input context, every action with its chords.
/// <list type="bullet">
///   <item><b>Rebind</b> / <b>Add chord</b> start a capture: the view hands the next key or pointer
///   chord to <see cref="SubmitCapturedChord"/>; Esc (<see cref="CancelCapture"/>) cancels.</item>
///   <item>A chord already bound to another action of the context is refused
///   (<see cref="ConflictMessage"/>); nothing changes.</item>
///   <item><b>Remove</b> drops a chord, <b>Reset</b> restores the context's defaults.</item>
/// </list>
/// Every change goes to <see cref="InputRegistry"/> (live at once) and to preferences.json.
/// </summary>
public partial class InputBindingsViewModel : ObservableObject
{
  private readonly InputRegistry _registry;
  private readonly PreferencesService? _preferences;
  private (string Context, string ActionId, InputChord? Replacing)? _capture;

  public ObservableCollection<BindingContextViewModel> Contexts { get; } = [];

  [ObservableProperty]
  private bool _isCapturing;

  /// <summary>"Press a key for <action> (Esc cancels)" while capturing.</summary>
  [ObservableProperty]
  private string _captureHint = string.Empty;

  [ObservableProperty]
  private string _conflictMessage = string.Empty;

  public InputBindingsViewModel(InputRegistry registry, PreferencesService? preferences)
  {
    _registry = registry;
    _preferences = preferences;
    Rebuild();
  }

  /// <summary>Human-readable action name: <c>viewport.switch_camera_mode</c> → "Switch camera mode".</summary>
  public static string ActionName(string actionId)
  {
    var tail = actionId.Contains('.') ? actionId.Substring(actionId.LastIndexOf('.') + 1) : actionId;
    var words = tail.Replace('_', ' ');
    return words.Length == 0 ? actionId : char.ToUpperInvariant(words[0]) + words.Substring(1);
  }

  public void Rebuild()
  {
    Contexts.Clear();
    var current = _registry.Snapshot();
    var all = _registry.Defaults.Concat(current);
    foreach (var ctx in all.Select(b => b.Context).Distinct().OrderBy(c => c, StringComparer.Ordinal))
    {
      var group = new BindingContextViewModel(ctx, this);
      foreach (var action in all.Where(b => b.Context == ctx).Select(b => b.ActionId).Distinct().OrderBy(a => a, StringComparer.Ordinal))
      {
        var row = new BindingActionViewModel(ctx, action, ActionName(action), this);
        foreach (var b in current.Where(b => b.Context == ctx && b.ActionId == action))
          row.Chords.Add(new BindingChordViewModel(ctx, action, b.Chord, this));
        group.Actions.Add(row);
      }
      Contexts.Add(group);
    }
  }

  internal void BeginCapture(string context, string actionId, InputChord? replacing)
  {
    _capture = (context, actionId, replacing);
    IsCapturing = true;
    ConflictMessage = string.Empty;
    CaptureHint = $"Press the new chord for \"{ActionName(actionId)}\" (Esc cancels)";
  }

  public void CancelCapture()
  {
    _capture = null;
    IsCapturing = false;
    CaptureHint = string.Empty;
  }

  /// <summary>The chord captured by the view. Returns whether the binding changed.</summary>
  public bool SubmitCapturedChord(InputChord chord)
  {
    if (_capture is not { } c)
      return false;
    var other = _registry.FindConflict(c.Context, chord);
    if (other is not null && other != c.ActionId)
    {
      ConflictMessage = $"{chord.DisplayText} is already bound to \"{ActionName(other)}\"";
      CancelCapture();
      return false;
    }
    if (c.Replacing is { } old)
      _registry.Unregister(c.Context, old);
    _registry.Register(c.Context, chord, new AppAction(c.ActionId));
    CancelCapture();
    Commit();
    return true;
  }

  internal void Remove(string context, InputChord chord)
  {
    if (_registry.Unregister(context, chord))
      Commit();
  }

  internal void Reset(string context)
  {
    _registry.ResetToDefaults(context);
    ConflictMessage = string.Empty;
    Commit();
  }

  private void Commit()
  {
    _preferences?.SaveBindings();
    Rebuild();
  }
}

public sealed partial class BindingContextViewModel(string context, InputBindingsViewModel owner) : ObservableObject
{
  public string Context { get; } = context;
  public ObservableCollection<BindingActionViewModel> Actions { get; } = [];

  [RelayCommand]
  private void Reset() => owner.Reset(Context);
}

public sealed partial class BindingActionViewModel(string context, string actionId, string name, InputBindingsViewModel owner)
  : ObservableObject
{
  public string Context { get; } = context;
  public string ActionId { get; } = actionId;
  public string Name { get; } = name;
  public ObservableCollection<BindingChordViewModel> Chords { get; } = [];

  [RelayCommand]
  private void AddChord() => owner.BeginCapture(Context, ActionId, null);
}

public sealed partial class BindingChordViewModel(string context, string actionId, InputChord chord, InputBindingsViewModel owner)
  : ObservableObject
{
  public InputChord Chord { get; } = chord;
  public string Text => Chord.DisplayText;

  [RelayCommand]
  private void Rebind() => owner.BeginCapture(context, actionId, Chord);

  [RelayCommand]
  private void Remove() => owner.Remove(context, Chord);
}
