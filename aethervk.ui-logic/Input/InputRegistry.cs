using System.Collections.Generic;
using System.Linq;

namespace AetherVk.Logic.Input;

/// <summary>One binding: in <see cref="Context"/>, <see cref="Chord"/> triggers the action <see cref="ActionId"/>.</summary>
public sealed record InputBinding(string Context, InputChord Chord, string ActionId);

public class InputRegistry
{
  private readonly Dictionary<string, Dictionary<InputChord, AppAction>> _bindings = new();
  private IReadOnlyList<InputBinding> _defaults = [];

  public void Register(string context, InputChord chord, AppAction action)
  {
    if (!_bindings.TryGetValue(context, out var map))
    {
      map = new Dictionary<InputChord, AppAction>();
      _bindings[context] = map;
    }
    map[chord] = action;
  }

  public AppAction? Resolve(string context, InputChord chord)
  {
    if (_bindings.TryGetValue(context, out var map) && map.TryGetValue(chord, out var action))
    {
      return action;
    }
    return null;
  }

  public IEnumerable<(string Context, InputChord Chord, AppAction Action)> GetAllMappings()
  {
    return from ctx in _bindings
           from binding in ctx.Value
           select (ctx.Key, binding.Key, binding.Value);
  }

  /// <summary>Removes the binding of <paramref name="chord"/> in <paramref name="context"/>; false when none.</summary>
  public bool Unregister(string context, InputChord chord) =>
    _bindings.TryGetValue(context, out var map) && map.Remove(chord);

  /// <summary>Removes every binding of <paramref name="context"/>.</summary>
  public void Clear(string context) => _bindings.Remove(context);

  /// <summary>The action <paramref name="chord"/> already triggers in <paramref name="context"/>, if any
  /// (a new binding of the chord would replace it).</summary>
  public string? FindConflict(string context, InputChord chord) => Resolve(context, chord)?.Id;

  /// <summary>Every binding, ordered by context, action, then chord text (stable for files and UIs).</summary>
  public IReadOnlyList<InputBinding> Snapshot() =>
    GetAllMappings()
      .Select(m => new InputBinding(m.Context, m.Chord, m.Action.Id))
      .OrderBy(b => b.Context, System.StringComparer.Ordinal)
      .ThenBy(b => b.ActionId, System.StringComparer.Ordinal)
      .ThenBy(b => b.Chord.DisplayText, System.StringComparer.Ordinal)
      .ToList();

  /// <summary>Replaces the whole table with <paramref name="bindings"/>.</summary>
  public void Replace(IEnumerable<InputBinding> bindings)
  {
    _bindings.Clear();
    foreach (var b in bindings)
      Register(b.Context, b.Chord, new AppAction(b.ActionId));
  }

  /// <summary>Remembers the current table as the defaults (call after registering them).</summary>
  public void CaptureDefaults() => _defaults = Snapshot();

  /// <summary>The bindings captured by <see cref="CaptureDefaults"/>.</summary>
  public IReadOnlyList<InputBinding> Defaults => _defaults;

  /// <summary>Restores the defaults of <paramref name="context"/> (other contexts untouched).</summary>
  public void ResetToDefaults(string context)
  {
    Clear(context);
    foreach (var b in _defaults.Where(b => b.Context == context))
      Register(b.Context, b.Chord, new AppAction(b.ActionId));
  }
}
