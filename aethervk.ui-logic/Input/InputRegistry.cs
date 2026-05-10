using System.Collections.Generic;
using System.Linq;

namespace AetherVk.Logic.Input;

public class InputRegistry
{
  private readonly Dictionary<string, Dictionary<InputChord, AppAction>> _bindings = new();

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
}
