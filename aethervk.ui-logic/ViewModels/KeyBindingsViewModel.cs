using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Input;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public record ShortcutDisplay(string ActionName, string ShortcutText, string Description);
public record ShortcutGroup(string ContextName, List<ShortcutDisplay> Mappings);

public partial class KeyBindingsViewModel : ViewModelBase, ISettingsCategory
{
    public IEnumerable<ShortcutGroup> Groups { get; }
    
    public string Name => "Key Bindings";

    public KeyBindingsViewModel(InputRegistry registry)
    {
        Groups = registry.GetAllMappings()
            .GroupBy(m => m.Context)
            .Select(g => new ShortcutGroup(
                g.Key,
                g.Select(m => new ShortcutDisplay(m.Action.DisplayName, m.Chord.DisplayText, m.Action.Description)).ToList()
            ))
            .ToList();
    }
}
