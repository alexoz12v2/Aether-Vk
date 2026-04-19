using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Models;

public interface IComponent
{
  string Name { get; }
}

public partial class Entity : ObservableObject
{
  public ulong Id { get; }

  [ObservableProperty]
  private string _name;

  public ObservableCollection<Entity> Children { get; } = new();
  public ObservableCollection<IComponent> Components { get; } = new();

  public Entity(ulong id, string name)
  {
    Id = id;
    _name = name;
  }
}
