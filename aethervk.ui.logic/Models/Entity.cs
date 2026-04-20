using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.Models;

public interface IComponent
{
  string Name { get; }
}

public class EntityVisibilityChangedMessage
{
  public Entity Entity { get; }
  public EntityVisibilityChangedMessage(Entity entity) => Entity = entity;
}

public class EntityOutlineChangedMessage
{
  public Entity Entity { get; }
  public EntityOutlineChangedMessage(Entity entity) => Entity = entity;
}

public partial class Entity : ObservableObject
{
  public ulong Id { get; }

  [ObservableProperty]
  private string _name;

  [ObservableProperty]
  private bool _isVisible = true;

  [ObservableProperty]
  private bool _isOutlined = false;

  public ObservableCollection<Entity> Children { get; } = new();
  public ObservableCollection<IComponent> Components { get; } = new();

  public Entity(ulong id, string name)
  {
    Id = id;
    _name = name;
  }

  partial void OnIsVisibleChanged(bool value)
  {
    WeakReferenceMessenger.Default.Send(new EntityVisibilityChangedMessage(this));
  }

  partial void OnIsOutlinedChanged(bool value)
  {
    WeakReferenceMessenger.Default.Send(new EntityOutlineChangedMessage(this));
  }
}
