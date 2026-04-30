using System.Collections.ObjectModel;
using System.Linq;
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
  public ulong SceneId { get; }
  public ulong Id { get; }

  [ObservableProperty]
  private string _name;

  [ObservableProperty]
  private bool _isVisible = true;

  [ObservableProperty]
  private bool _isOutlined = false;

  public ObservableCollection<Entity> Children { get; } = new();
  public ObservableCollection<IComponent> Components { get; } = new();

  public bool IsRoot => Name == "root" || Id == 1;
  public bool IsMeasurement => Components.Any(c => c is MeasurementComponent);

  public Entity(ulong sceneId, ulong id, string name)
  {
    SceneId = sceneId;
    Id = id;
    _name = name;
  }

  partial void OnNameChanged(string value)
  {
    var runtimeService = Services.ServiceLocator.Provider?.GetService(typeof(Services.NativeRuntimeService)) as Services.NativeRuntimeService;
    runtimeService?.SetEntityName(SceneId, Id, value);
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
