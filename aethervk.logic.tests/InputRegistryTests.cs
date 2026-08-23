using System.Linq;
using AetherVk.Logic.Input;
using Xunit;

namespace AetherVk.Logic.Tests;

public class InputRegistryTests
{
  [Fact]
  public void RegisterAndResolve_WorksCorrectly()
  {
    var registry = new InputRegistry();
    var chord = new InputChord(Key: "S", Shift: true);
    var action = new AppAction("test.action", "Test");

    registry.Register("TestContext", chord, action);

    var resolved = registry.Resolve("TestContext", chord);
    Assert.NotNull(resolved);
    Assert.Equal("test.action", resolved.Value.Id);

    var notFound = registry.Resolve("WrongContext", chord);
    Assert.Null(notFound);
  }

  [Fact]
  public void GetAllMappings_ReturnsAllRegistered()
  {
    var registry = new InputRegistry();
    registry.Register("Ctx1", new InputChord(Key: "A"), new AppAction("act1", "Act 1"));
    registry.Register("Ctx2", new InputChord(Key: "B"), new AppAction("act2", "Act 2"));

    var mappings = registry.GetAllMappings().ToList();

    Assert.Equal(2, mappings.Count);
    Assert.Contains(mappings, m => m.Context == "Ctx1" && m.Action.Id == "act1");
    Assert.Contains(mappings, m => m.Context == "Ctx2" && m.Action.Id == "act2");
  }
}
