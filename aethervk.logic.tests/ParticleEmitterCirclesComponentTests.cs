using AetherVk.Logic.Models;
using Xunit;

namespace AetherVk.Logic.Tests;

public class ParticleEmitterCirclesComponentTests
{
  [Fact]
  public void Constructor_InitializesEmptyList()
  {
    var comp = new ParticleEmitterCirclesComponent();
    Assert.Empty(comp.Circles);
  }

  [Fact]
  public void AddCircleCommand_AddsItemToCollection()
  {
    var comp = new ParticleEmitterCirclesComponent();
    comp.AddCircleCommand.Execute(null);

    Assert.Single(comp.Circles);
  }

  [Fact]
  public void RemoveCircleCommand_RemovesItemFromCollection()
  {
    var comp = new ParticleEmitterCirclesComponent();
    comp.AddCircleCommand.Execute(null);
    var item = comp.Circles[0];

    comp.RemoveCircleCommand.Execute(item);

    Assert.Empty(comp.Circles);
  }

  [Fact]
  public void ModifyingCircle_DoesNotCrash()
  {
    var comp = new ParticleEmitterCirclesComponent();
    comp.AddCircleCommand.Execute(null);
    var item = comp.Circles[0];
    
    // Changing property should trigger push (which returns early because ptr is zero)
    item.LatitudeDeg = 45.0f;
    Assert.Equal(45.0f, item.LatitudeDeg);
  }
}
