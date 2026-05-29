using System.Linq;
using AetherVk.Logic.Services;
using Xunit;

namespace AetherVk.Logic.Tests.Services
{
  public class NativeBufferPoolServiceTests
  {
    [Fact]
    public void Rent_ShouldReturnArrayOfSufficientSize()
    {
      // Arrange
      var poolService = new NativeBufferPoolService();

      // Act
      using var pooled = poolService.Rent<int>(10);

      // Assert
      Assert.NotNull(pooled.Array);
      Assert.True(pooled.Array.Length >= 10);
    }

    [Fact]
    public void Rent_MultipleTimes_ShouldReturnDistinctOrReusedArrays()
    {
      // Arrange
      var poolService = new NativeBufferPoolService();

      // Act
      var pooled1 = poolService.Rent<int>(10);
      pooled1.Array[0] = 42;
      pooled1.Dispose(); // Return to pool

      using var pooled2 = poolService.Rent<int>(10);

      // Assert
      // It might reuse the same array instance from ArrayPool
      Assert.NotNull(pooled2.Array);
      Assert.True(pooled2.Array.Length >= 10);
    }
  }
}
