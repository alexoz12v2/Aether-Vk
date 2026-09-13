using System.Numerics;
using Xunit;
using AetherVk.Logic.Services;

namespace AetherVk.Logic.Tests;

public class CameraTrackingTests
{
    [Fact]
    public void Test_AnimationTargetDTO_PacksLocalOffsetAndPivotCorrectly()
    {
        // Arrange
        double localOffsetX = 0.000042;
        double localOffsetY = 0.0;
        double localPosZ    = 0.0;
        var    rotation     = Quaternion.Identity;
        float  duration     = 0.4f;
        ulong  pivotId      = 42UL;

        // Act
        var target = new AnimationTarget(localOffsetX, localOffsetY, localPosZ, rotation, duration, pivotId);
        var dto    = target.ToDTO();

        // Assert — positions are packed as f64 in the DTO
        Assert.Equal(localOffsetX, dto.posX);
        Assert.Equal(localOffsetY, dto.posY);
        Assert.Equal(localPosZ,    dto.posZ);
        Assert.Equal(duration,     dto.durationS);

        // PivotEntityId must be propagated
        Assert.Equal(pivotId, dto.pivotEntityId);
    }

    [Fact]
    public void Test_AnimationTarget_NullPivot_PacksZeroIntoDTO()
    {
        // A null pivot entity means "no parent" — the DTO should carry 0.
        var target = new AnimationTarget(1.0, 2.0, 3.0, Quaternion.Identity, 1.0f, null);
        var dto    = target.ToDTO();

        Assert.Equal(0UL, dto.pivotEntityId);
    }
}
