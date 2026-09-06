using System;
using AetherVk.Logic.Utils;
using Xunit;

namespace AetherVk.Logic.Tests;

public class TimeUtilsTests
{
    [Theory]
    [InlineData("2025-10-10T00:00:00Z")]
    [InlineData("2025-10-01T00:00:00Z")]
    [InlineData("2000-01-01T12:00:00Z")]
    [InlineData("2000-01-01T00:00:00Z")]
    [InlineData("1970-01-01T00:00:00Z")]
    [InlineData("1900-01-01T00:00:00Z")]
    [InlineData("2099-12-31T23:59:59.999Z")]
    public void ToTaiParts_FromTaiParts_ShouldRoundtripExactly(string isoString)
    {
        // Arrange
        Assert.True(TimeUtils.TryParseIso8601(isoString, out var expectedTime));

        // Act
        var (centuries, nanoseconds) = TimeUtils.ToTaiParts(expectedTime);
        var actualTime = TimeUtils.FromTaiParts(centuries, nanoseconds);

        // Assert
        Assert.Equal(expectedTime, actualTime);
    }

    [Fact]
    public void FormatTaiEpoch_FormatsCorrectlyWithoutShift()
    {
        // Arrange
        var expectedTimeStr = "2025-10-10T00:00:00Z";
        Assert.True(TimeUtils.TryParseIso8601(expectedTimeStr, out var expectedTime));
        
        // Act
        var (centuries, nanoseconds) = TimeUtils.ToTaiParts(expectedTime);
        var formatted = TimeUtils.FormatTaiEpoch(centuries, nanoseconds);

        // Assert - The old flawed logic would yield "2025-10-09 23:59"
        Assert.Equal("2025-10-10 00:00", formatted);
    }
}
