using AetherVk.Logic.Utils;
using Xunit;

namespace AetherVk.Logic.Tests;

public class CursorWrapHelperTests
{
    [Fact]
    public void TryWrapCursor_AtLeftEdge_WrapsToRight()
    {
        bool wrapped = CursorWrapHelper.TryWrapCursor(2, 0, 1920, 2, 10, out int newX);
        Assert.True(wrapped);
        Assert.Equal(1910, newX); // 1920 - 10
    }

    [Fact]
    public void TryWrapCursor_AtRightEdge_WrapsToLeft()
    {
        bool wrapped = CursorWrapHelper.TryWrapCursor(1919, 0, 1920, 2, 10, out int newX);
        Assert.True(wrapped);
        Assert.Equal(10, newX); // 0 + 10
    }

    [Fact]
    public void TryWrapCursor_InMiddle_DoesNotWrap()
    {
        bool wrapped = CursorWrapHelper.TryWrapCursor(1000, 0, 1920, 2, 10, out int newX);
        Assert.False(wrapped);
        Assert.Equal(1000, newX);
    }
}
