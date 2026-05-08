using AetherVk.Logic.Input;
using Xunit;

namespace AetherVk.Logic.Tests;

public class InputChordTests
{
    [Fact]
    public void DisplayText_FormatsCorrectly()
    {
        var chord1 = new InputChord(Key: "S", Shift: true, Ctrl: true);
        Assert.Equal("Ctrl + Shift + S", chord1.DisplayText);
        
        var chord2 = new InputChord(Pointer: "MiddleButtonPressed", Alt: true);
        Assert.Equal("Alt + Middle Click", chord2.DisplayText);
    }
}
