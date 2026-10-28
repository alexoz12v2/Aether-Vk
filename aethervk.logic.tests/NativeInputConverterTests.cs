using AetherVk.Logic.Input;
using AetherVk.Logic.Services.NativeInput;
using Xunit;

namespace AetherVk.Logic.Tests;

/// <summary>Keypad shortcuts whatever the NumLock state (feedback #18).</summary>
public sealed class NativeInputConverterTests
{
  private static string? KeyOf(uint vk) =>
    NativeInputConverter.ToInputChord(
      new ComposedKeystroke([new NativeKeyEvent(vk, IsDown: true, NativeModifierFlags.None, 0)], 0))?.Key;

  [Theory]
  [InlineData(0x23u, "NumPad1")] // VK_END
  [InlineData(0x28u, "NumPad2")] // VK_DOWN
  [InlineData(0x22u, "NumPad3")] // VK_NEXT
  [InlineData(0x0Cu, "NumPad5")] // VK_CLEAR
  [InlineData(0x2Eu, "NumPadDecimal")] // VK_DELETE
  [InlineData(0x2Du, "NumPad0")] // VK_INSERT
  public void NumLockOff_KeypadKeys_MapToTheNumPadKeys(uint navigationVk, string expected)
  {
    uint vk = NativeInputConverter.NormalizeWin32NumPad(navigationVk, isExtended: false);
    Assert.Equal(expected, KeyOf(vk));
  }

  [Theory]
  [InlineData(0x23u)]
  [InlineData(0x28u)]
  [InlineData(0x2Eu)]
  public void DedicatedNavigationKeys_AreLeftAlone(uint navigationVk)
  {
    Assert.Equal(navigationVk, NativeInputConverter.NormalizeWin32NumPad(navigationVk, isExtended: true));
  }

  [Theory]
  [InlineData(0x61u)] // NumLock on: already VK_NUMPAD1
  [InlineData(0x31u)] // '1'
  [InlineData(0x56u)] // 'V'
  public void OtherKeys_PassThrough(uint vk)
  {
    Assert.Equal(vk, NativeInputConverter.NormalizeWin32NumPad(vk, isExtended: false));
  }
}
