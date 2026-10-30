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

  // X11 keycodes are evdev + 8. Unmapped, the keypad codes passed through raw and hit letters
  // (KP_1 = 87 = 'W'); keypad codes are physical, so NumLock does not matter (feedback #18).
  [Theory]
  [InlineData(90u, "NumPad0")]       // KP_0 / KP_Insert
  [InlineData(87u, "NumPad1")]       // KP_1 / KP_End (was 'W')
  [InlineData(88u, "NumPad2")]       // KP_2 / KP_Down (was 'X')
  [InlineData(89u, "NumPad3")]       // KP_3 / KP_Next (was 'Y')
  [InlineData(83u, "NumPad4")]       // KP_4 / KP_Left (was 'S')
  [InlineData(84u, "NumPad5")]       // KP_5 / KP_Begin (was 'T')
  [InlineData(85u, "NumPad6")]       // KP_6 / KP_Right (was 'U')
  [InlineData(79u, "NumPad7")]       // KP_7 / KP_Home (was 'O')
  [InlineData(80u, "NumPad8")]       // KP_8 / KP_Up (was 'P')
  [InlineData(81u, "NumPad9")]       // KP_9 / KP_Prior (was 'Q')
  [InlineData(91u, "NumPadDecimal")] // KP_Decimal / KP_Delete
  public void X11_KeypadKeycodes_MapToTheNumPadKeys(uint x11Keycode, string expected)
  {
    Assert.Equal(expected, KeyOf(NativeInputConverter.NormalizeX11Keycode(x11Keycode)));
  }

  [Theory]
  [InlineData(25u, 0x57u, "W")]  // the letter W, the key KP_1 used to be mistaken for
  [InlineData(10u, 0x31u, "D1")] // the top-row 1
  [InlineData(115u, 0x23u, null)] // the dedicated End key stays VK_END, not NumPad1
  public void X11_NonKeypadKeycodes_AreNotTheKeypad(uint x11Keycode, uint expectedVk, string? expectedKey)
  {
    uint vk = NativeInputConverter.NormalizeX11Keycode(x11Keycode);
    Assert.Equal(expectedVk, vk);
    Assert.Equal(expectedKey, KeyOf(vk));
  }

  [Fact]
  public void X11_UnmappedKeycode_PassesThroughRaw()
  {
    Assert.Equal(200u, NativeInputConverter.NormalizeX11Keycode(200u));
  }
}
