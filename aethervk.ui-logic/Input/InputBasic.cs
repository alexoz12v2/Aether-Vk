
using System;

namespace AetherVk.Logic.Input;

[Flags]
public enum InputModifiers : byte
{
  None = 0,
  Shift = 1 << 0,
  Ctrl = 1 << 1,
  Alt = 1 << 2,
  LeftMouse = 1 << 3,
  RightMouse = 1 << 4,
  MiddleMouse = 1 << 5,
}

public readonly struct InputState(bool isPressed, InputModifiers modifiers)
{
  public bool IsPressed { get; } = isPressed;
  public InputModifiers Modifiers { get; } = modifiers;
}

