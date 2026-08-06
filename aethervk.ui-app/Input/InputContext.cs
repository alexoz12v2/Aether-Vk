
namespace AetherVk.App.Input;

/// <summary>
/// List of hardcoded input contexts for <see cref="InputRegistry" /> as a string enum
/// </summary>
public enum InputContext
{
  Viewport
}

public static class InputContextExtensions
{
  extension(InputContext inputContext)
  {
    public string ToCtxString() => inputContext switch
    {
      InputContext.Viewport => "Viewport"
    };
  }

  extension(InputContext)
  {
    public static InputContext FromCtxString(string value) => value switch
    {
      "Viewport" => InputContext.Viewport,
      _ => throw new FormatException($"Invalid InputContext value ${value}")
    };
  }
}

