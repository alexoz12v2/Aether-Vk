namespace AetherVk.Logic.Input;

public interface IActionHandler
{
  bool Process(AppAction action, InputState inputState);
}

