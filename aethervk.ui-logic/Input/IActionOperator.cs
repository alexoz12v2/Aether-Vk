namespace AetherVk.Logic.Input;

public interface IActionOperator
{
  void OnEnter();
  void OnExit();
  bool ProcessAction(AppAction action, InputState inputState);
}
