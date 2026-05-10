namespace AetherVk.Logic.Input;

public interface IActionOperator
{
  void OnEnter();
  void OnExit();
  bool ProcessAction(AppAction action, bool isPressed);
  bool ProcessPointerDelta(float dx, float dy);
  bool ProcessPointerWheel(float deltaY);
}
