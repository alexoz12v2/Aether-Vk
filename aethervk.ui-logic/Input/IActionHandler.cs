namespace AetherVk.Logic.Input;

public interface IActionHandler
{
  bool ProcessAction(AppAction action, bool isPressed);
  bool ProcessPointerDelta(float dx, float dy);
  bool ProcessPointerWheel(float deltaY);
}
