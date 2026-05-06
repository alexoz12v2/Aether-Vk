namespace AetherVk.Logic.Messages;

public class CriticalErrorMessage
{
  public string Message { get; }

  public CriticalErrorMessage(string message)
  {
    Message = message;
  }
}
