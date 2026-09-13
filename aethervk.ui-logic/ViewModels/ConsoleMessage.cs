namespace AetherVk.Logic.ViewModels;

/// <summary>
/// A message containing a string to be logged to the console.
/// </summary>
public class ConsoleMessage
{
  public string Message { get; }

  public ConsoleMessage(string message)
  {
    Message = message;
  }
}
