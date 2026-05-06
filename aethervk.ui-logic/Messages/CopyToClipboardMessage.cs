namespace AetherVk.Logic.Messages;

public class CopyToClipboardMessage
{
  public string Text { get; }

  public CopyToClipboardMessage(string text)
  {
    Text = text;
  }
}
