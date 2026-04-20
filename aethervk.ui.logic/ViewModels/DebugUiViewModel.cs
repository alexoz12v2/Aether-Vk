using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

/// <summary>
/// A ViewModel for a tab that provides debugging tools.
/// </summary>
public partial class DebugUiViewModel() : TabItemViewModel("Debug UI")
{
  [ObservableProperty]
  [NotifyCanExecuteChangedFor(nameof(SendConsoleMessageCommand))]
  private string? _messageToSend;

  private bool CanSendConsoleMessage() => !string.IsNullOrWhiteSpace(MessageToSend);

  [RelayCommand(CanExecute = nameof(CanSendConsoleMessage))]
  private void SendConsoleMessage()
  {
    // The CanExecute check ensures MessageToSend is not null here.
    WeakReferenceMessenger.Default.Send(new ConsoleMessage(MessageToSend!));
    MessageToSend = string.Empty; // Clear the textbox after sending.
  }
}

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
