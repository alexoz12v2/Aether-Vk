using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;
using AetherVk.Logic.Messages;

namespace AetherVk.Logic.ViewModels;

public partial class VulkanViewportControlViewModel : ObservableObject
{
    public void ReportFatalError(string message)
    {
        WeakReferenceMessenger.Default.Send(new CriticalErrorMessage(message));
    }
}
