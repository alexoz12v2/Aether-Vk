using System.Collections.ObjectModel;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class ConsoleViewModel : TabItemViewModel, IRecipient<ConsoleMessage>
{
    private readonly ConsoleService _consoleService;

    public ObservableCollection<string> Messages => _consoleService.Messages;

    public ConsoleViewModel(ConsoleService consoleService) : base("Console")
    {
        _consoleService = consoleService;
        _consoleService.Log("Console initialized.");
        WeakReferenceMessenger.Default.Register<ConsoleMessage>(this);
    }

    [RelayCommand]
    private void ClearConsole()
    {
        _consoleService.Clear();
    }

    public void Receive(ConsoleMessage message)
    {
        _consoleService.Log($"[Debug UI] {message.Message}");
    }
}
