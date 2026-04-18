using System;
using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.Services;

public partial class ConsoleService : ObservableObject
{
    public ObservableCollection<string> Messages { get; } = new ObservableCollection<string>();

    public void Log(string message)
    {
        Messages.Add($"[{DateTime.Now:HH:mm:ss}] {message}");
    }

    public void Clear()
    {
        Messages.Clear();
    }
}
