using System;
using System.Collections.ObjectModel;
using System.Text;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public partial class ConsoleViewModel : TabItemViewModel, IRecipient<ConsoleMessage>, IDisposable
{
  private readonly ConsoleService _consoleService;
  private readonly IFileDialogService _fileDialogService;

  [ObservableProperty]
  private string _fullText = string.Empty;

  [ObservableProperty]
  private double _consoleFontSize = 14;

  public ObservableCollection<string> Messages => _consoleService.Messages;

  public ConsoleViewModel(
    ConsoleService consoleService,
    IFileDialogService fileDialogService,
    IConsoleMessenger consoleMessenger)
    : base("Console", consoleMessenger)
  {
    _consoleService = consoleService;
    _fileDialogService = fileDialogService;
    _consoleService.Messages.CollectionChanged += OnMessagesChanged;

    // Initial load
    SyncFullText();

    _consoleService.Log("Console initialized.");
    IsActive = true;  // → OnActivated() → registers ConsoleMessage
  }

  protected override void OnActivated()
  {
    Messenger.Register<ConsoleViewModel, ConsoleMessage>(this, (r, m) => r.Receive(m));
  }

  private void OnMessagesChanged(
    object? sender,
    System.Collections.Specialized.NotifyCollectionChangedEventArgs e
  )
  {
    SyncFullText();
  }

  private void SyncFullText()
  {
    var sb = new StringBuilder();
    foreach (var msg in _consoleService.Messages)
    {
      sb.AppendLine(msg);
    }
    FullText = sb.ToString();
  }

  [RelayCommand]
  private void ClearConsole()
  {
    _consoleService.Clear();
  }

  [RelayCommand]
  private void ZoomIn()
  {
    ConsoleFontSize += 1.0;
  }

  [RelayCommand]
  private void ZoomOut()
  {
    if (ConsoleFontSize > 6.0)
    {
      ConsoleFontSize -= 1.0;
    }
  }

  [RelayCommand]
  private async System.Threading.Tasks.Task ExportLogsAsync()
  {
    var savePath = await _fileDialogService.ShowSaveFileDialogAsync(
      "Export Console Logs",
      "txt",
      new[] { "*.txt" }
    );
    if (!string.IsNullOrEmpty(savePath))
    {
      try
      {
        var text = string.Join(System.Environment.NewLine, Messages);
        using (var stream = new System.IO.StreamWriter(savePath, false))
        {
          await stream.WriteAsync(text);
        }
      }
      catch (System.Exception)
      {
        // Ignore or log
      }
    }
  }

  public void Receive(ConsoleMessage message)
  {
    _consoleService.Log($"[Debug UI] {message.Message}");
  }

  public void Dispose()
  {
    IsActive = false;   // → OnDeactivated() → Messenger.UnregisterAll(this)
    _consoleService.Messages.CollectionChanged -= OnMessagesChanged;
  }
}
