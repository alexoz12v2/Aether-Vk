using System.Collections.Specialized;
using System.Linq;
using AetherVk.Logic.ViewModels;
using Avalonia.Controls;

namespace AetherVk.Views;

public partial class ConsoleView : UserControl
{
  public ConsoleView()
  {
    InitializeComponent();
    this.DataContextChanged += OnDataContextChanged;
  }

  private void OnDataContextChanged(object? sender, System.EventArgs e)
  {
    if (DataContext is ConsoleViewModel vm)
    {
      vm.Messages.CollectionChanged += OnMessagesCollectionChanged;
    }
  }

  private void OnMessagesCollectionChanged(object? sender, NotifyCollectionChangedEventArgs e)
  {
    // Scroll to the end when a new message is added
    if (e.Action == NotifyCollectionChangedAction.Add)
    {
      var textBox = this.FindControl<TextBox>("ConsoleTextBox");
      if (textBox != null)
      {
        textBox.CaretIndex = textBox.Text?.Length ?? 0;
      }
    }
  }

  private void OnConsolePointerWheelChanged(object? sender, Avalonia.Input.PointerWheelEventArgs e)
  {
    if (e.KeyModifiers.HasFlag(Avalonia.Input.KeyModifiers.Control))
    {
      if (DataContext is ConsoleViewModel vm)
      {
        if (e.Delta.Y > 0)
        {
          vm.ZoomInCommand.Execute(null);
        }
        else if (e.Delta.Y < 0)
        {
          vm.ZoomOutCommand.Execute(null);
        }
        e.Handled = true;
      }
    }
  }
}
