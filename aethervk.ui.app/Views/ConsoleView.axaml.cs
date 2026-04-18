using Avalonia.Controls;
using System.Collections.Specialized;
using System.Linq;
using AetherVk.Logic.ViewModels;

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
            var scrollViewer = this.FindControl<ScrollViewer>("ConsoleScrollViewer");
            if (scrollViewer != null)
            {
                scrollViewer.ScrollToEnd();
            }
        }
    }
}
