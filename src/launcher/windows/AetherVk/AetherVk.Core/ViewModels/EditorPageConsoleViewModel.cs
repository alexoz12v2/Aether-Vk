using AetherVk.Core.Interfaces;
using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using System.Collections.ObjectModel;
using System.Collections.Specialized;

namespace AetherVk.Core.ViewModels
{
    internal sealed class BulkObservableCollection<T> : ObservableCollection<T>
    {
        public void ReplaceAll(IEnumerable<T> items)
        {
            Items.Clear();
            foreach (T? item in items)
            {
                Items.Add(item);
            }
            OnCollectionChanged(
                new NotifyCollectionChangedEventArgs(
                    NotifyCollectionChangedAction.Reset));
        }
    }

    // ILogEventHub
    //      ↓
    // [ Internal buffer ]   ← always updated
    //      ↓ (on Activate)
    // [ObservableCollection] ← UI only
    //      ↓
    // ItemsRepeater / ListView
    public sealed partial class EditorPageConsoleViewModel : ObservableObject, IDisposable
    {
        // Internal, non-observable, always on
        private readonly List<LogEventEntry> _Buffer = [];
        private readonly Lock _Lock = new();

        // UI Facing observable projection
        private readonly BulkObservableCollection<LogEventEntry> _Entries = [];
        public ReadOnlyObservableCollection<LogEventEntry> Entries { get; }

        // other stuff
        private const int MaxEntries = 10000;

        private readonly IDisposable _Subscription;
        private readonly IUIDispatcher _Dispatcher;

        // whether or not to propagate logs to UI
        private bool _IsActive = true;

        public EditorPageConsoleViewModel(ILogEventHub hub, IUIDispatcher dispatcher)
        {
            Entries = new(_Entries);
            _Subscription = hub.Subscribe(OnLog);
            _Dispatcher = dispatcher;
        }

        // called by OnNavigatedFrom
        public void Deactivate() { _IsActive = false; }
        // called by OnNavigatedTo
        public void Activate()
        {
            _IsActive = true;
            // bring the UI Copy up to date
            List<LogEventEntry> snapshot;
            lock (_Lock)
            {
                snapshot = [.. _Buffer];
            }
            _Dispatcher.Enqueue(() =>
            {
                _Entries.ReplaceAll(snapshot);
            });
        }

        // note: This doesn't get called automatically, cause page caching needs to be accounted for
        // that's fine, cause we want the console to persist
        public void Dispose()
        {
            _Subscription.Dispose();
        }

        private void OnLog(LogEventEntry entry)
        {
            lock (_Lock)
            {
                _Buffer.Add(entry);
                if (_Buffer.Count > MaxEntries)
                {
                    // TODO: Change to Deque or something similiar?
                    _Buffer.RemoveAt(0);
                }
                if (!_IsActive) { return; }

                EnqueueUIUpdate(entry);
            }
        }

        private void EnqueueUIUpdate(LogEventEntry entry)
        {
            _Dispatcher.Enqueue(() =>
            {
                _Entries.Add(entry);
                if (_Entries.Count > MaxEntries)
                {
                    _Entries.RemoveAt(0);
                }
            });
        }
    }
}
