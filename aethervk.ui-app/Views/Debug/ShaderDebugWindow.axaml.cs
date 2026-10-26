using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using Avalonia.Controls;
using Avalonia.Threading;

namespace AetherVk.Views.Debug;

public partial class ShaderDebugWindow : Window
{
    private readonly ConcurrentQueue<string> _messageQueue = new();
    private readonly Timer _flushTimer;
    private int _isFlushing;
    private const int MaxLogLines = 2000;
    
    private TextBox? _logTextBox;
    private readonly List<string> _history = new();

    public ShaderDebugWindow()
    {
        InitializeComponent();
        _logTextBox = this.FindControl<TextBox>("LogText");
        _flushTimer = new Timer(FlushMessages, null, 100, 100);
    }

    public void AppendLog(string message)
    {
        _messageQueue.Enqueue(message);
    }

    private void FlushMessages(object? state)
    {
        if (Interlocked.CompareExchange(ref _isFlushing, 1, 0) != 0) return;
        try
        {
            if (_messageQueue.IsEmpty) return;

            var batch = new List<string>();
            while (_messageQueue.TryDequeue(out var msg)) batch.Add(msg);
            if (batch.Count == 0) return;

            Dispatcher.UIThread.Post(() =>
            {
                _history.AddRange(batch);
                if (_history.Count > MaxLogLines)
                {
                    _history.RemoveRange(0, _history.Count - MaxLogLines);
                }
                
                if (_logTextBox != null)
                {
                    _logTextBox.Text = string.Join("\n", _history);
                    _logTextBox.CaretIndex = _logTextBox.Text.Length;
                }
            });
        }
        finally
        {
            Interlocked.Exchange(ref _isFlushing, 0);
        }
    }

    protected override void OnClosed(EventArgs e)
    {
        _flushTimer.Dispose();
        base.OnClosed(e);
    }
}
