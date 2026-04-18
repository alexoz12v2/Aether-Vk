namespace AetherVk.Logic.Messages;

public class DragCompletedMessage
{
    // A simple interface to abstract away the specific view type
    // This allows the logic layer to instruct the view to clean up without taking a direct Avalonia dependency.
    public IDragSourceView View { get; }

    public DragCompletedMessage(IDragSourceView view)
    {
        View = view;
    }
}

public interface IDragSourceView
{
    void ClearDragState();
}
