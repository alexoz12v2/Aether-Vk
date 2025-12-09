using Microsoft.UI.Xaml;
using System.Windows.Input;

// the RegisterPropertyChangedCallback method.
// This enables application code to register for change notifications when the specified dependency property is changed

// To reset a value to be the default again,
// and also to enable other participants in precedence that might override the default but not a local value, call the ClearValue method

namespace AetherVk.UserControls.Shared
{
    public sealed class SplitActions
    {
        public static readonly DependencyProperty RequestSplitDependencyProperty =
            DependencyProperty.RegisterAttached(
                "RequestSplit",
                typeof(ICommand),
                typeof(SplitActions),
                new PropertyMetadata(null));
        public static void SetRequestSplit(UIElement element, ICommand command)
        {
            element.SetValue(RequestSplitDependencyProperty, command);
        }
        public static ICommand GetRequestSplit(UIElement element)
        {
            return (ICommand)element.GetValue(RequestSplitDependencyProperty);
        }
    }
}
