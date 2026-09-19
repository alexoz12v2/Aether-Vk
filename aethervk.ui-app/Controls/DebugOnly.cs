using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Templates;
using Avalonia.Metadata;

namespace AetherVk.UI.Controls;

/// <summary>
/// A decorator that defers the creation of its child controls.
/// In Release builds, the template is parsed, but the inner controls are never built and `Child` remains `null`.
/// </summary>
public class DebugOnly : Decorator
{
    public static readonly StyledProperty<ITemplate<Control?>?> ContentTemplateProperty =
        AvaloniaProperty.Register<DebugOnly, ITemplate<Control?>?>(nameof(ContentTemplate));

    [Content]
    public ITemplate<Control?>? ContentTemplate
    {
        get => GetValue(ContentTemplateProperty);
        set => SetValue(ContentTemplateProperty, value);
    }

#if DEBUG
    protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
    {
        base.OnPropertyChanged(change);

        if (change.Property == ContentTemplateProperty)
        {
            Child = ContentTemplate?.Build();
        }
    }
#endif
}
