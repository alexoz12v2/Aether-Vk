using Avalonia;
using Avalonia.Controls;

namespace AetherVk.Controls;

/// <summary>
/// A small circled-question-mark icon that opens a flyout with a header and descriptive
/// content when clicked. Designed for use next to property labels to explain computed
/// orbital element values.
/// </summary>
public partial class PropertyHelpButton : UserControl
{
  // ──────────────────────────────────────────────────────────────────────────
  // Dependency Properties
  // ──────────────────────────────────────────────────────────────────────────

  /// <summary>Bold heading shown at the top of the flyout.</summary>
  public static readonly StyledProperty<string> HelpHeaderProperty =
    AvaloniaProperty.Register<PropertyHelpButton, string>(
      nameof(HelpHeader),
      defaultValue: string.Empty);

  public string HelpHeader
  {
    get => GetValue(HelpHeaderProperty);
    set => SetValue(HelpHeaderProperty, value);
  }

  /// <summary>Descriptive body text shown below the header in the flyout.</summary>
  public static readonly StyledProperty<string> HelpContentProperty =
    AvaloniaProperty.Register<PropertyHelpButton, string>(
      nameof(HelpContent),
      defaultValue: string.Empty);

  public string HelpContent
  {
    get => GetValue(HelpContentProperty);
    set => SetValue(HelpContentProperty, value);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Constructor
  // ──────────────────────────────────────────────────────────────────────────

  public PropertyHelpButton()
  {
    InitializeComponent();

    // The Flyout popup lives in a detached visual root; DataContext doesn't
    // propagate reliably through the popup boundary via normal inheritance.
    // We subscribe to Opened and push DataContext = this explicitly onto the
    // flyout's Content (the Border), so {Binding HelpHeader/HelpContent} can
    // resolve against this control's own StyledProperties.
    // StyledProperty change notifications still flow, so the bindings stay
    // reactive when HelpHeader/HelpContent are updated by the ViewModel.
    if (HelpBtn.Flyout is Flyout flyout)
    {
      flyout.Opened += (_, _) =>
      {
        if (flyout.Content is Control content)
          content.DataContext = this;
      };
    }
  }
}
