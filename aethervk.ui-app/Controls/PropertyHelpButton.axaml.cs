using System;
using System.Collections.Generic;
using System.Text.RegularExpressions;
using System.Windows.Input;
using Avalonia;
using Avalonia.Automation;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Threading;

namespace AetherVk.Controls;

/// <summary>
/// A small circled-question-mark icon that opens a flyout with a header and descriptive
/// content when clicked. Designed for use next to property labels to explain computed
/// orbital element values.
/// </summary>
/// <remarks>
/// Opened with the pointer the flyout is "unpinned": it hides <see cref="DismissDelay"/> after the
/// pointer has left both the ? and the panel. A press inside the panel, or opening it with the
/// keyboard, pins it until a click outside or Escape. The optional footer action
/// (<see cref="ActionText"/> + <see cref="ActionCommand"/>) runs the command and closes the flyout.
/// </remarks>
public partial class PropertyHelpButton : UserControl
{
  /// <summary>Accessible name of the ? when <see cref="HelpHeader"/> is empty.</summary>
  public const string DefaultAccessibleName = "Help";

  /// <summary>Suffix of the footer action's AutomationId (<c>&lt;owner id&gt;Action</c>).</summary>
  public const string ActionAutomationIdSuffix = "Action";

  /// <summary>Grace time for the pointer to cross the gap between the ? and the panel.</summary>
  public static readonly TimeSpan DismissDelay = TimeSpan.FromMilliseconds(350);

  // ──────────────────────────────────────────────────────────────────────────
  // Dependency Properties
  // ──────────────────────────────────────────────────────────────────────────

  /// <summary>Bold heading shown at the top of the flyout.</summary>
  public static readonly StyledProperty<string> HelpHeaderProperty = AvaloniaProperty.Register<
    PropertyHelpButton,
    string
  >(nameof(HelpHeader), defaultValue: string.Empty);

  public string HelpHeader
  {
    get => GetValue(HelpHeaderProperty);
    set => SetValue(HelpHeaderProperty, value);
  }

  /// <summary>
  /// Descriptive body text shown below the header in the flyout. Paragraphs are separated by a
  /// blank line (<c>\n\n</c>, or <c>&amp;#x0A;&amp;#x0A;</c> in XAML); a single newline stays a line break.
  /// </summary>
  public static readonly StyledProperty<string> HelpContentProperty = AvaloniaProperty.Register<
    PropertyHelpButton,
    string
  >(nameof(HelpContent), defaultValue: string.Empty);

  public string HelpContent
  {
    get => GetValue(HelpContentProperty);
    set => SetValue(HelpContentProperty, value);
  }

  /// <summary>Footer button label; the button is hidden while this is null.</summary>
  public static readonly StyledProperty<string?> ActionTextProperty = AvaloniaProperty.Register<
    PropertyHelpButton,
    string?
  >(nameof(ActionText));

  public string? ActionText
  {
    get => GetValue(ActionTextProperty);
    set => SetValue(ActionTextProperty, value);
  }

  /// <summary>Command run by the footer button (the flyout closes afterwards).</summary>
  public static readonly StyledProperty<ICommand?> ActionCommandProperty = AvaloniaProperty.Register<
    PropertyHelpButton,
    ICommand?
  >(nameof(ActionCommand));

  public ICommand? ActionCommand
  {
    get => GetValue(ActionCommandProperty);
    set => SetValue(ActionCommandProperty, value);
  }

  public static readonly StyledProperty<object?> ActionCommandParameterProperty = AvaloniaProperty.Register<
    PropertyHelpButton,
    object?
  >(nameof(ActionCommandParameter));

  public object? ActionCommandParameter
  {
    get => GetValue(ActionCommandParameterProperty);
    set => SetValue(ActionCommandParameterProperty, value);
  }

  /// <summary>Maximum width of the flyout text column (default 240).</summary>
  public static readonly StyledProperty<double> FlyoutMaxWidthProperty = AvaloniaProperty.Register<
    PropertyHelpButton,
    double
  >(nameof(FlyoutMaxWidth), defaultValue: 240);

  public double FlyoutMaxWidth
  {
    get => GetValue(FlyoutMaxWidthProperty);
    set => SetValue(FlyoutMaxWidthProperty, value);
  }

  /// <summary>Where the flyout opens relative to the ? (default RightEdgeAlignedTop).</summary>
  public static readonly StyledProperty<PlacementMode> FlyoutPlacementProperty = AvaloniaProperty.Register<
    PropertyHelpButton,
    PlacementMode
  >(nameof(FlyoutPlacement), defaultValue: PlacementMode.RightEdgeAlignedTop);

  public PlacementMode FlyoutPlacement
  {
    get => GetValue(FlyoutPlacementProperty);
    set => SetValue(FlyoutPlacementProperty, value);
  }

  /// <summary><see cref="HelpContent"/> split into paragraphs (bound by the flyout body).</summary>
  public static readonly DirectProperty<PropertyHelpButton, IReadOnlyList<string>> ParagraphsProperty =
    AvaloniaProperty.RegisterDirect<PropertyHelpButton, IReadOnlyList<string>>(nameof(Paragraphs), o => o.Paragraphs);

  private IReadOnlyList<string> _paragraphs = Array.Empty<string>();

  public IReadOnlyList<string> Paragraphs
  {
    get => _paragraphs;
    private set => SetAndRaise(ParagraphsProperty, ref _paragraphs, value);
  }

  // ──────────────────────────────────────────────────────────────────────────
  // State
  // ──────────────────────────────────────────────────────────────────────────

  private readonly Flyout _flyout;
  private readonly Control _panel;
  private readonly Button _actionButton;
  private readonly DispatcherTimer _dismissTimer;
  private bool _pointerOpening;

  /// <summary>True while the flyout is shown.</summary>
  public bool IsFlyoutOpen => _flyout.IsOpen;

  /// <summary>True when the open flyout ignores the pointer leaving (click outside / Escape close it).</summary>
  public bool IsFlyoutPinned { get; private set; }

  /// <summary>The flyout's footer action button (hidden unless <see cref="ActionText"/> is set).</summary>
  public Button ActionButton => _actionButton;

  /// <summary>The inner clickable ? button.</summary>
  public Button TriggerButton => HelpBtn;

  // ──────────────────────────────────────────────────────────────────────────
  // Constructor
  // ──────────────────────────────────────────────────────────────────────────

  public PropertyHelpButton()
  {
    InitializeComponent();

    _flyout = (Flyout)HelpBtn.Flyout!;
    _panel = (Control)_flyout.Content!;
    _actionButton = ActionBtn;
    _dismissTimer = new DispatcherTimer { Interval = DismissDelay };
    _dismissTimer.Tick += (_, _) =>
    {
      _dismissTimer.Stop();
      if (_flyout.IsOpen && !IsFlyoutPinned && !HelpBtn.IsPointerOver && !_panel.IsPointerOver)
        _flyout.Hide();
    };

    // The Flyout popup lives in a detached visual root; DataContext doesn't
    // propagate reliably through the popup boundary via normal inheritance.
    // We subscribe to Opened and push DataContext = this explicitly onto the
    // flyout's Content (the Border), so {Binding HelpHeader/HelpContent} can
    // resolve against this control's own StyledProperties.
    // StyledProperty change notifications still flow, so the bindings stay
    // reactive when HelpHeader/HelpContent are updated by the ViewModel.
    _panel.DataContext = this;
    _flyout.Opened += (_, _) =>
    {
      _panel.DataContext = this;
      // opened with the keyboard (Enter/Space) or programmatically → pinned
      IsFlyoutPinned = !_pointerOpening;
      _pointerOpening = false;
    };
    _flyout.Closed += (_, _) =>
    {
      _dismissTimer.Stop();
      IsFlyoutPinned = false;
    };

    HelpBtn.AddHandler(PointerPressedEvent, (_, _) => _pointerOpening = !_flyout.IsOpen, RoutingStrategies.Tunnel, handledEventsToo: true);
    HelpBtn.PointerEntered += (_, _) => _dismissTimer.Stop();
    HelpBtn.PointerExited += (_, _) => ScheduleDismiss();
    _panel.PointerEntered += (_, _) => _dismissTimer.Stop();
    _panel.PointerExited += (_, _) => ScheduleDismiss();
    _panel.AddHandler(PointerPressedEvent, (_, _) => IsFlyoutPinned = true, RoutingStrategies.Tunnel, handledEventsToo: true);
    _panel.AddHandler(KeyDownEvent, (_, e) =>
    {
      if (e.Key == Key.Escape)
      {
        _flyout.Hide();
        e.Handled = true;
      }
    }, RoutingStrategies.Tunnel);

    _actionButton.Click += (_, _) => RunAction();

    UpdateAccessibleName();
    UpdateParagraphs();
  }

  // ──────────────────────────────────────────────────────────────────────────
  // Behaviour
  // ──────────────────────────────────────────────────────────────────────────

  /// <summary>Opens the flyout at the ?; <paramref name="pinned"/> as if opened by keyboard.</summary>
  public void OpenFlyout(bool pinned = true)
  {
    _pointerOpening = !pinned;
    _flyout.ShowAt(HelpBtn);
  }

  public void CloseFlyout() => _flyout.Hide();

  private void ScheduleDismiss()
  {
    if (!_flyout.IsOpen || IsFlyoutPinned)
      return;
    _dismissTimer.Stop();
    _dismissTimer.Start();
  }

  private void RunAction()
  {
    var command = ActionCommand;
    var parameter = ActionCommandParameter;
    _flyout.Hide();
    if (command != null && command.CanExecute(parameter))
      command.Execute(parameter);
  }

  protected override void OnPropertyChanged(AvaloniaPropertyChangedEventArgs change)
  {
    base.OnPropertyChanged(change);
    if (change.Property == HelpHeaderProperty)
      UpdateAccessibleName();
    else if (change.Property == HelpContentProperty)
      UpdateParagraphs();
    else if (change.Property == FlyoutPlacementProperty)
      _flyout.Placement = FlyoutPlacement;
    else if (change.Property == AutomationProperties.AutomationIdProperty)
    {
      // forwarded so UIA/e2e can find and click the inner button by the same id
      var id = AutomationProperties.GetAutomationId(this);
      AutomationProperties.SetAutomationId(HelpBtn, id);
      AutomationProperties.SetAutomationId(_actionButton, string.IsNullOrEmpty(id) ? null : id + ActionAutomationIdSuffix);
    }
  }

  private void UpdateAccessibleName()
  {
    var header = HelpHeader;
    var name = string.IsNullOrWhiteSpace(header) ? DefaultAccessibleName : header;
    AutomationProperties.SetName(HelpBtn, name);
    ToolTip.SetTip(HelpBtn, string.IsNullOrWhiteSpace(header) ? null : header);
  }

  private static readonly Regex ParagraphBreak = new(@"\n[ \t]*\n\s*", RegexOptions.Compiled);

  /// <summary>
  /// Splits help text into paragraphs at blank lines. CRLF and a literal backslash-n (as typed in
  /// a XAML attribute) count as newlines; single newlines stay inside the paragraph.
  /// </summary>
  public static IReadOnlyList<string> SplitParagraphs(string? text)
  {
    if (string.IsNullOrWhiteSpace(text))
      return Array.Empty<string>();
    var normalized = text!.Replace("\r\n", "\n").Replace("\\n", "\n");
    var parts = new List<string>();
    foreach (var p in ParagraphBreak.Split(normalized))
    {
      var trimmed = p.Trim('\n', ' ', '\t');
      if (trimmed.Length > 0)
        parts.Add(trimmed);
    }
    return parts;
  }

  private void UpdateParagraphs() => Paragraphs = SplitParagraphs(HelpContent);
}
