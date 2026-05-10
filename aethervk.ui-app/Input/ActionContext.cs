using AetherVk.Logic.Input;
using Avalonia;
using Avalonia.Controls;

namespace AetherVk.Input;

public static class ActionContext
{
  public static readonly AttachedProperty<string> IdProperty = AvaloniaProperty.RegisterAttached<
    Control,
    string
  >("Id", typeof(ActionContext));

  public static readonly AttachedProperty<IActionHandler?> HandlerProperty =
    AvaloniaProperty.RegisterAttached<Control, IActionHandler?>("Handler", typeof(ActionContext));

  public static void SetId(Control element, string value) => element.SetValue(IdProperty, value);

  public static string GetId(Control element) => element.GetValue(IdProperty);

  public static void SetHandler(Control element, IActionHandler? value) =>
    element.SetValue(HandlerProperty, value);

  public static IActionHandler? GetHandler(Control element) => element.GetValue(HandlerProperty);
}
