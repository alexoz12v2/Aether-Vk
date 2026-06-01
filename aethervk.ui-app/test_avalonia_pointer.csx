using System;
using System.Reflection;
using Avalonia;
using Avalonia.Controls;

var windowType = typeof(Window);
foreach (
  var method in windowType.GetMethods(
    BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic
  )
)
{
  if (
    method.Name.Contains("Cursor")
    || method.Name.Contains("Mouse")
    || method.Name.Contains("Pointer")
    || method.Name.Contains("Position")
  )
  {
    Console.WriteLine(method.Name);
  }
}
