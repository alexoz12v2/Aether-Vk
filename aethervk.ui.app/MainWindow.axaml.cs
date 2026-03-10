using System;
using System.Diagnostics;
using Avalonia.Controls;

namespace AetherVk;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        Debug.WriteLine("Hello Window");
        InitializeComponent();
        if (OperatingSystem.IsMacOS())
        {
            NativeMenu.SetMenu(this, CreateMacMenu());

            // hide dock panel
            MainMenu.IsVisible = false;
        }
    }

    private NativeMenu CreateMacMenu()
    {
        var fileMenu = new NativeMenuItem("File")
        {
            Menu =
            [
                new NativeMenuItem("Open"),
                new NativeMenuItem("Save"),
                new NativeMenuItemSeparator(),
                new NativeMenuItem("Exit")
            ]
        };

        var editMenu = new NativeMenuItem("Edit")
        {
            Menu =
            [
                new NativeMenuItem("Copy"),
                new NativeMenuItem("Paste")
            ]
        };

        return [fileMenu, editMenu];
    }
}