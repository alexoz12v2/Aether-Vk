using AetherVk.App.Helpers;
using Microsoft.UI;
using Microsoft.UI.Windowing; // needed for AppWindow
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using System;
using System.Threading.Tasks;
using Windows.Graphics;
using Windows.Services.Maps;
using WinRT.Interop; // XAML/HWND Interop
using WinUIEx;
using WinUIEx.Messaging;

// File scope namespace (From C# 10.0)
namespace AetherVk.App;

// XAML For winUI and WPF is basically the same, with modified bindings to
// the behind code: <https://learn.microsoft.com/en-us/dotnet/desktop/wpf/xaml/>
// Plus the WinUI 3 gallery

// for controls properties: <https://learn.microsoft.com/en-us/uwp/api/windows.ui.xaml.controls.grid?view=winrt-26100>

// for WinUIEx ❤: https://dotmorten.github.io/WinUIEx/concepts/Splashscreen.html

// for titlebar: https://learn.microsoft.com/en-us/windows/apps/develop/title-bar#full-customization

internal sealed partial class MainWindow : WinUIEx.WindowEx
{
    public MainWindow()
    {
        // Register to SizeChanged of yourself
        SizeChanged += OnWindowSizeChanged;

        // Initialize Component and members
        this.InitializeComponent();
        m_AppWindow = GetAppWindowForCurrentWindow();

        // Properly setup titlebar
        m_AppWindow.TitleBar.ExtendsContentIntoTitleBar = true;
        m_AppWindow.TitleBar.PreferredHeightOption = TitleBarHeightOption.Tall;
        m_AppWindow.TitleBar.ButtonBackgroundColor = Colors.Transparent;
        m_AppWindow.TitleBar.ButtonInactiveBackgroundColor = Colors.Transparent;

        // Convenience features from WinUIEx
        this.MinHeight = 400;
        this.MinWidth = 400;
        this.PersistenceId = "AetherVk-MainWindow";

        // If you want to get Raw window messages with WinUIEx
        WindowManager.Get(this).WindowMessageReceived += MainWindow_WindowMessageReceived;
    }

    private void MainWindow_WindowMessageReceived(object? sender, WindowMessageEventArgs e)
    {

    }

    private void Button_Click(object sender, RoutedEventArgs e)
    {
        Console.Beep();
    }

    private AppWindow GetAppWindowForCurrentWindow()
    {
        IntPtr hWnd = WindowNative.GetWindowHandle(this);
        WindowId wndId = Win32Interop.GetWindowIdFromWindow(hWnd);
        return AppWindow.GetFromWindowId(wndId);
    }

    private void OnWindowSizeChanged(object sender, WindowSizeChangedEventArgs args)
    {
        // Do something
    }

    private readonly AppWindow m_AppWindow;

    private async void Button_OpenDialog(object sender, RoutedEventArgs e)
    {
        ContentDialog dialog = new();
        dialog.XamlRoot = this.Content.XamlRoot;
        dialog.PrimaryButtonText = "Gira a Destra. Sempre destra";
        dialog.CloseButtonText = "Chiudi";
        dialog.IsPrimaryButtonEnabled = true;
        dialog.PrimaryButtonClick += (ContentDialog sender, ContentDialogButtonClickEventArgs args) =>
        {
            Console.Beep();
            args.Cancel = true;
        };
        dialog.Content = "Hello There";
        await dialog.ShowAsync();
    }

    private void Button_OpenWindow(object sender, RoutedEventArgs ev)
    {
        DetailWindow window = new();
        window.IsTitleBarVisible = false;
        window.ExtendsContentIntoTitleBar = true;
        window.SetWindowSize(300, 300);
        window.SetIsAlwaysOnTop(true);
        window.CenterOnScreen();
        window.HideFromTaskbar();

        // initial position
        var appWindow = AppWindow.GetFromWindowId(Win32Interop.GetWindowIdFromWindow(WindowNative.GetWindowHandle(this)));
        window.Move(appWindow.Position.X, appWindow.Position.Y);

        window.Activate();
        this.Closed += (_, _) => window.Close();
        this.PositionChanged += (object? self, PointInt32 pos) => window.Move(pos.X, pos.Y);
    }
}