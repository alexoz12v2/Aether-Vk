using Microsoft.UI;
using Microsoft.UI.Windowing; // needed for AppWindow
using System;
using WinRT.Interop; // XAML/HWND Interop
using WinUIEx;
using WinUIEx.Messaging;

// XAML For winUI and WPF is basically the same, with modified bindings to
// the behind code: <https://learn.microsoft.com/en-us/dotnet/desktop/wpf/xaml/>
// Plus the WinUI 3 gallery

// for controls properties: <https://learn.microsoft.com/en-us/uwp/api/windows.ui.xaml.controls.grid?view=winrt-26100>

// for WinUIEx ❤: https://dotmorten.github.io/WinUIEx/concepts/Splashscreen.html

// for titlebar: https://learn.microsoft.com/en-us/windows/apps/develop/title-bar#full-customization

namespace AetherVk.App
{
    internal sealed partial class MainWindow : WinUIEx.WindowEx
    {
        public MainWindow()
        {
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

            // Instantiate the main page inside the window's frame
            _ = ShellFrame.Navigate(typeof(Pages.ShellPage));
        }

        private void MainWindow_WindowMessageReceived(object? sender, WindowMessageEventArgs e)
        {

        }

        private AppWindow GetAppWindowForCurrentWindow()
        {
            IntPtr hWnd = WindowNative.GetWindowHandle(this);
            WindowId wndId = Win32Interop.GetWindowIdFromWindow(hWnd);
            return AppWindow.GetFromWindowId(wndId);
        }

        private readonly AppWindow m_AppWindow;
    }
}