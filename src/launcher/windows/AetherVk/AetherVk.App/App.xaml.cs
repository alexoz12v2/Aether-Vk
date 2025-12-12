using AetherVk.Core.Interfaces;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using System;

namespace AetherVk.App
{
    // https://albertakhmetov.com/posts/2025/how-to-properly-use-.net-build-in-dependency-injection-with-winui-apps/
#if DISABLE_XAML_GENERATED_MAIN
    internal static class Program
    {
        public static IHost? AppHost { get; private set; }

        public static IServiceProvider Services => AppHost!.Services;

        [STAThread]
        public static void Main(string[] args)
        {
            // this, together with the attribute, triggers the Creation of a Single Threaded COM
            // apartment and the necessary COM objects for the WinRT
            WinRT.ComWrappersSupport.InitializeComWrappers();
            try
            {
                Application.Start((p) =>
                {
                    // Basic Threading context initialization
                    DispatcherQueueSynchronizationContext context = new(DispatcherQueue.GetForCurrentThread());
                    System.Threading.SynchronizationContext.SetSynchronizationContext(context);

                    // Bootstrap our application
                    App app = new();
                    app.UnhandledException += (_, _) => StopHost();

                    // Host configuration for Global Services in our Dependency Injection system
                    AppHost = CreateHost(app);
                });
            }
            finally
            {
                StopHost();
            }
        }

        private static void StopHost()
        {
            if (AppHost is not null)
            {
                AppHost.StopAsync().GetAwaiter().GetResult();
                AppHost.Dispose();
            }
        }

        private static IHost CreateHost(IApp theApp)
        {
            HostApplicationBuilder builder = Host.CreateApplicationBuilder();

            // Singleton App and Main window
            // https://learn.microsoft.com/en-us/dotnet/core/extensions/dependency-injection#service-lifetimes
            // signature with automatic disposal and parameters is the service only
            _ = builder.Services.AddSingleton<IApp>(theApp);
            _ = builder.Services.AddSingleton<MainWindow>();

            return builder.Build();
        }
    }
#endif

    public partial class App : Application, IApp
    {
        public App()
        {
            this.InitializeComponent();
        }

        protected override async void OnLaunched(LaunchActivatedEventArgs args)
        {
            await Program.AppHost!.StartAsync();

            MainWindow theMainWindow = Program.Services.GetRequiredService<MainWindow>();
            theMainWindow.Activate();
        }
    }
}