using AetherVk.Core.Interfaces;
using AetherVk.Core.Types;
using AetherVk.Core.ViewModels;
using AetherVk.Launch;
using AetherVk.Pages;
using AetherVk.UserControls;
using CommunityToolkit.Mvvm.Messaging;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Serilog;
using Serilog.Filters;
using System;

namespace AetherVk.App
{
    internal sealed class UIDispatcher : IUIDispatcher
    {
        public void Enqueue(Action action)
        {
            // Throw if false?
            _ = DispatcherQueue.GetForCurrentThread().TryEnqueue(() => action());
        }
    }

    internal sealed class PanelHostPageFactory(Func<PanelHostPageViewModel, PanelHostPage> factory) : IPageFactory<PanelHostPageViewModel>
    {
        public object Create(PanelHostPageViewModel viewModel)
        {
            return factory(viewModel);
        }
    }

    // App level resource (not registered as a singleton because it's a App level resource)
    internal sealed class ViewModelLocator
    {
        // grid view model
        public static SplitContainerControlViewModel SplitContainerControlViewModel => Program.Services.GetRequiredService<SplitContainerControlViewModel>();

        // one view model for each editor. Note: there is no view model for the panel host cause it's created via DI
        // TODO remove
        public static EditorPageSplashScreenViewModel EditorPageSplashScreenViewModel => Program.Services.GetRequiredService<EditorPageSplashScreenViewModel>();
        public static EditorPageConsoleViewModel EditorPageConsole => Program.Services.GetRequiredService<EditorPageConsoleViewModel>();
    }


    // https://albertakhmetov.com/posts/2025/how-to-properly-use-.net-build-in-dependency-injection-with-winui-apps/
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
                Log.Information("Application Shutting Down");
                Log.ForContext("DevOnly", true).Information("Application Started Dev only");
                Log.CloseAndFlush();

                AppHost.StopAsync().GetAwaiter().GetResult();
                AppHost.Dispose();
            }
        }

        private static IHost CreateHost(IApp theApp)
        {
            HostApplicationBuilder builder = Host.CreateApplicationBuilder();

            // Logging configuration
            // 0. Add our hub, which intercepts the custom sink
            _ = builder.Services.AddSingleton<ILogEventHub, LogEventHub>();

            // 1. Create Serilog's logger
            Log.Logger = new LoggerConfiguration()
                .MinimumLevel.Debug()
                .WriteTo.Logger(lc =>
                    lc.Filter.ByExcluding(Matching.WithProperty<bool>("DevOnly", p => p))
                      .WriteTo.Sink(new UILogSink(builder.Services.BuildServiceProvider().GetRequiredService<ILogEventHub>())))
                .WriteTo.File(
                    System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData), "AetherVk.App", "log-.txt"),
                    rollingInterval: RollingInterval.Day)
                .CreateLogger();

            // 2. Remove default logging providers given by CreateApplicationBuilder
            _ = builder.Logging.ClearProviders();

            // 3. Plug Serlilog into .NET Logging
            _ = builder.Logging.AddSerilog();

            // -------------- Service Registration ------------------
            _ = builder.Services.AddSingleton<IUIDispatcher, UIDispatcher>();

            // factory for host page view model and for host page
            // TODO use scope
            _ = builder.Services.AddSingleton<IAbstractParamFactory<PanelHostPageViewModel, IMessenger>>(
                sp => new AbstractParamFactory<PanelHostPageViewModel, IMessenger>(
                    (messenger) =>
                    {
                        // resolve other dependencies here
                        // var someService = sp.GetRequiredService<ISomeService>();
                        return new PanelHostPageViewModel(messenger);
                    }));

            // view model and page for grid splitter
            _ = builder.Services.AddTransient<SplitContainerControlViewModel>();

            // view models and page for editor panels
            _ = builder.Services.AddTransient<EditorPageSplashScreenViewModel>();
            _ = builder.Services.AddTransient<EditorPageConsoleViewModel>();

            // Singleton App and Main window
            // https://learn.microsoft.com/en-us/dotnet/core/extensions/dependency-injection#service-lifetimes
            // signature with automatic disposal and parameters is the service only
            _ = builder.Services.AddSingleton<IApp>(theApp);
            _ = builder.Services.AddSingleton<MainWindow>();

            return builder.Build();
        }
    }

    public partial class App : Application, IApp
    {
        public App()
        {
            this.InitializeComponent();
        }

        protected override async void OnLaunched(LaunchActivatedEventArgs args)
        {
            await Program.AppHost!.StartAsync();
            Log.Information("Application Started");
            Log.ForContext("DevOnly", true).Information("Application Started Dev only");

            MainWindow theMainWindow = Program.Services.GetRequiredService<MainWindow>();
            theMainWindow.Activate();
        }
    }
}