using System;
using AetherVk.Logic.Services;
using AetherVk.Logic.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace AetherVk.Services
{
    public class ViewModelFactory : IViewModelFactory
    {
        private readonly IServiceProvider _serviceProvider;

        public ViewModelFactory(IServiceProvider serviceProvider)
        {
            _serviceProvider = serviceProvider;
        }

        public object CreateViewModel(string tabType)
        {
            var runtimeService = _serviceProvider.GetRequiredService<NativeRuntimeService>();
            var stateManager = _serviceProvider.GetRequiredService<SceneStateManager>();
            
            // Targeted spawn: fetch the first active scene or fallback
            var activeScene = System.Linq.Enumerable.FirstOrDefault(stateManager.AllScenes);
            ulong targetSceneId = activeScene != null ? activeScene.SceneId : 1UL;

            return tabType switch
            {
                "UITestPanel" => ActivatorUtilities.CreateInstance<UITestPanelViewModel>(_serviceProvider),
                "Console" => ActivatorUtilities.CreateInstance<ConsoleViewModel>(_serviceProvider),
                "DebugUI" => ActivatorUtilities.CreateInstance<DebugUiViewModel>(_serviceProvider),
                "HorizonJpl" => ActivatorUtilities.CreateInstance<HorizonJplViewModel>(_serviceProvider),
                "Outline" => ActivatorUtilities.CreateInstance<OutlineViewModel>(_serviceProvider, targetSceneId),
                "Properties" => ActivatorUtilities.CreateInstance<PropertiesViewModel>(_serviceProvider, targetSceneId),
                "Timeline" => ActivatorUtilities.CreateInstance<TimelineViewModel>(_serviceProvider),
                "Almanac" => ActivatorUtilities.CreateInstance<AlmanacExplorerViewModel>(_serviceProvider),
                "Viewport3D" => ActivatorUtilities.CreateInstance<Viewport3DViewModel>(_serviceProvider),
                _ => throw new ArgumentException($"Cannot find or construct type {tabType}")
            };
        }
    }
}
