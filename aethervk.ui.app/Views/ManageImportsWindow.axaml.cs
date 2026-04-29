using Avalonia.Controls;
using Avalonia.Interactivity;
using AetherVk.Logic.ViewModels;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Views;

public partial class ManageImportsWindow : Window
{
    public ManageImportsWindow()
    {
        InitializeComponent();

        WeakReferenceMessenger.Default.Register<OpenSpawnMeshDialogMessage>(this, async (r, m) =>
        {
            var dialog = new SpawnMeshDialogWindow
            {
                DataContext = new SpawnMeshViewModel(m.Model.Name + " Instance")
            };
            
            var result = await dialog.ShowDialog<bool>(this);
            if (result && dialog.DataContext is SpawnMeshViewModel vm)
            {
                var runtime = ServiceLocator.Provider?.GetService(typeof(NativeRuntimeService)) as NativeRuntimeService;
                if (runtime != null) {
                    await runtime.SpawnModelInstanceAsync(m.Model.Id, vm.EntityName, vm.PosX, vm.PosY, vm.PosZ);
                }
            }
        });
    }

    protected override void OnClosed(System.EventArgs e)
    {
        WeakReferenceMessenger.Default.Unregister<OpenSpawnMeshDialogMessage>(this);
        base.OnClosed(e);
    }

    private void CloseButton_Click(object? sender, RoutedEventArgs e)
    {
        Close();
    }
}
