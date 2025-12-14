using AetherVk.Core.Types;
using AetherVk.Core.ViewModels;
using AetherVk.Pages;
using AetherVk.UserControls.Shared;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.WinUI.Controls;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Markup;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.Linq;
using System.Windows.Input;

// the RegisterPropertyChangedCallback method.
// This enables application code to register for change notifications when the specified dependency property is changed

// To reset a value to be the default again,
// and also to enable other participants in precedence that might override the default but not a local value, call the ClearValue method

namespace AetherVk.UserControls
{
    public sealed partial class SplitContainerControl : UserControl
    {
        private SplitContainerControlViewModel ViewModel => (SplitContainerControlViewModel)DataContext;

        private readonly Dictionary<Guid, UIElement> ContainerPages = [];
        private readonly Dictionary<Guid, GridSplitter> ContainerSplitters = [];

        // an input or output is a dependency property, which you can set with GeneratedDependencyGenerator
        public SplitContainerControl()
        {
            InitializeComponent();
            Loaded += SplitContainerControl_OnLoaded;
            Unloaded += SplitContainerControl_OnUnloaded;
        }

        private void SplitContainerControl_OnUnloaded(object sender, RoutedEventArgs e)
        {
            ViewModel.PropertyChanged -= OnViewModelPropertyChanged;
        }

        private void SplitContainerControl_OnLoaded(object sender, RoutedEventArgs e)
        {
            RebuildColumns(ViewModel.ColumnDefinitions);
            RebuildRows(ViewModel.RowDefinitions);
            RebuildPages(ViewModel.Pages);
            RebuildSplitters(ViewModel.Splitters);

            ViewModel.PropertyChanged += OnViewModelPropertyChanged;
        }

        private void OnViewModelPropertyChanged(object? sender, PropertyChangedEventArgs ev)
        {
            switch (ev.PropertyName)
            {
                case nameof(ViewModel.ColumnDefinitions):
                    RebuildColumns(ViewModel.ColumnDefinitions);
                    break;
                case nameof(ViewModel.RowDefinitions):
                    RebuildRows(ViewModel.RowDefinitions);
                    break;
                case nameof(ViewModel.Pages):
                    RebuildPages(ViewModel.Pages);
                    break;
                case nameof(ViewModel.Splitters):
                    RebuildSplitters(ViewModel.Splitters);
                    break;
                default:
                    break;
            }
        }

        // Grid.ColumnDefinitions and Grid.RowDefinitions are not DependencyProperty, meaning we can't bind to them to
        // modify them dynamically directly.
        // Instead, since our ViewModel stores the current layout, we can react to changes in the layout inside our view model,
        // and manually modify the Grid's properties as we need them
        private void RebuildColumns(IReadOnlyList<GridDefinitionData> cols)
        {
            Container.ColumnDefinitions.Clear();
            foreach (GridDefinitionData c in cols)
            {
                Container.ColumnDefinitions.Add(new ColumnDefinition
                {
                    Width = c.IsSplitter ? new GridLength(8) : new GridLength(1, GridUnitType.Star),
                    MinWidth = c.IsSplitter ? 0 : 128
                });
            }
        }

        private void RebuildRows(IReadOnlyList<GridDefinitionData> rows)
        {
            Container.RowDefinitions.Clear();
            foreach (GridDefinitionData r in rows)
            {
                Container.RowDefinitions.Add(new RowDefinition
                {
                    Height = r.IsSplitter ? new GridLength(8) : new GridLength(1, GridUnitType.Star),
                    MinHeight = r.IsSplitter ? 0 : 128
                });
            }
        }

        private void RebuildPages(IReadOnlyList<GridElementData> pages)
        {
            HashSet<Guid> keep = [.. pages.Select(p => p.Id)];

            foreach (Guid id in ContainerPages.Keys.Where(id => !keep.Contains(id)))
            {
                // TODO: This is the place in which you unregister messages
                _ = Container.Children.Remove(ContainerPages[id]);
                _ = ContainerPages.Remove(id);

            }

            // 2) Add new pages or update existing pages (row/col/span)
            foreach (GridElementData pageVm in pages)
            {
                if (!ContainerPages.TryGetValue(pageVm.Id, out UIElement? pageElement))
                {
                    // PanelHostPage thePage = _ChildPanelFactory.Create(childViewModel);
                    ContentPresenter presenter = new()
                    {
                        ContentTemplate = (DataTemplate)Resources["ContentPresenterDataTemplate"],
                        Content = ViewModel.Children[pageVm.Id]
                    };

                    SetGrid(presenter, pageVm);

                    // AttachDependencyProperties(thePage);

                    ContainerPages.Add(pageVm.Id, presenter);
                    Container.Children.Add(presenter);
                }
                else
                {
                    SetGrid(pageElement, pageVm);
                }
            }
        }

        // TODO Existance check
        private static void SetGrid(UIElement element, GridElementData data)
        {
            element.SetValue(Grid.RowProperty, data.Row);
            element.SetValue(Grid.ColumnProperty, data.Column);
            element.SetValue(Grid.RowSpanProperty, data.RowSpan);
            element.SetValue(Grid.ColumnSpanProperty, data.ColumnSpan);
        }

        // TODO probably to remove in favour of templated controls and view model messaging
        // private static void AttachDependencyProperties(PanelHostPage thePage)
        // {
        //     // set attached dependency property
        //     // TODO: set the true command
        //     ICommand debugCommand = new RelayCommand<string>(theString =>
        //     {
        //         Debug.WriteLine($"Hello There From the splitter! {theString}");
        //     });
        //     SplitActions.SetRequestSplit(thePage, debugCommand);
        // }

        private void RebuildSplitters(IReadOnlyList<GridElementData> splitters)
        {
            // 1) Remove splitters that are no longer present
            HashSet<Guid> splitterIdsToKeep = [.. splitters.Select(s => s.Id)];
            List<Guid> splitterKeysToRemove = [.. ContainerSplitters.Keys.Where(k => !splitterIdsToKeep.Contains(k))];

            foreach (Guid id in splitterKeysToRemove)
            {
                if (ContainerSplitters.TryGetValue(id, out GridSplitter? splitter))
                {
                    // TODO: This is the place in which you unregister messages
                    _ = Container.Children.Remove(splitter);
                    _ = ContainerSplitters.Remove(id);
                }
            }

            // 2) Add new splitters or update existing ones
            foreach (GridElementData sVm in splitters)
            {
                if (!ContainerSplitters.TryGetValue(sVm.Id, out GridSplitter? existingSplitter))
                {
                    // Why XAML 😡: https://stackoverflow.com/questions/5755455/how-to-set-control-template-in-code
                    // Basically, Templated Controls have no template by default, hence you give it to them
                    string splitterXaml =
                        "<cu:GridSplitter xmlns='http://schemas.microsoft.com/winfx/2006/xaml/presentation'"
                        + " xmlns:x='http://schemas.microsoft.com/winfx/2006/xaml'"
                        + " xmlns:cu='using:CommunityToolkit.WinUI.Controls'"
                        + " ResizeBehavior='BasedOnAlignment'"
                        + " ResizeDirection='Columns'"
                        + " Background='Red'"
                        + " HorizontalAlignment='Stretch'"
                        + " VerticalAlignment='Stretch'"
                        + " Width='8'"
                        + " Height='8' />";

                    GridSplitter splitter = (GridSplitter)XamlReader.LoadWithInitialTemplateValidation(splitterXaml);

                    SetGrid(splitter, sVm);

                    ContainerSplitters.Add(sVm.Id, splitter);
                    Container.Children.Add(splitter);
                }
                else
                {
                    // update existing splitter attached properties if needed
                    SetGrid(existingSplitter, sVm);

                    // if orientation changed, update ResizeDirection
                    GridSplitter.GridResizeDirection newDirection = sVm.Orientation == AetherVk.Core.Types.Orientation.Horizontal
                        ? GridSplitter.GridResizeDirection.Rows
                        : GridSplitter.GridResizeDirection.Columns;

                    if (existingSplitter.ResizeDirection != newDirection)
                    {
                        existingSplitter.ResizeDirection = newDirection;
                    }
                }
            }
        }
    }
}
