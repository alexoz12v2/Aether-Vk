using AetherVk.Core.Private;
using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using System.Collections.Specialized;

namespace AetherVk.Core.ViewModels
{
    public sealed class GridElementViewModel(Guid id, int row, int col, int rowSpan, int colSpan)
    {
        public Guid Id { get; init; } = id;
        public int Row { get; init; } = row;
        public int Column { get; init; } = col;
        public int RowSpan { get; init; } = rowSpan;
        public int ColumnSpan { get; init; } = colSpan;
        public bool IsSplitter { get; init; } = false;
        // has meaning only if splitter
        public Orientation Orientation { get; init; } = Orientation.Horizontal;
    }

    public sealed class GridDefinitionViewModel
    {
        public bool IsSplitter { get; init; } = false;
    }

    public sealed class SplitCommandData(GridElementViewModel page, float ratio, Orientation orientation)
    {
        public GridElementViewModel Page { get; } = page ?? throw new ArgumentNullException(nameof(page));
        public float Ratio { get; } = ratio is <= 1 and >= 0 ? ratio : throw new ArgumentOutOfRangeException(nameof(ratio));
        public Orientation Orientation { get; } = orientation;
    }

    public sealed partial class SplitContainerControlViewModel : ObservableRecipient
    {
        private readonly LayoutTree _LayoutTree;

        [ObservableProperty]
        public partial IReadOnlyList<GridDefinitionViewModel> RowDefinitions { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridDefinitionViewModel> ColumnDefinitions { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridElementViewModel> Pages { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridElementViewModel> Splitters { get; set; } = [];

        // Messenger for split view to communicate with its panels (observable recipient gives it)
        // public StrongReferenceMessenger Messanger { get; }

        // ViewModels of child components are managed here for controlled DI
        public ObservableDictionary<Guid, PanelHostPageViewModel> Children { get; }

        public SplitContainerControlViewModel() : base(new StrongReferenceMessenger())
        {
            // Create child related objectsw
            Children = [];
            // Messanger = new();

            // initial layout with single leaf
            _LayoutTree = new LayoutTree(new LeafNode());

            // register for changes 
            _LayoutTree.TreeChanged += OnTreeChanged;
            Children.CollectionChanging += (sender, eventArgs) =>
            {
                // recompute layout _before_ a removal
                if (eventArgs.Action == NotifyCollectionChangedAction.Remove)
                {
                    RecomputeLayout();
                }
            };
            Children.CollectionChanged += (sender, eventArgs) =>
            {
                // recompute layout _after_ an addition
                if (eventArgs.Action == NotifyCollectionChangedAction.Add)
                {
                    RecomputeLayout();
                }
            };

            // manually recompute the initial layout
            _LayoutTree.PostConstructionTreeChanged();
        }

        // TODO: To be used when messages are written
        public void UnregisterChildFromMessenger<TMessage>(object recipient) where TMessage : class
        {
            Messenger.Unregister<TMessage>(recipient);
        }

        // TODO add more customization to command
        [RelayCommand]
        private void Split(SplitCommandData data)
        {
            if (data is null) { throw new ArgumentNullException(nameof(data)); }
            _ = _LayoutTree.FindNode(n => (n is LeafNode) && n.Id.Id == data.Page.Id)
                is LeafNode leaf
                ? _LayoutTree.SplitLeaf(leaf, data.Orientation, data.Ratio, new())
                : throw new InvalidOperationException("Couldn't find requested Node");
        }

        private void OnTreeChanged(TreeChangedAction action, Node? node)
        {
            switch (action)
            {
                case TreeChangedAction.Add:
                    Children.Add(node!.Id.Id, new(Messenger));
                    break;
                case TreeChangedAction.Remove:
                    _ = Children.Remove(node!.Id.Id);
                    break;
                case TreeChangedAction.RatioChanged:
                    RecomputeLayout();
                    break;
                default:
                    break;
            }
        }

        private void RecomputeLayout()
        {
            Layout layout = _LayoutTree.ComputeLayout();

            RowDefinitions = [.. layout.RowsDef.Select(r => new GridDefinitionViewModel { IsSplitter = r.IsSplitter })];
            ColumnDefinitions = [.. layout.ColumnsDef.Select(c => new GridDefinitionViewModel { IsSplitter = c.IsSplitter })];
            Pages = [.. layout.Pages.Select(p => new GridElementViewModel(
                id: p.Id, row: p.Row, col: p.Column, rowSpan: p.RowSpan, colSpan: p.ColumnSpan))];
            Splitters = [.. layout.Splitters.Select(s => new GridElementViewModel(
                id: s.Id, row: s.Row, col: s.Column, rowSpan: s.RowSpan, colSpan: s.ColumnSpan) { IsSplitter = true, Orientation = s.Orientation })];
        }
    }
}
