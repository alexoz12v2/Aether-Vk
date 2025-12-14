using AetherVk.Core.Private;
using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using System.Collections.Specialized;

namespace AetherVk.Core.ViewModels
{
    public sealed partial class SplitContainerControlViewModel : ObservableRecipient
    {
        private readonly LayoutTree _LayoutTree;
        private readonly IAbstractParamFactory<PanelHostPageViewModel, IMessenger> _ChildViewModelFactory;

        [ObservableProperty]
        public partial IReadOnlyList<GridDefinitionData> RowDefinitions { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridDefinitionData> ColumnDefinitions { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridElementData> Pages { get; set; } = [];
        [ObservableProperty]
        public partial IReadOnlyList<GridElementData> Splitters { get; set; } = [];

        // Messenger for split view to communicate with its panels (observable recipient gives it)
        // public StrongReferenceMessenger Messanger { get; }

        // ViewModels of child components are managed here for controlled DI
        public ObservableDictionary<Guid, PanelHostPageViewModel> Children { get; }
        private readonly Dictionary<object, Guid> _ChildReferenceToGuidMapping = [];

        public SplitContainerControlViewModel(
            IAbstractParamFactory<PanelHostPageViewModel, IMessenger> childViewModelFactory) : base(new StrongReferenceMessenger())
        {
            _ChildViewModelFactory = childViewModelFactory;

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
                switch (eventArgs.Action)
                {
                    case NotifyCollectionChangedAction.Add:
                        {
                            if (eventArgs?.NewItems?[0] is KeyValuePair<Guid, PanelHostPageViewModel> kv)
                            {
                                _ChildReferenceToGuidMapping.Add(kv.Value, kv.Key);
                            }
                        }
                        break;
                    case NotifyCollectionChangedAction.Remove:
                        {
                            if (eventArgs?.OldItems?[0] is KeyValuePair<Guid, PanelHostPageViewModel> kv)
                            {
                                _ = _ChildReferenceToGuidMapping.Remove(kv.Value);
                            }
                        }
                        break;
                    case NotifyCollectionChangedAction.Reset:
                        _ChildReferenceToGuidMapping.Clear();
                        break;
                    case NotifyCollectionChangedAction.Replace:
                        {
                            if (eventArgs?.NewItems?[0] is KeyValuePair<Guid, PanelHostPageViewModel> kvNew &&
                                eventArgs?.OldItems?[0] is KeyValuePair<Guid, PanelHostPageViewModel> kvOld)
                            {
                                _ = _ChildReferenceToGuidMapping.Remove(kvOld.Value);
                                _ChildReferenceToGuidMapping.Add(kvNew.Value, kvNew.Key);
                            }
                        }
                        break;
                    case NotifyCollectionChangedAction.Move:
                    default:
                        break;
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
                    Children.Add(node!.Id.Id, _ChildViewModelFactory.Create(Messenger));
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

            RowDefinitions = [.. layout.RowsDef.Select(r => new GridDefinitionData { IsSplitter = r.IsSplitter })];
            ColumnDefinitions = [.. layout.ColumnsDef.Select(c => new GridDefinitionData { IsSplitter = c.IsSplitter })];
            Pages = [.. layout.Pages.Select(p => new GridElementData(
                id: p.Id, row: p.Row, col: p.Column, rowSpan: p.RowSpan, colSpan: p.ColumnSpan))];
            Splitters = [.. layout.Splitters.Select(s => new GridElementData(
                id: s.Id, row: s.Row, col: s.Column, rowSpan: s.RowSpan, colSpan: s.ColumnSpan) { IsSplitter = true, Orientation = s.Orientation })];
        }
    }
}
