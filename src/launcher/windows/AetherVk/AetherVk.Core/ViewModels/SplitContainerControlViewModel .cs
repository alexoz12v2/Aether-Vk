using AetherVk.Core.Private;
using AetherVk.Core.Types;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using CommunityToolkit.Mvvm.Messaging;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics;

#region Coalescing Notes
// | Concern                    | Owner |
// | -------------------------- | ----- |
// | Gesture interpretation     | VM    |
// | Split vs coalesce decision | VM    |
// | Hysteresis                 | VM    |
// | Layout snapping bounds     | View  |
// | Tree mutation              | VM    |
// | Visual preview             | View  |
#endregion

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
                        Debug.Assert(false);
                        break;
                    default:
                        Debug.Assert(false);
                        break;
                }
            };

            // manually recompute the initial layout
            _LayoutTree.PostConstructionTreeChanged();

            // Register to messages coming from children view models
            // TODO: Unregister on disposal?
            Messenger.Register<SplitSessionBegin>(this, OnSplitSessionBegin);
            Messenger.Register<SplitSessionEnd>(this, OnSplitSessionEnd);
            Messenger.Register<SplitSessionUpdate>(this, OnSplitSessionUpdate);
        }

        #region MessageHandling
        [ObservableProperty]
        public partial SplitSessionState? ActiveSplitSession { get; private set; }

        private Guid? _ActivePageId = null;

        private void OnSplitSessionBegin(object recipient, SplitSessionBegin msg)
        {
            if (_ActivePageId is not null) { return; }
            if (!_ChildReferenceToGuidMapping.TryGetValue(msg.Sender, out Guid id)) { Debug.Assert(false); return; }

            _ActivePageId = id;

            Debug.WriteLine($"[BeginSplitSession] The Point: {msg.Start}");
            ActiveSplitSession = new(
                Source: msg.Sender,
                Start: msg.Start,
                Current: msg.Start,
                Bounds: msg.Bounds,
                Preview: new(
                    SplitPreviewKind.None,
                    Orientation.Horizontal,
                    0f));
        }

        private void OnSplitSessionUpdate(object recipient, SplitSessionUpdate msg)
        {
            if (_ActivePageId is null) { return; }
            if (!_ChildReferenceToGuidMapping.TryGetValue(msg.Sender, out Guid pageId)) { return; }

            SplitPreview updatedPreview = EvaluatePreview(ActiveSplitSession!.Start, msg.Current, ActiveSplitSession.Bounds, pageId);
            ActiveSplitSession = ActiveSplitSession with
            {
                Current = msg.Current,
                Preview = updatedPreview
            };
        }

        private void OnSplitSessionEnd(object recipient, SplitSessionEnd msg)
        {
            if (_ActivePageId is null) { return; }

            Debug.WriteLine($"[EndSplitSession] the Point: {msg.End}, cancelled: {msg.Cancelled}");
            if (!msg.Cancelled)
            {
                CommitIfValid(ActiveSplitSession!);
            }
            ActiveSplitSession = null;
            _ActivePageId = null;
        }
        #endregion
        #region SplitHandlingImpl

        private const double _SplitThreshold = 12; // epx

        // Coalescing behaviour configuration
        private Guid? _ActiveCoalesceTarget = null;
        private SplitNode? _ActiveCoalesceSplitNode = null;
        private Orientation? _ActiveCoalesceOrientation = null;
        private double _CoalesceExitDistance = 0; // hysteresis accumulator
        private Point _CoalesceReentryPoint;

        // Coalescing behaviour constants
        private const double _CoalescingEnterThreshold = 6; // epx past bounds
        private const double _CoalesceExitThreshold = 10; // epx back inside

        // if below split threshold → None
        // determine orientation
        // 
        // if currently coalescing:
        //     if still valid → stay coalescing
        //     else → cancel coalesce(with hysteresis)
        // 
        // else:
        //     if eligible for coalesce → enter coalesce
        //     else → valid split
        private SplitPreview EvaluatePreview(Point start, Point current, RectD bounds, Guid sourceId)
        {
            Debug.WriteLine($"[EvaluatePreview] Bounds: {bounds} current: {current}");
            double dx = Math.Abs(current.X - start.X);
            double dy = Math.Abs(current.Y - start.Y);
            if (Math.Max(dx, dy) < _SplitThreshold)
            {
                Debug.WriteLine("[EvaluatePreview] Resetting Splitting Mode: Too near to Starting point");
                ResetCoalesceState();
                return new(SplitPreviewKind.None, Orientation.Horizontal, 0);
            }

            // if you are coalescing, lock the orientation to give more stability
            Orientation orientation = _ActiveCoalesceTarget is not null
                ? _ActiveCoalesceOrientation!.Value
                : (dx > dy ? Orientation.Vertical : Orientation.Horizontal);
            bool vertical = orientation == Orientation.Vertical;

            double outside = DistanceOutsideBounds(current, bounds, orientation);
            Debug.WriteLine($"[EvaluatePreview] OUTSIDE --------------------------------->> {outside}");

            if (_ActiveCoalesceTarget is not null)
            {
                Debug.WriteLine($"[EvaluatePreview] We have a coalesce Target, outside: {outside}");
                if (outside > 0)
                {
                    // still coalescing in the same state. Stay that way
                    return new(SplitPreviewKind.Coalesce, orientation, 0, Children[_ActiveCoalesceTarget.Value]);
                }
                // inside bounds but you were coalescing: accumulate exit hysteresis
                _CoalesceExitDistance += vertical
                    ? Math.Abs(current.X - _CoalesceReentryPoint.X)
                    : Math.Abs(current.Y - _CoalesceReentryPoint.Y);
                Debug.WriteLine($"[EvaluatePreview] Coalescing Exit Distance Accumulation {_CoalesceExitDistance}");
                if (_CoalesceExitDistance < _CoalesceExitThreshold)
                {
                    // you didn't travel far enough inside the bounds, still coalescing
                    return new(SplitPreviewKind.Coalesce, orientation, 0, Children[_ActiveCoalesceTarget.Value]);
                }
                Debug.WriteLine("[EvaluatePreview] Coalescing: Exiting Coalescing State");
                // Exit coalesce
                ResetCoalesceState();
            }

            if (outside > _CoalescingEnterThreshold)
            {
                if (_LayoutTree.IsSingleton)
                {
                    Debug.WriteLine("[EvaluatePreview] Outside bounds but singleton node. Resetting Splitting mode");
                    ResetCoalesceState();
                    return new(SplitPreviewKind.None, Orientation.Horizontal, 0);
                }
                else
                {
                    // you travelled far enough outside bounds: enter coalesce state
                    Guid? target = FindNearestCoalesceTarget(sourceId, orientation, start, current);
                    if (target is not null)
                    {
                        Debug.WriteLine("[EvaluatePreview] Entering Coalescing State");
                        _ActiveCoalesceTarget = target;
                        _ActiveCoalesceOrientation = orientation;
                        _CoalesceExitDistance = 0;
                        _CoalesceReentryPoint = current;
                        return new(SplitPreviewKind.Coalesce, orientation, 0, Children[_ActiveCoalesceTarget.Value]);
                    }

                    Debug.Assert(false);
                    return new(SplitPreviewKind.Invalid, orientation, 0);
                }
            }

            // if you get here, then it's a split
            float ratio = (float)Math.Clamp(
                vertical
                    ? (current.X - bounds.Left) / (bounds.Right - bounds.Left)
                    : (current.Y - bounds.Top) / (bounds.Bottom - bounds.Top),
                0.1, 0.9);
            Debug.Assert(float.IsFinite(ratio));

            Debug.WriteLine("[EvaluatePreview] We are in the Splitting State");
            ResetCoalesceState();
            return new SplitPreview(SplitPreviewKind.ValidSplit, orientation, ratio);
        }

        private Guid? FindNearestCoalesceTarget(Guid sourceId, Orientation orientation, Point start, Point current)
        {
            Node? node = _LayoutTree.FindNode(n => n is LeafNode l && l.Id.Id == sourceId);
            if (node is not LeafNode source) { return null; }

            DragDir dir = GetDragDir(orientation, start, current);
            SplitNode? split = FindCoalesceSplit(source, orientation, dir);
            if (split is null) { return null; }

            // guard against finding direct ancestor and choosing yourself
            Node sibling = split.First == source ? split.Second : split.First;
            LeafNode target = FindBoundaryLeaf(sibling, orientation, dir);

            return target.Id.Id;
        }

        private void CommitIfValid(SplitSessionState session)
        {
            switch (session.Preview.Kind)
            {

                case SplitPreviewKind.ValidSplit:
                    CommitSplit(session);
                    break;
                case SplitPreviewKind.Coalesce:
                    CommitCoalesce(session);
                    break;
                case SplitPreviewKind.None:
                case SplitPreviewKind.Invalid:
                default:
                    break;
            }
        }

        private void CommitSplit(SplitSessionState session)
        {
            if (!_ChildReferenceToGuidMapping.TryGetValue(session.Source, out Guid pageId)) { Debug.Assert(false); return; }
            _ = _LayoutTree.FindNode(n => n is LeafNode ln && ln.Id.Id == pageId) is LeafNode leaf
                ? _LayoutTree.SplitLeaf(leaf, session.Preview.Orientation, session.Preview.Ratio, new())
                : throw new InvalidOperationException("Leaf not found");
        }

        // - A leaf can only be removed via RemoveLeaf(leaf)
        // - RemoveLeaf:
        // - removes the leaf
        // - promotes its sibling
        // - collapses the parent split
        // - preserves ancestor structure
        //
        // So coalescing = removing the source leaf, not “merging two arbitrary leaves”.
        // 
        // The target leaf is where the content visually snaps,
        // but the structural operation is: remove the dragged leaf.
        //
        // Preconditions for a valid coalesce
        //
        // - By the time CommitCoalesce runs, the following are already guaranteed:
        // - session.Preview.Kind == Coalesce
        // - _ActiveCoalesceTarget was validated during updates
        // - Orientation and direction were locked-in
        // - Target is adjacent in tree topology, not arbitrary
        // So commit logic should not re-evaluate anything.
        private void CommitCoalesce(SplitSessionState session)
        {
            if (!_ChildReferenceToGuidMapping.TryGetValue(session.Source, out Guid sourceId)) { Debug.Assert(false); return; }

            LeafNode sourceLeaf = GetLeaf(sourceId);

            // perform coalesce by removing source (or target?)
            _LayoutTree.RemoveLeaf(sourceLeaf);

            // Note: View Reformatting is up to the view code-behind
        }
        #endregion

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

        #region Helpers
        private void ResetCoalesceState()
        {
            _ActiveCoalesceTarget = null;
            _ActiveCoalesceSplitNode = null;
            _ActiveCoalesceOrientation = null;
            _CoalesceExitDistance = 0;
        }

        private static double DistanceOutsideBounds(Point p, RectD b, Orientation o)
        {
            switch (o)
            {
                case Orientation.Horizontal: // horizontal split → check Y axis
                    if (p.Y < Math.Min(b.Top, b.Bottom))
                    {
                        return Math.Min(b.Top, b.Bottom) - p.Y;
                    }

                    if (p.Y > Math.Max(b.Top, b.Bottom))
                    {
                        return p.Y - Math.Max(b.Top, b.Bottom);
                    }

                    return 0;

                case Orientation.Vertical: // vertical split → check X axis
                    if (p.X < Math.Min(b.Left, b.Right))
                    {
                        return Math.Min(b.Left, b.Right) - p.X;
                    }

                    if (p.X > Math.Max(b.Left, b.Right))
                    {
                        return p.X - Math.Max(b.Left, b.Right);
                    }

                    return 0;

                default:
                    throw new InvalidEnumArgumentException(nameof(o));
            }
        }

        // see if first ancenstor is common, if yes get the parent
        private static SplitNode? FindDirectSplit(LeafNode a, LeafNode b)
        {
            Node? cur = a.Parent;
            if (cur is SplitNode split)
            {
                if (split.First == a && split.Second == b) { return split; }
                if (split.First == b && split.Second == a) { return split; }
            }
            return null;
        }

        // throwing function to get a leaf out of the tree from Id, used when we are sure of its existance
        private LeafNode GetLeaf(Guid id)
        {
            return _LayoutTree.FindNode(n => n is LeafNode lf && lf.Id.Id == id) is LeafNode lf
                ? lf
                : throw new InvalidOperationException("No leaf");
        }
        #endregion

        #region Coalesce
        private enum DragDir { Negative, Positive }
        // Coalesce Step 1: Determine Drag Direction
        private static DragDir GetDragDir(Orientation o, Point start, Point current)
        {
            return o == Orientation.Vertical
                ? (current.X > start.X ? DragDir.Positive : DragDir.Negative)
                : (current.Y > start.Y ? DragDir.Positive : DragDir.Negative);
        }
        // Coalesce Step 2: Find Coalescible Ancestor Split
        private static SplitNode? FindCoalesceSplit(LeafNode source, Orientation orientation, DragDir dir)
        {
            Node child = source;
            Node? cur = source.Parent;
            while (cur is SplitNode split)
            {
                if (split.Orientation == orientation)
                {
                    bool sourceIsFirst = split.First == child;
                    // Direction Rules:
                    // Vertical:
                    // - Drag Right -> source must be first
                    // - Drag Left  -> source must be second
                    // Horizontal:
                    // - Drag Down  -> source must be first
                    // - Drag Up    -> source must be second
                    bool valid = dir == DragDir.Positive ? sourceIsFirst : !sourceIsFirst;
                    if (valid)
                    {
                        return split;
                    }
                }
                child = cur;
                cur = cur.Parent;
            }
            return null;
        }
        // Coalesce Step 3: Resolve Target Leaf (from chosen subtree, choose the leaf which touches the boundary)
        private static LeafNode FindBoundaryLeaf(Node subtree, Orientation o, DragDir dir)
        {
            Node cur = subtree;
            while (cur is SplitNode split)
            {
                if (split.Orientation == o)
                {
                    // keep walking toward boundary
                    cur = dir == DragDir.Positive
                        ? split.First
                        : split.Second;
                }
                else
                {
                    // Orientation mismatch: either child is fine, pick first
                    cur = split.First;
                }
            }
            return (LeafNode)cur;
        }
        #endregion
    }
}
