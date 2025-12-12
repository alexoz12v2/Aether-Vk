using AetherVk.Core.Types;

namespace AetherVk.Core.Private
{
    internal sealed class NodeId : IEquatable<NodeId>
    {
        public Guid Id { get; }

        public NodeId() { Id = Guid.NewGuid(); }

        public override string ToString() { return Id.ToString(); }

        public override int GetHashCode() { return Id.GetHashCode(); }

        public override bool Equals(object? obj) { return obj != null && GetType() == obj.GetType() && Equals(obj as NodeId); }

        public static bool operator ==(NodeId? first, NodeId? other) { return (first is null && other is null) || (first is not null && first.Equals(other)); }

        public static bool operator !=(NodeId? first, NodeId? other) { return !(first == other); }

        public bool Equals(NodeId? other) { return Id == other?.Id; }
    }

    // All measures here are in device independent "effective pixels"
    internal readonly struct Rect(double x, double y, double width, double height)
    {
        public double X { get; init; } = x;
        public double Y { get; init; } = y;
        public double Width { get; init; } = width;
        public double Height { get; init; } = height;

        public override string ToString() { return $"Rect({X:F1},{Y:F1},{Width:F1},{Height:F1})"; }
    }

    internal abstract class Node : IEquatable<Node>
    {
        public NodeId Id { get; } = new NodeId();
        public Node? Parent { get; internal set; } = null;

        public override bool Equals(object? obj) { return Equals(obj as Node); }

        public bool Equals(Node? other) { return other is not null && Id == other.Id; }

        public override int GetHashCode() { return Id.GetHashCode(); }

        public static bool operator ==(Node? a, Node? b) { return ReferenceEquals(a, b) || (a is not null && a.Equals(b)); }

        public static bool operator !=(Node? a, Node? b) { return !(a == b); }
    }

    internal sealed class LeafNode : Node
    {
        // panel-specific data
        public double MinWidth { get; set; } = 24;
        public double MinHeight { get; set; } = 24;

        // Additional Data
        public object? Tag { get; set; } = null;

        public LeafNode() { }
    }

    internal sealed class SplitNode : Node
    {
        public Orientation Orientation { get; set; }
        public Node First { get; private set; } // left or top
        public Node Second { get; private set; } // right or bottom

        // Setter should be called only from Tree class
        public double Ratio
        {
            get;
            internal set
            {
                if (value is > 1 or < 0)
                {
                    throw new ArgumentOutOfRangeException(nameof(Ratio));
                }
                field = value;
            }
        }

        public SplitNode(Orientation orientation, Node first, Node second, double ratio = 0.5)
        {
            Orientation = orientation;
            First = first ?? throw new ArgumentNullException(nameof(first));
            Second = second ?? throw new ArgumentNullException(nameof(second));
            Ratio = ratio;
            // WARNING: Parent wiring done by Tree class
        }

        public void ReplaceChild(Node oldChild, Node newChild)
        {
            if (oldChild == null) { throw new ArgumentNullException(nameof(oldChild)); }
            if (newChild == null) { throw new ArgumentNullException(nameof(newChild)); }

            if (First == oldChild)
            {
                First = newChild;
                newChild.Parent = this;
            }
            else if (Second == oldChild)
            {
                Second = newChild;
                newChild.Parent = this;
            }
            else { throw new ArgumentException($"{nameof(oldChild)} was not found in this split node"); }
        }
    }

    internal readonly struct LayoutElementDefinition()
    {
        public readonly bool IsSplitter { get; init; } = false;
    }

    internal readonly struct LayoutElementSpecification(Guid id, int row, int column, int rowSpan, int columnSpan)
    {
        public readonly int Row { get; } = row;
        public readonly int Column { get; } = column;
        public readonly int RowSpan { get; } = rowSpan;
        public readonly int ColumnSpan { get; } = columnSpan;
        public readonly Guid Id { get; } = id;
        // has meaning only for splitters
        public Orientation Orientation { get; init; } = Orientation.Horizontal;
    }

    internal sealed class Layout
    {
        public IList<LayoutElementDefinition> ColumnsDef { get; } = [];
        public IList<LayoutElementDefinition> RowsDef { get; } = [];
        public IList<LayoutElementSpecification> Pages { get; } = [];
        public IList<LayoutElementSpecification> Splitters { get; } = [];
    }

    internal enum TreeChangedAction
    {
        Add, Remove, RatioChanged
    }

    internal sealed class LayoutTree(LeafNode initialRoot) : IDisposable
    {
        public Rect Bounds { get; set; } = new Rect(0, 0, 1, 1);

        private Node _Root = initialRoot ?? throw new ArgumentNullException(nameof(initialRoot));
        private readonly ReaderWriterLockSlim _Lock = new();

        // Fired every time the tree structure changed
        public event Action<TreeChangedAction, Node?>? TreeChanged;

        // Ugly workaround such that you can react with the event at first node
        public void PostConstructionTreeChanged()
        {
            TreeChanged?.Invoke(TreeChangedAction.Add, _Root);
        }

        public Node? FindNode(Func<Node, bool> predicate)
        {
            _Lock.EnterReadLock();
            try
            {
                return FindNodeInternal(_Root, predicate);
            }
            finally { _Lock.ExitReadLock(); }
        }

        // split a leaf into two leaves by inserting a split node in place of the leaf. Returns the newly create.
        // TODO: Event on split
        public SplitNode SplitLeaf(LeafNode leaf, Orientation orientation, double firstRatio, LeafNode newLeaf)
        {
            if (leaf == null) { throw new ArgumentNullException(nameof(leaf)); }
            if (firstRatio is < 0 or > 1) { throw new ArgumentOutOfRangeException(nameof(firstRatio)); }
            if (newLeaf == null) { newLeaf = new(); }

            _Lock.EnterUpgradeableReadLock();
            try
            {
                if (FindNodeInternal(_Root, n => n == leaf) is null) { throw new InvalidOperationException($"{nameof(leaf)} not found in ${nameof(_Root)}"); }

                SplitNode split = new(orientation, first: leaf, second: newLeaf, firstRatio);

                _Lock.EnterWriteLock();
                try
                {
                    // replace existing leaf parent with split and wire up split
                    if (leaf.Parent is SplitNode parentSplit)
                    {
                        parentSplit.ReplaceChild(leaf, split);
                        split.Parent = parentSplit;
                    }
                    else // this is the new root
                    {
                        _Root = split;
                        split.Parent = null;
                    }

                    // Wire the existing leaf and the new leaf to the split
                    leaf.Parent = split;
                    newLeaf.Parent = split;
                }
                finally { _Lock.ExitWriteLock(); }

                // notify
                TreeChanged?.Invoke(TreeChangedAction.Add, newLeaf);
                return split;
            }
            finally { _Lock.ExitUpgradeableReadLock(); }
        }

        public void UpdateSplitter(SplitNode split, double newRatio)
        {
            if (split == null) { throw new ArgumentNullException(nameof(split)); }
            if (newRatio is < 0 or > 1) { throw new ArgumentOutOfRangeException(nameof(newRatio)); }

            _Lock.EnterUpgradeableReadLock();
            try
            {
                if (FindNodeInternal(_Root, n => n == split) is null) { throw new InvalidOperationException($"{nameof(split)} not found in ${nameof(_Root)}"); }
                _Lock.EnterWriteLock();
                try
                {
                    split.Ratio = newRatio;
                }
                finally { _Lock.ExitWriteLock(); }

                TreeChanged?.Invoke(TreeChangedAction.RatioChanged, null);
            }
            finally { _Lock.ExitUpgradeableReadLock(); }
        }

        // remove a leaf. it its simpling exists, the sibling replaces the parent split. if the root is the only leaf node,
        // throw
        public void RemoveLeaf(LeafNode leaf)
        {
            if (leaf == null) { throw new ArgumentNullException(nameof(leaf)); }
            _Lock.EnterUpgradeableReadLock();
            try
            {
                if (FindNodeInternal(_Root, n => n == leaf) is null) { throw new InvalidOperationException($"{nameof(leaf)} not found in ${nameof(_Root)}"); }
                SplitNode parent = leaf.Parent as SplitNode ?? throw new InvalidOperationException($"{nameof(leaf)} is Root and cannot be removed");

                // find sibling (guaranteed to be there if split exists)
                Node sibling = parent.First == leaf ? parent.Second : parent.First;

                _Lock.EnterWriteLock();
                try
                {
                    // if there's a grandParent, then rewire the sibling to it. Otherwise, the sibling becomes root
                    if (parent.Parent is SplitNode grandParent)
                    {
                        grandParent.ReplaceChild(parent, sibling);
                        sibling.Parent = grandParent;
                    }
                    else
                    {
                        _Root = sibling;
                        sibling.Parent = null;
                    }

                    // clear refs for garbage collection
                    leaf.Parent = null;
                }
                finally { _Lock.ExitWriteLock(); }

                TreeChanged?.Invoke(TreeChangedAction.Remove, leaf);
            }
            finally { _Lock.ExitUpgradeableReadLock(); }
        }

        public Layout ComputeLayout()
        {
            Layout layout = new();

            _Lock.EnterReadLock();
            try
            {
                // DEBUG: Sanity Check: No shared nodes
                EnsureNotShared(_Root);

                // Compute Spans with splitter cells
                Span span = ComputeSpans(_Root);

                // Initialize row/column definition
                for (int r = 0; r < span.Rows; r++) { layout.RowsDef.Add(new LayoutElementDefinition()); }
                for (int c = 0; c < span.Cols; c++) { layout.ColumnsDef.Add(new LayoutElementDefinition()); }

                // Assign Positions
                AssignPositions(_Root, layout, 0, 0);

                // Expand leaf spans upward
                ExpandLeafSpans(layout);
            }
            finally { _Lock.ExitReadLock(); }

            return layout;
        }

        // Debug Helper: Remove when properly tested ---------------------------------------------------------
        private static void EnsureNotShared(Node root)
        {
            HashSet<Node> seen = [];
            void Walk(Node n)
            {
                if (!seen.Add(n)) { throw new InvalidOperationException($"Shared Node detected: {n.Id}"); }
                if (n is SplitNode s)
                {
                    Walk(s.First ?? throw new InvalidOperationException($"Node id {n.Id} has a null child"));
                    Walk(s.Second ?? throw new InvalidOperationException($"Node id {n.Id} has a null child"));
                }
            }
            Walk(root);
        }

        // ------------------------------------------------------------------------------------------------------
        private readonly struct Span(int rows, int cols)
        {
            public int Rows { get; } = rows;
            public int Cols { get; } = cols;
        }
        // - Each subtree describes a rectangle of rowSpan ✖ columnSpan grid cells. (leaf ➡ 1 x 1)
        // - split vertical   ➡ children side-by-side, merge widths, add 1 splitter column
        // - split horizontal ➡ children stacked, merge heights, add 1 splitter row
        private Span ComputeSpans(Node node)
        {
            if (node is LeafNode) { return new Span(rows: 1, cols: 1); }

            SplitNode split = (SplitNode)node;
            Span first = ComputeSpans(split.First);
            Span second = ComputeSpans(split.Second);

            return split.Orientation switch
            {
                Orientation.Vertical => // left | splitter | right ➡ Add a column
                    new Span(
                        rows: Math.Max(first.Rows, second.Rows),
                        cols: first.Cols + 1 + second.Cols),
                Orientation.Horizontal => // top / splitter / bottom ➡ add a row
                    new Span(
                        rows: first.Rows + 1 + second.Rows,
                        cols: Math.Max(first.Cols, second.Cols)),
                _ => throw new InvalidOperationException(nameof(split.Orientation))
            };
        }

        // produce temporary specifications, each with unitary span, to be expanded on a later bottom up pass
        private void AssignPositions(Node node, Layout layout, int startRow, int startCol)
        {
            if (node is LeafNode)
            {
                layout.Pages.Add(new LayoutElementSpecification(
                    id: node.Id.Id,
                    row: startRow,
                    column: startCol,
                    rowSpan: 1,
                    columnSpan: 1));
                return;
            }

            SplitNode split = (SplitNode)node;
            Span first = ComputeSpans(split.First); // Note: Duplicated computation. if performance heavy, refactor
            Span second = ComputeSpans(split.Second);

            if (split.Orientation == Orientation.Vertical)
            {
                // left | splitter | right
                int splitterCol = startCol + first.Cols;

                // mark splitter column
                layout.ColumnsDef[splitterCol] = new() { IsSplitter = true };

                // add splitter element
                layout.Splitters.Add(new LayoutElementSpecification(
                    id: node.Id.Id,
                    row: startRow,
                    column: splitterCol,
                    rowSpan: Math.Max(first.Rows, second.Rows),
                    columnSpan: 1)
                { Orientation = split.Orientation });

                // recurse children
                AssignPositions(split.First, layout, startRow, startCol);
                AssignPositions(split.Second, layout, startRow, splitterCol + 1);
            }
            else // Horizontal; add splitter row
            {
                int splitterRow = startRow + first.Rows;

                layout.RowsDef[splitterRow] = new LayoutElementDefinition() { IsSplitter = true };

                layout.Splitters.Add(new LayoutElementSpecification(
                    id: node.Id.Id,
                    row: splitterRow,
                    column: startCol,
                    rowSpan: 1,
                    columnSpan: Math.Max(first.Cols, second.Cols))
                { Orientation = split.Orientation });

                AssignPositions(split.First, layout, startRow, startCol);
                AssignPositions(split.Second, layout, splitterRow + 1, startCol);
            }
        }

        private void ExpandLeafSpans(Layout layout)
        {
            // map from grid coordinates ➡ leaf node
            IDictionary<(int row, int col), LeafNode> leafMap = new Dictionary<(int col, int row), LeafNode>();
            BuildLeafPositionMap(_Root, 0, 0, leafMap);

            // now recompute spans for each leaf entry
            List<LayoutElementSpecification> updatedList = new(capacity: layout.Pages.Count);

            foreach (LayoutElementSpecification page in layout.Pages)
            {
                // ensure a page coordinate maps to a leaf
                if (!leafMap.TryGetValue((page.Row, page.Column), out LeafNode? leaf)) { throw new InvalidOperationException($"No leaf mapped at {page.Row}, {page.Column}"); }

                // start assuming the leaf has its own cell and is the deepest in the tree
                int rowSpan = 1;
                int colSpan = 1;

                Node child = leaf;
                Node? cur = leaf.Parent;

                // if the parent has a split, then we need to account for spans
                while (cur is SplitNode split)
                {
                    bool isFirst = split.First == child;
                    Span other = isFirst ? ComputeSpans(split.Second) : ComputeSpans(split.First);

                    if (split.Orientation == Orientation.Horizontal)
                    {
                        // The number of rows the leaf should occupy at the ancestor level
                        rowSpan = Math.Max(rowSpan, other.Rows);
                    }
                    else // Vertical
                    {
                        colSpan = Math.Max(colSpan, other.Cols);
                    }

                    child = cur;
                    cur = cur.Parent;
                }

                updatedList.Add(new LayoutElementSpecification(
                    id: leaf.Id.Id,
                    row: page.Row,
                    column: page.Column,
                    rowSpan: rowSpan,
                    columnSpan: colSpan));
            }

            layout.Pages.Clear();
            foreach (LayoutElementSpecification up in updatedList)
            {
                layout.Pages.Add(up);
            }
        }

        private void BuildLeafPositionMap(Node node, int row, int col, IDictionary<(int row, int col), LeafNode> map)
        {
            if (node is LeafNode leaf)
            {
                // check for duplicated insertion
                if (map.ContainsKey((row, col))) { throw new InvalidOperationException($"Two leaves mapped to the same grid cell: {leaf.Id}"); }

                map[(row, col)] = leaf;
                return;
            }

            SplitNode split = (SplitNode)node;
            Span first = ComputeSpans(split.First); // Note: Duplicate computation. If lags, rearrange

            if (split.Orientation == Orientation.Vertical)
            {
                // left | splitter | right
                BuildLeafPositionMap(split.First, row, col, map);
                // + first.Cols for the complete width of the first subtre +1 for the splitter
                BuildLeafPositionMap(split.Second, row, col + first.Cols + 1, map);
            }
            else
            {
                // top / splitter / bottom
                BuildLeafPositionMap(split.First, row, col, map);
                // + first.Rows for the complete height of the first subtree +1 for the splitter
                BuildLeafPositionMap(split.Second, row + first.Rows + 1, col, map);
            }
        }

        private static Node? FindNodeInternal(Node current, Func<Node, bool> predicate)
        {
            if (predicate(current)) { return current; }

            if (current is SplitNode s)
            {
                Node? r = FindNodeInternal(s.First, predicate);
                return r ?? FindNodeInternal(s.Second, predicate);
            }
            return null;
        }

        public void Dispose()
        {
            _Lock.Dispose();
        }
    }
}