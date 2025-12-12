using System.Collections;
using System.Collections.Specialized;
using System.ComponentModel;

namespace AetherVk.Core.Types
{
    public sealed class ObservableDictionary<TKey, TValue> : IDictionary<TKey, TValue>, INotifyCollectionChanged, INotifyPropertyChanged
        where TKey : notnull
    {
        private readonly Dictionary<TKey, TValue> _Dictionary = [];

        public event NotifyCollectionChangedEventHandler? CollectionChanging;
        public event PropertyChangedEventHandler? PropertyChanging;

        public event NotifyCollectionChangedEventHandler? CollectionChanged;
        public event PropertyChangedEventHandler? PropertyChanged;

        // The caller is responsible to perform checks on whether the collection actually changed
        private void RaisePreAdd(TKey theKey, TValue theValue)
        {
            CollectionChanging?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, new KeyValuePair<TKey, TValue>(theKey, theValue)));
            PropertyChanging?.Invoke(this, new PropertyChangedEventArgs(nameof(Count)));
            PropertyChanging?.Invoke(this, new PropertyChangedEventArgs(ItemPropertyName));
        }

        private void RaisePreRemove(TKey theKey, TValue theValue)
        {
            CollectionChanging?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Remove, new KeyValuePair<TKey, TValue>(theKey, theValue)));
            PropertyChanging?.Invoke(this, new PropertyChangedEventArgs(nameof(Count)));
            PropertyChanging?.Invoke(this, new PropertyChangedEventArgs(ItemPropertyName));
        }

        private void RaisePreReplace(TKey oldKey, TValue oldValue, TKey newKey, TValue newValue)
        {
            CollectionChanging?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Replace, new KeyValuePair<TKey, TValue>(newKey, newValue), new KeyValuePair<TKey, TValue>(oldKey, oldValue)));
            PropertyChanging?.Invoke(this, new PropertyChangedEventArgs(ItemPropertyName));
        }

        private void RaisePreReset()
        {
            // no need to notify PropertyChanged, as the size was already notified in the removes
            CollectionChanging?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset));
        }

        private void RaisePostAdd(TKey theKey, TValue theValue)
        {
            CollectionChanged?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, new KeyValuePair<TKey, TValue>(theKey, theValue)));
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Count)));
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(ItemPropertyName));
        }

        private void RaisePostRemove(TKey theKey, TValue theValue)
        {
            CollectionChanged?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Remove, new KeyValuePair<TKey, TValue>(theKey, theValue)));
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(nameof(Count)));
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(ItemPropertyName));
        }

        private void RaisePostReplace(TKey oldKey, TValue oldValue, TKey newKey, TValue newValue)
        {
            CollectionChanged?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Replace, new KeyValuePair<TKey, TValue>(newKey, newValue), new KeyValuePair<TKey, TValue>(oldKey, oldValue)));
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(ItemPropertyName));
        }

        private void RaisePostReset()
        {
            // no need to notify PropertyChanged, as the size was already notified in the removes
            CollectionChanged?.Invoke(this, new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset));
        }

        public TValue this[TKey key]
        {
            get => _Dictionary[key];
            set
            {
                if (_Dictionary.ContainsKey(key))
                {
                    TValue oldValue = _Dictionary[key];
                    RaisePreReplace(key, oldValue, key, value);
                    _Dictionary[key] = value;
                    RaisePostReplace(key, oldValue, key, value);
                }
                else
                {
                    RaisePreAdd(key, value);
                    _Dictionary[key] = value;
                    RaisePostAdd(key, value);
                }
            }
        }

        public const string ItemPropertyName = "Item[]";

        public int Count => _Dictionary.Count;
        public ICollection<TKey> Keys => _Dictionary.Keys;
        public ICollection<TValue> Values => _Dictionary.Values;
        public bool IsReadOnly => false;

        // Dictionary.Add raises a ArgumentException if key already present, so safe to call RaiseAdd after
        public void Add(TKey theKey, TValue theValue) { RaisePreAdd(theKey, theValue); _Dictionary.Add(theKey, theValue); RaisePostAdd(theKey, theValue); }
        public bool Remove(TKey theKey) { if (_Dictionary.TryGetValue(theKey, out TValue? val)) { RaisePreRemove(theKey, val); if (_Dictionary.Remove(theKey)) { RaisePostRemove(theKey, val); return true; } } return false; }
        public void Clear() { List<KeyValuePair<TKey, TValue>> oldItems = [.. _Dictionary]; RaisePreReset(); foreach ((TKey k, TValue v) in oldItems) { RaisePreRemove(k, v); } _Dictionary.Clear(); foreach ((TKey k, TValue v) in oldItems) { RaisePostRemove(k, v); } RaisePostReset(); }

        public bool ContainsKey(TKey key) { return _Dictionary.ContainsKey(key); }
        public bool TryGetValue(TKey key, out TValue theValue) { return _Dictionary.TryGetValue(key, out theValue!); }
        public void Add(KeyValuePair<TKey, TValue> item) { Add(item.Key, item.Value); }
        public bool Contains(KeyValuePair<TKey, TValue> item) { return _Dictionary.Contains(item); }
        public void CopyTo(KeyValuePair<TKey, TValue>[] array, int arrayIndex) { ((IDictionary<TKey, TValue>)_Dictionary).CopyTo(array, arrayIndex); }
        public bool Remove(KeyValuePair<TKey, TValue> item) { return Contains(item) && Remove(item.Key); }

        // TODO: Modifications through enumeration should notify
        public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator() { return _Dictionary.GetEnumerator(); }

        IEnumerator IEnumerable.GetEnumerator() { return GetEnumerator(); }
    }
}
