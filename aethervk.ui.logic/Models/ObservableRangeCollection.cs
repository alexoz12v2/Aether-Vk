using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;

namespace AetherVk.Logic.Models;

public class ObservableRangeCollection<T> : ObservableCollection<T>
{
  public void AddRange(IEnumerable<T> range)
  {
    if (range == null)
      throw new ArgumentNullException(nameof(range));
    var list = range as IList<T> ?? new List<T>(range);

    CheckReentrancy();
    var startingIndex = list.Count;

    // Optimization: ObservableCollection<T>.Items is usually a List<T>
    // casting it allows us to use allocation-friendly bulk operations
    if (Items is List<T> underlyingList)
    {
      underlyingList.AddRange(list);
    }
    else
    {
      foreach (var item in list)
        Items.Add(item);
    }
    OnPropertyChanged(new PropertyChangedEventArgs(nameof(Count)));
    OnPropertyChanged(new PropertyChangedEventArgs("Item[]"));

    // Avalonia perfectly supports multi-item add events natively. One event = 1 layout update
    OnCollectionChanged(
      new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, list, startingIndex)
    );
  }

  public void RemoveRange(int index, int count)
  {
    if (count == 0)
      return;
    if (index < 0 || index + count > Items.Count)
      throw new ArgumentOutOfRangeException(nameof(index));

    CheckReentrancy();
    IList<T> removed;

    // Bypassing an O(N^2) for-loop shifts memory instantanoeusly
    if (Items is List<T> underlyingList)
    {
      removed = underlyingList.GetRange(index, count);
      underlyingList.RemoveRange(index, count);
    }
    else
    {
      var removedList = new List<T>(count);
      for (var i = 0; i < count; i++)
      {
        removedList.Add(Items[i]);
        Items.RemoveAt(index);
      }

      removed = removedList;
    }

    OnPropertyChanged(new PropertyChangedEventArgs(nameof(Count)));
    OnPropertyChanged(new PropertyChangedEventArgs("Item[]"));

    OnCollectionChanged(
      new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Remove, removed, index)
    );
  }
}
