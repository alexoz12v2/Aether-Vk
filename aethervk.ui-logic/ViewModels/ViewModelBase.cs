using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Messaging;

namespace AetherVk.Logic.ViewModels;

public abstract class ViewModelBase : ObservableRecipient
{
  protected ViewModelBase() { }

  protected ViewModelBase(IMessenger messenger)
    : base(messenger) { }
}
