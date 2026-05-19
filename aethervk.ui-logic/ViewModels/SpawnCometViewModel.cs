using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;

namespace AetherVk.Logic.ViewModels;

public partial class SpawnCometViewModel : ObservableObject
{
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  [NotifyPropertyChangedFor(nameof(CanGoBack))]
  [NotifyPropertyChangedFor(nameof(IsStep1))]
  [NotifyPropertyChangedFor(nameof(IsStep2))]
  [NotifyPropertyChangedFor(nameof(IsStep3))]
  private int _currentStep = 1;

  public bool IsStep1 => CurrentStep == 1;
  public bool IsStep2 => CurrentStep == 2;
  public bool IsStep3 => CurrentStep == 3;

  public bool CanGoBack => CurrentStep > 1;

  public bool CanGoNext =>
    CurrentStep switch
    {
      1 => SelectedModel != null,
      2 => PhysicsType == "Static",
      3 => false,
      _ => false,
    };

  // --- Step 1 ---
  public ObservableCollection<ImportedModelItem> ImportedModels { get; } = new();

  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  private ImportedModelItem? _selectedModel;

  public bool HasNoModels => ImportedModels.Count == 0;

  // --- Step 2 ---
  [ObservableProperty]
  [NotifyPropertyChangedFor(nameof(CanGoNext))]
  private string _physicsType = "Static"; // Static, Kinematic, Dynamic

  // --- Step 3 ---
  [ObservableProperty]
  private float _posX = 0f;

  [ObservableProperty]
  private float _posY = 0f;

  [ObservableProperty]
  private float _posZ = 0f;

  [ObservableProperty]
  private float _scaleX = 1f;

  [ObservableProperty]
  private float _scaleY = 1f;

  [ObservableProperty]
  private float _scaleZ = 1f;

  [ObservableProperty]
  private float _pitch = 0f;

  [ObservableProperty]
  private float _yaw = 0f;

  [ObservableProperty]
  private float _roll = 0f;

  [ObservableProperty]
  private string _entityName = "New Comet";

  public SpawnCometViewModel(IEnumerable<ImportedModelItem> models)
  {
    foreach (var model in models)
    {
      ImportedModels.Add(model);
    }
    SelectedModel = ImportedModels.FirstOrDefault();
  }

  [RelayCommand]
  private void NextStep()
  {
    if (CanGoNext)
    {
      CurrentStep++;
    }
  }

  [RelayCommand]
  private void PreviousStep()
  {
    if (CanGoBack)
    {
      CurrentStep--;
    }
  }
}
