using System;
using System.Collections.Generic;
using System.Linq;
using AetherVk.Logic.Models;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels;

public partial class SnapObserverViewModel : ObservableObject
{
  [ObservableProperty]
  private double _latitude;

  [ObservableProperty]
  private double _longitude;

  [ObservableProperty]
  private KnownObservatory? _selectedObservatory;

  public IReadOnlyList<KnownObservatory> Observatories { get; }

  public SnapObserverViewModel()
  {
    Observatories = new List<KnownObservatory>
    {
      new KnownObservatory
      {
        Name = "Mauna Kea Observatory",
        Region = "Hawaii, USA",
        Latitude = 19.8206,
        Longitude = -155.4680,
      },
      new KnownObservatory
      {
        Name = "Paranal Observatory (VLT)",
        Region = "Cerro Paranal, Chile",
        Latitude = -24.6272,
        Longitude = -70.4042,
      },
      new KnownObservatory
      {
        Name = "Atacama Large Millimeter Array (ALMA)",
        Region = "Chajnantor Plateau, Chile",
        Latitude = -23.0234,
        Longitude = -67.7538,
      },
      new KnownObservatory
      {
        Name = "Roque de los Muchachos Observatory",
        Region = "Canary Islands, Spain",
        Latitude = 28.7594,
        Longitude = -17.8947,
      },
      new KnownObservatory
      {
        Name = "Palomar Observatory",
        Region = "California, USA",
        Latitude = 33.3563,
        Longitude = -116.8648,
      },
      new KnownObservatory
      {
        Name = "Green Bank Observatory",
        Region = "West Virginia, USA",
        Latitude = 38.4331,
        Longitude = -79.8397,
      },
      new KnownObservatory
      {
        Name = "Sydney Observatory",
        Region = "Sydney, Australia",
        Latitude = -33.8599,
        Longitude = 151.2003,
      },
      new KnownObservatory
      {
        Name = "Royal Observatory",
        Region = "Greenwich, UK",
        Latitude = 51.4769,
        Longitude = -0.0005,
      },
    };
  }

  partial void OnSelectedObservatoryChanged(KnownObservatory? value)
  {
    if (value != null)
    {
      Latitude = value.Latitude;
      Longitude = value.Longitude;
    }
  }

  public (double X, double Y, double Z) CalculateSimulationOffset()
  {
    double r = 6371.0;
    double latRad = Latitude * Math.PI / 180.0;
    double lonRad = Longitude * Math.PI / 180.0;

    double x = r * Math.Cos(latRad) * Math.Cos(lonRad);
    double y = r * Math.Cos(latRad) * Math.Sin(lonRad);
    double z = r * Math.Sin(latRad);

    // Convert km to simulation units (AU approx)
    // 1 km = 6.68458712e-9 AU
    double scale = 6.68458712e-9;

    return (x * scale, y * scale, z * scale);
  }
}
