using System;
using System.Timers;
using AetherVk.Logic.Services;
using CommunityToolkit.Mvvm.ComponentModel;

namespace AetherVk.Logic.ViewModels.Debug;

#if DEBUG
public sealed partial class DebugTelemetryPanelViewModel : ObservableObject
{
    private readonly INativeRuntimeService _runtime;
    private readonly Timer? _pollTimer;

    [ObservableProperty]
    private bool _isExpanded;

    [ObservableProperty]
    private double _physicalRamMb;

    [ObservableProperty]
    private double _virtualRamMb;

    [ObservableProperty]
    private double _cpuAllocatedMb;

    [ObservableProperty]
    private double _gpuAllocatedMb;

    [ObservableProperty]
    private double _logicThreadMs;

    [ObservableProperty]
    private double _renderCpuMs;

    [ObservableProperty]
    private double _gpuExecMs;

    /// <summary>The dust frame's GPU time (ms) of this viewport and its two heavy passes
    /// (<see cref="INativeRuntimeService.DustGpuTimings"/>; NaN when not measured).</summary>
    [ObservableProperty]
    private double _dustGpuMs = double.NaN;

    [ObservableProperty]
    private double _dustSplatMs = double.NaN;

    [ObservableProperty]
    private double _dustPropagateMs = double.NaN;

    [ObservableProperty]
    private double _dotNetHeapMb;

    [ObservableProperty]
    private int _gen0Collections;

    [ObservableProperty]
    private int _gen1Collections;

    [ObservableProperty]
    private int _gen2Collections;

    public DebugTelemetryPanelViewModel(INativeRuntimeService runtime)
    {
        _runtime = runtime;
        _pollTimer = new Timer(500);
        _pollTimer.Elapsed += (_, _) => PollTelemetry();
    }

    partial void OnIsExpandedChanged(bool value)
    {
        if (value)
        {
            PollTelemetry();
            _pollTimer?.Start();
        }
        else
        {
            _pollTimer?.Stop();
        }
    }

    private void PollTelemetry()
    {
        if (!_runtime.GetDebugTelemetryStats(out var stats))
            return;

        PhysicalRamMb = stats.OsPhysicalRamBytes / (1024.0 * 1024.0);
        VirtualRamMb = stats.OsVirtualRamBytes / (1024.0 * 1024.0);
        CpuAllocatedMb = stats.CpuAllocatedBytes / (1024.0 * 1024.0);
        GpuAllocatedMb = stats.GpuAllocatedBytes / (1024.0 * 1024.0);
        
        LogicThreadMs = stats.LogicThreadCpuTimeMs;
        RenderCpuMs = stats.RenderThreadCpuTimeMs;
        GpuExecMs = stats.ReservedGpuExecutionMs;

        if (_runtime.PresentationEngineId is { } pe && _runtime.DustGpuTimings(pe) is { } dust)
        {
            DustGpuMs = dust.TotalEmaMs;
            DustSplatMs = dust.SplatMs;
            DustPropagateMs = dust.PropagateMs;
        }

        DotNetHeapMb = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
        Gen0Collections = GC.CollectionCount(0);
        Gen1Collections = GC.CollectionCount(1);
        Gen2Collections = GC.CollectionCount(2);
    }
}
#endif
