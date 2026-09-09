using System;
using System.Collections.Concurrent;
using System.Reactive.Linq;
using System.Reactive.Subjects;

namespace AetherVk.Logic.Services;

public interface ICameraServiceRegistry
{
    void RegisterSelf(ulong cameraId, CameraService cameraService);
    void UnregisterSelf(ulong cameraId);
    CameraService? Get(ulong cameraId);
    
    IObservable<ulong> ViewportCreated { get; }
    IObservable<ulong> ViewportDestroyed { get; }
}

public sealed class CameraServiceRegistry : ICameraServiceRegistry, IDisposable
{
    private readonly ConcurrentDictionary<ulong, CameraService> _services = new();
    private readonly Subject<ulong> _viewportCreated = new();
    private readonly Subject<ulong> _viewportDestroyed = new();

    public IObservable<ulong> ViewportCreated => _viewportCreated.AsObservable();
    public IObservable<ulong> ViewportDestroyed => _viewportDestroyed.AsObservable();

    public void RegisterSelf(ulong cameraId, CameraService cameraService)
    {
        if (_services.TryAdd(cameraId, cameraService))
        {
            _viewportCreated.OnNext(cameraId);
        }
        else
        {
            // If already present, just update it, no new event
            _services[cameraId] = cameraService;
        }
    }

    public void UnregisterSelf(ulong cameraId)
    {
        if (_services.TryRemove(cameraId, out _))
        {
            _viewportDestroyed.OnNext(cameraId);
        }
    }

    public CameraService? Get(ulong cameraId)
    {
        return _services.TryGetValue(cameraId, out var service) ? service : null;
    }

    public void Dispose()
    {
        _viewportCreated.Dispose();
        _viewportDestroyed.Dispose();
    }
}
