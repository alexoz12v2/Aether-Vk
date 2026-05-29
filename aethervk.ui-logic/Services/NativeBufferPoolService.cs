using System;
using System.Buffers;

namespace AetherVk.Logic.Services;

public interface INativeBufferPoolService
{
    PooledArray<T> Rent<T>(int minimumLength);
    void Return<T>(T[] array, bool clearArray = false);
}

public readonly struct PooledArray<T> : IDisposable
{
    public readonly T[] Array;
    private readonly INativeBufferPoolService _pool;

    public PooledArray(T[] array, INativeBufferPoolService pool)
    {
        Array = array;
        _pool = pool;
    }

    public void Dispose()
    {
        _pool?.Return(Array);
    }
}

public class NativeBufferPoolService : INativeBufferPoolService
{
    public PooledArray<T> Rent<T>(int minimumLength)
    {
        return new PooledArray<T>(ArrayPool<T>.Shared.Rent(minimumLength), this);
    }

    public void Return<T>(T[] array, bool clearArray = false)
    {
        if (array != null)
        {
            ArrayPool<T>.Shared.Return(array, clearArray);
        }
    }
}
