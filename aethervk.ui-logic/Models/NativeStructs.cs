using System;
using System.Runtime.InteropServices;

namespace AetherVk.Logic.Models;

[StructLayout(LayoutKind.Sequential)]
public struct Float4
{
  public float X;
  public float Y;
  public float Z;
  public float W;

  public Float4(float x, float y, float z, float w)
  {
    X = x;
    Y = y;
    Z = z;
    W = w;
  }
}

[StructLayout(LayoutKind.Sequential, Pack = 1, CharSet = CharSet.Ansi)]
public struct FfiPhysicalMesh
{
  public byte IsProcedural;
  [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 256)]
  public string AssetPath;
}

[StructLayout(LayoutKind.Sequential, Pack = 16)]
public struct RationalBezierGpu
{
  public Float4 Cp0;
  public Float4 Cp1;
  public Float4 Cp2;
  public Float4 Cp3;
}

[StructLayout(LayoutKind.Sequential, Pack = 8)]
public struct TrajectoryGpu
{
  public ulong SegmentsPtr;
  public Float4 Color;
  public float LineWidth;
  public uint TextureId;
}

[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct FfiSphereGizmo
{
  public float Radius;
  public float Subdivisions;
  [MarshalAs(UnmanagedType.ByValArray, SizeConst = 16)]
  public float[] LocalFrame;
  public byte IsVisible;
}
