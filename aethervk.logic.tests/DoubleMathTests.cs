using System;
using System.Numerics;
using System.Runtime.InteropServices;
using AetherVk.Logic.Services;
using AetherVk.Logic.Utils;
using Xunit;

namespace AetherVk.Logic.Tests;

/// Double-precision camera math and the f64 rotation interop layouts (mirrored by Rust
/// `const _: () = assert!(size_of::<…>() == N)`): a mismatch would be silent otherwise.
public class DoubleMathTests
{
  [Fact]
  public void Quaterniond_MatchesSystemNumerics()
  {
    var a = Quaternion.Normalize(new Quaternion(0.1f, -0.7f, 0.3f, 0.6f));
    var b = Quaternion.CreateFromAxisAngle(Vector3.Normalize(new Vector3(1, 2, -1)), 0.8f);
    var prod = (Quaternion)((Quaterniond)a * (Quaterniond)b);
    var expected = a * b;
    Assert.Equal(expected.X, prod.X, 5);
    Assert.Equal(expected.Y, prod.Y, 5);
    Assert.Equal(expected.Z, prod.Z, 5);
    Assert.Equal(expected.W, prod.W, 5);

    var v = new Vector3(0.3f, -2f, 5f);
    var r = (Vector3)Vector3d.Transform(v, a);
    var re = Vector3.Transform(v, a);
    Assert.True((r - re).Length() < 1e-5f, $"{r} vs {re}");

    var ab = Quaterniond.CreateFromAxisAngle(new Vector3d(1, 2, -1), 0.8);
    Assert.True(Quaterniond.AngleBetween(ab, b) < 1e-6);
    Assert.True(Math.Abs(Quaterniond.AngleBetween(Quaterniond.Identity, ab) - 0.8) < 1e-12);
    var round = Quaterniond.Normalize(Quaterniond.Inverse(ab) * ab);
    Assert.True(Quaterniond.AngleBetween(round, Quaterniond.Identity) < 1e-15);
  }

  [Fact]
  public void InteropRotationDtos_HaveTheNativeSizes()
  {
    Assert.Equal(72, Marshal.SizeOf<HighResTransformDTO>());
    Assert.Equal(80, Marshal.SizeOf<AnimationTargetDTO>());
    Assert.Equal(64, Marshal.SizeOf<CRotoTranslateDTO>());
    Assert.Equal(48, Marshal.SizeOf<CDustTierStatsDTO>());
  }
}
