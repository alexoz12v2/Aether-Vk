using System;
using System.Numerics;

namespace AetherVk.Logic.Utils;

/// <summary>
/// Double-precision 3-vector. <see cref="Vector3"/> is float: at 1 AU its rounding is ~10 km,
/// more than an Earth observer's telescope field on a comet nucleus.
/// Widening from <see cref="Vector3"/> is implicit (lossless), narrowing is explicit.
/// </summary>
public readonly struct Vector3d : IEquatable<Vector3d>
{
  public readonly double X;
  public readonly double Y;
  public readonly double Z;

  public Vector3d(double x, double y, double z)
  {
    X = x;
    Y = y;
    Z = z;
  }

  public static Vector3d Zero => new(0, 0, 0);
  public static Vector3d UnitX => new(1, 0, 0);
  public static Vector3d UnitY => new(0, 1, 0);
  public static Vector3d UnitZ => new(0, 0, 1);

  public double LengthSquared() => X * X + Y * Y + Z * Z;

  public double Length() => Math.Sqrt(LengthSquared());

  public static double Dot(Vector3d a, Vector3d b) => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

  public static Vector3d Cross(Vector3d a, Vector3d b) =>
    new(a.Y * b.Z - a.Z * b.Y, a.Z * b.X - a.X * b.Z, a.X * b.Y - a.Y * b.X);

  /// <summary>Unit vector; zero stays zero.</summary>
  public static Vector3d Normalize(Vector3d v)
  {
    double l = v.Length();
    return l > 0 ? v / l : v;
  }

  /// <summary>Rotates <paramref name="v"/> by <paramref name="q"/> (q·v·q*), like <see cref="Vector3.Transform(Vector3, Quaternion)"/>.</summary>
  public static Vector3d Transform(Vector3d v, Quaterniond q)
  {
    var u = new Vector3d(q.X, q.Y, q.Z);
    var t = 2.0 * Cross(u, v);
    return v + q.W * t + Cross(u, t);
  }

  public static Vector3d operator +(Vector3d a, Vector3d b) => new(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

  public static Vector3d operator -(Vector3d a, Vector3d b) => new(a.X - b.X, a.Y - b.Y, a.Z - b.Z);

  public static Vector3d operator -(Vector3d a) => new(-a.X, -a.Y, -a.Z);

  public static Vector3d operator *(Vector3d a, double s) => new(a.X * s, a.Y * s, a.Z * s);

  public static Vector3d operator *(double s, Vector3d a) => a * s;

  public static Vector3d operator /(Vector3d a, double s) => new(a.X / s, a.Y / s, a.Z / s);

  public static implicit operator Vector3d(Vector3 v) => new(v.X, v.Y, v.Z);

  public static explicit operator Vector3(Vector3d v) => new((float)v.X, (float)v.Y, (float)v.Z);

  public bool Equals(Vector3d o) => X == o.X && Y == o.Y && Z == o.Z;

  public override bool Equals(object? obj) => obj is Vector3d o && Equals(o);

  public override int GetHashCode() => X.GetHashCode() ^ (Y.GetHashCode() << 2) ^ (Z.GetHashCode() >> 2);

  public static bool operator ==(Vector3d a, Vector3d b) => a.Equals(b);

  public static bool operator !=(Vector3d a, Vector3d b) => !a.Equals(b);

  public override string ToString() => $"<{X}, {Y}, {Z}>";
}

/// <summary>
/// Double-precision quaternion (x, y, z, w), Hamilton product like <see cref="Quaternion"/>.
/// An f32 quaternion rounds by ~1e-7 rad, ~15 km at 1 AU: the camera rotation travels in f64
/// end to end (Rust <c>HighResTransformComponent.rotation</c> is <c>Quat64</c>).
/// Widening from <see cref="Quaternion"/> is implicit (lossless), narrowing is explicit.
/// </summary>
public readonly struct Quaterniond : IEquatable<Quaterniond>
{
  public readonly double X;
  public readonly double Y;
  public readonly double Z;
  public readonly double W;

  public Quaterniond(double x, double y, double z, double w)
  {
    X = x;
    Y = y;
    Z = z;
    W = w;
  }

  public static Quaterniond Identity => new(0, 0, 0, 1);

  public double Length() => Math.Sqrt(X * X + Y * Y + Z * Z + W * W);

  public static Quaterniond Normalize(Quaterniond q)
  {
    double l = q.Length();
    return l > 0 ? new(q.X / l, q.Y / l, q.Z / l, q.W / l) : Identity;
  }

  public static Quaterniond Conjugate(Quaterniond q) => new(-q.X, -q.Y, -q.Z, q.W);

  public static Quaterniond Inverse(Quaterniond q)
  {
    double n = q.X * q.X + q.Y * q.Y + q.Z * q.Z + q.W * q.W;
    return new(-q.X / n, -q.Y / n, -q.Z / n, q.W / n);
  }

  public static Quaterniond CreateFromAxisAngle(Vector3d axis, double angle)
  {
    var a = Vector3d.Normalize(axis);
    double s = Math.Sin(angle * 0.5);
    return new(a.X * s, a.Y * s, a.Z * s, Math.Cos(angle * 0.5));
  }

  /// <summary>Rotation angle between two orientations, radians (sign of q ignored).</summary>
  public static double AngleBetween(Quaterniond a, Quaterniond b)
  {
    var d = a * Conjugate(b);
    double v = Math.Sqrt(d.X * d.X + d.Y * d.Y + d.Z * d.Z);
    return 2.0 * Math.Atan2(v, Math.Abs(d.W));
  }

  public static Quaterniond operator *(Quaterniond a, Quaterniond b) =>
    new(
      a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
      a.W * b.Y - a.X * b.Z + a.Y * b.W + a.Z * b.X,
      a.W * b.Z + a.X * b.Y - a.Y * b.X + a.Z * b.W,
      a.W * b.W - a.X * b.X - a.Y * b.Y - a.Z * b.Z
    );

  public static implicit operator Quaterniond(Quaternion q) => new(q.X, q.Y, q.Z, q.W);

  public static explicit operator Quaternion(Quaterniond q) =>
    new((float)q.X, (float)q.Y, (float)q.Z, (float)q.W);

  public bool Equals(Quaterniond o) => X == o.X && Y == o.Y && Z == o.Z && W == o.W;

  public override bool Equals(object? obj) => obj is Quaterniond o && Equals(o);

  public override int GetHashCode() =>
    X.GetHashCode() ^ (Y.GetHashCode() << 2) ^ (Z.GetHashCode() >> 2) ^ (W.GetHashCode() >> 1);

  public static bool operator ==(Quaterniond a, Quaterniond b) => a.Equals(b);

  public static bool operator !=(Quaterniond a, Quaterniond b) => !a.Equals(b);

  public override string ToString() => $"{{X:{X} Y:{Y} Z:{Z} W:{W}}}";
}
