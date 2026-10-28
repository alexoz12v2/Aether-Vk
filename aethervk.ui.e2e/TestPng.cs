using System.Buffers.Binary;
using System.IO.Compression;
using System.IO.Hashing;

namespace AetherVk.UI.E2E;

/// <summary>
/// A stand-in for the telescope image when <c>AETHERVK_E2E_IMAGES</c> has none: an RGBA PNG with a
/// soft radial glow and a crosshair, so the billboard is visibly translucent over the viewport.
/// </summary>
internal static class TestPng
{
  public static void Write(string path, int size = 512)
  {
    var raw = new byte[size * (size * 4 + 1)];
    double c = (size - 1) / 2.0;
    for (int y = 0; y < size; y++)
    {
      int row = y * (size * 4 + 1);
      raw[row] = 0; // filter: none
      for (int x = 0; x < size; x++)
      {
        double r = Math.Sqrt((x - c) * (x - c) + (y - c) * (y - c)) / c;
        bool cross = Math.Abs(x - c) < 1.5 || Math.Abs(y - c) < 1.5;
        byte glow = (byte)(255 * Math.Exp(-4 * r * r));
        int p = row + 1 + x * 4;
        raw[p] = cross ? (byte)255 : glow;
        raw[p + 1] = cross ? (byte)64 : glow;
        raw[p + 2] = cross ? (byte)64 : (byte)(glow / 2);
        raw[p + 3] = 255;
      }
    }

    using var file = File.Create(path);
    file.Write([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    var header = new byte[13];
    BinaryPrimitives.WriteInt32BigEndian(header, size);
    BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4), size);
    header[8] = 8; // bit depth
    header[9] = 6; // RGBA
    Chunk(file, "IHDR", header);
    using (var idat = new MemoryStream())
    {
      using (var z = new ZLibStream(idat, CompressionLevel.Fastest, leaveOpen: true))
        z.Write(raw);
      Chunk(file, "IDAT", idat.ToArray());
    }
    Chunk(file, "IEND", []);
  }

  private static void Chunk(Stream s, string type, byte[] data)
  {
    Span<byte> len = stackalloc byte[4];
    BinaryPrimitives.WriteInt32BigEndian(len, data.Length);
    s.Write(len);
    var typed = new byte[4 + data.Length];
    System.Text.Encoding.ASCII.GetBytes(type).CopyTo(typed, 0);
    data.CopyTo(typed, 4);
    s.Write(typed);
    Span<byte> crc = stackalloc byte[4];
    BinaryPrimitives.WriteUInt32BigEndian(crc, Crc32.HashToUInt32(typed));
    s.Write(crc);
  }
}
