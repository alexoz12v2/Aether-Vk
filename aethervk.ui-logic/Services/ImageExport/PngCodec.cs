using System;
using System.Buffers.Binary;
using System.IO;
using System.IO.Compression;
using System.Text;

namespace AetherVk.Logic.Services.ImageExport;

/// <summary>
/// A minimal PNG codec for the image export (netstandard 2.0, no image library): writes 8-bit RGB
/// PNGs with filter 0 on every row, and reads back such files (8-bit RGB or RGBA, filter 0 rows).
/// </summary>
public static class PngCodec
{
  private static readonly uint[] CrcTable = BuildCrcTable();

  /// <summary>Writes <paramref name="rgba"/> (<paramref name="width"/> × <paramref name="height"/>,
  /// rows from the top) as an opaque RGB PNG at <paramref name="path"/>.</summary>
  public static void WriteRgb(string path, byte[] rgba, int width, int height)
  {
    if (width <= 0 || height <= 0)
      throw new ArgumentOutOfRangeException(nameof(width), "empty image");
    if (rgba.Length < width * height * 4)
      throw new ArgumentException("pixel buffer shorter than width × height × 4", nameof(rgba));

    var raw = new byte[height * (1 + width * 3)];
    for (int y = 0, o = 0; y < height; y++)
    {
      raw[o++] = 0; // filter: none
      for (int x = 0; x < width; x++, o += 3)
      {
        int k = 4 * (y * width + x);
        raw[o] = rgba[k];
        raw[o + 1] = rgba[k + 1];
        raw[o + 2] = rgba[k + 2];
      }
    }

    var header = new byte[13];
    BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(0, 4), width);
    BinaryPrimitives.WriteInt32BigEndian(header.AsSpan(4, 4), height);
    header[8] = 8; // bit depth
    header[9] = 2; // colour type: RGB
    header[10] = 0; // compression
    header[11] = 0; // filter method
    header[12] = 0; // no interlace

    var dir = Path.GetDirectoryName(path);
    if (!string.IsNullOrEmpty(dir))
      Directory.CreateDirectory(dir);
    using var f = File.Create(path);
    f.Write(new byte[] { 0x89, (byte)'P', (byte)'N', (byte)'G', 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
    WriteChunk(f, "IHDR", header);
    WriteChunk(f, "IDAT", Zlib(raw));
    WriteChunk(f, "IEND", Array.Empty<byte>());
  }

  /// <summary>
  /// Reads a PNG written by <see cref="WriteRgb"/> (8-bit RGB or RGBA, every row with filter 0)
  /// as RGBA bytes, rows from the top. Other PNGs raise <see cref="InvalidDataException"/>.
  /// </summary>
  public static byte[] ReadRgba(string path, out int width, out int height)
  {
    var bytes = File.ReadAllBytes(path);
    if (bytes.Length < 8 || bytes[1] != (byte)'P' || bytes[2] != (byte)'N' || bytes[3] != (byte)'G')
      throw new InvalidDataException($"{path}: not a PNG");
    width = height = 0;
    int channels = 0;
    using var idat = new MemoryStream();
    for (int o = 8; o + 12 <= bytes.Length;)
    {
      int len = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(o, 4));
      string type = Encoding.ASCII.GetString(bytes, o + 4, 4);
      if (len < 0 || o + 12 + len > bytes.Length)
        throw new InvalidDataException($"{path}: truncated chunk {type}");
      if (type == "IHDR")
      {
        width = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(o + 8, 4));
        height = BinaryPrimitives.ReadInt32BigEndian(bytes.AsSpan(o + 12, 4));
        if (bytes[o + 16] != 8)
          throw new InvalidDataException($"{path}: not 8-bit");
        channels = bytes[o + 17] switch
        {
          2 => 3,
          6 => 4,
          _ => throw new InvalidDataException($"{path}: colour type {bytes[o + 17]} is not RGB/RGBA"),
        };
        if (bytes[o + 20] != 0)
          throw new InvalidDataException($"{path}: interlaced");
      }
      else if (type == "IDAT")
      {
        idat.Write(bytes, o + 8, len);
      }
      else if (type == "IEND")
      {
        break;
      }
      o += 12 + len;
    }
    if (width <= 0 || height <= 0 || channels == 0)
      throw new InvalidDataException($"{path}: no IHDR");

    idat.Position = 2; // the zlib header; the trailing Adler-32 is ignored by the inflater
    using var zs = new DeflateStream(idat, CompressionMode.Decompress);
    var raw = new byte[height * (1 + width * channels)];
    int read = 0;
    while (read < raw.Length)
    {
      int n = zs.Read(raw, read, raw.Length - read);
      if (n <= 0)
        break;
      read += n;
    }
    if (read != raw.Length)
      throw new InvalidDataException($"{path}: {read} of {raw.Length} raw bytes");

    var rgba = new byte[width * height * 4];
    for (int y = 0, o = 0; y < height; y++)
    {
      if (raw[o++] != 0)
        throw new InvalidDataException($"{path}: row {y} uses a PNG filter");
      for (int x = 0; x < width; x++, o += channels)
      {
        int k = 4 * (y * width + x);
        rgba[k] = raw[o];
        rgba[k + 1] = raw[o + 1];
        rgba[k + 2] = raw[o + 2];
        rgba[k + 3] = channels == 4 ? raw[o + 3] : (byte)255;
      }
    }
    return rgba;
  }

  /// <summary>zlib stream: header, deflate data, Adler-32 of the raw data.</summary>
  private static byte[] Zlib(byte[] raw)
  {
    using var z = new MemoryStream();
    z.WriteByte(0x78);
    z.WriteByte(0x9C);
    using (var ds = new DeflateStream(z, CompressionLevel.Optimal, leaveOpen: true))
      ds.Write(raw, 0, raw.Length);
    uint a = 1, b = 0;
    foreach (byte v in raw)
    {
      a = (a + v) % 65521;
      b = (b + a) % 65521;
    }
    var adler = new byte[4];
    BinaryPrimitives.WriteUInt32BigEndian(adler, (b << 16) | a);
    z.Write(adler, 0, 4);
    return z.ToArray();
  }

  private static void WriteChunk(Stream f, string type, byte[] data)
  {
    var be = new byte[4];
    BinaryPrimitives.WriteInt32BigEndian(be, data.Length);
    f.Write(be, 0, 4);
    var typed = new byte[4 + data.Length];
    Encoding.ASCII.GetBytes(type, 0, 4, typed, 0);
    Array.Copy(data, 0, typed, 4, data.Length);
    f.Write(typed, 0, typed.Length);
    BinaryPrimitives.WriteUInt32BigEndian(be, Crc32(typed));
    f.Write(be, 0, 4);
  }

  private static uint[] BuildCrcTable()
  {
    var table = new uint[256];
    for (uint n = 0; n < 256; n++)
    {
      uint c = n;
      for (int k = 0; k < 8; k++)
        c = (c & 1) != 0 ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[n] = c;
    }
    return table;
  }

  private static uint Crc32(byte[] data)
  {
    uint c = 0xFFFFFFFFu;
    foreach (byte b in data)
      c = CrcTable[(c ^ b) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
  }
}
