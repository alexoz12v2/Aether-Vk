//! texture_cache module.

use crate::simulation::comet::TexelFormat;
use aethervk_oshal_rlib::{hash::FnvHasher, os::files::Mmap};
use alloc::{collections::BTreeMap, string::String, vec::Vec};
use bytes::Bytes;
use core::hash::Hasher;

#[cfg(target_family = "unix")]
pub mod sys {
  use alloc::{ffi::CString, string::String};
  use libc::{O_CREAT, O_RDWR, O_TRUNC, c_void, close, fstat, open, pread, pwrite, rename};

  pub struct NativeFile(i32);
  impl NativeFile {
    pub fn open(path: &str) -> Option<Self> {
      let fd = unsafe {
        open(
          CString::new(path).unwrap().as_ptr(),
          O_RDWR | O_CREAT,
          0o644,
        )
      };
      if fd < 0 { None } else { Some(Self(fd)) }
    }
    pub fn create_trunc(path: &str) -> Option<Self> {
      let fd = unsafe {
        open(
          CString::new(path).unwrap().as_ptr(),
          O_RDWR | O_CREAT | O_TRUNC,
          0o644,
        )
      };
      if fd < 0 { None } else { Some(Self(fd)) }
    }
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> bool {
      unsafe {
        pread(
          self.0,
          buf.as_mut_ptr() as *mut c_void,
          buf.len(),
          offset as i64,
        ) == buf.len() as isize
      }
    }
    pub fn write_at(&self, offset: u64, buf: &[u8]) -> bool {
      unsafe {
        pwrite(
          self.0,
          buf.as_ptr() as *const c_void,
          buf.len(),
          offset as i64,
        ) == buf.len() as isize
      }
    }
    pub fn size(&self) -> u64 {
      let mut st: libc::stat = unsafe { core::mem::zeroed() };
      unsafe { fstat(self.0, &mut st) };
      st.st_size as u64
    }
  }
  impl Drop for NativeFile {
    fn drop(&mut self) {
      unsafe {
        close(self.0);
      }
    }
  }

  pub fn get_app_dir(app_name: &str) -> String {
    let home = unsafe { libc::getenv(b"HOME\0".as_ptr() as *const _) };
    let h_str = if home.is_null() {
      "/tmp"
    } else {
      unsafe { core::ffi::CStr::from_ptr(home) }.to_str().unwrap_or("/tmp")
    };
    let dir = alloc::format!("{}/.{}", h_str, app_name);
    unsafe {
      libc::mkdir(CString::new(dir.clone()).unwrap().as_ptr(), 0o755);
    }
    dir
  }
  pub fn rename_file(old: &str, new: &str) {
    unsafe {
      rename(
        CString::new(old).unwrap().as_ptr(),
        CString::new(new).unwrap().as_ptr(),
      );
    }
  }
}

#[cfg(windows)]
pub mod sys {
  use alloc::{string::String, vec::Vec};
  use windows::Win32::{
    Foundation::{CloseHandle, GENERIC_READ, GENERIC_WRITE, HANDLE},
    Storage::FileSystem::{
      CREATE_ALWAYS, CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, FILE_SHARE_WRITE,
      GetFileSizeEx, MOVEFILE_REPLACE_EXISTING, MoveFileExW, OPEN_ALWAYS, ReadFile, WriteFile,
    },
    System::{Environment::GetEnvironmentVariableW, IO::OVERLAPPED},
  };

  fn to_u16s(s: &str) -> Vec<u16> {
    core::iter::Iterator::chain(s.encode_utf16(), core::iter::once(0)).collect()
  }

  pub struct NativeFile(HANDLE);
  impl NativeFile {
    pub fn open(path: &str) -> Option<Self> {
      let handle = unsafe {
        CreateFileW(
          windows::core::PCWSTR(to_u16s(path).as_ptr()),
          GENERIC_READ.0 | GENERIC_WRITE.0,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          None,
          OPEN_ALWAYS,
          FILE_ATTRIBUTE_NORMAL,
          None,
        )
      };
      if handle.is_ok() {
        Some(Self(handle.unwrap()))
      } else {
        None
      }
    }
    pub fn create_trunc(path: &str) -> Option<Self> {
      let handle = unsafe {
        CreateFileW(
          windows::core::PCWSTR(to_u16s(path).as_ptr()),
          GENERIC_READ.0 | GENERIC_WRITE.0,
          FILE_SHARE_READ | FILE_SHARE_WRITE,
          None,
          CREATE_ALWAYS,
          FILE_ATTRIBUTE_NORMAL,
          None,
        )
      };
      if handle.is_ok() {
        Some(Self(handle.unwrap()))
      } else {
        None
      }
    }
    pub fn read_at(&self, offset: u64, buf: &mut [u8]) -> bool {
      let mut ol: OVERLAPPED = unsafe { core::mem::zeroed() };
      ol.Anonymous.Anonymous.Offset = (offset & 0xFFFFFFFF) as u32;
      ol.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
      let mut read = 0;
      unsafe {
        ReadFile(self.0, Some(buf), Some(&mut read), Some(&mut ol)).is_ok()
          && read as usize == buf.len()
      }
    }
    pub fn write_at(&self, offset: u64, buf: &[u8]) -> bool {
      let mut ol: OVERLAPPED = unsafe { core::mem::zeroed() };
      ol.Anonymous.Anonymous.Offset = (offset & 0xFFFFFFFF) as u32;
      ol.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
      let mut written = 0;
      unsafe {
        WriteFile(self.0, Some(buf), Some(&mut written), Some(&mut ol)).is_ok()
          && written as usize == buf.len()
      }
    }
    pub fn size(&self) -> u64 {
      let mut size = 0i64;
      unsafe {
        let _ = GetFileSizeEx(self.0, &mut size);
        size as u64
      }
    }
  }
  impl Drop for NativeFile {
    fn drop(&mut self) {
      unsafe {
        let _ = CloseHandle(self.0);
      }
    }
  }

  pub fn get_app_dir(app_name: &str) -> String {
    let mut buf = alloc::vec![0u16; 256];
    let len = unsafe { GetEnvironmentVariableW(windows::core::w!("USERPROFILE"), Some(&mut buf)) };
    let base = if len > 0 {
      String::from_utf16_lossy(&buf[..len as usize])
    } else {
      String::from("C:")
    };
    let dir = alloc::format!("{}\\.{}", base, app_name);
    unsafe {
      windows::Win32::Storage::FileSystem::CreateDirectoryW(
        windows::core::PCWSTR(to_u16s(&dir).as_ptr()),
        None,
      );
    }
    dir
  }
  pub fn rename_file(old: &str, new: &str) {
    unsafe {
      let _ = MoveFileExW(
        windows::core::PCWSTR(to_u16s(old).as_ptr()),
        windows::core::PCWSTR(to_u16s(new).as_ptr()),
        MOVEFILE_REPLACE_EXISTING,
      );
    }
  }
}

pub fn hash_path(path: &str) -> u64 {
  let mut hasher = FnvHasher::new();
  hasher.write(path.as_bytes());
  hasher.finish()
}

pub trait TexelFormatExt {
  fn to_u32(&self) -> u32;
  fn from_u32(val: u32) -> Self;
  fn byte_size(&self, width: u32, height: u32) -> u64;
}

impl TexelFormatExt for TexelFormat {
  fn to_u32(&self) -> u32 {
    match self {
      Self::R8_UNORM => 1,
      Self::R8G8_UNORM => 2,
      Self::R8G8B8_UNORM => 3,
      Self::R8G8B8A8_UNORM => 4,
      Self::BC7_UNORM_BLOCK => 5,
      Self::ETC2_R8G8B8_UNORM_BLOCK => 6,
      Self::ASTC_4x4_UNORM_BLOCK => 7,
      Self::Undefined => 0,
      Self::Unsupported(v) => *v,
    }
  }

  fn from_u32(val: u32) -> Self {
    match val {
      1 => Self::R8_UNORM,
      2 => Self::R8G8_UNORM,
      3 => Self::R8G8B8_UNORM,
      4 => Self::R8G8B8A8_UNORM,
      5 => Self::BC7_UNORM_BLOCK,
      6 => Self::ETC2_R8G8B8_UNORM_BLOCK,
      7 => Self::ASTC_4x4_UNORM_BLOCK,
      0 => Self::Undefined,
      v => Self::Unsupported(v),
    }
  }

  fn byte_size(&self, width: u32, height: u32) -> u64 {
    let (w, h) = (width as u64, height as u64);
    match self {
      Self::BC7_UNORM_BLOCK | Self::ASTC_4x4_UNORM_BLOCK => ((w + 3) / 4) * ((h + 3) / 4) * 16,
      Self::ETC2_R8G8B8_UNORM_BLOCK => ((w + 3) / 4) * ((h + 3) / 4) * 8,
      Self::R8_UNORM => w * h,
      Self::R8G8_UNORM => w * h * 2,
      Self::R8G8B8_UNORM => w * h * 3,
      _ => w * h * 4,
    }
  }
}

pub struct TextureHeader {
  pub id: u64,
  pub offset: u64,
  pub width: u32,
  pub height: u32,
  pub format: TexelFormat,
}

impl TextureHeader {
  pub fn to_bytes(&self) -> [u8; 28] {
    let mut buf = [0u8; 28];
    buf[0..8].copy_from_slice(&self.id.to_le_bytes());
    buf[8..16].copy_from_slice(&self.offset.to_le_bytes());
    buf[16..20].copy_from_slice(&self.width.to_le_bytes());
    buf[20..24].copy_from_slice(&self.height.to_le_bytes());
    buf[24..28].copy_from_slice(&self.format.to_u32().to_le_bytes());
    buf
  }

  pub fn from_bytes(b: &[u8]) -> Self {
    Self {
      id: u64::from_le_bytes(b[0..8].try_into().unwrap()),
      offset: u64::from_le_bytes(b[8..16].try_into().unwrap()),
      width: u32::from_le_bytes(b[16..20].try_into().unwrap()),
      height: u32::from_le_bytes(b[20..24].try_into().unwrap()),
      format: TexelFormat::from_u32(u32::from_le_bytes(b[24..28].try_into().unwrap())),
    }
  }
}

pub struct TextureCache {
  path_str: String,
  mmap: Option<Mmap>,
  // Memory Cache: id -> (width, height, format, file_offset, byte_size)
  index: BTreeMap<u64, (u32, u32, TexelFormat, u64, usize)>,
}

impl core::fmt::Debug for TextureCache {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_struct("TextureCache")
      .field("path_str", &self.path_str)
      .field("index_size", &self.index.len())
      .finish()
  }
}

impl TextureCache {
  pub fn new(app_name: &str) -> Self {
    let dir = sys::get_app_dir(app_name);
    let path_str = alloc::format!("{}/texfile", dir);

    if let Some(file) = sys::NativeFile::open(&path_str) {
      if file.size() == 0 {
        // Initialize the file with an empty list block: `count` (4 bytes) + `next_ptr` (8 bytes)
        file.write_at(0, &[0u8; 12]);
      }
    }

    let mut cache = Self {
      path_str,
      mmap: None,
      index: BTreeMap::new(),
    };
    cache.remap();
    cache
  }

  /// Drops any active File lock map, rebuilds map from disk, and indexes headers locally.
  fn remap(&mut self) {
    self.mmap = None;
    self.mmap = Mmap::open(&self.path_str).ok();
    self.index.clear();

    if let Some(mmap) = &self.mmap {
      let data = mmap.as_ref();
      let mut block_off = 0usize;

      loop {
        if block_off + 4 > data.len() {
          break;
        }
        let count = u32::from_le_bytes(data[block_off..block_off + 4].try_into().unwrap()) as usize;

        for i in 0..count {
          let slot = block_off + 4 + i * 28;
          if slot + 28 > data.len() {
            break;
          }
          let header = TextureHeader::from_bytes(&data[slot..slot + 28]);
          if header.id != 0 {
            let size = header.format.byte_size(header.width, header.height) as usize;
            self.index.insert(
              header.id,
              (
                header.width,
                header.height,
                header.format,
                header.offset,
                size,
              ),
            );
          }
        }

        let next_ptr_off = block_off + 4 + count * 28;
        if next_ptr_off + 8 > data.len() {
          break;
        }
        let next_ptr = u64::from_le_bytes(data[next_ptr_off..next_ptr_off + 8].try_into().unwrap());
        if next_ptr == 0 {
          break;
        }
        block_off = next_ptr as usize;
      }
    }
  }

  pub fn get(&self, id: u64) -> Option<(u32, u32, TexelFormat, Bytes)> {
    let &(w, h, fmt, offset, size) = self.index.get(&id)?;
    let data = self.mmap.as_ref()?.as_ref();

    let end = offset as usize + size;
    if end <= data.len() {
      let slice = &data[offset as usize..end];
      // Since we can't safely wrap Mmap data into Bytes without lifetime bounds, we copy.
      // (A true zero-copy would require Bytes to implement a custom memory mapped dropping behavior
      // or for Texture to reference lifetimes, which limits ECS usage.)
      Some((w, h, fmt, Bytes::copy_from_slice(slice)))
    } else {
      None
    }
  }

  pub fn decode_and_insert(
    &mut self,
    file_path: &str,
    file_data: &[u8],
  ) -> Result<u64, &'static str> {
    let id = hash_path(file_path);
    if self.index.contains_key(&id) {
      return Ok(id);
    }

    let ext = file_path.rsplit('.').next().unwrap_or("");
    let (format, width, height, pixels) = match ext {
      "ktx2" => {
        let reader = ktx2::Reader::new(file_data).map_err(|_| "KTX2 Parse")?;
        let header = reader.header();
        let fmt = match header.format {
          Some(ktx2::Format::BC7_UNORM_BLOCK) => TexelFormat::BC7_UNORM_BLOCK,
          Some(ktx2::Format::ETC2_R8G8B8_UNORM_BLOCK) => TexelFormat::ETC2_R8G8B8_UNORM_BLOCK,
          Some(ktx2::Format::ASTC_4x4_UNORM_BLOCK) => TexelFormat::ASTC_4x4_UNORM_BLOCK,
          _ => TexelFormat::Undefined,
        };
        let level = reader.levels().next().ok_or("KTX2 Lvl")?;
        (
          fmt,
          header.pixel_width,
          header.pixel_height,
          level.data.to_vec(),
        )
      }
      "png" => {
        let (header, image_data) = png_decoder::decode(file_data).map_err(|_| "PNG Decode")?;
        (
          if header.color_type == png_decoder::ColorType::RgbAlpha {
            TexelFormat::R8G8B8A8_UNORM
          } else {
            TexelFormat::R8_UNORM
          },
          header.width,
          header.height,
          image_data.into_flattened(),
        )
      }
      "jpg" | "jpeg" => {
        let mut decoder =
          zune_jpeg::JpegDecoder::new(zune_core::bytestream::ZCursor::new(file_data));
        let info = decoder.info().ok_or("JPEG Info")?;
        let data = decoder.decode().map_err(|_| "JPEG Decode")?;
        let format = match info.components {
          1 => TexelFormat::R8_UNORM,
          3 => {
            let mut rgba = Vec::with_capacity(data.len() / 3 * 4);
            for chunk in data.chunks_exact(3) {
              rgba.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
            }
            return Ok(self.decode_and_insert_raw(
              id,
              info.width as u32,
              info.height as u32,
              TexelFormat::R8G8B8A8_UNORM,
              &rgba,
            ));
          }
          4 => TexelFormat::R8G8B8A8_UNORM,
          _ => return Err("JPEG Format Unrecognized"),
        };
        (format, info.width as u32, info.height as u32, data)
      }
      _ => return Err("Format Unrecognized"),
    };

    self.insert(id, width, height, format, &pixels);
    Ok(id)
  }

  fn decode_and_insert_raw(
    &mut self,
    id: u64,
    width: u32,
    height: u32,
    format: TexelFormat,
    pixels: &[u8],
  ) -> u64 {
    self.insert(id, width, height, format, pixels);
    id
  }

  pub fn insert(&mut self, id: u64, width: u32, height: u32, format: TexelFormat, pixels: &[u8]) {
    self.mmap = None;
    let file = sys::NativeFile::open(&self.path_str).unwrap();

    let mut block_off = 0u64;
    let mut hole_off = None;
    let mut last_next_ptr_off = 0u64;

    // 1. Identify any tombstones ("holes") from previous `.remove()` calls
    loop {
      let mut count_buf = [0u8; 4];
      if !file.read_at(block_off, &mut count_buf) {
        break;
      }
      let count = u32::from_le_bytes(count_buf);

      for i in 0..count {
        let slot = block_off + 4 + (i as u64) * 28;
        let mut id_buf = [0u8; 8];
        if file.read_at(slot, &mut id_buf) && u64::from_le_bytes(id_buf) == 0 && hole_off.is_none()
        {
          hole_off = Some(slot);
        }
      }

      let next_ptr_off = block_off + 4 + (count as u64) * 28;
      let mut next_buf = [0u8; 8];
      if !file.read_at(next_ptr_off, &mut next_buf) {
        break;
      }
      let next_ptr = u64::from_le_bytes(next_buf);

      if next_ptr == 0 {
        last_next_ptr_off = next_ptr_off;
        break;
      }
      block_off = next_ptr;
    }

    // 2. Write pixel payload exactly at current EOF
    let data_offset = file.size();
    file.write_at(data_offset, pixels);
    let header = TextureHeader {
      id,
      offset: data_offset,
      width,
      height,
      format,
    };

    // 3. Patch hole, or Append a brand new Variable length Link List element Block.
    if let Some(hole) = hole_off {
      file.write_at(hole, &header.to_bytes());
    } else {
      let new_block_off = data_offset + pixels.len() as u64;
      let mut new_block = [0u8; 40]; // [Count (4)] + [1x Header (28)] + [Next_Ptr (8)]
      new_block[0..4].copy_from_slice(&1u32.to_le_bytes());
      new_block[4..32].copy_from_slice(&header.to_bytes());
      new_block[32..40].copy_from_slice(&0u64.to_le_bytes());

      file.write_at(new_block_off, &new_block);
      file.write_at(last_next_ptr_off, &new_block_off.to_le_bytes());
    }

    drop(file);
    self.remap();
  }

  pub fn remove(&mut self, id: u64) {
    self.mmap = None;
    let file = sys::NativeFile::open(&self.path_str).unwrap();
    let mut block_off = 0u64;

    loop {
      let mut count_buf = [0u8; 4];
      if !file.read_at(block_off, &mut count_buf) {
        break;
      }
      let count = u32::from_le_bytes(count_buf);

      for i in 0..count {
        let slot = block_off + 4 + (i as u64) * 28;
        let mut id_buf = [0u8; 8];
        if file.read_at(slot, &mut id_buf) && u64::from_le_bytes(id_buf) == id {
          file.write_at(slot, &0u64.to_le_bytes()); // Generates tombstone ID (0)
          drop(file);
          self.remap();
          return;
        }
      }

      let next_ptr_off = block_off + 4 + (count as u64) * 28;
      let mut next_buf = [0u8; 8];
      if !file.read_at(next_ptr_off, &mut next_buf) {
        break;
      }
      let next_ptr = u64::from_le_bytes(next_buf);
      if next_ptr == 0 {
        break;
      }
      block_off = next_ptr;
    }
    drop(file);
    self.remap();
  }

  /// Restructures the format into a single linear block file.
  pub fn defragment(&mut self) {
    let valid_headers: Vec<_> = self.index.keys().filter_map(|k| self.get_header(*k)).collect();
    self.mmap = None;

    let tmp_path = alloc::format!("{}.tmp", self.path_str);
    let old_file = sys::NativeFile::open(&self.path_str).unwrap();
    let new_file = sys::NativeFile::create_trunc(&tmp_path).unwrap();

    let count = valid_headers.len() as u32;
    let headers_size = 4 + (count as u64) * 28 + 8;
    let mut data_offset = headers_size;

    new_file.write_at(0, &count.to_le_bytes()); // Write single block header count

    for (i, mut header) in valid_headers.into_iter().enumerate() {
      let size = header.format.byte_size(header.width, header.height) as usize;
      let mut buf = alloc::vec![0u8; size];
      old_file.read_at(header.offset, &mut buf);

      new_file.write_at(data_offset, &buf); // Coalesce data linearly
      header.offset = data_offset;
      data_offset += size as u64;

      let slot = 4 + (i as u64) * 28;
      new_file.write_at(slot, &header.to_bytes());
    }

    new_file.write_at(4 + (count as u64) * 28, &0u64.to_le_bytes()); // Terminate with ptr=0

    drop(old_file);
    drop(new_file);
    sys::rename_file(&tmp_path, &self.path_str);
    self.remap();
  }

  fn get_header(&self, id: u64) -> Option<TextureHeader> {
    let &(width, height, format, offset, _sz) = self.index.get(&id)?;
    Some(TextureHeader {
      id,
      offset,
      width,
      height,
      format,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;

  #[test]
  fn test_texture_cache_lifecycle() {
    let mut cache = TextureCache::new("test_aethervk_tex_cache_1");
    let png_data = fs::read("../../test_assets/rain.png").unwrap();
    let id = cache.decode_and_insert("../../test_assets/rain.png", &png_data).unwrap();

    let (w, h, fmt, bytes) = cache.get(id).unwrap();
    assert!(w > 0);
    assert!(h > 0);
    assert_eq!(fmt.to_u32(), TexelFormat::R8G8B8A8_UNORM.to_u32());
    assert!(!bytes.is_empty());

    cache.remove(id);
    assert!(cache.get(id).is_none());

    // Clean up
    let _ = fs::remove_file(cache.path_str.clone());
  }

  #[test]
  fn test_texture_cache_multiple_inserts_and_defragment() {
    let mut cache = TextureCache::new("test_aethervk_tex_cache_2");
    let rain_data = fs::read("../../test_assets/rain.png").unwrap();
    let cloud_data = fs::read("../../test_assets/cloud.png").unwrap();
    let sun_data = fs::read("../../test_assets/sun.png").unwrap();

    let id1 = cache.decode_and_insert("../../test_assets/rain.png", &rain_data).unwrap();
    let id2 = cache.decode_and_insert("../../test_assets/cloud.png", &cloud_data).unwrap();
    let id3 = cache.decode_and_insert("../../test_assets/sun.png", &sun_data).unwrap();

    assert!(cache.get(id1).is_some());
    assert!(cache.get(id2).is_some());
    assert!(cache.get(id3).is_some());

    // Remove one to create a hole
    cache.remove(id2);
    assert!(cache.get(id2).is_none());

    // Defragment should remove the hole
    cache.defragment();

    assert!(cache.get(id1).is_some());
    assert!(cache.get(id2).is_none());
    assert!(cache.get(id3).is_some());

    let _ = fs::remove_file(cache.path_str.clone());
  }

  #[test]
  fn test_texture_cache_reopen() {
    let path_str;
    let id;
    {
      let mut cache = TextureCache::new("test_aethervk_tex_cache_3");
      let sun_data = fs::read("../../test_assets/sun.png").unwrap();
      id = cache.decode_and_insert("../../test_assets/sun.png", &sun_data).unwrap();
      path_str = cache.path_str.clone();
    }

    // Reopen cache
    let cache2 = TextureCache::new("test_aethervk_tex_cache_3");
    let res = cache2.get(id);
    assert!(res.is_some());

    let _ = fs::remove_file(path_str);
  }
}
