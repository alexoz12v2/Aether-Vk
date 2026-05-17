//! files module.

use crate::{
  os,
  os::{FsError, NativeError, NativeResult, fs::Path},
};

/// TODO: Document this item
pub struct MappedFile {
  ptr: *mut core::ffi::c_void,
  len: usize,
}

// Memory mappings are read-only byte arrays natively managed by the OS.
// They are inherently thread-safe and immutable.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
  /// TODO: Document this item
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, FsError> {
    let mut path_buf = path.as_ref().to_pathbuf();

    #[cfg(windows)]
    {
      use core::ffi::c_void;

      use windows::Win32::{
        Foundation::{CloseHandle, GENERIC_READ, HANDLE, INVALID_HANDLE_VALUE},
        Storage::FileSystem::{
          CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, GetFileSizeEx, OPEN_EXISTING,
        },
        System::Memory::{CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY},
      };

      // Polyfill traits to cleanly handle API breaks between `windows` >=0.59 and <=0.62
      // where pointer return values fluctuate between raw primitives and Result<T>.
      trait MapHandleExt {
        fn into_handle(self) -> HANDLE;
      }
      impl MapHandleExt for HANDLE {
        fn into_handle(self) -> HANDLE {
          self
        }
      }
      impl<E> MapHandleExt for Result<HANDLE, E> {
        fn into_handle(self) -> HANDLE {
          self.unwrap_or_default()
        }
      }

      trait MapPtrExt {
        fn into_ptr(self) -> *mut core::ffi::c_void;
      }
      impl MapPtrExt for *mut core::ffi::c_void {
        fn into_ptr(self) -> *mut core::ffi::c_void {
          self
        }
      }
      impl<E> MapPtrExt for Result<*mut core::ffi::c_void, E> {
        fn into_ptr(self) -> *mut core::ffi::c_void {
          self.unwrap_or(core::ptr::null_mut())
        }
      }

      let handle = unsafe {
        CreateFileW(
          windows::core::PCWSTR(path_buf.as_ptr_mut()),
          GENERIC_READ.0,
          FILE_SHARE_READ,
          None,
          OPEN_EXISTING,
          FILE_ATTRIBUTE_NORMAL,
          None,
        )
      }
      .map_err(|_| FsError::CouldNotOpenFile)?;

      if handle == INVALID_HANDLE_VALUE {
        return Err(FsError::CouldNotOpenFile);
      }

      let mut size: i64 = 0;
      if unsafe { GetFileSizeEx(handle, &mut size) }.is_err() {
        let _ = unsafe { CloseHandle(handle) };
        return Err(FsError::CouldNotGetFileSize);
      }
      let len = size as usize;

      // Zero-length files natively panic/fail memory mappers on both OSs. Trap and handle them safely here.
      if len == 0 {
        let _ = unsafe { CloseHandle(handle) };
        return Ok(MappedFile {
          ptr: core::ptr::NonNull::<core::ffi::c_void>::dangling().as_ptr(),
          len: 0,
        });
      }

      let mapping_handle =
        unsafe { CreateFileMappingW(handle, None, PAGE_READONLY, 0, 0, None) }.into_handle();

      let is_invalid = mapping_handle == INVALID_HANDLE_VALUE
        || unsafe { core::mem::transmute_copy::<_, isize>(&mapping_handle) } == 0;

      if is_invalid {
        let _ = unsafe { CloseHandle(handle) };
        return Err(FsError::CouldNotReadFile);
      }

      let view_ptr: *mut c_void =
        unsafe { MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, len) }.Value;

      // Prevent handle leaks! The OS map view retains an internal reference to the
      // file object behind the scenes, making it entirely safe to close handles immediately.
      let _ = unsafe { CloseHandle(mapping_handle) };
      let _ = unsafe { CloseHandle(handle) };

      if view_ptr.is_null() {
        return Err(FsError::CouldNotReadFile);
      }

      Ok(MappedFile { ptr: view_ptr, len })
    }

    #[cfg(not(windows))]
    {
      use libc::{MAP_FAILED, MAP_PRIVATE, O_RDONLY, PROT_READ, close, fstat, mmap, open};

      let fd = unsafe { open(path_buf.as_ptr_mut(), O_RDONLY) };
      if fd < 0 {
        return Err(FsError::CouldNotOpenFile);
      }

      let mut stat_buf: libc::stat = unsafe { core::mem::zeroed() };
      if unsafe { fstat(fd, &mut stat_buf) } != 0 {
        unsafe { close(fd) };
        return Err(FsError::CouldNotGetFileSize);
      }
      let len = stat_buf.st_size as usize;

      if len == 0 {
        unsafe { close(fd) };
        return Ok(MappedFile {
          ptr: core::ptr::NonNull::<core::ffi::c_void>::dangling().as_ptr(),
          len: 0,
        });
      }

      let mapped_ptr = unsafe { mmap(core::ptr::null_mut(), len, PROT_READ, MAP_PRIVATE, fd, 0) };

      // The POSIX spec guarantees memory mappings hold an internal reference to the `fd` until unmapped.
      unsafe { close(fd) };

      if mapped_ptr == MAP_FAILED {
        return Err(FsError::CouldNotReadFile);
      }

      Ok(MappedFile {
        ptr: mapped_ptr,
        len,
      })
    }
  }

  #[inline(always)]
  /// TODO: Document this item
  pub fn as_slice(&self) -> &[u8] {
    if self.len == 0 || self.ptr.is_null() {
      &[]
    } else {
      unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
  }
}

impl Drop for MappedFile {
  fn drop(&mut self) {
    if self.len == 0 || self.ptr.is_null() {
      return;
    }

    #[cfg(windows)]
    {
      use windows::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};

      let _ = unsafe {
        UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
          Value: self.ptr as *mut core::ffi::c_void,
        })
      };
    }
    #[cfg(not(windows))]
    {
      unsafe {
        libc::munmap(self.ptr, self.len);
      }
    }
  }
}

// Ergonomic integrations so `MappedFile` can be treated strictly as a byte slice array.
impl AsRef<[u8]> for MappedFile {
  #[inline(always)]
  fn as_ref(&self) -> &[u8] {
    self.as_slice()
  }
}

impl core::ops::Deref for MappedFile {
  type Target = [u8];

  #[inline(always)]
  fn deref(&self) -> &Self::Target {
    self.as_slice()
  }
}

impl core::borrow::Borrow<[u8]> for MappedFile {
  #[inline(always)]
  fn borrow(&self) -> &[u8] {
    self.as_slice()
  }
}

/// A cross-platform zero-copy memory mappeed file abstraction
pub struct Mmap {
  ptr: *mut u8,
  len: usize,
}

// memory mappings are thread safe for immutable read-only access
unsafe impl Send for Mmap {}
unsafe impl Sync for Mmap {}

// Required by `bytes::Bytes::from_owner()` so it knows the extent of the memory region
impl AsRef<[u8]> for Mmap {
  #[inline]
  fn as_ref(&self) -> &[u8] {
    if self.len == 0 {
      &[]
    } else {
      unsafe { core::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }
  }
}

#[cfg(target_family = "unix")]
impl Mmap {
  /// TODO: Document this item
  pub fn open<P: AsRef<os::fs::Path>>(path: P) -> NativeResult<Self> {
    use alloc::string::ToString;
    use core::{mem, ptr};
    use libc::{MAP_FAILED, MAP_PRIVATE, O_RDONLY, PROT_READ, close, fstat, mmap, open};

    let path_str = path.as_ref().to_str_unified().ok_or(NativeError::InvalidArgument)?.to_string();
    let c_path = alloc::ffi::CString::new(path_str).map_err(|_| NativeError::InvalidArgument)?;
    let fd = unsafe { open(c_path.as_ptr(), O_RDONLY) };
    if fd < 0 {
      return Err(NativeError::OsFsError(FsError::CouldNotOpenFile));
    }

    let mut stat_buf: libc::stat = unsafe { mem::zeroed() };
    if unsafe { fstat(fd, &mut stat_buf) } != 0 {
      unsafe { close(fd) };
      return Err(NativeError::OsFsError(FsError::CouldNotGetFileSize));
    }

    let size = stat_buf.st_size as usize;
    if size == 0 {
      unsafe { close(fd) };
      return Ok(Self {
        ptr: ptr::NonNull::dangling().as_ptr(),
        len: 0,
      });
    }

    let ptr = unsafe { mmap(ptr::null_mut(), size, PROT_READ, MAP_PRIVATE, fd, 0) };
    // POSIX docs: mapping survives even after closing the file descriptor
    unsafe { close(fd) };

    if ptr == MAP_FAILED {
      return Err(NativeError::OsFsError(FsError::CouldNotReadFile));
    }
    Ok(Self {
      ptr: ptr as *mut u8,
      len: size,
    })
  }
}

#[cfg(target_family = "unix")]
impl Drop for Mmap {
  fn drop(&mut self) {
    if self.len > 0 {
      unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
  }
}

#[cfg(windows)]
impl Mmap {
  /// TODO: Document this item
  pub fn open<P: AsRef<os::fs::Path>>(path: P) -> NativeResult<Self> {
    use windows::Win32::{
      Foundation::{CloseHandle, GENERIC_READ, INVALID_HANDLE_VALUE},
      Storage::FileSystem::{
        CreateFileW, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, GetFileSizeEx, OPEN_EXISTING,
      },
      System::Memory::{CreateFileMappingW, FILE_MAP_READ, MapViewOfFile, PAGE_READONLY},
    };
    let path = {
      let mut p: alloc::vec::Vec<u16> = path
        .as_ref()
        .to_str_unified()
        .map(|cow| cow.into_owned().encode_utf16().collect())
        .ok_or(NativeError::InvalidArgument)?;
      p.push(0u16);
      p
    };
    let handle = unsafe {
      CreateFileW(
        windows::core::PCWSTR(path.as_ptr()),
        GENERIC_READ.0,
        FILE_SHARE_READ,
        None,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        None,
      )
    }
    .map_err(|_| NativeError::OsFsError(FsError::CouldNotOpenFile))?;

    if handle == INVALID_HANDLE_VALUE || handle.is_invalid() {
      return Err(NativeError::OsFsError(FsError::CouldNotOpenFile));
    }

    let size = {
      let mut sz: i64 = 0;
      if unsafe { GetFileSizeEx(handle, &mut sz) }.is_err() {
        let _ = unsafe { CloseHandle(handle) };
        return Err(NativeError::OsFsError(FsError::CouldNotGetFileSize));
      }
      sz as usize
    };

    if size == 0 {
      let _ = unsafe { CloseHandle(handle) };
      return Ok(Self {
        ptr: core::ptr::NonNull::dangling().as_ptr(),
        len: 0,
      });
    }

    let mapping = unsafe {
      CreateFileMappingW(handle, None, PAGE_READONLY, 0, 0, None).map_err(|_| {
        let _ = CloseHandle(handle);
        NativeError::OsFsError(FsError::CouldNotReadFile)
      })?
    };

    // View retains reference to the mapping. It is safe to close the file
    let _ = unsafe { CloseHandle(handle) };

    let view = unsafe { MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0) };

    // Mapped memory view holds reference to mapping. Close mapping
    let _ = unsafe { CloseHandle(mapping) };

    if view.Value.is_null() {
      return Err(NativeError::OsFsError(FsError::CouldNotReadFile));
    }

    Ok(Self {
      ptr: view.Value as *mut u8,
      len: size,
    })
  }
}

#[cfg(windows)]
impl Drop for Mmap {
  fn drop(&mut self) {
    if self.len > 0 {
      use windows::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
      unsafe {
        let _ = UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
          Value: self.ptr as *mut core::ffi::c_void,
        });
      }
    }
  }
}

// TODO add tests for Mmap
