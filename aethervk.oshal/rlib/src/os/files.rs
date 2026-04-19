use crate::os::fs::Path;

// TODO move to os.rs
pub enum FsError {
  CouldNotOpenFile,
  CouldNotReadFile,
  CouldNotGetFileSize,
  CouldNotGetCurrentExe,
  CouldNotCreateFile,
  CouldNotWriteFile,
}

pub struct MappedFile {
  ptr: *mut core::ffi::c_void,
  len: usize,
}

// Memory mappings are read-only byte arrays natively managed by the OS.
// They are inherently thread-safe and immutable.
unsafe impl Send for MappedFile {}
unsafe impl Sync for MappedFile {}

impl MappedFile {
  pub fn new<P: AsRef<Path>>(path: P) -> Result<Self, FsError> {
    let mut path_buf = path.as_ref().to_pathbuf();

    #[cfg(windows)]
    {
      use windows::Win32::Foundation::{CloseHandle, GENERIC_READ, HANDLE, INVALID_HANDLE_VALUE};
      use windows::Win32::Storage::FileSystem::{
        CreateFileW, GetFileSizeEx, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ, OPEN_EXISTING,
      };
      use windows::Win32::System::Memory::{
        CreateFileMappingW, MapViewOfFile, FILE_MAP_READ, PAGE_READONLY,
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

      let view_ptr = unsafe { MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, len) }.into_ptr();

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
      use libc::{close, fstat, mmap, open, MAP_FAILED, MAP_PRIVATE, O_RDONLY, PROT_READ};

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
      use windows::Win32::System::Memory::UnmapViewOfFile;
      let _ = unsafe { UnmapViewOfFile(self.ptr as *const core::ffi::c_void) };
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
