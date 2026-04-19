use core::borrow::Borrow;

use alloc::vec::Vec;

use crate::os::FsError;

#[cfg(windows)]
#[allow(non_camel_case_types)]
pub type os_char = u16;
#[cfg(not(windows))]
#[allow(non_camel_case_types)]
pub type os_char = core::ffi::c_char;

pub const SEP: os_char = if cfg!(windows) {
  b'\\' as os_char
} else {
  b'/' as os_char
};

/// necessary utility to ensure equality and hash are not dependant on nul termination
pub(crate) fn strip_nul(slice: &[os_char]) -> &[os_char] {
  if let Some((&last, rest)) = slice.split_last() {
    if last == b'\0' as os_char {
      return rest;
    }
  }
  slice
}

pub trait FileSystemObject {
  fn is_valid(&self) -> bool;
  fn is_dir(&self) -> bool;
  fn is_file(&self) -> bool;
}

pub struct Path {
  inner: [os_char],
}

impl From<&str> for &Path {
  fn from(s: &str) -> Self {
    let the_ptr: *const os_char = s.as_ptr().cast();
    Path::from_slice(unsafe { core::slice::from_raw_parts(the_ptr, s.len()) })
  }
}

impl PartialEq for Path {
  fn eq(&self, other: &Self) -> bool {
    strip_nul(&self.inner) == strip_nul(&other.inner)
  }
}

impl Eq for Path {}

impl core::hash::Hash for Path {
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    strip_nul(&self.inner).hash(state);
  }
}

impl Path {
  pub fn from_slice(slice: &[os_char]) -> &Self {
    let ptr: *const Self = (slice as *const [os_char]) as *const Path;
    unsafe { &*ptr }
  }

  pub fn to_pathbuf(&self) -> PathBuf {
    let mut the_pathbuf = PathBuf::new();
    the_pathbuf.push_slice(&self.inner);

    the_pathbuf
  }
}

impl FileSystemObject for Path {
  fn is_valid(&self) -> bool {
    #[cfg(windows)]
    {
      use windows::Win32::Storage::FileSystem::{GetFileAttributesW, INVALID_FILE_ATTRIBUTES};

      let mut path_buf = self.to_pathbuf();
      let attrs = unsafe { GetFileAttributesW(windows::core::PCWSTR(path_buf.as_ptr_mut())) };
      attrs != INVALID_FILE_ATTRIBUTES
    }
    #[cfg(not(windows))]
    {
      let mut path_buf = self.to_pathbuf();
      let mut stat_buf: libc::stat = unsafe { core::mem::zeroed() };
      let result = unsafe { libc::stat(path_buf.as_ptr_mut(), &mut stat_buf) };
      result == 0
    }
  }

  fn is_dir(&self) -> bool {
    #[cfg(windows)]
    {
      use windows::Win32::Storage::FileSystem::{
        GetFileAttributesW, FILE_ATTRIBUTE_DIRECTORY, INVALID_FILE_ATTRIBUTES,
      };

      let mut path_buf = self.to_pathbuf();
      let attrs = unsafe { GetFileAttributesW(windows::core::PCWSTR(path_buf.as_ptr_mut())) };
      attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY.0) != 0
    }
    #[cfg(not(windows))]
    {
      let mut path_buf = self.to_pathbuf();
      let mut stat_buf: libc::stat = unsafe { core::mem::zeroed() };
      let result = unsafe { libc::stat(path_buf.as_ptr_mut(), &mut stat_buf) };
      if result != 0 {
        return false;
      }
      (stat_buf.st_mode & libc::S_IFMT) == libc::S_IFDIR
    }
  }

  fn is_file(&self) -> bool {
    #[cfg(windows)]
    {
      use windows::Win32::Storage::FileSystem::{
        GetFileAttributesW, FILE_ATTRIBUTE_DIRECTORY, INVALID_FILE_ATTRIBUTES,
      };

      let mut path_buf = self.to_pathbuf();
      let attrs = unsafe { GetFileAttributesW(windows::core::PCWSTR(path_buf.as_ptr_mut())) };
      attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_DIRECTORY.0) == 0
    }
    #[cfg(not(windows))]
    {
      let mut path_buf = self.to_pathbuf();
      let mut stat_buf: libc::stat = unsafe { core::mem::zeroed() };
      let result = unsafe { libc::stat(path_buf.as_ptr_mut(), &mut stat_buf) };
      if result != 0 {
        return false;
      }
      (stat_buf.st_mode & libc::S_IFMT) == libc::S_IFREG
    }
  }
}

impl<T> FileSystemObject for T
where
  T: AsRef<Path>,
{
  fn is_valid(&self) -> bool {
    self.as_ref().is_valid()
  }

  fn is_dir(&self) -> bool {
    self.as_ref().is_dir()
  }

  fn is_file(&self) -> bool {
    self.as_ref().is_file()
  }
}

// TODO: move to a utils
#[cfg(windows)]
fn utf8_to_utf16(s: &str, out: &mut Vec<u16>) {
  out.extend(s.encode_utf16());
}

#[derive(Clone)]
pub struct PathBuf {
  storage: PathStorage,
}

impl PartialEq for PathBuf {
  fn eq(&self, other: &Self) -> bool {
    strip_nul(self.as_slice()) == strip_nul(other.as_slice())
  }
}

impl Eq for PathBuf {}

impl core::hash::Hash for PathBuf {
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    strip_nul(self.as_slice()).hash(state);
  }
}

const PATH_SBO_CAP: usize = if cfg!(windows) { 32 } else { 64 };

#[derive(Clone)]
enum PathStorage {
  Inline(heapless::Vec<os_char, PATH_SBO_CAP>),
  Heap(Vec<os_char>),
}

impl PathBuf {
  pub fn new() -> Self {
    Self {
      storage: PathStorage::Inline(heapless::Vec::new()),
    }
  }

  pub fn clone(&self) -> Self {
    Self {
      storage: self.storage.clone(),
    }
  }

  pub fn parent(&self) -> Option<PathBuf> {
    let slice = self.as_slice();
    if let Some(pos) = slice.iter().rposition(|&c| c == SEP) {
      let mut parent_path = PathBuf::new();
      parent_path.push_slice(&slice[..pos]);
      Some(parent_path)
    } else {
      None
    }
  }

  pub fn join(&self, path: &str) -> PathBuf {
    let mut p = self.clone();
    p.pop_nul_if_present();
    if !p.as_slice().is_empty() && p.as_slice()[p.as_slice().len() - 1] != SEP {
      p.push_unit(SEP);
    }
    p.push(path);
    p
  }

  pub fn push_slice(&mut self, s: &[os_char]) {
    self.pop_nul_if_present();
    for &unit in s {
      #[cfg(windows)]
      let unit = if unit == b'/' as os_char {
        b'\\' as os_char
      } else {
        unit
      };
      self.push_unit(unit);
    }
  }

  pub fn push(&mut self, s: &str) {
    self.pop_nul_if_present();
    #[cfg(windows)]
    let the_iter = s.encode_utf16();
    #[cfg(not(windows))]
    let the_iter = s.bytes();
    for unit in the_iter {
      #[cfg(windows)]
      let unit = if unit == b'/' as os_char {
        b'\\' as os_char
      } else {
        unit
      };
      self.push_unit(unit as os_char);
    }
  }

  // nul terminated for ffi/os api compatibility
  pub(crate) fn as_ptr_mut(&mut self) -> *const os_char {
    self.ensure_nul_terminated();
    self.as_slice().as_ptr()
  }

  pub fn as_slice(&self) -> &[os_char] {
    match &self.storage {
      PathStorage::Inline(vec_inner) => &vec_inner,
      PathStorage::Heap(items) => &items,
    }
  }

  pub fn pop(&mut self) {
    while let Some(c) = self.pop_unit() {
      if c == SEP {
        return;
      }
    }
  }

  pub fn extension(&self) -> Option<&str> {
    let slice = self.as_slice();
    if let Some(pos) = slice.iter().rposition(|&c| c == b'.' as os_char) {
      #[cfg(windows)]
      {
        let s = unsafe {
          core::slice::from_raw_parts(slice.as_ptr().add(pos + 1), slice.len() - pos - 1)
        };
        Some(alloc::string::String::from_utf16_lossy(s))
      }
      #[cfg(not(windows))]
      {
        let s = unsafe {
          core::str::from_utf8_unchecked(core::slice::from_raw_parts(
            slice.as_ptr().add(pos + 1) as *const u8,
            slice.len() - pos - 1,
          ))
        };
        Some(s)
      }
    } else {
      None
    }
  }

  fn push_unit(&mut self, unit: os_char) {
    match &mut self.storage {
      PathStorage::Inline(inline_vec) => {
        if inline_vec.is_full() {
          self.promote_to_heap();
          self.push_unit(unit);
        } else {
          unsafe { inline_vec.push_unchecked(unit) };
        }
      }
      PathStorage::Heap(the_vec) => {
        the_vec.push(unit);
      }
    }
  }

  fn pop_unit(&mut self) -> Option<os_char> {
    match &mut self.storage {
      PathStorage::Inline(vec_inner) => {
        if !vec_inner.is_empty() {
          Some(unsafe { vec_inner.pop_unchecked() })
        } else {
          None
        }
      }
      PathStorage::Heap(items) => {
        if !items.is_empty() {
          Some(items.pop().unwrap())
        } else {
          None
        }
      }
    }
  }

  fn promote_to_heap(&mut self) {
    match &mut self.storage {
      PathStorage::Inline(inline_vec) => {
        let mut the_vec = Vec::with_capacity(2 * PATH_SBO_CAP);
        the_vec.extend_from_slice(&inline_vec);
        self.storage = PathStorage::Heap(the_vec);
      }
      _ => {}
    };
  }

  #[inline]
  fn pop_nul_if_present(&mut self) {
    let nul_char = b'\0' as os_char;

    match &mut self.storage {
      PathStorage::Inline(vec_inner) => {
        if vec_inner.last() == Some(&nul_char) {
          vec_inner.pop();
        }
      }
      PathStorage::Heap(items) => {
        if items.last() == Some(&nul_char) {
          items.pop();
        }
      }
    }
  }

  #[inline]
  fn ensure_nul_terminated(&mut self) {
    let nul_char = b'\0' as os_char;

    match &mut self.storage {
      PathStorage::Inline(vec_inner) => {
        // If it's empty, or the last char is NOT null, push a null
        if vec_inner.last() != Some(&nul_char) {
          self.push_unit(nul_char);
        }
      }
      PathStorage::Heap(items) => {
        // Fix: Changed '==' to '!=' to match the Inline logic
        if items.last() != Some(&nul_char) {
          items.push(nul_char);
        }
      }
    }
  }
}

impl<T: AsRef<str>> From<T> for PathBuf {
  fn from(s: T) -> Self {
    let mut buf = PathBuf::new();
    buf.push(s.as_ref());
    buf
  }
}

impl AsRef<Path> for PathBuf {
  fn as_ref(&self) -> &Path {
    Path::from_slice(strip_nul(self.as_slice()))
  }
}

pub fn current_exe() -> Result<PathBuf, FsError> {
  #[cfg(windows)]
  {
    use alloc::vec;
    use windows::Win32::System::LibraryLoader::GetModuleFileNameW;

    const MAX_PATH_WIN: usize = 260;
    let mut buffer: Vec<u16> = vec![0; MAX_PATH_WIN];
    loop {
      let written_len = unsafe { GetModuleFileNameW(None, &mut buffer) } as usize;
      if written_len == 0 {
        return Err(FsError::CouldNotGetCurrentExe);
      }
      if written_len < buffer.len() {
        buffer.truncate(written_len);
        let mut path_buf = PathBuf::new();
        path_buf.push_slice(&buffer);
        return Ok(path_buf);
      }
      buffer.resize(buffer.len() * 2, 0);
    }
  }
  #[cfg(target_os = "linux")]
  {
    use alloc::vec;

    let mut buffer: Vec<u8> = vec![0; 260];
    loop {
      let result = unsafe {
        libc::readlink(
          b"/proc/self/exe\0".as_ptr() as *const i8,
          buffer.as_mut_ptr() as *mut i8,
          buffer.len(),
        )
      };

      if result < 0 {
        return Err(FsError::CouldNotGetCurrentExe);
      }

      let len = result as usize;
      if len < buffer.len() {
        buffer.truncate(len);
        let s = unsafe { core::str::from_utf8_unchecked(&buffer) };
        return Ok(PathBuf::from(s));
      }
      buffer.resize(buffer.len() * 2, 0);
    }
  }
  #[cfg(target_os = "macos")]
  {
    use alloc::vec;
    use libc::{c_char, c_uint};

    let mut buf_size: c_uint = 0;
    unsafe {
      // This call will fail but will set buf_size to the required size.
      // TODO: does the mach2 crate have an equivalent which is not deprecated? well there's a FIXME that says
      // that maybe this function will get undeprecated
      libc::_NSGetExecutablePath(core::ptr::null_mut(), &mut buf_size)
    };

    if buf_size == 0 {
      return Err(FsError::CouldNotGetCurrentExe);
    }

    let mut buffer: Vec<c_char> = vec![0; buf_size as usize];
    let result = unsafe { libc::_NSGetExecutablePath(buffer.as_mut_ptr(), &mut buf_size) };

    if result != 0 {
      return Err(FsError::CouldNotGetCurrentExe);
    }

    // Find the length of the string, it might not be null terminated if buffer is full
    let len = buffer.iter().position(|&c| c == 0).unwrap_or(buffer.len());

    let s = unsafe {
      core::str::from_utf8_unchecked(core::slice::from_raw_parts(
        buffer.as_ptr() as *const u8,
        len,
      ))
    };
    Ok(PathBuf::from(s))
  }
  #[cfg(not(any(windows, target_os = "linux", target_os = "macos")))]
  {
    Err(FsError::CouldNotGetCurrentExe) // Unsupported platform
  }
}

pub fn read(path: &Path) -> Result<Vec<u8>, FsError> {
  #[cfg(windows)]
  {
    use windows::Win32::Storage::FileSystem::{
      CreateFileW, ReadFile, GetFileSizeEx, FILE_SHARE_READ, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
    };
    use windows::Win32::Foundation::{GENERIC_READ, CloseHandle, INVALID_HANDLE_VALUE};

    let mut path_buf = path.to_pathbuf();
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
      unsafe { CloseHandle(handle) }.map_err(|_| FsError::CouldNotOpenFile)?;
      return Err(FsError::CouldNotGetFileSize);
    }

    let mut buffer = {
      let mut the_vec = Vec::new();
      the_vec.resize(size as _, 0);
      the_vec
    };
    let mut bytes_read: u32 = 0;

    unsafe {
      ReadFile(
        handle,
        Some(&mut buffer),
        Some(core::ptr::from_mut(&mut bytes_read)),
        None,
      )
    }
    .map_err(|_| FsError::CouldNotReadFile)?;

    unsafe {
      buffer.set_len(bytes_read as usize);
      CloseHandle(handle)
    }
    .map_err(|_| FsError::CouldNotOpenFile)?;

    Ok(buffer)
  }
  #[cfg(not(windows))]
  {
    use libc::{open, fstat, read, close, O_RDONLY};
    use core::mem;

    let mut path_buf = path.to_pathbuf();
    let fd = unsafe { open(path_buf.as_ptr_mut(), O_RDONLY) };
    if fd < 0 {
      return Err(FsError::CouldNotOpenFile);
    }

    let mut stat_buf: libc::stat = unsafe { mem::zeroed() };
    if unsafe { fstat(fd, &mut stat_buf) } != 0 {
      unsafe { close(fd) };
      return Err(FsError::CouldNotGetFileSize);
    }

    let size = stat_buf.st_size as usize;
    let mut buffer = Vec::with_capacity(size);

    let bytes_read = unsafe { read(fd, buffer.as_mut_ptr() as _, size) };

    unsafe {
      buffer.set_len(bytes_read as usize);
      close(fd);
    }

    if bytes_read < 0 {
      Err(FsError::CouldNotReadFile)
    } else {
      Ok(buffer)
    }
  }
}

pub fn write(path: &Path, content: &[u8]) -> Result<(), FsError> {
  #[cfg(windows)]
  {
    use windows::Win32::Storage::FileSystem::{
      CreateFileW, WriteFile, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, FILE_SHARE_READ,
    };
    use windows::Win32::Foundation::{GENERIC_WRITE, CloseHandle, INVALID_HANDLE_VALUE};

    let mut path_buf = path.to_pathbuf();
    let handle = unsafe {
      CreateFileW(
        windows::core::PCWSTR(path_buf.as_ptr_mut()),
        GENERIC_WRITE.0,
        FILE_SHARE_READ,
        None,
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        None,
      )
    }
    .map_err(|_| FsError::CouldNotCreateFile)?;

    if handle == INVALID_HANDLE_VALUE {
      return Err(FsError::CouldNotCreateFile);
    }

    let mut bytes_written: u32 = 0;
    let success = unsafe {
      WriteFile(
        handle,
        Some(content),
        Some(core::ptr::from_mut(&mut bytes_written)),
        None,
      )
    };

    unsafe { CloseHandle(handle) }.map_err(|_| FsError::CouldNotCreateFile)?;

    if success.is_err() {
      Err(FsError::CouldNotWriteFile)
    } else {
      Ok(())
    }
  }
  #[cfg(not(windows))]
  {
    use libc::{open, write, close, O_WRONLY, O_CREAT, O_TRUNC};

    let mut path_buf = path.to_pathbuf();
    // 0o666 = read and write for owner, group, and others
    let fd = unsafe { open(path_buf.as_ptr_mut(), O_WRONLY | O_CREAT | O_TRUNC, 0o666) };
    if fd < 0 {
      return Err(FsError::CouldNotCreateFile);
    }

    let bytes_written = unsafe { write(fd, content.as_ptr() as _, content.len()) };

    unsafe {
      close(fd);
    }

    if bytes_written < 0 || bytes_written as usize != content.len() {
      Err(FsError::CouldNotWriteFile)
    } else {
      Ok(())
    }
  }
}

impl Borrow<Path> for PathBuf {
  fn borrow(&self) -> &Path {
    Path::from_slice(strip_nul(self.as_slice()))
  }
}

impl core::ops::Deref for PathBuf {
  type Target = Path;

  fn deref(&self) -> &Path {
    Path::from_slice(strip_nul(self.as_slice()))
  }
}

// --- Directory Iteration Types ---

pub struct DirEntry {
  path: PathBuf,
}

impl DirEntry {
  pub fn path(&self) -> PathBuf {
    self.path.clone()
  }
}

pub struct ReadDir {
  #[cfg(windows)]
  handle: windows::Win32::Foundation::HANDLE,
  #[cfg(windows)]
  find_data: windows::Win32::Storage::FileSystem::WIN32_FIND_DATAW,
  #[cfg(windows)]
  first: bool,
  #[cfg(windows)]
  parent: PathBuf,

  #[cfg(not(windows))]
  dirp: *mut libc::DIR,
  #[cfg(not(windows))]
  parent: PathBuf,
}

impl Iterator for ReadDir {
  type Item = Result<DirEntry, FsError>;

  fn next(&mut self) -> Option<Self::Item> {
    #[cfg(windows)]
    {
      use windows::Win32::Storage::FileSystem::FindNextFileW;

      loop {
        if self.first {
          self.first = false;
        } else {
          let success =
            unsafe { FindNextFileW(self.handle, core::ptr::from_mut(&mut self.find_data)) };
          if success.is_err() {
            return None;
          }
        }

        let file_name = &self.find_data.cFileName;

        // Skip "." and ".."
        if file_name[0] == b'.' as u16 && file_name[1] == 0 {
          continue;
        }
        if file_name[0] == b'.' as u16 && file_name[1] == b'.' as u16 && file_name[2] == 0 {
          continue;
        }

        // Determine length of null-terminated string
        let mut len = 0;
        while len < file_name.len() && file_name[len] != 0 {
          len += 1;
        }

        let mut path = self.parent.clone();
        path.pop_nul_if_present();

        if !path.as_slice().is_empty() && path.as_slice().last() != Some(&SEP) {
          path.push_slice(&[SEP]);
        }
        path.push_slice(&file_name[..len]);

        return Some(Ok(DirEntry { path }));
      }
    }

    #[cfg(not(windows))]
    {
      use core::ffi::CStr;

      loop {
        let entry = unsafe { libc::readdir(self.dirp) };
        if entry.is_null() {
          return None;
        }

        let d_name = unsafe { (*entry).d_name.as_ptr() };
        let file_name = unsafe { CStr::from_ptr(d_name) }.to_bytes();

        // Skip "." and ".."
        if file_name == b"." || file_name == b".." {
          continue;
        }

        let mut path = self.parent.clone();
        let slice =
          unsafe { core::slice::from_raw_parts(d_name as *const os_char, file_name.len()) };

        path.pop_nul_if_present();
        if !path.as_slice().is_empty() && path.as_slice().last() != Some(&SEP) {
          path.push_slice(&[SEP]);
        }

        path.push_slice(slice);

        return Some(Ok(DirEntry { path }));
      }
    }
  }
}

impl Drop for ReadDir {
  fn drop(&mut self) {
    #[cfg(windows)]
    {
      use windows::Win32::Storage::FileSystem::FindClose;
      let _ = unsafe { FindClose(self.handle) };
    }
    #[cfg(not(windows))]
    {
      unsafe { libc::closedir(self.dirp) };
    }
  }
}

// --- Implement read_dir ---

pub fn read_dir(path: &Path) -> Result<ReadDir, FsError> {
  #[cfg(windows)]
  {
    use windows::Win32::Storage::FileSystem::{FindFirstFileW, WIN32_FIND_DATAW};
    use windows::Win32::Foundation::INVALID_HANDLE_VALUE;

    let mut search_path = path.to_pathbuf();
    search_path.push_slice(&[SEP, b'*' as os_char]);
    search_path.ensure_nul_terminated();

    let mut find_data: WIN32_FIND_DATAW = unsafe { core::mem::zeroed() };
    let handle = unsafe {
      FindFirstFileW(
        windows::core::PCWSTR(search_path.as_ptr_mut()),
        core::ptr::from_mut(&mut find_data),
      )
    };

    if handle == INVALID_HANDLE_VALUE {
      return Err(FsError::CouldNotReadFile);
    }

    Ok(ReadDir {
      handle,
      find_data,
      first: true,
      parent: path.to_pathbuf(),
    })
  }

  #[cfg(not(windows))]
  {
    let mut path_buf = path.to_pathbuf();
    let dirp = unsafe { libc::opendir(path_buf.as_ptr_mut()) };
    if dirp.is_null() {
      return Err(FsError::CouldNotReadFile);
    }
    Ok(ReadDir {
      dirp,
      parent: path.to_pathbuf(),
    })
  }
}

// --- Supporting string conversions (`.to_str()`) ---

impl Path {
  /// Returns the path as a standard string.
  /// Note: Allocates a String on Windows due to UTF-16 decoding.
  #[cfg(windows)]
  pub fn to_str(&self) -> Option<alloc::string::String> {
    alloc::string::String::from_utf16(strip_nul(&self.inner)).ok()
  }

  #[cfg(not(windows))]
  pub fn to_str(&self) -> Option<&str> {
    let slice = strip_nul(&self.inner);
    let bytes = unsafe { core::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len()) };
    core::str::from_utf8(bytes).ok()
  }
}

/// Helper trait so `path.extension().and_then(|s| s.to_str())` compiles seamlessly,
/// bridging `impl AsRef<str>` directly to a usable `&str`.
pub trait ExtensionToStr {
  fn to_str(&self) -> Option<&str>;
}

impl<T: AsRef<str>> ExtensionToStr for T {
  fn to_str(&self) -> Option<&str> {
    Some(self.as_ref())
  }
}
