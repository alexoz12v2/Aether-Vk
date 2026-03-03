use alloc::vec::Vec;

#[cfg(windows)]
pub(super) type os_char = u16;
#[cfg(not(windows))]
pub(super) type os_char = core::ffi::c_char;

pub(super) const SEP: os_char = if cfg!(windows) {
  b'\\' as os_char
} else {
  b'/' as os_char
};

pub(super) trait FileSystemObject {
  fn is_valid(&self) -> bool;
  fn is_dir(&self) -> bool;
  fn is_file(&self) -> bool;
}

pub(super) struct Path {
  inner: [os_char],
}

impl Path {
  pub(super) fn from_slice(slice: &[os_char]) -> &Self {
    let ptr: *const Self = (slice as *const [os_char]) as *const Path;
    unsafe { &*ptr }
  }

  pub(super) fn to_pathbuf(&self) -> PathBuf {
    let mut the_pathbuf = PathBuf::new();
    the_pathbuf.push_slice(&self.inner);

    the_pathbuf
  }
}

impl FileSystemObject for Path {
  fn is_valid(&self) -> bool {
    todo!()
  }

  fn is_dir(&self) -> bool {
    todo!()
  }

  fn is_file(&self) -> bool {
    todo!()
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

pub(super) struct PathBuf {
  storage: PathStorage,
}

const PATH_SBO_CAP: usize = if cfg!(windows) { 32 } else { 64 };

enum PathStorage {
  Inline(heapless::Vec<os_char, PATH_SBO_CAP>),
  Heap(Vec<os_char>),
}

impl PathBuf {
  pub(super) fn new() -> Self {
    Self {
      storage: PathStorage::Inline(heapless::Vec::new()),
    }
  }

  pub(super) fn push_slice(&mut self, s: &[os_char]) {
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

  pub(super) fn push(&mut self, s: &str) {
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
  pub(super) fn as_ptr(&mut self) -> *const os_char {
    self.ensure_nul_terminated();
    self.as_slice().as_ptr()
  }

  pub(super) fn as_slice(&self) -> &[os_char] {
    match &self.storage {
      PathStorage::Inline(vec_inner) => &vec_inner,
      PathStorage::Heap(items) => &items,
    }
  }

  pub(super) fn pop(&mut self) {
    while let Some(c) = self.pop_unit() {
      if c == SEP {
        return;
      }
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
    match &mut self.storage {
      PathStorage::Inline(vec_inner) => {
        if *vec_inner.last().unwrap() == b'\0' as os_char {
          vec_inner.pop();
        }
      }
      PathStorage::Heap(items) => {
        if *items.last().unwrap() == b'\0' as os_char {
          items.pop();
        }
      }
    }
  }

  #[inline]
  fn ensure_nul_terminated(&mut self) {
    match &mut self.storage {
      PathStorage::Inline(vec_inner) => {
        if *vec_inner.last().unwrap() != b'\0' as os_char {
          self.push_unit(b'\0' as os_char);
        }
      }
      PathStorage::Heap(items) => {
        if *items.last().unwrap() == b'\0' as os_char {
          items.push(b'\0' as os_char);
        }
      }
    }
  }
}

impl AsRef<Path> for PathBuf {
  fn as_ref(&self) -> &Path {
    Path::from_slice(self.as_slice())
  }
}
