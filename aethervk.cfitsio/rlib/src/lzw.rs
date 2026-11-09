//! Unix `compress` (`.Z`, LZW) in pure Rust: cfitsio's `zuncompress.c` only inflates a `.Z` from a
//! `FILE *`, so the memory path decodes it here. The algorithm follows gzip's `unlzw.c` (the
//! de-facto reference for the format): 9 to `maxbits` (≤ 16) bit codes packed LSB first, block
//! mode with the `CLEAR` code 256, and the format's alignment quirk: the code stream is written
//! in groups of eight codes (`n_bits` bytes), so every width change and every `CLEAR` pads to the
//! next group boundary, counted from the previous boundary.

use alloc::vec::Vec;
use core::ffi::c_int;

use aethervk_cfitsio_sys as sys;

use crate::{FitsError, MAX_GUNZIP_BYTES};

const INIT_BITS: u32 = 9;
const CLEAR: u32 = 256;
const BIT_MASK: u8 = 0x1f;
const BLOCK_MODE: u8 = 0x80;

fn corrupt(what: &str) -> FitsError {
  FitsError::wrapper(sys::DATA_DECOMPRESSION_ERR as c_int, what)
}

/// Decodes a `.Z` stream (`1f 9d` magic). At most [`MAX_GUNZIP_BYTES`] of output.
pub fn unlzw(bytes: &[u8]) -> Result<Vec<u8>, FitsError> {
  if bytes.len() < 3 || !crate::is_unix_compress(bytes) {
    return Err(corrupt("not a compress (.Z) stream"));
  }
  let maxbits = u32::from(bytes[2] & BIT_MASK);
  let block_mode = bytes[2] & BLOCK_MODE != 0;
  if !(INIT_BITS..=16).contains(&maxbits) {
    return Err(corrupt("compress (.Z): unsupported number of bits"));
  }
  let maxmaxcode: u32 = 1 << maxbits;
  let first: u32 = if block_mode { 257 } else { 256 };
  let data = &bytes[3..];
  // the code stream, as gzip reads it: `base` moves to every padding boundary
  let mut base = 0usize;
  let mut posbits = 0usize;
  let mut n_bits = INIT_BITS;
  let mut maxcode: u32 = (1 << n_bits) - 1;
  let mut free_ent = first;
  let mut oldcode: Option<u32> = None;
  let mut finchar: u8 = 0;
  // prefix / suffix tables
  let mut prefix: Vec<u16> = alloc::vec![0; 1 << maxbits];
  let mut suffix: Vec<u8> = (0..=255u8).collect();
  suffix.resize(1 << maxbits, 0);
  let mut out: Vec<u8> = Vec::with_capacity(data.len() * 3);
  let mut stack: Vec<u8> = Vec::with_capacity(1 << maxbits);
  let round_up = |posbits: usize, n_bits: u32| -> usize {
    let group = (n_bits as usize) << 3;
    (posbits - 1) + (group - (posbits - 1 + group) % group)
  };
  loop {
    if free_ent > maxcode {
      // a width change: pad to the group boundary (with the old width), then restart the groups
      let p = if posbits == 0 { 0 } else { round_up(posbits, n_bits) };
      base += p >> 3;
      posbits = 0;
      n_bits += 1;
      maxcode = if n_bits == maxbits { maxmaxcode } else { (1 << n_bits) - 1 };
    }
    let avail_bits = (data.len().saturating_sub(base)) << 3;
    if posbits + n_bits as usize > avail_bits {
      break; // no whole code left
    }
    let byte = base + (posbits >> 3);
    let get = |i: usize| -> u32 { data.get(i).copied().map_or(0, u32::from) };
    let word = get(byte) | get(byte + 1) << 8 | get(byte + 2) << 16;
    let code = (word >> (posbits & 7)) & ((1 << n_bits) - 1);
    posbits += n_bits as usize;
    let Some(old) = oldcode else {
      if code > 255 {
        return Err(corrupt("compress (.Z): corrupt input (first code)"));
      }
      finchar = code as u8;
      oldcode = Some(code);
      out.push(finchar);
      continue;
    };
    if code == CLEAR && block_mode {
      prefix.iter_mut().for_each(|p| *p = 0);
      free_ent = first - 1;
      let p = if posbits == 0 { 0 } else { round_up(posbits, n_bits) };
      base += p >> 3;
      posbits = 0;
      n_bits = INIT_BITS;
      maxcode = (1 << n_bits) - 1;
      continue;
    }
    let incode = code;
    let mut code = code;
    stack.clear();
    if code >= free_ent {
      if code > free_ent {
        return Err(corrupt("compress (.Z): corrupt input (code past the table)"));
      }
      // KwKwK
      stack.push(finchar);
      code = old;
    }
    while code >= 256 {
      stack.push(suffix[code as usize]);
      code = u32::from(prefix[code as usize]);
    }
    finchar = suffix[code as usize];
    stack.push(finchar);
    if out.len() + stack.len() > MAX_GUNZIP_BYTES {
      return Err(FitsError::wrapper(sys::MEMORY_ALLOCATION as c_int, "compress (.Z): output over the size cap"));
    }
    out.extend(stack.iter().rev());
    if free_ent < maxmaxcode {
      prefix[free_ent as usize] = old as u16;
      suffix[free_ent as usize] = finchar;
      free_ent += 1;
    }
    oldcode = Some(incode);
  }
  Ok(out)
}
