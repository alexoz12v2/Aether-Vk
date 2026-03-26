// when it grows, this will be the structure
// os.rs
// os/
//   fs.rs

pub mod fs;
pub mod native;
pub mod time;
pub mod memory;
pub mod debug;
pub mod pool;



#[cfg(test)]
mod tests {
  #[test]
  fn should_work() {}
}